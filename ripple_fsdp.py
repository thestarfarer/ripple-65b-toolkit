#!/usr/bin/env python3
"""Ordered Ripple BF16/rsLoRA training with native FSDP, launched with torchrun.

Defaults: context 512, global batch 16, r128/alpha256, LR 5e-7,
100 warmup updates then constant LR. See README.md for installation and runs.
No Unsloth/TRL, quantization, implicit truncation, or embedded credentials.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from packed_data import PackedCollator, encode_packed_file, detect_format

BASE_MODEL = "ricursiveAI/Ripple-65B-Base-Preview"
BASE_REVISION = "6187d1108a25fcc31681ec942ec3c581721b9fe1"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FORMAT_VERSION = 2
STOP_REQUESTED = False


def env_int(name, default):
    return int(os.environ.get(name, default))


def arguments(argv=None, stage="probe"):
    defaults = {"probe": (512, 100, 50, "probe", "text", "mix_ctx512_ordered.jsonl"),
                "512": (512, 100, 50, "mixv2", "text", "mix_ctx512_ordered.jsonl"),
                "2048": (2048, 50, 100, "length", "auto", "finale_packed_2048_v3.jsonl")}
    length, warmup, eval_steps, kind, data_format, train_file = defaults[stage]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("RIPPLE_MODEL", BASE_MODEL))
    p.add_argument("--revision", default=os.environ.get("RIPPLE_REVISION", BASE_REVISION))
    p.add_argument("--train-file", default=os.environ.get("MIX_DATA", train_file))
    p.add_argument("--eval", action="append", default=None, metavar="NAME=PATH")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--output-dir", default=os.environ.get("LIL_OUTPUT"))
    p.add_argument("--rank", type=int, choices=[32, 64, 128], default=env_int("LIL_RANK", 128))
    p.add_argument("--alpha", type=int, default=int(os.environ["LIL_ALPHA"]) if "LIL_ALPHA" in os.environ else None)
    p.add_argument("--max-length", type=int, default=env_int("LIL_MAX_LEN", length))
    p.add_argument("--data-format", choices=["auto", "text", "packed"],
                   default="packed" if os.environ.get("LIL_PACKED") == "1" else data_format,
                   help="auto: any packed key selects isolated packed processing for train and eval")
    p.add_argument("--micro-batch", type=int, default=env_int("LIL_MICRO_BATCH", 1))
    p.add_argument("--global-batch", type=int, default=env_int("LIL_GLOBAL_BATCH", 16))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LIL_LR", "5e-7")))
    p.add_argument("--warmup", type=int, default=env_int("LIL_WARMUP", warmup))
    p.add_argument("--max-steps", type=int, default=env_int("LIL_MAX_STEPS", 0), help="0: one pass over the file")
    p.add_argument("--save-steps", type=int, default=env_int("LIL_SAVE_STEPS", 200))
    p.add_argument("--eval-steps", type=int, default=env_int("LIL_EVAL_STEPS", eval_steps))
    p.add_argument("--eval-batch", type=int, default=env_int("LIL_EVAL_BS", 1))
    p.add_argument("--eval-token-budget", type=int, default=env_int("LIL_EVAL_TOK_BUDGET", 15872))
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--overflow", choices=["error", "truncate"], default="error")
    p.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None,
                   help="Text default: append EOS if space remains. Packed: no added special tokens")
    p.add_argument("--include-attention-gate", action="store_true", help="Also adapt g_proj; off for the seven-target comparison")
    p.add_argument("--resume", default="auto" if os.environ.get("LIL_RESUME", "1") != "0" else "none",
                   help="auto, none, or a complete checkpoint directory; same topology required")
    p.add_argument("--init-adapter", default=os.environ.get("LIL_INIT_ADAPTER"), help="Ripple weights-only warm start")
    p.add_argument("--fresh-adapter", action="store_true", help="Explicitly start the 2048 stage without a 512 adapter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", choices=["none", "wandb"], default="none")
    p.add_argument("--run-kind", default=os.environ.get("LIL_RUN_KIND", os.environ.get("RUN_KIND", kind)))
    p.add_argument("--dry-run", action="store_true", help="CPU tokenizer/data checks only; does not load 65B weights")
    p.add_argument("--cpu-test", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.stage = stage
    a.alpha = round(math.sqrt(512 * a.rank)) if a.alpha is None else a.alpha
    if a.output_dir is None:
        a.output_dir = f"ripple-{a.run_kind}-ctx{a.max_length}-r{a.rank}-a{a.alpha}-lr{a.lr:g}-w{a.warmup}"
    a.output_dir = str(Path(a.output_dir).resolve())
    if min(a.max_length - 1, a.micro_batch, a.global_batch, a.eval_batch, a.eval_token_budget, a.keep_checkpoints, a.alpha) <= 0:
        p.error("length must be >=2 and batch sizes, budgets, checkpoint retention, and alpha must be positive")
    if min(a.warmup, a.max_steps, a.save_steps, a.eval_steps) < 0 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error("LR must be positive; step settings cannot be negative")
    if os.environ.get("LIL_SCHED", "constant") != "constant":
        p.error("This recipe uses constant-with-warmup; unset the previous LIL_SCHED override")
    if a.fresh_adapter and a.init_adapter:
        p.error("--fresh-adapter cannot be combined with --init-adapter")
    if a.model != BASE_MODEL and not Path(a.model).is_dir() and a.revision == BASE_REVISION:
        p.error("A different remote model needs an explicit --revision")
    if a.cpu_test and not Path(a.model).is_dir():
        p.error("--cpu-test requires a local tiny Ripple fixture")
    return a


def revision_kwargs(a):
    return {} if Path(a.model).is_dir() else {"revision": a.revision}


def log(message):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)


def canonical(name):
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")


def eval_paths(a):
    if a.no_eval:
        return {}
    entries = a.eval
    if entries is None and os.environ.get("EVAL_SETS"):
        entries = os.environ["EVAL_SETS"].split(",")
    if entries is None:
        entries = [f"lil={os.environ.get('EVAL_LIL', 'eval_lil_256.jsonl')}",
                   f"v3={os.environ.get('EVAL_V3', 'prose_eval_512.jsonl')}",
                   f"book={os.environ.get('EVAL_BOOK', 'eval_bookprose_256.jsonl')}"]
    result = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        if not sep or not name or not path or name in result:
            raise ValueError(f"Invalid or duplicate --eval NAME=PATH: {entry!r}")
        result[name] = path
    return result


def encode_file(path, tokenizer, cap, append_eos=True, overflow="error"):
    """Plain-text rows only. Preserve file order and retain a fingerprint per row."""
    rows, hashes = [], []
    overflow_count = 0
    longest = 0
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank row; ordered data must be explicit")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            if "packed" in row:
                raise ValueError(f"{path}:{line_no}: packed rows require --data-format packed or auto")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_no}: expected a nonempty 'text' string")
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            longest = max(longest, len(ids))
            if len(ids) > cap:
                overflow_count += 1
                if overflow == "truncate":
                    ids = ids[:cap]
                else:
                    # Finish counting before failing, without saving an oversized tensor.
                    ids = ids[:cap]
            eos = tokenizer.eos_token_id
            if append_eos and eos is not None and len(ids) < cap and (not ids or ids[-1] != eos):
                ids = ids + [eos]
            if len(ids) < 2:
                raise ValueError(f"{path}:{line_no}: fewer than two tokens; no next-token training target")
            rows.append(ids)
            hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    if overflow_count and overflow == "error":
        raise ValueError(f"{path}: {overflow_count}/{len(rows)} rows exceed {cap} Ripple tokens "
                         f"(longest {longest}). Rebuild windows, or explicitly opt into --overflow truncate.")
    report = {"rows": len(rows), "tokens": sum(map(len, rows)), "loss_tokens": sum(len(x)-1 for x in rows),
              "overlength_rows": overflow_count, "longest_before_eos": longest}
    return rows, hashes, report


def prefix_hash(hashes, count):
    return hashlib.sha256("".join(hashes[:count]).encode("ascii")).hexdigest()


def row_length(row):
    return len(row["input_ids"]) if isinstance(row, dict) else len(row)


def target_count(row):
    return sum(x != -100 for x in row["labels"][1:]) if isinstance(row, dict) else len(row) - 1


def collate(rows, indices, pad, device, mask_dtype=torch.float32):
    if isinstance(rows[0], dict):
        dummy = {"input_ids": [pad, pad], "labels": [-100, -100], "seg_lens": [2]}
        batch = PackedCollator(pad, mask_dtype)([rows[i] if i is not None else dummy for i in indices])
        return {key: value.to(device) for key, value in batch.items()}
    selected = [rows[i] if i is not None else [pad, pad] for i in indices]
    length = max(map(len, selected))
    ids = torch.full((len(selected), length), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for b, (index, seq) in enumerate(zip(indices, selected)):
        ids[b, :len(seq)] = torch.tensor(seq)
        mask[b, :len(seq)] = 1
        if index is not None:
            labels[b, :len(seq)] = torch.tensor(seq)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def micro_indices(start, stop, world, rank, micro, accumulation):
    """All ranks collectively visit each global-batch row once, including the tail."""
    for k in range(accumulation):
        first = start + k * world * micro + rank * micro
        yield [i if i < stop else None for i in range(first, first + micro)]


def loss_sum(logits, labels):
    # Explicit FP32 CE; sum is zero (not NaN) for an all-dummy microbatch.
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")


def ripple_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values):
    """Training-only mask path: preserve 4D isolation and causal/padding constraints."""
    if past_key_values is not None:
        raise ValueError("Training mask patch requires use_cache=False and no past_key_values")
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask.to(device=input_tensor.device, dtype=input_tensor.dtype)
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("Expected a 2D padding mask or a 4D additive mask")
    b, length = input_tensor.shape[:2]
    allowed = torch.ones((length, length), dtype=torch.bool, device=input_tensor.device).tril()
    allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
    result = torch.zeros((b, 1, length, length), dtype=input_tensor.dtype, device=input_tensor.device)
    return result.masked_fill(~allowed, torch.finfo(input_tensor.dtype).min)


def validate_adapter(path, a):
    config = json.loads((Path(path) / "adapter_config.json").read_text())
    expected = set(TARGETS + (["g_proj"] if a.include_attention_gate else []))
    if config.get("r") != a.rank or set(config.get("target_modules", [])) != expected:
        raise ValueError("Warm-start adapter rank/targets differ from this run")
    if config.get("lora_alpha") != a.alpha or config.get("use_rslora") is not True:
        raise ValueError("Warm-start adapter must have matching alpha and rsLoRA scaling")
    if config.get("base_model_name_or_path") != a.model:
        raise ValueError("Warm start must use an adapter for this exact Ripple base, not Ministral")
    if not Path(a.model).is_dir() and config.get("revision") != a.revision:
        raise ValueError("Warm-start adapter base revision differs from this pinned Ripple revision")
    return load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")


def build_model(a, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    config = AutoConfig.from_pretrained(a.model, trust_remote_code=True, **revision_kwargs(a))
    if config.model_type != "ricursive":
        raise ValueError("This implementation is specifically for Ripple's ricursive architecture")
    if a.cpu_test and (config.hidden_size > 128 or config.num_hidden_layers > 8):
        raise ValueError("CPU test fixture must have hidden_size <=128 and <=8 layers")
    dtype = torch.float32 if a.cpu_test else torch.bfloat16
    config.use_cache = False
    if rank == 0 or a.cpu_test:
        log("Loading frozen base on rank 0 CPU; other CUDA ranks use meta initialization.")
        base = AutoModelForCausalLM.from_pretrained(a.model, config=config, dtype=dtype,
                                           low_cpu_mem_usage=True, attn_implementation="sdpa",
                                           trust_remote_code=True, **revision_kwargs(a))
    else:
        with init_empty_weights(include_buffers=True):
            base = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(dtype=dtype)
    base.config.use_cache = False
    # Class patch is also present on every rank; inference exports remain ordinary PEFT adapters.
    type(base.model)._update_causal_mask = ripple_causal_mask
    decoder_class = type(base.model.layers[0])
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, use_rslora=True,
                      target_modules=TARGETS + (["g_proj"] if a.include_attention_gate else []),
                      bias="none", task_type=TaskType.CAUSAL_LM,
                      base_model_name_or_path=a.model, revision=None if Path(a.model).is_dir() else a.revision)
    # Meta adapters must be materialized by FSDP, then broadcast from rank 0.
    context = init_empty_weights(include_buffers=True) if rank != 0 and not a.cpu_test else contextlib.nullcontext()
    with context:
        peft_model = get_peft_model(base, lora)
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # FP32 persistent adapter weights/gradients/Adam moments
    adapter_names = sorted(n for n, p in peft_model.named_parameters() if p.requires_grad)
    if not adapter_names or any("lora_" not in n for n in adapter_names):
        raise RuntimeError("Unexpected trainable base parameters")
    count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log(f"rsLoRA r={a.rank}, alpha={a.alpha}, scale={a.alpha/math.sqrt(a.rank):.8f}; trainable={count:,}")
    if a.init_adapter and (rank == 0 or a.cpu_test):
        state = validate_adapter(a.init_adapter, a)
        expected = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
        if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state):
            raise ValueError("Warm-start adapter tensor names or shapes do not exactly match")
        set_peft_model_state_dict(peft_model, state)
    # Separate trainable LoRA A/B leaf modules from frozen decoder parameters.
    # Otherwise use_orig_params=True may allocate gradient storage for the frozen base.
    trainable_leaf = functools.partial(lambda_auto_wrap_policy,
        lambda_fn=lambda m: not list(m.children()) and any(p.requires_grad for p in m.parameters(recurse=False)))
    decoder_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={decoder_class})
    policy = functools.partial(_or_policy, policies=[trainable_leaf, decoder_policy])
    mp = None if a.cpu_test else MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.bfloat16, cast_forward_inputs=True)
    def materialize(module):
        if any(p.is_meta for p in module.parameters(recurse=False)) or any(b.is_meta for b in module.buffers(recurse=False)):
            module.to_empty(device=device, recurse=False)
    model = FSDP(peft_model, auto_wrap_policy=policy, use_orig_params=False,
                 sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mp,
                 device_id=device, sync_module_states=not a.cpu_test,
                 param_init_fn=materialize, limit_all_gathers=True, forward_prefetch=False)
    apply_activation_checkpointing(model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, decoder_class))
    # Exactly one uniform trainability/dtype group per FSDP unit.
    for unit in FSDP.fsdp_modules(model):
        direct = list(unit.parameters(recurse=False))
        if direct and len({p.requires_grad for p in direct}) != 1:
            raise RuntimeError("Mixed frozen/trainable FSDP unit")
    return model, peft_model, adapter_names


def contract(a, world):
    return {"format_version": FORMAT_VERSION, "model": a.model,
            "revision": None if Path(a.model).is_dir() else a.revision,
            "rank": a.rank, "alpha": a.alpha, "targets": TARGETS + (["g_proj"] if a.include_attention_gate else []),
            "world_size": world, "micro_batch": a.micro_batch, "global_batch": a.global_batch,
            "max_length": a.max_length, "data_format": a.data_format,
            "append_eos": a.append_eos, "overflow": a.overflow,
            "lr": a.lr, "warmup": a.warmup, "seed": a.seed, "cpu_test": a.cpu_test}


def complete_checkpoints(output):
    result = []
    for path in Path(output).glob("checkpoint-*"):
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit() and (path / "complete.json").is_file():
            result.append(path)
    return sorted(result, key=lambda x: int(x.name.split("-")[-1]))


def choose_resume(a):
    checkpoints = complete_checkpoints(a.output_dir)
    if a.resume == "auto":
        selected = checkpoints[-1] if checkpoints else None
    elif a.resume == "none":
        selected = None
        if checkpoints or (Path(a.output_dir) / "final").exists():
            raise ValueError("--resume none needs a fresh output directory")
    else:
        selected = Path(a.resume).resolve()
        if not (selected / "complete.json").is_file():
            raise ValueError(f"Not a complete checkpoint: {selected}")
    if selected is None and a.stage == "2048" and not a.init_adapter and not a.fresh_adapter:
        raise ValueError("Start 2048 with --init-adapter <512-output>/final and a new output directory. "
                         "Use --fresh-adapter only for an intentional fresh experiment.")
    return selected


def validate_resume(path, a, world, hashes, token_hashes):
    state = json.loads((path / "complete.json").read_text())
    expected = contract(a, world)
    saved_contract = dict(state["contract"])
    # The previous standalone plain-text rank probe used identical parameter names
    # and token fingerprints. Its v1 state can resume here without losing Adam state.
    if saved_contract.get("format_version") == 1 and "data_format" not in saved_contract:
        saved_contract.update(format_version=FORMAT_VERSION, data_format="text")
    if saved_contract != expected:
        differences = {k: (saved_contract.get(k), v) for k, v in expected.items()
                       if saved_contract.get(k) != v}
        raise ValueError(f"Resume configuration differs: {differences}. For a new topology, warm-start from checkpoint/adapter.")
    previous_rows = state["dataset_rows"]
    if len(hashes) < previous_rows or prefix_hash(hashes, previous_rows) != state["dataset_prefix_hash"]:
        raise ValueError("Training file was shortened, reordered, or changed; only append-only growth is resumable")
    if prefix_hash(token_hashes, previous_rows) != state["token_prefix_hash"]:
        raise ValueError("Tokenized training prefix changed; tokenizer/model/preprocessing must match")
    return state


def rng_state(device):
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def lr_for_update(step, peak, warmup):
    # As in LambdaLR + HF constant_with_warmup: update zero uses LR zero.
    return peak * min(1.0, step / max(1, warmup)) if warmup else peak


def export_adapter(model, peft_model, adapter_names, tokenizer, output, a):
    """Gather only LoRA leaf FSDP units. Never summon the frozen model."""
    raw = {}
    rank = dist.get_rank()
    for name, unit in model.named_modules():
        if not isinstance(unit, FSDP) or not isinstance(unit.module, torch.nn.Linear):
            continue
        clean_name = canonical(name)
        if ".lora_A." not in clean_name and ".lora_B." not in clean_name:
            continue
        with FSDP.summon_full_params(unit, recurse=False, writeback=False, rank0_only=True,
                                     offload_to_cpu=not a.cpu_test):
            if rank == 0:
                for key, value in unit.module.named_parameters():
                    raw[f"{clean_name}.{key}"] = value.detach().cpu().contiguous().clone()
    if rank == 0:
        if sorted(raw) != adapter_names:
            raise RuntimeError("Adapter export missed or renamed tensors: "
                               f"missing={set(adapter_names)-raw.keys()}, extra={raw.keys()-set(adapter_names)}")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(peft_model, state_dict=raw, save_embedding_layers=False)
        save_file(state, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
        peft_model.peft_config["default"].save_pretrained(output)
        tokenizer.save_pretrained(output)
        (output / "training_config.json").write_text(json.dumps(contract(a, dist.get_world_size()), indent=2))
    dist.barrier()


def save_checkpoint(model, peft_model, adapter_names, tokenizer, optimizer, a, device,
                    step, cursor, hashes, token_hashes, seen_tokens):
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(a.output_dir)
    final = output / f"checkpoint-{step}"
    if (final / "complete.json").exists():
        return final
    temporary = output / f"checkpoint-{step}.incomplete"
    if rank == 0:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    dist.barrier()
    # Save RNG before export; restore it after resume so checkpoint work is transparent.
    payload = {"trainable_shards": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
               "optimizer": optimizer.state_dict(), "rng": rng_state(device)}
    torch.save(payload, temporary / f"rank-{rank:05d}.pt")
    del payload
    export_adapter(model, peft_model, adapter_names, tokenizer, temporary / "adapter", a)
    if rank == 0:
        state = {"contract": contract(a, world), "step": step, "cursor": cursor,
                 "seen_loss_tokens": seen_tokens, "dataset_rows": len(hashes),
                 "dataset_prefix_hash": prefix_hash(hashes, len(hashes)),
                 "token_prefix_hash": prefix_hash(token_hashes, len(token_hashes))}
        (temporary / "complete.json").write_text(json.dumps(state, indent=2))
        os.replace(temporary, final)
        for old in complete_checkpoints(output)[:-a.keep_checkpoints]:
            shutil.rmtree(old)
        log(f"Saved resumable checkpoint: {final}")
    dist.barrier()
    return final


def load_checkpoint(path, model, optimizer, device):
    # Only load checkpoints made by this script and trusted by the user.
    state = torch.load(path / f"rank-{dist.get_rank():05d}.pt", map_location="cpu", weights_only=False)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state["trainable_shards"].keys():
        raise ValueError("FSDP parameter layout differs from the saved checkpoint")
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = state["trainable_shards"][name]
            if parameter.shape != saved.shape or parameter.dtype != saved.dtype:
                raise ValueError(f"FSDP shard shape/dtype changed: {name}")
            parameter.copy_(saved)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], device)
    dist.barrier()


def record(a, event):
    if dist.get_rank() == 0:
        with open(Path(a.output_dir) / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        log(json.dumps(event, allow_nan=False))


@torch.no_grad()
def evaluate(model, eval_sets, tokenizer, a, device, step):
    rank, world = dist.get_rank(), dist.get_world_size()
    was_training = model.training
    model.eval()
    event = {"event": "eval", "step": step}
    for name, rows in eval_sets.items():
        longest = max(map(row_length, rows))
        batch_size = max(1, min(a.eval_batch, a.eval_token_budget // longest))
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        # Identical forward-call count on every rank, even for a small or uneven eval set.
        for start in range(0, len(rows), batch_size * world):
            first = start + rank * batch_size
            indices = [i if i < len(rows) else None for i in range(first, first + batch_size)]
            batch = collate(rows, indices, tokenizer.pad_token_id, device,
                            torch.float32 if a.cpu_test else torch.bfloat16)
            labels = batch.pop("labels")
            logits = model(**batch, use_cache=False).logits
            totals[0] += loss_sum(logits, labels).double()
            totals[1] += (labels[:, 1:] != -100).sum()
        dist.all_reduce(totals)
        value = (totals[0] / totals[1]).item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite eval loss in {name}")
        event[f"eval/{name}/loss"] = value
        event[f"eval/{name}/tokens"] = int(totals[1].item())
    model.train(was_training)
    return event


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main(argv=None, stage="probe"):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    a = arguments(argv, stage)
    paths = eval_paths(a)
    for path in [a.train_file, *paths.values()]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Data file does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, **revision_kwargs(a))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if a.data_format == "auto":
        a.data_format = detect_format(a.train_file)
    if a.append_eos is None:
        a.append_eos = a.data_format == "text"
    if a.data_format == "packed" and (a.append_eos or a.overflow != "error"):
        raise ValueError("Packed mode preserves all segments with no added special tokens; "
                         "use --no-append-eos and --overflow error (the defaults). Repack overlength rows.")
    def encode(path):
        if a.data_format == "packed":
            return encode_packed_file(path, tokenizer, a.max_length)
        return encode_file(path, tokenizer, a.max_length, a.append_eos, a.overflow)
    rows, hashes, report = encode(a.train_file)
    token_hashes = [hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest() for x in rows]
    eval_sets, eval_reports = {}, {}
    for name, path in paths.items():
        eval_sets[name], _, eval_reports[name] = encode(path)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if a.global_batch % (world * a.micro_batch):
        raise ValueError(f"global-batch {a.global_batch} must be divisible by world_size {world} * micro-batch {a.micro_batch}")
    accumulation = a.global_batch // (world * a.micro_batch)
    if int(os.environ.get("RANK", "0")) == 0:
        log(json.dumps({"stage": a.stage, "data_format": a.data_format, "max_length": a.max_length,
                        "append_eos": a.append_eos, "train": report, "eval": eval_reports, "rank": a.rank, "alpha": a.alpha,
                        "global_batch": a.global_batch, "world_size": world, "accumulation": accumulation}, indent=2))
    if a.dry_run:
        return
    if "RANK" not in os.environ:
        raise RuntimeError("Launch training with torchrun --standalone --nproc_per_node=N train_rank_probe.py ...")
    if a.cpu_test:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Ripple training")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This training recipe requires native BF16-capable GPUs")
    dist.init_process_group("gloo" if a.cpu_test else "nccl", timeout=timedelta(minutes=60))
    try:
        run(a, device, tokenizer, rows, hashes, token_hashes, report, eval_sets, accumulation)
    finally:
        dist.destroy_process_group()


def run(a, device, tokenizer, rows, hashes, token_hashes, report, eval_sets, accumulation):
    rank, world = dist.get_rank(), dist.get_world_size()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    resume_path = choose_resume(a)
    resume_state = validate_resume(resume_path, a, world, hashes, token_hashes) if resume_path else None
    if resume_path and a.init_adapter:
        log("Complete checkpoint takes precedence over --init-adapter; restoring optimizer and cursor.")
        a.init_adapter = None
    cursor = resume_state["cursor"] if resume_state else 0
    step = resume_state["step"] if resume_state else 0
    seen_tokens = resume_state["seen_loss_tokens"] if resume_state else 0
    if rank == 0:
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.output_dir) / "run_config.json").write_text(json.dumps({**contract(a, world), "train_report": report}, indent=2))
    dist.barrier()
    model, peft_model, adapter_names = build_model(a, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    if resume_path:
        load_checkpoint(resume_path, model, optimizer, device)
        log(f"Resuming step {step}, next ordered row {cursor}, from {resume_path}")
    tracker = None
    if rank == 0 and a.report_to == "wandb":
        import wandb
        tracker = wandb.init(project=os.environ.get("WANDB_PROJECT", "ripple-fsdp"),
                             name=Path(a.output_dir).name, group=a.run_kind, config=contract(a, world))
    def emit(event):
        record(a, event)
        if tracker:
            tracker.log({k: v for k, v in event.items() if k not in {"event", "step"}}, step=event["step"])
    if eval_sets:
        emit(evaluate(model, eval_sets, tokenizer, a, device, step))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    while cursor < len(rows) and (not a.max_steps or step < a.max_steps):
        start_time = time.perf_counter()
        stop = min(cursor + a.global_batch, len(rows))
        global_tokens = sum(map(target_count, rows[cursor:stop]))
        if global_tokens <= 0:
            raise ValueError("Global batch has no next-token targets")
        update_lr = lr_for_update(step, a.lr, a.warmup)
        for group in optimizer.param_groups:
            group["lr"] = update_lr
        optimizer.zero_grad(set_to_none=True)
        running_loss = torch.zeros((), dtype=torch.float64, device=device)
        for indices in micro_indices(cursor, stop, world, rank, a.micro_batch, accumulation):
            batch = collate(rows, indices, tokenizer.pad_token_id, device,
                            torch.float32 if a.cpu_test else torch.bfloat16)
            labels = batch.pop("labels")
            logits = model(**batch, use_cache=False).logits
            summed = loss_sum(logits, labels)
            # FSDP averages across ranks; compensate and normalize across ALL valid tokens
            # in the optimizer update, including uneven lengths and partial final batches.
            (summed * (world / global_tokens)).backward()
            running_loss += summed.detach().double()
            del logits, summed, labels, batch
            # Deliberately no no_sync(): retain sharded gradients during accumulation.
        norm = model.clip_grad_norm_(1.0)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Non-finite gradient norm; checkpoint from the previous completed update remains usable")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        first_row = cursor
        cursor = stop
        seen_tokens += global_tokens
        dist.all_reduce(running_loss)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - start_time, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        event = {"event": "train", "step": step, "row_start": first_row, "row_stop": cursor,
                 "train/loss": running_loss.item() / global_tokens, "train/lr": update_lr,
                 "train/grad_norm": norm.item(), "train/loss_tokens": global_tokens,
                 "train/seen_loss_tokens": seen_tokens, "train/tokens_per_second": global_tokens / elapsed.item()}
        if device.type == "cuda":
            peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.int64, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            event["train/peak_allocated_GiB"] = peak.item() / 2**30
        emit(event)
        if eval_sets and a.eval_steps and step % a.eval_steps == 0:
            emit(evaluate(model, eval_sets, tokenizer, a, device, step))
        stopping = torch.tensor(int(STOP_REQUESTED), device=device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if (a.save_steps and step % a.save_steps == 0) or stopping.item():
            save_checkpoint(model, peft_model, adapter_names, tokenizer, optimizer, a, device,
                            step, cursor, hashes, token_hashes, seen_tokens)
        if stopping.item():
            log("Stop requested: saved at the completed optimizer-update boundary.")
            break
    if eval_sets and (not a.eval_steps or step % a.eval_steps != 0):
        emit(evaluate(model, eval_sets, tokenizer, a, device, step))
    save_checkpoint(model, peft_model, adapter_names, tokenizer, optimizer, a, device,
                    step, cursor, hashes, token_hashes, seen_tokens)
    export_adapter(model, peft_model, adapter_names, tokenizer, Path(a.output_dir) / "final", a)
    if rank == 0:
        (Path(a.output_dir) / "sweep_result.json").write_text(json.dumps({"rank": a.rank, "alpha": a.alpha,
            "lr": a.lr, "warmup": a.warmup, "steps": step, "rows_consumed": cursor,
            "loss_tokens_seen": seen_tokens, "complete_pass": cursor == len(rows)}, indent=2))
    if tracker:
        tracker.finish()
    log(f"Saved PEFT adapter: {a.output_dir}/final")


if __name__ == "__main__":
    main()
