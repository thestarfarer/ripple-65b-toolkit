"""Isolated causal packing for Ripple's SDPA training path.

The strings are retokenized with Ripple. No BOS/EOS is inserted. The first
token of every segment is excluded from the shifted language-model loss.
"""
from __future__ import annotations

import hashlib
import json

import torch

IGNORE = -100


def records(path):
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank row in ordered data")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, row


def detect_format(path):
    # Inspect every row; the first row may be an ordinary one-block text record.
    packed = False
    for _, row in records(path):
        packed = packed or "packed" in row
    return "packed" if packed else "text"


def segments(row):
    packed = row.get("packed")
    if packed is not None and not isinstance(packed, list):
        raise ValueError("'packed' must be a list of strings (or null for a text row)")
    texts = packed if packed else [row.get("text")]
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("expected nonempty segment strings, or a nonempty 'text' string")
    return texts


def encode_row(row, tok, cap):
    ids, labels, seg_lens = [], [], []
    for text in segments(row):
        tokens = tok(text, add_special_tokens=False)["input_ids"]
        if not tokens:
            raise ValueError("segment tokenized to zero tokens")
        ids.extend(tokens)
        labels.extend([IGNORE] + tokens[1:])
        seg_lens.append(len(tokens))
    if cap is not None and len(ids) > cap:
        raise ValueError(f"packed row has {len(ids)} Ripple tokens, exceeds cap {cap}; repack without dropping segments")
    if not any(label != IGNORE for label in labels[1:]):
        raise ValueError("row has no next-token targets (each segment has only one token)")
    return {"input_ids": ids, "labels": labels, "seg_lens": seg_lens}


def encode_packed_file(path, tokenizer, cap):
    rows, hashes = [], []
    overlength, longest, total_tokens, loss_tokens, segment_count = 0, 0, 0, 0, 0
    for line_no, row in records(path):
        try:
            encoded = encode_row(row, tokenizer, None)
        except ValueError as error:
            raise ValueError(f"{path}:{line_no}: {error}") from error
        length = len(encoded["input_ids"])
        longest = max(longest, length)
        overlength += length > cap
        total_tokens += length
        loss_tokens += sum(label != IGNORE for label in encoded["labels"][1:])
        segment_count += len(encoded["seg_lens"])
        rows.append(encoded)
        # Segment boundaries are part of the resume identity, not just joined text.
        source = json.dumps(segments(row), ensure_ascii=False, separators=(",", ":"))
        hashes.append(hashlib.sha256(source.encode("utf-8")).hexdigest())
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    if overlength:
        raise ValueError(f"{path}: {overlength}/{len(rows)} rows exceed {cap} Ripple tokens (longest {longest}). "
                         "Repack using Ripple's tokenizer. Packed segments are never truncated or discarded.")
    return rows, hashes, {"rows": len(rows), "segments": segment_count, "tokens": total_tokens,
                          "loss_tokens": loss_tokens, "overlength_rows": 0, "longest": longest}


class PackedCollator:
    def __init__(self, pad_token_id, dtype=torch.bfloat16):
        self.pad_token_id = pad_token_id
        self.dtype = dtype

    def __call__(self, feats):
        length = max(len(feat["input_ids"]) for feat in feats)
        ids = torch.full((len(feats), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, IGNORE)
        positions = torch.zeros_like(ids)
        mask = torch.full((len(feats), 1, length, length), torch.finfo(self.dtype).min, dtype=self.dtype)
        for b, feat in enumerate(feats):
            n = len(feat["input_ids"])
            if sum(feat["seg_lens"]) != n or len(feat["labels"]) != n or any(s <= 0 for s in feat["seg_lens"]):
                raise ValueError("inconsistent packed feature lengths")
            ids[b, :n] = torch.tensor(feat["input_ids"])
            labels[b, :n] = torch.tensor(feat["labels"])
            start = 0
            for size in feat["seg_lens"]:
                end = start + size
                positions[b, start:end] = torch.arange(size)
                block = mask[b, 0, start:end, start:end]
                block.masked_fill_(torch.ones(size, size, dtype=torch.bool).tril(), 0)
                start = end
            # Padding queries attend only themselves; no entirely masked softmax
            # rows. Real queries still cannot attend padding or other segments.
            padding = torch.arange(n, length)
            mask[b, 0, padding, padding] = 0
        return {"input_ids": ids, "labels": labels, "position_ids": positions, "attention_mask": mask}
