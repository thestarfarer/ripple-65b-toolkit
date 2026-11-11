"""SupportsLoRA for vllm ricursive - subclass + re-register. Import before LLM()."""
from vllm import ModelRegistry
from vllm.model_executor.models.ricursive import RicursiveForCausalLM
from vllm.model_executor.models.interfaces import SupportsLoRA


class RicursiveForCausalLMLoRA(RicursiveForCausalLM, SupportsLoRA):
    # packed_modules_mapping inherited from the in-tree class (qkg/gate_up).
    pass


def register():
    ModelRegistry.register_model("RicursiveForCausalLM", RicursiveForCausalLMLoRA)
