"""Locate decoder projections covered by a W4/W8 replacement artifact."""

from __future__ import annotations

from torch import nn


TARGET_PATH_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
    ".mlp.gate_proj",
)


def _decoder_layer_prefix(model: nn.Module) -> str:
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type.startswith("qwen3_5"):
        raise ValueError(
            "Qwen3.5 hybrid linear-attention coverage is not implemented; "
            "the optional model must not be partially quantized"
        )
    names = {name for name, _ in model.named_modules()}
    candidates = (
        "model.language_model.layers.",
        "model.layers.",
    )
    present = [prefix for prefix in candidates if any(name.startswith(prefix) for name in names)]
    if len(present) != 1:
        raise ValueError(
            "Expected exactly one supported decoder layer tree at "
            "model.layers or model.language_model.layers"
        )
    return present[0]


def iter_target_linears(
    model: nn.Module,
) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    """Return decoder Q/K/V/O and Gate/Up/Down linears in model order.

    The explicit decoder prefix keeps multimodal vision towers and projectors
    in their original precision.  Unsupported hybrid architectures fail
    closed instead of producing a misleading partial artifact.
    """

    module_map = dict(model.named_modules())
    result: list[tuple[nn.Module, str, str, nn.Linear]] = []
    decoder_prefix = _decoder_layer_prefix(model)

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not full_name.startswith(decoder_prefix):
            continue
        if not full_name.endswith(TARGET_PATH_SUFFIXES):
            continue

        if "." in full_name:
            parent_name, attr_name = full_name.rsplit(".", 1)
            parent = module_map[parent_name]
        else:
            parent = model
            attr_name = full_name
        result.append((parent, attr_name, full_name, module))

    return result
