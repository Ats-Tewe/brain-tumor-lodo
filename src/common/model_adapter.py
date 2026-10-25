"""
common/model_adapter.py
=======================
Lets the downstream analyses load EITHER hybrid architecture.

THE PROBLEM
-----------
`common.data.load_model_from_checkpoint` hardcodes `build_model(...)`, which
constructs a `DualHeadHybridCNNViT`. The pretrained variant is a different
class (`PretrainedDualHeadHybrid`, built on timm's vit_small_r26_s32_224), so
`evaluate.py`, `explain.py`, `uncertainty.py` and `temperature_scaling_fixed.py`
all fail to load it. Tables IV through VII cannot be regenerated for the model
the paper now reports.

THE FIX
-------
Detect the architecture from the checkpoint and build the right class. The two
models were deliberately given the same interface -- both return
{'vit_logits', 'cnn_logits', 'feature_map'} from forward() and both expose
enable_mc_dropout() -- so everything downstream works unchanged once the right
object is constructed.

Two things differ and are handled here:

  1. GRAD-CAM TARGET. The original model's convolutional features come from
     `model.backbone`; the pretrained model's come from `model.cnn_stem`.
     `gradcam_target()` returns whichever applies.

  2. ATTENTION ROLLOUT. The original model always returns per-layer attention.
     timm fuses attention by default and never materializes the matrix, so the
     pretrained model needs `enable_attention_capture()` called once before
     `forward(..., return_attention=True)` yields anything. Pass
     `enable_attention=True` when loading for explainability work.

USAGE
-----
    from common.model_adapter import load_any_checkpoint, gradcam_target

    model, ckpt, meta = load_any_checkpoint(path, device)
    model, ckpt, meta = load_any_checkpoint(path, device, enable_attention=True)

    target = gradcam_target(model)      # module to hook for Grad-CAM
    print(meta["kind"])                 # 'original' or 'pretrained'

To make the existing scripts use it without editing each one, see
`patch_loader()` at the bottom: importing this module and calling it once
redirects `common.data.load_model_from_checkpoint` to the dispatching version.
"""
from __future__ import annotations

from pathlib import Path

import torch

from common.config import NUM_CLASSES

PRETRAINED_BACKBONE = "vit_small_r26_s32_224"

def _checkpoint_kind(ckpt: dict) -> str:
    """
    Decide which class a checkpoint belongs to, from what it recorded.

    pretrained_hybrid_experiment.py stores an 'architecture' dict carrying
    'backbone'. The original train.py does not, so absence implies the original
    model. Falls back to inspecting the state_dict keys if the metadata is
    missing, because early checkpoints may predate the metadata field.
    """
    arch = ckpt.get("architecture")
    if isinstance(arch, dict) and arch.get("backbone") == PRETRAINED_BACKBONE:
        return "pretrained"

    state = ckpt.get("model_state", {})
    keys = list(state.keys())
    if any(k.startswith("vit.patch_embed.backbone") for k in keys):
        return "pretrained"
    if any(k.startswith("bridge.") or k.startswith("vit_blocks.") for k in keys):
        return "original"

    raise ValueError(
        "Cannot determine the architecture from this checkpoint. It records no "
        "'architecture' metadata and its state_dict matches neither model. "
        "Inspect the keys and load it explicitly rather than guessing."
    )

def load_any_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    enable_attention: bool = False,
) -> tuple[torch.nn.Module, dict, dict]:
    """
    Load either hybrid from a checkpoint. Returns (model in eval mode, raw
    checkpoint dict, metadata dict).

    enable_attention=True prepares the pretrained model for Attention Rollout.
    It is a no-op for the original model, which always returns attention.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    img_size = ckpt.get("img_size")
    if img_size is None:
        raise KeyError(
            f"{checkpoint_path} does not record 'img_size'. Refusing to guess: "
            f"building at the wrong resolution silently changes the token count "
            f"and the position-embedding shape."
        )

    kind = _checkpoint_kind(ckpt)

    if kind == "pretrained":
        from pretrained_hybrid_experiment import PretrainedDualHeadHybrid
        model = PretrainedDualHeadHybrid(
            num_classes=num_classes, img_size=img_size, pretrained=False
        )
    else:
        from module2_model.model import build_model
        model = build_model(num_classes=num_classes, pretrained=False, img_size=img_size)

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    n_attn_layers = 0
    if enable_attention and kind == "pretrained":
        n_attn_layers = model.enable_attention_capture()

    meta = {
        "kind": kind,
        "img_size": img_size,
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "architecture": ckpt.get("architecture"),
        "attention_enabled": bool(enable_attention and kind == "pretrained"),
        "attention_layers_patched": n_attn_layers,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    return model, ckpt, meta

def gradcam_target(model: torch.nn.Module) -> torch.nn.Module:
    """
    The module whose output Grad-CAM should hook.

    Original model  -> model.backbone   (EfficientNet-B0, 1280 channels)
    Pretrained model -> model.cnn_stem  (ResNet-26 stem, 2048 channels)

    Both produce a 10x10 spatial grid at 320px, so the downstream upsampling
    and masking code needs no change.
    """
    if hasattr(model, "cnn_stem"):
        return model.cnn_stem
    if hasattr(model, "backbone"):
        return model.backbone
    raise AttributeError(
        f"{type(model).__name__} exposes neither .cnn_stem nor .backbone; "
        f"cannot locate the Grad-CAM target."
    )

def get_attention(model: torch.nn.Module, output: dict) -> list:
    """
    Per-layer attention from a forward pass, for either model.

    The original returns it in output['attention_list'] when called with
    return_attention=True. The pretrained model stores it on its blocks and
    exposes collect_attention(). Returns [] if capture was never enabled, so
    callers can degrade rather than crash.
    """
    maps = output.get("attention_list") or []
    if maps:
        return maps
    if hasattr(model, "collect_attention") and getattr(model, "_attention_enabled", False):
        return model.collect_attention()
    return []

def patch_loader() -> None:
    """
    Redirect common.data.load_model_from_checkpoint to the dispatching version,
    so existing scripts pick up pretrained checkpoints without being edited.

    Call once, early, before anything loads a model:

        import common.model_adapter as adapter
        adapter.patch_loader()

    The signature is preserved: the original returns (model, ckpt), so this
    wrapper drops the metadata dict to stay compatible. Use
    load_any_checkpoint() directly when you want the metadata.
    """
    import common.data as _data

    def _dispatching(checkpoint_path, device, num_classes=NUM_CLASSES):
        model, ckpt, _meta = load_any_checkpoint(checkpoint_path, device, num_classes)
        return model, ckpt

    _dispatching.__doc__ = (
        "Patched by common.model_adapter.patch_loader(): dispatches on the "
        "checkpoint's recorded architecture instead of always building "
        "DualHeadHybridCNNViT."
    )
    _data.load_model_from_checkpoint = _dispatching

def describe_checkpoint(checkpoint_path: str | Path) -> dict:
    """Inspect a checkpoint without building the model. Useful for auditing."""
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    return {
        "path": str(checkpoint_path),
        "kind": _checkpoint_kind(ckpt),
        "img_size": ckpt.get("img_size"),
        "epoch": ckpt.get("epoch"),
        "architecture": ckpt.get("architecture"),
        "state_dict_tensors": len(ckpt.get("model_state", {})),
    }
