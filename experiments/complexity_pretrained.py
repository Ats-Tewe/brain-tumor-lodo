from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from computational_complexity import (  # noqa: E402
    DEVICE,
    IMG_SIZE,
    N_MEASURE,
    N_WARMUP,
    count_params,
    measure_inference_time,
)
from pretrained_hybrid_experiment import PretrainedDualHeadHybrid  # noqa: E402

def component_breakdown(model: torch.nn.Module) -> list[tuple[str, int]]:
    """
    Parameters grouped by top-level module.

    named_parameters() removes duplicates by default, so a module reachable
    under two names (cnn_stem and vit.patch_embed.backbone are the SAME object
    in this architecture) is counted once, under whichever name PyTorch
    encounters first. The groups therefore sum to the total without
    double-counting the shared convolutional stem.
    """
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        head = name.split(".", 1)[0]
        groups[head] = groups.get(head, 0) + p.numel()
    return sorted(groups.items(), key=lambda kv: -kv[1])

def main() -> int:
    print("=" * 72)
    print("  COMPLEXITY - PretrainedDualHeadHybrid (vit_small_r26_s32_224)")
    print("=" * 72)
    print(f"  device      : {DEVICE}")
    print(f"  input       : {IMG_SIZE}x{IMG_SIZE}x3, batch 1")
    print(f"  protocol    : {N_WARMUP} warmup, {N_MEASURE} timed")
    print("  (imported from computational_complexity.py, so identical to the")
    print("   scratch-model measurements already reported)")
    print()

    model = PretrainedDualHeadHybrid(
        num_classes=4, img_size=IMG_SIZE, pretrained=False
    ).to(DEVICE)

    total, trainable = count_params(model)
    print(f"  total parameters     : {total:,} ({total / 1e6:.2f} M)")
    print(f"  trainable parameters : {trainable:,} ({trainable / 1e6:.2f} M)")
    print()

    print("  by top-level module (duplicates removed, so these sum to total):")
    acc = 0
    for name, n in component_breakdown(model):
        acc += n
        print(f"    {name:24s} {n:>12,}  ({n / 1e6:6.2f} M)")
    print(f"    {'sum':24s} {acc:>12,}  ({acc / 1e6:6.2f} M)")
    if acc != total:
        print(f"    WARNING: group sum {acc:,} does not equal total {total:,}")
    print()

    shared = (
        getattr(model, "cnn_stem", None) is
        getattr(getattr(getattr(model, "vit", None), "patch_embed", None), "backbone", None)
    )
    print(f"  cnn_stem is vit.patch_embed.backbone : {shared}")
    print("  (if True, the auxiliary head and the transformer read one feature")
    print("   extractor, and its parameters are counted once above)")
    print()

    timing = measure_inference_time(model, DEVICE, batch_size=1)
    print(f"  latency : {timing['mean_ms']:.2f} +/- {timing['std_ms']:.2f} ms per image")
    print(f"            min {timing['min_ms']:.2f}, max {timing['max_ms']:.2f}, "
          f"n={timing['n_runs']}")
    print()

    print("=" * 72)
    print("  TABLE ROW")
    print(f"  Hybrid, pretrained | {total / 1e6:.2f} M | "
          f"{timing['mean_ms']:.2f} +/- {timing['std_ms']:.2f}")
    print("=" * 72)
    return 0
