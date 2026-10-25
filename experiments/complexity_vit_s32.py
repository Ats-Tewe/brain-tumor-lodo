from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from computational_complexity import (  # noqa: E402
    DEVICE,
    IMG_SIZE,
    N_MEASURE,
    N_WARMUP,
    measure_inference_time,
)

TIMM_NAME = "vit_small_patch32_224"
PATCH = 32
NUM_CLASSES = 4
EXPECTED_PARAMS = 22_515_076

class StandaloneViT(nn.Module):
    """Mirrors ablation_vit_standalone_fixed.StandaloneViT for the matched variant."""

    def __init__(self, num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE,
                 pretrained: bool = False):
        super().__init__()
        import timm
        self.vit = timm.create_model(TIMM_NAME, pretrained=pretrained,
                                     img_size=img_size, num_classes=0)
        self.head = nn.Linear(self.vit.embed_dim, num_classes)
        self.n_tokens = (img_size // PATCH) ** 2

    def forward(self, x):
        return self.head(self.vit(x))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/results/complexity_vit_s32.json")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE,
                    help="size the model is built for; the timing harness uses "
                         "IMG_SIZE from computational_complexity.py, so leave "
                         "this alone unless you change that too")
    ap.add_argument("--allow-param-mismatch", action="store_true",
                    help="proceed even if the parameter count differs from the "
                         "ablation run; use only if you know why it differs")
    args = ap.parse_args()

    print("=" * 68)
    print(f"  model      {TIMM_NAME}  (patch {PATCH}, token-matched)")
    print(f"  device     {DEVICE}")
    print(f"  img_size   {args.img_size}")
    print(f"  timing     {N_WARMUP} warmup + {N_MEASURE} measured, batch size 1")
    print("=" * 68)

    model = StandaloneViT(img_size=args.img_size, pretrained=False).to(DEVICE).eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  tokens     {model.n_tokens}")
    print(f"  parameters {total:,}")
    if total != EXPECTED_PARAMS:
        msg = (f"parameter count {total:,} does not match the ablation's "
               f"{EXPECTED_PARAMS:,}. The model is not built the same way, so "
               f"its latency would not belong in the same table.")
        if not args.allow_param_mismatch:
            print(f"  !! {msg}")
            print("     Re-run with --allow-param-mismatch only if you know why.")
            return 1
        print(f"  !! {msg} (continuing on request)")

    timing = measure_inference_time(model, DEVICE, batch_size=1)

    out = {
        "name": "ViT-S/32, token-matched",
        "timm_model": TIMM_NAME,
        "patch": PATCH,
        "img_size": args.img_size,
        "n_tokens": model.n_tokens,
        "total_params": total,
        "trainable_params": trainable,
        "inference_time": timing,
        "device": str(DEVICE),
        "note": ("Timed with the same function, warmup and iteration counts as "
                 "every other row of the complexity table, imported from "
                 "computational_complexity.py."),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print()
    mean = timing.get("mean_ms")
    std = timing.get("std_ms")
    print(f"  latency    {mean} +/- {std} ms")
    print(f"  wrote      {p}")
    print()
    print("  Table row to paste into Section 3.10:")
    print(f"    | ViT-S/32, token-matched | {total / 1e6:.2f} M | {mean} ± {std} |")
    print("=" * 68)
    return 0
