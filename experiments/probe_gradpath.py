from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

def resolve(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj

def grad_norm_through(model, module, x, logit_key: str):
    """L2 norm of d(top logit)/d(module output). Returns None if not reached."""
    captured = {}

    def hook(m, inp, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        if torch.is_tensor(t):
            t.retain_grad()
            captured["t"] = t

    h = module.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    out = model(x, return_attention=False)
    h.remove()

    t = captured.get("t")
    if t is None:
        return None

    logits = out[logit_key]
    top = logits[0, logits[0].argmax()]
    top.backward()

    if t.grad is None:
        return 0.0
    return float(t.grad.detach().norm().item())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="lodo3")
    args = ap.parse_args()

    ckpt_path = Path("outputs/models") / args.protocol / "best_model.pth"
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path.resolve()}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    img_size = ckpt.get("img_size", 224)

    from pretrained_hybrid_experiment import PretrainedDualHeadHybrid

    model = PretrainedDualHeadHybrid(num_classes=4, img_size=img_size, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    x = torch.randn(1, 3, img_size, img_size, device=device, requires_grad=True)

    candidates = [
        "cnn_stem",
        "backbone",
        "vit.patch_embed.backbone",
        "vit.patch_embed",
    ]

    print("=" * 78)
    print(f"  {ckpt_path}   img_size={img_size}   device={device}")
    print("=" * 78)
    print(f"  {'module':30s} {'out shape':22s} {'d(vit)':>12s} {'d(cnn)':>12s}")
    print("  " + "-" * 74)

    for name in candidates:
        mod = resolve(model, name)
        if mod is None:
            print(f"  {name:30s} {'(absent)':22s} {'-':>12s} {'-':>12s}")
            continue

        shape = "?"
        got = {}
        h = mod.register_forward_hook(
            lambda m, i, o: got.__setitem__("o", o[0] if isinstance(o, (list, tuple)) else o))
        with torch.no_grad():
            model(x, return_attention=False)
        h.remove()
        if torch.is_tensor(got.get("o")):
            shape = str(tuple(got["o"].shape))

        gv = grad_norm_through(model, mod, x, "vit_logits")
        gc = grad_norm_through(model, mod, x, "cnn_logits")

        fv = "not reached" if gv is None else f"{gv:.6e}"
        fc = "not reached" if gc is None else f"{gc:.6e}"
        print(f"  {name:30s} {shape:22s} {fv:>12s} {fc:>12s}")

    print("  " + "-" * 74)
    print()
    print("  A d(vit) of exactly 0 means Grad-CAM hooked there explains nothing")
    print("  about the ViT prediction -- it yields a blank map, and a cosine of")
    print("  exactly 0.0000 against attention rollout. That is what the run showed.")
    print("=" * 78)

    a = resolve(model, "cnn_stem")
    b = resolve(model, "vit.patch_embed.backbone")
    c = resolve(model, "backbone")
    print()
    print("  identity checks")
    print(f"    cnn_stem is vit.patch_embed.backbone : {a is b}")
    print(f"    backbone is cnn_stem                 : {c is a}")
    print(f"    backbone is vit.patch_embed.backbone : {c is b}")
    print()
    print("  If cnn_stem and vit.patch_embed.backbone are distinct, the auxiliary")
    print("  CNN head and the transformer read different feature extractors, and")
    print("  the paper cannot claim the CNN head explains the ViT decision.")
    return 0
