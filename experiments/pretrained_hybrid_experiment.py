"""
pretrained_hybrid_experiment.py
================================
The experiment that closes the "why this architecture?" question.

THE ARGUMENT SO FAR
-------------------
Measured on LODO R3 (maximum institutional shift):

    Standalone ViT-S, ImageNet-21k pretrained ....... 0.9547
    Standalone ViT-S, random initialization ......... 0.6850
    DualHeadHybridCNNViT (pretrained CNN + scratch ViT) 0.83-0.89

Pretraining is worth ~27 pp; the architectural gap is ~6 pp. The hybrid's
deficit against the pretrained standalone ViT is therefore most likely a
pretraining artifact: its EfficientNet backbone is pretrained but its 21.3M
transformer parameters are randomly initialized and trained on ~7,000 images.

THIS EXPERIMENT
---------------
Build the same dual-head design on a backbone that is pretrained END TO END,
and see whether the deficit disappears.

    timm 'vit_small_r26_s32_224'  =  ResNet-26 stem + ViT-Small,
    pretrained together on ImageNet-21k and fine-tuned on ImageNet-1k.

At 320px this backbone has total stride 32, giving a 10x10 grid = 100 tokens:
exactly the token count of the existing hybrid, so the comparison is direct.

The dual-head structure is preserved:

    Branch A (CNN auxiliary)  ResNet-26 feature map -> stop-gradient -> GAP
                              -> Dropout -> FC. Grad-CAM target.
    Branch B (ViT primary)    the pretrained transformer -> CLS
                              -> MC Dropout -> FC. Predictions + uncertainty.

If this reaches or exceeds 0.9547, the conclusion is clean and reportable:
the hybrid's disadvantage was pretraining, not architecture, and dual
explainability costs nothing once the transformer is pretrained too.

USAGE
-----
    python pretrained_hybrid_experiment.py --protocol lodo3
    python pretrained_hybrid_experiment.py --protocol all
    python pretrained_hybrid_experiment.py --protocol lodo3 --seed 123

    # sanity check the wiring without training
    python pretrained_hybrid_experiment.py --inspect-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import timm

from common.config import (
    BATCH_SIZE, CLASS_NAMES, EARLY_STOP_PATIENCE, FOCAL_GAMMA, IMG_SIZE,
    LABEL_SMOOTHING, LEARNING_RATE, NUM_CLASSES, NUM_EPOCHS, RANDOM_SEED,
    RESULTS_DIR, WARMUP_EPOCHS, WEIGHT_DECAY, protocol_config, resolve_protocols,
)
from common.data import (
    BrainTumorDataset, build_eval_transform, build_train_transform,
    make_generator, present_classes, seed_worker, stratified_train_val_split,
)
from common.experiment_utils import (
    cli_entrypoint, get_logger, resolve_device, set_seed, write_run_metadata,
)
from module2_model.model import FocalLoss

log = get_logger(__name__)

OUT_DIR = Path(RESULTS_DIR) / "pretrained_hybrid"
BACKBONE = "vit_small_r26_s32_224"
CNN_LOSS_WEIGHT = 0.3

def _patch_attention_for_capture(block) -> None:
    """
    Replace one timm Attention module's forward so it stores its attention
    weights on `.last_attn`.

    timm uses F.scaled_dot_product_attention when `fused_attn` is True, which
    never materializes the attention matrix, so rollout is impossible without
    this. We force the explicit path and keep a reference to the softmax
    output. Slightly slower, only enabled for explainability runs.
    """
    attn_mod = block.attn
    attn_mod.last_attn = None

    def forward_with_capture(x, *args, **kwargs):
        B, N, C = x.shape
        qkv = attn_mod.qkv(x).reshape(
            B, N, 3, attn_mod.num_heads, attn_mod.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn_mod, "q_norm"):
            q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

        attn = (q * attn_mod.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_mod.last_attn = attn.detach()
        attn = attn_mod.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return attn_mod.proj_drop(attn_mod.proj(out))

    attn_mod.forward = forward_with_capture

class PretrainedDualHeadHybrid(nn.Module):
    """
    The dual-head design rebuilt on a fully pretrained CNN-ViT backbone.

    Structurally equivalent to DualHeadHybridCNNViT: one shared convolutional
    feature map feeding a stop-gradient-isolated CNN head for Grad-CAM and a
    transformer head for prediction. The difference is that here the
    transformer arrives pretrained rather than randomly initialized.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE,
                 pretrained: bool = True, cnn_dropout: float = 0.3,
                 mc_dropout_rate: float = 0.5):
        super().__init__()
        self.img_size = img_size
        self.mc_dropout_rate = mc_dropout_rate

        self.vit = timm.create_model(
            BACKBONE, pretrained=pretrained, img_size=img_size, num_classes=0
        )

        pe = self.vit.patch_embed
        if not hasattr(pe, "backbone"):
            raise RuntimeError(
                f"{BACKBONE} does not expose patch_embed.backbone in this timm "
                f"version ({timm.__version__}). Inspect the model and adjust "
                f"the hook target."
            )
        self.cnn_stem = pe.backbone

        self._feature_map: torch.Tensor | None = None
        self.cnn_stem.register_forward_hook(self._capture)

        self.cnn_channels = self._infer_cnn_channels(img_size)
        self.embed_dim = int(self.vit.embed_dim)

        self.cnn_gap = nn.AdaptiveAvgPool2d(1)
        self.cnn_dropout = nn.Dropout(p=cnn_dropout)
        self.cnn_aux_fc = nn.Linear(self.cnn_channels, num_classes)

        self.mc_dropout = nn.Dropout(p=mc_dropout_rate)
        self.vit_primary_fc = nn.Linear(self.embed_dim, num_classes)

        for m in (self.cnn_aux_fc, self.vit_primary_fc):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    @property
    def backbone(self):
        """
        Alias for the convolutional stem.

        explain.py and the Grad-CAM code hook `model.backbone`, which exists on
        DualHeadHybridCNNViT but not here (the stem lives at .cnn_stem inside
        timm's hybrid patch embedding). Exposing it under the expected name lets
        every downstream script work unchanged. Channel count differs (2048 here
        against 1280 there) but Grad-CAM is agnostic to that, and both produce a
        10x10 grid at 320px.
        """
        return self.cnn_stem

    def _capture(self, module, inputs, output):
        self._feature_map = output[-1] if isinstance(output, (list, tuple)) else output

    @torch.no_grad()
    def _infer_cnn_channels(self, img_size: int) -> int:
        was_training = self.training
        self.eval()
        self.vit(torch.zeros(1, 3, img_size, img_size))
        self.train(was_training)
        if self._feature_map is None:
            raise RuntimeError("Forward hook did not fire; cannot infer channels.")
        return int(self._feature_map.shape[1])

    def forward(self, x, return_attention: bool = False):
        cls = self.vit(x)
        feature_map = self._feature_map

        cnn_in = feature_map.detach() if self.training else feature_map
        cnn_logits = self.cnn_aux_fc(self.cnn_dropout(self.cnn_gap(cnn_in).flatten(1)))

        vit_logits = self.vit_primary_fc(self.mc_dropout(cls))

        out = {"vit_logits": vit_logits, "cnn_logits": cnn_logits,
               "feature_map": feature_map}
        if return_attention:
            out["attention_list"] = (self.collect_attention()
                                     if getattr(self, "_attention_enabled", False)
                                     else [])
        return out

    def enable_mc_dropout(self):
        self.eval()
        self.mc_dropout.train()

    def enable_attention_capture(self) -> int:
        """
        Make the transformer expose per-layer attention, so Attention Rollout
        works on this model exactly as it does on the original hybrid. Call
        once before explainability runs. Returns the number of patched blocks.
        """
        blocks = getattr(self.vit, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                f"{BACKBONE} exposes no .blocks in timm {timm.__version__}; "
                f"cannot capture attention for rollout."
            )
        for blk in blocks:
            if not hasattr(blk, "attn"):
                raise RuntimeError("Transformer block has no .attn module.")
            blk.attn.fused_attn = False
            _patch_attention_for_capture(blk)
        self._attention_enabled = True
        return len(blocks)

    def collect_attention(self) -> list:
        """Attention maps from the most recent forward pass, one per layer."""
        if not getattr(self, "_attention_enabled", False):
            raise RuntimeError("Call enable_attention_capture() first.")
        return [blk.attn.last_attn for blk in self.vit.blocks
                if blk.attn.last_attn is not None]

    def describe(self) -> dict:
        n_tok = (self.img_size // 32) ** 2
        return {
            "backbone": BACKBONE,
            "img_size": self.img_size,
            "n_tokens": n_tok,
            "cnn_feature_channels": self.cnn_channels,
            "embed_dim": self.embed_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

def inspect(device, img_size: int) -> dict:
    """Build the model and verify the wiring without training anything."""
    log.info("=" * 70)
    log.info("INSPECTING %s", BACKBONE)
    log.info("=" * 70)

    model = PretrainedDualHeadHybrid(img_size=img_size).to(device)
    spec = model.describe()
    for k, v in spec.items():
        log.info("  %-22s %s", k, f"{v:,}" if isinstance(v, int) else v)

    x = torch.randn(2, 3, img_size, img_size).to(device)
    model.train()
    out = model(x)
    log.info("  feature map shape      %s", tuple(out["feature_map"].shape))
    log.info("  vit_logits shape       %s", tuple(out["vit_logits"].shape))
    log.info("  cnn_logits shape       %s", tuple(out["cnn_logits"].shape))

    model.zero_grad(set_to_none=True)
    crit = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    y = torch.tensor([0, 1]).to(device)
    (CNN_LOSS_WEIGHT * crit(out["cnn_logits"], y)).backward(retain_graph=True)
    stem_grads = [p.grad for p in model.cnn_stem.parameters() if p.grad is not None]
    max_g = max((float(g.abs().max()) for g in stem_grads), default=0.0)
    log.info("  CNN loss -> stem gradient max |g| = %.3e  (%s)",
             max_g, "isolated" if max_g == 0.0 else "LEAKING")

    tok = spec["n_tokens"]
    log.info("\n  %d tokens at %dpx. The existing hybrid uses 100 at 320px.",
             tok, img_size)
    log.info("  %s", "Matched." if tok == 100 else "NOT matched - comparison is confounded.")

    log.info("\n  Attention capture (required for Attention Rollout / C5):")
    attn_ok, n_layers, attn_shape = False, 0, None
    try:
        n_layers = model.enable_attention_capture()
        model.eval()
        with torch.no_grad():
            out_a = model(x, return_attention=True)
        maps = out_a["attention_list"]
        if maps:
            a = maps[0]
            attn_shape = tuple(a.shape)
            rows = a.sum(-1)
            row_ok = bool(torch.allclose(rows, torch.ones_like(rows), atol=1e-4))
            attn_ok = (len(maps) == n_layers
                       and a.shape[-1] == tok + 1
                       and row_ok)
            log.info("    patched blocks        %d", n_layers)
            log.info("    maps returned         %d", len(maps))
            log.info("    per-layer shape       %s  (B, heads, N, N)", attn_shape)
            log.info("    N == tokens + CLS     %s", a.shape[-1] == tok + 1)
            log.info("    rows sum to 1         %s", row_ok)
        else:
            log.warning("    no attention maps returned")
    except Exception as exc:
        log.warning("    attention capture FAILED: %s", exc)
    log.info("    %s", "Rollout will work." if attn_ok else
             "Rollout NOT available - C5 would need the original model.")

    return {"architecture": spec, "stopgradient_max_stem_grad": max_g,
            "tokens_match_existing_hybrid": tok == 100,
            "attention_capture": {"ok": attn_ok, "n_layers": n_layers,
                                  "map_shape": list(attn_shape) if attn_shape else None}}

def build_loaders(protocol: str, img_size: int, batch_size: int):
    cfg = protocol_config(protocol)
    train_full = BrainTumorDataset(cfg["train_dir"], transform=build_train_transform(img_size))
    eval_full = BrainTumorDataset(cfg["train_dir"], transform=build_eval_transform(img_size))
    train_sub, val_sub = stratified_train_val_split(train_full)
    val_sub = Subset(eval_full, val_sub.indices)

    labels = [train_full.samples[i][1] for i in train_sub.indices]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    w = [1.0 / max(counts[l], 1.0) for l in labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True, generator=make_generator())

    kw = dict(num_workers=4, pin_memory=True,
              worker_init_fn=seed_worker, generator=make_generator())
    return (DataLoader(train_sub, batch_size=batch_size, sampler=sampler, **kw),
            DataLoader(val_sub, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(BrainTumorDataset(cfg["test_dir"], transform=build_eval_transform(img_size)),
                       batch_size=batch_size, shuffle=False, **kw),
            counts, present_classes(cfg["test_dir"]))

def run_epoch(model, loader, criterion, device, cnn_weight, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    tot, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out = model(images)
            loss = criterion(out["vit_logits"], labels)
            if cnn_weight:
                loss = loss + cnn_weight * criterion(out["cnn_logits"], labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot += loss.item() * images.size(0)
            correct += (out["vit_logits"].argmax(1) == labels).sum().item()
            n += images.size(0)
    return tot / n, correct / n

@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    vp, cp, ys = [], [], []
    for images, labels in loader:
        out = model(images.to(device))
        vp.append(out["vit_logits"].argmax(1).cpu())
        cp.append(out["cnn_logits"].argmax(1).cpu())
        ys.append(labels)
    vp, cp, y = torch.cat(vp).numpy(), torch.cat(cp).numpy(), torch.cat(ys).numpy()

    f1s, per_class = [], {}
    for c in classes:
        i = CLASS_NAMES.index(c)
        tp = int(((vp == i) & (y == i)).sum())
        fp = int(((vp == i) & (y != i)).sum())
        fn = int(((vp != i) & (y == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        f1s.append(f1)
        per_class[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                        "f1": round(f1, 4), "support": int((y == i).sum())}

    return {"vit_accuracy": round(float((vp == y).mean()), 4),
            "cnn_accuracy": round(float((cp == y).mean()), 4),
            "vit_macro_f1": round(float(np.mean(f1s)), 4),
            "per_class": per_class, "n_samples": int(len(y))}

def train_protocol(protocol: str, args, device) -> dict:
    log.info("=" * 70)
    log.info("PRETRAINED HYBRID - %s", protocol.upper())
    log.info("=" * 70)

    set_seed(args.seed, deterministic=False)
    train_loader, val_loader, test_loader, counts, classes = build_loaders(
        protocol, args.img_size, args.batch_size)

    model = PretrainedDualHeadHybrid(img_size=args.img_size,
                                      pretrained=not args.no_pretrained).to(device)
    spec = model.describe()
    log.info("  %s | %d tokens | %.2fM params | pretrained=%s",
             spec["backbone"], spec["n_tokens"], spec["total_params"] / 1e6,
             not args.no_pretrained)

    total = counts.sum()
    safe = np.where(counts > 0, counts, 1.0)
    alpha = torch.tensor(total / (NUM_CLASSES * safe), dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=LABEL_SMOOTHING).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    warm = optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cos = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - WARMUP_EPOCHS,
                                                eta_min=1e-6)
    sched = optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[WARMUP_EPOCHS])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUT_DIR / f"best_pretrained_hybrid_{protocol}_seed{args.seed}.pth"

    best, best_ep, wait, hist = float("inf"), 0, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, device,
                                CNN_LOSS_WEIGHT, opt)
        va_l, va_a = run_epoch(model, val_loader, criterion, device, CNN_LOSS_WEIGHT)
        sched.step()
        hist.append({"epoch": ep, "train_loss": tr_l, "train_acc": tr_a,
                     "val_loss": va_l, "val_acc": va_a})
        if va_l < best:
            best, best_ep, wait = va_l, ep, 0
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "img_size": args.img_size, "architecture": spec}, ckpt)
        else:
            wait += 1
        if ep % 5 == 0 or ep == 1:
            log.info("  epoch %3d | train %.4f/%.4f | val %.4f/%.4f",
                     ep, tr_l, tr_a, va_l, va_a)
        if wait >= args.patience:
            log.info("  early stop at %d (best %d)", ep, best_ep)
            break

    mins = (time.time() - t0) / 60
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    m = evaluate(model, test_loader, device, classes)
    log.info("  TEST vit=%.4f cnn=%.4f macroF1=%.4f  (%.1f min, %d epochs)",
             m["vit_accuracy"], m["cnn_accuracy"], m["vit_macro_f1"], mins, len(hist))

    with open(OUT_DIR / f"log_{protocol}_seed{args.seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)

    return {"protocol": protocol, "seed": args.seed, "architecture": spec,
            "epochs_run": len(hist), "best_epoch": best_ep,
            "train_minutes": round(mins, 2), "test": m}

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default="lodo3",
                   choices=["all", "merged", "lodo1", "lodo2", "lodo3"])
    p.add_argument("--inspect-only", action="store_true",
                   help="Build the model and check the wiring, then exit.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p

def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = resolve_device()
    log.info("Device: %s | timm %s", device, timm.__version__)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = inspect(device, args.img_size)
    if args.inspect_only:
        (OUT_DIR / "inspection.json").write_text(json.dumps(info, indent=2))
        return 0

    results = []
    for proto in resolve_protocols(args.protocol):
        try:
            results.append(train_protocol(proto, args, device))
        except Exception:
            log.exception("Protocol %s failed", proto)

    payload = {"backbone": BACKBONE, "seed": args.seed,
               "inspection": info, "runs": results}
    (OUT_DIR / f"pretrained_hybrid_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2))
    write_run_metadata(OUT_DIR, config=vars(args))

    log.info("\n%s", "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("  Reference points on lodo3:")
    log.info("    standalone ViT-S, ImageNet-21k pretrained  0.9547")
    log.info("    standalone ViT-S, random init             0.6850")
    log.info("    existing hybrid (scratch transformer)     0.83-0.89")
    log.info("")
    for r in results:
        log.info("  %-8s pretrained hybrid: vit=%.4f  macroF1=%.4f",
                 r["protocol"], r["test"]["vit_accuracy"], r["test"]["vit_macro_f1"])
    r3 = next((r for r in results if r["protocol"] == "lodo3"), None)
    if r3:
        d = (r3["test"]["vit_accuracy"] - 0.9547) * 100
        log.info("\n  vs pretrained standalone ViT-S: %+.2f pp", d)
        if d >= -0.5:
            log.info("  The deficit is closed. It was pretraining, not architecture.")
        else:
            log.info("  A gap remains; report it rather than the architectural claim.")
    return 0

cli_entrypoint(main)
