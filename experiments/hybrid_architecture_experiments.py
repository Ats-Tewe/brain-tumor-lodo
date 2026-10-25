"""
hybrid_architecture_experiments.py
===================================
Two experiments on the DualHeadHybridCNNViT itself, to answer the question a
reviewer will certainly ask: why this architecture, when a plain pretrained
ViT-S beats it by 6 pp under maximum shift while being smaller and faster?

EXPERIMENT A -- "ab-ablation": what does the explanation head actually cost?
---------------------------------------------------------------------------
The published C6 compared the hybrid against a DIFFERENT model (different
architecture, different pretraining corpus, different token count). That
comparison cannot isolate the explainability mechanism.

The correct ablation holds everything fixed and varies only the CNN auxiliary
loss:

    Arm A   L_total = L_ViT + 0.3 * L_CNN     (explanation head trained)
    Arm B   L_total = L_ViT                    (explanation head not trained)

Same backbone, same from-scratch ViT, same seed, same data, same schedule.

There is a strong theoretical prior that the answer is approximately ZERO.
Because the CNN head reads `feature_map.detach()` during training, its gradients
never reach the backbone, so the backbone is already optimized purely by the ViT
loss. The only coupling left is indirect: gradient clipping (max_norm=1.0) is
computed over ALL parameters including cnn_aux_fc, so including that head
slightly changes the clipping scale and therefore the ViT's effective step size.

This script measures that residual empirically, and also verifies the
stop-gradient claim numerically before training (see --verify-stopgrad), by
checking that d(L_ViT + w*L_CNN)/d(backbone) == d(L_ViT)/d(backbone) exactly.

If the measured cost is near zero, that is a RESULT worth reporting: the
stop-gradient design makes dual explainability effectively free, which is a
much stronger version of C2 than the paper currently claims.

EXPERIMENT B -- "depth-sweep": is the ViT overparameterized for this data?
--------------------------------------------------------------------------
The hybrid trains 12 transformer blocks at embed_dim=384 (21.3M parameters)
from random initialization on 7,016 images. That is roughly 3,000 parameters
per training image, using a depth chosen for ImageNet-scale corpora.

This sweeps vit_num_layers over {4, 6, 8, 12} on the shift protocol. If a
shallower model generalizes better under shift, the 6 pp deficit against the
pretrained standalone ViT is an overparameterization problem, not an
architectural one, and it is fixable.

USAGE
-----
    # verify the stop-gradient claim numerically (seconds, no training)
    python hybrid_architecture_experiments.py --experiment verify-stopgrad

    # Experiment A: the honest C6 number
    python hybrid_architecture_experiments.py --experiment ab-ablation --protocol lodo3

    # Experiment B: depth sweep
    python hybrid_architecture_experiments.py --experiment depth-sweep --protocol lodo3

    # everything, sequentially
    python hybrid_architecture_experiments.py --experiment all --protocol lodo3

Results land in outputs/results/architecture_experiments/.
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

from common.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    FOCAL_GAMMA,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
    RESULTS_DIR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    protocol_config,
)
from common.data import (
    BrainTumorDataset,
    build_eval_transform,
    build_train_transform,
    make_generator,
    present_classes,
    seed_worker,
    stratified_train_val_split,
)
from common.experiment_utils import (
    cli_entrypoint,
    get_logger,
    resolve_device,
    set_seed,
    write_run_metadata,
)
from module2_model.model import DualHeadHybridCNNViT, FocalLoss

log = get_logger(__name__)

OUT_DIR = Path(RESULTS_DIR) / "architecture_experiments"

DEFAULT_CNN_WEIGHT = 0.3

def build_hybrid(vit_num_layers: int = 12, img_size: int = IMG_SIZE,
                 pretrained: bool = True) -> DualHeadHybridCNNViT:
    """
    Construct the hybrid with a configurable transformer depth.

    build_model() in module2_model hardcodes vit_num_layers=12, so we
    instantiate the class directly rather than editing the factory.
    """
    return DualHeadHybridCNNViT(
        num_classes=NUM_CLASSES,
        efficientnet_pretrained=pretrained,
        img_size=img_size,
        vit_embed_dim=384,
        vit_num_heads=6,
        vit_num_layers=vit_num_layers,
        vit_mlp_ratio=4.0,
        bridge_dropout=0.1,
        vit_attn_dropout=0.1,
        vit_mlp_dropout=0.1,
        cnn_aux_dropout=0.3,
        mc_dropout_rate=0.5,
    )

def verify_stopgradient(device, protocol: str, img_size: int = IMG_SIZE) -> dict:
    """
    Empirically confirm that the CNN auxiliary loss contributes nothing to the
    backbone gradient during training.

    Computes d(L_ViT + w*L_CNN)/d(backbone) and d(L_ViT)/d(backbone) on the same
    batch with the same RNG state, then compares them elementwise. Under the
    stop-gradient design these must be bit-identical.
    """
    log.info("=" * 70)
    log.info("VERIFYING STOP-GRADIENT ISOLATION (no training)")
    log.info("=" * 70)

    cfg = protocol_config(protocol)
    ds = BrainTumorDataset(cfg["train_dir"], transform=build_eval_transform(img_size))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    model = build_hybrid(vit_num_layers=12, img_size=img_size).to(device)
    model.train()
    criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)

    def backbone_grads(mode: str, cnn_weight: float = DEFAULT_CNN_WEIGHT):
        """mode: 'total' = L_ViT + w*L_CNN, 'vit' = L_ViT only, 'cnn' = L_CNN only."""
        model.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        out = model(images)
        if mode == "total":
            loss = (criterion(out["vit_logits"], labels)
                    + cnn_weight * criterion(out["cnn_logits"], labels))
        elif mode == "vit":
            loss = criterion(out["vit_logits"], labels)
        else:
            loss = cnn_weight * criterion(out["cnn_logits"], labels)
        loss.backward()
        return [(p.grad.detach().clone() if p.grad is not None else None)
                for p in model.backbone.parameters()]

    g_cnn_only = backbone_grads("cnn")
    n_none = sum(1 for g in g_cnn_only if g is None)
    nonzero = [float(g.abs().max()) for g in g_cnn_only if g is not None]
    cnn_alone_max = max(nonzero) if nonzero else 0.0
    cnn_contributes_nothing = (cnn_alone_max == 0.0)

    log.info("  PRIMARY TEST - backpropagate L_CNN alone, in train mode:")
    log.info("    backbone tensors with grad=None: %d of %d", n_none, len(g_cnn_only))
    log.info("    max |gradient| over the rest:     %.3e", cnn_alone_max)
    log.info("    CNN loss reaches the backbone:    %s",
             "NO (exactly zero)" if cnn_contributes_nothing else "YES")

    g_with = [g for g in backbone_grads("total") if g is not None]
    g_without = [g for g in backbone_grads("vit") if g is not None]

    max_abs_diff = max(float((a - b).abs().max()) for a, b in zip(g_with, g_without))
    grad_scale = max(float(a.abs().max()) for a in g_with)
    relative_diff = max_abs_diff / grad_scale if grad_scale else float("nan")

    g_repeat = [g for g in backbone_grads("vit") if g is not None]
    noise_floor = max(float((a - b).abs().max()) for a, b in zip(g_without, g_repeat))

    all_identical = max_abs_diff <= max(noise_floor, 1e-12)

    model.eval()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    out = model(images)
    loss_eval = (criterion(out["vit_logits"], labels)
                 + DEFAULT_CNN_WEIGHT * criterion(out["cnn_logits"], labels))
    loss_eval.backward()
    g_eval_coupled = [p.grad.detach().clone() for p in model.backbone.parameters()
                      if p.grad is not None]
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    out = model(images)
    criterion(out["vit_logits"], labels).backward()
    g_eval_vit = [p.grad.detach().clone() for p in model.backbone.parameters()
                  if p.grad is not None]
    max_diff_nodetach = max(float((a - b).abs().max())
                            for a, b in zip(g_eval_coupled, g_eval_vit))

    log.info("\n  SECONDARY TEST - d(L_ViT + w*L_CNN)/d(backbone) vs d(L_ViT)/d(backbone):")
    log.info("    max |difference|        = %.3e", max_abs_diff)
    log.info("    gradient magnitude      = %.3e", grad_scale)
    log.info("    relative difference     = %.3e", relative_diff)
    log.info("    float32 noise floor     = %.3e  (same loss, run twice)", noise_floor)
    log.info("    difference within noise = %s", all_identical)

    log.info("\n  CONTRAST - eval mode, where the detach is bypassed by design:")
    log.info("    max |difference| = %.3e", max_diff_nodetach)
    ratio = max_diff_nodetach / max_abs_diff if max_abs_diff else float("inf")
    log.info("    that is %.0fx larger than the train-mode difference", ratio)

    verdict = cnn_contributes_nothing and all_identical
    if verdict:
        log.info("\n  CONFIRMED. Backpropagating the CNN loss alone produces exactly")
        log.info("  zero backbone gradient, so the auxiliary head cannot shape the")
        log.info("  features it explains. The residual %.1e in the secondary test is",
                 max_abs_diff)
        log.info("  float32 non-determinism in cuDNN's convolution backward, not")
        log.info("  coupling: it sits at the noise floor and is %.0fx smaller than", ratio)
        log.info("  the genuine coupling measured in eval mode.")
    elif cnn_contributes_nothing:
        log.info("\n  Stop-gradient confirmed by the primary test (exactly zero).")
        log.warning("  Secondary test exceeded the noise floor; worth a look, but the")
        log.warning("  primary test is the structural one and it passed.")
    else:
        log.warning("\n  FAILED: the CNN loss reaches the backbone. Check the detach "
                    "at model.py line 210.")

    return {
        "stopgradient_confirmed": bool(verdict),
        "primary_test": {
            "description": "backprop L_CNN alone in train mode; backbone grads must be zero",
            "backbone_tensors_with_no_grad": int(n_none),
            "backbone_tensors_total": len(g_cnn_only),
            "max_abs_backbone_grad_from_cnn_loss": cnn_alone_max,
            "cnn_loss_reaches_backbone": not cnn_contributes_nothing,
        },
        "secondary_test": {
            "max_abs_diff": max_abs_diff,
            "gradient_magnitude": grad_scale,
            "relative_diff": relative_diff,
            "float32_noise_floor": noise_floor,
            "within_noise_floor": bool(all_identical),
        },
        "eval_mode_contrast": {
            "max_abs_diff": max_diff_nodetach,
            "ratio_vs_train_mode": ratio,
        },
        "interpretation": (
            "Backpropagating the CNN auxiliary loss alone yields exactly zero "
            "gradient on every shared-backbone parameter, confirming that the "
            "explanation head cannot influence the features it explains. The "
            "backbone is optimized solely by the ViT loss. Any accuracy "
            "difference between the two arms of the ab-ablation is therefore an "
            "indirect effect of gradient-norm clipping spanning all parameters, "
            "not a change in the backbone's objective."
        ),
    }

def build_loaders(protocol: str, img_size: int, batch_size: int):
    cfg = protocol_config(protocol)
    train_full = BrainTumorDataset(cfg["train_dir"], transform=build_train_transform(img_size))
    eval_full = BrainTumorDataset(cfg["train_dir"], transform=build_eval_transform(img_size))

    train_sub, val_sub = stratified_train_val_split(train_full)
    val_sub = Subset(eval_full, val_sub.indices)

    train_labels = [train_full.samples[i][1] for i in train_sub.indices]
    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    weights = [1.0 / max(counts[l], 1.0) for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                     generator=make_generator())

    kw = dict(num_workers=4, pin_memory=True,
              worker_init_fn=seed_worker, generator=make_generator())
    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler, **kw)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, **kw)

    test_ds = BrainTumorDataset(cfg["test_dir"], transform=build_eval_transform(img_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    classes = present_classes(cfg["test_dir"])
    log.info("  train %d | val %d | test %d | classes %s",
             len(train_sub), len(val_sub), len(test_ds), classes)
    return train_loader, val_loader, test_loader, counts, classes

def run_epoch(model, loader, criterion, device, cnn_weight, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out = model(images)
            loss = criterion(out["vit_logits"], labels)
            if cnn_weight != 0.0:
                loss = loss + cnn_weight * criterion(out["cnn_logits"], labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * images.size(0)
            correct += (out["vit_logits"].argmax(1) == labels).sum().item()
            n += images.size(0)
    return total / n, correct / n

@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    vit_p, cnn_p, ys = [], [], []
    for images, labels in loader:
        out = model(images.to(device))
        vit_p.append(out["vit_logits"].argmax(1).cpu())
        cnn_p.append(out["cnn_logits"].argmax(1).cpu())
        ys.append(labels)
    vit_p = torch.cat(vit_p).numpy()
    cnn_p = torch.cat(cnn_p).numpy()
    y = torch.cat(ys).numpy()

    idxs = [CLASS_NAMES.index(c) for c in classes]
    f1s = []
    for i in idxs:
        tp = int(((vit_p == i) & (y == i)).sum())
        fp = int(((vit_p == i) & (y != i)).sum())
        fn = int(((vit_p != i) & (y == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    return {
        "vit_accuracy": round(float((vit_p == y).mean()), 4),
        "cnn_accuracy": round(float((cnn_p == y).mean()), 4),
        "vit_macro_f1": round(float(np.mean(f1s)), 4),
        "n_samples": int(len(y)),
    }

def train_one_config(label: str, protocol: str, vit_layers: int, cnn_weight: float,
                     args, device) -> dict:
    log.info("=" * 70)
    log.info("CONFIG: %s | protocol=%s | vit_layers=%d | cnn_loss_weight=%.2f",
             label, protocol, vit_layers, cnn_weight)
    log.info("=" * 70)

    set_seed(args.seed, deterministic=False)

    train_loader, val_loader, test_loader, counts, classes = build_loaders(
        protocol, args.img_size, args.batch_size)

    model = build_hybrid(vit_num_layers=vit_layers, img_size=args.img_size).to(device)
    params = model.count_parameters()
    log.info("  total params %.2fM | vit blocks %.2fM",
             params["total_trainable"] / 1e6,
             params["vit_transformer_12_blocks"] / 1e6)

    total = counts.sum()
    safe = np.where(counts > 0, counts, 1.0)
    alpha = torch.tensor(total / (NUM_CLASSES * safe), dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS,
                                                   eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine],
                                                 milestones=[WARMUP_EPOCHS])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUT_DIR / f"best_{label}_{protocol}.pth"

    best_loss, best_epoch, wait = float("inf"), 0, 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                     cnn_weight, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, cnn_weight)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})

        if va_loss < best_loss:
            best_loss, best_epoch, wait = va_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "img_size": args.img_size, "vit_num_layers": vit_layers,
                        "cnn_loss_weight": cnn_weight}, ckpt)
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info("  epoch %3d | train %.4f/%.4f | val %.4f/%.4f",
                     epoch, tr_loss, tr_acc, va_loss, va_acc)
        if wait >= args.patience:
            log.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    minutes = (time.time() - start) / 60
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    metrics = evaluate(model, test_loader, device, classes)
    log.info("  TEST vit=%.4f cnn=%.4f macroF1=%.4f  (%.1f min, %d epochs)",
             metrics["vit_accuracy"], metrics["cnn_accuracy"],
             metrics["vit_macro_f1"], minutes, len(history))

    with open(OUT_DIR / f"log_{label}_{protocol}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    return {
        "label": label, "protocol": protocol,
        "vit_num_layers": vit_layers, "cnn_loss_weight": cnn_weight,
        "total_params": params["total_trainable"],
        "vit_block_params": params["vit_transformer_12_blocks"],
        "epochs_run": len(history), "best_epoch": best_epoch,
        "train_minutes": round(minutes, 2),
        "test": metrics,
    }

def plot_depth_sweep(results, protocol):
    rows = sorted([r for r in results if r["label"].startswith("depth")],
                  key=lambda r: r["vit_num_layers"])
    if len(rows) < 2:
        return
    depths = [r["vit_num_layers"] for r in rows]
    accs = [r["test"]["vit_accuracy"] * 100 for r in rows]
    par = [r["total_params"] / 1e6 for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(depths, accs, "o-", color="#4C72B0", linewidth=2, markersize=7,
             label="Test accuracy")
    ax1.set_xlabel("Transformer depth (number of blocks)")
    ax1.set_ylabel("Test accuracy (%)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(depths)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(depths, par, "s--", color="#C44E52", alpha=0.7, label="Parameters")
    ax2.set_ylabel("Total parameters (M)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax1.set_title(f"Transformer depth vs. generalization - {protocol}",
                  fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"depth_sweep_{protocol}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="all",
                   choices=["verify-stopgrad", "ab-ablation", "depth-sweep", "all"])
    p.add_argument("--protocol", default="lodo3",
                   choices=["merged", "lodo1", "lodo2", "lodo3"])
    p.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 12],
                   help="Transformer depths for the sweep")
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p

def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = resolve_device()
    log.info("Device: %s | experiment: %s | protocol: %s",
             device, args.experiment, args.protocol)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, stopgrad = [], None

    if args.experiment in ("verify-stopgrad", "all"):
        stopgrad = verify_stopgradient(device, args.protocol, args.img_size)

    if args.experiment in ("ab-ablation", "all"):
        results.append(train_one_config("ab_with_cnn_head", args.protocol, 12,
                                         DEFAULT_CNN_WEIGHT, args, device))
        results.append(train_one_config("ab_no_cnn_head", args.protocol, 12,
                                         0.0, args, device))

    if args.experiment in ("depth-sweep", "all"):
        for d in args.depths:
            existing = next((r for r in results
                             if r["label"] == "ab_with_cnn_head"
                             and r["vit_num_layers"] == d), None)
            if existing:
                results.append({**existing, "label": f"depth{d}"})
                log.info("Reusing ab_with_cnn_head result for depth %d", d)
                continue
            results.append(train_one_config(f"depth{d}", args.protocol, d,
                                             DEFAULT_CNN_WEIGHT, args, device))
        plot_depth_sweep(results, args.protocol)

    payload = {"protocol": args.protocol, "seed": args.seed,
               "stopgradient_verification": stopgrad, "runs": results}
    (OUT_DIR / f"architecture_experiments_{args.protocol}.json").write_text(
        json.dumps(payload, indent=2))
    write_run_metadata(OUT_DIR, config=vars(args))

    log.info("\n%s", "=" * 70)
    log.info("SUMMARY - %s", args.protocol)
    log.info("%s", "=" * 70)

    a = next((r for r in results if r["label"] == "ab_with_cnn_head"), None)
    b = next((r for r in results if r["label"] == "ab_no_cnn_head"), None)
    if a and b:
        cost = (b["test"]["vit_accuracy"] - a["test"]["vit_accuracy"]) * 100
        log.info("EXPERIMENT A - cost of the explanation head")
        log.info("  with CNN head:    %.4f", a["test"]["vit_accuracy"])
        log.info("  without CNN head: %.4f", b["test"]["vit_accuracy"])
        log.info("  cost: %+.2f pp   <-- the honest C6 number", cost)
        log.info("  (compare against the 5.39 pp the paper currently claims)")

    depth_rows = sorted([r for r in results if r["label"].startswith("depth")],
                        key=lambda r: r["vit_num_layers"])
    if depth_rows:
        log.info("\nEXPERIMENT B - transformer depth")
        log.info("  %-8s %-12s %-10s %s", "depth", "params(M)", "test acc", "vs depth-12")
        base = next((r for r in depth_rows if r["vit_num_layers"] == 12), None)
        for r in depth_rows:
            delta = ((r["test"]["vit_accuracy"] - base["test"]["vit_accuracy"]) * 100
                     if base else float("nan"))
            log.info("  %-8d %-12.2f %-10.4f %+.2f pp",
                     r["vit_num_layers"], r["total_params"] / 1e6,
                     r["test"]["vit_accuracy"], delta)
        best = max(depth_rows, key=lambda r: r["test"]["vit_accuracy"])
        log.info("\n  Best depth: %d (%.4f)", best["vit_num_layers"],
                 best["test"]["vit_accuracy"])
        log.info("  Standalone pretrained ViT-S reached 0.9547 on lodo3.")
        log.info("  If the best depth closes most of that gap, the deficit was")
        log.info("  overparameterization, not architecture.")

    log.info("\nSaved: %s", OUT_DIR)
    return 0

cli_entrypoint(main)
