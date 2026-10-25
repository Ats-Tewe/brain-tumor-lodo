from __future__ import annotations

import argparse
import csv
import json
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import timm

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
    resolve_protocols,
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

from module2_model.model import FocalLoss

log = get_logger(__name__)

OUT_DIR = Path(RESULTS_DIR) / "ablation_vit"

VARIANTS = {
    "matched": {
        "timm_name": "vit_small_patch32_224",
        "patch": 32,
        "note": "100 tokens at 320px - matches the hybrid's 10x10 feature grid",
    },
    "unmatched": {
        "timm_name": "vit_small_patch16_224",
        "patch": 16,
        "note": "400 tokens at 320px - the original ablation configuration",
    },
}

class StandaloneViT(nn.Module):
    """ViT-Small with a dropout + linear head, at a configurable patch size."""

    def __init__(self, variant: str = "matched", num_classes: int = NUM_CLASSES,
                 img_size: int = IMG_SIZE, dropout: float = 0.5,
                 pretrained: bool = True):
        super().__init__()
        spec = VARIANTS[variant]
        self.variant = variant
        self.pretrained = pretrained
        self.vit = timm.create_model(
            spec["timm_name"], pretrained=pretrained, img_size=img_size, num_classes=0
        )
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(self.vit.embed_dim, num_classes)
        self.n_tokens = (img_size // spec["patch"]) ** 2

    def forward(self, x):
        return self.head(self.drop(self.vit(x)))

    def describe(self) -> dict:
        return {
            "variant": self.variant,
            "pretrained": self.pretrained,
            "timm_model": VARIANTS[self.variant]["timm_name"],
            "n_tokens": self.n_tokens,
            "embed_dim": int(self.vit.embed_dim),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "note": VARIANTS[self.variant]["note"],
        }

def build_loaders(protocol: str, img_size: int, batch_size: int):
    """Train/val/test loaders using the project's authoritative class ordering."""
    cfg = protocol_config(protocol)
    train_dir, test_dir = cfg["train_dir"], cfg["test_dir"]

    train_full = BrainTumorDataset(train_dir, transform=build_train_transform(img_size))
    eval_full = BrainTumorDataset(train_dir, transform=build_eval_transform(img_size))

    train_sub, val_sub = stratified_train_val_split(train_full)
    val_sub = Subset(eval_full, val_sub.indices)

    train_labels = [train_full.samples[i][1] for i in train_sub.indices]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)

    weights = [1.0 / max(class_counts[lbl], 1.0) for lbl in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                     generator=make_generator())

    loader_kwargs = dict(num_workers=4, pin_memory=True,
                         worker_init_fn=seed_worker, generator=make_generator())
    train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, **loader_kwargs)

    test_classes = present_classes(test_dir)
    test_ds = BrainTumorDataset(test_dir, transform=build_eval_transform(img_size))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    log.info("  train %d | val %d | test %d", len(train_sub), len(val_sub), len(test_ds))
    log.info("  test classes present: %s", test_classes)
    return train_loader, val_loader, test_loader, class_counts, test_classes

def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass. Passing an optimizer trains; omitting it evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n

def train_model(model, train_loader, val_loader, class_counts, protocol, variant,
                device, epochs, patience):
    total = class_counts.sum()
    safe = np.where(class_counts > 0, class_counts, 1.0)
    alpha = torch.tensor(total / (NUM_CLASSES * safe), dtype=torch.float32).to(device)

    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS,
                                                   eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine],
                                                 milestones=[WARMUP_EPOCHS])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_scratch" if not model.pretrained else ""
    ckpt_path = OUT_DIR / f"best_vit_{variant}{suffix}_{protocol}.pth"

    best_loss, best_acc, best_epoch, wait = float("inf"), 0.0, 0, 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "lr": optimizer.param_groups[0]["lr"]})

        if va_loss < best_loss:
            best_loss, best_acc, best_epoch, wait = va_loss, va_acc, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "img_size": IMG_SIZE, "variant": variant,
                        "architecture": model.describe()}, ckpt_path)
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info("  epoch %3d | train %.4f/%.4f | val %.4f/%.4f | lr %.2e",
                     epoch, tr_loss, tr_acc, va_loss, va_acc,
                     optimizer.param_groups[0]["lr"])

        if wait >= patience:
            log.info("  early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    elapsed = time.time() - start
    log.info("  trained %d epochs in %.1f min | best epoch %d (val loss %.4f, acc %.4f)",
             len(history), elapsed / 60, best_epoch, best_loss, best_acc)
    return ckpt_path, history, {"best_epoch": best_epoch, "best_val_loss": best_loss,
                                 "best_val_acc": best_acc, "epochs_run": len(history),
                                 "train_minutes": round(elapsed / 60, 2)}

@torch.no_grad()
def evaluate(model, loader, device, test_classes):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    present_idx = [CLASS_NAMES.index(c) for c in test_classes]
    per_class, f1s = {}, []
    for idx in present_idx:
        tp = int(((preds == idx) & (labels == idx)).sum())
        fp = int(((preds == idx) & (labels != idx)).sum())
        fn = int(((preds != idx) & (labels == idx)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class[CLASS_NAMES[idx]] = {"precision": round(prec, 4), "recall": round(rec, 4),
                                        "f1": round(f1, 4), "support": int((labels == idx).sum())}

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    return {
        "accuracy": round(float((preds == labels).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(labels)),
    }

def save_curves(history, protocol, variant):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal loss")
    axes[0].set_title("Loss (same criterion for train and val)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"Standalone ViT ablation - {protocol} ({variant})", weight="bold")
    fig.tight_layout()
    path = OUT_DIR / f"curves_{variant}_{protocol}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def run_protocol(protocol: str, variant: str, args, device) -> dict:
    log.info("=" * 68)
    log.info("ABLATION - %s | variant=%s (%s)", protocol.upper(), variant,
             VARIANTS[variant]["note"])
    log.info("=" * 68)

    train_loader, val_loader, test_loader, class_counts, test_classes = build_loaders(
        protocol, args.img_size, args.batch_size
    )

    model = StandaloneViT(variant=variant, img_size=args.img_size,
                          pretrained=not args.no_pretrained).to(device)
    spec = model.describe()
    log.info("  %s | %d tokens | %.2fM params | pretrained=%s",
             spec["timm_model"], spec["n_tokens"], spec["total_params"] / 1e6,
             spec["pretrained"])

    ckpt_path, history, train_stats = train_model(
        model, train_loader, val_loader, class_counts, protocol, variant,
        device, args.epochs, args.patience
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    metrics = evaluate(model, test_loader, device, test_classes)
    log.info("  TEST accuracy %.4f | macro-F1 %.4f", metrics["accuracy"], metrics["macro_f1"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"log_{variant}_{protocol}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, protocol, variant)

    return {"protocol": protocol, "variant": variant, "architecture": spec,
            "training": train_stats, "test": metrics, "test_classes": test_classes}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Confound-free standalone ViT ablation")
    p.add_argument("--protocol", default="all",
                   choices=["all", "merged", "lodo1", "lodo2", "lodo3"])
    p.add_argument("--variant", default="matched", choices=["matched", "unmatched", "both"],
                   help="matched: 100 tokens (comparable to the hybrid). "
                        "unmatched: 400 tokens (the original configuration).")
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train the standalone ViT from random initialization. Use "
                        "this to isolate architecture from pretraining corpus: the "
                        "hybrid's transformer blocks are randomly initialized, so a "
                        "pretrained standalone ViT is not a like-for-like baseline.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p

def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device()
    log.info("Device: %s", device)

    variants = ["matched", "unmatched"] if args.variant == "both" else [args.variant]
    if args.no_pretrained:
        log.info("Training from RANDOM INITIALIZATION (--no-pretrained)")
    results = []
    for variant in variants:
        for protocol in resolve_protocols(args.protocol):
            try:
                results.append(run_protocol(protocol, variant, args, device))
            except Exception:
                log.exception("Protocol %s (variant %s) failed", protocol, variant)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))
    write_run_metadata(OUT_DIR, config=vars(args))

    log.info("\n%s", "=" * 68)
    log.info("SUMMARY")
    log.info("%s", "=" * 68)
    for r in results:
        log.info("  %-8s %-10s tokens=%3d  acc=%.4f  macro-F1=%.4f",
                 r["protocol"], r["variant"], r["architecture"]["n_tokens"],
                 r["test"]["accuracy"], r["test"]["macro_f1"])
    log.info("\nCompare the 'matched' rows against the hybrid's numbers in "
             "outputs/results/<protocol>/test_metrics.json. That difference is "
             "what contribution C6 should cite.")
    return 0

cli_entrypoint(main)
