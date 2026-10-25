# Temperature scaling for the dual-head hybrid.
#
# split_train_val is imported from src/training/train.py on purpose: fitting the
# source temperature requires the exact validation split the checkpoint was
# selected on. An earlier version of this file built its own stratified split
# instead, which overlapped train/ by roughly 90 percent, so the model was
# scored on images it had memorised. That version reported T* = 0.379 and
# source-val ECE 0.0112; both are artifacts of the overlap. Do not replace the
# import with a locally generated split.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import (
    BATCH_SIZE,
    ECE_N_BINS,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    protocol_config,
    resolve_protocols,
)
from common.data import (
    BrainTumorDataset,
    build_eval_transform,
    load_model_from_checkpoint,
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
from common.metrics import brier_score, compute_ece, confidence_bins, negative_log_likelihood

log = get_logger(__name__)

class TemperatureScaler(nn.Module):
    """A single scalar temperature, as in Guo et al. (2017)."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

def optimise_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit T* by L-BFGS minimization of NLL. Returns the scalar temperature."""
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=500)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler.scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().item())

@torch.no_grad()
def collect_logits(model, dataset, device, batch_size: int = BATCH_SIZE):
    """Run the ViT primary head over a dataset and return (logits, labels)."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, worker_init_fn=seed_worker, generator=make_generator(),
    )
    all_logits, all_labels = [], []
    model.eval()
    for images, labels in loader:
        out = model(images.to(device), return_attention=False)
        all_logits.append(out["vit_logits"].cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)

def _metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    probs = F.softmax(logits / max(temperature, 0.01), dim=1).numpy()
    preds = probs.argmax(axis=1)
    y = labels.numpy()
    return {
        "ece": compute_ece(probs, preds, y, ECE_N_BINS),
        "nll": negative_log_likelihood(probs, y),
        "brier": brier_score(probs, y),
        "accuracy": float((preds == y).mean()),
    }

def stratified_folds(labels: np.ndarray, k: int, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    """Split indices into k stratified folds without requiring scikit-learn."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for j, sample in enumerate(idx):
            folds[j % k].append(int(sample))
    return [np.array(sorted(f)) for f in folds]

def run_target_kfold(test_logits, test_labels, k: int) -> dict:
    """
    For each fold: fit T* on the other folds, evaluate that fold with it.
    Every test image is scored under a temperature fitted without it.
    """
    labels_np = test_labels.numpy()
    folds = stratified_folds(labels_np, k)

    per_fold, pooled_probs = [], np.zeros((len(labels_np), test_logits.shape[1]))

    for i, fold_idx in enumerate(folds):
        mask = np.ones(len(labels_np), dtype=bool)
        mask[fold_idx] = False
        fit_idx = np.where(mask)[0]

        t_star = optimise_temperature(test_logits[fit_idx], test_labels[fit_idx])

        held = _metrics(test_logits[fold_idx], test_labels[fold_idx], t_star)
        before = _metrics(test_logits[fold_idx], test_labels[fold_idx], 1.0)
        per_fold.append({
            "fold": i + 1,
            "n_fit": int(len(fit_idx)),
            "n_eval": int(len(fold_idx)),
            "T_star": round(t_star, 4),
            "ece_before": round(before["ece"], 4),
            "ece_after": round(held["ece"], 4),
        })
        pooled_probs[fold_idx] = F.softmax(
            test_logits[fold_idx] / max(t_star, 0.01), dim=1
        ).numpy()
        log.info("  fold %d/%d: T*=%.4f  ECE %.4f -> %.4f",
                 i + 1, k, t_star, before["ece"], held["ece"])

    pooled_preds = pooled_probs.argmax(axis=1)
    temps = [f["T_star"] for f in per_fold]

    return {
        "mode": "target-kfold",
        "k": k,
        "T_star_mean": round(float(np.mean(temps)), 4),
        "T_star_std": round(float(np.std(temps)), 4),
        "per_fold": per_fold,
        "before": {k2: round(v, 4) for k2, v in _metrics(test_logits, test_labels, 1.0).items()},
        "after": {
            "ece": round(compute_ece(pooled_probs, pooled_preds, labels_np, ECE_N_BINS), 4),
            "nll": round(negative_log_likelihood(pooled_probs, labels_np), 4),
            "brier": round(brier_score(pooled_probs, labels_np), 4),
            "accuracy": round(float((pooled_preds == labels_np).mean()), 4),
        },
        "_pooled_probs": pooled_probs,
    }

def run_source_val(model, cfg, device, test_logits, test_labels, img_size: int) -> dict:
    """
    Fit T* on the in-distribution validation split, then apply it to the shifted
    test set. Uses zero target-domain labels.

    IMPORTANT -- the split must be the SAME one train.py held out.
    The LODO rounds have no val/ directory on disk; train.py carves the
    validation set out of train/ in memory. If this function generates its own
    split instead, roughly 90% of the resulting "validation" images are ones the
    model was actually trained on, the model is near-perfect on them, and the
    fitted temperature is meaningless (an earlier version of this script made
    exactly that mistake: it produced T*=0.379 and a source-val ECE of 0.0112,
    both artifacts of evaluating on memorized data).

    So we import train.py's own split_train_val and call it, rather than
    reimplementing it. train.py seeds the global RNG and truncates with int()
    where a reimplementation would naturally round, which is enough to produce a
    different split.
    """
    train_dir = cfg["train_dir"]
    if not Path(train_dir).is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. Cannot build a source "
            f"validation split, and falling back to the test set is exactly the "
            f"bug this script exists to fix."
        )

    try:
        from module3_training.train import (
            BrainTumorDataset as TrainDataset,
            split_train_val as train_split_train_val,
        )
        full = TrainDataset(str(train_dir), transform=None)
        _, val_subset = train_split_train_val(full)
        log.info("  source validation split: %d images (from train.py's own "
                 "split_train_val, so it matches what training held out)",
                 len(val_subset))
        split_source = "module3_training.train.split_train_val"
    except Exception as exc:
        log.error("  Could not import train.py's split_train_val: %s", exc)
        log.error("  REFUSING to substitute a locally generated split: it would "
                  "overlap the training data and the resulting T* would be an "
                  "artifact. Fix the import, or run with --mode target-kfold.")
        raise

    val_logits, val_labels = collect_logits(model, val_subset, device)
    t_star = optimise_temperature(val_logits, val_labels)
    log.info("  T* fitted on source validation = %.4f", t_star)

    return {
        "mode": "source-val",
        "n_val": int(len(val_subset)),
        "split_source": split_source,
        "T_star": round(t_star, 4),
        "val_before": {k: round(v, 4) for k, v in _metrics(val_logits, val_labels, 1.0).items()},
        "val_after": {k: round(v, 4) for k, v in _metrics(val_logits, val_labels, t_star).items()},
        "before": {k: round(v, 4) for k, v in _metrics(test_logits, test_labels, 1.0).items()},
        "after": {k: round(v, 4) for k, v in _metrics(test_logits, test_labels, t_star).items()},
        "_pooled_probs": F.softmax(test_logits / max(t_star, 0.01), dim=1).numpy(),
    }

def plot_reliability(probs_before, probs_after, labels, ece_before, ece_after,
                     t_label: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, probs, ece, title in [
        (axes[0], probs_before, ece_before, "Before scaling"),
        (axes[1], probs_after, ece_after, f"After scaling ({t_label})"),
    ]:
        preds = probs.argmax(axis=1)
        lowers, uppers, acc, conf, count = confidence_bins(probs, preds, labels, ECE_N_BINS)
        centers = (lowers + uppers) / 2
        mask = count > 0

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.bar(centers[mask], acc[mask], width=1.0 / ECE_N_BINS * 0.9,
               edgecolor="black", color="#4C72B0", alpha=0.85, label="Accuracy")
        ax.plot(conf[mask], acc[mask], "o-", color="#C44E52", markersize=5,
                label="Bin accuracy vs. confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10, weight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved reliability diagram: %s", out_path)

def run_protocol(name: str, mode: str, k: int, device) -> dict:
    cfg = protocol_config(name)
    log.info("=" * 68)
    log.info("TEMPERATURE SCALING - %s (%s)", name.upper(), cfg["description"])
    log.info("=" * 68)

    checkpoint = Path(MODELS_DIR) / name / "best_model.pth"
    model, ckpt = load_model_from_checkpoint(checkpoint, device)
    img_size = ckpt["img_size"]
    log.info("  checkpoint: %s (img_size=%d)", checkpoint, img_size)

    test_dir = cfg["test_dir"]
    classes = present_classes(test_dir)
    log.info("  test classes present: %s", classes)

    test_ds = BrainTumorDataset(test_dir, transform=build_eval_transform(img_size))
    test_logits, test_labels = collect_logits(model, test_ds, device)
    log.info("  test images: %d", len(test_labels))

    results: dict = {
        "protocol": name,
        "description": cfg["description"],
        "img_size": img_size,
        "n_test": int(len(test_labels)),
        "present_classes": classes,
    }

    modes = ["target-kfold", "source-val"] if mode == "both" else [mode]
    for m in modes:
        if m == "target-kfold":
            r = run_target_kfold(test_logits, test_labels, k)
        else:
            r = run_source_val(model, cfg, device, test_logits, test_labels, img_size)

        pooled = r.pop("_pooled_probs")
        results[m] = r

        out_dir = Path(RESULTS_DIR) / name
        suffix = "" if len(modes) == 1 else f"_{m}"
        t_label = (f"T*={r['T_star_mean']:.3f}+/-{r['T_star_std']:.3f}"
                   if m == "target-kfold" else f"T*={r['T_star']:.3f}")
        plot_reliability(
            F.softmax(test_logits, dim=1).numpy(), pooled, test_labels.numpy(),
            r["before"]["ece"], r["after"]["ece"], t_label,
            out_dir / f"temperature_scaling_reliability{suffix}.png",
        )
        log.info("  %s: ECE %.4f -> %.4f  (accuracy unchanged at %.4f)",
                 m, r["before"]["ece"], r["after"]["ece"], r["after"]["accuracy"])

    return results

def write_outputs(name: str, results: dict) -> None:
    out_dir = Path(RESULTS_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "temperature_scaling_summary.json").write_text(
        json.dumps(results, indent=2)
    )

    lines = [
        "=" * 70,
        f"  TEMPERATURE SCALING - {name.upper()}",
        f"  {results['description']}",
        "=" * 70,
        "",
        f"  Test images: {results['n_test']}   img_size: {results['img_size']}",
        f"  Classes present: {', '.join(results['present_classes'])}",
        "",
    ]

    if "target-kfold" in results:
        r = results["target-kfold"]
        lines += [
            f"  MODE: {r['k']}-fold cross-validated calibration on the target domain",
            "  Each image is evaluated under a temperature fitted WITHOUT it.",
            "",
            f"    T* = {r['T_star_mean']:.4f} +/- {r['T_star_std']:.4f} (mean +/- std across folds)",
            f"    ECE   {r['before']['ece']:.4f}  ->  {r['after']['ece']:.4f}",
            f"    NLL   {r['before']['nll']:.4f}  ->  {r['after']['nll']:.4f}",
            f"    Brier {r['before']['brier']:.4f}  ->  {r['after']['brier']:.4f}",
            f"    Accuracy unchanged: {r['after']['accuracy']*100:.2f}%",
            "",
            "    Per-fold:",
        ]
        for f in r["per_fold"]:
            lines.append(
                f"      fold {f['fold']}: fit n={f['n_fit']:4d}  eval n={f['n_eval']:4d}  "
                f"T*={f['T_star']:.4f}  ECE {f['ece_before']:.4f} -> {f['ece_after']:.4f}"
            )
        lines.append("")

    if "source-val" in results:
        r = results["source-val"]
        lines += [
            "  MODE: source-domain validation split (zero target labels used)",
            "",
            f"    Validation images: {r['n_val']}",
            f"    T* = {r['T_star']:.4f}",
            f"    Source val ECE  {r['val_before']['ece']:.4f}  ->  {r['val_after']['ece']:.4f}",
            f"    Target test ECE {r['before']['ece']:.4f}  ->  {r['after']['ece']:.4f}",
            f"    Accuracy unchanged: {r['after']['accuracy']*100:.2f}%",
            "",
        ]

    lines.append("=" * 70)
    report = "\n".join(lines)
    (out_dir / "temperature_scaling_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temperature scaling with honest calibration/evaluation separation"
    )
    p.add_argument("--protocol", default="lodo3",
                   choices=["all", "merged", "lodo1", "lodo2", "lodo3"])
    p.add_argument("--mode", default="target-kfold",
                   choices=["target-kfold", "source-val", "both"],
                   help="target-kfold (recommended): K-fold calibration on the "
                        "target domain. source-val: fit on an in-distribution "
                        "validation split, use no target labels.")
    p.add_argument("--k", type=int, default=5, help="Number of folds for target-kfold")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p

def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device()
    log.info("Device: %s | mode: %s", device, args.mode)

    for name in resolve_protocols(args.protocol):
        try:
            results = run_protocol(name, args.mode, args.k, device)
            write_outputs(name, results)
            write_run_metadata(
                Path(RESULTS_DIR) / name,
                config={"script": "temperature_scaling_fixed",
                        "mode": args.mode, "k": args.k, "seed": args.seed},
            )
        except Exception:
            log.exception("Protocol %s failed", name)
    return 0

cli_entrypoint(main)
