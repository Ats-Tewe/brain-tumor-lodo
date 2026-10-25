"""
module6_uncertainty/mc_dropout_convergence.py
==============================================
Justifies T=50 Monte Carlo Dropout passes empirically, so the choice needs no
citation.

WHY THIS EXISTS
---------------
The paper cites reference [26] twice for the T=50 choice. That reference is
wrong: the PDF filed there is Damiano et al. on kidney-MRI radiomics
reproducibility, which makes no recommendation about the number of MC passes for
brain MRI. No correct replacement was found.

Rather than hunt for a citation, measure it. This sweeps T over
{5, 10, 25, 50, 100, 200} and reports how the uncertainty estimates stabilize.
If the quantities the paper depends on have converged by T=50, the choice is
defended by your own data and cannot be questioned.

WHAT IT MEASURES
----------------
For each T, on the held-out test set:

  * mean predictive std over all images
  * mean std for correct vs incorrect predictions, and their ratio
    (the paper reports 4.3x at T=50)
  * ECE
  * number of images flagged above the 0.10 uncertainty threshold
  * accuracy of the MC-mean prediction

Convergence is quantified as the absolute change from the previous T and,
separately, against the T=200 reference. A quantity is called converged when
successive changes fall below a stated tolerance.

Efficiency note: the sweep runs 200 stochastic passes ONCE and computes every T
from prefixes of that sample, rather than re-running inference six times. The
T=50 row therefore uses the first 50 passes, which is exactly what a T=50 run
would have produced with the same seed.

USAGE
-----
    python module6_uncertainty/mc_dropout_convergence.py --protocol lodo3
    python module6_uncertainty/mc_dropout_convergence.py --protocol lodo3 \
        --checkpoint outputs/results/pretrained_hybrid/best_pretrained_hybrid_lodo3_seed42.pth

Outputs to outputs/results/<protocol>/:
    mc_dropout_convergence.json
    mc_dropout_convergence.txt
    mc_dropout_convergence.png     <- Figure for the paper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import (
    BATCH_SIZE, ECE_N_BINS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR,
    UNCERTAINTY_THRESHOLD, protocol_config,
)
from common.data import (
    BrainTumorDataset, build_eval_transform, make_generator, present_classes, seed_worker,
)
from common.experiment_utils import (
    cli_entrypoint, get_logger, resolve_device, set_seed, write_run_metadata,
)
from common.metrics import compute_ece
from common.model_adapter import load_any_checkpoint

log = get_logger(__name__)

T_VALUES = [5, 10, 25, 50, 100, 200]
T_MAX = max(T_VALUES)

@torch.no_grad()
def collect_mc_passes(model, loader, device, n_passes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_passes stochastic forward passes with MC Dropout active.

    Returns softmax probabilities of shape [n_passes, N, C] and labels [N].
    Labels are captured during the first pass only; the loader must not shuffle.
    """
    model.enable_mc_dropout()

    all_passes, labels = [], None
    for t in range(n_passes):
        probs, ys = [], []
        for images, y in loader:
            out = model(images.to(device), return_attention=False)
            probs.append(F.softmax(out["vit_logits"], dim=-1).cpu().numpy())
            if t == 0:
                ys.append(y.numpy())
        all_passes.append(np.concatenate(probs, axis=0))
        if t == 0:
            labels = np.concatenate(ys, axis=0)
        if (t + 1) % 25 == 0:
            log.info("    %d/%d passes", t + 1, n_passes)

    return np.stack(all_passes, axis=0), labels

def metrics_at_T(passes: np.ndarray, labels: np.ndarray, T: int) -> dict:
    """Compute the reported uncertainty quantities from the first T passes."""
    sub = passes[:T]
    mean_probs = sub.mean(axis=0)
    std_probs = sub.std(axis=0)
    max_std = std_probs.max(axis=1)

    preds = mean_probs.argmax(axis=1)
    correct = preds == labels

    n_flagged = int((max_std > UNCERTAINTY_THRESHOLD).sum())
    mean_correct = float(max_std[correct].mean()) if correct.any() else float("nan")
    mean_incorrect = float(max_std[~correct].mean()) if (~correct).any() else float("nan")
    ratio = mean_incorrect / mean_correct if mean_correct else float("nan")

    return {
        "T": T,
        "accuracy": round(float(correct.mean()), 4),
        "mean_std": round(float(max_std.mean()), 5),
        "mean_std_correct": round(mean_correct, 5),
        "mean_std_incorrect": round(mean_incorrect, 5),
        "incorrect_correct_ratio": round(ratio, 3),
        "ece": round(compute_ece(mean_probs, preds, labels, ECE_N_BINS), 5),
        "n_flagged": n_flagged,
        "flagged_rate": round(n_flagged / len(labels), 5),
    }

def convergence_table(rows: list[dict]) -> list[dict]:
    """Absolute change from the previous T, and deviation from the T_MAX reference."""
    ref = rows[-1]
    tracked = ["mean_std", "incorrect_correct_ratio", "ece", "accuracy"]
    out = []
    for i, r in enumerate(rows):
        entry = {"T": r["T"]}
        for k in tracked:
            prev = rows[i - 1][k] if i > 0 else None
            entry[f"{k}_delta_prev"] = (round(abs(r[k] - prev), 5)
                                        if prev is not None else None)
            entry[f"{k}_delta_ref"] = round(abs(r[k] - ref[k]), 5)
        out.append(entry)
    return out

def plot_convergence(rows: list[dict], protocol: str, out_path: Path):
    Ts = [r["T"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(Ts, [r["mean_std"] for r in rows], "o-", color="#4C72B0", lw=2)
    axes[0].set_ylabel("Mean predictive std")
    axes[0].set_title("Uncertainty magnitude")

    axes[1].plot(Ts, [r["incorrect_correct_ratio"] for r in rows], "o-",
                 color="#C44E52", lw=2)
    axes[1].set_ylabel("std(incorrect) / std(correct)")
    axes[1].set_title("Error-detection ratio")

    axes[2].plot(Ts, [r["ece"] for r in rows], "o-", color="#55A868", lw=2)
    axes[2].set_ylabel("ECE")
    axes[2].set_title("Calibration error")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("MC Dropout passes (T)")
        ax.axvline(50, color="gray", ls="--", lw=1.2, alpha=0.8)
        ax.grid(alpha=0.3)
    axes[0].text(52, axes[0].get_ylim()[1]*0.95, "T=50", fontsize=8, color="gray")

    fig.suptitle(f"MC Dropout convergence - {protocol}", fontsize=12, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure: %s", out_path)

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default="lodo3",
                   choices=["merged", "lodo1", "lodo2", "lodo3"])
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint to use. Defaults to outputs/models/<protocol>/best_model.pth")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--tolerance", type=float, default=0.002,
                   help="A quantity counts as converged when successive changes "
                        "fall below this. Default 0.002.")
    return p

def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device()

    cfg = protocol_config(args.protocol)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        Path(MODELS_DIR) / args.protocol / "best_model.pth"

    log.info("=" * 70)
    log.info("MC DROPOUT CONVERGENCE - %s", args.protocol.upper())
    log.info("=" * 70)

    model, _ckpt, meta = load_any_checkpoint(ckpt_path, device)
    log.info("  checkpoint: %s", ckpt_path)
    log.info("  model kind: %s | img_size %d | %.2fM params",
             meta["kind"], meta["img_size"], meta["total_params"] / 1e6)

    ds = BrainTumorDataset(cfg["test_dir"], transform=build_eval_transform(meta["img_size"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                        pin_memory=True, worker_init_fn=seed_worker,
                        generator=make_generator(args.seed))
    log.info("  test images: %d | classes: %s", len(ds), present_classes(cfg["test_dir"]))

    log.info("  running %d stochastic passes once, then evaluating every T from prefixes",
             T_MAX)
    passes, labels = collect_mc_passes(model, loader, device, T_MAX)

    rows = [metrics_at_T(passes, labels, T) for T in T_VALUES]
    conv = convergence_table(rows)

    lines = [
        "=" * 78,
        f"  MC DROPOUT CONVERGENCE - {args.protocol.upper()}",
        "=" * 78,
        "",
        f"  Checkpoint : {ckpt_path.name}",
        f"  Model      : {meta['kind']}, {meta['img_size']}px",
        f"  Test images: {len(labels)}",
        f"  Seed       : {args.seed}",
        "",
        f"  {'T':>5} {'accuracy':>10} {'mean std':>10} {'ratio':>8} {'ECE':>8} {'flagged':>9}",
        "  " + "-" * 56,
    ]
    for r in rows:
        lines.append(f"  {r['T']:>5} {r['accuracy']*100:>9.2f}% {r['mean_std']:>10.5f} "
                     f"{r['incorrect_correct_ratio']:>8.2f} {r['ece']:>8.4f} "
                     f"{r['n_flagged']:>4}/{len(labels)}")

    lines += ["", "  Change from the previous T:", "",
              f"  {'T':>5} {'d(mean std)':>13} {'d(ratio)':>10} {'d(ECE)':>10}",
              "  " + "-" * 42]
    for c in conv:
        f = lambda v: f"{v:.5f}" if v is not None else "    -"
        lines.append(f"  {c['T']:>5} {f(c['mean_std_delta_prev']):>13} "
                     f"{f(c['incorrect_correct_ratio_delta_prev']):>10} "
                     f"{f(c['ece_delta_prev']):>10}")

    at50 = next(c for c in conv if c["T"] == 50)
    converged = (at50["mean_std_delta_prev"] is not None
                 and at50["mean_std_delta_prev"] < args.tolerance
                 and at50["ece_delta_prev"] < args.tolerance)
    r50 = next(r for r in rows if r["T"] == 50)
    rmax = rows[-1]

    lines += [
        "",
        f"  At T=50, deviation from the T={T_MAX} reference:",
        f"    mean std   {at50['mean_std_delta_ref']:.5f}",
        f"    ratio      {at50['incorrect_correct_ratio_delta_ref']:.3f}",
        f"    ECE        {at50['ece_delta_ref']:.5f}",
        f"    accuracy   {at50['accuracy_delta_ref']:.5f}",
        "",
        f"  T=50  : ratio {r50['incorrect_correct_ratio']:.2f}x, ECE {r50['ece']:.4f}, "
        f"{r50['n_flagged']} flagged",
        f"  T={T_MAX} : ratio {rmax['incorrect_correct_ratio']:.2f}x, ECE {rmax['ece']:.4f}, "
        f"{rmax['n_flagged']} flagged",
        "",
        ("  CONVERGED by T=50 (successive changes below "
         f"{args.tolerance}). The choice of T=50 is justified by this "
         "measurement and requires no citation.")
        if converged else
        ("  NOT converged by T=50 at the stated tolerance. Report the sweep and "
         "either raise T or state the residual sensitivity explicitly."),
        "=" * 78,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    out_dir = Path(RESULTS_DIR) / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mc_dropout_convergence.txt").write_text(report, encoding="utf-8")
    (out_dir / "mc_dropout_convergence.json").write_text(json.dumps({
        "protocol": args.protocol, "checkpoint": str(ckpt_path),
        "model": meta, "seed": args.seed, "n_test": int(len(labels)),
        "rows": rows, "convergence": conv, "converged_by_T50": bool(converged),
    }, indent=2))
    plot_convergence(rows, args.protocol, out_dir / "mc_dropout_convergence.png")
    write_run_metadata(out_dir, config=vars(args))
    log.info("  saved: %s", out_dir)
    return 0

cli_entrypoint(main)
