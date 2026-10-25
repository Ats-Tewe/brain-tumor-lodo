"""
aggregate_seeds.py
==================
Turns the multi-seed campaign into the numbers the paper reports.

WHY THIS MATTERS
----------------
Five runs of one fixed configuration on LODO R3 produced 78.43, 81.74, 83.33,
88.85 and 91.42 percent. Standard deviation 5.30 pp, range 13 pp. Any single
run therefore carries almost no information about a configuration's quality,
and the difference between two architectures reported from one run each is
uninterpretable unless it exceeds roughly 10 pp.

So every accuracy in the paper is reported as mean +/- standard deviation over
seeds, with n stated. This script produces those numbers and the LaTeX table.

USAGE
-----
    python aggregate_seeds.py
    python aggregate_seeds.py --results-dir outputs/results/pretrained_hybrid
    python aggregate_seeds.py --latex          # emit a LaTeX table body

Reads every pretrained_hybrid_seed*.json in the results directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

PROTOCOL_ORDER = ["merged", "lodo1", "lodo2", "lodo3"]
PROTOCOL_LABEL = {"merged": "Merged", "lodo1": "LODO R1",
                  "lodo2": "LODO R2", "lodo3": "LODO R3"}

SCRATCH_PUBLISHED = {"merged": 0.9927, "lodo1": 0.9513,
                     "lodo2": 0.9925, "lodo3": 0.8946}
SCRATCH_LODO3_RUNS = [0.7843, 0.8174, 0.8333, 0.8885, 0.9142]

def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """Collect every run, grouped by protocol."""
    by_protocol: dict[str, list[dict]] = {}
    files = sorted(glob.glob(str(results_dir / "pretrained_hybrid_seed*.json")))
    if not files:
        raise SystemExit(
            f"No pretrained_hybrid_seed*.json found in {results_dir}.\n"
            f"Copy outputs/results/pretrained_hybrid back from the server first."
        )
    for f in files:
        payload = json.loads(Path(f).read_text())
        for r in payload.get("runs", []):
            by_protocol.setdefault(r["protocol"], []).append(r)
    return by_protocol

def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = st.mean(values)
    sd = st.stdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd,
            "min": min(values), "max": max(values),
            "sem": sd / (n ** 0.5) if n else 0.0}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="outputs/results/pretrained_hybrid")
    ap.add_argument("--latex", action="store_true", help="Emit a LaTeX table body")
    args = ap.parse_args()

    by_protocol = load_runs(Path(args.results_dir))

    print("=" * 78)
    print("  MULTI-SEED RESULTS - pretrained dual-head hybrid")
    print("=" * 78)
    print(f"  {'protocol':<10}{'n':>3}{'mean':>10}{'sd':>8}{'min':>9}{'max':>9}"
          f"{'range':>8}   single-run (old)")
    print("  " + "-" * 74)

    summary = {}
    for proto in PROTOCOL_ORDER:
        runs = by_protocol.get(proto, [])
        if not runs:
            print(f"  {proto:<10}  no runs")
            continue
        accs = [r["test"]["vit_accuracy"] for r in runs]
        s = summarize(accs)
        summary[proto] = s
        old = SCRATCH_PUBLISHED.get(proto)
        old_s = f"{old*100:.2f}%" if old else "-"
        print(f"  {PROTOCOL_LABEL[proto]:<10}{s['n']:>3}{s['mean']*100:>9.2f}%"
              f"{s['sd']*100:>8.2f}{s['min']*100:>8.2f}%{s['max']*100:>8.2f}%"
              f"{(s['max']-s['min'])*100:>8.2f}   {old_s}")

    if "merged" in summary and "lodo3" in summary:
        m, l = summary["merged"], summary["lodo3"]
        gap = (m["mean"] - l["mean"]) * 100
        gap_sd = ((m["sd"] ** 2) / m["n"] + (l["sd"] ** 2) / l["n"]) ** 0.5 * 100
        print("\n" + "=" * 78)
        print("  CROSS-INSTITUTIONAL GENERALIZATION GAP")
        print("=" * 78)
        print(f"  Merged  {m['mean']*100:.2f}% +/- {m['sd']*100:.2f}  (n={m['n']})")
        print(f"  LODO R3 {l['mean']*100:.2f}% +/- {l['sd']*100:.2f}  (n={l['n']})")
        print(f"  gap     {gap:.2f} pp +/- {gap_sd:.2f}")
        print(f"\n  The paper currently claims 9.81 pp from single runs.")

    if "lodo3" in summary:
        p = summary["lodo3"]
        sc = summarize(SCRATCH_LODO3_RUNS)
        print("\n" + "=" * 78)
        print("  PRETRAINING EFFECT (LODO R3)")
        print("=" * 78)
        print(f"  {'scratch transformer':<24}{sc['mean']*100:7.2f}% +/- {sc['sd']*100:.2f}"
              f"   (n={sc['n']}, range {(sc['max']-sc['min'])*100:.2f} pp)")
        print(f"  {'pretrained transformer':<24}{p['mean']*100:7.2f}% +/- {p['sd']*100:.2f}"
              f"   (n={p['n']}, range {(p['max']-p['min'])*100:.2f} pp)")
        print(f"\n  accuracy gain        {(p['mean']-sc['mean'])*100:+.2f} pp")
        if p["sd"] > 0:
            print(f"  variance reduction   {sc['sd']/p['sd']:.2f}x")
        pooled = (((sc['n']-1)*sc['sd']**2 + (p['n']-1)*p['sd']**2)
                  / max(sc['n']+p['n']-2, 1)) ** 0.5
        if pooled:
            print(f"  effect size (Cohen d) {(p['mean']-sc['mean'])/pooled:.2f}")
        print("\n  Reference: standalone ViT-S pretrained 95.47%, random init 68.50%")
        print("  (single runs; the 26.97 pp difference is ~5x the noise floor)")

    if args.latex:
        print("\n" + "=" * 78)
        print("  LATEX TABLE BODY")
        print("=" * 78)
        print("% Accuracy as mean $\\pm$ standard deviation over independent")
        print("% training runs differing only in random seed.")
        for proto in PROTOCOL_ORDER:
            if proto not in summary:
                continue
            s = summary[proto]
            print(f"{PROTOCOL_LABEL[proto]} & {s['n']} & "
                  f"${s['mean']*100:.2f} \\pm {s['sd']*100:.2f}$ & "
                  f"[{s['min']*100:.2f}, {s['max']*100:.2f}] \\\\")

    out = {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in summary.items()}
    Path(args.results_dir, "multiseed_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {Path(args.results_dir, 'multiseed_summary.json')}")
    return 0
