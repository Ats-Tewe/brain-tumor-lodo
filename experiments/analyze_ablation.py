from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "outputs" / "results" / "ablation_vit" / "ablation_results.json"
PROTOCOL_ORDER = ["merged", "lodo1", "lodo2", "lodo3"]

def hybrid_accuracy(protocol: str):
    """ViT primary-head accuracy for the hybrid, from the evaluation outputs."""
    path = ROOT / "outputs" / "results" / protocol / "test_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())["vit"]["accuracy"]

def main() -> int:
    if not ABLATION.is_file():
        print(f"Not found: {ABLATION}")
        print("Copy outputs/results/ablation_vit back from the server first.")
        return 1

    results = json.loads(ABLATION.read_text())

    n_expected = 8
    if len(results) < n_expected:
        print(f"WARNING: found {len(results)} runs, expected {n_expected}.")
        print("This file may be the 2-epoch smoke test rather than the full run.")
        print("Check `grep -c 'TEST accuracy' ablation_log.txt` on the server "
              "reads 8 before trusting these numbers.\n")

    by_key = {(r["protocol"], r["variant"]): r for r in results}

    print("=" * 78)
    print("  ABLATION RESULTS — standalone ViT vs. the DualHead hybrid")
    print("=" * 78)
    header = (f"  {'protocol':<9}{'hybrid':>9}{'matched':>10}{'gap':>9}"
              f"{'unmatched':>11}{'gap':>9}{'tokens m/u':>13}")
    print(header)
    print("  " + "-" * 74)

    for proto in PROTOCOL_ORDER:
        hy = hybrid_accuracy(proto)
        m = by_key.get((proto, "matched"))
        u = by_key.get((proto, "unmatched"))

        hy_s = f"{hy*100:8.2f}%" if hy is not None else "      n/a"
        cells = [f"  {proto:<9}{hy_s}"]

        for r in (m, u):
            if r is None:
                cells.append(f"{'n/a':>10}{'':>9}")
                continue
            acc = r["test"]["accuracy"]
            cells.append(f"{acc*100:9.2f}%")
            if hy is not None:
                cells.append(f"{(acc - hy)*100:+8.2f}")
            else:
                cells.append(f"{'':>9}")

        tok_m = m["architecture"]["n_tokens"] if m else "-"
        tok_u = u["architecture"]["n_tokens"] if u else "-"
        cells.append(f"{f'{tok_m}/{tok_u}':>13}")
        print("".join(cells))

    print("  " + "-" * 74)
    print("  gap = standalone minus hybrid, in percentage points.")
    print("  Positive means the standalone ViT is more accurate, i.e. the hybrid")
    print("  paid that much for its guaranteed explanation pathway.")

    hy3 = hybrid_accuracy("lodo3")
    m3 = by_key.get(("lodo3", "matched"))
    u3 = by_key.get(("lodo3", "unmatched"))

    print("\n" + "=" * 78)
    print("  FOR CONTRIBUTION C6 (LODO R3, maximum shift)")
    print("=" * 78)

    if hy3 is None or m3 is None:
        print("  Missing lodo3 hybrid or matched-variant result.")
        return 1

    gap_m = (m3["test"]["accuracy"] - hy3) * 100
    print(f"  Hybrid:                      {hy3*100:.2f}%")
    print(f"  Standalone ViT (matched):    {m3['test']['accuracy']*100:.2f}%   "
          f"({m3['architecture']['n_tokens']} tokens)")
    print(f"  --> controlled gap:          {gap_m:+.2f} pp   <-- cite THIS in C6")

    if u3 is not None:
        gap_u = (u3["test"]["accuracy"] - hy3) * 100
        print(f"\n  Standalone ViT (unmatched):  {u3['test']['accuracy']*100:.2f}%   "
              f"({u3['architecture']['n_tokens']} tokens)")
        print(f"      uncontrolled gap:        {gap_u:+.2f} pp")
        print(f"      capacity contribution:   {gap_u - gap_m:+.2f} pp "
              f"(the part attributable to 4x the tokens, not the architecture)")
        print(f"\n  Original paper claimed 5.39 pp. Of that, roughly "
              f"{gap_u - gap_m:+.2f} pp was token budget.")

    print()
    return 0
