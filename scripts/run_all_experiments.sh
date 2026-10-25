#!/usr/bin/env bash

set -u

PROJECT_DIR="${PROJECT_DIR:-$HOME/brain_tumor_project}"
VENV="${VENV:-$PROJECT_DIR/venv}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_DIR/logs/$STAMP"

cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p "$LOG_DIR"

if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    conda deactivate 2>/dev/null || true
fi

source "$VENV/bin/activate" || { echo "Cannot activate $VENV"; exit 1; }

echo "======================================================================"
echo "  OVERNIGHT EXPERIMENT RUN"
echo "  started : $(date)"
echo "  project : $PROJECT_DIR"
echo "  python  : $(which python)"
echo "  logs    : $LOG_DIR"
echo "======================================================================"

python - <<'PY'
import torch
print(f"  torch {torch.__version__} | cuda {torch.cuda.is_available()}", end="")
print(f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " | NO GPU")
PY
echo

STEP_NAMES=()
STEP_STATUS=()
STEP_TIMES=()

run_step () {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    local start end elapsed rc

    echo "----------------------------------------------------------------------"
    echo "  [$(date +%H:%M:%S)] START  $name"
    echo "    -> $logfile"
    echo "----------------------------------------------------------------------"

    start=$(date +%s)
    "$@" > "$logfile" 2>&1 < /dev/null
    rc=$?
    end=$(date +%s)
    elapsed=$(( (end - start) / 60 ))

    if [ $rc -eq 0 ]; then
        echo "  [$(date +%H:%M:%S)] OK     $name  (${elapsed} min)"
    else
        echo "  [$(date +%H:%M:%S)] FAILED $name  (rc=$rc, ${elapsed} min)"
        echo "    last 15 lines:"
        tail -15 "$logfile" | sed 's/^/      /'
    fi
    echo

    STEP_NAMES+=("$name")
    STEP_STATUS+=("$rc")
    STEP_TIMES+=("$elapsed")
}

run_step "01_verify_stopgrad" \
    python hybrid_architecture_experiments.py \
        --experiment verify-stopgrad --protocol lodo3

run_step "02_ab_ablation_and_depth_sweep" \
    python hybrid_architecture_experiments.py \
        --experiment all --protocol lodo3 --depths 4 6 8 12

run_step "03_standalone_vit_scratch" \
    python ablation_vit_standalone_fixed.py \
        --protocol lodo3 --variant matched --no-pretrained

BEST_DEPTH=$(python - <<'PY' 2>/dev/null || echo 6
import json, pathlib
p = pathlib.Path("outputs/results/architecture_experiments/"
                 "architecture_experiments_lodo3.json")
rows = [r for r in json.loads(p.read_text())["runs"]
        if r["label"].startswith("depth")]
print(max(rows, key=lambda r: r["test"]["vit_accuracy"])["vit_num_layers"]
      if rows else 6)
PY
)
BEST_DEPTH="$(printf '%s' "$BEST_DEPTH" | tr -d '[:space:]')"
if ! [[ "$BEST_DEPTH" =~ ^[0-9]+$ ]] || [ "$BEST_DEPTH" -lt 1 ] || [ "$BEST_DEPTH" -gt 24 ]; then
    echo "  Could not read a valid depth from step 2 (got: '${BEST_DEPTH}'). Using 6."
    BEST_DEPTH=6
fi
echo "  Best transformer depth from step 2: $BEST_DEPTH"
echo

for proto in merged lodo1 lodo2; do
    run_step "04_depth${BEST_DEPTH}_${proto}" \
        python hybrid_architecture_experiments.py \
            --experiment depth-sweep --protocol "$proto" --depths "$BEST_DEPTH"
done

SUMMARY="$LOG_DIR/SUMMARY.txt"
{
    echo "======================================================================"
    echo "  OVERNIGHT RUN SUMMARY"
    echo "  finished: $(date)"
    echo "======================================================================"
    echo
    printf "  %-34s %-8s %s\n" "STEP" "STATUS" "MINUTES"
    printf "  %-34s %-8s %s\n" "----" "------" "-------"
    total=0
    failed=0
    for i in "${!STEP_NAMES[@]}"; do
        if [ "${STEP_STATUS[$i]}" -eq 0 ]; then st="OK"; else st="FAILED"; failed=$((failed+1)); fi
        printf "  %-34s %-8s %s\n" "${STEP_NAMES[$i]}" "$st" "${STEP_TIMES[$i]}"
        total=$((total + STEP_TIMES[$i]))
    done
    echo
    echo "  total: ${total} min   failures: ${failed}"
    echo
    echo "  Best transformer depth (step 2): ${BEST_DEPTH}"
    echo
    echo "  Key result files:"
    echo "    outputs/results/architecture_experiments/architecture_experiments_*.json"
    echo "    outputs/results/architecture_experiments/depth_sweep_lodo3.png"
    echo "    outputs/results/ablation_vit/ablation_results.json"
    echo
    echo "  Per-step logs: $LOG_DIR"
} | tee "$SUMMARY"

python - <<'PY' 2>/dev/null | tee -a "$SUMMARY"
import json, pathlib
p = pathlib.Path("outputs/results/architecture_experiments/"
                 "architecture_experiments_lodo3.json")
if not p.is_file():
    raise SystemExit
d = json.loads(p.read_text())
runs = d["runs"]

def get(label):
    return next((r for r in runs if r["label"] == label), None)

print("=" * 70)
print("  HEADLINE NUMBERS (lodo3)")
print("=" * 70)

sg = d.get("stopgradient_verification") or {}
if sg:
    print(f"  stop-gradient confirmed: {sg.get('stopgradient_confirmed')}")

a, b = get("ab_with_cnn_head"), get("ab_no_cnn_head")
if a and b:
    cost = (b["test"]["vit_accuracy"] - a["test"]["vit_accuracy"]) * 100
    print(f"  explanation-head cost:   {cost:+.2f} pp   (paper claims 5.39 pp)")

rows = sorted([r for r in runs if r["label"].startswith("depth")],
              key=lambda r: r["vit_num_layers"])
if rows:
    print("\n  depth   params(M)   test acc")
    for r in rows:
        print(f"  {r['vit_num_layers']:<7} {r['total_params']/1e6:<11.2f} "
              f"{r['test']['vit_accuracy']:.4f}")
    best = max(rows, key=lambda r: r["test"]["vit_accuracy"])
    print(f"\n  best depth {best['vit_num_layers']}: "
          f"{best['test']['vit_accuracy']:.4f}")
    print(f"  pretrained standalone ViT-S reached 0.9547")
    print(f"  remaining gap: "
          f"{(0.9547 - best['test']['vit_accuracy'])*100:+.2f} pp")
PY

touch "$PROJECT_DIR/EXPERIMENTS_DONE"
echo
echo "  Wrote marker: $PROJECT_DIR/EXPERIMENTS_DONE"
echo "  Done at $(date)"
