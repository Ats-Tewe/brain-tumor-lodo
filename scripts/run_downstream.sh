#!/usr/bin/env bash

set -u

cd "$(dirname "$0")" || exit 1
source venv/bin/activate

unset BTP_OUTPUTS_DIR

PROTOS="merged lodo1 lodo2 lodo3"

snapshot () {
    dest="$1"
    mkdir -p "$dest"
    for p in $PROTOS; do
        if [ -d "outputs/results/$p" ]; then
            cp -a "outputs/results/$p" "$dest/"
        fi
    done
    echo "  snapshot -> $dest"
}

banner () {
    echo
    echo "########################################################################"
    echo "##  $1"
    echo "########################################################################"
}

banner "STAGE 1/5  evaluate.py  (plain, no TTA)"
python run_with_adapter.py module4_evaluation/evaluate.py --protocol all 2>&1 | tee p_eval_plain.log
snapshot outputs_pretrained/eval_plain

banner "STAGE 2/5  evaluate.py  (5-view TTA)"
python run_with_adapter.py module4_evaluation/evaluate.py --protocol all --tta 2>&1 | tee p_eval_tta.log
snapshot outputs_pretrained/eval_tta

banner "STAGE 3/5  uncertainty.py  (MC Dropout, T=50)"
python run_with_adapter.py module6_uncertainty/uncertainty.py --protocol all --seed 42 2>&1 | tee p_unc.log

banner "STAGE 4/5  bootstrap_ci.py  (B=2000)"
python run_with_adapter.py module6_uncertainty/bootstrap_ci.py --seed 42 2>&1 | tee p_boot.log

banner "STAGE 5/5  explain.py  (Grad-CAM + Attention Rollout)"
python run_with_adapter.py module5_explainability/explain.py --protocol all 2>&1 | tee p_explain.log

banner "SUMMARY"
echo "Traceback counts per stage (any non-zero means that stage is NOT usable):"
for f in p_eval_plain p_eval_tta p_unc p_boot p_explain; do
    n=$(grep -c Traceback "$f.log" 2>/dev/null || echo missing)
    m=$(grep -c ERROR "$f.log" 2>/dev/null || echo 0)
    printf '  %-16s tracebacks=%-6s errors=%s\n' "$f" "$n" "$m"
done
echo
echo "Models built through the patched constructor (expect 4 per eval stage):"
grep -h models.built p_eval_plain.log p_eval_tta.log p_unc.log p_boot.log p_explain.log 2>/dev/null
echo
echo ALLDONE
