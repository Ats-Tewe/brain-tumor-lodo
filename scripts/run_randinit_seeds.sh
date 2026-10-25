#!/usr/bin/env bash

set -u

cd "$(dirname "$0")" || exit 1
source venv/bin/activate
unset BTP_OUTPUTS_DIR

ARCH=outputs/results/architecture_experiments
STORE="$ARCH/seed_runs"
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$STORE"

echo "======================================================================"
echo "  RANDOM-TRANSFORMER HYBRID, LODO R3, seeds 2024 and 31337"
echo "  (EfficientNet-B0 stem is ImageNet-initialized in this arm)"
echo "======================================================================"

if [ -f "$ARCH/architecture_experiments_lodo3.json" ]; then
    cp -a "$ARCH/architecture_experiments_lodo3.json" \
          "$STORE/lodo3_preexisting_$STAMP.json"
    echo "  preserved existing lodo3 json -> $STORE/lodo3_preexisting_$STAMP.json"
else
    echo "  no existing lodo3 json to preserve"
fi
echo

for S in 2024 31337; do
    echo "----------------------------------------------------------------------"
    echo "  SEED $S"
    echo "----------------------------------------------------------------------"

    python hybrid_architecture_experiments.py \
        --experiment depth-sweep \
        --depths 12 \
        --protocol lodo3 \
        --seed "$S" 2>&1 | tee "randinit_seed${S}.log"

    if [ -f "$ARCH/architecture_experiments_lodo3.json" ]; then
        cp -a "$ARCH/architecture_experiments_lodo3.json" "$STORE/lodo3_seed${S}.json"
        echo "  saved -> $STORE/lodo3_seed${S}.json"
    else
        echo "  WARNING: no result json produced for seed $S"
    fi
    echo
done

echo "======================================================================"
echo "  COLLECTED RESULTS"
echo "======================================================================"
echo "  from seed_check.log (seeds 42, 123, 777):"
grep -h -e "TEST vit=" seed_check.log 2>/dev/null | sed 's/^/    /'
echo
echo "  from this run:"
for f in "$STORE"/lodo3_seed*.json; do
    [ -f "$f" ] || continue
    echo "    $f"
    grep -h -e vit_accuracy -e '"seed"' "$f" | sed 's/^/      /'
done
echo
echo "  Feed all five accuracies into the mean, s.d., F-test and Welch test."
echo ALLDONE
