#!/usr/bin/env bash

set -u
PROJECT=/home/divclab/brain_tumor_project
cd "$PROJECT" || { echo "!! cannot cd to $PROJECT"; exit 1; }

echo "======================================================================"
echo "  looking for a Python with torch"
echo "======================================================================"

CANDIDATES=(
  "$PROJECT/venv/bin/python"
  "$HOME/venv/bin/python"
  "$HOME/miniconda3/bin/python"
  "$HOME/anaconda3/bin/python"
  "$HOME/miniforge3/bin/python"
  /usr/bin/python3
)
for d in "$HOME"/miniconda3/envs/*/bin/python \
         "$HOME"/anaconda3/envs/*/bin/python \
         "$HOME"/miniforge3/envs/*/bin/python \
         "$HOME"/.conda/envs/*/bin/python; do
  [ -x "$d" ] && CANDIDATES+=("$d")
done

PY=""
for p in "${CANDIDATES[@]}"; do
  [ -x "$p" ] || continue
  if "$p" -c "import torch" >/dev/null 2>&1; then
    ver=$("$p" -c "import torch,sys; print(torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null)
    echo "  torch OK : $p   ($ver)"
    [ -z "$PY" ] && PY="$p"
  else
    echo "  no torch : $p"
  fi
done

if [ -z "$PY" ]; then
  echo
  echo "!! No interpreter with torch found in the usual places."
  echo "   Run this to see what the training runs used, then tell me the path:"
  echo "       ls -d \$HOME/*/bin/python \$HOME/*/envs/*/bin/python 2>/dev/null"
  echo "       conda env list"
  exit 2
fi

echo
echo "  using   : $PY"
echo "======================================================================"
echo

"$PY" complexity_vit_s32.py "$@"
status=$?

echo
if [ $status -ne 0 ]; then
  echo "!! complexity_vit_s32.py exited with status $status"
fi
exit $status
