#!/usr/bin/env bash
set -euo pipefail
VENV=/workspace/host/lfm/venvs/sayso-trainer
BASE=/opt/unsloth-venv/lib/python3.12/site-packages
SITE=$VENV/lib/python3.12/site-packages
docker exec unsloth "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || docker exec unsloth /usr/bin/python3.12 -m venv --clear "$VENV"
docker exec unsloth sh -c 'printf "%s\n" "$1" > "$2"' sh "$BASE" "$SITE/zz-unsloth-venv-base.pth"
docker exec unsloth "$VENV/bin/python" -m pip install trl==1.18.1 transformers==5.23.0 datasets==5.2.0 tokenizers==0.23.3
docker exec unsloth "$VENV/bin/python" -c '
import datasets, torch, trl, transformers
for mod in (trl, transformers, datasets, torch):
    print(mod.__name__, mod.__version__, mod.__file__)
'
