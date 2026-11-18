#!/usr/bin/env bash
set -euo pipefail
ROOT=${SAYSO_TRAINING_HOST_ROOT:-/workspace/host/lfm/ajax-optimization/sayso-training}
VENV=/workspace/host/lfm/venvs/sayso-trainer
BASE=/opt/unsloth-venv/lib/python3.12/site-packages
SITE=$VENV/lib/python3.12/site-packages
docker exec sayso-trainer "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || docker exec sayso-trainer /usr/bin/python3.12 -m venv --clear "$VENV"
docker exec sayso-trainer sh -c 'printf "%s\n" "$1" > "$2"' sh "$BASE" "$SITE/zz-unsloth-venv-base.pth"
docker exec sayso-trainer "$VENV/bin/python" -m pip install -r "$ROOT/training/requirements-sayso-trainer.txt"
docker exec sayso-trainer "$VENV/bin/python" -c '
import sys
sys.path.insert(0, sys.argv[1])
import sayso_trainer
sayso_trainer.check_installed_stack()
from trl import SFTConfig, SFTTrainer
import datasets, tokenizers, torch, trl, transformers
for mod in (trl, transformers, datasets, tokenizers, torch):
    print(mod.__name__, mod.__version__, mod.__file__)
args = sayso_trainer.parse_args(["--canary", "--out", "/tmp/sayso-trainer-smoke"])
SFTConfig(bf16=False, max_length=args.cutoff, packing=False, padding_free=False, dataset_kwargs={"skip_prepare_dataset": True}, gradient_checkpointing=True, **sayso_trainer.sft_recipe(args, evaluating=True))
' "$ROOT/training/scripts"
