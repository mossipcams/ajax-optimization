#!/usr/bin/env bash
set -euo pipefail
VENV=/workspace/host/lfm/venvs/sayso-trainer
BASE=/opt/unsloth-venv/lib/python3.12/site-packages
SITE=$VENV/lib/python3.12/site-packages
docker exec unsloth "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || docker exec unsloth /usr/bin/python3.12 -m venv --clear "$VENV"
docker exec unsloth sh -c 'printf "%s\n" "$1" > "$2"' sh "$BASE" "$SITE/zz-unsloth-venv-base.pth"
docker exec unsloth "$VENV/bin/python" -m pip install trl==1.18.1 transformers==5.23.0 datasets==5.2.0 tokenizers==0.23.3
docker exec unsloth "$VENV/bin/python" -c '
from trl import SFTConfig, SFTTrainer
import datasets, tokenizers, torch, trl, transformers
for mod in (trl, transformers, datasets, tokenizers, torch):
    print(mod.__name__, mod.__version__, mod.__file__)
SFTConfig(output_dir="/tmp/sayso-trainer-smoke", bf16=False, max_length=8192, packing=False, padding_free=False, dataset_kwargs={"skip_prepare_dataset": True}, warmup_steps=0.1, train_sampling_strategy="group_by_length", optim="adamw_torch", remove_unused_columns=False, save_only_model=True, report_to="none")
'
