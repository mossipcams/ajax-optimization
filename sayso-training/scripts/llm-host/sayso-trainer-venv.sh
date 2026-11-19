#!/usr/bin/env bash
set -euo pipefail
ROOT=${SAYSO_TRAINING_HOST_ROOT:-/workspace/host/lfm/ajax-optimization/sayso-training}
VENV=/workspace/host/lfm/venvs/sayso-trainer
BASE=/opt/unsloth-venv/lib/python3.12/site-packages
SITE=$VENV/lib/python3.12/site-packages
CAUSAL_CONV1D_URL=https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/causal_conv1d-1.7.0.tar.gz
CAUSAL_CONV1D_SHA256=3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b
docker exec sayso-trainer "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || docker exec sayso-trainer /usr/bin/python3.12 -m venv --clear "$VENV"
docker exec sayso-trainer sh -c 'printf "%s\n" "$1" > "$2"' sh "$BASE" "$SITE/zz-unsloth-venv-base.pth"
if ! docker exec sayso-trainer "$VENV/bin/python" -c 'import importlib.metadata as m, causal_conv1d; assert m.version("causal-conv1d") == "1.7.0"' >/dev/null 2>&1; then
  docker exec sayso-trainer sh -eu -c '
    [ -f /opt/rocm/include/hipcub/hipcub.hpp ] || { apt-get update && apt-get install -y --no-install-recommends hipcub-dev rocprim-dev; }
    d=$(mktemp -d)
    cd "$d"
    "$2/bin/python" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$1" causal_conv1d-1.7.0.tar.gz
    echo "$3  causal_conv1d-1.7.0.tar.gz" | sha256sum -c -
    tar -xzf causal_conv1d-1.7.0.tar.gz
    sed -i "s/-std=c++17/-std=c++20/" causal_conv1d-1.7.0/setup.py
    grep -q -- "-std=c++20" causal_conv1d-1.7.0/setup.py
    CAUSAL_CONV1D_FORCE_BUILD=TRUE HIP_ARCHITECTURES=gfx1100 "$2/bin/python" -m pip install --no-build-isolation --no-deps "$d/causal_conv1d-1.7.0"
    cd /
    rm -rf "$d"
  ' sh "$CAUSAL_CONV1D_URL" "$VENV" "$CAUSAL_CONV1D_SHA256"
fi
docker exec sayso-trainer "$VENV/bin/python" -m pip install -r "$ROOT/training/requirements-sayso-trainer.txt"
docker exec sayso-trainer "$VENV/bin/python" -c '
import sys
sys.path.insert(0, sys.argv[1])
import sayso_trainer
sayso_trainer.check_installed_stack()
from trl import SFTConfig, SFTTrainer
import causal_conv1d, datasets, tokenizers, torch, trl, transformers
for mod in (trl, transformers, datasets, tokenizers, torch, causal_conv1d):
    print(mod.__name__, mod.__version__, mod.__file__)
args = sayso_trainer.parse_args(["--canary", "--out", "/tmp/sayso-trainer-smoke"])
SFTConfig(bf16=False, max_length=args.cutoff, packing=False, padding_free=False, dataset_kwargs={"skip_prepare_dataset": True}, gradient_checkpointing=True, **sayso_trainer.sft_recipe(args, evaluating=True))
' "$ROOT/training/scripts"
