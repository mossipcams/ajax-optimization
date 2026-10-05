# SaySo training and wake tooling

This directory contains the training tools imported from SaySo. Run the commands
in the older training and wake runbooks from this directory: `training/.venv`,
`training/configs`, `satellite/models`, and `scripts/` are relative to here.

| Path | Purpose |
|---|---|
| `training/` | Corpus generators, adapters, recipes, fixtures, tests, and training/export commands. |
| `scripts/` | Wake corpus, mining, training, and host utilities. |
| `satellite/models/` | Local wake recipes and generated model/data directories. |
| `satellite/eval` | Relative link to the upstream wake evaluation tools and fixtures. |
| `docs/` | Imported plans and runbooks; host paths describe the original deployment. |
| `vendor/SaySo/` | Git submodule supplying evals, `sayso_contract`, Home Assistant fetch/websocket helpers, and `satellite/sayso` runtime code. |
| `schemas`, `custom_components`, `evals` | Relative links to upstream resources used by existing recipes and compatibility checks. |

## Initialize and run

From the ajax-optimization root:

```bash
git submodule update --init sayso-training/vendor/SaySo
cd sayso-training
python3.12 -m venv training/.venv
source training/.venv/bin/activate
python -m pip install -r training/requirements.txt numpy scipy soundfile
source ./env.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python -m pytest training/tests scripts -q
python -m generators.cli --help
python scripts/wake_train.py --help
```

The submodule is pinned to SaySo `origin/main` commit
`738959a3b279a9d40c7f9258c97b8dd650472740`. Normal initialization uses the recorded
commit; it does not follow later upstream changes.

`training_paths.py` defines the shared Python search path. `env.sh` exports it
for direct scripts and their subprocesses; pytest and `./sayso` load the same
setup automatically. Local training and wake scripts take precedence over the
copies in the upstream checkout. The generator bootstrap also loads it when a
caller starts Python with a cleared environment. The two files in `training/scripts` named
`fetch_ha_home.py` and `ha_websocket.py` link to their submodule implementations,
so existing `scripts.fetch_ha_home` and `scripts.ha_websocket` imports keep working
without copied modules. `scripts/preflight.py` runs the upstream validator with
the local training packages and configuration.

Use `./sayso generate`, `validate`, `canary`, `promote-dataset`, `train`, `eval`,
and `promote-model` for the existing promotion workflow. Training and wake jobs
still require the GPU host and its lock. Install
`satellite/models/requirements-wake-train.txt` in the host training environment
for actual wake training; it is not needed for the basic CPU tests. The existing
`test_lfm_python_parse.py` imports the Home Assistant integration directly and
needs its runtime dependencies (see `vendor/SaySo/pyproject.toml`) in a compatible
Python environment. Without those dependencies, pytest stops during collection.
Some generator tests also need the Hugging Face tokenizer cache or network access;
optional wake trainer/model checks may skip when those dependencies are absent.

The promotion CLI defaults to the container mount
`/workspace/host/ajax-optimization/sayso-training`; set
`SAYSO_TRAINING_HOST_ROOT` if the host mounts this checkout elsewhere. The GPU
wrapper's compose default is
`/srv/llm/ajax-optimization/sayso-training/scripts/llm-host/llama-compose.yml`,
overridable with `GPU_LLAMA_COMPOSE`. Dataset, model, cache, and run paths under
`/srv/llm` remain host storage locations. No host deployment is performed by
initializing this tree.

`scripts/llm-host/llama-compose.yml` overlaps the parent repository's
`compose/llama-compose.yml`. They have not been merged; review the overlap before
deploying either. The parent's 4-slot/128k KV constraints apply to serving.

`scripts/publish_model.sh` targets releases in `mossipcams/SaySo` and updates
constants in the submodule checkout. Publishing therefore leaves that checkout
dirty for a separate upstream review; it does not update the recorded gitlink.
