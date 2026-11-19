# SaySo training and wake tooling

This directory contains the training tools imported from SaySo. Run the commands
in the older training and wake runbooks from this directory: `training/.venv`,
`training/configs`, `satellite/models`, and `scripts/` are relative to here.

The canonical end-to-end LFM training pipeline — stages, commands, artifacts, and the invariants a run must not deviate from — is documented in [docs/PIPELINE.md](docs/PIPELINE.md).

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
`f854c8c5b4bd7c763d27c91dc33c5a5c865984cb`. Normal initialization uses the recorded
commit; it does not follow later upstream changes.

The full test suite is the verification gate, and it runs locally. Use
`scripts/ci-local.sh` from anywhere in the checkout. It exports the merge of `HEAD`
with `origin/main` into a temp tree, fills `vendor/SaySo` from the pinned submodule
commit, and runs `python -m pytest training/tests -q` in a Python 3.12 venv with the
exact CI pins (`SAYSO_CI_VENV`, default `~/.cache/sayso-ci/venv`):

```bash
sayso-training/scripts/ci-local.sh
sayso-training/scripts/ci-local.sh --remote USER@HOST
sayso-training/scripts/ci-local.sh --post
```

`--remote` streams the same tree and script to a temp dir on the host and runs the
suite there (use the llm VM for heavy runs). The last line is `local-ci: PASS|FAIL
<counts> head=<sha> main=<sha>`, and the exit status is pytest's. The result is
recorded under the git dir for `HEAD`. `--post` sets a `local-ci` commit status on
`HEAD` from that record; it fails if `HEAD` has no record or is not pushed.

The GitHub pull request job (`.github/workflows/sayso-training.yml`) is light. It runs
`training/tests -m 'not slow'` with only the non-tokenizer pins, which skips the
Hugging Face tokenizer tests. There is no nightly job and no manual dispatch. The fast
set includes the runtime-contract behaviour tests, which check the generator output
against the live Home Assistant result shapes (query tools, timers, labels). They need
no network access to Home Assistant and no secrets. To run only the fast set locally:

```bash
python -m pytest training/tests -m 'not slow' -q
```

`training_paths.py` defines the shared Python search path. `env.sh` exports it
for direct scripts and their subprocesses; pytest and `./sayso` load the same
setup automatically. Local training and wake scripts take precedence over the
copies in the upstream checkout. The generator bootstrap also loads it when a
caller starts Python with a cleared environment. The two files in `training/scripts` named
`fetch_ha_home.py` and `ha_websocket.py` link to their submodule implementations,
so existing `scripts.fetch_ha_home` and `scripts.ha_websocket` imports keep working
without copied modules. `scripts/preflight.py` runs the upstream validator with
the local training packages and configuration.

Use `./sayso run --model <profile> --hypothesis "<text>"` to chain the promotion
pipeline unattended (`generate`, `validate`, `canary`, `promote-dataset`, `train`,
`select`, `eval`, then queue the model if it passes), and `./sayso status --json` to read
its state. The stage commands also run by hand: `generate`, `validate`, `canary`,
`promote-dataset`, `train`, `select`, `eval`, and `promote-model`. `select` is CPU
auto-eval of the final and mid-run checkpoints, with no dev suite. `promote-model` is
the manual step after a model is queued; deployment stays separate. Training and wake
jobs still require the GPU host and its lock. The stages, `run.json`, the ledger, and the
promotion queue are described in [docs/PIPELINE.md](docs/PIPELINE.md). Install
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

## Generation gates and snapshot backup

`./sayso generate` snapshots the working tree to `refs/sayso/snapshots/<tree>` and runs the fast
tests and the golden generator test against that snapshot. The contract is invariants 12–17 in
[docs/PIPELINE.md](docs/PIPELINE.md):

- A fast-test failure always stops generation.
- Golden drift stops generation.
- Changes to the protected gate files (`training/tests`, `training/generators/validation.py`,
  `training/preflight.py`, `evals/config/gates.yaml`, `training/dataset_gates.py`, `sayso`) since the
  last promoted dataset also stop it.
- `promote-dataset` regenerates the candidate in a separate checkout and requires a byte-for-byte
  match; `train` refuses unless the candidate reproduced and its input fingerprint still matches.

Flags for `./sayso generate`:

- `--config <path>` generates from a recipe other than `GENERATION_CONFIG`
  (`training/configs/generation/full_sft_v13.yaml`). Use a reviewed, versioned recipe under
  `training/configs/generation/`.
- `--accept-drift "<reason>"` accepts golden drift or protected gate-file changes for this candidate.
  The reason must be non-empty. Gate-file changes save the diff as
  `training/datasets/candidates/<run_id>.gate-diff.patch`. Use the flag only after reviewing the
  change and stating what changed and why the new output is wanted.

`scripts/backup-snapshot-refs.sh` runs on the dev machine. It fetches `refs/sayso/*` from the llm VM
clone (`SAYSO_VM_REPO`, default `LLM@192.168.1.76:ajax-optimization`) into the local `refs/sayso/*`,
then pushes `refs/sayso/*` to `origin`. It does not touch branches and does not force-push. Run it
periodically, for example from cron (the ssh key must work without a prompt):

```bash
*/30 * * * * cd /path/to/ajax-optimization && sayso-training/scripts/backup-snapshot-refs.sh >> "$HOME/Library/Logs/sayso-snapshot-backup.log" 2>&1
```
