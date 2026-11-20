# SaySo LFM training pipeline

Canonical, end-to-end description of the SaySo LFM fine-tuning pipeline and the invariants a run
must not deviate from. The base model and recipe come from a reviewed model profile in
`training/configs/models.yaml` (invariant 1). Design rationale lives in
[SAYSO_LFM_TRAINING_PLAN.md](SAYSO_LFM_TRAINING_PLAN.md); measured results live in
`training/TRAINING_LOG.md`. This file is authoritative for *what* a compliant run does and *in what order*.

```text
run:       generate → validate → canary → promote-dataset → train → select → eval → queue
             (corpus)  (gates)   (smoke)     (gate)        (full SFT) (CPU auto-eval) (champion A/B) (promotion queue)
by hand:   promote-model → publish
```

All commands run from the `sayso-training/` directory. `./sayso run` chains the stages above in
that order; each stage is also a standalone command. Host paths and the GPU lock are defined in `sayso`.

## Running the pipeline

- **Command:** `./sayso run --model <profile> --hypothesis "<text>" [--config <path>] [--accept-drift "<reason>"]`.
  `--model` (a profile from `training/configs/models.yaml`, invariant 1) and a non-empty `--hypothesis`
  are required. `--config` defaults to `GENERATION_CONFIG`, and `--accept-drift` is passed to `generate`.
- **Chain:** generate, validate, canary, promote-dataset, train, select, eval, queue. Canary and train
  both use the run's `--model`, so their profiles always match.
- **State:** `training/runs/promotion/run.json` records `run_key`, `model`, `hypothesis`, `config`,
  `accept_drift`, `stage`, `stages_done`, `dataset_run_id`, `train_id`, `verdict`, `failure`,
  `started_at`, and `updated_at`. A stage is added to `stages_done` only when it completes.
- **Failure and resume:** a failing stage stores `failure` in `run.json`, calls `gpu serve`, appends a
  `failed` ledger line, and exits nonzero. `./sayso run --resume` continues from the first stage not in
  `stages_done`. A new `run` is refused while `run.json` is unfinished, and `--resume` is refused once the
  run has finished.
- **Release gates are not a stage failure:** if the release evaluation fails its gates, `eval` still
  records the comparison, `queue` sets the verdict to `not_promotable`, and the run finishes normally.
- **Status — command:** `./sayso status [--json]`. Prints one object with `run` (`run.json`),
  `training_progress` (the latest training run's checkpoint step and `max_steps`, or null),
  `queued_promotions` (train ids in the promotion queue), and `next_command`. `next_command` is
  `./sayso run --resume` after a failed or unfinished run, `./sayso promote-model --train-id <id>` when a
  model is queued, and null otherwise. `--json` prints one compact line; without it the object is indented.
- **Ledger:** `training/runs/ledger.jsonl` is append-only. Each failing stage appends a line with verdict
  `failed`, and each finished run appends one line with verdict `queued` or `not_promotable`. A line holds
  the run key, model, hypothesis, config, dataset and train ids, git commit, `auto_eval`,
  `selected_checkpoint`, per-suite release counts (candidate passed, champion passed, total), `failure`,
  and `verdict`. A failure followed by a resume leaves one `failed` line and then the resumed run's own
  finished line.
- **Promotion queue:** a run whose release evaluation passes writes
  `training/runs/promotion/promotion_queue/<train_id>.json` (profile, hypothesis, selected checkpoint, GGUF
  path and sha256, release comparison). Nothing promotes or deploys it. A person runs
  `./sayso promote-model --train-id <train_id>` (stage 5), which writes `promoted_model.json` and leaves the
  queue entry in place. Deployment (stage 6) stays a separate manual step.

## 1. Corpus generation

- **Command:** `./sayso generate`. Internally runs
  `python -m generators.cli --config <generation-config> --output <dataset> --manifest <manifest>`
  with the working directory set to `training/` (see `cmd_generate` in `sayso`).
- **Config it reads:** `training/configs/generation/full_sft_v17.yaml`, named by the
  `GENERATION_CONFIG` constant in `sayso`. The generator configs under
  `training/configs/generation/` are versioned (`full_sft_v4.yaml` … `full_sft_v17.yaml`,
  plus `production.yaml` and `smoke.yaml`).
- **Inputs:** the generation config; the pinned tool contract
  `schemas/sayso-tool-schema-v2.json` (referenced by the config's `schema.path`); the real-home
  fixture `training/fixtures/real_home.json`; the area distribution
  `training/configs/area_distribution_v1.json`. All are read from within the generator
  (`training/generators/cli.py`). `schemas/` and `evals/` resolve through the pinned
  SaySo submodule (`vendor/SaySo`).
- **Outputs:** a candidate dataset and manifest in `training/datasets/candidates/`
  (`<run_id>.jsonl`, `<run_id>.manifest.json`) and a promotion state record under
  `training/runs/promotion/` (`current.json` pointing at `<run_id>.json`).

- **Snapshot and gates:** `./sayso generate` snapshots the working tree and runs the generation
  gates before writing a corpus (invariants 12–15). `--config <path>` overrides `GENERATION_CONFIG`
  for one run. `--accept-drift "<reason>"` accepts golden drift or protected gate-file changes.
  `promote-dataset` and `train` re-check the candidate (invariants 16–17).

## 2. Post-render validation and hard label gates

- **Command:** `./sayso validate`. Runs the dataset through
  `scripts/preflight.py <dataset>` (see `cmd_validate` in `sayso`).
- **Inputs:** the unchanged candidate dataset from stage 1. The CLI re-checks the
  dataset sha256 against the recorded value before running preflight.
- **Preflight config:** a generation YAML may set top-level `preflight: <path>`, which
  `sayso validate` passes to `scripts/preflight.py` as `--config` (default
  `training/configs/preflight.yaml`). `coverage.min_behavior_share` (default 0.005) sets the
  audit's minimum multi-call and alias row share (exclusion rows have no floor; the user does not prioritize exclusion phrasing, 2026-11-18), and `ordinary_ops` accepts a
  `cap.op: weight` mapping. v18 uses `preflight_v18.yaml` with multi_action 300 and GetDateTime
  150 and min_behavior_share 0.003, approved by the user on 2026-11-18; v18 compares drift against
  `training/configs/training_baseline_v18.json` (measured v18 family shares, user-approved 2026-11-18)
  instead of the v11-era baseline.
- **Label gates (enforced at generation, checked here):** every tool call must pass the
  pinned schema; `ALLOWED_HASS_TOOLS` in `training/adapters/schema.py` is the dataset-build
  gate for that contract. Coverage and eval-overlap gates live in
  `training/generators/validation.py`, including `check_quality_eval_overlap`, which rejects any
  training utterance present in the held-out set `excluded_train_utterances()` (defined in
  `evals.cases`). Generation is reproducible for a fixed seed.
- **Outputs:** the promotion record is marked `validated` (with `validated_sha256`) under
  `training/runs/promotion/`.

## 3. Training

Training proceeds in two gated steps: a canary, then full fine-tuning of an explicitly promoted dataset.
Both run the same trainer through the GPU lock, with the recipe of one model profile.

- **Canary — command:** `./sayso canary [--model <profile>] [--reuse-checkpoint]`. The profile defaults to
  `230m`. Renders and stages the split, runs the trainer with `--canary` (its first 250 optimizer steps,
  `CANARY_STEPS` in `training/scripts/train_common.py`), then scores the `smoke` suite with the canary
  gates (`eval.canary` in `training/configs/training_baseline.json`). The candidate records
  `canary_profile`, `canary_recipe`, and `canary_metrics`.
- **Gate — command:** `./sayso promote-dataset`. Requires a passing validation *and* a passing
  canary for the unchanged candidate; writes `training/runs/promotion/promoted_dataset.json`.
- **Full — command:** `./sayso train [--model <profile>]`. Blocked until a valid promoted
  dataset exists and its sha256 still matches. `--model` must equal the promoted canary profile.

Both canary and full invoke, via the GPU lock,
`gpu train lfm <TRAINER_PYTHON> <host>/training/scripts/sayso_trainer.py --data <staged> [--eval-data <staged-val>] …`
(`TRAINER_PYTHON` is env `SAYSO_TRAINER_PYTHON`, default the SaySo Trainer venv's `bin/python`; its exact pins live in `training/requirements-sayso-trainer.txt`)
with the profile recipe below. The full run also passes `--promotion-record`. Every GPU job runs inside a lock turn: `gpu run <label> -- <cmd>`
(the CLI calls the `gpu` binary at `/srv/llm/bin/gpu`, overridable with `GPU_LOCK`).

- **Model profiles and recipe:** `training/configs/models.yaml` holds the reviewed profiles. `train_recipe()`
  in `sayso` turns a profile into `--model <trainer_model>`, `--lr`, `--warmup-ratio`, `--batch`, `--accum`,
  followed by the profile's `extra_args`. Current values:

  | Profile | Base model (`trainer_model` basename) | `lr` | `warmup_ratio` | `batch` | `accum` | `extra_args` |
  |---|---|---|---|---|---|---|
  | `230m` (default) | `LFM2.5-230M-Base` | `5e-5` | `0.1` | `1` | `32` | none |
  | `350m` | `LFM2.5-350M-Base-sayso` | `2e-5` | `0.1` | `1` | `18` | `--merge-views` |

  Batch 1 with accumulation is used because batch 8 x accum 4 measured 23% slower per step on the 7900 XTX; ROCm SDPA has no flash kernel with a padding mask.

- **Trainer:** the trainer is full-parameter (`AutoModelForCausalLM.from_pretrained`, no adapter) with BF16 weights and
  compute (`dtype=torch.bfloat16`, `bf16=True`) and no FP32 master weights (`float32_mixed_precision` not set).
  `--model` comes from the profile; the trainer's own `--model` default is the `230m` base model.
- **Render + group-held-out split:** before training, the CLI renders both splits —
  `training/scripts/split_dataset.py` (whole-scenario-group held-out; the CLI gets `1 - VAL_RATIO`
  and `VAL_RATIO`, where `VAL_RATIO = 0.05` is defined in `sayso`, so a 95/5 train/val split) and `training/scripts/export_llamafactory_view.py` (the derived
  prompt/completion view rendered with the chat template of the canary profile's base model; the tool list is placed before
  the `Area context:` block, as SaySo's embedded runtime prompts the model since 1.17.0, with each tool's
  keys in template order: `type`, then `function` `name`, `description`, `parameters`; `adapters/test_lfm.py` and SaySo
  `tests/test_tool_key_order.py` pin the same bytes). The val split is passed as
  `--eval-data` so the trainer reports group-held-out validation loss (`eval_strategy="steps"`).
- **Outputs:** the canary checkpoint and smoke results under `/srv/llm/lfm/runs/sayso-promotion/<run_id>-canary/`;
  a full run's checkpoints at `<out>/checkpoint-*` and `<out>/final`, where `<out>` is
  `/srv/llm/lfm/runs/sayso-promotion/<train_id>-full`; plus a training-run record
  `training/runs/promotion/train/<train_id>.json` and `latest_train.json`.

## 4. Evaluation

- **Auto-eval — command:** `./sayso select` (`--train-id` optional; stage 6 of `run`). Chooses the checkpoint
  on CPU, without the GPU lock:
  1. Two checkpoints are scored: `final`, and the saved `checkpoint-*` nearest 57% of the trainer's `max_steps`
     (`MID_CHECKPOINT_FRACTION`). If the manifest has no `max_steps`, the highest saved step is used.
  2. Each is scored on the `promotion` suite (`training/scripts/eval_checkpoint_cpu.py --suite promotion`,
     run in the trainer container with GPUs hidden) and on 300 held-out rows (`HELDOUT_LIMIT`) of the run's own
     group-held-out validation split (`training/scripts/eval_rendered_cpu.py`; one view per source row, drawn
     with a fixed seed; a row passes on exact match of the output text).
  3. The winner has the fewest safety flags (the promotion suite's `zero_tolerance_flags`), then the highest
     promotion pass rate, then the highest held-out pass rate, then the earliest step.

  The winner is recorded as `selected_checkpoint` on the training run, clearing any earlier release
  evaluation for it. `report.json` (per-checkpoint scores and flags) is written to
  `/srv/llm/lfm/runs/sayso-promotion/<train_id>-full/`. No dev suite is used.
- **Release A/B — command:** `./sayso eval` (`--train-id` optional). Exports the selected checkpoint to F16 then to the
  release quantization (`RELEASE_QUANT = Q4_K_M`) through `training/scripts/export_gguf.py`, and scores the candidate *and* the champion on each of `promotion`, `context`, and `regression` (`RELEASE_SUITES`). The `regression` suite has no suite file; it is every case tagged `regression` (`evals/cases/regressions.jsonl` and the tagged cases in `evals/cases/context_lookalikes.jsonl`).
- **Counterfactual probe:** `training/scripts/counterfactual_probe.py` (probe kinds
  `device_absent`, `tool_removed`, `clarify_resolved`) for entity/tool grounding behavior. It is a standalone script; no stage runs it.
- **Config it reads:** gate thresholds from `evals/config/gates.yaml`; `safety_flags()` reads the
  `promotion` suite's `zero_tolerance_flags`. The deployed champion comes from
  `training/configs/training_baseline.json`. Suites are selected by tag via `evals.cases.select_cases` and
  gated by `evals.scorer.check_gates`.
- **Outputs:** per-run result directories under `/srv/llm/lfm/runs/sayso-promotion/<name>/` with
  `summary.json`, `outcomes.jsonl`, and `metadata.json`. Auto-eval results are named
  `<train_id>-auto-<checkpoint>-<timestamp>` and held-out results `<train_id>-heldout-<checkpoint>`.

## 5. Promotion against the deployed champion

- **Champion comparison — command:** `./sayso eval`. Compares the candidate GGUF against the
  deployed champion **case-for-case** (`compare_release` in `sayso`). The champion is the sha256-pinned
  GGUF named by `eval.champion` in `training/configs/training_baseline.json`; it is re-scored on
  the same cases so the comparison is paired, not historical.
- **Promotion gates (must all pass):** more wins than losses vs the champion; no per-category net
  worse than `-RELEASE_SLACK` (`RELEASE_SLACK = 2`); the `context` 100% gate holds; and no new
  zero-tolerance safety flag appears relative to the champion.
- **Promotion — command:** `./sayso promote-model` (`--train-id` optional; `--manual-override --reason <text>` only with an
  audit note). Requires a completed release evaluation. Writes `training/runs/promotion/promoted_model.json`.
  Within `run`, a model in the promotion queue is promoted only by running this command by hand.

## 6. Deployment

- **Command:** `scripts/publish_model.sh`. Pins the artifact by sha256 and updates constants in the
  SaySo submodule checkout for release.
- **Inputs:** the promoted model GGUF from `promote-model` (`promoted_model.json`).
- **Outputs / notes:** publishing targets releases in the upstream repo and leaves that submodule
  checkout dirty for a separate upstream review; it does not update the recorded gitlink. Promotion,
  deployment, and shipping are distinct decisions — record which applies; never infer promotion from
  the fact that a model is serving or published.

## Changing an invariant

Changing any pipeline invariant requires **explicit user approval** and an update to this file in
the **same change**. A run that would deviate from the invariants below must stop and request
direction rather than proceeding.

## Pipeline invariants

1. **Base model comes from a reviewed model profile.** `training/configs/models.yaml` defines the profiles, and each sets the trainer base model (`trainer_model`) and its recipe (`lr`, `warmup_ratio`, `batch`, `accum`, `extra_args`). Today `230m` = `LFM2.5-230M-Base` and `350m` = `LFM2.5-350M-Base-sayso`. `--model <profile>` selects one on `./sayso run`, `canary`, and `train` (default `230m`); the canary profile must match the train profile, and `train` refuses a mismatch. Defined by `model_profile` and `train_recipe` in `sayso`, the recipe table in stage 3, and the `--model` default in `training/scripts/sayso_trainer.py`.
2. **Full-parameter fine-tuning from fresh Base — no LoRA/adapter, no continued merge.** Defined by
   `AutoModelForCausalLM.from_pretrained` with no PEFT/LoRA/adapter call in `training/scripts/sayso_trainer.py`.
3. **BF16 weights and compute; no FP32 master weights.** Defined by `dtype=torch.bfloat16` on
   the `AutoModelForCausalLM.from_pretrained` call and `bf16=True` in `SFTConfig` in `training/scripts/sayso_trainer.py`, and the absence of `float32_mixed_precision=True`.
4. **Group-held-out validation split — a scenario/household group never spans two splits.** Defined
   by `_group_keys`, `assign_split`, and the violation checks in `training/scripts/split_dataset.py`;
   held at `VAL_RATIO = 0.05` in `sayso`.
5. **Schema-conditioned function-calling labels; no ChatML `<tool_call>` labels.** The label format is the
   pinned contract rendered by `training/scripts/export_llamafactory_view.py`, validated against
   `schemas/sayso-tool-schema-v2.json`.
6. **No eval case IDs or utterances from `evals/cases/` in training data.** Enforced by
   `check_quality_eval_overlap` in `training/generators/validation.py` against the held-out set
   `excluded_train_utterances()` (`evals.cases`).
7. **Corpora limited to reviewed, versioned generation configs — no ad-hoc concatenation.** Defined
   by the versioned YAML under `training/configs/generation/` referenced by `GENERATION_CONFIG` in
   `sayso`.
8. **Label gates enabled on every row: schema + coverage + eval-overlap.** Defined by
   `ALLOWED_HASS_TOOLS` in `training/adapters/schema.py` and the gates in
   `training/generators/validation.py`.
9. **`ALLOWED_HASS_TOOLS` is the training contract (a dataset-build gate, not a runtime boundary).**
   Defined by `ALLOWED_HASS_TOOLS: frozenset[str] = v2_tool_names()` in `training/adapters/schema.py`.
10. **GPU jobs only via the GPU lock** (`gpu run <label> -- <cmd>`; `gpu train lfm …` for training).
    Defined by `GPU_LOCK` in `sayso` and the "GPU host safety" section of the root `AGENTS.md`.
11. **Promotion only against the deployed champion, case-for-case.** Defined by `compare_release` in
    `sayso`, the champion entry in `training/configs/training_baseline.json`, and the
    zero-tolerance flags in `evals/config/gates.yaml`.
12. **Generation snapshots the working tree, never HEAD.** `./sayso generate` stages the whole
    working tree (staged, unstaged, and untracked files that are not ignored) into a temporary
    index, writes that tree, and wraps it in a parentless commit stored at
    `refs/sayso/snapshots/<tree>`. The candidate records that commit as `snapshot_commit`, and
    generation never reads `HEAD`. Defined by `snapshot_worktree` in `training/dataset_gates.py`
    and `cmd_generate` in `sayso`.
13. **Generation tests gate every candidate.** Against the snapshot, `generate` runs the fast tests
    (`training/tests -m "not slow"`) and the golden test (`training/tests/test_generator_golden.py`).
    A fast-test failure always stops generation. Golden drift (assertion failures only) stops
    generation unless `--accept-drift "<reason>"` is given with a non-empty reason; a golden run that
    fails without drift also stops. Results are cached in
    `training/runs/promotion/test-cache/<snapshot_tree>.json` and reused only while the input
    fingerprint for `training/configs/generation/smoke.yaml` (invariant 15) is unchanged. Defined by
    `check_snapshot` in `training/dataset_gates.py`.
14. **Changes to gate files since the last promoted snapshot require `--accept-drift`.** Protected
    paths: `training/tests` (including the golden snapshot), `training/generators/validation.py`,
    `training/preflight.py`, `evals/config/gates.yaml`, `training/dataset_gates.py`, and `sayso`.
    If any of them differ between the promoted dataset's snapshot and the new snapshot, generation
    stops unless `--accept-drift "<reason>"` is given. The diff is then saved as
    `<run_id>.gate-diff.patch` in `training/datasets/candidates/`, and its path and sha256 are recorded
    with the candidate. With no promoted dataset there is no baseline and this check does not run.
    Defined by `GATE_PATHS` and `check_gate_changes` in `training/dataset_gates.py`.
15. **Generation records an input fingerprint.** The candidate's `input_fingerprint` covers the
    generator tree (`training/generators`); the recipe and its config files (tool schema, area
    distribution, real-home fixture); the eval cases and SaySo commit; the Python version; the
    installed packages; and the tokenizer (model name, `tokenizers` and `transformers` versions, and
    tokenizer files). Each group has its own sha256, and the whole has a top-level `sha256`. Defined by
    `input_fingerprint` in `training/dataset_gates.py` and `cmd_generate` in `sayso`.
16. **Promotion reproduces the candidate byte-for-byte.** `./sayso promote-dataset` checks out the
    candidate's snapshot in a separate detached worktree, fills `vendor/SaySo` from the recorded
    `sayso_commit`, and regenerates the dataset there. The regenerated file's sha256 must equal the
    candidate's byte-for-byte, and the manifest `dataset_hash` must match; otherwise promotion stops.
    Only then are `reproduced` and `reproduced_sha256` recorded. Defined by `cmd_promote_dataset` in
    `sayso` and `materialize_snapshot` in `training/dataset_gates.py`.
17. **Full training refuses unless the candidate is reproduced and still matches.** `./sayso train`
    stops before the GPU lock unless: the candidate is `reproduced`; its `reproduced_sha256` equals the
    current dataset sha256; its input fingerprint recomputes to the recorded sha256 in the current
    environment (`PYTHON`) from its snapshot; and committed `HEAD` has the same `training/generators`
    tree and the same `vendor/SaySo` gitlink (`sayso_commit`) as the snapshot. Defined by
    `check_trainable` in `training/dataset_gates.py` and the call in `cmd_train` in `sayso`.

## Guard rails

The checkable invariants above are enforced by `training/tests/test_pipeline_invariants.py`, which
reads the constants from `sayso` and the trainer and fails when they drift from this file. It also
parses the doc's stated values (the base model and recipe of each model profile against
`training/configs/models.yaml`, `VAL_RATIO`, `RELEASE_SLACK`, `RELEASE_QUANT`, `RELEASE_SUITES`,
`GENERATION_CONFIG`), so changing the code or the doc alone fails
the test. Run it from `sayso-training/`: `python -m pytest training/tests/test_pipeline_invariants.py -q`.
Invariants 12–17 are also checked by number against the doc text, and each cited guard must still
exist in `training/dataset_gates.py` or `sayso`. Changing an invariant still requires explicit user approval, and the doc and this test must change in
the same change.
