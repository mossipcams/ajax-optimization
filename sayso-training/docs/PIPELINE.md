# SaySo LFM training pipeline

Canonical, end-to-end description of the SaySo `LFM2.5-230M-Base` fine-tuning
pipeline and the invariants a run must not deviate from. Design rationale lives in
[SAYSO_LFM_TRAINING_PLAN.md](SAYSO_LFM_TRAINING_PLAN.md); measured results live in
`training/TRAINING_LOG.md`. This file is authoritative for *what* a compliant run
does and *in what order*.

```text
generate → validate → canary → promote-dataset → train → select → eval → promote-model ─ publish
  (corpus)  (gates)   (smoke)     (gate)         (full SFT) (dev best)(champion A/B)(record)  (ship)
```

All commands run from the `sayso-training/` directory. The promotion CLI drives
each stage; host paths and the GPU lock are defined in `sayso`.

## 1. Corpus generation

- **Command:** `./sayso generate`. Internally runs
  `python -m generators.cli --config <generation-config> --output <dataset> --manifest <manifest>`
  with the working directory set to `training/` (see `cmd_generate` in `sayso`).
- **Config it reads:** `training/configs/generation/full_sft_v13.yaml`, named by the
  `GENERATION_CONFIG` constant in `sayso`. The generator configs under
  `training/configs/generation/` are versioned (`full_sft_v4.yaml` … `full_sft_v13.yaml`,
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

## 2. Post-render validation and hard label gates

- **Command:** `./sayso validate`. Runs the dataset through
  `scripts/preflight.py <dataset>` (see `cmd_validate` in `sayso`).
- **Inputs:** the unchanged candidate dataset from stage 1. The CLI re-checks the
  dataset sha256 against the recorded value before running preflight.
- **Label gates (enforced at generation, checked here):** every tool call must pass the
  pinned schema; `ALLOWED_HASS_TOOLS` in `training/adapters/schema.py` is the dataset-build
  gate for that contract. Coverage and eval-overlap gates live in
  `training/generators/validation.py`, including `check_quality_eval_overlap`, which rejects any
  training utterance present in the held-out set `excluded_train_utterances()` (defined in
  `evals.cases`). Generation is reproducible for a fixed seed.
- **Outputs:** the promotion record is marked `validated` (with `validated_sha256`) under
  `training/runs/promotion/`.

## 3. Training

Training proceeds in two gated steps: a canary, then full fine-tuning of an
explicitly promoted dataset. Both run the same trainer through the GPU lock.

- **Canary — command:** `./sayso canary` (optional `--reuse-checkpoint`). Renders and stages
  the split, then runs the trainer's first 250 steps and scores smoke + dev.
- **Gate — command:** `./sayso promote-dataset`. Requires a passing validation *and* a passing
  canary for the unchanged candidate; writes `training/runs/promotion/promoted_dataset.json`.
- **Full — command:** `./sayso train` (`--train-id` optional). Blocked until a valid promoted
  dataset exists and its sha256 still matches.

Both canary and full invoke, via the GPU lock,
`gpu train lfm python <host>/training/scripts/train_unsloth_full.py --data <staged> [--eval-data <staged-val>] …`
with the recipe `--lr 5e-5 --warmup-ratio 0.1` (`TRAIN_RECIPE` in `sayso`). The full run also
passes `--promotion-record`. Every GPU job runs inside a lock turn: `gpu run <label> -- <cmd>`
(the CLI calls the `gpu` binary at `/srv/llm/bin/gpu`, overridable with `GPU_LOCK`).

- **Config it reads / recipe:** base model `LFM2.5-230M-Base` (`BASE_MODEL` in `sayso`;
  `--model` default in `training/scripts/train_unsloth_full.py`); learning rate and warmup from
  `TRAIN_RECIPE`. The trainer is full-parameter (`full_finetuning=True`) with FP32 master weights
  (`float32_mixed_precision=True`) and BF16 compute (`bf16=True`).
- **Render + group-held-out split:** before training, the CLI renders both splits —
  `training/scripts/split_dataset.py` (whole-scenario-group held-out; the CLI gets `1 - VAL_RATIO`
  and `VAL_RATIO`, where `VAL_RATIO = 0.05` is defined in `sayso`, so a 95/5 train/val split) and `training/scripts/export_llamafactory_view.py` (the derived
  prompt/completion view rendered with the pinned Base template). The val split is passed as
  `--eval-data` so the trainer reports group-held-out validation loss (`eval_strategy="steps"`).
- **Outputs:** canary checkpoint + smoke/dev metrics under `/srv/llm/lfm/runs/sayso-promotion/`;
  a full run's checkpoints at `<out>/checkpoint-*` and `<out>/final`, plus a training-run record
  `training/runs/promotion/train/<train_id>.json` and `latest_train.json`.

## 4. Evaluation

- **Dev / checkpoint scoring — command:** `./sayso select` (`--train-id` optional). Scores every
  retained checkpoint on the `dev` suite via
  `training/scripts/eval_checkpoint_cpu.py --suite dev`, then picks the best (see stage 5).
- **Release A/B — command:** `./sayso eval`. Exports the selected checkpoint to F16 then to the
  release quantization (`RELEASE_QUANT = Q4_K_M`) through `training/scripts/export_gguf.py`, and
  scores the candidate *and* the champion on each of `promotion`, `context`, and `regression`
  (`RELEASE_SUITES`). Canary runs score the `smoke` suite instead.
- **Counterfactual probe:** `training/scripts/counterfactual_probe.py` (probe kinds
  `device_absent`, `tool_removed`, `clarify_resolved`) for entity/tool grounding behavior.
- **Config it reads:** gate thresholds and zero-tolerance flags from `evals/config/gates.yaml`
  (`suites.<suite>.zero_tolerance_flags`, read by `safety_flags()` in `sayso`); the deployed
  champion from `training/configs/training_baseline.json`. Suites are selected by tag via
  `evals.cases.select_cases` and gated by `evals.scorer.check_gates`.
- **Outputs:** per-run result directories under `/srv/llm/lfm/runs/sayso-promotion/<name>/` with
  `summary.json`, `outcomes.jsonl`, and `metadata.json`.

## 5. Checkpoint selection and promotion against the deployed champion

- **Selection — command:** `./sayso select`. Chooses the checkpoint with the fewest safety flags,
  then highest dev pass rate, then earliest step (`select_checkpoint` in `sayso`). Records
  `selected_checkpoint` on the training run.
- **Champion comparison — command:** `./sayso eval`. Compares the candidate GGUF against the
  deployed champion **case-for-case** (`compare_release` in `sayso`). The champion is the sha256-pinned
  GGUF named by `eval.champion` in `training/configs/training_baseline.json`; it is re-scored on
  the same cases so the comparison is paired, not historical.
- **Promotion gates (must all pass):** more wins than losses vs the champion; no per-category net
  worse than `-RELEASE_SLACK` (`RELEASE_SLACK = 2`); the `context` 100% gate holds; and no new
  zero-tolerance safety flag appears relative to the champion.
- **Promotion — command:** `./sayso promote-model` (`--manual-override --reason <text>` only with an
  audit note). Writes `training/runs/promotion/promoted_model.json`.

## 6. Deployment

- **Command:** `scripts/publish_model.sh`. Pins the artifact by sha256 and updates constants in the
  SaySo submodule checkout for release.
- **Inputs:** the promoted model GGUF from stage 5 (`promoted_model.json`).
- **Outputs / notes:** publishing targets releases in the upstream repo and leaves that submodule
  checkout dirty for a separate upstream review; it does not update the recorded gitlink. Promotion,
  deployment, and shipping are distinct decisions — record which applies; never infer promotion from
  the fact that a model is serving or published.

## Changing an invariant

Changing any pipeline invariant requires **explicit user approval** and an update to this file in
the **same change**. A run that would deviate from the invariants below must stop and request
direction rather than proceeding.

## Pipeline invariants

1. **Base model is `LFM2.5-230M-Base`.** Defined by `BASE_MODEL` in `sayso` and the `--model`
   default in `training/scripts/train_unsloth_full.py`.
2. **Full-parameter fine-tuning from fresh Base — no LoRA/adapter, no continued merge.** Defined by
   `full_finetuning=True` in `training/scripts/train_unsloth_full.py`.
3. **FP32 master weights with BF16 compute.** Defined by `float32_mixed_precision=True` and
   `bf16=True` in `training/scripts/train_unsloth_full.py`.
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

## Guard rails

The checkable invariants above are enforced by `training/tests/test_pipeline_invariants.py`, which
reads the constants from `sayso` and the trainer and fails when they drift from this file. It also
parses the doc's stated values (base model, `VAL_RATIO`, `TRAIN_RECIPE`, `RELEASE_SLACK`,
`RELEASE_QUANT`, `RELEASE_SUITES`, `GENERATION_CONFIG`), so changing the code or the doc alone fails
the test. Run it from `sayso-training/`: `python -m pytest training/tests/test_pipeline_invariants.py -q`.
Changing an invariant still requires explicit user approval, and the doc and this test must change in
the same change.
