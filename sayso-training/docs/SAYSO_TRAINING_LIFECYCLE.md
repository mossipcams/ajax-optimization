# SaySo training lifecycle

Run training from the SaySo checkout on the `llm` VM. The local `./sayso`
command wraps the existing v5 generator, static preflight, rendered-view
exporter, Unsloth trainer, and production SaySo scorer. It does not commit code,
publish weights, or deploy a model. See [training/README.md](../training/README.md)
for the observed host layout, [the model training plan](SAYSO_LFM_TRAINING_PLAN.md)
for design constraints, and [TRAINING_LOG.md](../training/TRAINING_LOG.md) for
historical runs and scores.

## Commands

Run these in order for each candidate:

```bash
# Fast CPU-only check while developing a dataset
python scripts/preflight.py

# Candidate -> promoted dataset -> evaluated model
./sayso generate
./sayso validate
./sayso canary
./sayso promote-dataset
./sayso train
./sayso select
./sayso eval
./sayso promote-model
```

`select`, `eval`, and `promote-model` act on a training run, not on the current
candidate: the latest `./sayso train` by default, or `--train-id <id>`.

`./sayso generate` builds the configured `full_sft_v12.yaml` dataset under
`training/datasets/candidates/<run-id>.jsonl`. Each run also has a manifest and
metadata record. Generation never changes the promoted-dataset pointer.

`./sayso validate` runs `scripts/preflight.py` against that candidate using
`training/configs/preflight.yaml` and the accepted reference in
`training/configs/training_baseline.json`. It checks row and tool-call structure,
pinned tools and targets, positive coverage, negative ratios, duplicates and
conflicts, family-distribution drift, required fixtures, and overlap with eval
utterances. A failed check leaves the candidate unvalidated.

`./sayso canary` requires successful validation. It renders and stages the
candidate, then runs the first 250 optimizer steps of the full-run schedule: the
trainer keeps the full run's cosine horizon and warmup and stops early, so step
250 of a canary is step 250 of a full run (the canary record's `canary_schedule`
shows the stop step, the schedule length, and the last logged learning rate).
The smoke suite is gated; the dev suite is recorded as a diagnostic. Apart from
smoke's 24 promotion cases, the canary doesn't score the locked promotion or
context suites. It is a compatibility and
early-warning gate, not a way to rank datasets: compare datasets with `select`
on full runs.

`./sayso promote-dataset` succeeds only when validation and canary both passed
for the unchanged dataset. It writes the local promoted-dataset record, including
the run ID, dataset and rendered-data hashes, generation commit, recipe, and
canary metrics.

`./sayso train` reads that explicit promotion record, checks the source and
rendered-data hashes, stages the exact rendered dataset, and launches full
training through `/srv/llm/bin/gpu train lfm`. The trainer independently requires
the promotion record and verifies the rendered-data hash before training. If a
newer candidate exists, this command still trains the latest explicitly promoted
dataset and leaves the newer candidate current. It refuses to start on tracked
uncommitted changes or a `vendor/SaySo` commit that is not on the SaySo remote.
Each run gets its own `train_id` and record under `training/runs/promotion/train/`.
The trainer writes `manifest.json` into the run directory (trainer, base-model
file, and data hashes, resolved training arguments, seed, package versions) and
the record embeds it. Every 200-step checkpoint is kept, weights only.

`./sayso select` scores every checkpoint of the run on the dev suite (CPU FP32)
and records the one with the fewest safety flags, then the highest dev pass
rate, then the earliest step. Selecting again clears any earlier release eval.

`./sayso eval` is the release gate. It converts the selected checkpoint with
llama.cpp's `convert_hf_to_gguf.py` (`GGUF_CONVERTER`), quantizes it to the
shipped type (`RELEASE_QUANT`, default `Q4_K_M`) with llama-cpp-python, and
scores that GGUF the way the integration runs it: llama-cpp-python in-process
with the integration's own message transform. The champion GGUF in
`training_baseline.json` (`eval.champion`, checked against its sha256) is scored
on the same cases. The candidate passes only if, against the champion:

- promotion: it wins more discordant cases than it loses, and no category
  loses more than 2 cases net;
- regression: it loses at most 2 cases net;
- every release suite: no case gains a zero-tolerance safety flag;
- context: it passes the 100% context gate.

The absolute `gates.yaml` thresholds are still reported per suite, but they no
longer decide release. The report includes per-suite wins, losses, and a sign-test
p-value.

`./sayso promote-model` writes `promoted_model.json` only for a GGUF whose
sha256 still matches the one that was evaluated, after the release eval passes.
The record holds the train and dataset IDs, both commits, the checkpoint, GGUF
path and sha256, metrics, and the champion comparison. It does not publish:
`scripts/publish_model.sh` refuses any GGUF whose sha256 is not in that record.

If an operator explicitly accepts a model despite failed release gates, pass
`--manual-override --reason "..."` to `promote-model`. The command still
requires a completed release eval of that exact GGUF and records its real
metrics and failed gate state; it never changes `final_eval_passed` to true.

## Eval suites

The SaySo cases have three roles. `promotion` (120) and `context` are locked
acceptance suites and are only scored by `./sayso eval`. Every other case is
split by category, with a fixed seed, into `dev` (checkpoint selection, canary
diagnostics) and `regression` (scored only by `./sayso eval`). `smoke` is still
a 24-case subset of `promotion`, and SaySo's tests require that.

## Local records and baselines

Candidate datasets are stored under ignored `training/datasets/candidates/`.
Per-run candidate records and the current-candidate pointer are under ignored
`training/runs/promotion/`; training runs are under its `train/` directory, with
`latest_train.json` pointing at the newest. The same directory holds the latest
`promoted_dataset.json` and `promoted_model.json`; each per-run JSON record keeps
the candidate's earlier gate results. These are local operational files, not a
database or Git-tracked approval system.

The accepted dataset distribution and model-eval baseline live in
`training/configs/training_baseline.json`. A run reads this file but never
rewrites it. Update it manually only after reviewing and accepting a dataset or
model; a failed run cannot replace the baseline.

`./train.sh` is a compatibility shortcut for `./sayso train`. It deliberately
fails unless an accepted dataset promotion already exists. Do not launch model
training from the Unsloth Studio UI because that bypasses the GPU lock and
promotion checks.

For code changes that affect generation, training rows, prompt/template
formatting, loss masking, or training configuration, generate a fresh candidate
and run the complete sequence. Unrelated code changes do not require a canary.
