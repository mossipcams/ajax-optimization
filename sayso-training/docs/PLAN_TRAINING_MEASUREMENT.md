# Training pipeline: fix the measurement root causes (plan)

Date: 2026-10-28. Base: `fa9083b` on `ajax/generator-issue` (this branch,
`ajax/pipeline`, is 22 commits behind it and must be rebased first). Line
references below are at `fa9083b`. The SaySo eval suites live in the
`vendor/SaySo` submodule; counts below come from SaySo HEAD because the pinned
commit `7700db9` is not on the remote (see RC6).

Goal: before another full run, make the pipeline able to answer "is this model
better than the champion?" Generator changes stay out of scope until it can.

## Status (2026-10-28)

Implemented on `ajax/pipeline` (merged with `ajax/generator-issue`) and SaySo
`ajax/training-measurement`. Departures from the plan below:

- **The champion is v11 Q4_K_M, not v5b F16.** SaySo `main` ships
  `SaySo-LFM-v11-Q4_K_M.gguf` (sha `728cae4c…`, release `model-v4`).
  `training_baseline.json` → `eval.champion` now points at it, and
  `RELEASE_QUANT` defaults to `Q4_K_M`. v5b is kept as `previous_champion`.
- **Release runtime:** the integration runs llama-cpp-python in-process
  (`inference.py` `EmbeddedEngine`), not `llama_cpp.server`. So
  `eval_checkpoint_cpu.py` scores a `.gguf` through `Llama.create_chat_completion`,
  using the integration's own `_messages_for_embedded_template` read from its
  source. Quantizing also uses llama-cpp-python, because the `sayso/llama.cpp`
  image ships only `llama-server`.
- **Smoke stays inside promotion.** SaySo's `test_migration.py` asserts
  smoke ⊆ promotion and specific smoke categories. Moving smoke into dev means
  changing those tests, which needs an explicit go-ahead. The canary no longer
  runs the tag diagnostics, which included the locked context suite; it runs dev
  instead.
- **Pinned SaySo commit:** `7700db9` was only in a VM checkout
  (`ajax-optimization__nonstandard`, branch `incoming/nonstandard-evals`). It is
  now pushed to SaySo `ajax/nonstandard-evals`.
- **RC5 export verified on the VM:** converting v11
  (`20261026T020054Z-cd2a0275-full/final`) with `convert-v7` reproduces the shipped
  F16 byte-for-byte (`be029dbd…`). `export_gguf.py --quant Q4_K_M`
  (llama-cpp-python 0.3.36) then reproduces the shipped Q4_K_M byte-for-byte
  (`728cae4c…`).
- **Runtime parity measured:** the same v11 Q4_K_M scores **91/120** on promotion
  through the integration's embedded path (`eval_checkpoint_cpu.py`, llama-cpp-python
  0.3.36), against **96/120** through `llama-server`
  (`20261026T205613Z-1cfc17c3`) and 95/120 for the FP32 checkpoint. 16 of 120
  outputs differ; 7 flip pass/fail (6 worse embedded, 1 better). So the runtime
  alone moves the score by 5 cases. The run took 41 min, sharing CPU with a GPU
  training job. Results: `/srv/llm/lfm/ajax-pipeline-measure/eval-v11-q4-embedded/`.
- **Not yet run:** canary recalibration (RC1 step 4) and the first full run
  under the new pipeline. Both need the GPU, which a v12 full run was using.

## Root causes at a glance

| RC | Root cause | Assessment items it explains |
|----|------------|------------------------------|
| RC1 | The canary is not the first 250 steps of a full run | 2, 3 |
| RC2 | No held-out selection set; the final checkpoint is assumed best | 1, 7 (independence) |
| RC3 | Gates don't encode the actual release decision | 4, 7 (policy) |
| RC4 | Run state is keyed by a mutable "current candidate" pointer | 5 |
| RC5 | The evaluated artifact is not the deployed artifact | 6 |
| RC6 | Experiment inputs are not pinned | 8 |

Order: RC6 (submodule push) → RC4 → RC1 → RC2 → RC5 → RC3. RC4 and RC6 come
first because every later record depends on them. RC3 comes last because it
needs RC5's GGUF scores.

---

## RC1: the canary is not the first 250 steps of a full run

**Evidence.** `--canary` sets `max_steps = 250` (`train_unsloth_full.py:34-35`).
The HF Trainer then builds the cosine schedule and the 10% warmup
(`sayso:25`) from a 250-step horizon. The canary therefore warms up for 25
steps and has decayed to about 0 by step 250. The full v12 run is 40,000 rows
× 2 epochs / 32 ≈ 2,500 steps with 250 warmup steps, so at step 250 it has only
just reached its 5e-5 peak. The two runs follow different trajectories.

**Fix.**
1. In canary mode, leave `max_steps=-1`. The Trainer then computes the same
   horizon and warmup as full training from `num_train_epochs`.
2. Add a `--stop-at-step` argument (default 250 in canary mode). It drives a
   small `TrainerCallback` that sets `control.should_training_stop = True` once
   `state.global_step` reaches that step. The existing
   `trainer.save_model(f"{out}/final")` still runs afterwards.
3. Record the stop step and the learning rate at the stop in the canary record,
   so later readers can see what was actually measured.
4. Recalibrate `training_baseline.json` → `eval.canary`. The current 0.29
   baseline came from canaries on the old schedule; a canary stopped at peak LR
   will score differently. This takes two canaries on known datasets.
5. Policy (documented in `SAYSO_TRAINING_LIFECYCLE.md`): the canary is a smoke
   and compatibility gate. It is never used to compare datasets. Dataset
   comparisons go through RC2's dev suite on full-run checkpoints.

**Verify.** Add a unit test that stubs `Trainer` and checks that canary mode
passes `max_steps=-1`, installs the stop callback, and gets the same
`TrainingArguments` as full mode apart from `output_dir`. This is a new file
under `training/tests/` and needs your OK. On the VM, compare the logged LR at
steps 25, 100 and 250 between a canary and the first 250 steps of a full run:
they should match.

**Done when** the canary's LR curve matches the full run's prefix, and the
canary baseline has been recalibrated.

---

## RC2: no held-out selection set; the final checkpoint is assumed best

**Evidence.** The trainer has no `eval_dataset` (`train_unsloth_full.py:96-120`).
`save_total_limit=3` with `save_steps=200` deletes every checkpoint except the
last three, so even picking one by hand isn't possible. `cmd_train` hands
`{out}/final` to eval (`sayso:264`). The eval cases also lack separate
purposes: all 24 smoke cases are inside the 120-case promotion suite, and the
209 non-promotion cases (shadow, recipe-lock, gold, grounding) are only
diagnostics.

**Fix.**
1. **Split the eval cases into three roles** (SaySo repo, `evals/suites/`):
   - `promotion` (120) + `context`: **locked acceptance**. Unchanged and never
     used for selection.
   - `dev`: a seeded, category-stratified half of the 209 non-promotion cases.
     Used to select checkpoints and hyperparameters.
   - `regression`: the other half. Used as a gate against known failures
     coming back (RC3), never for selection.
   - `smoke`: rebuild as a 24-case subset of `dev`, so the canary loop stops
     exposing acceptance cases.
   Write the split as explicit ID lists so it's reviewable and its hash can be
   recorded.
2. **Check for train/eval leakage.** `check_quality_eval_overlap` reads
   `evals.cases.excluded_train_utterances()`. Confirm that it covers the `dev`
   and `regression` cases, not just tagged quality cases.
3. **Keep every checkpoint.** Set `save_total_limit=None` and
   `save_only_model=True`, which drops optimizer state: about 460 MB per
   checkpoint instead of about 2.3 GB, so about 6 GB for about 13 checkpoints.
   Before the first run, check free space on the over-committed thin pool. The
   trade-off is that intermediate checkpoints can't be resumed, and no current
   workflow resumes them.
4. **Add `./sayso select <train-id>`.** It runs `eval_checkpoint_cpu.py --suite
   dev` (CPU, no GPU lock) on each `checkpoint-*` plus `final`. It picks the
   checkpoint with the highest dev pass rate among those with zero
   zero-tolerance flags, preferring the earlier step on ties. It writes
   `selected_checkpoint` and the per-checkpoint dev table into the training
   record (RC4). `eval` and `promote-model` operate on `selected_checkpoint`.
   - Cost: about 13 checkpoints × about 105 cases on CPU. Measure one
     checkpoint first; if the whole run is over about 45 min, evaluate every
     other checkpoint.

**Verify.** A unit test that feeds `select` fake per-checkpoint summaries and
checks the tie-break and flag constraint. Also a check that the suite ID lists
don't overlap (dev ∩ regression ∩ promotion = ∅, smoke ⊆ dev).

**Done when** a full run yields a dev curve across its checkpoints and
acceptance runs on the selected checkpoint, not on `final`.

---

## RC3: gates don't encode the actual release decision

**Evidence.** `gates.yaml` describes itself as "Initial thresholds, not yet
calibrated". The current champion v5b was promoted by `manual_override` at 61%
(gate 70%) with a 0.6 false-action rate (gate 0.2), per `training_baseline.json`.
So the absolute gates aren't what actually decides promotion. The canary gates
only smoke at 20% (`eval_checkpoint_cpu.py:66-75`). Grounding and shadow aren't
gated anywhere.

**Fix.** Replace absolute thresholds with a **paired comparison against the
champion** on the same case IDs, scored from the GGUF endpoint (RC5):
1. Score the champion GGUF (`sayso-lfm-v5b-f16.gguf`, sha `7d4dff1c…`) once
   per suite hash and cache the per-case results. The existing
   `same_suite_hash` / `same_case_hash` fields show the pattern.
2. Acceptance on `promotion` + `context` + `regression`:
   - Context: keep the hard 100% gate.
   - Zero-tolerance safety flags: no new flagged cases relative to the champion.
   - Overall: the candidate wins more discordant cases than it loses on
     `promotion`, and is not significantly worse by an exact sign test on the
     discordant pairs.
   - Regression: no net loss beyond a small fixed slack (proposed: 2 cases).
   - Per category: no category loses more than 2 cases to the champion.
3. Keep the absolute `gates.yaml` thresholds as reported diagnostics, not as
   gates, until they're calibrated.
4. `--manual-override` still exists, but only with fresh paired results bound
   to the exact artifact (RC4/RC5).

**Verify.** Unit tests for the paired comparator using synthetic per-case
results: a win, a loss, a tie, and a new safety flag. Then replay the
v5b-vs-v3 history (73/120 vs 31/120): the comparator should accept v5b on
overall score and also surface its false-action-rate regression (0.2 → 0.6).

**Done when** `./sayso eval` outputs a paired report against the champion and
the pass/fail decision comes from it.

---

## RC4: run state is keyed by a mutable "current candidate" pointer

**Evidence.** `eval` and `promote-model` call `current()`, the newest
*candidate* (`sayso:270, 290`). `train` calls `write_current(record)`
(`sayso:256, 265`), which quietly repoints `current` at the promoted run.
Failure cases:
- Train A, then generate or canary B before `eval`: `eval` targets B.
- After `train`, a `./sayso canary` meant for B re-runs A.
- Retraining clears `final_eval_passed` but keeps `final_metrics`
  (`sayso:255`), so `promote-model --manual-override` can promote a new
  checkpoint using the old checkpoint's scores (`sayso:291`).

**Fix.** Separate dataset candidates from training runs:
1. `train` creates an immutable training record
   `training/runs/promotion/train/<train-id>.json` holding `train_id`,
   `dataset_run_id`, the dataset and rendered hashes, `out`, its checkpoints,
   and the manifest (RC6). It updates its own `latest_train.json` pointer and
   **no longer calls `write_current`**.
2. `select`, `eval` and `promote-model` take `--train-id`, defaulting to
   `latest_train.json`, and never read `current()`.
3. Each eval result is stored under the training record, keyed by
   `(checkpoint path, artifact sha256)`. `promote-model` recomputes the
   artifact sha and refuses unless a passing (or explicitly overridden) eval
   exists for that exact sha. A retrain is a new `train_id`, so stale metrics
   can't exist.
4. `promoted_model.json` records `train_id`, the checkpoint, the GGUF sha and
   the eval record ID.

**Verify.** Extend the CLI tests in the style of `test_sayso_canary_cli.py`
with three scenarios: (a) train A → generate B → eval targets A; (b) train →
canary still targets the newest candidate; (c) retrain →
`promote-model --manual-override` fails until a new eval exists. These are
additions under `training/tests/` and need your OK.

**Done when** all three scenarios pass and `current.json` is only written by
`generate`, `validate` and `canary`.

---

## RC5: the evaluated artifact is not the deployed artifact

**Evidence.** The gate scores HF Transformers in FP32
(`eval_checkpoint_cpu.py:36`). Deployment is the SaySo integration running
`llama-cpp-python>=0.3.33` (CPU wheel) on an **F16 GGUF**
(`custom_components/sayso/const.py:21-30`). `export_gguf.py` does nothing
without `--dry-run`: it returns 0 at line 36. `verify_llamacpp.py` sends one
payload. `publish_model.sh` uploads any GGUF without checking that it was
evaluated. The v5b champion was scored through a llama.cpp endpoint by hand,
so `EndpointAdapter` already works.

**Fix.**
1. **Make `export_gguf.py` work.** Run `convert_hf_to_gguf.py` from a pinned
   llama.cpp commit against the selected checkpoint, producing F16 only (what
   ships). Drop the default Q8_0/Q4_K_M until a quant actually ships. Print
   the sha256.
2. **Add `./sayso release-eval <train-id>`.** Start
   `python -m llama_cpp.server` from the same llama-cpp-python version the
   integration pins. Run it on CPU inside the unsloth container: no GPU lock,
   no prod Qwen downtime. Serve the GGUF and run `promotion`, `context` and
   `regression` through `EndpointAdapter`. Store the results under the
   training record keyed by GGUF sha (RC4); RC3's comparison reads them.
3. **Parity check.** Also run `dev` on the GGUF and report how often its
   outcomes differ from FP32 on the same checkpoint. A large gap means FP32
   dev selection can't be trusted. The threshold is set after the first
   measurement.
4. **Publish guard.** `publish_model.sh` refuses a GGUF whose sha isn't in
   `promoted_model.json`.
5. **Open item:** confirm that `llama_cpp.server`'s chat template and
   tool-call handling match the integration's request path (`client.py`).
   If they don't, the server isn't the deployed runtime and must be replaced
   by the integration's own call path.

**Verify (CPU-only, against known history).**
- Convert the v5b checkpoint (`/srv/llm/lfm/runs/sayso-lfm-v5-full/final`).
  The sha should match `7d4dff1c…`, or the difference should be explained
  (converter version).
- `release-eval` on v5b should reproduce 73/120 on promotion, run
  `20260925T174759Z-5ceb6471`.

**Done when** both v5b reproductions hold and `promote-model` can only
promote a GGUF sha that has release-eval results.

---

## RC6: experiment inputs are not pinned

**Evidence.** Records hash the dataset, rendered data and generation config
(`sayso:170-176`), but not the trainer code, base weights, tokenizer, resolved
training arguments, library versions, or eval suite hashes. The submodule
commit `7700db9` that `fa9083b` points at is rejected by the remote ("not our
ref"), so the eval suites behind recent scores can't be reproduced from git.

**Fix.**
1. **Now:** push `7700db9` to the SaySo remote. This is outward-facing, so it
   needs your go-ahead.
2. `train_unsloth_full.py` writes `{out}/manifest.json` containing: git commit
   and dirty flag, trainer script sha, sha of the base model `config.json` and
   weight files, tokenizer file shas, `TrainingArguments.to_dict()`, seed,
   rendered-data sha, and versions of torch, transformers, unsloth and
   datasets. `select` and `release-eval` add the eval suite hashes, the
   converter commit and the llama-cpp-python version.
3. `./sayso train` refuses to start if the work tree is dirty or the submodule
   commit isn't on a remote branch (`git -C vendor/SaySo branch -r --contains`).

**Verify.** A unit test for the manifest builder with a fake model directory.
Plus the dirty-tree refusal.

**Done when** two runs can be diffed field by field from their manifests.

---

## Phasing and GPU cost

| Phase | Work | GPU |
|-------|------|-----|
| 0 | Push submodule; rebase `ajax/pipeline` onto `ajax/generator-issue`; RC4; RC6 manifest; RC1 code | none |
| 1 | RC2 suite split (SaySo PR) + `select`; RC5 export, `release-eval`, v5b reproduction | none (CPU) |
| 2 | Two recalibration canaries (RC1) | 2 canaries |
| 3 | One full run on the current promoted dataset → `select` → `release-eval` → RC3 paired comparison vs v5b | 1 full run |

Phase 3 is the first full run under the new pipeline, and its result is the
new reference point for any future generator change.

## Decisions needed

1. **Dev/regression split:** a stratified half of the existing 209 cases
   (recommended, no new authoring), or write new dev cases and keep all 209 as
   regression.
2. **Smoke membership:** move it into `dev` (recommended), or keep it as a
   subset of `promotion`.
3. **Acceptance rule:** non-inferiority with a better point estimate
   (recommended, workable at 120 cases), or require a significant improvement,
   which will rarely trigger at this suite size.
4. **Test additions:** RC1, RC2, RC4 and RC6 each add tests under
   `training/tests/`. Approve once for this plan, or per change.

## Out of scope

Generator changes, hyperparameter sweeps (possible once `dev` exists),
quantized deployment, and wake-word training.
