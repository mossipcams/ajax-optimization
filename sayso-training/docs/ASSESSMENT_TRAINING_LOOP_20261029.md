# Assessment: is the SaySo LFM training loop working?

Date: 2026-10-29. Report only: no pipeline, generator, eval or runtime change.
Read against `main` at `179f4ce` while the v13 full run
(`20261028T203602Z-342208c4`) was still training. Sources are the docs under
this directory, [TRAINING_LOG.md](../training/TRAINING_LOG.md), the `sayso`
CLI, `train_unsloth_full.py`, the generator recipes, the SaySo eval suites, and
the counterfactual probe reports listed below.

## Verdict

The training mechanics are right and most of them were earned by earlier
failures. The loop around them is not working:

- it changes the corpus against a measurement that cannot rank candidates;
- each run spends most of its 15 to 20 hours after the model has stopped
  improving;
- the last two corpora dropped the one training signal that the probe shows
  works.

Twelve corpus versions since v5 and neither v12 nor v13 beats v11 on smoke or
context.

| Model | Context (16) | Smoke (24) | Source |
|---|---:|---:|---|
| v11 champion Q4_K_M | 13 | 19 | `/srv/llm/lfm/v11-champion-eval` |
| v12 checkpoint-800 | 5 | 20 | `runs/sayso-promotion/v12-full-ckpt800-eval` |
| v12 checkpoint-1000 | 9 | 17 | `runs/sayso-promotion/v12-full-ckpt1000-eval` |
| v13 checkpoint-1200 | 12 | 15 | `runs/sayso-promotion/v13-full-ckpt1200-eval` |

## What is right

- Full SFT from Base, completion-only loss, and the pinned Base template
  rendered identically at train and serve time (`adapters/lfm.py`).
- The release eval scores the shipped Q4_K_M GGUF through the integration's
  own llama-cpp-python path. The runtime alone moves promotion by 5 of 120
  (91 embedded against 96 through `llama-server`,
  [PLAN_TRAINING_MEASUREMENT.md](PLAN_TRAINING_MEASUREMENT.md)).
- Paired comparison against the champion, hashed manifests, immutable train
  records.
- The counterfactual probe (`training/scripts/counterfactual_probe.py`) is the
  right primary metric for "does the model read the context".
- The Run 008 name-memorization diagnosis and its fix held.

## Findings, in priority order

### 1. Pairing works, and v12 and v13 stopped using it

The probe's `tool_removed` pairs keep the utterance and delete the expected
tool from the catalog. "Sticky" means the output was byte-identical after the
deletion. Splitting those pairs by whether the tool is in
`WITHHELD_PAIR_TOOLS` (`generators/pairs.py`: `HassLightSet`, media pause,
unpause, next, previous, set volume, relative volume, climate set temperature),
which the recipes pair at `pairs.withheld_tool: 1.0`:

| Checkpoint | Paired tools sticky | `HassTurnOn`/`HassTurnOff` sticky (unpaired) |
|---|---:|---:|
| v12 checkpoint-600 | 1 of 21 | 17 of 18 |
| v12 checkpoint-1000 | 0 of 17 | 16 of 18 |
| v13 checkpoint-600 | 1 of 22 | 19 of 24 |

Where training supplied a same-utterance sibling with the tool withheld, the
model reads the catalog. Where it did not, the model ignores it. On and off
carry most real use and have no pairs; device absence has no pairs at all.
The `contrast` family was 0.05 in `full_sft_v11.yaml` (the champion) and 0.0 in
v12 and v13. [PLAN_COUNTERFACTUAL_PAIRS.md](PLAN_COUNTERFACTUAL_PAIRS.md)
asked for 0.20. v13 trains 0.03 absence rows against 0.44 ordinary rows, which
teaches the prior that a named device exists.

Overall probe rates for reference (60 pairs per kind, 10 clarify pairs):

| Checkpoint | Appropriate change | Sticky | Original pass |
|---|---:|---:|---:|
| v12 canary (full-run schedule) | 0.09 | 0.44 | 0.35 |
| v12 checkpoint-600 | 0.19 | 0.38 | 0.50 |
| v12 checkpoint-1000 | 0.18 | 0.40 | 0.55 |
| v13 canary | 0.00 | 0.38 | 0.45 |
| v13 checkpoint-600 | 0.10 | 0.46 | 0.41 |

Reports: `counterfactual_probe.json` under
`/srv/llm/lfm/runs/sayso-promotion/*-cfprobe/`. The per-tool split reads each
`tool_removed` row's `original_output.calls[0].name` and buckets it against
`WITHHELD_PAIR_TOOLS`.

### 2. The measurement has no power

Suite sizes: promotion 120, context 16, smoke 24, and about 130 each for dev
and regression (the remaining 263 cases split in half). A sign test on a
handful of discordant pairs decides nothing, and 17 of the 110 v12 canary
failures were wrong expectations
([EVAL_CONTRACT_V12.md](EVAL_CONTRACT_V12.md)). The 5% group hold-out added in
`179f4ce` only produces a loss number. Scoring held-out generator rows
behaviorally, with held-out homes and vocabulary, would give thousands of cases
for free. [PLAN_V6_DATASET_ARCHITECTURE.md](PLAN_V6_DATASET_ARCHITECTURE.md)
proposed that on 2026-09-25 (val-ID, val-OOD, real-dev) together with the
in-distribution versus out-of-distribution check that decides whether to spend
on data or on model and prompt. Neither was built.

### 3. Most GPU time is spent after the plateau

- v12: probe rates flat from step 600 to 1000; smoke 20 of 24 at step 800,
  17 at step 1000.
- Run 013: step 2,500 to 5,000 lost 5 of 293. Run 005: epoch 3 scored below
  epoch 2.
- v13: held-out loss 0.078 at step 200, 0.045 at step 1,200, still falling
  while behavior does not move.

A v13 run is 4,332 steps at about 12.7 s per step, about 15 hours, with
`per_device_train_batch_size=1`, 32-step accumulation and no packing for a
230M model on 24 GB. One epoch at batch 4 to 8 should be a 2 to 4 hour run.
Verify batching with the 20-step smoke the training plan already prescribes
(loss parity against batch 1, finite gradients, padding masked in the LFM2
convolution blocks).

### 4. The scorer judges strings, not outcomes

`evals/scorer.py` requires an exact multiset match of canonical calls, so a
string `domain` fails against a list and `device_class` on a uniquely named
lock fails. The same gap makes generator labels hard to trust. One fix covers
both: resolve expected and emitted calls through Home Assistant's intent
matcher offline against the case's home and compare the affected entities.
The training venv already carries Home Assistant 2026.10.4.

### 5. Nothing can promote right now

- The context suite is gated at 100% and the champion scores 13 of 16. Cases
  06, 09 and 11 fail for every model; 06 targets `Kitchen light`, which is
  held out of training by design.
- `lfm_parse._parse_value` calls `int()` on `-` and raises a bare
  `ValueError` that nothing catches. It crashed the v13 step-1000 probe and
  will crash `./sayso eval`.

### 6. Real usage has never entered the loop

Five real transcripts sit in the context suite. The roughly 130 satellite
transcripts proposed as a hand-labeled dev set on 2026-09-25 never became
one. Only STT error shapes from the logs are used
([PLAN_STT_LOG_CORRUPTIONS.md](PLAN_STT_LOG_CORRUPTIONS.md)). Paraphrasing is
disabled in the v13 recipe, so every utterance comes from one Hassil grammar
and every eval failure becomes one more template (0 of 210 pause rows said
"hold", 0 of 370 start-timer rows named a room). LLM-generated labels and
judging were rejected in the v6 review, and that stands. Paraphrasing
utterances while the generator fixes the label is a different proposal and
needs a decision.

## Recommended order

1. Make the context gate relative to the champion and fix the parser crash.
   Run `./sayso select` on the v12 and v13 checkpoints; it is CPU-only and
   will show the plateau.
2. Build the dev set: held-out generator homes and vocabulary scored
   behaviorally, mechanically derived counterfactual pairs, and the real
   transcripts. Score v11 and v13 on it once to get in-distribution against
   out-of-distribution.
3. Switch the recipe to one epoch and batch 4 to 8 after a smoke.
4. Restore `contrast` at 0.20 with on, off and device absence covered. One
   change, one short run, read the probe. If sticky does not move, add a
   pairwise preference stage on the same pairs.
5. Only then revisit model size or prompt shape, with a measurement that can
   see the answer.

## Decisions needed

- Paraphrase utterances with the local 27B while labels stay generator-fixed,
  or keep the grammar only.
- Install Home Assistant's intent matcher as an offline label and scoring
  oracle (the v6 plan's executability check).
- Whether a one-off 700M run on the v13 corpus is worth a GPU day once the
  dev set exists, to bound what the 230M model can reach.
