# Plan: fix the SaySo command failures (v6)

Status date: 2026-10-16. Branch: `ajax/model-training-failure`.

## Problem

After "prefer handling commands locally" was turned off in Home Assistant, every command
reaches the SaySo model. The production model (v5b) failed simple on/off commands on the
living-room satellite. Five HA traces and the satellite STT captures show it:

| Transcript | v5b sent |
|---|---|
| turn off the living room lights | HassTurnOff `domain: sensor`, "Living room lights" |
| turn on the living room tv / turn on the tv | HassTurnOn `domain: sensor`, TV |
| turn on the bedroom tv | HassTurnOn `domain: sensor`, a person entity (there is no bedroom TV) |
| turn on the cd | no call; a junk reply |
| turn the livingium lights on / turn the little lights on | invented names |

## Root causes

1. **Look-alikes never appeared in training.** The v5 static context listed only controllable
   devices. Live Assist lists 27 non-controllable entries out of 48, with `TV` right next to
   `TV Active app` (sensor). The model copied the domain from the neighbouring line.
2. **The real home never trained refusals.** `REAL_HOME_EXCLUDED_FAMILIES` excludes
   absence, unsupported, unavailable and junk for the real home, so "turn on the bedroom tv"
   (absence) and "turn on the cd" (junk) on the deployment home had no examples.
3. **Requests that name a context-only entry were never trained.** 0 of 40,000 v6 rows have a
   user naming a context-only entry (for example "turn on Person One"), yet the `context`
   gate requires refusals for them.
4. **The eval gate measured a different prompt shape.** The runner rendered devices only, and
   the 120 promotion cases run on 5 devices-only households.
5. **The pipeline could not run v6.** The real-home quota starved the generator, the canary
   gate was uncalibrated, and there was no host deployment.

## Done

- Generator: context-only entries in synthetic and real homes; "Unassigned" removed with a
  `placeholder_area` guard; domain kept on labels; context layout seeded by the recipe;
  `token_budget` 8000; synthetic fallback when the real home saturates; production
  live-context wording. Commits `f0358f5` to `8e20dae`.
- Eval gate (SaySo #134, #135): the runner renders context entries and the production prompt;
  `live_home_1`; zero-tolerance `wrong_domain` and `context_entity_targeted`; a 16-case
  `context` suite (11 look-alikes plus the 5 real transcripts) gated at 100%; submodule at
  `6e536f1` (`7f79e3e`).
- Pipeline: `./sayso eval` requires the promotion and `context` suites; `generate` does not
  touch the GPU unless the recipe paraphrases; `REVISION` records the commit in `git archive`
  copies.
- v6 candidate `20261016T000918Z-7c1ef4ea`: generated, validated and audited (0 context-only
  targets, 0 wrong domains, 0 "Unassigned", 0 of 16 eval utterances in the corpus). Canary:
  smoke 7/24, `context` 7/16; failed the uncalibrated gate.

## 1. SaySo pipeline (`./sayso` and the LLM VM)

| # | Change | Why |
|---|---|---|
| 1.1 | Canary profile: `eval.canary` baseline and gates; `./sayso canary` passes `--profile canary`; `--reuse-checkpoint` re-judges an existing 250-step checkpoint | 250-step canaries score 8/24 and 7/24 against a 70% bar meant for a finished model. The final promotion and `context` gates stay strict. |
| 1.2 | One stable host deployment, `/srv/llm/lfm/ajax-optimization`, pointing at the current copy; the `HOST_TRAINING` default changes to its container path | Per-commit `git archive` copies plus an env var today; `/workspace/host/ajax-optimization` is not mounted in `unsloth`. |
| 1.3 | Repoint the cockpit backend `/srv/llm/bin/gpu-ctl` to 1.2 and the v6 recipe (backup, atomic swap) | It still points at the old `/srv/llm/sayso` and `full_sft_v5.yaml`, so its Data generation, train and eval buttons run old code. |
| 1.4 | Restore serving when `train` fails | `settle_gpu` leaves llama stopped on that path. |
| 1.5 | Define the rollout after `promote-model`: GGUF export, HA runtime model path, rollback | `promote-model` only writes a record. HA's sayso option names Gauntlet-v1 but the runtime loads v5b. |
| 1.6 | Record the v6 run in the training log, README and results | AGENTS.md. |

## 2. Data generator

| # | Change | Why |
|---|---|---|
| 2.1 | Before training: hard negatives in which the user names a context-only entry (person, sensor, sun, select) and the label is a refusal | 0 such rows in v6; the `context` gate has two at 100%. |
| 2.2 | Before training: absence and junk (and unsupported if feasible) on the real home, with a share cap. Real-home absence must not remove devices; it asks for a device type an area really lacks, so the static context stays byte-identical to production. | "turn on the bedroom tv" and "turn on the cd" on the deployment home. |
| 2.3 | Before training, if cheap: STT noise modelled on the real transcripts (phonetic slips such as "livingium" and "little lights", a leftover wake-word prefix such as "kda") | Satellite captures. Exact-match exclusion keeps eval utterances out of training. |
| 2.4 | Measure, then decide: real-home share (8.1% vs 10%) and 64-entity homes (31% vs 42% in v5) | Change only if the evals point at them. |
| 2.5 | Guard: a test that the training prompt equals the captured production prompt, and a recapture procedure for HA upgrades | HA wording already changed once. |
| 2.6 | Deferred: a generation progress file for the cockpit | Decision 2026-10-16. |

## 3. Everything else

| # | Change | Why |
|---|---|---|
| 3.1 | Promotion households get context-only entries (SaySo web task; the brief is ready); re-score v5b as the new checkpoint baseline | All 120 promotion cases run on devices-only homes. |
| 3.2 | A cross-check test that the training prompt equals the vendored eval-runner prompt | Two copies of the prompt already drifted once. |
| 3.3 | HA: fix the "Living room lights " trailing space; optionally re-enable local handling during the ~15 h training turn | Prod llama is down for the whole turn. |
| 3.4 | Release gate: replay the recovered transcripts through live HA with the new model before switching, then watch the SaySo traces | The test that matches production. |
| 3.5 | PR for `ajax/model-training-failure`; fast-forward the stale local `main` | The stale `main` makes the cockpit diff wrong. |

Not doing: a runtime domain-correction safety net in the SaySo integration (decided
2026-10-16).

## Order

1. Finish 1.1.
2. 2.1 and 2.2 (2.3 if cheap), then regenerate v6 (~15 min), validate, canary (~1 h).
3. 1.2 and 1.3, so the cockpit points at the right code before training.
4. `promote-dataset`, then `train` (~15 h, measured at 11.8 s/step). In parallel: 3.1, 3.2,
   2.5, 1.4, 1.5.
5. `eval` (promotion plus `context`), `promote-model`, the 3.4 replay, the 1.5 rollout, then
   1.6 and 3.5.
