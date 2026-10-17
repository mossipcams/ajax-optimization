# Plan: a full fine-tune that beats the stock instruct model (v7)

Status date: 2026-10-17. Branch: `ajax/model-training-failure`. Follows
`PLAN_v6_COMMAND_FAILURES.md` (steps 1-3 of its redesign are done: labels on the
rendered home, full-input dedup, real-home contrast refusals).

## Goal

One full fine-tune of LFM2.5-230M that:

1. beats the stock `LiquidAI/LFM2.5-230M` instruct model on every suite below;
2. beats v5b (the served model) on every suite;
3. passes the final promotion gates: `context` 100%, zero `wrong_domain`, zero
   `context_entity_targeted`.

## Where we are (same eval code, 2026-10-17)

| Model | smoke /24 | recipe-lock /38 | gold /35 | shadow /100 | grounding /36 | context /16 |
|---|---|---|---|---|---|---|
| Stock instruct, no fine-tune | pending | pending | pending | pending | pending | pending |
| v5b, Base, full run (served) | 13 | 20 | 19 | 49 | 29 | 6 |
| v6 canary, Base, LR 2e-5, 250 steps | 9 | 12 | 13 | 35 | 21 | 11 |
| v6 canary, instruct, LR 2e-4, 250 steps | 5 | 19 | 16 | 61 | 23 | 13 |
| **v6 canary, instruct, LR 5e-5, 250 steps** | **17** | 17 | **22** | **65** | 24 | 12 |

"Full fine-tune" means every weight is updated. Base vs instruct is only the
starting checkpoint; instruct at 5e-5 is the configuration to beat and the one
this plan trains.

The 2e-4 run collapsed onto `domain=['scene']` for lights (11 of 19 smoke
failures) although `scene` is 259 of 13,530 light-domain labels: a learning-rate
instability, not data.

## What the best run still gets wrong (instruct, 5e-5, 11 cases)

| Class | Cases | Cause in the data |
|---|---|---|
| Acts instead of clarifying | reading light, bedside lamp, "run the routine" | The real home has 0 clarify rows; synthetic clarify is 8% and never paired with a unique-name action on the same home |
| Invents a tool when the needed one is withheld | `light(...)`, `add_to_shopping_list(...)` | Withheld-tool refusals never share a request with the tool-offered action (dedup forbade it until 68a4496) |
| Alias distractor / exclusion | "sofa lamp"; "leave Pantry Light as it is" set brightness 0 | Few rows where the excluded entity is the nearest name |
| Real home: "Living room lights" sent to the TV | 1 | Real-home targets: TV 322, Living room lights 60. Targets follow tool count, not use |
| Real home: "Kitchen light" clarified instead of acted | 1 | `Kitchen light` is a holdout: visible, never a training target |
| Real home: "Turn off the Sun", "turn on the bedroom tv" acted | 2 | 161 context-only refusals and 149 absence rows, none paired with the action they contrast with |

## 1. Training recipe

| # | Change | Where | Check |
|---|---|---|---|
| 1.1 | Start from `/srv/llm/lfm/models/LFM2.5-230M` (instruct, downloaded 2026-10-17; tokenizer and config identical to Base) | `sayso` `cmd_canary`/`cmd_train` pass `--model` from `TRAIN_MODEL`, recorded in the candidate record | record shows the model path and SHA of `model.safetensors` |
| 1.2 | LR 5e-5 | same, `--lr` from `TRAIN_LR` | record shows LR |
| 1.3 | Warmup 10% instead of 3% | `train_unsloth_full.py` `--warmup-ratio`, default 0.03 kept for old runs | trainer log |
| 1.4 | Checkpoint selection: score `checkpoint-*` saved every 200 steps on smoke + `context` (CPU), keep the best by `context` then smoke; the same scoring decides 1 vs 2 epochs | new `./sayso select-checkpoint` or a script under `training/scripts/` | selection table written next to the run |
| 1.5 | Keep `--max-steps 250` canary as the cheap pre-check, now with the 1.1-1.3 recipe | `cmd_canary` | canary record |

## 2. Data (generator)

| # | Change | Where | Acceptance (audit on the 40k candidate) |
|---|---|---|---|
| 2.1 | Production model trains on every real entity: holdouts become an eval-only option (`real_home.holdouts: false` in the v6/v7 recipe) | `generators/real_home.py`, `row_generation.py`, recipe | `Kitchen light`, `Christmas tree`, `Desk Echo Dot`, `Rocky room cam` appear as targets |
| 2.2 | Balance real-home targets by expected use: cap per entity (`real_home.entity_cap`) and weight the satellite's common commands (living room lights, TV, porch and outdoor lights) | recipe + `pick_target` usage seeding | TV ≤ 2x Living room lights; no entity > 10% of real-home targets |
| 2.3 | Real-home contrast groups on one rendered context: act on `TV`; refuse `TV Application` / `TV Active app`; refuse an absent "bedroom TV"; refuse `Sun` / `Person One`; act on `Living room lights` (switch) | `row_generation.py` real-home path (one scenario, several rows sharing `contrast_group_id`) | ≥ 300 groups; each group's rows share a system prompt byte-for-byte |
| 2.4 | Withheld-tool pairs: the same request with the tool offered (action) and withheld (refusal), for LightSet, todo, media, climate; plus a few invented-tool negatives (home-llm's hallucinated-service idea) | `unavailable` family path | ≥ 1,000 pairs; preflight conflict rate 0 (key includes withheld tools, SaySo #136) |
| 2.5 | Clarify pairs on synthetic homes: unique name → action, shared alias/area → clarify, same home | `clarify` family path | ≥ 1,000 pairs |
| 2.6 | Exclusion rows where the excluded entity is the nearest name to the targets | `exclusion` family path | ≥ 500 rows; no label touches the excluded entity |

Not in v7 (worth doing after): home-llm style failure recovery (bad call, real HA
error text, retry) and "already in that state" refusals; varied final replies
instead of "Done."; unnamed timer calls (`HassPauseTimer()`), which only the
`regressions` suite covers. Train on home-llm data: no; 15% of its named calls
are wrong for Home Assistant (entity IDs as names, timer `start` mapped to
`HassVacuumStart`, humidifier intent on thermostats).

## 3. Verification

- Unit tests per change; full offline suite (baseline: 378 pass, 8 offline
  tokenizer failures, 1 known `test_capped_rows_are_replaced_not_dropped`).
- Corpus audit script (fabricated targets 0, eval utterances 0, actions on
  context-only 0, the 2.x acceptance counts) and `./sayso validate`.
- Canary (instruct, 5e-5, warmup 10%, 250 steps): must not regress the table
  above on any suite by more than 1 case, and must improve `context`.
- Full run, checkpoint selection, then the full promotion + `context` eval.

## Order and cost

1. 1.1-1.3 (code, ~1 h) and 2.1-2.6 (code, ~1 day), committed with tests.
2. Regenerate on the host clone (~25 min), audit, validate.
3. Canary (~1 h GPU, prod llama-server down).
4. Full run (~15 h GPU at 11.8 s/step, prod down) with checkpoints every 200 steps.
5. Checkpoint selection on CPU (~20 min per checkpoint scored).
6. Final eval; if all three goals hold, `promote-model`, the live-HA replay of the
   recovered transcripts (plan v6 3.4), then the GGUF rollout (plan v6 1.5).

## Risks

- Instruct starts can forget general chat; SaySo only needs tool calls and short
  replies, so this is acceptable, but watch `shadow`.
- 2.2 weights come from 406 satellite captures over two weeks; revisit with more.
- The `context` suite's household is the real home, so 2.1 trains on the eval's
  own entities. Utterances still never overlap (exact-match exclusion), but this
  measures in-home performance, which is the production case.
