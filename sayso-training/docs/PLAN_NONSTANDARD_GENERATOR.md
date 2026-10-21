# Plan: nonstandard voice commands in the generator

Source: the "SaySo — Updated Nonstandard Voice Command Training List" (24
families, P0–P2), 2026-10-20. The held-out evals are in SaySo
`evals/cases/nonstandard_v1.jsonl`; their plan is in SaySo
`docs/PLAN_nonstandard_evals.md`. The user approved this expansion of the eval
and training corpora on 2026-10-20.

## Cross-reference with the current generator

"Listed" is the status in the source list. "Actual" is what
`training/generators` produces today.

| # | Family | Listed | Actual | Evidence |
|---|---|---|---|---|
| 1 | TV volume up/down | Existing | Existing | `media_players.volume_up/down` → `HassSetVolumeRelative` `"up"/"down"` |
| 2 | TV volume by amount | Expand | Expand | Schema accepts an integer `volume_step` (−100..100), and the OHF grammar has `volume up by N%`, but `tools.build_volume_relative` emits only `"up"/"down"` |
| 3 | Satellite volume up/down | Expand | **New** | No home has a satellite media_player entity, and there is no self-reference phrasing ("speak louder", "your volume") |
| 4 | Satellite absolute volume | Expand | **New** | Same as #3. `HassSetVolume` exists for speakers |
| 5 | TV vs satellite | New | New | Needs #3 plus contrast pairs with `forbidden_entities` |
| 6 | Timer creation | Existing | **Partial** | `gold.py` forces a `name` on every timer call except cancel-all, so "Set a timer for 30 seconds" is never labelled unnamed |
| 7 | Extend timers | Expand | Partial | `HassIncreaseTimer` exists but is always named |
| 8 | Shorten timers | Expand | Partial | `HassDecreaseTimer` exists but is always named |
| 9 | Unnamed timer commands | Expand | **New** | No unnamed pause/cancel/status/increase rows (same `gold.py` cause) |
| 10 | Timer ambiguity | New | New | Timers have no state, and tool results are always `{"result": "Success"}` |
| 11 | Timer corrections | New | New | No history before the utterance |
| 12 | Timer failure handling | New | New | No error tool results |
| 13 | TV playback | Existing / expand | Existing | pause/unpause/next/previous exist; "stop the show" phrasing is missing |
| 14 | TV navigation | New | New, **tool needed** | No remote or key-press tool in the pinned v2 schema |
| 15 | TV app switching | New | New, **tool needed** | No source/app tool. `HassMediaSearchAndPlay` `media_class: app` exists in the schema; Roku support is unverified |
| 16 | Relative brightness | New | New | `HassLightSet` is absolute only. The synthetic GetLiveContext result carries no attributes |
| 17 | Contextual references | New | New | Multi-turn rendering exists only for the clarify→reply `follow_up` family |
| 18 | Correcting targets | Partial | Partial | The existing `correction` family is a tool-error retry, not a user correction |
| 19 | Assistant speech controls | New | Out of path | No tool. "Stop talking" is handled by the satellite; "say that again" is a plain reply |
| 20 | Multiple independent actions | Partial | Partial | `multi_action` exists, but timers (`targeting_modes=("context",)`) never join a multi-action |
| 21 | Group with exclusions | Existing | Existing | `exclusion` family. Floor words ("downstairs") are not grounded because the static context has no floors |
| 22 | Restore previous settings | New | New | Needs history with the prior value |
| 23 | Cover positioning | New tool support | Refusal only | `covers.set_position` is `_blocked` ("no position tool in Assist schema") |
| 24 | Conditional actions | New | New | Needs read-then-act turns |

Phrases alone do not fix #3–#5 (no self entity), #10–#12 (no timer state or
errors), #14/#15/#23 (no tool), or #16/#22/#24 (no realistic tool results).

## Verify before building

1. **HA tool-result shapes.** Capture these from a live SaySo trace:
   - intent success (`IntentResponse.as_dict()`)
   - GetLiveContext (`brightness` 0–255? `volume_level` 0–1?)
   - timer errors (`TimerNotFoundError`, `MultipleTimersMatchedError`)

   The evals assume these shapes. Fix the eval fixtures if the trace differs.
2. **Satellite media player naming.** Confirm the Linux Voice Assistant
   media_player entity name and whether it is exposed to Assist. The evals use
   `Kitchen Satellite`.
3. **Self-identification at runtime.** Nothing in the prompt tells the model
   which media player is "you". Recommended: add `satellite_media_player: <name>`
   to the area-context block (`sayso_contract.area_context`, set by
   `conversation.py` from `satellite_id`). This is a runtime contract change and
   needs its own PR. Until then the model has to infer from the name and the
   satellite area.
4. **Roku app path.** Check whether `media_player.play_media` with
   `media_content_type: app` is reachable from any v2 tool. If not, #15 stays a
   refusal.

## Generator changes

Each slice stays at four files or fewer and verifies on its own (router QWEN gate).

### Phase 1: P0 (no schema change)

1. **Unnamed timers** (`gold.py`, `utterances.py`). Make the forced timer `name`
   probabilistic, about 40% unnamed, for start/pause/unpause/cancel/status/
   increase/decrease. Add bare templates: "pause my timer", "how much longer",
   "add two minutes", "cancel the timer". The OHF `[the|my] timer` sentences
   already cover most of these.
2. **Volume by amount** (`tools.py`, `utterances.py`). Let
   `build_volume_relative` emit integer steps (±5..±30), with templates
   "up 10 percent", "down by five", "a lot louder" (keep vague words on
   `"up"/"down"`).
3. **Real tool results** (`rendering.py`). Replace `{"result": "Success"}` and
   the synthetic `{"entities": [...]}` with the shapes verified in step 1,
   including GetLiveContext attributes. Nearly every row passes through this,
   so dispatch it alone.
4. **Satellite self-volume** (`homes.py`, `capability_registry.py`,
   `utterances.py`).
   - In about 30% of homes, add a speaker media_player named after the
     satellite area (`<Area> Satellite`).
   - Add self-reference templates ("speak louder", "you're too loud", "set your
     volume to 40%", "half volume") labelled to that entity.
   - Add contrast pairs with a TV in the same home ("turn yourself down, not the
     TV", and the reverse), listing the other device in `forbidden_entities`.
5. **`timer_context` family** (`planning.py`, `row_generation.py`,
   `rendering.py`, config). Prefix history: one or two `HassStartTimer` turns
   with success results. Then one of:
   - bare command → unnamed call
   - two active timers + bare command → `MultipleTimersMatchedError` → trained
     clarification naming both timers from history
   - correction ("no, add five not ten" → `HassDecreaseTimer 5`; "make it 15" →
     `HassIncreaseTimer` by the difference)
   - missing or expired timer → `TimerNotFoundError` → trained honest refusal
     ("There's no pasta timer running")

   Reuse the `follow_up` multi-turn renderer, with `train_on_turn=False` on the
   history.

### Phase 2: P1

6. **Relative brightness.** Two-row pattern:
   - With no prior state, read first (`GetLiveContext` name + domain).
   - With the GetLiveContext result in context, set
     `HassLightSet brightness = clamp(round(b/255*100) + delta)`.

   Explicit deltas only. Vague words ("a little") need a fixed convention first
   (proposal: ±20). Decide that before labelling.
7. **`conversation_context` family.**
   - Previous-turn references ("turn it off", "make that louder", "the other
     one", grounded on a prior GetLiveContext answer).
   - User target corrections ("no, I meant the bedroom TV" → act on the new
     target only).
8. **TV remote and app refusals** (`capability_registry.py`). Add `_blocked`
   `navigate` and `launch_app` operations on `media_players` so the existing
   unavailable path produces refusals for "go home", "press back", and "open
   Netflix". Flip them to supported operations when a tool lands.
9. **Playback phrasing.** Add "stop the show/TV" → `HassMediaPause` and "resume
   the show" → `HassMediaUnpause`.

### Phase 3: P2

10. **Cross-domain `multi_action`.** Allow a timer start next to a device action
    ("mute the TV and start a 10-minute timer").
11. **`restore`.** History holds the state read and the change. Label: set back
    to the earlier value (`HassSetVolume`, `HassLightSet`).
12. **`conditional`.** First turn: GetLiveContext. With the result: act if the
    condition holds. If it does not hold, a spoken "already off". That branch
    has no eval response type yet; train it but don't score it.
13. **Cover position.** Keep the refusal and add "halfway" and "to 30 percent"
    phrasing. A real `HassSetPosition` needs the schema/tool contract bumped first.

### Out of scope

- Delayed actions, relative fan or thermostat changes, replay/seek, and
  per-device capability failures (from the source list).
- #19 assistant speech controls.

## Allocation

Start a `full_sft_v8.yaml` from v6. Take shares from `ordinary` (0.22 → about
0.12):

| Family | Share |
|---|---|
| `timer_context` | 0.04 |
| `conversation_context` | 0.03 |
| satellite volume (through `ordinary`/`settings` homes) | 0.02 |
| relative brightness | 0.01 |

`multi_action` keeps its share but now includes timer pairs. Run
`./sayso validate` and the allocation feasibility check before generating.

## Contract notes

- Never train on eval utterances. `excluded_train_utterances()` covers the
  utterance only, so do not copy the eval history lines into templates either.
- The evals relabel #14/#15/#23 from `refusal` to `action` only when a tool
  ships. Keep the generator's refusal labels in sync with that.

## Verification

- Unit tests per slice in `training/tests` (row shape, label rules, unnamed
  timer share, integer volume steps, multi-turn rendering).
- `./sayso validate` on a small smoke corpus per phase.
- Baseline: run `python -m evals.cli run --tag nonstandard` on the current
  champion before any retrain. Compare after the v8 canary, per
  `nonstandard_family_NN` tag. Promotion and smoke gates must not regress.
