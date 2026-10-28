# V12 canary: evaluation contract findings

Date: 2026-10-28
Checkpoint: v12 canary final (`runs/sayso-promotion/20261028T073043Z-3c85997b-canary/final`, 250 steps), not the full-run checkpoint-1000. Canary eval: `/tmp/v12-corpus/20261028T073043Z-3c85997b-canary-eval-20261028T084144Z/`
Result: 249 cases, 139 passed, 110 failed (`outcomes.jsonl` across six subdirectories, 24 + 38 + 35 + 100 + 36 + 16 = 249 lines; re-counted here, matches).

Failure outcomes across the 110: `unexpected_tool_call` 67, `missing_tool_call` 26, `unknown_tool` 7, `wrong_response_type` 6, `invalid_arguments` 4.

Runtime used for verification: the training venv at `training/.venv` (Python 3.14, Home Assistant 2026.10.4, `home_assistant_intents` 2026.10.27). The vendored SaySo `pyproject.toml` pins `homeassistant>=2026.8.3` (`vendor/SaySo/pyproject.toml:11`). The line references below are for 2026.10.4, which satisfies that floor. They were not re-read against an 8.3 install.

Scorer: `vendor/SaySo/evals/scorer.py`. `_canonical` (lines 48-51) compares `name` and `arguments` exactly. `_affects` (58-73) is name-, area- and domain-based. `classify` (76-100) requires exact multiset equality of canonical calls for `VALID_*` outcomes.

## Findings table

Path prefixes: `HA:` is `training/.venv/lib/python3.14/site-packages/homeassistant/`. `ICH:` is `home_assistant_intents`, not used below.

| Group | Cases (count) | Model emitted | Eval expects | Runtime evidence | Verdict | Proposed action |
|---|---|---|---|---|---|---|
| 1. GetLiveContext domain+name vs name only | 9: recipe_lock_06_a, recipe_lock_06_b, recipe_lock_06_d, v3_quality_gold_status_climate, v3_shadow_status_0085 to 0089 | `GetLiveContext` with string `domain` plus `name` (9 of 9) | 06_a: name only. Other 8: `domain` as a list plus `name` | `HA:components/homeassistant/llm.py:281-282` wraps a string `domain` in a list. Same filter when the name is unique. Brief cited `helpers/llm.py:281-282`; that line is `_tool_integration_domain`, so the path is corrected here. | Eval too strict | Scorer canonicalizes GetLiveContext: string domain equals list domain, and an extra domain matching the named entity's domain is equivalent. Scorer change, see below. |
| 2. HassCancelAllTimers with area, expects no args | 3: v3_shadow_timer_cancel_0035 (Attic), 0037 (Veranda), 0039 (Annex) | `HassCancelAllTimers` with `area=<room>` | `HassCancelAllTimers {}` | `HA:components/intent/timers.py:976-980` declares an `area` slot. `_find_timers` at `timers.py:675-708` filters on `area_name` (698-708), so area changes which timers are cancelled. Brief named only Attic; the other two cases use Veranda and Annex, same pattern. | Eval expectation wrong | Expectation changes to `area=<room>` for each case. Eval-side change, see below. |
| 3. Global cancel-all with area=home | 1: v3_quality_gold_timer_cancel_all | `HassCancelAllTimers` with `area='home'` | `HassCancelAllTimers {}` | Same area filter, `timers.py:698-708`. `home` is not an area name, so the filter drops timers in real areas. | Model error. Keep eval strict. | Training gap: v12 had no global cancel-all rows. Covered by the timer wording slice (see mapping). |
| 4. Lock expectations requiring device_class | 5: recipe_lock_03_b, 03_d, 04_b (lock call), 05_a, 05_d | Lock call with name only, no `device_class` | Lock call with `device_class: ['door']` | `lock/__init__.py` defines no `device_class` (grep finds none). `HA:components/homeassistant/llm.py:89-96` lists `device_class` as an interesting attribute and `:168-175` copies it into exposed-entity info only when the state carries it. Locks do not, so it is absent for lock entities. Brief's claim that it is "not rendered in the model-visible context" is not verified beyond this. | Eval too strict when the lock name is unique | Scorer drops `device_class` when the named entity is unique. Needs the entity inventory to confirm uniqueness. |
| 5. Garage door vs door | 2: recipe_lock_01_b, recipe_lock_04_d | `device_class: ['door']` on a garage door | `device_class: ['garage']` | Brief says `device_class` changes which entities match. Not re-verified here: the HA device_class filter path was not located. `llm.py:96` and `:168-175` show `device_class` is exposed for covers that carry it. | Keep strict | None. |
| 6. Timer status by name=room | 4: v3_shadow_timer_status_0031 (Cellar), 0032 (Attic), 0033 (Porch), 0034 (Veranda) | `HassTimerStatus` with `name=<room>` (4 of 4) | `HassTimerStatus {}` | `HA:components/intent/timers.py:1107-1112` status slots: `name`, `area`. `_find_timers` `:688-696` matches `name` against the timer's own name. Correct call is `area=Cellar` (`:698-708`). | Model error (wrong slot). The eval's `{}` should also accept `area=<room>`, but that does not flip these cases. | Eval change: accept `area=<room>`. Training gap: v12 had no area status rows (timer wording slice). |
| 7. Start timer with room as name | 5: v3_shadow_timer_start_0022 (Solarium), 0023 (Studio), 0024 (Terrace), 0025 (Balcony), 0026 (Cellar); utterance "Set a N minute timer in the <room>" | `HassStartTimer` with `minutes` plus `name=<room>` (5 of 5) | `HassStartTimer {minutes: N}` | `HA:components/intent/timers.py:846-849` `HassStartTimer` slots: `hours`/`minutes`/`seconds` (required one-of), `name`, `conversation_command`. No area slot. Room is context only. | Model error. Eval expectation right. | Training gap: 0 of 370 v12 HassStartTimer rows mention a room. Follow-up timer slice. |
| 8. Pause answered with unpause | 4: v3_shadow_timer_pause_0027 (Terrace), 0028 (Balcony), 0029 (Cellar), 0030 (Attic); utterance "Hold the <room> countdown timer" | `HassUnpauseTimer` with `name=<room>` | `HassPauseTimer {}` | Separate intents: pause `timers.py:1054-1063`, unpause `timers.py:1079-1088`. Both have `name` and `area` slots, so `HassPauseTimer(area=<room>)` is also a correct call. v12: 0 of 210 pause rows use hold/freeze phrasing. | Model error (wrong intent and wrong slot) | Training: hold/freeze pause phrasing (timer wording slice). Eval: also accept `area=<room>`; no flips. |

Groups 1 to 8 account for 33 failing cases (9 + 3 + 1 + 5 + 2 + 4 + 5 + 4). The remaining 77 failures were not adjudicated one by one in this document. The brief attributes them to model error.

## Summary arithmetic

- Cases that flip to pass under the proposed eval changes: group 1 (9), group 2 (3) and group 4 (5), 17 in total, so 139/249 becomes 156/249.
- Groups 6 and 8 need an expectation change (also accept `area=<room>`). All eight cases emitted `name=<room>`, so none of them flip.
- Common model error across groups 6, 7 and 8 (13 cases): a room mentioned in a timer request is put into the timer `name` slot.
- Group 4 assumes each lock name is unique in its case context. The model emitted the full lock name (for example "Patio Side Door Lock" or "Joe's Guest Room Door Lock").

## Proposed vendor/SaySo changes (pending approval)

These are proposals only. None has been made. They live in the `vendor/SaySo` submodule, a separate repository, and need separate approval.

1. Scorer (`vendor/SaySo/evals/scorer.py`, `_canonical` lines 48-51): canonicalize GetLiveContext so a string `domain` equals a single-element list, and drop a `domain` that matches the named entity's domain. Covers group 1 (9 cases).
2. Scorer (`_canonical`): drop `device_class` from both sides when the named entity is unique. Covers group 4 (5 cases). Needs uniqueness from the entity inventory.
3. Expectations for `v3_shadow_timer_cancel_0035`, `0037`, `0039`: expect `area=<room>`, not `{}`. Covers group 2 (3 cases).
4. Expectations for `v3_shadow_timer_status_0031` to `0034` and `v3_shadow_timer_pause_0027` to `0030`: also accept `area=<room>`. Covers groups 6 and 8. No flips, because the model used `name=`.
5. No change: group 3 (model error), group 5 (keep strict), group 7 (eval right).

After approval and any change, baseline scoring must be rerun. The 139/249 number is from the current scorer and expectations and is not valid for the proposed version.

## Model-error groups mapped to training-data fixes

The brief states these fixes exist in the timer wording slice. They were not re-verified in the repository configs in this worktree.

| Model-error group | Training gap in v12 | Fix |
|---|---|---|
| 3. Global cancel-all with area=home | No global cancel-all rows | Timer wording slice: global cancel-all rows |
| 6. Timer status with name=room | No area status rows | Timer wording slice: area status rows |
| 8. Pause answered with unpause | 0 of 210 pause rows with hold/freeze phrasing | Timer wording slice: hold/freeze pause phrasing |
| 7. Start timer with room as name | 0 of 370 v12 HassStartTimer rows mention a room | Follow-up timer slice: "set a N minute timer in the <area>", labelled with duration only |

## Open items and risks

- Floor argument risk: call floor arguments are not validated by the label gates, and floor names are not rendered in the prompt. A model can emit a floor name the runtime does not resolve, and no gate catches it.
- Training-row counts quoted from the brief (747 GetLiveContext rows with string domain, 203 lock rows with no device_class, 210 pause rows) were not re-counted here.
- `timers.py` imports `probatio` (line 11), the schema library in this Home Assistant release; the slot lines above are read from it.
