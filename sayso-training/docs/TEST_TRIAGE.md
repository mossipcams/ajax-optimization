# SaySo generator test triage (step 1)

Scope: `training/tests`, `training/generators`, this file. Full-suite runs were executed on the llm VM under Python 3.12.3 with the CI pins, with `--continue-on-collection-errors`. Logs live on the VM at `/srv/llm/data/testing-pipeline/logs/`; copies were read from there.

## Counts

| Run | VM log | Failed | Passed | Skipped | Errors |
|---|---|---|---|---|---|
| Before (HEAD `a7e78e1`, clean archive) | `base-full.log` | 56 | 899 | 2 | 11 |
| After generator fix only (`fix1`) | `fix1-full.log` | 27 | 936 | 2 | 3 |
| After generator fix and justified test edits (`fix3`) | `fix3-full.log` | 15 | 948 | 2 | 3 |

Remaining after `fix3`: 1 collection error and 2 setup errors plus 15 failures, all classified below. Environment (not installed on the VM, not a repo change): voluptuous (1 collection error), transformers (2 + 5 failures). Stale or unresolved (not environment): 10, which is 9 stale and 1 unresolved generator item (see table and the remaining-work list).

## Where the generator's dead-slot reasons come from

A correction row (`generators/rendering.py` `_render_correction`) has an untrained first assistant call with the wrong name (`correction_wrong_name` = "area + name"), a synthetic error tool reply, and a trained retry with the right entity. The first call is untrained (`train_on_turn: False`) because it is not a label.

- `unknown_canonical_entity` came from `generate_row` (`generators/row_generation.py`, final `validate_serialized_row`) running `gate_named_device_calls` on every assistant turn, including the untrained wrong-name call. The name is not an entity, so every correction attempt that reached this gate was rejected. The pipeline's own row check, `pipeline._row_rejection`, already filters to trained turns with the comment "a correction's untrained first try may name a missing entity on purpose" (`generators/pipeline.py:326-331`). The two checks disagreed, and the generation-time one ran first.
- `family_mismatch` for correction came from the correction branch of `generate_row`: `target_entity` is None, or `entity_name_contains_area(entity)` (`row_generation.py` correction branch). A name that already contains its area would make the wrong name a duplicate ("Living Room Living Room Lamp"), so this check is deliberate and is kept. It lowers yield but is not what killed the slots.
- Dead slots: `_pick_family_slot` raises `dead generation slots` after 50 failures per slot (`pipeline.py`, `SLOT_MAX_FAILURES`). Per-slot counters in the baseline (e.g. `correction/fans/turn_off #5: family_mismatch 35, unknown_canonical_entity 15`) show the gate reason dominating every correction slot.

### Fix (generation site)

`row_generation.py` adds `_labelled_gate_row(row)`: for `metadata.family == "correction"` it removes the untrained assistant tool-call turn and its tool reply (the untried first call) before `validate_serialized_row`. `generate_row` then also runs `validate_token_budget` on the full row whenever the view differs, so the token budget still counts the untried call. Other families are unchanged.

What was not changed: no gate in `generators/validation.py` changed. The round-1 attempt (every gate skipping untrained turns) was reverted. It would have stopped checking untrained history calls, which `9d34ee6` checks on purpose ("history calls included"), so it could not be shown to be safe. Because the gates are untouched, the fixtures in `test_label_gates.py`, `test_runtime_behavior.py` and `test_corpus_validation.py` still get rejected (they were not edited and still pass).

Review point: `PIPELINE.md` invariant 8 reads "label gates enabled on every row". The gates still run on every trained turn and on every non-correction row. For correction rows, the only exclusion is the untried first call (not a label). Flagged for review rather than treated as invisible.

## Test edits (each justified)

Every edit below changes only the expectation that a named commit made stale. No assertion was deleted or weakened, and no test was marked skip or xfail.

- `test_counterfactual_groups.py::test_feature_variant_withholds_the_unsupported_tool`: expected `device_unsupported` -> `unsupported`. Evidence: `fa9083b` ("keep feature contrasts as unsupported with the tool withheld"); fix1 output `assert 'unsupported' == 'device_unsupported'`.
- `test_coverage_gates.py::test_supported_actions_decide_between_acting_and_refusing`: `supported_action_family` now returns only the capable variant (`fa9083b`, "drop the incapable supported-action variant"); the refusal assertions were removed with it. Test name kept (documented, not renamed).
- `test_grounding_v2.py::test_every_generated_site_keeps_its_family_contrast`: features site checks the capable member only (`fa9083b`; baseline `KeyError: 'incapable'`).
- `test_grounding_v2.py::test_namespaced_rows_keep_prompt_schema_and_label_in_one_contract`: first assertion flipped to `not in prompt` (`8e20dae`, production live-context paragraph does not name the tool; the existing second assertion already expected the bare name absent).
- `test_nonstandard_generator.py::test_satellite_has_no_absolute_volume`: `volume_up` is now unsupported for a step-only satellite (`a73730b` made `volume_up`/`volume_down` require `volume`). The "no absolute volume" assertions are unchanged.
- `test_pipeline_linguistics.py::test_generated_requests_are_grammatical_english`: allow-list gained `hold`/`freeze` (`9d34ee6` phrases timer pauses as hold/freeze, `utterances.py`). "Hold my bread timer" is grammatical.
- `test_v11_templates.py::test_named_routine_action_uses_script_tool_and_name`: first assertion's allowed prefixes gained `turn on ` / `switch on ` (`9d34ee6`: scripts phrased "turn on"/"switch on" on purpose). Evidence: fix1 output `'turn on my Movie Mode routine'`. Only the first assertion was widened: `utterances.py:46` swaps only the leading `run ` for `turn on `/`switch on `, so those utterances still contain the script name and the second assertion keeps the original `run ` check.
- `test_v6_eval_fixes.py::test_clarify_settings_rows_clarify_with_candidates`: rows whose `metadata.family` is no longer `clarify` are skipped (`9d34ee6` media-ambiguity relabel re-labels `media_players volume_set` to ordinary "Done."). Clarify rows are still checked, and `checked >= 10` is kept.
- `test_preflight.py::test_valid_dataset_passes`: fixture utterance "Pause the TV" is verbatim eval case `nonstandard_13_pause` (`evals/cases/nonstandard_v1.jsonl`), which PIPELINE invariant 6 forbids in training data. Changed to "Pause the Living Room TV" (no eval match; fuzzy overlap is below the 0.8 threshold).

Tests attempted and reverted (documented, not left edited): `test_canonical_generator.py::test_unsupported_family_keeps_tool_and_labels_no_action` and `test_context_hard_negatives.py` (a retry helper turned the setup error into a share of 1.0, so the premise is still broken). Both were left as in the repo in step 1; step 2b (below) replaces them.

## Triage table

Group key: (a) generator bug, (b) stale test, (c) environment. Status is from the `fix3-full` run.

| test node id | group | root cause | action taken | evidence |
|---|---|---|---|---|
| `training/tests/test_area_scenarios.py::test_generation_is_deterministic_and_carries_area_rows` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_canonical_generator.py::test_generation_is_deterministic` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_canonical_generator.py::test_production_catalog_is_answer_independent` | b | test expects the rendered tools to equal the full production catalog, but the decoy-withhold step (_decoy_removals, tools.py/rendering.py) drops HassCancelAllTimers for this candidate | fixed in step 2c by owner decision A: the test asserts offered == catalog minus the decoy-withheld set (from `rendering._decoy_removals`), gold and core tools always offered, at most 2 withheld, withheld disjoint from needed; generator unchanged; the leak is documented in Step 2c | base-full.log FAILED; fix3-full.log: remains failing; step2c-full: see Step 2c evidence |
| `training/tests/test_canonical_generator.py::test_correction_row_shape` | a | correction row's untried wrong-name call rejected as unknown_canonical_entity | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_canonical_generator.py::test_unsupported_family_keeps_tool_and_labels_no_action` | b | fa9083b removed the scenario-level `device_unsupported` label for hidden gaps (`gold.py` `expected_device_unsupported` deleted; `gold.py:181-184` and `294-298` now reject `entity_lacks_capability`). The label is still produced on purpose by the context hard-negative branch (`row_generation.py:1377-1385`) | fixed in step 2b: the test asserts the scenario-level reject, then checks the labelled row from `generate_row` with the context rate at 1.0 (20 of 20 kept, tool offered, no tool calls, no unavailable tools); no planner or gate change | base-full.log FAILED; fix3-full.log: remains failing; step2b: see Step 2b |
| `training/tests/test_canonical_generator.py::test_area_rows_use_production_format` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_canonical_generator.py::test_no_eval_utterance_contamination` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_canonical_generator.py::test_cli_smoke_config_runs` | a | generator subprocess exits 1 on dead correction slots (same generator path as the in-process tests) | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_catalog_shortcuts.py::test_name_targeted_on_off_labels_carry_their_entity_domain` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_context_hard_negatives.py::test_home_without_context_entities_is_unchanged` | b | fa9083b: without context entities the unsupported media_players/turn_on draw is rejected (`entity_lacks_capability`), so the old helper's assumption that every call yields a row no longer holds | fixed in step 2b: the test compares first-attempt outcomes at rate 0.0 and 1.0; both give no row with reason `entity_lacks_capability`, so the rate has no effect, and the context-refusal check becomes a no-row check | base-full.log FAILED; fix3-full.log: remains failing; step2b: see Step 2b |
| `training/tests/test_corpus_validation.py::test_pipeline_rejects_row_with_unoffered_tool_and_retries` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_corpus_validation.py::test_pipeline_raises_on_forced_conflicting_pair` | a | test expects 'disagree on the label' but the run stops first on dead correction slots | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_counterfactual_groups.py::test_feature_variant_withholds_the_unsupported_tool` | b | no_action_reason is now 'unsupported' for feature contrasts (fa9083b: 'keep feature contrasts as unsupported with the tool withheld') | edited: expected reason 'device_unsupported' -> 'unsupported'; withheld-tool assertion kept; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_coverage.py::test_accepted_rows_span_capabilities` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_coverage_gates.py::test_supported_actions_decide_between_acting_and_refusing` | b | fa9083b dropped the incapable supported-action variant; supported_action_family returns one variant | edited: unpack one capable variant and keep the HassSetVolume assertion; refusal assertions removed (variant no longer exists); status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_defect_v4_corpus.py::test_count_row_tokens_counts_a_row_with_tool_calls` | c | transformers is not in the CI pin set; token counter import fails | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_defect_v4_corpus.py::test_cheap_token_bound_never_understates_the_real_count` | c | transformers is not in the CI pin set; token counter import fails | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_generator_pipeline.py::test_smoke_generation_small_n` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_generator_pipeline.py::test_generation_is_deterministic` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_generator_pipeline.py::test_generation_is_reproducible_across_processes` | a | generator subprocess exits 1 on dead correction slots (same generator path as the in-process tests) | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_generator_pipeline.py::test_rows_use_production_catalog` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_grounding_v2.py::test_every_generated_site_keeps_its_family_contrast` | b | fa9083b removed the 'incapable' features member, so members['incapable'] raises KeyError | edited: features site checks the capable member only; other site kinds untouched; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_grounding_v2.py::test_namespaced_rows_keep_prompt_schema_and_label_in_one_contract` | b | 8e20dae renders the production live-context paragraph without naming the tool; the prompt no longer contains homeassistant__GetLiveContext | edited: first assertion flipped to 'not in prompt', matching the existing bare-name check; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_llamafactory_view.py::test_smoke_fixture_yields_two_supervised_views` | c | transformers is not in the CI pin set; render and export import fail | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_llamafactory_view.py::test_first_supervised_view_contains_gold_tool_name` | c | transformers is not in the CI pin set; render and export import fail | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_llamafactory_view.py::test_correction_wrong_call_stays_in_prompt_not_output` | c | transformers is not in the CI pin set; render and export import fail | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_llamafactory_view.py::test_render_supervised_turn_includes_native_end_marker` | c | transformers is not in the CI pin set; render and export import fail | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_llamafactory_view.py::test_export_cli_writes_rendered_jsonl_and_dataset_info` | c | transformers is not in the CI pin set; render and export import fail | remaining: install transformers on the CI image (not installed); status: remains failing | base-full.log FAILED; fix3-full.log: remains failing |
| `training/tests/test_nonstandard_generator.py::test_satellite_has_no_absolute_volume` | b | a73730b: satellite (volume_step only) no longer supports volume_up through HassSetVolumeRelative | edited: volume_up assertion flipped to 'not supported'; no-absolute-volume assertions kept; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_nonstandard_generator.py::test_satellite_rows_from_generated_scenarios_target_the_satellite` | b | a73730b: volume_up/volume_down require `volume` (`capability_registry.py:131-132`), so the step-only satellite is never a volume target; the satellite self-volume premise is gone | rewritten in step 2b: checks pause/mute satellite rows (satellite profile supports both, `homes.py:617-620`), the tool name, `validate_row` passing, and `seen >= 50` (observed 114) | base-full.log FAILED; fix3-full.log: remains failing; sat-full: passes (see Step 2b) |
| `training/tests/test_nonstandard_generator.py::test_satellite_tv_exclusion_builds_contrast_rows_in_both_directions` | b | same a73730b premise: no satellite/TV volume contrast pairs exist | retired in step 2b (user decision): the satellite/TV volume contrast family is removed; replaced by `test_no_volume_call_targets_the_satellite` | base-full.log FAILED; fix3-full.log: remains failing; sat-full: test removed |
| `training/tests/test_nonstandard_generator.py::test_satellite_tv_contrast_rows_pass_validate_row` | b | same a73730b premise: no satellite contrast scenarios exist | retired in step 2b (user decision); replaced by `test_no_volume_call_targets_the_satellite`; `validation.py` satellite-contrast gates are unchanged | base-full.log FAILED; fix3-full.log: remains failing; sat-full: test removed |
| `training/tests/test_nonstandard_generator.py::test_exclusion_row_missing_excluded_device_fails_validation` | b | a73730b: the satellite fixture fails `validate_spec` with `entity_lacks_capability` (step-only satellite has no VOLUME_SET) | rewritten in step 2b: TV volume_down with the satellite excluded; the missing ', not you' clause still gives `missing_exclusion` (the same reason as before); ', not you' passes | base-full.log FAILED; fix3-full.log: remains failing; sat-full: passes |
| `training/tests/test_pipeline_linguistics.py::test_generated_requests_are_grammatical_english` | b | 9d34ee6 deliberately phrases timer pause as 'hold'/'freeze' (utterances.py:44 area); the verb allow-list predates it | edited: added hold and freeze to the allow-list in test_pipeline_linguistics.py; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_preflight.py::test_valid_dataset_passes` | b | fixture utterance 'Pause the TV' is verbatim eval case nonstandard_13_pause (PIPELINE invariant 6) | edited: fixture utterance -> 'Pause the Living Room TV' (not in any eval case; fuzzy overlap below 0.8); status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home.py::test_generated_rows_never_mention_unassigned` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home.py::test_mixing_is_off_by_default_in_generated_rows` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home.py::test_an_explicit_cap_is_never_exceeded` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home.py::test_capped_rows_are_replaced_not_dropped` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home.py::test_no_real_home_means_no_cap_bookkeeping` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home_mixing.py::test_the_manifest_records_requested_and_achieved_mixing` | a | real-home quota ends one row short: achieved_rate 0.2493 (374/1500 or 375/1504); requested 0.25 of 1500 = 375 target; probabilistic real-home selection in pipeline.py:537-543 | fixed in step 2c: the real-home forced check counted the datetime tail as eligible; it now counts only rows that can take a real home (`pipeline.py` eligible_remaining); no test edit; status: passes after fix | base-full.log FAILED; fix3-full.log: remains failing; step2c-targeted: passes |
| `training/tests/test_real_home_mixing.py::test_saturated_real_home_falls_back_to_synthetic` | a | correction row's untried wrong-name call rejected as unknown_canonical_entity | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_real_home_refusals.py::test_real_home_absence_and_junk_keep_the_production_context` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_sampling.py::test_generation_hits_family_allocations` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_sampling.py::test_generation_family_outcomes_match_slots` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_sampling.py::test_generation_unavailable_rows_withhold_namespaced_tool` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_sampling.py::test_quota_tracker_reports_requested_and_achieved` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v11_templates.py::test_named_routine_action_uses_script_tool_and_name[26]` | b | 9d34ee6 deliberately rephrases scripts as 'turn on'/'switch on' (utterances.py:552 area); test allowed only 'run ' forms | edited: first assertion's allowed prefixes extended to 'turn on ' and 'switch on '; second assertion kept as original 'run ' check; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v11_templates.py::test_named_routine_action_uses_script_tool_and_name[31]` | b | 9d34ee6 deliberately rephrases scripts as 'turn on'/'switch on' (utterances.py:552 area); test allowed only 'run ' forms | edited: first assertion's allowed prefixes extended to 'turn on ' and 'switch on '; second assertion kept as original 'run ' check; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v11_templates.py::test_named_routine_action_uses_script_tool_and_name[32]` | b | 9d34ee6 deliberately rephrases scripts as 'turn on'/'switch on' (utterances.py:552 area); test allowed only 'run ' forms | edited: first assertion's allowed prefixes extended to 'turn on ' and 'switch on '; second assertion kept as original 'run ' check; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v11_templates.py::test_named_routine_action_uses_script_tool_and_name[36]` | b | 9d34ee6 deliberately rephrases scripts as 'turn on'/'switch on' (utterances.py:552 area); test allowed only 'run ' forms | edited: first assertion's allowed prefixes extended to 'turn on ' and 'switch on '; second assertion kept as original 'run ' check; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v5_feasible_mix.py::test_v5_recipe_grounding_does_not_starve_families` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_clarify_settings_rows_clarify_with_candidates` | b | 9d34ee6 media-ambiguity relabel: media_players volume_set rows are re-labelled ordinary ('Done.'), not clarify | edited: rows whose metadata family is not clarify are skipped; clarify rows are still checked and checked >= 10 kept; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_withheld_pairs.py::test_pairs_exist_and_each_has_a_same_target_positive` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_withheld_pairs.py::test_sibling_withholds_exactly_the_parent_tool` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_withheld_pairs.py::test_v10_400_row_generation_passes_hygiene_gate` | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log FAILED; fix3-full.log: passes after fix |
| `training/tests/test_lfm_python_parse.py` (error) | c | the package `custom_components/sayso/__init__.py` imports `homeassistant` at module level (lines 11, 19), and Home Assistant needs Python 3.14; `lfm_parse.py` itself imports only `re` and `typing` | fixed in step 2b (lfm import): the test loads `lfm_parse.py` by file path from `training_paths.SAYSO`, without importing the package; test functions and assertions unchanged | base-full.log ERROR; fix3-full.log: remains erroring; step2b-lfm: see Step 2b (lfm import) |
| `training/tests/test_context_hard_negatives.py::test_context_hard_negative_share_and_labels` (error) | b | same fa9083b fixture: the hidden-gap reject aborted the helper's single-attempt setup | fixed in step 2b: the helper retries each slot as the pipeline does (`pipeline.py:645-650`), and the share is measured over attempts (`len(negatives) / attempts` ~ `CONTEXT_HARD_NEGATIVE_RATE`), since accepted rows are all context refusals for this pair | base-full.log ERROR; fix3-full.log: remains failing; step2b: see Step 2b |
| `training/tests/test_context_hard_negatives.py::test_context_hard_negatives_are_deterministic` (error) | b | same fa9083b fixture: `_sample` raised `entity_lacks_capability` in setup | fixed in step 2b: `_sample` returns rows and attempt count, and the determinism check compares both | base-full.log ERROR; fix3-full.log: remains failing; step2b: see Step 2b |
| `training/tests/test_v6_eval_fixes.py::test_enforce_hygiene_raises_on_duplicated_rows` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_smoke_hygiene_stats` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_no_smoke_row_lacks_a_core_tool` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_hygiene_datetime_requirement_follows_coverage_config` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_unavailable_rows_withhold_only_the_requested_tool` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_non_refusal_rows_offer_the_requested_tool` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_audit_reports_by_withheld_tool` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |
| `training/tests/test_v6_eval_fixes.py::test_smoke_audit_counts_exclusion_scopes` (error) | a | dead correction slots: untried first call (wrong name) rejected by gates as unknown_canonical_entity; family_mismatch also drops rows | fixed: generate_row validates the labelled view of correction rows (row_generation.py _labelled_gate_row); no gate changed; status: passes after fix | base-full.log ERROR; fix3-full.log: passes after fix |

## Remaining work (not fixed here)

- Environment (group c): done in step 2 (see CI split). `voluptuous`, `transformers`, `tokenizers`, and `Jinja2` are in the CI grep. `test_lfm_python_parse.py` is resolved in Step 2b (lfm import): it loads `lfm_parse.py` by file path, so Home Assistant is not needed.
- Group b, satellite premise: resolved in Step 2b (the satellite/TV volume contrast family is retired; see Step 2b (satellite)). The decoy-withhold answer-independence check is not satellite-related; it was resolved in Step 2c (decision A).
- Generator (group a): `test_the_manifest_records_requested_and_achieved_mixing` is fixed in Step 2c (the forced real-home check ignored the excluded datetime tail).

## CI split (step 2)

The PR and push job runs `python -m pytest training/tests -m 'not slow' -q`. The nightly job and `workflow_dispatch` run `python -m pytest training/tests -q`. Neither command passes `--continue-on-collection-errors`, so a new collection error fails the job. The `slow` marker is registered in `sayso-training/pytest.ini`.

Threshold: a test is `slow` when it runs the corpus generator end to end and its own call or setup time is at least 5 s. Times come from `--durations=0` runs on the llm VM, which shares the host with the training job, so they vary between runs. The threshold uses the larger of two runs (`dur-full`, which lacked Jinja2, and `dur2-full`, on the final dependency set). Tests under 5 s stay in the fast set even when they generate, for example the 400-row tests at 3 to 5 s.

Marked (18 tests): `test_coverage_gates.py` 3, `test_grounding_v2.py` 3, `test_real_home_mixing.py` 3, `test_data_quality.py` 2, and one each in `test_generator_pipeline.py`, `test_v5_feasible_mix.py`, `test_real_home_refusals.py`, `test_v6_eval_fixes.py`, `test_defect_v4_corpus.py`, `test_withheld_pairs.py`, and `test_pipeline_linguistics.py`.

Wall times on the VM (`fast-final`, `full-final`):

| Run | Tests | Result | Wall time |
|---|---|---|---|
| Fast set (`-m 'not slow'`) | 950 selected, 18 deselected | 939 passed, 8 failed, 3 errors | 127.6 s (2:07) |
| Full suite | 968 | 957 passed, 8 failed, 3 errors | 525.4 s (8:45) |

The full suite ran twice on the final dependency set (`dur2-full`, 531.7 s; `full-final`, 525.4 s) with identical failure sets. Job timeouts are 15 minutes for the fast job and 45 minutes for the full job.

Dependencies in the CI grep: `voluptuous==0.15.2` matches the `voluptuous==0.15.2` pin in Home Assistant 2026.8.3, the version SaySo's `homeassistant>=2026.8.3` requires. `transformers==5.18.0`, `tokenizers==0.23.2`, and `Jinja2==3.1.6` are needed by the token-count and chat-template tests (`transformers` `apply_chat_template` raises without Jinja2; the pipeline-invariants test only skips on a missing `transformers`, so it failed on the Jinja2 import once `transformers` was present). Torch is not installed and the tests do not need it. `requests`, `numpy`, `scipy`, `soundfile`, and `llama-cpp-python` are not added: only `training/scripts` imports them, and no test does.

Network: the tokenizer tests load `LiquidAI/LFM2.5-230M-Base` from the Hugging Face Hub (`test_defect_v4_corpus.py` token tests, `test_llamafactory_view.py`, and the render test in `test_pipeline_invariants.py`). `training/artifacts/` is gitignored, so CI fetches the tokenizer on every run. The VM serves it from its local cache, so the VM runs do not prove the CI fetch. CI has no Hub cache step.

Import error (resolved in Step 2b (lfm import)): `test_lfm_python_parse.py` imported `custom_components.sayso.lfm_parse`, and the package `__init__.py` imports `homeassistant` at module level. Home Assistant 2026.8.3 requires Python 3.14.2 or later, so it cannot be installed on the Python 3.12 CI or VM. The module was a collection error in both jobs.

Step-1 status changes after the grep change: the 7 transformers and token-count tests in `test_defect_v4_corpus.py` and `test_llamafactory_view.py` now pass on the VM, using its cache. `test_pipeline_invariants.py::test_rendered_training_view_has_no_tool_call_tags` failed once Jinja2 was missing and passes with it. The 8 failures remaining in the table above are unchanged from step 1 (`test_canonical_generator`, `test_context_hard_negatives`, `test_nonstandard_generator`, and `test_real_home_mixing`).

Failing in the fast (PR) set at step 2, before Step 2b, so both jobs were red then:

- `test_nonstandard_generator.py::test_satellite_rows_from_generated_scenarios_target_the_satellite` (resolved in Step 2b)
- `test_nonstandard_generator.py::test_satellite_tv_exclusion_builds_contrast_rows_in_both_directions` (resolved in Step 2b)
- `test_nonstandard_generator.py::test_satellite_tv_contrast_rows_pass_validate_row` (resolved in Step 2b)
- `test_nonstandard_generator.py::test_exclusion_row_missing_excluded_device_fails_validation` (resolved in Step 2b)
- `test_lfm_python_parse.py` (collection error, Home Assistant on Python 3.14; resolved in Step 2b (lfm import))

None of the 18 slow tests fails in either run.

Current state at `511c239`: the full suite on the VM is 969 passed, 0 failed, 0 errors (`full-lfm`, 530.67 s). The collection error above is gone, and the workflow no longer needs `--continue-on-collection-errors`. The fast-set count at `511c239` was not re-measured; the table above is from step 2.

## Step 2b: fa9083b unsupported slot and context tests

Decision: no planner or gate change. The unsupported media_players/turn_on slot still has a production path, so it is not a dead slot:

- `row_generation.py:1377-1385` sets the context hard-negative expectation (`device_unsupported`, the context name) before the reject check at `:1387`. Whenever the rate draw hits, the slot yields a row.
- Synthetic homes always carry context entities (`homes.py` `_context_entities_for`; `tests/test_homes.py::test_context_entities_present_and_noncolliding` asserts non-empty for every size). Real homes are excluded from this family (`planning.py` `REAL_HOME_EXCLUDED_FAMILIES`).
- The pipeline retries the slot with a new `attempt` (`pipeline.py:645-650`) and only reassigns it after 50 consecutive failures (`pipeline.py:705`). At about 30% per attempt, 50 consecutive rejects is not a realistic outcome.
- The hidden-gap reject (`gold.py:181-184`, `294-298`) is the intended gate after fa9083b, so it stays.

Why the pair was kept: no other pair fixes the premise. The only unsupported rows that are not context refusals are blocked requests (`planning.py` `BLOCKED_REQUEST_OPERATIONS`), and `row_generation.py` excludes them from the context branch (`is_blocked_request`), so they cannot exercise the context share.

Test changes (each keeps the original label, tool, share, minimum and determinism assertions):

- `test_canonical_generator.py::test_unsupported_family_keeps_tool_and_labels_no_action`: `build_scenario` now asserts the hidden-gap reject for each of the 20 seeds. The labelled no-action row (`does not support that.`, `HassTurnOn` offered, no tool calls, no unavailable tools) is checked on the row `generate_row` produces with the context rate at 1.0. Minimum: 20 kept.
- `test_context_hard_negatives.py::_sample`: retries each slot with the pipeline's attempt numbering and asserts that the only rejection is `entity_lacks_capability`. It returns the rows and the attempt count. Minimum: 300 rows.
- `test_context_hard_negative_share_and_labels`: share is measured over attempts (`len(negatives) / attempts`, tolerance 0.07 around `CONTEXT_HARD_NEGATIVE_RATE`). For this pair every accepted unsupported row is a context refusal, so a share over accepted rows is 1.0 by construction. This is the failure the step-1 retry helper hit. The label loop is unchanged.
- `test_context_hard_negatives_are_deterministic`: compares rows and attempt counts.
- `test_home_without_context_entities_is_unchanged`: compares first-attempt outcomes at rate 0.0 and 1.0. Both give no row with reason `entity_lacks_capability`, so the knob has no effect without context entities.

Evidence:

- VM targeted run `step2b-targeted` (the canonical test and the context file): 10 passed in 7.82 s; the longest test is 3.03 s (`test_other_families_and_rng_are_unchanged`, unchanged).
- VM full run `step2b-full` (`training/tests -q --continue-on-collection-errors`): 961 passed, 6 failed, 1 error in 532.19 s. Compared with `ef263f9` (8 failed, 3 errors), the two rewritten failures and the two setup errors are gone. No new failures. The remaining failures are the same as before: `test_production_catalog_is_answer_independent`, the four `test_nonstandard_generator.py` satellite tests, and `test_real_home_mixing.py::test_the_manifest_records_requested_and_achieved_mixing`. The remaining error is the `test_lfm_python_parse.py` collection error.

## Step 2b (satellite): retire the satellite/TV volume contrast family

Decision (user, explicit approval): retire the satellite/TV volume contrast row family and rewrite the four `test_nonstandard_generator.py` tests that depended on it. The `Step 2b` section above covers the fa9083b tests and is separate.

Premise gone (cited): `a73730b` changed `capability_registry.py:131-132` so `volume_up`/`volume_down` require the `volume` feature (`volume_set` has required it since `71e9c643`, line 130). The step-only satellite profile (`homes.py:617-620`, from `d0cef744`) has `volume_step` and no `volume`. `entities_supporting` therefore drops the satellite from `cap_entities` for every volume op, and `_satellite_contrast_pair` (`scenarios/core.py`, from `d0cef744`) never returned a pair.

Generator removal (`generators/scenarios/core.py`): removed `_satellite_contrast_pair`, the constant `_SATELLITE_CONTRAST_OPERATIONS` (used only by that helper), and the now-unused `is_satellite` import. The `if pair ... elif` became `if`. The pair branch was reachable only when a pair existed, and its `rng.random()` draw was short-circuited when none did, so the RNG sequence is unchanged.

Output identity, checked on the llm VM with `hash_gen.py` (outside the repo, not committed): for each of 400 scenarios at seed 20261020, the script serialises the scenario, the rendered utterance, and the `validate_row` reason as canonical JSON and takes sha256.

| Sample | sha256 before | sha256 after |
|---|---|---|
| exclusion, generated homes | `ab4e3fa4a743e1ccee7b5cd6c41c7fd90afc71b6765cb04fd3c4649f85b269c6` | identical |
| ordinary, generated homes | `e6bbced43848237ae810ceeae197a8e4b706a07406157c2a57ea4f4ef5ed0ea2` | identical |
| exclusion, fixed satellite plus TV home | `b5e25a8f665be8378b6039bf9bb46a60dcff10b1a4e2c5c99e497a6b69e5e14c` | identical |

The "before" run counted calls to the pair helper: 171 calls, every one with two entities and a non-volume op, and 0 non-None pairs. The "before" and "after" hashes were produced on the same sample by the same script. The gates in `validation.py` (`satellite_contrast_spoken`, `is_satellite_contrast`) and the phrasing in `utterances.py` are untouched.

Test changes (`training/tests/test_nonstandard_generator.py`; no slow marks existed on these tests):

- `test_satellite_tv_exclusion_builds_contrast_rows_in_both_directions` and `test_satellite_tv_contrast_rows_pass_validate_row` are replaced by `test_no_volume_call_targets_the_satellite`. It builds 300 fixed-seed scenarios (half with a fixed satellite plus TV home, half generated; both robustness modes) and asserts that no `HassSetVolume`/`HassSetVolumeRelative` call names a satellite. It asserts at least 100 satellite-bearing scenarios and at least 250 volume calls checked (observed 196 and 300). The test is a regression guard for the premise: if a satellite ever gained volume support, it would be picked as a volume target and fail here.
- `test_satellite_rows_from_generated_scenarios_target_the_satellite` now uses pause and mute, which the satellite supports (`homes.py:620`). It asserts the tool name (`HassMediaPause` or `HassMediaPlayerMute`), that `validate_row` returns None for each satellite row, and at least 50 satellite rows (observed 114: 63 pause, 51 mute).
- `test_exclusion_row_missing_excluded_device_fails_validation` uses a TV `volume_down` spec with the satellite excluded. Without the ", not you" clause, `validate_utterance` returns `missing_exclusion`, the same reason the original test expected. Wrong-device clauses also return `missing_exclusion`, and the ", not you" clause passes.

Evidence:

- VM targeted run `sat-targeted` (`training/tests/test_nonstandard_generator.py -q`): 87 passed in 8.71 s.
- VM full run `sat-full` (`training/tests -q --continue-on-collection-errors`): 964 passed, 2 failed, 1 error in 527.45 s.
- Against the `Step 2b` baseline (`step2b-full`, HEAD `981a826`: 961 passed, 6 failed, 1 error): the four satellite tests now pass, and no new failures appear. The remaining failures are `test_production_catalog_is_answer_independent` and `test_real_home_mixing.py::test_the_manifest_records_requested_and_achieved_mixing`, plus the `test_lfm_python_parse.py` collection error. All three were already in the baseline.
- The sensitivity of the rewritten tests is argued from their assertions, not checked by mutating the generator on the VM.

## Step 2c (quota and decoy): one generator fix, one decision

### Real-home quota: fixed in the generator

`test_real_home_mixing.py::test_the_manifest_records_requested_and_achieved_mixing` expected 375 real rows of 1500 and got 374. A VM diagnostic (outside the repo, not committed: `/srv/llm/data/testing-pipeline/diag/test_diag_rh.py`, seed 515, the same config as the test) showed:

- No synthetic fallback fired. All 374 real rows came from chains whose first draw was real and succeeded (`sel_true_ended_real` 374).
- The final accepted row is a `datetime` row from the `datetime_deficit` tail (`pipeline.py` around line 527). `datetime` is in `REAL_HOME_EXCLUDED_FAMILIES` (`planning.py` lines 36-46), so the row could not take a real home. The forced check (`real_home_remaining >= rows_remaining`) still counted it as eligible, so the last real row was owed when the run ended.

Fix (`generators/pipeline.py`, real-home block): the forced check and the draw use `eligible_remaining = max(1, rows_remaining - max(0, datetime_deficit))`. Rows still owed to the datetime tail cannot take a real home, so they are not counted. The number of RNG calls is unchanged; only the threshold moves.

Output change: only runs with a real home (`balance_real_home`). Where `datetime_deficit` is positive, real-home decisions change, and the RNG draws after them shift, so real-home rows change downstream. Runs without a real home are unchanged. Any corpus built with a real home (for example `full_sft_v*.yaml`) will not reproduce byte-for-byte. `fix3`-era hashes for those corpora are stale.

Not changed: no gate, no cap, no fallback reason, and the `e36ed8f` synthetic fallback is still in place. The fix is not a guarantee. Other excluded families (for example `unavailable` or `area`) drawn from the plan inside the forced stretch can still leave a row short. The test is deterministic on seed 515 and passes. No test file was edited.

### Decoy withholding: owner decision A, test rewritten

`test_production_catalog_is_answer_independent` expects the offered tools to equal the production catalog for an ordinary candidate. `rendering._decoy_removals` withholds one or two tools with p=0.3.

Findings:

- It leaks by construction. The candidate pool excludes `needed` (`rendering.py` lines 58-75: tools called anywhere in the row, the operation's tool, and the requested calls) and the core tools. The draw is seeded by `candidate_id`, but the pool, and therefore which tool a draw maps to, depends on the gold answer. The gold tool is never withheld, so a missing tool shows the row's answer is not that tool. For a 25-tool home, the per-tool effect is about 1-2 percent (my estimate from the draw sizes).
- It is deliberate, not stray. `705e661` ("remove v5 catalog-shape refusal shortcut") and `8497da1` ("v6 generator fixes") added it. `docs/PLAN_V5_CORPUS_SHORTCUT_FIXES.md` finding 1 records the shortcut it removes: 17, 23 or 29 offered tools gave 100%, 72% and 79% refusal. `docs/PLAN_V6_GENERATOR_EVAL_FIXES.md` Fix 3 makes "never withhold the requested operation's tool" an invariant, and line 300 makes the core tools never withheld.

So the test's premise (no removal for an ordinary candidate) was retired by deliberate commits, but the answer-independence the test asserts is not true for decoy rows. Removing the leak needs a choice this step could not make from code or commits. **The owner chose A** ("option A", round 2). The generator's decoy withholding is unchanged. Options considered:

- A. Keep the documented decoy and rewrite the test to assert `offered == catalog - {the documented decoy}`, keep the gold and core tools always offered, and keep the withheld set at most two tools, disjoint from the row's needed tools. This is the smallest honest change. The leak stays and is documented here.
- B. Restrict the decoy pool to tools that no row on this home can take as gold. This changes the output of every decoy row, and the decoy has little to withhold on homes with few spare tools.
- C. Drop the needed-tool protection. Rows then call tools the model is not offered. This breaks the `PLAN_V6` invariant and is rejected here.
- D. Drop decoy withholding. This reintroduces the v5 catalog-shape shortcut.

Test change under decision A (`training/tests/test_canonical_generator.py::test_production_catalog_is_answer_independent`): the old assertion `offered == catalog` is replaced by `offered == catalog - withheld`, where `withheld = set(rendering._decoy_removals(spec, row["messages"]))` is computed from the generator's own decoy draw, not by subtracting what is missing. The test also asserts: the explicit-removal path is not used (so the decoy is the only source), the withheld set is a subset of the catalog and has at most 2 tools, the gold tools (the tool calls in the row, non-empty) and the core tools (`CORE_TOOL_NAMES`) are offered, and the withheld set is disjoint from the needed tools. `GetDateTime` must still be offered. Every original assertion is kept; none was weakened.

### Evidence

- VM targeted run `rh-targeted` (`test_real_home_mixing.py`, `test_real_home.py`, `test_real_home_refusals.py`): 35 passed in 90.54 s. The cap tests in `test_real_home.py` pass unchanged.
- VM targeted run `tgt-step2c` (`test_canonical_generator.py`, `test_real_home_mixing.py`): 33 passed in 67.34 s, including both target tests.
- VM full run `step2c-final` (`training/tests -q --continue-on-collection-errors`, after the decoy test rewrite): 966 passed, 0 failed, 1 error in 528.09 s. The error is the `test_lfm_python_parse.py` collection error (Home Assistant on Python 3.12), unchanged.
- Against the `Step 2b` baseline (`step2b-full`: 961 passed, 6 failed, 1 error): the two Step 2c targets now pass and no new failure appears. An earlier run with the unchanged decoy test (`step2c-full`: 965 passed, 1 failed, 1 error) showed only the decoy assertion failing, which is the premise that decision A replaces.

## Step 2b (lfm import): load `lfm_parse` by file path

Evidence:

- `vendor/SaySo/custom_components/sayso/__init__.py` imports `homeassistant` at module level (lines 11 and 19). Importing `custom_components.sayso.lfm_parse` runs that `__init__.py` first, so the test could not collect on Python 3.12.
- `vendor/SaySo/custom_components/sayso/lfm_parse.py` imports only `from __future__ import annotations`, `re`, and `typing` (lines 2-5). The module does not need `homeassistant`, so loading it directly is safe.

Change (`training/tests/test_lfm_python_parse.py`, import block only): the package import is replaced by `importlib.util.spec_from_file_location` on `training_paths.SAYSO / "custom_components" / "sayso" / "lfm_parse.py"`, registered as `sayso_lfm_parse` in `sys.modules` before `exec_module`. The two names the tests use, `parse_lfm_python_tool_call` and `parse_lfm_python_tool_calls`, are bound from that module. The three test functions and every assertion are unchanged. No vendor file, generator, or validation gate was edited.

Evidence from the VM (Python 3.12.3, CI pins):

- Targeted run `lfm1` (`training/tests/test_lfm_python_parse.py -q`): 3 passed in 0.20 s.
- Full run `full-lfm` (`training/tests -q --continue-on-collection-errors`): 969 passed, 0 failed, 0 errors in 530.67 s. Against `ef263f9` (8 failed, 3 errors), the collection error is gone and no new failure appears. The `step2c-final` error on this module no longer occurs.


## Step 3: semantic label gates (referents and needless refusals)

Two gates in `training/generators/validation.py`, both run by `validate_serialized_row` on trained turns only:

- `gate_call_referents` → `unreferenced_call_target`. A trained call's `name` must be referred to by a user turn before it (directly, by alias, or by the entity's device type in a referred area), or its `area` must be referred to (named by a user turn, the context's `target_area` when no visible area is named, or contained in the call's `name`).
- `gate_unjustified_refusal` → `unjustified_refusal_or_clarify`. A trained refusal reply (`I can't help…`, `Sorry, I didn't catch that.`, absence replies) or a clarification whose candidates resolve to two or more distinct exposed entities is rejected when exactly one exposed entity matches the requested operation and utterance. Matching is name/alias, or (for non-absence replies) area and device type within the utterance's area scope (the spoken areas, or the context's `target_area` when no visible area is spoken).

The gates run after the existing gates, except `gate_unjustified_refusal`, which runs before `gate_clarification` so that a needless clarification is reported under the new reason. No existing gate was loosened.

### Measurement corpus

- Config: `training/configs/generation/full_sft_v13.yaml` with `count: 2000`, seed `20261028` unchanged. Scratch copy on the VM at `out/full_sft_v13_smoke2000.yaml`.
- The full coverage floors cannot be reserved at 2000 rows (`planning._reserve_tool_floors` raises), so the scratch copy scales them by 1/20: `min_positive_per_tool` 200→10, `min_positive_per_operation` 40→2, `get_datetime_positive_min` 400→20. The allocation shares are unchanged.
- Pre-gate corpus: generated from a pristine copy of HEAD `93aadc7` (the two edited files restored): `out/v13_smoke2000_pre.jsonl`, 2000 rows accepted.
- Both gate functions were run over every row by `out/measure_v13.py`. The sample is `random.Random(20261028).sample` over the rejected rows.

### Pre-tuning counts (first gate version, 2000 rows)

Total rows 2000. Rejected 131 (6.6%).

| Reason | Total | exclusion | ordinary | aliases | area | follow_up | clarify | absence | unsupported | unavailable |
|---|---|---|---|---|---|---|---|---|---|---|
| `unreferenced_call_target` | 100 | 24 | 55 | 7 | 14 | | | | | |
| `unjustified_refusal_or_clarify` | 31 | | | | | 7 | 11 | 4 | 8 | 1 |

Sampled classification (20 rejected rows, seed `20261028`). Verdicts use the first gate version's rule (name/alias match, no area scope), with entity areas and domains read from the row's static context:

| Row | Family | Reason | Utterance | Offending label | Verdict | Why |
|---|---|---|---|---|---|---|
| 55 | aliases | call | turn off smart tvs in my master bedroom | off, "Master Bedroom Smart TV" | FP | Same room and device type; name not spoken (plural "tvs") |
| 626 | ordinary | call | Turn down the master bedroom tv volume | volume down, "Smart TV" (Master Bedroom) | FP | Same room and type |
| 42 | ordinary | call | switch off the loft tv | off, "Television" (Loft) | FP | Same room and type |
| 370 | ordinary | call | mute the living room tv | mute, "Living Room QLED TV" | FP | Same room and type |
| 823 | ordinary | call | could you skip to the next track on the living room speaker? | next, "Echo Dot" (Living Room) | FP | Same room and type |
| 4 | ordinary | call | Please switch on the Den tv | on, "Beamer" (Den) | FP | Same room and type |
| 1791 | ordinary | call | Resume hallway speaker | unpause, "SaySo Satellite" (Hallway) | FP | Same room and type |
| 694 | area | call | could you switch off the lights in the lounge? | off, area "Living Room" | FP | "Lounge" is an area alias; `target_area` is "Living Room" |
| 31 | ordinary | call | Can you switch on the Craft Room light for me? | on, "Bench Lamp" (Craft Room) | FP | Same room and type |
| 3 | exclusion | call | switch off the kitchen lights, but leave kitchen downlights unchanged | off, "Kitchen Plinth Lights", "Kitchen lutron light" | FP | Same room and type |
| 1520 | area | call | turn off the dining room lights but leave the Dining Room Accent Light on | off, "Dining Room Ceiling Light", "Lamp" | FP | Same room and type |
| 598 | ordinary | call | Can you pause the Master Bedroom speaker for me? | pause, "Master Bedroom Sonos Speaker" | FP | Same room and type |
| 185 | ordinary | call | please pause nursery speaker | pause, "SaySo Satellite" (Nursery) | FP | Same room and type |
| 551 | area | call | leave the living room accent light on and switch off the other living room lights | off, "Living Room Downlight", "Lamp" | FP | Same room and type |
| 1947 | clarify | clarify | kda please switch on the hallway tv | "Did you mean the Hallway Fire TV or the Hallway TCL TV?" | FP | Matched only the "TV" entity, which is outside the hallway |
| 675 | unavailable | refusal | kda Mute SaySo Satellite | "I can't do that with the available Home Assistant tools." | FP | Requested tool withheld; the gate did not check the tool was offered |
| 1322 | unsupported | refusal | could you turn Foyer Lamp Power on? | "Foyer Lamp Power does not support that." | FP | Named refusal; loose substring match to "Foyer Lamp" |
| 585 | clarify | clarify | Switch off the Bathroom out let | "Did you mean the Bathroom Fountain Plug or the Bathroom Outlet?" | TP | Only "Bathroom Outlet" matches by name after STT folding |
| 1792 | unsupported | refusal | switch Spotlights Power off | "Spotlights Power does not support that." | FP | Named refusal; loose substring match to "Spotlights" |
| 195 | clarify | clarify | Please switch on the Entryway fan | "Did you mean the Entryway Table Fan or the Entryway Table Fan 2?" | TP | Only "Entryway Table Fan" matches (alias "Entryway fan") |

Pre-tuning totals in the sample: TP 2, FP 18. The FPs are the lesson: the first version judged referents by exact name only, and it judged refusals without an area scope or a tool-offered check.

### Tuning (new gates only)

Each change is a narrow exemption for a legitimate referent or a judgment the gate could not make, documented here:

1. **Area and type referent (call).** A name is referred when its area is in the utterance's area scope and its domain's type word appears in a user turn (`_CLARIFY_TYPES`, folded with `fold_stt_text`). Fixes the 14 call FPs above.
2. **Context area (call and refusal).** The scope is the visible areas the user turns name. The context's `target_area` (its own resolution, which covers aliases like "lounge") supplies the scope only for area-less requests. Fixed in the scope-fix round below.
3. **Name outside the area (call).** A name minus its area words is referred (e.g. "homepod mini" for "Living Room HomePod Mini"). Found in the suite's contrast families.
4. **Scoped matching (refusal and clarification).** An entity only matches when it is in the scope. Fixes the "TV" entity outside the hallway (1947).
5. **STT-folded type (refusal and clarification).** Type words match on `fold_stt_text`.
6. **Offered tool (refusal).** The requested tool must be offered, as the existing hidden-capability gate already requires.
7. **Named refusals skipped (refusal).** "X does not support that." is judged by `gate_hidden_capability_refusal`, which already checks the named entity's domain. Loose matches to other entities no longer count.
8. **Absence replies match by name only (refusal).** "There is no X in the Y" cannot be justified by a type-only match in the room. This drops the "tv" type matches on non-TV media players.
9. **"open"/"close" utterances skipped (refusal).** `_requested_tool` maps "open" to `HassTurnOn` (cover open), so "Open YouTube on the TV" looked like an unjustified refusal. Conservative: some real refusals on open/close requests are no longer judged.
10. **Clarification candidates must be distinct and exposed.** Otherwise the existing `gate_clarification` reasons (`clarify_unexposed_candidate`, `clarify_duplicate_candidate`) are preserved.

Earlier cuts were run through the full suite before these exemptions were final: `full_suite_step3` had 13 failures, all from the new gates. Dead contrast slots (`contrast/lights/turn_on` #12, `turn_off` #16, `media_players/next_track` #4: 350–400 `unreferenced_call_target` each); `launch_app` refusals (2 tests); counterfactual group tests (8: seven fail at row index 22 with `unjustified_refusal_or_clarify`, one allocation shortfall); withheld-pair and hygiene tests (3, dead contrast slots). The first cut's 2000-row generation also failed allocation (1844/2000 after 40000 attempts, 37694 `unreferenced_call_target`). Each cause above maps to one of the exemptions, and the final suite is green.

### Post-tuning counts (2000 rows, final gates)

Total rows 2000. Rejected 6 (0.3%), all `unjustified_refusal_or_clarify`:

| Reason | Pre-tuning | Post-tuning |
|---|---|---|
| `unreferenced_call_target` | 100 | 0 |
| `unjustified_refusal_or_clarify` | 31 | 6 (clarify 6) |

Post-tuning verdicts for all six (same rule as tuned: area scope, STT folding, type match):

- TP: 1748 "could you switch the cove light off?" (Bathroom), 1640 "Could you switch on the cove light?" (Bathroom), 478 "Can you switch the work light off for me?" (Garage), 734 "could you switch on the soft light?" (Hallway). Each offers two candidates outside the room while exactly one in-room device matches the type. The unmatched qualifier ("cove", "soft", "work") makes these borderline.
- FP: 1933 "Please switch on the Boiler Room out let" and 585 "Switch off the Bathroom out let". STT splits "outlet" into "out let", so the type word is lost and the second outlet/plug candidate is not counted.

### Generator true positives (listed, not fixed)

- **Clarify, candidates outside the room (4 of 158 clarify rows in the 2000-row corpus).** The candidate pair is drawn from another room, while exactly one device in the utterance's room matches. Examples: 1748, 1640, 478, 734 above. The fix is in clarify candidate selection (`_ambiguous_gold`/`shared_alias`), which is not a small change, so these rows stay listed. The gate rejects them at generation time; the rate does not starve any slot (the final generation accepted all 2000 rows).
- **Referent gate true positives: none found** in the 2000-row corpus. The generator emitted no call to a different device or wrong-area call beyond what the existing gates already reject. The gate's first two classes are covered by the unit tests instead.

### Known false positives left (not tuned)

- STT-split type words (1933, 585 above): the type word "outlet" is split by STT ("out let") and cannot be folded to a type match.
- Qualifier words are not checked. "cove light" counts a non-cove light in the same room as the type match (borderline TPs above). A stricter rule would require every content word to be accounted for.

### Final generation and verification

- Final gates, full 2000-row generation (`out/v13_smoke2000_post.jsonl`): accepted 2000, no dead slots. Post-hoc `measure_v13.py` over that corpus: 0 rejections, consistent with generation-time gates.
- VM targeted run `tests_gate4` (`training/tests/test_label_gates.py -q`): 171 passed (5 new tests: correct den-light call passes, different device, wrong area, unjustified refusal, needless clarification).
- VM full run `full_suite_final` (`training/tests -q`): 974 passed, 0 failed, 0 errors in 549.09 s. Baseline before Step 3: 969 passed (`full-lfm`); the 5 new tests account for the increase. No existing test was edited.

### Scope fix (second review round)

The review probe (`out/probe_scope.py`) found a hole in `_area_scope`: it always added the context's `target_area`. With utterance "turn on the den lights" and `target_area: Living Room`, `HassTurnOn {area: "Living Room", domain: ["light"]}` passed `gate_call_referents`. The same scope fed the type and outside-area referents, and `_utterance_matches_entity` in the refusal gate.

- Change: the context `target_area` supplies the scope only when no visible area is named by a user turn. If one is spoken, the scope is the spoken areas only. Nothing else was loosened.
- Tests (`test_label_gates.py`, 173 passed): `test_label_gate_spoken_area_beats_context_target_area` (den utterance, `target_area: Living Room`, call to Living Room area → `unreferenced_call_target`) and `test_label_gate_area_less_request_uses_context_target_area` (utterance "turn on the lights", `target_area: Den`, call to Den area passes `gate_call_referents` and `validate_serialized_row`).

Measurement with the fixed gates, same seed and config:

| Reason | Pre-tuning (first gate version) | Post-tuning (`fde19cf`) | Post-fix (this commit) |
|---|---|---|---|
| `unreferenced_call_target` | 100 | 0 | 0 |
| `unjustified_refusal_or_clarify` | 31 | 6 | 6 (clarify 6) |
| Rejected rows of 2000 | 131 | 6 | 6 |

- Pre-gate corpus `out/v13_smoke2000_pre.jsonl` (generated from `93aadc7`), `measure_v13.py` with the fixed gates: rejected 6, all `clarify`. The same 6 rows as the post-tuning verdicts above (TP 4, FP 2: 1933 and 585). All rejected rows are printed, since the sample of 20 exceeds the 6 rejections.
- The scope change does not change any count on this corpus. The wrong-area and spoken-area cases are covered by the unit tests above, not by a generated row.
- Regenerated corpus `out/v13_smoke2000_post_fix.jsonl` at the fixed gates: accepted 2000, no dead slots. `measure_v13.py` over it: 0 rejected.
- Full VM suite `full_suite_fix` (`training/tests -q`): 976 passed, 0 failed, 0 errors in 548.14 s. Baseline before this round: 974 passed (`full_suite_final`); the 2 new tests account for the increase.

## Step 4: gold oracle tests

Scope: `training/tests/test_gold_oracle.py` (new: 34 parametrized hand-labeled cases plus one person-refusal test, 35 tests) and `training/generators/gold.py`.

Method: every expectation was typed by hand from Home Assistant semantics and the contract (`docs/PIPELINE.md`, `docs/SAYSO_LFM_TRAINING_PLAN.md`, the pinned tool schema) before the run that first exercised it. Each case has a literal home. The only calls are `gold_from_scenario` and `expected_unexposed_domain`. No expected value was computed by gold.py or a helper that calls it. No hand label was corrected in this step; every disagreement was a gold bug.

### Hand label vs gold disagreements

Runs: `gold-oracle-1` (first run, 30 passed, 4 failed, against the step-4 gold.py as first committed); `gold-oracle-3` (the two-TV case added in the revision, run against the same gold.py before the revision fix: 34 passed, 1 failed).

| case id | hand label | gold output | verdict | evidence |
|---|---|---|---|---|
| `capability-light-without-brightness-rejects` | `reject`, `entity_lacks_capability` | `no_action` / `clarify` ("Which device did you mean?") | gold bug, fixed | Named device "Desk Lamp" lacks `brightness`, so `set_brightness` is not supported. Asking which device is wrong, since the device is unambiguous. The area and ambiguity branches already reject a present-but-unsupported entity. |
| `capability-named-light-without-brightness-no-fallback` | `reject`, `entity_lacks_capability` | `action` HassLightSet on "Kitchen Lamp", brightness 59 | gold bug, fixed | `_pick_entity` fell back to another light with `brightness` when the named target lacked it. That acts on a device the user did not name. |
| `capability-tv-without-volume-volume-up-rejects` | `reject`, `entity_lacks_capability` | `action` HassSetVolumeRelative on "Living Room TV", volume_step 30 | gold bug, fixed | `volume_up` requires the `volume` feature (`capability_registry`). The TV branch of `_ambiguous_gold` (added in 9d34ee6) skipped the feature filter for `tv` requests, so a single TV without volume got a call. The revision keeps the reject for one visible TV and keeps the two-TV exemption (row 5). |
| `capability-named-tv-without-pause-no-fallback` | `reject`, `entity_lacks_capability` | `action` HassMediaPause on "Bedroom Speaker" | gold bug, fixed | Same fallback as row 2. The named TV lacked `pause`, so `_pick_entity` picked a speaker in another room. |
| `ambiguity-two-visible-tvs-one-capable-clarifies` | `no_action`, `clarify`, candidates "Living Room TV", "Living Room Roku TV" | `action` HassSetVolumeRelative on "Living Room TV", volume_step 30 (`gold-oracle-3`, gold.py before the revision fix) | gold bug, fixed | Both TVs are visible in the area (name, domain and area are in the prompt; `volume` is not). Acting on the one capable TV makes the label depend on a feature the model cannot see (fa9083b). The two-visible-TV clarify rule from 9d34ee6 is restored. |

No hand label was corrected: 0 of 35 expectations changed.

### gold.py changes (revision)

1. Individual branch of `gold_from_scenario` (kept from the first commit): a named `target_entity` that lacks the operation's features returns `reject` / `entity_lacks_capability` before `_pick_entity`. The fallback to another entity no longer runs for a named target. This matches the area and ambiguity branches.
2. `_ambiguous_gold` (revised): for a `tv` intent on `media_players`, two or more visible TVs in the requested area are all candidates, as in 9d34ee6, so the label clarifies with both. The feature filter (`entities_supporting`) applies only when there is one visible TV, so one TV lacking the operation's features rejects with `entity_lacks_capability` (`capability-tv-without-volume-volume-up-rejects`). The first commit removed the exemption for every case, which made the two-TV label depend on a hidden feature (row 5).

Diff against 5078dc2 (the pre-step-4 gold.py), by behaviour:

- Two or more visible TVs in tv-intent ambiguity: unchanged (clarify with the visible TVs).
- One visible TV that lacks the operation: now rejects (5078dc2 acted on it).
- Named target lacking the operation's features: now rejects (5078dc2 fell back to another entity).

### Output identity (measured)

- Corpus: `training/configs/generation/full_sft_v13.yaml` copy `out/full_sft_v13_smoke2000.yaml` (count 2000, seed 20261028), `python -m generators.cli --config`.
- 5078dc2 tree (gold.py from `5078dc2`, everything else from the final tree): `out/v13_smoke2000_5078dc2.jsonl`, sha256 `5db5cfe95a7dae740df93f37dafb6f62f5315ab96f5ca21e4ee6b752c5c69421`, 2000 rows.
- Final code: `out/v13_smoke2000_final.jsonl`, sha256 `5db5cfe95a7dae740df93f37dafb6f62f5315ab96f5ca21e4ee6b752c5c69421`, 2000 rows.
- Byte-identical: 0 differing rows, same row count (`out/cmp_corpus.py`).
- This is one 2000-row corpus at one seed and config. It does not show that the two changed branches never fire on other seeds or on the full corpus.

### Verification

- `gold-oracle-1` (first run, first-commit gold.py): 30 passed, 4 failed (rows 1 to 4).
- `gold-oracle-3` (revision, two-TV case, gold.py before the revision fix): 34 passed, 1 failed (row 5).
- `gold-oracle-4` (`training/tests/test_gold_oracle.py -q`, final code): 35 passed.
- `full-step4-rev` (`training/tests -q`, final code): 1011 passed, 0 failed, 0 errors (4 hypothesis `.example()` warnings, not failures). Baseline at `5078dc2` was 976; step 4 adds 35 tests.
- Output identity: `gen-base` and `gen-final` (see above).

### Not covered

- Step-only satellite (`volume_up` without `volume_set`): the registry requires the `volume` feature for both, so the case cannot be expressed in the current gold. No case was written.
- Same-name entities in different areas: the contract forbids invented area-plus-name strings (issue #94), so no label was written.

## Step 5: hypothesis property tests

Scope: `generators/hypothesis_strategies.py` (strategy), `tests/test_hypothesis_strategies.py` (rewritten), `tests/test_gold_properties.py` (new), `tests/conftest.py` (Hypothesis profile). `generators/gold.py` was not changed: no property failed.

### Counterexamples

None. No property failed on any run, so no strategy or gold fix was needed and no counterexample was shrunk. The alias property was first written with a one-entity home, which could not fail if gold ignored names; a same-area distractor was added in the revision, and the property still passes (`s5-hyp2`, 13 passed).

Two development-time failures were test-harness problems, not generator or strategy bugs:

- A coverage test built on `hypothesis.find` (400 draws per pair) reported 3 of 42 pairs not found (`covers/close`, `scripts/run`, `media_players/pause`). A probe with `@given` over the same strategy counted all 42 pairs at 400 draws. The sampler is not uniform, so a fixed small budget can miss pairs. `find` was dropped.
- A nested `@given` collector at 400 examples then missed 3 different pairs (`timers/decrease`, `media_players/play`, `climate/set_temperature`). At 2000 examples all 42 were drawn. The profile is derandomized, so the run is repeatable.

### Strategy

- `valid_spec_strategy()` samples `(capability, operation)` from `trainable_pairs()`, which is built from `CAPABILITIES` and `trainable_operations()`, the same source the generator uses. It covers 42 pairs. It previously covered `lights/turn_on` only.
- `build_scenario` already injects a capable entity with the operation's required features when the home has none. The new `test_valid_spec_target_supports_operation` checks the target named in each call supports the operation, and the expected kind is `action`.
- `corrupted_spec_strategy` stays on `lights/turn_on`. `corrupt_spec` rewrites the first call's `name`, which the wrong-entity assertion (`unknown_canonical_entity`) assumes is an entity name. Timer calls skip the entity check in `validate_spec`, and script calls take no arguments, so the same corruption gives a different reason there. Widening it would change what the existing assertion tests, so it stays narrow.

### Hypothesis settings

`tests/conftest.py` registers and loads a `sayso` profile: `derandomize=True`, `deadline=None`, `database=None`. Both CI jobs run pytest on `training/tests`, so both use it. Per-test `max_examples` are set on each test. The coverage test sets 2000 examples; every other property uses 20 to 60.

### Property tests (gold)

All use `gold_from_scenario` with `random.Random(0)` on both sides.

| test | input transformation | property |
|---|---|---|
| `test_renaming_entity_renames_call_target` | rename the drawn target entity to `Renamed <token>` | after == before with the old name replaced by the new name, and the call names are exactly `[new]` |
| `test_unrelated_entity_leaves_label_unchanged` | add an entity in another area with another domain | label unchanged |
| `test_alias_request_resolves_to_canonical_entity` | home with the aliased entity plus a same-area, same-capability distractor (`Pantry Shelf`, operation features present) that matches neither the alias nor the canonical word; request by alias vs by canonical name | alias label == canonical label, kind `action`, call name is the canonical name (never the distractor) |

Scope limits: the rename and unrelated properties skip timers (the gold label has no entity target) and scripts (the script tool name is derived from the entity name, so a rename also renames the call name). The alias property has a distractor in the same area with the same capability, so a gold that ignored names would clarify over two candidates and fail `kind == action`. Ambiguity between two entities that share the requested name is not covered by this property.

### Measurements (llm VM)

- Property files (`test_hypothesis_strategies.py` and `test_gold_properties.py`, 13 tests): 13 passed, 3.8 s (`s5-hyp2`, after the alias distractor).
- Coverage test: 2.9 s, below the 5 s threshold, so not marked `slow`. No Step 5 test reached 5 s, so no `slow` mark was added.
- Fast set (`-m 'not slow'`): 999 passed, 18 deselected, 138.8 s (`s5-fast2`) at this commit. The Step 2 figure of 127.6 s was for a set with 939 passing tests, so the growth is not isolated here; the property files account for about 4 s of it.
- Full suite (`training/tests -q`): 1017 passed, 0 failed, 0 errors, 552.7 s (`s5-full2`).
