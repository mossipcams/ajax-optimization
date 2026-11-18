
from __future__ import annotations

from collections import Counter

import copy
import json
import random
from typing import Any

from generators.capability_registry import CAPABILITIES, operation_spec, trainable_operations
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker, pair_hash
from generators.grounding import is_blocked_request, pick_variant
from generators.homes import prepare_media_tv_ambiguity
from generators.correction import entity_name_contains_area
from generators.gold import expected_action, expected_no_action, expected_status, expected_unexposed_domain, gold_from_scenario, gold_matches_family, target_names_from_expected
from generators.labels import render_example, scenario_to_spec
from generators.real_home import load_real_home
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.scenarios.core import home_areas, semantic_id
from generators.scenarios.discrimination import (
    call_carries_value,
    pick_discriminating_description,
)
from generators.planning import CORE_TOOL_NAMES, REAL_HOME_EXCLUDED_FAMILIES, REAL_HOME_REFUSAL_FAMILIES
from generators.scenarios.unavailable import unique_no_action_hint
from generators.shared_alias import SHARED_ALIAS_RATE, build_generic_phrase_spec, build_shared_alias_spec
from generators.context_rows import (
    _CONDITIONAL_READ,
    _CONDITIONAL_TEMPLATES,
    _CONVERSATION_CONTEXT_CORRECT,
    _CONVERSATION_CONTEXT_HISTORY,
    _CONVERSATION_CONTEXT_OTHER,
    _CONVERSATION_CONTEXT_PRONOUN,
    _RELATIVE_BRIGHTNESS_DELTAS,
    _RELATIVE_BRIGHTNESS_DOWN,
    _RELATIVE_BRIGHTNESS_READ,
    _RELATIVE_BRIGHTNESS_UP,
    _RESTORE_BACK,
    _RESTORE_CHANGE,
    _RESTORE_READ,
    _TIMER_CONTEXT_ADJUST_HISTORY,
    _TIMER_CONTEXT_BARE,
    _TIMER_CONTEXT_LOWER,
    _TIMER_CONTEXT_MINUTES,
    _TIMER_CONTEXT_NAMED,
    _TIMER_CONTEXT_NAMES,
    _TIMER_CONTEXT_RAISE,
    _TIMER_CONTEXT_START_NAMED,
    _TIMER_CONTEXT_START_UNNAMED,
    _balanced_upper,
    _conditional_plan,
    _conditional_row,
    _conversation_context_row,
    _conversation_context_targets,
    _relative_brightness_plan,
    _relative_brightness_row,
    _restore_plan,
    _restore_row,
    _timer_context_row,
    casing_kind,
)
from generators.stt_noise import apply_log_stt_noise, apply_stt_noise
from generators.talk import (
    _CHATTER_PREFIX_FILLERS,
    _CHATTER_SUFFIX_FILLERS,
    _chatter_filler,
    _conversational_sentence,
    _garbled_device_fragment,
    _junk_sample,
    _junk_text,
    _junk_utterance,
    WRAPPED_COMMAND_RATE,
    talk_has_command_verb,
    talk_text_violation,
    wrap_utterance_with_chatter,
)
from generators.utterances import (
    _DATETIME_PHRASES,
    apply_generic_wording,
    discrimination_utterance,
    expand_utterance,
    finalize_training_utterance,
    request_seed_from_spec,
)
from generators.validation import (
    check_quality_eval_overlap,
    validate_row,
    validate_serialized_row,
    validate_token_budget,
)

CONTEXT_HARD_NEGATIVE_RATE = 0.30
UNEXPOSED_DOMAIN_RATE = 0.25
REAL_HOME_REPHRASE_LIMIT = 8
GENERIC_PHRASE_SHARE = 0.5

_DATETIME_UTTERANCES = tuple(dict.fromkeys((
    "what time is it",
    "tell me the time",
    "what's the time right now",
    "what is the current time",
    "what time is it now",
    "what date is it",
    "what's the date right now",
    "do you know what the date is",
    "can you tell me the date",
    "what's the day today",
    "is it Tuesday today",
    "what day of the week is it today",
    "what weekday is it",
    "which weekday is today",
    "do you know what weekday it is",
) + _DATETIME_PHRASES))


def _datetime_utterance(rng: random.Random, index: int) -> str:
    return rng.choice(_DATETIME_UTTERANCES)


def _follow_up_reply(chosen: str, rng: random.Random) -> str:
    templates = (chosen, f"the {chosen}", f"{chosen}, please")
    return rng.choice(templates)


def _withholds_core_tool(capability: str, operation: str) -> bool:
    op = operation_spec(capability, operation)
    return bool(op and op.tool_name and op.tool_name in CORE_TOOL_NAMES)


def contrast_slot_position(index: int) -> tuple[int, int]:
    from generators.grounding import CONTRAST_GROUP_SIZE

    return divmod(index, CONTRAST_GROUP_SIZE)


def _offered_call_reason(calls: list[dict[str, Any]], offered: list[dict[str, Any]]) -> str | None:
    from adapters.schema import tool_schema_map, validate_tool_arguments
    from generators.tools import namespaced_tool_name

    schemas = tool_schema_map(offered)
    for call in calls:
        name = namespaced_tool_name(call["name"])
        if name not in schemas:
            return "call_tool_not_offered"
        reason = validate_tool_arguments(name, call["arguments"], schemas)
        if reason:
            return reason
    return None


def _contrast_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    *,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
) -> tuple[dict[str, Any] | None, str | None]:
    from generators.grounding import behaviour_class, build_spec, seeded_contrast_group
    from generators.tools import production_catalog

    group_index, position = contrast_slot_position(slot["index"])
    try:
        group = seeded_contrast_group(group_index, config.seed)
    except ValueError as exc:
        return None, f"contrast_group_unbuildable: {exc}"
    if position >= len(group):
        return None, "contrast_variant_missing"
    item = group[position]
    spec = build_spec(item, seed=config.seed, index=group_index)
    if behaviour_class(spec["expected"]) != item["expected_class"]:
        return None, "contrast_label_mismatch"
    name = item["target_name"]
    present = {entity["name"] for entity in spec["home"]["entities"]}
    spec.update(
        family="contrast",
        utterance=item["utterance"],
        spoken_targets={name: f"the {item['noun'].casefold()}"} if name in present else {},
        removed_tools=item["withheld_tools"],
        full_tool_catalog=True,
        semantic_id=f"{item['contrast_group']}_{position}",
    )
    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"
    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    row["tools"] = production_catalog(spec["home"], removed_tools=item["withheld_tools"])
    reason = _offered_call_reason(spec["expected"].get("calls") or [], row["tools"])
    if reason:
        return None, reason
    row["metadata"].update(
        family="contrast",
        offered_tool_count=len(row["tools"]),
        real_home=False,
        # Every contrast variant is a counterfactual of the same command, so it counts toward grounding.
        grounding_family=f"ground_generated_{item['contrast_axis']}",
        discrimination=False,
        contrast_group=item["contrast_group"],
        contrast_axis=item["contrast_axis"],
        contrast_variant=item["contrast_variant"],
    )
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


def _rephrase_past_eval_overlap(spec: dict[str, Any]) -> None:
    # A real-home phrasing is keyed by its candidate id, so a rejected phrasing would repeat on every
    # retry. Re-render under fresh phrasing seeds until the utterance is off the exact eval prompts.
    base = spec.get("phrasing_seed") or spec.get("candidate_id", "")
    for attempt in range(1, REAL_HOME_REPHRASE_LIMIT + 1):
        if not check_quality_eval_overlap(spec["utterance"]):
            return
        spec["phrasing_seed"] = f"{base}:rephrase{attempt}"
        spec["utterance"] = expand_utterance(spec)


def _shared_alias_row(
    spec: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    family: str | None,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    casing_counts: dict[str, Counter[str]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    spec["utterance"] = finalize_training_utterance(
        spec["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
    )
    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"
    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )
    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)
    if family:
        row["metadata"]["family"] = family
    row["metadata"]["real_home"] = False
    row["metadata"]["grounding_family"] = None
    row["metadata"]["discrimination"] = False
    row["metadata"]["shared_alias"] = spec["shared_alias"]
    row["metadata"]["shared_candidates"] = list(spec["expected"]["candidates"])
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason:
        return None, reason
    dup_tracker.record(spec)
    return row, None


def _labelled_gate_row(row: dict[str, Any]) -> dict[str, Any]:
    if row["metadata"].get("family") != "correction":
        return row
    untried_ids = {
        call["id"]
        for message in row["messages"]
        if message["role"] == "assistant" and not message.get("train_on_turn")
        for call in message.get("tool_calls") or []
    }
    messages = [
        message
        for message in row["messages"]
        if message.get("tool_call_id") not in untried_ids
        and not untried_ids.intersection(call["id"] for call in message.get("tool_calls") or [])
    ]
    return {**row, "messages": messages}


def generate_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    attempt: int = 0,
    retry: int = 0,
    stt_remaining: int = 0,
    stt_log_remaining: int = 0,
    rows_remaining: int = 1,
    real_home_targets: Any | None = None,
    real_home_entity_cap: int = 0,
    real_home_names: frozenset[str] = frozenset(),
    real_home_usage: Any | None = None,
    quota: Any | None = None,
    grounding_required: bool = False,
    discrimination_required: bool = False,
    real_home_selected: bool | None = None,
    grounding_variants: list[dict[str, Any]] | None = None,
    grounding_rate: float = 0.0,
    casing_counts: dict[str, Counter[str]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    area_scenario = slot.get("area_scenario")
    if area_scenario:
        from generators.scenarios.area import build_area_spec

        # Area specs are seeded by slot index, so a retry would rebuild the same
        # eval-overlapping utterance; shift the index until it clears.
        for shift in range(8):
            spec = build_area_spec(area_scenario, slot["index"] + shift * 100_000, config.seed)
            if not check_quality_eval_overlap(spec["utterance"]):
                break
        spec["full_tool_catalog"] = True
        if check_quality_eval_overlap(spec["utterance"]):
            return None, "quality_eval_overlap"
        spec["utterance"] = finalize_training_utterance(
            spec["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
        )
        if spec["utterance"].casefold() in excluded:
            return None, "excluded_prompt"
        if check_quality_eval_overlap(spec["utterance"]):
            return None, "quality_eval_overlap"
        reason = validate_row(spec, token_budget=config.token_budget)
        if reason:
            return None, reason
        reject = dup_tracker.would_reject(spec)
        if reject:
            return None, reject
        spec["candidate_id"] = (
            f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
            f"_{dup_tracker.occurrences(spec)}"
        )
        try:
            row = render_example(spec)
        except ValueError as exc:
            return None, str(exc)
        if spec.get("clarify_question"):
            row["messages"][-1]["content"] = spec["clarify_question"]
        row["metadata"]["area_scenario"] = area_scenario
        row["metadata"]["family"] = "area"
        row["metadata"]["real_home"] = False
        row["metadata"]["grounding_family"] = None
        row["metadata"]["discrimination"] = False
        reason = validate_serialized_row(
            row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
        )
        if reason:
            return None, reason
        dup_tracker.record(spec)
        return row, None

    capability = slot["capability"]
    operation = slot["operation"]
    family = slot.get("family")
    if (
        family == "clarify"
        and (slot.get("robustness") or "ambiguity") == "ambiguity"
        and slot.get("capability") == "lights"
        and rng.random() < SHARED_ALIAS_RATE
    ):
        index = slot["index"] + attempt * 10000
        if rng.random() < GENERIC_PHRASE_SHARE:
            shared = build_generic_phrase_spec(rng, index, slot["home_size"])
        else:
            shared = build_shared_alias_spec(rng, index, slot["home_size"])
        if shared is not None:
            return _shared_alias_row(
                shared, config, rng, family=family, excluded=excluded,
                dup_tracker=dup_tracker, casing_counts=casing_counts,
            )
    if family == "contrast":
        return _contrast_row(slot, config, excluded=excluded, dup_tracker=dup_tracker)
    if family == "timer_context":
        return _timer_context_row(
            slot,
            config,
            rng,
            attempt=attempt,
            excluded=excluded,
            dup_tracker=dup_tracker,
            casing_counts=casing_counts,
        )
    if family == "relative_brightness":
        return _relative_brightness_row(
            slot,
            config,
            rng,
            attempt=attempt,
            excluded=excluded,
            dup_tracker=dup_tracker,
            casing_counts=casing_counts,
        )
    if family == "conversation_context":
        return _conversation_context_row(
            slot,
            config,
            rng,
            attempt=attempt,
            excluded=excluded,
            dup_tracker=dup_tracker,
            casing_counts=casing_counts,
        )
    if family == "restore":
        return _restore_row(
            slot,
            config,
            rng,
            attempt=attempt,
            excluded=excluded,
            dup_tracker=dup_tracker,
            casing_counts=casing_counts,
        )
    if family == "conditional":
        return _conditional_row(
            slot,
            config,
            rng,
            attempt=attempt,
            excluded=excluded,
            dup_tracker=dup_tracker,
            casing_counts=casing_counts,
        )
    home_size = slot["home_size"]
    home = None
    real_home_context = None
    absence_area = None
    real_home_row = False
    grounding_family = None
    variant = None
    if capability == "datetime":
        robustness = "datetime"
        targeting = "context"
        inject_missing = True
        scenario_index = slot["index"] + attempt * 10000
    else:
        cap = CAPABILITIES[capability]
        robustness = slot.get("robustness") or pick_robustness(rng, 0.85)
        if robustness in {"unsupported", "unavailable"} and _withholds_core_tool(
            capability, operation
        ):
            robustness = "ordinary"
        if (
            not slot.get("family")
            and attempt % 125 == 0
            and robustness == "ordinary"
            and capability in {"lights", "switches", "fans", "covers"}
            and operation in {"turn_on", "turn_off", "open", "close"}
        ):
            robustness = "exclusion"
        targeting = pick_targeting(cap, rng, robustness, operation=operation)
        if robustness == "large_home":
            home_size = max(home_size, 64)
        if (
            robustness == "exclusion"
            and not family
            and config.real_home_entity_cap > 0
        ):
            home_size = max(home_size, 64)
        inject_missing = family not in {"absence", "unsupported"}
        select_real_home = (
            rng.random() < config.real_home_rate
            if real_home_selected is None else real_home_selected
        )
        if family in REAL_HOME_EXCLUDED_FAMILIES:
            select_real_home = False
        variant = (
            pick_variant(capability, operation, slot["index"] + attempt, variants=grounding_variants)
            if grounding_variants else None
        )
        if variant is not None:
            pass
        elif config.real_home_path and select_real_home:
            home = load_real_home(config.real_home_path, split="train")
            real_home_context = json.loads(config.real_home_path.read_text(encoding="utf-8"))
            for entity in [*real_home_context["entities"], *real_home_context.get("context_entities", [])]:
                entity["name"] = " ".join(entity["name"].split())
                if entity.get("aliases"):
                    entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
            if family in REAL_HOME_REFUSAL_FAMILIES:
                home = copy.deepcopy(real_home_context)
                inject_missing = False
                robustness = "ambiguity"
                targeting = "area"
                if family == "absence":
                    areas, _ = home_areas(home)
                    absent_areas = [
                        area for area in areas
                        if area != "Unassigned" and not any(
                            entity["domain"] == cap.domain and entity.get("area") == area
                            for entity in home["entities"]
                        )
                    ]
                    if not absent_areas:
                        return None, "family_mismatch"
                    absence_area = rng.choice(absent_areas)
            real_home_row = True
        elif grounding_rate and grounding_required:
            variant = pick_variant(capability, operation, slot["index"] + attempt)
        elif grounding_rate and rng.random() < grounding_rate:
            variant = pick_variant(capability, operation, slot["index"] + attempt)
        if variant is not None:
            home = copy.deepcopy(variant["home"])
            targeting = variant["targeting"]
            robustness = variant["robustness"]
            grounding_family = variant["family"]
            if family not in {"absence", "unsupported"}:
                inject_missing = variant.get("inject_missing", True)
        scenario_index = slot["index"] + attempt * 10000
        if variant is not None and variant.get("rng_index") is not None:
            scenario_index = variant["rng_index"]

    if robustness == "exclusion" and operation == "query_state":
        # An exclusion ("but leave X alone") is an action concept; a status question has no clause to
        # carry it. Give the exclusion slot a trainable action op on the same capability instead.
        actions = sorted(op.name for op in trainable_operations(CAPABILITIES[capability]))
        if actions:
            operation = rng.choice(actions)
        else:
            robustness = "ordinary"
    scenario = build_scenario(
        index=scenario_index,
        seed=config.seed ^ (attempt << 16) ^ (retry << 24),
        capability=capability,
        operation=operation,
        home_size=home_size,
        targeting=targeting,
        robustness=robustness,
        split=config.split,
        attempt=attempt,
        home=home,
        inject_missing=inject_missing,
        target_usage=real_home_usage if real_home_row else None,
        family=None if real_home_row and family in REAL_HOME_REFUSAL_FAMILIES else family,
        bare_name_rate=config.bare_name_rate,
        request_intent={"area": absence_area} if absence_area else (
            variant.get("request_intent") if variant else None
        ),
    )
    if real_home_row:
        targets = [*(scenario.get("target_entities") or [])]
        if scenario.get("target_entity"):
            targets.append(scenario["target_entity"])
        if any(entity not in real_home_context["entities"] for entity in targets):
            return None, "family_mismatch"
        real_home_names = {entity["name"] for entity in real_home_context["entities"]}
        if any(name not in real_home_names for name in scenario["expected"].get("candidates") or []):
            return None, "family_mismatch"
        scenario["home"] = real_home_context
    if (
        capability == "media_players"
        and family in {"clarify", "follow_up"}
        and scenario.get("robustness") == "ambiguity"
    ):
        intent_area = (scenario.get("request_intent") or {}).get("area") or scenario["home"]["sayso_entity_area"]
        prepare_media_tv_ambiguity(scenario["home"], operation, intent_area, rng, slot["index"])
    if (
        (
            capability == "media_players"
            and family in {"clarify", "follow_up"}
            and scenario.get("robustness") == "ambiguity"
        )
        or (
            real_home_row
            and family in REAL_HOME_REFUSAL_FAMILIES | {"clarify", "follow_up"}
            and scenario["robustness"] == "ambiguity"
        )
    ):
        scenario["expected"] = gold_from_scenario(scenario, rng)
        scenario["semantic_id"] = semantic_id(scenario)
    if real_home_row:
        if family == "junk":
            scenario["expected"] = expected_no_action("not_understood")
        scenario["semantic_id"] = semantic_id(scenario)
        if absence_area:
            scenario["semantic_id"] += f"_{pair_hash(absence_area, real_home_context)[:8]}"
    if grounding_family:
        scenario["phrasing_seed"] = variant["phrasing_seed"]

    media_ambiguity = capability == "media_players" and family in {"clarify", "follow_up"}
    if media_ambiguity and scenario["expected"].get("response") != "clarify":
        kind = scenario["expected"].get("kind")
        family = {"action": "ordinary", "status": "status", "no_action": "absence"}.get(kind, family)
        scenario["family"] = family
    spec = scenario_to_spec(scenario)
    if media_ambiguity:
        spec["request_intent"] = scenario["request_intent"]
    if family == "absence" and not media_ambiguity and not real_home_row and rng.random() < UNEXPOSED_DOMAIN_RATE:
        unexposed = expected_unexposed_domain(spec["home"], rng)
        if unexposed is not None:
            spec["expected"] = unexposed
            spec["subcategory"] = "unexposed_domain"
    if family:
        spec["family"] = family
    spec["full_tool_catalog"] = True
    context_name = None
    if (
        family == "unsupported"
        and spec["home"].get("context_entities")
        and not is_blocked_request(family, slot.get("capability"), slot.get("operation"))
        and rng.random() < CONTEXT_HARD_NEGATIVE_RATE
    ):
        context_name = rng.choice(spec["home"]["context_entities"])["name"]
        spec["expected"] = expected_no_action("device_unsupported", unsupported_names=[context_name])
    expected = spec.get("expected") or {}
    if expected.get("kind") == "reject":
        return None, expected["reason"]
    if family and not gold_matches_family(expected, family):
        return None, "family_mismatch"
    follow_up_clarify_candidates: list[str] | None = None
    if family == "correction":
        entity = scenario.get("target_entity")
        if entity is None or entity_name_contains_area(entity):
            return None, "family_mismatch"
        spec["correction_wrong_name"] = f"{entity['area']} {entity['name']}"
        area = entity["area"]
        name = entity["name"]
        spec["spoken_targets"] = {name: f"the {area} {name}"}
    if family == "follow_up":
        candidates = list(expected.get("candidates") or [])
        if expected.get("response") != "clarify" or len(candidates) < 2:
            return None, "family_mismatch"
        follow_up_clarify_candidates = candidates[:2]
    if expected.get("kind") == "no_action":
        hint_spec = spec
        if absence_area:
            hint_spec = {**spec, "home": {**spec["home"], "sayso_entity_area": absence_area}}
        spec["request_hint"] = unique_no_action_hint(hint_spec, rng)
    elif robustness == "ambiguity":
        apply_generic_wording(spec)
    context_hard_negative = False
    if family == "datetime":
        spec["utterance"] = _datetime_utterance(rng, slot["index"])
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "GetDateTime"}]
    elif family == "junk":
        spec["utterance"] = _junk_utterance(rng, slot["index"])
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "not_understood"}]
        spec["expected"] = {**expected, "response": "not_understood"}
        spec.pop("request_hint", None)
        spec.pop("clarify_question", None)
    elif spec.get("subcategory") == "unexposed_domain":
        context_hard_negative = True
        target = spec["expected"]["unavailable"]["name"]
        state = rng.choice(("on", "off"))
        spec["utterance"] = rng.choice((
            f"turn {state} {target}",
            f"switch {target} {state}",
            f"power {state} {target}",
            f"set {target} to {state}",
            f"could you turn {target} {state}?",
            f"can you switch {target} {state} for me",
        ))
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "unexposed_domain"}]
        spec.pop("request_hint", None)
        spec.pop("clarify_question", None)
    elif context_name is not None:
        context_hard_negative = True
        state = rng.choice(("on", "off"))
        template = rng.choice((
            "turn {state} {name}",
            "switch {name} {state}",
            "could you turn {name} {state}?",
        ))
        spec["utterance"] = template.format(name=context_name, state=state)
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "device_unsupported"}]
        spec.pop("request_hint", None)
        spec.pop("clarify_question", None)
    elif family == "follow_up":
        clarify_expected = expected
        clarify_spec = {**spec, "expected": clarify_expected}
        spec["utterance"] = expand_utterance(clarify_spec)
        chosen = rng.choice(follow_up_clarify_candidates or [])
        entity = next(
            (item for item in scenario["home"]["entities"] if item["name"] == chosen),
            None,
        )
        if entity is None:
            return None, "family_mismatch"
        spec["expected"] = (
            expected_status(entity) if media_ambiguity and operation == "query_state"
            else expected_action(entity, operation, rng)
        )
        if spec.get("hint_call"):
            hint_args = spec["hint_call"].get("arguments") or {}
            gold_args = spec["expected"]["calls"][0]["arguments"]
            for key, value in hint_args.items():
                if key in {"name", "area", "floor", "domain", "device_class"} or key not in gold_args:
                    continue
                gold_args[key] = value
        spec["follow_up_clarify_candidates"] = follow_up_clarify_candidates
        spec["follow_up_reply"] = _follow_up_reply(chosen, rng)
        spec["target_names"] = target_names_from_expected(spec["expected"])
        spec["_follow_up_gold_targets"] = spec["target_names"]
        spec["target_names"] = []
    else:
        spec["utterance"] = expand_utterance(spec)

    discrimination = False
    if (
        discrimination_required
        and variant is None
        and expected.get("kind") == "action"
        and scenario.get("targeting") == "individual"
        and len(expected.get("calls") or []) == 1
        and not call_carries_value((expected.get("calls") or [{}])[0])
    ):
        seed_text = request_seed_from_spec(spec)
        description = pick_discriminating_description(scenario, rng)
        if description:
            spec["utterance"] = discrimination_utterance(
                expected["calls"][0]["name"], seed_text, description
            )
            discrimination = True
            spec["discrimination"] = True

    if grounding_required and not grounding_family:
        return None, "grounding_variant_miss"

    if real_home_row and not family and check_quality_eval_overlap(spec["utterance"]):
        _rephrase_past_eval_overlap(spec)
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    if family != "junk" and not context_hard_negative:
        spec["utterance"] = finalize_training_utterance(
            spec["utterance"], rng, upper=_balanced_upper(casing_counts, spec)
        )

    if family != "junk" and not context_hard_negative and rows_remaining > 0:
        if stt_log_remaining > 0 and rng.random() < stt_log_remaining / rows_remaining:
            corrupted, kind = apply_log_stt_noise(
                spec["utterance"], rng, target_names=spec.get("target_names")
            )
            if kind:
                trial = dict(spec)
                trial["utterance"] = corrupted
                trial["stt_corruption"] = kind
                if validate_row(trial, token_budget=config.token_budget) is None:
                    spec = trial
        elif stt_remaining > 0 and rng.random() < stt_remaining / rows_remaining:
            corrupted, kind = apply_stt_noise(
                spec["utterance"],
                rng,
                target_names=spec.get("target_names"),
                force_transform=True,
            )
            if kind:
                trial = dict(spec)
                trial["utterance"] = corrupted
                trial["stt_corruption"] = kind
                if validate_row(trial, token_budget=config.token_budget) is None:
                    spec = trial

    chatter_placement = None
    if (
        expected.get("kind") == "action"
        and expected.get("calls")
        and not context_hard_negative
        and family not in {"junk", "datetime", "follow_up"}
        and rng.random() < WRAPPED_COMMAND_RATE
    ):
        spec["utterance"], chatter_placement = wrap_utterance_with_chatter(
            spec["utterance"], rng
        )

    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason

    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject

    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )

    targets = spec.get("target_names") or []
    if real_home_row and real_home_entity_cap and real_home_targets is not None:
        if any(real_home_targets[name] >= real_home_entity_cap for name in targets):
            return None, "real_home_entity_cap"

    if family == "follow_up" and spec.get("_follow_up_gold_targets") is not None:
        spec["target_names"] = spec.pop("_follow_up_gold_targets")

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    row["metadata"]["real_home"] = real_home_row
    row["metadata"]["grounding_family"] = grounding_family
    row["metadata"]["discrimination"] = discrimination
    if chatter_placement:
        row["metadata"]["chatter_wrap"] = chatter_placement
    if family:
        row["metadata"]["family"] = family
    gate_row = _labelled_gate_row(row)
    reason = validate_serialized_row(
        gate_row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
    if reason is None and gate_row is not row:
        reason = validate_token_budget(row, config.token_budget, tokenizer_model=config.tokenizer_model)
    if reason:
        return None, reason
    if quota is not None and family not in {"datetime", "junk"}:
        from generators.coverage import classify_row

        reason = quota.wants(classify_row(row))
        if reason:
            return None, reason
    dup_tracker.record(spec)
    if real_home_row and real_home_targets is not None:
        real = [name for name in targets if not real_home_names or name in real_home_names]
        real_home_targets.update(real)
        if real_home_usage is not None:
            real_home_usage.update(real)
    if config.pair_rates.get("withheld_tool") and not (real_home_row or grounding_family or discrimination):
        row["_pair_spec"] = spec
    return row, None
