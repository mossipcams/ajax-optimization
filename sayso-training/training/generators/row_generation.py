
from __future__ import annotations

from collections import Counter

import copy
import json
import random
import re
from typing import Any

from generators.capability_registry import CAPABILITIES, operation_spec
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker, input_hash, pair_hash
from generators.grounding import pick_variant
from generators.correction import entity_name_contains_area
from generators.gold import expected_action, expected_no_action, gold_from_scenario, gold_matches_family, target_names_from_expected
from generators.labels import render_example, scenario_to_spec
from generators.real_home import holdout_entity_names
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.scenarios.core import home_areas, semantic_id
from generators.scenarios.discrimination import (
    call_carries_value,
    pick_discriminating_description,
)
from generators.planning import CORE_TOOL_NAMES, REAL_HOME_EXCLUDED_FAMILIES, REAL_HOME_REFUSAL_FAMILIES
from generators.scenarios.unavailable import unique_no_action_hint
from generators.stt_noise import apply_log_stt_noise, apply_stt_noise
from generators.utterances import (
    apply_generic_wording,
    expand_utterance,
    finalize_training_utterance,
    request_seed_from_spec,
)
from generators.validation import (
    check_quality_eval_overlap,
    validate_row,
    validate_serialized_row,
)

CONTEXT_HARD_NEGATIVE_RATE = 0.30

_DATETIME_UTTERANCES = (
    "what time is it",
    "tell me the time",
    "what's the time right now",
    "what is the current time",
    "what time is it now",
)


def _datetime_utterance(rng: random.Random, index: int) -> str:
    return rng.choice(_DATETIME_UTTERANCES)


_JUNK_HEADS = (
    "but you're being", "and then he said", "i'll translate", "we were talking",
    "she said something about", "it was just", "they're going to", "i think it was",
    "so anyway", "that's what i", "you know how", "he's been",
)
_JUNK_TAILS = (
    "safe away", "ready in the meal", "for the weekend", "over at the place",
    "the other one", "about half of it", "before they left", "kind of thing",
    "in a minute or two", "all the way down", "sort of like that", "on the way back",
)
_JUNK_FRAGMENTS = (
    "war", "for gv", "uh huh", "hmm", "okay so", "right then", "no no",
    "wait", "mm hmm", "yeah but", "oh", "hang on", "what the", "and um",
)


def _follow_up_reply(chosen: str, rng: random.Random) -> str:
    templates = (chosen, f"the {chosen}", f"{chosen}, please")
    return rng.choice(templates)


def _junk_utterance(rng: random.Random, index: int) -> str:
    text = _junk_text(rng)
    return text[:1].upper() + text[1:] if rng.random() < 0.5 else text


def _junk_text(rng: random.Random) -> str:
    shape = rng.random()
    if shape < 0.30:
        return f"{rng.choice(_JUNK_FRAGMENTS)}."
    if shape < 0.55:
        return f"{rng.choice(_JUNK_FRAGMENTS)}, {rng.choice(_JUNK_FRAGMENTS)}."
    if shape < 0.80:
        return f"{rng.choice(_JUNK_HEADS)} {rng.choice(_JUNK_TAILS)}..."
    return (
        f"{rng.choice(_JUNK_HEADS)} {rng.choice(_JUNK_TAILS)} and... "
        f"{rng.choice(_JUNK_FRAGMENTS)}?"
    )


def casing_kind(has_calls: bool) -> str:
    return "call" if has_calls else "no_call"


def _withholds_core_tool(capability: str, operation: str) -> bool:
    op = operation_spec(capability, operation)
    return bool(op and op.tool_name and op.tool_name in CORE_TOOL_NAMES)


def _balanced_upper(casing_counts: dict[str, Counter[str]] | None, spec: dict[str, Any]) -> bool | None:
    if casing_counts is None:
        return None
    counts = casing_counts[casing_kind(bool((spec.get("expected") or {}).get("calls")))]
    if counts["upper"] - counts["lower"] >= 2:
        return False
    if counts["lower"] - counts["upper"] >= 2:
        return True
    return None


def normalize_entity_name(name: str) -> str:
    return " ".join(str(name).split())


def scaled_target_usage(usage: Any | None, target_weights: dict[str, float] | None) -> Counter:
    base = Counter(usage or ())
    if not target_weights:
        return base
    weights = {normalize_entity_name(name): float(weight) for name, weight in target_weights.items()}
    values = {}
    for name, count in base.items():
        key = normalize_entity_name(name)
        values[name] = count / weights[key] if key in weights else count
    return Counter(values)


def exceeds_max_target_share(real_home_targets: Counter, targets: list[str], max_target_share: float) -> bool:
    bound = max(25, max_target_share * (sum(real_home_targets.values()) + len(targets)))
    return any(real_home_targets[target] + 1 > bound for target in targets)


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
            f"{spec['semantic_id']}_{input_hash(spec)[:8]}"
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
    home_size = slot["home_size"]
    home = None
    real_home_context = None
    real_context_refusal = None
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
        targeting = pick_targeting(cap, rng, robustness)
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
            # Labels are decided on the exact home that is rendered: the full
            # snapshot, holdouts included. Holdouts are only barred as named targets.
            real_home_context = json.loads(config.real_home_path.read_text(encoding="utf-8"))
            for entity in [*real_home_context["entities"], *real_home_context.get("context_entities", [])]:
                entity["name"] = " ".join(entity["name"].split())
                if entity.get("aliases"):
                    entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
            home = copy.deepcopy(real_home_context)
            inject_missing = False
            if family in REAL_HOME_REFUSAL_FAMILIES:
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

    real_home_target_usage = None
    if real_home_row:
        frozen_entities = copy.deepcopy(home["entities"])
        holdouts = holdout_entity_names(config.real_home_path) if config.real_home_holdouts else frozenset()
        # pick_target takes the least-used candidate, so holdouts lose every tie.
        real_home_target_usage = scaled_target_usage(real_home_usage, config.real_home_target_weights)
        real_home_target_usage.update(dict.fromkeys(holdouts, 10**9))
    # Scenario builders pad or rewrite a home they are given; a real-home row
    # must show production's home unchanged. Another seed picks another area,
    # so retry a few builds before giving up on the real home for this slot.
    for shift in range(4 if real_home_row else 1):
        scenario = build_scenario(
            index=scenario_index + shift * 7919,
            seed=config.seed ^ (attempt << 16) ^ (retry << 24),
            capability=capability,
            operation=operation,
            home_size=home_size,
            targeting=targeting,
            robustness=robustness,
            split=config.split,
            attempt=attempt,
            home=copy.deepcopy(home) if real_home_row else home,
            inject_missing=inject_missing,
            target_usage=real_home_target_usage,
            family=None if real_home_row and family in REAL_HOME_REFUSAL_FAMILIES else family,
            bare_name_rate=config.bare_name_rate,
            request_intent={"area": absence_area} if absence_area else (
                variant.get("request_intent") if variant else None
            ),
        )
        if not real_home_row or scenario["home"]["entities"] == frozen_entities:
            break
    if real_home_row:
        if scenario["home"]["entities"] != frozen_entities:
            return None, "family_mismatch"
        if family in REAL_HOME_REFUSAL_FAMILIES and scenario["robustness"] == "ambiguity":
            scenario["expected"] = gold_from_scenario(scenario, rng)
        if family == "junk":
            scenario["expected"] = expected_no_action("not_understood")
        if family == "unsupported":
            context_only = real_home_context.get("context_entities") or []
            if not context_only:
                return None, "family_mismatch"
            device_tokens = {
                token for entity in real_home_context["entities"] for token in entity["name"].lower().split()
            } - {area.lower() for area in home_areas(real_home_context)[0]} - {"room", "the"}
            lookalikes = [
                entity for entity in context_only if device_tokens & set(entity["name"].lower().split())
            ]
            pool = lookalikes if lookalikes and rng.random() < 0.7 else context_only
            real_context_refusal = rng.choice(pool)["name"]
            scenario["expected"] = expected_no_action(
                "device_unsupported", unsupported_names=[real_context_refusal]
            )
        scenario["semantic_id"] = semantic_id(scenario)
        if absence_area:
            scenario["semantic_id"] += f"_{pair_hash(absence_area, real_home_context)[:8]}"
    if grounding_family:
        scenario["phrasing_seed"] = variant["phrasing_seed"]

    spec = scenario_to_spec(scenario)
    if family:
        spec["family"] = family
    spec["full_tool_catalog"] = True
    expected = spec.get("expected") or {}
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
    elif (
        family == "unsupported"
        and spec["home"].get("context_entities")
        and (real_context_refusal is not None or rng.random() < CONTEXT_HARD_NEGATIVE_RATE)
    ):
        context_hard_negative = True
        name = real_context_refusal or rng.choice(spec["home"]["context_entities"])["name"]
        state = rng.choice(("on", "off"))
        template = rng.choice((
            "turn {state} {name}",
            "switch {name} {state}",
            "could you turn {name} {state}?",
        ))
        spec["utterance"] = template.format(name=name, state=state)
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "device_unsupported"}]
        spec["expected"] = expected_no_action("device_unsupported", unsupported_names=[name])
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
        spec["expected"] = expected_action(entity, operation, rng)
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
            verb = "turn on"
            for candidate in ("turn on", "turn off", "open", "close", "lock", "unlock", "set", "run"):
                if seed_text.lower().startswith(candidate):
                    verb = candidate
                    break
            spec["utterance"] = f"{verb} {description}"
            discrimination = True
            spec["discrimination"] = True

    if grounding_required and not grounding_family:
        return None, "grounding_variant_miss"

    if (
        check_quality_eval_overlap(spec["utterance"])
        and expected.get("kind") == "action"
        and family not in {"follow_up", "datetime", "junk"}
        and not context_hard_negative
    ):
        # Canonical phrasing is deterministic, so a retry rebuilds the same
        # eval sentence (e.g. "turn on the living room tv"); rephrase instead.
        # Rephrase the action itself; finalize_training_utterance adds the one
        # conversational wrapper later, so wrapping here would nest two.
        action = request_seed_from_spec(spec)
        variants = [
            re.sub(r"^turn (on|off) (.+)$", r"switch \1 \2", action),
            re.sub(r"^turn (on|off) (.+)$", r"turn \2 \1", action),
            re.sub(r"^turn (on|off) (.+)$", r"switch \2 \1", action),
            f"please {action}",
        ]
        for candidate in rng.sample(variants, len(variants)):
            if candidate != action and not check_quality_eval_overlap(candidate):
                spec["utterance"] = candidate
                break
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

    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason

    if real_home_row:
        # ponytail: production runs on this one home, so its commands may take
        # 4x the phrasings of a synthetic one; raise if TV-style rows still cap out.
        spec["_near_limit"] = config.near_duplicate_limit * 4
    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject

    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{input_hash(spec)[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )

    targets = spec.get("target_names") or []
    if real_home_row and real_home_targets is not None:
        over_share = (
            config.real_home_max_target_share
            and exceeds_max_target_share(real_home_targets, targets, config.real_home_max_target_share)
        )
        over_cap = real_home_entity_cap and any(
            real_home_targets[name] >= real_home_entity_cap for name in targets
        )
        if over_cap or over_share:
            return None, "real_home_entity_cap"

    if family == "follow_up" and spec.get("_follow_up_gold_targets") is not None:
        spec["target_names"] = spec.pop("_follow_up_gold_targets")

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    if real_home_row and any(
        json.loads(call["function"]["arguments"]).get("name") in holdouts
        for message in row["messages"]
        for call in message.get("tool_calls") or []
    ):
        return None, "family_mismatch"
    row["metadata"]["real_home"] = real_home_row
    row["metadata"]["grounding_family"] = grounding_family
    row["metadata"]["discrimination"] = discrimination
    if family:
        row["metadata"]["family"] = family
    reason = validate_serialized_row(
        row, token_budget=config.token_budget, tokenizer_model=config.tokenizer_model
    )
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
    return row, None
