from __future__ import annotations

import copy
import random
import re
from typing import Any

from generators.context import area_context_for
from generators.coverage import bare_tool_name
from generators.gold import _type_label, expected_action, expected_no_action
from generators.scenarios.core import home_areas
from generators.utterances import finalize_training_utterance, request_seed_from_spec
from generators.validation import check_quality_eval_overlap

WITHHELD_PAIR_TOOLS = frozenset(
    {
        "HassLightSet",
        "HassMediaPause",
        "HassMediaUnpause",
        "HassMediaNext",
        "HassMediaPrevious",
        "HassSetVolume",
        "HassSetVolumeRelative",
        "HassClimateSetTemperature",
    }
)


def withheld_sibling(spec: dict[str, Any]) -> dict[str, Any] | None:
    expected = spec.get("expected") or {}
    if expected.get("kind") != "action":
        return None
    calls = expected.get("calls") or []
    if len(calls) != 1:
        return None
    tool = bare_tool_name(calls[0].get("name"))
    if tool not in WITHHELD_PAIR_TOOLS:
        return None
    sibling = copy.deepcopy(spec)
    original_expected = copy.deepcopy(expected)
    sibling["expected"] = expected_no_action(
        "unsupported", requested=original_expected, unavailable_tools=[tool]
    )
    sibling["family"] = "unavailable"
    base_id = spec.get("candidate_id") or ""
    sibling["candidate_id"] = f"{base_id}_withheld"
    sibling["semantic_id"] = f"{spec.get('semantic_id') or base_id}_withheld"
    return sibling


CLARIFY_PAIR_SUFFIX = "_clarify_pair"
_CLARIFY_ONLY_KEYS = (
    "clarify_question",
    "follow_up_clarify_candidates",
    "follow_up_reply",
    "request_hint",
    "hint_call",
    "spoken_targets",
)


def clarify_sibling(spec: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    expected = spec.get("expected") or {}
    candidates = list(expected.get("candidates") or [])
    if expected.get("kind") != "no_action" or expected.get("response") != "clarify" or len(candidates) < 2:
        return None
    home = spec.get("home") or {}
    if spec.get("real_home") or str(home.get("home_id") or "").startswith("real_home"):
        return None
    entity = next((item for item in home.get("entities") or [] if item["name"] == candidates[0]), None)
    operation = spec.get("operation")
    if entity is None or not operation:
        return None
    sibling = copy.deepcopy(spec)
    for key in _CLARIFY_ONLY_KEYS:
        sibling.pop(key, None)
    sibling["family"] = "ordinary"
    sibling["expected"] = expected_action(entity, operation, rng)
    sibling["target_names"] = [entity["name"]]
    sibling["utterance"] = finalize_training_utterance(request_seed_from_spec(sibling), rng)
    if check_quality_eval_overlap(sibling["utterance"]):
        return None
    if area_context_for(home, sibling["utterance"]) != area_context_for(home, spec["utterance"]):
        return None
    base_id = spec.get("candidate_id") or ""
    sibling["candidate_id"] = f"{base_id}{CLARIFY_PAIR_SUFFIX}"
    sibling["semantic_id"] = f"{spec.get('semantic_id') or base_id}{CLARIFY_PAIR_SUFFIX}"
    return sibling


_CONTRAST_STOP_WORDS = frozenset({"the", "room"})
_CONTRAST_DOMAINS = frozenset({"sun", "person"})


def _contrast_words(name: str, ignored: set[str]) -> set[str]:
    return {word for word in re.split(r"\W+", name.casefold()) if word and word not in ignored}


def real_home_contrast_siblings(spec: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if not spec.get("real_home") or expected.get("kind") != "action" or len(calls) != 1:
        return []
    target_name = (calls[0].get("arguments") or {}).get("name")
    utterance = spec.get("utterance") or ""
    if not target_name:
        return []
    index = utterance.casefold().find(str(target_name).casefold())
    if index < 0:
        return []
    home = spec.get("home") or {}
    entity = next((item for item in home.get("entities") or [] if item.get("name") == target_name), None)
    if entity is None:
        return []
    domain = entity.get("domain")
    areas, _floors = home_areas(home)
    stop_words = set(_CONTRAST_STOP_WORDS) | {area.casefold() for area in areas}
    context_entities = home.get("context_entities") or []
    lookalikes = [
        item["name"]
        for item in context_entities
        if domain and _contrast_words(target_name, stop_words) & _contrast_words(item["name"], stop_words)
    ]
    non_device = [item["name"] for item in context_entities if item.get("domain") in _CONTRAST_DOMAINS]
    absent_areas = [
        area
        for area in areas
        if area != "Unassigned"
        and not any(item.get("area") == area and item.get("domain") == domain for item in home.get("entities") or [])
    ]

    def build(suffix: str, replacement: str, label: dict[str, Any], family: str) -> dict[str, Any]:
        sibling = copy.deepcopy(spec)
        sibling["utterance"] = utterance[:index] + replacement + utterance[index + len(target_name):]
        sibling["expected"] = label
        sibling["family"] = family
        sibling["candidate_id"] = f"{spec.get('candidate_id') or ''}_{suffix}"
        sibling["semantic_id"] = f"{spec.get('semantic_id') or spec.get('candidate_id') or ''}_{suffix}"
        sibling["target_names"] = []
        sibling["linguistics"] = [{"source": "sayso_fallback"}]
        return sibling

    siblings: list[dict[str, Any]] = []
    if lookalikes:
        name = rng.choice(lookalikes)
        siblings.append(
            build("contrast_a", name, expected_no_action("device_unsupported", unsupported_names=[name]), "unsupported")
        )
    if non_device:
        name = rng.choice(non_device)
        siblings.append(
            build("contrast_b", name, expected_no_action("device_unsupported", unsupported_names=[name]), "unsupported")
        )
    if absent_areas:
        area = rng.choice(absent_areas)
        label = expected_no_action(
            "area_unavailable",
            unavailable={"area": area.casefold(), "type": _type_label(spec.get("capability") or "")},
        )
        siblings.append(build("contrast_c", f"{area} {target_name}", label, "absence"))
    return siblings


__all__ = [
    "CLARIFY_PAIR_SUFFIX",
    "WITHHELD_PAIR_TOOLS",
    "clarify_sibling",
    "real_home_contrast_siblings",
    "withheld_sibling",
]
