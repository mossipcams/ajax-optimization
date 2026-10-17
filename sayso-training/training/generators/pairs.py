from __future__ import annotations

import copy
import random
from typing import Any

from generators.context import area_context_for
from generators.coverage import bare_tool_name
from generators.gold import expected_action, expected_no_action
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
    if str(home.get("home_id") or "").startswith("real_home"):
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


__all__ = ["CLARIFY_PAIR_SUFFIX", "WITHHELD_PAIR_TOOLS", "clarify_sibling", "withheld_sibling"]
