from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

from generators.capability_registry import CAPABILITIES, SupportLevel
from generators.coverage import bare_tool_name, expected_tool
from generators.gold import expected_no_action, target_names_from_expected
from generators.utterances import finalize_training_utterance, request_seed_from_spec

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

PAIR_KINDS = ("withheld_tool",)
PAIR_SLOT_BIAS = 0.1
PAIR_PHRASING_ATTEMPTS = 4


def withheld_pair_targets() -> list[dict[str, str]]:
    return [
        {"capability": capability.name, "operation": operation.name}
        for capability in CAPABILITIES.values()
        if capability.support == SupportLevel.SUPPORTED
        for operation in capability.operations
        if operation.support == SupportLevel.SUPPORTED
        and operation.tool_name
        and bare_tool_name(operation.tool_name) in WITHHELD_PAIR_TOOLS
    ]


def retarget_keeps_tool_floor(
    slot: dict[str, Any], accepted_by_tool: Mapping[str, int], tool_floor: int
) -> bool:
    tool = expected_tool(slot.get("capability", ""), slot.get("operation", ""))
    return tool is None or tool_floor <= 1 or accepted_by_tool.get(tool, 0) >= tool_floor


def withheld_sibling(spec: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    expected = spec.get("expected") or {}
    if expected.get("kind") != "action":
        return None
    calls = expected.get("calls") or []
    if len(calls) != 1:
        return None
    tool = bare_tool_name(calls[0].get("name"))
    if tool not in WITHHELD_PAIR_TOOLS:
        return None
    utterance = _refusal_utterance(spec, expected, rng)
    if utterance is None:
        return None
    sibling = copy.deepcopy(spec)
    sibling.pop("request_hint", None)
    sibling["expected"] = expected_no_action(
        "unsupported", requested=copy.deepcopy(expected), unavailable_tools=[tool]
    )
    sibling["family"] = "unavailable"
    sibling["utterance"] = utterance
    base_id = spec.get("candidate_id") or ""
    sibling["candidate_id"] = f"{base_id}_withheld"
    semantic_id = spec.get("semantic_id") or base_id
    sibling["semantic_id"] = f"{semantic_id}_withheld"
    return sibling


def _refusal_utterance(spec: dict[str, Any], requested: dict[str, Any], rng: random.Random) -> str | None:
    parent = (spec.get("utterance") or "").casefold()
    base = spec.get("phrasing_seed") or spec.get("candidate_id") or ""
    for attempt in range(1, PAIR_PHRASING_ATTEMPTS + 1):
        probe = {
            **spec,
            "expected": requested,
            "target_names": target_names_from_expected(requested),
            "phrasing_seed": f"{base}:withheld{attempt}",
        }
        text = finalize_training_utterance(request_seed_from_spec(probe), rng)
        if text and text.casefold() != parent:
            return text
    return None
