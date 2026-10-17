from __future__ import annotations

import copy
from typing import Any

from generators.coverage import bare_tool_name
from generators.gold import expected_no_action

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


__all__ = ["WITHHELD_PAIR_TOOLS", "withheld_sibling"]
