
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from training_paths import SAYSO

_SPEC = importlib.util.spec_from_file_location(
    "sayso_lfm_parse", SAYSO / "custom_components" / "sayso" / "lfm_parse.py"
)
_LFM_PARSE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LFM_PARSE
_SPEC.loader.exec_module(_LFM_PARSE)
parse_lfm_python_tool_call = _LFM_PARSE.parse_lfm_python_tool_call
parse_lfm_python_tool_calls = _LFM_PARSE.parse_lfm_python_tool_calls


def test_parse_apostrophe_names_in_single_quoted_values() -> None:
    omalleys = parse_lfm_python_tool_call(
        "HassTurnOff(name='O'Malley's Study Blinds', device_class=['blind'])"
    )
    assert omalleys["name"] == "HassTurnOff"
    assert omalleys["arguments"]["name"] == "O'Malley's Study Blinds"
    assert omalleys["arguments"]["device_class"] == ["blind"]

    kids = parse_lfm_python_tool_call("HassTurnOff(name='Kids' Room Light', domain=['light'])")
    assert kids["arguments"]["name"] == "Kids' Room Light"
    assert kids["arguments"]["domain"] == ["light"]


def test_parse_normal_name_without_interior_apostrophe() -> None:
    parsed = parse_lfm_python_tool_call("HassTurnOn(name='Office Main Light', domain=['light'])")
    assert parsed["arguments"]["name"] == "Office Main Light"


def test_parse_multiple_tool_calls_in_bracketed_block() -> None:
    calls = parse_lfm_python_tool_calls(
        "[HassLightSet(name='Den Desk Lamp', domain=['light'], brightness=40), "
        "HassTurnOff(name='Foyer Outlet', domain=['switch'])]"
    )
    assert len(calls) == 2
    assert calls[0]["name"] == "HassLightSet"
    assert calls[1]["arguments"]["name"] == "Foyer Outlet"
