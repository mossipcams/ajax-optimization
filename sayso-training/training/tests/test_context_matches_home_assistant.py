
from __future__ import annotations

import re
from pathlib import Path

import yaml

from generators.context import (
    SAYSO_SYSTEM_PROMPT,
    exposed_entities,
    serialize_context,
)

REPO = Path(__file__).resolve().parents[2]

HOME = {
    "sayso_entity_area": "Kitchen",
    "entities": [
        {
            "name": "Kitchen Light",
            "aliases": ["Counter Light"],
            "domain": "light",
            "device_class": None,
            "area": "Kitchen",
            "floor": "Ground",
            "state": "off",
            "capabilities": ["brightness"],
        },
        {
            "name": "Front Door Lock",
            "aliases": [],
            "domain": "lock",
            "device_class": "door",
            "area": "Hallway",
            "floor": "Ground",
            "state": "locked",
            "capabilities": [],
        },
    ],
}


def test_sayso_prompt_matches_the_integration_default() -> None:
    const = (REPO / "custom_components" / "sayso" / "const.py").read_text(encoding="utf-8")
    match = re.search(r'DEFAULT_SYSTEM_PROMPT = """(.*?)"""', const, re.DOTALL)
    assert match, "DEFAULT_SYSTEM_PROMPT not found in const.py"
    assert SAYSO_SYSTEM_PROMPT == match.group(1)


def test_entity_names_are_deduplicated() -> None:
    home = {
        "sayso_entity_area": "Kitchen",
        "entities": [
            {
                "name": "Great Room Thermostat",
                "aliases": ["Great Room Thermostat"],
                "domain": "climate",
                "area": "Great Room",
                "floor": "Ground",
                "state": "heat",
            }
        ],
    }
    assert exposed_entities(home)[0]["names"] == "Great Room Thermostat"


def test_overview_carries_no_state_and_uses_home_assistant_fields() -> None:
    rows = exposed_entities(HOME)
    assert [row["names"] for row in rows] == ["Front Door Lock", "Kitchen Light, Counter Light"]
    for row in rows:
        assert set(row) <= {"names", "domain", "areas"}
        assert "state" not in row
        assert "capabilities" not in row


def test_context_entities_render_sorted_together_with_devices() -> None:
    home = {
        "sayso_entity_area": "Living Room",
        "entities": [
            {
                "name": "Living Room Light",
                "aliases": ["Counter Light"],
                "domain": "light",
                "area": "Kitchen",
                "floor": "Ground",
                "state": "off",
                "capabilities": [],
            },
            {
                "name": "TV",
                "aliases": ["TV"],
                "domain": "media_player",
                "area": "Living Room",
                "floor": "Ground",
                "state": "idle",
                "capabilities": ["on"],
            },
        ],
        "context_entities": [
            {"name": "TV Active app", "domain": "sensor", "area": "Living Room"},
            {"name": "TV Application", "domain": "select", "area": "Living Room"},
            {"name": "Ben Carter", "domain": "person"},
            {"name": "Sun", "domain": "sun"},
        ],
    }
    rows = exposed_entities(home)
    assert [row["names"] for row in rows] == [
        "Ben Carter",
        "Living Room Light, Counter Light",
        "Sun",
        "TV",
        "TV Active app",
        "TV Application",
    ]
    tv_index = next(i for i, row in enumerate(rows) if row["names"] == "TV")
    assert rows[tv_index + 1] == {"names": "TV Active app", "domain": "sensor", "areas": "Living Room"}
    person_row = rows[0]
    assert set(person_row) == {"names", "domain"}, person_row


def test_context_entities_appear_in_serialized_prompt() -> None:
    home = dict(HOME)
    home["context_entities"] = [
        {"name": "Sun", "domain": "sun"},
        {"name": "Kitchen Motion", "domain": "binary_sensor", "area": "Kitchen"},
    ]
    prompt = serialize_context(home)
    block = prompt.split("smart home:\n", 1)[1].split("\nWhen controlling Home Assistant", 1)[0]
    parsed = yaml.safe_load(block)
    assert [
        {"names": "Front Door Lock", "domain": "lock", "areas": "Hallway"},
        {"names": "Kitchen Light, Counter Light", "domain": "light", "areas": "Kitchen"},
        {"names": "Kitchen Motion", "domain": "binary_sensor", "areas": "Kitchen"},
        {"names": "Sun", "domain": "sun"},
    ] == parsed


def test_home_without_context_entities_renders_unchanged() -> None:
    home = {key: value for key, value in HOME.items() if key != "context_entities"}
    prompt = serialize_context(home)
    block = prompt.split("smart home:\n", 1)[1].split("\nWhen controlling Home Assistant", 1)[0]
    assert yaml.safe_load(block) == [
        {"names": "Front Door Lock", "domain": "lock", "areas": "Hallway"},
        {"names": "Kitchen Light, Counter Light", "domain": "light", "areas": "Kitchen"},
    ]


def test_serialized_context_is_yaml_and_omits_live_state() -> None:
    prompt = serialize_context(HOME)
    assert "Static Context: An overview of the areas and the devices in this smart home:" in prompt
    assert "You are in area" not in prompt
    assert prompt.endswith(
        "Area context:\nsatellite_area: Kitchen\ntarget_area: Kitchen\ntarget_area_source: satellite"
    )
    assert "homeassistant__GetLiveContext" in prompt

    block = prompt.split("smart home:\n", 1)[1].split("\nWhen controlling Home Assistant", 1)[0]
    assert "locked" not in block and "off" not in block
    parsed = yaml.safe_load(block)
    assert parsed == [
        {"names": "Front Door Lock", "domain": "lock", "areas": "Hallway"},
        {"names": "Kitchen Light, Counter Light", "domain": "light", "areas": "Kitchen"},
    ]
