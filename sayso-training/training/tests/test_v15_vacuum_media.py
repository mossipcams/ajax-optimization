from __future__ import annotations

import random

from adapters.schema import tool_schema_map, v2_openai_tools, validate_tool_arguments
from generators.homes import generate_home
from generators.scenarios.core import build_scenario
from generators.tools import build_call_for_operation, build_media_pause

SCHEMAS = tool_schema_map(v2_openai_tools())
AREAS = ("Kitchen", "Living Room", "Hallway", "Garage")
FLOORS = ("Upstairs", "Main Floor", "Basement")
MEDIA_DEVICE_CLASSES = {"receiver", "projector"}


def test_vacuum_area_calls_validate_against_v2_schema_over_seeds() -> None:
    for seed in range(50):
        rng = random.Random(seed)
        for area in AREAS:
            call = build_call_for_operation(None, "vacuums", "clean_area", rng, area=area)
            assert call == {"name": "HassVacuumCleanArea", "arguments": {"area": area}}, call
            assert validate_tool_arguments(call["name"], call["arguments"], SCHEMAS) is None, call


def test_vacuum_floor_calls_validate_against_v2_schema_over_seeds() -> None:
    for seed in range(50):
        rng = random.Random(seed)
        for floor in FLOORS:
            call = build_call_for_operation(None, "vacuums", "clean_area", rng, floor=floor)
            assert call == {"name": "HassVacuumStart", "arguments": {"domain": ["vacuum"], "floor": floor}}, call
            assert validate_tool_arguments(call["name"], call["arguments"], SCHEMAS) is None, call


def test_vacuum_calls_in_generated_scenarios_validate_against_v2_schema() -> None:
    checked = 0
    for seed in range(3):
        for index in range(60):
            scenario = build_scenario(
                index=index, seed=20261103 + seed, capability="vacuums", operation="clean_area",
                home_size=24, family="ordinary",
            )
            for call in scenario["expected"].get("calls", []):
                if not call["name"].startswith("HassVacuum"):
                    continue
                checked += 1
                assert validate_tool_arguments(call["name"], call["arguments"], SCHEMAS) is None, call
    assert checked > 0


def test_receiver_and_projector_players_are_generated_with_schema_device_classes() -> None:
    names_by_class: dict[str, set[str]] = {device_class: set() for device_class in MEDIA_DEVICE_CLASSES}
    for index in range(300):
        home = generate_home(index, 40, random.Random(index))
        for entity in home["entities"]:
            if entity["capability"] == "media_players" and entity["device_class"] in MEDIA_DEVICE_CLASSES:
                names_by_class[entity["device_class"]].add(entity["name"])
    schema_classes = set(SCHEMAS["HassMediaPause"]["properties"]["device_class"]["items"]["enum"])
    assert MEDIA_DEVICE_CLASSES <= schema_classes, schema_classes
    for device_class, names in names_by_class.items():
        assert names, device_class
        for name in names:
            assert device_class in name.casefold(), name


def test_receiver_and_projector_media_calls_validate_against_v2_schema() -> None:
    checked = 0
    for index in range(300):
        home = generate_home(index, 40, random.Random(index))
        for entity in home["entities"]:
            if entity["capability"] != "media_players" or entity["device_class"] not in MEDIA_DEVICE_CLASSES:
                continue
            checked += 1
            call = build_media_pause(entity)
            assert validate_tool_arguments(call["name"], call["arguments"], SCHEMAS) is None, call
            call = build_call_for_operation(entity, "media_players", "mute", random.Random(index))
            assert validate_tool_arguments(call["name"], call["arguments"], SCHEMAS) is None, call
    assert checked > 0
