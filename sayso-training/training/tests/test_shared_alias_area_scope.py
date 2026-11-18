from __future__ import annotations

from generators.validation import shared_alias_action_conflict, validate_spec


def _entity(name: str, area: str, aliases: list[str] | None = None) -> dict:
    entity = {"name": name, "area": area}
    if aliases is not None:
        entity["aliases"] = aliases
    return entity


def _action(utterance: str, name: str, entities: list[dict]) -> dict:
    return {
        "utterance": utterance,
        "home": {"entities": entities},
        "expected": {"kind": "action", "calls": [{"name": "HassTurnOff", "arguments": {"name": name}}]},
    }


def _lamp_plug_home() -> list[dict]:
    return [
        _entity("Master Bedroom Lamp Plug", "Master Bedroom"),
        _entity("Kitchen Table Lamp Plug", "Kitchen"),
        _entity("Kitchen Floor Lamp Plug", "Kitchen"),
    ]


def test_name_tail_shared_only_in_another_area_is_not_flagged():
    spec = _action("please switch the lamp plug off", "Master Bedroom Lamp Plug", _lamp_plug_home())
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) is None


def test_name_tail_shared_in_the_targets_own_area_is_flagged():
    entities = [
        _entity("Master Bedroom Table Lamp Plug", "Master Bedroom"),
        *_lamp_plug_home(),
    ]
    spec = _action("please switch the lamp plug off", "Master Bedroom Lamp Plug", entities)
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_shared_alias_shared_in_another_area_still_flags_home_wide():
    entities = [
        _entity("Master Bedroom Lamp Plug", "Master Bedroom", ["lamp plug"]),
        _entity("Kitchen Table Plug", "Kitchen", ["lamp plug"]),
        _entity("Kitchen Floor Plug", "Kitchen", ["lamp plug"]),
    ]
    spec = _action("please switch the lamp plug off", "Master Bedroom Lamp Plug", entities)
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_shared_alias_in_the_targets_own_area_is_flagged():
    entities = [
        _entity("Master Bedroom Table Plug", "Master Bedroom", ["lamp plug"]),
        _entity("Master Bedroom Lamp Plug", "Master Bedroom", ["lamp plug"]),
        _entity("Kitchen Table Plug", "Kitchen", ["lamp plug"]),
    ]
    spec = _action("please switch the lamp plug off", "Master Bedroom Lamp Plug", entities)
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_phrase_in_another_area_with_no_match_in_the_action_area_is_flagged():
    entities = [
        _entity("Kitchen Table Lamp Plug", "Kitchen"),
        _entity("Kitchen Floor Lamp Plug", "Kitchen"),
        _entity("Lounge Ceiling Fan", "Lounge"),
    ]
    spec = _action("please switch the lamp plug off in the kitchen", "Lounge Ceiling Fan", entities)
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_shared_alias_with_no_match_in_the_action_area_is_flagged():
    entities = [
        _entity("Kitchen Table Plug", "Kitchen", ["lamp plug"]),
        _entity("Kitchen Floor Plug", "Kitchen", ["lamp plug"]),
        _entity("Lounge Ceiling Fan", "Lounge"),
    ]
    spec = _action("please switch the lamp plug off in the kitchen", "Lounge Ceiling Fan", entities)
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"
