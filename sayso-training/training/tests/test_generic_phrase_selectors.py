from __future__ import annotations

from generators.validation import shared_alias_action_conflict, validate_spec


def _entity(name: str, area: str, capability: str, features: tuple[str, ...] = ()) -> dict:
    return {"name": name, "area": area, "aliases": [name], "capability": capability, "capabilities": list(features)}


def _spec(utterance: str, name: str, entities: list[dict], tool: str, capability: str, operation: str) -> dict:
    return {
        "utterance": utterance,
        "capability": capability,
        "operation": operation,
        "home": {"entities": entities},
        "expected": {"kind": "action", "calls": [{"name": tool, "arguments": {"name": name}}]},
    }


def test_sharer_that_cannot_perform_the_operation_does_not_make_phrase_ambiguous():
    entities = [
        _entity("Guest Bedroom Streaming TV", "Guest Bedroom", "media_players", ("mute",)),
        _entity("Kitchen Streaming TV", "Guest Bedroom", "media_players"),
    ]
    spec = _spec("mute the streaming tv", "Guest Bedroom Streaming TV", entities,
                 "HassMediaPlayerMute", "media_players", "mute")
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_sharers_that_can_both_perform_the_operation_are_still_ambiguous():
    entities = [
        _entity("Guest Bedroom Streaming TV", "Guest Bedroom", "media_players", ("mute",)),
        _entity("Kitchen Streaming TV", "Guest Bedroom", "media_players", ("mute",)),
    ]
    spec = _spec("mute the streaming tv", "Guest Bedroom Streaming TV", entities,
                 "HassMediaPlayerMute", "media_players", "mute")
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_longer_suffix_in_text_selects_the_one_sharer_it_belongs_to():
    entities = [
        _entity("Nursery left bedside lamp", "Nursery", "lights"),
        _entity("Nursery bedside lamp", "Nursery", "lights"),
    ]
    spec = _spec("switch the left bedside lamp on", "Nursery left bedside lamp", entities,
                 "HassTurnOn", "lights", "turn_on")
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_bare_tail_that_matches_no_longer_suffix_is_still_ambiguous():
    entities = [
        _entity("Nursery left bedside lamp", "Nursery", "lights"),
        _entity("Nursery bedside lamp", "Nursery", "lights"),
    ]
    spec = _spec("switch the bedside lamp on", "Nursery left bedside lamp", entities,
                 "HassTurnOn", "lights", "turn_on")
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_generic_floor_lamp_with_two_capable_same_area_lamps_is_still_flagged():
    entities = [
        _entity("Living Floor Lamp", "Living Room", "lights"),
        _entity("Kitchen Floor Lamp", "Living Room", "lights"),
    ]
    spec = _spec("turn on the floor lamp", "Living Floor Lamp", entities,
                 "HassTurnOn", "lights", "turn_on")
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"
