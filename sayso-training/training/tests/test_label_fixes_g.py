from __future__ import annotations

import random

from generators.homes import (
    _NICKNAMES,
    FLOOR_THERMOSTAT_RATE,
    FRESH_WORD_RATE,
    _eval_entity_names,
    _eval_word_tokens,
    _generator_word_tokens,
    fresh_word_name,
    fresh_words,
    generate_home,
)
from generators.validation import shared_alias_action_conflict, validate_spec

HOME_SIZES = (16, 24, 32)


def _homes(count: int = 240, *, holdout: bool = False) -> list[dict]:
    return [
        generate_home(index, HOME_SIZES[index % len(HOME_SIZES)],
                      random.Random(f"label_fixes_g:{index}"), holdout=holdout)
        for index in range(count)
    ]


def _entryway_home() -> dict:
    return {"entities": [
        {"name": "Entryway Plug", "area": "Entryway", "aliases": ["Entryway Plug", "Entryway outlet"]},
        {"name": "Entryway Lamp Plug", "area": "Entryway", "aliases": ["Entryway Lamp Plug", "Entryway outlet"]},
        {"name": "Kitchen Plug", "area": "Kitchen", "aliases": ["Kitchen Plug", "Entryway outlet"]},
    ]}


def _action(utterance: str, targets: list[str]) -> dict:
    return {
        "utterance": utterance,
        "home": _entryway_home(),
        "expected": {"kind": "action", "calls": [
            {"name": "HassTurnOn", "arguments": {"name": target, "domain": ["switch"]}}
            for target in targets
        ]},
    }


# --- validation: shared same-area alias in any action spec -------------------------------------

def test_area_wide_action_on_the_sharing_area_is_allowed():
    spec = _action("turn on the entryway outlet", [])
    spec["expected"]["calls"] = [{"name": "HassTurnOn", "arguments": {"area": "Entryway", "domain": ["switch"]}}]
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_area_wide_action_on_another_area_does_not_cover_the_sharers():
    spec = _action("turn on the entryway outlet", [])
    spec["expected"]["calls"] = [{"name": "HassTurnOn", "arguments": {"area": "Kitchen", "domain": ["switch"]}}]
    assert shared_alias_action_conflict(spec)


def test_plain_action_naming_a_same_area_shared_alias_is_rejected():
    spec = _action("turn on the entryway outlet", ["Entryway Plug"])
    assert "shared_alias" not in spec
    assert shared_alias_action_conflict(spec)
    assert validate_spec(spec) == "shared_alias_action"


def test_shared_alias_action_targeting_every_sharer_is_allowed():
    spec = _action("turn on the entryway outlet", ["Entryway Plug", "Entryway Lamp Plug"])
    assert not shared_alias_action_conflict(spec)
    assert validate_spec(spec) != "shared_alias_action"


def test_alias_shared_only_across_areas_is_allowed():
    spec = _action("turn on the entryway outlet", ["Entryway Plug"])
    spec["home"]["entities"] = spec["home"]["entities"][:1] + spec["home"]["entities"][2:]
    assert not shared_alias_action_conflict(spec)


def test_name_action_on_a_unique_alias_is_allowed():
    spec = _action("turn on the kitchen plug", ["Kitchen Plug"])
    assert not shared_alias_action_conflict(spec)


def test_generated_homes_reject_unambiguous_shared_area_alias_in_actions():
    for home in _homes(60):
        for entity in home["entities"]:
            for alias in entity["aliases"]:
                sharers = [other for other in home["entities"]
                           if other["area"] == entity["area"]
                           and alias.casefold() in {a.casefold() for a in other["aliases"]}]
                if len(sharers) < 2:
                    continue
                spec = {"utterance": f"turn on the {alias}", "home": home,
                        "expected": {"kind": "action", "calls": [
                            {"name": "HassTurnOn", "arguments": {"name": entity["name"]}}]}}
                assert shared_alias_action_conflict(spec)


# --- homes: floor-named thermostats ------------------------------------------------------------

def test_some_climate_entities_are_named_after_their_floor():
    climates = []
    for home in _homes():
        floors = set(home["area_floors"].values())
        for entity in home["entities"]:
            if entity["capability"] == "climate":
                climates.append((entity, floors))
    floor_named = [(e, floors) for e, floors in climates
                   if any(e["name"] == f"{floor} Thermostat" for floor in floors)]
    assert floor_named
    assert 0.05 <= len(floor_named) / len(climates) <= 0.5
    for entity, floors in floor_named:
        assert entity["floor"] == entity["name"][: -len(" Thermostat")]
        assert entity["floor"] in floors


def test_floor_named_thermostats_respect_eval_name_avoidance():
    reserved = _eval_entity_names()
    for home in _homes():
        for entity in home["entities"]:
            assert entity["name"].casefold() not in reserved


def test_floor_thermostat_rate_is_modest_and_names_are_deterministic():
    assert 0 < FLOOR_THERMOSTAT_RATE <= 0.3
    first = [e["name"] for h in _homes(30) for e in h["entities"]]
    second = [e["name"] for h in _homes(30) for e in h["entities"]]
    assert first == second


def test_holdout_homes_get_no_floor_named_thermostats():
    for home in _homes(60, holdout=True):
        floor_names = {f"{floor} Thermostat" for floor in home["area_floors"].values()}
        for entity in home["entities"]:
            if entity["capability"] == "climate":
                assert entity["name"] not in floor_names


# --- homes: fresh descriptor words -------------------------------------------------------------

def test_fresh_word_pool_is_large_and_free_of_eval_only_words():
    words = fresh_words()
    assert len(words) >= 100
    eval_only = _eval_word_tokens() - _generator_word_tokens()
    assert not {word.casefold() for word in words} & eval_only


def test_fresh_word_rate_is_about_ten_percent():
    assert FRESH_WORD_RATE == 0.10
    changed = 0
    trials = 2000
    for seed in range(trials):
        rng = random.Random(seed)
        name = fresh_word_name("Joe's Workshop Fan", "Office", rng, set())
        changed += name != "Joe's Workshop Fan"
    assert 0.05 <= changed / trials <= 0.16


def test_fresh_word_never_replaces_the_device_noun_or_area_prefix():
    pool = set(fresh_words())
    for seed in range(1500):
        name = fresh_word_name("Basement Ceiling Light", "Basement", random.Random(seed), set())
        words = name.split()
        assert words[0] == "Basement" and words[-1] == "Light"
        assert len(words) == 3
        assert words[1] in pool or words[1] == "Ceiling"
        name = fresh_word_name("Joe's Workshop Fan", "Office", random.Random(seed), set())
        words = name.split()
        assert words[-1] == "Fan"
        assert len(words) == 3


def test_holdout_homes_use_no_fresh_words():
    pool = {word.casefold() for word in fresh_words()}
    for home in _homes(60, holdout=True):
        for entity in home["entities"]:
            tokens = {token.casefold() for token in entity["name"].split()}
            assert not tokens & pool


def test_generated_names_never_equal_a_nickname_alias():
    nicknames = {nickname.casefold() for values in _NICKNAMES.values() for nickname in values}
    for home in _homes(120):
        assert not {e["name"].casefold() for e in home["entities"]} & nicknames


def test_generated_names_still_avoid_eval_names_with_fresh_words():
    reserved = _eval_entity_names()
    for home in _homes(120):
        names = [e["name"].casefold() for e in home["entities"]]
        assert len(names) == len(set(names))
        assert not set(names) & reserved
