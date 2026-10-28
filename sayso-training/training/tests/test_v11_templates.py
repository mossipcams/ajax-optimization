from __future__ import annotations

import pytest

from generators.utterances import _phrase_for_call, request_seed_from_spec
from generators.validation import check_fuzzy_eval_overlap

_SEEDS = [str(seed) for seed in range(200)]


def _phrases(target: str, call: dict, seeds=_SEEDS) -> set[str]:
    return {_phrase_for_call(target, call, seed) for seed in seeds}


def test_dock_phrasings_map_to_return_to_base() -> None:
    call = {"name": "HassVacuumReturnToBase", "arguments": {"name": "Roomba"}}
    phrases = _phrases("Roomba", call)
    assert {"send Roomba back to its dock", "tell Roomba to dock", "send Roomba back to its base", "tell Roomba to go back and charge"} <= phrases
    assert all("Roomba" in phrase for phrase in phrases)


def test_clean_area_phrasings_keep_the_area_argument() -> None:
    call = {"name": "HassVacuumCleanArea", "arguments": {"name": "Roomba", "area": "Kitchen"}}
    phrases = _phrases("Roomba", call)
    assert {"clean the Kitchen with Roomba", "tell Roomba to vacuum the Kitchen", "tell Roomba to clean the Kitchen"} <= phrases
    assert all("Kitchen" in phrase for phrase in phrases)


def _script_spec(entities: list[dict], target_names: list[str], call_name: str, candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "home": {"entities": entities},
        "target_names": target_names,
        "spoken_targets": {},
        "expected": {"kind": "action", "calls": [{"name": call_name, "arguments": {}}]},
    }


def _script(name: str, object_id: str) -> dict:
    return {
        "name": name,
        "entity_id": f"script.{object_id}",
        "domain": "script",
        "capability": "scripts",
        "aliases": [name],
    }


@pytest.mark.parametrize("index", range(40))
def test_named_routine_action_uses_script_tool_and_name(index: int) -> None:
    entities = [_script("Movie Mode", "movie_mode"), _script("Welcome Home", "welcome_home")]
    spec = _script_spec(entities, ["Movie Mode"], "movie_mode", f"routine-{index}")
    utterance = request_seed_from_spec(spec)
    assert utterance in {
        "run my Movie Mode routine", "start the Movie Mode routine",
        "run the routine called Movie Mode", "go ahead and kick off Movie Mode", "run Movie Mode",
    } or utterance.startswith("run ")
    assert "Movie Mode" in utterance or utterance.startswith("run ")


def test_named_routine_templates_appear_for_unique_script() -> None:
    entities = [_script("Movie Mode", "movie_mode"), _script("Welcome Home", "welcome_home")]
    utterances = {
        request_seed_from_spec(_script_spec(entities, ["Movie Mode"], "movie_mode", f"routine-{index}"))
        for index in range(200)
    }
    assert {"run my Movie Mode routine", "start the Movie Mode routine", "run the routine called Movie Mode", "go ahead and kick off Movie Mode"} <= utterances


def test_named_routine_templates_skip_duplicate_names() -> None:
    entities = [_script("Movie Mode", "movie_mode"), _script("Movie Mode", "movie_mode_two")]
    spec = _script_spec(entities, ["Movie Mode"], "movie_mode", "routine-dup")
    assert all("routine called" not in request_seed_from_spec({**spec, "candidate_id": f"d{index}"}) for index in range(50))


def test_brightness_fractions_use_correct_integers() -> None:
    expected = {25: "a quarter", 33: "a third", 50: "half"}
    for level, word in expected.items():
        call = {"name": "HassLightSet", "arguments": {"name": "Kitchen Light", "brightness": level}}
        phrases = _phrases("the kitchen light", call)
        assert any(word in phrase for phrase in phrases), level
        assert all(str(level) not in phrase for phrase in phrases if word in phrase)
        for other_level, other_word in expected.items():
            if other_level != level:
                assert not any(other_word in phrase for phrase in phrases), (level, other_word)


def test_percent_down_and_up_phrasings_use_the_call_value() -> None:
    down = {"name": "HassLightSet", "arguments": {"name": "Kitchen Light", "brightness": 40}}
    up = {"name": "HassLightSet", "arguments": {"name": "Kitchen Light", "brightness": 80}}
    assert any("down to 40 percent" in phrase for phrase in _phrases("the kitchen light", down))
    assert any("up to 80 percent" in phrase for phrase in _phrases("the kitchen light", up))
    assert all("80" not in phrase for phrase in _phrases("the kitchen light", down))


def test_fan_fraction_and_percent_speed_phrasings() -> None:
    half = {"name": "HassFanSetSpeed", "arguments": {"name": "Desk Fan", "percentage": 50}}
    quarter = {"name": "HassFanSetSpeed", "arguments": {"name": "Desk Fan", "percentage": 25}}
    down = {"name": "HassFanSetSpeed", "arguments": {"name": "Desk Fan", "percentage": 30}}
    assert any(phrase.endswith("half") for phrase in _phrases("the desk fan", half))
    assert any("a quarter" in phrase for phrase in _phrases("the desk fan", quarter))
    assert any("down to 30 percent" in phrase for phrase in _phrases("the desk fan", down))


def test_new_phrasings_pass_eval_overlap_check() -> None:
    calls = [
        ({"name": "HassVacuumReturnToBase", "arguments": {"name": "Roomba"}}, "Roomba"),
        ({"name": "HassVacuumCleanArea", "arguments": {"name": "Roomba", "area": "Kitchen"}}, "Roomba"),
        ({"name": "HassLightSet", "arguments": {"name": "Kitchen Light", "brightness": 25}}, "the kitchen light"),
        ({"name": "HassFanSetSpeed", "arguments": {"name": "Desk Fan", "percentage": 30}}, "the desk fan"),
    ]
    for call, target in calls:
        for phrase in _phrases(target, call, seeds=_SEEDS[:40]):
            assert not check_fuzzy_eval_overlap(phrase), phrase
