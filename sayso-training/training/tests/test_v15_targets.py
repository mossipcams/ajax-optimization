from __future__ import annotations

import random
import re

from generators.capability_registry import CAPABILITIES, is_satellite, is_satellite_name
from generators.context import exposed_entities
from generators.homes import generate_home
from generators.rendering import scenario_to_spec
from generators.scenarios.core import build_scenario
from generators.utterances import request_seed_from_spec
from generators.validation import validate_spec

SEEDS = range(3)
INDICES = range(400)
CONTRAST_CAPABILITIES = ("lights", "media_players")


def _generic_phrases(scenario) -> list[str]:
    area = scenario["home"]["sayso_entity_area"].casefold()
    return [
        f"the {area} {noun}" if form == 0 else f"the {noun} in the {area}"
        for noun in ("light", "speaker", "satellite")
        for form in (0, 1)
    ]


def _contrast_rows(capability: str, operation: str = "turn_off"):
    for seed in SEEDS:
        for index in INDICES:
            scenario = build_scenario(
                index=index, seed=20261103 + seed, capability=capability, operation=operation,
                home_size=24, family="ordinary",
            )
            phrases = _generic_phrases(scenario)
            spoken = {phrase.casefold() for phrase in (scenario.get("spoken_targets") or {}).values()}
            if spoken & set(phrases):
                yield scenario


def test_generated_satellite_names_never_contain_sayso_brand() -> None:
    names = []
    for index in range(600):
        home = generate_home(index, 24, random.Random(index))
        names += [e["name"] for e in home["entities"] if is_satellite(e)]
    assert names, "no satellites generated"
    for name in names:
        assert "SaySo" not in name, name


def test_generated_media_player_names_never_contain_sayso_brand() -> None:
    names = []
    for index in range(600):
        home = generate_home(index, 24, random.Random(index))
        names += [e["name"] for e in home["entities"] if e["capability"] == "media_players"]
    assert names, "no media players generated"
    for name in names:
        assert "SaySo" not in name, name


def test_contrast_injected_satellites_never_contain_sayso_brand() -> None:
    for capability in CONTRAST_CAPABILITIES:
        for scenario in _contrast_rows(capability):
            for entity in scenario["home"]["entities"]:
                if is_satellite(entity):
                    assert "SaySo" not in entity["name"], entity["name"]


def test_generic_contrast_rows_exist_and_gold_is_the_one_matching_entity() -> None:
    for capability in CONTRAST_CAPABILITIES:
        rows = list(_contrast_rows(capability))
        assert rows, capability
        phrases = {phrase for scenario in rows for phrase in scenario["spoken_targets"].values()}
        assert any(" in the " in phrase for phrase in phrases), phrases
        assert any(" in the " not in phrase for phrase in phrases), phrases
        domain = CAPABILITIES[capability].domain
        for scenario in rows:
            target = scenario["target_entity"]
            assert target["domain"] == domain, target
            calls = scenario["expected"]["calls"]
            assert len(calls) == 1, calls
            assert calls[0]["arguments"]["name"] == target["name"], calls
            context = [
                info for info in exposed_entities(scenario["home"])
                if target["name"] in info["names"].split(", ") and info["domain"] == domain
            ]
            assert context, target
            same_domain = [
                e for e in scenario["home"]["entities"]
                if e["domain"] == domain and e["area"] == target["area"]
            ]
            assert same_domain == [target], same_domain


def test_generic_contrast_rows_act_without_clarify_and_validate() -> None:
    for capability in CONTRAST_CAPABILITIES:
        for scenario in _contrast_rows(capability):
            spec = scenario_to_spec(scenario)
            assert spec["expected"]["kind"] == "action", spec["expected"]
            assert spec["expected"].get("response") != "clarify", spec["expected"]
            assert "clarify_question" not in spec, spec
            utterance = request_seed_from_spec(spec)
            phrase = next(iter(scenario["spoken_targets"].values()))
            assert phrase.casefold() in utterance.casefold(), utterance
            assert validate_spec({**spec, "utterance": utterance}) is None, utterance


def test_same_domain_clarify_rows_still_clarify() -> None:
    clarified = 0
    for index in range(200):
        scenario = build_scenario(
            index=index, seed=20261103, capability="lights", operation="turn_off",
            home_size=24, family="clarify",
        )
        expected = scenario["expected"]
        if expected.get("response") == "clarify":
            clarified += 1
            domains = {
                e["domain"] for e in scenario["home"]["entities"] if e["name"] in expected["candidates"]
            }
            assert len(domains) == 1, domains
    assert clarified > 0


def _generated_satellite_form(name: str, area: str) -> str | None:
    for suffix in ("Satellite", "Voice Assistant", "Voice PE"):
        if name == f"{area} {suffix}":
            return suffix
    if re.fullmatch(r"Home Assistant Voice [0-9a-f]{4}", name):
        return "Home Assistant Voice"
    return None


def test_every_generated_satellite_is_recognized() -> None:
    forms = set()
    for index in range(600):
        home = generate_home(index, 24, random.Random(index))
        for entity in home["entities"]:
            if entity["capability"] != "media_players":
                continue
            form = _generated_satellite_form(entity["name"], entity["area"])
            if form is None:
                continue
            forms.add(form)
            assert is_satellite(entity), entity["name"]
    assert forms == {"Satellite", "Voice Assistant", "Voice PE", "Home Assistant Voice"}, forms


def test_satellite_name_rejects_ordinary_speakers_and_players() -> None:
    for name in ("Kitchen Speaker", "Living Room Echo Dot", "Sonos One", "TV", "Satellite", "Home Assistant Voice 0a1"):
        assert not is_satellite_name(name), name


def test_satellite_name_accepts_generated_forms_case_insensitively() -> None:
    for name in ("Kitchen Satellite", "kitchen voice assistant", "Hallway Voice PE", "Home Assistant Voice 0a1f", "HOME ASSISTANT VOICE 0A1F"):
        assert is_satellite_name(name), name


def test_sayso_satellite_name_is_recognized_as_satellite() -> None:
    assert is_satellite_name("Sayso Satellite")


def test_sayso_satellite_appears_among_generated_satellites_at_about_a_quarter() -> None:
    satellites = []
    for index in range(600):
        home = generate_home(index, 24, random.Random(index))
        satellites += [e["name"] for e in home["entities"] if is_satellite(e)]
    sayso = satellites.count("Sayso Satellite")
    assert sayso > 0, satellites
    assert 0.15 < sayso / len(satellites) < 0.35, (sayso, len(satellites))


def test_generated_names_and_aliases_never_use_saysocase_brand() -> None:
    for index in range(600):
        home = generate_home(index, 24, random.Random(index))
        for entity in home["entities"]:
            for text in (entity["name"], *entity.get("aliases", [])):
                assert "SaySo" not in text, text
