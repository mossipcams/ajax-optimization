
from __future__ import annotations

import json

import pytest

from pathlib import Path

from generators.config import GeneratorConfig
from generators.real_home import (
    HOLDOUT_STRIDE,
    derive_entity_cap,
    holdout_entity_names,
    load_real_home,
    split_entities,
)
from generators.scenarios import build_scenario

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIXTURE = ROOT / "fixtures" / "real_home.json"
ASSIST_STATIC_CONTEXT = ROOT / "fixtures" / "real_home_assist_static_context.yaml"
SMOKE = ROOT / "configs" / "generation" / "smoke.yaml"


def _cfg(**overrides) -> GeneratorConfig:
    config = GeneratorConfig.from_yaml(SMOKE, repo_root=REPO)
    config.area_distribution_path = None
    for key, value in overrides.items():
        if key == "real_home_path" and value is not None:
            path = Path(value)
            config.real_home_path = path if path.is_absolute() else ROOT / path
        else:
            setattr(config, key, value)
    return config


def _entity(entity_id: str, capability: str, name: str) -> dict:
    return {
        "entity_id": entity_id,
        "name": name,
        "aliases": [name],
        "domain": entity_id.split(".")[0],
        "kind": entity_id.split(".")[0],
        "capability": capability,
        "device_class": None,
        "area": "Kitchen",
        "floor": "Main Floor",
        "state": "on",
        "capabilities": ["on", "off"],
    }


ENTITIES = [_entity(f"light.l{i}", "lights", f"Light {i}") for i in range(10)] + [
    _entity("climate.only", "climate", "Only Thermostat")
]


def test_splits_are_disjoint_and_cover_everything():
    train = split_entities(ENTITIES, "train")
    holdout = split_entities(ENTITIES, "holdout")
    train_ids = {e["entity_id"] for e in train}
    holdout_ids = {e["entity_id"] for e in holdout}
    assert not train_ids & holdout_ids
    assert train_ids | holdout_ids == {e["entity_id"] for e in ENTITIES}
    assert len(holdout) == len(ENTITIES) // HOLDOUT_STRIDE


def test_a_lone_entity_of_its_capability_stays_in_training():
    assert "climate.only" in {e["entity_id"] for e in split_entities(ENTITIES, "train")}
    assert "climate.only" not in {e["entity_id"] for e in split_entities(ENTITIES, "holdout")}


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split must be one of"):
        split_entities(ENTITIES, "validation")


def test_loading_returns_an_independent_copy():
    first = load_real_home(FIXTURE)
    first["entities"].append(_entity("light.injected", "lights", "Injected"))
    second = load_real_home(FIXTURE)
    assert len(second["entities"]) == len(first["entities"]) - 1


def test_the_real_fixture_splits_without_losing_capabilities():
    train = load_real_home(FIXTURE, split="train")
    holdout = load_real_home(FIXTURE, split="holdout")
    assert holdout["entities"]
    assert train["home_id"].endswith("_train")
    assert train["size"] == len(train["entities"])
    assert train["sayso_entity_area"] in {e["area"] for e in train["entities"]}
    fixture = json.load(open(FIXTURE, encoding="utf-8"))
    assert {e["capability"] for e in fixture["entities"]} == {
        e["capability"] for e in train["entities"]
    }


def test_holdout_names_never_appear_in_the_training_split():
    train_names = {e["name"] for e in load_real_home(FIXTURE, split="train")["entities"]}
    assert not train_names & holdout_entity_names(FIXTURE)


def test_build_scenario_uses_the_home_it_is_given():
    home = load_real_home(FIXTURE, split="train")
    scenario = build_scenario(
        index=0,
        seed=1,
        capability="lights",
        operation="turn_on",
        home_size=16,
        home=home,
    )
    assert scenario["home"]["home_id"] == home["home_id"]
    assert scenario["target_entity"]["entity_id"] in {e["entity_id"] for e in home["entities"]}


def test_build_scenario_injects_capabilities_the_real_home_lacks():
    home = load_real_home(FIXTURE, split="train")
    assert not any(e["capability"] == "locks" for e in home["entities"])
    scenario = build_scenario(
        index=3,
        seed=1,
        capability="locks",
        operation="lock",
        home_size=16,
        home=home,
    )
    assert scenario["target_entity"]["capability"] == "locks"

def _full_fixture_home() -> dict:
    raw = json.load(open(FIXTURE, encoding="utf-8"))
    entities = split_entities(raw["entities"], "train") + split_entities(raw["entities"], "holdout")
    for entity in entities:
        entity["name"] = " ".join(entity["name"].split())
        if entity.get("aliases"):
            entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
    context_entities = []
    for context_entity in raw.get("context_entities") or []:
        context_entity = dict(context_entity)
        context_entity["name"] = " ".join(context_entity["name"].split())
        context_entities.append(context_entity)
    home = dict(raw)
    home["entities"] = entities
    home["context_entities"] = context_entities
    return home


def test_full_fixture_renders_the_captured_live_static_context():
    from generators.context import _dump, exposed_entities

    rendered = _dump(exposed_entities(_full_fixture_home()))
    assert rendered == ASSIST_STATIC_CONTEXT.read_text(encoding="utf-8")


def test_generated_rows_never_mention_unassigned():
    from generators.pipeline import run_generation

    config = _cfg(
        count=80,
        seed=97,
        real_home_path=FIXTURE,
        real_home_rate=1.0,
        synthetic_only=False,
        allocations={"ordinary": 0.35, "status": 0.2, "settings": 0.15, "multi_action": 0.15, "correction": 0.15},
    )
    rows = run_generation(config)["rows"]
    assert any(row["metadata"].get("real_home") for row in rows), "no rows used the real home"
    for row in rows:
        blob = json.dumps(row["messages"]).casefold()
        assert "unassigned" not in blob, blob


def test_validate_serialized_row_rejects_placeholder_area():
    from generators import validation

    def make_row(area):
        return {
            "tools": [{"type": "function", "function": {"name": "HassSetVolume", "description": "", "parameters": {}}}],
            "messages": [
                {"role": "user", "content": f"Turn volume up in {area}"},
                {"role": "assistant", "content": ""},
            ],
        }

    rejected = validation.validate_serialized_row(
        make_row("Unassigned"), token_budget=10_000, tokenizer_model="LiquidAI/LFM2.5-230M-Base"
    )
    assert rejected == "placeholder_area"
    kitchen = validation.validate_serialized_row(
        make_row("Kitchen"), token_budget=10_000, tokenizer_model="LiquidAI/LFM2.5-230M-Base"
    )
    assert kitchen != "placeholder_area"


def test_build_scenario_without_a_home_is_unchanged():
    kwargs = dict(index=5, seed=2, capability="lights", operation="turn_on", home_size=16)
    assert build_scenario(**kwargs) == build_scenario(**kwargs, home=None)


def test_real_home_settings_reach_the_manifest():
    config = _cfg(real_home_path=FIXTURE, real_home_rate=0.05, synthetic_only=False)
    recorded = config.to_dict()
    assert recorded["real_home_path"] == str(FIXTURE)
    assert recorded["real_home_rate"] == 0.05


def test_default_config_does_not_mix_in_a_real_home():
    config = GeneratorConfig()
    assert config.real_home_path is None
    assert config.real_home_rate == 0.0


def test_mixing_is_off_by_default_in_generated_rows():
    from generators.pipeline import run_generation

    rows = run_generation(_cfg(count=80, seed=3, synthetic_only=True))["rows"]
    assert not any(row["metadata"].get("real_home") for row in rows)


def test_derived_cap_scales_with_count_and_rate():
    assert derive_entity_cap(40_000, 0.10, 56) == 288
    assert derive_entity_cap(40_000, 0.05, 56) == 144
    assert derive_entity_cap(40_000, 0.0, 56) == 0
    assert derive_entity_cap(40_000, 0.10, 0) == 0


def test_an_explicit_cap_is_never_exceeded():
    from generators.pipeline import run_generation

    config = _cfg(
        count=120,
        seed=41,
        real_home_path=FIXTURE,
        real_home_rate=0.2,
        real_home_entity_cap=3,
        synthetic_only=False,
    )
    report = run_generation(config)["stats"]
    counts = dict(report["real_home"]["most_common"])
    assert counts, "expected the real home to contribute rows"
    assert max(counts.values()) <= 3
    assert report["accepted"] == 120


def test_capped_rows_are_replaced_not_dropped():
    from generators.pipeline import run_generation

    report = run_generation(
        _cfg(
            count=80,
            seed=41,
            real_home_path=FIXTURE,
            real_home_rate=0.15,
            real_home_entity_cap=3,
            synthetic_only=False,
        )
    )["stats"]
    assert report["accepted"] == 80
    assert max(dict(report["real_home"]["most_common"]).values()) <= 3


def test_no_real_home_means_no_cap_bookkeeping():
    from generators.pipeline import run_generation

    report = run_generation(_cfg(count=80, seed=3))["stats"]
    assert "real_home" not in report


def test_context_entities_survive_both_splits_identical():
    train = load_real_home(FIXTURE, split="train")
    holdout = load_real_home(FIXTURE, split="holdout")
    assert len(train["context_entities"]) == 27
    assert train["context_entities"] == holdout["context_entities"]


def test_context_entity_names_never_collide_with_targets():
    home = load_real_home(FIXTURE, split="train")
    context_names = {c["name"] for c in home["context_entities"]}
    target_names = set()
    for entity in home["entities"]:
        target_names.add(entity["name"])
        target_names.update(entity.get("aliases", []))
    assert not {n.casefold() for n in context_names} & {t.casefold() for t in target_names}


def test_context_entity_names_are_whitespace_normalised(tmp_path):
    fixture = json.load(open(FIXTURE, encoding="utf-8"))
    fixture["context_entities"] = [{"name": "  TV   Active   app ", "domain": "sensor"}]
    path = tmp_path / "home.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    home = load_real_home(path)
    assert [c["name"] for c in home["context_entities"]] == ["TV Active app"]


def test_exposed_entities_list_context_entries_alongside_targets():
    from generators.context import exposed_entities

    home = load_real_home(FIXTURE, split="train")
    names = [entry["names"] for entry in exposed_entities(home)]
    assert "TV Active app" in names
    assert names[names.index("TV")] == "TV"
    assert names[names.index("TV") + 1] == "TV Active app"
