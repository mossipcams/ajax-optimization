from __future__ import annotations

import json
from pathlib import Path

import yaml

from generators.config import GeneratorConfig




def _make_real_home(tmp_path: Path) -> Path:
    entities = [
        {"entity_id": f"ent_{i}", "name": f"Entity {i}", "capability": "lights", "area": "Kitchen"}
        for i in range(10)
    ]
    home = {
        "home_id": "test_home",
        "entities": entities,
        "sayso_entity_area": "Kitchen",
        "areas": ["Kitchen"],
    }
    path = tmp_path / "real_home.json"
    path.write_text(json.dumps(home), encoding="utf-8")
    return path


def _base_recipe() -> dict:
    return {
        "version": 7,
        "count": 10,
        "seed": 42,
        "output": "training/datasets/test.jsonl",
        "allocations": {"lights": 1.0},
        "generation": {"synthetic_only": False},
    }


def test_config_default_holdouts_true(tmp_path: Path):
    _make_real_home(tmp_path)
    data = _base_recipe()
    data["real_home"] = {"path": str(tmp_path / "real_home.json"), "rate": 0.5}
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")

    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)
    assert config.real_home_holdouts is True


def test_config_explicit_holdouts_false(tmp_path: Path):
    _make_real_home(tmp_path)
    data = _base_recipe()
    data["real_home"] = {
        "path": str(tmp_path / "real_home.json"),
        "rate": 0.5,
        "holdouts": False,
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")

    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)
    assert config.real_home_holdouts is False


def test_config_to_dict_includes_holdouts(tmp_path: Path):
    _make_real_home(tmp_path)
    data = _base_recipe()
    data["real_home"] = {
        "path": str(tmp_path / "real_home.json"),
        "rate": 0.5,
        "holdouts": False,
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")

    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)
    d = config.to_dict()
    assert "real_home_holdouts" in d
    assert d["real_home_holdouts"] is False


def test_holdout_names_empty_when_disabled(tmp_path: Path):
    home_path = _make_real_home(tmp_path)

    data = _base_recipe()
    data["real_home"] = {
        "path": str(home_path),
        "rate": 0.5,
        "holdouts": False,
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")

    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)

    from generators.real_home import holdout_entity_names

    if config.real_home_holdouts:
        assert config.real_home_path is not None
        holdouts = holdout_entity_names(config.real_home_path)
    else:
        holdouts = frozenset()

    assert holdouts == frozenset()


def test_pipeline_uses_all_entities_when_holdouts_disabled(tmp_path: Path):
    home_path = _make_real_home(tmp_path)
    all_names = {f"Entity {i}" for i in range(10)}

    data = _base_recipe()
    data["real_home"] = {
        "path": str(home_path),
        "rate": 0.5,
        "holdouts": False,
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")

    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)
    assert config.real_home_path is not None

    from generators.pipeline import real_home_entity_names

    names = real_home_entity_names(config.real_home_path, config.real_home_holdouts)
    assert names == all_names


def test_pipeline_entity_list_uses_normalized_fixture_names():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "real_home.json"

    from generators.pipeline import real_home_entity_names
    from generators.real_home import holdout_entity_names, load_real_home

    names = real_home_entity_names(fixture, hold_out=False)
    assert "Living room lights" in names
    expected = {
        entity["name"]
        for split in ("train", "holdout")
        for entity in load_real_home(fixture, split=split)["entities"]
    }
    assert names == expected
    assert holdout_entity_names(fixture) <= names
