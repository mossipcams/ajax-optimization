from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from generators.config import GeneratorConfig
from generators.row_generation import (
    exceeds_max_target_share,
    normalize_entity_name,
    scaled_target_usage,
)


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


def _load_recipe(tmp_path: Path, real_home: dict | None) -> GeneratorConfig:
    data = {
        "version": 7,
        "count": 10,
        "seed": 42,
        "output": "training/datasets/test.jsonl",
        "allocations": {"lights": 1.0},
        "generation": {"synthetic_only": False},
    }
    if real_home is not None:
        data["real_home"] = real_home
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(data), encoding="utf-8")
    return GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)


def test_config_defaults_leave_balance_off(tmp_path: Path):
    _make_real_home(tmp_path)
    config = _load_recipe(tmp_path, {"path": str(tmp_path / "real_home.json"), "rate": 0.5})
    assert config.real_home_target_weights == {}
    assert config.real_home_max_target_share == 0.0


def test_config_parses_target_balance_keys(tmp_path: Path):
    _make_real_home(tmp_path)
    config = _load_recipe(
        tmp_path,
        {
            "path": str(tmp_path / "real_home.json"),
            "rate": 0.5,
            "target_weights": {"Living room lights": 4.0, "Porch_Light": 1.5},
            "max_target_share": 0.10,
        },
    )
    assert config.real_home_target_weights == {"Living room lights": 4.0, "Porch_Light": 1.5}
    assert config.real_home_max_target_share == pytest.approx(0.10)


def test_config_manifest_dict_includes_balance_keys(tmp_path: Path):
    _make_real_home(tmp_path)
    config = _load_recipe(
        tmp_path,
        {
            "path": str(tmp_path / "real_home.json"),
            "rate": 0.5,
            "target_weights": {"Living room lights": 4.0},
            "max_target_share": 0.10,
        },
    )
    d = config.to_dict()
    assert d["real_home_target_weights"] == {"Living room lights": 4.0}
    assert d["real_home_max_target_share"] == pytest.approx(0.10)


def test_config_rejects_nonpositive_weight(tmp_path: Path):
    _make_real_home(tmp_path)
    with pytest.raises(ValueError, match="must be > 0"):
        _load_recipe(
            tmp_path,
            {
                "path": str(tmp_path / "real_home.json"),
                "rate": 0.5,
                "target_weights": {"Living room lights": 0.0},
            },
        )


def test_config_rejects_share_out_of_range(tmp_path: Path):
    _make_real_home(tmp_path)
    for share in (-0.1, 1.0, 2.5):
        with pytest.raises(ValueError, match="real_home_max_target_share"):
            _load_recipe(
                tmp_path,
                {
                    "path": str(tmp_path / "real_home.json"),
                    "rate": 0.5,
                    "max_target_share": share,
                },
            )


def test_normalize_entity_name_collapses_whitespace():
    assert normalize_entity_name("  Living   room  lights ") == "Living room lights"


def test_scaled_usage_noop_without_weights():
    usage = Counter({"TV": 322, "Thermostat": 275})
    assert scaled_target_usage(usage, None) == usage
    assert scaled_target_usage(usage, {}) == usage


def test_scaled_usage_divides_named_entities_by_weight():
    usage = Counter({"Living room lights": 60, "TV": 322})
    weights = {"Living room lights": 4.0}
    scaled = scaled_target_usage(usage, weights)
    assert scaled["Living room lights"] == pytest.approx(15.0)
    assert scaled["TV"] == 322


def test_scaled_usage_matches_whitespace_variants():
    usage = Counter({"Living room lights": 60})
    weights = {"  Living   room  lights ": 4.0}
    scaled = scaled_target_usage(usage, weights)
    assert scaled["Living room lights"] == pytest.approx(15.0)


def test_exceeds_share_floor_allows_small_corpora():
    targets = Counter({"TV": 24})
    assert not exceeds_max_target_share(targets, ["TV"], 0.10)


def test_exceeds_share_rejects_over_bound_target():
    targets = Counter({"TV": 101, "Other": 900})
    assert exceeds_max_target_share(targets, ["TV", "Other"], 0.10)


def test_exceeds_share_allows_under_bound_target():
    targets = Counter({"TV": 99, "Other": 900})
    assert not exceeds_max_target_share(targets, ["TV"], 0.10)


def test_exceeds_share_ignores_other_targets():
    targets = Counter({"TV": 5, "Other": 900})
    assert not exceeds_max_target_share(targets, ["TV"], 0.10)
