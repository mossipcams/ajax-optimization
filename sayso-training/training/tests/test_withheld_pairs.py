from __future__ import annotations

import dataclasses
import random
from functools import lru_cache
from pathlib import Path

import pytest

from generators import pipeline
from generators.config import GeneratorConfig
from generators.coverage import bare_tool_name
from generators.gold import expected_action, expected_no_action, target_names_from_expected
from generators.homes import make_entity
from generators.pairs import WITHHELD_PAIR_TOOLS, withheld_sibling
from generators.rendering import render_example

REPO_ROOT = Path(__file__).resolve().parents[2]
V10_RECIPE = REPO_ROOT / "training" / "configs" / "generation" / "full_sft_v10.yaml"
SAMPLE = 400
REFUSAL = "I can't do that with the available Home Assistant tools."


def _action_spec(capability: str = "lights", operation: str = "set_brightness") -> dict:
    rng = random.Random(1234)
    target = make_entity(
        name="Kitchen Light",
        capability=capability,
        area="Kitchen",
        floor="Ground",
        rng=rng,
    )
    expected = expected_action(target, operation, rng)
    return {
        "candidate_id": "lights_set_brightness_ab12cd34_0007",
        "semantic_id": "lights_set_brightness_ab12cd34",
        "family": "ordinary",
        "utterance": "Set the kitchen light to 40 percent",
        "home": {"home_id": "test_home", "size": 1, "entities": [target]},
        "expected": expected,
    }


def _offered_tool_names(row: dict) -> set[str]:
    return {bare_tool_name(tool["function"]["name"]) for tool in row.get("tools") or []}


def _user_text(row: dict) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def _final_reply(row: dict) -> str:
    return next(
        message["content"]
        for message in reversed(row["messages"])
        if message["role"] == "assistant" and not message.get("tool_calls")
    )


def _v10_config(count: int) -> GeneratorConfig:
    base = GeneratorConfig.from_yaml(V10_RECIPE, repo_root=REPO_ROOT)
    return dataclasses.replace(
        base,
        count=count,
        min_positive_per_tool=0,
        min_positive_per_operation=0,
        get_datetime_positive_min=0,
        output_path=Path("/tmp/withheld_pairs_test.jsonl"),
        manifest_path=Path("/tmp/withheld_pairs_test.manifest.json"),
    )


@lru_cache(maxsize=1)
def _generated() -> dict:
    return pipeline.run_generation(_v10_config(SAMPLE))


def _siblings() -> list[dict]:
    return [row for row in _generated()["rows"] if row["metadata"].get("pair_kind") == "withheld_tool"]


def _parent_of(sibling: dict) -> dict:
    group = sibling["metadata"]["contrastive_group"]
    return next(
        row for row in _generated()["rows"]
        if row["metadata"].get("contrastive_group") == group and row["metadata"].get("pair_kind") is None
        and row["metadata"]["family"] != "unavailable"
    )


def test_withheld_sibling_shares_home_and_target_with_different_utterance():
    spec = _action_spec()
    sibling = withheld_sibling(spec, random.Random(5))
    assert sibling is not None
    assert sibling["home"] == spec["home"]
    assert target_names_from_expected(sibling["expected"]["requested"]) == target_names_from_expected(spec["expected"])
    assert sibling["utterance"] != spec["utterance"]
    assert sibling["family"] == "unavailable"
    assert sibling["candidate_id"].endswith("_withheld")
    assert sibling["semantic_id"].endswith("_withheld")


def test_withheld_sibling_withholds_exactly_the_parent_tool():
    spec = _action_spec()
    tool = bare_tool_name(spec["expected"]["calls"][0]["name"])
    assert tool in WITHHELD_PAIR_TOOLS
    sibling = withheld_sibling(spec, random.Random(5))
    assert sibling["expected"]["kind"] == "no_action"
    assert sibling["expected"]["response"] == "unsupported"
    assert sibling["expected"]["unavailable_tools"] == [tool]
    assert sibling["expected"]["requested"] == spec["expected"]

    row = render_example(sibling)
    assert tool not in _offered_tool_names(row)
    assert len(_offered_tool_names(row)) > 0
    assert _final_reply(row) == REFUSAL


def test_withheld_sibling_rejects_multi_call():
    spec = _action_spec()
    spec["expected"]["calls"].append({"name": "HassGetState", "arguments": {"name": "Kitchen Light"}})
    assert withheld_sibling(spec, random.Random(5)) is None


def test_withheld_sibling_rejects_unlisted_tool():
    spec = _action_spec(capability="switches", operation="turn_on")
    assert bare_tool_name(spec["expected"]["calls"][0]["name"]) not in WITHHELD_PAIR_TOOLS
    assert withheld_sibling(spec, random.Random(5)) is None


def test_withheld_sibling_rejects_non_action():
    spec = _action_spec()
    spec["expected"] = expected_no_action("unsupported")
    assert withheld_sibling(spec, random.Random(5)) is None


def test_config_pairs_section_parsed(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "count: 40\n"
        "seed: 1\n"
        "output: out.jsonl\n"
        "pairs:\n"
        "  withheld_tool: 0.5\n",
        encoding="utf-8",
    )
    config = GeneratorConfig.from_yaml(recipe, repo_root=tmp_path)
    assert config.pair_rates == {"withheld_tool": 0.5}
    assert config.to_dict()["pair_rates"] == {"withheld_tool": 0.5}


def test_config_pairs_default_off_keeps_recipe_hash_fields():
    config = GeneratorConfig(count=10, seed=1)
    assert config.pair_rates == {}
    assert "pair_rates" not in config.to_dict()


def test_config_pairs_rate_out_of_range():
    with pytest.raises(ValueError):
        GeneratorConfig(count=10, seed=1, pair_rates={"withheld_tool": 1.2})


def test_config_pairs_unknown_kind_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(count=10, seed=1, pair_rates={"clarify": 1.0})


def test_v10_recipe_enables_withheld_pairs_and_keeps_v9_shape():
    config = GeneratorConfig.from_yaml(V10_RECIPE, repo_root=REPO_ROOT)
    assert config.count == 40000
    assert config.seed == 20261024
    assert config.pair_rates == {"withheld_tool": 1.0}
    assert config.allocations["unavailable"] == pytest.approx(0.04)


def test_pairs_exist_and_each_has_a_same_target_positive():
    siblings = _siblings()
    assert siblings
    for sibling in siblings:
        parent = _parent_of(sibling)
        assert sibling["metadata"]["home_id"] == parent["metadata"]["home_id"]
        assert sibling["metadata"]["expected_target_names"] == parent["metadata"]["expected_target_names"]
        assert _user_text(sibling) != _user_text(parent)


def test_sibling_withholds_exactly_the_parent_tool():
    for sibling in _siblings():
        parent = _parent_of(sibling)
        parent_calls = [
            call for message in parent["messages"] if message["role"] == "assistant"
            for call in message.get("tool_calls") or []
        ]
        assert len(parent_calls) == 1
        tool = bare_tool_name(parent_calls[0]["function"]["name"])
        assert tool in WITHHELD_PAIR_TOOLS
        assert sibling["metadata"]["unavailable_tools"] == [tool]
        assert tool not in _offered_tool_names(sibling)
        assert tool in _offered_tool_names(parent)
        assert _final_reply(sibling) == REFUSAL


def test_v10_400_row_generation_passes_hygiene_gate():
    stats = _generated()["stats"]["hygiene"]
    assert stats["duplicate_rate"] <= 0.01
    assert stats["conflict_rate"] <= 0.001
    assert stats["invalid_rows"] == 0
    assert stats["contaminated"] == 0
