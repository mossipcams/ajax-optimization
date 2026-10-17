from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from generators.config import GeneratorConfig  # noqa: E402
from generators.coverage import bare_tool_name  # noqa: E402
from generators.gold import expected_action, expected_no_action  # noqa: E402
from generators.homes import make_entity  # noqa: E402
from generators.pairs import WITHHELD_PAIR_TOOLS, withheld_sibling  # noqa: E402
from generators.rendering import render_example  # noqa: E402


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


def _rendered_tool_names(row: dict) -> set[str]:
    return {tool["function"]["name"] for tool in row.get("tools") or []}


def test_withheld_sibling_shape_and_rendering():
    spec = _action_spec()
    sibling = withheld_sibling(spec)
    assert sibling is not None
    tool = bare_tool_name(spec["expected"]["calls"][0]["name"])
    assert tool in WITHHELD_PAIR_TOOLS

    expected = sibling["expected"]
    assert expected["kind"] == "no_action"
    assert expected["response"] == "unsupported"
    assert expected["unavailable_tools"] == [tool]
    assert expected["requested"]["kind"] == "action"
    assert expected["requested"]["calls"][0]["arguments"] == spec["expected"]["calls"][0]["arguments"]
    assert sibling["family"] == "unavailable"
    assert sibling["utterance"] == spec["utterance"]
    assert sibling["candidate_id"].endswith("_withheld")
    assert sibling["semantic_id"].endswith("_withheld")

    row = render_example(sibling)
    offered = _rendered_tool_names(row)
    assert not any(bare_tool_name(name) == tool for name in offered)
    assert len(offered) > 0

    base_row = render_example(spec)
    assert base_row["metadata"]["contrastive_group"] is None


def test_withheld_sibling_rejects_multi_call():
    spec = _action_spec()
    spec["expected"]["calls"].append({"name": "HassGetState", "arguments": {"name": "Kitchen Light"}})
    assert withheld_sibling(spec) is None


def test_withheld_sibling_rejects_unlisted_tool():
    assert WITHHELD_PAIR_TOOLS.isdisjoint({"HassTurnOn"})
    spec = _action_spec(capability="switches", operation="turn_on")
    assert bare_tool_name(spec["expected"]["calls"][0]["name"]) not in WITHHELD_PAIR_TOOLS
    assert withheld_sibling(spec) is None


def test_withheld_sibling_rejects_non_action():
    spec = _action_spec()
    spec["expected"] = expected_no_action("unsupported")
    assert withheld_sibling(spec) is None


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


def test_config_pairs_default_empty():
    config = GeneratorConfig(count=10, seed=1)
    assert config.pair_rates == {}
    assert config.to_dict()["pair_rates"] == {}


def test_config_pairs_rate_out_of_range():
    with pytest.raises(ValueError):
        GeneratorConfig(count=10, seed=1, pair_rates={"withheld_tool": 1.2})


def test_withheld_sibling_row_helper(monkeypatch):
    from generators import pipeline


    monkeypatch.setattr(pipeline, "validate_serialized_row", lambda row, **kwargs: None)
    config = GeneratorConfig(count=10, seed=1)
    config.pair_rates = {"withheld_tool": 1.0}

    spec = _action_spec()
    row = render_example(spec)
    row["_final_spec"] = spec
    sibling_row = pipeline._withheld_sibling_row(row, config, random.Random(3), None, remaining=4)
    assert sibling_row is not None
    assert "_final_spec" not in row
    group_id = f"withheld_{spec['candidate_id']}"
    assert row["metadata"]["contrastive_group"] == group_id
    assert sibling_row["metadata"]["contrastive_group"] == group_id
    assert sibling_row["metadata"]["pair_kind"] == "withheld_tool"
    assert sibling_row["metadata"]["family"] == "unavailable"

    for key in ("real_home", "grounding_family", "discrimination"):
        assert sibling_row["metadata"][key] == row["metadata"].get(key)
    spec2 = _action_spec()
    spec2["candidate_id"] = "lights_set_brightness_ab12cd34_0008"
    row2 = render_example(spec2)
    row2["_final_spec"] = spec2
    row2["metadata"]["real_home"] = True
    row2["metadata"]["grounding_family"] = "near_duplicate"
    row2["metadata"]["discrimination"] = True
    sibling_row2 = pipeline._withheld_sibling_row(row2, config, random.Random(3), None, remaining=4)
    assert sibling_row2 is not None
    assert sibling_row2["metadata"]["real_home"] is True
    assert sibling_row2["metadata"]["grounding_family"] == "near_duplicate"
    assert sibling_row2["metadata"]["discrimination"] is True

    spec_full = _action_spec(capability="fans", operation="set_speed")
    row_full = render_example(spec_full)
    row_full["_final_spec"] = spec_full
    assert pipeline._withheld_sibling_row(row_full, config, random.Random(3), None, remaining=0) is None

    spec_off = _action_spec(capability="switches", operation="turn_on")
    row_off = render_example(spec_off)
    row_off["_final_spec"] = spec_off
    config.pair_rates = {}
    assert pipeline._withheld_sibling_row(row_off, config, random.Random(3), None, remaining=4) is None
    assert row_off["metadata"]["contrastive_group"] is None


def test_pipeline_emits_withheld_siblings():
    from generators import pipeline

    config = GeneratorConfig(count=80, seed=5)
    config.pair_rates = {"withheld_tool": 0.2}

    result = pipeline.run_generation(config)
    rows = result["rows"]
    assert len(rows) == config.count
    siblings = [row for row in rows if row.get("metadata", {}).get("pair_kind") == "withheld_tool"]
    assert siblings
    by_id = {row["metadata"]["candidate_id"]: row for row in rows}
    for sibling in siblings:
        group_id = sibling["metadata"]["contrastive_group"]
        base_id = group_id.removeprefix("withheld_")
        original = by_id.get(base_id)
        assert original is not None
        assert original["metadata"].get("contrastive_group") == group_id
        assert sibling["metadata"]["family"] == "unavailable"


def test_pipeline_no_pairs_section_leaves_corpus_unchanged():
    from generators import pipeline

    config = GeneratorConfig(count=80, seed=5)
    rows = pipeline.run_generation(config)["rows"]
    assert len(rows) == 80
    assert not any(row.get("metadata", {}).get("pair_kind") for row in rows)


def test_helper_without_rate_does_not_consume_rng():
    from generators import pipeline

    def drain(rng):
        return [rng.random() for _ in range(4)]

    config = GeneratorConfig(count=10, seed=1)
    baseline = drain(random.Random(9))
    control = random.Random(9)
    spec = _action_spec()
    row = render_example(spec)
    row["_final_spec"] = spec
    assert pipeline._withheld_sibling_row(row, config, control, None, remaining=4) is None
    assert drain(control) == baseline


def test_pipeline_allocations_tracker_path_accepts_siblings():
    from generators import pipeline

    config = GeneratorConfig(
        count=80,
        seed=7,
        allocations={"ordinary": 0.7, "unavailable": 0.3},
        min_positive_per_operation=0,
        min_positive_per_tool=0,
        get_datetime_positive_min=0,
    )
    config.pair_rates = {"withheld_tool": 1.0}

    result = pipeline.run_generation(config)
    rows = result["rows"]
    assert len(rows) == config.count
    siblings = [row for row in rows if row.get("metadata", {}).get("pair_kind") == "withheld_tool"]
    assert siblings
    by_id = {row["metadata"]["candidate_id"]: row for row in rows}
    for sibling in siblings:
        group_id = sibling["metadata"]["contrastive_group"]
        base_id = group_id.removeprefix("withheld_")
        original = by_id.get(base_id)
        assert original is not None
        assert original["metadata"].get("contrastive_group") == group_id
        assert sibling["metadata"]["family"] == "unavailable"
    tracker_summary = result["stats"]["quota"]
    assert tracker_summary["achieved_family"]["unavailable"] > 0
    assert sum(tracker_summary["achieved_family"].values()) >= config.count


def test_pipeline_allocations_tracker_path_no_pairs_unchanged():
    from generators import pipeline

    config = GeneratorConfig(
        count=80,
        seed=7,
        allocations={"ordinary": 0.7, "unavailable": 0.3},
        min_positive_per_operation=0,
        min_positive_per_tool=0,
        get_datetime_positive_min=0,
    )

    rows = pipeline.run_generation(config)["rows"]
    assert len(rows) == config.count
    assert not any(row.get("metadata", {}).get("pair_kind") for row in rows)
