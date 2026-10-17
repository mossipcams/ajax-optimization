from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from generators.config import GeneratorConfig  # noqa: E402
from generators.gold import expected_no_action  # noqa: E402
from generators.homes import make_entity  # noqa: E402
from generators.pairs import CLARIFY_PAIR_SUFFIX, clarify_sibling  # noqa: E402
from generators.rendering import render_example  # noqa: E402


def _clarify_spec(home_id: str = "test_home", candidates: tuple[str, ...] = ("Kitchen Light", "Kitchen Lamp")) -> dict:
    rng = random.Random(1234)
    entities = [
        make_entity(name=name, capability="lights", area="Kitchen", floor="Ground", rng=rng)
        for name in ("Kitchen Light", "Kitchen Lamp")
    ]
    return {
        "candidate_id": "lights_clarify_ab12cd34_0003",
        "semantic_id": "lights_clarify_ab12cd34",
        "family": "clarify",
        "capability": "lights",
        "operation": "set_brightness",
        "utterance": "Set the kitchen light to 40 percent",
        "home": {"home_id": home_id, "size": len(entities), "entities": entities},
        "expected": expected_no_action("clarify", candidates=list(candidates)),
    }


def test_clarify_sibling_acts_on_first_candidate():
    spec = _clarify_spec()
    sibling = clarify_sibling(spec, random.Random(5))
    assert sibling is not None

    expected = sibling["expected"]
    assert expected["kind"] == "action"
    assert len(expected["calls"]) == 1
    assert "candidates" not in expected
    assert sibling["target_names"] == ["Kitchen Light"]
    assert expected["calls"][0]["arguments"]["name"] == "Kitchen Light"
    assert "Kitchen Light".casefold() in sibling["utterance"].casefold()
    assert "Kitchen Lamp".casefold() not in sibling["utterance"].casefold()
    assert sibling["family"] == "ordinary"
    assert sibling["candidate_id"] == f"{spec['candidate_id']}{CLARIFY_PAIR_SUFFIX}"
    assert sibling["semantic_id"] == f"{spec['semantic_id']}{CLARIFY_PAIR_SUFFIX}"
    assert spec["expected"]["kind"] == "no_action"


def test_clarify_sibling_renders_with_identical_system_message():
    spec = _clarify_spec()
    sibling = clarify_sibling(spec, random.Random(5))
    assert sibling is not None

    original_row = render_example(spec)
    sibling_row = render_example(sibling)
    assert original_row["messages"][0]["role"] == "system"
    assert sibling_row["messages"][0]["content"].encode() == original_row["messages"][0]["content"].encode()
    assert sibling_row["metadata"]["no_action_reason"] is None
    assert sibling_row["metadata"]["family"] == "ordinary"


def test_clarify_sibling_returns_none_for_non_clarify():
    assert clarify_sibling(_clarify_spec(candidates=("Kitchen Light",)), random.Random(5)) is None

    action_spec = _clarify_spec()
    action_spec["expected"] = {
        "kind": "action",
        "calls": [{"name": "HassLightSet", "arguments": {"name": "Kitchen Light"}}],
    }
    assert clarify_sibling(action_spec, random.Random(5)) is None

    unavailable_spec = _clarify_spec()
    unavailable_spec["expected"] = expected_no_action("unsupported")
    assert clarify_sibling(unavailable_spec, random.Random(5)) is None


def test_clarify_sibling_returns_none_for_real_home():
    spec = _clarify_spec(home_id="real_home_test")
    assert clarify_sibling(spec, random.Random(5)) is None


def test_pipeline_clarify_pairs_share_group_without_shortfall():
    from generators import pipeline

    config = GeneratorConfig(
        count=80,
        seed=7,
        allocations={"clarify": 0.5, "ordinary": 0.5},
        min_positive_per_operation=0,
        min_positive_per_tool=0,
        get_datetime_positive_min=0,
    )
    config.pair_rates = {"clarify": 1.0}

    result = pipeline.run_generation(config)
    rows = result["rows"]
    assert len(rows) == config.count

    siblings = [row for row in rows if row["metadata"].get("pair_kind") == "clarify"]
    assert siblings
    by_group = {
        row["metadata"]["contrastive_group"]: row
        for row in rows
        if row["metadata"].get("family") == "clarify" and row["metadata"].get("contrastive_group")
    }
    for sibling in siblings:
        assert sibling["metadata"]["family"] == "ordinary"
        assert sibling["metadata"]["real_home"] is False
        original = by_group.get(sibling["metadata"]["contrastive_group"])
        assert original is not None
        assert sibling["metadata"]["contrastive_group"] == f"clarify_{original['metadata']['candidate_id']}"
        assert sibling["messages"][0]["content"] == original["messages"][0]["content"]
