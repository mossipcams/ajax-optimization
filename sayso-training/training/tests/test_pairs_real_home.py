from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from generators.config import GeneratorConfig
from generators.gold import expected_action
from generators.pairs import real_home_contrast_siblings
from generators.pipeline import _pair_sibling_rows
from generators.rendering import render_example


def load_fixture_home() -> dict:
    home = json.loads((HERE.parent / "fixtures" / "real_home.json").read_text(encoding="utf-8"))
    for entity in [*home["entities"], *home.get("context_entities", [])]:
        entity["name"] = " ".join(entity["name"].split())
        if entity.get("aliases"):
            entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
    return home


def make_tv_spec(home: dict) -> dict:
    tv = next(entity for entity in home["entities"] if entity["name"] == "TV")
    return {
        "candidate_id": "cand-realhome-0001",
        "semantic_id": "sem-realhome-tv-on",
        "family": "ordinary",
        "capability": "media_players",
        "operation": "turn_on",
        "utterance": "turn on the TV",
        "home": home,
        "expected": expected_action(tv, "turn_on", random.Random(7)),
        "target_names": ["TV"],
        "real_home": True,
        "full_tool_catalog": True,
    }


def test_tv_siblings_are_lookalike_context_and_absent():
    home = load_fixture_home()
    spec = make_tv_spec(home)
    siblings = real_home_contrast_siblings(spec, random.Random(11))
    by_suffix = {sibling["candidate_id"].removeprefix("cand-realhome-0001_"): sibling for sibling in siblings}
    assert set(by_suffix) == {"contrast_a", "contrast_b", "contrast_c"}

    lookalike = by_suffix["contrast_a"]
    context_names = [item["name"] for item in home.get("context_entities", [])]
    expected_lookalikes = {name for name in context_names if any(word in name.split() for word in ("TV", "tv"))}
    assert lookalike["utterance"].startswith("turn on the ")
    lookalike_name = lookalike["utterance"][len("turn on the "):]
    assert lookalike_name in expected_lookalikes and lookalike_name != "TV"
    assert lookalike["expected"] == {
        "kind": "no_action",
        "response": "device_unsupported",
        "calls": [],
        "unsupported_names": [lookalike_name],
    }
    assert lookalike["family"] == "unsupported"

    non_device = by_suffix["contrast_b"]
    non_device_name = non_device["utterance"][len("turn on the "):]
    context_domain = {item["name"]: item.get("domain") for item in home.get("context_entities", [])}
    assert context_domain[non_device_name] in {"sun", "person"}
    assert non_device["expected"] == {
        "kind": "no_action",
        "response": "device_unsupported",
        "calls": [],
        "unsupported_names": [non_device_name],
    }

    absent = by_suffix["contrast_c"]
    replacement = absent["utterance"][len("turn on the "):]
    assert replacement.endswith(" TV")
    replacement_area = replacement[: -len(" TV")]
    area_entities = [entity for entity in home["entities"] if entity.get("area") == replacement_area]
    assert all(entity.get("domain") != "media_player" for entity in area_entities)
    assert absent["expected"] == {
        "kind": "no_action",
        "response": "area_unavailable",
        "calls": [],
        "unavailable": {"area": replacement_area.casefold(), "type": "media players"},
    }
    assert absent["family"] == "absence"

    for sibling in siblings:
        assert sibling["target_names"] == []
        assert sibling["linguistics"] == [{"source": "sayso_fallback"}]


def home_prompt(row: dict) -> str:
    return row["messages"][0]["content"].split("Area context:")[0]


def test_rendered_system_prompt_and_tools_identical():
    home = load_fixture_home()
    spec = make_tv_spec(home)
    original_row = render_example(spec)
    siblings = real_home_contrast_siblings(spec, random.Random(13))
    assert siblings
    for sibling in siblings:
        sibling_row = render_example(sibling)
        assert sibling_row["tools"] == original_row["tools"]
        if sibling["family"] == "absence":
            assert home_prompt(sibling_row) == home_prompt(original_row)
            assert "target_area: Bedroom\ntarget_area_source: explicit" in sibling_row["messages"][0]["content"]
        else:
            assert sibling_row["messages"][:1] == original_row["messages"][:1]


def test_empty_for_non_real_home_multi_call_or_missing_name():
    home = load_fixture_home()
    spec = make_tv_spec(home)

    not_real_home = {**spec, "real_home": False}
    assert real_home_contrast_siblings(not_real_home, random.Random(1)) == []

    multi_call = dict(spec)
    multi_call["expected"] = {**spec["expected"], "calls": list(spec["expected"]["calls"]) * 2}
    assert real_home_contrast_siblings(multi_call, random.Random(1)) == []

    missing_name = dict(spec)
    missing_name["utterance"] = "please start something else"
    assert real_home_contrast_siblings(missing_name, random.Random(1)) == []


def test_pipeline_pair_sibling_rows_groups_contrasts():
    home = load_fixture_home()
    spec = make_tv_spec(home)

    def original_row(candidate_id: str) -> dict:
        base = {**spec, "candidate_id": candidate_id}
        row = render_example(base)
        row["metadata"]["real_home"] = True
        row["_final_spec"] = base
        return row

    config = GeneratorConfig(
        count=20,
        real_home_rate=0.5,
        real_home_path=HERE.parent / "fixtures" / "real_home.json",
        pair_rates={"real_home_contrast": 1.0},
    )
    originals = [original_row("cand-a"), original_row("cand-b")]
    accepted: list[dict] = []

    def accept(sibling_row: dict) -> bool:
        accepted.append(sibling_row)
        return True

    for original in originals:
        _pair_sibling_rows(original, config, random.Random(5), None, 64, accept=accept)
    assert len(accepted) >= 2

    groups: dict[str, list[dict]] = {}
    for row in originals + accepted:
        group = row["metadata"].get("contrastive_group")
        if group:
            groups.setdefault(group, []).append(row)
    assert any(len(members) >= 2 for members in groups.values())
