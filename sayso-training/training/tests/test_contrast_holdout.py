from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import pytest

from generators import homes
from generators.capability_registry import is_satellite

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SUFFIXES = ("lights off", "lights on", "evening lighting")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reserved() -> dict[str, frozenset[str]]:
    return homes.holdout_slice()


def _role_of(entity: dict) -> str:
    return re.sub(r" \d+$", "", entity["name"]).casefold()


def _ends_with_any(name: str, nouns: frozenset[str]) -> bool:
    return any(name == noun.casefold() or name.endswith(f" {noun.casefold()}") for noun in nouns)


def _has_brand(name: str, brands: frozenset[str]) -> bool:
    return any(f" {brand.casefold()} " in f" {name} " for brand in brands)


def _sample(*, holdout: bool, count: int = 60) -> list[dict]:
    rng = random.Random(20261021)
    return [
        homes.generate_home(index, size, rng, holdout=holdout)
        for index, size in enumerate([16, 32, 64] * (count // 3))
    ]


def _training_nouns() -> set[str]:
    areas = {*homes._AREAS, *homes._OUTDOOR_AREAS, *(a for allowed in homes._DEVICE_AREAS.values() for a in allowed)}
    return {noun for capability in homes._ROLES for area in areas for noun in homes._roles(capability, area)}


def test_holdout_pools_are_separate_from_training_pools() -> None:
    reserved = _reserved()
    training_brands = {brand for brands in homes._BRANDS.values() for brand in brands}
    assert set(homes._HOLDOUT_ROLES) == set(homes._ROLES)
    assert reserved["areas"] and not reserved["areas"] & set(homes._AREAS)
    assert reserved["nouns"] and not reserved["nouns"] & _training_nouns()
    assert reserved["owners"] and not reserved["owners"] & set(homes._OWNERS)
    assert reserved["brands"] and not reserved["brands"] & training_brands


def test_default_generation_never_uses_reserved_names_areas_or_nouns() -> None:
    reserved = _reserved()
    for home in _sample(holdout=False, count=150):
        assert not set(home["areas"]) & reserved["areas"]
        assert not set(home["owners"]) & reserved["owners"]
        assert not {timer["area"] for timer in home["active_timers"]} & reserved["areas"]
        for entity in home["entities"]:
            name = _role_of(entity)
            assert entity["area"] not in reserved["areas"], entity["name"]
            assert not _ends_with_any(name, reserved["nouns"]), entity["name"]
            assert not _has_brand(name, reserved["brands"]), entity["name"]
            assert not any(name.startswith(owner.casefold()) for owner in reserved["owners"]), entity["name"]


def test_holdout_homes_draw_only_from_the_reserved_slice() -> None:
    reserved = _reserved()
    for home in _sample(holdout=True):
        assert set(home["areas"]) <= reserved["areas"]
        assert set(home["owners"]) <= reserved["owners"]
        for entity in home["entities"]:
            assert entity["area"] in reserved["areas"], entity["name"]
            if is_satellite(entity):
                continue
            name = _role_of(entity)
            assert (
                _ends_with_any(name, reserved["nouns"])
                or _has_brand(name, reserved["brands"])
                or (entity["capability"] == "scripts" and name.endswith(SCRIPT_SUFFIXES))
            ), entity["name"]


def test_holdout_entity_names_never_appear_in_training_homes() -> None:
    def names(sampled: list[dict]) -> set[str]:
        return {
            entity["name"].casefold()
            for home in sampled for entity in home["entities"] if not is_satellite(entity)
        }

    assert not names(_sample(holdout=True)) & names(_sample(holdout=False, count=150))


def test_holdout_pools_context_restores_default_mode() -> None:
    with homes.holdout_pools():
        inside = homes.generate_home(1, 32, random.Random(1))
    after = homes.generate_home(1, 32, random.Random(1))
    assert set(inside["areas"]) <= _reserved()["areas"]
    assert not set(after["areas"]) & _reserved()["areas"]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, list[dict]]:
    module = _script("build_contrast_holdout")
    out = tmp_path_factory.mktemp("contrast_holdout")
    case_path, _ = module.write(module.build(8, 7), out)
    return out, [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines()]


def test_built_cases_follow_the_eval_case_contract(built) -> None:
    from evals.cases import _parse_case

    out, cases = built
    assert len({case["contrast_group"] for case in cases}) == 8
    assert len({case["id"] for case in cases}) == len(cases)
    reserved = _reserved()
    for line, payload in enumerate(cases, start=1):
        case = _parse_case(payload, source="contrast_holdout.jsonl", line=line)
        assert case.tags == ("contrast", f"contrast_axis_{payload['contrast_axis']}")
        home = json.loads((out / "homes" / f"{case.household}.json").read_text(encoding="utf-8"))
        assert home["home_id"] == case.household
        assert {entity["area"] for entity in home["entities"]} <= reserved["areas"]
    for group in {case["contrast_group"] for case in cases}:
        members = [case for case in cases if case["contrast_group"] == group]
        assert sorted(case["contrast_variant"] for case in members) == list(range(len(members)))
        assert members[0]["contrast_axis"] == "base"
        assert len({case["utterance"] for case in members}) == 1


def test_built_expected_calls_pass_the_production_tool_contract(built) -> None:
    runner = pytest.importorskip("evals.runner")
    from sayso_contract import tool_contract

    out, cases = built
    actions = 0
    for case in cases:
        home = json.loads((out / "homes" / f"{case['household']}.json").read_text(encoding="utf-8"))
        removed = (case["unavailable"] or {}).get("removed_tools") or []
        catalog = runner.production_catalog(home, removed_tools=removed)
        calls = [(call["name"], call["arguments"]) for call in case["expected"]["calls"]]
        domains = {entity["domain"] for entity in home["entities"]}
        assert tool_contract.check_tool_calls(calls, catalog, domains) == [], case["id"]
        actions += bool(calls)
    assert actions


def test_scorer_reports_variant_axis_and_group_consistency(tmp_path) -> None:
    module = _script("score_contrast_holdout")
    cases = [
        {"id": "a0", "contrast_group": "a", "contrast_axis": "base", "contrast_variant": 0},
        {"id": "a1", "contrast_group": "a", "contrast_axis": "count", "contrast_variant": 1},
        {"id": "b0", "contrast_group": "b", "contrast_axis": "base", "contrast_variant": 0},
        {"id": "b1", "contrast_group": "b", "contrast_axis": "presence", "contrast_variant": 1},
        {"id": "c0", "contrast_group": "c", "contrast_axis": "base", "contrast_variant": 0},
        {"id": "c1", "contrast_group": "c", "contrast_axis": "count", "contrast_variant": 1},
        {"id": "d0", "contrast_group": "d", "contrast_axis": "base", "contrast_variant": 0},
    ]
    outcomes = [
        {"case_id": "a0", "passed": True}, {"case_id": "a1", "passed": True},
        {"case_id": "b0", "passed": True}, {"case_id": "b1", "passed": False},
        {"case_id": "c0", "passed": False}, {"case_id": "c1", "passed": True},
    ]
    cases_path, run_dir = tmp_path / "cases.jsonl", tmp_path / "run"
    run_dir.mkdir()
    cases_path.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    (run_dir / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"
    assert module.main([str(run_dir), str(cases_path), "--output", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["group_consistency"] == {
        "passed": 1, "total": 3, "accuracy": 0.3333, "incomplete_groups": 1,
    }
    assert report["per_variant"]["0"] == {"passed": 2, "total": 3, "accuracy": 0.6667}
    assert report["per_variant"]["1"] == {"passed": 2, "total": 3, "accuracy": 0.6667}
    assert report["per_axis"]["count"] == {"passed": 2, "total": 2, "accuracy": 1.0}
    assert report["per_axis"]["presence"] == {"passed": 0, "total": 1, "accuracy": 0.0}
    assert report["cases_missing"] == 1
    assert "GROUP CONSISTENCY" in module.table(report)
