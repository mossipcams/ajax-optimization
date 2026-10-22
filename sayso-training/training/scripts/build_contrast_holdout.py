#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

CASE_FILE = "contrast_holdout.jsonl"
DEFAULT_GROUPS = 300
DEFAULT_SEED = 20261021
PREFIX = "holdout_"


def eval_home(home: dict[str, Any]) -> dict[str, Any]:
    areas = {area: [] for area in home.get("areas") or ()}
    return {
        "home_id": f"{PREFIX}{home['home_id']}",
        "satellite_area": home["sayso_entity_area"],
        "sayso_entity_area": home["sayso_entity_area"],
        "areas": areas,
        "area_aliases": dict(areas),
        "area_floors": home.get("area_floors") or {},
        "entities": home["entities"],
        "context_entities": home.get("context_entities") or [],
        "exposure_source": "synthetic_holdout",
    }


def eval_expected(variant: dict[str, Any]) -> dict[str, Any]:
    from generators.tools import namespaced_tool_name

    gold = variant["expected"]
    behaviour = variant["expected_class"]
    calls = [
        {"name": namespaced_tool_name(call["name"]), "arguments": call["arguments"]}
        for call in gold.get("calls") or []
    ] if behaviour == "action" else []
    spoken = f"the {variant['noun'].casefold()}"
    return {
        "response_type": {"action": "action", "clarify": "clarification"}.get(behaviour, "refusal"),
        "calls": calls,
        "clarification": {"candidates": gold["candidates"]} if behaviour == "clarify" else None,
        "forbidden_entities": [],
        "spoken_aliases": {
            call["arguments"]["name"]: spoken for call in calls if "name" in call["arguments"]
        },
    }


def eval_case(variant: dict[str, Any]) -> dict[str, Any]:
    unavailable = dict(variant["expected"].get("unavailable") or {})
    if variant["withheld_tools"]:
        unavailable["removed_tools"] = list(variant["withheld_tools"])
    axis = variant["contrast_axis"]
    return {
        "id": f"{PREFIX}{variant['contrast_group']}_{variant['contrast_variant']}",
        "category": "contrast",
        "tags": ["contrast", f"contrast_axis_{axis}"],
        "household": f"{PREFIX}{variant['home']['home_id']}",
        "utterance": variant["utterance"],
        "unavailable": unavailable or None,
        "expected": eval_expected(variant),
        "provenance": {
            "source": "training/scripts/build_contrast_holdout.py",
            "held_out": True,
            "family": variant["family"],
            "expected_class": variant["expected_class"],
        },
        "contrast_group": f"{PREFIX}{variant['contrast_group']}",
        "contrast_axis": axis,
        "contrast_variant": variant["contrast_variant"],
    }


def build(groups: int, seed: int) -> list[list[dict[str, Any]]]:
    from generators.grounding import seeded_contrast_group
    from generators.homes import holdout_pools

    built: list[list[dict[str, Any]]] = []
    index = 0
    with holdout_pools():
        while len(built) < groups:
            if index >= groups * 4:
                raise SystemExit(f"only {len(built)} of {groups} contrast groups after {index} tries")
            try:
                built.append(seeded_contrast_group(index, seed))
            except ValueError:
                pass
            index += 1
    return built


def write(groups: list[list[dict[str, Any]]], out: Path) -> tuple[Path, Path]:
    cases_dir, homes_dir = out / "cases", out / "homes"
    cases_dir.mkdir(parents=True, exist_ok=True)
    homes_dir.mkdir(parents=True, exist_ok=True)
    case_path = cases_dir / CASE_FILE
    with case_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            for variant in group:
                home = eval_home(variant["home"])
                (homes_dir / f"{home['home_id']}.json").write_text(
                    json.dumps(home, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                handle.write(json.dumps(eval_case(variant), ensure_ascii=False) + "\n")
    return case_path, homes_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the held-out contrast eval set.")
    parser.add_argument("out", type=Path, help="output directory for cases/ and homes/")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    groups = build(args.groups, args.seed)
    case_path, homes_dir = write(groups, args.out)
    print(json.dumps({
        "groups": len(groups),
        "cases": sum(len(group) for group in groups),
        "cases_path": str(case_path),
        "homes_dir": str(homes_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
