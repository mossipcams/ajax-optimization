#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"
for _path in (str(TRAINING_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generators.validation import check_quality_eval_overlap


def normalize(name: object) -> str:
    return " ".join(str(name).split())


HOLDOUT_TARGETS = ("Kitchen light", "Christmas tree", "Desk Echo Dot", "Rocky room cam")
NEAREST_NAME_STOP_WORDS = frozenset({"the", "light", "lights", "lamp"})


def load_real_home(path: Path) -> tuple[set[str], set[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load real-home fixture {path}: {exc}") from exc
    entity_names: set[str] = set()
    for entry in data.get("entities") or []:
        entity_names.add(str(entry["name"]).lower())
        for alias in entry.get("aliases") or []:
            entity_names.add(str(alias).lower())
    context_names = {str(entry["name"]).lower() for entry in data.get("context_entities") or []}
    return entity_names, context_names


def tool_call_targets(messages: list[dict]) -> list[str]:
    targets: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            name = arguments.get("name") if isinstance(arguments, dict) else None
            if isinstance(name, str):
                targets.append(normalize(name))
    return targets


def run_audit(corpus_path: Path, real_home_path: Path) -> dict:
    entity_names, context_names = load_real_home(real_home_path)

    fabricated_targets = 0
    context_only_actions = 0
    eval_overlap = 0
    withheld_pairs = 0
    clarify_pairs = 0
    nearest_name_exclusions = 0
    real_home_targets: Counter[str] = Counter()
    contrast_systems: dict[str, str | None] = {}

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{corpus_path}:{index}: invalid JSON: {exc}") from exc
        messages = row.get("messages") or []
        meta = row.get("metadata") or {}
        system_text = next(
            (m["content"] for m in messages if m.get("role") == "system" and isinstance(m.get("content"), str)),
            "",
        )
        system_lower = normalize(system_text).lower()
        targets = tool_call_targets(messages)

        fabricated_targets += sum(1 for t in targets if t.lower() not in system_lower)

        for utterance in (m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)):
            if check_quality_eval_overlap(utterance):
                eval_overlap += 1

        pair_kind = meta.get("pair_kind")
        if pair_kind == "withheld_tool":
            withheld_pairs += 1
        elif pair_kind == "clarify":
            clarify_pairs += 1

        excluded = [normalize(n).lower() for n in meta.get("excluded_names") or []]
        expected = [normalize(n).lower() for n in meta.get("expected_target_names") or []]
        if excluded and expected:
            shared = {word for word in excluded[0].split()} - NEAREST_NAME_STOP_WORDS
            if any(shared & {word for word in target.split()} for target in expected):
                nearest_name_exclusions += 1

        if pair_kind == "real_home_contrast" and meta.get("contrastive_group") is not None:
            group = str(meta["contrastive_group"])
            first = contrast_systems.setdefault(group, system_text)
            if first is not None and first != system_text:
                contrast_systems[group] = "__mismatch__"

        if meta.get("real_home"):
            for t in targets:
                lowered = t.lower()
                real_home_targets[lowered] += 1
                if lowered in context_names:
                    context_only_actions += 1

    holdout_counts = {t: real_home_targets[t.lower()] for t in HOLDOUT_TARGETS}
    tv_count = real_home_targets["tv"]
    living_room_count = real_home_targets["living room lights"]
    entity_total = sum(count for name, count in real_home_targets.items() if name in entity_names)
    largest_entity_count = max(
        (count for name, count in real_home_targets.items() if name in entity_names),
        default=0,
    )
    max_share = largest_entity_count / entity_total if entity_total else 0.0
    contrast_groups = len(contrast_systems)
    systems_identical = all(v != "__mismatch__" for v in contrast_systems.values())

    gates = {
        "fabricated_targets": {
            "value": fabricated_targets,
            "threshold": 0,
            "pass": fabricated_targets == 0,
        },
        "context_only_actions": {
            "value": context_only_actions,
            "threshold": 0,
            "pass": context_only_actions == 0,
        },
        "eval_overlap": {
            "value": eval_overlap,
            "threshold": 0,
            "pass": eval_overlap == 0,
        },
        "holdout_targets": {
            "value": holdout_counts,
            "threshold": "each >= 1",
            "pass": all(count >= 1 for count in holdout_counts.values()),
        },
        "tv_vs_living_room": {
            "value": {"TV": tv_count, "Living room lights": living_room_count},
            "threshold": "count(TV) <= 2 * count('Living room lights')",
            "pass": tv_count <= 2 * living_room_count,
        },
        "max_real_home_target_share": {
            "value": max_share,
            "threshold": 0.10,
            "pass": max_share <= 0.10,
        },
        "real_home_contrast_groups": {
            "value": {"groups": contrast_groups, "identical_systems": systems_identical},
            "threshold": ">= 300 and byte-identical system per group",
            "pass": contrast_groups >= 300 and systems_identical,
        },
        "withheld_pairs": {
            "value": withheld_pairs,
            "threshold": 1000,
            "pass": withheld_pairs >= 1000,
        },
        "clarify_pairs": {
            "value": clarify_pairs,
            "threshold": 1000,
            "pass": clarify_pairs >= 1000,
        },
        "nearest_name_exclusions": {
            "value": nearest_name_exclusions,
            "threshold": 500,
            "pass": nearest_name_exclusions >= 500,
        },
    }
    return {"gates": gates, "passed": all(g["pass"] for g in gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a v7 SFT corpus against the plan-v7 verification gates")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--real-home", type=Path, default=REPO_ROOT / "training/fixtures/real_home.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run_audit(args.corpus, args.real_home)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
