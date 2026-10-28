#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(passed: int, total: int) -> dict[str, Any]:
    return {"passed": passed, "total": total, "accuracy": round(passed / total, 4) if total else None}


def _bucket_rates(buckets: dict[Any, list[bool]]) -> dict[str, dict[str, Any]]:
    return {str(key): _rate(sum(values), len(values)) for key, values in sorted(buckets.items())}


def score(cases: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    passed = {row["case_id"]: bool(row["passed"]) for row in outcomes}
    by_variant: dict[int, list[bool]] = defaultdict(list)
    by_axis: dict[str, list[bool]] = defaultdict(list)
    groups: dict[str, list[bool | None]] = defaultdict(list)
    for case in cases:
        if "contrast_group" not in case:
            continue
        result = passed.get(case["id"])
        groups[case["contrast_group"]].append(result)
        if result is None:
            continue
        by_variant[case["contrast_variant"]].append(result)
        by_axis[case["contrast_axis"]].append(result)
    complete = [results for results in groups.values() if None not in results]
    return {
        "cases_scored": sum(len(values) for values in by_axis.values()),
        "cases_missing": sum(results.count(None) for results in groups.values()),
        "overall": _rate(
            sum(sum(values) for values in by_axis.values()),
            sum(len(values) for values in by_axis.values()),
        ),
        "per_variant": _bucket_rates(by_variant),
        "per_axis": _bucket_rates(by_axis),
        "group_consistency": {
            **_rate(sum(all(results) for results in complete), len(complete)),
            "incomplete_groups": len(groups) - len(complete),
        },
    }


def table(report: dict[str, Any]) -> str:
    lines = [f"{'bucket':<28}{'passed':>8}{'total':>8}{'accuracy':>10}"]

    def row(label: str, rate: dict[str, Any]) -> None:
        accuracy = "-" if rate["accuracy"] is None else f"{rate['accuracy']:.1%}"
        lines.append(f"{label:<28}{rate['passed']:>8}{rate['total']:>8}{accuracy:>10}")

    row("overall", report["overall"])
    for variant, rate in report["per_variant"].items():
        row(f"variant {variant}", rate)
    for axis, rate in report["per_axis"].items():
        row(f"axis {axis}", rate)
    row("GROUP CONSISTENCY", report["group_consistency"])
    return "\n".join(lines)


def _outcomes_path(path: Path) -> Path:
    return path / "outcomes.jsonl" if path.is_dir() else path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a held-out contrast eval run.")
    parser.add_argument("results", type=Path, help="eval run dir or its outcomes.jsonl")
    parser.add_argument("cases", type=Path, help="contrast_holdout.jsonl from build_contrast_holdout.py")
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    args = parser.parse_args(argv)
    report = score(_read_jsonl(args.cases), _read_jsonl(_outcomes_path(args.results)))
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(table(report), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
