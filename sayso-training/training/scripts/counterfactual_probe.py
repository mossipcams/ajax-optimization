#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENDOR_SAYSO = ROOT / "vendor" / "SaySo"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(VENDOR_SAYSO))

import evals.runner as runner
from evals.adapters import InMemoryAdapter
from evals.cases import Case, load_all_cases, load_home
from evals.outcomes import PRODUCTION_MAX_OUTPUT_TOKENS, QUERY_TOOLS, CaseResult, Turn
from evals.runner import production_catalog
from evals.scorer import is_action_call, summarize

PROBE_KINDS = ("device_absent", "tool_removed", "clarify_resolved")
SCORING = {
    "device_absent": "refusal",
    "tool_removed": "refusal",
    "clarify_resolved": "action_names_only_remaining",
}
KIND_ORDER = {kind: index for index, kind in enumerate(PROBE_KINDS)}


@dataclass(frozen=True)
class Pair:
    kind: str
    original: Case
    variant: Case
    variant_home: dict[str, Any] | None
    removed_entity: str | None
    removed_tool: str | None
    remaining_entity: str | None

    @property
    def pair_id(self) -> str:
        return f"{self.kind}:{self.original.id}"


def _expected(response_type: str, calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "response_type": response_type,
        "calls": calls or [],
        "clarification": None,
        "forbidden_entities": [],
        "spoken_aliases": {},
    }


def _named(entities: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [entity for entity in entities if str(entity["name"]).casefold() == name.casefold()]


def _labels(entity: dict[str, Any]) -> set[str]:
    return {str(entity["name"]).casefold(), *(str(alias).casefold() for alias in entity.get("aliases") or [])}


def _phrase_in(phrase: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None


def _mentioned(utterance: str, labels: set[str]) -> bool:
    return any(_phrase_in(label, utterance) for label in labels)


def _clashes(first: set[str], second: set[str]) -> bool:
    return any(_phrase_in(a, b) or _phrase_in(b, a) for a in first for b in second)


def _without(home: dict[str, Any], entity: dict[str, Any], household: str) -> dict[str, Any]:
    name = str(entity["name"]).casefold()
    variant = copy.deepcopy(home)
    variant["home_id"] = household
    for key in ("entities", "context_entities"):
        if key in variant:
            variant[key] = [
                item
                for item in variant[key]
                if not (isinstance(item, dict) and str(item.get("name")).casefold() == name)
            ]
    return variant


def _single_action_call(case: Case) -> dict[str, Any] | None:
    if case.history or case.after_utterance or case.unavailable is not None:
        return None
    if case.expected["response_type"] != "action":
        return None
    calls = case.expected["calls"]
    if len(calls) != 1 or calls[0]["name"] in QUERY_TOOLS:
        return None
    return calls[0]


def _variant_case(
    case: Case,
    kind: str,
    household: str,
    unavailable: dict[str, Any] | None,
    expected: dict[str, Any],
) -> Case:
    return replace(
        case,
        id=f"{case.id}::{kind}",
        category=f"counterfactual_{kind}",
        tags=("counterfactual", kind),
        household=household,
        unavailable=unavailable,
        expected=expected,
        provenance={"source": "counterfactual_probe", "derived_from": case.id, "kind": kind},
        source=f"counterfactual_probe:{kind}",
        history=(),
        after_utterance=(),
    )


def _device_absent_pair(case: Case, home: dict[str, Any]) -> Pair | None:
    call = _single_action_call(case)
    if call is None:
        return None
    name = call["arguments"].get("name")
    if not isinstance(name, str):
        return None
    matches = _named(home["entities"], name)
    if len(matches) != 1:
        return None
    target = matches[0]
    target_labels = _labels(target)
    for other in home["entities"]:
        if other is target:
            continue
        other_labels = _labels(other)
        if _clashes(target_labels, other_labels) or _mentioned(case.utterance, other_labels):
            return None
    household = f"{case.household}::device_absent::{case.id}"
    variant = _variant_case(
        case,
        "device_absent",
        household,
        {"kind": "device_absent", "capability": name, "removed_tools": []},
        _expected("refusal"),
    )
    return Pair(
        "device_absent",
        case,
        variant,
        _without(home, target, household),
        target["name"],
        None,
        None,
    )


def _tool_removed_pair(case: Case, home: dict[str, Any]) -> Pair | None:
    call = _single_action_call(case)
    if call is None:
        return None
    tool = call["name"]
    try:
        production_catalog(home, removed_tools=[tool])
    except ValueError:
        return None
    variant = _variant_case(
        case,
        "tool_removed",
        case.household,
        {"kind": "tool_removed", "capability": tool, "removed_tools": [tool]},
        _expected("refusal"),
    )
    return Pair("tool_removed", case, variant, None, None, tool, None)


def _clarify_resolved_pair(case: Case, home: dict[str, Any]) -> Pair | None:
    if case.history or case.after_utterance or case.unavailable is not None:
        return None
    if case.expected["response_type"] != "clarification":
        return None
    clarification = case.expected.get("clarification") or {}
    candidates = clarification.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        return None
    if not all(isinstance(candidate, str) for candidate in candidates):
        return None
    matches = [_named(home["entities"], candidate) for candidate in candidates]
    if any(len(found) != 1 for found in matches):
        return None
    removed, remaining = matches[0][0], matches[1][0]
    if removed is remaining:
        return None
    household = f"{case.household}::clarify_resolved::{case.id}"
    variant = _variant_case(case, "clarify_resolved", household, None, _expected("action"))
    return Pair(
        "clarify_resolved",
        case,
        variant,
        _without(home, removed, household),
        removed["name"],
        None,
        remaining["name"],
    )


def build_pairs(cases: Sequence[Case]) -> list[Pair]:
    homes: dict[str, dict[str, Any]] = {}
    pairs: list[Pair] = []
    for case in sorted(cases, key=lambda item: item.id):
        if case.household not in homes:
            homes[case.household] = load_home(case.household)
        home = homes[case.household]
        for builder in (_device_absent_pair, _tool_removed_pair, _clarify_resolved_pair):
            pair = builder(case, home)
            if pair is not None:
                pairs.append(pair)
    return sorted(pairs, key=lambda pair: (pair.original.id, KIND_ORDER[pair.kind]))


def limit_pairs(pairs: Sequence[Pair], limit: int) -> list[Pair]:
    kept: list[Pair] = []
    counts = {kind: 0 for kind in PROBE_KINDS}
    for pair in pairs:
        if counts[pair.kind] < limit:
            kept.append(pair)
            counts[pair.kind] += 1
    return kept


def build_all_pairs(limit_per_kind: int | None = None) -> list[Pair]:
    pairs = build_pairs(load_all_cases())
    return pairs if limit_per_kind is None else limit_pairs(pairs, limit_per_kind)


def pair_counts(pairs: Sequence[Pair]) -> dict[str, int]:
    return {kind: sum(1 for pair in pairs if pair.kind == kind) for kind in PROBE_KINDS}


def pair_record(pair: Pair) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "kind": pair.kind,
        "scoring": SCORING[pair.kind],
        "removed_entity": pair.removed_entity,
        "removed_tool": pair.removed_tool,
        "remaining_entity": pair.remaining_entity,
        "original": pair.original.as_dict(),
        "variant": pair.variant.as_dict(),
        "variant_home": pair.variant_home,
    }


def _call_target(call: dict[str, Any]) -> str | None:
    arguments = call.get("arguments")
    name = arguments.get("name") if isinstance(arguments, dict) else None
    return name.casefold() if isinstance(name, str) else None


def names_only(turn: Turn, entity_name: str) -> bool:
    if turn.parse_error is not None or turn.violations:
        return False
    actions = [call for call in turn.calls if is_action_call(call)]
    return bool(actions) and all(_call_target(call) == entity_name.casefold() for call in actions)


def _output(turn: Turn) -> dict[str, Any]:
    return {"text": turn.text, "calls": turn.calls, "parse_error": turn.parse_error}


def _probe_cases(pairs: Sequence[Pair]) -> list[Case]:
    cases: dict[str, Case] = {}
    for pair in pairs:
        cases.setdefault(pair.original.id, pair.original)
        cases[pair.variant.id] = pair.variant
    return [cases[key] for key in sorted(cases)]


@contextmanager
def _households(variants: dict[str, dict[str, Any]]) -> Iterator[None]:
    original = runner.load_home

    def resolve(home_id: str) -> dict[str, Any]:
        if home_id in variants:
            return copy.deepcopy(variants[home_id])
        return original(home_id)

    runner.load_home = resolve
    try:
        yield
    finally:
        runner.load_home = original


def _row(pair: Pair, by_id: dict[str, CaseResult]) -> dict[str, Any]:
    original = by_id[pair.original.id]
    variant = by_id[pair.variant.id]
    if pair.kind == "clarify_resolved":
        variant_passed = names_only(variant.turn, pair.remaining_entity or "")
    else:
        variant_passed = variant.passed
    original_output = _output(original.turn)
    variant_output = _output(variant.turn)
    identical = original_output == variant_output
    return {
        "pair_id": pair.pair_id,
        "kind": pair.kind,
        "original_case_id": pair.original.id,
        "variant_case_id": pair.variant.id,
        "original_passed": original.passed,
        "variant_passed": variant_passed,
        "original_outcome": original.outcome.value,
        "variant_outcome": variant.outcome.value,
        "original_output": original_output,
        "variant_output": variant_output,
        "identical_output": identical,
        "appropriate_change": original.passed and variant_passed,
        "sticky": identical,
    }


def evaluate_pairs(pairs: Sequence[Pair], adapter: Any) -> tuple[list[CaseResult], list[dict[str, Any]]]:
    cases = _probe_cases(pairs)
    variants = {
        pair.variant.household: pair.variant_home for pair in pairs if pair.variant_home is not None
    }
    with _households(variants):
        results = runner.evaluate(cases, adapter)
    by_id = {result.case_id: result for result in results}
    return results, [_row(pair, by_id) for pair in pairs]


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _share(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return _rate(sum(1 for row in rows if row[key]), len(rows))


def _rates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "appropriate_change_rate": _share(rows, "appropriate_change"),
        "sticky_rate": _share(rows, "sticky"),
        "original_pass_rate": _share(rows, "original_passed"),
        "variant_pass_rate": _share(rows, "variant_passed"),
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _rates(rows),
        "kinds": {kind: _rates([row for row in rows if row["kind"] == kind]) for kind in PROBE_KINDS},
    }


def _load_predictor(checkpoint: str, threads: int) -> Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    end = "<|im_end|>"

    def predict(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=PRODUCTION_MAX_OUTPUT_TOKENS, do_sample=False)
        text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False).split(end)[0]
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return predict


def _pairs_payload(pairs: Sequence[Pair]) -> dict[str, Any]:
    return {"counts": pair_counts(pairs), "pairs": [pair_record(pair) for pair in pairs]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Counterfactual probe: identical request, changed device list or tool catalog.")
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-per-kind", type=int, default=None)
    args = parser.parse_args()

    if args.limit_per_kind is not None and args.limit_per_kind < 1:
        parser.error("--limit-per-kind must be at least 1")
    if not args.build_only and not args.checkpoint:
        parser.error("checkpoint is required unless --build-only is set")

    pairs = build_all_pairs(args.limit_per_kind)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.build_only:
        dest = args.out / "pairs.json"
        dest.write_text(
            json.dumps(_pairs_payload(pairs), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = pair_counts(pairs)
        print(" ".join(f"{kind}={counts[kind]}" for kind in PROBE_KINDS) + f" -> {dest}")
        return 0

    predict = _load_predictor(args.checkpoint, args.threads)
    adapter = InMemoryAdapter(predict)
    runner.RESULTS_DIR = args.out
    results, rows = evaluate_pairs(pairs, adapter)
    dest = runner.write_run(
        cases=_probe_cases(pairs),
        results=results,
        adapter=adapter,
        suite=None,
        extra_metadata={"checkpoint": args.checkpoint, "gate_label": "counterfactual_probe"},
    )
    report = {
        "checkpoint": args.checkpoint,
        "run_dir": str(dest),
        "counts": pair_counts(pairs),
        "summary": summarize_rows(rows),
        "run_summary": summarize(results),
        "pairs": rows,
    }
    report_path = args.out / "counterfactual_probe.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = report["summary"]["overall"]
    print(
        f"overall: pairs={overall['pairs']} appropriate_change={overall['appropriate_change_rate']} "
        f"sticky={overall['sticky_rate']} original_pass={overall['original_pass_rate']} "
        f"variant_pass={overall['variant_pass_rate']} -> {report_path}"
    )
    for kind in PROBE_KINDS:
        stats = report["summary"]["kinds"][kind]
        print(
            f"{kind}: pairs={stats['pairs']} appropriate_change={stats['appropriate_change_rate']} "
            f"sticky={stats['sticky_rate']} original_pass={stats['original_pass_rate']} "
            f"variant_pass={stats['variant_pass_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
