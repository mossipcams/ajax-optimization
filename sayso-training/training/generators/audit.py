
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from adapters.schema import tool_schema_map, validate_tool_arguments
from generators.capability_registry import SCRIPT_ACTION_TOOL
from generators.coverage import ABSENCE, bare_tool_name, classify_row, expected_tool
from generators.validation import (
    check_label_conflicts,
    check_quality_eval_overlap,
    gate_areas,
    gate_clarification,
    gate_hidden_capability_refusal,
    gate_named_device_calls,
    gate_offered_tools,
    gate_on_off_polarity,
    gate_scripts,
    is_satellite_contrast,
    satellite_contrast_spoken,
)


def _shares_long_word(name: str, alias: str) -> bool:
    name_words = {word for word in re.findall(r"[a-z]+", name.lower()) if len(word) >= 3}
    alias_words = {word for word in re.findall(r"[a-z]+", alias.lower()) if len(word) >= 3}
    return bool(name_words & alias_words)


def audit_rows(
    rows,
    *,
    expected_count=None,
    required_operations=None,
    min_positive_per_operation=0,
    min_positive_per_tool=0,
    max_absence_rate=None,
    get_datetime_positive_min=0,
    allowed_operation_shortfall=0,
):
    counts = Counter()
    casing = {"call": Counter(), "no_call": Counter()}
    seen = set()
    grammar_sources = Counter()
    grammar_templates = set()
    requests = set()
    positive_by_operation = Counter()
    positive_by_tool = Counter()
    by_outcome = Counter()
    by_targeting = Counter()
    by_tool = Counter()
    by_domain = Counter()
    by_withheld_tool = Counter()
    negatives_by_reason = Counter()
    real_home_rows = 0
    get_datetime_positive = 0
    for row in rows:
        facets = classify_row(row)
        by_outcome[facets["outcome"]] += 1
        by_targeting[facets["targeting"]] += 1
        real_home_rows += bool(row.get("metadata", {}).get("real_home") or facets["real_home"])
        for tool in facets["tools"]:
            by_tool[tool] += 1
        if facets["positive"]:
            key = (facets["tier"], facets["capability"], facets["operation"])
            positive_by_operation[key] += 1
            by_domain[(facets["capability"], facets["operation"], facets["targeting"])] += 1
            tool = expected_tool(facets["capability"], facets["operation"])
            if tool:
                positive_by_tool[tool] += 1
        else:
            negatives_by_reason[facets["no_action_reason"] or facets["outcome"]] += 1
        meta = row.get("metadata", {})
        for tool in meta.get("unavailable_tools", []):
            by_withheld_tool[tool] += 1
        row_id = meta.get("candidate_id")
        if not row_id or row_id in seen:
            raise ValueError(f"missing or duplicate candidate ID: {row_id}")
        seen.add(row_id)
        messages = row["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        requests.add(user.casefold())
        grammar = [entry for entry in meta.get("linguistics", [])
                   if entry.get("source", "").startswith("sentences/en/")]
        counts["ohf_rows"] += bool(grammar)
        counts["fallback_rows"] += any(entry.get("source") == "sayso_fallback"
                                       for entry in meta.get("linguistics", []))
        for entry in grammar:
            grammar_sources[entry["source"]] += 1
            grammar_templates.add((entry["source"], entry["block"], entry["template"]))
        if not user.strip() or check_quality_eval_overlap(user):
            raise ValueError(f"empty request or eval overlap: {row_id}")
        calls = [c["function"] for m in messages for c in m.get("tool_calls", [])]
        if any(bare_tool_name(call["name"]) == "GetDateTime" for call in calls):
            get_datetime_positive += 1
            positive_by_tool["GetDateTime"] += 1
        schemas = tool_schema_map(row["tools"])
        offered_bare = {bare_tool_name(name) for name in schemas}
        final = messages[-1].get("content", "")
        if not calls and meta.get("capability") == "timers":
            required = expected_tool("timers", meta.get("operation") or "")
            if required in offered_bare or re.search(r"has no .* available", final, re.I):
                raise ValueError(f"contradictory timer refusal: {row_id}")
        if any(bare_tool_name(call["name"]) == "GetLiveContext" for call in calls) and final.strip().lower() == "done.":
            raise ValueError(f"state query labeled as action: {row_id}")
        call_names = []
        for call in calls:
            arguments = call["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            reason = validate_tool_arguments(call["name"], arguments, schemas)
            if reason:
                raise ValueError(f"{row_id}: {reason}")
            if arguments.get("name") in meta.get("excluded_names", []):
                raise ValueError(f"excluded target called: {row_id}")
            call_names.append(str(arguments.get("name") or ""))
        if set(meta.get("unavailable_tools", [])) & offered_bare:
            raise ValueError(f"blocked tool is offered: {row_id}")
        excluded = meta.get("excluded_names", [])
        if excluded and not satellite_contrast_spoken(user, call_names, excluded) and (
            "leave" not in user.lower() or any(name.lower() not in user.lower() for name in excluded)
        ):
            raise ValueError(f"missing exclusion wording: {row_id}")
        if meta.get("category") in {"multi_action", "exclusion"} and len(calls) < 2 and not is_satellite_contrast(call_names, excluded):
            raise ValueError(f"false multiple-action tag: {row_id}")
        if meta.get("category") == "exclusion" and not excluded:
            raise ValueError(f"false exclusion tag: {row_id}")
        kind = "call" if calls else "no_call"
        casing[kind]["upper" if user[0].isupper() else "lower"] += 1
        counts["rows"] += 1
        counts[kind] += 1
        counts["tool_calls"] += len(calls)
        counts["multi_call_rows"] += len(calls) > 1
        counts["exclusion_rows"] += bool(excluded)
        counts["exclusion_same_area"] += bool(excluded and meta.get("exclusion_scope"))
        counts["exclusion_named_list"] += bool(excluded and not meta.get("exclusion_scope"))
        counts["alias_rows"] += meta.get("category") != "ambiguity" and any(alias.lower() in user.lower() and name.lower() not in user.lower()
                                    for name, alias in meta.get("spoken_targets", {}).items())
        counts["nickname_alias_rows"] += meta.get("category") != "ambiguity" and any(
            alias.lower() in user.lower() and not _shares_long_word(name, alias)
            for name, alias in meta.get("spoken_targets", {}).items()
        )
        counts["conversational_rows"] += bool(re.search(r"\b(?:please|could you|can you)\b", user, re.I))
    if expected_count is not None and counts["rows"] != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {counts['rows']}")
    for kind, cases in casing.items():
        total = sum(cases.values())
        if total >= 10 and not 0.25 <= cases["upper"] / total <= 0.75:
            raise ValueError(f"label-dependent casing for {kind}: {dict(cases)}")
    if counts["rows"] >= 1000:
        for behavior in ("multi_call_rows", "exclusion_rows", "alias_rows"):
            if counts[behavior] < counts["rows"] * 0.005:
                raise ValueError(f"insufficient actual {behavior}: {counts[behavior]}")

    missing_operations = {}
    missing_tools = {}
    if required_operations:
        for key in sorted(required_operations):
            got = positive_by_operation.get(key, 0)
            if got < min_positive_per_operation:
                missing_operations[str(key)] = got
        required_tools = {
            tool
            for (_tier, capability, operation) in required_operations
            if (tool := expected_tool(capability, operation)) and tool != SCRIPT_ACTION_TOOL
        }
        if any(cap == "scripts" and op == "run" for _t, cap, op in required_operations):
            required_tools.add(SCRIPT_ACTION_TOOL)
        for tool in sorted(required_tools):
            got = positive_by_tool.get(tool, 0)
            if got < min_positive_per_tool:
                missing_tools[tool] = got
    if missing_operations:
        shortfall = len(missing_operations)
        if shortfall > allowed_operation_shortfall:
            raise ValueError(f"missing positive supervision for operations: {missing_operations}")
    if missing_tools:
        raise ValueError(f"missing positive supervision for tools: {missing_tools}")
    absence_rate = by_outcome.get(ABSENCE, 0) / max(counts["rows"], 1)
    if max_absence_rate is not None and absence_rate > max_absence_rate:
        raise ValueError(
            f"absence answers are {absence_rate:.1%} of the corpus, above the "
            f"{max_absence_rate:.1%} cap ({by_outcome.get(ABSENCE, 0)}/{counts['rows']})"
        )
    if get_datetime_positive_min and get_datetime_positive < get_datetime_positive_min:
        raise ValueError(
            f"missing positive supervision for GetDateTime: "
            f"got {get_datetime_positive}, need {get_datetime_positive_min}"
        )

    return {**counts, "unique_requests": len(requests),
            "ohf_source_count": len(grammar_sources), "ohf_template_count": len(grammar_templates),
            "ohf_sources": dict(sorted(grammar_sources.items())),
            "casing": {kind: dict(cases) for kind, cases in casing.items()},
            "real_home_rows": real_home_rows,
            "by_outcome": dict(sorted(by_outcome.items())),
            "by_targeting": dict(sorted(by_targeting.items())),
            "by_tool": dict(sorted(by_tool.items())),
            "by_withheld_tool": dict(sorted(by_withheld_tool.items())),
            "negatives_by_reason": dict(sorted(negatives_by_reason.items())),
            "absence_rate": round(absence_rate, 4),
            "positive_by_tool": dict(sorted(positive_by_tool.items())),
            "get_datetime_positive": get_datetime_positive,
            "positive_by_operation": {str(k): v for k, v in sorted(positive_by_operation.items())},
            "positive_by_domain_targeting": {
                f"{cap}/{op}/{mode}": n for (cap, op, mode), n in sorted(by_domain.items())
            }}


HARD_GATES = (
    ("gate_offered_tools", gate_offered_tools),
    ("gate_scripts", gate_scripts),
    ("gate_areas", gate_areas),
    ("gate_named_device_calls", gate_named_device_calls),
    ("gate_clarification", gate_clarification),
    ("gate_on_off_polarity", gate_on_off_polarity),
    ("gate_hidden_capability_refusal", gate_hidden_capability_refusal),
)
_FAMILY_FIELDS = ("family", "template_family", "grounding_family", "category")
_EXAMPLE_LIMIT = 3


def _row_id(row: dict, line_number: int):
    return (row.get("metadata") or {}).get("candidate_id", line_number)


def _conflict_view(row: dict) -> dict:
    messages = []
    for message in row.get("messages") or []:
        role = message.get("role")
        if role in {"system", "user"}:
            messages.append({"role": role, "content": message.get("content")})
        elif role == "assistant" and message.get("train_on_turn"):
            messages.append({
                "role": role,
                "train_on_turn": True,
                "tool_calls": [
                    {"function": {"name": call["function"]["name"], "arguments": call["function"].get("arguments")}}
                    for call in message.get("tool_calls") or []
                ],
            })
    return {
        "messages": messages,
        "tools": [{"function": {"name": tool["function"]["name"]}} for tool in row.get("tools") or []],
        "metadata": {"no_action_reason": (row.get("metadata") or {}).get("no_action_reason")},
    }


def audit_corpus(path: Path, *, limit: int | None = None) -> dict:
    scanned = 0
    rejected_rows = 0
    row_ids: list = []
    conflict_views: list[dict] = []
    gate_stats = {name: {"rejected": 0, "reasons": Counter(), "examples": []} for name, _ in HARD_GATES}
    rejected_by_field: dict[str, Counter] = {field: Counter() for field in _FAMILY_FIELDS}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and scanned >= limit:
                break
            row = json.loads(line)
            scanned += 1
            row_id = _row_id(row, line_number)
            row_ids.append(row_id)
            conflict_views.append(_conflict_view(row))
            rejected = False
            for name, gate in HARD_GATES:
                reason = gate(row)
                if reason is None:
                    continue
                rejected = True
                stats = gate_stats[name]
                stats["rejected"] += 1
                stats["reasons"][reason] += 1
                if len(stats["examples"]) < _EXAMPLE_LIMIT:
                    stats["examples"].append(row_id)
            if rejected:
                rejected_rows += 1
                metadata = row.get("metadata") or {}
                for field in _FAMILY_FIELDS:
                    if metadata.get(field) is not None:
                        rejected_by_field[field][str(metadata[field])] += 1

    conflict_groups = check_label_conflicts(conflict_views)
    conflict_examples = [
        {
            "command": group["command"][:160],
            "rows": {label[:160]: [row_ids[index] for index in indexes[:_EXAMPLE_LIMIT]] for label, indexes in group["labels"].items()},
        }
        for group in conflict_groups[:_EXAMPLE_LIMIT]
    ]
    return {
        "corpus": str(path),
        "limit": limit,
        "rows_scanned": scanned,
        "rows_rejected": rejected_rows,
        "gates": {
            name: {
                "rejected": stats["rejected"],
                "reasons": dict(sorted(stats["reasons"].items())),
                "examples": stats["examples"],
            }
            for name, stats in gate_stats.items()
        },
        "label_conflicts": {
            "groups": len(conflict_groups),
            "rows": sum(len(indexes) for group in conflict_groups for indexes in group["labels"].values()),
            "examples": conflict_examples,
        },
        "rejected_by_metadata": {
            field: dict(sorted(counts.items())) for field, counts in rejected_by_field.items() if counts
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-absence-rate", type=float, default=None)
    args = parser.parse_args()
    with args.dataset.open() as handle:
        report = audit_rows(
            (json.loads(line) for line in handle),
            expected_count=args.count,
            max_absence_rate=args.max_absence_rate,
        )
    print(json.dumps(report, indent=2))
