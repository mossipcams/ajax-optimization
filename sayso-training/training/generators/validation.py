
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any
from pathlib import Path

import yaml

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools
from generators.context import STATIC_CONTEXT_HEADER
from generators.tools import HA_TOOL_NAMESPACES, namespaced_tool_name, script_tool_name
from generators.capability_registry import CAPABILITIES, SupportLevel, entity_supports, is_satellite_name
from generators.config import DEFAULT_TOKEN_BUDGET
from generators.gold import _type_label
from generators.stt_noise import _int_to_words, utterance_contains_target

_BANNED = re.compile(r"<tool_call>|evals/cases/|tool_call_start", re.I)
_PLACEHOLDER_AREA = "unassigned"


def _normalize_prompt(text: str) -> str:
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


@lru_cache(maxsize=1)
def quality_eval_prompts() -> frozenset[str]:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    from evals.cases import excluded_train_utterances
    return frozenset(_normalize_prompt(prompt) for prompt in excluded_train_utterances())


def check_quality_eval_overlap(utterance: str) -> bool:
    return _normalize_prompt(utterance) in quality_eval_prompts()


def corrupt_spec(spec: dict[str, Any], field: str) -> dict[str, Any]:
    corrupted = dict(spec)
    expected = dict(corrupted.get("expected", {}))
    calls = list(expected.get("calls") or [])
    if calls and field in {"wrong_tool", "wrong_entity"}:
        calls[0] = dict(calls[0])
        if field == "wrong_tool":
            calls[0]["name"] = "NonexistentTool"
        else:
            calls[0]["arguments"] = {**calls[0].get("arguments", {}), "name": "Invented Device"}
        expected["calls"] = calls
        corrupted["expected"] = expected
    return corrupted

_TIMER_NAME_TOOLS = frozenset({
    "HassStartTimer", "HassPauseTimer", "HassUnpauseTimer", "HassCancelTimer",
    "HassIncreaseTimer", "HassDecreaseTimer", "HassTimerStatus",
})


def _entity_can_run(entity: dict[str, Any], tool: str) -> bool:
    capability = entity.get("capability")
    cap = CAPABILITIES.get(capability)
    if cap is None:
        return True
    candidates = [op for op in cap.operations if op.tool_name == tool]
    if not candidates:
        return True
    return any(entity_supports(entity, capability, op.name) for op in candidates)


def _is_satellite_name(name: str) -> bool:
    return is_satellite_name(name)


def is_satellite_contrast(call_names: list[str], excluded: list[str]) -> bool:
    if len(call_names) != 1 or len(excluded) != 1:
        return False
    return _is_satellite_name(call_names[0]) != _is_satellite_name(excluded[0])


def satellite_contrast_spoken(utterance: str, call_names: list[str], excluded: list[str]) -> bool:
    if not is_satellite_contrast(call_names, excluded):
        return False
    lowered = utterance.casefold()
    if _is_satellite_name(excluded[0]):
        return lowered.endswith(", not you")
    return lowered.endswith(f", not the {excluded[0].casefold()}")


def shared_alias_action_conflict(spec: dict[str, Any]) -> bool:
    if (spec.get("expected") or {}).get("kind") != "action":
        return False
    text = (spec.get("utterance") or "").casefold()
    entities = (spec.get("home") or {}).get("entities") or []
    for entity in entities:
        for alias in entity.get("aliases") or []:
            needle = alias.casefold()
            if not re.search(rf"\b{re.escape(needle)}\b", text):
                continue
            if any(
                other is not entity
                and other.get("area") == entity.get("area")
                and needle in {name.casefold() for name in other.get("aliases") or []}
                for other in entities
            ):
                return True
    return False


def validate_spec(spec: dict[str, Any]) -> str | None:
    if spec.get("shared_alias") and shared_alias_action_conflict(spec):
        return "shared_alias_action"
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    cap = CAPABILITIES.get(spec.get("capability"))
    operation = next((op for op in cap.operations if op.name == spec.get("operation")), None) if cap else None
    if expected.get("kind") == "no_action":
        if spec.get("capability") == "timers" and expected.get("response") != "unsupported":
            return "timer_inventory_refusal"
        if expected.get("response") == "unsupported" and operation and operation.support != SupportLevel.UNAVAILABLE:
            requested = expected.get("requested", {}).get("calls", [])
            withheld = set(expected.get("unavailable_tools", []))
            if not requested or any(call["name"] not in withheld for call in requested):
                return "unsupported_without_withheld_tool"
        if expected.get("response") == "device_unsupported":
            named = expected.get("unsupported_names") or []
            by_name = {e["name"]: e for e in spec.get("home", {}).get("entities", [])}
            context_names = {e["name"] for e in spec.get("home", {}).get("context_entities") or []}
            if not named:
                return "device_unsupported_without_names"
            for name in named:
                if name not in by_name:
                    if name in context_names:
                        continue
                    return "unknown_canonical_entity"
                if entity_supports(by_name[name], spec.get("capability"), spec.get("operation")):
                    return "contradictory_device_support"
    if any(call.get("name") == "GetLiveContext" for call in calls) and expected.get("kind") != "status":
        return "state_query_requires_status_label"
    if expected.get("kind") == "status" and not any(call.get("name") == "GetLiveContext" for call in calls):
        return "status_requires_state_query"
    entities = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "").casefold()
        domains = {cap.domain for cap in CAPABILITIES.values() if _type_label(cap.name) == unavailable.get("type")}
        if any(
            e.get("area", "").casefold() == area
            and (e.get("domain") in domains or unavailable.get("type") == "devices")
            for e in entities.values()
        ):
            return "contradictory_absence"
    call_names = [str((call.get("arguments") or {}).get("name") or "") for call in calls]
    contrast = is_satellite_contrast(call_names, spec.get("excluded_names") or [])
    if spec.get("category") in {"multi_action", "exclusion"} and len(calls) < 2 and not contrast:
        return "missing_multiple_actions"
    if spec.get("category") == "exclusion" and not spec.get("excluded_names"):
        return "missing_excluded_target"
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    script_names = {
        script_tool_name(entity)
        for entity in spec.get("home", {}).get("entities", [])
        if entity.get("domain") == "script"
    }
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        if name in script_names:
            if arguments:
                return "script_tool_takes_no_arguments"
            if any(
                script_tool_name(e) == name and e["name"] in excluded
                for e in entities.values()
                if e.get("domain") == "script"
            ):
                return "excluded_entity_called"
            continue
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities and name not in _TIMER_NAME_TOOLS:
            return "unknown_canonical_entity"
        if target in excluded:
            return "excluded_entity_called"
        if target in entities and not _entity_can_run(entities[target], name):
            return "entity_lacks_capability"
    for canonical, spoken in (spec.get("spoken_targets") or {}).items():
        if canonical not in entities or not str(spoken).strip():
            return "invalid_spoken_target"
    return None


_STATUS_QUERY = re.compile(r"status|\bwhat|^\W*(?:please\s+)?(?:is|are|did|does|do|how)\b|\?\W*$")


def _names_one_clarify_candidate(spec: dict[str, Any], lowered: str) -> bool:
    expected = spec.get("expected") or {}
    if spec.get("phrasing_seed"):
        return False
    if expected.get("response") == "clarify":
        candidates = list(expected.get("candidates") or [])
    else:
        candidates = list(spec.get("follow_up_clarify_candidates") or [])
    if len(candidates) < 2:
        return False
    entities = {e["name"]: e for e in (spec.get("home") or {}).get("entities", [])}

    def spoken(name: str) -> bool:
        phrases = [name, *(entities.get(name) or {}).get("aliases", [])]
        return any(
            len(p.split()) >= 2 and re.search(r"(?<!\w)" + re.escape(p.casefold()) + r"(?!\w)", lowered)
            for p in phrases
        )

    return sum(spoken(name) for name in candidates) == 1


def validate_utterance(spec: dict[str, Any]) -> str | None:
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        return "missing_utterance"
    if _BANNED.search(utterance):
        return "banned_marker"
    expected = spec.get("expected") or {}
    lowered = utterance.casefold()
    if _names_one_clarify_candidate(spec, lowered):
        return "clarify_target_named"
    if spec.get("area_scenario") or spec.get("category") == "area_grounding":
        for name in spec.get("excluded_names") or []:
            if "leave" not in lowered or name.casefold() not in lowered:
                return "missing_exclusion"
        return None
    if expected.get("kind") == "action":
        calls = expected.get("calls") or []
        for call in calls:
            arguments = call.get("arguments") or {}
            for scope in ("area", "floor"):
                if arguments.get(scope) and str(arguments[scope]).casefold() not in lowered:
                    return "missing_expected_scope"
            for key, value in arguments.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key in {"hours", "minutes", "seconds"} and value == 0:
                    continue
                if key == "volume_step":
                    value = abs(value)
                spellings = [str(value)]
                if float(value).is_integer():
                    spellings.append(str(int(value)))
                    words = _int_to_words(int(value))
                    if words:
                        spellings.append(words)
                if not any(
                    re.search(r"(?<![\w.])" + re.escape(word) + r"(?!\w|\.\d)", lowered)
                    for word in spellings
                ):
                    return "missing_expected_value"
        has_area_target = any(
            isinstance(c.get("arguments"), dict) and c["arguments"].get("area") and not c["arguments"].get("name")
            for c in calls
        )
        if (
            calls
            and not has_area_target
            and calls[0]["name"]
            not in {
                "HassCancelAllTimers",
                "HassStartTimer",
                "HassPauseTimer",
                "HassTimerStatus",
            }
        ):
            for name in spec.get("target_names") or []:
                if _is_satellite_name(name):
                    continue
                spoken = spec.get("spoken_targets", {}).get(name, name).casefold()
                if (
                    spoken not in lowered
                    and not utterance_contains_target(lowered, name)
                    and not utterance_contains_target(lowered, spoken)
                ):
                    area = (calls[0].get("arguments") or {}).get("area")
                    if not area or str(area).casefold() not in lowered:
                        if spec.get("discrimination"):
                            continue
                        return "missing_expected_target"
        excluded = spec.get("excluded_names") or []
        if not satellite_contrast_spoken(utterance, [
            str((call.get("arguments") or {}).get("name") or "") for call in calls
        ], excluded) and any("leave" not in lowered or name.casefold() not in lowered for name in excluded):
            return "missing_exclusion"
    if expected.get("kind") == "status" and not _STATUS_QUERY.search(lowered):
        return "status_not_query"
    return None


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _call_with_parsed_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return call
    return {**call, "function": {**function, "arguments": arguments}}


def count_row_tokens(row: dict[str, Any], *, model_name: str) -> int:
    tokenizer = _tokenizer(model_name)
    tools = []
    for tool in row.get("tools") or []:
        fn = tool.get("function", tool)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            }
        )
    messages = []
    for message in row.get("messages") or []:
        entry = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            entry["tool_calls"] = [_call_with_parsed_arguments(c) for c in message["tool_calls"]]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        messages.append(entry)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def row_characters(row: dict[str, Any]) -> int:
    total = len(json.dumps(row.get("tools") or []))
    for message in row.get("messages") or []:
        total += len(str(message.get("content") or ""))
        total += len(json.dumps(message.get("tool_calls") or []))
    return total


_MIN_CHARS_PER_TOKEN = 3.0


def validate_token_budget(
    row: dict[str, Any],
    budget: int,
    *,
    tokenizer_model: str,
) -> str | None:
    if row_characters(row) / _MIN_CHARS_PER_TOKEN <= budget:
        return None
    estimated = False
    try:
        tokens = count_row_tokens(row, model_name=tokenizer_model)
    except Exception:
        from generators.context import serialize_context

        spec_len = len(serialize_context(row.get("metadata", {}))) + sum(
            len(str(m.get("content", ""))) for m in row.get("messages", [])
        )
        tokens = spec_len // 4
        estimated = True
    row.setdefault("metadata", {})["_token_length"] = tokens
    row["metadata"]["_token_length_estimated"] = estimated
    if tokens > budget:
        return "token_budget_exceeded"
    return None


def validate_row(
    spec: dict[str, Any],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    tokenizer_model: str = "LiquidAI/LFM2.5-230M-Base",
    rendered: dict[str, Any] | None = None,
) -> str | None:
    reason = validate_spec(spec)
    if reason:
        return reason
    reason = validate_utterance(spec)
    if reason:
        return reason
    if rendered is not None:
        return validate_token_budget(rendered, token_budget, tokenizer_model=tokenizer_model)
    return None


def validate_serialized_row(
    row: dict[str, Any],
    *,
    token_budget: int,
    tokenizer_model: str,
) -> str | None:
    if _PLACEHOLDER_AREA in json.dumps(row.get("messages") or []).casefold():
        return "placeholder_area"
    if not row.get("tools"):
        return "missing_tools"
    for message in row.get("messages", []):
        if message.get("role") == "assistant" and message.get("train_on_turn"):
            if message.get("tool_calls"):
                for call in message["tool_calls"]:
                    args = call.get("function", {}).get("arguments")
                    if isinstance(args, str) and len(args) > 4096:
                        return "truncated_supervision"
    return validate_token_budget(row, token_budget, tokenizer_model=tokenizer_model)


# Row-level checks for counterfactual groups (plan Step 5).

DEFAULT_JACCARD_THRESHOLD = 0.8
_BARE_TOOL = {namespaced_tool_name(tool): tool for tool in HA_TOOL_NAMESPACES}


def _system_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and messages[0].get("role") == "system":
        return str(messages[0].get("content") or "")
    return ""


def _offered_schemas(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return tool_schema_map(row.get("tools") or [])


def _parsed_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@lru_cache(maxsize=512)
def _static_entities(system_prompt: str) -> tuple[dict[str, Any], ...]:
    """Entities listed in the static context: names (aliases included), domain, areas."""
    _, header, rest = system_prompt.partition(STATIC_CONTEXT_HEADER)
    if not header:
        return ()
    block: list[str] = []
    for line in rest.split("\n"):
        if line.startswith(("- ", " ")):
            block.append(line)
        elif block:
            break
    entities = []
    for entry in yaml.safe_load("\n".join(block)) or []:
        names = [name.strip() for name in str(entry.get("names") or "").split(", ") if name.strip()]
        areas = [area.strip() for area in str(entry.get("areas") or "").split(", ") if area.strip()]
        entities.append({"names": names, "domain": entry.get("domain"), "areas": areas})
    return tuple(entities)


@lru_cache(maxsize=None)
def _domain_supports(domain: str, tool: str) -> bool | None:
    """True/False when the domain has registry capabilities; None when it has none (not judged)."""
    caps = [cap for cap in CAPABILITIES.values() if cap.domain == domain]
    if not caps:
        return None
    return any(
        op.tool_name == tool and op.support != SupportLevel.UNAVAILABLE
        for cap in caps
        for op in cap.operations
    )


def check_unsupported_calls(row: dict[str, Any], *, home: dict[str, Any] | None = None) -> str | None:
    """Reason the row's assistant tool calls are not servable, or None.

    The row carries no entity features, so support is judged per domain unless
    `home` is given; with `home`, a named entity found there is checked with
    `_entity_can_run` (features included).
    """
    schemas = _offered_schemas(row)
    domains_by_name: dict[str, set] = defaultdict(set)
    for entity in _static_entities(_system_prompt(row)):
        for name in entity["names"]:
            domains_by_name[name].add(entity["domain"])
    home_by_name = {e["name"]: e for e in (home or {}).get("entities", [])}
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if name not in schemas:
                return "unoffered_tool"
            arguments = _parsed_arguments(function.get("arguments"))
            if arguments is None:
                return "invalid_call_shape"
            reason = validate_tool_arguments(name, arguments, schemas)
            if reason:
                return reason
            bare = _BARE_TOOL.get(name, name)
            target = arguments.get("name")
            if target is None or bare in _TIMER_NAME_TOOLS:
                continue
            if not isinstance(target, str):
                return "invalid_call_shape"
            domains = domains_by_name.get(target)
            if not domains:
                return "unknown_canonical_entity"
            if home is not None and target in home_by_name:
                if not _entity_can_run(home_by_name[target], bare):
                    return "entity_lacks_capability"
            elif all(_domain_supports(domain, bare) is False for domain in domains):
                return "unsupported_operation"
    return None


def _label(row: dict[str, Any]) -> str:
    calls = sorted(
        (call["function"]["name"], json.dumps(_parsed_arguments(call["function"].get("arguments")), sort_keys=True))
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and message.get("train_on_turn")
        for call in message.get("tool_calls") or []
    )
    if calls:
        return json.dumps({"calls": calls})
    return json.dumps({"no_call": (row.get("metadata") or {}).get("no_action_reason")})


def _context_hash(row: dict[str, Any]) -> str:
    tools = sorted(tool["function"]["name"] for tool in row.get("tools") or [])
    return hashlib.sha256("\0".join([_system_prompt(row), *tools]).encode()).hexdigest()


def _command(row: dict[str, Any]) -> str:
    return "\n".join(
        _normalize_prompt(str(message.get("content") or ""))
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def check_label_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups that share a normalized command and model-visible context but disagree on the label."""
    groups: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups[(_command(row), _context_hash(row))][_label(row)].append(index)
    return [
        {"command": command, "labels": dict(labels)}
        for (command, _), labels in groups.items()
        if len(labels) > 1
    ]


@lru_cache(maxsize=1)
def _eval_index() -> dict[str, Any]:
    prompts = sorted(quality_eval_prompts())
    token_sets = [frozenset(prompt.split()) for prompt in prompts]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings[token].append(index)
    return {
        "sets": token_sets,
        "df": Counter({token: len(ids) for token, ids in postings.items()}),
        "postings": postings,
    }


def _jaccard_overlap(tokens: frozenset[str], index: dict[str, Any], threshold: float) -> bool:
    # Prefix filter: a pair with Jaccard >= t shares a token among the first
    # |U| - ceil(t|U|) + 1 tokens of U, ordered rarest-first in the eval set.
    ordered = sorted(tokens, key=lambda token: (index["df"].get(token, 0), token))
    prefix = len(ordered) - math.ceil(threshold * len(ordered)) + 1
    candidates: set[int] = set()
    for token in ordered[:prefix]:
        candidates.update(index["postings"].get(token, ()))
    return any(
        len(tokens & index["sets"][i]) / len(tokens | index["sets"][i]) >= threshold
        for i in candidates
    )


def check_fuzzy_eval_overlap(
    utterance: str,
    *,
    context: str = "",
    threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> bool:
    """Eval leakage: the utterance matches an eval prompt after normalization, or has token-set Jaccard >= threshold with one.

    `context` (the row's system prompt) is accepted for existing callers but not used for
    rejection. Masking entity names by context made generic shapes such as "turn off the <E>"
    match eval templates and rejected ordinary commands.
    """
    tokens = frozenset(_normalize_prompt(utterance).split())
    if not tokens:
        return False
    return _jaccard_overlap(tokens, _eval_index(), threshold)
