import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module("counterfactual_probe", ROOT / "scripts" / "counterfactual_probe.py")


@pytest.fixture(scope="module")
def pairs():
    return probe.build_all_pairs()


def _first(pairs, kind):
    return next(pair for pair in pairs if pair.kind == kind)


def _names(entities):
    return [str(entity["name"]).casefold() for entity in entities]


def _tool_call(name, arguments):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


def _text(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_build_produces_every_kind(pairs):
    counts = probe.pair_counts(pairs)
    assert all(counts[kind] > 0 for kind in probe.PROBE_KINDS)


def test_device_absent_variant_removes_target_and_expects_refusal(pairs):
    pair = _first(pairs, "device_absent")
    target = pair.removed_entity
    assert target.casefold() in _names(load_original_home(pair))
    assert target.casefold() not in _names(pair.variant_home["entities"])
    assert pair.variant.household != pair.original.household
    assert pair.variant.expected["response_type"] == "refusal"
    assert pair.variant.expected["calls"] == []
    assert pair.variant.unavailable == {"kind": "device_absent", "capability": target, "removed_tools": []}


def test_tool_removed_variant_removes_only_the_called_tool(pairs):
    pair = _first(pairs, "tool_removed")
    assert pair.variant.household == pair.original.household
    assert pair.variant_home is None
    assert pair.variant.unavailable["removed_tools"] == [pair.removed_tool]
    assert pair.variant.expected["response_type"] == "refusal"


def test_clarify_resolved_variant_keeps_exactly_one_candidate_and_expects_action(pairs):
    pair = _first(pairs, "clarify_resolved")
    candidates = pair.original.expected["clarification"]["candidates"]
    present = [name for name in candidates if name.casefold() in _names(pair.variant_home["entities"])]
    assert present == [pair.remaining_entity]
    assert pair.variant.expected["response_type"] == "action"


def test_sticky_identical_output_is_not_an_appropriate_change(pairs):
    pair = _first(pairs, "device_absent")
    call = pair.original.expected["calls"][0]

    def sticky(messages, tools):
        return _tool_call(call["name"], call["arguments"])

    _, rows = probe.evaluate_pairs([pair], probe.InMemoryAdapter(sticky))
    row = rows[0]
    assert row["original_passed"] is True
    assert row["variant_passed"] is False
    assert row["identical_output"] is True
    assert row["sticky"] is True
    assert row["appropriate_change"] is False
    summary = probe.summarize_rows(rows)["overall"]
    assert summary["sticky_rate"] == 1.0
    assert summary["appropriate_change_rate"] == 0.0


def test_context_sensitive_change_is_an_appropriate_change(pairs):
    pair = _first(pairs, "device_absent")
    call = pair.original.expected["calls"][0]
    target = pair.removed_entity

    def contextual(messages, tools):
        if re.search(rf"names: {re.escape(target)}\s*$", messages[0]["content"], re.MULTILINE):
            return _tool_call(call["name"], call["arguments"])
        return _text("I can't find that device in your home.")

    _, rows = probe.evaluate_pairs([pair], probe.InMemoryAdapter(contextual))
    row = rows[0]
    assert row["original_passed"] is True
    assert row["variant_passed"] is True
    assert row["identical_output"] is False
    assert row["appropriate_change"] is True
    assert row["sticky"] is False
    summary = probe.summarize_rows(rows)
    assert summary["overall"]["appropriate_change_rate"] == 1.0
    assert summary["kinds"]["device_absent"]["sticky_rate"] == 0.0


def load_original_home(pair):
    from evals.cases import load_home

    return load_home(pair.original.household)["entities"]
