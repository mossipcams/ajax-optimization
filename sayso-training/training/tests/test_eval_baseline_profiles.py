import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_baseline = _load_module("eval_baseline", ROOT / "scripts" / "eval_baseline.py")

V6_CANARY_SUMMARY = {
    "total": 24,
    "passed": 7,
    "overall_pass_rate": 0.2917,
    "category_pass_rate": {},
    "malformed_output_rate": 0.0,
    "unknown_tool_rate": 0.1667,
    "invalid_argument_rate": 0.1667,
    "flags": {
        "action_on_ambiguity": 3,
        "tool_call_on_unavailable": 2,
        "wrong_domain": 2,
    },
}


def _baseline():
    return json.loads((ROOT / "configs" / "training_baseline.json").read_text(encoding="utf-8"))["eval"]


def test_canary_profile_passes_v6_canary_summary():
    assert eval_baseline.baseline_failures(V6_CANARY_SUMMARY, _baseline(), "canary") == []


def test_canary_profile_fails_collapsed_summary():
    collapsed = dict(V6_CANARY_SUMMARY)
    collapsed["overall_pass_rate"] = 0.05
    collapsed["malformed_output_rate"] = 0.5
    failures = eval_baseline.baseline_failures(collapsed, _baseline(), "canary")
    assert any("overall_pass_rate" in f for f in failures)


def test_canary_profile_flags_respect_slack():
    drifted = dict(V6_CANARY_SUMMARY)
    drifted["flags"] = dict(V6_CANARY_SUMMARY["flags"])
    drifted["flags"]["wrong_domain"] = 3
    assert eval_baseline.baseline_failures(drifted, _baseline(), "canary") == []
    broken = dict(drifted)
    broken["flags"] = dict(drifted["flags"])
    broken["flags"]["wrong_domain"] = 5
    failures = eval_baseline.baseline_failures(broken, _baseline(), "canary")
    assert any("wrong_domain" in f for f in failures)


def test_checkpoint_profile_still_fails_v6_canary_summary():
    failures = eval_baseline.baseline_failures(V6_CANARY_SUMMARY, _baseline(), "checkpoint")
    assert failures
    assert all(f.startswith("smoke ") for f in failures)


def test_smoke_baseline_values_unchanged():
    baseline = _baseline()
    assert baseline["smoke"]["overall_pass_rate"] == 0.7083
    assert baseline["max_drop"] == 0.12


def test_canary_profile_configuration_present():
    canary = _baseline()["canary"]
    assert isinstance(canary["note"], str) and canary["note"].strip()
    assert canary["source_run"] == "20261016T000918Z-7c1ef4ea"
    assert canary["overall_pass_rate"] == 0.2917
    assert canary["tool_validity"] == 0.6666
    assert canary["flag_slack"] == 1
    assert canary["flags"] == {"action_on_ambiguity": 3, "tool_call_on_unavailable": 2, "wrong_domain": 2}
    gates = canary["gates"]
    assert gates["min_overall_pass_rate"] == 0.2
    assert gates["category_min_pass_rate"] == {}
    assert gates["max_rates"] == {
        "malformed_output_rate": 0.05,
        "unknown_tool_rate": 0.25,
        "invalid_argument_rate": 0.25,
    }
    assert gates["zero_tolerance_flags"] == ["transport_error", "malformed_treated_as_abstention"]
