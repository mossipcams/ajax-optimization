from __future__ import annotations

from typing import Any


def baseline_failures(summary: dict[str, Any], baseline: dict[str, Any], profile: str) -> list[str]:
    failures: list[str] = []
    if profile == "canary":
        accepted = baseline["canary"]
        tolerance = baseline["max_drop"]
        overall = summary["overall_pass_rate"]
        if overall < accepted["overall_pass_rate"] - tolerance:
            failures.append(
                f"canary overall_pass_rate: {overall} < baseline {accepted['overall_pass_rate']:.4f} - {tolerance:.2f}"
            )
        tool_validity = (
            1 - summary["malformed_output_rate"] - summary["unknown_tool_rate"] - summary["invalid_argument_rate"]
        )
        if tool_validity < accepted["tool_validity"] - tolerance:
            failures.append(
                f"canary tool_validity: {tool_validity:.4f} < baseline {accepted['tool_validity']:.4f} - {tolerance:.2f}"
            )
        slack = accepted.get("flag_slack", 0)
        for flag, old_count in accepted.get("flags", {}).items():
            count = summary["flags"].get(flag, 0)
            if count > old_count + slack:
                failures.append(f"canary {flag}: {count} > baseline {old_count} + slack {slack}")
        return failures
    accepted = baseline["smoke"]
    tolerance = baseline["max_drop"]
    rates = {"overall_pass_rate": summary["overall_pass_rate"], **summary["category_pass_rate"]}
    for metric, old_rate in {
        "overall_pass_rate": accepted["overall_pass_rate"],
        **accepted["category_pass_rate"],
    }.items():
        actual = rates.get(metric)
        if actual is None or actual < old_rate - tolerance:
            failures.append(f"smoke {metric}: {actual} < baseline {old_rate:.4f} - {tolerance:.2f}")
    tool_validity = 1 - summary["malformed_output_rate"] - summary["unknown_tool_rate"] - summary["invalid_argument_rate"]
    if tool_validity < accepted["tool_validity"] - tolerance:
        failures.append(
            f"smoke tool_validity: {tool_validity:.4f} < baseline {accepted['tool_validity']:.4f} - {tolerance:.2f}"
        )
    for flag, old_count in accepted.get("flags", {}).items():
        count = summary["flags"].get(flag, 0)
        if count > old_count:
            failures.append(f"smoke {flag}: {count} > baseline {old_count}")
    return failures
