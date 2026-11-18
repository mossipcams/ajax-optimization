from __future__ import annotations

import json

from generators.config import GeneratorConfig, default_allocations
from generators.coverage import bare_tool_name
from generators.pipeline import run_generation


def _area_live_context_args(row: dict) -> list[dict]:
    args = []
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            if bare_tool_name(call["function"]["name"]) != "GetLiveContext":
                continue
            arguments = json.loads(call["function"]["arguments"])
            if "area" in arguments:
                args.append(arguments)
    return args


def test_area_query_rows_emit_area_live_context_with_summary() -> None:
    result = run_generation(GeneratorConfig(
        count=1500, seed=7, paraphrase_enabled=False, allocations=default_allocations(),
    ))
    rows = [row for row in result["rows"] if _area_live_context_args(row)]
    assert len(rows) >= 10, len(rows)
    for row in rows:
        system_prompt = row["messages"][0]["content"]
        for arguments in _area_live_context_args(row):
            assert arguments["area"] in system_prompt, (arguments, system_prompt)
        final = row["messages"][-1]
        assert final["role"] == "assistant", final
        assert final["content"].strip(), row["messages"]
        assert "unknown" not in final["content"], final["content"]


def test_area_live_context_domains_are_phrasable() -> None:
    from generators.utterances import _AREA_STATUS_WORDS

    result = run_generation(GeneratorConfig(
        count=1500, seed=7, paraphrase_enabled=False, allocations=default_allocations(),
    ))
    for row in result["rows"]:
        for arguments in _area_live_context_args(row):
            if "domain" in arguments:
                assert arguments["domain"] in _AREA_STATUS_WORDS, (arguments, row["messages"][0]["content"])
