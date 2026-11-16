from __future__ import annotations

import re
from pathlib import Path

from generators.config import GeneratorConfig, default_allocations
from generators.pipeline import run_generation

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "real_home.json"
QUESTION = re.compile(r"^Did you mean (?P<candidates>.+)\?$")
CANDIDATE_SEPARATOR = re.compile(r", or the | or the |, the ")


def _clarify_candidates(question: str) -> list[str]:
    match = QUESTION.match(question)
    assert match, question
    body = match.group("candidates")
    assert body.startswith("the "), question
    return [part.removeprefix("the ") for part in CANDIDATE_SEPARATOR.split(body)]


def test_real_home_clarify_candidates_are_context_entities() -> None:
    result = run_generation(GeneratorConfig(
        count=600, seed=11, real_home_path=FIXTURE, real_home_rate=0.6,
        paraphrase_enabled=False, allocations=default_allocations(),
    ))
    assert any(row["metadata"]["real_home"] for row in result["rows"])
    rows = [
        row for row in result["rows"]
        if row["metadata"]["real_home"] and row["metadata"].get("family") == "clarify"
    ]
    for row in rows:
        system_prompt = row["messages"][0]["content"]
        question = row["messages"][-1]["content"]
        candidates = _clarify_candidates(question)
        assert len(candidates) >= 2, question
        for name in candidates:
            assert name in system_prompt, (name, question)
