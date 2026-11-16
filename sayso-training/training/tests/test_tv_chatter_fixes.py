from __future__ import annotations

import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators import row_generation  # noqa: E402
from generators.config import GeneratorConfig  # noqa: E402
from generators.deduplication import DuplicateTracker  # noqa: E402

SEED = 7


def _ordinary_slot(index: int) -> dict:
    return {
        "index": index,
        "family": "ordinary",
        "robustness": "ordinary",
        "capability": "lights",
        "operation": "turn_on",
        "tier": 1,
        "home_size": 16,
        "area_scenario": None,
        "grounding": False,
    }


def _generate_one(index: int, count: int) -> tuple[dict | None, str | None]:
    config = GeneratorConfig(count=count, seed=SEED)
    return row_generation.generate_row(
        _ordinary_slot(index),
        config,
        random.Random(index),
        excluded=set(),
        dup_tracker=DuplicateTracker(),
    )


@lru_cache(maxsize=1)
def _ordinary_rows() -> tuple[dict, ...]:
    tracker = DuplicateTracker()
    config = GeneratorConfig(count=1200, seed=SEED)
    rows = []
    for index in range(1200):
        row, reason = row_generation.generate_row(
            _ordinary_slot(index),
            config,
            random.Random(index),
            excluded=set(),
            dup_tracker=tracker,
        )
        # Rejected candidates are skipped, not asserted on: the rate and the
        # tool-call checks are about the rows that were actually accepted.
        if row is not None and reason is None:
            rows.append(row)
    return tuple(rows)


def _single_turn_action_rows(rows) -> list[dict]:
    return [
        row
        for row in rows
        if sum(m["role"] == "user" for m in row["messages"]) == 1
        and any(m.get("tool_calls") for m in row["messages"])
    ]


def _wrapped_rows(rows) -> list[dict]:
    return [r for r in rows if (r.get("metadata") or {}).get("chatter_wrap")]


def test_generated_junk_utterances_have_no_command_verbs():
    for seed in range(3000):
        text = row_generation._junk_utterance(random.Random(seed), seed)
        assert text.strip()
        assert row_generation.talk_has_command_verb(text) is None, text


def test_junk_filter_allows_devices_but_blocks_verbs():
    assert row_generation.talk_has_command_verb("I'm past the TV.") is None
    assert row_generation.talk_text_violation("I'm past the TV.") is not None
    assert row_generation.talk_has_command_verb("did grandpa turn off the television") is not None
    assert row_generation.talk_text_violation("fantastic settled closet") is None
    assert row_generation.talk_has_command_verb("fantastic settled closet") is None


def test_chatter_fillers_have_no_device_or_command_words():
    fillers = (
        row_generation._CHATTER_PREFIX_FILLERS + row_generation._CHATTER_SUFFIX_FILLERS
    )
    assert fillers
    for filler in fillers:
        assert row_generation.talk_text_violation(filler) is None
    for seed in range(600):
        rng = random.Random(seed)
        for prefix in (True, False):
            filler = row_generation._chatter_filler(rng, prefix=prefix)
            assert filler.strip()
            assert row_generation.talk_text_violation(filler) is None


def test_wrap_preserves_command_text_and_placement_mix():
    placements: Counter[str] = Counter()
    for seed in range(1200):
        wrapped, placement = row_generation.wrap_utterance_with_chatter(
            "turn off the kitchen lights", random.Random(seed)
        )
        assert "turn off the kitchen lights" in wrapped
        assert placement in {"prefix", "suffix", "both"}
        placements[placement] += 1
    total = sum(placements.values())
    for name, low, high in (
        ("prefix", 0.30, 0.50),
        ("suffix", 0.30, 0.50),
        ("both", 0.12, 0.28),
    ):
        share = placements[name] / total
        assert low <= share <= high


def test_wrapped_rows_keep_original_tool_calls():
    rows = _ordinary_rows()
    actions = _single_turn_action_rows(rows)
    wrapped = _wrapped_rows(actions)
    assert len(wrapped) > 5
    for row in wrapped:
        with_calls = [m for m in row["messages"] if m.get("tool_calls")]
        assert with_calls, "wrapped action row must keep its tool call"
        names = {c["function"]["name"] for c in with_calls[0]["tool_calls"]}
        assert {name.split("__")[-1] for name in names} == {"HassTurnOn"}


def test_wrapped_rate_is_within_one_to_five_percent():
    actions = _single_turn_action_rows(_ordinary_rows())
    assert actions
    wrapped = _wrapped_rows(actions)
    rate = len(wrapped) / len(actions)
    assert 0.01 <= rate <= 0.05


def test_generation_is_deterministic_per_seed():
    compared = 0
    for seed in range(24):
        first, first_reason = _generate_one(seed, 24)
        second, second_reason = _generate_one(seed, 24)
        assert first_reason == second_reason
        if first is None or second is None:
            continue
        compared += 1
        assert first["messages"] == second["messages"]
    assert compared > 0
