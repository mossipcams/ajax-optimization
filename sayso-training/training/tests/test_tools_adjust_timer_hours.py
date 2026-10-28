import random

from generators.tools import build_adjust_timer

UNITS = {"seconds", "minutes", "hours"}


def test_build_adjust_timer_emits_all_units_one_per_call():
    seen = set()
    for seed in range(500):
        rng = random.Random(seed)
        tool = build_adjust_timer(rng, direction="increase")
        args = tool["arguments"]
        unit_keys = [key for key in args if key in UNITS]
        assert len(unit_keys) == 1
        seen.add(unit_keys[0])
    assert seen == UNITS
