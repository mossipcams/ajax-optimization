from __future__ import annotations

import random

from generators.capability_registry import CAPABILITIES
from generators.scenarios.core import pick_targeting


def test_clean_area_never_picks_floor_targeting() -> None:
    cap = CAPABILITIES["vacuums"]
    for seed in range(2000):
        rng = random.Random(seed)
        targeting = pick_targeting(cap, rng, "ordinary", operation="clean_area")
        assert targeting != "floor", seed


def test_start_can_still_pick_floor_targeting() -> None:
    cap = CAPABILITIES["vacuums"]
    picked = {
        pick_targeting(cap, random.Random(seed), "ordinary", operation="start")
        for seed in range(2000)
    }
    assert "floor" in picked
