import random

from generators.stt_noise import apply_log_stt_noise, utterance_contains_target


def test_log_noise_reproduces_the_satellite_slips():
    seen = {}
    for seed in range(200):
        corrupted, kind = apply_log_stt_noise(
            "turn the living room lights on", random.Random(seed), target_names=["Living room lights"]
        )
        seen[kind] = corrupted
        assert utterance_contains_target(corrupted, "Living room lights")
    assert seen["log_living_livingium"] == "turn the livingium lights on"
    assert seen["log_living_little"] == "turn the little lights on"
    assert seen["log_wake_prefix"].split(" ", 1)[0] in {"kda", "koda"}


def test_little_only_folds_before_lights():
    assert not utterance_contains_target("turn on the little lamp", "Living room lamp")
