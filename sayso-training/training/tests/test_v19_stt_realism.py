from __future__ import annotations

import random

from generators.stt_noise import (
    WAKE_PREFIX_RATE,
    _WAKE_PREFIXES,
    apply_log_stt_noise,
    apply_stt_noise,
    utterance_contains_target,
)

DEVICE_NOUN_CASES = {
    "homophone_lamp": ("turn on the hallway lamp", ["hallway lamp"]),
    "homophone_switch": ("turn off the porch switch", ["porch switch"]),
    "vowel_speaker": ("mute the office speaker", ["office speaker"]),
    "phonetic_vacuum": ("start the bedroom vacuum", ["bedroom vacuum"]),
    "dropped_letter_heater": ("set the patio heater to 70", ["patio heater"]),
    "transposed_thermostat": ("set the basement thermostat to 68", ["basement thermostat"]),
    "phonetic_plug": ("unplug the kitchen plug", ["kitchen plug"]),
}


def test_new_device_noun_corruptions_appear() -> None:
    seen: set[str] = set()
    for seed in range(300):
        rng = random.Random(seed)
        for kind, (source, names) in DEVICE_NOUN_CASES.items():
            corrupted, got = apply_stt_noise(source, rng, target_names=names)
            if got == kind:
                seen.add(kind)
                assert corrupted != source
                assert utterance_contains_target(corrupted, names[0])
    for kind in DEVICE_NOUN_CASES:
        assert kind in seen, f"never drew {kind}"


def test_device_noun_corruption_keeps_verbs() -> None:
    switch_noun_seen = False
    for seed in range(300):
        rng = random.Random(seed)
        corrupted, kind = apply_stt_noise(
            "switch on the porch switch", rng, target_names=["porch switch"]
        )
        assert "swish on" not in corrupted.casefold()
        assert corrupted.casefold().startswith("switch on ")
        if kind == "homophone_switch":
            switch_noun_seen = True
            assert corrupted == "switch on the porch swish"
        vacuum, _kind = apply_stt_noise("vacuum the kitchen", random.Random(seed))
        assert vacuum.casefold().startswith("vacuum ")
    assert switch_noun_seen


def test_device_noun_corruption_never_removes_target_name() -> None:
    for seed in range(150):
        for source, names in DEVICE_NOUN_CASES.values():
            corrupted, kind = apply_stt_noise(source, random.Random(seed), target_names=names)
            if kind:
                assert utterance_contains_target(corrupted, names[0])


def test_log_stt_wake_prefix_rate() -> None:
    source = "hallway lamp on"
    draws = 1000
    hits = 0
    prefixes: set[str] = set()
    for seed in range(draws):
        corrupted, _kind = apply_log_stt_noise(source, random.Random(seed), target_names=["hallway lamp"])
        assert utterance_contains_target(corrupted, "hallway lamp")
        first = corrupted.split(" ", 1)[0]
        if first in _WAKE_PREFIXES:
            hits += 1
            prefixes.add(first)
    fraction = hits / draws
    assert 0.20 <= fraction <= 0.30, f"wake prefix rate {fraction}"
    assert prefixes and prefixes <= set(_WAKE_PREFIXES)
