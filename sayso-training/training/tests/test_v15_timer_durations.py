import random
import re

from generators.ohf import render_call
from generators.stt_noise import apply_stt_noise
from generators.tools import build_start_timer
from generators.utterances import _HALF_DURATION_PHRASES, _phrase_for_call

COMPOUND_SHAPES = {
    "hours_minutes": {"hours", "minutes"},
    "minutes_seconds": {"minutes", "seconds"},
}

FORBIDDEN_DURATION = (
    r"\d+ (hours|minutes|seconds) \d+-(hour|minute|second)",
    r"\d+-(hours|minutes|seconds)\b",
    r"\d+ (hours|minutes|seconds)\b.*\btimer\b",
)

HALF_PHRASES = ("an hour and a half", "half an hour", "a minute and a half")

COMPOUND_CALLS = [
    ({"hours": 2, "minutes": 40}, ("2", "40")),
    ({"minutes": 2, "seconds": 30}, ("2", "30")),
    ({"hours": 1, "minutes": 15}, ("1", "15")),
]

HALF_SPEC = {
    "an hour and a half": {"hours": 1, "minutes": 30},
    "half an hour": {"minutes": 30},
    "a minute and a half": {"minutes": 1, "seconds": 30},
}


def _shape(arguments):
    keys = {key for key in arguments if key != "name"}
    for shape, units in COMPOUND_SHAPES.items():
        if keys == units:
            return shape
    return "single" if len(keys) == 1 else "other"


def test_build_start_timer_emits_three_shapes_in_target_shares():
    counts = {"single": 0, "hours_minutes": 0, "minutes_seconds": 0, "other": 0}
    for seed in range(500):
        arguments = build_start_timer(random.Random(seed))["arguments"]
        counts[_shape(arguments)] += 1
        if _shape(arguments) == "hours_minutes":
            assert 1 <= arguments["hours"] <= 3
            assert 1 <= arguments["minutes"] <= 59
        if _shape(arguments) == "minutes_seconds":
            assert 1 <= arguments["minutes"] <= 59
            assert 5 <= arguments["seconds"] <= 55
    assert counts["other"] == 0
    assert abs(counts["single"] / 500 - 0.60) < 0.07
    assert abs(counts["hours_minutes"] / 500 - 0.25) < 0.07
    assert abs(counts["minutes_seconds"] / 500 - 0.15) < 0.07


def test_build_start_timer_half_values_in_target_shares():
    calls = [build_start_timer(random.Random(seed))["arguments"] for seed in range(2000)]
    hours_minutes = [a for a in calls if _shape(a) == "hours_minutes"]
    minutes_seconds = [a for a in calls if _shape(a) == "minutes_seconds"]
    singles = [a for a in calls if _shape(a) == "single" and "minutes" in a]
    assert 0.25 <= sum(a["minutes"] == 30 for a in hours_minutes) / len(hours_minutes) <= 0.37
    assert 0.25 <= sum(a["seconds"] == 30 for a in minutes_seconds) / len(minutes_seconds) <= 0.37
    assert 0.07 <= sum(a["minutes"] == 30 for a in singles) / len(singles) <= 0.15


def test_build_start_timer_is_deterministic_and_keeps_name():
    first = build_start_timer(random.Random(7), name="pasta")
    second = build_start_timer(random.Random(7), name="pasta")
    assert first == second
    assert first["name"] == "HassStartTimer"
    assert first["arguments"]["name"] == "pasta"


def test_compound_durations_render_grammatically_through_ohf_and_fallback():
    for arguments, numbers in COMPOUND_CALLS:
        call = {"name": "HassStartTimer", "arguments": arguments}
        for seed in range(200):
            for text in (render_call(call, "", seed), _phrase_for_call("", call, str(seed))):
                assert text is not None
                for pattern in FORBIDDEN_DURATION:
                    assert not re.search(pattern, text), text
                for number in numbers:
                    assert re.search(rf"\b{number}\b", text), text


def test_adjective_durations_use_hyphen_and_spaced_forms():
    call = {"name": "HassStartTimer", "arguments": {"hours": 1, "minutes": 15}}
    texts = [render_call(call, "", seed) for seed in range(200)]
    assert any(re.search(r"\ba 1-hour 15-minute timer\b", text) for text in texts), texts
    assert any(re.search(r"\ba 1 hour 15 minute timer\b", text) for text in texts), texts


def test_half_phrases_are_at_least_three_percent_of_start_timer_phrasings():
    hits = 0
    for seed in range(2000):
        call = build_start_timer(random.Random(seed))
        text = _phrase_for_call("", call, str(seed))
        if any(phrase in text for phrase in HALF_PHRASES):
            hits += 1
    assert hits / 2000 >= 0.03


def test_spelled_out_number_words_come_from_existing_stt_noise():
    call = {"name": "HassStartTimer", "arguments": {"hours": 1, "minutes": 30}}
    kinds = set()
    for seed in range(200):
        text = render_call(call, "", seed).replace("-", " ")
        if "half" in text:
            continue
        noisy, kind = apply_stt_noise(text, random.Random(seed))
        if kind == "number_variant":
            kinds.add(kind)
            assert re.search(r"\bone hour\b", noisy), noisy
            assert re.search(r"\bthirty minutes?\b", noisy), noisy
    assert kinds == {"number_variant"}


def test_half_duration_phrases_map_as_specified():
    mapping = {phrase: shape for shape, phrases in _HALF_DURATION_PHRASES for phrase in phrases}
    assert mapping == HALF_SPEC


def test_half_duration_phrases_render_for_their_shape_only():
    for phrase, arguments in HALF_SPEC.items():
        call = {"name": "HassStartTimer", "arguments": arguments}
        seen = set()
        for seed in range(200):
            text = _phrase_for_call("", call, str(seed))
            assert text is not None
            if phrase in text:
                seen.add(text)
                assert re.search(r"^\w+ (?:(?:a|the|my) )?timer for ", text), text
        assert seen, phrase
        for other_phrase in set(HALF_SPEC) - {phrase}:
            for seed in range(20):
                assert other_phrase not in _phrase_for_call("", call, str(seed))
