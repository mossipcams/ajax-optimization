from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.gold import expected_no_action
from generators.row_generation import (
    _conversational_sentence,
    _garbled_device_fragment,
    _junk_sample,
    _junk_text,
    generate_row,
)

SEEDS = range(2000)
PROBE_SEEDS = range(3000)
TARGET_MIX = {"filler": 0.40, "conversation": 0.40, "garbled": 0.20}
MIX_TOLERANCE = 0.04
COMMAND_OVERLAP = re.compile(
    r"\b(play|plays|played|playing|turn|turns|turned|turning|pause|paused|pausing|stop|stopped|"
    r"next|skip|skipped|set|open|opened|close|closed|light|lights|fan|tv|show|game|music|"
    r"volume|timer|kids)\b"
)
DEVICE_NOUN = re.compile(
    r"\b(tv|lights|light|fan|speaker|switch|echo dot|lamp|door|volume|timer|music|show)\b"
)
LIVE_PROBES = (
    "play the tv", "resume the tv", "unpause the tv", "turn down the tv", "turned on the tv",
    "turned on the tv right and we are 4th right 4th right with that we are 4th right with that yeah yeah 4th round right everybody welcome to the bad sense kitchen no it is for adults it is for adults yeah hey you are welcome and",
    "turn down the cdda turn down the cd like what is the wave series it is the company that we now are w",
    "pause the kitchen echo dot", "turn down the desk echo dot", "mute the speaker", "next song",
    "Two qualities there.", "What time is it?",
    "Rocky. But just like you I need to do all that student teaching stuff.",
    "Oopsies did not have some data next weekend.",
    "That's your sleep square like how do you calculate it",
    "that really turned me off that foodyland.",
    "It's like the same little kids. You play with kid one time",
    "Turn off the kitchen light.",
    "But you're being safe away and... You've been ready in the meal?",
    "We'll turn on the living room TV.", "Check out the bigger of TV.",
    "five minute timer", "timer for ten minutes", "set a timer for 1 hour and 30 minutes",
    "set a 2 minute 30 second timer", "pause the living room timer",
    "add five minutes to the living room timer", "cancel the living room timer",
    "is anything on in the living room", "are the lights on in the kitchen",
    "set the air purifier to 50 percent", "turn on the switches in the living room",
    "what's the date today", "play some jazz on the kitchen echo dot",
    "set the kitchen lights to 59 percent", "set a timer for two hours", "pause the timer",
    "resume the timer", "how much time is left on the timer", "add an hour to the timer",
    "take an hour off the timer", "cancel the timer", "turn on the T V", "turn on the T.V.",
    "turn on the teevee", "is the T V on", "lower the tv volume", "turn the tv up",
    "turn down the sayso satellite", "mute the satellite", "pause the everywhere speaker",
    "I told her we would be there by six.",
    "The kids played outside all day and now they're exhausted.",
    "Honestly that show really turned me off after the second season.",
    "She paused for a second and then just kept talking.",
    "Did you see the game last night?", "turn on the living room light the uh the",
    "wait what was the hm tv", "what time is it", "what day is it",
    "set a timer for one hour and thirty minutes", "set a timer for an hour and a half",
    "timer for 2 minutes and 30 seconds", "set a 1 hour 15 minute timer",
    "set a timer for 20 minutes", "how much time is left on the living room timer",
    "cancel the timer in the living room", "which lights are on in the kitchen",
    "is anything on in the bedroom", "set the levoit fan speed to 30 percent", "is the tv on",
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split())


def _word_count(text: str) -> int:
    return len(text.split())


def _junk_row(seed: int):
    config = GeneratorConfig(count=1, seed=seed, token_budget=8192)
    rng = random.Random(seed)
    tracker = DuplicateTracker(near_limit=8)
    for attempt in range(60):
        slot = {
            "index": attempt,
            "family": "junk",
            "robustness": None,
            "capability": "lights",
            "operation": "turn_on",
            "tier": 1,
            "home_size": 16,
            "area_scenario": None,
        }
        row, _ = generate_row(
            slot, config, rng, excluded=set(), dup_tracker=tracker, attempt=attempt
        )
        if row is not None:
            return row
    return None


def test_junk_mix_matches_target_shares_over_seeds():
    kinds = Counter(_junk_sample(random.Random(seed))[0] for seed in SEEDS)
    total = len(SEEDS)
    for kind, share in TARGET_MIX.items():
        assert abs(kinds[kind] / total - share) <= MIX_TOLERANCE, (kind, kinds)


def test_conversational_sentences_are_distinct_and_speech_length():
    sentences = [_conversational_sentence(random.Random(seed)) for seed in SEEDS]
    assert len(set(sentences)) >= 1500
    assert all(6 <= _word_count(text) <= 25 for text in sentences), [
        text for text in sentences if not 6 <= _word_count(text) <= 25
    ][:5]


def test_conversational_sentences_use_command_overlap_words_in_speech():
    sentences = [_conversational_sentence(random.Random(seed)) for seed in SEEDS]
    overlapping = sum(bool(COMMAND_OVERLAP.search(text.lower())) for text in sentences)
    assert overlapping / len(sentences) >= 0.30


def test_garbled_fragments_name_a_device_without_an_action():
    fragments = [_garbled_device_fragment(random.Random(seed)) for seed in SEEDS]
    assert all(DEVICE_NOUN.search(text.lower()) for text in fragments)
    assert len(set(fragments)) >= 300


def test_junk_text_is_deterministic_for_a_fixed_seed():
    assert [_junk_text(random.Random(seed)) for seed in range(50)] == [
        _junk_text(random.Random(seed)) for seed in range(50)
    ]


def test_junk_text_never_reproduces_a_live_probe_utterance():
    probes = {_normalize(probe) for probe in LIVE_PROBES}
    texts = [_junk_text(random.Random(seed)) for seed in SEEDS]
    texts += [_conversational_sentence(random.Random(seed)) for seed in SEEDS]
    texts += [_garbled_device_fragment(random.Random(seed)) for seed in SEEDS]
    leaked = sorted({text for text in texts if _normalize(text) in probes})
    assert leaked == []


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _spaced(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def test_talk_and_garble_text_stays_away_from_live_probes_over_3000_seeds():
    probe_tokens = [(probe, _tokens(probe)) for probe in LIVE_PROBES]
    probe_spaced = [_spaced(probe) for probe in LIVE_PROBES]
    texts = [_conversational_sentence(random.Random(seed)) for seed in PROBE_SEEDS]
    texts += [_garbled_device_fragment(random.Random(seed)) for seed in PROBE_SEEDS]
    too_similar = []
    substrings = []
    for text in set(texts):
        tokens = _tokens(text)
        spaced = _spaced(text)
        for (probe, other), probe_text in zip(probe_tokens, probe_spaced):
            union = tokens | other
            if union and len(tokens & other) / len(union) >= 0.5:
                too_similar.append((text, probe))
            if spaced in probe_text or probe_text in spaced:
                substrings.append((text, probe))
    assert too_similar == []
    assert substrings == []


def test_junk_rows_stay_no_action_not_understood():
    expected = expected_no_action("not_understood")
    for seed in range(3):
        row = _junk_row(20260922 + seed)
        assert row is not None, f"no junk row generated for seed {seed}"
        assistant = [m for m in row["messages"] if m["role"] == "assistant"]
        assert assistant[-1]["content"] == "Sorry, I didn't catch that."
        assert not any(m.get("tool_calls") for m in row["messages"])
        assert row["metadata"]["family"] == "junk"
        assert row["metadata"]["no_action_reason"] == expected["response"] == "not_understood"
