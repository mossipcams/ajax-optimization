from __future__ import annotations

import copy
import json
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators import pipeline
from generators.config import GeneratorConfig
from generators.context import STATIC_CONTEXT_HEADER
from generators.deduplication import DuplicateTracker
from generators.grounding import CONTRAST_GROUP_SIZE
from generators.pipeline import run_generation
from generators.row_generation import generate_row
from generators.tools import namespaced_tool_name, production_catalog
from generators.validation import (
    check_fuzzy_eval_overlap,
    check_label_conflicts,
    check_unsupported_calls,
    quality_eval_prompts,
)

DEN_TV = "Den Smart Bedroom TV"
DEN_LIGHT = "Den Ceiling Lights"

HOME = {
    "entities": [
        {"name": DEN_LIGHT, "domain": "light", "capability": "lights", "capabilities": [],
         "area": "Den", "entity_id": "light.den_ceiling_lights"},
        {"name": "Master Bedroom Echo Dot", "domain": "media_player", "capability": "media_players",
         "capabilities": ["on", "off", "mute", "volume"], "area": "Master Bedroom",
         "entity_id": "media_player.master_bedroom_echo_dot"},
    ]
}


def _system(entities: list[dict]) -> str:
    block = yaml.dump(entities, default_flow_style=False).rstrip()
    return "\n".join(["You are SaySo, a local Home Assistant voice agent.", STATIC_CONTEXT_HEADER, block])


def _context_entities(home: dict) -> list[dict]:
    return [
        {"names": e["name"], "domain": e["domain"], "areas": e["area"]}
        for e in home["entities"]
    ]


@lru_cache(maxsize=1)
def _tools():
    return production_catalog(HOME)


def _row(calls: list[tuple[str, dict]], *, command: str = "turn on the den light",
         system: str | None = None, tools=None, label_reason: str | None = None) -> dict:
    system = system or _system(_context_entities(HOME))
    assistant = {"role": "assistant", "content": "", "train_on_turn": True}
    if calls:
        assistant["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for i, (name, arguments) in enumerate(calls)
        ]
    else:
        assistant = {"role": "assistant", "content": "Did you mean the Den light or the Den lamp?",
                     "train_on_turn": True}
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": command, "train_on_turn": False},
            assistant,
        ],
        "tools": tools if tools is not None else _tools(),
        "metadata": {"no_action_reason": label_reason},
    }


TURN_ON_LIGHT = ("intent__HassTurnOn", {"name": DEN_LIGHT, "domain": ["light"]})
MUTE_ECHO = ("media_player__HassMediaPlayerMute", {"name": "Master Bedroom Echo Dot"})


def _generated(family: str, count: int, start: int = 0) -> list[dict]:
    config = GeneratorConfig(count=4, seed=7)
    tracker = DuplicateTracker()
    rows: list[dict] = []
    index = start
    while len(rows) < count:
        slot = {"index": index, "family": family, "robustness": "ambiguity", "capability": "lights",
                "operation": "turn_on", "tier": 1, "home_size": 32}
        row, _ = generate_row(slot, config, random.Random(index), excluded=set(), dup_tracker=tracker)
        if row is not None:
            rows.append(row)
        index += 1
    return rows


def _trained_calls(row: dict) -> str:
    return json.dumps([
        (call["function"]["name"], call["function"]["arguments"])
        for m in row["messages"] if m["role"] == "assistant" and m.get("train_on_turn")
        for call in m.get("tool_calls") or []
    ])


def _set_calls(row: dict, calls: list[dict]) -> dict:
    assistant = next(m for m in row["messages"] if m["role"] == "assistant" and m.get("train_on_turn"))
    assistant["tool_calls"] = calls
    return row


# --- label conflicts -------------------------------------------------------

def test_conflicting_pair_is_caught():
    acting = _row([TURN_ON_LIGHT])
    asking = _row([], label_reason="clarify")
    conflicts = check_label_conflicts([acting, asking])
    assert len(conflicts) == 1
    assert sorted(i for rows in conflicts[0]["labels"].values() for i in rows) == [0, 1]
    assert len(conflicts[0]["labels"]) == 2


def test_same_command_and_context_with_different_arguments_conflicts():
    other = ("intent__HassTurnOn", {"name": "Master Bedroom Echo Dot", "domain": ["media_player"]})
    assert len(check_label_conflicts([_row([TURN_ON_LIGHT]), _row([other])])) == 1


def test_identical_labels_do_not_conflict():
    assert check_label_conflicts([_row([TURN_ON_LIGHT]), _row([TURN_ON_LIGHT])]) == []


def test_proper_contrast_pair_is_not_a_conflict():
    # Same command, one fact changed in the context: a second light is removed.
    with_light = _row([TURN_ON_LIGHT])
    without_light = _row([], label_reason="unsupported",
                         system=_system(_context_entities({"entities": HOME["entities"][1:]})))
    assert check_label_conflicts([with_light, without_light]) == []


def test_generated_contrast_group_is_not_a_conflict():
    rows = _generated("contrast", CONTRAST_GROUP_SIZE)
    assert len({row["metadata"]["contrast_group"] for row in rows}) == 1
    assert len({row["messages"][0]["content"] for row in rows}) > 1
    assert len({_trained_calls(row) for row in rows}) > 1
    assert check_label_conflicts(rows) == []


# --- unsupported calls -----------------------------------------------------

def test_hand_built_valid_row_passes():
    assert check_unsupported_calls(_row([TURN_ON_LIGHT])) is None
    assert check_unsupported_calls(_row([MUTE_ECHO])) is None


def test_generated_rows_pass():
    for row in _generated("ordinary", 12):
        assert check_unsupported_calls(row) is None, row["metadata"]["candidate_id"]


def test_off_schema_call_is_caught():
    row = _set_calls(_row([TURN_ON_LIGHT]), [{
        "id": "c", "type": "function",
        "function": {"name": "intent__HassTurnOn",
                     "arguments": json.dumps({"name": DEN_LIGHT, "bogus": 1})},
    }])
    assert check_unsupported_calls(row) == "extra_argument"


def test_unoffered_tool_is_caught():
    offered = {tool["function"]["name"] for tool in _tools()}
    name = "intent__HassSetAlarm"
    assert name not in offered
    row = _row([(name, {"name": DEN_LIGHT})])
    assert check_unsupported_calls(row) == "unoffered_tool"


def test_unsupported_entity_is_caught():
    # Muting a light: HassMediaPlayerMute is offered, but no light supports it.
    row = _row([("media_player__HassMediaPlayerMute", {"name": DEN_LIGHT})])
    assert check_unsupported_calls(row) == "unsupported_operation"


def test_call_to_name_outside_the_context_is_caught():
    row = _row([("intent__HassTurnOn", {"name": "Invented Lamp"})])
    assert check_unsupported_calls(row) == "unknown_canonical_entity"


def test_entity_features_are_checked_when_the_home_is_given():
    # The media player lacks the mute feature, so the home-level check refuses it.
    row = _row([MUTE_ECHO])
    home = {"entities": [{**HOME["entities"][1], "capabilities": ["on"]}]}
    assert check_unsupported_calls(row) is None
    assert check_unsupported_calls(row, home=home) == "entity_lacks_capability"


# --- fuzzy train/eval overlap ----------------------------------------------

def _eval_prompt(min_tokens: int) -> str:
    return next(p for p in sorted(quality_eval_prompts()) if len(p.split()) >= min_tokens)


def test_fuzzy_overlap_catches_reordered_eval_prompt():
    prompt = _eval_prompt(4)
    reordered = " ".join(reversed(prompt.split()))
    assert check_fuzzy_eval_overlap(reordered)


def test_fuzzy_overlap_catches_near_eval_prompt():
    prompt = _eval_prompt(5)
    assert check_fuzzy_eval_overlap(prompt + " ok")


def test_fuzzy_overlap_catches_a_device_name_added_to_an_eval_command():
    # Jaccard: {turn, on, the, den, tv} vs {turn, on, the, tv} is 4/5 = 0.8.
    assert "turn on the tv" in quality_eval_prompts()
    assert check_fuzzy_eval_overlap("Turn on the Den TV")


def test_fuzzy_overlap_does_not_ban_a_generic_command_shape():
    # Jaccard: {turn, on, the, den, smart, bedroom, tv} vs {turn, on, the, bedroom, tv} is 5/7.
    assert "turn on the bedroom tv" in quality_eval_prompts()
    assert not check_fuzzy_eval_overlap("Turn on the Den Smart Bedroom TV")


def test_fuzzy_overlap_ignores_entity_names_in_the_context():
    # Masking by context used to turn "turn off the <E>" into an eval template; names no longer matter.
    assert "turn off the kitchen light" in quality_eval_prompts()
    context = _system([{"names": "Kitchen Lamp, Porch Light", "domain": "light", "areas": "Kitchen"}])
    assert not check_fuzzy_eval_overlap("turn off the Kitchen Lamp", context=context)
    assert not check_fuzzy_eval_overlap("switch off the Porch Light", context=context)
    assert not check_fuzzy_eval_overlap("Turn on the Lounge Screen in the Living Room",
                                        context=_system([{"names": "Lounge Screen, TV",
                                                          "domain": "media_player",
                                                          "areas": "Living Room"}]))


def test_fuzzy_overlap_does_not_ban_generic_device_commands_with_different_names():
    assert not check_fuzzy_eval_overlap("turn off the Kitchen Lamp")
    assert not check_fuzzy_eval_overlap("switch off the Porch Light")


def test_fuzzy_overlap_does_not_flag_an_unrelated_command():
    assert not check_fuzzy_eval_overlap("set a timer for ten minutes in the kitchen")
    assert not check_fuzzy_eval_overlap("what is the weather like in lisbon tomorrow")


def test_fuzzy_overlap_screens_5k_utterances_quickly():
    verbs = ["turn on", "switch off", "mute", "set", "open", "lock", "start", "pause"]
    nouns = ["den light", "bedroom tv", "kitchen fan", "hallway lamp", "garage door", "timer"]
    rng = random.Random(3)
    utterances = [
        f"{rng.choice(verbs)} the {rng.choice(nouns)} {rng.randrange(1000)} "
        f"{rng.choice(['please', 'now', 'again', 'later'])}"
        for _ in range(5000)
    ]
    context = _system(_context_entities(HOME))
    started = time.perf_counter()
    flagged = sum(check_fuzzy_eval_overlap(u, context=context) for u in utterances)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, elapsed
    assert flagged < len(utterances)


# --- pipeline wiring -------------------------------------------------------

def _pipeline_cfg(count: int) -> GeneratorConfig:
    config = GeneratorConfig.from_yaml(
        TRAINING_ROOT / "configs" / "generation" / "smoke.yaml", repo_root=TRAINING_ROOT.parent
    )
    config.count = count
    config.seed = 99
    config.area_distribution_path = None
    return config


def test_pipeline_rejects_row_with_unoffered_tool_and_retries(monkeypatch):
    real_generate_row = pipeline.generate_row
    tampered: list[dict] = []

    def tamper_first_call(slot, config, rng, **kwargs):
        row, reason = real_generate_row(slot, config, rng, **kwargs)
        if row is None or tampered or not any(
            m.get("tool_calls") and m.get("train_on_turn") for m in row["messages"]
        ):
            return row, reason
        row = copy.deepcopy(row)
        for message in row["messages"]:
            if message.get("train_on_turn"):
                for call in message.get("tool_calls") or []:
                    call["function"]["name"] = "NoSuchTool"
        tampered.append(row)
        return row, None

    monkeypatch.setattr(pipeline, "generate_row", tamper_first_call)
    result = run_generation(_pipeline_cfg(60))
    assert tampered
    assert result["stats"]["rejection_reasons"]["unsupported_call"] >= 1
    assert len(result["rows"]) == 60
    assert all("NoSuchTool" not in json.dumps(row) for row in result["rows"])


def test_pipeline_raises_on_forced_conflicting_pair(monkeypatch):
    real_generate_row = pipeline.generate_row
    original: list[dict] = []

    def force_conflicting_twin(slot, config, rng, **kwargs):
        row, reason = real_generate_row(slot, config, rng, **kwargs)
        if row is None:
            return row, reason
        if not original:
            if any(m.get("tool_calls") for m in row["messages"]):
                original.append(copy.deepcopy(row))
            return row, reason
        if len(original) == 1:
            # Same command and same model-visible context, but the call is dropped: a different label.
            twin = copy.deepcopy(original[0])
            for message in twin["messages"]:
                message.pop("tool_calls", None)
            original.append(twin)
            return twin, None
        return row, reason

    monkeypatch.setattr(pipeline, "generate_row", force_conflicting_twin)
    with pytest.raises(RuntimeError, match="disagree on the label"):
        run_generation(_pipeline_cfg(60))


def _user_row(utterance: str, *, real_home: bool) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are SaySo, a local Home Assistant voice agent."},
            {"role": "user", "content": utterance},
        ],
        "metadata": {"real_home": real_home},
    }


def test_pipeline_fuzzy_screen_spares_real_home_but_not_synthetic_rows():
    prompt = _eval_prompt(6)
    near = f"{prompt} now"  # not an exact eval match; token Jaccard stays above the threshold
    assert pipeline._row_rejection(_user_row(near, real_home=True)) is None
    assert pipeline._row_rejection(_user_row(near, real_home=False)) == "fuzzy_eval_overlap"


def test_pipeline_exact_screen_still_rejects_real_home_eval_prompt():
    prompt = _eval_prompt(6)
    assert pipeline._row_rejection(_user_row(prompt, real_home=True)) == "fuzzy_eval_overlap"


def test_real_home_row_on_an_exact_eval_prompt_is_rephrased_not_dropped(monkeypatch):
    from generators import row_generation

    def phrase(spec):
        # The first phrasing is the exact eval prompt; a fresh phrasing seed draws a different one.
        return "pause the tv" if spec.get("phrasing_seed") is None else "pause TV"

    monkeypatch.setattr(row_generation, "expand_utterance", phrase)
    assert row_generation.check_quality_eval_overlap("pause the tv")
    spec = {"utterance": "pause the tv", "candidate_id": "pause_tv_semantic"}
    row_generation._rephrase_past_eval_overlap(spec)
    assert spec["utterance"] == "pause TV"
    assert not row_generation.check_quality_eval_overlap(spec["utterance"])


def test_real_home_rephrasing_never_accepts_an_eval_prompt(monkeypatch):
    from generators import row_generation

    monkeypatch.setattr(row_generation, "expand_utterance", lambda spec: "pause the tv")
    spec = {"utterance": "pause the tv", "candidate_id": "pause_tv_semantic"}
    row_generation._rephrase_past_eval_overlap(spec)
    assert row_generation.check_quality_eval_overlap(spec["utterance"])
