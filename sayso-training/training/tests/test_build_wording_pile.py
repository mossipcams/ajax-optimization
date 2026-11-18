
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from generators.capability_registry import operation_spec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wording_pile.py"
_spec = importlib.util.spec_from_file_location("build_wording_pile", SCRIPT)
bwp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bwp  # @dataclass looks the module up by name
_spec.loader.exec_module(bwp)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(bwp.time, "sleep", lambda seconds: None)


def _op(capability: str, operation: str):
    return bwp.CAPABILITIES[capability], operation_spec(capability, operation)


def _seed_pile(path: Path) -> None:
    entries = [
        {"id": bwp.template_id(t), "op": op, "style": "direct", "template": t, "slot": "name", "writer": "w"}
        for t, op in [
            ("turn on {name}", "lights.turn_on"),
            ("turn off {name}", "lights.turn_off"),
            ("dim {name}", "lights.set_brightness"),
        ]
    ]
    path.write_bytes(bwp.pile_bytes(entries))


def _answers_from_prompt(answers: dict[str, str]):
    def fake_chat(endpoint, prompt, temperature, timeout=600):
        templates = json.loads(prompt.split("Templates: ", 1)[1])
        return json.dumps([answers[template] for template in templates])

    return fake_chat


def test_slot_check_requires_exactly_one_known_slot():
    assert bwp.slot_check("turn on {name}", area_ok=False) is None
    assert bwp.slot_check("turn on {area}", area_ok=True) is None
    assert bwp.slot_check("turn on the lights", area_ok=True) == "slot"
    assert bwp.slot_check("turn {name} and {name} on", area_ok=True) == "slot"
    assert bwp.slot_check("turn {name} or {area}", area_ok=True) == "slot"
    assert bwp.slot_check("turn on {device}", area_ok=True) == "slot"


def test_slot_check_rejects_area_when_op_has_no_area_targeting():
    assert bwp.slot_check("turn on {area}", area_ok=False) == "area_slot"


def test_slot_check_rejects_stray_braces_and_bad_word_counts():
    assert bwp.slot_check("turn on {name} {", area_ok=True) == "braces"
    assert bwp.slot_check("on {name}", area_ok=True) is None
    assert bwp.slot_check("{name}", area_ok=True) == "words"
    fifteen_words = "please could you kindly turn the {name} now for me thanks a lot my friend"
    assert len(fifteen_words.split()) == 15
    assert bwp.slot_check(fifteen_words, area_ok=True) == "words"


def test_admit_dedupes_by_normalized_template_and_drops_eval_overlap():
    cap, op = _op("lights", "turn_on")
    counts = Counter()
    seen: set[str] = set()
    banned = {bwp.template_key("turn on the {name}")}
    accepted = bwp.admit(
        ["turn on {name}", "Please turn on {name}.", "turn on the {name}", "turn on {area} lights", "switch on {name}"],
        op_key="lights.turn_on", cap=cap, op=op, style="direct",
        writer="w", seen=seen, banned=banned, counts=counts,
    )
    assert [entry["template"] for entry in accepted] == ["turn on {name}", "turn on {area} lights", "switch on {name}"]
    assert counts["duplicate"] == 1
    assert counts["eval_overlap"] == 1
    assert counts["candidates"] == 5
    assert accepted[1]["slot"] == "area"
    assert accepted[0]["id"] == bwp.template_id("Please turn on {name}")


def test_admit_rejects_area_for_ops_without_area_targeting():
    cap, op = _op("timers", "cancel_all")
    counts = Counter()
    accepted = bwp.admit(
        ["cancel the {area} timers", "cancel the {name} timer"],
        op_key="timers.cancel_all", cap=cap, op=op, style="direct",
        writer="w", seen=set(), banned=set(), counts=counts,
    )
    assert [entry["template"] for entry in accepted] == ["cancel the {name} timer"]
    assert counts["area_slot"] == 1


def test_noun_check_rejects_device_noun_doubled_after_name():
    lights, _ = _op("lights", "turn_on")
    media, _ = _op("media_players", "pause")
    assert bwp.noun_check("switch on the {name} light", lights) == "doubled_noun"
    assert bwp.noun_check("turn on {name} lights please", lights) == "doubled_noun"
    assert bwp.noun_check("pause the {name} tv", media) == "doubled_noun"
    assert bwp.noun_check("turn on {name}", lights) is None
    assert bwp.noun_check("switch on {name} now", lights) is None


def test_noun_check_exempts_timer_noun_only_for_timer_ops():
    timers, _ = _op("timers", "pause")
    lights, _ = _op("lights", "turn_on")
    assert bwp.noun_check("pause the {name} timer", timers) is None
    assert bwp.noun_check("cancel the {name} timer", timers) is None
    assert bwp.noun_check("turn on the {name} timer", lights) == "doubled_noun"
    assert bwp.noun_check("pause the {name} light", timers) == "doubled_noun"


def test_noun_check_requires_capability_noun_for_area_templates():
    lights, _ = _op("lights", "turn_on")
    fans, _ = _op("fans", "turn_on")
    covers, _ = _op("covers", "close")
    vacuums, _ = _op("vacuums", "clean_area")
    assert bwp.noun_check("Could you turn on {area}", lights) == "area_noun"
    assert bwp.noun_check("Turn on {area} lights", lights) is None
    assert bwp.noun_check("Turn on the {area} fans", fans) is None
    assert bwp.noun_check("Close {area} blinds", covers) is None
    assert bwp.noun_check("Close {area}", covers) == "area_noun"
    assert bwp.noun_check("Clean {area}", vacuums) == "area_noun"
    assert bwp.noun_check("Clean the {area} with the vacuum", vacuums) is None


def test_area_noun_map_covers_every_area_targeted_capability():
    for cap in bwp.CAPABILITIES.values():
        if "area" in cap.targeting_modes and bwp.trainable_operations(cap):
            assert cap.name in bwp.AREA_NOUNS, cap.name


def test_admit_counts_doubled_and_area_noun_rejections():
    cap, op = _op("lights", "turn_on")
    counts = Counter()
    accepted = bwp.admit(
        ["switch on the {name} light", "turn on {area}", "turn on {area} lights"],
        op_key="lights.turn_on", cap=cap, op=op, style="direct",
        writer="w", seen=set(), banned=set(), counts=counts,
    )
    assert [entry["template"] for entry in accepted] == ["turn on {area} lights"]
    assert counts["doubled_noun"] == 1
    assert counts["area_noun"] == 1


def test_eval_template_replaces_names_then_areas():
    names = {"Kitchen Counter Lights", "Kitchen Lights"}
    areas = {"Kitchen"}
    text = bwp.eval_template("Switch the kitchen counter lights on in the Kitchen", names, areas)
    assert text == "Switch the {name} on in the {area}"


def test_eval_template_respects_word_boundaries():
    assert bwp.eval_template("kitchenette lights", set(), {"Kitchen"}) == "kitchenette lights"


def test_parse_json_array_tolerates_fences_and_think_blocks():
    text = "<think>plan first [</think>\n```json\n[\"turn on {name}\", 3]\n```"
    assert bwp.parse_json_array(text) == ["turn on {name}", 3]
    assert bwp.parse_json_array("no array here") is None
    assert bwp.parse_json_array("[not json]") is None
    assert bwp.parse_json_array('{"a": 1}') is None


def test_write_drops_malformed_items_and_uses_mocked_http(monkeypatch, tmp_path):
    calls = []

    def fake_chat(endpoint, prompt, temperature, timeout=600):
        calls.append((endpoint.model, temperature))
        return '```json\n["turn on {name}", 7, "switch on {name}"]\n```'

    monkeypatch.setattr(bwp, "chat", fake_chat)
    monkeypatch.setattr(bwp, "eval_keys", lambda: set())
    out = tmp_path / "pile.jsonl"
    assert bwp.main(["write", "--url", "http://fake/v1", "--model", "writer-model",
                     "--ops", "lights.turn_on", "--n", "3", "--out", str(out)]) == 0
    assert len(calls) == len(bwp.STYLES)
    assert all(model == "writer-model" and temperature == 0.9 for model, temperature in calls)
    entries = bwp.read_pile(out)
    assert {entry["template"] for entry in entries} == {"turn on {name}", "switch on {name}"}
    assert len(entries) == 2  # the same templates repeated across styles collapse by dedupe
    assert all(entry["writer"] == "writer-model" and entry["op"] == "lights.turn_on" for entry in entries)


def test_write_rejects_unknown_ops(monkeypatch, tmp_path):
    monkeypatch.setattr(bwp, "eval_keys", lambda: set())
    with pytest.raises(SystemExit, match="unknown operations"):
        bwp.main(["write", "--url", "u", "--model", "m", "--ops", "lights.dance", "--out", str(tmp_path / "x")])


def test_verify_marks_agreement_blind_and_is_resumable(monkeypatch, tmp_path):
    pile = tmp_path / "pile.jsonl"
    _seed_pile(pile)
    prompts = []
    answer = _answers_from_prompt({"turn on {name}": "lights.turn_on", "turn off {name}": "lights.turn_on",
                                   "dim {name}": "ambiguous"})

    def recording_chat(endpoint, prompt, temperature, timeout=600):
        prompts.append((prompt, temperature))
        return answer(endpoint, prompt, temperature)

    monkeypatch.setattr(bwp, "chat", recording_chat)
    argv = ["verify", "--url", "u", "--model", "verifier", "--pile", str(pile)]
    assert bwp.main(argv) == 0
    entries = {entry["template"]: entry for entry in bwp.read_pile(pile)}
    assert entries["turn on {name}"]["verify"] == {"model": "verifier", "op": "lights.turn_on", "agree": True}
    assert entries["turn off {name}"]["verify"] == {"model": "verifier", "op": "lights.turn_on", "agree": False}
    assert entries["dim {name}"]["verify"] == {"model": "verifier", "op": "ambiguous", "agree": False}
    assert len(prompts) == 1
    prompt, temperature = prompts[0]
    assert temperature == 0.0
    assert "lights.turn_off" in prompt and "ambiguous" in prompt
    assert "direct" not in prompt  # blind: the writer's style never reaches the verifier

    assert bwp.main(argv) == 0
    assert len(prompts) == 1  # every entry already verified by this model, nothing re-asked


def test_verify_prompt_offers_unnatural_and_timer_name_hint():
    timers = bwp.verify_prompt(["pause {name}"], bwp.verify_options("timers"), "timers")
    assert bwp.UNNATURAL in timers and bwp.AMBIGUOUS in timers
    assert "timer's name (such as 'laundry')" in timers
    assert "not a device name" in timers
    lights = bwp.verify_prompt(["turn on {name}"], bwp.verify_options("lights"), "lights")
    assert bwp.UNNATURAL in lights and bwp.AMBIGUOUS in lights
    assert "laundry" not in lights
    assert "{name} stands for a device name" in lights
    assert "plausibly say the template to a voice assistant" in lights
    assert "Trigger the fan action assigned to {name}" in lights


def test_verify_records_unnatural_and_counts_it(monkeypatch, tmp_path, capsys):
    pile = tmp_path / "pile.jsonl"
    _seed_pile(pile)
    answer = _answers_from_prompt({"turn on {name}": "unnatural", "turn off {name}": "lights.turn_off",
                                   "dim {name}": bwp.AMBIGUOUS})
    monkeypatch.setattr(bwp, "chat", answer)
    assert bwp.main(["verify", "--url", "u", "--model", "verifier", "--pile", str(pile)]) == 0
    entries = {entry["template"]: entry for entry in bwp.read_pile(pile)}
    assert entries["turn on {name}"]["verify"] == {"model": "verifier", "op": "unnatural", "agree": False}
    assert entries["turn off {name}"]["verify"]["agree"] is True
    assert entries["dim {name}"]["verify"]["agree"] is False
    err = capsys.readouterr().err
    assert "{" in err and "'unnatural': 1" in err and "'agree': 1" in err and "'disagree': 1" in err


def test_verify_keeps_going_after_http_errors(monkeypatch, tmp_path):
    pile = tmp_path / "pile.jsonl"
    _seed_pile(pile)

    def failing_chat(endpoint, prompt, temperature, timeout=600):
        raise OSError("connection refused")

    monkeypatch.setattr(bwp, "chat", failing_chat)
    assert bwp.main(["verify", "--url", "u", "--model", "verifier", "--pile", str(pile)]) == 0
    assert all("verify" not in entry for entry in bwp.read_pile(pile))


def test_verify_retries_wrong_length_then_succeeds(monkeypatch, tmp_path):
    pile = tmp_path / "pile.jsonl"
    _seed_pile(pile)
    good = _answers_from_prompt({"turn on {name}": "lights.turn_on", "turn off {name}": "lights.turn_off",
                                 "dim {name}": "lights.set_brightness"})
    replies = iter(['["lights.turn_on"]'])

    def flaky_then_good(endpoint, prompt, temperature, timeout=600):
        return next(replies, None) or good(endpoint, prompt, temperature)

    monkeypatch.setattr(bwp, "chat", flaky_then_good)
    assert bwp.main(["verify", "--url", "u", "--model", "verifier", "--pile", str(pile)]) == 0
    agreed = {entry["template"]: entry["verify"]["agree"] for entry in bwp.read_pile(pile)}
    assert agreed == {"turn on {name}": True, "turn off {name}": True, "dim {name}": True}


def _agreed_pile(path: Path, order: list[int]) -> None:
    rows = []
    for index in range(60):
        template = f"turn on {{name}} number {index} now"
        rows.append({
            "id": bwp.template_id(template), "op": "lights.turn_on", "style": "terse",
            "template": template, "slot": "name", "writer": "writer-model",
            "verify": {"model": "verifier", "op": "lights.turn_on", "agree": True},
        })
    rows.append({
        "id": bwp.template_id("dim {name}"), "op": "lights.set_brightness", "style": "casual",
        "template": "dim {name}", "slot": "name", "writer": "writer-model",
        "verify": {"model": "verifier", "op": "lights.turn_on", "agree": False},
    })
    shuffled = [rows[index] for index in order]
    path.write_bytes("".join(json.dumps(row, sort_keys=True) + "\n" for row in shuffled).encode("utf-8"))


def test_finalize_is_deterministic_and_writes_manifest(tmp_path):
    manifests, shas = [], []
    for label, order in (("a", list(range(61))), ("b", list(reversed(range(61))))):
        pile = tmp_path / f"in_{label}.jsonl"
        _agreed_pile(pile, order)
        out = tmp_path / f"pile_{label}.jsonl"
        manifest = tmp_path / f"manifest_{label}.json"
        assert bwp.main(["finalize", "--pile", str(pile), "--out", str(out), "--manifest", str(manifest)]) == 0
        manifests.append(json.loads(manifest.read_text(encoding="utf-8")))
        shas.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert shas[0] == shas[1]
    assert manifests[0]["sha256"] == shas[0]
    assert manifests[0]["count"] == 60
    assert manifests[0]["per_op"] == {"lights.turn_on": 60}
    assert manifests[0]["writer"] == "writer-model"
    assert manifests[0]["verifier"] == "verifier"
    assert set(manifests[0]) == {"sha256", "count", "per_op", "writer", "verifier", "created"}


def test_finalize_review_sample_is_seeded(tmp_path, capsys):
    pile = tmp_path / "in.jsonl"
    _agreed_pile(pile, list(range(61)))
    samples = []
    for _ in range(2):
        bwp.main(["finalize", "--pile", str(pile), "--out", str(tmp_path / "p.jsonl"),
                  "--manifest", str(tmp_path / "m.json"), "--seed", "7"])
        samples.append(capsys.readouterr().out.splitlines()[1:])
    assert samples[0] == samples[1]
    assert len(samples[0]) == bwp.REVIEW_SAMPLE


def test_finalize_refuses_when_nothing_agrees(tmp_path):
    pile = tmp_path / "in.jsonl"
    _seed_pile(pile)
    with pytest.raises(SystemExit, match="no entries agreed"):
        bwp.main(["finalize", "--pile", str(pile), "--out", str(tmp_path / "p"), "--manifest", str(tmp_path / "m")])


def test_eval_keys_load_from_real_eval_cases():
    keys = bwp.eval_keys()
    assert keys
    assert all("{" not in key for key in keys)


class _Response:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


def _capture_urlopen(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=600):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"choices": [{"message": {"content": "<think>x</think>[]"}}]})

    monkeypatch.setattr(bwp.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_chat_sends_enable_thinking_false_only_when_no_think(monkeypatch):
    captured = _capture_urlopen(monkeypatch)
    text = bwp.chat(bwp.Endpoint(bwp.DEFAULT_URL, "m", None, no_think=True), "p", 0.0)
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bwp.parse_json_array(text) == []

    bwp.chat(bwp.Endpoint(bwp.DEFAULT_URL, "m", None), "p", 0.9)
    assert "chat_template_kwargs" not in captured["body"]


def test_think_defaults_per_subcommand_and_can_be_overridden():
    parser = bwp.build_parser()
    write = parser.parse_args(["write", "--model", "m", "--out", "o"])
    verify = parser.parse_args(["verify", "--model", "m", "--pile", "p"])
    assert write.no_think is False and verify.no_think is True
    assert write.url == bwp.DEFAULT_URL == "http://127.0.0.1:8000/v1"
    assert parser.parse_args(["write", "--model", "m", "--out", "o", "--no-think"]).no_think is True
    assert parser.parse_args(["verify", "--model", "m", "--pile", "p", "--think"]).no_think is False
