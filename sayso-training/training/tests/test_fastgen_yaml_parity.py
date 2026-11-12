"""The optional Rust emitter must produce exactly what PyYAML produces, or decline (None)."""

import random

import pytest
import yaml

fastgen = pytest.importorskip("sayso_fastgen")


def _pyyaml(data):
    return yaml.dump(
        data,
        Dumper=getattr(yaml, "CDumper", yaml.Dumper),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).replace(": null\n", ":\n")


_ALPHABET = list("abcXYZ019 _-.,:#'\"?![]{}&*|>%@`~=+<\t\n") + [
    "é", "ü", "日", " ", " ", "😀", "﻿", "\u0085",
]
_TRICKY = [
    "", "yes", "No", "on", "OFF", "null", "~", "true", "1", "0x1F", "1.5", "-.inf", ".nan",
    "2024-01-02", "2024-1-2 3:04:05", "<<", "=", "---", "...", "- x", "? x", ": x", "#x", "a #b",
    "a: b", "1_000", "012", "+1:30", " lead", "trail ", "it's", "'q'",
]


def _value(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.3:
        return rng.choice(_TRICKY)
    if roll < 0.6:
        return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 12)))
    words = _ALPHABET[:9] + ["'", ":", " "]
    return " ".join(
        "".join(rng.choice(words) for _ in range(rng.randint(1, 14)))
        for _ in range(rng.randint(1, 20))
    )


def test_matches_pyyaml_or_declines():
    rng = random.Random(0)
    emitted = 0
    for _ in range(20000):
        value = _value(rng)
        data = [
            {"names": value, "domain": rng.choice(["light", "on", "lock"])},
            {"names": "Desk Lamp", "domain": "light", "areas": rng.choice([None, value])},
        ]
        out = fastgen.dump_entities(data)
        if out is None:
            continue
        emitted += 1
        assert out == _pyyaml(data), repr(value)
    assert emitted > 15000


def test_folds_long_names_like_pyyaml():
    names = ", ".join(f"living room ceiling light {i}" for i in range(8))
    data = [{"names": names, "domain": "light", "areas": "Living Room"}]
    assert fastgen.dump_entities(data) == _pyyaml(data)
