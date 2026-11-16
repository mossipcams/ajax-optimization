from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators import talk  # noqa: E402

SEEDS = range(2000)
MEDIA_CHATTER = {sentence.casefold() for sentence in talk._MEDIA_CHATTER_SENTENCES}


def test_media_chatter_sentences_have_no_command_verb():
    for sentence in talk._MEDIA_CHATTER_SENTENCES:
        assert talk.talk_has_command_verb(sentence) is None, sentence


def test_media_chatter_appears_in_junk_utterances_for_some_seeds():
    samples = [talk._junk_sample(random.Random(seed)) for seed in SEEDS]
    media = [text for kind, text in samples if kind == "conversation" and text.casefold() in MEDIA_CHATTER]
    conversation = [text for kind, text in samples if kind == "conversation"]
    assert media
    assert 0.08 <= len(media) / len(conversation) <= 0.22, (len(media), len(conversation))


def test_junk_utterances_with_media_chatter_are_deterministic():
    assert [talk._junk_text(random.Random(seed)) for seed in range(200)] == [
        talk._junk_text(random.Random(seed)) for seed in range(200)
    ]


def test_chatter_filler_never_returns_media_chatter():
    returned: Counter[str] = Counter()
    for seed in SEEDS:
        for prefix in (True, False):
            filler = talk._chatter_filler(random.Random(seed), prefix=prefix)
            returned[filler.casefold()] += 1
            assert filler.casefold() not in MEDIA_CHATTER, filler
    assert returned
