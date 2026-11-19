from __future__ import annotations

from generators import talk
from generators.validation import check_fuzzy_eval_overlap, check_quality_eval_overlap


def test_suffix_fillers_pass_talk_guards():
    for filler in talk._CHATTER_SUFFIX_FILLERS:
        assert talk.talk_has_command_verb(filler) is None, filler
        assert talk.talk_text_violation(filler) is None, filler


def test_media_chatter_sentences_pass_talk_guards():
    for sentence in talk._MEDIA_CHATTER_SENTENCES:
        assert talk.talk_has_command_verb(sentence) is None, sentence


def test_media_chatter_sentences_have_no_eval_overlap():
    for sentence in talk._MEDIA_CHATTER_SENTENCES:
        assert check_quality_eval_overlap(sentence) is False, sentence
        assert check_fuzzy_eval_overlap(sentence) is False, sentence


def test_pools_are_deduped_and_reach_target_sizes():
    fillers = list(talk._CHATTER_SUFFIX_FILLERS)
    sentences = list(talk._MEDIA_CHATTER_SENTENCES)
    assert len(fillers) == len({f.casefold() for f in fillers})
    assert len(sentences) == len(set(s.casefold() for s in sentences))
    assert len(fillers) >= 18
    assert len(sentences) >= 66


def test_wrapped_command_rate_is_doubled():
    assert talk.WRAPPED_COMMAND_RATE == 0.05
