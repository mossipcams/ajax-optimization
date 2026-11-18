from __future__ import annotations

import random
import re

_JUNK_HEADS = (
    "but you're being", "and then he said", "i'll translate", "we were talking",
    "she said something about", "it was just", "they're going to", "i think it was",
    "so anyway", "that's what i", "you know how", "he's been",
)
_JUNK_TAILS = (
    "safe away", "ready in the meal", "for the weekend", "over at the place",
    "the other one", "about half of it", "before they left", "kind of thing",
    "in a minute or two", "all the way down", "sort of like that", "on the way back",
)
_JUNK_FRAGMENTS = (
    "war", "for gv", "uh huh", "hmm", "okay so", "right then", "no no",
    "wait", "mm hmm", "yeah but", "oh", "hang on", "what the", "and um",
)


def _junk_utterance(rng: random.Random, index: int) -> str:
    text = _clean_junk_text(rng)
    return text[:1].upper() + text[1:] if rng.random() < 0.5 else text


_JUNK_FILLER_SHARE = 0.40
_JUNK_CONVERSATION_SHARE = 0.40

_TALK_SPEAKERS = (
    "I", "we", "she", "he", "my brother", "my sister", "the kids", "Dad", "Grandma",
    "Mike", "Jen", "the coach", "the announcer", "Tom", "my roommate",
    "the neighbors", "Kim", "they", "the guy at work",
)
_TALK_NARRATIVE_OPENERS = ("", "", "Well, ", "So ", "Honestly, ", "Anyway, ", "Then ")
_TALK_NARRATIVES = (
    "played cards with the neighbors until midnight",
    "played the same song on repeat all afternoon",
    "turned off the show halfway through the episode",
    "turned the TV on and then forgot about it",
    "paused the game to grab a snack",
    "stopped the car at the light and waited",
    "skipped the intro on every single episode",
    "skipped breakfast and went straight to work",
    "set the table for six people",
    "set the timer on the stove and walked away",
    "opened the windows because it got so hot",
    "closed the store early for the holiday",
    "left the fan running all night",
    "kept the lights on for the dog",
    "watched the game with the whole family",
    "listened to the music in the car the whole way",
    "lowered the volume when the baby finally fell asleep",
    "had a long talk about the kids and school",
    "cooked dinner while the kids watched cartoons",
    "fixed the bathroom faucet over the weekend",
    "drove to the lake and back in one day",
    "wrote a long letter to the landlord",
    "forgot about the pot on the stove again",
    "told the story about the old farm once more",
    "finished the whole book on the train",
    "went for a walk along the river",
    "missed the bus again this morning",
    "got a new phone and lost the old one",
    "changed the channel during the big finish",
)
_TALK_NARRATIVE_TAILS = (
    "", "", " last night", " on Sunday", " with my cousin", " after dinner", " for hours",
    " all day long", " without saying a word", " the other day", " because it was raining",
    " in the middle of the night", " and then nobody talked", " for about ten minutes",
    " like it was nothing", " when we got home", " at the lake house",
)
_TALK_VERB_PAIRS = (
    ("catch", "caught"), ("watch", "watched"), ("play", "played"), ("turn off", "turned off"),
    ("turn on", "turned on"), ("pause", "paused"), ("set", "set"), ("open", "opened"),
    ("close", "closed"), ("skip", "skipped"), ("stop", "stopped"), ("finish", "finished"),
    ("fix", "fixed"),
)
_TALK_QUESTION_OBJECTS = (
    "the game", "that show", "the new episode", "the kids", "the television", "the lights", "the fan",
    "the music", "the volume", "the kitchen timer", "the playoffs", "the dog", "the car", "the window",
    "the dishes", "the bills", "the kitchen", "that movie",
)
_TALK_QUESTION_TAILS = (
    " yesterday", " this morning", " before we left", " when you got home",
    " at the game", " with the kids",
)
_TALK_QUESTION_WHO = ("he", "she", "they", "Tom", "Jen", "your brother", "Grandpa", "Mike")
_TALK_DIALOGUE_OPENERS = ("Look, ", "I mean, ", "No, ", "Listen, ", "Okay, ", "You know what, ", "Honestly, ", "Right, ", "Well, ")
_TALK_DIALOGUE_CLAUSES = (
    "we can pause this and come back to it later",
    "the show is about to start and you missed it",
    "that is not how the game is supposed to work",
    "I said I would be home for dinner",
    "the timer is going off in the kitchen again",
    "nobody wants to set the table tonight",
    "the kids were up until two watching the TV",
    "I can't believe they turned off the lights in the parking lot",
    "the team is down by three with two minutes left",
    "we are going to stop at the next town",
    "that song always gets stuck in my head",
    "the new season is way better than the last one",
    "your sister skipped the whole first act",
    "the fan in the bedroom has been making that noise for weeks",
    "I left the volume low so the baby could sleep",
    "the coach kept calling the same play all night",
    "the music was too loud for the whole neighborhood",
    "she only watched the game for the halftime show",
    "he opened the box and found nothing inside",
    "we closed the deal on the house last spring",
    "the kids are going to be so tired tomorrow",
    "I need to call the dentist before Friday",
    "that guy has been talking about the same thing for an hour",
    "the store is open until nine on Saturdays",
    "nobody knows who left the back door unlocked",
    "the show got canceled after one season",
    "my mom always stops by on Thursdays",
    "I don't think the timer on that thing ever worked",
    "we played the long version of the game again",
)
_MEDIA_CHATTER_SENTENCES = (
    "that song from the movie was stuck in my head all day",
    "the episode last night had a great ending",
    "my brother wants a new TV for the game",
    "the playlist my sister made is way too long",
    "I can't believe how many lights they put up on the house this year",
    "the channel keeps showing the same old movie every Sunday",
    "that show was canceled after one season and nobody cared",
    "the music at the wedding was way too loud for my ears",
    "Grandpa only watches the game when the weather is bad",
    "the new TV in the den is bigger than the whole couch",
    "the lights in the parking lot are being replaced next month",
    "the movie we saw last weekend was about a dog and a boat",
    "we need a better song for the video we are making",
    "the kids keep quoting that movie at dinner",
    "the game went to overtime and nobody in the house slept",
    "my favorite channel moved to a different number this week",
    "that episode made me cry in the middle of the grocery store",
    "the lights at the stadium were so bright I could not see the field",
    "my roommate has a whole playlist of old country music",
    "is the new show any good or is it a waste of time",
    "the TV guide listed the game for eight o'clock",
    "the music festival movie was a total mess",
)
_MEDIA_CHATTER_SHARE = 0.15
_TALK_GARBLE_LEADS = (
    "so", "and", "oh", "like", "the uh", "hang on", "um", "okay", "and then", "so yeah",
    "hm and", "and the uh",
)
_TALK_GARBLE_DEVICES = (
    "the den tv", "the porch lights", "the bedroom fan", "the office speaker",
    "the hallway switch", "the garage door", "the upstairs speaker", "the bedroom lamp",
    "the basement lights", "the hallway light", "the back porch light",
    "the upstairs fan", "the family room tv", "the den speaker", "the volume on the set",
)
_TALK_GARBLE_TAILS = (
    "were kind of", "uh huh", "and um", "kind of", "of", "hm", "the uh the", "like",
    "you know", "I think", "or something", "and stuff",
)


def _talk_narrative(rng: random.Random) -> str:
    speaker = rng.choice(_TALK_SPEAKERS)
    opener = rng.choice(_TALK_NARRATIVE_OPENERS)
    text = f"{opener}{speaker} {rng.choice(_TALK_NARRATIVES)}{rng.choice(_TALK_NARRATIVE_TAILS)}."
    return text[:1].upper() + text[1:]


def _talk_question(rng: random.Random) -> str:
    base, past = rng.choice(_TALK_VERB_PAIRS)
    obj = rng.choice(_TALK_QUESTION_OBJECTS)
    tail = rng.choice(_TALK_QUESTION_TAILS)
    shape = rng.random()
    if shape < 0.40:
        return f"Did {rng.choice(_TALK_QUESTION_WHO)} {base} {obj}{tail}?"
    if shape < 0.70:
        return f"Have you ever {past} {obj}{tail}?"
    return f"Wait, who {past} {obj}{tail}?"


def _talk_dialogue(rng: random.Random) -> str:
    return f"{rng.choice(_TALK_DIALOGUE_OPENERS)}{rng.choice(_TALK_DIALOGUE_CLAUSES)}."


def _conversational_sentence(rng: random.Random) -> str:
    shape = rng.random()
    if shape < 0.50:
        return _talk_narrative(rng)
    if shape < 0.80:
        return _talk_question(rng)
    return _talk_dialogue(rng)


def _garbled_device_fragment(rng: random.Random) -> str:
    text = (
        f"{rng.choice(_TALK_GARBLE_LEADS)} {rng.choice(_TALK_GARBLE_DEVICES)} "
        f"{rng.choice(_TALK_GARBLE_TAILS)}"
    ).strip()
    return f"{text}{rng.choice(('.', '...', '?', ''))}"


def _junk_filler_text(rng: random.Random) -> str:
    shape = rng.random()
    if shape < 0.30:
        return f"{rng.choice(_JUNK_FRAGMENTS)}."
    if shape < 0.55:
        return f"{rng.choice(_JUNK_FRAGMENTS)}, {rng.choice(_JUNK_FRAGMENTS)}."
    if shape < 0.80:
        return f"{rng.choice(_JUNK_HEADS)} {rng.choice(_JUNK_TAILS)}..."
    return (
        f"{rng.choice(_JUNK_HEADS)} {rng.choice(_JUNK_TAILS)} and... "
        f"{rng.choice(_JUNK_FRAGMENTS)}?"
    )


def _junk_conversation_sentence(rng: random.Random) -> str:
    # Media chatter lives in its own pool so _chatter_filler never wraps a command with it.
    if rng.random() < _MEDIA_CHATTER_SHARE:
        return rng.choice(_MEDIA_CHATTER_SENTENCES)
    return _conversational_sentence(rng)


def _junk_sample(rng: random.Random) -> tuple[str, str]:
    roll = rng.random()
    if roll < _JUNK_FILLER_SHARE:
        return "filler", _junk_filler_text(rng)
    if roll < _JUNK_FILLER_SHARE + _JUNK_CONVERSATION_SHARE:
        return "conversation", _junk_conversation_sentence(rng)
    return "garbled", _garbled_device_fragment(rng)


def _junk_text(rng: random.Random) -> str:
    return _junk_sample(rng)[1]


_JUNK_CLEAN_ATTEMPTS = 8

TALK_DEVICE_NOUNS = (
    "tv", "tvs", "television", "televisions", "telly",
    "light", "lights", "lamp", "lamps",
    "fan", "fans",
    "lock", "locks",
    "thermostat",
    "speaker", "speakers",
    "vacuum",
    "blind", "blinds",
)
TALK_COMMAND_VERBS = (
    "turn", "turns", "turned", "turning",
    "switch", "switches", "switched", "switching",
    "set", "sets",
    "pause", "pauses", "paused", "pausing",
    "play", "plays", "played", "playing",
    "stop", "stops", "stopped", "stopping",
    "start", "starts", "started", "starting",
    "resume", "resumes", "resumed",
    "unpause", "unpaused",
    "mute", "muted", "muting",
    "unmute",
    "volume",
    "dim", "dims", "dimmed", "dimming",
    "open", "opens", "opened", "opening",
    "close", "closes", "closed", "closing",
    "lock", "locks", "locked", "locking",
    "unlock", "unlocks", "unlocked",
)

_TALK_VERB_RE = re.compile(r"\b(?:" + "|".join(TALK_COMMAND_VERBS) + r")\b")
_TALK_BANNED_RE = re.compile(
    r"\b(?:" + "|".join(dict.fromkeys(TALK_DEVICE_NOUNS + TALK_COMMAND_VERBS)) + r")\b"
)


def talk_has_command_verb(text: str) -> str | None:
    match = _TALK_VERB_RE.search(text.casefold())
    return match.group(0) if match else None


def talk_text_violation(text: str) -> str | None:
    match = _TALK_BANNED_RE.search(text.casefold())
    return match.group(0) if match else None


def _clean_junk_text(rng: random.Random) -> str:
    for _ in range(_JUNK_CLEAN_ATTEMPTS):
        text = _junk_text(rng)
        if talk_has_command_verb(text) is None:
            return text
    return _junk_filler_text(rng)


WRAPPED_COMMAND_RATE = 0.025

_CHATTER_PREFIX_FILLERS = ("okay so", "um", "hold on", "wait", "oh and")
_CHATTER_SUFFIX_FILLERS = ("thanks", "please", "okay", "real quick")


def _chatter_filler(rng: random.Random, *, prefix: bool) -> str:
    if rng.random() < 0.25:
        for _ in range(4):
            sentence = _conversational_sentence(rng).casefold().rstrip(".?! ")
            if talk_text_violation(sentence) is None:
                return sentence
    return rng.choice(_CHATTER_PREFIX_FILLERS if prefix else _CHATTER_SUFFIX_FILLERS)


def wrap_utterance_with_chatter(text: str, rng: random.Random) -> tuple[str, str]:
    roll = rng.random()
    base = text.rstrip().rstrip(".!?")
    if roll < 0.40:
        return f"{_chatter_filler(rng, prefix=True)}, {base}", "prefix"
    if roll < 0.80:
        return f"{base}, {_chatter_filler(rng, prefix=False)}", "suffix"
    return (
        f"{_chatter_filler(rng, prefix=True)}, {base}, {_chatter_filler(rng, prefix=False)}",
        "both",
    )
