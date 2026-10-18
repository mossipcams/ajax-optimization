# Check: [[...]] phrases reach espeak verbatim and produce the intended IPA; plain words unchanged.
import json, subprocess, sys
from livekit.wakeword.data.piper.text import normalize_phrases_for_piper, is_phoneme_phrase
from livekit.wakeword.data.piper.synthesis import get_phonemes
cfg = json.load(open("/srv/llm/data/wake/runs/koda-phonics-v6-20261018/data/piper/en-us-libritts-high.json"))
ipa = lambda t: subprocess.run(["espeak-ng", "--ipa", "-q", "-v", "en-us", t], capture_output=True, text=True).stdout.strip()
cases = {"[[k'oUdE]]": "kˈoʊdɛ", "[[k'oUdA:]]": "kˈoʊdɑː", "[[koUd'@]]": "koʊdˈə", "Soda": "sˈoʊdə", "Goda": "ɡˈoʊdə"}
norm = normalize_phrases_for_piper(list(cases))
assert norm[:3] == ["[[k'oUdE]]", "[[k'oUdA:]]", "[[koUd'@]]"], norm      # untouched, case kept
assert norm[3:] == ["soda", "goda"], norm                                  # plain path unchanged
assert is_phoneme_phrase(" [[k'oUd@]] ") and not is_phoneme_phrase("Koda")
for text, want in zip(norm, cases.values()):
    assert ipa(text) == want, (text, ipa(text), want)
# different pronunciations must give different Piper phoneme ids (lowercasing used to merge E/e)
assert get_phonemes(cfg, "[[k'oUdE]]") != get_phonemes(cfg, "[[k'oUde]]")
assert get_phonemes(cfg, "[[k'oUd@]]") == get_phonemes(cfg, "coda")
print("ok")
