from livekit.wakeword.data.piper import text as T
from livekit.wakeword.data import generate as G
T.register_pronunciations({"koda": "K OW1 D AH0"})
adv = G.generate_adversarial_phrases(["Koda"], include_input_words=0, max_replace=1, whole_word=True)
assert {"goda", "toda", "boda", "cota", "cody", "coder", "soda", "kona", "koza"} <= set(adv), sorted(adv)
assert "coda" not in adv and "abandon" not in adv and len(adv) < 60, len(adv)
default = G.generate_adversarial_phrases(["Koda"], include_input_words=0)
assert len(default) > 1000                                         # defaults unchanged (upstream behaviour)
print("ok", len(adv), len(default))
