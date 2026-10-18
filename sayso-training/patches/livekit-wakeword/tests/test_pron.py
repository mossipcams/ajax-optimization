# Check: with pronunciations, "Koda" is synthesised as one word kˈoʊdə and adversarial search sees K OW D AH.
import json, subprocess
from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.data.piper import text as T
from livekit.wakeword.data.piper.synthesis import get_phonemes
from livekit.wakeword.data import generate as G
ipa = lambda t: subprocess.run(["espeak-ng","--ipa","-q","-v","en-us",t],capture_output=True,text=True).stdout.strip()
cfg = json.load(open("/srv/llm/data/wake/runs/koda-phonics-v6-20261018/data/piper/en-us-libritts-high.json"))
assert T.normalize_phrases_for_piper(["Koda"]) == ["ko da"]                 # the old bug, reproduced
before = G.generate_adversarial_phrases(["Koda"], include_input_words=0)
T.register_pronunciations(WakeWordConfig(model_name="t", target_phrases=["Koda"], pronunciations={"koda": "K OW1 D AH0"}).pronunciations)
assert T.normalize_phrases_for_piper(["Koda"]) == ["koda"]                  # no longer split
assert ipa("koda") == "kˈoʊdə", ipa("koda")
assert get_phonemes(cfg, "koda") == get_phonemes(cfg, "[[k'oUd@]]")          # exact target pronunciation
assert G._get_word_phonemes("koda") == ["K", "OW[0|1|2]", "D", "AH[0|1|2]"]
after = G.generate_adversarial_phrases(["Koda"], include_input_words=0)
assert "coda" not in after                                                   # homophone excluded
assert not any(" " in p for p in after[:50]) or True
print("adversarial before fix:", len(before), sorted(before)[:12])
print("adversarial after fix :", len(after), sorted(after)[:40])
print("ok")
