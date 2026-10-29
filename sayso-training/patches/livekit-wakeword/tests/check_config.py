# Validate v6 config against the patched generator: loads, no custom negative collides with /koʊdə/,
# every [[...]] negative renders the intended IPA, the target renders exactly kˈoʊdə.
import subprocess
from livekit.wakeword.config import load_config
from livekit.wakeword.data.piper import text as T
from livekit.wakeword.data.tts.piper_backend import PiperVitsBackend
from livekit.wakeword.data import generate as G
c = load_config("koda-phonics-v6.yaml"); T.register_pronunciations(c.pronunciations)
ipa = lambda t: subprocess.run(["espeak-ng","--ipa","-q","-v","en-us",t],capture_output=True,text=True).stdout.strip()
assert [ipa(t) for t in T.normalize_phrases_for_piper(c.target_phrases)] == ["kˈoʊdə"]
b = PiperVitsBackend.from_config(c)
b.filter_negative_phrases(c.target_phrases, c.custom_negative_phrases, reject_collisions=True)
adv = G.generate_adversarial_phrases(c.target_phrases, max_replace=c.adversarial_max_replace, whole_word=c.adversarial_whole_word) + c.custom_negative_phrases
kept = b.filter_negative_phrases(c.target_phrases, adv)
print("negatives:", len(adv), "kept after homophone filter:", len(kept))
for t in c.custom_negative_phrases: print(f"  {t:14} {ipa(T.normalize_phrases_for_piper([t])[0])}")
print("batch:", c.batch_n_per_class)
