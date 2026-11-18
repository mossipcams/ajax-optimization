# livekit-wakeword: Koda pronunciation fix

The Koda wake-word models v2–v5 were trained on the wrong word. The patched
livekit-wakeword copy on the llm VM (`/home/LLM/livekit-wakeword-koda`, and the
per-run `src/` copies made from it) splits any word missing from CMUDict into
dictionary subwords before espeak sees it:

| Phrase | Sent to espeak | Piper spoke |
|---|---|---|
| Koda | `ko da` | kˈoʊ dˈɑː |
| Kota | `ko ta` | kˈoʊ tˈɑː |
| Coda | `coda` | kˈoʊdə |

So two of the three target phrases were not /koʊdə/. Spelled negatives were
mangled the same way (`Hoda` → hˈoʊ dˈɑː).

`0001-livekit-wakeword-koda-pronunciation-fix.patch` (config.py, data/generate.py,
data/piper/text.py) adds:

- `pronunciations:` config (word → ARPAbet). Registered words are not split, so
  `Koda` is synthesised as kˈoʊdə and is searchable for adversarial neighbours.
  The target stays the text `Koda`, so the miner's ASR wake-word guard still works.
- `[[...]]` phrases pass to espeak verbatim (case-sensitive phoneme input) for
  exact negatives such as `[[k'oUdE]]` → kˈoʊdɛ.
- `adversarial_whole_word` / `adversarial_max_replace`. Upstream defaults match
  CMUDict patterns anywhere inside a word (9,269 hits for Koda, e.g. "abandon");
  whole-word with one slot changed gives the 43 real neighbours.

Defaults keep upstream behaviour; only configs that set the new keys change.

## Apply

```bash
cd /home/LLM/livekit-wakeword-koda   # or a run's copy, from the directory above src/
git apply -p1 0001-livekit-wakeword-koda-pronunciation-fix.patch
```

## Check

`tests/` runs on the llm VM against the run copy at
`/srv/llm/data/wake/runs/koda-phonics-v6-20261018` (needs espeak-ng and the Piper
model there):

```bash
cd /srv/llm/data/wake/runs/koda-phonics-v6-20261018
for t in test_phoneme_phrases test_pron test_adv check_config; do
  PYTHONPATH=$PWD/src /srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python -B staging/$t.py
done
```

`koda-phonics-v6.yaml` is the first config using the fix: target `Koda` only,
exact one-slot negatives of /koʊdə/, live false-positive shapes, and rebalanced
real-data batch classes.
