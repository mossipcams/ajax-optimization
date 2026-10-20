# Koda wake model v6b

v6b is the first Koda model that beats the deployed v5 on real eMeet audio with
checkable ground truth. It uses the fixed generator
(`../../../patches/livekit-wakeword/`, /koʊdə/ only, exact one-slot negatives)
plus real room data, re-labelled by what people said after each wake.

Model (not in git; lives on the llm VM and the Pi):
`koda-phonics-v6b.onnx`, 99,190 bytes,
sha256 `b7ecd1a1c72dc8970eda6fd24e4725b0ba2f5d64c397bde504701a7787f646a9`.
Built at `/srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v6b/`.

## Result (locked sessions neither v6 nor v6b trained on)

45 real Koda and 430 room-mic negative windows, live threshold 0.20, per-window scores:

| | v5 (sha256 163df9ad…) | v6 | v6b |
|---|---|---|---|
| Real Koda caught | 18/45 | 28/45 | 29/45 |
| Room negatives passing | 1/430 | 2/430 | 1/430 |
| Held-out TV hours (2 h, streaming) | 0 FA/hr | 0 FA/hr | 0 FA/hr |

Paired, v6b vs v5 at 0.20: v6b catches 12 Koda that v5 misses and v5 catches 1
that v6b misses (exact binomial p = 0.003). The same single room window passes both.

Limits: 430 negative windows is about 15 minutes of mined near-misses, not hours of
conversation; labels are a judgement of the follow-on speech, not a human listening
pass; the 5 confirmed live false positives are in training and cannot be scored.

## Labels (`spool_labels_v2.json`, `relabel.py`)

1,616 Pi spool windows. Each room-mic wake that fired was judged by its follow-on
utterance (SaySo trace or satellite STT): a home command, even garbled ("Pause the
few days"), or Koda heard in the window = real wake; conversation, TV, or a lone
Whisper "Thank you." hallucination = false wake; no usable follow-on = unknown
(excluded). Near-threshold windows followed within 10 s by a real wake =
missed Koda (repeat-after-miss). TV sessions and pre-Koda audio stay negatives by
construction. Totals: 116 positives (19 misses), 1,480 negatives, 20 unknown.

`relabel.py` assigns judgements by index into the room fired wakes sorted by UTC.
`spool_labels_v2.json` is its output and is the record to use.

The SaySo agent's own outcome is not a usable label: through ISO week 41 it called
tools on TV speech and failed real commands at tool_parse; from week 43 it refuses
non-commands and completes commands.

## Files

- `run_v6b.sh` — re-stage real data, mine hard negatives, train, export, eval
  (reuses v6's cached synthetic features; CPU stages at nice 19 on 4 cores).
- `stage_real_v2.py` — real positives (misses oversampled with real context shifts
  and ±6 dB) and room hard negatives from the v2 labels; reuses the locked test
  sessions in `data/splits.json` and never reselects them.
- `eval_v2.py` — v5 / v6 / v6b on the locked sessions with bootstrap intervals.
- `koda-phonics-v6b.yaml` — v6 config with batch real_positive 20, hard_real_words 50.
