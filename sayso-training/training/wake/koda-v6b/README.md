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

## Live threshold 0.31 (2026-10-29)

The Pi now fires at 0.31 instead of 0.20. The model is unchanged.

- **Config:** `/etc/sayso-satellite/config.yaml`, changed at 11:44 CDT. The previous config is saved as
  `config.yaml.bak-pre-threshold-031-20261029`.
- **Rollback:** copy that backup over `config.yaml`, then
  `sudo systemctl restart sayso-satellite.service`.

**Evidence.** Satellite recordings were replayed on the satellite's own scoring grid. The window end is a live hop, and
±80 ms steps stay on its mel/embedding grid, so replayed scores equal the live scores exactly.

- Script: `/srv/llm/data/wake/runs/fn-20261029-live.py` on the llm VM.
- Output: `/srv/llm/data/wake/runs/fn-20261029/live.json`.

| | Peak score |
|---|---|
| v6b live false wakes (10-21, and three on 10-29) | 0.21, 0.22, 0.27, 0.28 |
| Real Koda since deploy (10-28 timer commands) | 0.86, 0.90 |

The table below uses the same 45 locked-session Koda as above. Here a Koda counts as caught if any hop within ±0.5 s
of its window passes, so the 0.20 row differs from the per-window table.

| Rule | Koda caught | v6b false wakes passing |
|---|---|---|
| One hop ≥ 0.20 (previous live rule) | 35/45 | 4/4 |
| Two hops in a row, 80 ms apart, ≥ 0.20 | 33/45 | 3/4 |
| Two hops in a row, 160 ms apart, ≥ 0.20 | 29/45 | 1/4 |
| One hop ≥ 0.30 | 33/45 | 0/4 |

Requiring several hops in a row only stops false wakes that flicker. Most v6b false wakes stay just above 0.20 for
0.16–0.4 s instead. What separates them from real Koda is their low peak.

**Limits:**
- Only 4 false wakes support this.
- 0.31 sits just above the highest of them (0.28).
- The expected cost is the 2 locked Koda that peak at 0.22 and 0.28.
