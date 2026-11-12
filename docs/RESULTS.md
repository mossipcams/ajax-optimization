# Results

All numbers: RX 7900 XTX. Prod is Swift-1.5 Qwen3.8-27B Q4_K_M, MTP n=3 + ngram-map-k 16, 2 slots non-unified 2 x 81920, q8_0 KV.
Rows dated 2026-09-30 and earlier were measured on the older UD-Q4_K_M, 4-slot, 128k unified layout.
"SOLO" = one stream at ~30k depth, "PAIR" = two concurrent streams at ~30k each (combined tok/s).
Since 2026-11, SOLO edit/review come from `hfbench.py` with Pi sampling (T1.0, top_p 0.95, top_k 20, repeat 1.05, freq 0.3) at ~24k depth.

## Deployed

### No mmproj (2026-11-11)
`-hf` auto-loaded the 0.93 GB F16 mmproj. VRAM was 24493 of 24560 MiB, and 1716 MiB sat in GTT.
With `--no-mmproj`: GTT 381 MiB (583 after bench), VRAM 24285 MiB. Live SOLO edit 154.0 -> 160.1, review 60.9 -> 63.0.
Images are used only with the atomic profile.

### ngram-map-k size_n 16 (2026-11-11)
Replaces ngram-mod. Edit SOLO 135.9 -> 152.6 (+12%), review 59.7 -> 64.6 (+8%). Pi sampling.

### Non-unified KV, 2 slots (2026-10-20)
Same-day PAIR at 30k x 2, 3 samples each. Unified 96.2 / 84.0 / 87.6 vs non-unified 103.4 / 103.4 / 97.5 combined t/s.
GTT 309 -> 189 MiB. Per-slot context raised from 61440 to 81920 on 2026-10-21.
A single stream reads only its own KV, and Pi folds context near 40k, so unified's larger shared pool goes unused.

### Swift-1.5 model (2026-10)
`ukisai/Swift-1.5-Qwen3.8-27B-GGUF:Q4_K_M`, a reasoning-efficient fine-tune of Qwen3.8-27B.
Anti-loop sampling since 2026-10-05: T1.0, top_p 0.95, top_k 20, repeat 1.05, freq 0.3, reasoning effort medium.

### GQA token packing in the scalar FA kernel (2026-09-30)
The Vulkan scalar FA kernel gave every query token its own workgroup, so verify batches re-scanned the KV once per token.
Cost was ~26 µs per workgroup pass over 30k KV regardless of rows per tile, so fewer passes is what matters.
The patch packs T tokens x 6 GQA heads into one tile (T=4 when there are exactly 4 tokens, else 2; 32-row tile).
T rides in bits 16-23 of `mask_n_head_log2` because the FA push constants are at the 128-byte limit.

Kernel (per layer, 30k KV, hd256, GQA 6, q8_0):

| Query rows | Before | After |
|---|---|---|
| 1 | 132 µs | 132 µs |
| 2 | 243 µs | 172 µs |
| 4 | 428 µs | 289 µs |
| 8 | 807 µs | 572 µs |

End to end (two alternating A/B rounds, acceptance unchanged):

| | `d280808-fa2` | `d280808-fa2pack` |
|---|---|---|
| SOLO | 59.6 | 61.6 |
| PAIR | 76.5 | 88.7 |

Traffic is ~70% single-stream, so the weighted gain is smaller than the PAIR figure.

### Earlier
- Idle-slot eviction fix: `--no-cache-idle-slots` (unified KV was clearing idle slots on every new request, forcing 30k re-prefills).
- Model: unsloth UD-Q4_K_M; MTP head rebuilt with Q8_0 layer and Q4_0 output; server sampling defaults T0.7/p0.8/k20.
- FA2: scalar path forced for small GQA row counts, split-k tuned (12-22% on RDNA3).

## Rejected (measured)
- Q2_K draft-head output: acceptance 0.669 vs 0.692, tok/s within noise.
- Mesa 26.2.3, `--spec-draft-p-min 0.5`, MTP n4/n5, q4_0 KV, UD-IQ4_XS, unsloth MTP head.
- `GGML_VK_FA_SPLITK_MULT` sweep: the default is already best.
- Turning GQA packing off for small batches: 8 rows 689 µs vs 808 µs, but worse at 4 rows.
- f16 KV: same FA time as q8_0 at 4-8 rows, so FA is not bandwidth- or dequant-bound.

### Stock unsloth V3.1 quants vs Swift (2026-11-11)

| arm | edit SOLO r1/r2 | review SOLO r1/r2 | GTT loaded/after bench MiB | VRAM MiB | SaySo-code PPL |
|---|---|---|---|---|---|
| Swift-1.5 Q4_K_M | 153.3 / 151.7 | 60.7 / 60.9 | 385 / 587 | 24285 | 2.196 |
| V3.1 UD-Q4_K_M | 169.2 / 169.0 | 61.0 / 60.9 | 233 / 342 | 23600 | 2.112 |
| V3.1 UD-Q4_K_XL | 164.7 / 164.4 | 55.4 / 55.6 | 385 / 738 | 24492 | 2.117 |

- Every arm passed toolcheck 6/6. Idle GTT was ~20 MiB. The run alternated S Q X X Q S without mmproj (S = Swift, Q = V3.1 UD-Q4_K_M, X = XL).
- XL is out: review -9% and VRAM full.
- Stock Q4_K_M is +11% on edit and ties on review, but it was rejected. Over 21 days of Pi sessions (4554 Swift replies), thinking was 64% of Swift's output by characters. Thinking runs at review speed, where the models tie. Swift's card claims 16-58% fewer thinking tokens; stock wins only if that reduction is under ~3%.
- PPL is not comparable across the fine-tune and the base model.

## Notes
- GPU power cap (2026-10-05): the `llm` VM's RX 7900 XTX allows 294–327 W;
  the cap was set by hand to the 294 W minimum. No tok/s impact has been measured.
  `host/gpu-power-cap.service` provides boot persistence; the unit has not been
  installed on any host as part of this change. See the README for installation
  and revert commands.
- The FA sparse-gather path exists but is dead here: the graph always passes `n_kv_max = 0` and the host restricts sparse to f16 KV.
- The full `FLASH_ATTN_EXT` suite in `test-backend-ops` already fails ~1800 cases on this build without the patch (sinks, odd head sizes). The Qwen shapes in `patches/0002` pass, including with packing.
- Remaining per-cycle cost (solo, ~36 ms): weights ~25 ms, dispatch ~7 ms, draft steps ~4 ms. The draft lm_head (q4_0, 248k x 5120) is bandwidth-bound at ~880 GB/s.
- Open ideas: fine-tune the MTP head on real traffic (higher acceptance, est. +15-20%, needs training and a parity harness), and dispatch/launch overhead.
- `hfbench.py` disables thinking, but about two-thirds of real output is thinking, so review SOLO is the closest proxy for real traffic.
