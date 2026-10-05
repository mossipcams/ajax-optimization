# Results

All numbers: RX 7900 XTX, Qwen3.8-27B UD-Q4_K_M, MTP n=3 + ngram-mod, 4 slots / 128k unified, q8_0 KV.
"SOLO" = one stream at ~30k depth, "PAIR" = two concurrent streams at ~30k each (combined tok/s).

## Deployed

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
- Layout changes (2 x 64k slots, non-unified KV) are faster for pairs but violate the fixed-layout constraint.

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
