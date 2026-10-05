# Ajax Optimization

llama.cpp serving configuration, Vulkan kernel changes and benchmarks for running
Qwen3.8-27B (UD-Q4_K_M + MTP draft + ngram-mod) on a single AMD RX 7900 XTX (RADV, Vulkan).
The repository also hosts [SaySo training and wake tooling](sayso-training/README.md).

Serving-only goal and constraints: maximum decode tok/s inside a fixed layout — 4 slots sharing one 128k unified KV pool, q8_0 KV,
all layers on GPU. No CPU or RAM offload, and no host-side work added.

## Layout

| Path | What |
|---|---|
| `patches/0001-*.patch` | Vulkan changes on upstream llama.cpp `d280808f5`: scalar flash-attention (FA2) tuning and GQA **token packing**. Also carries a small unrelated `qwen35*.cpp` reshape change that the prod image was built with. |
| `patches/0002-*.patch` | `test-backend-ops` cases for the Qwen decode/verify shapes (perf + correctness). |
| `tools/dump_h.cpp` | Dumps the target model's pre-norm hidden states per token (`h_nextn`). Groundwork for an MTP-head fine-tune; not used in serving. |
| `compose/llama-compose.yml` | Production llama-server config (flags, image, sampling defaults). |
| `bench/` | Bench and profiling scripts (see below). |
| `docs/RESULTS.md` | Measurements, what worked, what was rejected. |
| `sayso-training/` | Imported SaySo training, generation, and wake tooling; shared runtime code comes from its pinned `vendor/SaySo` submodule. |

Initialize the shared SaySo checkout with
`git submodule update --init sayso-training/vendor/SaySo`.
The imported `sayso-training/scripts/llm-host/llama-compose.yml` overlaps
`compose/llama-compose.yml`; these configurations have not been merged.

## Reproduce the kernel change

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout d280808f5
git apply ../ajax-optimization/patches/0001-vulkan-fa2-and-gqa-token-packing.patch
git apply ../ajax-optimization/patches/0002-test-backend-ops-qwen-fa-cases.patch
docker build -f .devops/vulkan.Dockerfile --target server -t sayso/llama.cpp:d280808-fa2pack .
```

`GGML_VK_FA_TOK_PACK=1` disables token packing at runtime; `=N` forces N tokens per tile.

## Benchmarks

Everything runs on the GPU host inside a lock turn (`gpu run <label> -- <cmd>`); prod is stopped for the duration.

- `bench/knobbench.py` — SOLO and PAIR (two streams, ~30k depth each) decode at a fixed depth.
- `bench/fapack_ab.sh`, `bench/fa2rebase.sh` — A/B two images, alternating.
- `bench/fa_perf.sh` / `bench/fa_correctness.sh` — flash-attention microbench and CPU-reference check in a `test-backend-ops` container (needs the tests build, see `patches/0002`).
- `bench/pairprof.sh`, `bench/soloprof.sh` — per-op Vulkan timings (`GGML_VK_PERF_LOGGER`).
- `bench/mtp_ab.sh`, `bench/quant_ab.sh`, `bench/slotbench.sh`, `bench/specbench.sh`, `bench/cachebench.sh` — draft head, quant, slot layout, speculative and cache experiments.

Paths in the scripts (`/srv/llm/...`) are the GPU host's; adjust for another machine.
