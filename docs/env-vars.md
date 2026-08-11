# Environment variable registry

**Convention: every goinfer environment variable is prefixed `GOINFER_`.** (The stray `GINFER_`
prefix — 10 test/bench knobs — was consolidated into `GOINFER_` on 2026-08-11; `GINFER_SPEC_TARGET`
and `GINFER_TEST_MODEL` merged into their pre-existing `GOINFER_` twins, which the decoder and gpu
test suites had been setting inconsistently.)

This registry curates the **operator-facing** knobs — the ones you might set when running or tuning a
deployment. The full set (≈130, most of them per-family test-model overrides and diagnostic probes)
is grep-derivable and enumerated at the bottom.

## Serving

| Var | Purpose |
|---|---|
| `GOINFER_API_KEY` | Bearer token the OpenAI-compatible server requires on requests (unset = open). |

(Most serve configuration is CLI flags on `cmd/serve`, not env vars — see the README Build section.)

## Residency & MoE (device-memory tuning)

| Var | Purpose |
|---|---|
| `GOINFER_NO_RESIDENCY` | Force the staged (non-resident) GPU path — disables whole-model device residency. |
| `GOINFER_GEMMA4_RESIDENT` | Opt Gemma-4 into the resident path (bring-up gate; now default, kept as an override). |
| `GOINFER_MOE_CACHE_SLOTS` / `GOINFER_MOE_CACHE_EXPERTS` | Size the resident MoE expert-slot cache (VRAM ↔ per-token DMAs trade). |
| `GOINFER_MOE_NOCACHE` | Disable the MoE expert cache (always stage experts per token). |
| `GOINFER_MOE_WILLNEED` / `GOINFER_MOE_PREAD` | MoE expert-paging readahead strategy (madvise WILLNEED / pread). |
| `GOINFER_METAL_MOE_SLOTS` | Metal resident MoE slot count. |
| `GOINFER_SPLITKV_MIN_KEYS` | Override the split-KV decode-attention key-count threshold (per-geometry default otherwise). |

## Decode-path escape hatches (default is the fast/bit-exact path; these opt out for A/B or safety)

| Var | Purpose |
|---|---|
| `GOINFER_NO_GREEDY_FASTPATH` | Disable the on-device greedy argmax fast path (force full-logits readback). |
| `GOINFER_NO_KVONLY_PREFILL` | Disable KV-only prefill (run the full-logits prefill on every prompt token). |
| `GOINFER_BATCHED_PREFILL` | Toggle the batched prefill path (`=0` disables). |
| `GOINFER_METAL_BATCHED_PREFILL` | Opt into Metal batched prefill — NOT bit-identical to decode (54% divergence); measurement only. |
| `GOINFER_INT4_SLOWPATH` / `GOINFER_INT4_F16_SCALES` | int4 unpack path selectors. |
| `GOINFER_CUDA_NO_FUSE` | Disable CUDA kernel fusion (debug/A-B). |
| `GOINFER_MLA_NAIVE` | Use the naive (un-optimized) MLA attention path. |

## CUDA graphs (perf, opt-in)

| Var | Purpose |
|---|---|
| `GOINFER_CUDA_GRAPHS` | Enable CUDA graph capture/replay on the static decode segments. |
| `GOINFER_CUDA_GRAPHS_ONLY` / `_SYNC` / `_UNSAFE` | Graph-capture scope / sync / bounds-check-elision variants (investigation flags). |

## Diagnostics, probes & parity debugging

| Var | Purpose |
|---|---|
| `GOINFER_DECODE_TIMING` | Per-phase decode timing on the serve/decode path. |
| `GOINFER_DUMP_LOGITS` | Dump logits for cross-checking. |
| `GOINFER_MEM_PROBE` | RSS/heap attribution probe around the test suite. |
| `GOINFER_FAKEQUANT` (`_ACT` / `_EXPERTS` / `_PERROW`) | Parity-debugging: simulate a quantization on the f32 path to isolate a numeric gap. |
| `GOINFER_G4_CAPTURE` / `GOINFER_MOE_PROF_SPLIT` | Gemma-4 / MoE internal capture + profiling splits. |
| `GOINFER_RESIDENT_DEBUG` | Print why `BuildResident` declined (module/device/arch). |

## CI & test gates

| Var | Purpose |
|---|---|
| `GOINFER_HEAVY_TESTS=1` | Opt into the ~120 heavy (real-checkpoint) tests that `go test ./...` otherwise skips. |
| `GOINFER_MANIFEST_EMIT` / `GOINFER_MANIFEST_MACHINE` | Emit parity-manifest rows / stamp the generating machine. |
| `GOINFER_PAR_THRESHOLD` / `GOINFER_PAR_WIDTH` | Sweep the CPU matmul parallel threshold (MACs) / fan-out width. |

## Test-model overrides (a large class — point tests at your local checkpoints)

Generic: `GOINFER_TEST_MODEL`, `GOINFER_PREQUANT_GGUF`, `GOINFER_MODELS_DIR`, `GOINFER_CUDA_MODEL`,
`GOINFER_METAL_MODEL`, `GOINFER_E2E_MODEL`, `GOINFER_EMBED_MODEL`. Per-family: one `GOINFER_<FAMILY>_GGUF`
(and occasional `_DIR`/`_INT4`/`_LABEL`) for deepseek, glm, granite, gptoss, llama3/4, nemotron, phi3,
gemma3/4, cohere/aya, mellum, mla, moe*, eagle/spec, giw, chatml, matrix, cpuint4, … — each skips its
test when unset.

## The exhaustive list

This registry is curated, not generated. To enumerate every variable the code reads:

```sh
grep -rhoE 'GOINFER_[A-Z0-9_]+' --include='*.go' . | sort -u
```
