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
| `GOINFER_P13_OFF` | Keep the safetensors SOURCE mapping resident for the model's life, as the loader did before P13. The loader now closes it at end of load when no tensor dtype can alias it (BF16/F16 widen on read; anything else may be a zero-copy view). Set this only to diagnose a suspected use-after-free, or to reproduce the old memory profile — it is the control arm the P13 measurement used. |
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

## Prefill/MoE defaults (2026-08-31 → 09-01) — the four default-ON changes and their opt-outs

**These four turned ON by operator decision and every one of them can change greedy output.** The
only contract naming them used to be a CHANGELOG bullet (N-42). Each is a developer A/B handle
rather than a user setting, and each is listed here because an operator who sees an output change
after upgrading needs one place to look.

| Var | Default | Opt out with | What it changes |
|---|---|---|---|
| `GOINFER_CPU_FAST_ATTENTION` | ON | `=0` | f32 prefill attention above a 512-token suffix. NOT split-invariant: a warm session's long suffix can differ in the last ulps from a cold prefill, which at temperature 0 can flip a near-tie. `serve` sets this from `--cpu-exact-prefill`. |
| `GOINFER_FUSED_ATTENTION` | ON | `=0` | FlashAttention-style fused prefill schedule (P19). Re-associates the softmax denominator, so it is not bit-identical to the materialized path. |
| `GOINFER_MOE_EXPERT_MAJOR` | ON | `=0` | Expert-major MoE MLP traversal (P18). |
| `GOINFER_NO_OPTFWD` | (unset) | `=1` | Disables the optimistic-forward speculation gate. Same convention as `GOINFER_NO_GREEDY_FASTPATH`. `GOINFER_OPTFWD_MAX_TEMP` bounds the temperature at which it engages. |

## Operator knobs not in the sections above

| Var | Purpose |
|---|---|
| `GOINFER_NO_RESIDENT_MEM_GUARD` | Bypass Metal's resident memory guard — the only remedy the decline message prints. |
| `GOINFER_SSM_RESIDENT` | Opt Granite/Nemotron SSM layers into the resident path. |
| `GOINFER_SPLITKV_ATTN` | CUDA split-KV attention rollback knob (sibling of `GOINFER_SPLITKV_MIN_KEYS`). |
| `GOINFER_MODELS` / `GOINFER_MODEL_TMP` | Model asset root / temp dir for downloads. |
| `GOINFER_PREFILL_ATTN_WORKERS` / `GOINFER_ATTN_ROW_TILE` | Prefill attention fan-out, row tile. |
| `GOINFER_ATTN_KEYS` | WebGPU **decode** key-split attention kernel kill switch — set to `0` to force the old dim-split kernel instead (A/B in the same binary). Not a prefill knob and not a key count (V-23, docs/review-2026-09-04.md). |
| `GOINFER_W4A8_SPLITHALF` | Select the split-half W4A8 kernel. |
| `GOINFER_W4A8_BATCH` | Opt into the fused q/k/v and gate/up batched W4A8 matmul (audit R-06). Default off — measured 1.08x decode on this box, ambiguous against the 1.05x park / 1.15x ship bar, so it ships parked rather than as the default. |
| `GOINFER_MOE_RESIDENCY` / `GOINFER_MOE_RESIDENCY_SCOPE` | Metal MoE residency mode and scope. |
| `GOINFER_PRECISE_MATH` | Metal: precise (non-fast) math in generated shaders. |
| `GOINFER_NO_RESIDENT_REUSE` | Disable resident-forward reuse across requests. |
| `GOINFER_NVRTC_DIRS` | Extra directories to search for NVRTC when building CUDA kernels. |
| `GOINFER_CUDA_GRAPHS_SYNC` | Force a synchronize around CUDA graph launches. |

## Diagnostics and experiment knobs — NOT contract

Listed so the registry is complete and `TestEnvVars_docAndCodeAgree` has somewhere to put things
that are not operator-facing. These may change or disappear without notice:

`GOINFER_A10_PROBE`, `GOINFER_DELTANET_TIMING`, `GOINFER_ROUTER_CAPTURE`,
`GOINFER_MOE_PREFILL_SCRATCH`, `GOINFER_MOE_CACHE_PROF`,
`GOINFER_FAKEQUANT_ACT`, `GOINFER_FAKEQUANT_EXPERTS`, `GOINFER_FAKEQUANT_PERROW`,
`GOINFER_SSM_W8A16`, `GOINFER_SSM_F16MAMBA`, `GOINFER_SSM_NOMUL`, `GOINFER_SSM_Q8CPU`,
`GOINFER_SSM_SKIPFFN`, `GOINFER_SSM_STOP_LAYER`.

Gate/CI knobs read by `cmd/gate` and the harnesses: `GOINFER_GATE_BACKEND`,
`GOINFER_GATE_HEARTBEAT`, `GOINFER_GATE_SKIP_HEAVY`, `GOINFER_GATE_SKIP_WEBGPU`,
`GOINFER_REQUIRE_FIXTURES`, `GOINFER_TEST_NOTHINK`, `GOINFER_SPEC_PROBE_GIW`.

## CUDA graphs (perf, opt-in)

| Var | Purpose |
|---|---|
| `GOINFER_CUDA_GRAPHS` | Enable CUDA graph capture/replay on the static decode segments. |
| `GOINFER_CUDA_GRAPHS_ONLY` | Restrict graph capture to a scope (investigation flag). |
| `GOINFER_CUDA_GRAPHS_SYNC` | Force a synchronize around graph launches (investigation flag). |
| `GOINFER_CUDA_GRAPHS_UNSAFE` | Bounds-check-elision variant. **It also BYPASSES THE TENANCY GATE** (`cuda/graphs_safe.go`), so it is not a pure performance switch — N-42. |

## Diagnostics, probes & parity debugging

| Var | Purpose |
|---|---|
| `GOINFER_DECODE_TIMING` | Per-phase decode timing on the serve/decode path. |
| `GOINFER_DUMP_LOGITS` | Dump logits for cross-checking. |
| `GOINFER_MEM_PROBE` | RSS/heap attribution probe around the test suite. |
| `GOINFER_FAKEQUANT` (`_ACT` / `_EXPERTS` / `_PERROW`) | Parity-debugging: simulate a quantization on the f32 path to isolate a numeric gap. |
| `GOINFER_G4_CAPTURE` / `GOINFER_MOE_PROF_SPLIT` | Gemma-4 / MoE internal capture + profiling splits. |


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

This registry is curated, not generated — but it is no longer allowed to drift from the code.
`TestEnvVars_docAndCodeAgree` (decoder/envvars_registry_test.go) fails when a variable is read by
non-test code and missing here, or documented here and referenced by no `.go` file at all. Both
directions had drifted: 40 variables were missing (including the escape hatches for all four
default-ON changes), one was documented while nothing in the tree read it, and three CUDA graph
knobs were written as suffix shorthand so their full names never appeared at all (N-42).

To enumerate every variable the code reads:

```sh
grep -rhoE 'GOINFER_[A-Z0-9_]+' --include='*.go' . | sort -u
```
