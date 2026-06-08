# goinfer feature plan — v0.4 planning (updated 2026-06-07, post-v0.3.0)

> **Audience:** internal planning doc (docs/internal/ is gitignored). Started
> as the v0.2 gap analysis vs llama.cpp / Ollama / mistral.rs; v0.2.0 and
> v0.3.0 both shipped everything it scoped. This revision: fresh peer survey
> (2026-06-07) + the v0.4 sketch. Historical v0.2/v0.3 detail lives in
> CHANGELOG.md and the git log; the original survey conclusions are at the
> bottom.

## State of play

**v0.3.0 is tagged and released** (release notes:
`docs/internal/release-notes-v0.3.0.md`). Cumulative since v0.1.3, goinfer
gained: Gemma 4 (E2B/E4B/12B, parity-gated), sampler completeness (min-p,
penalties, logit bias, logprobs), the `chat` package (template detection +
byte-exact native renderers), JSON Schema constrained decoding
(`GrammarFromStruct` — the flagship), tool calling (per-family render +
parse + grammar-constrained calls), `cmd/serve` (OpenAI-compatible, pure
stdlib, warm KV sessions, `/v1/embeddings`), `decoder.Session`
prefix caching with `.giw-kv` snapshots, LoRA adapter loading, Q6_K/Q4_K_S
parity pins, and aikit v1.0.0 across both modules.

**Every item from the original gap table is closed** except the two
deliberate deferrals: continuous batching/PagedAttention and multimodal.
The catch-up phase is over; v0.4 should play offense.

## Fresh peer survey (2026-06-07)

The field's competition moved while we closed the library gap. Five
battlegrounds now:

| Battleground | Who's pushing | goinfer position |
|---|---|---|
| **MTP speculative decoding** (models ship multi-token heads; no draft model; ~2x on dense) | llama.cpp (Qwen 3.6 MTP, PR #22673), Ollama 0.23.1 (Gemma 4 MTP via MLX, 2x on 31B), LM Studio 0.4.14 (stable), vLLM (Gemma 4 MTP; EAGLE 3.1 in v0.22) | Greedy draft-model spec exists, parked on CPU. **MTP doesn't change the math**: CPU decode is compute-bound, so an M=K verify pass costs ~K sequential decodes regardless of how cheap drafting is. Stays parked; revisit with a bandwidth-bound (GPU) backend. Caveat from the field: even llama.cpp sees *no* MoE single-stream win on consumer hardware. |
| **Model freshness** | Qwen 3.6 (dense 27B + 35B-A3B MoE), DeepSeek V4, GLM-4 MoE, Gemma 3n, Nemotron 3, Mellum2 | This is the race the descriptor design lets us run cheaply — and the one to enter (see v0.4). Gemma 4 already in, incl. PLE (which llama.cpp struggled with). |
| **Hardware acceleration depth** | CUDA kernel fusion (+24% on 4090), Vulkan now beating CUDA on some hardware, M5 Neural Accelerators (MLX-only, up to 4x TTFT) | Structural gap, accepted trade. Pure-Go CPU SIMD + opt-in WebGPU. The counter-position *is* portability: one static binary, every platform, no driver. Don't fight on kernels. |
| **Serving surface** | mistral.rs v0.7: dynamic model load/unload, **multi-model serving**, initial **OpenAI Responses API**, prefix caching; LM Studio: MCP host + OAuth; Ollama: `launch` app integrations | `cmd/serve` is one model per process, Chat Completions + embeddings. Prefix caching: at parity (landed same season as mistral.rs's). Multi-model + Responses API are the live gaps. |
| **Quantization frontier** | Native FP8/FP4 (DeepSeek V4 in llama.cpp), TurboQuant (4.9x vs f16, tracked in llama.cpp #20969), NVFP4 on Apple | int8/int4 runtime + broad K-quant dequant coverage. FP8/FP4 is GPU-hardware-driven — not our fight on CPU. Watch TurboQuant: a CPU-implementable 3–4 bit quant with near-paper MSE could matter for the embed-demo size/quality curve. |

### Timely, high-fit openings

1. **Mellum2 (JetBrains, released 2026-06-06)** — 12B MoE, **2.5B active**,
   Apache 2.0, code-focused. We already run Mellum 1 *and* two MoE families
   (Mixtral, Qwen2-MoE). 2.5B active is the right compute envelope for CPU
   decode, and "first pure-Go runtime to run Mellum2" is a fast, ownable
   claim with a JetBrains-audience hook (Go developers, IDE crowd).
2. **Qwen 3.6** — the model the entire field optimized for in May. Dense
   27B is the MTP-era flagship; the 35B-A3B MoE is the local-inference
   darling. If the architecture is descriptor-close to Qwen3 (likely),
   support is high-leverage. The Coder variants slot straight into the
   demo/serve story.

## v0.4 — theme: "new models + serve polish"

### Track A: model freshness — LANDED (2026-06-07), with follow-ons

- ✅ **Mellum2** (`f70cf51`, `4b0e7be`) — fully closed per
  `task-mellum2-close.md`: chat template (named-ChatML alias, byte-exact
  golden), HF logit-parity golden, sliding-window eviction exercised.
- ✅ **Qwen 3.5/3.6-MoE** (`f70c013`) — **the largest forward-pass addition
  to date**: Gated DeltaNet hybrid (3:1 linear/softmax attention), new
  `deltanet.go` sequence-mixing primitive, hybrid cache (per-layer
  deltaState alongside KV), 256-expert + shared MoE. Bit-exact vs the HF
  oracle (argmax + cosine 1.0) on a generated tiny checkpoint.
- **Open follow-ons from the commit itself** (Track A isn't *practically*
  done until these land):
  - [ ] GGUF loading for `qwen3_5_moe` + int8/int4 quant path (parity-first
        f32 only today) — without it the real 35B-A3B isn't runnable at
        size.
  - [ ] Real-checkpoint parity (the tiny generated model proves the
        algorithm; the released weights prove the loader).
  - [ ] **Hybrid models opt out of prefix-reuse and speculative** (the
        recurrent deltaState isn't position-truncatable — fallback is
        correct but silent). Document the limitation in Session/serve
        docs; a deltaState snapshot-at-position scheme is possible but
        priced separately.
  - [ ] Chunked/parallel DeltaNet scan for prefill throughput.
- Process note (validated twice now): each family lands as descriptor +
  loader + parity golden, and the whole v0.2/v0.3 surface (templates,
  tools, constrain, serve) inherits automatically. The moat compounds.

### Track B: serve polish — IN PROGRESS (execution doc: `task-serve-polish.md`; commits `3baead4` → `b516209` held locally)

### Track C (unplanned, landed): GPU decode investigation — REOPENED

Seven commits (`7ad1fd5` → `18cd52d`, held locally) executed the
`gpu-assessment.md` Stages 1–3: W8A8 WGSL bit-exact, staged shape 4× CPU
on the FFN block, full fusion measured slower. **Roofline review against
the actual hardware (RTX 2070 SUPER vs 3700X: ceiling ~8–10×) showed the
measured 1.83× decode is ~20% of the card — a software gap (sync/submission
structure), not physics.** Next: the one-command-buffer-per-token
experiment + llama.cpp CUDA/Vulkan calibration on the same box. See
`gpu-assessment.md` §0.5.

- **Multi-model serving + dynamic load/unload** — serve N models from one
  process; `/v1/models` lists them, requests route by `model` field;
  load/unload at runtime (mistral.rs v0.7's headline, Ollama's core UX).
  Fits the one-binary story: one goinfer process = a local model zoo.
  Memory discipline via prequant `.giw` mapping keeps N models cheap.
- **OpenAI Responses API** (`/v1/responses`) — the API surface is shifting
  under Chat Completions; mistral.rs already started. Getting in early
  keeps "point your SDK at goinfer" true through 2026.
- Carried from last revision (as usage appears): request queue + N decode
  workers (today: one mutex); true incremental tool-call streaming (today:
  buffered, then deltas).

### Watching, still not doing

- MTP speculative — parked (compute-bound CPU; see survey). Re-evaluate the
  moment the WebGPU backend graduates from matmul-only.
- TurboQuant — track #20969; CPU-friendly, could earn a spot if it survives
  scrutiny.
- Continuous batching / PagedAttention, multimodal — unchanged deferrals.

### The v1.0 question (rolling — updated 2026-06-07)

Track A's verdict cuts both ways. Mellum2 landed clean (descriptor held —
good evidence). But Qwen 3.6 *grew* the surface by its largest increment
ever: a second sequence-mixing primitive, a second cache type, per-layer-kind
descriptors. The loader/descriptor contract now contains its newest,
least-settled member — and hybrid/linear-attention models are clearly a
trend (Qwen, Granite Hybrid, GLM), not a one-off. Freezing now would lock
day-one shapes for the hybrid abstraction. Revised criterion: v1.0 waits
until a *second* hybrid family lands on the existing deltanet/hybrid-cache
shapes without breaking them — same bar the transformer descriptor already
passed.

---

## Appendix: original peer survey (2026-06-06, pre-v0.2)

Kept for the record — this table drove the v0.2/v0.3 scope. Every "✗ then"
cell is now ✓ except the two deliberate deferrals.

| Feature | llama.cpp | Ollama | mistral.rs | goinfer then | now |
|---|---|---|---|---|---|
| min-p sampling | ✓ | ✓ | ✓ | ✗ | ✓ |
| repetition/presence/frequency penalties | ✓ | ✓ | ✓ | ✗ | ✓ |
| logit bias / logprobs | ✓ | ✓ | ✓ | ✗ | ✓ |
| chat template from model metadata | ✓ | ✓ | ✓ | demo heuristic | ✓ |
| JSON **Schema** constrained output | ✓ | ✓ | ✓ | JSON only | ✓ |
| OpenAI-compatible HTTP server | ✓ | ✓ | ✓ | ✗ | ✓ |
| tool calling | ✓ | ✓ | ✓ (+MCP) | ✗ | ✓ |
| prefix/prompt caching | ✓ | ✓ | ✓ | `TruncateTo` only | ✓ |
| LoRA adapters | ✓ | ✓ | ✓ | ✗ | ✓ |
| K-quants beyond Q4_K_M | ✓ | ✓ | n/a | ✗ | ✓ |
| speculative decoding | ✓ (variants) | ✗→MTP | ✓ | ✓ greedy | ✓ greedy |
| continuous batching / PagedAttention | ✓ | ✓ | ✓ | ✗ | deferred |
| multimodal | ✓ | ✓ | ✓ | ✗ | deferred |

### Positioning (unchanged, still the filter)

The "Go LLM inference" field is purego/cgo *bindings* to llama.cpp
(gollama.cpp, yzma) — nobody else is a real pure-Go runtime. goinfer wins by
being the best **embeddable Go library**: idiomatic API, single static
binary, parity-gated numerics, structured outputs straight into Go structs —
not by chasing kernel throughput.

### Survey sources

- 2026-06-07: codersera Local AI Runtime Update (May 2026 — Ollama 0.23/0.24,
  vLLM 0.21, llama.cpp MTP b9196+, MLX/M5, LM Studio 0.4.14);
  mistral.rs v0.7.0 release notes; Mellum2 announcement coverage.
- 2026-06-06: llama.cpp server/speculative docs; Ollama structured-outputs +
  streaming-tool blogs; mistral.rs README; gollama.cpp / yzma repos.
