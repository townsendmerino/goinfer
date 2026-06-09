# goinfer roadmap (rolling; updated 2026-06-08, post-v0.3.0 → v0.4 planning)

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

### Timely, high-fit openings — BOTH LANDED (detail in Track A)

1. ✅ **Mellum2 (JetBrains)** — 12B MoE / 2.5B active, code-focused. Fully
   closed: parity golden, chat template, window eviction.
2. ✅ **Qwen 3.6 (35B-A3B)** — landed as the largest forward-pass addition to
   date (Gated DeltaNet hybrid). **Real-checkpoint int8 full-model parity
   validated** (Gate 2, `e3eb033`). Remaining before a headline "Qwen 3.6
   support" claim: the GGUF loader (loads safetensors today). See Track A.

## v0.4 — theme: "new models + serve polish"

### Track A: model freshness — LANDED (2026-06-07), with follow-ons

- ✅ **Mellum2** (`f70cf51`, `4b0e7be`) — fully closed: chat template
  (named-ChatML alias, byte-exact golden), HF logit-parity golden,
  sliding-window eviction exercised.
- ✅ **Qwen 3.5/3.6-MoE** (`f70c013`) — **the largest forward-pass addition
  to date**: Gated DeltaNet hybrid (3:1 linear/softmax attention), new
  `deltanet.go` sequence-mixing primitive, hybrid cache (per-layer
  deltaState alongside KV), 256-expert + shared MoE. Bit-exact vs the HF
  oracle (argmax + cosine 1.0) on a generated tiny checkpoint.
- ✅ **Real-checkpoint parity — DONE** (`e3eb033`, Gate 2 GREEN). The real
  Qwen3.6-35B-A3B ships as a **VL model**; goinfer loads its **text decoder**
  (`language_model.` prefix + fused/stacked 256-expert MoE unpack; vision/MTP
  ignored). Gate 1 (slice, f32) cosine 1.0 bit-exact; **Gate 2 (full 40-layer,
  int8, ~39 GB resident via streaming-quant fused experts, vs bf16 golden):
  argmax 74/80 (92.5%), 8/10 prompts reproduce the golden continuation, logit
  cosine min 0.99466 / mean 0.99859** (both misses near-ties, <3% of range).
  **It runs at int8 from safetensors** — not f32-only.
- **Open follow-ons** (Track A isn't *practically* done until these land):
  - [ ] **GGUF loading for `qwen3_5_moe`** — loads safetensors today; the GGUF
        path (the common distribution format) + a chunked DeltaNet scan are
        the named follow-ons. This is the gate before "Qwen 3.6 support" is a
        release headline.
  - [x] **Hybrid models opt out of prefix-reuse and speculative** — the
        recurrent deltaState isn't position-truncatable; fall back to the
        staged path, documented. (A deltaState snapshot-at-position scheme is
        possible but priced separately.)
- Process note (validated twice now): each family lands as descriptor +
  loader + parity golden, and the whole v0.2/v0.3 surface (templates,
  tools, constrain, serve) inherits automatically. The moat compounds.

### Track B: serve polish — LANDED (in CHANGELOG `[Unreleased]`)

`cmd/serve` grew from one-model/Chat-Completions to a model host, pure stdlib:

- ✅ **Multi-model** — `--model name=path` repeatable; requests route on the
  OpenAI `model` field; `/v1/models` lists all; per-model mutex (distinct
  models run in parallel) + per-model warm-KV dir.
- ✅ **Admin API** (`--allow-admin`, default off — RCE-adjacent):
  `POST /admin/models/{load,unload}`; unload refuses a busy model (409) and
  snapshots its warm KV.
- ✅ **OpenAI Responses API** (`/v1/responses`): `input`/`instructions`/
  `text.format`(→constrain)/`tools`, streaming event shapes,
  `store`/`previous_response_id` (in-memory ring riding the per-model
  sessionLRU).
- ✅ **Backpressure** — bounded per-model queue (`--max-queue`, default 8) →
  429 + Retry-After when full (single decode worker/model; not continuous
  batching, which stays deferred).

### Track C (unplanned): GPU decode investigation — CLOSED (2026-06-08)

**Full writeup + the decision matrix: `docs/gpu-assessment.md` (§0.0, §1).**

The arc that began as "is the staged hybrid optimal?" closed in the opposite
place: a **full-token on-GPU residency forward** wins. Shipped:

- **GPU full-residency decode wired into `decoder.Generate`** (webgpu backend,
  dense Qwen2/Llama). Real int4 `.giw` models now run **pure-GPU** end-to-end,
  not the per-matmul staged path. Greedy output matches the CPU decode.
- **W4A8 (int4) is the capability unlock:** a **7B int4 fits and runs pure-GPU
  on the 8 GB card at ~51 tok/s** — the model class that does NOT fit at int8 —
  at **~71% of llama.cpp-CUDA** tok/s at equal 4-bit quant (1.5B: 102 tok/s,
  ~55%). The engine gap *narrows* with model size; this is a footprint/capability
  win, not a speed record (the WebGPU dispatch model still trails CUDA's
  megakernel). int8 decode peaks ~89.7 tok/s on the 1.5B (3.5× the staged hybrid).

Deferred follow-ons (named, **not scheduled** — pick up if a use case pulls):

- **Batched on-device GPU prefill** — today's prefill is option-(a) sequential
  GPU `Run`, O(prompt-len); a batched M=len pass would fix long-prompt TTFT.
- **f16 KV cache** — unlocks 32k context for a 7B on 8 GB (v1 caps at 16k, f32).
- **CPU-int4 decode — RESOLVED (2026-06-08, aikit v1.1.1).** The §1 matrix
  measured CPU int4 at ~0.15–0.18 tok/s because `MatmulBTQ4` (f32 activation) is
  dequant-bound at M=1. aikit v1.1.1's `MatmulBTW4A8` (int4×int8 integer decode
  kernel) replaced it for M=1 decode (`MatmulBTQ4` stays prefill); re-measured at
  **2.1–4.3 tok/s, 1.4–1.9× of CPU int8int8** — CPU int4 is now usable, not only a
  GPU footprint win.
- **W4A8-on-disk format** — f16 group scales in the `.giw` (smaller files; the
  kernel is ALU-bound so it wouldn't speed decode).

**Pivot (updated 2026-06-08):** the GPU arc is closed AND Track A/B largely
landed. The one real open item across both tracks is the **`qwen3_5_moe` GGUF
loader** (it runs int8 from safetensors today; GGUF is the distribution-format
gate). Smaller carried items: true incremental tool-call streaming (today
buffered-then-deltas); N decode workers per model (today one).

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
