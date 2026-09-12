# goinfer roadmap (rolling; rewritten 2026-09-12)

> **Audience:** internal direction — the *where-to* doc, kept short on purpose. It points at owners
> rather than duplicating them: open work is [`QUEUE.md`](QUEUE.md) and the four queues; the
> performance backlog is [`ollama-chase.md`](ollama-chase.md); measured claims are
> [`benchmarks.md`](benchmarks.md); what runs where is the generated
> [`capability-matrix.md`](capability-matrix.md) / [`hardware-matrix.md`](hardware-matrix.md).
> The June 2026 roadmap this replaces — peer survey, Tracks A/B/C, the KV-memory program, all
> landed or closed — is archived verbatim at
> [`completed/roadmap-2026-06.md`](completed/roadmap-2026-06.md).

## Where we are — v0.17.2, 2026-09-12

- **Runtime.** Pure-Go CPU (NEON / AVX2) plus three GPU backends: WebGPU (`-tags gpu`; the one
  cgo backend, quarantined in its own module), **cgo-free CUDA** (gocudrv, since v0.10) and
  **cgo-free Metal** (purego `objc`). Every tag ships release binaries — six platform targets each
  for `goinfer-chat` and `goinfer-serve`, plus model-embedded chat builds at 0.5B and 1.5B.
- **Coverage.** 36 families; five sequence mixers — softmax GQA, Gated DeltaNet, Mamba-2, MLA,
  KDA (Ling 3.0) — plus dense and sparse MoE; vision-in for Gemma 3 and the Qwen VL pair, no
  audio. Resident on 27 / 26 / 23 families (Metal / CUDA / WebGPU). Loaders: safetensors, GGUF,
  GPTQ, AWQ, `.giw`; fp8 e4m3 reads (blockwise f32 scales).
- **Standings vs Ollama** (dated rows in `benchmarks.md`'s TL;DR): CUDA decode **ahead** on small
  models at short context (1.13× on the 1.5B), **behind at depth** (0.71× by 3900); CUDA prefill
  **1.9–3.2× behind** at depth since the 2026-09-05 tensor-core landing (was 12–15×); Mac CPU
  prefill at parity-to-ahead, Mac CPU decode **0.57–0.77× behind**; **Mac Metal prefill 3.3–8.8×
  behind on TTFT** — the largest remaining gap, and it is on the main development machine.
- **Serving.** Single-request by design: one decode worker per model behind a bounded queue.
  Resident prefix reuse since 2026-09-02 (exact-extension only for recurrent families);
  speculative decoding still off for hybrids pending the state snapshot
  (`task-recompute-audit.md` R-01 phase 1).

## The gate that orders everything

**Owner decision, 2026-09-11: goinfer is not promoted until it is at least as fast as Ollama** on
the machines people actually have. That inverts June's frame — performance leads, breadth is the
tiebreaker — and it is why the programs below are ordered the way they are. Tagging v1.0 and
promoting are separate decisions with separate gates (see the last section).

## Open programs — each owned by a live doc

1. **Prefill gap** — [`task-prefill-gap.md`](task-prefill-gap.md). CUDA: L2 + L3 shipped
   2026-09-05, next lever is P24 (`attn_fused` at 1.72% of tensor peak). Metal: the L1 default flip
   is still open pending one fresh-prompt gate run; L2-Metal fused attention and the W2 Mac peer row
   are unrun. This is the program that moves the headline.
2. **Mac decode** — CPU: aikit's `task-simd-audit.md` S-02 remedies landed (dynamic chunking,
   `MatmulBTW4A8Batch`); the remaining named gap is 4-bit matmul to bandwidth class. Metal: no
   decode campaign is open; the last verdict is `completed/metal-verdict.md`.
3. **MoE over capacity** — P20 (DMA-bound prefill on the streaming path),
   [`task-moe-streaming.md`](task-moe-streaming.md). The 26B/35B streaming path is near its
   structural ceiling on 8 GB; more VRAM, not more kernel, is the lever.
4. **Fit to hardware** — [`task-fit-to-hardware.md`](task-fit-to-hardware.md): phases 0–3 done,
   4 partial (`goinfer-chat fit` dry run and self-measure ship; startup banner and `pull` verdict
   open). `completed/task-model-pull.md` phase 1 shipped.
5. **First hour / cold user** — [`task-embed-and-harness-ux.md`](task-embed-and-harness-ux.md)
   (the five modes of use); the cold-user protocol on published tags; owed: the formal Mac
   two-scenario re-run on v0.17.2.
6. **Peer measurement** — [`task-peer-benchmarks.md`](task-peer-benchmarks.md) (the matrix, with
   the 10-turn agent-turn TTFT as headline) and
   [`task-llamacpp-inproc.md`](task-llamacpp-inproc.md) (in-process llama.cpp via purego, drafted
   2026-09-12 — the fidelity column and per-token cost vs depth).
7. **Speculative decoding** — `spec/` (13 docs). Adaptive verify width failed all four ship gates
   and was removed; the grammar-prior premise is dead; DSpark/DFlash block drafters queued
   (P10/P15); MTP heads (`spec/09`) phase 3 untouched.
8. **Multimodal** — [`multimodal.md`](multimodal.md) P6–P11.
9. **Correctness programs** — [`audit-2026-09-10.md`](audit-2026-09-10.md)'s 8-step program in
   progress; [`task-recompute-audit.md`](task-recompute-audit.md) R-01 phase 1;
   [`task-int4-layout-2026-09.md`](task-int4-layout-2026-09.md) (one int4 layout per tensor, L1–L5)
   and [`task-halt-2026-09.md`](task-halt-2026-09.md) (cancel / halt / lease, K1–K9), both drafted
   this week.
10. **The book** — `book/`, the chaptered inference primer for a Go engineer new to ML.

## Decided and parked — each with its trigger

- **Continuous batching / paged attention** — not this engine's weight class; unchanged since
  v0.2. Its small cousin, N decode workers or batched multi-request decode, gets a kill-or-earn
  measurement before any task doc (decode is bandwidth-bound; a second stream mostly shares the
  same bytes).
- **Bindings** (sidecar for desktop, c-archive for mobile) — scoped in
  [`task-bindings.md`](task-bindings.md), not started. Gate: the B0.1 on-device iPhone spike.
- **Browser / WASM** (`GOOS=js` → `navigator.gpu`, cgo-free) — a demo, not a binding strategy;
  unscheduled.
- **WebGPU dp4a / batched WebGPU prefill** — upstream-blocked on the binding exposing
  `dot4I8Packed`; WebGPU has no batched prefiller today.
- **Native AMD / Intel** — WebGPU is the answer; the reasoning and the AMD/HIP-first gradient are
  in [`gpu-vendor-coverage.md`](gpu-vendor-coverage.md).
- **New families** — [`next-models.md`](next-models.md). DeepSeek V4-Flash is parked as
  unvalidatable on hardware here (G8); a family that cannot get one whole forward on a box we own
  is parked, not queued.
- **int4 KV, TurboQuant (NO-GO), `.giw` f16 scales, W4A8 VNNI** — triggers as written in the June
  archive; none has fired.
- **Pure-Go reference oracle** replacing the `pin_*.py` generators —
  [`task-oracle-refforward.md`](task-oracle-refforward.md); post-1.0 infrastructure, and the
  carve-out named in the v1.0 gate's "no Python in the repo" line.

## Superseded — June claims and what replaced them

Retractions kept visible so the archive is read correctly:

| June 2026 said | Now |
|---|---|
| hybrids (`qwen3_5_moe`) "stay on the staged path" | Gated DeltaNet resident on WebGPU (08-19), CUDA and Metal (08-20); Qwen3.5-MoE / 3.8 / Next resident on all three |
| MoE and Gemma 4 "excluded from residency" | MoE resident on all three backends; Gemma 4 resident on CUDA and Metal (WebGPU declines on per-layer geometry) |
| the resident path is "stateless (no prefix-reuse)" | resident prefix reuse shipped 2026-09-02 (`3358e6b`); agent turn 3 on the 1.5B 9.13 s → 0.42 s |
| a fused W4A8 batch "needs an aikit `MatmulBTW4A8Batch` (doesn't exist yet)" | exists (aikit v1.34.0) and is on goinfer's decode path |
| FP8 "is GPU-hardware-driven — not our fight" | fp8 e4m3 blockwise reader shipped (`decoder/fp8.go`); the `ue8m0` scale format is the remaining half |
| "Llama 4 remains as the last popularity-momentum pick" | `llama4_text` landed (CPU on every backend); its residency lift is scoped in `completed/residency-port-triage.md` |
| "three efficient-attention axes" | five mixers — KDA landed as `bailing_hybrid` (v0.17.0) |
| "one no other pure-Go runtime can make" | `goccy/go-llama` is a real cgo-free lane (single-threaded, GGUF-only, CPU-only); the README stopped claiming the lane is empty 2026-08-27 |
| MTP "stays parked; revisit with a bandwidth-bound (GPU) backend" | two such backends exist; the spec program (`spec/`) is where that revisit lives, and adaptive width already failed its gates |
| "don't grind WebGPU" (~90–100 tok/s ceiling) | withdrawn 2026-09-02 (G35/G36): 118.4 tok/s from a bit-identical reduce fix, 2.39× on the token at 1k from attention |

## The v1.0 question

June's criterion — a second hybrid family lands on the deltanet/hybrid-cache shapes without
breaking them — was met in June. v1.0 is now a checklist, not a feeling:
[`queue-release.md`](queue-release.md) **E1**, one of five boxes ticked (parity coverage complete,
2026-08-15). Open: verification machinery sound; loader and descriptor surface *actually* frozen;
a clean out-of-tree audit against the release candidate; no Python in the repo (E7).
[`api-tiers.md`](api-tiers.md) (signed off 2026-08-18) fixes which surfaces semver-bind. Not
scheduled. It is not gated on performance — that gate is promotion's, above — but there is no
reason to tag 1.0 before there is something to promote.
