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
  audio. Resident on 26 / 28 / 23 families (Metal / CUDA / WebGPU; `hardware-matrix.md`, 2026-09-25 — Phi-3 / Phi-4 now decline every GPU backend, queue-engineering.md H2). Loaders: safetensors, GGUF,
  GPTQ, AWQ, `.giw`; fp8 e4m3 reads (blockwise f32 scales).
- **Standings vs Ollama** (the pre-registered sweep, nobara half re-run 2026-09-29 at `754f12d3`,
  `measurements/peer-sweep-2026-09-29.md`; dated rows in `benchmarks.md`'s TL;DR): CUDA greedy decode **AHEAD in all 12 cells**
  (0.5B / 1.5B / 7B, 128 to 8000 tokens; 1.05–1.47×), none void; CUDA TTFT on the 1.5B far ahead at K=512 and level at 3900;
  CPU amd64 decode behind on the 0.5B only (0.91×; 1.5B level, 7B ahead); **Phi-3 on CUDA behind** at its real `q4k` default
  (0.90× / 0.72× at 128 / 3900); a top-p sampling regression on CUDA, found by the sweep and fixed 2026-09-30. Mac Metal decode: the 2026-09-25
  sweep had it behind at every depth ≥ 2048 (0.58–0.75×); after R18b a same-session run, not pre-registered, put it **ahead at
  every depth** (1.03–1.19×); the Mac half of the re-run (2026-09-30) is partial, with cell g one cell (0.5B at 128, AHEAD 1.296×) and
  cell i not run. Mac Metal TTFT, 1.05× / 1.96× behind at K=512 / 3900 after R16 (2026-09-25), is **level at K=3900 (0.983) and
  AMBIGUOUS-HIGH at K=512** after R19 (cell h, 2026-09-30); Mac CPU decode unresolved (0.5B AMBIGUOUS-LOW, 1.5B void). Mac CPU
  prefill at parity-to-ahead (2026-09 row, not re-measured). *(Until 2026-09-25 this line read "ahead on
  small models at short context (1.13×), behind at depth (0.71× by 3900)" and "Mac CPU decode 0.57–0.77×
  behind"; both are superseded.)*
- **Serving.** Single-request by design: one decode worker per model behind a bounded queue.
  Resident prefix reuse since 2026-09-02 (exact-extension only for recurrent families);
  speculative decoding still off for hybrids pending the state snapshot
  (`tasks/task-recompute-audit.md` R-01 phase 1).

## The gate that orders everything

**Owner decision, 2026-09-11: goinfer is not promoted until it is at least as fast as Ollama** on
the machines people actually have. That inverts June's frame — performance leads, breadth is the
tiebreaker — and it is why the programs below are ordered the way they are. Tagging v1.0 and
promoting are separate decisions with separate gates (see the last section).

## Open programs — each owned by a live doc

1. **Prefill gap** — [`task-prefill-gap.md`](completed/task-prefill-gap.md) (archived, COMPLETE). CUDA: L2 + L3 shipped
   2026-09-05, next lever is P24 (`attn_fused` at 1.72% of tensor peak). Metal: the L1 default flip
   is still open pending one fresh-prompt gate run; L2-Metal fused attention and the W2 Mac peer row
   are unrun. This is the program that moves the headline.
2. **Mac decode** — CPU: aikit's `task-simd-audit.md` S-02 remedies landed (dynamic chunking,
   `MatmulBTW4A8Batch`); the remaining named gap is 4-bit matmul to bandwidth class. Metal: no
   decode campaign is open; the last verdict is `completed/metal-verdict.md`.
3. **MoE over capacity** — P20 (DMA-bound prefill on the streaming path),
   [`task-moe-streaming.md`](completed/task-moe-streaming.md). The 26B/35B streaming path is near its
   structural ceiling on 8 GB; more VRAM, not more kernel, is the lever.
4. **Fit to hardware** — [`tasks/task-fit-to-hardware.md`](tasks/task-fit-to-hardware.md): phases 0–3 done,
   4 partial (`goinfer-chat fit` dry run and self-measure ship; startup banner and `pull` verdict
   open). `completed/task-model-pull.md` phase 1 shipped.
5. **First hour / cold user** — [`task-embed-and-harness-ux.md`](tasks/task-embed-and-harness-ux.md)
   (the five modes of use); the cold-user protocol on published tags; owed: the formal Mac
   two-scenario re-run on v0.17.2.
6. **Peer measurement** — [`task-peer-benchmarks.md`](tasks/task-peer-benchmarks.md) (the matrix, with
   the 10-turn agent-turn TTFT as headline) and
   [`task-llamacpp-inproc.md`](tasks/task-llamacpp-inproc.md) (in-process llama.cpp via purego, drafted
   2026-09-12 — the fidelity column and per-token cost vs depth).
7. **Speculative decoding** — `spec/` (13 docs). Adaptive verify width failed all four ship gates
   and was removed; the grammar-prior premise is dead; DSpark/DFlash block drafters queued
   (P10/P15); MTP heads (`spec/09`) phase 3 untouched.
8. **Multimodal** — [`multimodal.md`](multimodal.md) P6–P11.
9. **Correctness programs** — [`audit-2026-09-10.md`](audit-2026-09-10.md)'s 8-step program in
   progress; [`task-recompute-audit.md`](tasks/task-recompute-audit.md) R-01 phase 1;
   [`task-int4-layout-2026-09.md`](tasks/task-int4-layout-2026-09.md) (one int4 layout per tensor, L1–L5)
   and [`task-halt-2026-09.md`](tasks/task-halt-2026-09.md) (cancel / halt / lease, K1–K9), both drafted
   this week.
10. **The book** — `book/`, the chaptered inference primer for a Go engineer new to ML.

## Decided and parked — each with its trigger

- **Continuous batching / paged attention** — not this engine's weight class; unchanged since
  v0.2 (MC5 of `tasks/parked/task-concurrency-2026-09.md`, parked with a trigger). Its small cousin got its kill-or-earn
  measurement (MC2 earned) and shipped in 2026-09:
  - N CPU decode workers (MC3c), then batched CPU decode behind them (MC3c step 2: 2.19× the workers at 4 clients
    on the 7B);
  - batched multi-request decode on Metal (MC3: 1.59× at 4 clients, bit-identical to serving each alone) and on CUDA
    (1.380× on the 1.5B, 1.826× on the 7B);
  - per-conversation resident KV slots on Metal, CUDA and WebGPU (MC1).
- **Bindings** (sidecar for desktop, c-archive for mobile) — scoped in
  [`task-bindings.md`](tasks/parked/task-bindings.md), not started. Gate: the B0.1 on-device iPhone spike.
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
  [`task-oracle-refforward.md`](tasks/parked/task-oracle-refforward.md); post-1.0 infrastructure, and the
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
