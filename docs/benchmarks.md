# goinfer — capability & performance vs the field

> **What this page is.** An evaluator-facing, provenance-gated comparison of goinfer
> against the engines people actually weigh it against. **Rule:** no number or ✓/✗
> without a traceable source — every cell is either measured (with the goinfer commit +
> date + machine), copied from an in-repo measurement with its doc/commit cited, or
> marked `—` (not applicable / not verified). If you can't trace it, it isn't here.

## TL;DR — 2026-08-31

<!-- OPEN DECISION, filed 2026-09-05, carried over from 2026-09-01: should this TL;DR section move
     ABOVE the "What this page is" blockquote? The Apple Silicon rows below already exist either
     way (§A) — this is a pure ordering/presentation call, maintainer's judgment. -->

**Re-anchored, with three stated exceptions.** Every row the 2026-08-25 re-anchor scoped — the
peer comparisons in §B2/§B4/§B5/§B6/§B7 — has been re-measured on the current stack (RTX 2070
SUPER, driver `595.91.07`, Nobara 44 / kernel 7.2.0, CUDA 13.2) against **Ollama v0.32.5**, or
deliberately withdrawn. Mac rows (§A) were never affected; last measured 2026-08-24. **Not covered,
and each says so in place:** peer *prefill* (no harness exists — see the table), the Mellum2 MoE
prefill row, and the SigLIP vision-prefill row. The last two are Linux rows from June that the
re-anchor never scoped; they are marked stale rather than quietly carried.

| | verdict | source |
|---|---|---|
| **CUDA decode, short context** (≤512 tok) | **goinfer wins on small models** — 0.5B **1.24×**, 1.5B **1.13×**; **parity at 7B** (1.00×) | §B8 |
| **CUDA decode, deep context** (2048+) | **Ollama wins, and the gap widens with depth** — goinfer 0.95×→0.78× (0.5B), 0.89×→0.71× (1.5B), 0.82×→0.71× (7B) by 3900 | §B8 |
| ↳ *unless the model is windowed* | gemma3-1b **does not degrade at all** — goinfer ahead 1.06–1.12× at *every* depth. The depth loss is a property of full attention over a growing KV, not of the engine | §B5.1 |
| **Prefill** | **Depth-dependent, and it crosses over.** goinfer is FASTER to first token below ~600–1000 prompt tokens (0.13× at K=128) and slower above, reaching 4.8×/6.1× behind at K=3900 (0.5B/1.5B). On overhead-free *throughput* the deficit is larger: marginal cost per token is **12–15× behind at depth**, and it GROWS with K while Ollama's is flat. **Re-anchored 2026-09-01** on the §B8 stack | §B2 |
| **Total request time** | goinfer wins prompts up to **~320 tokens** at 1.5B, loses beyond — decode edge vs prefill cost. Also pre-re-anchor | archive §B2 |
| **26B MoE on an 8 GB card** | **both engines run it.** goinfer keeps **every expert on the GPU** (host↔VRAM streaming) at 16.1 tok/s, 17.6 at ctx 2048; Ollama is **faster (~24.5)** by offloading 58% to CPU. An architecture distinction, not a capability peers lack | §B4.1 |
| **Apple Silicon CPU prefill** | **vs Ollama: 1.54× behind at K=512, reaching 0.91× (AHEAD) at K=3900; whole-curve marginal ratio 0.86×, goinfer faster** — aikit v1.34.0's S-01 int4 tile roughly doubled it (67.6→141.7 tok/s at K=512, measured pre/post on one box). Supersedes the 2026-09-01 row of 2.98×/1.80×, which the pre-tile arm reproduced to within 4% | §A |
| ↳ *and against our own past* | **8.61× faster than the pre-2026-09-01 record at 3020 tokens** (334.9 s → 38.9 s); the rate no longer falls with length (78.4 → 77.7 tok/s where it used to collapse 51.5 → 9.0) | §A |
| **Apple Silicon CPU decode** | **goinfer is behind** — 0.75–0.77× (0.5B) and 0.57–0.60× (1.5B) of Ollama CPU on an M1 Pro. `int4` is the right default there | §A |
| **Cold start & footprint** | **goinfer alone** — first token in **0.48 s**, **77 MB** resident, model compiled *into* the binary | §A, Table 1 |
| **Peer-independent** | pure Go, `CGO_ENABLED=0` (no libcuda/libnvrtc linked), **bit-identical** decode, HF logit-parity gate as a contract | Table 1 |
| **goinfer does not have** | continuous batching · GPU breadth · broad multimodal (vision-in only, no audio) · 11 architectures vs peers' dozens | Table 1 |

**One-line reading:** goinfer is a *small-model, short-context, single-request* engine that trades
throughput and breadth for a static binary, no native dependency, and a decode you can reproduce
bit-for-bit. Where it loses it loses honestly, and the losses are in this table rather than below it.

> **The lane.** goinfer runs open-weight model weights *in-process, in pure Go* — the
> single-file, zero-install, HF-parity-gated lane. The native engines (llama.cpp,
> Ollama, vLLM, mistral.rs) are far broader, and — against **current** Ollama (v0.32.5, 2026-07) —
> faster or at parity almost everywhere; the cgo-free CUDA backend holds a real edge only on
> **small-model** dense 4-bit decode at short context (**0.5B 1.24×, 1.5B 1.13×** at 128 tokens,
> launch/issue-bound) and reaches **parity at 7B**, while losing long-context decode and prefill
> (**§B8**, the current anchor). *An earlier draft of this paragraph said `0.5B ~1.7×` and `parity
> at 1.5B`, from the 476-vs-268 pairing whose goinfer half was retired as a methodology mismatch —
> both figures are withdrawn; §B8 measured 332.7 vs 268.7 server-to-server.* Its durable wins are
> peer-independent: pure-Go/no-native-dep, model-in-binary, bit-identical decode, and running a 26B
> **fully GPU-resident** on an 8 GB card (§B4 — current Ollama also runs it on 8 GB, but via CPU
> offload and faster; goinfer's distinction is all-experts-on-GPU, not that peers can't run it). The Go *bindings*
> (gollama.cpp, yzma) reach llama.cpp's speed but still ship its native `.so`.
> **The pure-Go lane is occupied.** `goccy/go-llama` (MIT, active) runs llama.cpp in-process with
> no cgo, no shared library and no wasm runtime at execution time, by compiling it to
> `wasm64-wasip1` and transpiling that to Go (`goccy/llamawasm2go` v0.2.4, llama.cpp base b10223 /
> `11924d4c`). It inherits llama.cpp's kernels and model coverage. Its engine is single-threaded —
> no `pthread_create` and no ggml threadpool symbols in the bundle, `ContextParams.NThreads`
> (`llama.go`) documents the clamp to 1, and its own `scripts/bench-compare.sh` runs native
> `llama-bench` with `-t 1` to match — GGUF-only, with no GPU backend. What separates goinfer is
> narrower than the lane: it implements the forward pass rather than inheriting one, which is what
> carries multi-threaded CPU decode, a GPU backend, checkpoint formats beyond GGUF (safetensors,
> GPTQ, AWQ), and compiling the model **into** the binary. No head-to-head numbers exist in either
> direction; neither project has published any. It trades peak throughput and breadth (no continuous batching,
> vision-in only — no audio, CPU-slow — 11 architectures) for a static binary that
> boots in ~0.5 s.

> **Re-anchored, and CLOSED 2026-08-31.** The box's OS and driver stack was replaced on 2026-08-25
> (Nobara 43 → 44: driver `595.58.03` → **`595.91.07`**, kernel `7.0.5` → **`7.2.0`**, glibc fc43 →
> **2.43-8.fc44**, CUDA 13.2 reported by the driver), which under this repo's rules invalidates
> comparability and forces a deliberate re-anchor. **That work is finished.** Every section is now
> either re-anchored — §B4.1/§B4.2, §B5.1, §B6.3, §B7.1, §B8 — or deliberately withdrawn (§B, and
> the v0.11.0 qualification). Parity was re-established *before* any timing: `gate gpu` PASS at
> `a161bd6` with **24/24 PTX byte-identical**, so the driver's new compiler did not move the
> numerics.
>
> Full chronology — what moved, per-leg discharge notes, and the retirements — in
> [`benchmarks-archive.md`](benchmarks-archive.md). Re-measure with `scripts/bench_peer.py` (peer
> Ollama v0.32.5 at `~/ollama-0325`, both sides over HTTP, interleaved) — **not**
> `bench_compare.sh`, which by its own design note never drives the peer.

---

## Methodology (non-negotiable — applies to every number on this page)

A measurement enters a table **only if it satisfies all of**:

- **Same machine** for goinfer and any peer it's compared against (rig named per row).
- **Same model checkpoint** and **same quant** — the exact GGUF/`.giw` named.
- **Greedy decode, fixed seed** (we measure the engine, not sampling luck).
- **Pinned versions**: goinfer commit + each peer's version, inline.
- **Date** of the run, and a **thermal note** (plugged in, warm, repeated runs, median).
- **Local disk only** — the checkpoint is read from local NVMe/SSD. A model read from the
  SMB archive mount is **not a measurement**; see *Model storage* below.
- **COUNT TOKENS FROM THE ENGINE'S OWN `usage`, NEVER BY COUNTING SSE FRAMES — AND THE ERROR DOES
  NOT CANCEL IN A RATIO.** A chunk is not a token: streams hold bytes back for UTF-8 continuation and
  stop-string matching, so several tokens can share one frame and frame-counting under-reads, always
  downward. The tempting argument is that a peer ratio is safe because both sides are biased the same
  way. **Measured 2026-08-27/28 across 130 recorded cells, they are not:**

  | engine | tok/chunk range | worst single cell |
  |---|---|---|
  | goinfer | 1.0000 – 1.0456 | phi3-mini T=0.6 |
  | Ollama | 1.0000 – **1.0909** | 1.5B T=0.4 |

  On the SAME cell the two engines differ by up to **9 points** — goinfer at 1.0000 where Ollama
  reads 1.0909 — so a chunk-counted paired difference carries a residual in the peer's favour rather
  than cancelling. `bench_peer.py` now takes each side's reported `usage` where the engine gives one
  and records `tokens_per_chunk` per cell, so this is a checked number and not an assumption. **Read
  that column on every peer run.**

  Those same rows exposed a second thing worth checking for: Ollama returned **576 tokens where 3072
  were requested** on one cell (1.5B, T=0.4), i.e. its completions terminated early. A cell that
  generated a fifth of the requested tokens is not the cell it claims to be, whatever its rate says.
- **`scripts/prompts.json` IS CALIBRATED FOR TOKEN DEPTH AND IS NOT VALID INPUT FOR ANY
  CONTENT-DEPENDENT MEASUREMENT.** Every prompt in it has **four unique words**
  (`"Continue this text. the the the ... the"`). For throughput that is correct and deliberate —
  decode cost per token does not depend on which tokens they are, and filler hits exact token
  counts. For anything whose value depends on WHAT IS GENERATED — speculation acceptance, drafter
  hit rates, optimistic-forward overlap — it measures a regime the engine never serves.
  **Measured 2026-08-27/28:** the same optFwd A/B on filler vs a 127-token prose paragraph moved by
  up to **9.4 percentage points on one cell**, turning a measured 5.1% WIN into a 4.3% LOSS and
  shifting a model's break-even temperature from 0.95 to 0.37. Use a realistic prompt and say which
  one; `docs/spec/10-optfwd-gate.md` has the procedure and the raw cells.
- **A longer measurement window does NOT reliably buy precision — check before spending on it.**
  The intuition is that averaging over N tokens shrinks the spread as 1/sqrt(N), so a noisy cell can
  be fixed by measuring for longer. **Measured 2026-08-27, it does not hold here.** Per-window
  coefficient of variation over decode windows of 64 / 128 / 256 tokens:

  | model | 64 | 128 | 256 | 1/sqrt(N) would predict at 256 |
  |---|---|---|---|---|
  | phi3-mini (sampled) | 3.29% | 2.59% | 1.75% | 1.65% |
  | 1.5B (sampled) | 7.86% | 6.40% | **6.89%** | 3.93% |

  The 1.5B gets **worse** from 128 to 256. There is a persistent per-window component that averaging
  cannot touch, so past some length a longer window costs more tokens for the *same* confidence —
  total cost scales as CV²·N, and CV stops falling while N keeps rising. In that case 64-token
  windows were **3.3x cheaper** than 256 for equal power.

  **The general rule: treat "measure for longer" as a hypothesis to test on two window lengths, not
  a lever you can pull.** Where the noise is a property of the mechanism rather than of sampling —
  a speculative hit/miss lottery here — it survives averaging and the floor is where you land.
  Raw: `docs/measurements/g26-winvar-*.json`; the reasoning it fed is `docs/spec/10-optfwd-gate.md`.
- **Verified-idle box, with the machine state recorded beside the number** — load average at the
  start of the run, and nothing else of ours running. Added 2026-08-25 after this list failed to
  catch a bad number: a 3020-token prefill timing came back **1587.1 s** where three later
  measurements of the identical thing give ~350 s. It satisfied every other bullet on this list —
  same machine, same checkpoint, same quant, dated — and was wrong by 4.5×, most likely because an
  abandoned prefill from a killed client was still burning a core (the defect now fixed as G18).
  It survived into three documents and a release note before a disagreeing instrument caught it.
  **A number with no recorded machine state cannot be argued with later**; that is what made this
  one expensive rather than merely wrong. See the withdrawn G15 in `docs/queue-performance.md`.

Anything not matching all of these is `—` and a re-measure, never a guess. This page
exists *because* a sloppy comparison is worse than none: `docs/completed/gpu-assessment.md`
caught one of its own early runs comparing a **Qwen1.5-1.8B q4** against the
**Qwen2.5-1.5B** target (a 191 tok/s number) and discarded it. Same-checkpoint,
same-quant, same-machine is the whole discipline.

Reproduce goinfer's side end-to-end (and get the verbatim peer commands) with
[`scripts/bench_compare.sh`](../scripts/bench_compare.sh).

**Two pinned peers — a CURRENT one and a historical one.** A live competitive claim must be
measured against the *current* peer; a *reproducible* historical row is kept beside it. As of
2026-08-04:
- **Current: Ollama v0.32.5** (2026-07-27) at `~/ollama-0325` (`OLLAMA_HOST=127.0.0.1:11436
  OLLAMA_MODELS=~/ollama-0325/models LD_LIBRARY_PATH=~/ollama-0325/lib/ollama
  ~/ollama-0325/bin/ollama serve`). **All current §B2 claims use this** (see the re-anchor box in §B2).
- **Historical: Ollama 0.5.7** (2025-01-16) at `~/ollama-587/bin/ollama` (port 11435). Kept ONLY to
  reproduce the labeled-historical rows — it is **~18 months stale** and must **never** back a current
  claim. The original §B2 tables were measured against it; that was the bug corrected on 2026-08-04.

Both import the 1.5B as `q15`. When re-measuring, use v0.32.5; keep 0.5.7 for the historical rows.

**Forward vs serve split (new — future rows may state both).** Every published row here is a
**serve-path, client-wall-clock** number (prefill, sampling, detok, HTTP all inside it) — the
methodology-symmetric bar, and it stays. But a direct decomposition now exists
(`GOINFER_DECODE_TIMING=1`, `docs/task-moe-streaming.md`): on the 26B the greedy **forward** is
~36 ms/tok while the **serve** number is ~57 ms/tok, the gap being **prefill amortization** (sequential
full-logits prefill, no batched `PrefillLast`) plus context-depth growth — *not* a per-token serve tail
(greedy `sample`/`embed` ≈ 0). So future rows may state both a forward rate and a wall-clock rate, the
way §B2 already prints Ollama's decode-only rate alongside its wall clock — with the wall-clock number
remaining the one that counts for a peer comparison. Existing rows (incl. §B4's 16.98 tok/s, re-confirmed)
are correctly measured and unchanged.

---

## Model storage — archive remote, benchmark local

> **This section is the authority on where a measurement may read its checkpoint from.**
> `CLAUDE.md` states the rule and the two forbidden roots and points here; nothing else should
> restate the detail. Consolidated 2026-08-26 after the rule was found in three places, complete
> in only this one.

**Models are stored on the Linux box and benchmarked from local disk only. Nothing that
produces a number is ever read over the network mount.** Paged decode over SMB turns
weight loading into network round-trips at arbitrary times during the run, so the result
silently measures the LAN and the server's disk instead of the engine — and it fails
*quietly*, producing a plausible-looking number rather than an error. That is exactly the
class of bad row the Methodology gate above exists to reject.

| | Path | Role |
|---|---|---|
| Archive | `//192.168.1.240/models` → `/srv/models` (5.5 TB ext4) | **Cold storage. A bench surface for neither machine** — it is a 5400 rpm SMR disk. |
| MacBook bench set | `~/models` (internal SSD) | The only place a darwin/Metal row may be measured from. |
| Linux bench set | `/home/francis/models` (NVMe) | The only place an amd64/CUDA row may be measured from. Unchanged by the archive. |

Note the Linux box benches from its **own NVMe `~/models`**, not from `/srv/models`. The
archive is storage on both machines, never a read path for a timed run.

**`/srv/models` needs saying out loud, because the `/Volumes/` rule does not reach it.** On the
MacBook the archive is only ever a network mount, so "don't benchmark over the network" and "don't
benchmark from the archive" are the same sentence. On the Linux box they are not: `/srv/models` is
a local mount point on the very machine that measures every CUDA row, so a reader applying "measure
from local disk" can satisfy it and still be reading the 5400 rpm SMR archive. Same wrong number,
no network involved, and no rule violated as the older phrasing was written. Both roots are
therefore named wherever the rule appears.

### Working from the archive

`~/bin/models-pull` copies a checkpoint from the archive onto local disk before you
benchmark it; `models-push` archives a local one and refuses to report success unless the
byte counts match on both sides. Both ride **rsync over SSH**, not SMB, so the share does
not need to be mounted.

```sh
models-pull -l                     # list the archive
models-pull -l qwen                # filtered
models-pull qwen3.6-35b-a3b-int4.giw   # archive -> ~/models, resumable
models-push my-new-download.gguf       # ~/models -> archive, size-verified
```

They honour `MODELS_HOST` (default `nobara`), `MODELS_ARCHIVE` (`/srv/models`) and
`MODELS_LOCAL` (`~/models`).

**Mount the share only to browse it, and unmount when done.** It is deliberately not
automounted: a permanently-present `/Volumes/models` is precisely how a checkpoint path on
the mount ends up in a benchmark by accident. **If a row's model path starts with `/Volumes/`
or with `/srv/models`, the row is void** — re-measure it from the local bench set (on the
MacBook, after a `models-pull`; on the Linux box the file is already on the NVMe, so this is a
path mistake rather than a missing copy).

---

## Table 1 — Capability matrix

Durable, mostly-boolean axes. goinfer's `✗` stay in the table. Legend: **✓** yes ·
**✗** no · **~** partial / caveated · **—** not verified or N/A. Peer cells verified
against each project's repo/docs on **2026-06-10** (citations in *Sources*); the `go-llama`
column on **2026-08-27**; re-verify at each goinfer tag. Its `—` cells are unchecked, not
absent — this pass read the engine and packaging, not the library surface.

| Capability | **goinfer** | llama.cpp | Ollama | mistral.rs | vLLM | Go bindings¹ (gollama.cpp · yzma) | pure-Go ports² (llama2.go) | go-llama³ (goccy) |
|---|---|---|---|---|---|---|---|---|
| Runs weights in-process, **no native dep** | ✓ pure Go, no cgo ᵃ | ✗ *is* the C/C++ lib | ✗ spawns native `llama-server` (GGUF) + MLX ᵇ | ✗ Rust; GPU links CUDA/Metal | ✗ Python + PyTorch + CUDA | ✗ dlopen prebuilt llama.cpp `.so/.dll/.dylib` ᶜ | ✓ genuinely pure Go | ✓ pure Go, transpiled wasm ᵏ |
| Single static binary, zero install | ✓ ᵃ | ~ standalone binary, model separate | ~ binary + `llama-server` companion ᵇ | ~ one CLI binary; GPU needs CUDA/Metal | ✗ pip / Python env | ✗ needs the native lib ᶜ | ✓ (but toy) | ✓ (bundle ~126 MB) ᵏ |
| Model **embeddable in the binary** | ✓ `.giw` mapped from the image ᵈ | ✗ | ✗ | — | ✗ | ✗ | ✗ | — |
| HF logit-parity gate (a contract) | ✓ ᵉ | — | — | — | — | — | — | — |
| Constrained / structured decode | ✓ **struct-derived** grammar ᶠ | ✓ GBNF + JSON-schema | ✓ JSON schema | ✓ grammar + strict schema | ✓ xgrammar / `guided_json` | — | ✗ | — |
| OpenAI-compatible server | ✓ pure stdlib ᵍ | ✓ `llama-server` | ✓ | ✓ (+ Anthropic) | ✓ (heavy deps) | ✗ (library only) | ✗ | — |
| LoRA adapters | ✓ PEFT, merged at load ʰ | ✓ | ✓ | ✓ | ✓ | — | ✗ | — |
| GPU | ~ WebGPU (broad residency) + **cgo-free CUDA & Metal** (dense + MoE; `features.go`-gated) ⁱ | ✓ CUDA/Metal/Vulkan | ✓ CUDA/ROCm/Vulkan/Metal | ✓ CUDA/Metal | ✓ CUDA/TPU/+ | ✓ inherits llama.cpp | ✗ CPU only | ✗ no GPU backend ᵏ |
| Continuous batching | ✗ | ✓ | ~ parallel slots via llama-server ᵇ | ✓ | ✓ PagedAttention | — | ✗ | — |
| Multimodal (vision/audio) | ~ **vision in** (Gemma 3 VL, pure-Go SigLIP → serve + agent; ~171 s/image CPU or **18.8 s on `-tags gpu`**, no audio) | ✓ | ✓ | ✓ | ✓ | ~ (yzma VLMs; gollama —) | ✗ | — |
| Model coverage | ~ **11 architectures** ʲ | ✓ dozens | ✓ broad | ✓ broad | ✓ 200+ | ✓ inherits llama.cpp | ✗ Llama-2 toy | ✓ inherits llama.cpp (GGUF only) ᵏ |
| Multi-threaded CPU decode | ✓ | — | — | — | — | — | — | ✗ single-threaded ᵏ |

**Reading it:** *no-native-dep pure-Go execution* is no longer goinfer's alone — `go-llama`
reaches it by transpiling llama.cpp, and what separates the two is where the forward pass lives
(implemented vs inherited), which is what carries threading, GPU and non-GGUF formats. goinfer is
still alone here on *model-in-binary*, and on
cold-start/footprint (Table 2). It ties on the library surface (constrained decode,
LoRA, OpenAI server). It **loses**, honestly, on GPU breadth, continuous batching,
multimodal, and model coverage — those are the native engines' and vLLM's turf.

¹ Go *bindings*: "no CGO" (purego/ffi) but still download/bundle and `dlopen` a native
llama.cpp shared library — not pure-Go inference. ² nikolaydubina/llama2.go, the
in-process port this column was built around, is **archived (2024-11-30)**, CPU-only,
Llama-2-in-`llama2.c`-format, ~0.87 tok/s on 7B. That column described the pure-Go
lane when it was written; ³ is why it no longer describes the lane.

³ **`goccy/go-llama`** (MIT, active) with `goccy/llamawasm2go` v0.2.4: llama.cpp built for
`wasm64-wasip1` and transpiled to Go by `goccy/wasm2go`, so the engine is ordinary Go — no cgo, no
shared library, and no wasm runtime at execution time. Base llama.cpp b10223 (`11924d4c`).

ᵏ go-llama cells, read from the two repos on 2026-08-27. **Single-threaded:** no `pthread_create`
and no ggml threadpool symbols anywhere in the ~126 MB bundle; `ContextParams.NThreads` (`llama.go`)
documents the clamp to 1; `scripts/bench-compare.sh` runs native `llama-bench` with `-t 1` "to match
the engine's single-threaded wasm build". **Bundle:** p0 46 MB + p1 45 MB + p2 35 MB, including
~15 MB `arm64.s` and ~14 MB `amd64.s`. Hand-spliced arm64 assembly covers **q8_0 only**
(`DbgGemvQ8_0_4x4` / `DbgGemmQ8_0_4x4` in `llamawasm2go/wasm2go.go`, exercised by
`go-llama/internal/gemv_repack_numeric_test.go`); q4 takes the generic transpiled path. **No
published numbers** in either repo — a harness exists (`bench_test.go`: `BenchmarkDecode`,
`BenchmarkPromptEval`, plus `bench-compare.sh`) but no results, so every performance cell here is
*not measured* rather than unfavourable. Note go-llama's README still describes a wasm32 4 GiB
ceiling and a ~3B-at-Q4 limit; the shipped bundle is wasm64 (int64 pointers, `MaxMem` is `uint64`,
and `WithMaxMemory` documents a 64 GiB default of reserved address space), so that text is stale
and is not repeated here.

Cell provenance: ᵃ `README.md` (pure Go, no cgo, cross-compiled static binary) ·
ᵇ Ollama PR #16031 (CGO runner removed → upstream `llama-server` for GGUF + MLX for
safetensors; merged 2026-05-29 — so continuous batching is whatever `llama-server`'s
`-np` slots give, not an Ollama-documented feature) · ᶜ gollama.cpp / yzma READMEs
(purego/ffi that `dlopen` a prebuilt llama.cpp lib) · ᵈ `ARCHITECTURE.md` §2 (`.giw`
weights mapped from the binary's read-only image) · ᵉ `README.md` + `CHANGELOG.md`
(forward-pass numerics parity-gated vs HF; per-family logit-parity tests) ·
ᶠ `README.md` (`GrammarFromStruct` — grammar derived from a Go struct) · ᵍ `README.md` /
`cmd/serve` (OpenAI-compatible server in pure stdlib) · ʰ `README.md` (LoRA PEFT,
merged at load) · ⁱ `ARCHITECTURE.md` §2 + `docs/completed/gpu-assessment.md` (WebGPU full
residency; cgo quarantined behind `-tags gpu`) + §B2/§B3 below (`cuda/`, `metal/`:
driver-JIT / MSL, **CGO_ENABLED=0**, admission-gated by
`decoder/features.go`) ·
ʲ `decoder/registry.go` — **13 registered `model_type` keys, 11 distinct architectures**
(`gemma3_text`/`qwen3_5_moe_text` are text-decoder aliases of `gemma3`/`qwen3_5_moe`).

---

## Table 2 — Measured performance

Two rigs, both the repo's existing ones. Apple-Silicon CPU is the pure-Go lane; the
RTX section is the GPU-residency story. Peer columns are filled **only** from
same-machine/same-quant runs; where we have none, the cell is `—` and the script +
peer commands are how you fill it.

**The two rigs, precisely** — stated once here so rows can say "M1 Pro" or "the Ryzen box"
without the reader having to guess what that means:

| | `apple-m1pro` | `linux-62gb` / `nobara-pc` |
|---|---|---|
| machine | MacBook Pro `MacBookPro18,3` (14", 2021) | desktop |
| CPU | Apple M1 Pro — **8 cores: 6 performance + 2 efficiency** | Ryzen 7 3700X — 8c/16t |
| RAM | **16 GB** unified | **62 GB** |
| GPU | integrated (Metal) | RTX 2070 SUPER, **8 GB** | ✗ no GPU backend ᵏ |
| OS at the time of writing | macOS 26.6.2 | Nobara 44, kernel 7.2.0, CUDA 13.2 |

**Several numbers on this page are properties of those rigs, not of the code, and the
distinction is easy to lose.** The clearest cases:

- **`maxAttnWorkers = 6`** is the M1 Pro's *performance*-core count, not `GOMAXPROCS`; the 2
  efficiency cores measured harmful for this class of work. A part with more P-cores has more
  headroom than any speedup here shows.
- **The 16 GB** is why prefill scratch had to be budgeted and then tiled at all (G16/G20): six
  workers × an untiled 8k `scores` buffer is ~1.6 GB on a machine that was already swapping.
  On a 64 GB part the untiled version would simply have worked, and the tiling would read as
  premature.
- **The 8 GB card** is what makes "a 35B decodes resident" a story rather than a footnote.

So a *larger* Apple Silicon part should show **bigger** parallel speedups and hit the tiling
threshold later — the CPU rows here are a floor for that family, not a ceiling. Re-measuring on a
different Mac is a new rig and a new row, not an update to these.

### A. Apple Silicon CPU (the pure-Go lane)

Rig: **Apple M1 Pro** (6P+2E), prequant `.giw` int8, greedy + fixed prompt/seed,
plugged in. Source: `docs/ARCHITECTURE.md` §2 + `docs/completed/perf-campaign.md`, "after the
v0.5.0 perf work." goinfer commit for these: the v0.5.0-era CPU campaign (see
completed/perf-campaign.md). Peers were **not** run on this rig at the time → `—` (use the script).

**Peer run added 2026-08-22** (`docs/measurements/mac-cpu-decode-vs-ollama-2026-08-22.md`,
`bench_peer.py` method, decode-only, GGUF weights verified tensor-identical): ollama/llama.cpp
CPU (native Q4_K_M, its own default 6 threads) vs goinfer CPU at depth 128, same two models —

| model | goinfer int4 (default) | goinfer int8int8 | ollama Q4_K_M | int4 ratio | int8int8 ratio |
|---|---|---|---|---|---|
| 0.5B | 34.5 tok/s | 56.3 tok/s | 109.0 tok/s | 0.32x | 0.52x |
| 1.5B | 17.0 tok/s | 26.7 tok/s | 68.3 tok/s | 0.25x | 0.39x |

Diagnosis: the gap is not thread-count/E-core related (measured — capping goinfer to 6 threads
made no real difference) and only partly a quant-format artifact (int4→int8int8 recovers ~60% by
itself, reproducing a 2026-06-14 aikit finding that W4A8's NEON kernel is compute-bound on its
nibble-unpack). A residual ~2x gap survives even at the closest comparable format; see the
measurement doc for the bandwidth analysis and open questions.

**SUPERSEDED 2026-08-24 — the ranking flipped, do not use the int4/int8int8 ratio above.** The
diagnosis above was correct when written: at that point the int8int8 ratio (0.52x/0.39x) beat the
int4 ratio (0.32x/0.25x) specifically because int8int8's own LM head happened to already run the
fast W8A8 path, while int4's LM head ran a slow weight-only-Q8 path that was, at the time,
undiagnosed — int8int8 wasn't faster because int4 was slow at matmul, it was faster because int4's
*head* was slow and int8int8's wasn't. Once the W4A8 NEON kernel (`docs/task-w4a8-neon-bandwidth.md`,
item-3+4 harness) and the int4-mode LM head (same doc's LM-head follow-up, `embedding()` moved from
weight-only Q8 to full W8A8) both shipped, that asymmetry closed. Provenance: **Apple M1 Pro**,
`bench_peer.py` method, decode-only, greedy, depth 128, quiet box, goinfer commit **`a11c56b`**
(2026-08-24, the LM-head W8A8 default) —

| model | goinfer int4 | goinfer int8int8 | ollama Q4_K_M | int4 ratio | int8int8 ratio |
|---|---|---|---|---|---|
| 0.5B | 81.9-83.75 tok/s | 85.25 tok/s (Step-0 cell, unaffected by the LM-head fix — see below) | 109.0 tok/s | 0.75-0.77x | 0.78x |
| 1.5B | 39.1-40.7 tok/s | 37.56 tok/s (Step-0 cell, unaffected) | 68.3 tok/s | 0.57-0.60x | 0.55x |

**int4 now matches or beats int8int8 at both sizes, at half the weight RAM.** The int8int8 cells
here are the item-3+4 harness's own Step-0 baseline (`docs/task-w4a8-neon-bandwidth.md`), not
re-measured for this correction: the LM-head fix only touches `embedding()`'s `quantInt4`/
`quantInt4Mix` branch — int8int8's base mode passes through unchanged, so its LM head was already
full W8A8 before and after that fix, and its decode rate did not move. **Current guidance: `int4`
is the right default on Apple Silicon CPU decode** — equal-or-better speed at half the RAM, the
inverse of what this section said as of `d469c7c`. That advice is not deleted (above) because it
was an honest, correctly-diagnosed reading of the box at the time; it is superseded because the
thing it was measuring around — the LM head's drag — no longer exists.

**Re-measured under load 2026-08-02 (`14dfc47`) — consistent with the baseline.** The
table's ~70/~36 remain the headline: they were measured on a clean rig (v0.5.0 campaign).
A spot re-run of `BenchmarkDecode` this pass came back at ~68 (best 69.8) tok/s for 0.5B
and ~34 (best 34.5) for 1.5B (best-of-8 × 2 s, `qwen2.5-coder-{0.5b,1.5b}-instruct-q4_k_m.gguf`
loaded `int8int8`), but that run had the IDE holding ~4 cores (load avg 3.8), so those are a
*lower bound*, not a replacement — landing just under the clean baseline is exactly what a
contended re-run should do. The takeaway is that the ~70/~36 held across both
decode-parallelism-threshold fixes (the M1-Pro int8 crossover was already optimal, so unlike
the Ryzen path it had no room to move; see `tune.go` sweep note); a clean re-measure would be
needed to *revise* the headline. The end-to-end demo row (~57/~26) was **not** re-measured.

| metric (M1 Pro · `.giw` int8 · greedy/fixed seed) | goinfer 0.5B | goinfer 1.5B | peers |
|---|---|---|---|
| decode tok/s — `BenchmarkDecode` (pure forward+sample) | ~70 | ~36 | — |
| decode tok/s — end-to-end demo (incl. UI/stream) | ~57 | ~26 | — |
| cold start → first token | **0.48 s** | **1.23 s** | — |
| resident heap (`phys_footprint`) | **77 MB** | **87 MB** | — |
| install footprint | one static binary, model included | one static binary, model included | — |

Context for the footprint win (same 0.5B, M-series, `docs/ARCHITECTURE.md` §2):
embedded-GGUF path → prequant `.giw` cuts **cold start 2.30 s → 0.48 s (~5×)** and
**resident heap 772 MB → 78 MB (~10×)** — the weights are mapped from the binary's
read-only image, not heap-copied.

**Prefill (relative, CPU):** vectorizing prefill attention onto the SIMD path gave
**~3.4× dense prompt prefill** and **~1.7× Gemma 4** (M1 Pro; `CHANGELOG` v0.5.0,
`7fa82c2`/`88b7aaa`).

> ## ⚠ STALE as of 2026-09-01 — CPU prefill changed four times in one day and this block predates all of it
>
> **Everything below is retained as the record of a state the engine has left.** Four changes
> landed 2026-09-01, none of them reflected in the numbers or the prose here:
>
> | change | effect | record |
> |---|---|---|
> | **A3 head fan-out** — f32 prefill attention parallelised over query heads | 1.58× @K=2048, **1.92× @K=4096** (dense 0.5B, e2e) | `measurements/a3-f32-attention-fanout-2026-09-01.md` |
> | **P18 expert-major MoE prefill** (default ON) | **4.364×** at K=4096 on 28-layer Mellum2, bit-identical | `measurements/p18-expert-major-e2e-2026-09-01.md` |
> | **P19 fused attention** (default ON, changes output) | +8.0% e2e; 1.69–1.73× at the kernel | `measurements/p19-fused-attention-2026-09-01.md` |
> | (earlier) f32 prefill attention became the default above a 512-token floor | 1.15×–2.28× by depth | CHANGELOG |
>
> **THE PROSE BELOW IS NOW FACTUALLY WRONG, not merely dated.** Specifically: *"CPU prefill is
> SINGLE-THREADED"*, *"the attention half is serial … attention's heads do not [fan out] — A1's
> deferral"*, and *"the process sits at ~100% CPU through a large prefill"*. A1's deferral was
> resolved by A3; measured utilization on the same class of run is now **1.67× before the fan-out
> and 5.27× after** it, and the profile arms today ran at 250–300% CPU.
>
> ### RE-MEASURED 2026-09-01 — and the RATE NO LONGER FALLS WITH LENGTH
>
> Same configuration as the original cells (dense **1.5B**, `int8int8`, prefill + 1 token, M1 Pro),
> best of 3 rather than the original's single shot — an improvement to the method, stated so the
> two are not read as identically obtained. goinfer `cfec302`.
>
> | prompt | recorded (pre-2026-09-01) | **measured** | **tok/s** | speedup |
> |---|---|---|---|---|
> | 170 | 3.3 s | **2.2 s** | **78.4** | 1.52× |
> | 620 | 19.7 s | **7.0 s** | **89.0** | 2.83× |
> | 1520 | 93.2 s | **18.1 s** | **84.1** | 5.16× |
> | 3020 | 334.9 s | **38.9 s** | **77.7** | **8.61×** |
>
> **The structural claim is the flat rate, not the speedup.** The recorded cells fell from
> 51.5 tok/s to 9.0 tok/s across this range — a 5.7× collapse — and the prose below attributes that
> to the serial attention half. It is now **78.4 → 77.7 tok/s, flat**, because that is precisely
> what A3's head fan-out removed. The speedup grows with length (1.52× → 8.61×) for the same
> reason: the term that was superlinear is the one that got fixed.
>
> Attribution, since three changes landed together: the f32 prefill default and **A3's head
> fan-out** do the work here, plus **P19's fused schedule** (+8%). **P18 is inert for this row** —
> it is MoE-only and this is a dense model.
>
> ### AND THE FIRST PEER CPU-PREFILL COMPARISON — 2026-09-01
>
> > **SUPERSEDED 2026-09-05 — CPU prefill is now at parity and past it on the marginal; do not
> > quote the 2.98×/1.80× ratios or the int4/int8int8 inversion below.** aikit's S-01
> > register-blocked int4 tile landed in v1.34.0. Re-measured on the same box against the same
> > Ollama 0.32.5: **1.54× behind at K=512, 0.91× at K=3900 (AHEAD), whole-curve marginal ratio
> > 0.86×.** The table below was taken on aikit v1.31.0. It is stale, not wrong — a pre-tile arm
> > re-run on 2026-09-05 reproduced it to within 4% (3.10× / 1.81×), which is what makes the new
> > row comparable to it. **The QUANT NOTE below is also superseded**: int8int8 no longer beats
> > int4 at M>1; the tile removed the unpack repetition that caused that inversion, and int4 now
> > wins by 1.10–1.14×. Record: `measurements/cpu-peer-prefill-2026-09-05.md`.
>
> The table above is goinfer-vs-goinfer. Against **Ollama v0.32.5** (same GGUF both sides, Ollama
> forced to CPU with `num_gpu: 0`, goinfer at **int4** so the weights match q4_K_M, 4 distinct
> prompts per cell, engines interleaved) on the same M1 Pro:
>
> | K | goinfer | Ollama | ratio |
> |---|---|---|---|
> | 512 | 68.4 tok/s | 203.7 | **2.98×** |
> | 1024 | 68.7 | 187.9 | **2.74×** |
> | 2048 | 67.6 | 145.4 | **2.15×** |
> | 3900 | 61.9 | 111.2 | **1.80×** |
>
> **The gap closes with depth**, and the scaling says why: over 512→3900 goinfer slows 8.35× while
> Ollama slows **13.37×**. At the deepest interval the marginal rates are 56.5 vs 87.8 tok/s.
>
> **QUANT NOTE, because the first attempt got it wrong.** These use goinfer at int4 to match the
> peer's q4_K_M. A first sweep ran int8int8 — mirroring §A's own table — which is 8-bit against
> 4-bit and not a peer comparison at all. Re-run weight-matched. The difference is itself worth
> knowing: **int8int8 CPU prefill is 25–33% FASTER than int4** (90.7 vs 68.4 tok/s at K=512),
> because the W4A8 unpack costs more at M>1 than the halved weight bytes save. §A's absolute table
> is int8int8 and this peer table is int4; they are not the same configuration and must not be read
> as one series.
>
> **Ollama caches prompts hard on CPU** — repeat/fresh 0.05 → 0.00 across these depths. Every
> request here carries a unique prefix; reusing one would have compared our prefill against its
> cache lookup. Record: `measurements/cpu-peer-prefill-2026-09-01.md`.
>
> "Long prompts on the CPU backend are much slower than the speedup suggests" — the practical
> warning below — no longer holds in the form it is written. A 3020-token prompt is 38.9 s, not
> 334.9 s.
>
> **Everything from here to the end of this block is the pre-2026-09-01 record:**
>
> **Read those as RELATIVE speedups over their own predecessor — they are not a rate, and CPU
> prefill is SINGLE-THREADED.** Both facts are compatible and both were true when written; the
> second was simply never stated anywhere a reader could see it (added 2026-08-25, queue G16).
> Absolute numbers on an M1 Pro (dense 1.5B, `int8int8`, prefill + 1 token): **170 tok 3.3 s
> (51.5 tok/s) · 620 tok 19.7 s · 1520 tok 93.2 s · 3020 tok 334.9 s (9.0 tok/s)** — the best-case
> rate is the same order as this model's *decode* rate, and it falls as the prompt grows because
> the attention half is serial (aikit's weight matmuls do fan out; attention's heads do not — A1's
> deferral, `docs/task-attention-decode-cost.md`). On a 6-performance-core box the process sits at
> ~100% CPU through a large prefill. **Practical consequence:** long prompts on the CPU backend are
> much slower than "3.4× faster prefill" suggests — size expectations from the absolute table, not
> the speedup. GPU backends do not share this (`PrefillLast` is batched on-device). Sparse-MoE prefill is now batched too — **2.4×** on Mellum2
12B-A2.5B (**3.36 → 8.11 tok/s** at a 1024-token prompt), measured on the RTX-box CPU
(**Ryzen 7 3700X**, `08acc11`, 2026-06-10) — a *different* rig, listed separately so it
isn't conflated with the M1 numbers above. **⚠ STALE (flagged 2026-08-31): measured on the
pre-2026-08-25 stack (Nobara 43) and never scoped by the re-anchor, which covered the peer
comparisons only. The 2.4× is a goinfer-vs-goinfer ratio on one binary, so it is the safer half;
the absolute 3.36 → 8.11 tok/s is not a current rate.**

**Vision prefill (SigLIP, gemma-3-4b-it, 896², 4096 patches):** ~171 s/image on
CPU (compute-bound matmul; int8 is a wash on AVX2 — no VNNI). On `-tags gpu` with
`--backend webgpu` the resident GPU encoder runs the whole tower on-device:
**18.8 s/image** (~9×) on an **RTX 2070 SUPER**, parity cosine 1.000000 vs the CPU
W8A8 encoder (`886c8fd`/`5d7c572`, 2026-06-11). **⚠ STALE (flagged 2026-08-31): also a
pre-2026-08-25 Linux row the re-anchor did not scope — the ~9× is same-binary, the absolute
18.8 s/image is not current.** The attention matmuls are still naive f32 — a tiled GEMM there is the next lever (`docs/completed/task-gpu-vision-tower.md`).

### B. GPU residency vs native CUDA, at equal quant

> **RETIRED 2026-08-27 — both rows withdrawn, not corrected, and not re-measured.** What stood here
> (WebGPU int8 vs native q8_0, and the 60–70%-of-CUDA headline derived from it) is preserved with
> its retirement reasoning in [`benchmarks-archive.md`](benchmarks-archive.md).
> Current GPU-vs-GPU numbers: **§B8**, whose backend table places goinfer's WebGPU against its own
> CUDA (38 / 41 / 63% at 0.5B / 1.5B / 7B) — a cross-backend row, not a peer cell.

### B2. cgo-free CUDA (`-tags cuda`) vs Ollama-CUDA — 4-bit both sides

> **The historical rows moved.** The peer tables against **Ollama 0.5.7** (2025-01-16), the
> 2026-08-04 v0.32.5 re-anchor box, and the full batched-prefill engineering chronology are in
> [`benchmarks-archive.md`](benchmarks-archive.md).
> **Decode is superseded by §B8.** What remains below is what §B8 does not carry.

The `cuda/` backend is a **driver-JIT, CGO_ENABLED=0** path (dlopen `libcuda` via purego; PTX
embedded, no toolkit at build or run time — `ldd` shows no `libcuda`/`libnvrtc` linked). It admits
an architecture only when it implements every feature that architecture needs, else declines to the
staged/CPU path (`decoder/features.go`, the authoritative set).

**Prefill: RE-ANCHORED 2026-09-01** on the §B8 stack (driver `595.91.07`, Nobara 44, goinfer
`fb43caf`, Ollama v0.32.5), via the new `scripts/bench_peer_prefill.py`. Full record:
[`docs/measurements/cuda-prefill-reanchor-2026-09-01.md`](measurements/cuda-prefill-reanchor-2026-09-01.md).

**There is no single ratio. It crosses over.** TTFT rate (`prompt_tokens / TTFT`), medians of 6
distinct prompts per cell, engines interleaved with a server restart between:

| K | 0.5B | 1.5B |
|---|---|---|
| 128 | **0.13×** | **0.27×** |
| 512 | **0.51×** | **0.91×** |
| 1024 | 1.06× | 1.76× |
| 2048 | 2.33× | 3.37× |
| 3900 | **4.82×** | **6.13×** |

(<1 means goinfer is faster.) goinfer wins below ~1024 prompt tokens at 0.5B and ~600 at 1.5B.

> **TTFT RATE IS NOT PREFILL THROUGHPUT, AND THE GAP IS WORSE THAN THE TABLE.** TTFT carries each
> engine's fixed per-request overhead, and that overhead is not common-mode: Ollama's fitted floor
> is **340–356 ms** against goinfer's tens of ms. That floor is most of why goinfer "wins" at
> K=128, and it flatters Ollama's rate at depth. Overhead-free, the **marginal cost per prefill
> token** is 0.377 → 0.932 ms on goinfer across the ladder (**2.5× growth**, the O(K²) attention
> signature) against Ollama's flat 0.064 → 0.063. At the deepest interval that is **14.8× (0.5B)
> and 12.7× (1.5B) behind** — the honest number for a prefill-speed claim.
>
> The retired `4.7×` was not wrong at the depth it was taken; it was one point on a curve running
> 0.13× → 6.13×, published without a depth. It also hid a real win: goinfer is 2–8× faster to
> first token on short prompts, which this page had never said.

> **METHOD NOTE — Ollama caches prompts and goinfer does not** (repeat/fresh 0.53 vs 1.00,
> measured). Reusing one prompt per cell, as the decode harness does, would have compared our
> prefill against the peer's cache lookup and reported ~6.3× where that cell reads ~3.4×. Every
> request in this sweep carries a unique prefix; the check is recorded per cell in the results.

**Consequence for total request time** — ⚠ the paragraph below is computed from the RETIRED
pre-re-anchor prefill figure and its crossover numbers have not been recomputed against the
2026-09-01 sweep. The direction survives (a decode edge cannot cover a per-token prefill deficit at
depth) but the specific token counts should be treated as stale until re-derived, and the re-anchor
moves them in goinfer's favour at the short end. On the same pre-re-anchor stack: with a small short-context
decode edge and ~4–5× slower prefill, goinfer wins *total* time only up to **~320-token prompts**
at 1.5B (~128 before the batched-prefill campaign, ~230 mid-campaign). Past that Ollama wins. At 7B
the crossover is **below 128 tokens** — the ~10% decode edge there cannot cover a 5–7× per-token
prefill penalty. **goinfer's total-time advantage is a small-model, short-prompt phenomenon.**

**What is peer-independent, and where the release should lead:**

- **cgo-free** (CGO_ENABLED=0, driver-only `ldd`), **portable**, **bit-identical** decode.
- **The batched-prefill campaign as an absolute engineering result:** real qwen2.5-coder-1.5b
  **2048-token TTFT 13.1 s → 2.1 s (6.17×)** vs the old sequential path — a measured TTFT win with
  **no peer involved**, and **bit-identical to sequential prefill**. *That bit-identity claim was
  briefly FALSE on real models (84% token-stream divergence from an fma-vs-mul+add contraction
  difference); it is restored and enforced at build time by `cuda.TestKernelFMALint`, evidenced by
  `TestPrefillDivergenceRate` = 0/50 on the real 1.5B. Write-up:
  `docs/completed/task-batched-prefill-bitidentity.md`.*
- **§B4's 26B-A4B on an 8 GB card, fully GPU-resident.** Current Ollama also runs it, faster, via a
  42/58 GPU/CPU split — so the honest claim is the architecture one: all experts on the GPU.
- **The prefill ceiling, stated plainly.** Kernel tuning will not close the 2048-token gap: GEMV
  Compute sits at 54% of the *dp4a* peak and dp4a is ~1/3 of Turing IMMA, so the ceiling itself is
  dp4a. Closing it needs the tensor-core GEMM in `docs/completed/task-rotation-perrow-imma.md`,
  which reorders a group-scaled float sum and therefore **cannot be bit-identical** — scoped and
  unfunded. A stated trade of the cgo-free, bit-identical thesis, not a deficiency.
- **Correctness is gated, not assumed.** CUDA decode is held to the repo's 3% near-tie parity rule
  against the CPU path on a real q4_k_m checkpoint (9/10 exact argmax, 0 hard fails).

**Scope:** these are dense-lane numbers. Coverage is `decoder/features.go`, which grants `cuda` MoE
(routed + ungated shared) + partial rotary + the Gemma set; MoE-resident *speed* is unmeasured here.

*The archive's most reusable content is not a number: it is the record of the prefill GEMV gap being
**mis-attributed five times**, each refuted by the next measurement, before `ncu` named it
(L1TEX latency from 49.99%-bytes-per-sector uncoalesced loads). Read it before recording a mechanism
as a bound.*

### B3. cgo-free Metal (darwin, `--backend metal`) vs Ollama-Metal — 4-bit both sides

> **✅ RE-ANCHORED 2026-08-04 against current peer (Ollama v0.32.5, FA-on) and current
> goinfer (`38e5cd7`, post-Aug decode work: f16 scales, encode-ahead, half4 coalescing).**
> The old 0.32.0 rows (`~128`/`~61`, 2026-07-16) are superseded by the table below; the
> ratios barely moved (0.5B 1.03×→0.96×, 1.5B 0.77×→0.74×) — the peer got a bit faster on
> this box, goinfer's server-path figure is a touch lower than the stale approximate. The
> conclusion is unchanged: parity-ish on 0.5B, a real deficit on 1.5B, and **Metal's story
> is cgo-free, not raw speed.**

Provenance: **Apple M1 Pro, 16 GB**, macOS **26.5.2** · peer **Ollama v0.32.5** (FA `auto`,
`Flash Attention enabled` in the load log — the M0 on-box confirmation) · both engines driven
from the **identical local q4_K_M GGUF** (Ollama models `ollama create`d `FROM` the same file
goinfer loads — same checkpoint, not a different quantizer's build), **goinfer W4A8** (int4
weights / int8 activations — the 4-bit lane, speed-equivalent to int8int8 on this issue-bound
chip, §metal-verdict §4), warm, greedy (`temperature:0`), **short prompt**, 256-token
completions (both hit the cap), **client wall clock**, interleaved reps for thermal control,
first (session-start) run dropped, best-of-3. Measured on the Mac, **2026-08-04**. The `metal/`
package is `//go:build darwin && metal` and is selected at runtime with `--backend metal`; it
registers via `decoder.RegisterBackend` and must be blank-imported by the binary.

| model · q4_K_M (M1 Pro, warm, greedy, short prompt, wall-clock) | goinfer (Metal, W4A8) | Ollama-Metal v0.32.5 | goinfer vs peer |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | **~116 tok/s** | **~121** | **0.96×** |
| Qwen2.5-Coder-1.5B | **~54 tok/s** | **~73** | **0.74×** |

- **Metal's story is not raw speed — it's cgo-free / no-Xcode.** goinfer is near parity on
  0.5B and **loses ~1.35×** on 1.5B. Apple GPUs have no DP4A, so integer decode is issue-bound;
  the CUDA ratio does **not** carry over. The figure is size-dependent (0.96× → 0.74×).
- Do not quote a Metal speed *multiple* as a headline. The defensible Metal claims are
  portability (no Xcode, no toolchain, static binary) and correctness parity.
- **This is a SHORT-prompt number, deliberately.** goinfer's Metal backend **declines batched
  prefill by default** (it is not bit-identical to sequential decode — 54% stream divergence,
  §A2-Metal / `metal-verdict` §2c), so it prefills the prompt sequentially through the decode
  path. Ollama batches prefill (llama.cpp GEMM), so **on long prompts the wall-clock gap widens**
  (measured: a ~70-token prompt drops 0.5B to 0.83× and 1.5B to 0.66×). The short-prompt rows
  above isolate decode+serving, matching the §B2 method; the long-prompt penalty is a real
  serving trade of the bit-identity default, not a decode-kernel deficit.
- **Where the 1.5B wall-clock (~54) sits vs the decode-only spike (73.6, `task-metal-cgofree-spike`):**
  the spike is best-of-40 at a *shallow fixed* depth with zero serving cost; the wall-clock
  averages decode over depth ~10→266 (decode decays with KV depth — §A1-Metal 63.8@128→39.8@1024)
  and adds prefill + HTTP/JSON + detokenize + sampling. The two are consistent; the served rate
  asymptotes ~55–58 tok/s as fixed per-request overhead amortizes (16-tok req 30 tok/s → 512-tok 55).

#### Metal greedy decode by KV depth — qwen2.5-coder-1.5b (the depth axis, current binary)

The §B3 rows above are a single short-prompt point; this is the depth curve, the Metal analogue of
the CUDA §B6/§B7 sweep, so the Metal side stops being a pre-P6a single number. **Split-KV was never
enabled on Metal** (built, measured a regression, reverted — ollama-chase §A2-Metal), so unlike the
CUDA 0.5B curve these rows carry **no split-KV caveat**: this is the single attention path Metal
ships. Provenance: **Apple M1 Pro**, macOS 26.6.1, **qwen2.5-coder-1.5b W4A8**, greedy decode-only
(the resident `ForwardArgmax` on-device-argmax path, no serving/prompt overhead — isolates the depth
term), KV warmed incrementally, min-of-batches, current binary, 2026-08-09 (`TestZZ_metalDepthBench`,
opt-in `GOINFER_METAL_DEPTH_BENCH=1`). 4000 is near `metalCtxCap=4096`; the top cell is clamped
inside the resident KV.

| depth | goinfer 1.5B (Metal, decode-only) | µs/pos vs previous |
|---|---|---|
| 128  | 62.0 tok/s | — |
| 512  | 47.8 tok/s | +12.46 |
| 2048 | 27.2 tok/s | +10.30 |
| 4000 | 18.2 tok/s | +9.33 |

The per-position term is a **~9–12 µs/pos plateau** — roughly an order of magnitude above the same
model's CUDA coefficients (§B7: +0.55 / +0.99 µs/pos) and far above the peer's ~0.03–0.09. Decode
falls 3.4× over 128→4000 (vs 1.8× on CUDA), i.e. Metal degrades harder at depth. This is the
latency/occupancy-bound attention the half-width KV probe confirmed (plan §P4: q8 costs 88% of full,
so byte reduction is not the lever) plus the absence of split-KV — not a bandwidth wall. The lever is
the occupancy/latency rewrite, not KV-quant; q8 on Metal buys VRAM/reachability, not decode speed.

### B4. Host↔VRAM MoE streaming — a 26B that does not fit the card (cgo-free CUDA)

> **RE-ANCHORED 2026-08-27 on driver `595.91.07` / Nobara 44 — see §B4.1 below.** The throughput did
> not move: at the same 16 slots/layer this stack measures **11.42 tok/s** against the 11.3 recorded
> here, and the hit rate is identical at 57.3%. What DID move is what the card will grant — the same
> `GOINFER_MOE_CACHE_SLOTS=48` request now caps to **30** slots where it once reached 38, so the
> **16.98 tok/s @ 38 figure below is not reproducible on this stack because that configuration is no
> longer grantable**, not because the path got slower. The rows below are kept as the
> `595.58.03` record they are.

> **⚠ RE-VERIFIED against current Ollama (v0.32.5) — the "peers fail to load it" claim was
> false and has been retracted.** On the same RTX 2070 SUPER, Ollama **v0.32.5 loads and runs**
> Gemma 4 26B-A4B (google QAT q4_0 GGUF) by **splitting it 42% GPU (6.35 GB VRAM) / 58% CPU-RAM**,
> decoding at **~24.5 tok/s** — coherent (*"…the city of Paris."*, via `raw:true`; the bare
> `FROM gguf` Modelfile's empty templated output was a chat-template gap, not a load failure).
> That is **faster than goinfer's 16.98 tok/s** here. Current mainstream loaders offload
> oversized models to CPU rather than refusing them, so "peers can't run it" is not true of the
> 2026 peer. What remains true and specific to goinfer: it runs this model **fully GPU-resident**
> (every expert executes on the GPU, streamed host→VRAM per token) instead of running the
> oversized 58% on CPU — an *architecture* difference, not a capability the peer lacks. On **this**
> 8 GB / 15 GB-model mismatch, CPU offload (Ollama) wins on rate; goinfer's GPU-resident bet only
> pays off on a card the model nearly fits (see the capacity analysis below).

This is a **capability/architecture** row. **Gemma 4 26B-A4B** (128 experts, top-8, ~11.4 GB of
int4 experts) does not fit an 8 GB card. goinfer decodes it **fully GPU-resident** by keeping the
~1.3 GB non-expert core in VRAM and streaming the experts from pinned host memory into a VRAM slot
cache (host↔VRAM "C′" path, `docs/task-moe-streaming.md`) — the experts execute on the GPU, not
CPU. Current Ollama runs the same model by CPU-offloading the part that doesn't fit (measured
above); the two take opposite approaches to the same over-capacity problem.

| RTX 2070 SUPER · 8 GB · driver 595.58.03 · 2026-08-02 | value |
| --- | --- |
| model | Gemma 4 **26B-A4B**, int4 (128 experts, top-8, 30 layers) — experts ~11.4 GB, **does not fit 8 GB** |
| decode | **16.98 tok/s** (64-tok greedy, capture-free, synchronous H2D) — *see the reproducibility note below; not currently reproducible on this card* |
| expert cache | 38 slots/layer (auto-capped from 48 to measured free VRAM), **81.6% hit rate over the whole run** (17816 / 4024) — the steady-state decode figure is higher, **89.1%**, because this one includes the cold-cache fill (see §"production-config decomposition", `docs/task-moe-streaming.md`) |
| **configuration (required — not the default)** | `GOINFER_MOE_CACHE_EXPERTS=1` (host→VRAM expert streaming) + `GOINFER_MOE_CACHE_SLOTS=48` (auto-caps to 38). Omitting the second leaves the cache at its `topK` default — fresh-load per token, ~5 tok/s, not 17. *(`GOINFER_GEMMA4_RESIDENT=1` was also required when this was measured; Gemma-4 residency became unconditional in `a5ebb35` and the variable is now inert.)* |
| resident VRAM | ~1.3 GB core + ~3.8 GB slots + KV — the 11.4 GB of experts live in host RAM |
| coherence | greedy through the real chat template: distinct-trigram 0.818, *"…**Paris**… the Eiffel Tower, the Louvre Museum… **Gastronomy:**"* |
| peers | **Ollama v0.32.5: loads + runs** via 42% GPU / 58% CPU-RAM split, **~24.5 tok/s** (faster than goinfer here) — the old "fail to load" claim was outdated and is retracted |
| commit | `cuda/gemma4_26b_cache_test.go` (`GOINFER_HEAVY_TESTS=1`), host↔VRAM track through `2f51449` |

- The number is a **floor**: synchronous H2D with no async overlap yet, and it still clears
  fieldfare's 5.1–6.3 tok/s on its own 8 GB M2 Air (the closest analogue either project has —
  *different silicon, so a floor in the peer's constrained regime, not a comparison row*).
- **REPRODUCIBILITY (2026-08-12).** This row is **not currently reproducible on the same card**. It
  was measured at 38 slots/layer, which requires materially more free VRAM than the 26B gates now
  observe; at the free VRAM they see, the cap lands at **34 slots, and 34 fails outright** with
  `CUDA_ERROR_OUT_OF_MEMORY` at the first forward (both real-26B gates, plus a direct sweep — 30
  runs, 34 does not). The measurement was real; the configuration was narrow, running with roughly
  the forward's own demand left over. Why the cap can choose a value that allocates and then cannot
  launch is an open defect (A1, `docs/QUEUE.md`). Until it resolves, the README recommends the
  highest **measured-safe** value (30) rather than a computed one, and this row should be read as a
  ceiling reached once, not a configuration to aim at.
- **Reproducing this row needs three environment variables** (listed in the table above); a default
  build runs this model on CPU. The three hit-rate figures in the docs are three different
  measurements, not a disagreement: **77.5%** is 32 slots whole-run, **81.6%** is 38 slots whole-run
  (same 21840 expert reads, more slots), and **89.1%** is 38 slots *steady-state decode only*,
  which excludes the cold-cache fill. Quote the basis with the number.
- The 81.6%-hit-rate-at-30%-residency is the empirical basis of the design: trained MoE routing
  is a stationary skew, so an LRU cache over a small resident fraction captures most reads
  (`turbo-fieldfare` reaches the same conclusion via 16-slot LFU — two mechanisms, one property).
- **The bottleneck is CAPACITY, not MoE and not the kernels — and MoE is what would make it fast.**
  MoE is *cheap per token by design*: only top-8 of 128 experts activate, so ~4B of the 26B
  parameters are touched per token. A 26B-A4B that FIT VRAM would therefore decode *faster* than a
  dense 7B (fewer active weights/token). This one is at 16.98 tok/s only because it does not fit:
  the ~714 MB/token of routed experts stream host→VRAM over **PCIe (~12 GB/s, ~30× slower than
  VRAM's ~450 GB/s)**, and that DMA — not compute, router, or quantization — is the wall. Put the
  same model on a card that holds it (16 GB+) and it should beat the dense 7B, not trail it. So the
  26B's low rate is a **hardware-mismatch** result (`docs/completed/task-26b-prefill-bound.md`), and "other
  MoE models run faster" reduces to "other MoE models fit." Do not read this as an MoE or kernel
  deficiency and do not point IMMA/kernel work at it — the fix is memory, or a model that fits.
- **Which number to quote: 16.98 tok/s** (capture-free, the headline). The **4.98 tok/s** that also
  appears in `docs/task-moe-streaming.md` is the `GOINFER_G4_CAPTURE` readback / fresh-load
  (`nSlots=topK`, no reuse) FLOOR — informative, not the benchmark. Don't cite 4.98 as the rate.

> **serve caveat (both GPU backends).** GPU-resident models bypass the session cache in
> `cmd/serve` (`7557723`), so they skip prompt-prefix reuse and speculative decode. That
> is a deliberate trade — resident decode is worth far more than the TTFT optimization —
> but it means a resident server re-prefills each request. The numbers above include
> that re-prefill on goinfer's side.

> **Fresh re-measure (2026-07-14, CUDA-spike step 0):** the 1.5B-int8 row is re-measured
> on this 2070 SUPER after the buffer-coalescing win (`f8ef42b`/`5c3777f`) that postdated
> the 89.7 figure — **89.7 → 111.6 tok/s** (best of 6 × 48-tok greedy, resident int8,
> `TestDecodeRealModel_throughput`), closing the native-CUDA gap **61% → 76%**. This is
> the number the CUDA-megakernel-spike GO bar keys off: **≥1.3× = 145 tok/s int8**. The
> 7B-int4 row was not re-measured (also predates coalescing — treat as stale).

- The 7B-int4 row is the *footprint* headline: it **fits and decodes pure-GPU on an
  8 GB card** — the model class that does not fit at int8.
- goinfer-GPU is portable **WebGPU** (cgo quarantined behind `-tags gpu`) measured
  against years-tuned **native CUDA** — 60–70% of the CUDA engines at equal quant is
  the ceiling `gpu-assessment.md` predicted and hit (decode is glue-serialization-
  bound near a WebGPU wall, not bandwidth-bound; see §0.5 there).
- **Provenance gap (disclosed):** the 2026-06-08 source run did not pin the exact
  Ollama / llama.cpp **version**. Treat the ratios as "as-of-2026-06-08"; re-pin peer
  versions on the next run via `scripts/bench_compare.sh`. The checkpoints, quants,
  card, and greedy/warm conditions *are* matched.

---

## Reproduce it

`scripts/bench_compare.sh` runs **goinfer's** side end-to-end on your machine —
`BenchmarkDecode` (decode tok/s), `BenchmarkPrefillLong` (prefill), cold-start wall
clock to first token, and resident memory (`phys_footprint` on macOS, max-RSS on
Linux) — stamping each with the goinfer commit + date + machine. It then **prints the
peer commands verbatim** (`ollama run --verbose`, `llama-bench -ngl 99`, vLLM) for you
to run yourself on the same machine; it never drives the peers (their install is
yours). Peers absent → it still emits goinfer's column.

```bash
GOINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q8_0.gguf \
  scripts/bench_compare.sh            # add GOINFER_GPU=1 for the -tags gpu residency row
```

---

### B4.1 — Re-anchored 2026-08-27 (driver `595.91.07`, Nobara 44)

**Provenance.** RTX 2070 SUPER 8 GB · driver `595.91.07` · Nobara 44 · 2026-08-27 · goinfer
`d559c82` · Gemma 4 26B-A4B int4 (128 experts, top-8), checkpoint in `~/models/gemma-4-26b-a4b-it`
· `GOINFER_HEAVY_TESTS=1`, `-tags "cuda goinfer_testhooks"`, `TestGemma4_26B_cache_B` · 64 greedy
tokens, capture OFF, synchronous H2D · logs `b4-26b-cache-slots48_d559c82.log`,
`b4-26b-cache-slots16_d559c82.log`.

| `GOINFER_MOE_CACHE_SLOTS` | slots granted | ms/tok | tok/s | cache hit rate | load |
|---|---|---|---|---|---|
| 16 | 16 | 88 | **11.42** | 57.3% (12524 / 9316) | 4m56s |
| 48 | **30** (capped) | 62 | **16.12** | 76.1% (16611 / 5229) | 3m57s |

**The like-for-like row reproduces.** 16 slots gives 11.42 tok/s against the 11.3 recorded on
`595.58.03`, with the hit rate landing on 57.3% both times. The host↔VRAM streaming path is
unchanged by the driver, kernel, libc and distro move.

**What changed is the cap, not the throughput.** The cache sizes itself against free VRAM, and on
this stack the driver reports less of it:

    C′ cache: 48 slots/layer would need 4.9 GB VRAM but only 3.6 GB free — capping to 30 (3.1 GB)

38 slots is therefore not requestable here, which is the specific reason **16.98 tok/s is not
reproducible** — the configuration cannot be granted, rather than the path having regressed. Forcing
it is not an option worth taking: granting a cap one step past what fits is exactly the A1 defect
(`TestSlotAllocation_matchesGranularityForm`), where the forward died in a routing kernel with
189.6 MiB still nominally free.

**Hit rate scales with slots as the mechanism predicts** — 57.3% at 16, 76.1% at 30, 81.6% at 38
historically — and tok/s tracks it, since each hit is one expert DMA not paid.

**Why the cap moved, established by experiment rather than inferred.** The resident KV cache is
allocated BEFORE the expert cache is sized, and at the `cudaCtxCapDefault` of 4096 positions it
takes 2.01 GB of the card (30 layers × 8 kv heads × 256 head_dim × 2 × 4 B = 480 KB per position).
Halving the context to 2048 should therefore free ~1.0 GB, and at ~0.103 GB per slot buy ~9–10 more
of them. **Predicted 39–40 before running it; measured 40:**

| `GOINFER_26B_CTX` | KV | free VRAM | slots granted | hit rate | ms/tok | tok/s |
|---|---|---|---|---|---|---|
| 4096 (default) | 2.01 GB | 3.6 GB | 30 | 76.1% | 62 | 16.12 |
| **2048** | 1.01 GB | 4.5 GB | **40** | 82.2% | 57 | **17.62** |

So **the host↔VRAM path is not slower than it was on `595.58.03`; the default context starves the
cache.** 40 slots overshoots the historical 38, at a hit rate that brackets it (82.2% vs 81.6%),
and 17.62 tok/s sits above the 16.98 recorded there. Trading half the resident context for 33% more
slots is worth 9.3% of decode rate on this card.

That makes context-vs-slots a real knob rather than a footnote, and it is the *only* one that moves
the slot count here: everything else in the 8 GB is either the model core (~1.3 GB), the CUDA
context and allocator overhead, or the desktop. `GOINFER_26B_CTX` (test-only) and
`Options.ResidentContext` / `--ctx` (shipping) are the same lever.

**Still a capability number, not a benchmark.** The gate's own docstring says so: this is a
correctness-plus-latency run over a synchronous H2D path, with ~714 MB/token crossing PCIe. Both
runs pass their coherence floor (distinct-trigram 0.789 against a 0.70 bar). Current Ollama runs
this model faster on the same card by CPU-offloading 58% of it (~24.5 tok/s, §B4 above); goinfer's
distinction here is all-experts-on-GPU, which is an architecture difference rather than a rate win.

### B4.2 — What the expert-cache cap actually is, and why 38 was never safe

**This section is arithmetic plus already-published measurements — it adds no new timing row.** The
closed form is A1's (`docs/queue-performance.md`); the measured caps and rates are §B4 and §B4.1.

The cache asks for `GOINFER_MOE_CACHE_SLOTS` slots per layer and the runtime caps the request to
what free VRAM allows. **The cap is not a property of the card or of the build** — it is a function
of free VRAM at the moment `allocSlots` runs, and that number moves with the driver, the distro, and
the resident KV cache. A slot count is therefore only meaningful with the free-VRAM figure beside it.

Per slot, per layer, the backend allocates four buffers of `123,904 × {1, 2, 8, 16}` bytes, and the
driver rounds **each independently** up to its 2 MiB quantum. So the cost is a step function, not a
line:

```
  x   = n × 123,904 / 2,097,152
  Q(n) = ceil(x) + ceil(2x) + ceil(8x) + ceil(16x)          quanta per layer
  requirement(n) = 30 layers × Q(n) × 2,097,152
  grantable      when  requirement(n) + 402,653,184 ≤ free   (the margin)
```

**The form predicts the runtime's own log lines exactly, on a stack it was not derived on.** It was
fitted on driver `595.58.03`; these are `595.91.07`:

| the runtime printed | the form gives |
|---|---|
| "48 slots/layer would need **4.9 GB**" | requirement(48) = 4,907,335,680 |
| "capping to 30 (**3.1 GB**)" | requirement(30) = 3,145,728,000 |
| capped to 30 at "**3.6 GB** free" | cap 30 ⟺ free ∈ [3,548,381,184 , 3,611,295,744) |
| capped to 40 at "**4.5 GB** free" | cap 40 ⟺ free ∈ [4,492,099,584 , 4,617,928,704) |

Both bracket checks contain the logged figure, so the form survived a driver and distro major
upgrade without refitting.

| slots/layer | Q(n) | requirement | free needed (= req + 384 MiB margin) |
|---|---|---|---|
| 8 (default) | 14 | 880,803,840 | 1,283,457,024 |
| 16 | 27 | 1,698,693,120 | 2,101,346,304 |
| **30** | 50 | 3,145,728,000 | 3,548,381,184 |
| 31 | 51 | 3,208,642,560 | 3,611,295,744 |
| 33 | 54 | 3,397,386,240 | 3,800,039,424 |
| 34 | 58 | 3,649,044,480 | 4,051,697,664 |
| 38 | 62 | 3,900,702,720 | 4,303,355,904 |
| **40** | 65 | 4,089,446,400 | 4,492,099,584 |
| 48 | 78 | 4,907,335,680 | 5,309,988,864 |

Note 33 → 34: four quanta at once, because at n = 34 all four buffers cross a boundary together. 34
is the worst step in the range, and it is the value that was once recommended.

**What each stack actually granted, with the leftover column that distinguishes safe from lucky:**

| stack | context | free at `allocSlots` | cap | leftover after `allocSlots` | hit rate | tok/s |
|---|---|---|---|---|---|---|
| `595.58.03` | 4096 | 3,847,880,704 | **33** | 450,494,464 | *not measured* | *not measured* |
| `595.91.07` | 4096 | ~3.6 GB | **30** | 402,653,184 – 465,567,744 | 76.1% | **16.12** |
| `595.91.07` | 2048 | ~4.5 GB | **40** | 402,653,184 – 528,482,304 | 82.2% | **17.62** |

**Leftover is what the 38-slot figure was missing.** `16.98 tok/s @ 38 slots` (§B4) is recorded as
**measured-but-unsafe, not retracted** — it ran, and it produced coherent output. It was granted by
the old, quantum-blind cap, which sized the request without rounding each of the four buffers up
independently and so believed 38 fit.

**Two records of that run disagree, and the disagreement is left visible rather than resolved by
choosing.** `queue-performance.md`'s A2 records it surviving on *~133 MB of leftover*; the closed
form puts requirement(38) at 3,900,702,720, which already exceeds the 3,847,880,704 free recorded for that machine state, before the 384 MiB margin is even considered.
Both cannot be exact. The likeliest reading is that the ~133 MB was observed at a free-VRAM figure
that was never written down and differed from the one A1 later used to derive the form, which is
precisely why free VRAM now appears as a column here instead of being assumed. **The safety verdict
does not depend on which is right**: under either reading 38 was granted without the margin the cap
exists to preserve, and 34 — one step below it — is the worst quantum boundary in the whole range.

**The current best configuration is not 38 and is not on the historical stack.** Halving the
resident KV context frees ~1 GB, which buys ten more slots, which is worth more than the context
was: **40 slots at 17.62 tok/s** beats the old 16.98 outright. So the honest statement of this
row's history is that the peak was never reproduced *at 38* and did not need to be — the
configuration that beats it is a different, and safe, one.

**Operating guidance.** Request generously and let the cap decide (`GOINFER_MOE_CACHE_SLOTS=48`
grants whatever fits); the default of 8 is **inert** on this model, because top-8 routing fills it
exactly and nothing survives to the next token. If you want more slots, the lever is the resident
context, not the request.

## B5 — Anchored re-measure, 2026-08-09 (goinfer `686c9f8` vs Ollama v0.32.6)

> **SUPERSEDED — moved to [`benchmarks-archive.md`](benchmarks-archive.md).**
> Greedy rows are superseded by **§B8**; the sampled (`temp` / `temp+top_p`) and phi3-mini /
> gemma3-1b cells by **§B5.1** immediately below, which re-measured them on the current stack.
> The **v0.11.0 release qualification** that sat here was **RETIRED** rather than re-measured — its
> numbers were §B6/§B7 by code-identity, and both are now re-anchored.
>
> Two findings in the archived block are mechanism rather than measurement and were not withdrawn:
> the per-segment coefficients showing **the depth step is a kernel switch, not depth**, and the
> note that Ollama's depth behaviour must **not** be attributed to flash attention.

### B5.1 — Re-anchored 2026-08-27 (driver `595.91.07`, Nobara 44)

**Provenance.** RTX 2070 SUPER · driver `595.91.07` · Nobara 44 · 2026-08-27 · harness
`scripts/bench_peer.py` at goinfer `61b1e03`, serve binaries **built from `74718c2`** (cpu / cuda /
webgpu — the binaries are what was measured, not the harness commit) · peer **Ollama v0.32.5**
(`~/ollama-0325`); the superseded rows above used **v0.32.6** · both sides over HTTP, interleaved,
servers restarted per cell · 64 tokens × 8 completions × 2 runs, spread shown · sampling sent
explicitly to both sides, never assumed · int4 / q4_K_M · raw cells `b5-reanchor-61b1e03.json`
(34 cells) and `b5-reanchor-05b-61b1e03.json` (15), zero errors.

**Sampled configurations, depth 128, CUDA** — the rows §B5 could not carry forward:

| config | model | goinfer | Ollama v0.32.5 | verdict | old goinfer |
|---|---|---|---|---|---|
| temp 1.0, no truncation | qwen2.5-coder-0.5b | 237.7 | 268.7 | Ollama 1.13× | 219.2 → **+8.4%** |
| temp 1.0, no truncation | gemma3-1b | 143.7 | 148.2 | Ollama 1.03× | 131.7 → **+9.1%** |
| temp 1.0, no truncation | phi3-mini | 109.8 | 125.6 | Ollama 1.14× | 116.6 → **−5.8%** |
| temp 0.8 + top_p 0.95 | qwen2.5-coder-0.5b | 227.2 | 265.2 | Ollama 1.17× | 190.3 → **+19.4%** |
| temp 0.8 + top_p 0.95 | gemma3-1b | 124.7 | 148.2 | Ollama 1.19× | 115.2 → **+8.2%** |
| temp 0.8 + top_p 0.95 | phi3-mini | 102.1 | 125.7 | Ollama 1.23× | 99.4 → +2.7% |
| temp 0.8 + top_k 40 | gemma3-1b | **156.4** | 148.0 | **goinfer 1.06×** | — (new cell) |
| temp 0.8 + top_k 40 | phi3-mini | 112.1 | 125.7 | Ollama 1.12× | — (new cell) |

**The peer did not move, so the goinfer-side deltas are attributable.** Ollama reads 125.6 → 125.6
on phi3-mini and 149.1 → 148.2 on gemma3-1b across the two anchors, despite the point-release
difference. Five of six comparable cells improved, three of them by 8–19%.

**One regression, recorded rather than explained: phi3-mini at `temperature 1.0` is down 5.8%**
(116.6 → 109.8), well outside the old ±0.5 spread — while gemma3-1b and the 0.5B *gained* 8–9% on
the same configuration. That rules out a uniform sampler change and makes it model-specific. No
cause is offered here: guessing one in the same sitting as the measurement is how a wrong
explanation gets attached to a right number.

> **RESOLVED 2026-08-28 (G26) — the cause is optimistic forward, and HALF THE MAGNITUDE WAS THE
> ANCHOR'S OWN SPREAD.** Leaving the paragraph above as written: declining to guess was right, and
> the number it reports is a real measurement at its protocol. Both halves of it need amending.
>
> **The −5.8% overstates it by about 2x.** Re-measured at n=15, the anchor build's own temp1.0 cell
> spans **109.6 → 116.9**, and the 116.6 recorded above is the 14th of 15 sorted values — roughly its
> **90th percentile**. It was compared against a low-to-central draw of the newer build. Built from
> the same toolchain on the same driver and measured together, the two differ by **−2.6%**
> (114.40 ± 2.07 → 111.40 ± 0.55), not −5.8%.
>
> **The cause is `6a4e0ae`, optimistic forward**, gated `!fastGreedy` so it runs on sampled decode and
> never on greedy — which is why greedy was unaffected across 746 commits (−0.15%). A kill-switch A/B
> settles it: disabling the feature is worth **+5.8%** on that cell, landing HEAD 3.0% above the
> anchor. The sampler itself is exonerated — it is 12% *faster* at this vocab.
>
> **Fixed in `c9a79b4`**: the overlap is now capped at T ≤ 0.2, below the lowest measured break-even.
> **This row is therefore historical for a second reason** — it measures a binary whose sampled-decode
> behaviour has since changed.
>
> **And a caveat that outlives this row:** these cells use `scripts/prompts.json`, in which every
> prompt has four unique words. Correct for throughput, wrong for anything content-dependent. On a
> realistic prose prompt the same A/B moves by up to **9.4 points** on one cell. See
> `docs/spec/10-optfwd-gate.md` and G28.

**Greedy backends, depth 128** (the other cells §B5 could not carry):

| model | goinfer CPU | Ollama CPU | goinfer CUDA | Ollama CUDA | goinfer WebGPU |
|---|---|---|---|---|---|
| phi3-mini | 8.5 ±0.0 | 10.8 ±0.0 | 124.9 ±0.1 | 125.9 ±0.1 | 10.0 ±0.0 |
| gemma3-1b | 23.9 ±0.0 | 29.8 ±0.0 | **170.7** ±0.6 | 149.1 ±0.5 | 15.3 ±0.0 |

**Depth curve, CUDA greedy — it splits by architecture, which is the useful part:**

| depth | goinfer phi3-mini | Ollama | goinfer gemma3-1b | Ollama |
|---|---|---|---|---|
| 512 | 112.6 | 113.5 (Ollama 1.01×) | **164.3** | 148.4 (goinfer 1.11×) |
| 2048 | 78.3 | 92.2 (Ollama 1.18×) | **159.0** | 147.3 (goinfer 1.08×) |
| 3900 | 56.2 | 73.3 (Ollama 1.30×) | **165.1** | 147.9 (goinfer 1.12×) |

phi3-mini degrades with depth exactly as §B8 records for the dense qwen models — parity at 512,
then 1.18× and 1.30× behind. **gemma3-1b does not degrade at all**: 164 / 159 / 165, ahead at every
depth. Its sliding window is 512, so attention stops growing past it and the decode-depth penalty
never arrives. Stated as the mechanism rather than a general claim: goinfer's depth problem is a
property of full attention over a growing KV, and a windowed model does not have one.

## B6 — Split-KV decode attention, re-gated (2026-08-09, P6a)

> **SUPERSEDED — moved to [`benchmarks-archive.md`](benchmarks-archive.md).**
> Re-measured on the current stack with a committed harness (`scripts/bench_splitkv.py`); the ratios
> did not move, most cells agreeing to ±0.005. Current rows: **§B6.3** below. The archived block
> keeps B6.1 (what the new default recovers) and B6.2 (anchored depth rows on the fixed binary),
> including the note that these thresholds are **not device-portable**.

### B6.3 — Re-anchored 2026-08-27 (driver `595.91.07`, Nobara 44)

**Provenance.** RTX 2070 SUPER · driver `595.91.07` · Nobara 44 · 2026-08-27 · int4 q4_K_M,
greedy, `max_tokens=64`, one warm request discarded then 2 blocks of 8, decode-only rate timed
client-side from the first streamed token · a freshly started `serve` per cell · prompts
token-calibrated per model against `usage.prompt_tokens` · arms **paired** per (geometry, depth)
and adjacent in time, with arm order alternating so a first-cell effect cannot land on the same
arm every time · `scripts/bench_splitkv.py`, 48 cells per mode, ~53 min each.

**The binary is the anchor, not the commit field.** Both modes ran against one binary built from
`bb42106` (`serve-cuda-bb42106`, built once and reused). The JSON's `goinfer_commit` reads
`d1d3505` / `30a63f7` with `goinfer_tree_dirty: true`, because HEAD moved under the runs during
unrelated release work that day. Those fields record the working tree at write time and are **not**
what was measured; the pinned binary is.

**Two modes, because `GOINFER_SPLITKV_ATTN=1` does not force the split path** — it enables the
gate, which then decides per geometry from the table this section produced. Conflating the two
yields a table of 1.000s that looks like a null result.

**Mode `force` — split-KV itself (`GOINFER_SPLITKV_MIN_KEYS=0`) ÷ off.** This is the question the
2026-08-09 table asked, and the comparison against it is direct at 256+:

| geometry | 128 | 256 | 512 | 1024 | 2048 | 3900 |
|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | 0.858 | 0.843 | 0.824 | 0.825 | 0.967 | 1.111 |
| qwen2.5-coder-1.5b | 0.933 | 0.943 | 0.948 | 1.080 | 1.189 | 1.282 |
| gemma3-1b (win 512) | 0.915 | 0.888 | 0.900 | 0.935 | 0.959 | 1.090 |
| phi3-mini (MHA) | 0.990 | 0.992 | 0.968 | 0.915 | 0.814 | 0.746 |

Against 2026-08-09, cell by cell at 256+: 0.5B `0.843/0.824/0.825/0.967/1.111` vs
`0.839/0.819/0.869/0.955/1.197`; 1.5B `0.943/0.948/1.080/1.189/1.282` vs
`0.941/0.939/1.078/1.191/1.280`; gemma3-1b `0.888/0.900/0.935/0.959/1.090` vs
`0.890/0.909/0.919/0.941/1.084`; phi3-mini `0.992/0.968/0.915/0.814/0.746` vs
`0.993/0.969/0.919/0.815/0.754`. **The driver, kernel, libc and distro upgrade did not move these
ratios** — most cells agree to ±0.005, and phi3-mini's monotone decline to ~0.75 reproduces almost
exactly. The two visible gaps are 0.5B at 1024 and 3900.

**The 128 column is NOT comparable and is new information.** In 2026-08-09 it was a control: the
old gate fired only from 256 up, so its "ON" arm did not split at 128 and read ~1.000 by
construction. Forcing the split path at 128 costs 7–14% on three of the four geometries — the
shallow cost the control could not show.

**Mode `gate` — the shipped default ÷ off.** Not a re-anchor of anything; it asks whether the
per-geometry table now in `cuda/resident.go` costs anything against not splitting at all. **~1.000
is the pass condition here, not a null result.**

| geometry | 128 | 256 | 512 | 1024 | 2048 | 3900 |
|---|---|---|---|---|---|---|
| qwen2.5-coder-0.5b | 1.020 | 0.981 | 1.020 | 0.999 | 0.995 | 1.114 |
| qwen2.5-coder-1.5b | 1.008 | 0.984 | 1.009 | 1.083 | 1.193 | 1.281 |
| gemma3-1b (win 512) | 1.001 | 1.020 | 0.974 | 0.996 | 0.994 | 1.150 |
| phi3-mini (MHA) | 1.000 | 1.000 | 1.001 | 0.999 | 1.000 | 1.000 |

**Every regression the 2026-08-09 table recorded is gone, and the wins are kept.** phi3-mini's
`0.919 / 0.815 / 0.754` at 1024/2048/3900 — a −25% loss at depth, paid in production under the old
one-constant gate — is now flat `1.000` across the row: the "never" class declines to split, so
both arms run the same kernel. gemma3-1b's `0.890/0.909/0.919/0.941` return to ~1.0. And 1.5B's
wins survive intact: `1.083 / 1.193 / 1.281` against the forced arm's `1.080 / 1.189 / 1.282`,
because the gate correctly takes the split path exactly where it pays.

The three cells reading 0.974–0.984 (0.5B and 1.5B at 256, gemma3-1b at 512) are small and sit
near the crossover; they are recorded, not explained.

## B7 — Deep context: 8k/16k/32k decode (2026-08-09, cap-raise leg)

> **SUPERSEDED — moved to [`benchmarks-archive.md`](benchmarks-archive.md).**
> Current rows: **§B7.1** below — goinfer unchanged (39.1 vs 39.0 at the depth-matched 32000 cell),
> so the improved ratio is the peer moving, not a goinfer win.
>
> The archived block keeps two things that were not withdrawn: the per-segment µs-per-KV-position
> coefficients, and the finding that **the deep gap is NOT DRAM bandwidth** — where the alternative
> is falsified rather than merely assumed. *(Its `B7.1 — Control` subsection was renumbered to
> `B7 control` in the move; it collided with the §B7.1 re-anchor below.)*

## B7.1 — Re-anchored 2026-08-27 (driver `595.91.07`, Nobara 44)

**Provenance.** RTX 2070 SUPER · driver `595.91.07` · Nobara 44 · 2026-08-27 · goinfer `444b9a9`,
serve binaries built from `74718c2` · peer **Ollama v0.32.5** (`~/ollama-0325`) — the superseded
rows used **v0.32.6** · `-ctx 32768` on goinfer, `num_ctx 32768` and `OLLAMA_FLASH_ATTENTION=false`
on the peer, matching §B7's recorded configuration · deep protocol as §B7 defined it (`num_predict`
128, 1 warm + 2×2, not the shallow 17 completions) · token-calibrated prompts · decode-only,
client-timed from the first streamed token · 11 cells, zero errors · raw cells
`b7-deep-444b9a9.json`.

| depth | goinfer | Ollama v0.32.5 | verdict | goinfer 2026-08-09 | Ollama 2026-08-09 |
|---|---|---|---|---|---|
| 8000 ᵃ | 127.1 | 223.1 | Ollama 1.76× | 124.4 (@8192) | 259.1 (@8192) |
| 16000 ᵃ | 72.3 | 175.7 | Ollama 2.43× | 70.6 (@16384) | 239.7 (@16384) |
| **32000** | **39.1** | **128.4** | Ollama 3.28× | **39.0** | 215.9 |

ᵃ **Depth mismatch, stated rather than smoothed:** the anchor used 8192 / 16384; these prompts were
calibrated at 8000 / 16000. At the observed slope that is worth about 1% on the goinfer side and
under 1% on the peer's, so it does not carry the comparison — but **32000 is depth-matched exactly**
and is the row to read.

**goinfer did not move.** 39.1 against 39.0 at identical depth, and the two mismatched rows are
within ~2% once the depth difference is allowed for. The driver, kernel, libc and distro upgrade did
nothing to deep-context decode, which is the same answer §B4 and §B6 gave.

**The peer moved, hard: −40% at 32000** (215.9 → 128.4), −27% at ~16k, −14% at ~8k. That is a
different Ollama version (v0.32.6 → v0.32.5, i.e. the current bench peer is the OLDER build) on a
new driver stack, so the cause is not isolated here and is not attributed.

**Read the ratio in that light.** Ollama 5.54× → 3.28× at 32000 looks like goinfer halving a gap.
It is not: goinfer's number is the same to within 0.3%. Every part of the change is on the peer's
side. A ratio has two operands and this page has been bitten before by moving one and reporting the
quotient.

**Shallow backends at depth 128** (Phase A, same run, deep-ctx protocol): goinfer CPU 22.4, Ollama
CPU 57.4; goinfer CUDA **339.6**, Ollama CUDA 258.3 (goinfer 1.31×); goinfer WebGPU 126.9. The CUDA
row is consistent with §B8's 332.7 at the same depth on the same stack.

**Scope kept as the original set it.** The 1.5B/32000 cell is still skipped — 45–70 minutes for a
cell whose own note says no decision turns on it, and the 0.5B 32000 pair is the bend-detection
probe. It shows no second regime change past 16k: the goinfer curve continues its established
decline rather than bending.

## B8 — RE-ANCHORED to the Nobara 44 / driver 595.91.07 stack (2026-08-26, goinfer `a161bd6`)

**This section is the current anchor for goinfer-vs-Ollama greedy CUDA decode.** It supersedes the
peer rows in §B2 and §B5. §B4, §B6 and §B7 are **not** re-anchored by it and stay marked STALE —
`bench_peer.py` does not produce them, and marking them off this run would bless rows nothing
measured. The sampled (`temp` / `temp+top_p`) rows and the phi3-mini / gemma3-1b cells in §B5 are
likewise **not** covered: this sweep is greedy-only, qwen2.5-coder-only.

**Provenance, every row below:** goinfer **`a161bd6`** · peer **Ollama v0.32.5** (`~/ollama-0325`) ·
RTX 2070 SUPER, **driver 595.91.07**, **Nobara 44 / kernel 7.2.0-202.fc44 / glibc 2.43-8**, CUDA
13.2 reported by the driver · Ryzen 7 3700X · qwen2.5-coder **0.5B / 1.5B / 7B** at **q4_K_M**, same
weights both sides, from local NVMe under `~/models` · **2026-08-26** · **greedy** (`temperature 0`
sent explicitly to both; peer also `seed 1`) · **decode-only, prefill excluded** (inter-token rate
timed client-side from the first streamed token) · servers restarted per cell, **interleaved cell by
cell** · 64 tokens × 8 completions × 2 runs per cell, spread shown · **33/33 cells, zero errors** ·
GPU exclusive at start (the supervisor refuses to launch if anything beyond the compositor holds the
card) · **parity established first**: `gate gpu` PASS at the same sha, 39 minutes before the sweep.

> **Machine state on this row-set is ATTESTED, not instrument-read, and that is a defect being
> retired rather than a convention.** `bench_peer.py` recorded no load average until the change
> committed alongside these numbers; the operator attests the box was idle for the duration, and the
> GPU-exclusivity above *was* machine-enforced. Later runs carry a real header (driver, distro,
> kernel, commit, tree-dirty, peer version, binary mtimes, and load average + GPU temperature at
> every cell) and the harness now **refuses to start** on a non-idle box. Do not read this
> attestation as equivalent to that record — the archived results JSON marks the header
> `RECONSTRUCTED` and separates instrument-read from attested fields, field by field.

### Greedy decode by KV depth — the anchor table

| depth | goinfer 0.5B | Ollama 0.5B | goinfer 1.5B | Ollama 1.5B | goinfer 7B | Ollama 7B |
|---|---|---|---|---|---|---|
| 128 | **332.7** ±4.9 | 268.7 ±0.9 | **220.8** ±1.0 | 195.8 ±0.1 | **73.1** ±0.0 | 72.8 ±0.1 |
| 512 | **304.0** ±10.7 ᵍ | 267.9 ±0.5 | **196.5** ±1.9 | 171.2 ±14.8 ʰ | 69.6 ±0.1 | 72.3 ±0.0 |
| 2048 | 253.3 ±0.1 | 266.4 ±0.4 | 159.3 ±0.3 | 179.2 ±0.2 | 58.4 ±0.1 | 70.9 ±0.0 |
| 3900 | 202.5 ±0.2 | 258.6 ±0.0 | 123.1 ±0.3 | 174.2 ±0.3 | 49.0 ±0.0 | 69.5 ±0.1 |

| depth | 0.5B | 1.5B | 7B |
|---|---|---|---|
| 128 | **1.24×** | **1.13×** | 1.00× |
| 512 | **1.13×** | **1.15×** | 0.96× |
| 2048 | 0.95× | 0.89× | 0.82× |
| 3900 | 0.78× | 0.71× | 0.71× |

ᵍ 3.5% spread — the widest cell in the set and the only one near this page's 5% threshold. Indicative.
ʰ 8.6% spread on the PEER side, reproducing the instability §B5 recorded at this exact cell
(spread 27.0 there, 146–182 over ten runs in the campaign before it). Three campaigns, three
binaries, two OS stacks, same cell: this is a property of Ollama at 1.5B/512 on this card, not of a
session. Treat the 1.15× at that cell as indicative.

**The picture is unchanged by the upgrade**: a real win on tiny models at short context, parity at
7B/128, and a widening loss with depth on every model. Depth still stops at 3900 (`cudaCtxCap`).

### The backend table, 128 context, greedy

| model | goinfer CPU | Ollama CPU | goinfer CUDA | Ollama CUDA | goinfer WebGPU ⁱ |
|---|---|---|---|---|---|
| 0.5B | 23.5 ±0.1 | 57.9 ±0.1 | **332.7** ±4.9 | 268.7 ±0.9 | 127.6 ±1.0 |
| 1.5B | 17.6 ±0.0 | 24.2 ±0.0 | **220.8** ±1.0 | 195.8 ±0.1 | 90.1 ±0.4 |
| 7B | 4.9 ±0.0 | 6.0 ±0.0 | 73.1 ±0.0 | 72.8 ±0.1 | 46.1 ±0.0 |

ⁱ **Cross-backend, not a peer cell.** Ollama has no WebGPU build, so this column has no counterpart
and must never be presented as a like-for-like comparison. It is here to place goinfer's portable
backend against its own native one: WebGPU runs at **38 / 41 / 63%** of goinfer's own CUDA at 0.5B /
1.5B / 7B — the gap narrows as the model grows, which is what a dispatch-overhead story predicts.

### The peer is the control, and it held to <0.5%

The point of re-measuring both sides is that the peer becomes a control on everything that is not
goinfer. Across a **distro major upgrade** — new driver, new kernel, new libc, new graphics stack —
Ollama reproduced its 2026-08-09 numbers cell for cell:

| cell | Ollama 2026-08-09 (driver 595.58.03) | Ollama 2026-08-26 (driver 595.91.07) | Δ |
|---|---|---|---|
| 0.5B @128 | 269.4 | 268.7 | −0.3% |
| 0.5B @512 | 269.6 | 267.9 | −0.6% |
| 0.5B @2048 | 266.4 | 266.4 | 0.0% |
| 0.5B @3900 | 259.8 | 258.6 | −0.5% |
| 1.5B @128 | 195.4 | 195.8 | +0.2% |
| 1.5B @2048 | 179.5 | 179.2 | −0.2% |
| 1.5B @3900 | 174.3 | 174.2 | −0.1% |

Seven cells, every one inside **0.7%** (largest deviation 0.63%, at 0.5B @512), on a box whose
documented between-session drift is ~3.5%. Two
things follow, and only two. **The stack upgrade did not move decode throughput** for an engine that
did not change. And **the harness and the box are stable enough that a goinfer-side delta of more
than ~1% is attributable** rather than dismissible as session noise.

goinfer's own cells moved **+0.7% to +4.3%** against 2026-08-09 at every depth except 0.5B @512
(243.2 → 304.0) and 1.5B @512 (181.9 → 196.5). Those two are not claimed as a speedup here: the
0.5B @512 cell carries this set's widest spread, the old 0.5B row was **non-monotonic** (512 read
*below* 2048, 243.2 vs 244.0, which a depth curve should not do), and the binaries differ — so the
honest reading is that the old 512 cells were suspect, not that something got 25% faster. **The
comparison in this subsection is cross-session and indicative; the anchor is the table above it.**

## B9 — tables relocated from the README front page (2026-08-27)

> **STALE, and moved to [`benchmarks-archive.md`](benchmarks-archive.md).**
> Measured against **Ollama v0.32.6** on the pre-2026-08-25 stack. Greedy rows are superseded by
> **§B8**; the sampled configurations the block was kept for are superseded by **§B5.1**, which
> re-measured them on the current stack the same day the block was written — so its own claim to
> hold the only sampling table no longer holds. Nothing here is a current claim.

## Measurement notes worth keeping

- **Interleaving is not optional, and skipping it is invisible in the output.** Absolute sampled
  numbers on this box drift ~3.5% between sessions (proven: the same binary read 112.4 in one session
  and 116.5 in another, while a *different* binary read 116.4 alongside it). A ratio built from two
  engines measured in different sessions silently absorbs that drift — which is what made the original
  phi3-mini row read 1.12× where the same-session pair reads 1.08×. If a cell's two sides were not
  measured back-to-back, the verdict column is not a measurement of the engines.
- **A fresh server per cell does not protect you from the PREVIOUS engine.** Interleaved peer
  benchmarking kills engine A and starts engine B, but Ollama keeps a model resident for minutes after
  its last request, so B's first cells can contend with A still unloading. This produced a −52.9%
  phantom regression in §B7.1 that was arm-specific — the tell that it was an artifact, not a result.
  Wait for a verified-idle device (`nvidia-smi` at ~0% and baseline memory) between engines, and treat
  any anomaly in the FIRST cells of a run as suspect until re-run.
- **Early EOS at `temperature 1.0` shortens completions.** In the 2026-08-09 sweep, **4 of 16**
  completions at goinfer's default sampling terminated before the 64-token cap (11/36/45/61 tokens);
  greedy hit the cap all 16 times. Short completions make a per-token rate noisier. This is a
  property of *that configuration* (untruncated sampling reaches EOS sooner), **not a measurement
  fault** — the rate itself is unaffected. Any future sweep at `temperature > 0` without truncation
  should expect it and report run counts.
- **One peer cell is genuinely high-variance.** Ollama v0.32.5 at 1.5B/512 measured 146–182 tok/s
  across ten runs (mean 162.7, stdev 11.2) while goinfer was stable at 184.7 ±1.2 in the same cell,
  interleaved. A two-run sample there read as a near-tie; ten runs read as a ~1.11× goinfer win.
  Do not quote that cell without its run count.

## Maintenance rules (so this page never rots into a lie)

- **Every number carries its date + goinfer commit + peer version + sampling config, inline.**
  No floating "~90 tok/s" without the run that produced it — and no number without its
  **sampling configuration** (greedy, or `temperature`/`top_p` with their values), for goinfer
  *and* the peer. Sampling config is a REQUIRED per-number field, on the same footing as machine,
  driver, peer version, and date: a row's throughput is only interpretable once you know whether
  it was sampled greedily or with temperature+top_p (the two can differ by an order of magnitude
  on the same engine). Any existing row that does not state one is marked **sampling: unrecorded**
  and must **not** be assumed greedy.
- **A host-stack change re-anchors; it never carries forward.** The NVIDIA driver is already a
  required per-row field, and the rule extends to what sits under it: **kernel, libc and distro**.
  When any of them moves, mark every affected row STALE *the same day* — before any re-measurement
  exists — and record what moved from the package manager's own log (`dnf history info <id>`), not
  from memory. A stale row that still reads as current is the failure mode this page exists to
  prevent, and the window between "the stack moved" and "the numbers are back" is exactly where it
  happens. Re-measure with `scripts/bench_peer.py`; `bench_compare.sh` does not drive the peer and
  cannot produce a comparison.
- **Re-run `scripts/bench_compare.sh` at each tag.** A number more than one minor
  version stale is re-measured or struck — never silently carried forward.
- **Re-verify the capability matrix against peer release notes at each tag** (cheap —
  it's mostly booleans). Peers move: Ollama dropped its CGO runner for a `llama-server`
  subprocess (PR #16031, merged 2026-05-29); mistral.rs shipped multi-model + an OpenAI
  Responses API. A stale ✓/✗ is as misleading as a stale tok/s.
- **The bar is provenance, not flattery.** If a cell can't be traced to a measurement
  or a peer's own docs, cut it.

---

## Sources

**goinfer (in-repo):** `docs/completed/gpu-assessment.md` §0.0 (GPU residency: 89.7 / 51.7 tok/s;
61% / 71% of CUDA at equal quant; the discarded wrong-model 191 tok/s; commit `eaf9a6c`,
2026-06-08) · `docs/ARCHITECTURE.md` §2 (cold-start / heap / binary tiers; embedded-GGUF
vs `.giw`) · `docs/completed/perf-campaign.md` (M1 Pro CPU decode, measurement conventions) ·
`CHANGELOG.md` v0.4.0/v0.5.0 (prefill 3.4× / 1.7×; MoE prefill `08acc11`) ·
`decoder/registry.go` (13 `model_type` keys, 11 distinct architectures) · `CHANGELOG.md`
v0.5.0 (the 7B int4 row: **51.7 vs llama.cpp-CUDA 72.8 tok/s = ~71%** — this peer
figure lives in CHANGELOG, not gpu-assessment) · `README.md` (capabilities) ·
**§B2 (cgo-free CUDA)** — measured 2026-07-16 on this repo at commit `7557723`, RTX 2070
SUPER / driver 595.58.03, peer Ollama 0.5.7 via `/api/generate`; goinfer via `cmd/serve`
`/v1/chat/completions`; both wall-clock, best of 3 warm; lab notes in
`docs/completed/task-cuda-cgofree-spike.md` · **§B3 (cgo-free Metal)** — **re-anchored 2026-08-04** on
Apple M1 Pro 16 GB / macOS 26.5.2, goinfer `38e5cd7` (W4A8) via `cmd/serve` `/v1/chat/completions`,
peer **Ollama v0.32.5** (FA-on) via `/api/chat`, both from the identical local q4_K_M GGUF, same
server-to-server wall-clock method; notes in `docs/completed/metal-model-coverage.md`.

**Peers (verified 2026-06-10, against each project's repo/docs):**
llama.cpp — github.com/ggml-org/llama.cpp (README; `tools/server/README.md`;
`grammars/README.md`; `docs/multimodal.md`).
Ollama — github.com/ollama/ollama PR #16031 (CGO runner removed → `llama-server` + MLX,
merged 2026-05-29); docs.ollama.com (modelfile/LoRA, structured outputs, OpenAI API).
mistral.rs — github.com/EricLBuehler/mistral.rs (README; `guides/serve/openai-responses-api.md`;
`guides/serve/multiple-models.md`; v0.8.3, 2026-06-01).
vLLM — github.com/vllm-project/vllm (README, 200+ archs); docs.vllm.ai
(structured outputs / xgrammar); en.wikipedia.org/wiki/VLLM (V1 engine).
gollama.cpp — github.com/dianlight/gollama.cpp (README: purego bindings, dlopen native
llama.cpp; v0.2.2-llamacpp.b6862).
yzma — github.com/hybridgroup/yzma (README: purego+ffi, `llama.Load(libPath)`, VLM
examples; v1.16.1, 2026-06-08).
llama2.go — github.com/nikolaydubina/llama2.go (README; **archived 2024-11-30**, CPU-only).

> *Peer-state note:* the web sweep that populated the peer columns read each project's
> canonical repo files/READMEs (GitHub) and search summaries; a couple of cells were
> marked "unverified" by the sweep and are `—` here rather than guessed. Before a
> release that leans on this page, re-open the flagged primary URLs to confirm wording.

## B10 — decode-path decision matrix, re-measured (2026-09-02, goinfer `cd6895c2`)

Replaces `docs/completed/gpu-assessment.md` §1, whose table dates from 2026-06-08. That page is
archived and its numbers are left as the record they are; **this is the current one.** Two
things moved since: the G35/G36 kernel work (a serial `quantize` reduce and an attention kernel
that reduced once per key), and resident prefix reuse, which splits TTFT into two numbers that
mean different things.

**Provenance.** RTX 2070 SUPER, NVIDIA driver `595.91.07`, Nobara 44 · Ryzen 3700X ·
goinfer `cd6895c2`, aikit `v1.31.0` · Qwen2.5-Coder-1.5B: `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`
and `~/models/qwen15-w4a8-int4.giw`, both on local NVMe · greedy (temperature 0) ·
`gpu/matrix_bench_test.go` via `decoder.Generate`, so token counts come from the engine, not
from counting SSE frames · warm (a discarded generation compiles the pipelines first) ·
**two full runs, both reported** — no peer is involved, so these are goinfer-vs-goinfer only.

| path | fits / VRAM | decode tok/s | TTFT 256-tok, cold | TTFT, warm |
|---|---|---|---|---|
| CPU int8 | yes (host) | 13.7 / 13.7 | 4773 / 4721 ms | — |
| CPU int4 | yes (host) | 16.6 / 16.4 | 5580 / 5633 ms | — |
| GPU staged (int8) | yes / 485 MiB | 20.5 / 22.4 | 5686 / 5374 ms | — |
| **GPU residency int8** | yes / 3237 MiB | **118.3 / 113.9** | 1942 / 1938 ms | **7 / 8 ms** |
| **GPU residency int4** | yes / 2535 MiB | **137.9 / 137.4** | 1748 / 1713 ms | **7 / 7 ms** |

**Decode is up ~1.25–1.35× on the resident paths** against the June table's 95.1 (int8) and
102.8 (int4) — the G35/G36 kernels, measured here through a different harness than the one that
produced those results, which is the corroboration worth having. The staged and CPU paths are
unchanged within noise, as expected: neither runs the rewritten kernels.

**The two TTFT columns are both real and describe different moments.** Cold is a new
conversation, or one whose prompt diverged from what the resident cache holds. Warm is every
subsequent turn of an agent loop, where prefix reuse prefills only what changed. Quoting cold
alone overstates a loop's cost by ~250× here; quoting warm alone overstates a first request.
Non-resident paths show `—` rather than 0: they cannot reuse at all (it is gated on
`m.resident != nil`), and a 0 would read as "instant" instead of "not applicable".

**Read `tok/s` and `TTFT` as separate questions.** int4 decodes faster than int8 (137 vs 118)
*and* has a lower cold TTFT, but the gap between them is far smaller than the gap either has to
its own warm number — which is the practical point: on a multi-turn workload the prefill term,
not the quant, dominates what a user feels.

---

## Peer matrix 2026-09 — tier 1/2, first pass (2026-09-04/05, IN PROGRESS)

Executes `docs/task-peer-benchmarks.md`'s redone matrix via `scripts/bench_peer.py`. **This is a
first pass, not the finished matrix** — W2 (prefill), W4 (agent-turn replay), the fidelity column,
and the pass@1 row are all still unbuilt (see "Not done yet" below), and one re-measurement is
still in flight as this section is written. Raw provenance-stamped JSON for every cell below is at
`docs/measurements/peer-matrix-2026-09/*.json`; this section will be updated in place as the
missing pieces land — treat a stale-looking gap here as "not yet written up", not "not measured".

**Provenance (both boxes).** Nobara: RTX 2070 SUPER, driver `595.91.07`, Nobara Linux 44 · CUDA ·
goinfer built from `abcdd1fe` (product code; harness commits on top only change
`scripts/bench_peer.py`, not the binary) · Ollama `0.32.5` · llama-server `0.4.0-dev (build 1,
commit 427291b)`, built from source with CUDA. Mac: Apple M1 Pro, macOS 25.6.0/Darwin, 16 GB RAM ·
Metal · goinfer built from `abcdd1fe` · Ollama `0.32.5` · llama-server `0.3.0 (build 10621, commit
c1d0e7a00)` · MLX (`mlx-lm` 0.31.3, `mlx` 0.32.2, version not yet in the harness's own provenance
record — a gap to close). All cells: greedy (temperature 0), depth-128 unless stated, n=5 runs per
cell except the depth-8000 cells (n=2, ncomp=2 — a single 8k completion costs minutes, see below),
same-session interleaved with a server restart between cells, idle-gated (box refuses to measure
above 1-min loadavg 1.0).

### D7 (Qwen2.5-7B-Instruct, GGUF Q4_K_M) — W1, depth 128

| box | backend | goinfer | Ollama | llama.cpp | MLX |
|---|---|---|---|---|---|
| nobara | CUDA | 73.0 | 74.3 | 80.4 | — |
| Mac | Metal | 21.8 | 25.5 | 26.1 | 37.9 |

### S (Qwen2.5-Coder-1.5B, GGUF Q4_K_M) — W1, depth 128

| box | backend | goinfer | Ollama | llama.cpp | MLX |
|---|---|---|---|---|---|
| Mac | Metal | 72.7 | 84.3 | 86.2 | 109.8 |
| nobara | **CPU** (tier-2 pure-Go lane) | 17.7 | 24.3 | 27.1 | n/a |

`go-llama`/`goccy`, the task doc's other CPU-lane peer, is not installed on either box yet — the
CPU-lane row above is Ollama/llama.cpp/goinfer only.

### M35 (Qwen3.6-35B-A3B) and M26 (Gemma-4-26B-A4B) — W1, depth 128, nobara CUDA only

Both are MoE models past this card's 8 GB VRAM, so goinfer runs `-moe-cache-experts` streaming and
llama.cpp needs `--fit`'s auto-offload rather than its `-ngl 99` default (see the fix below).
**Quant note:** neither checkpoint had a real Q4_K_M GGUF in the archive. Both were requantized
locally with `llama-quantize --allow-requantize` — M35 from Q8_0 (near-lossless), **M26 from an
existing Q4_0 (a double quantization — flag this wherever M26's row is quoted for quality, not
just speed)**. goinfer itself does not load either GGUF: it runs its own kind-4 `.giw` bundle for
both (the shipped-default configuration), so goinfer's column and the peer columns are not reading
byte-identical files, only the same nominal quant tier.

| model | goinfer | Ollama | llama.cpp (`-ngl 99`, as shipped) | llama.cpp (fixed) |
|---|---|---|---|---|
| M35 | 23.5 | 23.9 | never loaded (900s timeout) | **32.9** |
| M26 | 24.6 | 22.2 | never loaded (900s timeout) | **27.8** |

**The llama.cpp fix, worth keeping as a finding on its own:** llama-server's own `--fit` is on by
default and auto-places layers across host/GPU — but only for arguments left *unset*. The harness
was passing `-ngl 99` unconditionally, which forces full GPU offload of a 20 GB/16.8 GB model into
an 8 GB card regardless of `--fit`, and both cells ran their full 900s load-wait and never came up.
Dropping `-ngl` for these two model keys on the CUDA backend (CPU-forcing elsewhere untouched) lets
`--fit` place layers automatically — the exact "zero-flag" mode `docs/task-fit-to-hardware.md`
is about. llama.cpp went from *unable to run these cells at all* to *winning both of them.*

### W3 — long-context decode at depth 8000

| model | box | goinfer decode tok/s (pre-fix → post-fix) | goinfer wall-clock (pre-fix → post-fix) | Ollama | llama.cpp |
|---|---|---|---|---|---|
| D7 | nobara CUDA | 35.7 | — | 56.7 | 58.7 |
| M35 | nobara CUDA | 21.4 → 21.6 | 1528.8s → 1489.9s (**2.5%, noise-level**) | 23.2 | 30.8 |
| M26 | nobara CUDA | 15.5 → 15.5 | 384.3s → 368.4s (**4.1%, real**) | 19.8 | 22.7 |
| S | Mac Metal, `--cpu-fast-attention` | 16.84 (`true`) / 16.53 (`false`) | — | — | — |

**M35/M26 wall-clock, not the tok/s column, was the real finding — and it's now been re-measured
post-fix.** `tok/s` is decode-only by construction (timed from the first streamed token), so it
never showed prefill cost. The *cell wall-clock* did: goinfer's M35 cell took 1528.8s (~25.5 min)
and M26's took 384.3s (~6.4 min) despite being CUDA-resident, consistent with repaying the full 8k
prefill on every completion via a **sequential (non-batched) MoE prefill path** — while Ollama and
llama.cpp's much shorter wall-clocks on the same cells suggest both reuse the repeated prefix. A
separate, unrelated fix landed on the nobara box the same night this was found (`4ee59e15`/
`654fa481`/`a9c23c67`/`5cc48545` — "MoE models take the batched prefill path (P20 blocker 3)"),
and a same-night correctness fix to it (`9453430e`, see below) — **re-run against a binary built
at `9453430e` specifically, not the earlier commit, so the numbers reflect the guard's final,
correct form:**

- **M35: no real change (2.5%, noise-level).** Expected and correct — M35 is Gated-DeltaNet and
  `9453430e` makes it decline the batched path explicitly, the same path it was already (by
  accident) taking before the fix. This row was never going to move; it's here as confirmation the
  guard behaves as designed, not as a finding.
- **M26: 4.1% faster, a real if modest win.** Consistent with `a9c23c67`'s own attribution finding
  that the host→VRAM expert DMA is ~59.5% of M26's prefill wall-clock and untouched by batching —
  batching only speeds up the row-loop compute around it. The in-process paired measurement found
  8.3–8.5% there; 4.1% end-to-end (which also includes the unaffected decode phase, diluting it
  further) is consistent with that, not a discrepancy.
- **Neither model's decode tok/s moved**, as expected — the fix is prefill-only.

**A latent correctness bug in that same batched-prefill work was caught and fixed same-night,
before any shipped row was actually wrong — worth recording here because it bears directly on
M35's row.** `654fa481` removed an `r.moe` refusal that was, by accident, the only thing keeping
M35 (Gated-DeltaNet) off the batched path at all — batched prefill has no notion of recurrent
state, and a DeltaNet layer's conv/matrix state must advance one token at a time; a batched pass
would silently return plausible-but-wrong logits. `ForwardN` already excluded this deliberately
(`r.prefillReady && r.dnet == nil`); the new `PrefillLast` path never had to, because `r.moe` was
still declining M35 for an unrelated reason. Checked, not assumed: no shipped commit was actually
unsafe, because DeltaNet layers load no q/k/o weights, so an existing empty-string sentinel in
`nonBatchableKind` still caught M35 by coincidence — **an accident of this family's weight layout,
not a guard.** A hybrid whose recurrent layers also carried q/k/o would have sailed through into
the dense attention stack — the same silent-wrong-computation bug class as the LFM2 incident this
repo's `CLAUDE.md` already tracks (reached `main` twice). Fixed in `9453430e`:
`prefillStaticDecline` now refuses `r.dnet != nil` directly, for the true reason, with a
mutation-proven test fixture (a synthetic DeltaNet model carrying valid int4 q/k/o — the one shape
the accidental guard could not have caught, which no real checkpoint here produces). **Net effect
on the numbers above: none** — M35 declined the batched path both before and after, so its M35 row
is unaffected either way; the fix is about correctness-by-construction going forward, not a
retraction of anything measured here.

**S's `--cpu-fast-attention` row is decode-only and does not yet answer the question it was run
for.** The flag is documented as prefill-only, so the ~2% decode gap above is expected and not
the finding — but the harness that produced it reused `bench_peer.py`'s decode-only timing design
without adding TTFT/prefill instrumentation, so the actual prefill-time comparison this row exists
to answer is still open. Cost 3h38m wall-clock to learn that. A corrected re-run with real TTFT
capture is a follow-up, not done here.

### G20 (gpt-oss-20b, MXFP4) — W1, depth 128, tier 2

The task doc frames this model as "fits the Mac, not the card" — a resident cell on one box, an
offload cell on the other. Measured tonight, that framing held on nobara and did **not** hold on
the Mac.

| box | backend | goinfer | Ollama | llama.cpp |
|---|---|---|---|---|
| nobara | CUDA (`-moe-cache-experts`) | **62.4** | 26.5 | 31.6 |
| Mac | CPU | declined — see below | not attempted | not attempted |

**Mac capability boundary, not a bug.** goinfer's `gpt_oss` architecture has no resident
CUDA/Metal/WebGPU backend today (`decoder/registry.go`, `decoder/features.go`), so this cell must
run `-backend cpu` on any box. On the Mac, a plain CPU decode of the 13.8 GB checkpoint — no
`-stream-weights` paging, the model nominally fits 16 GB RAM on paper — drove swap to 22.6–22.9 GB
on a 23.5 GB swap file. Caught via a single-request smoke test and killed before a real measurement
was taken, rather than letting it run — this same night already had one kernel-panic incident from
a related failure mode (see "M35/M26 on the Mac" below), and this was judged the same class of
risk. Ollama/llama.cpp legs on the Mac were not attempted for the same reason. Nobara's CUDA
result stands on its own: goinfer wins by a wide margin here.

### M35/M26 on the Mac — parked, a real capability boundary, not a partial result

The Mac has 16 GB RAM; goinfer's Metal backend measured-declines full residency for both models
(M35 needs 20.6 GB, M26 14.95 GB, against an 11.2 GB budget — 70% of 16 GB) and falls back
automatically to a CPU-staged, `-stream-weights` decode path. That path was run for real and killed
after **2h10min with zero completions** — RSS stayed at ~3.2 GB against a 20 GB model the whole
time, consistent with re-reading weights from disk per token rather than holding them resident.
This measured, real slowness is very likely what produced a genuine kernel panic
(`panic(...): watchdog timeout: no checkins from watchdogd in 92 seconds`) shortly afterward on
this machine. **M35, M26, and H27 (also too large for this Mac) are off-limits on this box going
forward, on any path** — not a temporary skip, a hardware boundary for this machine as configured.
llama.cpp and Ollama, which don't go through goinfer's residency guard, ran fine on the Mac for
M35/M26 at depth 128 (see the W1 table above) — the boundary is specific to goinfer's own
CPU-staged fallback, not to running these models on this hardware at all.

### Tier-2 goinfer variants, Mac only

| variant | model | tok/s | vs. int4/on baseline |
|---|---|---|---|
| `--quant int8int8` | S | 72.8 | 72.7 (int4) — no meaningful difference |
| `--quant int8int8` | D7 | 21.9 | 21.8 (int4) — no meaningful difference |

The task doc's own note that "the ordering flipped with the tile" for int8int8 vs int4 did **not**
replicate on this Mac for either S or D7 — both land within noise of the int4 baseline. Worth a
second look on the CUDA side before treating this as settled either way.

### Not done yet

- **W2** (prefill at 512/3900 tokens, TTFT) — no cells run.
- **W4** (the agent-turn transcript replay) — the harness for it doesn't exist; this is the
  workload the task doc calls the one that "matters most" and hasn't been started.
- **Fidelity column** (teacher-forced top-1 agreement) and **pass@1** (HumanEval+/MBPP+) — neither
  scorer is built. Every row above is speed-only; no quality claim should be read into any of them.
- **FreeToken** (nobara's platform-specialist peer) — investigated and declined: needs CUDA 13
  (box has 12.6) and only supports RTX 30/40/50-series GPUs (this box is a Turing 2070 SUPER). A
  real hardware mismatch, not an oversight.
- **`go-llama`/`goccy`** (the Go-lane CPU peer) — not installed on either box.
- **W3 at 2k/32k** — only 8k has been run.
- **M35/M26 W3 post-fix re-run** — in progress, see the note above.
