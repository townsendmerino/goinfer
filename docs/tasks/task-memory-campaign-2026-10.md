# Task: the memory campaign — what is left after the KV program (MEM0–MEM17), 2026-10

> **Status 2026-10-21 (the close-out sweep).** The aikit side shipped (aikit v1.71.0, `gpu/annmetal/v0.2.1`): MEM3, MEM14, MEM15
> and MEM16 are done, and so is MEM13's job-store and token-bytes half (#94, cacaa8ec). MEM0's nobara half is read (33ef5a21);
> its Mac half is not. The rest of the goinfer items have not started. **The premise that the KV program is finished does not
> hold** (KV4 has not started), but only MEM9(c) waits on it. **Decided by the owner on 2026-10-21:** D1, library callers get
> the mapped sidecar through a `decoder.Options` field that is **on by default**, so a library `.gguf` load transcodes once
> and writes the `.giw` beside the model, as the apps do. D2: after the GPU attaches, the towers' host copy is dropped and
> reloaded lazily from the mapping on a CPU fallback. D3: bf16 kinds for the f32 islands first, then int8 per family behind
> its parity bar. D4: aikit's `chunk.Chunk` becomes a file and a byte range read through a mapping, and ken stores chunks that
> way. D5 stays with O3 (`docs/tasks/parked/task-metal-audit-followups-2026-10.md`), parked with a "defer" call. **Moved here
> the same day:** MEM12 takes the unused f32 KV reservation from the archived cpu-kv-f16 doc (`NewKVCache` reserves f32
> capacity and `setQuant` keeps it when the storage form changes; int8 has the same leftover). MEM13 takes the two staging
> copies from the archived never-swap doc (Metal's int4 lane on a direct `.gguf` load, `metal/model.go:int4Concat`, and
> CUDA's dense host pack), decided: remove both. **Stale below:** MEM13's "in PR #94" (merged); §8's "Next" line; §0's MEM3
> row; MEM5's "blocked on G40-1" (answered 2026-10-15, 90d4c789: a single pass kept); MEM9's ".giw v16 work" (v17 shipped);
> MEM10's full host copy (gone for dense weights in #95 and for expert stacks in #103); and the M35 figure that
> `decoder/fitguard_giw_pinned_test.go:measuredM35Experts` calls measured, which is §1.1's arithmetic (MEM0 ran no M35 cell).

**Ledger, 2026-10-21.** One row per item, the current state; it supersedes the status notes below where they
disagree, and is updated in the same commit as the work. `python3 scripts/task_docs.py check` reads it.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| MEM0-nobara | done | — | 33ef5a21; `docs/measurements/mem0-2026-10/README.md` |
| MEM0-Mac | done | — (no 7B cell: the coder 7B is not on the Mac, and pulling it and writing its sidecar would take 9 of the disk's 12 GB free) | `docs/measurements/mem0-2026-10/README.md` §4 |
| MEM0-tack | done | — | 5310f789 |
| MEM1 | done | — | branch `mem1-sidecar-2026-10`; MEM1-G0..G5 PASS, `docs/measurements/mem1-2026-10/README.md` |
| MEM2 | done | — | branch `mem2-2026-10`; MEM2-G1..G4 PASS (G1 as amended), `docs/measurements/mem2-2026-10/README.md` |
| MEM3 | done | — | aikit v1.71.0; 58d5dc3e, ba54d206 |
| MEM4(a) | blocked | MEM0's Mac tower cell | aikit's `WrapBF16` has no caller |
| MEM4(b) | open | D11 (a), owner, 2026-10-21: aikit first (head-only loads for the Qwen3.5/Qwen3-VL, GLM-OCR, Qwen2.5-VL and Gemma 4 towers, on an aikit branch; the owner tags), then goinfer's bump and serve's switch | §7 D11 |
| MEM5-VL | open | either box: chunk the prefill of the families G40 left out; L in total | `KVCache.prefillChunkable` |
| MEM5-ring | blocked | G43: fast attention on a sliding-window layer is not chunk-invariant | 90d4c789 |
| MEM6 | open | nobara for the CUDA cells, the Mac for the Metal routes: embed per chunk and return the memory; its gate is registered (ede47b89); S | — |
| MEM7 | open | Mac: size the Metal prefill block to the machine, now that B41's reading is on main; S | `prefillBlockScratch` |
| MEM8(a) | moved | O3, with D5, parked (defer) | `docs/tasks/parked/task-metal-audit-followups-2026-10.md` |
| MEM8(b) | open | D10 (a), owner, 2026-10-21: Gemma 4's per-layer table int4 on Metal by default, a Metal quality cell on E2B, the sidecar variant tag moved, the Metal goldens re-pinned; the untied input embedding stays at the pin | §7 D10 |
| MEM8(b′) | open | either box: stream the logit and PLE tables during the transcode; M | `streamQuantizedEmbed` |
| MEM9(a,b) | blocked | parked; trigger: a Mac that runs an MoE unpaged, or a user report | — |
| MEM9(c) | moved | KV11 in `docs/tasks/task-kv-architecture-2026-10.md` | — |
| MEM10 | open | nobara: a pre-permuted `cuda` target registered from the mapping; M–L | B36 (#103) covered the rest |
| MEM11-bf16 | open | either box: measure LFM2's resident bytes, then bf16 kinds per family (D3); S–M | — |
| MEM11-int8 | open | after the bf16 kinds: int8 per family behind its parity bar (D3, owner 2026-10-21) | — |
| MEM12-scratch | done | — | branch `mem12-scratch-2026-10`; SCR-G1 PASS per buffer (amended 2026-10-22), G3, G4 PASS, G2 PASS as amended, `docs/measurements/mem12-scratch-2026-10/README.md` |
| MEM12-f32res | done | — | branch `mem12-f32res-2026-10`; F32R-G1, G2 PASS, `docs/measurements/mem12-f32res-2026-10/README.md` |
| MEM12-ringres | open | either box: a ring layer's unused append-forever reservation (f32 from `NewKVCache`, f16/int8 from `setQuant`); S | `docs/measurements/mem12-f32res-2026-10/README.md` |
| MEM12-arena | killed | it waited on KV9, which closed 2026-10-21; reopen with KV9 | — |
| MEM13-jobs, MEM13-adapters | done | — | b1402a9f; #94 (cacaa8ec) |
| MEM13-tokenizer | done | — | branch `mem13-tokenizer-2026-10`; TK-G1..G3 PASS, `docs/measurements/mem13-tokenizer-2026-10/README.md` |
| MEM13-features | done | — | sized from the fit plan: up to 256 MiB, less when the plan leaves less after its KV (`featureCacheBudgetFor`); #120 |
| MEM13-heapquant | moved | D8a-build and D8b-build below (D8 decided: build both) | §4 MEM13, "Heap-only quant, after MEM1" |
| D8a-build | done | — | branch `d8a-single-copy-2026-10`; D8A-G1..G4 PASS, `docs/measurements/d8a-2026-10/README.md` |
| D8b-build | done | — | branch `d8b-q4k-giw-2026-10` at v17 (D9); D8B-G1..G4 PASS, `docs/measurements/d8b-2026-10/README.md` |
| D8b-cuda | open | nobara: D8b's CUDA cell, after D8b-build merges | — |
| MEM13-copies-Metal | done | — (removed before the decision was recorded: the int4 lane writes each part into its device buffers) | 1cc27f17 (2026-10-18); `metal/int4_upload_alloc_test.go` |
| MEM13-copies-CUDA | open | nobara: remove CUDA's dense host pack (owner, 2026-10-21, from never-swap); S–M | — |
| MEM14, MEM15, MEM16 | done | — | aikit v1.71.0 |
| MEM17 | open | aikit, then ken: `chunk.Chunk` as a file and a byte range through a mapping; ken stores chunks that way (D4, owner 2026-10-21); an aikit minor bump | `chunk.Chunk.Text` |
| ken | open | ken: `StaticModel.Close` on reload, and `Builder.Add` | — |
| D1–D4 | done | decided 2026-10-21 | the status block above |
| D8 | done | — | owner, 2026-10-21: build both halves (§7) |
| D9 | done | — | owner, 2026-10-21: stay at v17 (§7) |
| D5 | moved | O3 (parked, defer) | — |
| D6, D7 | done | — | aikit 5e4d3c5, a15bfa9 |


> **Status 2026-10-17 (late): the aikit side SHIPPED in aikit v1.71.0 and `gpu/annmetal/v0.2.1`** — MEM3, MEM14, MEM15 and
> MEM16 done, D6 decided (§5's status blocks carry the commits and gate results). Still open on the aikit side: MEM17 (D4)
> and MEM4(a) (after MEM0). **goinfer, 2026-10-18: the aikit bump and MEM3's `Close` follow-ups are on `main`** (`3763d83e` aikit
> v1.70.2 → v1.71.0 in all five modules; `58d5dc3e` the whisper decoder and `Transcriber` gain a `Close`; `ba54d206` serve's unload
> closes the lazily loaded Voxtral and Qwen3-ASR encoders and Voxtral's attached device; `635d63ee` the parity refresh; `69fc3bfc`
> the Voxtral device-close test), so **MEM3 is closed on both sides**. The rest of MEM0–MEM13 is not started; MEM0's nobara half
> is read (`docs/measurements/mem0-2026-10/README.md`; the Metal cells are the Mac's), and MEM13's job-store and per-adapter token-bytes items are in PR #94.
>
> **Scoped 2026-10-17.** Written on the owner's ask after a Cowork
> conversation the same day ("can you list the large data structures that we need in aikit and goinfer, like weights,
> vocabulary, etc. with a typical percent of memory, so we can see if we're handling them in the most efficient way?
> could we do better memory management? assume the kv work is completed", then "yes to task doc, 'memory campaign'").
>
> - Code read at goinfer `main` `f0d11c70` and aikit `main` `fd12e14` (goinfer pins aikit v1.70.2). Static reading by
>   three parallel passes plus hand checks of the headline claims; the claims marked *checked* were re-read in the tree
>   by the author of this doc.
> - **The KV program is assumed finished** (`task-kv-architecture-2026-10.md`, KV0–KV11, owner's framing). KV appears
>   below only as the reference row that other terms are sized against. KV11's single residency manager is the home for
>   anything here that is a residency policy (MEM9).
> - Every figure is byte arithmetic from the code and the configs unless it names a measurement record. Every gate is a
>   sketch: whoever starts an item registers its gates first (rule, void conditions, stopping rule, cost basis, tier), as
>   `CLAUDE.md` requires.
> - One doc for both repos, on the owner's standing preference for one consolidated task doc. The aikit items
>   (MEM14–MEM17) are worked in aikit and released through its own ritual; goinfer picks them up on the next bump.
>
> **Siblings.** [`../completed/task-never-swap-2026-09.md`](../completed/task-never-swap-2026-09.md) (S0–S6: why the default app path is already
> file-backed; this doc starts where it stopped) · [`task-memory-accounting-2026-09.md`](task-memory-accounting-2026-09.md)
> (one bytes-needed function per backend; every new term here must land in it) ·
> [`task-cpu-prefill-memory-2026-10.md`](task-cpu-prefill-memory-2026-10.md) (G40: the CPU prefill chunking MEM5
> extends) · [`task-kv-architecture-2026-10.md`](task-kv-architecture-2026-10.md) · [`../ideas-weight-memory.md`](../ideas-weight-memory.md)
> (the June weight-memory menu; its shelved items are listed in §6 so nobody re-proposes them) ·
> [`task-int4-layout-2026-09.md`](task-int4-layout-2026-09.md) (the up-front representation policy).

## 0. Summary

**What is already right.** On the app path (`serve`, `chat`, `fit`), a `.gguf` or safetensors source becomes a sidecar
`.giw` by default on darwin and linux, and the weights are one physical copy: zero-copy aliases of a read-only mapping
on the CPU, no-copy MTLBuffers over a `MAP_SHARED` mapping on Metal (S6). After load, a 7B on Metal holds about 0.1 GB
that is not file-backed (`docs/measurements/s6-alias-2026-09-24.md`). The Gemma 4 E-model per-layer embedding table is
gathered on the host from the mapping, so only the rows a prompt uses are resident. MoE experts page on the CPU (pool
mode on darwin) and through slot pools on Metal and CUDA. G40 chunks the plain CPU prefill and returns its memory.

**What is left** falls in three groups:

1. **Paths that miss the sidecar** — the library entry point (`decoder.Load` from another module), unpaged MoE on Metal,
   CUDA's expert cache, and the f32 "parity-first" islands (MEM1, MEM9–MEM11).
2. **Multimodal towers** — f32 heap copies of bf16 checkpoints, a host copy kept beside the GPU copy, and Metal tower
   scratch that is never freed (MEM2–MEM4).
3. **Prefill scratch** — whole-prompt passes for the families G40 did not reach, whole-suffix host embeddings on the
   resident paths, and a fixed 512 MiB Metal block (MEM5–MEM7).

Plus the Metal int8 head pin (MEM8), the CPU heap's GC headroom (MEM12), small heap items (MEM13), the aikit index and
encoder items (MEM14–MEM17), and one correctness item in aikit's audio loaders (MEM3).

| item | what | repo / backend | size | bytes at stake (one example) |
|---|---|---|---|---|
| MEM0 | footprint census of the app path and of tack's load path | both | S | — (it sizes the rest) |
| MEM1 | a public mapped-load entry point for library callers | goinfer, all | S–M | 7B on Metal: ~9 GB anonymous → ~0.1 GB |
| MEM2 | Metal tower scratch: release on grow, a real `Close` | goinfer, Metal | S | up to ~2.8 GiB stranded after a 4K image |
| MEM3 | audio loaders: the mappings leaked to process exit; now held and closeable (**shipped, aikit v1.71.0**; goinfer `Close` follow-up open) | aikit + goinfer | S | each audio checkpoint's mapping |
| MEM4 | towers: alias bf16/f32, drop the host copy after GPU attach | aikit + goinfer | M | Qwen2.5-VL on Metal: ~5.2 GB → ~1.35–2.7 GB |
| MEM5 | chunk the prefill of the families G40 left whole | goinfer, CPU | M–L | 2–4.5 GiB at a 20K prompt |
| MEM6 | resident prefill embeds per chunk, not per suffix | goinfer, CUDA/Metal | S | E4B at 20K: ~1.0 GiB |
| MEM7 | Metal prefill block sized to the machine | goinfer, Metal | S | 512 MiB fixed = ~21% of a 1.5B session |
| MEM8 | Metal int4 LM head; gather-only tables decoupled from the head pin | goinfer, Metal | M | tied tables −44%; E2B PLE file 2.35 → 1.32 GB |
| MEM9 | MoE on Metal: stack-major aliasable experts; writer gaps; pool → KV11 | goinfer, Metal | M | M26 unpaged: 12.85 GB anonymous |
| MEM10 | CUDA expert cache registered from the mapping | goinfer, CUDA | M–L | M26 12.85 GB / M35 18.1 GB pinned host |
| MEM11 | f32 islands: LFM2 conv, MLA, drafters | goinfer, all | S–M each | LFM2: over half of resident weight bytes |
| MEM12 | CPU heap: GC headroom and per-request churn | goinfer, CPU | M | up to ~1× the live heap (KV-dominated) |
| MEM13 | small heap items (tokenizer, sampler, job store, feature cache, …) | goinfer | S each | ~15–60 MB each |
| MEM14 | BM25: incremental build and one postings arena (**shipped, v1.71.0**) | aikit | M | index −20%; build 4.4× the index (was ~4.7× plus the corpus) |
| MEM15 | HNSW graph as CSR, mapped (**shipped, v1.71.0**, blob v5) | aikit | S–M | mapped load 6.9 → 0.001 MiB heap |
| MEM16 | encoder tables (BERT word embeddings, Q8 `WordEmb`, Model2Vec, SPLADE, annmetal) (**shipped, v1.71.0 / annmetal v0.2.1**) | aikit | S each | see §5 |
| MEM17 | chunk text by reference, not by copy | aikit API + ken | S (owner decision) | 1M chunks: ~1.4 GB (39% of the index side) |

## 1. Where the memory is (the grounding)

### 1.1 goinfer, one conversation, app path, M1 Pro 16 GB

Metal, int4, f16 KV at 32K context, in the middle of a 20K-token prompt. Arithmetic from the configs and the code; the
weight totals agree with the measured `.giw` sizes (1.29 / 5.18 GB, `docs/measurements/cpu-giw-vs-direct-2026-09-24.md`,
CPU target) and with `docs/measurements/memory-accounting-metal-2026-09-25.md` (5.170 GB for the 7B metal `.giw`).

| structure | how it is held | Qwen2.5-Coder-7B | Qwen2.5-Coder-1.5B |
|---|---|---|---|
| MLP projections | int4, group 32, f16 scales (0.5625 B/weight), aliased | 3.19 GB · 47% | 0.65 GB · 25% |
| attention projections | same | 0.48 GB · 7% | 0.09 GB · 3% |
| LM head | int8 on Metal (the pin, MEM8); int4 on CPU/CUDA since 2026-09-28 | 0.55 GB · 8% | 0.23 GB · 9% (tied: the same table) |
| input embedding | gathered on the host from the mapping; touched rows only | ~0.02 GB · <1% | (tied) |
| KV — reference row, the KV program's | f16 | 1.88 GB · 28% | 0.94 GB · 37% |
| Metal prefill scratch | one 512 MiB block per pass (`metal/backend.go:prefillBlockScratch`) | 0.54 GB · 8% | 0.54 GB · 21% |
| tokenizer, sampler, logits, runtime | heap | ~0.1 GB · 1.5% | ~0.1 GB · 4% |
| **total** | | **~6.8 GB** | **~2.6 GB** |

Other models, weights only (`.giw` payload; share of weight bytes):

| model | the biggest structures |
|---|---|
| Gemma 4 E2B | per-layer embedding table 2.35 GB int8 on Metal (62%) / 1.32 GB int4 elsewhere (51%) — gather-only, so its *resident* share is small; MLP 23–34% |
| Gemma 4 26B-A4B (M26) | routed experts 12.85 GB (88–90%), 3.35 MB per expert, one slot across 30 layers ≈ 100 MB |
| Qwen3.6-35B-A3B (M35) | routed experts 18.12 GB (90–92%), of which int4 group scales 2.0 GB |

Other runtime terms that matter when they are in play: vision and audio towers (0.35–2.8 GB f32 each; Qwen2.5-VL on Metal
≈ 5.2 GB host + device), DFlash/DSpark drafters (f32 from bf16; 2.1 GB for the 4B pairing), and on the CPU the Go
heap's GC headroom over a KV-dominated live heap (no `GOGC`/`GOMEMLIMIT` is set).

### 1.2 aikit, a code-search index (what ken holds)

Corpus shapes: the aikit tree itself as the small case (1,905 chunks in the M1 Pro bench corpus, Model2Vec 256-d); the
BM25 rates — 111 postings and 256 tokens per chunk, 1.32× append slack — come from a Python emulation of the BM25
tokenizer over aikit's tree, not a Go measurement.

| | aikit tree, Model2Vec, FlatI8 | 100k chunks × 768-d, CodeRankEmbed Q8 | 1M chunks × 768-d |
|---|---|---|---|
| embedding model | 64 MB · 87% | 209 MB · 36% | 209 MB · 6% |
| tokenizer + memo | ~4 MB · 5% | ~20 MB · 3% | ~20 MB · <1% |
| vectors (`ann/flat_i8.go:FlatI8`) | 0.5 MB · 1% | 77 MB · 13% | 772 MB · 21% |
| BM25 (`bm25/index.go:Index`) | 3 MB · 4% | ~134 MB · 23% | ~1.2 GB · 33% |
| chunk text (held by ken) | 2 MB · 3% | ~139 MB · 24% | ~1.39 GB · 39% |
| **total** | **~74 MB** | **~580 MB** | **~3.6 GB** |
| build-time transient | <15 MB | tokens ~560 MB + f32 vectors 310 MB | tokens ~5.6 GB + f32 vectors 3.1 GB |

## 2. What must not change

1. **Numerics.** At the same precision, every item here is bit-identical to the tree before it (aliasing, chunking at
   invariant boundaries, freeing, pooling). An item that changes precision (MEM8's int4 head, MEM11's int8 islands) is
   an opt-in or an owner-decided default with its own quality gate, as `--embed-int4` was.
2. **The never-swap rules.** Gates that read memory read the test process (`VmHWM`/`VmSwap` on Linux, `footprint` on
   macOS), not the system, per `CLAUDE.md`. A budget guard keys on a quantity we compute, never on RSS (the
   inverted-guard lesson in `CLAUDE.md` "Measurement discipline").
3. **One accounting path.** Any term this campaign adds, moves or removes lands in `decoder/residentneed.go` (the
   `ResidentNeedBytes` family) in the same change, so `fit` and the guards keep agreeing.
4. **No new production env reads.** A new choice is a `decoder.Options` field (`docs/env-vars.md`).
5. **Tiers.** Most items here are answerable at tier 0 (allocation counters, bytes reserved, bit-identity on tiny
   fixtures). Footprint on a real model is tier 1 (0.5B or 1.5B, by day, under 10 minutes). Nothing here needs a timed
   night run except where an item changes a kernel or a block size (MEM4's widen, MEM7).

## 3. MEM0 — a footprint census first (S)

**Why first.** §1 is arithmetic. Before any build, one tier-1 run per cell measures what the arithmetic predicts, so each
item is sized by the tree and not by this doc.

- Cells (Mac, Metal and CPU): 1.5B and 7B int4 at a 20K prompt; Gemma 4 E2B at a 20K prompt (the per-layer table's
  touched pages); one image turn on the Qwen2.5-VL or Gemma 4 tower; a library-path load (MEM1's "before").
- Read per cell: `footprint` categories after load, at the end of prefill and 10 s after; Go heap in use; the Metal
  device ledger.
- **tack's load path.** Read how tack loads its models (its repo is not in this doc's reading). If it calls
  `decoder.Load` on a `.gguf`, MEM1 is the first build item; if it goes through the sidecar, MEM1 drops to "library
  users only".
  - **Read 2026-10-18, from nobara's night-queue jobs and serve logs (tack's repo itself not read): through the sidecar.** Tack's
    E0 and E1 jobs run the `goinfer-serve` binary as a subprocess (`-goinfer-serve <binary> -model-path <file>.gguf`), not
    `decoder.Load`, and the serve logs show the `.giw` sidecar being loaded (`swap guard (load): not armed for …int4.e4h.cuda.v17.giw`).
    So MEM1 is **library users only**, not the first build item.

## 4. The goinfer items

### MEM1 · A public mapped-load entry point (S–M)

**Problem (checked).** The sidecar policy lives under `internal/`: `internal/prequant/prequant.go:EnsureCachedGIW`
and `internal/prequant/prequant.go:DefaultToSidecar`, reached only through `internal/modelload/modelload.go`. A caller
in another module that calls `decoder.Load` on a `.gguf` or a safetensors dir gets the weights built into the Go heap.
`docs/use-from-go.md` shows `decoder.Load(path, decoder.Options{})`, which loads f32. Two multipliers sit on that heap:

- `decoder/weightmat.go:wantsCanonicalInt4` returns true for any backend name other than `"cpu"`, including the empty
  default, and `decoder/weightmat.go:wantsRow4Fallback` adds a row4 copy everywhere except Metal, so an `Options{}`
  int4 load on arm64 keeps canonical *and* row4 (`decoder/fitguard.go:quantBytesPerElem` prices it at about twice the
  encoding).
- With `Backend: "metal"` the heap `WeightMat`s stay beside the MTLBuffers — the host-copy term
  `task-memory-accounting-2026-09.md` measured at the whole dense term (2.1 GB on the 1.5B from a direct `.gguf`).

Example: a 7B int4 library load on Metal ≈ 4.5 GB heap + ≈ 4.5 GB in buffers; the app path holds ≈ 0.1 GB.

**Fix shape.** Export the sidecar step: either a `decoder.Options` field that asks `Load` to resolve a fresh sidecar
(transcoding once when absent), or a small public package over `EnsureCachedGIW` that returns the `.giw` path. Update
`docs/use-from-go.md` to lead with it. Owner decision **D1**: API shape, and whether a library `.gguf` load should take
the sidecar by default as the apps do.

**Gates (sketch).** Tier 0: on a tiny fixture through the public entry point, heap in use after load stays under a
stated bound and the weights report mmap-aliased; logits byte-identical to the internal path. Tier 1: 0.5B footprint
after load, library path before vs after.

**Gates, registered 2026-10-21 before the code (MEM1-G).** The shape built: `decoder.Options.DirectLoad` (zero value: the
sidecar), and the sidecar machinery moved from `internal/prequant` into `decoder` (prequant imports decoder, so `Load`
cannot reach it where it was; prequant keeps its API as wrappers). A test binary's zero `Options` keeps the direct load
(`testing.Testing()`, the `swapguard` precedent): the ~300 test files that load a `.gguf` test the direct path today, and
under the default each would transcode and write a `.giw` beside its fixture or real checkpoint.

| gate | tier | instrument | PASS iff | void |
|---|---|---|---|---|
| G0 production default | 0 | a non-test program (`go run`) loads a temp copy of `testdata/glm-tiny.gguf` with `Options{Quant: "int8int8"}` | the sidecar `SidecarPath` names exists beside the copy afterwards, and the program's greedy tokens equal the in-process load of that sidecar | the fixture is absent (a SKIP, counted as a failure here) |
| G1 library default | 0 | in-package, the test-binary default turned off by a test hook; same copy and options | the model is the mapped `.giw` (its source is the sidecar, its mapping non-nil); a second load does not transcode (the sidecar's mtime unchanged); logits byte-identical to `Load(EnsureCachedGIW(...))`, the app path's two steps | — |
| G2 opt-out | 0 | `Options{DirectLoad: true}` on the copy | no file is written beside it and the model is a heap load (no mapping) | — |
| G3 test binaries | 0 | the zero `Options` in a test binary, no hook | no file is written beside the copy | — |
| G4 refusal | 0 | the copy in a directory made read-only | `Load` returns an error naming `DirectLoad`, as the apps' refusal names `-direct-load`, and writes nothing | running as root (the chmod does not bind): SKIP |
| G5 tier 1 | 1, Mac | `docs/measurements/mem0-2026-10/testdata/libload.go` on `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `Quant: "int4"`, one run | Go heap in use after `runtime.GC` + `FreeOSMemory` ≤ 100 MiB (MEM0 §4.2 read 647 MiB on this Mac) | the sidecar was not written (the load went direct) |

> **Read 2026-10-21: all six PASS** (`docs/measurements/mem1-2026-10/README.md`). Each tier-0 gate went red under a
> planted defect. G5: 1 MiB of Go heap in use after the library load of the 0.5B at int4 on the Mac, against MEM0's 647.

G5's cost: one 0.5B transcode and load, about 1–2 min wall; no timed figure, so no interleaving. Stopping rule: one
run each; a FAIL is diagnosed, not re-run. The f32 `Options{}` cell is not run: the int4 cell answers the mechanism,
and an f32 sidecar writes ~2 GB beside the model for no new information.

### MEM2 · Metal tower scratch: free what grows, and close (S)

**Problem (checked).** The Metal tower accelerators reassign their buffers on growth without releasing the old ones —
`metal/gemma4_vision.go:g4vAccel.grow`, `metal/gemma4_audio.go:g4aAccel.grow`, `metal/embeddinggemma2.go:eg2Accel.grow`,
and the grid tower in `metal/grid_vision.go` — and all four `Close` methods return nil
(`metal/gemma4_vision.go:g4vAccel.Close`, `metal/grid_vision.go:gridVAccel.Close`, `metal/gemma4_audio.go:g4aAccel.Close`,
`metal/embeddinggemma2.go:eg2Accel.Close`). Each accelerator also owns its own device ledger, so the resident's
release-all never reaches it. Scratch at about 90 KiB per patch (Qwen3-VL-8B-class tower, f32) makes a 1080p frame
~0.7 GiB and a 4K screenshot ~2.8 GiB, held for the life of the process; a video sums its frames. CUDA's towers
allocate per call and release; Metal's text prefill releases per pass.

**Fix shape.** Release the old buffers in `grow`; a `Close` that frees scratch and uploaded weights; one ledger with the
resident (or `Close` wired into unload); optionally a patch ceiling with per-segment passes.

**Gates.** Tier 0: device bytes live after N grows equal the last capacity; after `Close`, zero (mutation: drop the
release, red). Tier 1: one 4K image turn then unload, Metal footprint returns to the pre-load figure.

**MEM2 gates, registered 2026-10-21 before the code (MEM2-G).** The build: each Metal tower accelerator (the Gemma 4
vision and audio towers, EmbeddingGemma 2's text encoder, and the grid base under the Qwen3.5+, GLM-OCR, SigLIP and
Qwen2.5-VL towers) releases the scratch a `grow` replaces, and the rope tables EmbeddingGemma 2's cache drops; `Close`
waits out the forward in flight (the accelerator's lock), releases every buffer and object on the accelerator's own
device ledger, and is idempotent; a forward after `Close` is an error. The SigLIP and Qwen2.5-VL resident wrappers'
`Close` closes their accelerator (aikit's `Encoder.Close` already calls it). serve's unload (`closeEntryNatives`)
closes the Gemma 4 vision tower, the Qwen3.5+ and GLM-OCR towers' accelerators and the Gemma 4 audio accelerator.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| MEM2-G1 grow | 0, Mac | each of the seven towers on its committed tiny fixture: accelerator A runs three inputs of increasing size, accelerator B the same three largest first, so B grows once; the device ledger's buffer count (`LedgerLen`) and Metal's own allocated size (`CurrentAllocatedSize`, net of the reading before each build) | A's ledger count equals B's exactly, A's allocated size is within 64 KiB of B's, and A's output at every size is byte-equal to B's |
| MEM2-G2 close | 0, Mac | the same seven, after `Close` | the ledger reads (0, 0); Metal's allocated size is back within 64 KiB of its reading before the build; a second `Close` returns nil; a forward after `Close` returns an error |
| MEM2-G3 unload | 0, any | serve's admin unload through the real drain, an entry holding a counting fake Gemma 4 tower, Qwen3.5+ and GLM-OCR accelerators and a Gemma 4 audio accelerator | each fake is closed exactly once and the entry holds none afterwards; an entry that never built one closes nothing |
| MEM2-G4 tier 1 | 1, Mac | the Qwen3-VL-2B tower (`~/models/qwen3-vl-2b-instruct`) on Metal, float32, in-process: a 1080p grid, then a 3840×2160 grid, then `Close`; Metal's allocated size at each step | after the 4K forward the allocated size is the weights plus one 4K scratch (within 1%: the 1080p scratch is gone, against main, where it stays); after `Close` it is back within 16 MiB of the reading before the build |

Planted defects, each run once on the committed tree and reverted: the release dropped from the grid tower's `grow` (G1
red), `Close` left a no-op (G2 red), and one of serve's new closes dropped (G3 red). The existing matches-CPU gates of
every tower (`TestG4VMetal_matchesCPU`, `TestGridVisionMetal_matchesCPU`, `TestS3Towers_tiny`, `TestG4AMetal_matchesCPU`,
`TestEG2Metal_matchesCPU`) stay green. Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for
G1–G3; about 2 min for G4 (a 32,400-patch tower forward; not a timed run, so it holds the timing lock shared).

> **Read 2026-10-21: G2–G4 PASS; G1 FAILED as registered on its byte cross-check and PASSES as amended**
> (`docs/measurements/mem2-2026-10/README.md`). The ledger counts matched on all seven towers; Metal's own reading was
> 256 KiB over on the process's first Gemma 4 vision build (present before any forward, absent on a second build) and
> 128 KiB over on EmbeddingGemma 2 with the same buffers (suballocation). **Amendment, after the reading:** each tower is
> built and closed once first, and G1's byte bar is 256 KiB; G2's stays 64 KiB. G4: Qwen3-VL-2B holds one 4K scratch
> (2622.5 MiB) after 1080p then 4K and returns to +0.3 MiB after `Close`; main strands 665.5 MiB and keeps 3875.4 MiB
> after `Close`. All three plants red.

### MEM3 · aikit audio loaders hold their checkpoint and can close it (S) — SHIPPED in aikit v1.71.0

> **Released 2026-10-17 in aikit v1.71.0. D6 decided the same day: no KeepAlive discipline.** The finalizer, the
> `Tensor.owner` back-pointer and the KeepAlives that existed only for it were deleted (aikit `5e4d3c5`); `Close` is the only
> release, and a pin test reads a dropped file's view after the file is collected (red 8/8 with the old finalizer forced to
> run). Consequence for goinfer: a `SafetensorsFile` dropped without `Close` now keeps its mapping until exit, by contract.
> Checked at goinfer `main`: every other opener closes after copying or keeps its file (main weights, LoRA, DFlash, DSpark,
> embeddinggemma2, the clef head, the multimodal projectors), so **the goinfer follow-up is exactly: close serve's lazily
> loaded `qwenASR` and `voxtral` encoders on unload, and give `internal/whisper`'s decoder and `Transcriber` a `Close`**
> (the whisper decoder keeps views and drops its file).

> **Status 2026-10-17: fixed on aikit branch `mem3-audio-mapping-lifetime` (`db44648`, unpushed), and the finding was
> corrected in the doing.** The use-after-unmap this entry first described is **not reachable today**: each tensor keeps
> a pointer back to its file (the `owner` field of aikit's safetensors `Tensor`), the file holds its tensors, and Go
> never runs a finalizer on an object in a cycle — so a dropped file is never unmapped, and the mappings **leaked until
> process exit** instead. Verified there: a file holding one tensor was still reachable after 10 GCs; one holding none
> was collected. The fix: the three encoders keep their file and gain an idempotent `Close` (a forward after `Close`
> returns an error); Voxtral maps its checkpoint once instead of twice; Qwen ASR uses the shared opener. The
> weak-pointer test cannot fail on today's code (the cycle keeps the file alive) and guards the day the cycle goes away;
> with the cycle broken, each loader's original bug fails it cleanly, 3/3 rounds per loader. The text below is the
> entry as first written, kept for the sweep.
>
> **Open, owner decision (D6):** the finalizer `OpenSafetensorsMmap` installs never runs for a file that holds any
> tensor, so its comment's promise (a guard for callers who forget `Close`) is not kept. Making it real (for example
> with `runtime.AddCleanup`) would turn every caller that drops a file early into a real crash, so it needs an audit of
> the callers first. Not changed.

**Problem, as first written (superseded by the status above).** For an F32 tensor,
`embed/safetensors.go:SafetensorsFile.TensorF32` returns a zero-copy view into the mapping. Three loaders open the
checkpoint mmap'd, keep F32 views, and keep no reference to the file:

| loader | keeps the file? (before `db44648`) |
|---|---|
| `audio/whisper_encoder.go:LoadWhisperEncoder` | no |
| `audio/voxtral.go:LoadVoxtralAudio` (projector w1/w2) | no |
| `audio/qwen3asr_encoder.go:LoadQwenASREncoder` | no |

Checked safe, so the sweep is complete at these three: `audio/gemma4_encoder.go:LoadGemma4AudioEncoder` (closes the
file but copies every tensor out first); all six vision loaders (SigLIP, Qwen2.5-VL, Qwen3-VL, Pixtral, GLM-OCR,
Gemma 4 — each copies out, then closes); aikit `encoder` (keeps the file on its weights); and goinfer's own loaders
(main weights, LoRA, DFlash, DSpark — each keeps its `SafetensorsFile`). BF16 checkpoints are safe in any loader: they
are widened into the heap. goinfer reaches all three bad loaders: serve loads Qwen ASR and Voxtral lazily
(`internal/serveapp/vision_serve.go`) and `internal/whisper/transcribe.go` loads Whisper. Not reproduced.

**Fix shape.** Retain the `SafetensorsFile` in each of the three encoders, with a `Close`. On the goinfer side, at the
next aikit bump: close serve's lazily loaded Qwen ASR and Voxtral encoders on model unload, and close the Whisper
encoder after a transcription, so the mappings do not live to process exit.

**Gates.** Tier 0. A read of unmapped memory is a SIGSEGV that kills the test binary rather than failing a test, so
either run the encoder under `debug.SetPanicOnFault(true)` (the fault becomes a recoverable panic), or — preferred,
crash-free and deterministic — a test hook records the file the loader opened, the test takes a `weak.Pointer` to it,
runs several `runtime.GC()` cycles with the encoder still alive, and asserts the pointer is not nil (red before the fix,
since nothing holds the file). Confirm Go's weak-pointer semantics for an object with a finalizer before relying on the
second form.

### MEM4 · Towers: alias instead of widening, drop the host copy after GPU attach (M)

**Problem.** The vision and audio towers (aikit `vision/`, `audio/`) load f32 into the heap: bf16 checkpoints are
widened 2× (`vision/qmat.go:newQMat` copies; f32 checkpoints are copied too), and a GPU-resident tower keeps its host
blocks (`vision/qwen_resident.go:QwenVisionEncoder.EnableResident` sets a flag; goinfer's Metal and CUDA tower builders
read the host weights and keep the encoder). Only SigLIP has a head-only load with streamed blocks. Examples: Qwen2.5-VL
ViT ≈ 676M params = 2.70 GB host f32 + ≈ 2.52 GB device on Metal; Voxtral's audio tower ≈ 2.55 GB heap (1.27 GB as
stored). goinfer's default `-vision-quant` is f32.

**Fix shape.** Two independent halves:

- (a) Alias. aikit already has zero-copy bf16/f16 `WeightMat` wrappers over a mapping (`WrapBF16`/`WrapF16`, bit-identical
  to f32 `MatmulBT`, no caller yet); route `newQMat` through them, and alias f32 checkpoints. Caveat: the arm64 widen is
  scalar per tile, so this is a speed question on the Mac (in-process A/B by day, TE5(b)).
- (b) After GPU attach, release the host blocks; the CPU fallback reloads them lazily from the mapping. Generalize
  SigLIP's head-plus-streamed-blocks shape to the other towers. Owner decision **D2**: is a lazy reload on CPU fallback
  acceptable latency?

**Gates.** Tier 0: logits/features bit-identical aliased vs widened on each tower's tiny fixture; heap after attach
under a stated bound. Tier 1: one image turn on the Mac, footprint before/after. Speed: in-process A/B for (a) on arm64.

**MEM4(b) gates, registered 2026-10-22 before the code (TWR-G), on D11 (a).** The aikit half: the Qwen3.5/Qwen3-VL
(`LoadQwen3VisionEncoder`, `LoadQwen3VisionEncoderMMProj`), GLM-OCR, Qwen2.5-VL and Gemma 4 towers gain `ReleaseBlocks`,
which drops their encoder blocks (the bulk of a tower) from the host, and `BlocksReleased`; the next host forward or weight
export (`Weights`, `GPUWeights`) reloads them by running the same loader again, so the values are the same. A tower built from a
caller's `TensorSource` (`LoadQwen3VisionEncoderFrom`) cannot reopen it and refuses, keeping its blocks. The goinfer half,
after the owner tags aikit: serve calls `ReleaseBlocks` once a device tower is built, and the CPU fallback reloads.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| TWR-G1 release (aikit) | 0 | each tower on its tiny fixture (aikit's `testdata/qwen35vl-vision-tiny`, `qwen25vl-vision-tiny`, `glm-ocr-vision-tiny`; Gemma 4's `gemma4-vision-tiny` where present), float32: a CPU forward and the weight export, then `ReleaseBlocks`, then both again; the Go heap after `runtime.GC` before and after the release | `BlocksReleased` reads true after the release and false after the next use; the heap falls by ≥ 80% of the blocks' bytes (summed from the export); the forward after the release is byte-equal to the one before; the export after it equals the one before; a `LoadQwen3VisionEncoderFrom` tower's `ReleaseBlocks` returns an error and its forward is unchanged |
| TWR-G2 serve (goinfer, after the tag) | 0 | serve's device-tower attach for the four towers, and `deviceFallback`'s CPU fallback after a device memory error, with a fake accelerator | each attach releases the blocks once; the fallback's features are byte-equal to the CPU tower's on a never-released encoder |
| TWR-G3 tier 1 (goinfer, after the tag) | 1, Mac | the Qwen3-VL-2B tower (`~/models/qwen3-vl-2b-instruct`) attached on Metal, Go heap in use after `runtime.GC`, branch against main | the heap falls by ≥ 80% of the blocks' float32 bytes |

Planted defect, run once in aikit and reverted: `ensureBlocks` not reloading (TWR-G1 red: the forward after the release
fails or differs). Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for G1 and G2; about a
minute for G3. aikit's `preflight` runs before its PR.

### MEM5 · Chunk the prefill of the families G40 left whole (M–L)

**Problem (checked).** `decoder/kvcache.go:KVCache.prefillChunkable` is false for a cache with image blocks, m-RoPE
positions, DeepStack, captures, tree verify, or `manualPos` — and `manualPos` is set for Gemma 4, the Qwen3.5 hybrids,
LFM2, Granite-H, Nemotron-H, MLA and KDA (`decoder/model.go`). The fast-attention ring case also keeps one pass. Those
prompts run `runLayersFromEmbedN` over every row: at about 108 KiB of activations per row on the 1.5B shape and
236 KiB on the 7B, a 20K prompt is 2.1–4.5 GiB — the pre-G40 shape (`task-cpu-prefill-memory-2026-10.md` §2), on the
newest families and on every image turn.

**Fix shape.** Per family, not at once:

- recurrent / `manualPos`: the state already carries across calls in decode; chunking needs a per-family
  chunk-invariance gate;
- ring + fast attention: blocked on G40-1's invariance question, or chunk with exact attention;
- VL: chunk boundaries outside image blocks.

**Gates.** Tier 0 per family: chunked vs whole logits byte-identical on the tiny fixture at two chunk sizes, one
boundary inside each layer kind. Tier 1: one family's 20K footprint before/after (the G40 harness).

### MEM6 · Resident prefill embeds per chunk (S)

**Problem.** On the resident paths that do not block the prompt (CUDA's chunked prefill, Metal's layer-major and E-model
routes), the host embeds the whole suffix up front, one allocation per row
(`decoder/prefill_chunk_scratch.go` says the path "does allocate in full"). Gemma 4 E4B's resident embed row is
hidden + layers × 256 floats ≈ 52 KiB → ~1.0 GiB at 20K; CUDA dense 117–273 MiB at 20K.

**Fix shape.** Embed per chunk into one reused slab (a callback, or a cut at the chunk size).

**Gates.** Tier 0: bytes allocated by a 2,048-row prefill on the tiny fixture fall from O(prompt) to O(chunk); logits
byte-identical.

**Added 2026-10-18 (owner; MEM0's nobara half):** the host embeddings of a 20K prompt were measured **held after the request**
(+218 MiB on the 1.5B, +476 MiB on the 7B, still there 10 s later: `docs/measurements/mem0-2026-10/README.md`). The fix must embed per chunk
**and return the memory afterwards**; its gate adds: `RssAnon` 10 s after a 20K cold prompt is within 64 MiB of its value after load plus the
session's retained state, on the 1.5B and the 7B (`mem0_linux.py`'s cells). Registered with the CUDA streaming change
(`docs/measurements/cuda-stream-pack-2026-10-18/README.md`).

### MEM7 · Size the Metal prefill block to the machine (S)

**Problem.** `metal/backend.go:prefillBlockScratch` is a fixed 512 MiB per pass, allocated and released each pass. On a
1.5B 32K session that is about a fifth of the process; on a 7B it is 8%. It does not consult the KV plan's free-memory
reading.

**Fix shape.** Derive block rows from the same free-memory probe the fit plan uses, with the current 512 MiB as a
ceiling; count the block in `ResidentNeedBytes`. Open question: does a persistent arena beat allocate-and-zero per pass
on a 16 GB machine?

**Gates.** Tier 0: logits byte-identical across block sizes (blocking is already invariant; pin it at two sizes).
Speed: TTFT at the smaller block, in-process A/B by day (TE5(b)); a served gate only if unresolved.

### MEM8 · Metal int4 LM head; gather-only tables decoupled from the head pin (M)

**Problem (checked).** `internal/loadflags/loadflags.go:Flags.embedInt4` returns false on Metal unless the flag is set
explicitly, because Metal has no int4 head GEMV — an int4 table made a plain Metal load decline to the CPU. So tied
tables stay int8 on Metal (1.5B 234 vs 131 MB; M26 739 vs 415 MB), and every table that shares the policy follows:
Gemma 4 E2B's per-layer embedding table is 2.35 GB int8 against 1.32 GB int4. That table is only gathered on the host
(`decoder/residency.go:Model.embedResidentInto`), so its resident cost is small either way; its full size shows up on
disk and once at transcode (MEM8b).

**Fix shape.**

- (a) A Metal int4 head GEMV (W4 over f16 activations, the existing int4 projection kernel's shape at N = vocab), then
  let `embedInt4` follow the CPU/CUDA default on Metal.
- (b) Independently: key gather-only tables (per-layer embedding, an untied input embedding) on their own precision
  instead of the head pin.
- (b′) The transcode builds logit tables whole in heap before the layers stream (E2B's per-layer table: 2.35 GB once);
  stream them in row blocks.

**Gates.** (a) Tier 0: kernel bit-identity against the CPU int4 head reference; quality is the existing embed-int4 bar
(it is already the CPU/CUDA default), plus `ideas-weight-memory.md` §3's rare-token caution. (b) tier 0: gathered rows
identical to a dequantized reference at each precision. (b′) tier 1: transcode peak heap on E2B.

**MEM8(b) gates, registered 2026-10-21 before the code (PLE-G), on D10 (a).** The build: at quant int4, Gemma 4's
per-layer token table is int4 whatever `EmbedInt4` says (both loaders: `decoder/gguf.go`'s `per_layer_token_embd` and the
safetensors one in `decoder/weights.go`); the token embedding and the LM head keep the pin. At any other quant nothing
changes. Every load that had the pin on at int4 moves: Metal's default, and also a library `decoder.Options{}` load and an
explicit `--embed-int4=false` on CPU or CUDA. The sidecar of a source with a per-layer table, at int4 with the pin on, is
named with a `.ple4` segment (`<base>.int4.ple4.<target>.v17.giw`), so a sidecar written before this change (int8 table)
is never opened for it, and no legacy name is adopted for it; every other sidecar name is unchanged. `metal/` is not
touched (the table is gathered on the host).

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| PLE-G1 table | 0 | the synthetic E2B-shaped GGUF (`tinyGemma4GGUF`, `ple`) and `testdata/gemma4-emodel-tiny` (safetensors; generated, so CI skips it), int4, `EmbedInt4` false and true | the per-layer table is int4 under both and byte-equal between them, and every gathered row equals the int4 dequantization of the source row; the token embedding is int8 under false and int4 under true, as before |
| PLE-G2 sidecar | 0 | the sidecar path for those sources and for `llama-tiny`, at int4 and int8, pin on and off; a transcode of the synthetic GGUF at the `.ple4` name and its load | `.ple4` appears exactly for a per-layer source at int4 with the pin on; every other name equals main's; the transcoded bundle's table is int4 and its logits over 8 tokens are byte-equal to the direct load's |
| PLE-G3 quality | 1, Mac, Metal | E2B (`~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf`): main's sidecar (int8 table, `.int4.metal.v17.giw`) against this branch's (`.int4.ple4.metal.v17.giw`), resident, one process, one arm at a time; the five G3 prompts, each free-run 32 tokens greedy on the int8 arm, then that sequence teacher-forced through the int4 arm | top-1 agreement over every position ≥ 95.0% and the mean KL(int8 ‖ int4) ≤ 0.05 nats: PASS. Agreement 90.0–95.0% or KL 0.05–0.10: AMBIGUOUS, parked for the owner. Worse: FAIL. Also read, not gated: the two sidecars' sizes (the arithmetic: the table falls from about 2.35 to 1.32 GB) |
| PLE-G4 goldens | 0–1 | the tests that pin Gemma 4 E-model outputs at int4 (`decoder` and `metal`, tiny and real E2B), then `scripts/refresh_parity_hashes.sh` | each either passes unchanged or moves only through the table's precision and is re-pinned in the same commit, with the moved figure in the record; the parity refresh is green |

Planted defect, run once on the committed tree and reverted: the per-layer table following the pin again (G1 red, and G2's
name check red on the sidecar's contents). Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds
for G1, G2 and G4's tiny tests; G3 about 10 min plus a one-time transcode of E2B (about 2.8 GB written; 6.8 GB free at
registration), not a timed run, so it holds the timing lock shared.

### MEM9 · MoE on Metal: aliasable experts, the two writer gaps, the pool to KV11 (M)

**Problem.**

- Unpaged MoE copies every expert into stacked anonymous buffers (`metal/moe.go:buildMoELayer`): concatenation cannot
  alias because the writer emits per-expert records. M26 unpaged = 12.85 GB anonymous. On the 16 GB Mac M26 always
  pages, so this matters on larger Macs.
- The shared-expert gate|up and qwen35's K‖V are still copied (~50 MB on M35; S6's "two writer gaps").
- The paged pool is per layer with a fixed N, unlike the CPU's single global pool (`decoder/moepool.go:expertBufferPool`);
  it cannot follow skew across layers.

**Fix shape.** A stack-major expert layout under `-target metal`, the fused-group trick applied to experts; close the two
writer gaps in the same format bump (folds into the planned `.giw` v16 work); the per-layer pool goes into KV11's
residency manager rather than its own rewrite.

**Gates.** Tier 0: byte-identical logits aliased vs copied on the MoE tiny fixtures; device ledger shows no expert
copy. Tier 1: footprint on the smallest MoE that runs unpaged on the Mac.

### MEM10 · CUDA expert cache from the mapping (M–L, nobara)

**Problem.** With `--moe-cache-experts`, the whole expert stack is packed into a fresh heap copy and registered as
pinned host memory (M26 12.85 GB, M35 18.1 GB page-locked), and the build repacks each tensor through a full host
copy (`cuda/resident.go:packWeight`) with the scales round-tripped f16 → f32 → f16. The `"cuda"` `.giw` target exists
in the cache key but still writes the canonical kind.

**Fix shape.** A pre-permuted cuda target, so registration and upload read the mapping directly; the scales stay f16 end
to end.

**Gates.** Tier 0 (GPU, tiny fixtures): uploaded bytes identical to today's packing; tier 1 on nobara: host footprint
of the 26B with the expert cache before/after (`VmHWM`, the CUDA box rules in `CLAUDE.md`).

### MEM11 · f32 islands (S–M each)

**Problem.**

- **LFM2 short-conv.** `decoder/lfm2.go:shortConvWeights` holds `inProj` [3·convDim × hidden] and `outProj` at f32,
  with convDim = hidden. That is 4·hidden² f32 per conv layer, which by arithmetic is over half of an LFM2 model's
  resident weight bytes at int4 — the loader comment's "~0.6% of the model's bytes" counts the depthwise kernel, not the
  projections. Measure with `ResidentWeightBytes` on a real checkpoint before acting.
- **MLA.** `decoder/weights.go:mlaWeights` is all f32 and copied on `.giw` load (DeepSeek-V2-Lite ≈ 1.49 GB, ~15% of an
  int4 load); Metal builds int8 copies on the GPU (`metal/mla.go:int8Proj`) while the host keeps the f32.
- **Drafters.** `decoder/dflash.go:LoadDFlashDrafter` widens bf16 to f32 heap (2.1 GB for the 4B pairing); CUDA keeps
  the host f32 beside its int8 device copy.

**Fix shape.** For bf16 sources, the bf16 `WeightMat` kind (half the bytes, bit-identical to the widened f32); int8
where a family's parity bar allows, as an owner decision per family (**D3**); release host copies after upload.

**Gates.** Tier 0 per family: bf16 kind vs widened f32 bit-identical; int8 only behind the family's existing parity
gate.

### MEM12 · The CPU heap: GC headroom and per-request churn (M)

**Problem.** The CPU KV lives on the Go heap (f32 by default), and no `GOGC` or `GOMEMLIMIT` is set, so the heap may
grow to about twice a KV-dominated live heap (1.1 GiB of KV at 20K on the 1.5B, 2.3 GiB on the 7B). G40 measured GOGC=10
cutting a 6K footprint from 1,039 to 684 MiB but fixed only the prefill, with targeted returns. Churn that keeps the
pacer busy: the MoE expert-major branch allocates a K × hidden output per layer per pass (22 MiB/layer per 2,048 rows,
215 MiB/layer on the unchunked 26B at 20K), the LM head allocates M × vocab per verify round and per prefill
(4.6–8 MiB / 0.6–1 MiB), the sampler builds six vocab-sized buffers per request (7–12 MiB), and tokenization allocates
~13 MB per 24K-token prompt.

**Fix shape.** Measure first (MEM0): live heap vs footprint at 20K on the CPU. Then, in order: hold the MoE output in
`decoder/forwardn.go:prefillPassScratch`, a cache-owned K × vocab logits buffer, pooled sampler scratch; only then
consider off-heap (anonymous-mmap) arenas for KV and session scratch, or a pacing rule. `GOMEMLIMIT` is refuted (§6).

**Gates.** Tier 0: `testing.AllocsPerRun` / bytes allocated per decode token and per verify round fall to the stated
bound. Tier 1: the 1.5B 20K footprint vs live heap, before/after.

**MEM12-f32res gates, registered 2026-10-21 before the code (F32R-G).** The build: a CPU cache that stores K/V as f16 or
int8 reserves no f32 capacity. `NewCache` passes no f32 hint to `NewKVCache` when it will call `setQuant`, and `setQuant`
replaces any f32 capacity the cache holds with an empty, non-nil slice (the snapshot restore keeps an empty slice non-nil),
so its own callers cannot leave one either. CPU KV pricing (`decoder/residentneed.go:Model.ResidentKVBytes`) already
counts only the quantized bytes, so nothing there moves. Tier 0 answers it: the term is a reservation, so its size is a
counter, and its residency (an untouched `make` on fresh pages may never be resident) is not the question; its weight in
the Go heap's accounting, which paces the collector, is.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| F32R-G1 bytes | 0 | `llama-tiny` and `gemma2-tiny` (its global layers append, its local ones ring), each with f16 and with int8 KV where the family allows it, and f32: `NewCache(512)`; every layer's `cap` of the f32 and quantized K/V slices; the bytes `NewCache(512)` allocates over `NewCache(0)` (`runtime.MemStats.TotalAlloc`, GC off) | under f16 and int8 every f32 K/V capacity is 0 and the quantized ones are 512 × the layer's KV width, as before; under f32 the f32 capacities are unchanged; and the bytes over `NewCache(0)` are within 1% of the quantized reservation alone (on main they also carry the f32 reservation) |
| F32R-G2 identity | 0 | the f16 and int8 KV tests (`TestKVF16_*`, `TestKVI8_*`), the prefix- and session-share tests, then `scripts/refresh_parity_hashes.sh` | all PASS (not skip); the parity refresh is green |

Planted defect, run once on the committed tree and reverted: `setQuant` keeping the f32 capacity with `NewCache` still
passing the hint (G1 red). Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for G1, about 4
min for the parity refresh.

> **Read 2026-10-21: both PASS** (`docs/measurements/mem12-f32res-2026-10/README.md`). The hint costs the quantized
> reservation to the byte; on the tree before, an f16 cache's hint cost three times its own storage. G2's three heavy
> int8 tests could not load `gemma-3-4b-it` past the fit guard by evening; on the 0.5B they read the same on the branch
> and on main. Found on the way: ring layers carry an unused append-forever reservation (MEM12-ringres).

**MEM12-scratch gates, registered 2026-10-21 before the code (SCR-G).** The build, three parts. (1) The MoE expert-major
branch of the batched pass takes its K × hidden output, its done flags and `moeMLPBatch`'s buffers (router logits, the
per-rank outputs, the gather, gate, up and down buffers) from `prefillPassScratch`, so a chunked prefill allocates them once;
callers without a pass scratch allocate as before. (2) The LM head writes into a buffer the cache owns: a verify round's
K × vocab into one the cache grows on demand, prefill's last row into the cache's decode logits buffer, which is what the
sequential fallback already returns (valid until the next forward on that cache; every consumer reads it before then:
`generateInto`, the three speculative loops, the VL prefill paths). Callers that keep rows (`PromptLogitsMany`, the
batched CPU decode, audio) keep their own. (3) The sampler's six vocabulary-sized buffers come from a pool and go back when
the generation ends (`generateInto`, the n-gram and grammar speculative loops); a sampler never released allocates as before.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| SCR-G1 MoE | 0 | `mixtral-tiny` (4 layers, hidden 64, 8 experts, top 2), expert-major on, `prefillChunkRows` 16, a 64-token prompt: bytes allocated (`TotalAlloc`, GC off) between the first chunk's end and the last's, branch against `origin/main` | the branch allocates at least 90% of the arithmetic MoE buffers less than main: 3 chunks × 4 layers × 4 B × (16·64 output + 16·8 router logits + 16·2·64 per-rank) = 153,600 bytes, so a fall of ≥ 138,240 |
| SCR-G2 logits | 0 | `llama-tiny` (vocab 256): a warm verify round (`forwardN`, K = 4) and a 16-token `prefillLogits`, bytes allocated, branch against main | the round falls by ≥ 0.99 × 4 × 256 × 4 B and the prefill by ≥ 0.99 × 256 × 4 B |
| SCR-G3 sampler | 0 | `internal/sampling`, vocabulary 151,936, ten requests each `NewSampler`, `Observe` of 64 ids, 8 draws with top-k, top-p, min-p and a repetition penalty, then the release; GC off | requests 2–10 each allocate ≤ 5% of request 1's bytes (main: each the same as request 1) |
| SCR-G4 identity | 0 | on `mixtral-tiny` with expert-major on and on `llama-tiny`: the chunked batched prefill's last logits against the sequential decode's, byte-equal; the speculative tests (`TestSpec*`, `TestGenerateNgram*`), the sampler package's tests, then `scripts/refresh_parity_hashes.sh` | all byte-equal and PASS (not skip); the goldens green |

Planted defects, each run once on the committed tree and reverted: the MoE branch allocating its output again (G1 red), the
LM head allocating again (G2 red), the release not returning the buffers (G3 red). Stopping rule: one pass each; a FAIL is
diagnosed, not re-run. Cost: seconds for G1–G3, a few minutes for G4's test sets and about 4 for the parity refresh.

> **SCR-G1 amended 2026-10-22, before its re-read (owner, 2026-10-21: a per-buffer bar, so a regression in one term
> cannot hide inside the sum).** The bar is now per buffer: each of the expert-major path's eight buffers (the output, the
> done flags, the router logits, the per-rank outputs, and the gather, gate, up and down buffers) counts its own fresh
> allocations (`moeScratchGrows`), and in the same `mixtral-tiny` prefill (64 tokens, chunks of 16, expert-major on) **each
> is allocated exactly once**: at least once (it is in use) and never again in a later chunk or layer. For that to be a
> property of the code rather than of the routing, the gather, gate, up and down buffers are sized by the chunk's rows (the
> bound on any one expert's rows) instead of the largest expert's rows in that chunk, which can grow from chunk to chunk.
> The `TotalAlloc` reading against main stays as context, not a bar. Plants, each run once and reverted: each buffer in
> turn allocating fresh on every call (its own row red, the others green), including the output-only plant that left the
> summed bar green.

> **Read 2026-10-21: G1, G3 and G4 PASS; G2 FAILED as registered and PASSES as amended**
> (`docs/measurements/mem12-scratch-2026-10/README.md`). G2's first reading found a defect of the build (the MoE scratch
> embedded by value cost every pass about 200 bytes; now started by the first MoE layer) and an instrument that could not
> resolve 1 KB (main moved by 1.4 KB between runs; per-P pool caches). **Amendment, after the reading:** G2 reads the mean of
> 50 warm iterations at `GOMAXPROCS=1` on both arms, bars unchanged: the round falls 4,096 bytes and the prefill 1,024,
> exactly the logits. G1: 410,384 → 94,592 bytes. G3: later requests ≤ 1.9% of the first's 5.6 MB. The output-only MoE
> plant left G1 green (its bar sums the buffers); the whole-path plant turned it red.

### MEM13 · Small heap items (S each)

| item | where | bytes | fix |
|---|---|---|---|
| tokenizer kept size and load spike | `tokenizer/sentencepiece.go:Tokenizer`; aikit's GGUF metadata reader boxes every string | 25.8 MB kept, 92.9 MB allocated per load (measured on a Go 1.24 copy) | merge ranks keyed by `(leftID, rightID)` as one uint64 (10.2 → 4.7 MB), a size hint for the rank map, a typed metadata reader (~half the spike); ~15–18 MB per model |
| job store keeps prompt ids with no journal | `internal/serveapp/job.go:jobStore` | ~41 MB at its cap with 20K prompts | `[]int32`, or keep them only with `-job-dir` |
| feature cache not priced | `internal/serveapp/feature_cache.go:featureCacheBudget` | 256 MiB per vision model | count it in the fit plan, or size it from free memory |
| token bytes per adapter | `internal/serveapp/openai.go:loadedModel` (`tokenBytes`) | 5.4 MB per adapter | share with the base, as the tokenizer is |
| heap-only quant | q4k (phi3's default on CPU/CUDA) has no `.giw` form; on the arm64 heap path o_proj, down_proj and experts keep the dual copy even with `Backend: "cpu"` | model-sized | after MEM1, decide whether the heap path still matters |

**D8b-build gates, registered 2026-10-21 before the code (D8B-G).** The build: a `.giw` tensor kind 7 for a native Q4_K
matrix (the header, then the raw super-blocks, aligned and aliased from the mapping through `linalg.WrapQ4K`), the
weights format version 17 → 18 (every sidecar of an older version is rebuilt on its next load, as the version rule
already does), the writer no longer refusing q4k, a q4k `.gguf` routed to its sidecar by `decoder.Load`'s default and
by `internal/modelload` (q4k leaves the direct-only list for a `.gguf`; a safetensors directory at q4k stays direct), and
a q4k bundle's per-32 activation group stamped from its baked quant label when `Options` does not set it.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D8B-G1 round trip | 0 | a one-layer llama `.gguf` written by the test with Q4_K attention and MLP tensors (`q4kTestMat`'s blocks; the table f32), loaded at q4k directly and through `TranscodeGIW(q4k)` then `Load` of the bundle, `Backend: "cpu"` | the bundle's Q4_K tensors are kind 7, aliased from the mapping (no heap copy of their bytes); the logits of an 8-token prompt token by token and 8 greedy tokens are byte-equal between the two loads |
| D8B-G2 routing | 0 | `decoder.Load` (default on, the test hook) and `modelload.Plan`/`Load` on the same q4k `.gguf` | both open the q4k sidecar; a safetensors directory at q4k still loads directly |
| D8B-G3 version | 0 | the existing version tests (`TestEnsureCachedGIW_olderFormatIsRebuiltOnce` and its siblings) at version 18, and a kind-7 record read by a reader of the previous version's rules | green, and a v17 file is rebuilt once |
| D8B-G4 tier 1 | 1, Mac | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` at `Quant: "q4k"`, `Backend: "cpu"`: the direct load against the sidecar load, one run each: Go heap in use after `runtime.GC` + `FreeOSMemory`, and 16 greedy tokens | tokens equal; the sidecar load's heap ≤ 0.25 × the direct load's |
| D8B-cuda | — | nobara's cell, ledger row D8b-cuda | not run here |

> **Amended during the build, before any verdict.** The kind is **8**: kind 7 was a pre-v16 kind (`decoder/serialize.go`'s
> kind list). The **version stays 17**: `docs/giw-bundles.md` § "From v17: a new encoding needs no new version" names
> q4k's form as the case that needs none (an older build refuses only the tensor it cannot read and rebuilds that
> sidecar), and a bump re-transcodes every cached sidecar on both boxes. "A format version" was this doc's own D8 wording,
> written before that rule was read, so the version was put back to the owner as **D9** (§7); he chose v17 the same day. D8B-G3 then reads: the version tests green at 17, and a kind-8 tensor refused by name by a reader
> that does not know kind 8.

> **Read 2026-10-21: all four PASS at v17** (`docs/measurements/d8b-2026-10/README.md`): the round trip byte-equal; the
> 0.5B's heap 583.6 MiB direct → 1.7 MiB through the q4k sidecar, same tokens; main's reader refuses kind 8 by name.

Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for G1–G3; about 2 min for G4 (one
0.5B q4k transcode, ~0.5 GB on disk). Plants: kind 7 written as an empty record (G1 red); the routing left direct (G2 red).

**D8a-build gates, registered 2026-10-21 before the code (D8A-G).** The build: under `Backend: "cpu"` (the stated promise
`decoder/weightmat.go:wantsCanonicalInt4` reads), the int4 layer tensors outside the batched Q/K/V/gate/up dispatch and the
embedding tables (`o_proj`, `down_proj`, routed and shared experts, and the router and family-specific layer tensors in the
same class, the one `quantizeWMSkipRow4` already names) are built as one row4 copy where `linalg.Int4Row4Usable` accepts the
shape, through `repackedOnlyOrCanonical`, as Q/K/V/gate/up already are. The reading behind it (aikit v1.71.0,
`linalg/weightmat_row4_arm64.go`): when a row4 layout is present, `WeightMat.matmulBTW4A8Q` dispatches to it at every M,
grouped activations included, so the canonical copy of these tensors is never read by a CPU kernel; they never reach
`matmulW4A8Batch`. amd64 is unchanged (no row4 there). Phi-3's fused-split tensors are batched-dispatch tensors and stay out.
The `.giw` writer refuses a repacked-only int4 tensor by name instead of writing it as an empty f32 record (it needs the
canonical bytes; a `Backend: "cpu"` heap model already reaches that with Q/K/V/gate/up today).

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| D8A-G1 bytes | 0, arm64 | every tracked tiny fixture under `testdata/` and `decoder/testdata/` that `Load` reads as a text model, at int4, `Backend: "cpu"` against `Backend: ""` | every in-class int4 tensor whose shape row4 accepts is row4-only (`Int4Layout() == "row4"`), and its bytes are the row4 bytes alone; ≥ 20 families load and ≥ 4 of them are MoE, else the gate is void |
| D8A-G2 bit-identity | 0, arm64 | the same loads: the logits of an 8-token prompt token by token, and 8 greedy tokens through `Generate` (batched CPU prefill) | byte-equal between `Backend: "cpu"` and `Backend: ""` on every family |
| D8A-G3 writer | 0 | `SerializeWeights` of a `Backend: "cpu"` int4 tiny model | an error naming the repacked-only tensor; no bytes claimed written |
| D8A-G4 tier 1 | 1, Mac | `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, `DirectLoad`, `Backend: "cpu"`, int4: Go heap in use after `runtime.GC` + `FreeOSMemory`, and 16 greedy tokens, main against the branch, one run each | the heap falls by ≥ 50 MiB (the arithmetic canonical bytes of `o_proj` and `down_proj`: 24 × (896·896 + 896·4864)/2 × 17/16 ≈ 63 MiB), and the tokens are equal |

> **Read 2026-10-21: all four PASS** (`docs/measurements/d8a-2026-10/README.md`): 40 families, 15 of them MoE, one copy
> and byte-equal logits; the 0.5B's heap 522.0 → 455.3 MiB under `Backend: "cpu"`, same tokens. Both plants red.

Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for G1–G3; about 1 min for G4. The planted
defects (a canonical copy kept; a row4 tile path forced off) are run once each and must turn G1 or G2 red.

**MEM13-tokenizer gates, registered 2026-10-21 before the code (TK-G).** The build: merge ranks keyed by the two pieces'
ids packed in a uint64 (a merge whose piece is not in the vocabulary keeps a string key, so no merge is lost), the merge
loop carrying each node's id, a size hint on the GGUF path's rank map, and a typed metadata reader in `tokenizer` that
walks the GGUF key-value section once and decodes only the keys the tokenizer reads, each to its own type, instead of
aikit's `map[string]any` of boxed values. Fixture: the committed `tokenizer/testdata/qwen2-0.5b-coder-meta.gguf` (Qwen's
151,646-token vocabulary and its merges), and the other two committed tokenizer GGUFs for identity.

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| TK-G1 identity | 0 | the `tokenizer` package's tests; a differential test that the typed reader returns what aikit's reader returns for every key the tokenizer reads, on each committed tokenizer GGUF; token ids of a fixed corpus (English, code, CJK, emoji, digits, whitespace runs) on each committed tokenizer GGUF, pinned from `main` before the change | every test passes with no new skip; every key equal; every id equal |
| TK-G2 memory | 0 | `LoadGGUF` of the Qwen fixture in a test: bytes allocated by the load (`TotalAlloc` delta) and heap kept with the tokenizer alive (`HeapAlloc` after `runtime.GC`, less the same before), each the median of 5 loads, read on `main` and on the branch | allocated ≤ 0.70 × main's, and kept ≤ main's − 3 MiB |
| TK-G3 encode speed | 0, by day | a new `BenchmarkEncode` (the TK-G1 corpus, Qwen fixture), built as a test binary from `main` plus the benchmark file and from the branch, run A B B A × 3 (`-benchtime 2s`), one at a time under the timing lock | the median of the six new/old ns/op ratios ≤ 1.02 (encode no more than 2% slower) |

> **Read 2026-10-21: all three PASS** (`docs/measurements/mem13-tokenizer-2026-10/README.md`): a load allocates 24.36 MiB
> against 88.58 and keeps 15.32 against 24.61 (two GCs; the instrument was fixed before the verdict, as the record says);
> encode 0.9935 of main's time.

Void: a TK-G3 cell whose 1-minute load average is above 4.0 at its start (CLAUDE.md's heavy-load rule) is re-run in its
place. Stopping rule: one pass each; a FAIL is diagnosed, not re-run. Cost: seconds for G1 and G2; about 2 min for G3.

**Heap-only quant, after MEM1 (read 2026-10-21, code at `58cd3021`).** With MEM1 merged, a load builds weights in the heap
only when it is asked to or has no `.giw` form: `Options.DirectLoad` (the apps' `-direct-load`), a q4k `.gguf`, a
safetensors directory with a LoRA or at bf16/f16, a platform other than darwin and linux, and test binaries. The item has
two halves, and the reading separates them:

- **The arm64 dual copy is now opt-in.** It is `decoder/weightmat.go:wantsRow4Fallback`'s CPU-fallback safety net on the
  heap path; the sidecar path never builds it (a `.giw` is mapped and gets no row4 copy). MEM0 §4.2 measured what it costs
  where it applies: arm64's int4 library load held 647 MiB on the 0.5B and 1,854 on the 1.5B, against amd64's 454 and
  1,150 (42–61% more; `Backend` empty, so every projection kept both forms). After MEM1 the same library load holds 1 MiB
  (`docs/measurements/mem1-2026-10/README.md`). Only a caller who sets `DirectLoad` pays it, and that caller asked for
  the heap build.
- **q4k still has no `.giw` form, so phi3 on the CPU or CUDA is still a heap load by default.** `internal/modelload`'s
  `activationSafeQuant` picks q4k for a phi3 `.gguf` on the CPU or CUDA (phi3 is the only family
  `decoder.ActivationQuantHazard` names), and q4k routes direct. Phi-3-mini's `.gguf` is 2.39 GB, and the heap build
  should hold about that (Q4_K tensors wrapped exact, the rest int8; arithmetic, not measured). On Metal, phi3 loads int8int8 through its sidecar and is not
  affected.

So the dual-copy half has no default caller left, and the q4k half is one family on two backends. **D8** (§7) asks which
of them to build.

## 5. The aikit items

### MEM14 · BM25: an incremental build and one postings arena (M)

**Problem.** The BM25 build takes the whole tokenized corpus at once — about 5.6 KB per chunk, ~4.7× the finished index
(~5.6 GB peak at 1M chunks), and a rebuild-and-swap holds the old index, the new tokens and the new index together. The
postings are one slice per term with ~1.32× append slack plus a 24 B header each, and there is no index format
(roadmap N4, N6 deferred as API decisions).

**Fix shape.** A builder that adds documents one at a time (N6), so peak is the index plus one document; one flat
`[]posting` arena with per-term offsets (~20–30% smaller), which is also the shape an N4 format can map.

**Gates.** Tier 0: identical scores on the existing BM25 golden corpus; bytes allocated by a build fall to O(index).

> **SHIPPED in aikit v1.71.0.** Postings arena (`74a9365`): scores bit-identical, index 20% smaller. `Builder.Add` and a saved,
> mappable index (`06c5619`): identical to `Build`; a build allocates 4.4× the index (bar 5×); a mapped load adds 0.66 MiB of
> heap. Night A/B against v1.70.2 on nobara, pre-registered at ≤ 1.10×: `Build` 1.026× (100k synthetic chunks) and 1.030×
> (aikit's tree) — ACCEPT; `Builder.Add` costs the same as `Build` (1.000× / 1.020×), so ken can move to it for the memory win
> at no time cost.

### MEM15 · HNSW graph as CSR, mapped (S–M)

**Problem.** `ann/hnsw.go:HNSW` holds per-node neighbor slices (~58 B of headers on ~100 B of ids), decodes the graph
into heap even under the mapped load, and in-process builds carry append slack (~260 B/node measured vs ~160 loaded). At
256-d int8 the graph is about 60% of the index.

**Fix shape.** Offsets plus one flat id array, written in the v-next blob and aliased from the mapping; pre-size the
build from n.

**Gates.** Tier 0: identical search results on the HNSW goldens; heap after a mapped load under a stated bound.

> **SHIPPED in aikit v1.71.0.** Build pre-sizing (`e9f1c41`): results and saved graph byte-identical; allocations per `Add`
> 69 → 2. CSR graph, blob v5 (`50f9780`): results identical, v4 blobs still load, a mapped load adds 0.001 MiB of heap (6.9
> before).

### MEM16 · Encoder tables and small copies (S each)

> **SHIPPED in aikit v1.71.0 and `gpu/annmetal/v0.2.1`, with one correction to this entry.** The "saves 0.5–0.77 GB" line
> below was wrong for F32 checkpoints: a gather-only F32 table read from the mapping already costs only the touched rows,
> as clean reclaimable pages, which an f16 heap copy cannot beat (+386 MiB on arctic2-m). What shipped instead:
>
> - **Word-embedding tables native-precision, mapped** (`a15bfa9`): F32 stays a mapped view; F16/BF16 are now mapped 16-bit
>   views widened one row per token (before: widened whole onto the heap); a misaligned data offset falls back to the copy.
>   Bit-identical across all 65,536 F16/BF16 patterns; minilm as F16/BF16 holds 43.3 MiB after load against 88.0 widened.
>   `WordEmbF16`/`WordEmbInt8` stay opt-ins (f16 cosine ≥ 0.9999999, int8 ≥ 0.99992 on four fixtures, `cb0c08b`).
> - **SPLADE row tiling** (`6e9609d`): bit-identical; a 512-token expand allocates 89.4 → 7.0 MiB (32-row tiles).
> - **Q8 `WordEmb` from the mapping** (`7bfd399`): bit-identical; heap after load 200.0 → 110.5 MiB.
> - **Model2Vec mapped by default** (`6284cb0`): bit-identical across 1,753 texts; time to first result 0.74× on the M1 and
>   0.71×/0.68× on nobara (July's +17% did not reproduce). A dropped mapped model is no longer freed: callers that reload
>   must call `StaticModel.Close` (ken).
> - **annmetal** (`0f20f32`): a mapped `FlatI8` is aliased, not copied (`EnableGPU` adds 0.45 MiB against 49.3 copied), batch
>   scratch reused, still opt-in. Released as `gpu/annmetal/v0.2.1` (v0.2.0 is the same code; its tag message failed the
>   evidence format check).

- **BERT/GTE have no int8 path** (`encoder/bert.go:BERT`); the word-embedding table is 45–69% of those models (bge-m3:
  1.02 of 2.27 GB) and is only gathered, so per-row int8 or f16 saves 0.5–0.77 GB.
- **`encoder/weights_q8.go:WeightsQ8` clones `WordEmb` to heap f32**: 94 MB, ~45% of the Q8 model; alias it or store it
  per-row int8.
- **Model2Vec defaults to a heap copy** (`embed/model.go:StaticModel`, 64 MB); make the mapped load the default for app
  bundles such as tack.
- **SPLADE materializes L × vocab logits** only to max-reduce them (`encoder/splade.go:SPLADE.expandIDs`): 62.5 MB per
  512-token expand per worker; max over row tiles is exact and ~2 MB.
- **annmetal** copies the int8 index into a device buffer while the host copy stays (2× on unified memory) and allocates
  an M × N buffer per call; since it loses to the CPU on Apple, the cheap fix is not enabling it there; the full one is a
  no-copy buffer over the mapped index.

### MEM17 · Chunk text by reference (owner decision)

**Problem.** `chunk/chunk.go:Chunk` copies its text out of the source, and ken keeps it: ~1.4 GB at 1M chunks, the
largest term on the index side.

**Fix shape.** A file + byte-range form of a chunk, with the text read through a mapping when needed. This is an API
choice in aikit and a storage choice in ken (**D4**).

## 6. Already decided or refuted — do not re-propose

| idea | status | where |
|---|---|---|
| native K-quant `WeightMat` kind, frequency row-blocked embed, 3-bit rotation, calibration scorer, cross-model dedup | shelved / rejected | `../ideas-weight-memory.md` |
| `DropCanonical` | rejected for the up-front representation policy | `task-int4-layout-2026-09.md` |
| `GOMEMLIMIT` after load | dropped: halved throughput where it bound | `../completed/task-never-swap-2026-09.md` S4 item 4 |
| `MADV_DONTNEED` eviction on darwin | a no-op on a read-only mapping; why pool mode is the darwin default | never-swap S5; aikit perf-dead-ends |
| `F_NOCACHE` on the expert pread fd | measured, no effect | Metal code notes |
| f32 routers, f32 matrices copied on `.giw` load | deliberate (discrete top-k; small) | the `.giw` writer's kind choice |
| embed-int4 as the CPU/CUDA default | owner decision 2026-09-28 | `internal/loadflags/loadflags.go` |
| PQ / IVF in aikit | int8 already took the 4× | aikit perf-dead-ends, "Reasoned out, not built" |
| CPU vision scratch pooling; folding worker scratch | measured, no gain | aikit roadmap §3 (A-4, A-5) |
| RoPE tables | none are precomputed; nothing to save | — |
| LM head on every prompt token; fused-attention scratch per decode token; KV regrowth garbage | fixed | audit-metal-2026-09-12; G40 |

## 7. Owner decisions

- **D1** (MEM1): the public API shape, and whether a library `.gguf` load takes the sidecar by default.
- **D2** (MEM4b): a lazy host reload of tower weights on CPU fallback — acceptable?
- **D3** (MEM11): per family, bf16 only (bit-identical) or int8 behind the parity bar.
- **D4** (MEM17): chunk text by reference in aikit's API, and ken's storage.
- **D5** (MEM8a): whether embed-int4 becomes the Metal default once the head kernel exists, on the CPU/CUDA evidence.
- **D6** (MEM3): **DECIDED 2026-10-17** — no KeepAlive discipline; the finalizer and `Tensor.owner` are deleted and `Close`
  is the only release (aikit `5e4d3c5`). The audit found one goinfer caller (the whisper decoder) that would have crashed
  under a real cleanup.
- **D7** (MEM16): **DECIDED 2026-10-17** — word-embedding tables load native-precision and mapped; f16/int8 heap tables
  are opt-ins.

- **D8** (MEM13-heapquant): **DECIDED by the owner, 2026-10-21: build both**, against the recommendations below. (a) single-copy
  weights for `o_proj`, `down_proj` and experts under `Backend: "cpu"` on arm64, every CPU kernel that reads them taking
  the single copy, gated on bytes and bit-identity; (b) a `.giw` form for q4k, a new tensor kind and a format version, so
  phi3 on the CPU and CUDA maps too. Order: (a), then (b); (b)'s CUDA cell is nobara's. The question as asked: (a) the arm64 dual copy on the heap path: close it now that the heap path is
  opt-in (recommended), or build row4-only `o_proj`/`down_proj`/experts under `Backend: "cpu"` (S–M: every CPU kernel
  that reads them must take row4-only, gated on bytes and bit-identity)? (b) q4k: give it a `.giw` form, a new tensor kind
  and a format version (M–L; phi3 on the CPU and CUDA then maps too), or leave phi3's CPU/CUDA load on the heap until a
  user meets it (recommended: one family, and its Metal load already maps)?

- **D9** (D8b-build): **DECIDED by the owner, 2026-10-21: (a), stay at v17.** The question as asked: the `.giw` format version for q4k's new kind 8. (a) **Stay at v17**, as
  `docs/giw-bundles.md` § "From v17" prescribes for a new encoding: only q4k sidecars are new, and nothing else rebuilds
  (recommended). (b) **Bump to v18**, as D8's wording said: every cached sidecar on both boxes re-transcodes once on its
  next load (each about the model's size on disk; the Mac is at 97% disk), with nothing gained that (a) lacks. The
  branch is built at v17; (b) is a one-line change.

- **D10** (MEM8(b)): **DECIDED by the owner, 2026-10-21: (a)**, the per-layer table only; `metal/model.go` stays untouched (if (a) needs it, stop and say so). The question as asked: Keying the gather-only tables on their own precision is a no-op on CPU and
  CUDA, where `--embed-int4` already defaults on and both tables are int4. It changes only Metal, where the head pin keeps
  them int8, so it is a default precision change on Metal (§2 rule 1) and not plumbing. Two further facts bound it. Metal's
  greedy chain uploads an untied input embedding to the device as int8 (`metal/greedy_chain.go`, through `int8BufA` and
  `metal/model.go:int8Buf`), so that table cannot leave int8 without Metal changes, in a file the unmerged
  `kv8a-metal-2026-10` branch also edits. And the Metal `.giw` sidecar's name does not carry the per-layer table's
  precision, so existing sidecars and fresh loads would disagree unless a variant tag moves. Options: (a) Gemma 4's
  per-layer table only, int4 on Metal by default (E2B's on disk and at transcode 2.35 → 1.32 GB; CPU and CUDA already run
  it at int4), with a Metal quality cell on E2B, the sidecar variant tag moved (each Metal Gemma 4 E sidecar re-transcodes
  once) and the Metal goldens re-pinned; the untied input embedding stays at the pin. (b) Plumbing only: a
  `decoder.Options` field for the gather-only tables' precision whose default follows the head pin, so nothing changes until
  it is set. (c) Fold it into D5 with MEM8(a) (parked, defer). **Recommended: (a)**; it is the only option that saves
  anything by default, and the evidence for int4 on that table is CPU and CUDA's default already.
- **D11** (MEM4(b)): **DECIDED by the owner, 2026-10-21: (a)**, aikit first, on an aikit branch under aikit's own `CLAUDE.md`; the aikit PR is opened, its release push and tag are the owner's, and goinfer's bump and serve's switch wait for the tag. The question as asked: D2 decided the policy (drop the towers' host copy after attach; reload lazily
  on a CPU fallback), but goinfer cannot carry it out alone. aikit v1.71.0 has the head-only shape for SigLIP only
  (`vision.LoadEncoderHead`, blocks loaded by `ensureBlocks` on the CPU path's first call). The Qwen3.5/Qwen3-VL, GLM-OCR,
  Qwen2.5-VL and Gemma 4 encoders keep their blocks in unexported fields, with no release and no lazy reload, and their
  exports (`Weights`, `GPUWeights`) alias those fields. Options: (a) aikit first: head-only loaders (or a block release
  plus `ensureBlocks`) for those four towers on SigLIP's model, an aikit minor release (its push and tag are the owner's),
  then goinfer's bump and the switch of serve's device-tower loads to head-only, gated on the bytes after attach and
  bit-identity of the CPU fallback after a reload; (b) leave MEM4(b) in aikit's queue until MEM4(a)'s aliasing work opens
  the same files. **Recommended: (a)**, with the aikit half done on an aikit branch for the owner to tag.

## 8. Sequencing

1. **MEM0** (by day, tier 1) — and its tack reading decides MEM1's rank.
2. **MEM2, MEM3** — bugs, small, tier 0.
3. **MEM1** if tack is on the library path; **MEM6, MEM7** — small, tier 0/1.
4. **MEM4** (towers) and **MEM5** (per family) — the large anonymous terms on image turns and new families.
5. **MEM8**, **MEM9** (with the `.giw` v16 format work), **MEM11** per family, **MEM12**.
6. **MEM10** on nobara when that box is free.
7. aikit: **MEM16, MEM14, MEM15 shipped** in v1.71.0 (2026-10-17); **MEM17** after D4; **MEM4(a)** after MEM0 (the mapped
   16-bit view MEM16 built for word embeddings is the same shape the towers need).

**Next, goinfer:** MEM0 (nobara half, then the Mac's cells); then MEM13's job-store and per-adapter token-bytes items. The aikit bump and MEM3's follow-ups are done (status block).

<!-- doc-reviewed: 2026-10-21 -->
