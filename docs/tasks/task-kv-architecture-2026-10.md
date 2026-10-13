# Task: one KV architecture — paged, content-addressed, tiered (KV0–KV10) — 2026-10

> **Status: SCOPED 2026-10-13. KV0 DONE 2026-10-13 (§6, KV0); KV1 is next.** Written on the owner's ask after a Cowork conversation the same day ("seems
> like we should consider a new KV architecture, given what goinfer is like now, feature-wise, what would be the best
> architecture if we were writing it from scratch. how close is the current code?", then "please write the task doc for
> the new architecture as proposed, be thorough").
>
> **D1 DECIDED 2026-10-13: paged attention is un-parked** (owner: "un park paged attention for sure"). It had been
> parked in `docs/roadmap.md` ("not this engine's weight class") with MC5's trigger, "the owner reverses the
> positioning, not a benchmark result" (`docs/tasks/parked/task-concurrency-2026-09.md`). The roadmap and MC5 now
> record the decision. The scope is paged KV as a **memory and reuse** mechanism for one user's conversations and
> agents. Continuous batching stays parked. `docs/positioning.md` keeps "no paged attention" until KV4 ships it.
>
> The facts about the code were read at main `e2ffb953` (2026-10-13). Nothing in this doc has been measured for it. Every
> gate below is a sketch: whoever starts a phase registers its gates first (rule, void conditions, stopping rule, cost
> basis), as `CLAUDE.md` requires.
>
> **Context.** This follows a 2026-10-12/13 thread on context size that listed six levers:
> 1. flags at run time;
> 2. Metal fit-by-default (M2, merged 2026-10-13);
> 3. f16 KV by default on CUDA;
> 4. raising Metal's 32,768 ceiling;
> 5. one shared KV pool;
> 6. KV in host memory.
>
> **Lever 3 is in the tree as of 2026-10-13, after this doc was written (KV0 found it).** f16 K/V is now the CUDA default for
> llama, mistral, qwen2, qwen3 and phi3 (`cuda/kvpolicy.go:cudaKVF16Decline` lists what keeps f32; owner override,
> commit 60222412). The text below that says CUDA is f32-only (this note as first written, §1, §4, §5.9, KV4b, D7) described the
> tree at `e2ffb953`. R6 phase 3 had been parked on 2026-09-26 on a *speed* case: about 1.05× on a GQA 7B at depth 3,900
> (`docs/prompts/cuda-r6-flash-decode.md`); the case for it was *capacity*, and the override took it on that ground. What
> is left for KV4b is f16 *pools* and the families still on f32. Lever 4 is KV-pre plus KV10, lever 5 is this doc, and lever
> 6 is KV7.

## 0. Summary

**What.** One KV manager in the decoder owns every piece of KV metadata: sequences, per-layer-group block tables, a
content-addressed prefix index, reference counts, eviction and tiers. The backends shrink to block storage plus kernels
that read KV through a block table. Recurrent state (DeltaNet, Mamba-2, short conv, KDA) becomes a second pool, with
checkpoints keyed by the same hashes.

**Why.** Today KV is a fixed per-slot reservation with per-slot bookkeeping, laid out differently on each backend. That
one shape causes five measured limits (§1): a conversation gets 1/N of the KV memory; a shared agent preamble is
prefilled once per conversation; hybrids get one slot and exact-extension reuse only; a conversation that loses its GPU
slot is gone; and every KV feature is built three or four times.

**What a user gets.**
- One conversation can use all of the KV memory, not 1/N of it.
- Conversations and subagents share what they have in common. Claude Code's first request is 19,692–20,199 tokens, most
  of it system prompt and tool schemas. KV0's traffic study checks how much of that stays the same across conversations.
- A parked conversation resumes without a re-prefill.
- Hybrids (Qwen3.5/3.6/3.8 and the other recurrent families) get prefix reuse and more than one conversation.
- CUDA gets an f16 KV cache, which it has never had.

**How.** Eleven phases, each shippable alone. At the same KV precision, each is bit-identical to the tree before it.
- KV1–KV3 change bookkeeping, copies and admission only. They deliver cross-conversation reuse and fix Metal's long-prompt
  413 without touching an attention kernel.
- KV4–KV5 are the kernel work.

| phase | what | backends | size | needs |
|---|---|---|---|---|
| KV-pre | long-prompt prefill in blocks, admission per block (**in flight**, not this doc's) | Metal | S | — |
| KV0 | prior art, block size, invariance and traffic measurements, owner decisions | — | S | — |
| KV1 | the manager over today's slots: logical blocks, hashed index, no storage change | CUDA, Metal, WebGPU | M | KV0 |
| KV2 | cross-conversation reuse by device copy between slots | CUDA, Metal, WebGPU | S–M | KV1 |
| KV3 | prefill and admission in blocks everywhere (absorbs KV-pre) | all resident | S | KV1 |
| KV4 | paged storage and block-table kernels; f16 KV | CUDA | L | KV3 |
| KV5 | paged storage | Metal | L | KV4 |
| KV6 | page or keep slots: decide, then build or record | WebGPU | M | KV5 |
| KV7 | tiers: host and disk, parking and resume | CUDA first, then all | M–L | KV4 |
| KV8 | recurrent state pool and checkpoints | CUDA, Metal | L | KV4 (KV5 for Metal) |
| KV9 | the CPU path onto the manager | CPU | M–L | KV2, D2 |
| KV10 | long context and lower-precision blocks | all | M | KV4/KV5 |

## 1. Why: what the current KV design costs

1. **One conversation gets 1/N of the KV memory.** Every resident slot reserves its whole context up front, so slots and
   context trade against each other.
   - CUDA shrinks context until the requested slots fit (`cuda/resident.go:ctxForSlots`, `cuda/resident.go:kvSlotsFit`).
     On the 8 GB card, the 2026-10-05 cold-user run's 7B at `-ctx 16384` got 1 slot of the 4 requested
     (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`).
   - Metal keeps 2 slots by default, and M2 picks the largest context up to 16,384 that fits beside the weights and both
     slots.
   - **The I7 night (2026-10-12):** the 7B resolved 31,328 positions in 2 slots (3.3 GB of KV), which left 2.5 GB
     available. A 20,199-token request was then refused with a 413, because admission priced its prefill at 3.21 GB
     (`docs/tasks/task-integration-harness-2026-10.md`, I7). The second slot's reservation held memory that no
     conversation was using.
2. **The same preamble is computed once per conversation.**
   - First requests: Claude Code 19,692 tokens (2.1.268, the Mac) and 20,199 (2.1.296, nobara); opencode 7,165; Codex
     5,850 (`docs/measurements/integration-harness-2026-10/harness-census-2026-10-12.md`,
     `docs/measurements/integration-harness-2026-10/claude-code-census-2026-10-12.md`).
   - Every new conversation or subagent prefills its preamble again. The 1.5B on the Mac took 34.2 s for a 20,954-token
     prompt (I7 night).
   - Resident reuse is per slot (`decoder/resident_reuse.go:Model.reuseLenOf`). It declines a "shared lead" on purpose
     (`decoder/resident_reuse.go:pickResidentSlot`), as serve's session LRU does
     (`internal/serveapp/sessions.go:pickSession`), because taking it would overwrite another conversation's history.
   - That is a mutable-slot problem. Immutable shared blocks do not have it (§5.5).
3. **Hybrids get one slot and exact-extension reuse only.**
   - `decoder/resident_reuse.go:Model.residentSlotCount` returns 1 when `decoder/forwardn.go:Model.hasRecurrentState`.
     `reuseLenOf` accepts only a strict extension of the committed sequence (R-01 phase 0).
   - An edited message or a second conversation therefore means a full re-prefill or the CPU path. On Qwen3.6-35B-A3B the
     CPU path decodes 5.6× slower (4.19 against 23.5 tok/s;
     `docs/measurements/r01-phase2-cpu-fallback-cost-2026-09-23.md`).
   - R-01 phase 2 and FreeToken Lead 1 are both open (`docs/tasks/task-recompute-audit.md`,
     `docs/tasks/task-freetoken-techniques.md`).
4. **A conversation that loses its GPU slot is gone.**
   - The CPU path has a RAM → NVMe tier (`internal/serveapp/sessions.go:sessionLRU`, `.giw-kv` v3 snapshots,
     `-session-dir`, `-kv-idle-demote`). The GPU paths have nothing.
   - R-04's resident half is open. QUEUE §A's figures, quoted there: about 257 MiB and 43 ms to park a 2.3k-token
     conversation, against about 8.9 s to recompute it.
5. **Every backend lays KV out its own way, so every KV feature is built three or four times.** (§4 has the table.)
   - Windowed KV: CUDA uses a shifted view with compaction (`cuda/kvwindow.go`, 2026-10-10). Metal and WebGPU allocate
     the whole context and mask it.
   - Precision: CUDA f16 for five families and f32 for the rest, with no int8; Metal f16 or i8; WebGPU f32, f16 or i8; the CPU f32 or i8.
   - Slots: a pointer swap on CUDA, contiguous views on Metal, whole runners on WebGPU.
   - Accounting: `docs/tasks/task-memory-accounting-2026-09.md` found five KV byte formulas. Today it is one function,
     `decoder/residentneed.go:Model.ResidentKVBytes`, with a "cuda" branch.
   - Context ceilings are allocation ceilings: Metal's 32,768 (`decoder/fitplan.go`), WebGPU's 16k/32k/64k by precision
     (`decoder/fitplan.go:WebGPUCtxCeiling`). Under paging they become table length and kernel limits (KV10).

## 2. What must not change

1. **Numerics.**
   - **Storage invariance:** at the same KV precision, paged storage returns logits bit-identical to slot storage, on
     every family fixture and real checkpoint each backend runs.
   - **The CPU reference stays exact.** The f32 CPU forward is the project's reference, and KV9 changes its storage only
     if the result is bit-identical.
2. **The four resident-reuse rules** (`decoder/resident_reuse.go`, header), restated as properties of the block key
   (§5.5):
   1. match on token ids, never text;
   2. nothing written by an incomplete pass is ever reused;
   3. at least one token is always recomputed for the seed logits;
   4. the weights, including the adapter, are part of the identity.
3. **Recurrent reuse stays exact-extension-only until KV8 replaces it with checkpoints.** A restore must be bit-exact,
   conv window included: spec/09 showed that skipping it gives wrong logits with no error.
4. **The cache-state grid stays fail-closed** (`decoder/cachestate.go:cacheStateGrid`: 14 kinds of state × 5 lifecycle
   paths). The new lifecycle paths (publish, share, demote, promote, restore) join it, and every kind gets a declared
   cell for each.
5. **One memory formula** (`docs/tasks/task-memory-accounting-2026-09.md`). The pool size, admission and the allocator
   compute from one function, and the never-swap rules hold on every gate (`docs/tasks/task-never-swap-2026-09.md`).
6. **The Hard API tier** (`docs/api-tiers.md`):
   - `decoder.KVCache` stays an opaque handle;
   - `Session`'s methods, `Model.NewCache`, `Model.NewSession` and `Model.LoadSession` keep their signatures;
   - the `-ctx`, `-kv-sessions` and `-session-dir` flags keep their meaning (§8, D6).
7. **MC3 identity.** A batched step's rows equal each sequence's own calls.
8. **Schedule independence.** A reply does not depend on what other conversations did, beyond the continuation effect
   that exists today (§5.6). This extends MC4's identity gate (`docs/measurements/mc4-candidate-cuda-2026-10-01.md`).
9. **Lossy is opt-in until gated.** A precision change goes through a fidelity gate before it can become a default.

## 3. What the KV layer has to carry

| feature | families (examples) | where it runs today | what the design does |
|---|---|---|---|
| full attention, GQA | most | all | full group |
| sliding window W | Gemma 2/3/4 local layers, Mistral, Cohere2, gpt-oss, Mellum, Phi-3 | CPU ring; CUDA windowed view (on in the CLIs); Metal, WebGPU full + mask | window group: blocks behind the window are released |
| chunked attention C | Llama 4 local layers (`attention_chunk_size`) | CPU (`attnChunkStart`) | chunk group: blocks before the current chunk are released |
| NoPE layers | Cohere2 globals, Llama 4 | all | nothing extra: K is stored as computed |
| MLA latent | DeepSeek V2/V3, Kimi K2, Ling 3.0's MLA layers | CPU, CUDA (one latent buffer), WebGPU | latent group (a row is the latent width) |
| KV-shared layers | Gemma 4 E2B/E4B | CPU, CUDA, Metal, WebGPU | no group of their own: they read their source's table |
| K = V layers | Gemma 4 globals (`attention_k_eq_v`) | CUDA copies k into v (audit R-23) | group flag: store K once (halves those layers' KV); measured before it is on |
| attention sinks | gpt-oss | all GPU | nothing (a learned logit, not KV) |
| recurrent state | DeltaNet (Qwen3.5/3.6/3.8, Qwen3-Next), Mamba-2 (Granite-4.0-H, Nemotron-H), short conv (LFM2), KDA (Ling 3.0) | CPU all; DeltaNet on CUDA, Metal, WebGPU; Mamba-2 on WebGPU | state pool and checkpoints (§5.7) |
| image blocks, bidirectional | Gemma 3/4 vision | CPU and the residents | spans are atomic: a hit never ends inside one |
| m-RoPE positions | Qwen2.5-VL, Qwen3-VL, Qwen3.5 with images | CPU and the residents | the image hash is in the key; the decode delta lives in sequence metadata |
| deepstack rows | Qwen3-VL | transient during the image prefill | covered by the image hash |
| LoRA at compute time | any | CPU and the residents (`ResidentAdapter`) | adapter content id in the key |
| speculation (n-gram, EAGLE, draft model, block drafters) | any | `ForwardN` plus a positional truncate | verify rows land in the private tail; a truncate frees whole blocks |
| batched decode | dense (MC3, MC3c) | CUDA, Metal; CPU workers | per-row block tables |
| chunked prefill | any | CUDA always (512-row passes); Metal under MC3 | chunks are a multiple of the block size |
| KV precision | — | f32 / f16 / i8 by backend | per pool, in the key |
| hidden-state routes | embeddings, decision heads (`HiddenLast`, `ResidualAll`) | transient | transient sequences, never published |
| sessions, persistence | — | CPU only | tiers (§5.8) |
| fit and admission | — | `Plan`, `AdmitPrefillMemory`, `kvSlotsFit`, `resolveMetalCtxCap` | one pool size; admission in blocks plus chunk scratch |

## 4. Today, per backend, and how close it is

| | CPU | CUDA | Metal | WebGPU |
|---|---|---|---|---|
| storage | per sequence, per layer `[]float32` growing by append; a ring per window layer (`decoder/kvcache.go:KVCache`) | per slot, per layer `ctxCap × kvDim`; window layers hold window + 512 (`cuda/kvwindow.go`) | per layer, slots in one allocation (`kvContig`), padded | per slot a whole `runModel` and `DecodeRunner` with bind groups baked over its KV (`gpu/kv_slots.go:webgpuKVSlot`) |
| precision | f32, or int8 with a scale per position and KV head | **f16 by default for llama, mistral, qwen2, qwen3 and phi3, f32 otherwise** (`decoder/residency.go:Model.ResidentKVPrecision`, `cuda/kvpolicy.go:cudaKVF16Decline`); no int8, no scales | f16, or int8 (f32 compiled out) | f32, f16 or int8 |
| context when unpinned | the model's window; KV allocated per request | fit up to 16,384, floor 4,096 (`cuda/resident.go:resolveCtxCapFit`) | fit up to 16,384 (M2), floor 4,096; `-ctx` up to 32,768 (`metal/model.go:resolveMetalCtxCap`) | a ceiling by precision |
| several conversations | sessions (`-kv-sessions 4`), MC3c workers / batched decode | MC1 slots (`cuda/resident.go:cudaResident.UseKVSlot`), MC3 (`cuda/batchstep.go:cudaResident.StepBatch`) | MC1, 2 slots by default; MC3 (`metal/batch.go:metalResident.StepBatch`) | MC1 only (clamp-only on discrete GPUs) |
| hybrids | all four recurrent kinds | DeltaNet, one slot | DeltaNet, one slot | DeltaNet, Mamba-2, one slot |
| prefix reuse | `decoder/session.go:Session.rewindForReuse` behind `bestExtend` / `pickSession` | per-slot id lists, longest common prefix (`reuseLenOf`), slot choice (`pickResidentSlot`) | the same | the same |
| persistence | `.giw-kv` v3 snapshots (`decoder/kvsnapshot.go:Session.Snapshot`), NVMe cold tier | none | none | none |
| prefill scratch | admission prices an upper bound over the whole prompt (`decoder/prefill_budget.go:prefillScratchBytes`) | 512-row passes, about 146 MB (`cuda/prefill.go:prefillChunkRows`) | the whole padded prompt in one pass (KV-pre is changing this) | allocated per call |

**What is already right and carries over:**
- **The cache-state grid** (`decoder/cachestate.go`). It is exactly the correctness skeleton the new manager needs.
- **The four reuse rules.** They become key properties instead of per-site checks.
- **Kernel-class awareness.** `decoder/residency.go:ResidentFastPrefill`, `decoder/residency.go:ResidentPrefillKernelFloor`
  and `decoder/resident_reuse.go:Model.declineShortLeadReuse` become the class component of the key (§5.5, §5.6).
- **MC1 and MC3.** Binding a slot and stepping several rows become "pass a sequence's block table" and "pass a table per row".
- **Chunked prefill.** It is chunk-invariant on Metal: 0 of 14,336,000 KV elements and 0 of 151,936 last-token logits
  differ at chunks of 64–384 (`metal/mc5_chunk_invariance_test.go:TestMC5_prefillChunkInvariance`).
- **CUDA's windowed KV.** Built 2026-10-10 and on in the CLIs (`docs/tasks/task-cuda-windowed-kv-2026-10.md`). Its
  `TestWindowedKV_*` suite becomes the window group's identity oracle.
- **One memory function**, `Model.ResidentKVBytes` / `Model.ResidentNeedBytes`.

**What does not carry over:**
- the per-slot reservation and per-slot id lists;
- slot-level LRU;
- the shifted view and compaction;
- whole-runner slots on WebGPU;
- the CPU/GPU split in reuse and persistence.

**How close is it?** The *semantics* are perhaps 70% there and well tested: which state exists, when reuse is safe, which
layers store what, and which kernel class wrote a row. The *storage* is close to zero, and that is where the effort is.
- **Kernels that read or write KV by absolute position**, each gaining a table lookup (counts corrected by KV0's census,
  `docs/measurements/kv0-tile-census-2026-10-13.md`; the first version of this list said "about 30" and "about 20"):
  - **CUDA:** 8 writers (`kv_store`, which is only ever launched at position 0 as a contiguous copy, `rope_kv`,
    `rope_kv_pw`, `rope_kv_batched`, `rope_kv_batched_pw`, `rope_kv_mrope_batched` and its pairwise twin, `mla_latent_store`) and
    43 reader entry points, about 19 distinct bodies: `attention`, `attn_batched`, `attn_block_full`, `attn_img_batched`, eight
    `attn_fused*` (the other two of the first list's ten, `attn_vit_*`, read vision-tower scratch), the three `splitkv_*` that read
    KV (`splitkv_scores`, `splitkv_vsum`, `splitkv_vsum_partial`), the flash-decode `fa_partial*` and 24 `fa_partial_rows_*`
    instantiations, `mla_attn`. There is no int8 KV on CUDA;
  - **Metal:** 4 live stores (`kv_store`, `kv_store_i8`, `kv_store_f16`, `mc3_kv_store_rows`) and 14 live reader entry points
    (`attention`, `attention_i8`, the `attention_fa*` family and its MC3 row twins, the four prefill kernels). Not KV kernels:
    `g4a_attn`, `attention_fa_combine`, `mc3_attention_fa_combine_rows`. Compiled but never dispatched: `kv_store_f32`,
    `attention_f32`, `rope2_kv`;
  - **WebGPU:** its attention pipelines (`gpu/attention.go`, `gpu/attn_keys_rows.go`, `gpu/attn_keys_wide.go`, `gpu/mla.go`) and
    the KV stores, about a dozen WGSL shaders.

## 5. The design

### 5.1 Overview

```
request ─► tokens ─► KV manager: Admit
                       ├─ prefix index (hash chain per layer group) ─► hit: blocks [0, k) shared, refcount+1
                       ├─ recurrent: restore the deepest checkpoint ≤ k (hybrids)
                       └─ private blocks for [k, n) + one block of headroom
forward passes read/write KV through a SeqView { table per group, pos, state slot }
a pass completes (device fence) ─► its full blocks are published: immutable, indexed
generation ends ─► references drop ─► blocks stay cached (refcount 0, LRU) ─► host ─► disk ─► evicted
```

```
full group (B = 64)      seq A: [ 7 | 12 |  3 | 40 | 41 ]      blocks 7, 12, 3: the shared preamble,
                         seq B: [ 7 | 12 |  3 | 55 ]           stored once, refcount 2
window group (W = 1024)  seq A: [ 66 | 67 | 68 | 69 | 70 ]     only ceil(W/B) + 1 blocks are held;
                                                               older ones go back to the pool (or the cache)
```

### 5.2 Layer groups

- **Derived at load from the `Architecture`**, never configured. A group is a set of layers with the same KV geometry
  (`kvDim`, element type, scales) and the same retention policy:
  - **full:** every position is kept;
  - **window(W):** a layer at position p reads [p − W + 1, p];
  - **chunk(C):** a layer at position p reads from the start of p's C-sized chunk (Llama 4);
  - **latent:** MLA's per-position latent, with a row width of `kv_lora_rank + qk_rope_head_dim`.
- **Aliases.** A KV-shared layer (Gemma 4 E-models) belongs to no group: it reads its source layer's table. A K = V layer
  is flagged, and may store K once (a measured option).
- **Recurrent layers** contribute a state spec (bytes per sequence), not a group.
- **Pools are per group** (option A). The planner sizes the window and chunk groups for the running sequences' needs and
  gives the rest to the full or latent group.
  - **Option B:** one arena with a page size uniform across groups, which is vLLM's choice. It makes memory fungible between
    groups, at the cost of padding (vLLM's own doc calls the result "some memory waste").
  - **KV0 chose option A** (`docs/measurements/kv0-tile-census-2026-10-13.md` §3): a window group's need is bounded per sequence
    (17 blocks for Gemma-3-4B's 1,024-token window at B = 64, 123 MiB of 763 MiB for a 32K sequence), so the fungibility buys little;
    option B pads where layer counts do not divide (Gemma-3's 29 local layers: 3.3% of the local group) and where `kvDim` differs
    by layer (Gemma 4); and the recurrent state is a different size class in both.

### 5.3 Blocks and block tables

- **A block is B consecutive positions** of one sequence in one group, at the same block index in every layer of the
  group. Storage per layer is `[numBlocks][B][kvDim]`, plus `[numBlocks][B][nKV]` scales for int8. That is today's
  row-major `[pos][kvDim]`, cut into pieces.
- **B = 64** (chosen in KV0; the working assumption had been 256). The census
  (`docs/measurements/kv0-tile-census-2026-10-13.md`) found that **B must be a multiple of 64 and nothing more**, and B = 64, 128
  and 256 all satisfy every kernel:
  - the only contiguous key tile aligned from absolute key 0 is CUDA's `attn_fused*` at 64 keys (full-attention layers); Metal's
    `attention_prefill_fused` needs B % 8 == 0 (B % 32 == 0 to keep its 32-key group in one block); no WebGPU kernel loads a
    multi-key tile. `attention_fa_blk64` is the head-dim-64 twin of `attention_fa_blk`; both load 32 keys a block, not 64;
  - B divides the prefill chunk (CUDA's 512; Metal's `PrefillBlockRows`, a multiple of 64 and at least 256, so 64 needs no
    change there and 128 or 256 would) and CUDA's 512-position window slack;
  - small enough that a window group stays close to its window (17 blocks of 64 for a 1,024-token window, against 5 blocks of
    256 that hold 1,280 positions) and that a shared preamble loses at most 63 tokens at its end; large enough that a lookup
    is a shift and 512 table entries cover 32K positions. 128 is the fallback if KV1's index measures worse than the arithmetic.
- **A table** is an int32 array of block ids per sequence per group, mirrored to the device. An MC3 step passes a
  `[rows][maxBlocks]` table.
- **The kernel change** is one indirection: key k lives at `table[k / B]·B + k mod B`, applied **per key**. Every reader on
  every backend already addresses each key's row separately, so no kernel changes the grouping of its arithmetic. That is the
  bit-identity argument, and it is *not* "tiles are aligned to blocks": that holds for `attn_fused*` on full-attention layers
  only. Sliding-window readers, flash-decode and the split-KV chunks (`lo = winStart + sp·per`, `per` moving every few tokens),
  Metal's decode flash-attention blocks and prefill writes at an arbitrary `startPos` are unaligned by construction. What keeps
  the bits is that **every tile, partition and thread-to-key map stays defined in absolute key units**; re-cutting one at a block
  edge is a numerical change, and a per-tile base-pointer optimisation is bit-safe only for `attn_fused` on full layers. The
  KV4/KV5 gates test this rather than trust it.
- **Waste** is at most one partial block per sequence per group: B/2 positions on average, 0.10% of a 32K sequence's KV at B = 64
  on the 7B, 0.62 MiB (`block_arithmetic.py`).

### 5.4 Sequences and their lifecycle

A sequence is one conversation's KV: its tables, its position, its recurrent state slot, its image spans and m-RoPE delta,
its adapter, and its namespace.

| operation | what happens |
|---|---|
| **Admit** | the longest valid hit is shared (refcount +1); private blocks are allocated for the rest of the prompt, plus one block of headroom; hybrids restore a checkpoint (§5.7) |
| **Append** | decode and verify write into the private tail; a new private block is taken when the tail fills |
| **Publish** | a full block becomes immutable and enters the index once the device work that wrote it has **completed** (fence), never on encode. This is rule 2's new form. A cancelled request still leaves its finished blocks reusable, because they are what those tokens produce. |
| **Truncate(p)** | speculative rollback or an edit: blocks wholly at or past p are released; the partial tail stays private |
| **Fork** | full blocks are shared; the private tail block is copied (n > 1 sampling, a branching agent) |
| **Release** | refcounts drop; published blocks stay cached at refcount 0 (evictable, LRU); unpublished ones are freed |
| **Park / resume** | release, then admit again through the index. With tiers, parked blocks move down a tier rather than vanish. |

There is no copy-on-write in the general sense. Only full, published blocks are ever shared, and a sequence only writes
into blocks it owns.

### 5.5 The prefix index

- **The key is a hash chain:** `h_i = H(h_{i−1}, group, tokens of block i, extras_i)`.
  - **The root** hashes the model's numerics identity (§5.8), the namespace (§5.15), the adapter's content id and scale,
    and the KV precision.
  - **The per-block extras** are the content hash and offset of every image span that touches the block, plus the
    **producer kernel class** of the block's rows (§5.6).
  - A block written by two classes, such as the end of a prompt and the start of its reply, records both and the row
    where they meet. It matches only the same mix.
- **H is SHA-256**, truncated for the map key. SHA-256 is vLLM's default hash since v0.10.2; its v0.8.5 doc estimated
  100–200 ns per token for the Python path, and an exploratory Go chain here measured 3–9 ns (§11a). The hash input needs a
  canonical byte encoding (vLLM's pickle default is not reproducible across versions, which is why `sha256_cbor` exists), which
  the disk tier (§5.8) depends on.
- **Only full blocks are indexed,** as in vLLM. A partial tail is private.
- **A lookup, across groups:**
  1. Take the longest chain of hits from block 0 in the full (or latent) group: L_full.
  2. Take the largest block-aligned L ≤ L_full at which every window and chunk group still holds the blocks its layers
     read at L.
  3. For a hybrid, take the deepest recurrent checkpoint at or below L.
  4. Apply the caps: L ≤ ⌊(n − 1)/B⌋·B (rule 3: the last token's logits are recomputed), and L never inside an image span.
  
  This is vLLM's intersection (`HybridKVCacheCoordinator.find_longest_cache_hit` iterates to a fixed point over any number of
  groups; its design doc says "exactly two", the code does not).
- **Eviction** takes refcount-0 blocks in LRU order, a sequence's later blocks before its earlier ones, so a chain
  shortens from its end. A block whose parent has gone can never be reached again, and is evicted first.
- **What it removes:**
  - The **shared-lead problem.** Shared blocks are immutable, so sharing a preamble can never truncate another
    conversation. `pickResidentSlot`'s shared-lead rule and serve's preamble floor go away.
  - **Per-slot id lists.** `resIDs`, `resSlots` and `residentCommitIDs` / `residentForgetIDs` become sequence state and
    publish/release.

### 5.6 Determinism: what the system promises

There are three properties, and each has a gate.

1. **Storage invariance (KV4, KV5; hard).** For the same reused blocks, paged and slot storage give bit-identical logits.
2. **Sharing invariance (KV2 on; hard).** A block shared across sequences is bit-identical to the one the requester would
   compute cold. Two things make that true:
   - **The producer class is in the key.** A request only matches blocks written by the kernel class its own cold
     prefill would use for those rows.
     - On CUDA a prompt of 512 tokens or more runs the fast kernels for every row (`FastPrefillFloor`). On Metal,
       `PrefillKernelFloor` marks the same kind of boundary.
     - So a 300-token request never takes rows a 20,000-token request computed. This is
       `declineShortLeadReuse`'s rule, generalised: its 64-token margin came from the MC4 graded run's root cause.
   - **Each backend's batched prefill is invariant to where a chunk starts.** Measured: Metal (`TestMC5_prefillChunkInvariance`),
     and by KV0 on CUDA and WebGPU (`docs/measurements/kv0-invariance-2026-10-13.md`): chunk widths of 64 to 512 with ragged
     tails and splits at 512, 768 and 984 are bit-identical on both; a split whose first call is under 512 tokens, or whose
     second pass is under 16 rows, is **not** on CUDA, which is the class rule below, not a defect. A second condition joins the
     key: **a pass of 15 rows or fewer is exact class** whatever the prompt's length.
3. **Continuation, as today (not bit-identical to cold).** A conversation's own decode-written rows are reused when its
   next prompt extends them, as resident reuse does now. Their class is "decode", which no cold prefill produces, so a
   match on them is accepted only as a continuation of the same chain. If both versions of the same content exist, the
   one equal to the requester's cold class is preferred.

**Consequence.** Cross-conversation sharing adds no reply variation beyond property 3's. Property 3's variation exists
today: whether a slot still holds a conversation already depends on what else ran. The schedule-independence gate (KV2)
is MC4's identity gate extended to shared preambles.

### 5.7 Recurrent state

- **A state spec per model.** spec/09 measured 62.8 MiB per sequence for Qwen3.6-35B-A3B and 149.6 MiB for the 27B
  (`docs/spec/09-mtp-heads.md`, quoted in `task-recompute-audit.md` R-01).
- **Two stores:**
  - a pool of **state slots**, one per running sequence. This lifts `residentSlotCount`'s one-slot rule for hybrids;
  - a **checkpoint store** keyed by `(chain hash at position c, c)`.
- **When to checkpoint.** Every block costs too much: at 62.8 MiB each, 40 blocks would be 2.5 GB. Checkpoints are taken
  only:
  1. **at commit**, the end of a completed generation. That covers the next turn and an edit back to this one;
  2. **at a preamble boundary** that serve marks from the rendered template (the end of the system message and tools).
     That is where cross-conversation sharing pays;
  3. **optionally every N blocks** under a byte budget, with admission weighed by expected reuse. Marconi's policy, read
     in KV0.
- **Restore** copies the checkpoint into the sequence's state slot, conv window included (spec/09's lesson), and
  recomputes from c. The hit is `min(attention hit, deepest checkpoint ≤ it)`.
- **Out of scope:** checkpointing per verify round for speculative rollback on hybrids. R-01 phase 1 priced it at 11.0%
  of a decode step and it was closed.

### 5.8 Tiers

| backend | tiers |
|---|---|
| CUDA, WebGPU (discrete) | device → pinned host → disk |
| Metal, WebGPU (integrated), CPU | one memory tier → disk |

- **Demotion.** Idle refcount-0 blocks go down a tier under pressure, and proactively for a parked conversation after
  `-kv-idle-demote` (that flag's CPU meaning today).
- **Promotion** happens on a hit, before use, with the rest of the chain prefetched.
- **The host tier's budget** comes from the never-swap live ceiling. Pinned memory cannot page, which is the point.
- **The disk tier** is a content-addressed block store under `-session-dir/<model fingerprint>/`.
  - Each block carries a CRC, a geometry guard, its precision, its producer class and the numerics epoch.
  - Files are owner-only (0700/0600), as `.giw-kv` blobs are today.
  - It stays opt-in, as `-session-dir` is today.
- **The numerics identity** is today's `internal/serveapp/main.go:modelFingerprint` (name, size, mtime, quant) plus the
  backend and a **numerics epoch** (the build's parity `deps_hash` or its version). A block written by another build is
  never trusted, because kernels change numerics between versions.
- **What it replaces:**
  - **at KV7:** the GPU parking gap (R-04's resident half);
  - **at KV9 (if D2 says so):** the CPU session LRU's `.giw-kv` cold tier. The v3 snapshot format stays readable for one
    release.

### 5.9 Precision

- **Per pool:** f32, f16 or i8. An int8 pool keeps one scale per position and KV head inside the block. Precision is in
  the root key, so blocks of different precisions never match.
- **CUDA's f16 is already the default** for llama, mistral, qwen2, qwen3 and phi3 (owner override 2026-10-13, ahead of this
  doc's KV4b); `--kv f32` keeps the reference. D7 is therefore decided for those families. KV4b extends f16 to f16 *pools* and to
  the families that still decline (`cuda/kvpolicy.go:cudaKVF16Decline`: windowed, MLA, MoE, pairwise RoPE, m-RoPE).
- **Lower precision, KV10, research only:** keys and values at different precisions (K8V4), or keys quantized per channel
  (KIVI/KVQuant style).

### 5.10 Memory: sizing, admission, pressure

- **One function prices the load:** weights, the KV pool, the recurrent state pool, the largest prefill chunk's scratch,
  and the margin. It extends `decoder/residentneed.go:Model.ResidentNeedBytes`; the planner, admission and the allocator
  all call it.
- **The pool** is the budget minus everything else.
  - The window and chunk groups are sized for the maximum number of running sequences; the full and latent groups get
    the rest.
  - On a paged-MoE load the expert cache keeps an explicit share. Today such a load holds KV at 4,096 so the pager gets
    the leftover, and the new rule keeps that behaviour unless a measurement says otherwise.
- **Admission** asks for blocks for [hit, prompt) plus one block of headroom, plus the chunk scratch, priced from the
  allocator's own figure, not `prefillScratchBytes`'s whole-prompt bound.
  - If the blocks are not free yet, the request waits in serve's existing FIFO (`internal/serveapp/admission.go:admission`).
  - A prompt over the per-sequence cap (the model's window, `-ctx`, the kernel limit, the pool) gets 400
    `context_length_exceeded`, as today.
  - A request that could never fit gets a 413.
- **No up-front reservation for `max_tokens`.** Claude Code sends 32,000. Decode grows a block at a time.
- **Under pressure (D5),** in order:
  1. evict idle cached blocks;
  2. demote them to the host or disk tier;
  3. preempt the most recently admitted running sequence by releasing it. Its published blocks stay; it re-admits through
     the index and continues. That is property 3's continuation, so its reply can differ from an unpreempted run by the
     continuation effect.
  
  Today the same request would have been refused at admission.

### 5.11 Prefill, batching, graphs

- **Prefill runs in chunks of k·B rows,** with chunk boundaries on block boundaries. The scratch is bounded by the chunk,
  not the prompt (KV-pre, KV3).
- **MC3's step takes a table per row.** Metal's bind limit is 31 in the pinned aikit (the "16" in `metal/batch_rows.go` is a
  stale comment; `mc3_attention_fa*_rows` binds 16 today, so a table buffer fits): one pool buffer per layer plus one table
  buffer. On CUDA there is no rows kernel to extend: `stepAttnRows` loops over rows and rebinds a slot per call, so a table per
  row is new kernel API there.
- **CUDA graphs.** No KV launch is captured today (graphs capture only the segments around attention), so a table pointer is a launch argument; a table the captured segments read would be a device buffer written before replay. Windowed
  models get graphs back; today graphs are declined when windowed KV engages (`task-cuda-windowed-kv-2026-10.md` §7). A
  measured bonus, not a promise.

### 5.12 Speculation and drafters

- **Verify rows go into the private tail.** A rollback is `Truncate`.
- **Window and chunk groups keep one block of slack** behind the window, so a rollback inside a verify width stays valid.
  This is the windowed doc's slack rule in block form.
- **Drafters' own caches stay outside the pool** in this doc: `cuda/drafter.go`, EAGLE, and the block drafters.

### 5.13 The backend contract (sketch; names provisional)

```go
// The decoder owns this metadata; backends never see hashes, refcounts or tiers.
type SeqView struct {
	Tables    [][]int32 // per group: block ids, logical order
	Pos       int
	StateSlot int       // -1: no recurrent state
}

// BlockStore is what a resident implements in place of ResidentKVSlotter.
type BlockStore interface {
	Groups() []GroupSpec                          // what it allocated, per group: block count, bytes per block
	Copy(g int, pairs [][2]int32) error           // device copies: fork tails, KV2's cross-slot reuse
	Demote(g int, ids []int32, dst HostBlocks) error
	Promote(g int, src HostBlocks, ids []int32) error
	StateSlots() int
	SaveState(slot int, dst HostState) error      // checkpoints (KV8)
	LoadState(slot int, src HostState) error
}
```

**What the decoder seam becomes:**
- `ResidentForward`'s `Forward`, `ForwardN` and `PrefillLast` take a `SeqView` instead of acting on "the bound slot".
- `UploadKV` writes through the table.
- `TruncateTo` and `Reset` become manager operations, which they already nearly are: no-ops on a positional cache.
- `ResidentCapped`'s cap comes from the manager.
- `ResidentReusableKV`'s windowed floor becomes the window group's retention policy.

**A backend not yet paged** keeps its slots behind a slot adapter. Logical block i of the sequence bound to slot s lives
at `(s, i·B)`, so KV1–KV3 run unchanged on every backend.

**The backends are separate modules** (`cuda/`, `metal/`, `gpu/`, each requiring a tagged goinfer). A phase that changes
the seam lands with `RELEASING.md`'s two-step tag.

**The manager is its own package** (amended 2026-10-13, owner decision). Sequences, block tables, the prefix index,
refcounts, the LRU and the slot adapter live in a package under `internal/` that imports nothing from `decoder`;
`decoder` imports it, and its oracle property test and wrong-hit mutations live beside it. The reason is the day loop: Go
re-runs a package's tests when anything it imports changes, so a manager inside `decoder` would re-run its property tests
on every `decoder` edit and every `decoder` test on every manager edit. Measured on 2026-10-13, 13 tests in `decoder`
were 143 of its 160 s, held there only by package placement (`docs/measurements/test-wait-2026-10.md` §2). KV1's
**Build** list starts with this package; the geometry it reads from the `Architecture` comes in as a plain value, not
the type.

### 5.14 The CPU path

- **What it is today:** a `KVCache` per conversation, `sessionLRU` choosing which conversation to keep, `.giw-kv` on disk,
  MC3c for concurrency.
- **What it costs:** four CPU sessions that share Claude Code's 19,692-token preamble hold it four times. At f32 on the 7B,
  about 115 KB per position (`docs/completed/task-cpu-kv-quant.md`), that is about 2.3 GB each.
- **D2 has two options:**
  - (a) the CPU keeps its storage and joins the index only for cross-session copies (KV2's shape);
  - (b) the CPU moves to host blocks, so a preamble is stored once, with a contiguous reference kept behind a test switch
    for parity.
- **Recommendation:** (a) when KV2 lands; (b) at KV9 if the memory case is measured on a real multi-session workload.
- `docs/tasks/task-cpu-kv-f16-2026-10.md` (CPU f16 by default) is independent of both.

### 5.15 Sharing scope and the timing side channel

- **The risk.** A shared prefix makes time-to-first-token observable: a prompt someone else already sent answers faster,
  and repeated probes can reconstruct parts of it (Chu et al., "Selective KV-Cache Sharing to Mitigate Timing
  Side-Channels in LLM Inference", arXiv 2508.08438).
- **goinfer is single-user by positioning,** but `serve` can face a LAN.
- **The design:**
  - the root key carries a **namespace**;
  - by default (D4) a server has one namespace, and one per API key when keys are configured;
  - `-kv-share off` turns cross-sequence sharing off and keeps per-conversation continuation.

### 5.16 Observability

- **`usage.prefill_reused_tokens` stays.** A vendor extension says where the reuse came from (own continuation, shared,
  host, disk) and how many blocks were promoted. These are `task-turn-telemetry-2026-09.md`'s consumers.
- **The banner** names the pool in tokens, the block size, the groups, the precision and the sharing scope. For example:
  "KV pool: 58,000 tokens (f16, 3.3 GB); one conversation can use all of it."
- **A `serve check` row:** a second conversation with the first's preamble reuses it.

### 5.17 Not in this design

- **Continuous batching beyond MC3.** It stays parked.
- **Position-independent caching:** reusing a block at a different absolute position (RAG chunk caching with re-rotation).
  K is stored after RoPE, so a block is valid only where it was computed.
- **Token eviction or compression that changes outputs:** StreamingLLM-style sinks, H2O, SnapKV.
- **Sharing across models or across precisions.**
- **A network tier** (LMCache-style remote KV).
- **Drafter KV in the pool.**

## 6. The phases

### KV-pre — Metal long-prompt prefill in blocks, admission per block (in flight, not this doc's)

vscode-claude's steps 1–2 of 2026-10-13, recommended to go ahead now. They came out of last night's I7 413.
- **The cause:** Metal's batched prefill sizes its f16 activation scratch to the whole padded prompt (`metal/prefill.go`,
  "sized to the padded prompt"). Read that day from the allocation: about 144 KB per prompt token on the 7B, so about
  3 GB at 21K tokens and 9.5 GB at 64K.
- **The fix:** prefill one long prompt in blocks, through the chunking mechanism that already exists, and price admission
  per block from Metal's own figure.
- **This doc asks one thing of it:** the block size is a multiple of B = 64 (it already is: `PrefillBlockRows` is a multiple of 64).

### KV0 — Prior art, block size, two invariance measurements, the traffic study, owner decisions (no product code)

1. **Read the prior art** (§11) and write a short section here: what to take, what to leave.
2. **Tile census.** Every KV-reading kernel's contiguous key tile and alignment, per backend. Choose B, and pools per group
   (A) or uniform pages (B).
3. **Invariance on CUDA and WebGPU.**
   - The twin of `TestMC5_prefillChunkInvariance`: a prompt prefilled whole and in chunks, KV compared element by
     element, with a planted one-token control.
   - Each backend's **class map**: for a prompt of n tokens prefilled from offset s, which kernel class writes each row.
   - A backend that is not chunk-invariant either gets a fix, or sharing on it stays within one conversation.
4. **The traffic study.** This is also the data FreeToken Lead 1 has waited for.
   - Real multi-turn sessions of Claude Code, opencode and Codex against goinfer on nobara (7B, CUDA), recorded with the
     I4 instruments extended to replies.
   - Per request: tokens shared with the same session's previous request; tokens shared with any other session's
     requests; non-extending edits; concurrency (title requests, subagents).
   - The output is projected time-to-first-token saved per new conversation and per turn, on both boxes, from measured
     prefill rates.
   - **Unknown today, and the study answers it:** whether the per-turn date Claude Code sends as a mid-conversation
     `system` message stays at the tail, which keeps the large preamble a stable shared prefix.
5. **Owner decisions D1–D8** (§8), recorded here.

- **Size:** S.
- **Decision rule, registered with the study before it runs:**
  - KV1–KV3 go ahead (they add no paged attention).
  - KV4 goes ahead (D1 is yes) when either the capacity arithmetic or the traffic study clears the bar written before
    the study runs.

### KV1 — The manager over today's slots (no storage change)

- **Build:**
  - the manager's own package (§5.13, "The manager is its own package");
  - the geometry (groups, state spec) from the `Architecture`;
  - sequences, logical blocks, the hash chain and index, refcounts, LRU;
  - the slot adapter;
  - `resident_reuse.go`'s bookkeeping replaced by manager calls, with the four rules and the class moved into the key;
  - the cache-state grid extended with the new lifecycle paths.
- **Not changed:** serve's admission and every backend kernel.
- **Gates (sketch):**
  - **Identity:** every existing reuse, slot, MC3 and MC4 test passes unchanged, and every parity golden passes
    unchanged.
  - **Oracle property test:** random sequences of admit, append, publish, truncate, fork and release against a plain
    per-sequence array model. Refcounts, tables and published contents are checked after every operation.
  - **Wrong-hit mutations:** drop the adapter, an image hash, the class or the precision from the key, and an
    adversarial test must go red. Each is shown once.
  - **Speed:** a lone request's time-to-first-token and decode within 0.99–1.01 of main.
- **Size:** M. **Closes:** nothing visible. It is the foundation.

### KV2 — Cross-conversation reuse by copy (still slot storage)

- **Build:**
  - a hit that lives in another slot is copied into the bound slot (`CopyDevice` / `CopyDeviceBatch`) and the rest is
    prefilled;
  - slots are reassigned by block LRU instead of slot LRU.
  - The copy is small next to the prefill it saves. A 19,692-token preamble on the 7B at f32 (112 KB a position) is about
    2.2 GB of device copy, around 10 ms on the 8 GB card, against seconds of prefill.
- **Gates (sketch):**
  - **Schedule independence (hard):** N conversations sharing a preamble, in every schedule the MC4 harness runs; every
    reply equals that conversation's reply alone.
  - **Time-to-first-token** for a new conversation sharing a preamble of 16K tokens or more: a band registered from
    KV0's projection, on both boxes.
  - **A lone request** at 0.99 or better.
  - **The copy cost** reported.
- **Size:** S–M. **Closes:** most of §1.2, without kernel work, on CUDA, Metal and WebGPU.

### KV3 — Prefill and admission in blocks on every backend (absorbs KV-pre)

- **Build:**
  - every resident's chunk size is a multiple of B (CUDA's `prefillChunkRows` default is already 512);
  - admission is in blocks plus the chunk scratch, from each allocator's own figure, replacing the whole-prompt bound on
    the resident paths.
- **Gates (sketch):**
  - the 20,199-token request answers on the Mac's 7B, which today is the I7 413;
  - chunk invariance holds per backend;
  - time-to-first-token at K = 512 and K = 3,900 is 0.97 or better against main.
- **Size:** S.

### KV4 — Paged storage on CUDA, then f16

- **KV4a: f32 paged, bit-identical.**
  - pools per group; device tables;
  - every KV kernel through the table;
  - the window and chunk groups replace `kvwindow.go`'s view and compaction;
  - the MLA latent group; KV-shared aliases;
  - MC3 tables per row; graphs for windowed models;
  - the planner sizes the pool in place of context × slots.
- **KV4b: f16 pools.** This reopens R6 phase 3 (parked 2026-09-26 on speed) on capacity.
  - The fidelity gate takes the registered shape `docs/completed/task-gpu-f16-kv.md` used on WebGPU: argmax agreement and
    cosine on a held-out set.
  - `docs/prompts/cuda-r6-flash-decode.md` notes that no family is known to need f32 KV: Metal runs f16 for every family.
  - The default is D7.
- **Gates (sketch):**
  - **Identity (hard):** every CUDA fixture and real checkpoint, with `TestWindowedKV_*` kept as oracles.
  - **Speed,** each the median of paired blocks:
    - decode at depth 128, 2,048, 3,900 and 8,000: 0.98 or better;
    - prefill time-to-first-token at K = 512 and 3,900: 0.97 or better;
    - MC3 aggregate at 4 clients: 0.98 or better.
  - **Capacity:**
    - one sequence reaches at least 90% of the pool's tokens;
    - four sequences sharing a 16K preamble store it once, bytes measured;
    - the 8 GB card's single-conversation context at least doubles at the default slot count, against today's.
  - **Never-swap** flat throughout.
- **Kill:** if decode at depth 2,048 or more stays under 0.98 after the registered tuning budget, KV4 parks with its
  measurement and KV2's copies stay.
- **Size:** L. **Closes:** lever 3 (CUDA f16), the slots-against-context question, and the windowed shifted view.

### KV5 — Paged storage on Metal

- **Build:** KV4's shape with Metal's kernels and its f16/i8 pools. Unified memory means no host tier.
- **Gates:** KV4's, with Mac numbers.
- **Closes:** the context-against-slots trade that M2 and I7 both make on Metal. I7's open question (how far context
  should give way to slots) dissolves.
- **Size:** L.

### KV6 — WebGPU: decide, then build or record

- **The facts:** a slot is a whole runner with bind groups baked over its KV, and the ceilings are per precision.
- **Options:** page it (pool buffers plus a table storage buffer), or keep slots with KV2's copies.
- **Recommendation:** keep slots, unless KV4/KV5's measured cost is low and a WebGPU user case exists.
- **Size:** M if built.

### KV7 — Tiers, parking and resume

- **Build:**
  - CUDA's pinned host tier;
  - a disk tier for every GPU backend;
  - promotion on hit;
  - the fingerprint plus numerics epoch.
- **Gates (sketch):**
  - **R-04's own rule:** resume against the fallback a lost slot really takes today, not against a cold prefill
    (`task-recompute-audit.md`, R-01's phase-2 rule);
  - promoted blocks are bit-exact;
  - persistence across a restart;
  - file permissions;
  - never-swap.
- **Size:** M–L. **Closes:** R-04's resident half.

### KV8 — Recurrent state pool and checkpoints

- **Build:** state slots, a checkpoint store, the preamble-boundary mark from serve, Marconi-style admission.
- **Gates (sketch):**
  - a bit-exact restore, conv window included;
  - hybrids' time-to-first-token for (a) a new conversation sharing a preamble and (b) an edited last message;
  - the checkpoint byte budget held;
  - more than one sequence for hybrids on CUDA and Metal, with MC3 identity.
- **Size:** L. **Closes:** R-01 phase 2, FreeToken Lead 1, and MC4's recurrent item.

### KV9 — The CPU path (D2)

- **Option (b) of §5.14.**
- **Gates:**
  - the parity manifest bit-identical (the CPU is the reference);
  - the memory saved with four sessions sharing a preamble, measured.
- **Size:** M–L.

### KV10 — Long context and lower precision

- **Build:**
  - raise Metal's and WebGPU's ceilings to their kernel limits;
  - a 64K/128K fidelity harness (lever 4's third step), with a YaRN-configured checkpoint for the long end;
  - the families with no fused prefill kernel, listed or given one;
  - K8V4 as research.
- **Size:** M.

## 7. Order, and what is useful if the program stops early

- **KV0 → KV1 → KV2 → KV3** deliver the agent-preamble win and fix the Metal 413, with no kernel changes and no paged
  attention. They pay for themselves before KV4 starts.
- **KV4** is the gate on the rest. If it parks on speed, KV2's copies and KV3's admission stay, and the doc records why.
- **KV7 and KV8 can come before KV5** for CUDA users. **KV9 waits on D2.** **KV10 waits on KV4/KV5.**
- **Every phase leaves the tree consistent:** a backend not yet paged keeps working behind the slot adapter.

## 8. Owner decisions

| id | question | recommendation |
|---|---|---|
| **D1** | Reverse the paged-attention parking narrowly, for memory and reuse within one user's conversations and agents? `positioning.md`'s "no paged attention" would change; continuous batching stays parked. | **DECIDED 2026-10-13: yes** ("un park paged attention for sure"). |
| **D2** | Does the CPU path join (KV9)? | Index and copies when KV2 lands; storage later, only on a measured memory case. |
| **D3** | WebGPU: page it or keep slots? | Keep slots; KV6 decides with numbers. |
| **D4** | Sharing scope by default? | One namespace per server, one per API key when keys are configured; `-kv-share off` available. |
| **D5** | Policy under pressure? | Evict idle → demote → preempt the newest by release (§5.10). |
| **D6** | CLI meanings before v1.0 (`-ctx`, `-kv-sessions` are Hard tier)? | Keep both: `-ctx` is a per-conversation cap, `-kv-sessions` the number of conversations kept warm and running on a resident. New flags only for sharing scope and the host-tier budget. |
| **D7** | CUDA's default KV precision once f16 passes its gate? | f16 by default with f32 kept as the reference, Metal's precedent. |
| **D8** | Where this sits in the queue? | KV0–KV2 soon (they pay for agents now). Where KV4 sits against the multimodal track is yours to set. |

## 9. Risks, and how each is contained

| risk | containment |
|---|---|
| **A silent wrong hit**, the worst failure: confident, wrong output with no error | every input that changes KV bytes is in the key; the cache-state grid extended and fail-closed; wrong-hit mutation tests for each key part (KV1) |
| **Replies varying with what else ran** | the producer class in the key; chunk invariance measured per backend (KV0); the schedule-independence gate (KV2) |
| **Indirection slowing the kernels** | tiles aligned to blocks; speed bands registered before the code; KV4's kill criterion |
| **Three GPU backends × about 50 kernels** | one backend at a time; the slot adapter keeps the others working; identity gates per kernel family |
| **Memory accounting drifting again** | one function prices load, admission and allocation (§5.10); never-swap on every gate |
| **CUDA graphs and table updates** | the table is a device buffer with a stable pointer, written before replay; measured in KV4 |
| **Disk blocks from another build** | the numerics epoch in the identity; CRC and geometry guard per block |
| **Privacy of persisted and shared KV** | owner-only files, opt-in persistence, namespaces, `-kv-share off` |
| **The v1.0 API freeze** | the Hard-tier surfaces keep their meaning (§2.6); the library surface review sees this doc |
| **Cross-module seam changes** | each seam change lands with the two-step tag |
| **Test budget** | heavy gates on the night queue under the timing lock, per `CLAUDE.md`'s run budget |

## 10. What this absorbs, closes or changes elsewhere

- **Absorbs:**
  - lever 5 (a shared KV pool);
  - MC5's paged-KV half (`docs/tasks/parked/task-concurrency-2026-09.md`);
  - R-01 phase 2 and R-04's resident half (`docs/tasks/task-recompute-audit.md`);
  - FreeToken Lead 1 (`docs/tasks/task-freetoken-techniques.md`);
  - the reopen condition of `docs/tasks/parked/task-kv-cache-streaming.md` (its host-KV idea becomes KV7);
  - lever 4's third step (KV10).
- **Replaces when built:**
  - `cuda/kvwindow.go`'s shifted view (KV4);
  - the per-slot bookkeeping in `decoder/resident_reuse.go` (KV1);
  - `pickResidentSlot` and serve's preamble floor (KV1–KV2);
  - WebGPU's runner-per-slot if KV6 says so.
- **Interacts with:**
  - **KV-pre:** its block size;
  - **I7** (`docs/tasks/task-integration-harness-2026-10.md`): its slots-against-context question dissolves at KV4/KV5,
    so its code should stay small;
  - `docs/tasks/task-cpu-kv-f16-2026-10.md`: independent;
  - `docs/tasks/task-library-surface-review-2026-10.md`: D6;
  - `docs/tasks/task-option-path-admission-2026-10.md`: new options join its grid.
- **Doc amendments when phases land:**
  - `docs/roadmap.md` and MC5 record D1 (done 2026-10-13); `docs/positioning.md`'s "no paged attention" changes when
    KV4 ships;
  - `docs/ARCHITECTURE.md` "The KV cache". It is already stale for CUDA: it says CUDA allocates the full context on
    sliding-window layers, which stopped being the default on 2026-10-10. Fix that in KV1's doc pass;
  - `docs/hardware-matrix.md` and the banner text.

## 11. Prior art, to read in KV0

- **PagedAttention** (Kwon et al., SOSP 2023, arXiv 2309.06180): blocks and tables, the origin of the idea.
- **vLLM's automatic prefix caching** (design doc, v1):
  - hash = parent hash + the block's tokens + extra keys (LoRA ids, multimodal hashes, a cache salt on block 0), with one chain for
    all groups and the group id appended to the map key (§11a);
  - full blocks only;
  - a free queue in LRU order with blocks released in reverse;
  - SHA-256 by default since v0.10.2 (`sha256_cbor` is the reproducible encoding).
- **vLLM's hybrid KV cache manager** (design doc):
  - KV cache groups with one page size;
  - sliding-window blocks freed outside the window;
  - a hit is the intersection across groups;
  - Mamba prefix caching: "work in progress" in the docs up to v0.14.0; on main since 2026-01 (align mode) and 2026-06 (Marconi-style
    admission), with defaults still moving (§11a);
  - KV-sharing layers patched in the model runner.
- **SGLang's RadixAttention** (Zheng et al., arXiv 2312.07104): a radix tree over token prefixes, eviction by leaves.
- **Marconi** (MLSys 2025, arXiv 2411.19379): prefix caching for hybrid models, with checkpoint admission and eviction
  weighed by reuse and the compute saved.
- **llama.cpp's memory module** (`llama-kv-cache` with a `unified` flag, `llama-kv-cache-iswa`, `llama-memory-hybrid`; the
  `-unified` file names are gone): per-cell sequence ids, sliding-window and recurrent memories. Read for how sequences share cells.
- **Ollama's Go runner:** per-slot caches matched by longest prefix, structurally like MC1 today. **Confirmed for its default mode and
  corrected (KV0, §11a):** the runner files were removed on 2026-05-29 (Ollama now runs `llama-server`); before that its
  `OLLAMA_MULTIUSER_CACHE` mode already did cross-slot prefix fork by sharing cells, which is KV2's shape.
- **TensorRT-LLM's KV cache reuse and host offload, and LMCache's tiers:** the tiering half.
- **Chu et al., arXiv 2508.08438 (SafeKV):** the timing side channel of shared prefix caches, and selective sharing.

## 11a. What we take and what we leave (read for KV0, 2026-10-13)

Everything below was read on 2026-10-13 at the revisions in the last table. The doc's §11 list was written before the reading, and
seven of its lines changed: see "What §5 and §11 get wrong" at the end.

| source | what we take | what we leave | basis |
|---|---|---|---|
| vLLM prefix caching (v1) | hash chain of (parent, token ids, extras); extras per block (LoRA, mm id + offset in block, salt on block 0); full blocks only; free blocks in reverse so chain tails go first; hit capped at n − 1; SHA-256 with a fixed seed | pickle/CBOR serialization; the per-process random seed (non-crypto hashes only); publishing at schedule time instead of after the device fence | READ; the key gaps are INFERRED |
| vLLM hybrid manager | per-group hit then intersection by fixed point; full group looked up once, others only shrink; window lookup right to left; freed window blocks become null entries; sparse recurrent checkpoints plus "semantic" ones; hash granularity finer than block size | one uniform page size with padding (Case 3 and 4); one LRU shared by all groups; the doc text (it says "exactly two types"; the code does not) | READ; Mamba status READ from main and open issues |
| SGLang RadixAttention | namespace in the child key; lock refs along the matched path; lock only the window segment for window layers; recurrent state restored by copy into the request's slot; pluggable eviction priority | the radix tree itself (token-granular hits cost splits and a second index we do not need); leaf-only eviction | READ |
| Marconi | admit two recurrent checkpoints per sequence: the last decoded state, and branch points; utility = recency + alpha x FLOP efficiency; checkpoint at a prefill chunk boundary | speculative insertion into a radix tree (a shared prefix only pays on the 3rd occurrence); alpha auto-tuning by replay; the headline 34.4x (its baseline checkpoints every 32 tokens) | READ |
| llama.cpp memory | cells hold a position plus a set of sequence ids, so a unified cache shares by metadata; iswa = two caches behind one interface; hybrid = attention cache + recurrent cache; recurrent state is one cell per sequence, rollback only through a bounded snapshot count; whole-state checkpoints with a minimum spacing | no hashing: reuse is longest common prefix per slot; SWA and recurrent reuse falls back to full re-processing when no checkpoint fits; RoPE-shift chunk reuse (our §5.17 excludes it) | READ |
| Ollama (Go runner, removed) | the multi-user mode: longest prefix across all slots, then fork into the oldest free slot by sharing cells | everything else; Ollama no longer has this code | READ; behaviour of the current build INFERRED from the launch flags |
| SafeKV | private-by-default visibility, checked on every lookup; per-user isolation keeps intra-user reuse | the three-tier sensitivity classifier and the diversity-ratio guard (they exist because tenants must share across users) | READ |
| PagedAttention | block size is a tradeoff between kernel utilisation and fragmentation; large blocks suit swapping | their 16-token default (their sequences were short) | READ |
| TensorRT-LLM, LMCache | priority-plus-LRU retention per token range with a duration; evicted blocks stay in the search tree on the host; pools per window size sized by a ratio | the network tier, and V1's static pool fractions | READ (docs only) |

### vLLM automatic prefix caching

[Design doc](https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html) (v0.8.5, the one §13 links) and `docs/design/prefix_caching.md` at main;
code `vllm/v1/core/kv_cache_utils.py`, `block_pool.py`, `kv_cache_manager.py`, `vllm/utils/hashing.py` at `ac71b175` (main, 2026-10-13).

READ:
- **What is hashed.** `hash_block_tokens` hashes `(parent_block_hash, tuple(token_ids), extra_keys)`; a missing parent becomes `NONE_HASH`.
- **Extra keys** (`generate_block_hash_extra_keys`): `("lora", name, path)` on every block; `("mm", identifier, offset_in_block)` for each image touching the block;
  `("cache_salt", salt)` on block 0 only; `("prompt_embeds", sha256 of the block's embeddings)`.
- **The group id is not in the chain.** It is appended to the hash as 4 big-endian bytes to form the map key (`make_block_hash_with_group_id`), so one chain serves every group.
- **Hash functions** (`PrefixCachingHashAlgo`): `sha256` (pickle, the default), `sha256_cbor`, `xxhash`, `xxhash_cbor`. There is no "builtin" option any more.
  The v0.14.0 doc says SHA-256 became the default in v0.10.2 and that pickle hashes "may not be reproducible across different Python or vLLM versions"; CBOR is the reproducible one.
- **The seed.** `NONE_HASH = hash_fn(seed)`. The seed is `PYTHONHASHSEED` if set, else the fixed string `"vllm-none-hash"` for cryptographic functions, else 32 random bytes for xxhash.
- **Full blocks only** ("We only cache full blocks"). A partial entry now exists only for Mamba align mode (`cache_partial_block`).
- **Free order.** `free_blocks` appends hashed blocks at the tail (LRU) and prepends unhashed ones at the head (reused first). A request's blocks are freed with `reversed(...)`,
  so the chain tail goes first ("the last block of a request must hash more tokens and is less likely to be reused"). A hit calls `touch`, which pulls the block off the free queue.
- **Rule 3 exists there too.** `get_computed_blocks` caps the hit at `num_tokens - 1` "to obtain logits".
- **Publishing.** A full block is cached as soon as it is allocated, "so that the block can be reused by other requests in the same batch". For Mamba this was a bug:
  PR #29387 (merged 2026-02-10) moves a request to the next step when it hits a block "computed in the current step", because the state "has not been computed yet".
- **The cost claim.** The v0.8.5 doc says SHA-256 "comes with a performance impact of about 100-200ns per token (~6ms for 50k tokens)". The v0.14.0 doc says "about 75ns per token (<4ms for 50k)".
  The main doc has no figure. All three are doc estimates for the Python path, not measurements.

INFERRED:
- The key has no model, kernel-class or precision component: one vLLM process holds one model at one precision. Our root and producer class are additions, not differences of taste.
- vLLM does not evict orphaned chain blocks first. §5.5's rule is ours.
- One chain per token stream (group id appended at insert) is cheaper than §5.5's `H(h_{i-1}, group, ...)`, which implies a chain per group. Worth deciding in KV1.
- **Exploratory, one run, not a result:** a Go `crypto/sha256` chain over int32 tokens on nobara (3700X, SHA-NI, go1.26.2) took 8.6 / 4.1 / 3.2 ns per token at B = 16 / 64 / 256.
  That is about 65 µs for a 20,000-token prompt at B = 256. The throwaway program is in the scratchpad, not the repo.

### vLLM hybrid KV cache manager

[Design doc](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) (the file says it was written at commit `458e74` and is "in its early stage");
code `kv_cache_coordinator.py` (`HybridKVCacheCoordinator`), `single_type_kv_cache_manager.py`, `config/cache.py` at `ac71b175`.

READ:
- **Groups and pages.** One pool, blocks of one page size; a group is layers of one attention type with the same page size. Case 2 repeats a regular ratio (2 window : 1 full).
  Case 3 (no ratio) groups by the smallest layer count: Gemma-3-27b, 52 window + 10 full, becomes groups of 10 with the last padded by 8 layers, "some memory waste and is not perfect".
  Case 4 (Mamba) raises the attention block size until `block_size x kv_hidden >= mamba state`, which "can lead to more than 400 block_size". Case 5, KV sharing, is invisible to the manager and "patched in the model runner".
- **Hit lookup.** The doc says the algorithm "applies to models with exactly two attention types". The code does not: `find_longest_cache_hit` loops over all attention groups (full first),
  each either accepts the candidate length or shrinks it, any shrink restarts the loop, and it stops when a pass shrinks nothing ("converges because length monotonically decreases").
  The full group is looked up once ("downward-closed"); a one-full-plus-one-other model breaks after one pass.
- **Window groups.** `SlidingWindowManager.find_longest_cache_hit` scans right to left for `cdiv(window - 1, block)` contiguous cached blocks. `remove_skipped_blocks` frees blocks behind
  `n - window + 1 - extra_retained_tokens` and puts the null block in their table slots; the freed blocks stay hashed in the LRU.
- **Hash granularity.** `hash_block_size` (default: the GCD of the groups' block sizes) can be finer than a group's block size, so a hit need not end on a big block's edge.
- **Where Mamba stands (read 2026-10-13).** Merged to main: align mode, PR #30877 (2026-01-23): the Mamba state is cached only at block boundaries; `mamba_cache_mode` is `none` or `align`;
  Marconi-style admission, PR #37898 (2026-06-10): last state, plus a shared prefix detected when the attention index hits further than the Mamba index;
  sparse retention, PR #52216 (2026-08-17): `prefix_cache_retention_interval`, default 0 = only "semantic" checkpoints (the replay boundary and shared-prefix junctions).
  Behind a flag: `--enable-mamba-shared-prefix-checkpoint` (off; needs EAGLE/MTP). The docs say "work in progress" up to v0.14.0 and no longer do on main.
  Open as of today: RFC #55697 (application-directed checkpoints through a `<|mamba_checkpoint|>` marker in the input, the same idea as our serve-marked preamble boundary),
  #58303 (a dense default reverts to 0% reuse under interleaved long conversations), #60838 (`--mamba-block-size` has no effect in align mode), #55403 (block priorities).

INFERRED:
- Our option A (pools per group) avoids the Case 3 and 4 padding by construction; the price is a planner that sizes pools. TensorRT-LLM V2 took the same road for the same models (below).
- A single LRU across groups is a consequence of one pool; under option A each pool has its own.
- vLLM's Mamba block size above 400 is a data point that B = 256 is not outlandish for a hybrid.

### SGLang RadixAttention

[Paper](https://arxiv.org/abs/2312.07104) (v2, 6 Jun 2024), §3 and §6.3; code `python/sglang/srt/mem_cache/` at `e9400c5a` (2026-10-13): `radix_cache.py`, `evict_policy.py`,
`unified_radix_cache.py`, `unified_cache/components/README.md`, `components/mamba.py`.

READ:
- **Tree.** Edges carry token sequences; KV sits "in a non-contiguous, paged layout, where the size of each page is equivalent to one token". `--page-size` defaults to 1; with a larger page, keys and matches are rounded down to the page (`RadixKey.page_aligned`).
- **Namespace.** `RadixKey(token_ids, extra_key, cache_salt)`; both are folded into every child key, so another namespace never reaches the same nodes.
- **Eviction and locks.** "least recently used leaf first"; each node has a reference counter and "a node is evictable if its reference counter is zero". Cache and running requests share one pool, with no preallocated cache. `inc_lock_ref` walks the matched node up to the root. The default policy is LRU; the code also has LFU, FIFO, MRU, FILO, priority, SLRU and a tail-optimised LRU.
- **Overhead and traffic.** 0.2 s of tree management in a 74.3 s run with no reuse (< 0.3%). One month on one Chatbot Arena worker per model: 52.4% hit rate (LLaVA-Next-34B) and 74.1% (Vicuna-33B).
- **Hybrid.** `UnifiedRadixCache` holds Full, SWA and Mamba components in one tree. A match advances only where every component's validator passes (an intersection). A window component keeps "tombstone" nodes (the node stays, its window data is gone)
  and locks only a window-long segment. Mamba state lives on a leaf, is restored by copy-on-write into the request's own slot (`finalize_match_result_in_cache`), and is stored at Mamba-chunk-aligned positions.
  Evicting Full cascades to SWA and Mamba; evicting Mamba cascades nowhere.

INFERRED:
- A radix tree gives token-granular hits; a fixed-block hash chain loses up to B − 1 tokens at the point where two prompts diverge, and at the end of a shared preamble. At B = 256 against a 20,000-token preamble that is about 1%, but it is a real cost.
  KV1–KV3 copy by position, so they can index at a finer hash unit than B (vLLM's `hash_block_size`) at no storage cost. Worth a line in KV0's block-size decision.

### Marconi

[Paper](https://arxiv.org/abs/2411.19379) (v3, 10 Apr 2025, MLSys 2025), §4 and §5; the authors' code (github.com/ruipeterpan/marconi) was not read.

READ:
- **Admission.** SSM state "cannot be rolled back". Marconi admits at most two states per sequence: the state after the last decoded token (conversations resume from it) and the state at a branch point of the radix tree.
  To find branch points it does a speculative insertion of the input before prefill; if that creates an intermediate node, the state at that node is saved during prefill.
  Cost, in the paper's words: "purely input" prefixes "only benefit from reusing starting from the third occurrence".
- **Getting the state during prefill.** For chunked-state models, save the state at the second-to-last chunk boundary (the hit may then fall a little short); otherwise a two-pass prefill.
- **Eviction.** `S(n) = recency(n) + alpha x flop_efficiency(n)`, both normalised to [0, 1], evict the lowest. FLOP efficiency is the compute saved by the entry divided by its bytes. Alpha 0 is LRU.
  Alpha starts at 0, then a replay grid search over the requests seen before the first eviction picks it (a few seconds).
- **Numbers.** Against vLLM+ (a checkpoint every 32-token block): token hit rate 4.5x, 7.3x, 34.4x (LMSys, ShareGPT, SWE-Bench); P95 TTFT down 36.1–71.1% (103.3–617.0 ms).
  Against SGLang+ (the same admission, LRU eviction) the eviction policy alone wins 45.6%, 19.0% and 219.7% at P95 hit rate. Setup: a 7B hybrid (4 attention, 24 SSM, 28 MLP layers), A100-40GB, traces replayed at varied arrival rates.

INFERRED:
- The admission rule is where the 34.4x comes from, and its baseline is a strawman for us. §5.7's rules 1 and 2 already are Marconi's two admission cases, and rule 2 reads the boundary from the rendered template, so it pays on the second conversation, not the third.
- vLLM's PR #37898 shows a tree-free way to detect a shared prefix: the attention index hits further than the checkpoint index.
- Rule 3 ("every N blocks under a byte budget") should start as LRU with the preamble checkpoint pinned. FLOP-aware eviction is a measured follow-up: the paper's own gain over LRU is the smaller one, and a single-user workload has far fewer entries to rank.
- Checkpoint positions should be prefill chunk boundaries, so a checkpoint costs no extra pass. That constrains the chunk size, as B does.

### llama.cpp memory module

[Repository](https://github.com/ggml-org/llama.cpp) at `77c4540c` (master, 2026-10-13): `src/llama-kv-cache.cpp`, `llama-kv-cache-iswa.cpp`, `llama-memory-hybrid.cpp`, `llama-memory-recurrent.cpp`, `llama-kv-cells.h`, `tools/server/server-context.cpp`, `server-task.cpp`, `common/common.h`.

READ:
- **Names.** `llama-kv-cache-unified*` are gone: `llama_kv_cache` takes a `unified` flag. Unified means one stream, so cells are shared by every sequence; not unified means `n_stream = n_seq_max` streams, one per sequence.
- **Sharing.** A cell has a position and a bitset of sequence ids (`llama_kv_cells`). Within a stream, `seq_cp` only adds the destination id to the cells: no data moves. Across streams it queues a buffer copy and is allowed only for a full cache.
- **iswa.** `llama_kv_cache_iswa` owns a base cache and an SWA cache and forwards every `seq_*` call to both. The SWA cache holds `min(size_base, n_swa x (unified ? n_seq_max : 1) + n_ubatch)` cells, padded to 256; `swa_full` makes it as large as the base.
- **Hybrid.** `llama_memory_hybrid` holds one `llama_kv_cache` and one `llama_memory_recurrent`. `seq_rm` tries the recurrent side first "since it may fail", then the attention side.
- **Recurrent.** One cell per sequence (a `tail` id), possibly shared by several sequences. A partial removal fails unless it is a rollback within `n_rs_seq` stored snapshots, and a shared cell "cannot move back for only one of them".
- **The server does no hashing.** No hash appears in `server-context.cpp` or `server-task.cpp`. `get_available_slot`: a slot given by id; else, if `--slot-prompt-similarity` (default 0.1) is non-zero, the idle slot with the best longest-common-prefix fraction; else the least recently used idle slot.
  With a slot chosen, `n_past` = common prefix with that slot's tokens.
- **Window and recurrent models.** If the slot's `pos_min` is past `pos_next - n_swa`, the server looks for a context checkpoint; with none it logs "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)" and sets `n_past = 0`.
  Checkpoints are whole-state blobs per slot: `n_ctx_checkpoints` 32, `checkpoint_min_step` 8192 tokens, oldest erased, ones too close to an earlier one erased.
- **A RAM prompt cache** (`--cache-ram`, default 8192 MiB) saves a slot's whole state when the slot is about to lose more than half of it or is taken by LRU. `server_prompt_cache::load` picks the saved prompt with the best common-prefix fraction (skipping any that keep under 25%) and removes it from the cache when it is loaded.
- **Shifting.** `ctx_shift` is off by default. `n_cache_reuse` reuses matching chunks by shifting positions; it needs a memory that can shift, which excludes SWA and recurrent memories and multimodal prompts.

INFERRED:
- llama.cpp has our "shared lead" problem and solves it per slot (similarity threshold, saved prompts), not with immutable shared blocks. Its SWA and recurrent fallback is the failure mode §5.5's group intersection and §5.7's checkpoints exist to avoid.
- A cell that carries a set of sequence ids is the ref count by another name. Under KV2 (slot storage, copy on hit) we need only the copy.

### Ollama

[Repository](https://github.com/ollama/ollama). Main is `665f9e07` (v0.40.3, 2026-10-13). The Go runners were last present at `f63eea3d` (2026-05-24).

READ:
- **The files in §11 and §13 do not exist at main.** Commit `9db4bdba` (2026-05-29, PR #16031) is "runner: Remove CGO engines, use llama-server exclusively for GGML models". `llm/llama_server.go` now starts upstream `llama-server` (pinned in `LLAMA_CPP_VERSION`) with `-np <numParallel> -c NumCtx x numParallel`, so each slot gets `NumCtx`; `--kv-unified --cache-ram 0` is used only for "decision" requests; requests send `cache_prompt: true`. `OLLAMA_NUM_PARALLEL` defaults to 1 in `envconfig/config.go`.
- **At `f63eea3d`, `runner/ollamarunner/cache.go`:**
  - Default (`OLLAMA_MULTIUSER_CACHE` unset): `findLongestCacheSlot`, the free slot with the longest common prefix of token ids (plus multimodal hashes); the match is reused in place and the slot's tail is cut with `Remove`.
  - With `OLLAMA_MULTIUSER_CACHE`: `findBestCacheSlot` takes the longest prefix over all slots, busy ones included. If that slot is free and wholly a prefix of the prompt it is reused; otherwise the oldest free slot is chosen and `CopyPrefix` forks the prefix into it. In `kvcache/causal.go`, `CopyPrefix` adds the destination sequence id to the source's cells: shared, not copied.
  - Window models: `CanResume` accepts a position only if the new window lies inside the stored window. Recurrent models: `PrepareRestore` picks a checkpoint (conv and recurrent state per position) or resets to 0.
  - Context shift (`ShiftCacheSlot`) drops the oldest half beyond `numKeep`; if the cache cannot remove a middle range it clears the slot and returns the inputs to be reprocessed (`ErrReprocessInputs`). `llamarunner/cache.go` has the same slot logic over llama.cpp.

INFERRED:
- The doc's line is right for the default mode and understates the multi-user one, which is KV2's shape (cross-slot prefix reuse by fork) behind an off-by-default env var. It is not a description of Ollama today: with one slot by default, what a new conversation can reuse is llama-server's RAM prompt cache.

### SafeKV (Chu et al.)

[arXiv 2508.08438](https://arxiv.org/abs/2508.08438) v2, 9 Feb 2026, §2.3, §4, §7.

READ:
- **Threat model.** An adaptive black-box adversary with arbitrary accounts, timed and batched probes, coordinating across accounts, observing TTFT and per-token latency; it cannot read prompts, weights or KV and cannot run code on the server.
- **Mechanism.** One radix tree across HBM, DRAM and SSD. Each node has `private_tag` and `creator_id`, checked "at each hop" of a lookup. New nodes are private; an asynchronous detector promotes a block to shareable.
  Detector: tier 1 patterns (under 0.2 ms, flags 60–70% of private blocks), tier 2 a Llama-3.2-1B classifier (under 125 ms), tier 3 a context-aware check for under 8% of requests. A reuse-diversity ratio (distinct users over hits) demotes a shareable block whose access pattern looks like probing.
- **Numbers.** Defence success above 94% on average; against per-user isolation, TTFT overhead cut by up to 40.58% (50.41% → 11.74% on Qwen3-235B-A22B; 118% → 34.28% on DeepSeek-R1) and throughput 1.36–2.66x. Full isolation itself inflated TTFT by about 38% in their multi-tenant setup. Per-user isolation "preserves KV-cache reuse within each user".

INFERRED, for §5.15:
- Our namespace in the chain root is the simpler form of their per-node check: a block of another namespace is unreachable, not merely refused. It is also the paper's "isolation" baseline, whose cost came from many users with nothing in common. Here the sharers are one user's agents, so the cost does not apply.
- We should not build detection. Their classifier exists because tenants must share across users; ours do not.
- Promotions from the host and disk tiers are timing-visible too; the namespace covers them.
- D4's default (one namespace per server when no API keys are set) lets any LAN client probe. A safer default is `-kv-share off` for cross-namespace sharing whenever `serve` binds beyond loopback and no keys are set. This is a proposal for the owner, not a source finding.

### PagedAttention, TensorRT-LLM, LMCache

- **PagedAttention** ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180), §7.2) READ: "If the block size is too small, vLLM may not fully utilize the GPU's parallelism"; "if too large, internal fragmentation increases and the probability of sharing decreases". On ShareGPT, 16 to 128 were best; on Alpaca (short sequences) 16 and 32 were good and larger "significantly degrade". They chose 16 as "large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads".
  For swapping, small blocks give "numerous small data transfers" that limit PCIe bandwidth; recomputation does not depend on block size and "is never higher than 20% of swapping's latency" in their comparison. INFERRED: their sequences were hundreds of tokens; ours are 6,000–20,000, so the waste of at most one partial block per group at B = 256 is 1.3% of a 20,000-token sequence and 4.3% of a 6,000-token one. Their swap result favours a large B for KV7's host tier.
- **TensorRT-LLM** ([`docs/source/features/kvcache.md`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) at `eaefddce`, 2026-10-13) READ: blocks go into a radix search tree when filled; eviction is "prioritized LRU" with priorities 0–100 set per token range and a duration after which they revert (default 35); an evicted block is copied to host memory and "remains in the search tree"; V1 makes one pool per (window, head count) with a static memory fraction, "not optimal". The V2 manager is chosen for hybrid Mamba, GPT-OSS and Gemma models and sizes window and full pools independently (`pool_ratio`).
  The "retention priority with expiry" is a ready form for the preamble pin in §5.7.
- **LMCache** ([README](https://github.com/LMCache/LMCache) at `4d69cdc4`) READ: a layer that stores KV persistently and reuses it across engines, with P2P CPU-memory sharing. Not read further; it is the network tier §5.17 excludes.

### What §5 and §11 get wrong (or say too weakly)

- **§5.5, "vLLM … prices it at about 100–200 ns per token."** True of the v0.8.5 doc only; the v0.14.0 doc says about 75 ns; main has no figure. All are estimates for the Python path. Say "vLLM's v0.8.5 doc estimated 100–200 ns per token in Python; an exploratory Go run here measured 3–9 ns", or drop the figure.
- **§5.5, "vLLM recommends SHA-256 for multi-tenant use."** It has been the default since v0.10.2. The options are `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`; the "builtin" hash of the v0.8.5 doc is gone. §5.8's disk tier needs a canonical byte encoding for the hash input; vLLM's pickle default is not reproducible across versions, which is why `sha256_cbor` exists.
- **§5.5, "This is vLLM's intersection, extended to more than two kinds of layer."** The design doc says two; the code (`HybridKVCacheCoordinator.find_longest_cache_hit`) already iterates to a fixed point over any number of groups. Cite the code.
- **§11, "Mamba prefix caching 'work in progress'."** Stale since the doc moved on: align mode is on main (PR #30877, 2026-01-23), Marconi-style admission since 2026-06-10 (PR #37898), with defaults still being fixed in open issues (#58303, #60838).
- **§11 and §13, Ollama.** The two runner files were deleted on 2026-05-29 (PR #16031). The line is accurate for the default of the old code, understates its `OLLAMA_MULTIUSER_CACHE` mode (cross-slot prefix fork, KV2's shape), and does not describe Ollama today (llama-server, `-np N`, a RAM prompt cache). Re-point the §13 paths at `f63eea3d` or drop them.
- **§11, llama.cpp file names.** `llama-kv-cache-unified.cpp` and `-unified-iswa.cpp` are now `llama-kv-cache.cpp` and `llama-kv-cache-iswa.cpp`, with `unified` a runtime flag; there are also `llama-memory-hybrid-iswa.cpp` and `-hybrid-idx.cpp`.
- **§5.4, publish after the fence.** Supported by vLLM's own bug: it publishes at schedule time and PR #29387 had to stop same-step hits for Mamba. Cite it as the reason.

### Revisions read

| source | revision | date |
|---|---|---|
| vLLM | `ac71b175172f2c092d7141f41f0777b914788cfb` (main); docs v0.8.5, v0.12.0, v0.14.0 | 2026-10-13 |
| SGLang | `e9400c5a8f3a0d0312a37aecef694f5edfd69341` (main) | 2026-10-13 |
| llama.cpp | `77c4540ce4667f4577753fe034a3a8042f510b8f` (master) | 2026-10-13 |
| Ollama | main `665f9e074c99fe5abd57f6c2c2948dd60543a637` (v0.40.3); Go runners at `f63eea3d273816ffc27fbeb0662ab44d236abc45` | 2026-10-13; 2026-05-24 |
| TensorRT-LLM | `eaefddcecd752d9a76a7bd43536643a5d6110c1f` | 2026-10-13 |
| LMCache | `4d69cdc43ac9bab9c0c1cd5cac380c480cb32224` (dev), README only | 2026-10-13 |
| Papers | arXiv 2312.07104 v2, 2411.19379 v3, 2508.08438 v2, 2309.06180 (HTML and PDF text) | 2026-10-13 |

Not read: Marconi's own code; the vLLM PR bodies beyond #29387, #37898 and the two issues quoted (titles and dates of the others come from GitHub search); LMCache beyond its README.

## 12. When this closes

The design goes where the engine's description lives: `docs/ARCHITECTURE.md`, "The KV cache", rewritten phase by phase
as each lands. If it outgrows that section, it becomes `docs/kv.md`, linked from there. This doc then moves to
`docs/completed/` with its measurements, as the other programs have. There is no separate architecture doc while this is
a proposal: it would describe an engine that does not exist yet, and two documents of the same design drift apart.

## 13. Sources

Code, read at `e2ffb953`:
- **Decoder:**
  - `decoder/kvcache.go:KVCache`, `decoder/kvcache.go:KVCache.TruncateTo`, `decoder/cachestate.go:cacheStateGrid`;
  - `decoder/resident_reuse.go:Model.reuseLenOf`, `decoder/resident_reuse.go:pickResidentSlot`,
    `decoder/resident_reuse.go:Model.residentAcquireSlotAll`, `decoder/resident_reuse.go:Model.declineShortLeadReuse`,
    `decoder/resident_reuse.go:Model.residentSlotCount`, `decoder/resident_reuse.go:Model.residentReuseFloor`;
  - `decoder/residency.go:ResidentForward`, `decoder/residency.go:ResidentKVSlotter`,
    `decoder/residency.go:ResidentBatchStepper`, `decoder/residency.go:ResidentFastPrefill`,
    `decoder/residency.go:ResidentPrefillKernelFloor`, `decoder/residency.go:Model.ResidentKVPrecision`;
  - `decoder/session.go:Session.rewindForReuse`, `decoder/kvsnapshot.go:Session.Snapshot`,
    `decoder/kvsnapshot.go:Model.LoadSession`;
  - `decoder/forwardn.go:Model.hasRecurrentState`, `decoder/forwardn.go:Model.specRollbackSafe`;
  - `decoder/model.go:Model.generateInto`, `decoder/model.go:Model.mc3Prefill`;
  - `decoder/fitplan.go:Model.Plan`, `decoder/fitplan.go:WebGPUCtxCeiling`;
  - `decoder/residentneed.go:Model.ResidentKVBytes`, `decoder/residentneed.go:Model.ResidentNeedBytes`;
  - `decoder/prefill_budget.go:Model.AdmitPrefillMemory`, `decoder/prefill_budget.go:prefillScratchBytes`.
- **CUDA:**
  - `cuda/resident.go:kvSlotsFit`, `cuda/resident.go:ctxForSlots`, `cuda/resident.go:kvBytesForCap`,
    `cuda/resident.go:resolveCtxCapFit`, `cuda/resident.go:cudaResident.checkKVFits`, `cuda/resident.go:cudaLayer`,
    `cuda/resident.go:cudaResident.UseKVSlot`;
  - `cuda/kvwindow.go:cudaResident.kvEnsure`, `cuda/kvwindow.go:cudaResident.ReusableKV`;
  - `cuda/prefill.go:prefillChunkRows`, `cuda/batchstep.go:cudaResident.StepBatch`.
- **Metal:** `metal/model.go:resolveMetalCtxCap`, `metal/backend.go:metalResident.UseKVSlot`,
  `metal/batch.go:metalResident.StepBatch`, `metal/mc5_chunk_invariance_test.go:TestMC5_prefillChunkInvariance`.
- **WebGPU:** `gpu/kv_slots.go:webgpuKVSlot`.
- **Serve:** `internal/serveapp/sessions.go:sessionLRU`, `internal/serveapp/sessions.go:bestExtend`,
  `internal/serveapp/sessions.go:pickSession`, `internal/serveapp/admission.go:admission`,
  `internal/serveapp/main.go:modelFingerprint`.

Docs and records:
- **Positioning and policy:** `docs/positioning.md`, `docs/roadmap.md`, `docs/api-tiers.md`, `docs/ARCHITECTURE.md`.
- **Task docs:**
  - `docs/tasks/parked/task-concurrency-2026-09.md` (MC1, MC3, MC5), `docs/tasks/task-recompute-audit.md` (R-01,
    R-04), `docs/tasks/task-freetoken-techniques.md` (Lead 1);
  - `docs/tasks/task-cuda-windowed-kv-2026-10.md`, `docs/tasks/task-memory-accounting-2026-09.md`,
    `docs/tasks/task-never-swap-2026-09.md`;
  - `docs/tasks/task-integration-harness-2026-10.md` (I4, I7), `docs/tasks/task-cpu-kv-f16-2026-10.md`,
    `docs/tasks/parked/task-kv-cache-streaming.md`;
  - `docs/completed/task-gpu-f16-kv.md`, `docs/completed/task-cpu-kv-quant.md`, `docs/spec/09-mtp-heads.md`.
- **Measurements:**
  - `docs/measurements/integration-harness-2026-10/harness-census-2026-10-12.md`,
    `docs/measurements/integration-harness-2026-10/claude-code-census-2026-10-12.md`;
  - `docs/measurements/cold-user-2026-10-05-nobara-pc.md`,
    `docs/measurements/r01-phase2-cpu-fallback-cost-2026-09-23.md`,
    `docs/measurements/mc4-candidate-cuda-2026-10-01.md`, `docs/measurements/chunked-prefill-2026-09-27.md`;
  - `docs/measurements/concurrency-mc3-2026-09-26.md`, `docs/measurements/concurrency-mc3-cuda-2026-09-27.md`.

External:
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html)
- [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)
- [Marconi](https://arxiv.org/abs/2411.19379)
- [Selective KV-cache sharing (SafeKV)](https://arxiv.org/abs/2508.08438)

<!-- doc-reviewed: 2026-10-13 -->
