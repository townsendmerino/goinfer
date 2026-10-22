# Task: one KV architecture — paged, content-addressed, tiered (KV0–KV11) — 2026-10

> **Status 2026-10-21 (the close-out sweep).** KV0, KV1, KV2 and KV3 are done; KV3's tiled exact prefill kernel merged on
> 2026-10-18 (#102), after v0.23.0. KV7's first two steps (#57, #86) and KV8a on CUDA (#115) are merged; KV8a on Metal is on
> branch `kv8a-metal-2026-10`. KV4 has not started, and KV5, the rest of KV7, KV8, KV10 and KV11 wait on it. **Decided by
> the owner on 2026-10-21:** KV4 goes first among the CUDA KV tracks. KV precision's phase 1 (f16 for the CUDA families that
> refuse it) folds into KV4b, and its phase 2 (int8, B39) waits for KV4's pools, so no KV kernel is written twice. KV9
> closes per its step 0: host blocks would save 405 MiB of a 2,159 MiB four-session footprint at 6K on the 0.5B; reopen it
> on a measured CPU case well beyond that. KV3's WebGPU half (`origin/kv3-webgpu-2026-10`) is rebased and merged. The
> Gemma 3 GEMM fidelity gate runs by day on nobara (G46 in `docs/queue-correctness.md`). I7, the default context toward
> 32,768, moves here from the integration doc, into KV5. **Stale below:** KV-pre is merged (64e13303), not in flight; KV8a
> is built on CUDA, not a sketch; KV7 has two steps merged, not one; KV9's step 0 exists (`docs/measurements/kv9-2026-10/`);
> KV11's first measurement was made and parked on 2026-09-22; KV3 is CUDA and Metal until its WebGPU half merges. A split
> of this doc is proposed for the next sweep: the finished KV0–KV3 record to `docs/completed/`, a slim live doc for KV4–KV11,
> keeping §5's numbering, which code comments cite.

**Ledger, 2026-10-21.** One row per item, the current state; it supersedes the status notes below where they
disagree, and is updated in the same commit as the work. `python3 scripts/task_docs.py check` reads it.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| KV-pre | done | — | 64e13303 (as I8) |
| KV0, KV0-H | done | — | the kv0 tile-census, invariance and traffic records; kv0h-hybrid-traffic-2026-10.md |
| KV1 | done | — | f08b47ca; kv1-2026-10-13.md |
| KV2 | done | — | 772b1fb2 (#26); G5-C PASS, kv2-cuda-2026-10-15.md §7 |
| KV2-FU1 | done | — | built on branch `kv2-fu1-2026-10`, merged through `kv2-fu1-merge-2026-10` (gates re-run green on that merge with main, 2026-10-21); `TestKV2FU1_ownRollbackScheduleIndependence`, `TestKV2FU1_cpuOwnRollback`; §6, KV2 follow-up 1, readings |
| KV2-FU2 | done | — | 11b29ba5 (#37) |
| KV3 | done | — (CUDA and Metal) | 356fa99a (#14), in v0.23.0; 973527ef (#102) |
| KV3-WebGPU | done | — | d555247c, 49898b2f; kv3-webgpu-2026-10-13.md (day gates re-read on the merged tree 2026-10-21: price x1.040 to x1.056, 1.65 GB whole vs 0.180 GB per chunk) |
| KV3-E4B | open | nobara: read the registered E4B cold-TTFT run | kv3-e4b-cold-ttft-2026-10-19 |
| KV3-GEMM | moved | G46 in `docs/queue-correctness.md`: run by day on nobara (owner, 2026-10-21) | — |
| KV3-S1 | open | nobara: `glu_quant_batched` writes in place (a variant without `__restrict__`, a PTX rebuild, `PrefillScratchBytes` with it); S–M | KV3, "Inside a pass", item 1 |
| KV3-S2 | blocked | KV3-S1, and only if the KV budget is at the margin (the doc's own trigger) | KV3, item 2 |
| KV4 | active | nobara: gates registered 2026-10-21; KV4a then KV4b. Pool = the VRAM main gives KV at the same flags (D-KV4-1 (a)); only a borrower past its share is stopped, as at the context cap (D-KV4-2) (owner, 2026-10-21); L | §6; `docs/measurements/kv4-cuda-2026-10-21.md` |
| KV4a | active | nobara: built (KV4a-1 to -3), draft PR #138; G1a (with its registered plants red), G1b (tiny and real assets), G1c, G3a to G3d (G3b served included), G4 and TTFT pass; MC3 0.973 to 0.983 and Gemma-3-1B decode ambiguous: shipped on the capacity arm with the ~2% cost at four clients recorded, MC3 on decode's bands (D-KV4-3 (b), owner, 2026-10-22); G1d's heavy partitions read (three failures were KV4a's and are fixed, one fails on `main` too) | `kv4-cuda-2026-10-21.md` §3, §5 |
| KV4b | active | nobara, branch `kv4b-2026-10-22`: f16 delivered on request per family group, `auto` per group on G4 PASS (D7). G4 keeps both cells: K=3,900 by day with ten prompts, K=8,000 on the night queue at four prompts (D-KV4-4 (b), owner, 2026-10-22); tier 0 re-run alone under the exclusive timing lock (owner, 2026-10-22) | `kv4-cuda-2026-10-21.md` §2 G5; `docs/tasks/task-kv-precision-everywhere-2026-10.md` |
| KV5 | blocked | KV4; takes I7, the default context toward 32,768 (owner, 2026-10-21) | §6 |
| KV6 | done | — (WebGPU keeps slots, D3) | §8 |
| KV7-s1, KV7-s2 | done | — | 60cfcba7 (#57); 671b54bb (#86) |
| KV7-rest | blocked | KV4: the disk tier, promotion, the block host pool; and D5's step 3, releasing a running conversation so it re-admits through the index (moved here from KV4 by D-KV4-2, owner, 2026-10-21) | §6; `docs/measurements/kv4-cuda-2026-10-21.md` §5 |
| KV8 | blocked | KV4 (KV5 for Metal) | §6 |
| KV8a-CUDA | done | — | fac209c2 (#115) |
| KV8a-CUDA-cc28k | done | — | the 28K Claude Code chain on CUDA at main `799b98df`: PASS, 1 tested, 0 of 7 differ (`docs/measurements/kv8a-2026-10/README.md`, "The 28K Claude Code chain on CUDA") |
| KV8a-Metal | done | — (owner, 2026-10-22: M2(b) graded on its 20K cells; the 28K chain is past a 16 GB Mac's swap guard and ran on CUDA instead, KV8a-CUDA-cc28k) | branch `kv8a-metal-2026-10`; M1–M5 PASS, `docs/measurements/kv8a-metal-2026-10/README.md` § "The grade" |
| KV9 | killed | closed per step 0 (owner, 2026-10-21); reopen on a measured CPU case where host blocks save well beyond 405 MiB | `docs/measurements/kv9-2026-10/` |
| KV10 | blocked | KV4, KV5 | §6 |
| KV11 | blocked | KV4 and KV7; its KV-share trade curve needs no pool and can run now (either box, M) | §6; lead4-perlayer-hitrate-2026-09-22.md |
| §5.16 | blocked | KV4's build list: the reuse-source field, the pool banner, the check row | §5.16 |
| §12 | open | desk: `docs/ARCHITECTURE.md`'s KV sections for KV7 and KV8a | §12 |
| CPU-ring | open | needs an owner: the CPU ring's one-position rollback, left as a finding in the prefix-reuse record §3 | from the archived windowed-KV doc |
| D1–D8, Q1–Q7 | done | — | §8 |


> **2026-10-20: KV8a added, ahead of KV4.** KV0-H (`docs/measurements/kv0h-hybrid-traffic-2026-10.md`) drove opencode
> and Codex on Qwen3.5-9B (CUDA, nobara). 29 of its 69 follow-up requests diverged inside the previous reply, where today's
> exact-extension rule re-prefills the whole prompt. Its registered rule pulls an end-of-prompt checkpoint ahead of KV4
> (71.7% of the prompt tokens processed would be skipped, 34.2 s median per affected turn; both projections). KV8a (§6)
> is that checkpoint on today's single slot; KV8 keeps the rest. Parking was not measured (no C requests) and stays in
> KV7 and KV8. Added on the owner's word ("yes write it") after the KV0-H readings.

> **Status: SCOPED 2026-10-13. KV0 DONE 2026-10-13 (§6, KV0). KV1 DONE 2026-10-14: every gate read PASS
> (`docs/measurements/kv1-2026-10-13.md`), merged to main. KV2 DONE 2026-10-15: the CUDA and Metal halves merged
> (`772b1fb2`, PR #26); G5-C, the sharing gate on nobara, PASS (`docs/measurements/kv2-cuda-2026-10-15.md`); the donor
> choice (follow-up 2) merged 2026-10-15 (#37). KV3 DONE: shipped in v0.23.0 (2026-10-15), with the rollback-budget fix
> (CUDA's windowed KV compacts in 64-token steps, so a client's rollback keeps at least 448 tokens).** Written on the owner's ask after a Cowork conversation the same day ("seems
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
> llama, mistral, qwen2, qwen3 and phi3 (`decoder/kvresolve.go:CUDAKVF16Decline` lists what keeps f32; owner override,
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
  of it system prompt and tool schemas. KV0's traffic study (`docs/measurements/kv0-traffic-2026-10-13.md`) found that 90% of it
  is identical across conversations *except* for a per-conversation hash in its first line, which ends a prefix match at token
  22 until the harness setting `CLAUDE_CODE_ATTRIBUTION_HEADER=0` (or a server-side normalisation) removes it.
- A parked conversation resumes without a re-prefill.
- Hybrids (Qwen3.5/3.6/3.8 and the other recurrent families) get prefix reuse and more than one conversation.
- CUDA gets f16 KV pools and extends f16 to the remaining families.

**How.** Twelve phases, each shippable alone. At the same KV precision, each is bit-identical to the tree before it.
- KV1–KV3 change bookkeeping, copies and admission only. They deliver cross-conversation reuse and fix Metal's long-prompt
  413 without touching an attention kernel.
- KV4–KV5 are the kernel work.

| phase | what | backends | size | needs |
|---|---|---|---|---|
| KV-pre | long-prompt prefill in blocks, admission per block (**in flight**, not this doc's) | Metal | S | — |
| KV0 | prior art, block size, invariance and traffic measurements, owner decisions (**done**) | — | S | — |
| KV1 | the manager over today's slots: logical blocks, hashed index, no storage change | CUDA, Metal, WebGPU | M | KV0 |
| KV2 | cross-conversation reuse by device copy between slots (**done** 2026-10-15: CUDA and Metal, `772b1fb2`; G5-C PASS) | CUDA, Metal, WebGPU | S–M | KV1 |
| KV3 | prefill and admission in blocks everywhere, CUDA 12,288-key fix, 16-row rule (absorbs KV-pre; **done**: shipped in v0.23.0, with the rollback-budget fix) | all resident | S | KV0 |
| KV4 | paged storage and block-table kernels; f16 pools | CUDA | L | KV2, KV3 |
| KV5 | paged storage | Metal | L | KV4 |
| KV6 | keep slots with KV2's copies (**decided**, D3) | WebGPU | — | KV2 |
| KV7 | tiers: host and disk, parking and resume | CUDA first, then all | M–L | KV4 |
| KV8 | recurrent state pool and checkpoints | CUDA, Metal | L | KV4 (KV5 for Metal) |
| KV8a | an end-of-prompt recurrent checkpoint on today's single slot (**added 2026-10-20**, pulled ahead of KV4 by KV0-H's rule) | CUDA, then Metal | S–M | KV0-H |
| KV9 | the CPU path onto the manager (deferred by D2: CPU joins through KV2's copies) | CPU | M–L | KV2, D2 |
| KV10 | long context and lower-precision blocks | all | M | KV4/KV5 |
| KV11 | one residency manager for KV blocks and MoE experts (added 2026-10-13) | CUDA, then Metal | L | KV4, KV7 |

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
   compute from one function, and the never-swap rules hold on every gate (`docs/completed/task-never-swap-2026-09.md`).
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
| sliding window W | Gemma 2/3/4 local layers, Mistral, Cohere2, gpt-oss, Mellum, Phi-3 | CPU ring, except gpt-oss: its sink attention is not on the batched kernel `decoder/arch.go:Architecture.kvRingsOK` admits, so its sliding layers keep append-forever f32 KV on the CPU (corrected 2026-10-22, kernel survey KS-F2); CUDA windowed view (on in the CLIs); Metal, WebGPU full + mask | window group: blocks behind the window are released |
| chunked attention C | Llama 4 local layers (`attention_chunk_size`) | CPU (`attnChunkStart`) | chunk group: blocks before the current chunk are released |
| NoPE layers | Cohere2 globals, Llama 4 | all | nothing extra: K is stored as computed |
| MLA latent | DeepSeek V2/V3, Kimi K2, Ling 3.0's MLA layers | CPU, CUDA (one latent buffer), WebGPU | latent group (a row is the latent width) |
| KV-shared layers | Gemma 4 E2B/E4B | CPU, CUDA, Metal, WebGPU | no group of their own: they read their source's table |
| K = V layers | Gemma 4 globals (`attention_k_eq_v`) | CUDA copies k into v (audit R-23) | group flag: store K once (halves those layers' KV); measured before it is on |
| attention sinks | gpt-oss | all GPU | nothing (a learned logit, not KV) |
| recurrent state | DeltaNet (Qwen3.5/3.6/3.8, Qwen3-Next), Mamba-2 (Granite-4.0-H, Nemotron-H), short conv (LFM2), KDA (Ling 3.0) | CPU all; DeltaNet on CUDA, Metal, WebGPU; Mamba-2 on WebGPU | state pool and checkpoints (§5.7) |
| image blocks, bidirectional | Gemma 3/4 vision | CPU and the residents | spans are atomic: a hit never ends inside one |
| m-RoPE positions | Qwen2.5-VL, Qwen3-VL, Qwen3.5 with images | CPU and the residents | the image hash is in the key; the decode delta lives in sequence metadata (`SeqView.MRoPEDelta`, §5.13) |
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
| precision | f32, or int8 with a scale per position and KV head | **f16 by default for llama, mistral, qwen2, qwen3 and phi3, f32 otherwise** (`decoder/residency.go:Model.ResidentKVPrecision`, `decoder/kvresolve.go:CUDAKVF16Decline`); no int8, no scales | f16, or int8 (f32 compiled out) | f32, f16 or int8 |
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
- **CUDA's windowed KV.** Built 2026-10-10 and on in the CLIs (`docs/completed/task-cuda-windowed-kv-2026-10.md`). Its
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
  - **Block 0 is a reserved, zero-filled null block** in every pool, and every released window or chunk table entry and every
    unused tail entry points to it, never to `-1` or to a block that has been given to another sequence (vLLM's
    `remove_skipped_blocks` puts its null block in freed window slots, §11a). Metal's `attention_prefill_fused` and
    `attention_prefill_steel` read up to 31 and 15 keys *below* a sliding window's start and up to the next multiple of 8 past
    the last key, and mask the scores after the load (`docs/measurements/kv0-tile-census-2026-10-13.md` §5). A masked weight of
    0 times a NaN is NaN, so what those reads land on must be mapped and finite.
- **The kernel change** is one indirection: key k lives at `table[k / B]·B + k mod B`, applied **per key** (with the physical row
  offset widened to 64-bit `int64_t` / `size_t` where CUDA's `attention` and `mla_attn` compute `s * kvDim` as 32-bit `int` today,
  so a large pool cannot overflow 2 Gi elements). Every reader on every backend already addresses each key's row separately, so no
  kernel changes the grouping of its arithmetic. That is the bit-identity argument, and it is *not* "tiles are aligned to blocks":
  that holds for `attn_fused*` on full-attention layers only. Sliding-window readers, flash-decode and the split-KV chunks
  (`lo = winStart + sp·per`, `per` moving every few tokens), Metal's decode flash-attention blocks and prefill writes at an
  arbitrary `startPos` are unaligned by construction. What keeps the bits is that **every tile, partition and thread-to-key map
  stays defined in absolute key units**; re-cutting one at a block edge is a numerical change, and a per-tile base-pointer
  optimisation is bit-safe only for `attn_fused` on full layers. The KV4/KV5 gates test this rather than trust it.
- **Waste** is at most one partial block per sequence per group: B/2 positions on average, 0.10% of a 32K sequence's KV at B = 64
  on the 7B, 0.62 MiB (`block_arithmetic.py`).

### 5.4 Sequences and their lifecycle

A sequence is one conversation's KV: its tables, its position, its recurrent state slot, its image spans and m-RoPE delta,
its adapter, and its namespace.

| operation | what happens |
|---|---|
| **Admit** | the longest valid hit is shared (refcount +1); private blocks are allocated for the rest of the prompt, plus one block of headroom; hybrids restore a checkpoint (§5.7) |
| **Append** | decode and verify write into the private tail; a new private block is taken when the tail fills (or one block ahead for a device-chained decode, §5.13) |
| **Publish** | a full block becomes immutable and enters the index once the device work that wrote it has **completed** (fence), never on encode (vLLM publishes at schedule time and had to patch same-step hits on uncomputed Mamba state in PR #29387, §11a). This is rule 2's new form. A cancelled request still leaves its finished blocks reusable, because they are what those tokens produce. |
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
  - **The per-block extras** are the content hash, offset and length of every image span that touches the block, plus the
    **producer kernel class** of the block's rows (§5.6).
  - **The m-RoPE delta needs no key part of its own.** An earlier image shifts the rotary position of every later row
    (Qwen2.5-VL, Qwen3-VL, Qwen3.5), but the delta is a function of the earlier spans' grids, and the chain already carries
    every earlier span's content hash, offset and length: the same link means the same images in the same places, and so the
    same delta. A request recomputes its delta from its own prompt before any reuse decision, with no vision tower
    (`decoder/rope.go:mropeDelta`).
  - A block written by two classes, such as the end of a prompt and the start of its reply, records both and the row
    where they meet. It matches only the same mix.
- **H is SHA-256**, truncated for the map key. SHA-256 is vLLM's default hash since v0.10.2; its v0.8.5 doc estimated
  100–200 ns per token for the Python path, and an exploratory Go chain here measured 3–9 ns (§11a). The hash input needs a
  canonical byte encoding (vLLM's pickle default is not reproducible across versions, which is why `sha256_cbor` exists), which
  the disk tier (§5.8) depends on.
- **Only full blocks are indexed,** as in vLLM. A partial tail is private.
- **A lookup, across groups:**
  1. Take the longest chain of hits from block 0 in the full (or latent) group: `L_full`. A model with no such group (every
     layer windowed: Mistral, Phi-3) has nothing to walk there, and needs nothing kept in the pool.
     - A cold request computes `h_0 … h_{k−1}` from its own tokens, with its cold producer class.
     - A continuing or forked sequence cannot: its earlier blocks may carry the `decode` class or a mix at a turn boundary
       (§5.6 property 3), which its tokens do not determine. So the sequence keeps its own hash list from block 0 (32 B a
       block, 16 KB at 32K), and the pool keeps no tombstone for a block whose window data is released.
     - `L_full` is then the request's own block count, and step 2 does the work. SGLang's tombstones exist because it walks a
       tree from its root; a hash chain is looked up by value.
  2. Take the largest block-aligned `L ≤ L_full` at which every window and chunk group still holds the live blocks its layers
     read at `L` (`[L − ceil(W/B)·B, L)` for a window W), scanning from the right, as vLLM's `SlidingWindowManager` does (§11a).
  3. For a hybrid, take the deepest recurrent checkpoint at or below `L`.
  4. Apply the caps:
     - `L ≤ ⌊(n − 1)/B⌋·B` (rule 3: at least one token is recomputed for the seed logits);
     - for a hit on another sequence's blocks, when the requester's cold prefill class is fast (`n ≥ FastPrefillFloor`):
       **`n − L` is at least the backend's fast-pass minimum (16 rows on CUDA, so `L ≤ ⌊(n − 16)/B⌋·B`)**. Otherwise the rows
       the requester computes itself, `[L, n)`, run the exact kernels where its cold prefill would have run the fast ones
       (KV0's `split@985`), and the reply is not the cold one (§5.6 property 2). A sequence continuing its own rows keeps
       today's rule (property 3);
     - `L` never lands inside an image span.
  
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
     key, the chunker (KV3: a prefill never ends in a pass shorter than the resident's own minimum, 16 rows on CUDA) and the
     hit cap (§5.5 step 4): **a pass of 15 rows or fewer is exact class** whatever the prompt's length, so a fast-class
     prefill never cuts such a chunk or takes a hit from another sequence that leaves such a suffix.
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
     **Corrected 2026-10-20 by KV0-H:** a checkpoint at commit covers only what the live state already continues (a strict
     extension of the whole sequence). All 29 of KV0-H's B divergences sat inside the previous reply, 15 to 123 tokens in,
     because the harness sends the reply back other than as generated (opencode drops the think block). The checkpoint that
     serves the next turn is the one taken **where the prompt ends, before the reply** (KV8a). A checkpoint at commit pays
     only for a client that sends the reply back exactly, and the live state already serves that client;
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
  embedding/head-table precision (`Model.HeadTable()`, since `--embed-int4` changes `m.w.Embed` while `m.Quant()` stays `"int4"`),
  the backend, the GPU device and driver (for the disk tier), every operator knob that changes which kernel computes a row at
  a given position (a class boundary needs none: a row's class is in its block key), and a **numerics epoch** (the build's
  parity `deps_hash` or its version). A block written by another build, device or table precision is never trusted, because
  kernels and embeddings change numerics between them.
- **What it replaces:**
  - **at KV7:** the GPU parking gap (R-04's resident half);
  - **at KV9 (if D2 says so):** the CPU session LRU's `.giw-kv` cold tier. The v3 snapshot format stays readable for one
    release.

### 5.9 Precision

- **Per pool:** f32, f16 or i8. An int8 pool keeps one scale per position and KV head inside the block. Precision is in
  the root key, so blocks of different precisions never match.
- **CUDA's f16 is already the default** for llama, mistral, qwen2, qwen3 and phi3 (owner override 2026-10-13, ahead of this
  doc's KV4b); `--kv f32` keeps the reference. D7 is therefore decided for those families. KV4b extends f16 to f16 *pools* and to
  the families that still decline (`decoder/kvresolve.go:CUDAKVF16Decline`: windowed, MLA, MoE, pairwise RoPE, m-RoPE).
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
    the leftover, and the new rule keeps that behaviour unless a measurement says otherwise. KV11 replaces the fixed split
    with one budget for KV blocks and experts (§6).
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
  models get graphs back; today graphs are declined when windowed KV engages (`../completed/task-cuda-windowed-kv-2026-10.md` §7). A
  measured bonus, not a promise.

### 5.12 Speculation and drafters

- **Verify rows go into the private tail.** A rollback is `Truncate`.
- **Window and chunk groups keep rollback slack behind the window, sized for a client's rollback, not for a verify
  width.** One block (64 tokens) covers a speculative verify but not a conversation that drops or edits its last reply.
  - **The measurement (2026-10-14).** On CUDA's windowed KV (1,024 window + 512 slack), the base moved in full
    512-token jumps. So the budget for rolling back into a slot cycled between 0 and 512 tokens, depending on where the
    last decode landed, and any larger rollback went cold. Replaying that arithmetic predicted warm or cold for 59 of 59
    Codex requests from the I3 night. 35 of the 59 went cold, about 2,430 s of re-prefill.
  - **The rule chosen for CUDA (the ring fix, built 2026-10-14; `kvwindow-ring-2026-10`, speed guard PASS):**
    - The base compacts in 64-token steps, so the budget never drops below 448. It costs no VRAM; the extra copying is
      about 0.1% of a token's time on Mellum2.1.
    - Extra rollback, up to 1,024, comes only from VRAM left after planning. It never costs residency, context or a slot.
      An expert-cache load gets it too: since 2026-10-19 the grant is taken just before the expert cache is sized, out
      of the cache and never below its minimum (P34, cause 2). Before that, an expert-cache load got none.
    - Limit: right after a prefill pass of 448 rows or more, an edit inside that pass has less than 448 until decode
      catches up. Dropping the last reply is always covered.
  - **The paged design keeps the same guarantee.** A window group retains at least the rollback budget behind the
    window, not one block. KV2's registration (`kv2-2026-10`) is amended to match, and KV4 sizes window groups with it.
  - **What removes most rollback in the first place** is the client sending its reasoning back. `/v1/responses` returns
    reasoning items and accepts them back (PR #12). The ring rule covers every other client that edits late in its
    prompt.
- **Drafters' own caches stay outside the pool** in this doc: `cuda/drafter.go`, EAGLE, and the block drafters.

### 5.13 The backend contract (sketch; names provisional)

```go
// The decoder owns this metadata; backends never see hashes, refcounts or tiers.
type SeqView struct {
	Tables     [][]int32 // per group: block ids, logical order
	Pos        int
	MRoPEDelta int       // ropePos - pos past the prompt's image spans (0 for text); the caller's, from its own prompt
	StateSlot  int       // -1: no recurrent state
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
- `ResidentForward` (`Forward`, `ForwardN`, `UploadKV`) and its extensions (`Prefiller.PrefillLast`,
  `ResidentPrefillKV.ForwardNoLogits`, `ResidentMRoPE.ForwardMRoPE`, `ResidentGreedy.ForwardArgmax`,
  `ResidentTopK.ForwardTopK`, `ResidentSample.ForwardSample`, `ResidentHiddenLast.HiddenLast`,
  `ResidentResidualAll.ResidualAll`, and `ResidentBatchStepper.StepBatch`'s `ResidentBatchSeq`) take a `SeqView` (or bind it
  for the step) instead of acting on "the bound slot".
- **Device-chained decode** (`decoder/residency.go:ResidentGreedyChain`, `decoder/residency.go:ResidentSampleChain`): a chain
  queues the next token's forward on the device before the host has seen the token. Under paged storage (KV4/KV5), `Append` must either keep
  at least one uncommitted lookahead block mapped in the device's block table before `pos` reaches the last row of the current
  block (`(pos + 1) % B == 0`), or close and re-open the chain at block boundaries so a queued lookahead forward never writes
  past the end of the mapped table.
- `TruncateTo` and `Reset` become manager operations, which they already nearly are: no-ops on a positional cache.
- `ResidentCapped`'s cap comes from the manager.
- `ResidentReusableKV`'s windowed floor becomes the window group's retention policy.

**A backend not yet paged** keeps its slots behind a slot adapter. Logical block `i` of the sequence bound to slot `s` lives
at `(s, i·B)`, so KV1–KV3 run unchanged on every backend. **CUDA's windowed layers (`cuda/kvwindow.go`) need care in KV2,
not in KV1.** A windowed layer of slot `s` holds only its recent rows, at a shifted view (`kvBases[s]`): a kernel still
addresses position `p` as `p`, so the adapter's statement holds, but rows behind the base are gone. KV1 copies nothing and
asks the bound slot what it still holds (`ResidentReusableKV`). KV2's copy from another slot has two new needs there, both
the backend's: the same question answered for a slot that is not bound, and a copy that reads and writes through each
slot's own base. KV4 replaces `kvwindow.go` with the window group's block pool.

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
- `docs/completed/task-cpu-kv-f16-2026-10.md` (CPU f16 by default) is independent of both.

### 5.15 Sharing scope and the timing side channel

- **The risk.** A shared prefix makes time-to-first-token observable: a prompt someone else already sent answers faster,
  and repeated probes can reconstruct parts of it (Chu et al., "Selective KV-Cache Sharing to Mitigate Timing
  Side-Channels in LLM Inference", arXiv 2508.08438).
- **goinfer is single-user by positioning,** but `serve` can face a LAN.
- **The design:**
  - the root key carries a **namespace**;
  - by default (D4) a server has one namespace, and one per API key when keys are configured;
  - `-kv-share off` turns cross-sequence sharing off and keeps per-conversation continuation.
- **Known limit, stated and accepted (owner, 2026-10-14).** A continuation is a request that extends a slot's whole
  sequence, giving up less than one block. So under `-kv-share off`, or across namespaces, a request still takes a
  slot's rows when its prompt matches everything the slot holds except up to 63 trailing tokens. A client that
  already knows all but the last 63 tokens of another conversation can therefore learn from its time to first token
  how far its guess matched. It learns nothing about a conversation it cannot already reproduce up to that tail.
  `docs/server.md` states the same limit.

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

### KV0 — Prior art, block size, two invariance measurements, the traffic study, owner decisions (no product code) — DONE 2026-10-13

Size S. Done in one session, on nobara; test files and measurement records only (the branch `kv0-2026-10`).

1. **Prior art: done.** §11a, from the sources (each fact marked read or inferred, seven lines of §5 and §11 corrected).
2. **Tile census: done** (`docs/measurements/kv0-tile-census-2026-10-13.md`, per-kernel tables beside it).
   - Every reader on every backend already addresses each key separately, and no kernel loads a contiguous key tile that a block
     boundary could split except CUDA's `attn_fused*` at 64 keys (full-attention layers) and Metal's `attention_prefill_fused`
     (8 keys, 32-key groups). **B must be a multiple of 64 and nothing more.**
   - **B = 64** (§5.3), **option A, pools per group** (§5.2). The block arithmetic for Qwen2.5-7B, Gemma-3-4B and Qwen3.6-35B-A3B is in
     the record: at B = 64 and f16, 3.67 MB a block and 512 blocks per 32K sequence for the 7B; Gemma's window group holds 17
     blocks (123 MiB) beside 512 global blocks (640 MiB); Qwen3.6's 10 attention layers are 1.31 MB a block and its recurrent
     state is a 62.8 MiB pool on top.
   - §4's kernel counts, the Metal bind limit and the bit-identity argument of §5.3 were wrong or too strong and are corrected.
3. **Invariance on CUDA and WebGPU: done** (`docs/measurements/kv0-invariance-2026-10-13.md`; rule and predictions registered before
   the runs).
   - **Chunk invariance holds on both backends** (widths 64 to 512, ragged tails, tiny fixture and the real 1.5B at int4, CUDA at f32
     and f16): 0 of 14,336,000 K/V elements and 0 of 151,936 logits differ.
   - **A split at a block-aligned offset is invariant on CUDA exactly when the first call was in the class the whole prompt
     would use**: splits at 512, 768 and 984 are bit-identical; at 256 and 511 (first call under the 512-token floor) and at 985 (second
     pass of 15 rows) they are not, starting at position 0, 0 and 985, and `chunk = 496` (an 8-row tail) from 992. The class
     rule: fast kernels iff the pass has 16 rows or more and its prompt length is 512 or more, the length being the pass's own end
     for a prefill in two calls. **Added to the key (§5.5, §5.6): the 16-row rule**, which `declineShortLeadReuse` does not state.
   - **WebGPU: invariant in every arm** (13 arms, both fixtures); no kernel class depends on the prompt.
   - **The continuation effect, measured on CUDA** (§5.6 property 3): decode-written rows differ from fast-prefill rows by 0.7% rms
     (3.7 to 4.6 at the largest activation of 360); next-token logits cosine 0.992 to 0.999, argmax equal in 49 of 50 trials; the exact
     class equals decode to the bit.
   - No backend measured needs a chunk-invariance fix.
4. **Traffic study: done, with deviations** (`docs/measurements/kv0-traffic-2026-10-13.md`). Qwen3-4B replaced the brief's models
   (the Coder-7B gave 14 of 14 single-turn sessions; gpt-oss-20b does not resolve for a 20k first request on this card: the batched
   prefill declines past 12,288 keys). 13 study sessions, 101 requests, token counts validated against the server's own log.
   - Within a conversation a request strictly extends its previous request (opencode 40 of 41, Codex 26 of 26); new conversations
     of opencode and Codex share 99.7% to 100% of their first request with an earlier conversation's.
   - **Claude Code's per-conversation hash in its first line ends the shared prefix at token 22**; with
     `CLAUDE_CODE_ATTRIBUTION_HEADER=0` the first requests of different conversations share 18,213 of about 20,154 tokens. Its
     date line sits at the end of the first request's preamble, not per turn.
   - Today's single slot loses the preamble to opencode's title request on every new conversation, and Claude Code thread
     switches re-prefill 8,000 to 17,000 tokens each (78% of its request time).
5. **Owner decisions D1–D8 and Q1–Q6** (§8): all decided and answered on 2026-10-13.

**Decision rule, read against what was registered** (`kv0-traffic-2026-10-13.md` §1 and §2.4):
- KV1–KV3 go ahead (they add no paged attention). The traffic study supports them: the sharing is real for opencode and Codex, and
  for Claude Code once its header is out of the way.
- **KV4: the capacity arm clears, the sharing arm clears conditionally.** The 7B at default flags is 4 conversations × 11,784
  positions; one conversation could have the model's 32,768 from the same memory (2.78 ×). The saving per new Claude Code
  conversation is 0 s as shipped and 29.7 s on the Mac (4.5 s at the 1.5B's batched nobara rate) with the header off. Two cautions
  go with the verdict: the sharing arm is delivered by KV2's copy and not by paging, and the capacity arm's headline context is
  reachable today with `--ctx 32768 --kv-sessions 1` at the price of the other slots. The owner's call whether they change the
  verdict.

### KV1 — The manager over today's slots (no storage change) — DONE 2026-10-14, merged to main (`f08b47ca`)

Branch `kv1-2026-10`, merged. Gates and readings: `docs/measurements/kv1-2026-10-13.md` (registered before the code; G2
and G5b amended before their runs). **Every gate read and passing:** identity by day on both boxes, the windowed tests at
awkward offsets on the exact class, the oracle property test, the wrong-hit mutations, the manager's own cost, the
fast-class windowed measurement (which measures, and does not gate KV1), the served speed gate on both boxes and the
Metal device gate.

- **Built:**
  - `internal/kvmgr`, importing nothing of goinfer's: the geometry types; the key (a root of model, namespace, adapter
    and KV precision; a SHA-256 chain per block over its tokens, the image spans that touch it and the class of each
    row); blocks of 64 with reference counts, publish, the index and the LRU of blocks nobody references; sequences
    with admit, append, publish, truncate, fork and release; and the slot adapter (`kvmgr.Slots`), where logical block
    i of the sequence in slot s is at `(s, 64i)`.
  - `decoder/kvgeometry.go`: `Architecture.KVGeometry` derives the groups (full, window, chunk, latent; K = V; the
    layers that read another layer's KV) and the recurrent state from the architecture, and agrees with `kvDimAt` and
    `kvPositionsAt` on ten family shapes. The manager's types are re-exported under `decoder.KV*` names for the backends.
  - `decoder/resident_reuse.go` is now calls into the manager. The four rules are `kvmgr.Slots.Score`'s and the root's;
    the slot choice is `kvmgr.PickSlot`; the short-prefix guard is `kvmgr.ClassMap.DeclineShortLead` and states the
    16-row rule; a completed generation publishes its full blocks (`kvmgr.Slots.Commit`), and a forget takes them out of
    the index at once.
  - `decoder/cachestate.go`: publish, share and evict are lifecycle paths, with a declared cell for each of the 14 kinds
    and a test that checks each cell against the reuse code.
  - `docs/ARCHITECTURE.md`, "The KV cache": corrected (CUDA's windowed layers, the fit-by-default contexts) and extended
    with the manager.
- **Not changed:** serve's admission, every backend kernel and allocation, and what a request reuses.
- **Choices made in the build** (each is in the record's §1 or §3):
  - **The names the tests hold are kept.** `residentReuseLen`, `residentAcquire`, `residentCommitIDs`,
    `residentForgetIDs`, `pickResidentSlot`, `declineShortLeadReuse` and the bound sequence's three fields (`resIDs`,
    `resIDsLora`, `resImgBlocks`) remain, as calls into the manager and as its record of the bound slot, because the
    identity gate says the existing tests pass unchanged and twelve of their sites assign those fields. The parked slots
    (`resSlots`, `resCur`, `resTick`) are gone.
  - **One chain for every group** (the question §11a left for KV1): the group is not hashed into the chain. Under slots
    every group's blocks are published together; the group id joins the map key when groups get their own pools (KV4).
  - **The adapter's id is its load identity**, as rule 4's pointer comparison was. A content hash is KV7's need.
  - **A forget unindexes the whole slot**, as rule 2 does today. §5.4's "a cancelled request still leaves its finished
    blocks reusable" is not adopted here: it would change what a request reuses.
  - **A sequence is reused only by continuing it in its own slot, at token granularity.** The index is built and
    published to, and nothing takes a hit from it yet. In production no block has a reference count above zero until
    KV2 pins a copy's source; the counts are exercised by the oracle test through the pool API.
- **Left for KV2, which must settle each before it takes another sequence's rows:**
  - **The class of a row is recorded per pass the decoder sees.** CUDA cuts a long prefill into 512-row passes
    internally, and a last pass of 15 rows or fewer is exact class (KV0's `chunk=496` arm). The decoder records the
    whole call as one pass, so such a tail is recorded as fast. KV3 moves the 16-row rule into the chunker, which
    removes that tail; KV2's gate should check the recorded classes against launch counts, as KV0's test does.
  - **Rows written by a path that does not go through `residentAcquireSlot`** (the image prefills, which compute on the
    CPU and upload) are recorded as class unknown, which matches no other request. That is safe and loses the reuse.
  - **The namespace is in the key and nothing sets it.** D4's defaults (one per API key; off beyond loopback without
    keys) need a namespace on the request, from serve.
  - **Window groups.** The geometry knows which blocks a window group holds (`KVGroup.FirstBlock`), but the adapter
    leaves the answer to the backend (`ResidentReusableKV`), as today.
  - **The slot choice is still by slot** (`PickSlot`'s last-use tick). The block LRU is maintained beside it and is
    what KV2 switches to.
  - **A hit from another sequence must leave a suffix the fast kernels will run**: at least 16 rows on CUDA when the
    requester's cold prefill is fast class (KV0's `split@985`; the KV3 session's request). `Manager.Admit` caps a hit
    at n − 1 only, and the cap needs the requester's class map.
  - **CUDA's windowed layers.** A slot that is not bound must be able to say what it still holds, and a copy between
    slots must go through each slot's shifted view. Both are the backend's; KV1 copies nothing.
  - **A windowed layer on the fast kernels is invariant only to pass starts that are multiples of 64** (measured,
    the record's G2b). A hit from another sequence is block-aligned, so the pass after it is safe. A continuation's
    pass starts anywhere, so on a model with a sliding window its fast rows are recorded with no producer named and
    the index never returns them. A row with no producer named matches nothing.
  - **An image span with no content hash matches nothing in the index** (built: a lookup stops at the first block it
    touches). KV2 should keep that when it adds the copy.
- **Size:** M. **Closes:** nothing visible. It is the foundation.

### KV2 — Cross-conversation reuse by copy (still slot storage) — DONE 2026-10-15

**Done 2026-10-15.** The CUDA and Metal halves merged in `772b1fb2` (PR #26). G5-C, the sharing gate on nobara's 1.5B at
`-ctx 24576 -kv-sessions 2` (`docs/measurements/kv2-cuda-2026-10-15.md`):
- **T-idle: PASS 1.926** (1.915 to 1.946 by block).
- **T-busy: delivers 0.926 of the projected saving,** inside [0.80, 1.05]. The median time to first token went from
  12.29 s on `main` to 2.05 s, a saving of 10.25 s, on an 18,112-row hit in every block.

The donor choice (follow-up 2 below) merged the same day (#37).

Branch `kv2-2026-10`. Gates, and the choices made before the code: `docs/measurements/kv2-2026-10-14.md`. The Mac
builds the decoder, serve, Metal, WebGPU and the CPU; nobara builds CUDA from `docs/prompts/nobara-kv2-cuda-2026-10.md`.
KV2 merges when every arm on both boxes is read.

- **Build:**
  - one optional resident interface copies rows `[start, end)` of every layer from one slot to another; a backend that
    cannot returns a named decline and the rows are prefilled;
  - a hit that lives in another slot is copied into the request's slot and the rest is prefilled;
  - a prompt's blocks are published when its prefill ends, so a conversation arriving while another generates can
    take its preamble;
  - a request no longer takes another conversation's slot in place when it would cut a block or more of it;
  - the namespace comes from serve (D4), and `-kv-share off` is added;
  - slots are reassigned by block LRU instead of slot LRU;
  - the CPU's sessions copy last (D2).
  - The copy is small next to the prefill it saves. A 19,692-token preamble on the 7B at f32 (112 KB a position) is about
    2.2 GB of device copy, around 10 ms on the 8 GB card, against seconds of prefill.
- **Sharing keys on token ids, so equal text with different ids does not share (owner, 2026-10-15: accepted).** The
  block key chains the token ids (§2, rule 1: ids, never text). Under the id round trip's proposal 2, a reply's generated
  ids come back in a signed opaque field (Responses' `encrypted_content`, Anthropic's `thinking.signature`) and are used
  in place of re-tokenizing the echoed text (PR #24's step-0 record, `id-roundtrip-step0-2026-10-15.md` on branch
  `id-roundtrip-step0-2026-10`). Two conversations whose prompts carry the same text but different ids (one carries a
  reply's non-canonical generated ids, the other re-tokenizes the same text canonically, or carries ids from a different
  generation) hash differently from the first differing id, so a copy stops at the block before it. That is acceptable:
  - what crosses conversations is the prefill-class preamble (system prompt, tools, environment), which no carried id
    touches and which both tokenize identically;
  - a reply's rows are decode class, which property 3 (§5.6) never shares across conversations whatever the ids;
  - matching on text would need the two tokenizations to give the same KV, which they do not (different ids are
    different inputs), so it would break sharing invariance (§5.6, property 2).
  KV2's gates need no change for it: the CUDA registration (`docs/measurements/kv2-cuda-2026-10-15.md` §3) states it.
- **Gates (registered):**
  - G0: split invariance on the Mac's backends.
  - G1: identity.
  - G2: the copy exact, per backend.
  - G3: schedule independence (hard).
  - G4: KV1's carry-overs, each with a planted defect.
  - G5: the sharing gate, which decides KV2 (Q3). The busy-source case is graded against KV0's projection, and the
    idle case against `main`.
  - G6: a lone request at 0.99 against `main`, on Metal and CUDA.
  - G7: copy and admission cost, reported.
- **Size:** S–M. **Closes:** most of §1.2, without kernel work, on CUDA, Metal and WebGPU.
- **Follow-up 1, after KV2 merges (owner, 2026-10-14; gate registered 2026-10-21, before the code, branch
  `kv2-fu1-2026-10`).** With the first-prompt anchor (`kvmgr.Slots.FirstPrompt`), an own rollback that gives up more
  than it keeps could truncate in place instead of re-prefilling.
  - Main's keep-at-least-as-much-as-it-overwrites rule (`kvmgr.PickSlot`; on the CPU, `pickSession`) exists only
    because main could not tell a conversation's own request from another's.
  - The item: drop that rule for a request that keeps the slot's whole first prompt, on the GPU and the CPU.
  - Gate it like G3: same reply in every schedule, and replies unchanged against KV2's merge.
  - **The change, as registered.**
    - GPU: `kvmgr.PickSlotFor` takes the best-scoring slot in place, with no keep-versus-overwrite test and no
      allowance, when the request keeps that slot's whole first prompt but its last token (its existing own test).
      Every other request keeps main's rule and KV2's allowance.
    - CPU: serve's session LRU records each session's first prompt length, kept by a request that keeps all of it but
      its last token and restarted by any other, as `kvmgr.Slots.FirstPrompt` is. `pickSession`'s room rule (a fresh
      session when reuse would discard more than it keeps) does not apply to such a request. A session loaded from
      disk or faulted back from the cold tier has no length until its next request, so it keeps main's rule.
    - Unchanged: busy slots and sessions, the donor choice and copies, the recurrent families' scoring, and
      `Model.residentReuseFloor` (CUDA's windowed ring still floors an in-place rollback past its budget).
  - **Gate FU1-G3. Tier: day. Testing tier 0 (counters and row identity on a fake and a tiny fixture).**
    - *Instrument:* G3's decoder instrument (`TestKV2_scheduleIndependence`'s copying fake) with a second kernel
      class. Its batched pass (`PrefillLast`) writes rows that differ from decode's for the same input, as Metal's and
      CUDA's do. It reports a kernel floor of 1 and an exact tail, so the decoder records batched rows fast and decode
      rows exact, and a copy takes only batched rows. KV2's fake has one class, where a re-prefilled reply equals an
      in-place one, so a reply check on it cannot see this defect.
    - *Conversations:* three, sharing KV2's 327-token preamble, each with a 397-token first prompt, greedy, 12 tokens a
      turn. Five turns each:
      1. the first prompt;
      2. its reply and 500 tokens;
      3. a rollback to the first prompt and its reply plus 2 new tokens: it keeps 409 rows and gives up more than 500;
      4. that turn's reply and 500 tokens;
      5. a second rollback to the same point with other tokens, which checks that the anchor survives an in-place
         rollback.
    - *Schedules,* at 1, 2 and 4 slots: (a) round robin by turn; (b) the same, conversations reversed; (c) an
      interleaving in which each rollback follows its own long turn while the others take slots; (d) each
      conversation run through before the next; and 20 seeded random interleavings.
    - *Reference, G3's (§5.6):* a turn is its conversation continuing when the slot its previous turn used has served
      no other turn since. Its reference is that conversation's lineage, replayed alone on a fresh one-slot model.
      Any other turn's reference is its prompt prefilled cold, and the lineage restarts from it.
    - *Rule, all four or red:*
      1. every emitted token equals the reference's (the fake's logits are one-hot, so equal tokens are equal logits);
      2. after every turn, the bound slot's rows equal the reference's, read back from the fake;
      3. every continuing turn binds its previous slot, copies nothing (`PrefillShared` 0, no `CopyKV` call) and reuses
         its common prefix with that slot's sequence, capped at n − 1;
      4. coverage: at 4 slots every schedule has at least one rollback that continues in place giving up more than it
         keeps; at 2 slots the schedules together have at least one.
    - **Amended after the first run, before any plant: rule 3 over-claimed.**
      - *As registered,* it named every continuing turn. 34 turns failed it, in 16 schedules, all at 2 slots
        (`docs/measurements/kv2-2026-10-14/fu1-g3-first-run-rule3.txt`). Rules 1, 2 and 4 held on every turn.
      - *Every failing turn's conversation had lost its slot at an earlier turn,* which re-prefilled into another slot.
        `kvmgr.Slots.Commit` then records that turn's prompt as the slot's first prompt. A later rollback to the first
        turn's reply diverges inside it, so it is held to main's rule. That is the item as defined ("keeps the slot's
        whole first prompt"); the anchor cannot know the conversation's first prompt once its slot is gone.
      - *Rule 3 now reads:* every continuing turn that keeps its lineage's first prompt but its last token binds its
        previous slot, copies nothing and reuses its common prefix. A lineage starts at the conversation's first turn
        or at its last turn that could not continue. Rule 4 counts those turns.
      - Reported as a finding: a conversation that loses its slot once loses the in-place rollback to any point before
        the turn that lost it.
  - **Readings, 2026-10-21 (Mac, by day, branch `kv2-fu1-2026-10`): every arm PASS, every plant red.**
    - *Decoder* (`TestKV2FU1_ownRollbackScheduleIndependence`): 72 of 72 subtests pass, 24 schedules at each of 1, 2
      and 4 slots. Every turn's tokens and rows equal its reference's. Rollbacks that continued in place giving up more
      than they kept: 144 at 4 slots (6 in every schedule), 28 at 2 and 11 at 1.
    - *CPU arm* (`TestKV2FU1_cpuOwnRollback`): all three runs pass. Each reply equals the conversation's alone in its
      token ids and in every step's chosen and top-4 log-probabilities, as float bits. B's first turn, which copies 64
      rows of A's preamble, also equals B alone.
    - *Units:* `TestPickSlotFor_deepOwnRollback`, `TestPickSlotFor_ownRollback` (as amended) and
      `TestPickSessionFor_ownRollback` pass.
    - *Identity:* `decoder` `-run 'KV2|ResidentReuse|MC3|Session'` gives 49 PASS, 4 SKIP and 0 FAIL. The four skips need the
      gemma-3-270m fixture, which this box does not have, and skip on `main` too. `internal/kvmgr` gives 29 PASS and
      `internal/serveapp` `-run 'Session|KVShare|PickSession'` 12 PASS. The only existing test file changed is
      `internal/kvmgr/share_test.go`, in the named case.
    - *Plants* (`docs/measurements/kv2-2026-10-14/planted-fu1.txt`):
      1. the branch point's rule in `kvmgr.pickSlot`: red in all 24 schedules at 4 slots and in 7 at 2 (rules 1 and 2
         on 136 turns, rule 3 on 137), and in both `kvmgr` unit tests. A one-slot model never calls `PickSlot`, so 1
         slot stays green;
      2. `kvmgr.Slots.Commit` restarting the anchor: red in the same 31 schedules;
      3. the CPU LRU ignoring the anchor: red in both runs with room, green in the control. The token ids were equal
         and only the log-probability bits differed, so a check of ids alone would not have seen it.
    - *Other checks:* `scripts/touched.py` green; `go vet` untagged, with `realckpt`, with `goinfer_testhooks` and for
      linux/amd64; staticcheck v0.8.1 native and for linux/amd64, after its canary went red on both; `gofmt -l` clean.
      `TestParityManifest_fresh` passes (43 of 43 families). The parity refresh refreshed no `deps_hash`
      (forward goldens 33 passed, 31 skipped, 0 failed): `decoder/resident_reuse.go` is in no family's dependencies.
    - *Not run:* a device arm (registered out above).
    - **Defect found on main, 2026-10-21, and fixed the same day (branch `fix-kv2fu1-own-test-2026-10`).** The merge
      (#127) turned main's CI red in `TestResidentAcquire_interleavedConversationsKeepTheirSlots` and
      `TestResidentAcquire_bindFailureGoesCold`; the merge's re-run had named its tests by pattern and missed them. The
      own test, "keeps the first prompt but its last token", also accepts another conversation whose prompt differs from
      the first prompt only in that last token: a reuse score leaves the last token to compute, so the two score alike.
      Before FU1 that only lifted the allowance; with FU1 it skips the keep-versus-overwrite rule too, so the other
      conversation overwrote the slot. Fix: `kvmgr.Slots.KeepsFirst` checks the prompt begins with the whole first prompt,
      and the decoder passes a slot's first-prompt length only when it does; the CPU LRU tests its uncapped common prefix
      against the whole first prompt. `TestPickSessionFor_ownRollback` gains the case; both decoder tests and it are red
      without the fix and green with it; the whole `decoder` package and FU1-G3 pass on the fix.
    - *CPU arm,* serve's session LRU over llama-attnfa-tiny's real CPU forward. Conversation A has a 600-token first
      prompt (a 300-token preamble and 300 of its own); B shares the preamble. Turns: the first prompt; its reply and
      700 tokens; a rollback to the first prompt and reply plus 2 tokens (it keeps 606 and gives up 706). Three runs:
      an LRU of 2 holding A alone (room); an LRU of 3, A and B interleaved (room); and an LRU of 2 holding both (no
      room, the control, where main also reuses in place).
      - Rule: every turn after the first is handed the session its previous turn used, reuses its common prefix with
        it, copies nothing, and replies exactly as that conversation does alone in an LRU of one.
      - The rollback's cold prompt is 608 tokens, past the CPU's 512-row fast class, so a re-prefill writes the reply's
        rows on another kernel than the decode that wrote them in place, and the reply check can see the defect.
    - *Identity (replies unchanged against KV2's merge):* KV2's tests of the slot choice, the copy and the sessions
      pass, with their test files byte-identical to the branch point (`48ca55f8`): `decoder`
      `-run 'KV2|ResidentReuse|MC3|Session'`, `internal/kvmgr` whole, and `internal/serveapp`
      `-run 'Session|KVShare|PickSession'`.
      - **Named under G1's exception, before the change:** `TestPickSlotFor_ownRollback`'s last case ("own rollback
        that keeps less than it overwrites", want 1, "main's rule stands") asserts the rule this item removes, and
        becomes want 0.
      - `TestResidentReuse_equalsTheBookkeepingItReplaced` stays byte-identical. With choice 3 off
        (`kvAllowanceOverride`) the decoder passes no first-prompt lengths, so its walk still compares the slot choice
        with KV1's bookkeeping.
    - *Plants,* each shown red once and removed, its output kept beside KV2's record:
      1. `kvmgr.pickSlot` keeps the keep-versus-overwrite test for the own conversation (the branch point's rule):
         rules 1 to 3 red at 2 and 4 slots;
      2. `kvmgr.Slots.Commit` restarts the anchor on every request: the rollbacks red;
      3. the CPU LRU ignores the anchor: the CPU arm red.
    - *Void:* a listed test skipped (read `-v` for `--- PASS`); a test file changed that is not named here; coverage zero.
    - *Not in this gate: a device arm.* No backend code changes. An in-place rollback is the path every one-slot model
      already takes, and the Metal, WebGPU and CUDA G3 arms' edits keep more than they give up, so their expected
      results do not change. CUDA takes the change through the same decoder path; its windowed floor is nobara's.
    - *Stopping rule:* fixed cases, run to completion, no sampling. *Cost:* seconds. KV2's decoder G3 runs in 0.11 s on
      its fake; the CPU arm's prompts are under 1,400 tokens on the tiny fixture.
- **Follow-up 2, the donor choice: built 2026-10-15** (found by nobara's CUDA arm,
  `docs/measurements/kv2-cuda-2026-10-15.md` §6).
  - **The defect:** with several slots holding a prefix and windowed KV, the copy took the first of the longest hits. When
    that slot's window had moved past the rows, it declined and prefilled cold, although another slot still held them.
    It cost time, never a wrong row.
  - **The fix:** `decoder/resident_reuse.go:kvChooseDonor`. Of the slots other than the destination whose capped hit
    beats the reuse, take the longest hit, the lower slot on a tie, that `ResidentReusableKVSlot` says still holds the
    rows. Decline only when none does.
  - **Tier 0 gate, on the owner's brief:** `TestKV2_donorWhoseWindowHoldsTheRows`.
    - Three slots. Slots 0 and 1 hold the preamble, and slot 0's window has moved past it.
    - The third conversation's copy comes from slot 1, and its reply equals its reply alone.
  - **`TestKVChooseDonor`** covers the rule's cases.
  - **The planted defect, the old first-match choice:** both tests red ("copied nothing"; "got slot -1"), then restored.
  - **G3's schedule independence passes,** with the 23 KV2 and MC3 tests near the change.
  - **A windowed family's served G5 cell is not run:** it is tier 2, and the owner set this gate at tier 0.

### KV3 — Prefill and admission in blocks on every backend (absorbs KV-pre) — DONE (v0.23.0)

**Done: shipped in v0.23.0 (2026-10-15).** That includes the rollback-budget fix in §5.12: CUDA's windowed KV compacts in
64-token steps, so a client's rollback keeps at least 448 tokens, and up to 1,024 from VRAM left after planning.

**Started 2026-10-13 (owner decision, §8 D8): alongside KV1, ahead of KV2.** It does not depend on KV1. The CUDA half is on the branch
`kv3-cuda-2026-10` with its record, `docs/measurements/kv3-cuda-2026-10-13.md`; the Metal half is the KV-pre session's (`metal/`).

- **Build:**
  - every resident's chunk size is a multiple of B (CUDA's `prefillChunkRows` default is already 512);
  - admission is in blocks plus the chunk scratch, from each allocator's own figure, replacing the whole-prompt bound on
    the resident paths;
  - **the 12,288-key limit on CUDA** (added 2026-10-13, KV0's biggest time-to-first-token finding): the batched prefill declined past 12,160
    attended keys because `checkPrefillShmem` held every layer to the exact attention kernel's shared-memory bound, including the layers the
    fused attention serves, whose shared memory does not grow with the span. The guard now asks which kernel serves the layer;
  - **the 16-row rule in the chunker**: a prefill never ends in a pass of fewer than 16 rows (a shorter pass is exact class); the pass before the last gives it
    64 rows, so every pass still starts on a multiple of 64 (the KV1 record's G2b: a windowed fused pass is invariant only from such a start).
- **CUDA half, done on the branch (day readings; see the record for the numbers):**
  - the guard change, with identity (digests bit-identical to `main` at 300, 3,900 and 12,000 tokens; the one prompt with a short last pass
    changed to the bits of its single-pass prefill, as registered) and fidelity (tiny fixture, 18 prompts of 12,400 to 16,384 tokens, cosine
    0.99986 or better against the per-token path; the real 1.5B at 12,500 tokens, cosine 0.996, argmax equal); the passes now run fused;
  - admission by the resident's own per-chunk figure (`decoder.ResidentPrefillChunks`; CUDA's `PrefillScratchBytes`, held to the pass by a test):
    0.19 GB against 1.65 GB for a 20,954-token request on the 1.5B;
  - **WebGPU's half** (branch `kv3-webgpu-2026-10`, `docs/measurements/kv3-webgpu-2026-10-13.md`): the same interface, the pass's own figure as an upper bound (1.04 to 1.06 times
    the bytes the pass creates), 0.18 GB against 1.65 GB for a 20,954-token request; pricing only, no execution change;
  - **night jobs queued** (under the timing lock): the real fidelity set (six prompts to 20,000 tokens), the Claude Code first-request A/B
    (registered band: PASS at 45 s or less, against 268.7 s before; the one exploratory reading is 25.1 s), TTFT at K = 512 and 3,900 against
    `main` (0.97 or better), and the whole CUDA suite with the ignored fixtures linked. **The branch merges when they read green.**
  - still declining past the bound, by name: head dimension other than 64 or 128 (Gemma 3 and 4, measured in the census), MLA, the Gated-DeltaNet
    families (the census logged no decline for Qwen3.6-35B-A3B at 12,242 keys, which is an inference, not a measured pass), the all-logits, embedding and residual tails, image blocks. The tiled exact kernel below is the work that removes the first of these.
- **The request to the KV1 session (§8 Q5) is met on its branch.** `kvmgr.ClassMap.DeclineShortLead` states the 16-row condition, and KV1 leaves the
  rest to KV2 by name: a hit on another sequence's rows must leave a suffix of 16 rows or more when the requester's cold prefill is fast class (§5.5
  step 4), and the class of a row is recorded per pass the decoder sees, so a 512-row-pass tail of 15 rows or fewer was recorded as fast. KV3's chunker
  removes that tail, so the two branches meet there: KV2's gate checks the recorded classes against launch counts, as KV0's test does. The decoder's
  own loops read the resident's tail minimum (`ResidentPrefillTailMin`: CUDA 16, everything else 8), so a manager that cuts passes should too.
- **Merging with KV1.** `origin/kv1-2026-10` merges into this branch with no textual conflict (checked 2026-10-13; `decoder/model.go`, the changelog and this
  doc auto-merge). Both change non-test `decoder/*.go`, so the parity refresh (`scripts/refresh_parity_hashes.sh`) runs once, after the later of the two
  merges and before its push.
- **After the census (`docs/measurements/kv3-long-prompt-census-2026-10-13.md`): a tiled exact attention kernel, owner-approved 2026-10-13 (§8 Q7).** The
  dense Gemma families (head dim 256, and 512 on Gemma 4's global layers) still fall to the per-token path past 12,160 keys, at least 183 s and 276 s for a
  20,000-token prompt against about 17 s and 49 s batched. The paged-expert families do not decline; their cost is expert paging. **Built 2026-10-13 on
  `kv3-tiled-2026-10`** (record: `docs/measurements/kv3-tiled-exact-2026-10-13.md`, registered before the code, three amendments before their runs): three
  row-batched kernels in their own module (`cuda/prefill_splitkv.cu`, decode's split-KV attention with the query row on the grid), taken by a layer whose
  exact attention is past the bound; nothing below the bound changes. **Day gates pass:** the kernels equal `attn_batched` bit for bit in 360 cases, a forced
  route equals `attn_batched` over a whole model pass, a batched exact pass past the bound equals the per-token path bit for bit on the tiny fixture (logits and
  every K and V row), and admission's figure equals the pass's allocations. **Night readings (2026-10-14, record §10):** speed PASS (Gemma-3-4B 5.68x, Gemma-4-E4B
  3.31x against the per-token path at 12,400 tokens), no-regression TTFT PASS, the real-model identity PASS on both exact arms (Gemma-4-E4B, Gemma-3-4B levers off,
  bit for bit) and **FAIL on Gemma-3-4B at default settings: cosine 0.9865 against the 0.99 bar**. The difference is the tensor-core GEMM's, which already reads
  0.9901 on Gemma-3 at 12,000 tokens below the bound; the branch is the first to run it past 12,160 keys. **Owner decision 2026-10-14: exact** (record §11): a prompt that takes the
  row kernels runs every pass on the exact kernels. **Read 2026-10-14 (record §12):** G4 on Gemma-3-4B bit-identical in both arms at 12,400 tokens, G5 re-read 4.13x
  (PASS; 5.68x with the GEMM on). The CUDA suite timed out at 2 hours with only the four reds of the branch below (three fresh-process tests and the tail-rule defect,
  whose fix is merged here); its remainder, in the gate's partition, read no failure (its job's red was a test list built with `realckpt`, fixed). The branch merges after `kv3-cuda-2026-10`.
- **Not started: a fidelity gate for the CUDA tensor-core GEMM at depth on Gemma-3 (option (b) of the tiled record's §10; not chosen 2026-10-14).**
  Why it is owed even with the exact rule: on one prompt Gemma-3-4B's default batched prefill read cosine 0.9963 at 1,200 tokens, 0.9901 at 12,000 and 0.9865 at
  12,400 against the per-token path, so `main`'s Gemma-3 prefill between 3,900 and 12,160 keys is unmeasured and may already sit under the 0.99 bar the KV3 work
  uses. The lever's own gate (`docs/completed/task-prefill-gap.md`, L2/L3) was graded on the Qwen models to K = 3,900, never on Gemma-3. What it would register:
  Gemma-3-4B (and Gemma-2 / dense Gemma-4 if their checkpoints are local), prompts at about 2,000, 4,000, 8,000 and 12,000 tokens, at least eight realistic prompts
  per depth from a hashed set; default batched prefill against the per-token path; last-token logits cosine and argmax with the near-tie rule (3% of |top-1|), and
  per-layer KV cosine to name where it diverges; a bar per depth (cosine 0.99 on every prompt and argmax equal outside near ties), a parked band 0.985 to 0.99, and
  what each outcome triggers (pass: the GEMM may return past the bound under a new registration; fail below the bound: the floor for Gemma-3 moves or the lever goes off
  for the family). Cost: the per-token side is about 90 rows a second, so about 40 minutes for the four depths on one model (eight prompts each); a night job. Owner to decide whether and when.
- **Gates (sketch, registered per half in the record):**
  - the 20,199-token request answers on the Mac's 7B, which today is the I7 413 (the KV-pre session);
  - chunk invariance holds per backend (KV0: CUDA and WebGPU measured, Metal's existing test);
  - time-to-first-token at K = 512 and K = 3,900 is 0.97 or better against main (CUDA: night job);
  - a Claude Code-sized first request on CUDA prefills in batched passes (night job; 45 s or less on Qwen3-4B).
- **Size:** S.
- **Inside a pass: what is left after chunking (added 2026-10-15; not started).** With passes bounded at 512 rows,
  `cuda/prefill_scratch.go:cudaResident.PrefillScratchBytes` is about 69 MiB on the 1.5B and 148.5 MiB on the 7B (Qwen2.5
  shapes, before the allocator's rounding), and the MLP is most of it: three `M × inter` f32 buffers, against 12–13% on the
  attention side. Two reductions, both bit-identical by construction. Each lowers the per-chunk admission figure, and so frees
  KV pool, since the pool is the budget minus everything else (§5.10). Neither is large; together they are about 1,000 tokens
  of f16 KV on either model.
  1. **CUDA: `glu_quant_batched` writes its product into the gate buffer.** Each thread reads `g[i]` and writes `dscr[i]` at
     the same index, and the quantize loop reads `dscr` only after `__syncthreads`, so `dScrB` can be `gOb` (and `pleGB` on
     the PLE path, `cudaResident.prefillCore`'s second `bSw` launch). That drops one `M × inter` buffer per pass: 17.5 MiB
     on the 1.5B, 37 MiB on the 7B, about 25%, roughly 640 tokens of f16 KV on either. The kernel declares `g` and `dscr`
     `__restrict__`, and aliasing restrict pointers is undefined behaviour, so this needs an in-place variant without it, a
     PTX rebuild (`cuda/build_ptx.sh`), a bit-identity check against today's pass, and `PrefillScratchBytes` changed in the
     same commit (`TestKV3_prefillScratchBytes_matchesThePass` holds the two together).
  2. **Phase aliasing, the pattern aikit v1.70.2 shipped for its encoder.** The attention-side buffers (`aqB`, `qBb`, `kBb`,
     `vBb`, `cctxB`, `cqB` and their scales) and the MLP-side ones become sub-buffers of one allocation sized to the larger
     phase. It saves only the attention side, 12–13% of a pass (8.5 MiB on the 1.5B, 19.5 MiB on the 7B). Which buffers
     cross the phase boundary has not been checked in this tree; aikit's `TestScratchPhases_poisonedBoundaryIsInert` (NaN
     written into each phase's buffers at the boundary, output bit-identical to a fresh run) is the model for its gate.
     Worth doing only after 1, and only if the KV budget is at the margin.
  - **Metal is not the same fix.** Its gate and up share one fused `[M, 2·maxI]` buffer (`guF`), and the product goes to a
    separate `[M, maxI]` (`dqF`), which the down GEMM reads. Writing the product compactly into `guF` would overwrite rows
    not yet read, so the Metal equivalent needs a down GEMM that reads A at a 2·I row stride, or a different kernel order.
    Not scoped.

### KV4 — Paged storage on CUDA, then f16

- **KV4a: f32 paged, bit-identical.**
  - pools per group (with block 0 reserved and zero-filled as the null block for released window/chunk entries and tail padding,
    §5.3); device tables;
  - every KV kernel through the table, with 64-bit (`int64_t` / `size_t`) row-offset arithmetic in `attention` and `mla_attn`
    (`kv0-tile-census-2026-10-13.md` §5);
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
- **Size:** L. **Closes:** lever 3 (CUDA f16 pools and remaining families), the slots-against-context question, and the windowed
  shifted view.

### KV5 — Paged storage on Metal

- **Build:** KV4's shape with Metal's kernels and its f16/i8 pools (including the zero-filled block 0 null block so
  `attention_prefill_fused` and `attention_prefill_steel`'s out-of-window and tail over-reads never load `NaN`, §5.3). Unified
  memory means no host tier.
- **Gates:** KV4's, with Mac numbers.
- **Closes:** the context-against-slots trade that M2 and I7 both make on Metal. I7's open question (how far context
  should give way to slots) dissolves.
- **Size:** L.

### KV6 — WebGPU: keep slots with KV2's copies (DECIDED 2026-10-13, D3)

- **The facts:** a slot is a whole runner with bind groups baked over its KV, `maxStorageBuffersPerShaderStage` is 8 (the default, which the device request does not raise; inferred from the spec,
  not read from a device) and int8 attention already binds 7, and the ceilings are per precision (`kv0-tile-census-2026-10-13.md` §5).
- **Decision (D3):** keep slots on WebGPU with KV2's cross-slot copies; do not page it unless a WebGPU user case appears after
  KV4/KV5.

### KV7 — Tiers, parking and resume

- **First slice built 2026-10-16 (CUDA, over slot storage, pinned host RAM only, one restore):** prefix snapshots,
  `docs/tasks/task-kv7-prefix-snapshot-2026-10.md`. The disk tier, promotion policy and block-granular host pool below are still to do.
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
- **KV8a ships first (2026-10-20):** the end-of-prompt checkpoint on the single slot. KV8 builds the pool, the other
  checkpoint positions and admission on it, and keys its store as KV8a does.
- **Note, 2026-10-20 (KV0-H check, `docs/measurements/kv0h-hybrid-traffic-2026-10.md`, amendment A5):** in the Claude Code check on Qwen3.5-9B (check data, 10 requests, one
  thread), **2 of the 8 D divergences sat 3 tokens into a new user message**, 25 tokens before the end of the previous prompt. **A message-boundary checkpoint would serve them; an
  end-of-prompt checkpoint would not.** The other 6 sat 1.5k to 3.6k tokens before the end (a rewritten history). It is two requests, under that study's bar of three, and it bears on
  where the checkpoint is taken, not only on whether.

### KV8a — An end-of-prompt checkpoint on today's single slot (added 2026-10-20)

- **Why (KV0-H, `docs/measurements/kv0h-hybrid-traffic-2026-10.md` §2).** On Qwen3.5-9B under opencode and Codex (CUDA,
  nobara, f32 KV), 29 of 69 follow-up requests extended the previous prompt but not the previous reply (category B). All 29
  diverge inside the reply, 15 to 123 tokens in (median 26). opencode sends no reasoning back, and its template re-renders
  the reply without its think block; half of Codex's B requests carried ids, so carried ids do not remove B. Today each
  re-prefills its whole 6.1k–9.4k-token prompt: median time to first token 37.6 s (opencode) and 27.3 s (Codex), against
  2.2 s and 0.6 s for A. Restoring the state saved where the previous prompt ended would re-process 71 to 416 tokens
  (median 201). Projected: 71.7% of the prompt tokens processed skipped, 34.2 s median per affected turn, 1,006 s over the
  study's 29 B turns. Both bars of the registered rule (25%, 5 s) pass, so the rule pulls this ahead of KV4.
- **What it is.** One checkpoint for the one resident slot hybrids already have (`decoder/resident_reuse.go:Model.residentSlotCount`
  stays 1):
  - **Save.** Once per request, when prefill has processed the whole prompt and before the first decode step, copy the slot's
    recurrent state at position `c = len(prompt)`: every recurrent layer's state and its conv window, for both mixers
    (`decoder/residency.go:ResidentMixer.Recurrent`, DeltaNet and Mamba). Saving at the end of prefill leaves every
    prefill pass shaped as it is today.
  - **Keep** the attention layers' KV as today. It is positional, so truncating to `c` is enough.
  - **Restore** when the next request matches the slot's committed tokens through `c` but not through the end of the
    committed sequence, and carries at least one token past `c`: copy the checkpoint into the live state, truncate the
    attention KV to `c`, and prefill from `c`. §2's rule that at least one token is always recomputed holds, since the
    request has tokens past `c`. A strict extension of the whole sequence keeps today's path (continue from the live state).
    A divergence before `c`, an identical re-request, or a request with image spans stays cold, as today.
  - **Score.** `internal/kvmgr/slots.go:scoreRecurrent` returns `c` in the restore case, where today it returns 0. Its image
    rules are unchanged.
  - **Where it lives.** Device memory if the load's memory plan admits it, host memory otherwise; the choice is registered
    per backend before building. Its size comes from the state spec (§5.7: 62.8 MiB for Qwen3.6-35B-A3B, 149.6 MiB for the
    27B; the 9B is measured in gate 1). The bytes are charged to the memory plan. When they do not fit, KV8a declines and the
    slot behaves as today: fail closed.
  - **Keyed** as §5.7's checkpoint store is keyed, `(chain hash at c, c)`, with one entry, so KV8 extends it rather than
    replacing it.
- **Not in KV8a** (KV8 keeps them): more than one state slot; checkpoints at commit, at a preamble boundary, at a message
  boundary (KV8's 2026-10-20 note) or every N blocks; Marconi admission; cross-conversation reuse of recurrent state
  (`decoder/session_share.go` and `decoder/kvsnap.go` keep refusing hybrids); MC3 batching; the image path
  (`decoder/generate_vl.go` keeps its recurrent refusal); speculative rollback on hybrids (§5.7, out of scope). Parking
  stays in KV7 and KV8. KV0-H measured no C requests, and its A5 records that this is not evidence against parking for
  Claude Code on a larger card.
- **Gates (sketch; whoever starts KV8a registers them first, as `CLAUDE.md` requires):**
  1. **Bit-exact save and restore**, conv window included (spec/09: skipping the conv window gives wrong logits with no
     error). The restored state equals the saved state byte for byte, every recurrent layer, on CUDA. Record the checkpoint's
     bytes for the 9B.
  2. **Agreement with a cold prefill.** On B-shaped fixtures (a conversation whose next prompt re-renders the previous reply
     differently), compare the token stream after restore and suffix prefill against a cold prefill of the same prompt. Hold
     it to the bar the strict-extension path (R-01 phase 0) was accepted on, since that path already continues from a prompt
     prefilled in two parts. If that bar is not on record, register one before building.
  3. **The cache-state grid** (`decoder/cachestate.go:cacheStateGrid`, §2) gains the save and restore paths, with a declared
     cell for every kind of state. The recurrent `lcShare` cell ("no checkpoint to restore") changes with it. The grid stays
     fail-closed.
  4. **The cost on every turn.** The save runs on every request, used or not. Its added time to first token on A turns stays
     inside noise against KV0-H's A medians (2.2 s opencode, 0.6 s Codex). With device-chained decode
     (`decoder/residency.go:ResidentGreedyChain`, `decoder/residency.go:ResidentSampleChain`), the save is ordered before the
     first chained decode step.
  5. **The measured win.** Rerun KV0-H's opencode and Codex configuration with KV8a on: same model, card, tasks, f32 KV and
     ids key. The bar: B turns' median time to first token falls by at least half the projected 34.2 s, and A turns do not
     regress beyond noise. KV0-H's readings are the do-nothing arm.
- **Order.** CUDA first, on nobara, where KV0-H ran. Metal follows with gates 1 to 4 on the Mac once CUDA passes; gate 5 on
  Metal is a separate reading, if the owner wants it.
- **Size:** S–M per backend (an estimate). **Needs:** KV0-H only, nothing from KV4. **Closes:** the B half of R-01 phase 2
  and of FreeToken Lead 1; KV8 keeps the rest.

### KV9 — The CPU path (deferred by D2, 2026-10-13)

- **Decision (D2):** the CPU path joins only through KV2's copies for now; moving CPU storage to host blocks (option (b) of §5.14)
  waits on a measured multi-session memory case.
- **Gates (if reopened):**
  - the parity manifest bit-identical (the CPU is the reference);
  - the memory saved with four sessions sharing a preamble, measured.
- **Step 0, measured 2026-10-14 (exploratory):** `docs/measurements/kv9-2026-10/README.md`.
  - **Recommendation:** close KV9 with KV2's copies as enough. Owner's call.
  - At 6K on the 0.5B, host blocks would save 405 MiB of a measured 2,159 MiB four-session footprint.
  - The CPU cold prefill's retained high-water is the larger term.
- **Size:** M–L if reopened.

### KV10 — Long context and lower precision

- **Build:**
  - raise Metal's and WebGPU's ceilings to their kernel limits;
  - a 64K/128K fidelity harness (lever 4's third step), with a YaRN-configured checkpoint for the long end;
  - the families with no fused prefill kernel, listed or given one;
  - K8V4 as research.
- **Size:** M.

### KV11 — One residency manager for KV blocks and MoE experts (added 2026-10-13)

Added on the owner's word the same day, from the conversation that produced `docs/tasks/task-forward-architecture-2026-10.md`
(the compute half; this is its memory item).

- **Why: two managers split one memory by a fixed rule.**
  - A load that pages MoE experts keeps KV at 4,096 positions on CUDA and on Metal. The expert cache takes whatever is left
    after everything pinned, so every byte given to KV is a byte the experts never see.
    - On CUDA, growing the context there "cut its slots and decode speed sharply" (`cuda/resident.go:resolveCtxCapFit`).
    - Metal keeps the same rule: "paged MoE keeps the default" (M2, `CHANGELOG.md`).
  - Three expert pagers keep three budgets:
    - CUDA's expert-slot cache, per layer (`-moe-cache-experts`, `-moe-cache-slots`);
    - Metal's `metal/expertpool.go`, also per layer, under the same flags;
    - the CPU's `decoder/moepaging.go:expertPager`, one cache across layers (`-moe-pager`, and `-stream-weights` with
      `-weight-cache`).
  - So a paged-MoE user picks context or expert hit rate at load, by flag. Neither choice fits a mixed workload: a long agent
    prompt wants KV, and a short chat wants experts.
- **Design.**
  - An expert slot becomes a second kind of unit in §5.10's pool. KV blocks and experts share one budget, one allocator and
    one pressure policy.
  - **The two units differ in size.** A KV block is 64 rows across a group's layers (§5.2, option A); an expert slot is one
    expert's gate, up and down weights and scales in one layer. Both depend on the model and its quantization. So they share
    memory through a common slab arena, or by trading coarse device buffers under one byte budget, not by packing experts
    into KV blocks.
  - **One concrete form of the common slab arena: two-ended (added 2026-10-15).** From demofox's "double-headed stack
    allocator" (2026-10-03): KV blocks are carved from the bottom of one device allocation and expert slots from the top,
    and the boundary moves as the mix shifts. A long agent prompt moves it up by evicting the expert at the boundary; a
    short chat moves it down by releasing the KV block there. Three things decide whether it fits:
    - **Only the boundary moves.** Every expert slot except the one at the boundary keeps its address, which meets the
      graph constraint below for all but one slot: only the boundary slot's segments need re-capture, or the boundary
      slot stays out of captured segments.
    - **KV blocks are relocatable.** Kernels reach them through the block table, a launch argument (§5.11), so a busy KV
      block at the boundary can be copied into a free block lower in the pool and its table entry rewritten, instead of
      waiting for it to free. Holes in the KV side stay usable by KV, since every block is one size; they only hold the
      boundary still until it compacts.
    - **Against it:** option A's per-group pools (§5.2) make several KV sides, not one, so the two-ended form fits one
      group's pool (the full-attention group, which holds most of the bytes) or needs option B's uniform page; and
      compaction is device copy traffic the eviction cost model has to charge.

    A design option for the build, not a decision. "First, measure" below still decides whether there is a build.
  - **Resizing is new.** Today CUDA sizes its expert cache once, at load and after KV, to the free memory it measures
    (`cuda/resident.go:cudaResident.allocSlots`, `cuda/resident.go:expertCache`). Any buffer a captured CUDA graph segment
    reads must keep its address, so slot buffers are allocated at their largest size up front, or the segments that read
    them are re-captured when they change.
  - An expert's lower tier is the mapping or host copy it is paged from (§5.8).
  - Eviction compares expected cost per unit: a KV block's re-prefill or demotion cost, against an expert's miss cost (a
    host-to-device copy per use, times its measured miss rate, per layer).
  - D5's order still holds among KV blocks. An expert goes before a cached KV block only when its expected miss cost is lower.
  - The CPU pager keeps its own cache (D2).
- **First, measure.** The build goes ahead only if these show the fixed split costs something:
  - per-layer expert hit rates at a fixed total budget on the real Gemma 4 26B-A4B, which is FreeToken Lead 4's first
    step (`docs/tasks/task-freetoken-techniques.md`);
  - the trade curve on one paged-MoE model per GPU box: decode speed, and the longest prompt that fits, against the KV share.
- **Gates (sketch):**
  - tokens identical to today's at the same split, since paging changes where weights live and not what is computed;
  - a prompt that today's 4,096 cap refuses now fits, up to the fit default (16,384) where memory allows;
  - decode at short prompts within 0.97 of today's 4,096-position load;
  - never-swap throughout.
- **Owner question when it starts:** do `-moe-cache-experts` and `-moe-cache-slots` keep their meanings, or become caps on
  the shared budget? D6 covers only `-ctx` and `-kv-sessions`.
- **Needs:** KV4 (a pool to share) and KV7 (the tiers). CUDA first, then Metal after KV5.
- **Absorbs:** FreeToken Lead 4, pooling the CUDA expert cache across layers.
- **Size:** L.

## 7. Order, and what is useful if the program stops early

- **KV0 → header fix + KV3 alongside KV1 → KV2** (decided in D8 and Q2) deliver the agent-preamble win, fix CUDA's 12,288-key
  batched-prefill decline, and fix the Metal 413, with no attention-kernel paging. They pay for themselves before KV4 starts.
- **KV4** is the gate on the rest (decided on the capacity bar, Q3). If it parks on speed, KV2's copies and KV3's admission stay,
  and the doc records why.
- **KV7 and KV8 can come before KV5** for CUDA users. **KV6 is closed (keep slots, D3).** **KV9 is deferred (D2).** **KV10 waits on
  KV4/KV5.**
- **KV8a comes before KV4** (added 2026-10-20). KV0-H's registered rule pulled the end-of-prompt checkpoint forward. It
  works on today's single slot and leaves the pool and the rest of the checkpoint policy to KV8.
- **KV11 follows KV4 and KV7,** CUDA first, and is built only if its first measurement shows the fixed split costs
  something.
- **Every phase leaves the tree consistent:** a backend not yet paged keeps working behind the slot adapter.

## 8. Owner decisions

| id | question | recommendation |
|---|---|---|
| **D1** | Reverse the paged-attention parking narrowly, for memory and reuse within one user's conversations and agents? `positioning.md`'s "no paged attention" would change; continuous batching stays parked. | **DECIDED 2026-10-13: yes** ("un park paged attention for sure"). |
| **D2** | Does the CPU path join (KV9)? | Index and copies when KV2 lands; storage later, only on a measured memory case. **DECIDED 2026-10-13: no, not now; the CPU joins only through KV2's copies.** |
| **D3** | WebGPU: page it or keep slots? | Keep slots; KV6 decides with numbers. **DECIDED 2026-10-13: keep slots on WebGPU**, as the census supports. |
| **D4** | Sharing scope by default? | One namespace per server, one per API key when keys are configured; `-kv-share off` available. **DECIDED 2026-10-13: one namespace on loopback, one per API key, and no sharing between namespaces (off) when `serve` binds beyond loopback without API keys.** The conservative choice; it costs a single local user nothing. **As built in KV2 (2026-10-14), against serve as it is:** serve has one shared API key, so "one namespace per key" is one namespace, and its holders are one trust domain. Serve refuses to start beyond loopback without a key, so that case cannot arise. The network-facing case that remains, a loopback server answering an `-allowed-hosts` name without a key, shares nothing. `--kv-share off` turns sharing off for any server. |
| **D5** | Policy under pressure? | Evict idle → demote → preempt the newest by release (§5.10). **DECIDED 2026-10-13: as proposed.** |
| **D6** | CLI meanings before v1.0 (`-ctx`, `-kv-sessions` are Hard tier)? | Keep both: `-ctx` is a per-conversation cap, `-kv-sessions` the number of conversations kept warm and running on a resident. New flags only for sharing scope and the host-tier budget. **DECIDED 2026-10-13: keep the meanings of `-ctx` and `-kv-sessions`.** |
| **D7** | CUDA's default KV precision once f16 passes its gate? | f16 by default with f32 kept as the reference, Metal's precedent. **DECIDED 2026-10-13: extend f16 to the remaining families as each clears its fidelity gate** (it is already the default for llama, mistral, qwen2, qwen3 and phi3). |
| **D8** | Where this sits in the queue? | KV0–KV2 soon (they pay for agents now). Where KV4 sits against the multimodal track is yours to set. **DECIDED 2026-10-13: the header fix today, then KV3 (including the 12,288-key fix and the 16-row rule) alongside KV1, then KV2.** KV3 does not depend on KV1; it is the CUDA twin of the Metal long-prompt work (KV-pre) already in flight. |

### Raised by KV0 (all answered by the owner, 2026-10-13)

| id | question | KV0's reading |
|---|---|---|
| **Q1** | Remove Claude Code's per-conversation header: `connect claude-code` writing `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, a server-side strip of the `x-anthropic-billing-header` line, or leave it? | Without it Claude Code conversations share 22 tokens; with it, 18,213 of about 20,154. This is a `connect` or `serve` change (product code), outside KV0. **ANSWERED: `connect claude-code` writes `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, and the recipe says so. Serve does not silently rewrite a client's prompt; if wanted later it is an opt-in flag.** Done: `docs/measurements/kv3-cuda-2026-10-13.md` §1. |
| **Q2** | The batched prefill declines past 12,288 attended keys on this card (a 49,664 B shared-memory need against 49,152 B) and the prompt falls to the per-token path at about 66 tokens a second: a Claude Code first request takes about 4.5 minutes. Fix it before KV1, inside KV3, or leave it? | It dominates Claude Code's cost on CUDA: sharing cannot help the first conversation. **ANSWERED: fix it first, inside KV3, and start KV3 now.** It does not depend on KV1; it is the CUDA twin of the Metal long-prompt work in flight, and the biggest time-to-first-token item KV0 found. |
| **Q3** | KV4's bar, as registered, mixes two things (§6, KV0's decision rule). Should the sharing arm decide KV2 and the capacity arm decide KV4? | Capacity 2.78 ×; sharing arm clears only with the header off, and KV2 delivers it. **ANSWERED: split the bar. The sharing result decides KV2 (copies deliver the sharing), the capacity result decides KV4. On those terms both go ahead.** |
| **Q4** | D4's default when `serve` binds beyond loopback with no API keys: the SafeKV reading (§11a) suggests cross-namespace sharing off by default. | A proposal from the reading, not from a source. **ANSWERED: yes, default it off then** (see D4). |
| **Q5** | Should `declineShortLeadReuse` (and the key) state the 16-row condition, and should KV1's windowed-layer tests run chunk and split arms at a start that is not a multiple of 64? | Measured for the first; INFERRED by the CUDA census for the second. **ANSWERED: yes to both, and the first is a correctness rule, not tuning:** the 16-row condition goes into the short-prefix guard and the key (KV1's files), and into the chunker (KV3, here); KV1's windowed tests start at offsets that are not multiples of 64. |
| **Q6** | The I3 night plan's configurations run Qwen2.5-7B-Instruct and Mellum2.1 through opencode and Codex. | In KV0's pilots Qwen2.5-7B-Instruct made 1 to 6 requests a session and passed none; the Coder-7B made 1 or 2. Expect the same stall in those cells. **ANSWERED: swap Qwen2.5-7B-Instruct for Qwen3-4B in the I3 night cells** (KV3 item 4; `task-integration-harness-2026-10.md`). |
| **Q7** | After the long-prompt census: build a tiled exact attention kernel for CUDA, or only a head-dim-256 fused instantiation? | The exact kernel covers 256 and 512 and removes the 12,160 bound for any family; the fused instantiation covers Gemma 3 only. **ANSWERED 2026-10-13: the tiled exact kernel (the census's recommendation).** Registered before code, in KV3's record. |

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
  - lever 4's third step (KV10);
  - FreeToken Lead 4, pooling the CUDA expert cache across layers (KV11).
- **Replaces when built:**
  - `cuda/kvwindow.go`'s shifted view (KV4);
  - the per-slot bookkeeping in `decoder/resident_reuse.go` (KV1);
  - `pickResidentSlot` and serve's preamble floor (KV1–KV2);
  - WebGPU's runner-per-slot if KV6 says so;
  - the fixed 4,096-position context of a paged-MoE load, and the GPU expert caches' per-layer slot counts (KV11).
- **Interacts with:**
  - **KV-pre:** its block size;
  - **I7** (`docs/tasks/task-integration-harness-2026-10.md`): its slots-against-context question dissolves at KV4/KV5,
    so its code should stay small;
  - `docs/completed/task-cpu-kv-f16-2026-10.md`: independent;
  - `docs/tasks/task-library-surface-review-2026-10.md`: D6;
  - `docs/option-state-grid.md` (its Policy section; design record `docs/completed/task-option-path-admission-2026-10.md`): new options join its grid;
  - `docs/tasks/task-forward-architecture-2026-10.md`: its RF2 and RF4 read KV4's and KV5's block tables, and its OP4
    memory planner prices through §5.10's one function.
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
seven of its lines changed: see "What the KV0 reading corrected in §5 and §11" at the end.

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

### What the KV0 reading corrected in §5 and §11 (applied above)

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
  - `docs/completed/task-cuda-windowed-kv-2026-10.md`, `docs/tasks/task-memory-accounting-2026-09.md`,
    `docs/completed/task-never-swap-2026-09.md`;
  - `docs/tasks/task-integration-harness-2026-10.md` (I4, I7), `docs/completed/task-cpu-kv-f16-2026-10.md`,
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

<!-- doc-reviewed: 2026-10-21 -->
