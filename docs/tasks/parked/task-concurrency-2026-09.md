# Task: concurrency — stop the resident-KV thrash, then earn batched multi-request decode (MC0–MC5) — 2026-09

> **PARKED, 2026-10-01 (owner), with one graded run queued.** Moved to `docs/tasks/parked/`; a pointer stub stays at the
> old path because ~60 code comments and several docs cite it. MC0–MC3 (including MC3c) and chunked prefill shipped
> (below), and nothing in this track is an unblocked build. Open:
> - **The MC4 candidate's graded run on CUDA — RAN 2026-10-01 night: IDENTITY FAILED; ROOT-CAUSED AND FIXED 2026-10-02 (`2f685d7e`:
>   an adaptive round held `resBusy` but not the batcher's `busy` flag, so two generations wrote each other's KV slot); RERUN RAN 2026-10-02: IDENTITY STILL FAILS** (one reproducible
>   candidate-only divergence in the staggered workload, reply 21 of 30 in all three rounds; copy at 4 clients also varies, but batch differs from itself there, so that one is not the candidate's). ROOT-CAUSED AND FIXED the same day (a reuse under 64 tokens is declined on a fast-prefill prompt; 0 differing replies in every arm and schedule at the default configuration; the graded rerun on the fixed build is not yet registered or run): the staggered reply is a near-tie flip caused by reused KV rows computed by a different kernel class (a short earlier request's exact rows under a long fast-prefilled prompt; a 3-token header reuse is enough), not by the candidate and not by the kernels (they are start-offset-invariant);
>   with `GOINFER_CUDA_FAST_PREFILL=0` every arm and schedule is 30/30 identical to the sequential reply (`DIAGNOSIS` and `ROOT CAUSE` sections); the gate is not re-graded;
>   the speed numbers are recorded, not read (chat at 4 clients 0.850x batch, below the 0.97 hard bar, would park it on its own). The first run's speed gates are recorded, not read; see the RESULT, ROOT CAUSE,
>   RERUN and RERUN RESULT sections of [`mc4-candidate-cuda-2026-10-01.md`](../../measurements/mc4-candidate-cuda-2026-10-01.md).
>   Do not treat `-spec-adaptive` as lossless. The run was queued as: (`mc4-candidate-cuda`, nobara, ~45 min). The build is
>   `-spec-adaptive` (`6e9fcb99`, opt-in); the gates and the decision rule were registered before the run in
>   [`mc4-candidate-cuda-2026-10-01.md`](../../measurements/mc4-candidate-cuda-2026-10-01.md). The likeliest outcome is a
>   copy-only miss that goes to the owner. The Metal run of the same gates (the Mac's night queue, ~60 min) is not queued.
> - **Owner decisions:** whether `-spec-adaptive` ever becomes a default; "spec inside a batch" (CUDA copy L = 1.54×; needs a
>   ragged-batch kernel, no design); P10 on Metal (projected, not started).
> - **Owed, low priority:** the 413 prefill-share fix's end-to-end re-run (the 2026-09-28 attempt was inconclusive: its memory
>   ballast was compressible, so the ballast is fixed first; a unit test pins the fix).
> - **Optional, unqueued:** item 29, the W7 run on Vulkan (reported only; WebGPU's MC1 shipped regardless).
> - **Parked with a trigger:** Lever B (an aikit small-M kernel for batched CPU decode); MC4's MoE, recurrent and mixed-adapter
>   items; MC5's continuous batching and paged KV (the owner reverses the positioning); J6 re-run (8 or more concurrent
>   conversations on one model).
>
> **Current, 2026-09-28 (doc sweep):**
> - **MC1 (resident KV slots)** shipped on Metal, CUDA and WebGPU. On discrete WebGPU GPUs it is clamp-only by design,
>   since step 2 was reverted.
> - **MC3 (batched multi-request decode)** shipped on Metal and CUDA, with sampled batching (S2), step polish (S3/S4),
>   chunked prefill (`-prefill-chunk 512`) and the step-kernel spec verify.
> - **MC3c (CPU)** shipped: workers, then batched decode (`-cpu-batch auto`).
> - **MC2 earned**, with its Linux cells in.
> - **The 413 prefill-share fix's end-to-end check** ran 2026-09-28 and was INCONCLUSIVE: its memory ballast was
>   compressible, so it could not hold the band. The fix is owed before a re-run.
> - **The per-pass prefill cut is CLOSED:** the measured share is 0.151, not ~⅓.
> - **MC4 candidate "speculate when alone, batch under load"**: built 2026-09-29 as `-spec-adaptive` (`6e9fcb99`, off by
>   default). CUDA premise measured 2026-09-29: S 1.253×, L 0.790× on chat. CUDA's copy workload overshot the candidate's own
>   condition (L 1.540×, spec beats batching under load), a separate, larger "spec inside a batch" question (investigated
>   2026-09-29, parked; the shipped candidate underperforms plain `-spec ngram` on copy traffic). P10 on Metal is projected,
>   not started.
> - ~~**Since 9ccf7fb1** (embed-int4 on by default), a default Metal load declines the resident~~ — **fixed 2026-09-30**
>   (`0ca36756`: `--embed-int4` defaults off on `--backend metal`, so a default Metal load stays on the GPU). The old
>   `-embed-int4=false` workaround is no longer needed.
>
> **Status: FILED 2026-09-23; owner decisions taken 2026-09-26 (1: yes, 2: fold). MC0 DONE 2026-09-26: the
> thrash is confirmed on Metal, and a CPU session-LRU bug it uncovered is fixed. MC1 SHIPPED 2026-09-26 on Metal**
> (`c2f1532e`): 4 resident KV slots hold the 1-client aggregate at 2 and 4 clients (0.99× / 1.01×), and the 4-client
> aggregate is 1.22–1.24× the previous build's. **MC1 SHIPPED on CUDA 2026-09-27** (`9fddaf7e`,
> [`concurrency-mc1-cuda-2026-09-27.md`](../../measurements/concurrency-mc1-cuda-2026-09-27.md)): on the 1.5B, 4 clients
> reach 1.250× the one-slot build, and a lone request is unchanged. **MC1 SHIPPED on WebGPU 2026-09-27** (`3926f927`,
> [`concurrency-mc1-webgpu-2026-09-27.md`](../../measurements/concurrency-mc1-webgpu-2026-09-27.md)): on the Mac, 4 clients
> reach 2.805× the one-slot build (Qwen2.5 prefills per token there, so a thrash costs more), and a lone request is
> unchanged. **MC2 EARNS on the Mac CPU**
> (1.69–2.04× at B = 4, bit-identical; J8's 4 independent workers reach 2.00–2.48×). The Linux cells followed (below). **MC3
> SHIPPED 2026-09-26 on Metal** (`d4b708b5` + fixes `2b1cc280`, `d2225ec4`;
> [`concurrency-mc3-2026-09-26.md`](../../measurements/concurrency-mc3-2026-09-26.md)). All five pre-registered W7 gates
> pass: 4 clients at **1.593×** the serialized aggregate (76.2 → 121.5 tok/s), p99 turn **0.659×**, a lone request
> **1.002× / 1.004×** (p50 / p99), and every reply identical. It is on under serve's default `-max-concurrent` 4.
> **MC3 S0: no fidelity gate is needed.** Test-only `simdgroup_matrix` kernels carry 8 sequences for
> 1.0–2.7 GEMVs on every decode matmul (qkv, o, gate/up, down, int8 LM head). Every output is bit-identical to
> production's GEMV ([`concurrency-mc3-s0-2026-09-26.md`](../../measurements/concurrency-mc3-s0-2026-09-26.md)). **S1**:
> a whole batched step, in sequence, is bit-identical to production and reaches 1.73–1.84× at B = 4. The owner set
> the p99 bar's reading (same load + a lone-request guard) before the build. The owner chose "cpu first":
> **MC3c step 1 SHIPPED** (`serve -max-concurrent N`, CPU models): 1.86–1.97× at 4 clients, p99 halved,
> byte-identical, **default 4** (owner, 2026-09-26). The MC2 Linux cells are in: the 1.5B earns, and there batching beats
> the workers; the 0.5B is in the owner band.
>
> **Open, 2026-09-26:**
> - ~~MC3c step 2 (batching behind the same admission)~~ — **SHIPPED 2026-09-27** (`420d655b`, graded at `2c1d89ec`;
>   [`concurrency-mc3c-step2-2026-09-27.md`](../../measurements/concurrency-mc3c-step2-2026-09-27.md)). All five W7 gates
>   pass on the 7B (CPU): 4 clients 2.19× the step-1 workers, p99 turn 0.48×, a lone request 1.000× / 1.000×, every
>   reply identical. `-cpu-batch auto` (the default) batches models of ≥ 2 GiB of weights; off on darwin until a Mac
>   7B cell;
> - ~~a Mac 7B MC2 cell~~ — done 2026-09-27: batched B = 4 ÷ J8 N = 4 reads 1.540× / 1.383× (depths 128 / 512), so
>   `-cpu-batch auto` batches on darwin too;
> - ~~fused projections in the batched CPU step (S1)~~ — shipped 2026-09-27 (`000efe2e`): the step 1.058× at B = 4
>   on the 7B, W7 4 clients 1.051×. **Lever B, an aikit small-M kernel for the batched step: PARKED by the owner,
>   2026-09-27.**
>   - The ceiling after S1, estimated from the probes, is ~+11–12% at 4 clients (~+7% at 2), realistically about half
>     that. A lone request gains 0.
>   - It needs an aikit assembly kernel and a release.
>   - Re-open if 7B-class concurrent CPU serving becomes a priority, or if a peer comparison puts the batched CPU path
>     behind. Per-shape fan-out width was tried and closed: a negative result
>     ([`concurrency-mc3c-s1-2026-09-27.md`](../../measurements/concurrency-mc3c-s1-2026-09-27.md));
> - MC3 follow-ons: ~~the 7B's batched-step cost~~ — S4 shipped 2026-09-27, 2 clients 1.121× on the 7B and 1.059× on the
>   1.5B (B ≥ 3 keeps the fragment: no bit-identical kernel beats it); ~~encode-ahead~~ — parked on its measured
>   headroom (~4% / ~1.3%); ~~a per-pass prefill cost cut~~ — CLOSED 2026-09-28 by the owner: measured, prefill passes are 0.151 of a 4-client W7 cell on the 1.5B (0.152 on the 7B), not the estimated ~⅓ ([`mc3-prefill-attr-2026-09-28.md`](../../measurements/mc3-prefill-attr-2026-09-28.md)); ~~CUDA~~ — **SHIPPED 2026-09-27**
>   (`7a44a58e`; [`concurrency-mc3-cuda-2026-09-27.md`](../../measurements/concurrency-mc3-cuda-2026-09-27.md)). All five
>   W7 gates pass on the 1.5B: 4 clients 1.380× the one-at-a-time build, p99 turn 0.755×, a lone request 1.004× /
>   0.999×, every reply identical. The 7B reads 1.826× at 4 clients. It is on under serve's defaults. ~~Follow-on
>   lever: a batched head~~ — **SHIPPED 2026-09-28**: PrefillLastN at M=4 moved 1.75×→2.07× (1.5B), 2.04×→2.34× (7B),
>   0 differing logits at every M tested; W7 rerun queued. Per-call scratch in `prefillCore` is unmeasured, not
>   pursued this pass;
> - ~~the 7B end to end~~ — done 2026-09-27: all five W7 gates pass, 4 clients at 1.785× the serialized build, p99 turn
>   0.592× ([`concurrency-mc3-7b-w7-2026-09-27.md`](../../measurements/concurrency-mc3-7b-w7-2026-09-27.md));
> - ~~MC1 on CUDA~~ — shipped 2026-09-27: 4 clients at 1.250× the one-slot build, every hard gate passes
>   ([`concurrency-mc1-cuda-2026-09-27.md`](../../measurements/concurrency-mc1-cuda-2026-09-27.md)); ~~MC1 on WebGPU~~ —
>   shipped 2026-09-27: 4 clients at 2.805× the one-slot build on the Mac, every hard gate passes
>   ([`concurrency-mc1-webgpu-2026-09-27.md`](../../measurements/concurrency-mc1-webgpu-2026-09-27.md)); ~~the
>   discrete-GPU clamp on real Vulkan hardware~~ — measured 2026-09-28 on nobara: CLEAN (a real
>   out-of-memory buffer allocation on Vulkan fails as a normal Go error, no device loss;
>   [`mc1-webgpu-nobara-2026-09-28/`](../../measurements/mc1-webgpu-nobara-2026-09-28/)); ~~slots-before-context step 2
>   (discrete GPUs)~~ — attempted 2026-09-28, REVERTED: a real in-process release does not return VRAM the same
>   process can reuse (LiveBufferBytes 0, wgpu-native still refuses the identical re-allocation after Close + 1s of
>   Poll), so a rebuild-and-retry can only ever LOSE the working clamped build, not improve it. Discrete GPUs keep
>   today's clamp-only behavior; "slots before context" ships on darwin only. Context against slots on WebGPU:
>   decided 2026-09-27, slots first as on CUDA; done on darwin, discrete GPUs stay clamp-only by design now;
>   ~~CUDA `resolveCtxCapFit` keys on `ResidentContextPinned()`~~ — shipped 2026-09-28 (item 28): a guard-pinned
>   context (not a genuine `-ctx`) now correctly runs fit-by-default and the slots rule, clamped so it can only
>   land at or under the guard's own pin, including below `cudaCtxCapDefault`, which the old code got backwards;
> - ~~context against slots, owner's call~~ — decided 2026-09-27: a slot request shrinks the unpinned default
>   context (`947e06ce`). The 7B now starts at 4096 with 4 slots and reads 1.34–1.35× at its default, where it
>   thrashed on 2 slots at 8192. ~~`Plan`'s conservative weight estimate~~ — fixed 2026-09-27 (`4e230601`): it
>   priced an untied model's host-side embedding table as device memory (4930 against 4444 MB on the 7B). The build
>   now trims an unpinned context to fit every slot, and the 7B starts at 4984 tokens with 4 slots;
> - ~~one MC2 Linux cell to re-run clean~~ — done 2026-09-27: the 0.5B depth-128 cell reads 1.185× clean (was 1.230×
>   overlapping an rsync), still in the owner band.
>
> Closed 2026-09-26:
> - MC1's clamp is pinned by `TestKVSlotsWithin` and logs when it clamps;
> - J8's latency half is 0.51× the serialized p99 under the same load (2.0× a lone request);
> - the J6 re-run is closed by analysis (see MC1's follow-on).
>
> MC2 was the kill-or-earn measurement `roadmap.md` required before any batched decode work, and it earned, so MC3
> went ahead (decision 1) and shipped. MC5 stays parked with its trigger. MC4 is parked, except the one candidate
> registered 2026-09-28, which was built 2026-09-29 and is queued for its CUDA graded run. The speed bars were loosened 2026-09-26, before any measurement, per the owner's
> standing guidance; see "Amendments".
>
> **What this is.** R12's W7 measurement (`docs/measurements/w7-plain-concurrency-2026-09-19.md`,
> Metal, qwen2.5-coder-1.5b q4_k_m, plain 6-turn conversations) recorded goinfer's aggregate
> decode at 60.08 → 36.14 → 36.39 tok/s for 1/2/4 clients, against llama-server's 84.82 → 95.85 →
> 149.72 with `-np N -cb`. That record reads the whole gap as serialization. This doc separates it
> into two problems with different costs:
>
> 1. **The drop from 60 to 36 is probably not serialization.** A single worker serving N
>    conversations in turn should hold its aggregate near the one-client figure. The code suggests
>    the loss comes from the GPU-resident path keeping **one** KV cache per model (`m.resIDs`,
>    `decoder/resident_reuse.go`), so each conversation's turn overwrites the previous one's KV and
>    the next turn re-prefills its whole history — on Metal, where prefill is still ~2.5× behind
>    Ollama at K=512 (red-october R4). The plateau at 2 and 4 clients fits: once turns alternate,
>    every turn misses regardless of N. The CPU path does not have this problem —
>    `sessionLRU` keeps `-kv-sessions` (default 4) conversations warm (**wrong when filed; see MC0: an LRU bug from
>    `e1c867f6` made it thrash too, fixed 2026-09-26**) — but `Session.Generate`'s
>    own comment in `decoder/session.go` records that the resident/GPU path takes
>    `prefillFrom == 0` and relies on the resident's single KV for reuse. **This is read from the
>    code, not measured; MC0 measures it.**
> 2. **Scaling past the one-client figure needs batched decode.** Decode is bandwidth-bound; one
>    forward that carries B sequences reads the weights once for B tokens. That is the mechanism
>    behind llama-server's numbers, and goinfer has no path for it today.
>
> MC0–MC1 address the first problem and stay inside goinfer's stated niche (one generation at a
> time). MC2–MC3 address the second and bend it; see "House rules this doc bends."
>
> **Siblings.** [`red-october.md`](../red-october.md) R12 (owns W7; this doc is its follow-on) ·
> [`task-work-queue-2026-09.md`](../task-work-queue-2026-09.md) (J1 fair admission, which MC3 changes;
> J6 prefix-aware scheduling, measured 1.024× and not shipped, which MC1 may re-open; J8, which
> MC2 absorbs if the owner agrees) · [`../positioning.md`](../../positioning.md) ("not a serving
> engine") · [`../roadmap.md`](../../roadmap.md) §"Decided and parked" (continuous batching) ·
> [`task-fit-to-hardware.md`](../task-fit-to-hardware.md) (owns the fit guard MC1 must price) ·
> [`task-never-swap-2026-09.md`](../task-never-swap-2026-09.md) (MC1's extra KV is anonymous GPU
> memory on a 16 GB Mac).

---

## House rules this doc bends, stated

- `roadmap.md` parks continuous batching and paged attention as "not this engine's weight class,"
  and says N decode workers or batched multi-request decode get **a kill-or-earn measurement
  before any task doc**. W7 measured a peer's batching, not goinfer's. This doc is filed anyway
  because MC0–MC1 are not throughput items (they fix a reuse miss), and because it is useful to
  have MC2's gate and MC3's scope written down before the measurement, so the measurement is held
  to a pre-registered band. **Nothing past MC1 is built until MC2 earns.** MC5 stays parked
  exactly as the roadmap has it.
- `positioning.md` says a model "serves one generation at a time behind a bounded queue." That
  stays true through MC1. MC3 would make it false for dense models on one backend; the positioning
  amendment is owner decision 1.

## Owner decisions

**Decided 2026-09-27, after MC3 shipped: the next three, in order** (owner: "start 1, then Metal step polish then
Chunked prefill"):
1. **MC3 S2: sampled tokens in steps.** A temperature-only token is drawn on-device and so ran as its own call, never
   in a step.
2. **MC3 S3: Metal step polish.** B = 2 reads 1.07×; the step's host time is 2.7 of 25 ms.
3. **Chunked prefill, UNPARKED from MC5** by this decision. Continuous batching and paged KV stay parked. A
   newcomer's prefill is the remaining stall MC3 leaves: it runs whole between steps, pausing every other
   conversation's decode.

**Decided 2026-09-26: 1 is yes, 2 is fold.** One user's parallel agents are in the niche, so MC3 fits and
`positioning.md` gains its sentence (added the same day). J8's N-independent-workers cell runs inside MC2's session;
J8 is not built as a feature.

1. **Does "single-user" include one user's parallel agents?** An agent harness that fans out
   subagents is one user on one machine, but it is not batch-1. If yes, MC3 fits the niche and
   `positioning.md` gains one sentence; if no, this doc ends at MC1 and MC2's result is recorded
   as a finding.
2. **J8 as written, or folded into MC2?** J8 measures N independent decode workers per model. Its
   own text calls batching "the real alternative." MC2 can run J8's cell in the same session at no
   extra build cost, so the two answers land side by side. Recommendation: fold.

---

## MC0 — confirm the diagnosis (measurement only)

**Result, 2026-09-26: CONFIRMED** ([`concurrency-mc0-2026-09-26.md`](../../measurements/concurrency-mc0-2026-09-26.md)).
- Metal at 2 clients reuses 7 tokens per turn (the template lead) against the full history at 1 client, and runs at
  0.88× the 1-client aggregate (65.5 → 57.7 tok/s).
- The CPU control thrashed as well, at 0.69×. The cause was a session-LRU bug from `e1c867f6`: with one resident
  session, the shared preamble counted as reuse.
- MC0 fixed it (`pickSession`). With the fix the CPU control holds 1.00× with full reuse, and 2-client CPU serving is
  1.58× faster.
- MC1's prize on this short workload is ~12% at 2 clients; it grows with history length.

**Why first.** MC1 is only worth building if the thrash is real. The API already reports the
evidence: `usage.prefill_reused_tokens` on every response.

**Run.** `scripts/bench_w7_plain.py` at 1 and 2 clients, Metal, the same checkpoint as W7, with the
per-turn `prefill_reused_tokens` and prompt length recorded for every turn. Add one control cell:
the same two-client run on the **CPU** backend, where `sessionLRU` already holds four
conversations.

**Pre-registered reading.**
- **Confirmed** if, on Metal at 2 clients, turns 2+ report `prefill_reused_tokens` near zero while
  the 1-client run reuses the full prior history, **and** the CPU control's 2-client aggregate stays
  within 0.85–1.0× of its own 1-client figure.
- **Not confirmed** if Metal reuse holds at 2 clients and the aggregate still drops. Then the loss
  is elsewhere (admission overhead, scheduling, memory pressure), MC1 is not built, and this
  section records what the numbers point at instead.

**Cost.** One session on the Mac; no code beyond logging the field the harness already receives.

## MC1 — multi-slot resident KV

**Result, 2026-09-26: SHIPPED on Metal**
([`concurrency-mc1-2026-09-26.md`](../../measurements/concurrency-mc1-2026-09-26.md)).
- On the W7 workload with 4 slots, the aggregate at 2 and 4 clients is 0.989× / 1.007× the 1-client figure.
- The 4-client aggregate is 70.9 against the previous build's 57.0 tok/s (1.24×; the repeat reads 1.22×).
- Each turn prefills only its new user turn at every client count.
- Outputs are bit-identical to each conversation served alone, on the tiny fixture and on qwen2.5-coder-1.5b.
- Recurrent families keep one; VL turns use the bound slot. CUDA and WebGPU shipped 2026-09-27; see "MC1 on CUDA" and
  "MC1 on WebGPU" below.
- The fit guard's clamp is implemented, but no clamping load was run.
- **Fixed 2026-09-26 (found by MC3 S1): the clamp was priced after the build had allocated its own buffers.**
  - Its budget reads live available memory, and the base it compares against (`residentNeedBytes`) already includes
    the weights, so the weights were counted twice.
  - Measured on the 1.5B at a 1024 context: 2 of 8 slots granted (28 MB each) against a 4,044 MB base and 5,531 MB
    live before the build, where 8 fit.
  - `buildResident` now prices the slots first. `TestMetalKVSlots_pricedBeforeTheBuild` stubs the live figure to
    fall by what the device allocates: it grants 2 of 4 before the fix and 4 of 4 after.
  - This errs only toward fewer slots. MC1's logs do not record the count granted, but its W7 cells show every
    turn prefilling only its new user turn at 4 clients, which fewer than 4 slots would have made thrash. So its
    numbers ran at 4 and stand.

**Goal.** N conversations' KV stay resident on the GPU at once, so interleaved turns reuse their
own prefix instead of evicting each other. Still one generation at a time; no batching.

**Design.**
- Replace the model's single resident KV and its bookkeeping (`resIDs`, `resIDsLora`, the recorded
  image blocks) with N slots, each carrying its own copy of that bookkeeping.
- `residentReuseLen` runs against every slot and returns the best match plus its slot; the
  generation binds that slot. No match → the least-recently-used slot is reset and taken.
- **Slot count.** Reuse `-kv-sessions` as the requested count; the backend clamps it to what the
  fit guard allows and says so in the banner. The fit guard's KV term (`residentKVBytes` in
  `metal/backend.go`, and the CUDA and WebGPU equivalents) is multiplied by the slot count.
  Rough size: KV bytes per token = layers × kvDim × 2 (K and V) × bytes per element. For
  Qwen2.5-1.5B (28 layers, kvDim 256, f16) that is about 28 KB per token, ~117 MB per 4k-context
  slot. Larger models scale with their own geometry; the guard does the arithmetic, not a table.
- **Recurrent and hybrid families** (Gated-DeltaNet, KDA, Mamba-style mixers — the set
  `hasRecurrentState` marks) carry per-sequence state that mutates in place. Either each slot holds
  its own state buffers, or these families keep one slot and decline the rest. Start with the
  decline; the resident-prefix-reuse history (`3e45469`, the recurrent-state exclusion that was
  missing once) is the reason to be conservative.
- **LoRA.** A fine-tune entry shares its base's `*decoder.Model`, so base and adapter requests
  thrash the same single KV today. Per-slot `resIDsLora` fixes this as a side effect; no separate
  work.
- **Backends.** Metal first (W7's machine, the owner's machine), then CUDA, then WebGPU. A backend
  that has not been converted keeps one slot.

**Gates (pre-registered).**
- Per-conversation output **bit-identical** to the same conversation served alone — reuse must not
  change bytes, the same bar the existing resident prefix reuse holds.
- On the W7 plain workload, Metal: aggregate at 2 and 4 clients within **0.90–1.0×** of the
  1-client figure (band derived from "serialized with full reuse costs only admission and the
  per-turn suffix prefill"). Below 0.85× at 4 clients is a finding to explain, not a ship.
- Per-turn `prefill_reused_tokens` at 2 and 4 clients equals the 1-client run's, turn for turn.
- The fit guard declines a slot count that does not fit, with a message naming the count it chose.

**Follow-on.** With more than one slot, J6's prefix-aware scheduling has something to exploit (now true on Metal,
CUDA and WebGPU).
**Closed by analysis 2026-09-26, not re-measured:**
- J6's 1.024× (2026-09-15) was measured on the CPU, where the session LRU already held 4 conversations. The "single
  slot" premise above was wrong: J6 had multiple warm prefixes to schedule around, and it still did not pay.
- J6's mechanism was reverted and conflicts with MC3c's N-wide admission.
- MC1 and MC3c give every active conversation, up to the slot or session count, its own warm prefix and its own turn,
  which removes the contention J6 reorders.

**Re-open condition:** more concurrently active conversations than `-kv-sessions`, for example an agent fan-out of 8
or more on one model.
Re-run J6's measurement once after MC1 ships; its 1.024× was taken against a single slot.

### MC1 on CUDA (nobara-pc) — built 2026-09-27, W7 grading pre-registered below before any timing

**Result, 2026-09-27: SHIPPED** ([`concurrency-mc1-cuda-2026-09-27.md`](../../measurements/concurrency-mc1-cuda-2026-09-27.md)).
- Every hard gate passes.
  - Identity: every compared turn matches.
  - Reuse: every client prefills only its new user turn.
  - Ship: the 4-client aggregate reads **1.250×** the one-slot build (174.2 → 217.9 tok/s; pairs 1.250 / 1.255 /
    1.249).
  - Solo guard: p50 1.007×, p99 1.002×.
- The expected band read 1.008× / 1.014×, just above its top edge.
- **The 7B clamp binds:** 2 of 4 slots at the fit-by-default 8192 context, so 4 round-robin clients still thrash
  (1.006×). An exploratory pair at `-ctx 4096` fits 4 slots and reads 1.33×.
- **Owner decision, 2026-09-27: a slot request shrinks the unpinned default context** (`947e06ce`, `ctxForSlots`).
  - The 7B now starts at 4096 with 4 slots and reads 74.7 against 55.6 tok/s (1.34–1.35×, two interleaved pairs).
  - The 1.5B is unchanged: 8192 tokens and 4 slots.
  - An explicit `-ctx` is never shrunk.
- The decoder suite was stopped before timing and re-run after it, by owner decision; no `decoder/` file changed.

**What was built** (`9fddaf7e`; prompt `docs/prompts/nobara-mc1-cuda-2026-09.md`). The CUDA resident implements
`decoder.ResidentKVSlotter`; the decoder half is Metal's, unchanged. Each CUDA specific the prompt named was checked in
the code:
- **Storage.** A slot is a full copy of the per-layer f32 `kc`/`vc` set (one latent buffer for an MLA layer, nothing
  for a DeltaNet layer), `kvBytesForCap` each. `TestResidentKVBytes_matchesCUDAAllocation` now builds every fixture
  with 2 slots and checks each one.
- **Pricing.** CUDA's `checkKVFits` already runs after the weights are on the device and prices KV alone against
  what is left, so the slots are priced there too (`kvSlotsFit`: as many as fit beside the companion reserve and the
  384 MiB margin). Both sides of that comparison exclude the weights, so it cannot count them twice.
  `TestCUDAKVSlots_pricedAgainstWhatIsLeft` stubs the probe to fall by what the device really allocates from a figure
  where exactly 2 fit, and asks for 3. A double count would grant fewer, and pricing before the weights would grant 3.
  On the 1.5B (1.105 GB before KV, 470 MB per slot) it grants 2. At `9fddaf7e` the context was resolved first, as
  before (fit by default picks up to 8192), and the slots were clamped after it. Since `947e06ce` (owner decision,
  same day), a slot request shrinks the unpinned context first; see the result above. A clamp logs its numbers, and the banner names
  it. `TestCUDAKVSlots_clampedBuild` is a real clamping build: 2 of 4 granted, both usable.
- **Graphs.** No change is needed, and none was made: the captured segments (segA/B/C) touch no KV. rope_kv and every
  attention kernel run live in the gap and bind `r.kc[l]`/`r.vc[l]` when issued, so a slot switch is a pointer swap
  on the executor. `TestCUDAKVSlots_graphs` runs the identity scenario with graphs forced on (this box is in the
  Default compute mode) and requires that they stayed on.
- **Host KV writers.** `UploadKV`, the batched prefill, the MLA latent store and the flash-decode/split-KV lanes all
  index the bound slot's slices at call time. Nothing caches a KV buffer across calls.
- **Other per-sequence state.**
  - A block drafter's own KV follows one conversation; the decoder already clears `resDrafterSynced` on a switch.
  - The C′ expert cache is keyed by expert, not by history.
  - Prefill pass lengths and captures are per call.
  - A DeltaNet resident is held to one slot on CUDA as well as by the decoder (`TestCUDAKVSlots_recurrentKeepsOne`).
  - Expert streaming (`MoECacheExperts`) also keeps one slot: that cache takes whatever VRAM is left, so a KV slot
    would cost it expert slots. That is the same trade `resolveCtxCapFit` declines to make with the context.
- **CUDA does not batch.** It has no `ResidentBatchStepper`, so generations still run one at a time; the slots only
  stop the thrash.

**Identity (before timing), all pass.** `TestCUDAKVSlots_interleavedMatchesAlone` covers dense (llama-tiny), sliding
window (mistral-tiny-window) and MLA (deepseek-tiny), plus qwen2.5-coder-1.5b int4 with 48 tokens × 3 turns × 2
conversations. The interleaved ids equal the ids alone. Reuse per turn is 0 / 55 / 105 on the 1.5B, and the one-slot
control thrashes (3 / 3). The tagged CUDA suite: 171 pass, 0 fail.

**W7 grading, pre-registered.**
- *old* = `serve-cuda` @ `b2168bdc` (one slot); *new* = `serve-cuda` @ `9fddaf7e`. Each is built once with
  `CGO_ENABLED=0 go -C cuda build -tags cuda`, and the binary is named by its hash.
- Machine: nobara-pc, RTX 2070 SUPER 8 GB, NVIDIA driver 595.91.07, compute mode Default.
- Workload: `scripts/bench_w7_plain.py --engines goinfer --backend cuda --fixed-nonce`, with
  qwen2.5-coder-1.5b-instruct q4_k_m from `~/models` (NVMe), `-quant int4` and serve defaults. `-kv-sessions` 4 means
  new asks for 4 slots. 6 turns × 128 greedy tokens per client, and a fresh server per cell.
- Cells: 1, then 2, then 4 clients. Each is old/new × 3 pairs, in the order old new new old old new.
- Every cell is idle-gated: load1 ≤ 2.0 from `/proc/loadavg`, no CUDA compute process, and GPU memory within 256 MiB
  of the run's starting baseline.
- Gates (hard unless marked):
  1. Identity:
     - at 1 client, `content_sha` old == new on every turn of every pair;
     - at 2 and 4 clients, new's client 0 == new's 1-client run on every turn;
     - old vs new at 2 and 4 clients is reported only.
  2. Reuse: at 2 and 4 clients, every new client's turn 1 reuses what the 1-client run's turn 1 reuses (0, cold), and
     its turns 2–6 prefill exactly what the 1-client run's do.
  3. *(Expected band, not a gate.)* new's 2- and 4-client aggregate is 0.90–1.0× its own 1-client aggregate. Below
     0.85× at 4 clients is a finding to explain.
  4. Ship: 4-client aggregate new ÷ old, median of 3 pairs, ≥ 1.03×.
  5. Solo guard: 1-client p50 and p99 turn (nearest rank over the cell's 6 turns), new ÷ old, median of 3 pairs,
     ≤ 1.05× each.
- It ships if 1, 2, 4 and 5 hold. A hard-gate miss is recorded as a negative result and not shipped.
- Also reported: one 4-client pair (old, then new) on qwen2.5-7b-instruct q4_k_m. Where the clamp should bind:
  the banner's slot count, and the aggregate.
- Gates are computed by `gates.py`, archived with the raw JSON in `docs/measurements/concurrency-mc1-cuda-2026-09-27/`.

### MC1 on WebGPU (the MacBook) — built 2026-09-27, W7 grading pre-registered below before any timing

**Result, 2026-09-27: SHIPPED** ([`concurrency-mc1-webgpu-2026-09-27.md`](../../measurements/concurrency-mc1-webgpu-2026-09-27.md)).
- Every hard gate passes.
  - Identity: every compared turn matches; old and new also agree on all 108 turns at 2 and 4 clients.
  - Reuse: every client prefills only its new user turn.
  - Ship: the 4-client aggregate reads **2.805×** the one-slot build (9.27 → 26.00 tok/s; pairs 2.642 / 2.805 /
    2.814). 2 clients read 2.803×.
  - Solo guard: p50 1.001×, p99 1.008×.
- The expected band reads 0.998× / 1.000×.
- The win is larger than CUDA's because Qwen2.5 prefills one token at a time on this backend (`PrefillLast` declines
  q/k/v bias off Vulkan), so each thrashing turn re-prefilled its whole conversation at decode speed.
- The slowest turn at 4 clients falls from ~99 s to ~23.6 s. What remains is admission: WebGPU has no batch stepper,
  so generations still run one at a time.
- Owed: the discrete-GPU clamp (a failed allocation plus the headroom probe) on real Vulkan hardware
  (`docs/prompts/nobara-mc1-webgpu-2026-09.md`).
- **Owner decision, 2026-09-27: slots before context on WebGPU too, in two steps.**
  - **Step 1, done (darwin):** `slotsBeforeContext` (`gpu/kv_slots.go`). When more than one slot is requested and
    the context is unpinned, the context gives up positions, down to 4096, until every slot fits under
    `darwinKVSlots`. Below the floor the slots clamp. It is priced from `kvBytesPerPosition`, which is pinned against
    the real allocation on every KV layout.
  - "Unpinned" is the new `decoder.Model.ResidentContextPinned()`: false when the caller left the context 0,
    including when the load-time fit guard auto-pinned it (R13). That pin is a one-slot ceiling, not a choice. An
    explicit `-ctx` is never shrunk.
  - On qwen2.5-coder-1.5b, with the memory available before the build stubbed to room for 4 slots at 8000
    positions, an unpinned load lands on exactly 8000 with 4 slots. An explicit 16384 keeps 16384 with 1 slot
    (`TestWebGPUKVSlots_slotsBeforeContext`).
  - The graded W7 configuration (4 slots fit at 16k) is unchanged.
  - **Step 2 (discrete GPUs):** only if the nobara run shows a failed allocation on Vulkan is clean. What fits is
    learned from the build there, and the prompt carries the pre-registered decision rule.
    - **Outcome, 2026-09-28 (items 26–27):** the clamp is clean on real Vulkan. But step 2 was attempted and
      REVERTED: an in-process release does not return VRAM the process can reuse. Discrete GPUs stay clamp-only by
      design, and "slots before context" ships on darwin only (the header's MC1 entry).
  - ~~CUDA's `resolveCtxCapFit` still treats a guard-pinned context as explicit~~ — fixed 2026-09-28 (item 28,
    item 4 of the same prompt). See its own result below.

**Item 28, SHIPPED 2026-09-28: CUDA `resolveCtxCapFit` now keys on `ResidentContextPinned()`, not `request > 0`.**
Before this, `request > 0` alone made the function return `resolveCtxCap(request, modelCtx)` untouched — the same
path a genuine `-ctx` takes — whether `request` came from the caller's own choice or from the load-time fit
guard auto-pinning a smaller context for an unrequested load (R13). A guard pin is a ceiling the guard already
proved safe against host RAM, not a choice; MC1's slots rule is supposed to shrink exactly that kind of
context, and it silently never ran for it.

- **The bug had a second edge fit-by-default's own comments hadn't anticipated:** `decoder.ctxFloor` (the
  general fit guard's own floor, 2048) sits BELOW `cudaCtxCapDefault` (4096, CUDA's historical floor). The old
  code's `return cudaCtxCapDefault` fallbacks would have floored a guard pin of, say, 2048 back UP to 4096 —
  raising the context past exactly what the guard had just determined host RAM could hold. Fixed by having both
  the ceiling (`candidate`) and the floor CUDA's own logic returns to follow a guard pin down together, never
  just the ceiling.
- **Fix:** `resolveCtxCapFit`'s guard is now `request > 0 && m.ResidentContextPinned()` for the genuine-pin
  shortcut; a guard pin (`request > 0`, not pinned) clamps both `candidate` and a new `floor` variable to
  `request` before Plan runs, and `ctxForSlots` takes that `floor` as a parameter instead of hardcoding
  `cudaCtxCapDefault`, so the slots search still respects a sub-4096 guard pin as its own lower bound.
- **Found and fixed two existing tests that were passing for the wrong reason.** Both `TestResolveCtxCapFit_shortcuts`
  (`cuda/resident_cap_test.go`) and `TestResolveCtxCapFit_slotsShrinkTheContext` (`cuda/kv_slots_test.go`) called
  `resolveCtxCapFit(m, 8192, ...)` on an `m` loaded WITHOUT `ResidentContext` set — i.e. `m.ResidentContextPinned()`
  was already false before this fix existed to check it, so under the corrected logic these cases would silently
  stop testing "a genuine pin never shrinks" and start testing the (different, also-correct) guard-pin path
  instead, without a single assertion changing shape — `fitDefaultCtx` is itself 8192 and llama-tiny is small
  enough that both paths happen to land on 8192 for the unshrunk case. Fixed by loading a genuinely-pinned `m`
  (`Options.ResidentContext: request`) for every case meant to exercise the explicit-pin shortcut, with a
  `m.ResidentContextPinned()` assertion on the setup itself so a future edit here can't reintroduce the same
  silent drift. Added a new case pinning the sub-cudaCtxCapDefault floor-follows-down behavior directly
  (`request=2048`, unpinned `m`, want 2048 — the exact scenario the old code got backwards).
- Full tagged CUDA suite: 177 pass, 0 fail (same as item 9's baseline). `gofmt`, `go vet` and the pinned
  staticcheck 0.8.0 all clean on `-tags 'cuda goinfer_testhooks'`.

**Item 26, RESULT 2026-09-28 (nobara-pc, RTX 2070 SUPER 8 GB, WebGPU through wgpu-native's Vulkan backend, driver
595.91.07): CLEAN — go to step 2.** Logs: [`docs/measurements/mc1-webgpu-nobara-2026-09-28/`](../../measurements/mc1-webgpu-nobara-2026-09-28/).

- **1. Identity on Vulkan.** `mistral-tiny-window` ran here for the first time anywhere (`identity.log`) and passed
  — the sliding-window layout has now been through the interleaved-vs-alone scenario. The full non-heavy `gpu`
  suite: **135 pass, 0 fail** (`full-gpu-suite.log`; the Mac's own run was 132/0 — the difference is fixture
  availability, not a platform gap). The two darwin-only tests (`_pricedAgainstMemory`, `_slotsBeforeContext`) skip
  here as expected. `TestWebGPUKVSlots_interleavedMatchesAlone` on the real 1.5B (`GOINFER_HEAVY_TESTS=1
  GOINFER_WEBGPU_KVSLOTS_MODEL=...`) reused 0/55/105 per turn, matching the CUDA figure exactly
  (`identity-1.5b.log`).
- **2. The real clamp** (`TestWebGPUKVSlots_realClamp`, `gpu/kv_slots_test.go`, committed). Loads with 4 slots
  requested at the default 16k f32-KV context, runs 2 turns × 32 tokens on every granted slot, checks it against a
  fresh one-slot load, and checks `LiveBufferBytes()` and `nvidia-smi` return to their pre-load baseline after
  `Close`.
  - **qwen2.5-coder-1.5b** (`~/models`, ~0.94 GB/slot): all 4 requested slots fit, no clamp message
    (`realclamp-1.5b.log`).
  - **qwen2.5-7b-instruct** (`~/models`, ~4.4 GB weights + ~1.79 GB/slot): **1 of 4 granted.** Slot 1's build (the
    second slot; slots are 0-indexed) failed on layer 8's K/V buffer — a raw allocation failure, not the 384 MiB
    headroom probe (the probe never ran) — with wgpu-native's own error text: `wgpuDeviceCreateBuffer ... label =
    'kvcache-slot' ... Not enough memory left` (`realclamp-7b.log`). That answers this job's first question: **a
    real out-of-memory buffer allocation on Vulkan fails as a normal Go `error` from `TryCreateBuffer`** — no
    panic, no device loss, no corrupted state. Slot 0's generation is bit-identical to a fresh one-slot load,
    `LiveBufferBytes()` matched exactly, and `nvidia-smi` read +3 MiB after `Close` (well under the 128 MiB margin
    `cuda/lifecycle_test.go` already uses for the same kind of check).
  - Read against the pre-registered rule: the build's own clamp message is the "logged reason"; the granted slot(s)
    are bit-identical on both checkpoints; nothing leaked on either checkpoint; the only "validation error" that
    appeared is the deliberate, handled `TryCreateBuffer` probe the clamp exists to catch, not an unexpected later
    failure — no subsequent allocation, dispatch, generation or `Close` failed on either checkpoint. That is
    **Clean** by the rule's own text ("a LATER allocation or dispatch fails" is what would make it not clean).
- Step 2 (below) is therefore in scope, per the rule.

**Item 27, RESULT 2026-09-28: attempted, found UNSAFE on real hardware, REVERTED — not shipped.**
Step 1 being clean does not make step 2 automatic; step 2 asked a genuinely different question and the answer
was no, for a reason step 1 could not have surfaced.

- **Design tried.** `BuildResident` split into a thin wrapper and `buildResidentAt(m, forceCtx int)`, the whole
  original body plus one override point (`forceCtx > 0` replaces `slotsBeforeContext`'s resolution outright). On a
  shortfall (`rd.KVSlots() < want`, unpinned, above the floor), the wrapper priced the largest context that would
  fit `want` slots in the room the first build actually measured (`ctxWhereSlotsFit`, reusing the same search
  `slotsBeforeContext`'s darwin path already uses), closed the first resident, and called `buildResidentAt` again
  at that smaller context — the exact shape the prompt suggests, built to reuse the already-tested eligibility/
  weights/KV code rather than a second, easier-to-drift path.
- **First bug, found immediately: a self-deadlock.** `buildResidentAt` holds `b.mu` (a plain, non-reentrant
  `sync.Mutex`) for its whole body via `defer b.mu.Unlock()`. The first cut called the retry from INSIDE
  `buildResidentAt`, before that defer could run — `TestWebGPUKVSlots_realClamp/qwen2.5-7b` hung past its 10-minute
  timeout instead of failing fast (one goroutine, blocked on its own lock). Fixed by moving the retry to the
  wrapper, entirely outside any call that holds `b.mu` — structurally impossible to deadlock this way, not just
  observed to not deadlock.
- **Second bug, the one that killed the approach: real VRAM is not returned for reuse inside the same process.**
  With the deadlock fixed, the retry reliably reproduced on the real 7B: `Close()` on the first (1-slot) resident,
  then the SAME weight upload that had JUST succeeded (~4.4 GB, unchanged) failed immediately with wgpu-native's
  own `wgpuDeviceCreateBuffer ... Not enough memory left` — the identical error the ORIGINAL allocation failure
  used to justify the clamp in the first place. `LiveBufferBytes()` (this package's own buffer-object accounting,
  independent of the driver) read **0** immediately after `Close()` — so nothing was leaked at the Go/wgpu-object
  level; the buffers were correctly released as objects. The VRAM itself was not usable again regardless: neither
  one `device.Poll(true, nil)` (this codebase's established GPU/CPU fence, used everywhere else for exactly this
  kind of wait) nor ten of them with 100 ms sleeps between each (1 full second) changed the outcome. This reads as
  wgpu-native's Vulkan allocator not returning freed device memory to a state a same-process, back-to-back
  allocation can draw from — not a bug in this codebase's release bookkeeping, which the zero `LiveBufferBytes()`
  reading rules out directly.
- **Why this had to be reverted, not shipped with a caveat.** The failure mode is not "no better than before" —
  it is WORSE. On the retry's failure, `buildResidentAt` returns a decline, and the wrapper returned that decline
  as-is: `decoder.Model.withResidency()` saw a failed resident build and fell back to the CPU/staged path
  ENTIRELY — losing the working 1-of-4-slot GPU-resident build the FIRST attempt had already produced, correctly,
  before step 2 ever touched it. A user who would have gotten a clamped-but-GPU-resident 1-slot serve got the
  CPU-staged path instead, silently, the moment their card was too small for the requested slot count at the
  default context. Recovering by rebuilding at the ORIGINAL context after a failed retry was considered and
  rejected: given the driver-level reclaim problem just measured, there is no confidence that attempt would
  succeed either, on a device now in whatever state the failed retry left it.
- **Reverted in full**, before commit: `buildResidentAt`/`forceCtx`, `webgpuDiscreteSlotsShrink`, its test, the
  `TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches` watch-target update, and the retry call in
  `BuildResident`. `gpu/` is back to exactly item 26's committed state (`8c9b647f`); `git diff` against it is
  empty for every file step 2 touched.
- **Proposed default, left to the owner, per the prompt's own decision rule for this outcome:** keep today's
  behavior unconditionally on discrete GPUs — the clamp grants as many slots as fit at the requested context and
  stops there (a real, working, GPU-resident build, just fewer slots than asked), and a user who wants more slots
  on a card too small at the default context passes a smaller explicit `--ctx` themselves and reloads, rather than
  the backend silently retrying in-process. "Slots before context" therefore ships on darwin only; a discrete GPU
  is not a smaller version of the same mechanism, because darwin's version never allocates before it prices.
- **What would actually be needed to revisit this**, if the owner wants slots-before-context on discrete GPUs
  later: either (a) probe context sizes downward BEFORE any allocation, the way darwin does — which needs a real
  free-VRAM query this backend does not have (unlike CUDA's driver `MemInfo()`; that gap is exactly why step 2 had
  to be measured after the fact instead of priced up front), or (b) accept that a shrink means restarting the
  process at a smaller `-ctx`, never an in-process rebuild — which is closer to "propose a safe default" than an
  engineering fix, since it hands the decision back to the operator rather than solving it silently.

**Item 29, PRE-REGISTERED 2026-09-28, before any timing: W7 on Vulkan (optional, reported only).** Items 26–28 are
all resolved (26 clean, 27 measured-and-reverted with a clear writeup, 28 shipped), so this is unblocked per the
prompt's own §5, run at the owner's word — never on its own initiative (queued overnight, not run by day; see
CLAUDE.md's run budget).

- **Hypothesis, from the prompt itself, stated before running:** the win should look more like CUDA's 1.25× than
  the Mac's 2.8×. Qwen2.5 has a batched prefill on Vulkan (`PrefillLast` admits q/k/v bias there, unlike Metal,
  where it declines and every thrashing turn re-prefills one token at a time at decode speed) — so a thrashing
  turn here costs less to begin with, and MC1 has less thrash to remove.
- **old** = `serve-webgpu` at `68f2dbdf` (one slot, pre-MC1); **new** = `serve-webgpu` at `a08bc26d` (this tree,
  item 28 included). Each built once with `go -C gpu build -tags gpu -o <bin> ./cmd/serve`, named by its hash.
- **Machine:** nobara-pc, RTX 2070 SUPER 8 GB, WebGPU through wgpu-native's **Vulkan** backend
  (`oliverbestmann/webgpu` v1.36.0 — same version the Mac graded), NVIDIA driver 595.91.07.
- **Workload:** `scripts/bench_w7_plain.py --engines goinfer --backend webgpu --fixed-nonce`, with
  qwen2.5-coder-1.5b-instruct q4_k_m from `~/models` (NVMe — never `/srv/models`), `-quant int4` and serve
  defaults. `-kv-sessions` 4 means new asks for 4 slots at the default 16k f32 context (~0.94 GB/slot). 6 turns ×
  128 greedy tokens per client, a fresh server per cell.
- **Cells:** 1, then 2, then 4 clients. Each is old/new × 3 pairs, in the order old new new old old new.
- **Idle gate: the CUDA one, not the Mac's** (this job's own §5 instruction) — `/proc/loadavg` load1 ≤ 2.0, GPU
  compute-process count at or below the run's starting baseline, and GPU memory within 256 MiB of the starting
  baseline (`concurrency-mc1-cuda-2026-09-27/run-w7.sh`'s gate, adapted in
  `concurrency-mc1-webgpu-nobara-2026-09-28/run-w7.sh`).
- **Gates (the Mac cell's, hard unless marked — same arithmetic, `gates.py` unchanged):**
  1. Identity: at 1 client, `content_sha` old == new on every turn of every pair; at 2 and 4 clients, new's
     client 0 == new's 1-client run on every turn; old vs new at 2 and 4 clients is reported only.
  2. Reuse: at 2 and 4 clients, every new client's turn 1 reuses what the 1-client run's turn 1 reuses (0, cold),
     and its turns 2–6 prefill exactly what the 1-client run's do.
  3. *(Expected band, not a gate.)* new's 2- and 4-client aggregate is 0.90–1.0× its own 1-client aggregate. Below
     0.85× at 4 clients is a finding to explain.
  4. Ship: 4-client aggregate new ÷ old, median of 3 pairs, ≥ 1.03×.
  5. Solo guard: 1-client p50 and p99 turn (nearest rank over the cell's 6 turns), new ÷ old, median of 3 pairs,
     ≤ 1.05× each.
- **This item is "reported only" per the prompt's own §5**, not a ship/park decision: 1, 2, 4 and 5 are graded and
  stated as pass/fail exactly like the shipped CUDA/Mac cells, but nothing in this codebase currently branches on
  the verdict — WebGPU's MC1 already shipped 2026-09-27 regardless of this result. A hard-gate miss is a finding
  to record, not a blocker.
- Also reported: the slot count new's banner states, and whether item 26's discrete-GPU clamp (1 of 4 on the real
  7B at the default context) visibly costs anything in THIS workload's 1.5B cells, which fit all 4 slots.
- Estimate: the graded CUDA/Mac cells each ran 18 server starts (3 cell-sizes × 6 keys) at ~1–2 min per cell
  (6 turns × 128 tokens, plus idle-gate settling) — roughly 30–45 min end to end on comparable hardware. Over
  CLAUDE.md's ~10-minute daytime quick-check bound: **queued to the night queue**
  (`scripts/night.py add mc1-webgpu-w7-nobara --est 45 ...`), not run by day.
- Raw output and gates land in `docs/measurements/concurrency-mc1-webgpu-nobara-2026-09-28/`.

**What was built** (`gpu/kv_slots.go`). The WebGPU resident implements `decoder.ResidentKVSlotter`; the decoder half
is Metal's, unchanged.
- **A slot is a whole decode runner.** This is not a pointer swap, as it is on CUDA and Metal. `newDecodeRunner` bakes
  every KV buffer into bind groups it builds once: `ropeStore`, `vStore`, `qkvFinalize`, the attention binds, and
  the MLA store and attention. A runner can therefore only use the caches it was built over.
  - Each slot has a `runModel` whose layers share every weight buffer with slot 0 and own only their KV: K and V
    at the resident precision, the int8 scales, and the MLA latent.
  - Each slot also has its own runner and its own lazily grown `ForwardN` pool.
  - `UseKVSlot` swaps the bound `rm`/`runner`/`batch`. `UploadKV`, `Forward`, `ForwardN` and `PrefillLast` read those
    at call time, so none of them changed.
- **Adapters.** `generateInto` binds an adapter before it acquires the slot. A switch therefore binds the adapter on
  the new slot's runner first (on an error the binding is unchanged), then clears the old runner's copy.
- **Pricing.** WebGPU has no free-memory query.
  - On darwin, where device memory is host RAM, the count is clamped against two ceilings, the smaller of which
    applies, as in Metal's guard:
    - 70% of RAM, against the weights, their host copy and every slot;
    - the memory available when the build started (read before its first upload), against the weights and the
      slots. The host copy is not counted twice here.
  - Elsewhere, a slot whose allocation fails stops the count at the slots already built. Each slot must also leave a
    384 MiB probe allocatable, CUDA's margin, for the per-call prefill scratch.
  - Both clamps log the count they chose, and the banner names it.
  - A slot request does **not** shrink the resident context here, unlike CUDA's `947e06ce`. The WebGPU context is
    `WebGPUCtxCeiling`'s fixed 16k (f32 KV), clamped to the model's own window or `-ctx`. Applying "slots before
    context" to this backend is an owner decision, open.
- **Families.** A runModel carrying Mamba or DeltaNet state keeps one slot, on WebGPU as well as by the decoder.

**Identity (before timing).** `gpu/kv_slots_test.go`:
- The interleaved-vs-alone scenario with a one-slot thrash control runs on llama-tiny with f32, int8 (scales) and
  f16 KV, and on deepseek-tiny (MLA). mistral-tiny-window is not built on this Mac (gitignored fixture), so it skips
  here; nobara has it.
- The same scenario runs with conversation A on a LoRA adapter.
- A slot's `ForwardN` pool never serves another slot.
- Every per-sequence buffer is the slot's own and every weight is shared.
- The recurrent families (qwen35-tiny, nemotron-tiny) keep one slot.
- A clamping build (a failure injected halfway through slot 2 of 4) grants 2, both usable, with zero live device
  bytes after Close.
- Both darwin ceilings are exercised on a real build, and there is an arithmetic table.
- Mutation-checked: dropping the adapter move fails the adapter scenario on A and on B, and sharing slot 0's KV
  fails identity and the `ForwardN` test.
- On qwen2.5-coder-1.5b (48 tokens × 3 turns × 2 conversations), the result is recorded in the measurement.

**W7 grading, pre-registered.**
- *old* = `serve-webgpu` at `68f2dbdf` (one slot); *new* = `serve-webgpu` at the MC1-WebGPU commit. Each is built
  once with `go -C gpu build -tags gpu -o <bin> ./cmd/serve`, and the binary is named by its hash.
- Machine: the MacBook Pro (M1 Pro, 16 GB, macOS 26.6.2), WebGPU through wgpu-native's Metal backend
  (`oliverbestmann/webgpu` v1.36.0).
- Workload: `scripts/bench_w7_plain.py --engines goinfer --backend webgpu --fixed-nonce`, with
  qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`, `-quant int4` and serve defaults.
  - `-kv-sessions` 4 means new asks for 4 slots, at the default 16k f32 context: ~0.94 GB of KV each.
  - 6 turns × 128 greedy tokens per client, and a fresh server per cell.
  - Prefill on this backend is per token for Qwen2.5: `PrefillLast` declines q/k/v bias off Vulkan. So a thrashing
    turn re-prefills its whole conversation one token at a time.
- Cells: 1, then 2, then 4 clients. Each is old/new × 3 pairs, in the order old new new old old new.
- Every cell is idle-gated: load1 ≤ 2.0 (`sysctl vm.loadavg`), and no other serve process. The bench also waits for
  3000 MB free before each server.
- Gates, CUDA's (hard unless marked):
  1. Identity:
     - at 1 client, `content_sha` old == new on every turn of every pair;
     - at 2 and 4 clients, new's client 0 == new's 1-client run on every turn;
     - old vs new at 2 and 4 clients is reported only.
  2. Reuse: at 2 and 4 clients, every new client's turn 1 reuses what the 1-client run's does (0, cold), and its turns
     2–6 prefill exactly what the 1-client run's do.
  3. *(Expected band, not a gate.)* new's 2- and 4-client aggregate is 0.90–1.0× its own 1-client aggregate. Below
     0.85× at 4 clients is a finding to explain.
  4. Ship: 4-client aggregate new ÷ old, median of 3 pairs, ≥ 1.03×.
  5. Solo guard: 1-client p50 and p99 turn (nearest rank over the cell's 6 turns), new ÷ old, median of 3 pairs,
     ≤ 1.05× each.
- It ships if 1, 2, 4 and 5 hold. A hard-gate miss is recorded as a negative result and not shipped.
- Also reported: the slot count new's banner states. If the darwin clamp grants fewer than 4 at the defaults, the
  gates still read at the defaults, and the clamp is the finding.
- Gates are computed by `gates.py` (CUDA's, unchanged in its arithmetic), archived with the raw JSON in
  `docs/measurements/concurrency-mc1-webgpu-2026-09-27/`.

**Estimate.** One to two weeks across the three backends, mostly bookkeeping and the fit guard.

## MC2 — kill or earn: batched decode on CPU

**Result on the Mac, 2026-09-26: EARN** ([`concurrency-mc2-2026-09-26.md`](../../measurements/concurrency-mc2-2026-09-26.md)).
- Batched B = 4 against the production single-token forward: 1.945× / 1.690× on the 0.5B (depth 128 / 512) and 2.042× /
  1.798× on the 1.5B, every rep above 1.25×.
- The prototype `decodeMultiStep` is bit-identical to single-sequence decode (the tiny fixture and the 0.5B at int4 and
  int8int8), and costs 0.97–1.00× at B = 1.
- J8's cell, 4 independent workers, reaches 2.00–2.48× aggregate. Its per-request latency is not measured.
- **Linux CPU (`nobara`, 2026-09-26 evening):**
  - the 1.5B earns (1.40–1.48×, where batching beats the workers' 1.33–1.36×);
  - the 0.5B falls in the owner band (1.11–1.23×, where the workers win at 1.64–1.73×);
  - no cell kills;
  - the prototype's M = 1 path is 0.76–0.93× production decode on amd64, so a tuned small-M kernel would lift batching.

**S0, 2026-09-26** ([`concurrency-mc2-s0-2026-09-26.md`](../../measurements/concurrency-mc2-s0-2026-09-26.md)): on the
Mac's CPU, one existing forward carrying n rows (`forwardN`, bit-identical to n sequential forwards) costs 1.77–1.93×
a one-row forward at n = 4 (int4, 0.5B and 1.5B, depth 128 and 512). Batching's ceiling at B = 4 is therefore ≈ 2.1–2.3×,
far above the earn line. It is an upper bound: B separate sequences do not share attention the way one sequence's
rows do. The registered grade still needs the multi-sequence prototype below. On the identity question below:
`forwardN`'s own contract already shows the M = B projection kernels are bit-identical to M = 1 row for row, so the
identity gate applies.

**Question.** Does one forward carrying B sequences beat B sequential forwards by enough to be
worth MC3's cost? Asked on CPU first because the batched matmul already exists in aikit
(`MatmulBTW4A8Batch`), each `decoder.Session` already owns a separate KV cache, and the result can
be checked token for token.

**Build (prototype, behind a test or a bench, not the serving path).**
- A step function that takes B sequences, each with its own `Session` cache and position, and
  produces one token each: projections as one M=B batched matmul; attention per sequence over its
  own KV length (a loop is fine for the measurement); per-sequence RoPE position and sampling.
- Plain dense families only.
- Note for the builder: `MatmulBTW4A8Batch` has no row4 tile (see the comment in
  `decoder/weightmat.go`), so small M may underperform what a tuned kernel would do. Record the
  result with that caveat rather than tuning inside the measurement.
- Check first whether W4A8 activation quantization is per row. If it is, batched output should be
  bit-identical to single-sequence decode; if not, pre-register a reference gate instead before
  measuring.

**Cells.** Mac CPU and Linux CPU; 0.5B and 1.5B int4; B = 1, 2, 4, 8. If decision 2 is "fold,"
add J8's N-independent-workers cell at N = 2 and 4 on the same box in the same session.

**Pre-registered gate.** Aggregate tok/s at B=4 vs B=1: **earn at ≥ 1.6×**, projection band
1.6–3.0× (weights read once per step; the ceiling is set by where the CPU turns compute-bound).
**Kill below 1.3×.** Between the two, the result goes to the owner with the numbers. Per-sequence
output must meet whichever identity or reference gate was registered above; a throughput win that
fails it is a kill.

**Estimate.** One to two weeks.

## MC3 — batched decode on Metal (only if MC2 earns and decision 1 is yes)

**S1, 2026-09-26: the batched step holds in sequence, bit-identically**
([`concurrency-mc3-s1-2026-09-26.md`](../../measurements/concurrency-mc3-s1-2026-09-26.md)).
- A test-only step runs production's per-sequence kernels (norms, RoPE, KV, attention, SwiGLU) and S0's batched
  matmuls, each sequence on its own MC1 slot.
- It matches production's single-token forward on every logit: 1.5B and 7B, 4 sequences × 12 steps, 0 differ.
- In sequence, B = 4 reaches 1.73–1.84× one stream's aggregate (the step costs 2.1–2.3× a token) and B = 8 reaches
  2.4–3.2×, within ~5% of S0's projection.
- B = 2 is a wash (0.98–1.10×) and B = 1 loses (0.51–0.64×), so production's path runs below B = 3.
- S1 also found and fixed an MC1 slot-clamp double count.

**S0, second pass, 2026-09-26: the matrix units pay AND stay bit-identical, so MC3 keeps the identity gate**
([`concurrency-mc3-s0-2026-09-26.md`](../../measurements/concurrency-mc3-s0-2026-09-26.md)).
- The kernels have an 8-token fragment column, weights built in registers through `thread_elements()`, and exact
  integer group sums in the f32 fragments.
- Production's reduction order is reproduced: residue-ordered partials, then `simd_sum`'s own xor-butterfly tree,
  which was measured on the M1 Pro.
- They carry 8 sequences for 1.0–2.7× one production GEMV. That is 2.9–8.0× faster than 8 sequential GEMVs, with 0
  differing outputs on every 1.5B and 7B shape. The int8 LM head is integer-exact in any order.
- The tree form needs a build-time self-check, with a robust `simd_sum`-per-output fallback (2.1–3.0×).
- A fragment costs the same at M = 2 as at M = 8, so M = 2 is about a wash.
- Projected (not measured) for the 1.5B at depth 128: B = 4 ≈ 1.9× one stream's aggregate, with the step ≈ 2.1× a
  single step. That projects to clearing the W7 aggregate gate and **failing the p99 bar as written** (≤ 1.5× a lone
  request). At the same 4-client load it projects to ~0.5× MC1's p99.

**S0, first pass, 2026-09-26 (superseded in its conclusion above)**
([`concurrency-mc2-2026-09-26.md`](../../measurements/concurrency-mc2-2026-09-26.md), MC3 S0).
- A bit-identical small-M W4A8 kernel on Metal (R18's rows kernel extended to M activation rows) amortises little. On
  the dominant gate/up shape M = 4 gains 1.30× (1.5B) and 1.15× (7B), because these kernels are bound by per-weight
  work. MLX's masked form shares the unpack but spills registers.
- It concluded that a paying MC3 likely needed the matrix units with f16 products, and so a fidelity gate. The second
  pass keeps the matrix units and drops the fidelity gate: int4 and int8 operands are exact in half, and the order
  is reproduced.
- The alternative is to put the batching effort on the CPU path, where MC2 earns bit-identically.

**Scope.** Dense families, one backend, one adapter per batch. Everything else declines to the
serialized MC1 path, the same shape fast prefill used.

**New pieces.**
- **Small-M W4 GEMM** for 2–8 rows. Neither the decode GEMV nor the prefill f16-MMA tile fits this
  shape, and the prefill GEMM is already Metal's weakest kernel (R4). Prior-art sweep required:
  MLX's and llama.cpp's Metal small-batch matmul paths.
- **Multi-sequence attention** over B slots of different lengths. MC1's slots are the KV storage,
  which is why MC1 comes first. The CUDA flash-decode multi-row lane (`cuda/flash_decode.go`)
  batches verify rows of **one** sequence at consecutive positions; it is a reference for the
  kernel shape, not a ready answer.
- **Per-sequence** positions, sampling, stop tokens and grammar masks (the masker runs host-side;
  one per sequence).
- **Scheduler.** Admission (J1) changes from "grant one turn" to "join the next step at a token
  boundary." In this stage a newcomer's prefill runs between steps, whole; chunked prefill
  interleaved with decode is MC5.

**Declines to the serialized path:** MoE, recurrent and hybrid families, vision turns,
speculative decode (a sequence in a batch runs without spec), and a batch whose requests name
different adapters.

**Gates (pre-registered).**
- Per-sequence output meets the identity or reference gate chosen in MC2, on Metal.
- W7 plain workload, 4 clients: aggregate **≥ 1.5×** MC1's 4-client figure (loosened to ≥ 1.2×, see "Amendments").
- ~~p99 per-request latency no worse than **1.5×** MC1's 1-client figure (J8's latency bar, kept).~~ Replaced before the
  build, by the owner 2026-09-26 (see "Amendments"). Both of these are hard:
  - **p99 under load:** MC3's 4-client per-turn p99 must be no worse than MC1's 4-client per-turn p99 (the same offered
    load).
  - **the solo guard:** a lone request (1 client) under the MC3 build must be within noise of MC1's 1-client figure.
    That means a p50 and p99 per-turn ratio ≤ 1.05×, as the median of 3 paired, interleaved, idle-gated cells.

  The 4-client p99 ÷ a lone request is reported, not gated.

**Estimate.** Four to eight weeks for Metal. A CUDA port follows only on its own measurement.

**SHIPPED 2026-09-26: all five W7 gates pass** ([`concurrency-mc3-2026-09-26.md`](../../measurements/concurrency-mc3-2026-09-26.md)).
- The build is `d4b708b5`, plus `2b1cc280` (serve printed the concurrency line before deciding it) and `d2225ec4` (the
  straggler window phase-locked 4 generations into 3-wide + solo runs). Both fixes were found before the grading;
  the grading ran on `d2225ec4`.
- 4 clients: 1.593× (1.583–1.599) aggregate, and p99 turn 0.659×.
- A lone request: p50 1.002×, p99 1.004×.
- Identity and reuse equal on every turn of all 14 cells. 2 clients read 1.071× (reported).

**The 7B end to end — W7 grading, pre-registered 2026-09-27 before any W7 timing on the 7B.** Every W7 grading so far ran
the 1.5B. On the 7B, MC3 has been measured only below serve: S0's kernels, and S1's step in sequence (1.76–1.84× at
B = 4, before S2 and S3 changed the step).
- *old* = `serve-metal` at `9efc3185`, pre-MC3: MC1 plus R18b plus the slot-pricing fix, serialising its generations.
- *new* = at `cc5f8c2c`, the current stack: MC3, S2, S3 and chunked prefill.
- Both run at serve's defaults (`-max-concurrent` 4, `-kv-sessions` 4, context 4096). Both banners, checked without
  timing, keep 4 conversations resident on the 7B; *new*'s also reports batched decode.
- **Before any timing:** `TestMC3Step_bitIdentical`, `_Deep`, `_drawsMatchForwardSample` and both concurrent-vs-alone
  tests (greedy, sampled) run on the 7B. A difference there is a bug and stops the grading.
  - The tests load its `.int4.metal.giw` with weights aliased, as S1 did: the `.gguf`'s own int4 load path does not fit
    this Mac.
  - The sampled test had hard-coded the 1.5B. It now reads `GOINFER_METAL_MC3_MODEL` like the others, plus
    `GOINFER_METAL_MC3_TOKENIZER`, since a bundle is named with its `.gguf` for the tokenizer.
- **Workload:** `scripts/bench_w7_plain.py` with `BENCH_W7_MODEL` = qwen2.5-7b-instruct q4_k_m from `~/models` (serve
  aliases its `.int4.metal.giw`). 6 turns × 128 greedy tokens per client, `--fixed-nonce`, a fresh server per cell,
  idle-gated per cell (load1 ≤ 2.0).
- **Cells:** 4, then 1, then 2 clients, each old/new × 3 pairs in the order old new new old old new.
- **Gates.** 1, 2, 4 and 5 are hard.
  1. identity: every turn's `content_sha` equal between new and old;
  2. reuse equal;
  3. 4-client aggregate new ÷ old, median of 3 ≥ 1.2× (MC3's bar);
  4. 4-client p99 turn new ÷ old, median ≤ 1.0;
  5. lone request p50 and p99, new ÷ old, median ≤ 1.05× each.
- **Reported, not gated:** the 2-client cells, and the 4-client p99 ÷ a lone request's.
- **Decision.**
  - All pass: MC3 is recorded as paying on the 7B, and the open "7B end to end" item closes.
  - Gate 1 or 2 fails: a bug, fixed before anything else.
  - Aggregate 1.03–1.2× with the hard gates passing: goes to the owner with a recommendation.
  - Below 1.03×, or gate 4 or 5 fails: the owner decides whether batched decode stays on by default for a model of
    this size.
- **Graded 2026-09-27: all five pass**
  ([`concurrency-mc3-7b-w7-2026-09-27.md`](../../measurements/concurrency-mc3-7b-w7-2026-09-27.md)).
  - 4 clients at **1.785×** (25.5 → 45.5 tok/s), p99 turn 0.592×.
  - A lone request 1.002× / 1.014×; every reply identical.
  - 2 clients read 1.018×. In sequence, a step on the 7B costs ~1.9× a token at any B ≤ 4, so B = 2 breaks even.
    The lever for this size is the batched matmuls' own cost.

**MC3 S4 — the 7B's batched-step cost, and encode-ahead (2026-09-27). Registered before any build or W7 timing.**

*Exploratory, measured first* (`docs/measurements/mc3-7b-lever-2026-09-27/`, M1 Pro):
- **Why a 7B step costs ~1.9× a token at any B ≤ 4.** The step's matmul kernels (`mc3_bt`, `mc3_btd`) run four 8×8×8
  `simdgroup_multiply_accumulate` per 32-k group over all 8 token columns, whatever B is. On the 7B's large shapes,
  production's GEMV is bandwidth-bound (gate/up 0.433 ms, ~157 GB/s), and the fragment is ALU-bound at a fixed cost
  (1.17 ms). The fragment costs 2.3× the GEMV on qkv, 1.9× on o, 2.7× on gate/up and 1.7× on down (S0's table,
  re-read). Those four are ~56 ms of the 64 ms one-sequence step.
- **No kernel family found does better below B = 4.** `TestMC3SmallMProbe`, re-run on this build: S0's bit-identical
  M-row GEMV costs more than M sequential GEMVs on the 7B (gate/up M = 2: 1.02–1.17 ms against 2 × 0.433; M = 4:
  2.99 ms). At B = 4 the fragment's half-empty columns cannot be refilled: its columns are token vectors.
- **So the one clean saving is B = 2:** qkv and gate/up as two production GEMVs (0.146 against 0.155 ms, and 0.866
  against 1.167).
  - Projected on the 7B: −8.8 ms of a 65.3 ms B = 2 step, so the in-sequence aggregate goes 1.02× → ~1.17×.
  - On the 1.5B only gate/up qualifies (0.224 against 0.244): about 3%.
  - B ≥ 3 is unchanged.
- **Encode-ahead: parked, by the numbers** (`TestMC3StepWallVsGPU`, depth 128).
  - A batched step's non-GPU time is 1.26–1.66 ms on the 1.5B (6.8–8.1% of the step) and 1.41–2.18 ms on the 7B
    (2.1–3.1%).
  - Production's own encode-ahead recovers about half of its token's gap (0.83 → 0.45 ms, 0.92 → 0.50).
  - So a step executor would buy ~4% on the 1.5B and ~1.3% on the 7B, and it needs a guess at the next run's
    makeup, because the batcher regroups sequences between steps. That is at or under the park line.
  - What the served gap mostly is (estimated from W7's own numbers, not measured): newcomer suffix prefills, ~⅓ of a
    4-client 1.5B cell. That is the per-pass prefill cost lead.

*The build (S4).*
- In `forwardMulti`, qkv and gate/up run as B per-row production GEMVs (production's own kernel and dispatch, `saRowsPick`)
  when B × t_GEMV < t_fragment for that shape, and as the fragment otherwise.
- The two costs are measured once at build, on layer 0's real weights (median of 7 after 2 warm-ups), and logged
  with the choice. Every server's log records what it chose.
- Output is bit-identical to production by construction: it is production's kernel on each row. Tests force either
  path.

*Correctness before timing:* the step identity tests (`_bitIdentical`, `_Deep`, `_drawsMatchForwardSample`) with the
per-row path forced on at B = 2 and 3, and both concurrent-vs-alone tests, on the 1.5B and the 7B.

*Grading.*
- *old* = `serve-metal` at the commit before S4; *new* = S4. Serve defaults, W7 greedy, `--fixed-nonce`.
- 7B: 2, then 4, then 1 client(s), 3 interleaved pairs each (old new new old old new).
- 1.5B, a guard: 2 clients, 3 pairs.
- Gates:
  1. identity and reuse equal, new vs old, every cell (hard);
  2. the 7B's 2-client aggregate new ÷ old, median ≥ 1.05×;
  3. the 7B's 4-client aggregate ≥ 0.98× and p99 turn ≤ 1.02× (hard: B ≥ 3 is meant to be untouched);
  4. the 7B's lone request, p50 and p99 ≤ 1.05× (hard; its path is production's and unchanged);
  5. the 1.5B's 2-client aggregate ≥ 0.98× (hard).
- Decision: all pass ships. Gate 2 at 1.03–1.05× goes to the owner; below 1.03× parks.
- **SHIPPED 2026-09-27 (`7fa344b2`): all five pass** ([`concurrency-mc3-s4-2026-09-27.md`](../../measurements/concurrency-mc3-s4-2026-09-27.md)).
  - The 7B at 2 clients reads **1.121×** (1.093–1.152), p99 turn 0.74–0.90×.
  - The 1.5B at 2 clients reads **1.059×**.
  - The 7B at 4 clients and alone is unchanged (0.998×; 1.001× / 0.998×). Every reply is identical.
  - All 12 new servers calibrated B ≤ 2 for both shapes.

**MC3 S3 SHIPPED 2026-09-27: all five W7 gates pass** ([`concurrency-mc3-s3-2026-09-27.md`](../../measurements/concurrency-mc3-s3-2026-09-27.md)).
- 4 clients at **1.261×** S2 (121.6 → 153.1 tok/s), with p99 turn 0.784×.
- 2 clients 1.08–1.11×; a lone request 0.999× / 1.015×; every reply identical.

**MC3 S3 — step polish (2026-09-27).** An exploratory breakdown (`TestMC3StepBreakdown`, in sequence, each category
no-op'd in turn) showed the batched matmuls already amortised (14.4 ms at any B). What grew with B was the per-row
work, run one sequence after another:
- the small per-row kernels, ~1 ms per sequence (4.2 of a B = 4 step's 22.6 ms at depth 128);
- attention, 2.6 ms at B = 4 and 12.7 at B = 8 depth 512.

**The build.**
- **S3a:** multi-row forms of rmsnorm_quant, quant_vec, swiglu_quant and rope2 (`metal/batch_rows.go`). Their bodies
  are production's byte for byte, derived at init by renaming the signature and adding a per-row prologue; nothing is
  retyped, because these kernels round differently when their code shape changes.
- **S3b:** multi-row kv_store and attention, the per-head kernel. A row map selects the rows below
  attnFADepthFloor; attention_fa rows keep their own dispatches.
  - This needs every row's KV slot reachable from one dispatch, and aikit binds at most 16 buffers. So with several
    slots a layer's KV slots are now **one allocation**, slot views at fixed offsets.
  - aikit's host views ignore a view's offset, so the host KV writes (`UploadKV`, a test hook) add the bound slot's
    offset (`kvHostOff`).

GPU step time, before → after (`s3b-breakdown.log`):

| depth, B | before | after |
|---|---:|---:|
| 128, 4 | 22.59 ms | 17.69 ms |
| 128, 8 | 30.06 ms | 18.72 ms |
| 512, 4 | 26.58 ms | 20.24 ms |
| 512, 8 | 37.75 ms | 23.36 ms |
| 128, 2 | 18.85 ms | 17.20 ms |

**Correctness before any timing:**
- `TestMC3Step_bitIdentical`, `_Deep` (mixed attention_fa and per-head rows in one step) and
  `_drawsMatchForwardSample`: 0 differ.
- Both end-to-end concurrent identity tests (greedy and sampled) pass.
- MC1's slot tests and the tagged Metal suite (182 / 0) pass.

**Grading, pre-registered 2026-09-27 before any W7 timing of the S3 build.**
- *old* = `serve-metal` at `4954f978` (S2); *new* = the S3 commit.
- The MC3 W7 workload, greedy, `--fixed-nonce`, serve defaults.
- Cells: 4 clients old/new × 3 pairs (old new new old old new); 2 clients the same; 1 client the same.
- Gates:
  1. identity: `content_sha` equal, new vs old, every cell (hard);
  2. reuse equal (hard);
  3. 4-client aggregate new ÷ old, median of 3 ≥ 1.03× (a polish bar under the owner's guidance: a clean win beyond
     noise ships);
  4. 4-client p99 turn new ÷ old ≤ 1.0 (hard);
  5. lone request p50 and p99 ≤ 1.05× (hard).
- Reported: the 2-client aggregate and p99.
- Decision: all pass ships; aggregate below 1.03× parks.

**MC3 S2 SHIPPED 2026-09-27: all five W7 gates pass** ([`concurrency-mc3-s2-2026-09-27.md`](../../measurements/concurrency-mc3-s2-2026-09-27.md)).
- 4 sampled clients (T = 0.8, fixed seeds) at **1.623×** the shipped MC3, with p99 turn 0.635×.
- A lone request 1.002× / 1.005×; every reply identical, the pre-MC3 reference included.
- The reference cell (74.1 tok/s) shows the shipped MC3 had given sampled requests nothing.

**MC3 S2 — sampled tokens in steps (2026-09-27).** Under MC3, a temperature-only request's tokens are drawn on the
device (ResidentSample), and each such token ran as its own per-sequence call. So a plain `temperature > 0` chat
request, the default shape of most chat clients, got no batching at all.
- **The build.** A step row can carry the draw: `ResidentBatchSeq.Draw`. The Metal step runs ForwardSample's two
  gumbel dispatches on that row of the batch logits, with the row's own (temperature, seed, draw).
  `generateInto` takes the sampler's draw once per token before submitting, in both modes, so the RNG stream
  advances as it always has.
- **Correctness, run before any timing:**
  - `TestMC3Step_drawsMatchForwardSample`: 24 batched draws against ForwardSample on twin slots, mixed with 24 logits
    rows; 0 differ.
  - `TestMC3_concurrentSampledMatchesAloneOnMetal`: 4 conversations at temperature 0.8 with fixed seeds, concurrent
    against alone; ids identical, 163 tokens served in steps.
  - On the fake: `TestMC3_sampledConcurrentMatchesAlone`, which goes red when the batched draw uses the wrong
    counter.
- **Grading, pre-registered 2026-09-27 before any W7 timing of the S2 build.**
  - *old* = `serve-metal` at `d2225ec4`, the shipped MC3: sampled tokens run solo.
  - *new* = the S2 commit.
  - *ref*, reported only = `9efc3185`, pre-MC3.
  - The W7 workload as for MC3, at `--temperature 0.8 --seed 1000 --fixed-nonce`.
  - Cells: 4 clients old/new × 3 pairs, interleaved old new new old old new; 1 client the same; one ref cell at 4
    clients.
  - Gates:
    1. every turn's `content_sha` equal between new and old at the same client count and index (hard);
    2. reuse equal (hard);
    3. 4-client aggregate new ÷ old, median of 3 ≥ 1.2×;
    4. 4-client p99 turn new ÷ old, median ≤ 1.0 (hard);
    5. lone request p50 and p99, new ÷ old, median ≤ 1.05× each (hard).
  - Decision as for MC3: ship when all pass; aggregate 1.03–1.2× goes to the owner; below 1.03× parks.

**The W7 grading, pre-registered 2026-09-26 before any W7 timing of the MC3 build.**

*Builds.*
- *old* = `serve-metal` from `9efc3185`: MC1 plus R18b plus the slot-pricing fix. A Metal resident serialises its
  generations there.
- *new* = the MC3 commit; its hash goes in the run log.
- Both run at serve's defaults (`-max-concurrent` 4, `-kv-sessions` 4), the release configuration: old serialises,
  new batches.

*Workload.*
- `scripts/bench_w7_plain.py` on qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`;
- 6 turns × 128 greedy tokens per client, `--fixed-nonce`;
- a fresh server per cell, idle-gated per cell (load1 ≤ 2.0).

*Cells.*
- 4 clients: old and new × 3 pairs, in the order old new new old old new.
- 1 client: the same, 3 pairs.
- 2 clients: one pair, reported only.

*Gates.* 1, 2, 4 and 5 are hard.
1. **Identity:** every turn's `content_sha` equal between new and old, at the same client count and client index, in
   every cell.
2. **Reuse:** every turn's prompt − `prefill_reused_tokens` equal between new and old.
3. **Aggregate:** 4-client aggregate tok/s, new ÷ old paired per pair; the median of 3 must be ≥ 1.2×.
4. **p99 under load:** new's 4-client per-turn p99 ÷ old's, paired per pair; the median of 3 must be ≤ 1.0.
5. **Solo guard:** 1-client per-turn p50 and p99, new ÷ old paired per pair; the median of 3 must be ≤ 1.05× for
   each.

*Reported, not gated.* The 2-client cells, and the 4-client p99 ÷ a lone request's.

*Decision.*
- All gates pass: MC3 ships enabled under serve's existing default.
- Gate 1 or 2 fails: it is a bug, not shipped.
- Aggregate between 1.03× and 1.2× with gates 1, 2, 4 and 5 passing: goes to the owner with a ship recommendation,
  per the standing guidance (the amendment above).
- Below 1.03×: parked.
- Gate 4 or 5 fails: not shipped on by default; the owner decides.

**Docs that change when MC3 ships:** `positioning.md` (the "one generation at a time" sentence),
`roadmap.md` §"Decided and parked," `docs/server.md`, red-october R12's row, `QUEUE.md`.

### MC3 on CUDA — design, correctness gates and W7 grading (pre-registered 2026-09-27, before any code)

**Result, 2026-09-27: SHIPS** (`7a44a58e`,
[`concurrency-mc3-cuda-2026-09-27.md`](../../measurements/concurrency-mc3-cuda-2026-09-27.md)).
- Every correctness gate passes. That includes a 2200-key sequence on the flash-decode lane inside a step, and the
  concurrent run against each conversation alone, greedy and sampled, on the tiny fixtures, the 1.5B and the 7B.
- W7 on the 1.5B: gates 1–2 identical; 4 clients **1.380×** (218.0 → 300.6 tok/s); p99 turn **0.755×**; a lone
  request **1.004× / 0.999×** (p50 / p99).
- Reported: 2 clients 1.085×; the 7B at 4 clients 1.826× (p99 0.573×, every turn identical); 99.8% of 4-client tokens
  ran in 4-row steps.
- On under serve's defaults, per the table below.

**Why now.** The S0 ([`concurrency-mc3-cuda-s0-2026-09-27.md`](../../measurements/concurrency-mc3-cuda-s0-2026-09-27.md))
found both halves:
- A batched pass through CUDA's exact kernels is bit-identical to decode per row.
- At B = 4 it is 1.75× (1.5B) and 2.04× (7B) cheaper than 4 decodes, with every row's logits.

MC1 on CUDA (the KV slots) shipped the same day. The decoder side (`ResidentBatchStepper`, MC3's batcher,
`EnableResidentConcurrency`, serve's `setConcurrency` and banner) is backend-generic and unchanged.

**Design.** The CUDA resident implements `BatchStepRange` / `StepBatch`.
1. **The layer stack is `prefillCore`'s, in a multi-sequence mode.** The step runs B rows, one per sequence, through the
   same batched exact path the verify tail uses: rms+quant, `gemv_w4a8_rn` for q/k/v, o, gate/up and down, the
   batched qk-norm, SwiGLU and residuals. `forceExactKernels` holds, as for every non-ordinary tail. No new kernel.
2. **Rope, KV store and attention run per sequence, through decode's own code.**
   - The dynamic gap of `launchToken` (rope_kv, then the attention choice: single-block, split-KV or the
     flash-decode lane) moves into a method, as a pure move with decode unchanged.
   - For row b, the step points decode's single-row buffers (`qB`, `kB`, `vB`, `cctx`) at row b's views of the batched
     buffers (`Buffer.At`), binds sequence b's KV slot, and calls that method at b's own position. Then it restores
     them.
   - Every sequence therefore gets exactly the kernel, and the position-dependent scales, its own decode would use.
     That includes the flash-decode lane past 2048 keys, which is not bit-identical to the exact path; using decode's
     own path makes the choice identical by construction. Launch arguments are captured at launch, and everything
     runs on one stream in order.
3. **The head is per row, as `PrefillLastN`'s tail does it today** (bit-identical to `Forward`, per the S0).
   - A row that carries a `Draw` then runs `gumbelPick` on that row's logits, which is `ForwardSample`'s own draw. An
     infinite-temperature draw takes `ForwardArgmax`'s argmax, as `ForwardSample` does.
   - A batched head is a later lever, only with bit-identity proven: the S0 puts the per-row head at 4.2 of the 7B's
     27.0 ms at B = 4.
4. **Scope, as Metal's MC3 had it.** Dense families whose batched prefill engages (`prefillReady`).
   - `BatchStepRange` reports hi = 0, so the model keeps one generation at a time, for: MoE (including C′ expert
     streaming), MLA, Gated DeltaNet and the other recurrent families, and a model with no batched prefill.
   - Speculation, adapters and vision take the resident exclusively (the decoder's existing rule).
   - The range is lo = 2, hi = the slot count.

**Correctness, before any timing (hard; a failure is a bug):**
1. **Step identity.** For B = 2, 3 and 4 sequences on their own slots at different depths, each `StepBatch` row equals
   that sequence's own `Forward` bit for bit, or `ForwardSample`'s id for a `Draw` row.
   - Fixtures: the tiny CUDA fixtures the resident serves (at least llama-tiny and mistral-tiny-window, the latter for
     the window); qwen2.5-coder-1.5b, including one sequence past 2048 keys so the flash-decode lane runs inside a
     step; and the 7B.
2. **Through the production path.** 4 concurrent `Generate` calls on a CUDA model with `EnableResidentConcurrency(4)`
   emit exactly the ids each conversation emits alone, turn for turn, greedy and sampled (fixed seeds).
   - A control reads `ResidentBatchStats`: steps of ≥ 2 must have run.
   - One conversation's prompt is longer than 512 tokens, so chunked prefill (`-prefill-chunk`) runs while the others
     decode.
   - Tiny fixture, 1.5B and 7B.
3. **Decode unchanged.** The tagged CUDA suite passes, since the gap's move touches the decode hot path. So do vet,
   gofmt and CI's pinned staticcheck with the cuda tags.

**W7 grading, pre-registered.**
- *Builds:* *old* = `serve-cuda` at the commit before this work (MC1: slots, one generation at a time); *new* = the
  MC3-CUDA commit. Both are built once from clean worktrees, named by hash, and run at serve's defaults
  (`-max-concurrent` 4, `-kv-sessions` 4, `-prefill-chunk` 512).
- *Machine:* `nobara`, RTX 2070 SUPER, NVIDIA driver 595.91.07 (CUDA rows are anchored to it).
- *Workload:* `scripts/bench_w7_plain.py --engines goinfer --backend cuda --fixed-nonce --server-log`,
  qwen2.5-coder-1.5b-instruct q4_k_m from `~/models` (never `/srv/models`), 6 turns × 128 greedy tokens per client, a
  fresh server per cell.
- *Idle gate, every cell:* load1 ≤ 2.0; no CUDA compute process beyond those present at the start; GPU memory within
  256 MiB of the starting baseline.
- *Cells:* 4 clients old/new × 3 pairs (old new new old old new); 1 client the same; 2 clients one pair, reported.
- *Gates* (1, 2, 4 and 5 are hard):
  1. **Identity:** every turn's `content_sha` is equal between old and new at the same client count and index, in
     every cell.
  2. **Reuse:** every turn's prompt − `prefill_reused_tokens` is equal between old and new.
  3. **Aggregate:** 4-client aggregate, new ÷ old paired, median of 3 ≥ **1.2×**.
  4. **p99 under load:** new's 4-client per-turn p99 ÷ old's, paired, median ≤ **1.0**.
  5. **Solo guard:** 1-client p50 and p99 turn, new ÷ old, median ≤ **1.05×** each.
- *Reported, not gated:*
  - one 4-client pair on qwen2.5-7b q4_k_m, which starts at 4984 tokens with 4 slots (MC1-CUDA);
  - `ResidentBatchStats` per cell (a shutdown log line, as MC3c's);
  - the 4-client p99 ÷ a lone request's.
- *Decision (MC3's table):*
  - all gates pass and ≥ 1.2×: ships on under serve's defaults;
  - gate 1 or 2 fails: a bug, not shipped;
  - 1.03–1.2× with 1, 2, 4 and 5 passing: the owner, with a ship recommendation;
  - < 1.03×: parked;
  - 4 or 5 fails: not on by default; the owner decides.

**Docs when it ships:** `docs/server.md` (the concurrency paragraph: CUDA joins Metal), the banner's wording if it
names Metal, this section and the status header, and `docs/README.md`'s count.

**Follow-on lever, SHIPPED 2026-09-28: the batched head.** The S0's own "Reading" section named this "the first
lever to test," with a quantified ceiling: batching the head should move M=4 from PrefillLastN's 1.75×/2.04×
(1.5B/7B) toward `PrefillLastNArgmax`'s already-batched 2.20×/2.42×. Measured, not just estimated, this time.

- **What was actually redundant.** `prefillCore`'s head, for `tailAllLogits` (every row needs it — PrefillLastN,
  speculative verify, and MC3's own `StepBatch`), ran the LM head GEMV once PER ROW in a loop (`r.doG`), reading
  the SAME `lm_head` weights M times. `batchedHeadArgmax` already had the fix for the argmax-only case
  (`bNormB` + one `bGemvB` call over all M rows) — `tailAllLogits` had never been given the same treatment, only
  the argmax tail had.
- **`batchedHeadFull`** (`cuda/prefill.go`) reuses those same two calls — `bGemvB`'s own doc comment documents its
  int8 kernel as "bit-identical to gemv_w8a8_fwd by construction" (exact int32 accumulation, so tiling M changes
  no element), and the S0 already proved the same for the int4 kernels used elsewhere in this pass. Only the GEMV
  is batched; per-row post-processing is untouched: a draw row's logits are copied into the single-row `r.logits`
  buffer first (`gpu.CopyDevice`, an existing documented device-to-device verb) so the unmodified `stepDraw`/
  `gumbelPick` kernels never needed to change, and every other row still downloads to the same pinned host buffer
  and gets softcap/logit-scale applied exactly as before — those were never the redundant part.
  - `tailLastLogits`/`tailHiddenLast` (only the LAST of M rows is ever read — a long prompt's ordinary
    `PrefillLast`) are explicitly excluded from batching: batching there would compute M−1 rows nobody reads, the
    regression this scoping exists to avoid.
- **Correctness (nobara-pc, RTX 2070 SUPER, driver 595.91.07, idle):**
  [`batchedhead-correctness-2026-09-28.log`](../../measurements/concurrency-mc3-cuda-s0-2026-09-27/batchedhead-correctness-2026-09-28.log),
  [`batchedhead-7b-2026-09-28.log`](../../measurements/concurrency-mc3-cuda-s0-2026-09-27/batchedhead-7b-2026-09-28.log).
  - `TestCUDAStepBatch_matchesForward`, `TestCUDAStepBatch_concurrentMatchesAlone`: pass, unchanged — MC3's own
    step-identity and concurrent-matches-alone gates.
  - `TestMC3CUDAS0_batchedRowsVsDecode` (the S0's own bit-identity check, re-run against the new code): **0
    differing logits** at M = 2/4/8 on both the 1.5B (303,872 / 607,744 / 1,215,488 compared) and the 7B
    (304,128 / 608,256 / 1,216,512), same as before this change.
  - Full tagged CUDA suite: 177 pass, 0 fail. `gofmt`, `go vet`, pinned staticcheck 0.8.0 all clean.
- **Speed — a real number, not the estimate:**

  | model | M | PrefillLastN before | PrefillLastN after | before | after |
  |---|---:|---:|---:|---:|---:|
  | 1.5B | 2 | 7.48 ms | 7.255 ms | 1.30× | **1.44×** |
  | 1.5B | 4 | 10.61 ms | 9.046 ms | 1.75× | **2.07×** |
  | 1.5B | 8 | 16.76 ms | 12.977 ms | 2.22× | **2.86×** |
  | 7B | 2 | 18.73 ms | 17.905 ms | 1.47× | **1.54×** |
  | 7B | 4 | 27.00 ms | 23.526 ms | 2.04× | **2.34×** |
  | 7B | 8 | 45.78 ms | 38.722 ms | 2.41× | **2.85×** |

  At M=4 (MC3's default slot count) the 7B lands at 2.34×, inside the 2.2–2.4× band the S0 predicted, and within
  0.07× of `PrefillLastNArgmax`'s own 2.41× ceiling — the head is now nearly as cheap as the argmax-only variant
  for every family this scoping covers.
- **In-process, by day, per TE5(b)** (owner decision 2026-09-28, `docs/tasks/task-test-efficiency-2026-09.md`): a
  resolved in-process kernel-speed result may be acted on by day, and this one resolves cleanly — real,
  bit-identical, and consistent across both models and every M tested. The served, end-to-end W7 number (how much
  of this reaches the 4-client aggregate) is queued for tonight, not assumed from the per-step figure above.
- **Owed:** a W7 rerun on the 1.5B (the graded MC3-CUDA configuration) to see how much of this reaches the
  4-client aggregate beyond the already-shipped 1.380×/1.826× (1.5B/7B) — queued, `mc3-cuda-batchedhead-w7`.
- **Not pursued this pass: per-call scratch in `prefillCore`.** Unlike the batched head, this had no prior
  measurement anywhere in the codebase (the S0 named it only as an estimate). `prefillCore` allocates its
  M-sized scratch fresh from the device allocator on every call — including every MC3 `StepBatch` — and frees it
  at return (`cuda/prefill.go`'s own "M-sized scratch (device), freed at the end" comment), roughly 13 `af`/`ai`
  calls per call, all before the per-layer loop (not multiplied by depth). Whether that allocate/free cycle is a
  meaningful fraction of a 2-4-row step's time was not measured here; sizing it (a category on `prefillProf`, or an
  isolated allocation microbenchmark at MC3's actual M) is the natural next step before deciding whether to build
  a persistent scratch pool for the batched-step case.

## MC3c — concurrent decode on CPU (owner: "cpu first", 2026-09-26)

**Step 1 result, 2026-09-26: all five gates pass**
([`concurrency-mc3c-2026-09-26.md`](../../measurements/concurrency-mc3c-2026-09-26.md)).
- `serve -max-concurrent 4` on the W7 CPU workload gives 1.97× the serialized 4-client aggregate (repeats 1.86× and
  1.91×), and p99 per-turn latency falls from 12.8–13.4 s to 6.5–7.0 s.
- Output is byte-identical to each conversation served alone, and the concurrency tests pass under `-race`.
- It shipped default 1. **Owner decision 2026-09-26: default 4** (still capped by `-kv-sessions`). With that, a
  request's prefill-memory share counts the generations running or queued ahead of it, so a lone request keeps the
  whole margin. Step 2 (batching) is not started.

**Why this, before MC3.** MC2 earned on the CPU with batching (1.69–2.04× at B = 4), and J8's cell, 4 independent
decode workers, matched or beat it (2.00–2.48×) at a fraction of the build. MC3 on Metal needed a matrix-unit kernel,
and at the time (MC3 S0, first pass) a fidelity gate, which the second pass removed. The owner chose the CPU first. **Step 1 is the workers**: N CPU generations of one model
run at once, each on its own session KV. **Step 2 is batched decode behind the same admission**, registered here and
started only if the `nobara` cells or larger client counts show it beating the workers.

**Design (step 1).**
- `serve -max-concurrent N` (default 1, which is today's behaviour exactly). It applies to a CPU model only: a
  GPU-resident model keeps one generation at a time, because it has one claimed KV (`resBusy`). So does a streamed
  model (expert or layer pager), whose pager is shared.
- Admission (J1) becomes N-wide, still FIFO: up to N hold a turn, and the rest queue in arrival order.
- The session LRU gains **check-out**. `acquire` never hands out a session another generation holds, and never evicts
  one to make room. The LRU's own lock is held only around LRU operations, not across the generation. Background work
  (idle demotion, save, restore) skips a checked-out session, and shutdown's save waits for check-ins.
- No change inside the decoder: distinct sessions already run concurrently on one `*Model`
  (`Model`'s own contract), and a race test pins that.

**Gates (pre-registered; speed bar per the owner's standing guidance).**
1. **Bit-identical:** each conversation's response, served concurrently, is byte-identical to the same conversation
   served alone (greedy).
2. **`-race` clean** across concurrent generations and the LRU's check-out.
3. **Reuse unchanged:** on the W7 plain workload, CPU, every turn's prompt − reused equals the 1-client run's.
4. **Speed:** the W7 4-client aggregate is ≥ 1.2× the serialized build's in the same session, with N = 4.
5. **Latency under load:** p99 per-turn latency at 4 clients is no worse than the serialized build's at 4 clients.
   This is the same offered load, not a lone request's latency.

**Default.** `-max-concurrent` stays 1 until the gates pass. Then a default > 1 for CPU models is an owner decision,
because it trades per-request latency for throughput.

**Step-2 trigger cell, pre-registered 2026-09-27.**
- What runs: `TestMC2_batchedDecodeThroughput` on `nobara` (Ryzen 7 3700X, 16 threads), qwen2.5-7b-instruct q4_k_m
  from `~/models`, int4, at depths 128 and 512. The harness's defaults apply: 16 steps, 5 reps, and arms interleaved
  and rotated.
- Metric: batched B = 4 aggregate ÷ J8 N = 4 aggregate, per rep, median of 5, at each depth.
- Decision:
  - the LOWER of the two depths' medians ≥ 1.15× → build step 2;
  - the HIGHER < 1.05× → park step 2, and rewrite its trigger to "a model size where batching beats the workers by
    ≥ 1.15×";
  - anything else → the owner decides.
- Reported, not gated: B = 8, B = 2, the serial arms, and the identity test.
- A caveat, recorded and not adjusted for: on amd64 the prototype's batched path at M = 1 costs 0.76–0.93× production
  decode, because its kernels are untuned for M > 1 (MC2 record, "Linux CPU cells"). So the batched numbers are a
  floor.
- The identity test (`TestMC2_decodeMultiStepBitIdentical` on the 7B) runs first and is hard: a difference stops the
  run as a bug.
- Each depth runs only with load1 ≤ 1.0 and no other `go test` / `go build` / serve on the box.

**Step-2 trigger result, 2026-09-27: BUILD STEP 2** ([`concurrency-mc2-2026-09-26.md`](../../measurements/concurrency-mc2-2026-09-26.md),
"Linux 7B cell").
- Identity passes on the 7B at int4 and int8int8.
- Batched B = 4 ÷ J8 N = 4 is **2.41×** at depth 128 and **2.25×** at depth 512. That is the median of 5 paired reps;
  every rep reads the same at the logged resolution, 2.22–2.27× at the worst rounding.
- The lower depth's median clears the 1.15× bar, so the registered rule says build.
- Why the gap is so large:
  - On this CPU a 7B decode is already bandwidth-bound, so 4 independent workers reach only 1.13–1.15× one decode.
  - Batching reads each weight once per step and reaches 2.59–2.73× at B = 4 (3.1–3.3× at B = 8).
  - The batched numbers are a floor: the prototype's M = 1 path runs at 0.965× production decode on the 7B.
- Across model sizes, batching's lead over the workers grows 0.64–0.75× (0.5B) → 1.02–1.11× (1.5B) → 2.25–2.41× (7B).
- Step 2 is not started. Its design, gates and pre-registration follow.

### MC3c step 2 — batched CPU decode behind the same admission (design and grading pre-registered 2026-09-27, before any code)

**Result, 2026-09-27: SHIPPED, all five W7 gates pass**
([`concurrency-mc3c-step2-2026-09-27.md`](../../measurements/concurrency-mc3c-step2-2026-09-27.md)).
- Code `420d655b`; graded build `2c1d89ec`, which adds the shutdown stats line; census fix `d16f5845`.
- On the 7B (CPU, 4 clients) the aggregate is **2.19×** the step-1 workers (5.44 → 11.92 tok/s), and the p99 turn is
  **0.48×** (~96 → ~46 s).
- A lone request is **1.000× / 1.000×** (p50 / p99): it ran no batched step.
- Every reply is identical to the step-1 build's, turn for turn, at 1 and 4 clients.
- Reported: 2 clients 1.47×. The 0.5B with `-cpu-batch on` reads 0.73× the workers, with identical replies, which is
  why `auto` leaves it on them.
- One change from the registered design: an ineligible cache never enters the batcher, so no run mixes eligible and
  ineligible tokens.

**Step 2 follow-on S1 — result, 2026-09-27: SHIPPED, every gate passes**
([`concurrency-mc3c-s1-2026-09-27.md`](../../measurements/concurrency-mc3c-s1-2026-09-27.md)).
- Code `000efe2e`. On the 7B the batched step at B = 4 is **1.058×** the per-projection build at depth 128 and 512
  (bar 1.03×); B = 2 is 1.04× and B = 8 is 1.06×, and every step's logits are bit-identical between the arms.
- W7 at 4 clients (reported): **1.051×** (11.93 → 12.54 tok/s), p99 turn 0.96×, every reply identical.
- The 1.5B's step gains 1.142×, under `-cpu-batch on` only.
- S1 takes ~5.8% of the ~17–19% ceiling. Lever B (an aikit small-M kernel) is the rest. It is **parked by the owner
  (2026-09-27)**: realistically ~+5–6% end to end at 4 clients and nothing for a lone request, for an aikit release.
  The trigger is in the status header's open list.

**Step 2 follow-on S1 — fused batched projections in the step (pre-registered 2026-09-27, before any code).**
- *Why.* An exploratory probe (`TestCPUBatchS0_matmulScaling`, 7B layer 0, amd64, canonical int4) timed each shape at
  M rows against one row:
  - gate/up (18944×3584) cost 1.03–1.09× one row at M = 2–4, and the LM head 1.00–1.02×: already amortised;
  - down cost 1.11–1.34×;
  - q/o (3584×3584) cost 1.6–1.9×, and k/v (512×3584) **2.0–3.0×**. At those sizes the time is fork/join overhead and
    compute, not bandwidth.
- *Ceiling.* Perfect amortisation everywhere would save ~45 ms of a ~290 ms B = 4 step, ≈ +17–19% batched throughput;
  any real change gets part of it.
- *Lever A (decoder-only, this item).*
  - `decodeMultiStep` runs q‖k‖v as one `matmulW4A8Batch` over B rows, and gate‖up as another: one fork/join each
    instead of three and two.
  - The kernel already exists (`linalg.MatmulBTW4A8Batch`, the fused path production's single-token forward uses
    behind `w4a8BatchEnabled`). Its contract is "numerically identical to calling MatmulBTW4A8Into once per op", with
    the M ≥ 4 tile intact.
  - It is used only when every op has canonical bytes and they share a group size and K; anything else keeps the
    separate calls.
  - An in-process switch (a test-hook package variable, no environment read) lets both arms run interleaved.
- *Lever B (an aikit small-shape kernel)* is not started, and only follows if A leaves headroom.
- *Gates:*
  1. **Identity (hard):** with A on, `TestCPUBatch_everyEligibleFixtureBitIdentical`,
     `TestCPUBatch_concurrentMatchesAlone` (tiny; 1.5B on; 7B auto) and `TestMC2_decodeMultiStepBitIdentical` on the
     7B (int4 and int8int8) all pass.
  2. **Speed:** `TestCPUBatchS1_fusedVsUnfused` on `nobara`, qwen2.5-7b q4_k_m int4, at depths 128 and 512.
     - Setup: 16 steps, 5 reps; arms fused and unfused interleaved rep by rep with rotating order; B = 2, 4 and 8.
     - Metric: batched B = 4 aggregate, fused ÷ unfused, paired per rep, median of 5, at each depth.
     - The lower depth's median ≥ **1.03×** ships A on by default.
     - The higher depth's median < **1.01×** parks A and removes it.
     - Anything else goes to the owner.
  3. **No regression (hard for ship):** the B = 2 and B = 8 medians are ≥ 0.99× at both depths.
- *Reported, not gated:* the B = 2 and B = 8 ratios; the 1.5B's B = 4 ratio; and a W7 4-client confirmation, step-2
  build (`2c1d89ec`) against the A build, 3 interleaved pairs, idle-gated. Identity there is hard, and the aggregate is
  reported.
- *Idle gate:* every timed run starts at load1 ≤ 1.0 with no other go, test or serve process.

**Goal.** When several CPU generations of one model are decoding, their decode tokens run as one batched forward
instead of N independent forwards. Each reply stays bit-identical to the same conversation served alone. The step-1
admission, session check-out and prefill are unchanged; only what runs at a decode token changes.

**Where it plugs in.** `generateInto`'s CPU path decodes with `logits, err = m.forward(next, cache)`, once per token, on
its own session's cache. Step 1 runs N of those concurrently. Step 2 routes that call through a CPU coordinator when
batching is enabled for the model and the cache is eligible:
- a run of ≥ 2 tokens goes through one batched step;
- a single token goes through `m.forward` unchanged, so a lone request takes exactly today's path.

**New pieces.**
1. **The batched step, promoted from the MC2 prototype.** `decodeMultiStep` (`decoder/batchdecode_mc2_test.go`, test-only
   today) moves into production.
   - Projections, o-proj, MLP and LM head run as M = B matmuls through `forwardN`'s kernels, which are bit-identical to
     M = 1 row for row.
   - Attention runs per sequence over its own cache.
   - `TestMC2_decodeMultiStepBitIdentical` is already its identity gate. It passes on llama-tiny and the 0.5B (int4 and
     int8int8) on both arm64 and amd64, and on the 7B (int4 and int8int8) on amd64.
   - Its scope is `mc2Eligible`: the generic forward, plain pre-norm, dense MLP, no attention output gate, no learned
     positions, no weight streaming; f32 append-forever caches with no adapter, tree mask or manual position.
2. **The coordinator.** It shares MC3's token coalescing rather than copying it, factored out of `residentBatcher`:
   - pending tokens; the straggler window, opened when a run could start (the phase-lock lesson measured 2026-09-26);
   - decode-loop enter/exit, so a run waits only for generations inside their decode loop;
   - the stats counters (runs, steps, step sizes, solo tokens, straggler runs).

   What it does not share is exclusivity. A CPU model has no single device to own, so prefill and non-batched work run
   outside it, as in step 1.
3. **Per-sequence state stays per sequence.** The step returns each row's logits, and each generation samples with its
   own sampler. That covers repetition penalties, logit processors, grammar masks and stop tokens, on the host exactly
   as today. Nothing is drawn inside the step.
4. **The operator choice** is `decoder.Options.CPUBatchDecode` (an int, so `Options` stays comparable), with a
   per-model accessor. There is no environment read. serve gets the flag `-cpu-batch auto|on|off`, and the banner's
   concurrency line says which mode runs:
   - `off` is step 1 exactly: independent workers;
   - `on` batches every eligible model;
   - `auto`, the default, batches an eligible model whose `ResidentDenseWeightBytes()` is at least **2 GiB**, and runs
     workers below that.
   - Why 2 GiB: the measured cells put batching's lead over the workers at 0.64–0.75× on the 0.5B (0.47 GB) and
     1.02–1.11× on the 1.5B (1.23 GB) on `nobara`. On the Mac the 1.5B workers lead. On the 7B (4.93 GB) batching
     leads by 2.25–2.41×.
   - So `auto` leaves the small and marginal models on the workers, where they are no worse off, and batches where
     the lead is large.
   - The 2–4.9 GB range is unmeasured, and `auto` batches it; recorded, not tested.
   - **darwin:** `auto` means `off` until a Mac 7B MC2 cell shows batched B = 4 ÷ J8 N = 4 ≥ 1.15×. Batching loses on the
     Mac's 0.5B and 1.5B, and its 7B has not been measured.
   - **The Mac 7B cell, pre-registered 2026-09-27 before any timing.**
     - `TestMC2_batchedDecodeThroughput` on the M1 Pro (8P + 2E cores), qwen2.5-7b-instruct q4_k_m int4 from
       `~/models`, depths 128 and 512, the harness's defaults (16 steps, 5 reps, arms interleaved and rotated).
     - Each depth runs in its own `go test`, idle-gated: load1 ≤ 2.0 and no other go test or serve process.
     - It loads the 7B's `.int4.cpu-arm64.giw`, transcoded with `cmd/prequant` for this cell. The `.gguf`'s CPU int4
       load needs ~13.3 GB, which this Mac's fit guard refuses, and the guard is not bypassed.
     - **Identity first (hard):** `TestMC2_decodeMultiStepBitIdentical`'s int4 subtest on the 7B. int8int8 is not
       the metric's quant, and a 7B at int8int8 is ~7.6 GB of heap.
     - **Metric:** batched B = 4 ÷ J8 N = 4, per rep, median of 5, at each depth.
     - **Decision:** both depths' medians ≥ 1.15× → `auto` batches on darwin too (the `runtime.GOOS` clause in
       `decoder/cpu_batch.go` goes). Both < 1.05× → darwin stays off, with the numbers recorded. Otherwise the owner
       decides.
     - **Measured 2026-09-27: `auto` now batches on darwin.**
       - Batched B = 4 ÷ J8 N = 4 reads **1.540×** at depth 128 and **1.383×** at 512
         ([`concurrency-mc2-2026-09-26.md`](../../measurements/concurrency-mc2-2026-09-26.md), "Mac 7B cell").
       - Identity passes on the 7B.
       - The darwin clause is removed. The 2 GiB threshold keeps the Mac's 1.5B on the workers.

**Declines, per token, to production's own `m.forward` (the step-1 worker path):**
- a cache that fails `mc2Eligible`: `--kv i8`, a sliding-window ring, an adapter session, a tree mask;
- MoE, recurrent, hybrid and own-forward families, and weight streaming (`layerPager`);
- speculative decode, which drives its own verify;
- vision turns.

A run mixing eligible and ineligible tokens batches the eligible ones and runs the rest solo.

**Correctness, before any timing (hard; a failure is a bug):**
1. **Through the production path:** N concurrent `Session.Generate` calls routed through the coordinator emit exactly
   the ids each conversation emits alone, turn for turn, with the same `PrefillReused`.
   - Fixtures: llama-tiny (CI) and qwen2.5-coder-1.5b with `on`; the 7B with `auto` (heavy).
   - A control asserts from the stats that steps of ≥ 2 actually ran, so the test cannot pass on solo tokens.
2. **Sampled:** the same with temperature and a fixed seed per conversation, since each sampler draws from identical
   logits.
3. **Declines:** an `--kv i8` session and an adapter session interleaved with eligible ones still match alone. The
   stats show their tokens ran solo.
4. **`-race` clean:** concurrent generations through the coordinator, plus step 1's session check-out.
5. **The prototype's gate stays:** `TestMC2_decodeMultiStepBitIdentical` is kept, pointed at the production step.

**W7 grading, pre-registered.**
- *Builds:* *old* = `serve` (CPU) at the commit before step 2, which is step 1 (workers, `-max-concurrent` 4); *new* =
  the step-2 commit, where `auto` batches the 7B. Both are built once, from clean worktrees, and named by hash.
- *Machine:* `nobara`, CPU backend. `-backend cpu`; it must not touch the GPU.
- *Workload:* `scripts/bench_w7_plain.py --engines goinfer --backend cpu --fixed-nonce --server-log`, on
  qwen2.5-7b-instruct q4_k_m from `~/models` (never `/srv/models`) at int4.
  - 6 turns × 128 greedy tokens per client, a fresh server per cell, serve defaults otherwise.
- *Idle gate, every cell:* load1 ≤ 1.0 (this is CPU timing), and no other `go`, test-binary or serve process.
- *Cells:*
  - 4 clients: old/new × 3 pairs, in the order old new new old old new;
  - 1 client: the same, 3 pairs;
  - 2 clients: one pair, reported only.
- *Expected duration:* ~1–1.5 h. A 4-client old cell is ~3,100 tokens at ~5.7 tok/s.
- *Gates* (1, 2, 4 and 5 are hard):
  1. **Identity:** every turn's `content_sha` is equal between old and new at the same client count and client index,
     in every cell. Both builds are bit-identical to a conversation served alone.
  2. **Reuse:** every turn's prompt − `prefill_reused_tokens` is equal between old and new.
  3. **Aggregate:** 4-client aggregate tok/s, new ÷ old paired per pair, with a median of 3 ≥ **1.2×**.
  4. **p99 under load:** new's 4-client per-turn p99 ÷ old's, paired, with a median of 3 ≤ **1.0**.
  5. **Solo guard:** 1-client per-turn p50 and p99, new ÷ old, paired, with a median of 3 ≤ **1.05×** each.
- *Reported, not gated:*
  - the 2-client pair;
  - the coordinator's stats per cell (step sizes and solo tokens);
  - one 4-client pair on the 0.5B with `-cpu-batch on` against `off`, to show why `auto` leaves it on the workers.
- *Decision:*
  - All hard gates pass and the aggregate is ≥ 1.2×: it ships with `auto` as the default.
  - Gate 1 or 2 fails: it is a bug, not shipped.
  - Aggregate 1.03–1.2× with 1, 2, 4 and 5 passing: it goes to the owner with a ship recommendation.
  - Aggregate below 1.03×: parked, with `-cpu-batch on` kept only if the owner asks.
  - Gate 4 or 5 fails: not on by default; the owner decides.
- *Record:* `docs/measurements/concurrency-mc3c-step2-<date>.md`, with raw JSON and logs beside it. The NVIDIA driver is
  irrelevant here (CPU); the CPU governor and kernel version are recorded instead.

**Not in scope:**
- A small-M amd64 kernel. The batched path's M = 1 costs 0.76–0.97× production, so every batched number above is a
  floor, and a tuned kernel is a follow-on on its own measurement.
- Mixing workers and batches, for example two batches of 4 at 8 clients.
- Batched prefill across generations.
- CUDA/Metal: MC3 and MC1 cover those.

**Docs that change when it ships:**
- `docs/server.md` (the `-max-concurrent` paragraph, and the new `-cpu-batch`);
- the banner's concurrency line;
- this section and the status header;
- `docs/README.md`'s measurement count.

**Estimate.** A few days. The step and the coalescing exist, and the work is promotion, a shared coalescer, the
option and the tests.

## MC4 — broadening (parked)

Each item waits on MC3 shipping and on a measured request for it:
- **MoE in a batch** — tokens in one step route to different experts; interacts with the expert
  pager and with never-swap's memory rules. Trigger: MC3 shipped and an MoE model is the one
  harness users run.
- **Recurrent and hybrid families** — per-slot state is MC1's job; batching the state update is
  this item's. Trigger: same.
- **Speculative decode inside a batch.** Trigger: MC3 shipped and spec's single-stream win is
  still larger than batching's per-sequence share. **Trigger fired on CUDA, 2026-09-29** (copy L
  = 1.540×, [`spec-vs-batching-cuda-2026-09-29.md`](../../measurements/spec-vs-batching-cuda-2026-09-29.md)
  §6). Investigated the same day: the real fix is a ragged-batch kernel (unscoped, real backend
  work); the cheap hypothesis (a workload-aware yield policy in the switch candidate, zero new
  kernels) does NOT work — measured worse than doing nothing, because the cost is concurrent
  admission itself, not the per-round policy. **Parked, no design exists yet, owner's call.**
- **Mixed adapters in one batch.** Trigger: a real multi-adapter workload.

**The spec trigger, measured on Metal — pre-registered 2026-09-27, before any timing.** Today a model served with
`--spec ngram` takes the resident exclusively, so under load it gives up MC3's batching. Its verify is `ForwardBatch`,
production's decode kernels run layer-major in one command buffer, so an extra verified row costs 0.71–0.96 of a token
(`thetaFor("metal")` = 0.96). The question is what a Metal user gains or loses by turning spec on.
- One binary, `serve-metal` at `cc5f8c2c`, two arms, a fresh server per cell:
  - *batch*: serve's defaults (MC3 batching; a lone request runs production's own path);
  - *spec*: the same with `-spec ngram` (one generation at a time).
- Two workloads, qwen2.5-coder-1.5b int4 from `~/models`, greedy:
  - *copy*, where n-gram drafting does best: `scripts/bench_spec_copy.py`. Each request hands back a ~1000-token
    section of `decoder/model.go` at `cc5f8c2c` verbatim, 256 tokens; each client sends 2 requests in turn.
  - *chat*: the W7 workload as graded for MC3 (6 turns × 128 tokens, `--fixed-nonce`).
- Cells: 4 clients, then 1, per workload, batch/spec × 3 pairs in the order batch spec spec batch batch spec, idle-gated
  per cell (load1 ≤ 2.0).
- Metrics, paired per pair, median of 3:
  - **S**, spec's single-stream win: 1-client aggregate, spec ÷ batch;
  - **L**, under load: 4-client aggregate, spec ÷ batch. The 4-client p99 request or turn latency is reported.
- Identity is reported, not gated: every reply's hash, spec vs batch at the same client count. A difference is a finding
  about spec's lossless claim on Metal, recorded and examined before anything is built on it.
- Reported only: the copy workload on qwen2.5-7b-instruct at 1 and 4 clients, 3 pairs each.
- **Decision** (the 1.5B, per workload):
  - S ≤ 1.05 on both → spec has no single-stream win on Metal to keep. The spec item stays parked for Metal, and the
    docs say that `--spec ngram` on Metal gives up batching under load for no gain.
  - S > 1.05 on a workload, with L < 1.0 → "speculate when alone, batch under load" has value there. It is
    registered as an MC4 candidate, and the owner decides.
  - L ≥ 1.0 on a workload → spec beats batching even at 4 clients, and MC4's spec trigger is met there.
- **Measured 2026-09-27: S ≤ 1.05 on both, so the spec item stays parked for Metal**
  ([`spec-vs-batching-metal-2026-09-27.md`](../../measurements/spec-vs-batching-metal-2026-09-27.md)).
  - S: copy 0.979×, chat 0.928×. Spec gives a lone Metal request nothing.
  - L: copy 0.610×, chat 0.489×. Under 4-client load it forfeits MC3's batching.
  - Found on the way, both open:
    - with `--spec ngram`, chat turns after the first differ from plain decode (turn 0 equal in every conversation).
      **Fixed 2026-09-27:** the loop now forwards a round's trailing token when `max_tokens` ends the generation, as
      plain decode does. The next turn had re-prefilled it through Metal's non-bit-identical f16 prefill. Every W7
      turn now matches (`TestGenNgramInto_residentCommitMatchesPlain`, `TestSpecNgram_multiTurnMatchesPlain`).
      The other loops were checked the same day.
      - The block drafter (`--drafter`, CUDA) committed one position more than it wrote, so the next turn reused a
        stale one. It is fixed and pinned by `TestBlockSpecGenerate_commitsOnlyWrittenPositions`. ~~its CUDA check on
        nobara is owed~~ — **run 2026-09-28** (item 30, `docs/prompts/nobara-cuda-spec-trailing-token-2026-09.md`):
        new `cuda/spec_twoturn_test.go` on the real qwen3-4b + DFlash drafter, N ∈ {1, 17, 48}, both fixes together
        (the n-gram one too). 14/14 pass with the fixes in; reverted, n-gram is exactly one token short at every N
        and the block drafter's turn 2 visibly diverges at N=17/48 (N=1's stale position happens not to corrupt
        that particular continuation — see the record for the caveat). Full results and logs:
        [`spec-vs-batching-metal-2026-09-27.md`](../../measurements/spec-vs-batching-metal-2026-09-27.md)'s own
        "the CUDA check" update.
      - The grammar-fused loop was one short and is fixed.
      - The two-model loop never commits, and is unchanged;
    - on the 7B, serve's prefill-memory share 413'd MC3 batch cells, because it divides the margin as if prefills ran
      concurrently. **Fixed 2026-09-27:** a request that prefills on a resident keeps the whole live margin
      (`TestPrepare_prefillShare`). The end-to-end rerun is owed: the attempt ran on a Mac already swapping.

**Metal spec verify on the step kernels — registered 2026-09-27, before any build or timing.** This replaces
`docs/prompts/metal-verify-curve-remeasure-2026-09.md`, deleted by the owner 2026-09-27. That prompt would have
re-timed the f16-MMA prefill kernel as a verify path for P10's block drafters (`docs/spec/08-dspark-dflash.md`). That
kernel is not bit-identical to decode, so its ceiling could unlock nothing on its own. Two findings of the day moved
the question:
- **Metal's present verify is lossless but expensive.** `ForwardN` is bit-identical to `Forward`
  (`TestSpecVerify_forwardNMatchesForward`), but each extra row costs 0.71–0.96 of a token. Served n-gram spec
  measured 0.98× plain decode on copy-heavy traffic (above).
- **MC3's step kernels are a bit-identical batched path that costs little per row.** A step of 8 rows costs 1.82× a
  token on the 1.5B and 2.06× on the 7B at depth 128 (S4's in-sequence run): an extra row is ~0.12–0.15 of a token,
  about CUDA's measured 0.16–0.25.
  - A verify is 1 + k consecutive positions of one sequence on one slot.
  - The step already stores every row's K/V before its attention runs, layer by layer. So row i sees rows below it,
    as sequential decode does.
  - The one visible blocker: `forwardMulti` refuses a slot that appears on two rows.

*First, exploratory, and deciding whether anything is built:*
1. **Identity (hard, stops the item).** A step whose rows are consecutive positions of one sequence on one slot gives
   every logit and every K/V element bit-identical to sequential production decode.
   - Rows: M = 2, 4, 8 at depths 128 and 2048, plus a run straddling `attnFADepthFloor`.
   - Models: the 1.5B and the 7B.
2. **Cost:** the step's GPU time at M = 1, 2, 4, 8 rows against a production token, at depths 128, 512 and 2048, on
   both models.
   - The build starts if an extra row averages ≤ 0.30 of a token at M = 8 up to depth 2048 on both.
   - Otherwise the item parks with the curve recorded.

*The build* (only if 1 and 2 pass): the Metal resident gains an argmax-only verify (`PrefillLastNArgmax`, which the
n-gram loop already prefers when present).
- Rows run on the step kernels in pieces of at most 8, the fragment's width.
- A one-row round runs production's own `Forward` plus a host argmax, so a round with no draft costs what plain
  decode does.
- `thetaFor("metal")` becomes the step's measured marginal cost, the conservative end.
- The fragment's cost is nearly flat in M, which the controller's linear model understates for shallow drafts; the
  chat gate below is what catches that.

*Grading* (registered now, run only after the build):
- One binary, three arms at 1 client (spec is single-stream; under load it forfeits batching either way):
  - *plain*: serve's defaults;
  - *spec-old*: `-spec ngram` on the current verify, the build's parent;
  - *spec-new*: `-spec ngram` on the step verify.
- The copy and chat workloads of the spec measurement above. The 1.5B is graded, the 7B reported. 3 interleaved
  rounds, idle-gated.
- Gates:
  1. every reply equals plain's, every turn (hard);
  2. copy: spec-new ÷ plain, median ≥ 1.25×. That is the level at which `--spec ngram` becomes worth recommending on
     Metal;
  3. chat: spec-new ÷ plain ≥ 0.97× (hard: spec must not cost ordinary traffic).
- Decision: all pass ships, and `server.md`'s "leave it off on Metal" is rewritten. Copy at 1.03–1.25× goes to the
  owner. A chat failure is fixed (for example, a minimum draft depth for step verifies) before anything ships.
- P10's block drafters follow only on this result, with their own drafter port and measurement.
- **SHIPPED 2026-09-27 (`1e153876`): both deciding gates and all three served gates pass**
  ([`metal-spec-step-verify-2026-09-27.md`](../../measurements/metal-spec-step-verify-2026-09-27.md)).
  - Identity: same-slot rows are bit-identical to sequential decode, 0 differ in 7 cases on both models.
  - Cost: an extra row is 0.11–0.19 (1.5B) and 0.16–0.24 (7B) of a token at M = 8.
  - Served, 1 client, on the 1.5B: copy **2.082×** plain decode and chat **1.068×**, every reply identical. The 7B
    reads 1.846× and 1.014×. The old verify read 0.977× and 0.956×.
  - Next, by the spec measurement's own registered rule (S > 1.05× on both workloads, with L < 1): "speculate when
    alone, batch under load" is now worth registering. P10's block drafters have a Metal verify.
  - **MC4 candidate: "speculate when alone, batch under load". Pre-registered 2026-09-28, before any design code or
    timing; building it is the owner's decision.**
    - **Why its condition is met.** The 2026-09-27 rule registers this when S > 1.05 on a workload with L < 1.
      - When that rule ran, S was 0.979× / 0.928×. The step-kernel verify (1e153876) then made a lone request 2.082×
        plain on copy and 1.068× on chat (1.5B; the 7B 1.846× / 1.014×), so S > 1.05 on both workloads.
      - L is still < 1. `-spec` forces one generation at a time (`internal/serveapp/openai.go`, `setConcurrency`), so
        under 4 clients it gives up MC3's batching (0.610× / 0.489× before the step verify, and still one stream).
    - **The candidate.** With `-spec ngram` on a resident that batches, the model keeps MC3's concurrency. A
      generation speculates only while it is alone.
      - **At every spec round boundary**, the loop checks whether any other generation is decoding, waiting to
        prefill, or queued. If one is, it stops drafting and joins MC3's decode loop from its committed position,
        on the same slot and KV. Each round runs in `exclusive`, so a newcomer's prefill waits at most one round, not
        a whole generation.
      - **It resumes speculating** after 8 consecutive steps alone (hysteresis, so a briefly idle second client
        does not thrash it).
      - **Rejected:** choosing at admission and holding the resident for the whole generation. A lone request that
        starts long makes every later arrival wait for it, the p99 cost the 2026-09-27 run measured (1.64× / 1.98×).
    - **Why it can be lossless.** A spec round on the step verify is bit-identical to sequential decode, and so is an
      MC3 step (`TestMC3Step_bitIdentical`, `TestMC3Verify_sameSlotRowsBitIdentical`). A switch at a round boundary changes only who runs
      the next committed token. That is a claim to prove, not to assume: see step 0.
    - **Step 0, by day, before any served timing. Hard; it stops the item.**
      1. **Identity:** a generation forced to switch spec → batch → spec at round boundaries, beside 1–3 others,
         emits exactly plain decode's ids, and its KV is bit-identical.
         - The 1.5B and the 7B, at depths 128 and 2048.
         - Two turns per conversation. The Metal re-prefill lesson: a cache one token short changes every later turn.
      2. **The alone cost:** a lone speculating generation that takes `exclusive` every round keeps ≥ 0.97× of
         today's exclusive spec rate, in process on the copy prompt.
    - **The graded run, after the build, Mac night queue (~60 min).**
      - **Arms:** *batch* (serve's defaults, the do-nothing arm), *spec-exclusive* (today's `-spec ngram`), and
        *candidate*. One binary.
      - **Workloads:** `bench_spec_copy.py`'s copy and W7's chat at 1 and at 4 clients, plus a staggered workload.
        Clients join at 0 / 5 / 10 / 15 s and leave at different times, the case that exercises the switch.
      - 3 interleaved rounds, idle-gated. The 1.5B is graded, the 7B reported.
    - **Gates, candidate against batch:**
      1. every reply equals batch's (plain decode's), every turn, every arm and workload (hard);
      2. alone: copy ≥ 1.25× and chat ≥ 0.97× (hard for chat);
      3. at 4 clients: ≥ 0.97× on both workloads (hard; it must not give up batching);
      4. staggered: aggregate ≥ 1.00×, and p99 turn latency ≤ 1.10× batch's.
    - **Decision:**
      - All pass: `-spec ngram` becomes safe under load on Metal, and `docs/server.md` says so. Whether it becomes
        serve's default on Metal is then a separate owner decision.
      - Copy alone at 1.03–1.25×: the owner's call.
      - Any hard gate fails: parked with the numbers.
    - **CUDA premise, measured 2026-09-29** (queued 2026-09-28, graded 2026-09-29;
      [`spec-vs-batching-cuda-2026-09-29.md`](../../measurements/spec-vs-batching-cuda-2026-09-29.md)): a split verdict,
      by the same rule.
      - **chat clears it:** S = 1.253×, L = 0.790× — the candidate applies on CUDA for chat traffic. Its
        build (`6e9fcb99`, opt-in `-spec-adaptive`) gets the same graded run as Metal's (above), on nobara: registered
        2026-10-01 in [`mc4-candidate-cuda-2026-10-01.md`](../../measurements/mc4-candidate-cuda-2026-10-01.md) and queued overnight.
      - **copy overshoots it:** S = 2.104×, and **L = 1.540× — spec beats MC3's batching even at 4
        clients**, which the rule reads as MC4's own "spec inside a batch" trigger, a different and larger feature
        than this candidate. Parked as its own item, owner's call; not folded into "speculate when alone, batch
        under load". **Investigated 2026-09-29** (§6 of the same record): this candidate's own yielding policy
        underperforms plain `-spec ngram` on copy traffic (206 vs 402 tok/s, exploratory), and a workload-aware
        fix within the switch does not work (concurrent admission is the cost, not the per-round policy) — a
        real reason to caveat `-spec-adaptive` for copy-heavy traffic, not just a missed opportunity elsewhere.
      - Identity held perfectly under load (0 differences, every turn, every workload, both models) — unlike
        Metal's 2026-09-27 record, where later turns diverged before item 30's trailing-token fix existed.
      - The embed-int4 default did not knock the CUDA resident off; every cell read `cuda-resident (int4)`.
  - **P10 on Metal, projected 2026-09-28, not started (owner):**
    [`p10-metal-projection-2026-09-28.md`](../../measurements/p10-metal-projection-2026-09-28.md).
    - Code 1.14–1.49×, math 1.33–1.73×; chat a loss, unguarded.
    - The one unmeasured term is the drafter trunk's Metal cost. Its by-day in-process measurement is step 0 when
      the item is picked up.

## MC5 — continuous batching, paged KV, chunked prefill (parked)

Unchanged from `roadmap.md`: not this engine's weight class. **Trigger:** the owner reverses the
positioning, not a benchmark result.

**Chunked prefill unparked by the owner, 2026-09-27** ("Owner decisions" above). Continuous batching and paged KV
stay parked.

**Chunked prefill SHIPPED 2026-09-27: the revised candidate passes all five gates** ([`chunked-prefill-2026-09-27.md`](../../measurements/chunked-prefill-2026-09-27.md)).
- 512-token chunks with the tail fix, `serve -prefill-chunk 512` by default.
- The decoders' longest stall during a ~3k-token newcomer's prefill fell **5.38 → 1.23 s (0.229×)**.
- Newcomer TTFT 1.107×, wall 1.045×, a lone request 0.998×, every reply identical.

**Chunked prefill, first candidate (256-token chunks): NOT SHIPPED** ([`chunked-prefill-2026-09-27.md`](../../measurements/chunked-prefill-2026-09-27.md)).
- The decoders' longest stall during a ~3k-token newcomer's prefill fell **5.35 s → 1.14 s (0.213×)**.
- Newcomer TTFT 1.137×; every reply identical; a lone request's TTFT 1.000×.
- **But the cell's wall time rose 1.052× against its ≤ 1.05× gate.** By the registered rule it does not ship.
- Chunking is now `decoder.Options.ResidentPrefillChunk` / `serve -prefill-chunk`, 0 (off) by default.
- The diagnosis, exploratory and after the grading:
  - a pass's floor is the prefill GEMM's 64-row tile (~70 ms per 64 tokens on the 1.5B);
  - the candidate's "cut while two chunks remain" rule left a final pass of up to 2C − 1 tokens at full depth, which
    was the 1.14 s longest stall itself (a 1.20 s pass alone).

**Chunked prefill, revised candidate — registered 2026-09-27, before any timing of it.**
- **Two changes, each from the diagnosis:**
  - 512-token chunks: half the passes, so less of chunking's per-pass cost; a 512-token chunk alone costs +5.8% on a
    3000-token prefill, against +7.9% at 256;
  - the tail rule: cut while a chunk plus 8 tokens remain, so the last pass is 8 .. C + 7 tokens, never the
    up-to-2C tail.
- The build is the commit carrying `Options.ResidentPrefillChunk`; *new* = that build with `-prefill-chunk 512`.
- *old* = `serve-metal` at `731f4f4e` (S3), as before.
- The bench, cells, pairs and all five gates are exactly the first candidate's, bars unchanged.
- Decision: all pass ships, with `serve -prefill-chunk` defaulting to 512; anything else leaves it at 0.

**Chunked prefill — registered 2026-09-27, before any timing.**

*The stall it removes.* Under MC3, a newcomer's prefill runs whole in the resident's exclusive section, so every
decoding conversation stops for the entire prompt. W7 cannot show this: its turns reuse their history and prefill
~150-token suffixes. A long prompt arriving mid-decode, such as a pasted file, a RAG context, or an agent's first
turn, stalls the others for the whole prefill.

*The question the design turns on, measured first.* Is Metal's fast prefill chunk-invariant? Is
`PrefillLast(chunk, startPos)` one after another the same bits as one `PrefillLast` over the whole prompt?
- `TestMC5_prefillChunkInvariance` prefilled a 1000-token prompt whole into one slot and in chunks of 64, 128, 256
  and 384 (ragged tails) into another. **0 of 14,336,000 KV elements and 0 of 151,936 last-token logits differ**, at
  every chunk size.
- Its control changes one token at position 700: 0 elements differ before it, 3.7 M from it on, so the comparison
  sees a difference and places it.
- So chunking changes no reply, and it can be switched on only under load.

*The build* (`decoder/model.go` `mc3Prefill`, `decoder/mc3_batch.go`).
- Under MC3, a prompt suffix of ≥ 512 tokens (`prefillChunkMin`) is prefilled in 256-token chunks
  (`prefillChunkTokens`), and only while another generation is decoding.
- Each chunk runs in its own exclusive section, and one decode step is yielded between chunks (`yieldToDecode`,
  bounded at 50 ms).
- A chunk is cut only while two chunks' worth remains, so the pass that ends the prefill is never shorter than a
  chunk. A shorter tail would take the sequential path, whose numerics differ.
- With nobody decoding, the rest goes in one pass, exactly as before. A declined batched prefill falls back to the
  ordinary path.
- Tested on the fake resident: `TestMC3_longPrefillChunksWhileOthersDecode`. A 1200-token newcomer arrives in 4
  chunks with decode steps between them, and every reply equals the alone run. It goes red with no yield, and red
  with chunking off.

*Grading, pre-registered.*
- *old* = `serve-metal` at `731f4f4e` (S3); *new* = the chunked-prefill commit.
- `scripts/bench_prefill_stall.py`:
  - 3 decoder clients each stream 400 greedy tokens from a short prompt;
  - after 2 s, 3 newcomers arrive one after another, each with a ~2400-word prompt (~3k tokens) and 16 answer tokens;
  - a fresh server per cell, serve defaults, qwen2.5-coder-1.5b int4 from `~/models`, idle-gated.
- Cells: old/new × 3 pairs (old new new old old new). A solo arm (`--decoders 0`, the 3 newcomers alone) runs old/new
  × 3 pairs likewise.
- Gates:
  1. identity: every reply's hash equal, new vs old, decoders and newcomers, every cell (hard);
  2. the stall: the decoders' max inter-token gap, new ÷ old, median of 3 pairs ≤ 0.5×;
  3. newcomer TTFT: new ÷ old, median over newcomers and pairs ≤ 1.5× (chunking interleaves decode steps into it);
  4. the cell's wall time: new ÷ old, median ≤ 1.05× (no throughput cost);
  5. solo guard: the solo arm's newcomer TTFT, new ÷ old, median ≤ 1.05× (hard; nothing chunks with nobody
     decoding).
- Reported: the decoders' p50 and p99 gaps.
- Decision: all pass ships (on by default under MC3). Gate 2 between 0.5× and 0.8× goes to the owner. Gate 1 or 5
  failing is a bug.

---

## Amendments

*2026-09-26, before any measurement: speed bars loosened per the owner's standing guidance.* In the owner's words:
"i'm happy to park stuff if it's a couple percent above current code, but giving up gains seems stupid … in the
future let's be a lot more permissive". That came after R18, which parked a bit-identical 1.17× on a ship line derived
from a theoretical bound.

Hard gates are unchanged: bit-identity or the registered reference gate, reuse equal turn for turn, the fit guard's
decline, and p99 latency. Only the speed thresholds move:

| stage | as filed 2026-09-23 | now |
|---|---|---|
| MC1 aggregate at 2/4 clients | ship within 0.90–1.0× of 1 client; below 0.85× at 4 clients is "a finding, not a ship" | **Ships** if the hard gates hold and the 4-client aggregate beats today's by more than noise (W7: 36.39 tok/s). The 0.90–1.0× band stays as the *expected* reading and is reported; a result below it is explained in the record, not grounds to hold the fix back. |
| MC2 (B=4 vs B=1) | earn ≥ 1.6×, kill < 1.3×, owner between | **earn ≥ 1.25×, kill < 1.1×**, owner between. Still an investment gate for MC3's 4–8 weeks, so the gray zone goes to the owner with the numbers. |
| MC3 (4 clients vs MC1's 4-client aggregate) | ≥ 1.5× | **≥ 1.2×**, with the p99 latency bar (≤ 1.5× MC1's 1-client figure) kept as a hard gate |
| J8's cell (folded into MC2) | ≥ 1.25× at 4 requests, p99 ≤ 1.5× | unchanged. It is reported beside MC2 and is not built as a feature either way. |

*2026-09-26, before the MC3 build, from S0's projection: the reading of MC3's p99 bar (owner).*
- S0 ([`concurrency-mc3-s0-2026-09-26.md`](../../measurements/concurrency-mc3-s0-2026-09-26.md)) projects a B = 4 Metal
  step at ~2.1× a single step, and B = 2 at ~1.8×. So under load a batched request takes ~2× as long as a lone one,
  and the bar as filed (≤ 1.5× MC1's *1-client* figure) projects to failing for any batching.
- MC3c was graded on the same J8 bar at the same offered load (0.51–0.52×), and reported the lone-request ratio
  (2.0–2.2×) without gating on it.
- Asked which reading MC3 is held to, the owner chose **"same load + solo guard"**:
  - MC3's 4-client p99 ≤ MC1's 4-client p99;
  - and a lone request within noise of MC1's (≤ 1.05×, as registered in the MC3 gates above).

  Both are hard gates. The aggregate gate (≥ 1.2×) and the identity gate are unchanged.

## Not in scope, stated

- **N independent decode workers per model** as a shipped feature. MC2 may measure it (decision 2);
  it is not built here.
- **Persisting GPU slots across restarts.** `-session-dir` persistence stays with the CPU
  `sessionLRU`; an evicted GPU slot is simply gone.
- **Moving KV between GPU slots and the CPU session tier.** A possible follow-on to MC1 if MC0
  shows eviction still hurts at realistic client counts.
- **Multi-model scheduling.** Distinct models already run in parallel; nothing here changes that.

<!-- doc-reviewed: 2026-10-01 -->
