# Task: concurrency — stop the resident-KV thrash, then earn batched multi-request decode (MC0–MC5) — 2026-09

> **Status: FILED 2026-09-23; owner decisions taken 2026-09-26 (1: yes, 2: fold). MC0 DONE 2026-09-26: the
> thrash is confirmed on Metal, and a CPU session-LRU bug it uncovered is fixed. MC1 SHIPPED 2026-09-26 on Metal**
> (`c2f1532e`): 4 resident KV slots hold the 1-client aggregate at 2 and 4 clients (0.99× / 1.01×), and the 4-client
> aggregate is 1.22–1.24× the previous build's. **MC1 SHIPPED on CUDA 2026-09-27** (`9fddaf7e`,
> [`concurrency-mc1-cuda-2026-09-27.md`](../measurements/concurrency-mc1-cuda-2026-09-27.md)): on the 1.5B, 4 clients
> reach 1.250× the one-slot build, and a lone request is unchanged. WebGPU is not converted. **MC2 EARNS on the Mac CPU**
> (1.69–2.04× at B = 4, bit-identical; J8's 4 independent workers reach 2.00–2.48×). The Linux cells are owed. **MC3
> SHIPPED 2026-09-26 on Metal** (`d4b708b5` + fixes `2b1cc280`, `d2225ec4`;
> [`concurrency-mc3-2026-09-26.md`](../measurements/concurrency-mc3-2026-09-26.md)). All five pre-registered W7 gates
> pass: 4 clients at **1.593×** the serialized aggregate (76.2 → 121.5 tok/s), p99 turn **0.659×**, a lone request
> **1.002× / 1.004×** (p50 / p99), and every reply identical. It is on under serve's default `-max-concurrent` 4.
> **MC3 S0: no fidelity gate is needed.** Test-only `simdgroup_matrix` kernels carry 8 sequences for
> 1.0–2.7 GEMVs on every decode matmul (qkv, o, gate/up, down, int8 LM head). Every output is bit-identical to
> production's GEMV ([`concurrency-mc3-s0-2026-09-26.md`](../measurements/concurrency-mc3-s0-2026-09-26.md)). **S1**:
> a whole batched step, in sequence, is bit-identical to production and reaches 1.73–1.84× at B = 4. The owner set
> the p99 bar's reading (same load + a lone-request guard) before the build. The owner chose "cpu first":
> **MC3c step 1 SHIPPED** (`serve -max-concurrent N`, CPU models): 1.86–1.97× at 4 clients, p99 halved,
> byte-identical, **default 4** (owner, 2026-09-26). The MC2 Linux cells are in: the 1.5B earns, and there batching beats
> the workers; the 0.5B is in the owner band.
>
> **Open, 2026-09-26:**
> - **MC3c step 2 (batching behind the same admission): BUILD, by the pre-registered trigger** (2026-09-27, `nobara`
>   CPU, 7B). Batched B = 4 ÷ J8 N = 4 is 2.41× at depth 128 and 2.25× at 512, against a 1.15× bar. On the 7B the
>   workers barely scale (1.13–1.15×) and batching reaches 2.59–2.73× ([`concurrency-mc2-2026-09-26.md`](../measurements/concurrency-mc2-2026-09-26.md),
>   "Linux 7B cell"). Design, correctness gates and W7 grading pre-registered 2026-09-27 (MC3c, "step 2"); not built;
> - MC3 follow-ons, none registered: a hybrid B = 2 (it reads 1.07× on the 1.5B, ~1.0× on the 7B), an encode-ahead
>   executor for steps, and CUDA (only on its own measurement);
> - ~~the 7B end to end~~ — done 2026-09-27: all five W7 gates pass, 4 clients at 1.785× the serialized build, p99 turn
>   0.592× ([`concurrency-mc3-7b-w7-2026-09-27.md`](../measurements/concurrency-mc3-7b-w7-2026-09-27.md));
> - ~~MC1 on CUDA~~ — shipped 2026-09-27: 4 clients at 1.250× the one-slot build, every hard gate passes
>   ([`concurrency-mc1-cuda-2026-09-27.md`](../measurements/concurrency-mc1-cuda-2026-09-27.md)); MC1 on WebGPU;
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
> MC4 and MC5 stay parked. MC2 is the kill-or-earn
> measurement `roadmap.md` requires before any batched decode work, and now carries J8's cell. MC3
> is in the niche (decision 1) but must not start until MC2 earns. MC4 and MC5 are parked with
> triggers. The speed bars were loosened 2026-09-26, before any measurement, per the owner's
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
> **Siblings.** [`red-october.md`](red-october.md) R12 (owns W7; this doc is its follow-on) ·
> [`task-work-queue-2026-09.md`](task-work-queue-2026-09.md) (J1 fair admission, which MC3 changes;
> J6 prefix-aware scheduling, measured 1.024× and not shipped, which MC1 may re-open; J8, which
> MC2 absorbs if the owner agrees) · [`../positioning.md`](../positioning.md) ("not a serving
> engine") · [`../roadmap.md`](../roadmap.md) §"Decided and parked" (continuous batching) ·
> [`task-fit-to-hardware.md`](task-fit-to-hardware.md) (owns the fit guard MC1 must price) ·
> [`task-never-swap-2026-09.md`](task-never-swap-2026-09.md) (MC1's extra KV is anonymous GPU
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

**Result, 2026-09-26: CONFIRMED** ([`concurrency-mc0-2026-09-26.md`](../measurements/concurrency-mc0-2026-09-26.md)).
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
([`concurrency-mc1-2026-09-26.md`](../measurements/concurrency-mc1-2026-09-26.md)).
- On the W7 workload with 4 slots, the aggregate at 2 and 4 clients is 0.989× / 1.007× the 1-client figure.
- The 4-client aggregate is 70.9 against the previous build's 57.0 tok/s (1.24×; the repeat reads 1.22×).
- Each turn prefills only its new user turn at every client count.
- Outputs are bit-identical to each conversation served alone, on the tiny fixture and on qwen2.5-coder-1.5b.
- Not converted: WebGPU (one slot); recurrent families keep one; VL turns use the bound slot. CUDA shipped
  2026-09-27; see "MC1 on CUDA" below.
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

**Follow-on.** With more than one slot, J6's prefix-aware scheduling has something to exploit (now true on Metal).
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

**Result, 2026-09-27: SHIPPED** ([`concurrency-mc1-cuda-2026-09-27.md`](../measurements/concurrency-mc1-cuda-2026-09-27.md)).
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

**Estimate.** One to two weeks across the three backends, mostly bookkeeping and the fit guard.

## MC2 — kill or earn: batched decode on CPU

**Result on the Mac, 2026-09-26: EARN** ([`concurrency-mc2-2026-09-26.md`](../measurements/concurrency-mc2-2026-09-26.md)).
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

**S0, 2026-09-26** ([`concurrency-mc2-s0-2026-09-26.md`](../measurements/concurrency-mc2-s0-2026-09-26.md)): on the
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
([`concurrency-mc3-s1-2026-09-26.md`](../measurements/concurrency-mc3-s1-2026-09-26.md)).
- A test-only step runs production's per-sequence kernels (norms, RoPE, KV, attention, SwiGLU) and S0's batched
  matmuls, each sequence on its own MC1 slot.
- It matches production's single-token forward on every logit: 1.5B and 7B, 4 sequences × 12 steps, 0 differ.
- In sequence, B = 4 reaches 1.73–1.84× one stream's aggregate (the step costs 2.1–2.3× a token) and B = 8 reaches
  2.4–3.2×, within ~5% of S0's projection.
- B = 2 is a wash (0.98–1.10×) and B = 1 loses (0.51–0.64×), so production's path runs below B = 3.
- S1 also found and fixed an MC1 slot-clamp double count.

**S0, second pass, 2026-09-26: the matrix units pay AND stay bit-identical, so MC3 keeps the identity gate**
([`concurrency-mc3-s0-2026-09-26.md`](../measurements/concurrency-mc3-s0-2026-09-26.md)).
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
([`concurrency-mc2-2026-09-26.md`](../measurements/concurrency-mc2-2026-09-26.md), MC3 S0).
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

**SHIPPED 2026-09-26: all five W7 gates pass** ([`concurrency-mc3-2026-09-26.md`](../measurements/concurrency-mc3-2026-09-26.md)).
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
  ([`concurrency-mc3-7b-w7-2026-09-27.md`](../measurements/concurrency-mc3-7b-w7-2026-09-27.md)).
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

**MC3 S3 SHIPPED 2026-09-27: all five W7 gates pass** ([`concurrency-mc3-s3-2026-09-27.md`](../measurements/concurrency-mc3-s3-2026-09-27.md)).
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

**MC3 S2 SHIPPED 2026-09-27: all five W7 gates pass** ([`concurrency-mc3-s2-2026-09-27.md`](../measurements/concurrency-mc3-s2-2026-09-27.md)).
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

## MC3c — concurrent decode on CPU (owner: "cpu first", 2026-09-26)

**Step 1 result, 2026-09-26: all five gates pass**
([`concurrency-mc3c-2026-09-26.md`](../measurements/concurrency-mc3c-2026-09-26.md)).
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

**Step-2 trigger result, 2026-09-27: BUILD STEP 2** ([`concurrency-mc2-2026-09-26.md`](../measurements/concurrency-mc2-2026-09-26.md),
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
  still larger than batching's per-sequence share.
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
  ([`spec-vs-batching-metal-2026-09-27.md`](../measurements/spec-vs-batching-metal-2026-09-27.md)).
  - S: copy 0.979×, chat 0.928×. Spec gives a lone Metal request nothing.
  - L: copy 0.610×, chat 0.489×. Under 4-client load it forfeits MC3's batching.
  - Found on the way, both open:
    - with `--spec ngram`, chat turns after the first differ from plain decode (turn 0 equal in every conversation).
      **Fixed 2026-09-27:** the loop now forwards a round's trailing token when `max_tokens` ends the generation, as
      plain decode does. The next turn had re-prefilled it through Metal's non-bit-identical f16 prefill. Every W7
      turn now matches (`TestGenNgramInto_residentCommitMatchesPlain`, `TestSpecNgram_multiTurnMatchesPlain`).
      The other loops were checked the same day.
      - The block drafter (`--drafter`, CUDA) committed one position more than it wrote, so the next turn reused a
        stale one. It is fixed and pinned by `TestBlockSpecGenerate_commitsOnlyWrittenPositions`, but its CUDA check on
        nobara is owed.
      - The grammar-fused loop was one short and is fixed.
      - The two-model loop never commits, and is unchanged;
    - on the 7B, serve's prefill-memory share 413'd MC3 batch cells, because it divides the margin as if prefills ran
      concurrently. **Fixed 2026-09-27:** a request that prefills on a resident keeps the whole live margin
      (`TestPrepare_prefillShare`). The end-to-end rerun is owed: the attempt ran on a Mac already swapping.

## MC5 — continuous batching, paged KV, chunked prefill (parked)

Unchanged from `roadmap.md`: not this engine's weight class. **Trigger:** the owner reverses the
positioning, not a benchmark result.

**Chunked prefill unparked by the owner, 2026-09-27** ("Owner decisions" above). Continuous batching and paged KV
stay parked.

**Chunked prefill SHIPPED 2026-09-27: the revised candidate passes all five gates** ([`chunked-prefill-2026-09-27.md`](../measurements/chunked-prefill-2026-09-27.md)).
- 512-token chunks with the tail fix, `serve -prefill-chunk 512` by default.
- The decoders' longest stall during a ~3k-token newcomer's prefill fell **5.38 → 1.23 s (0.229×)**.
- Newcomer TTFT 1.107×, wall 1.045×, a lone request 0.998×, every reply identical.

**Chunked prefill, first candidate (256-token chunks): NOT SHIPPED** ([`chunked-prefill-2026-09-27.md`](../measurements/chunked-prefill-2026-09-27.md)).
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
- S0 ([`concurrency-mc3-s0-2026-09-26.md`](../measurements/concurrency-mc3-s0-2026-09-26.md)) projects a B = 4 Metal
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

<!-- doc-reviewed: 2026-09-23 -->
