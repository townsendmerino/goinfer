# Task: concurrency — stop the resident-KV thrash, then earn batched multi-request decode (MC0–MC5) — 2026-09

> **Status: FILED 2026-09-23; owner decisions taken 2026-09-26 (1: yes, 2: fold). MC0 DONE 2026-09-26: the
> thrash is confirmed on Metal, and a CPU session-LRU bug it uncovered is fixed. MC1 SHIPPED 2026-09-26 on Metal**
> (`c2f1532e`): 4 resident KV slots hold the 1-client aggregate at 2 and 4 clients (0.99× / 1.01×), and the 4-client
> aggregate is 1.22–1.24× the previous build's. CUDA and WebGPU are not converted. **MC2 EARNS on the Mac CPU**
> (1.69–2.04× at B = 4, bit-identical; J8's 4 independent workers reach 2.00–2.48×). The Linux cells are owed. **MC3
> S0 is done, 2026-09-26: no fidelity gate is needed.** Test-only `simdgroup_matrix` kernels carry 8 sequences for
> 1.0–2.7 GEMVs on every decode matmul (qkv, o, gate/up, down, int8 LM head). Every output is bit-identical to
> production's GEMV ([`concurrency-mc3-s0-2026-09-26.md`](../measurements/concurrency-mc3-s0-2026-09-26.md)). One
> owner question is open before the build: the reading of the p99 bar. The owner chose "cpu first":
> **MC3c step 1 SHIPPED** (`serve -max-concurrent N`, CPU models): 1.86–1.97× at 4 clients, p99 halved,
> byte-identical, **default 4** (owner, 2026-09-26). The MC2 Linux cells are in: the 1.5B earns, and there batching beats
> the workers; the 0.5B is in the owner band.
>
> **Open, 2026-09-26:**
> - MC3c step 2 (batching behind the same admission; the Linux 1.5B data argues for it on larger models);
> - MC3 (Metal batched decode): S0 done, bit-identical; the p99 bar's reading is the owner's before the build;
> - MC1 on CUDA and WebGPU;
> - one MC2 Linux cell to re-run clean.
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
- Not converted: CUDA and WebGPU (one slot); recurrent families keep one; VL turns use the bound slot.
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

## MC5 — continuous batching, paged KV, chunked prefill (parked)

Unchanged from `roadmap.md`: not this engine's weight class. **Trigger:** the owner reverses the
positioning, not a benchmark result.

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
