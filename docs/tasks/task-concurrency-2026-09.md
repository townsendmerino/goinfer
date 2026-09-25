# Task: concurrency — stop the resident-KV thrash, then earn batched multi-request decode (MC0–MC5) — 2026-09

> **Status: FILED 2026-09-23, nothing built.** MC0 (measurement only) and MC1 (multi-slot resident
> KV) are unblocked. MC2 is the kill-or-earn measurement `roadmap.md` requires before any batched
> decode work; MC3 is written but must not start until MC2 earns. MC4 and MC5 are parked with
> triggers. Two owner decisions are flagged below; neither blocks MC0 or MC1.
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
>    `sessionLRU` keeps `-kv-sessions` (default 4) conversations warm — but `Session.Generate`'s
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

1. **Does "single-user" include one user's parallel agents?** An agent harness that fans out
   subagents is one user on one machine, but it is not batch-1. If yes, MC3 fits the niche and
   `positioning.md` gains one sentence; if no, this doc ends at MC1 and MC2's result is recorded
   as a finding.
2. **J8 as written, or folded into MC2?** J8 measures N independent decode workers per model. Its
   own text calls batching "the real alternative." MC2 can run J8's cell in the same session at no
   extra build cost, so the two answers land side by side. Recommendation: fold.

---

## MC0 — confirm the diagnosis (measurement only)

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

**Follow-on.** With more than one slot, J6's prefix-aware scheduling has something to exploit.
Re-run J6's measurement once after MC1 ships; its 1.024× was taken against a single slot.

**Estimate.** One to two weeks across the three backends, mostly bookkeeping and the fit guard.

## MC2 — kill or earn: batched decode on CPU

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
- W7 plain workload, 4 clients: aggregate **≥ 1.5×** MC1's 4-client figure.
- p99 per-request latency no worse than **1.5×** MC1's 1-client figure (J8's latency bar, kept).

**Estimate.** Four to eight weeks for Metal. A CUDA port follows only on its own measurement.

**Docs that change when MC3 ships:** `positioning.md` (the "one generation at a time" sentence),
`roadmap.md` §"Decided and parked," `docs/server.md`, red-october R12's row, `QUEUE.md`.

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

## Not in scope, stated

- **N independent decode workers per model** as a shipped feature. MC2 may measure it (decision 2);
  it is not built here.
- **Persisting GPU slots across restarts.** `-session-dir` persistence stays with the CPU
  `sessionLRU`; an evicted GPU slot is simply gone.
- **Moving KV between GPU slots and the CPU session tier.** A possible follow-on to MC1 if MC0
  shows eviction still hurts at realistic client counts.
- **Multi-model scheduling.** Distinct models already run in parallel; nothing here changes that.

<!-- doc-reviewed: 2026-09-23 -->
