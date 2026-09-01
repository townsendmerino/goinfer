# Engineering queue — CLOSED ENTRIES, archived 2026-08-31

Closed entries from [`docs/queue-engineering.md`](../queue-engineering.md), moved here 2026-08-31
so the live queue holds only open work. The same split the performance and correctness queues use.

**Thirteen entries, moved verbatim — headers, evidence and all.** Read them for what was learned,
not for status; several record a mechanism that outlived the task.

**What was deliberately NOT moved, and why it matters more than what was.** Four entries LOOK
closed in their headers and are not, so they stayed live:

| entry | header reads | residual |
|---|---|---|
| `B8` position-keyed pins | "audited 2026-08-12" | "the gates are clean, the PROSE is not" |
| `B7` off-origin work | "swept, 2026-08-12" | the sweep's action column is not all discharged |
| `F2` §2/§3 criticals | "SWEPT 2026-08-12" | "two lack an anchor" |
| `F3` G-01 class | — | "status unconfirmed" |

Archiving on a header keyword would have moved all four and buried the residual. That is the same
stale-header defect the correctness queue collected four times; here it would have been caused BY
the cleanup rather than found by it, which is why the classification was read entry by entry
instead of grepped.

Also left live: `E7` (inventory done, migrations pending), `E8` (in progress), `B3`'s successor
`B18`, and `B4` (reopened).

---

**G21 · Incremental tool-call-aware streaming (Finding 1, stage 2)** — `mac`, **DONE 2026-08-26.**
The real fix behind G19's heartbeats.

G19 stopped the tool path from being *silent* (comment frames while it buffers). It did not make it
*incremental*: the whole generation is still buffered, so a client sees no prose until the model
finishes. This emits prose deltas as they generate and holds back only what could still turn into a
tool call.

**The safety property, which is what makes this delicate:** anything streamed early must be a
BYTE-EXACT PREFIX of the `lead` that `ParseToolCalls` ultimately computes. A delta cannot be
unsent, so an over-eager emit is unrecoverable corruption, not a cosmetic bug.

**Survey of the four families (2026-08-26) — the scope follows from it:**

| family | how `lead` is computed | streamable |
|---|---|---|
| `chatml` / `mellum2` | `Cut(out, "<tool_call>")` — raw prefix, untrimmed | **YES** |
| `gemma4` | `Cut(out, "<\|tool_call>")` — raw prefix, untrimmed | **YES** |
| `mistral` | `TrimSpace(before "[TOOL_CALLS]")` | **NO** — trimming means the streamed bytes would not equal `lead` |
| `llama3` | `TrimSpace(out)`, strips `<\|python_tag\|>`, then `lead` only if the JSON at the first `{` parses | **NO** — normalized AND parse-dependent; nothing is known until the object parses |

So: incremental for the two families whose `lead` is literally a prefix of the raw output, buffered
(exactly as today) for the two where it is not. ChatML covers the Qwen family — the common tool
case, and the one the dsh run used — so this is most of the value at none of the risk.

**Mechanism:** hold back the longest suffix of pending output that is a proper prefix of the
family's opener literal; emit everything before it. On seeing the opener, stop emitting and revert
to buffering for the rest of the generation. `ParseToolCalls` still runs over the FULL buffer at the
end, unchanged — this changes only *when* prose leaves, never how calls are parsed.

**THE SURVEY WAS WRONG IN ONE PLACE, AND THE CONTRACT TEST CAUGHT IT BEFORE ANY BYTE SHIPPED.**
The table above says `chatml`/`gemma4` compute `lead` as a raw untrimmed prefix. They do not — both
return `strings.TrimSpace(lead)`. A streamer built on the raw-prefix assumption would have emitted
leading and trailing whitespace the parser discards, breaking the byte-exact prefix property in the
one direction that cannot be repaired. `TestToolCallOpenerMatchesParser` asserted the contract
against the real parsers rather than trusting the switch statement, and failed on the first run.

The design absorbed it: `chat.ProseStreamer` normalizes exactly as the parsers do — never emits
leading whitespace, holds a trailing whitespace run until a non-space follows, drops it if the
opener arrives instead, and holds any partial opener at the tail (`StreamableLen`).

**Landed:** `Template.ToolCallOpener` (the per-family contract), `chat.StreamableLen` (hold-back),
`chat.ProseStreamer` (the normalizing streamer), and the chat tool path emitting prose deltas as
they generate, with the final send emitting only the REMAINDER of the lead. `ParseToolCalls` still
runs over the full buffer, unchanged — this changes *when* prose leaves, never how calls are parsed.

**Gates:**

| gate | where |
|---|---|
| streamed bytes are a prefix of the parsed `lead`, byte-at-a-time, every streamable family, incl. whitespace shapes | `chat.TestProseStreamerMatchesParser` |
| hold-back releases exactly the prose before the opener under bytewise chunking | `chat.TestStreamableLen_bytewiseNeverOverruns` |
| partial openers held; opener at 0; empty opener | `chat.TestStreamableLen_holdsBackPartialOpeners` |
| non-streamable families stay non-streamable | same contract test |
| a call-only generation leaks NO content delta | `serveapp.TestToolStreamNeverLeaksCallAsProse` |
| tool call still parses; `[DONE]` still terminates; non-streaming unchanged | the G19 tests, still green |

Mutation-checked: removing the hold-back fails all three chat gates.

**Limit of the end-to-end coverage, stated.** The 1.5B available here emits a tool call for *every*
prompt tried — prose requests, arithmetic, "say one word" — and never any prose, so "prose arrives
before the generation ends" cannot be observed through it. That property is proven exhaustively at
the `chat` level instead, byte-at-a-time against the real parsers, which is stronger evidence than
one model's behavior. The serveapp test asserts the corruption case that model IS ideal for: with a
call-only generation, zero content deltas.

**G19 · Tool-path streaming is SILENT while it buffers — SSE heartbeats (Finding 1, stage 1)** —
`mac`, **DONE 2026-08-25. Ships in v0.15.0** (decision:
`docs/measurements/dsh-tier0-decisions.md`, which said v0.14.0 — that tag was already cut on
2026-08-19; v0.15.0 is the open release).

**The buffering is correct; the silence is the defect.** `tools.go` says it outright — *"Tool
decisions need the whole output, so buffer (even when streaming)"* — because a tool call can only
be parsed from the complete output. But it means `stream: true` with tools declared sends **zero
bytes until the generation finishes**. Measured against the real dsh agent request: **first byte at
1682.6s**, all 495 bytes at once. dsh's default `streamIdleTimeoutMs` is **300000ms** and `TIMEOUT`
is in its retry set, so any generation slower than five minutes cannot succeed through it however
correct the output — and each retry stacked another generation (that half is G18).

**Stage 1 (this item):** emit SSE **comment frames** (`: ping`) while the buffer holds. Comments are
protocol-legal, content-free, and invisible to every SSE parser, so they defeat idle timeouts —
dsh's and every other harness's — without touching the tool-call parsing the buffering exists for.

**Two sites, both buffer-then-stream:** `internal/serveapp/tools.go` (chat completions) and
`respondTools` in `internal/serveapp/responses.go`.

**One consequence to state, not smuggle:** heartbeats require `sseStart` BEFORE `drive`, so a
generation error can no longer be a 500 on those paths — headers are already flushed. That is not a
new convention: `sseErr` exists for exactly this ("the response is already 200 with headers
flushed, so a status code is no longer available", M1) and the non-tool streaming paths already use
it. Non-streaming requests keep the 500 unchanged.

**Gates — landed** in `internal/serveapp/heartbeat_test.go`, against qwen2.5-coder-1.5b:

| gate | result |
|---|---|
| frames flow while the buffer holds | **71 heartbeats before the first data frame** (3 data frames total) |
| no content delta emitted early | asserted per-frame; the single whole-message delta is still the only content |
| tool call parses identically | `tool_calls` delta seen, stream ends with `[DONE]` |
| non-streaming path unchanged | 200, and `Content-Type` is never `event-stream` |

Mutation-checked: disable the heartbeat and the count goes to **0** with the same 3 data frames —
the gate fails on exactly the property it exists for. `sseHeartbeatInterval` is a package var
(10s in production) so the test drives the mechanism rather than the wall clock.

**Stage 2, queued (NOT this item):** incremental tool-call-aware streaming — emit prose deltas,
hold back only from a potential tool-call opening, per family via the `chat` package's existing
parsers. Real work; deliberately not rushed into the release.

**G18 · Prefill ignores cancellation — an abandoned client leaves a core burning** — `mac`,
**DONE 2026-08-25. Ships in v0.15.0** (decision:
`docs/measurements/dsh-tier0-decisions.md`, which said v0.14.0 — already tagged 2026-08-19).

Correctness / consumer-trust class, **not** perf — the class C4-soak exists to catch, found early
by a real harness instead (the dsh Tier-0 run). Measured before-state, twice: goinfer climbed
**14:59 → 19:53 → 22:01** of CPU with no client attached, and reached **47:38** after a second
killed client. A retrying harness stacks generations — dsh retries a `TIMEOUT` five times, and each
retry opens a new prefill while the abandoned one keeps running.

**The seam is already right and that is the trap.** `internal/serveapp/tools.go` passes
`r.Context()` into `lm.drive`, so the cancellation looks wired at review. It does not reach the
work: `generateInto` (`decoder/model.go`) had the ctx but called `m.prefillLogits(prompt[prefillFrom:],
cache)` without it, and `prefillLogits` / `forwardLayersN` / `runLayersFromEmbedN` took no context at
all. (Cited by function, not line — this entry's own fix shifted those line numbers and broke the
citation lint.) The resident prefill loop in `generateInto`
(`for i, id := range prompt`) does not check it either.

**Change:** thread ctx through the prefill chain and check it per layer in the batched path, per
token in the sequential fallback and the resident loop. At ~28 layers the check is free (the
decisions doc: "at 25 tok/s the check is free") and bounds wasted work to one layer.

**Gates (from the decision) — all three landed** in `decoder/prefillcancel_test.go`, measured on an
M1 Pro against qwen2.5-coder-1.5b:

| gate | result |
|---|---|
| cancelled before start | returns in **746µs** (a 512-token prefill is seconds) |
| cancelled mid-flight (retry-storm shape) | cancel at 300ms, observed at **12.34s** |
| decode's existing behavior — checked, not assumed | stopped **0 tokens** after cancel; `gen.Err() = context canceled` |

Mutation-checked: with the per-layer check removed, the cancelled-context test returns `err = nil`
after doing the whole prefill.

**The bound is one LAYER, and it is not instant — say so.** 12.34s at a 3072-token prompt is the
measured tail, against a full prefill of ~335s (int8int8) or ~1587s (int4) before the fix. That is
enough to stop retry-storm accumulation (dsh retries on a 300s timeout) and it is not the same
claim as "cancellation is immediate". Tightening it further means checking per-head inside
`attendBatchedHeads`; filed here as the known tail rather than left to be discovered.

**Scope confirmed rather than assumed:** decode already honored cancellation — the defect was
prefill-only, exactly as the decision stated.

**Parity:** the edit touches `core` forward files, so `TestParityManifest_fresh` re-staled 27
validated families. Discharged the sanctioned way — `scripts/refresh_parity_hashes.sh`, which runs
the forward goldens as independent numeric ground truth first and refuses on any failure or on a
vacuous zero-ran green. Goldens **27 passed / 21 skipped / 0 failed**; diff verified deps_hash-only,
`validated_at` preserved. The check is a uniform early-return in the shared layer loop, so it is
not a path a skipped family's golden would have exercised differently.

**G14 · Tier 0 — the tested "Use goinfer with DeepSeek Harness" recipe** — `mac`+`linux`, **DONE
2026-08-26. The gate is MET and the recipe is written** (`README.md`, "Use goinfer with DeepSeek
Harness"). Qualifying run: dsh on the mac driving a CUDA-resident goinfer on the linux box over
Tailscale through a two-tool, multi-step task — `glob` → `read` → synthesis — completed in 277 s
with **zero retries**, verified from the session transcript rather than from the answer. Prefill
270 tok/s (2116 tokens in 7.84 s) vs ~30 tok/s on mac CPU. Full record in
`docs/measurements/dsh-tier0-run-2026-08-25.md`. Scope, tiers and the standing "nothing in goinfer may depend on dsh" rule
are in `docs/scoping-dsh-goinfer.md`; this is its Tier 0 only.

**Gate (unchanged):** the recipe may be written ONLY from a real end-to-end run — dsh web driving
goinfer through a multi-turn, tool-using agent task — with every friction point fixed, worked
around in the recipe, or filed. Findings to `docs/measurements/`.

**Unblocked by G12** (`4ca19e9`): the `developer` alias means the run uses dsh's default flags, and
the recipe gets no workaround section — at most one line for anyone pinned to a pre-`4ca19e9`
goinfer.

**Model decision, made 2026-08-25 because nothing at the doc's recommended bar is on this box**
(no Qwen3.8-27B / gpt-oss-20b / DeepSeek-lite): **two passes.** Wire-level shakeout on
`qwen2.5-coder-1.5b` (1.0 GB, 32k, has tools) where an iteration costs seconds and the friction
list gets found cheaply; then the QUALIFYING multi-turn run on `gemma4-26b-int4` (15 GB, closest to
the bar on disk, and the one that actually exercises the resident re-prefill trade-off the recipe
warns about). The recipe rests on the second run. If the recipe ends up quoting the 1.5B for
anything, that must be said in the recipe. `Qwen3.5-35B-A3B` is on disk and was ruled out here on a
figure that has since been superseded TWICE — see the correction below before reusing this call.

**CORRECTION (2026-08-28).** The "~1.2-1.4 tok/s" that disqualified `Qwen3.5-35B-A3B` was already
withdrawn in `task-zeno-compare.md` (noise-contaminated, superseded by 1.605 CPU-paged) before this
line was written, and is now superseded again by direct measurement: the CPU pager runs it at
**1.52-1.73 tok/s**, and the Metal expert pager at **1.97-2.02 tok/s**
(`task-metal-expert-streaming-at-scale.md`). Whether ~2 tok/s makes an agent loop practical is a
judgement this doc should make deliberately — but it should not keep making it on a retracted number.
The two-pass decision above is otherwise untouched; only its exclusion premise was stale.

**A12 · The CUDA heavy tier does not fit in 8 GB in one process — the gate cannot pass here** —
`linux`, **CLOSED `e682eb2` (2026-08-13). No capacity/leak issue ever existed — see the resolution
block immediately below before reading the investigation.**

**RESOLUTION (the actual final word — `e682eb2`, 2026-08-13 09:58:30, 30-50 min after every
"below" section, which none of them incorporate; this doc was never updated to say so until
2026-08-19).** The four failures had **four independent causes**, not one shared mechanism:

1. `TestSplitKV_bitIdentical_gemma3` — a test defect. It panicked on a CORRECT resident decline
   (context didn't fit) via an unchecked type assertion; now skips with the decline named.
2. `TestResidentLaunchVRAMProbe` — a test defect. It already printed "the launch path was never
   reached" and still reported FAIL; now reports SKIP, matching its own words (B8's rule).
3. `TestMoERouteFirstLaunchReservation` — a test defect. It required being the FIRST test in the
   process to launch a given kernel (a context-level reservation only exists on first launch) with
   nothing asserting or declaring that precondition; now asserted, with instructions to run alone.
4. `TestAttention_HeadDimWidths` — the one REAL bug, and not a capacity/leak issue: every CUDA call
   in the block discarded its error (`_ = ...`), so a resource failure left the output buffer
   untouched (all zeros), which the assertion reported as `cosine 0.000000` — a dropped-error bug
   wearing a numerics bug's clothes. Errors are now checked and named per call.

All four pass alone AND in the tier after the fixes. **The meta-lesson, stated in the fix commit:
"SHARED PRECONDITION IS NOT SHARED MECHANISM."** `scripts/gpu_gate.sh`'s own header supplied a
mechanism for one symptom (VRAM contention → bogus `cosine 0.000000`) from a real past incident, and
it was inherited as the explanation for these four rather than measured — framing every candidate
below for most of a day before anyone actually grepped for a dropped CUDA error. The "does the tier
fit in 8 GB" question this entry opened with, the "no leak" retraction, and the "Close() leaks on
the 7B path, this selects option (b)" verdict below are all superseded by this: there was no
capacity question to answer and no leak to hunt. **Confirmed empirically, not just by the fix
commit's own claim:** the archived GREEN CUDA gate for `v0.14.0` at `c6760d7`
(`docs/measurements/gpu_gate_cuda_v0.14.0_c6760d7_PASS.log`) shows the full heavy tier passing
clean — `TestB2DenseFlagship` PASS, `TestRealForwardParity` PASS, "heavy tier (real models) — 2325s"
PASS — and the suite now includes `cuda/zz_a12_baseline_test.go`'s `TestZZ_A12ContextBaseline`, the
per-context-reservation bound `e682eb2` added as this investigation's lasting instrument.

**Below is the investigation as it happened, kept for the reasoning** (the "STILL OPEN" candidate
list, the retraction, the later "declining ceiling" re-measurement, and the three-option decision
with "the tag is blocked on this" are ALL superseded by the resolution above, not just the sections
already marked "Superseded" — none of them had seen `e682eb2` yet).

`scripts/gpu_gate.sh` is red, and after A11 and the two leak fixes the remaining failures are **all
VRAM exhaustion presenting as something else**:

| test | in-suite | alone |
|---|---|---|
| `TestAttention_HeadDimWidths` | FAIL — `attention cosine 0.000000 vs CPU ref` | **PASS**, cosine ≈ 1.0 |
| `TestSplitKV_bitIdentical_gemma3` | FAIL | **PASS** — bit-identical at depth 1536 |
| `TestMoERouteFirstLaunchReservation` | FAIL (0.13 s) | **PASS** |
| `TestResidentLaunchVRAMProbe` | FAIL (248 s) | VRAM-sensitive by construction |

**`cosine 0.000000` is the signature the script's own header names**: *"parallel packages contend for
VRAM and the failures come back as bogus numerics ('cosine 0.000000') rather than 'you are out of
memory'."* The other symptom is `panic: interface conversion: decoder.ResidentForward is nil` — a
`BuildResident` that could not allocate. Neither is a numerics defect.

**It is CUMULATIVE, not a single leaker.** Two genuine leaks were found and fixed (`5ece205`:
`TestAllocFloor`, `TestA10ReportingGap` — both drained the device and never released). Fixing them
stopped the tier aborting at 91 s and let it run its full ~26 minutes, which is what exposed these.
Beyond that: `TestMoERouteDemandThreshold` + `TestMoERouteFirstLaunchReservation` pass as a pair; the
pressure builds across the whole tier rather than at one site.

**And there is no systematic missing-`Close`.** Audited: 75 `Load` calls against 175 `defer …Close()`
across the CUDA tests; only three small files load without any (`cuda/gemma_bos_build_test.go`,
`cuda/graphs_safe_test.go`, `cuda/mustalloc_test.go`). Worth tidying, but not the cause.

**So this reads as a capacity limit of the 8 GB card for the full heavy tier in one process, not a
defect in the tree.** Every test passes on its own; the tier as a whole does not fit.

**CANDIDATE LIST AFTER THE RETRACTION — two cheapest CANDIDATES REFUTED BY MEASUREMENT
(2026-08-13).**

**1. Parallelism — REFUTED by reading the invocation.** `scripts/gpu_gate.sh` passes **`-p 1` on
every CUDA invocation** (four of them), targets the **single `./cuda/` package** rather than
`./cuda/...`, and the package contains **zero `t.Parallel()` calls**. Tests are strictly sequential;
no two test binaries touch the card at once. This was the cheapest candidate and it would have
explained every symptom, which is why it was checked first.

**2. Asynchronous teardown — REFUTED by measurement** (`cuda/closesettle_test.go`, 7B + 16-step
decode):

    free before Close        2418.2 MiB
    Close() returned after   123.1 ms
    free at Close's return   7310.2 MiB   +4892.0 MiB released SYNCHRONOUSLY
    free once stable         7310.2 MiB   +0.0 MiB more, asynchronously
    settle time after return 15.7 ms      (polling overhead only)

**`Close()` is fully synchronous — everything is back before it returns, and there is no tail.** The
2.4 → 4.3 → 6.2 GiB climb the tracer saw was the sampler catching Close *during* its own 123 ms
execution, which is consistent with this and not with an async release. So no stabilisation wait is
warranted, and "the next Load starts inside the previous teardown" is not the mechanism.

**STILL OPEN, and the list is genuinely short now:**

- **Per-context kernel reservations.** The one mechanism that survives both refutations, and it is
  already measured in this queue: A9/A10 established that a kernel's **first launch** reserves
  local-memory backing store sized for occupancy, that it is a **context** property rather than a
  model one, and that `Close()` tears down a *model* without destroying the *context*. `moe_route`
  alone was 138,412,032 B. A tier that touches many kernels for the first time would accumulate
  reservations that no model-level `Close` can return — which fits sequential execution, no leak,
  full synchronous release, and order-dependence simultaneously. **Not yet tested.** The test is
  cheap: read free VRAM at process start, run tests exercising disjoint kernel sets with every model
  closed, and see whether the baseline steps down and stays down.
- **Genuine peak** — one test's high-water mark plus whatever is legitimately resident exceeding the
  card.
- **Fragmentation** — refuted earlier against a *different* observation, which does not refute it
  here.

**RETRACTED 2026-08-13 — THERE IS NO LEAK. The "leak" was a probe-position artifact, and the
authoritative experiment says `Close()` returns everything.**

`TestResidentCloseFreesVRAM_7B` (`cuda/lifecycle7b_test.go`) runs the sibling gate's shape at the
scale that supposedly reproduced: **qwen2.5-7B, int4, backend cuda, a real 16-step decode loop, three
Load→Forward→Close cycles in one process**, reading free VRAM after each Close from a held probe
context:

    baseline 474 MiB
    cycle 0: loaded 5366 MiB (+4892), after Close 474 MiB (+0)
    cycle 1: loaded 5366 MiB (+4892), after Close 474 MiB (+0)
    cycle 2: loaded 5366 MiB (+4892), after Close 474 MiB (+0)

**Zero retention on every cycle, including the first** — a FOURTH outcome, outside all three
pre-registered branches. Not per-cycle growth (no leak), and not even the one-time context retention
the A9/A10 mechanism predicted.

**What produced the false −1344 MiB.** The tracer's final sample is taken at an *uncontrolled point
relative to teardown*. Re-running `TestB2DenseFlagship` under the tracer and reading the tail shows
free VRAM still climbing as the process exits:

    free=2,535,653,376   (2.4 GiB)
    free=2,535,653,376
    free=4,475,518,976   (4.3 GiB)   <- deferred Close running
    free=6,509,756,416   (6.2 GiB)   <- still recovering when the process ended

`Close()` was working the whole time; the sampler simply stopped watching before it finished. **This
is the Position class** — the probe sat on one side of the event and reported the other side's state
— which this queue has recorded twice before and which I walked into a third time while building an
instrument specifically to avoid guessing.

**Consequences, stated rather than quietly dropped:**

- **The per-test table's `after` column is not trustworthy for tests with large teardowns.** Every
  such reading may have been taken mid-`Close`. The raw data stays at
  `docs/measurements/a12-vram-per-test.json` with this caveat attached; the *shape* claim (sawtooth,
  declining ceiling) rests on those same readings and is therefore **unproven**, not merely uncertain.
- **The "correctly scoped, silently narrow" criticism of `TestResidentCloseFreesVRAM` does not
  survive.** It was green because it is right. Its scope is still worth printing (below), but it was
  not hiding anything.
- **Option (b) is off the table** — there is no leak to hunt. What blocks the gate is still open, and
  the candidate list reopens as the third pre-registered branch said it would.

**What the new gate is still worth keeping for:** it is the 7B/decode-loop coverage that did not
exist, it prints its scope with its verdict, and it asserts steady state from cycle 2 so a real
per-cycle leak would fail it. It just happens to be green today.

*(Superseded reading below, kept because the reasoning was sound and only the probe was not.)*

**MEASURED 2026-08-13 — the reading is the THIRD branch: NEITHER shape, and the diagnosis is
accumulation LOCALIZED to a few tests.** 186 tests traced, `cuMemGetInfo` at 50 ms, joined to `-v`
boundaries.

**It sawtooths** — 33 tests end with *more* free VRAM than they started, 38 with less — so most of the
tier frees correctly and a blanket "environment limit" is wrong. **But the ceiling declines and never
recovers:**

| fifth of run | ceiling (max free after any test) |
|---|---|
| 1 | 7310 MiB |
| 2 | 5966 |
| 3 | 5828 |
| 4 | 5822 |
| 5 | **3400** |

**Eight tests take a step down that is never regained.** The two largest early ones were re-run
**alone, each in its own process with nothing else on the card**, and both reproduce:

| test | free before | free after | net |
|---|---|---|---|
| `TestB2DenseFlagship` | 7310 MiB | 5966 | **−1344** |
| `TestRealForwardParity` | 7310 MiB | 6144 | **−1166** |

(The late entries in the eight — `TestSplitKV_bitIdentical_gemma3` is the final test — are confounded
by having few successors, so "never regained" is trivially true for them. The two above are not.)

**AND BOTH ALREADY `defer Close()`.** So this is not a missing cleanup in a test: **closing a
CUDA-backed model does not return all of its VRAM** on these paths. That is a product-level finding,
not a test-hygiene one.

**The gate that should catch it passes, and the gap is visible.** `TestResidentCloseFreesVRAM`
(`cuda/lifecycle_test.go`) does three Load+Forward+Close cycles and asserts used ≤ baseline+128 MiB —
a well-built gate, and green. It exercises the **0.5B coder** model with a **single one-token
forward**. `TestB2DenseFlagship` loads **qwen2.5-7B** and drives a real decode loop. So the gate
covers a model an order of magnitude smaller on a path that never populates whatever is being
retained. Not a tautological gate — a **correctly-scoped gate whose scope is narrower than the claim
it is read as supporting**.

**This selects option (b), and it now has a target** rather than being a fishing expedition: find
what `Close()` does not release on the 7B/decode-loop path, and widen
`TestResidentCloseFreesVRAM` to a model and workload that would have caught it. Options (a), (c) and
the partition all become unnecessary if the leak is real and fixable — and the measurement says it is
real.

**Superseded by the above, kept for the reasoning:**

**BEING MEASURED BEFORE CHOOSING** (`cuda/vramtrace_test.go`, `7fa09da`). Free VRAM is sampled
in-process by `cuMemGetInfo` — the A-chain's own instrument — every 50 ms, joined to test boundaries
from `-v` output by wall clock, because Go exposes no per-test hook and the four failing tests share
no helper. **Pre-registered readings:**

| shape | reading |
|---|---|
| monotonic decline, not recovering | **accumulation** — something is not freeing, the `defer` audit missed it, the gate is correctly red. Option (b), with a target |
| sawtooth, recovering per test, high-water above the card | **genuine environment limit.** Option (c) is honest, and statable with a number rather than as a judgement |
| neither | report it — both candidates are wrong and the candidate list reopens |

The per-test table is the deliverable, not a conclusion.

**OPTION (a) HAS A PRICE THAT MUST BE PAID KNOWINGLY.** Per-test process isolation removes the
symptom **by removing the conditions under which a real leak is observable**. A genuine leak looks
exactly like today — green alone, red in suite — so after isolation a leak and an environment limit
**both read green**, on the gate that covers the resident CUDA path. It buys a **permanent blind
spot**, and it may still be right; it must not be chosen as the option that "removes the class",
which is how I first described it.

**A FOURTH OPTION, and probably the best of them: PARTITION.** Split the tier into two or three
groups by memory footprint, each its own process. Fits the card, **preserves cross-test detection
within a group**, and `scripts/gpu_gate.sh` already orchestrates multiple `go test` invocations so it
costs no product code. Weaker than one process, much stronger than N.

**Options, none taken yet — this is a decision, not a fix:**

1. **Run each heavy test in its own process.** Bounds VRAM by construction, since exit reclaims
   everything. Costs process-start time per test, needs the runner reworked, and — the part that
   matters — **buys a permanent blind spot for leaks** (see above). Not "the one that removes the
   class"; the one that removes the *observation*.
2. **Tidy the three no-`Close` files and re-measure.** Cheap, and it may or may not be enough — say
   which before running it, or it becomes a fishing expedition.
3. **Declare the heavy tier out of scope for an 8 GB box** and record the gate as environment-limited,
   with the per-test isolation runs as the evidence. Honest, but it weakens the gate.

**The tag is blocked on this.** Not because numerics are wrong — they are demonstrably right test by
test — but because `scripts/gpu_gate.sh` says *do not tag*, and a gate that is overridden by the person it
was written to constrain is not a gate. B8 applies here too: the gate cannot presently distinguish
"failed" from "could not evaluate", and these four are the second kind.

**B0 · Repo-hygiene group must run what CI runs** — `linux`, **DONE `0c54e35`**

`scripts/ci_checks.py` parses `.github/workflows/ci.yml` and emits the hygiene-class steps; group 5
runs them. **Derived, not duplicated** — a check CI gains appears in the gate with no edit to the
gate. 21 steps across 7 jobs; 13 run here and pass, 8 are darwin-only and are a **counted skip**
naming the platform.

The old block was a strict subset of CI: no staticcheck at all, `vet` without the
`goinfer_testhooks` tag and over narrower packages, no build, no module-boundary guard.

**The environment turned out to be part of the check**, found rather than reasoned. CI's root job has
no `go.work`, so the module-boundary guard sees the root module graph in isolation; this box has a
committed `go.work` that unions every submodule, so the guard reported a **false red** on its first
run, naming `cuda`, `gpu` and `webgpu` as leaks. Derived fix: a job with a `workspace` step runs with
one, a job without runs `GOWORK=off`. **Reproducing the command without reproducing the environment
is not reproducing the check** — worth carrying to any other "run what CI runs" work.

Mutation-checked both directions: a gofmt violation turns root gofmt and staticcheck red; breaking
the derivation's selector makes it **refuse** ("derived ZERO hygiene steps") rather than report an
empty set as a pass. The second is the one that matters — a derivation that degrades to nothing
looks exactly like a clean run.

**B0a · A guard that cannot find its tool must fail, not skip** — `linux`, **AUDITED — no live
instance in the repo. Closed with the residual risk named.**

The shape: a check ran as `command -v staticcheck >/dev/null && staticcheck ... | head`. The binary
exists but is not on `PATH`, so `command -v` failed, the `&&` short-circuited, and the whole check
evaluated to nothing while the surrounding output looked exactly like a clean run. It was reported as
"clean".

**Audit result: the repo does not do this.** Three `command -v` uses, all in `scripts/gpu_gate.sh`,
all `nvidia-smi` **backend detection** rather than tool-guarding a check — and each emits a counted
verdict on the absent path (group 0: `skip "clean-GPU check (no nvidia-smi; ...)"`). Absence there is
a real condition about the machine, not a missing instrument.

**And the class is structurally prevented**, which is the better answer than "we checked". Group 6
reconciles **emitted** verdicts against a **declared** set, so a group that dies or short-circuits
without emitting fails the gate — silence is detectable by construction, not by remembering.

**B0's new group 5 is correct by the same standard**, mutation-checked two ways: PyYAML unavailable →
`scripts/ci_checks.py` exits 2 with a message and group 5 **fails**; the script missing entirely → non-zero
and empty output, group 5 **fails**. Neither degrades to the old hand-written list, which would have
looked like a pass.

**Residual risk, named because it is the one that actually bit:** the instance was an **ad-hoc shell
command typed at a prompt**, not repo code. No gate polices that. The mitigation is the habit the
gate exists to replace — run `scripts/gpu_gate.sh` rather than hand-rolling the check — which is exactly what
B0 makes worth doing.

**B5 · `RELEASING.md` must reference `QUEUE.md`** — **DONE (2026-08-12).**

A file nothing reads is accurate today and inert the first week nobody opens it — the pattern this
queue was written to fix, applied to itself. A tag is the natural moment to review what is
outstanding, and it is a checkpoint that already happens. **Landed:** `RELEASING.md` now has a
"Queue-gated follow-ups — consult QUEUE.md at each tag" section (after the GitHub Release step),
whose **first concrete customer is C3** (Metal consumer window fires on a release carrying an aikit
bump). The abstract "reference QUEUE.md" and its first real trigger landed together, so the reference
is not itself an inert line.

**B3 · Re-tier by cost — SUPERSEDED, folded into B18's T4.** `linux`

`GOINFER_HEAVY_TESTS` gates "needs a real model" and is used as "slow".
`TestSplitKV_bitIdentical` asserts bit-identity at 2048 context in 13 seconds behind two flags,
while 26B streaming runs 5m16s behind the same one.

Rule: **anything asserting a claim the README makes runs by default.** Census is gathered — 26
heavy-gated tests, with `TestSplitKV_bitIdentical`, `TestPrefillDivergenceRate` and
`TestArgmaxTieBreak` all backing published claims. Report the resulting tier membership so the
split is reviewable.

The idea survives as-is inside B18's T4 ("this is B3, promoted") — a measured-wall-clock
requirement added, sequenced against the rest of the release-path restructure rather than done
alone. Kept here, marked, rather than deleted — see B18 for the live version.

**E9 · Autonomous kernel-optimization loop (autoresearch) over the gates** — `mac`, **RUN 2026-08-20/21, results landed on main.** Decided 2026-08-13 by Francis. (Stale until this edit: this entry still read "PLAN DRAFTED; not started" after the loop had already run and landed — found 2026-08-22 while investigating stray `autoresearch/*` branches, which turned out to be superseded exploration trails, not unmerged work.)

An *execution method* for kernel campaigns, not a new campaign: an agent runs **edit → benchmark → keep/revert** unattended (~40 exp/hr), gated by goinfer's existing correctness harness. Filed engineering (it's tooling); its **target** is the performance FA lever (`docs/ollama-chase.md`, "remaining long-context lever") and GEMV micro-tuning. **goinfer is unusually suited** because the bit-identity + parity gates make "fast but wrong" auto-revertable — the contract that costs speed is what makes the autonomous search safe.

**What actually ran, first target GEMV/norm-class micro-tuning (Metal, per §6's sequencing):** 14 `autoresearch/*` scratch branches, one campaign each — `delta-rule`, `gemv-w8a8-coal`, `gemv-w8a8-amax`, `quant-vec`, `layernorm-quant`, `rmsnorm-f32`, `rmsnorm-quant`, `act-quant`, `swiglu-quant`, `qk-norm`, `gpu-delta-rule`, plus two stale-pointer branches with no divergence from main. Per §5(d) survivors were reviewed and landed as clean, individually-authored commits — not by merging the scratch branches — then main kept going further than any single branch: vectorized loads (`fb05e8f`, `bd1aa27`, `3b6e28a`, `e13d71c`, `1f4d510`, `198ab66`, `4090dc4`), then a second SIMD-shuffle-reduction pass on top of several of those (`128d759`, `ec70519`, `bdc921f`, `84049cb`, `befb7e7`), a `precise::` numerical pin where fast-math reassociation was unsafe (`5ed07ce`, `9014fa5`), and one more win outside the original target list (`4966ef5`, attention's V-read unroll). `rmsnorm-quant`/`swiglu-quant`/`qk-norm`/`act-quant`/`gpu-delta-rule` correctly reverted real attempts that didn't hold (drift or no measured win) — written-refutation artifacts per §3's item 3, not gaps. The `autoresearch/*` branches are now stale (fully superseded by main) and can be deleted.

**Synthesis with the relay (load-bearing):** the relay decides *what* to search and defines the gates (it catches premise errors — Lazy Z, KV-quant — a loop cannot); the loop does the *mechanical search* inside that validated frame. Do not point a loop at an unvalidated premise, at Metal expecting wins (5 refutations — re-derives M3), or at the exact path expecting big wins (Campaign A exhausted the bit-identical levers; value is the `--mode fast` lane). **Guards:** adversarial/near-boundary cases in the per-candidate gate (or it reward-hacks the accuracy floor — P2/P3); tiered gate (fast inner check + full parity confirm on survivors) built on **E8's gate-runner**; order-alternated best-of-N (P6a clock-ramp); the loop is **gate-read-only** (editing the gate to pass is the automated vacuous-gate trap); survivors go through normal review, the loop replaces the grind not the judgment.

**Full plan + setup recipe: `docs/task-autoresearch-loop.md`** (the single-number CORRECT|WRONG bench, the git keep/revert harness, the agent instruction, results.tsv audit trail). First target: GEMV (bounded, byte-identical) to prove the harness, then the FA fast-lane.

### F. Audit backlog

**F1 · §4 gates — SWEPT 2026-08-12: all five were already FIXED** — `linux`, **CLOSED**

Every entry here was stale, the same shape as C-08 and five times over. Swept against the tree with a
content-keyed citation added to each, so the next sweep is the lint rather than a person:

| gate | state | anchor |
|---|---|---|
| G-01 `TestResidentAdmission_matrix` tautological | **fixed** — compares against a reviewed golden and errors on any family missing a row | `decoder/features_test.go:215` |
| G-02 Metal snapshot golden applies no embed scale | **fixed** — `Forward`/`ForwardArgmax` apply the arch scale, with a named regression gate | `metal/snapshot_golden_test.go:77` |
| G-03 `buildMatrix` env-pinning | already closed | — |
| G-04 `case "slots"` doesn't assign `residencyBufs` | **fixed** — the switch populates `pinned` and it is assigned after | `metal/model.go:927` |
| G-05 tokenizer/chat tests probe a developer home | **fixed** — `GOINFER_MODELS_DIR`, defaulting to `$HOME/models` | `decoder/modelsdir_test.go:13` |
| G-06 hardcoded developer-home paths | **substantially fixed** — same mechanism; residue is a literal `/home/francis/models` reached only when `os.UserHomeDir()` *fails*, in the four per-package `modelsdir` test helpers | `decoder/modelsdir_test.go:13` |

G-06's residue is the only thing left and it is a last-resort fallback, not a probe path. Recorded
rather than struck, because "substantially fixed" is a different state from "fixed".

**G13 · `/v1/messages` silently restructures the conversation for ANY illegal role — validate and
reject instead** — `mac`, **DONE 2026-08-26.**

**Landed:** `anthropicTurns` rejects any role that is not `user`/`assistant` with a
`400 invalid_request_error` naming the offending role and pointing at the top-level `system` field
(the likeliest mistake). Validating there rather than in the handler covers `/v1/messages`,
`/v1/messages/count_tokens` and the vision path at once — all three already surfaced the `*apiErr`.

**Gates:** eight illegal shapes rejected (`developer`, `system`, `Assistant`, `USER`, `sytem`,
`tool`, `""`, `function`), each asserted to be a 400 that names the role; legal roles still accepted
including a `cache_control`-bearing block array (the shape Claude Code actually sends), so the
validation is not a compatibility break in disguise; and an HTTP-level test proving the 400 reaches
the wire on BOTH routes — a function-level test alone would not have shown that.
Mutation-checked: removing the check fails the rejection test.

`anthropicRole`'s doc comment now records that its "everything else is a user turn" fallback is no
longer load-bearing, so nobody reads it as a license to accept new roles silently. Small. Does not block Tier 0 of
`docs/scoping-dsh-goinfer.md`; the interim state is visible (pinned), which is the point.

**Before-state, already recorded as a passing test** — `TestAnthropicDeveloperRoleStaysUser` in
`internal/serveapp/developerrole_test.go`, written during G12. `anthropicRole` maps everything that
is not `"assistant"` to a user turn, and there is **no role validation anywhere in
`internal/serveapp`**. So a typo'd, invented, or wrong-API role (`"developer"`, `"Assistant"`,
`"sytem"`, anything) does not fail — it is silently folded into the conversation as a user turn,
restructuring what the model sees. `developer` was not a special case; it was the instance that
happened to get caught.

**The change:** on `/v1/messages`, accept only `user` and `assistant` in the `messages` array and
return a clean `invalid_request` 400 for anything else, which is what the upstream Anthropic API
does. G12 framed this surface as demote-vs-alias and that was the wrong menu: upstream's actual
behavior for an illegal role is neither — it is rejection. Rejecting is *more* faithful to the
"works for the apps that matter" bar, not less, because real Anthropic-shape clients only ever send
legal roles, so a 400 costs them nothing and buys everyone else a loud failure instead of a quiet
restructuring. Flip `TestAnthropicDeveloperRoleStaysUser` from a pin to the rejection assertion as
part of the change; keep a case per illegal-role shape.

**Scope note:** this is the Anthropic surface only. The OpenAI surfaces' `default:`-to-user arm is
deliberate and stays (G12's non-goal guard pins it) — OpenAI's own API is lenient there, and
`developer` is now aliased rather than swallowed.

**R1 · The refresh script's history — two corrections to the record** — `linux`, **CLOSED, both
answered from the log**

**Correction 1: "the refresh script had never been usable" is WRONG.** I wrote that in `eea7f29`'s
commit body. The log says otherwise — **nine commits carry its goldens proof**, with counts rising as
fixtures were added:

| date | commit | goldens |
|---|---|---|
| 2026-07-26 | `2e91607` | 14 passed |
| 2026-07-28 | `9624dd9` | 14 passed |
| 2026-08-01 | `ecc5af2` | 119 passed |
| 2026-08-02 | `e58ac8a` | 17 passed |
| 2026-08-02 | `1f6dbe0` | 4 passed |
| 2026-08-03 | `2922468` | 17 passed |
| 2026-08-04 | `7cc2f0d` | 17 passed |
| 2026-08-09 09:58 | `ed81e13` | 19 passed |
| 2026-08-09 15:10 | `ca29d6c` | goldens=19 |

**It worked, was used, and broke on 2026-08-09 at 20:59** — `6edd1ca` introduced `method: null`,
which the writer could not round-trip. `eea7f29` (2026-08-12) is the **first refresh attempted after
that**, three days later, and it aborted. The abort was the guard working on its first real
opportunity, not a guard that had never let anything through.

**Correction 2: the precedent cited in this queue was the wrong commit.** `9e5f8fa` was described
here as "a metadata field addition re-staled `decoder/weights.go` and the refresh ran 19 goldens". It is
`fix(quant): reject --quant that conflicts with a prequant .giw at startup` and **touches the manifest
not at all** — its five files are `CHANGELOG.md`, `decoder/giwquant_test.go`, `decoder/serialize.go` and two `an aikit benchmarks entrypoint`s.
The real precedents are the nine above. I repeated the wrong SHA several times from this file without
opening it.

**Second-order check: a mangled write DID land, and lived ~7 weeks.** The HTML-escaping defect was not
caught by the guard, because it predates the script and arrived through `go test -run ParityManifest
-update`:

    2026-06-20  82b39cc  \u003e appears (1)
    2026-07-26  93eb7d4  (2)
    2026-07-28  99b3f95  (1)
    2026-08-09  6edd1ca  0 — cleaned

So the answer to "did the guard hold from the start" is **no**: escaped sequences were in the tree
from 2026-06-20 to 2026-08-09. They are cosmetic — a `>` inside a `reference` string — and changed no
hash or verdict, but the claim "a clean result means the guard held" would have been false.
`method: ""` never landed (checked; 4 `null`s today, 0 empty strings).

**What is durable:** the writer is now faithful (`SetEscapeHTML(false)`, `Method` as `RawMessage`), so
neither defect can recur through either route.

**R2 · The refresh `arch=` stamp worked on its first real use — `mac`, CLOSED.** Added `2026-08-12`
(`a163150`) because last time the machine that ran a refresh was **unrecoverable from the record** —
the `2e8dfb6` arm64-vs-amd64 question could only be inferred. The `2026-08-13` arm64 gate run is the
first refresh since, and its record answers the question **directly**: proof block and trailer both
read `arch=arm64`. The record now says which arch ran the goldens instead of leaving it to inference.
