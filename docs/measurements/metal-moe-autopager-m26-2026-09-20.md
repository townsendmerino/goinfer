# R11(c) — Metal MoE pager on M26: two near-incidents, not a clean measurement

**Result: the pager mechanism genuinely engages correctly (confirmed `metal-resident`, confirmed
`g4moe.paged`) at both an auto-sized N=64 and a manually-forced N=32, but BOTH runs drove this
machine into a severe swap spiral before the timed decode loop got far — the second time even
after halving the slot count from the first. Both killed manually. No served rate obtained at
either N.** This is recorded with the same care as a clean result, per this repo's own measurement
discipline — a near-miss that isn't written down teaches nothing to whoever reads this next.

Box: `apple-m1pro` (M1 Pro, 6P+2E, 16 GB, already under ordinary desktop/editor memory pressure —
not an idle machine). goinfer `fd1a9cb2`. Model: `gemma4-26b-int4.giw` (Gemma-4-26B-A4B, int4,
16,119,795,238 bytes), freshly pulled from the archive for this run. `num_experts=128`,
`top_k_experts=8` (from the model's own `config.json`, confirmed after the fact — see "What this
almost got wrong" below).

## Why this was attempted at all

`docs/tasks/red-october.md` R11(c): re-run the M35/M26/G20 rows against the auto-sized Metal MoE
pager "with the fallback provably not engaged." The prior incident on this exact machine
(`docs/benchmarks.md` "M35/M26 on the Mac") was the CPU-staged fallback path — a *different*
mechanism from the GPU-resident expert pager (`decoder.Options.MoECacheExperts`), which the
record already showed producing a real, if slow, decode rate (~2 tok/s) rather than hanging. The
open question R11(c) asks is whether the auto-sizer, when actually exercised (not hand-picked via
`GOINFER_METAL_MOE_SLOTS`, which the existing `TestGemma4_26B_pagedRuns` test uses and which
bypasses the auto-sizer entirely — see `metal/backend.go`'s `metalMoESlotsRequest`), is safe on
this hardware.

## What ran

Both runs used an independent external monitor (`vm_stat`/`sysctl vm.swapusage` polled every 2s,
outside the test process itself) specifically so a spiral *before* either test's own first RSS
check would still be visible, and both were killed manually (`pkill -9 -f metal.test`) as soon as
the monitor showed sustained (not single-tick) swap growth — not left to run to Go's own 10-minute
test timeout.

**Run 1 — auto-sized (N=64).** New test, `metal/gemma4_26b_autopaged_test.go`
(`TestGemma4_26B_autoPagedRuns`, heavy-gated): `decoder.Load(giw, decoder.Options{Backend: "metal",
Quant: "int4", MoECacheExperts: true})` — `MoECacheSlots` left at 0, the real auto-size path
(`autoMoESlots` in `metal/backend.go`), not a hand-picked N. Checks `DecodePath()` reports
`metal-resident` (not `-staged`/`cpu`) and `r.g4moe.paged` is true *before* touching anything else;
bounded to 4 timed decode steps; an in-process RSS kill switch at 15,000 MB.

**Run 2 — manually forced (N=32).** The existing, previously-shipped
`TestGemma4_26B_pagedRuns` (`metal/gemma4_26b_paged_test.go`), which sets
`GOINFER_METAL_MOE_SLOTS=32` directly — bypassing the auto-sizer entirely, so this is a genuinely
different, lower, hand-picked commitment (half of run 1's N), not a re-run of the same thing.
Attempted after run 1's writeup, at the user's request, specifically to see whether a smaller
slot count was survivable on this machine right now.

## Run 1 — second by second (from the external monitor)

| time (PDT) | event | swap used | swap total |
|---|---|---|---|
| 19:52:16 | test starts, `decoder.Load` begins | 1,854 MB | 3,072 MB |
| 19:52:20–19:52:44 | RSS climbs to ~6.5 GB (mmap read of the .giw), then fluctuates as the OS reclaims | 1,854 MB (flat) | 3,072 MB |
| 19:52:5x | `DecodePath()` logged: `"metal-resident (int4)"` — the safe path confirmed | — | — |
| 19:52:5x | `buildResident` returns; auto-sizer logged: **N = 64 slots/layer (the `autoMoESlotsMax` ceiling)** | — | — |
| 19:53:06 | first swap growth | 2,285 MB | 3,072 MB |
| 19:53:08 | | 4,030 MB | 4,096 MB (macOS just grew the swap file) |
| 19:53:10 | | 7,749 MB | 8,192 MB (grew again) |
| 19:53:22 | peak observed | 12,063 MB | 12,288 MB |
| ~19:53:1x | **test process killed manually** (`pkill -9 -f metal.test`), before any decode step | | |
| 19:53:27 | first tick after the kill | 4,671 MB | 11,264 MB |
| 19:53:58+ | system fully recovered: 6.4 GB free, load back to baseline within ~1 minute | | |

Total elapsed from process start to kill: **~55 seconds**. Not one of the 4 timed decode steps
ran — the spiral happened entirely during load/build, before `ForwardEmb` was ever called.

## Run 2 (N=32) — second by second

| time (PDT) | event | swap used | swap total |
|---|---|---|---|
| 20:02:54 | test starts, `decoder.Load` begins | 3,146 MB | 4,096 MB |
| 20:02:56–20:03:37 | RSS climbs to ~7.0 GB, then declines as the OS reclaims (same shape as run 1) | 3,074–3,138 MB (flat) | 4,096 MB |
| 20:03:2x | `t.Logf` confirms: **"26B paged build OK (N=32 slots/layer, 30 MoE layers) — RSS 7 MB → 892 MB after build"** — `buildResident` succeeded, further than run 1 got | — | — |
| 20:03:39 | first swap growth | 3,214 MB | 4,096 MB |
| 20:03:41 | | 4,348 MB | 5,120 MB (grew) |
| ~20:03:41 | **killed manually** at this first confirmed (non-single-tick) growth, proactively — before the previous run's much longer wait-and-see | | |
| 20:03:44–20:03:54 | swap kept climbing for ~13s *after* the kill signal (already-in-flight I/O/reclaim), peaking at 7,907 MB / 8,192 MB total | | |
| 20:03:56+ | system recovers: free pages jump from ~4k to 573,922 in one tick, swap total settles at 6,144 MB (not back to the original 3,072–4,096 MB — see note below) | | |

Total elapsed from process start to kill: **~47 seconds**. `buildResident` completed this time
(run 1 never got that far) and confirmed the smaller, real commitment its own log line reports —
892 MB RSS immediately after build, not the multi-GB figure the external monitor was reading
moments earlier (that earlier reading is `decoder.Load`'s own mmap-read transient, already settling
by the time `buildResident` returns — consistent between both runs). The spiral this time started
about 2 seconds after that log line, i.e. during or immediately after the warm/seed step
(`r.ForwardEmb` for the first token) rather than during load/build — a later point of failure than
run 1, but still before any of the 4 *timed* decode steps this test also bounds itself to.

**Both kills left the swap file permanently larger than it started** (run 1: 3,072 → 11,264 →
settled ~5,120–6,144 MB; run 2: 4,096 → 8,192 → settled 6,144 MB). macOS does not appear to shrink
an expanded swap file back down promptly, so each attempt left the machine with less true headroom
for the next one — worth knowing before a third attempt on this same session.

## Why this reads as a real finding, not noise

`DecodePath()` and `r.g4moe.paged` both confirmed the *correct* mechanism engaged in both runs —
this machine did not repeat the CPU-staged-fallback disaster either time. That part of R11(c)'s
premise holds: the paged path is real and distinguishable from the dangerous one, and checking for
it explicitly before decoding is a real, working safeguard (it would have caught the *other*
failure mode; it does nothing for either of these, because both are memory-budget problems
*inside* the safe path, not a wrong-path selection).

Run 1 alone would have pointed at `autoMoESlotsFor`'s budget check (`metal/backend.go`) specifically:
it solves `needFixed + N·perSlot ≤ 0.7·RAM` against a **snapshot of available memory taken at one
instant**, and this machine's free-memory reading is measured (that run) to swing by multiple GB
within single-digit seconds under ordinary desktop load — the external monitor's own free-page
column goes 8,556 → 15,517 → 3,640 pages across three consecutive 2-second ticks near the start of
run 1. A budget check keyed to one such instant can read as far more generous than the real,
sustained margin, and it picked the formula's own maximum ceiling (N=64).

**Run 2 changes the diagnosis.** N=32 — half the slot count, chosen by hand, not through the
auto-sizer at all — hit the same shape of spiral. That rules out "the auto-sizer's snapshot timing
picked too large a number" as the *whole* story: a human-chosen, conservative-by-convention N (this
exact value is `TestGemma4_26B_pagedRuns`'s own existing default, previously used successfully
enough to be the shipped test's default) still wasn't safe *this session*, on *this* machine, under
its *current* ambient load. The more accurate read is that this 26B model's paged-resident
footprint — dense terms + KV + N expert slots + host-side overhead, before a single token
decodes — is simply too large for what this machine currently has free, and the specific N mostly
changes how far into the load/build/first-token sequence the spiral starts (run 1: during
load/build; run 2: during/after the first token), not whether it happens.

## What this almost got wrong

Before writing up run 1, the first hypothesis was that N=64 (the ceiling) *equalled or exceeded*
this model's real expert count and therefore silently fell back to full non-paged residency —
which would have been a different, and differently-actionable, bug (the guard failing to re-check
its own escape hatch). Checked directly against the model's real `config.json`
(`num_experts=128`), not assumed: 64 < 128, so `g.paged = true` genuinely held and this was a
paged run, just a memory-heavy one. Recorded here because it's exactly the kind of plausible
wrong turn that gets written into a conclusion if not checked — and it's also why run 2 (a real N
well below any boundary condition) was worth doing before settling on a diagnosis: it ruled out
"boundary case" as thoroughly as this ruled out "silent full fallback."

## Decision

**Not shipped at either N. Not retried a third time this session.** Both a machine-chosen ceiling
(N=64) and a human-chosen, previously-used-elsewhere conservative value (N=32) produced the same
shape of near-incident on this machine today. Continuing to lower N by hand and re-attempting is
not obviously safe given that pattern, and each attempt has left this machine's swap file
permanently larger (less true headroom) than before it — a third attempt is a real decision, not
a reflex, and isn't taken here without it being asked for again.

## Out of scope, not established here

- A served tok/s number for M26 under any paging configuration — never reached, at N=64 or N=32.
- The floor of what *would* work on this machine right now (N=16? N=8, the `top_k` minimum?) —
  untested. Given N=32 already failed, there is no strong reason to expect a modest further
  reduction succeeds, but it is genuinely unknown, not ruled out.
- Whether either N would be safe on a quieter box (this machine had ordinary desktop/editor load
  throughout both runs, not an idle baseline) — these runs cannot separate "this N is too large in
  general" from "this machine's ambient load makes any N this size too large right now."
- `autoMoESlotsFor`'s formula itself is not shown wrong by run 1 alone, and run 2 shows the
  problem isn't confined to its snapshot-timing behavior at all — the formula is one input to a
  larger question (does this class of model fit on this class of machine under real, non-idle
  conditions), not the standalone defect it looked like after run 1 alone.
- M35 and G20 — not attempted this session.

## A fix this suggests, not built here

Two separable levers, since run 2 shows the auto-sizer's own snapshot-timing behavior is not the
whole story: (1) `autoMoESlotsFor`'s live-memory reading could be sampled more than once, or taken
from a trailing/conservative statistic rather than an instant — this addresses run 1's specific
mechanism, not run 2's; (2) independent of the auto-sizer, a lower recommended/default N for this
model class on a 16 GB Metal box, treated as a real hardware-fit question rather than a formula
tuning question — CUDA's VRAM doesn't compete with the OS and every other running application for
the same pool the way Metal's "available RAM" does, so a budget fraction tuned against CUDA's
behavior may simply not transfer. Neither attempted here; this record exists so whoever picks it
up starts from a real, two-run timeline instead of a guess drawn from one.
