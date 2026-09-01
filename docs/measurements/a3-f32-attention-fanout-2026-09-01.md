# A3 — f32 prefill attention fans out over heads (2026-09-01)

**Result: 1.58× at K=2048 and 1.92× at K=4096 end-to-end prefill, bit-identical output.**
The item was costed at ~13%.

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| goinfer | `2e3d018` + the change under test (committed as the child of it) |
| toolchain | go1.27.0 darwin/arm64 |
| model | qwen2.5-coder-0.5b-instruct **q4_k_m**, from `~/models` on the internal SSD |
| kernel arm geometry | Mellum2 — nH=32, nKV=4, hd=128 (the model the 8k f32 runs used) |
| method | paired + interleaved, alternating which arm leads; medians of n=3 |
| thermal | box quiesced first; see "The first run was contended" below |

No peer is involved anywhere on this page. Both arms are goinfer, so
`bench_compare.sh`'s goinfer-vs-goinfer scope is the right one and no
`bench_peer.py` number appears here.

## What the item claimed, and what was actually true

The queue and the Atlas carried this as *"the f32 branch is single-threaded by
construction — its per-kv-group gather is shared mutable state a concurrent
split would race on"*, priced at **~13% end-to-end**. The sentence came from
G24's harness comment and is true **about the head loop**. It was then read as
if it were true about the work.

It is not. The f32 arm's two matmuls are `linalg.MatmulBT`, which fans out
internally over its N output columns (`parallelCols`) above `parThreshold =
1<<24` MACs. At the K=8192 tile shape each call is ~268M MACs — 16× over that
line — and nothing in goinfer overrides the threshold. So the f32 path was
already using several cores, at the **column** level rather than the head level.

Measured, rather than argued (`TestA3FanoutUtilization`, utilization = CPU
time ÷ wall time — the discriminator an Amdahl estimate cannot talk its way
past):

| arm | wall | utilization |
|---|---|---|
| acc64 (head-parallel, 6 workers) | 16 930 ms | **4.75×** |
| f32 as shipped (column-parallel only) | 6 299 ms | **1.67×** |
| f32 with linalg forced serial | 9 500 ms | 1.00× |

So the premise was neither right (1.67 ≠ 1.0) nor harmless. **The pre-registered
band called this AMBIGUOUS** — the reading written before the first run was
"~1× ⇒ premise holds, >2× ⇒ premise false, between ⇒ parked pending a profile"
— and 1.67× landed in the middle. The park was resolved the way the band said
to resolve it: by finding where the missing parallelism actually was. Solving
Amdahl on the two f32 rows puts ~58% of that arm in serial code — the gather,
the per-row softmax and the scatter, none of which are inside a matmul, and
none of which column-level fan-out can reach.

**How the ~13% was derived, since the derivation is the reusable lesson.** G24
measured its two arms with `MatmulBT` forced serial — deliberately and
correctly, to get an arithmetic-plus-gather ratio of 8.37×. The error was one
step downstream: that **serial-vs-serial** ratio was fed into an Amdahl estimate
against a profile share measured on the **parallel** production path, and the
gap booked as recoverable headroom. Two different parallelism states either side
of one division sign. It is the same shape as dividing a kernel throughput by an
end-to-end number, which CLAUDE.md already warns about for `bench_compare.sh`,
and the same shape as the 4-layer slice that read 3.11× and measured 1.52× at
full depth.

## The change

Fan out the f32 path over **query heads**, exactly as the acc64 path already
does. Each worker gathers K/V into **its own** `kh`/`vt` — which cost no new
memory, because `prefillAttnWorkers` had been budgeting `2*nKeys*hd` per slot
for those buffers all along, on every prefill, including the acc64 path that
never touches them. Heads are assigned in contiguous runs so a worker walks
whole kv groups and re-gathers only when `kvh` changes: total gathers ≤
`nKV + workers` rather than `nH`.

The two parallelism levels must not nest — 6 head workers each spawning
GOMAXPROCS column goroutines is oversubscription — so the head-parallel arm
takes its matmul through a per-slot `linalg.Workspace` with the threshold pinned
above any reachable MAC count, making it serial. The serial arm (1 slot, below
the fan-out floor) keeps the package-level column-parallel `MatmulBT`, so the
pre-A3 shape is preserved exactly where it was already the right one.

## Kernel result

Mellum2 geometry, nKeys=8192, K=2048, tile=256, 6 workers, GOMAXPROCS=8:

| arm | wall | utilization |
|---|---|---|
| f32 column-parallel (pre-A3) | 6 299 ms | 1.67× |
| **f32 head-parallel (A3)** | **1 927 ms** | **5.27×** |
| acc64 | 16 930 ms | 4.75× |

**3.27× and 3.30×** on two interleaved repeats. Utilization 1.67× → 5.27×,
which is the mechanism stated as a number rather than as a story.

## End-to-end result — the citable one

A kernel ratio is what mis-priced this item in the first place, so the shipping
claim comes from the whole forward. `TestA3FanoutEndToEnd`, paired and
interleaved through the real caller (`forwardLayersN`), pre-A3 reproduced with
`GOINFER_PREFILL_ATTN_WORKERS=1` — an A/B handle that already existed, so no
measurement-only knob was added to the production path:

| K | pre-A3 | A3 | speedup | pairs |
|---|---|---|---|---|
| 1024 | 7.08 s | 5.04 s | **1.40×** | 1.32 1.33 1.59 |
| 2048 | 18.86 s | 11.96 s | **1.58×** | 1.57 1.58 1.63 |
| 4096 | 58.79 s | 30.64 s | **1.92×** | 1.85 1.87 1.92 |

The win grows with K, as attention's share of prefill does. **K=8192 is NOT
measured here** and is deliberately not extrapolated to — that extrapolation is
the exact move this page exists to correct.

**K=1024 is the soft row.** Its three pairs are 1.32 / 1.33 / 1.59, so the
median sits on an outlier and moves between runs (1.31× on the contended run,
1.40× here). Read it as ~1.3–1.4× and do not quote three significant figures
off n=3.

## Bit-identity

The output is byte-identical, not close:

- `TestAttendF32Fanout_bitIdentical` — serial arm (1 slot, column-parallel
  matmul) vs fan-out arm (6 slots, serial matmul per worker), asserted `!=` on
  every element, plus a non-vacuity check that the buffers are not both zeros.
  **Mutation-proven**: seeding a worker's `lastKVH` to 0 so it skips its first
  gather turns it red at `ctx[0]`.
- It gates a second thing for free: aikit's contract that `MatmulBT` is
  numerically inert to fan-out width. If that ever breaks, it breaks here rather
  than silently inside a generated token.
- `TestLongPromptFast_forwardParity` — the committed long-prompt golden, which
  pins real token ids through the f32 prefill path — passes **unchanged**.
- `TestA3FanoutEndToEnd` asserts the two arms' logits match bitwise at every
  depth before timing anything, so the timing is like-for-like by construction.

Because it is bit-identical, the `parity_manifest.json` staleness the core edit
triggers is a genuine false positive of the whole-file content hash, and
`scripts/refresh_parity_hashes.sh` is the sound way to clear it.

## The first run was contended, and the ratios survived it

The first pass of both measurements ran 08:26–08:41 while an aikit job was on
the same box (caught by the operator, not by the harness; load averages were
still decaying at 1.61 / 4.56 / 7.59 twenty minutes later). Re-run on the quiet
box:

| K | contended | quiet | absolute pre-A3 |
|---|---|---|---|
| 1024 | 1.31× | 1.40× | 6.65 s → 7.08 s |
| 2048 | 1.60× | 1.58× | 18.82 s → 18.86 s |
| 4096 | 1.97× | 1.92× | 66.16 s → 58.79 s |

The absolute K=4096 time moved 12%, so the contention was real and measurable.
The **ratios** moved ≤3% at K=2048/4096, which is what paired, interleaved,
alternating-lead measurement is for: both arms ate the same interference within
each pair. This is recorded as corroboration of the method, **not** as licence
to measure on a busy box — K=1024, the row with the least work per pair, is also
the row that disagrees most between the two runs.

**And the re-run nearly did not happen.** The first attempt to reproduce on the
quiet box returned `ok ... (cached)` — Go replayed the contended result, with
identical wall-clock start and end timestamps and the *old* run's internal clock
in its own output. `-count=1` is not optional for a timing test. This repo has
already recorded that exact trap once (rev 7: "four runs that agreed to the
centisecond were go test cache replays"); this is the second time, which is why
it is written down here as well.

## What this does not claim

- **Not measured at K=8192**, the depth the f32 flag's own headline uses.
- **Not measured on a MoE**, and MoE prefill is where attention's profile share
  is highest — so if anything this understates the win there, but that is a
  prediction, not a result.
- **CPU only.** CUDA and Metal attention are separate implementations and are
  untouched.
- The 0.5B end-to-end model is smaller than the Mellum2 geometry the kernel arm
  used. The two tables are consistent in direction and are not two measurements
  of one quantity.
