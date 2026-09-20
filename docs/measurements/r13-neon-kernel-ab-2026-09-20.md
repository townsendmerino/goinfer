# R13 Build (aikit kernel A/B) — G=6 NEON grouped kernels beat per-head calls 1.5-2.4×

`docs/tasks/red-october.md` R13's Build, first step: "aikit's kernel A/B bench first (per-head as
the do-nothing arm, A's two-head shape, B's full group) — it nominates." The pure-Go grouped
kernels (`MatmulQKAcc64Group`/`MatmulAVAcc64Group`, aikit) measured 6-10× SLOWER than the shipped
per-head kernels — the register-budget analysis in the Build text ("on NEON's 32 registers, QK at
8 keys × G=6 is 24 accumulators + 4 widened K + 2 q = 30") is explicitly about NEON vector
registers, not generic Go, so that negative result was expected and the real performance question
could only be answered once the assembly existed. This record is that answer.

**Bottom line: the G=6 NEON port is a real win, 1.53-2.39× faster than G separate per-head calls,
growing with depth — the opposite of the pure-Go result, and consistent with the brief's own
register-budget argument that this was always a NEON-assembly story.**

## What was built

`linalg/attn_acc64_group_arm64.s` (aikit, commit `af926e3`): hand-written WORD-encoded ARM64 NEON
for `avAcc64NEON8G6` (one 8-dim V block, all G=6 queries) and `qkAcc64NEON8G6` (one 8-key block,
all G=6 queries), following the register layout the Build text specifies: 24 f64 lane-pair
accumulators (one per (query, dim-pair) or (query, key-pair)), K/V widened once per key/dim and
shared across all 6 queries' FMLA chains (the entire point — G separate calls reload/rewiden K/V
per query), q/scores re-derived cheaply per query since they're tiny and stay cache-resident. Wired
into `MatmulAVAcc64Group`/`MatmulQKAcc64Group` as a G=6 fast path with a Go fallback for every other
G, mirroring `attn_acc64_arm64.go`'s existing `avAcc64Blocks`/`qkAcc64Keys` dispatch convention.

**A real bug, found and fixed before trusting any number.** The QK kernel's store-phase narrowing
initially reused V16/V17 as scratch (copied from the per-head kernel's own convention, where V16/17
are pure scratch). In the grouped kernel they are themselves live accumulators — query g=4's
key-pairs 0 and 1, since accumulator index = g×4+kp = 16, 17 — so query g=3's store step (runs
first in program order) silently clobbered g=4's not-yet-read accumulators, producing wrong output
only for g=4's first four keys and nothing else. Root-caused by progressive isolated-probe
bisection (isolated single FMLA at Vd=16: correct; VEOR-zero at V16: correct; a faithful
standalone copy of the real kp=0 instruction sequence: correct; kp=0+kp=1 together: correct — the
bug only reproduced inside the real kernel's full store phase, which none of the probes exercised
until the last one), not by inspection alone.

## Gates

Gate 1 (528+ cases before this work, now extended): raw-bit equality of the grouped kernels against
G separate per-head calls, G ∈ {1,2,4,6,7,8}, nKeys/hd sweep, nonzero `bOff`/`headOff` — green,
including new NEON-vs-Go direct tests (`TestAVAcc64NEON8G6_matchesGo`,
`TestQKAcc64NEON8G6_matchesGo`) with mutation-detection guards (dropping the last key/dim changes
the output — the oracle is not vacuous) and end-to-end tests through the public
`MatmulAVAcc64Group`/`MatmulQKAcc64Group` dispatch with the NEON path actually firing (G=6). Full
`linalg` suite green (`go test ./linalg/...`, 15.9s), `gofmt -l` clean, `staticcheck` clean (one
pre-existing, untouched-by-this-work U1000 in `checks_off.go`).

## Data

`BenchmarkMatmulQKAVGroupAB`, G=6/hd=128/nKV=2 (aikit's own registered shape), `-benchtime=200x
-count=3`, Apple M1 Pro. Median of 3:

| depth | ungrouped (G separate calls) | grouped (NEON, G=6) | speedup |
|---|---|---|---|
| 130  | 32,268 ns | 19,786 ns | **1.63×** |
| 2048 | 306,408 ns | 199,775 ns | **1.53×** |
| 8192 | 2,259,904 ns | 944,421 ns | **2.39×** |

(Full 3-repeat spread: depth130 ungrouped 30.6-37.6μs / grouped 18.6-20.3μs; depth2048 ungrouped
306.2-313.1μs / grouped 199.6-200.5μs; depth8192 ungrouped 2206-2380μs / grouped 930-945μs — tight
within each arm, no overlap between arms at any depth.)

## Reading

Speedup is not flat with depth: 1.63× at the shallowest depth, dips slightly to 1.53× at 2048, then
climbs to 2.39× at 8192. This is a DIFFERENT shape than R13 step 0(iii)'s distinct-bytes probe
(`r13-distinct-bytes-probe-2026-09-19.md`), which found the group's shared-read win only appears
past a cache-capacity crossover around K∈(512,2048] and grows steadily past it — here depth130 (well
below that crossover) already shows a strong win, and depth2048 is a local minimum rather than a
step up. This kernel's win is NOT purely the distinct-bytes-probe's cache-sharing mechanism, since
that mechanism alone doesn't explain a win at K=130; the register-level argument (fewer loads AND
fewer total instructions per shared byte, not just fewer distinct bytes touched) applies at every
depth, and the cache-crossover effect likely reinforces it on top at depth8192 — consistent with
this being two effects layered, not one. This has not been decomposed further (would need a
cache-miss counter, same instrument gap noted in the distinct-bytes record).

## What is and isn't established

**Established:** the G=6 NEON kernels are bit-exact against the Go oracle and the shipped per-head
kernels (gate 1), and measurably faster in isolation at every depth tested — the aikit-level
nomination this step exists to make. **Not established:** whether this survives inside the real
`attendBatchedHeads` tiled/pooled decode path (this bench isolates the kernel calls only, same
caveat the distinct-bytes probe carried); the served, end-to-end effect once softmax's ~40-46% share
(step 0(ii)) and the category-band ceiling (~1.39-1.54× projected overall, not 1.5-2.4× on QK+AV
alone) are applied; AVX2/amd64 parity (no x86 port exists yet, so this Build currently only clears
the Mac's arm64 half of R13's two-box requirement).

## Next step

Per the Build text's own sequencing: `BenchmarkDecodeAtDepth` (three arms interleaved, both boxes,
all four models) next, then the served `bench_peer.py` paired-interleaved measurement against the
step-0 peer row, which is the one that actually elects. Goinfer-side wiring (Arm A/B,
`GOINFER_ATTN_GROUPED`, the four goinfer gates including `go test -race` and the call-counter wiring
proof) has not been started — this record covers only the aikit kernel nomination step. aikit's
`af926e3` is committed locally, not yet pushed or version-bumped in goinfer's `go.mod`.
