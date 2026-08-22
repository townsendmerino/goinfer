# W4A8 ops-per-byte — answer to docs/prompts/aikit-w4a8-ops-per-byte.md

Box: Ryzen 7 3700X (8c/16t, AVX2, no VNNI), the same one the 0.656 vs 1.674 tok/s gap was
measured on. Kernel under test: `dotW4A8FoldAVX2` (`linalg/dot_w4a8_amd64.s`) at the FFN
gate/up/down shape, K=5120, the largest single per-token contributor in the profile.
Method and code: `linalg/w4a8_opsperbyte_bench_test.go` (`TestW4A8OpsPerByte`,
`TestW4A8IssueWidthProbe`) in the aikit repo.

## The number the task asked for

|regime|throughput|
|---|--:|
|hot (L1-resident, 1 row reused) | 16.62 GMAC/s |
|cold (streaming 55.7 MB, real DRAM reads) | 15.90 GMAC/s (9.94 GB/s) |
|`dotI8AVX2` reference, hot, same K (no nibble-unpack, no per-group scale fold) | 51.20 GMAC/s |

**Achieved/theoretical: `dotW4A8FoldAVX2` reaches ~32% of the no-unpack/no-fold reference's
throughput (3.08× slower per MAC).** Cold is only 1.05× slower than hot at 9.94 GB/s against
this box's ~51 GB/s DDR4-3200 peak — **neither regime is memory-bound**; the kernel is
compute-limited in both, confirming your own bandwidth math (16.8 GMAC/s / 11.7 GB/s
in-the-wild, nowhere near the DDR ceiling).

## Where the remainder goes

The marginal-FMA issue-width probe (aikit's `priors-microgpt-c.md` §1, same technique already
run against `dequantRowInt8` and your `rmsNorm`/`applyRoPE`) on the **cold** kernel call gives
ratio **0.91 — NOT issue-limited**. Idle execution-port capacity exists even while streaming
from DRAM, so raw instruction/uop count competing for ports is not the ceiling.

Leading hypothesis, from the assembly (not perf-counter-confirmed — `perf` wasn't installed on
the box and I didn't want to add packages to a machine mid your 1.0 prep sweep):
`dotW4A8FoldAVX2` accumulates into **one** f32 register (`Y10`, via `VFMADD231PS`, once per
group, 160 times) — a serial dependency chain across the whole K-loop. `dotI8AVX2` uses
**four** independent accumulators, and its own comment states why: "breaks the ... dependency
chain so the four interleaved groups issue independently." A single accumulator caps the loop
at roughly one iteration per FMA-latency count regardless of how many ports are free —
consistent with "not issue-limited" (ports sit idle) while still running well below
`dotI8AVX2`'s throughput (latency-bound on the chain, not port-bound).

## Update, same day: the accumulator fix was tried and measured negative

Built the 4-independent-accumulator variant (`dotW4A8Fold4AVX2`) the recommendation below
originally proposed. Correctness held (1e-5 rel-err vs the scalar oracle, and vs the
production kernel), but throughput did not move: hot 17.12 → 17.36 GMAC/s (+1.4%), cold
16.31 → 16.15 GMAC/s (−1.0%) — both inside noise, `nvidia-rtx2070s`, K=5120. **The
dependency-chain hypothesis was wrong**, or at least not the dominant factor.

Why the issue-width probe pointed the wrong way: "not issue-limited" from marginal-FMA
injection means idle capacity on the ports *FMA instructions* use. It does not rule out
contention on a *different* port — and the nibble-unpack prologue (8 shuffle/logic
instructions per group: `VPAND`/`VPSRLW`/`VPUNPCKLBW`/`VPUNPCKHBW`/`VPSUBB`×2, feeding the 3
MAC+fold instructions) is exactly the kind of work that would saturate a shuffle port while
leaving FMA ports idle. The injected dead FMAs get absorbed for free because they compete for
a different resource than whatever is actually full — the same reading a genuinely
memory-bound kernel would give. The probe distinguishes busy-vs-waiting only for the port
class it injects into; it does not localize which resource, if any, is saturated.

Recorded as a measured dead end, not a re-triable one: aikit's
`docs/internal/perf-dead-ends.md` §8.9.

## Second update, same day: the unpack cost isolated and quantified — it's half the story

Built the "pre-unpacked-weights, still-scaled" micro-benchmark the first update called for
(diagnostic-only — weights as one int8/weight instead of packed nibbles, never shippable,
doubles the weight footprint). Held everything else fixed (single accumulator, per-group
scale-fold) and isolated unpack directly:

|kernel|ns/call|GMAC/s|
|---|--:|--:|
|`dotW4A8FoldAVX2` (production: unpack + per-group scale) | 295 | 17.36 |
|unpack-free (diagnostic: no unpack, still per-group scale) | 188 | 27.23 |
|`dotI8AVX2` (reference: no unpack, no per-group scale) | 106 | 48.30 |

Removing unpack alone recovers **1.57×**. The remaining **1.77×** to the reference is the
per-group scale-fold itself (`VCVTDQ2PS`+`VBROADCASTSS`+`VFMADD231PS`, one f32 convert +
broadcast + FMA per 32-weight group) — real, not noise, and `dotI8AVX2` doesn't pay it (one
overall scale at the end, not one per 32 weights). **Roughly a 57/43 split of the total
overhead, no third factor** — unpack + scale-fold fully account for the gap to the reference.
This confirms the first update's suspicion and adds the number it was missing: unpack is the
larger piece, but not the whole story, and a fix that only addresses one of the two leaves
real throughput on the table.

## Third update, same day: the VPMADDUBSW saturation math worked out — safe, but the win shrinks once priced correctly

Before writing any assembly: is unsigned-nibble `VPMADDUBSW` actually safe here, given
`dotI8AVX2`'s own comment defers it for full int8×int8 ("u8×i8 pair sums can exceed int16 and
it SATURATES")? **Yes, provably.** That concern is about the general u8 range — worst case
`2×255×128=65,280`, over int16's ±32,767 ceiling. A nibble caps the unsigned operand at 15, not
255: worst case `2×15×128=3,840`, over 8× inside the ceiling regardless of activation values.
No saturation risk, confirmed before implementation, not assumed from "the range looks
narrower."

But raw (uncentered) nibbles compute `Σnib·act`, not the true `Σ(nib-8)·act` — a per-group
correction `8·Σact` has to go somewhere. Priced optimally (precomputing `Σact` per group once
per token, the same "quantize once, reuse across all N rows" shape goinfer's own
`QuantizeActivationsInto` already uses, since the activation row is shared across every weight
row in one M=1 matmul — not recomputed per row), the realistic instruction count is **18/group
vs the current 20** — ~10%, not the ~50% "skip 4 ops" naively implied — and even that needs a
real calling-convention change to `MatmulBTW4A8Into` (a precomputed per-group activation-sum
array), not a drop-in kernel swap like the two prior experiments.

**Decision: stopped here, not built.** Given the accumulator experiment already measured a
plausible-sounding instruction-count argument land at ~0% real speedup, a ~10% reduction that
needs an API change wasn't judged worth building blind. Math and the revised estimate are
recorded so this isn't re-derived from scratch if revisited with new evidence (e.g. hardware
counters, or a reason to believe 10% clears the noise floor here specifically).

## Recommendation (final for this round)

**Still: do not start the Q4_K-style format work** — nothing across three rounds of
measurement weakens that. The kernel has two real, separately-quantified inefficiencies
(unpack ~57%, per-group scale-fold ~43% of the overhead to `dotI8AVX2`), one closed negative
lead (the accumulator dependency chain, §8.9), and one worked-out-but-not-built lever (the
VPMADDUBSW unpack rewrite — safe, but ~10% for real engineering cost). The remaining
promising lever is **VNNI** (`VPDPBUSD`) — would remove the unpack-then-widen sequence AND the
scale-fold's separate convert step in one instruction class, not just one of the two — but it
is hardware-gated: no VNNI-capable box available to validate it. This is a stopping point, not
a closed investigation: revisit if a VNNI box becomes available, or if there's new reason to
think the 10% unpack-rewrite estimate is worth the API-change cost after all.
