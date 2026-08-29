# Expert-major MoE prefill batching is NOT a compute lever — and the attention exclusion it competes with was never measured

**2026-08-28, M1 Pro (8 CPU, 16 GB), Mellum2 4-layer slice, int8int8, `forwardLayersN` only.**
Pre-registration: `mellum2-moe-prefill-split-PREREGISTERED.md`, written before the arms ran.
Harness: `decoder/mellum2_prefill_profile_test.go`. Raw run logs and pprof profiles are
archived outside the repo, in a per-machine log directory under the operator's home — described
rather than cited, because a concrete path there resolves for nobody else.

## Verdict

| pre-registered rule | measured | outcome |
|---|---|---|
| bound ≥ 20% at K ≥ 4096 → fund on compute | — | no |
| 10–20% → ambiguous, parked | — | no |
| **< 10% → not a compute lever** | **≤ 1.4%, not resolvable** | **this one** |

**Lever 4 (`task-moe-streaming.md`) is parked as a compute lever.** Its case now has to be
made on streaming I/O — where the same expert is re-fetched once per row — and measured there.
This result does **not** retire it; it relocates the argument.

The second, independent pre-registration (the *direction* of the bound across K, which could
have overturned the first) confirms rather than contradicts: the bound FALLS with K.

## The two arms

`uniform` (one repeated id) is both the control and the **ceiling**: a chunk whose rows all
select the same experts is exactly what expert-major batching manufactures, so the lever
cannot beat it. `varied` is real routing. Attention cost is content-independent, so it is a
matched constant across the pair and the difference isolates the routing term.

| K | uniform (ceiling) | varied (real) | pass 1 | pass 2 |
|---|---|---|---|---|
| 1024 | 10.3 s | 10.8 s | +3.77% | +5.41% |
| 2048 | 26.4 s | 27.8 s | +4.40% | +5.67% |
| 4096 | 79.7 s | 80.8 s | +2.73% | +0.12% |
| 8192 | 307.3 s | 311.0 s | +2.66% | **−0.29%** |

**Read the passes, not the means.** At K=1–2k the two passes agree and the routing term is a
real ~4–5%. At K ≥ 4096 they disagree by 20× and, at 8192, **in sign** — the term has fallen
below the run-to-run spread (±0.4–3.5%). The honest statement is not "1.2% at K=8192" but
*"not resolvable"*: at agentic prompt lengths the effect Lever 4 attacks cannot be detected at
all, let alone recovered.

## Why: the two levers scale in opposite directions

Share of prefill work, real-routing arm (park samples excluded — `pthread_cond` is the known
idle-M artifact a CPU profiler miscounts):

| K | attention | ALL weight matmul (upper bound on the MoE FFN) |
|---|---|---|
| 1024 | 77.3% | 22.7% |
| 2048 | 88.8% | 11.2% |
| 4096 | 93.9% | 6.1% |
| 8192 | **97.1%** | **2.9%** |

The weight-matmul column includes the q/k/v/o projections, so the MoE FFN alone is strictly
less. This also strikes the `~17%` that stood at `decoder/forwardn.go`: a K≈1k-era figure,
quoted as constant, naming the expert matmuls for a bucket that holds the projections too.

## The finding that outranks it: the A3/G24 MoE exclusion was never measured

`--cpu-fast-attention` (G24, 2.28× end-to-end at K=8192 on dense) **refuses MoE**, so the 97%
term has no shipped lever on this path. The refusal rests on a stated mechanism — an f32 QKᵀ
reassociation flips a top-k expert at a near-tie and cascades — that is real in kind and had
**never been measured on a MoE**:

- no MoE model appears anywhere in `attention-a3-kernel-ratio-2026-08-26.md`;
- both G24 tests load the **dense** bench checkpoint;
- `TestA3FastAttentionDivergence`'s doc comment says it pins *"MoE excluded: the flag cannot
  turn f32 attention on for a MoE arch at all"* — `MoE` occurs **once** in that file, in the
  comment, and `arch.MoE` occurs **zero** times. It asserts nothing about MoE.

Measured now (`decoder/a3_moe_exclusion_test.go`, `-tags goinfer_testhooks`; the seam is an
untyped `const false` in shipping builds, so the compiler proves the production guard is
unchanged). **Varied ids deliberately** — a constant-id prompt collapses the top-k and cannot
produce the near-tie flip the guard exists to prevent:

| K | cosine (acc64 vs f32) | maxAbs | acc64 | f32 | speedup |
|---|---|---|---|---|---|
| 2048 | **0.999788** | 1.149 | 27.6 s | 17.9 s | 1.54× |
| 8192 | **0.999787** | 1.358 | 319.2 s | 102.8 s | **3.11×** |

Dense ships behind this same flag at **0.9976**. The MoE divergence is an **order of magnitude
smaller** — and it is *larger* on dense, the case the flag already permits.

The speedup is **bigger than dense's 2.28×**, and for the reason the profile predicts: attention
is 97.1% of MoE prefill at K=8192 against ~70% on dense, so the same kernel swap reaches more of
the total.

**Cosine is flat across depth** — 0.999788 at K=2048, 0.999787 at K=8192 — which is the direct
counter to the cascade argument as it applies to prompt LENGTH.

### The mechanism, measured — and it is REAL

A cosine cannot tell a routing flip from ordinary numeric drift, and the guard is a claim about
routing specifically. `decoder/a3_moe_routeflip_test.go` decomposes it with three arms on one
prompt, using the existing `moeSelOverride` replay seam:

| arm | attention | routing | cosine vs A |
|---|---|---|---|
| A | acc64 | natural (recorded) | — |
| B | f32 | natural | 0.999788085 |
| C | f32 | **A's, replayed** | 0.999931014 |

**211 of 8192 moeMLP calls (2.576%) flip their top-k set, and removing the routing term recovers
67.4% of the divergence.** The mechanism `forwardn.go` names is not hypothetical: flips happen,
and they are the DOMINANT contributor to what divergence there is.

**This corrects the earlier reading of the cosine alone.** "The exclusion looks over-conservative"
was the right conclusion from the wrong evidence. The correct statement is narrower and more
useful:

- the guard is **right in kind** — routing flips are real, and they dominate;
- the guard is **not supported in degree** — with the flips included, MoE still diverges an order
  of magnitude LESS than the dense case the flag already permits (0.999788 vs 0.9976). The
  kernel comment's own bar for shipping f32 attention is cosine ≥ 0.99.

So "NOT negotiable at any flag setting" is stronger than the evidence carries. What the evidence
supports is that MoE needs its own bar and its own measurement, not that it needs a categorical
refusal.

The flip-impact figures are a BOUND, not a margin: the smallest kept top-k weight is p50 0.0768,
p90 0.0960, max 0.1147 (against 0.125 for a uniform top-8). Those are not vanishing weights — but
the realized effect is ~1.4e-4, far below what a 0.077 weight would permit, which is itself
evidence that the swapped experts produce similar outputs. That is what a near-tie predicts.

### The depth run — and it OVERTURNS the magnitude claim above

Same test, same K, same prompt, on the FULL 28-layer Mellum2 (nobara-pc, 62 GB, idle;
2026-08-28 21:12-22:03 PDT, 3023 s):

| | 4-layer slice | 28-layer full |
|---|---|---|
| moeMLP calls | 8,192 | 57,344 |
| top-k flips | 211 (2.576%) | **8,303 (14.479%)** |
| cos(A,B) total | 0.999788085 | **0.997874393** |
| cos(A,C) routing removed | 0.999931014 | 0.999364978 |
| routing explains | 67.4% | 70.1% |

**Divergence grew exactly 10.0x on 7x the layers** (1-cos: 2.12e-4 -> 2.13e-3). The flip rate
rose 5.6x, roughly proportional to depth, and the routing term's SHARE held flat at ~70%. So the
mechanism is depth-invariant in character and compounds in magnitude — which is precisely what
the guard's word "cascades" asserted.

**The "order of magnitude safer than dense" reading is retracted.** It was an extrapolation from
4 layers and it was wrong by 10x. At full depth:

| | 1 - cosine |
|---|---|
| dense, which the flag ALREADY ships | 2.400e-3 |
| 28-layer MoE | 2.126e-3 |

MoE is **0.89x** dense — parity, not an order of magnitude. Both clear the kernel comment's own
>= 0.99 bar with room to spare.

**What survives, and what does not.**

- SURVIVES: the exclusion is not supported as a CATEGORICAL refusal. At full depth MoE diverges
  slightly LESS than the dense case the flag permits, so "NOT negotiable at any flag setting" is
  still stronger than the evidence carries.
- DOES NOT SURVIVE: any claim that MoE is comfortably safer. The margin is parity. A decision to
  extend the flag to MoE is now a close call to be argued on the >= 0.99 bar, not a formality.
- **DO NOT EXTRAPOLATE TO DEEPER MoEs.** Divergence tracked depth almost linearly here, and
  Qwen3.5-35B-A3B has 40 layers. The obvious multiplication is exactly the move that just failed:
  this section exists because a 4-layer extrapolation was off by 10x. A deeper family needs its
  own run, not arithmetic.

**Method note.** The 4-layer slice was correctly labelled as unable to test depth, and that label
was load-bearing: every number in the section above it is right, and the conclusion drawn from
them was wrong. A caveat that names the axis a measurement cannot reach is worth more than the
measurement.

### EXTENDED 2026-08-29 — the two things that closed it

**Token level: the divergence never reaches the output.** Full 28-layer Mellum2, K=2048, 48 greedy
tokens, arms differing only in prefill (`decoder/a3_moe_tokenlevel_test.go`):

    continuations are IDENTICAL (48/48 tokens agree)

That was the last real objection. Cosine on hidden states is not what a user sees, and MoE's error
is discontinuous where dense's is smooth, so equal cosines did not have to mean equal behaviour.
They did.

**A properly MATCHED divergence pair, on the third attempt.** The earlier comparisons were each
matched on one axis and disagreed about the sign:

| pairing | dense | MoE | ratio |
|---|---|---|---|
| K-matched only | 0.5B, 24 layers, K=2048: 1.631e-3 | 28L, K=2048: 2.126e-3 | 1.30x WORSE |
| depth-matched only | 1.5B, 28 layers, K=8192: 2.400e-3 | 28L, K=2048: 2.126e-3 | 0.89x better |
| **both matched** | **1.5B, 28 layers, K=2048: 2.352e-3** | **28L, K=2048: 2.126e-3** | **0.90x better** |

The habit worth naming: each time, the pairing that happened to be available got called "matched".
Only the third is. It confirms the original direction — but it was not entitled to be believed
until it existed.

**The exclusion is dropped.** `useAcc64 := !fastAttn`; the probe seam is deleted as dead code, and
`a3_divergence_test.go`'s comment — which claimed to pin "MoE excluded" while asserting nothing of
the sort — is corrected to describe what its body actually checks.

**What the speedup actually is, stated as a range because it varies by an order of magnitude:**

| machine | model | quant | K | f32 speedup |
|---|---|---|---|---|
| M1 Pro | 4-layer slice | int8int8 | 1024 | 1.35x |
| M1 Pro | 4-layer slice | int8int8 | 2048 | 1.54x |
| M1 Pro | 4-layer slice | int8int8 | 8192 | 3.11x |
| Ryzen 3700X | full 28-layer | int8int8 | 2048 | 1.08x |
| **Ryzen 3700X** | **full 28-layer** | **int8int8** | **8192** | **1.52x** |
| **M1 Pro** | **full 28-layer** | **int4** | **8192** | **1.59x** |

**THE ANSWER IS ~1.5x, NOT 3.11x** — see the full-model section below. Never a slowdown. The flag
remains OFF by default, so extending removes a refusal rather than turning anything on.

### The caveat that decides whether this is actionable

**A 4-layer slice was the one thing that could not test the cascade claim — and the depth run
above has now answered it.** Retained as the reasoning that led to running it, because it was
right: the flip rate was the input a cascade needed, and over 7x the depth it compounded 10x. The mechanism the guard
names is compounding: a flipped expert perturbs the hidden state, which perturbs the next layer's
routing. That is a phenomenon in DEPTH, and the slice preserves the layer-type *ratio* while
discarding 24 of 28 layers — exactly the axis the argument lives on. The flat cosine across K
rules out compounding along the sequence; it says nothing about compounding across 28 layers.

So the honest status: the exclusion is **unsupported by evidence and looks over-conservative**,
not yet **refuted**. What would settle it, in order:

1. The full 28-layer Mellum2 on the 64 GB Linux box (it does not fit here) — same two arms.
2. A routing-flip COUNT, not a cosine: how many top-k selections differ, at what gate-weight gap.
   A flip at a near-tie where `norm_topk_prob` renormalizes is bounded by the smallest of the 8
   weights; a flip at a real margin is the thing the guard is right about.
3. Token-level divergence over a greedy continuation, which is what a user actually experiences.

Extending G24 to MoE was worth **1.52×** on the full model at K=8192 (measured below), against
Lever 4's sub-5%. Still the right call by a wide margin — but the 3.11× that appears above is a
4-layer-slice figure and was quoted as the model-level number until it was checked.

## Scope limits

1. **Resident weights, no streaming.** Only Lever 4's compute half was measured.
2. **4-layer slice, not the 28-layer model.** Valid here because Mellum2's `layer_types` has
   period 4, so layers [0,4) reproduce the 21-sliding/7-full mix exactly, and every layer is
   `sparse`. **This does not transfer** — see the pre-registration's warning before reusing it.
3. **CPU only, M1 Pro.** Says nothing about Metal or CUDA.
4. The full checkpoint (12 GB at int8int8) would not fit against 8.1 GB of swap already in use;
   swap was recorded before and after every run and stayed flat, so nothing paged.

---

## The full model at long K — the cell neither machine had (2026-08-29)

Both earlier speedup figures came from configurations nobody runs: **3.11× on a 4-layer slice at
K=8192**, **1.08× on the full model at K=2048**. This is the cell that decides whether the MoE
extension matters in practice. Run on both machines simultaneously, full 28-layer Mellum2, K=8192,
`TestA3MoEExclusionIsMeasured`:

| machine | quant | commit | acc64 | f32 | speedup | 1 − cosine |
|---|---|---|---|---|---|---|
| Ryzen 3700X, 62 GB | int8int8 | `9fa307c` | 8411.6 s | 5540.1 s | **1.52×** | 2.777e-3 |
| M1 Pro, 16 GB | int4 | `03908c6` | 3935.2 s | 2480.5 s | **1.59×** | 6.370e-3 |

**~1.5×, and the slice overstated by about 2×.** The strongest evidence here is not either number
but their *convergence*: two architectures, two quants, and very different memory conditions landing
within 0.07× of each other.

**Why the slice lied — leading explanation, UNVERIFIED.** The slice's ~1.6 GB of weights fit in
cache, so weight matmul was cheap and attention read as 97.1% of prefill work at K=8192. The full
model's 6–12 GB does not fit; weight matmul becomes bandwidth-bound, attention's share falls, and an
attention-only kernel swap buys correspondingly less. Confirming this needs a profile at K=8192 on
the full model — another multi-hour run, not done.

**The Mac arm paged and its absolute times are inflated.** Swapouts rose 304,524,379 → 307,177,129
(+2.65M pages ≈ 43 GB) and macOS grew the swap file from 8 GB to 13 GB mid-run. Pre-flight checked
the machine *at rest* and correctly found no active paging — but the run itself creates the
pressure, so the pre-flight tested the wrong condition. Both arms paged alike and the ratio matches
the clean box run, so it is kept as corroboration; **the citable figure is the Ryzen 1.52×.**

### A correction this run forces on the divergence claim

"MoE diverges 0.90× dense" was a **K=2048** statement and does not survive to long context:

| K | MoE (full 28L, int8int8) | dense (28L) | ratio |
|---|---|---|---|
| 2048 | 2.126e-3 | 2.352e-3 | 0.90× — MoE better |
| 8192 | 2.777e-3 | 2.400e-3 (recorded) | **1.16× — MoE marginally worse** |

Both remain ~4× inside the ≥ 0.99 bar, which is what the extension decision rests on. But the dense
K=8192 figure is cross-session, so the *direction* is not load-bearing, and **"MoE diverges less
than dense" should not be repeated unqualified** — it was true of the one K where it was measured.

The int4 arm's 6.370e-3 is the largest divergence recorded for this flag on any model. Still inside
the bar, but int4 + f32 attention is the worst combination measured and deserves its own line in any
future scoping.
