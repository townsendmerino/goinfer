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

### The caveat that decides whether this is actionable

**A 4-layer slice is the one thing that cannot test the cascade claim.** The mechanism the guard
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

If those hold, extending G24 to MoE is a **3.11×** lever against 97% of long-context prefill
cost — dominating anything Lever 4 could have returned against the remaining 3%.

## Scope limits

1. **Resident weights, no streaming.** Only Lever 4's compute half was measured.
2. **4-layer slice, not the 28-layer model.** Valid here because Mellum2's `layer_types` has
   period 4, so layers [0,4) reproduce the 21-sliding/7-full mix exactly, and every layer is
   `sparse`. **This does not transfer** — see the pre-registration's warning before reusing it.
3. **CPU only, M1 Pro.** Says nothing about Metal or CUDA.
4. The full checkpoint (12 GB at int8int8) would not fit against 8.1 GB of swap already in use;
   swap was recorded before and after every run and stayed flat, so nothing paged.
