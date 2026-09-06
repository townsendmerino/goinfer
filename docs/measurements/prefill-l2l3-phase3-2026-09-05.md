# CUDA prefill §3 fidelity gate — each lever SHIPS alone; the COMBINATION fails one cell (2026-09-05)

**The first fidelity data CUDA prefill has ever had against an f32-activation reference, and it
says three things the doc did not anticipate.** (1) At depth — K=1024 and K=3900 — the fast path
is **closer to the truth than the exact path that ships today**, on every criterion. (2) **L2 alone
and L3 alone each pass the whole decision set**, both beating exact even at K=256. (3) The
**combination** fails at K=256 only, which no additive model of the two levers predicts.

**Pre-registered verdict, recorded as measured: the COMBINATION DOES NOT SHIP on the decision set
AS ORIGINALLY WRITTEN**, because that set includes K=256 and the combination fails there. §3 calls
that a result, not a failure.

**Superseded the same day by an explicit operator decision, recorded in §2.6 below and in
`task-prefill-gap.md` §3:** §3 mandates a prompt-length floor AND names a decision cell below any
plausible floor — an inconsistency in the document itself. Resolved toward the Floor clause, with
the floor set at **512** and justified by a cell measured at that depth. The K=256 failure stands
on the record and is not withdrawn.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER, driver **595.91.07**, Nobara 44; 16 cores, 62 GB |
| goinfer | `f7bc4dae` (harness) with the L2/L3 kernels from `46d5b3a8` / `6ae846bb` |
| reference | **CPU, f32 weights AND f32 activations** (`Options{Backend:"cpu", Quant:""}`), `GOINFER_CPU_FAST_ATTENTION=0` so attention is the exact f64-accumulating kernel. **Both models at f32** — this box's 62 GB fits the 28 GB f32 D7, so unlike the Mac's 16 GB run no weight quantisation enters the reference at all |
| reference build | own process, 10 real-prose prompts per cell, 8 concurrent workers, **concurrency preflight passed** (identical prompt through 8 workers → bit-identical seed logits). 66 min wall, 50 files, 1.9 GB, `~/goinfer-logs/prefill-ref/` (not the repo) |
| arms | exact = `PrefillLast` with both levers off (**this is what CUDA ships today** — §3: "the exact arm sets the bar because it is what ships today"); fast = `PrefillLast` with the selected levers |
| scoring | `decoder/fidelity_testhook.go`; both arms teacher-forced on the **reference's own** 64 tokens |
| logs | `~/goinfer-logs/prefill-l2l3/p3a-*.log`, `p3b-*.log` |

## 1. S — the decision model

Mean teacher-forced top-1 agreement vs the reference, hard flips (3% near-tie rule) out of 640
continuation positions, and mean KL(reference ‖ arm):

| K | arm | agreement | hard flips | mean KL | cell |
|---|---|---|---|---|---|
| **256** | exact | 93.75% | 4/640 | 0.03708 | — |
| | L2 only | **94.06%** | **3** | **0.03700** | **SHIPS** |
| | L3 only | **94.22%** | 4 | **0.03638** | **SHIPS** |
| | **both** | **92.34%** | **3** | **0.03678** | **DOES NOT SHIP** |
| **1024** | exact | 89.22% | 10/640 | 0.04583 | — |
| | L2 only | 89.38% | 8 | 0.04409 | SHIPS |
| | L3 only | 89.69% | 9 | 0.04495 | SHIPS |
| | **both** | **89.84%** | **7** | **0.04382** | **SHIPS** |
| **3900** | exact | 91.41% | 12/640 | 0.05103 | — |
| | L2 only | 91.41% | 11 | 0.05076 | SHIPS |
| | L3 only | 91.72% | 8 | 0.05047 | SHIPS |
| | **both** | **92.03%** | **8** | **0.05022** | **SHIPS** |

**Decision-set verdicts (K ∈ {256, 1024}; K=3900 confirms only): L2-only SHIPS. L3-only SHIPS.
BOTH DOES NOT SHIP.**

### 1.1 At depth the fast path is closer to the truth than the exact path

At K=1024 and K=3900 the combined fast path beats the shipped exact path on **all three**
criteria — hard flips **7 vs 10** and **8 vs 12**, lower KL, higher agreement. §3 was designed to
ask "is the fast path *not further* from the reference than the exact path?"; the answer at depth
is that it is **nearer**. That is a claim about the *exact* path too: the batched W4A8 prefill that
ships today is measurably further from f32 than the tensor-core path is.

### 1.2 The K=256 failure is an INTERACTION, and no pre-registered suspect predicted it

§3.1 pre-registered a suspect for exactly this case: *"If fast is measurably worse, run L2-only and
L3-only arms — the f16 conversion of K/V is the first suspect."* Both arms were run. **Neither
lever is the cause.** L2 alone is +0.31 pt over exact at K=256; L3 alone is +0.47 pt. Both are
*improvements*. The combination is **−1.41 pt**.

An additive model of two individually-positive effects predicts about **+0.78**. The measurement is
**−1.41**, a discrepancy of ~2.2 pt.

**Both clauses of criterion B failed**, not one: the cell mean is −1.41 pt against a −1.0 bar, and
fast beat exact on **4 of 10** prompts against a ≥5 requirement. Per-prompt diffs:
`−1.6, 0, −4.7, −3.1, +3.1, −1.6, 0, −3.1, −3.1, 0`.

**How much of this can n=10 resolve? Not all of it, and that is stated rather than resolved.** The
per-prompt sd is ~2.2 pt, so the standard error of the cell mean is ~0.70 pt. The −1.41 is ~2 SE
from zero, and the departure from additivity is ~3 SE. That is suggestive of a real interaction and
**short of conclusive**. Resolving it needs more prompts — which is a follow-up, not something to
do now, because re-running a failed cell with a larger sample until it passes is how a gate stops
meaning anything.

**A gap in the pre-registration, worth recording against §3 rather than against this run.** §0
requires that a decision rule "include an explicit *ambiguous → parked* band", and §3's gate has
none: it is binary. This cell — 0.41 pt outside a bar, within ~2 SE of it — is precisely the zone
that band exists for. The verdict below applies the rule as written; the missing band is a defect
in the rule, and noting it after the fact is not the same as widening it.

### 1.3 Mechanism: why a *smaller* numerical error can *lower* top-1 agreement

L3's unit gate puts it within **1.1e-6** of the exact GEMV, and it performs 4× FEWER float roundings
(one fold per 32-group in int32 vs one per 8-element word). Yet it moves agreement. There is no
contradiction: KL is continuous in the logits and a 1e-6 perturbation barely moves it — which the
data shows, L3-only having the *lowest* KL of any arm at K=256 — while teacher-forced top-1
agreement is a **near-tie-sensitive** metric, and an arbitrarily small perturbation flips an
argmax whenever two logits are within it. The two metrics disagreeing is the expected signature of
a tiny perturbation, not of a defect, and it is why §3 gates on three criteria rather than one.

## 2. D7 — every arm fails, and it exposes a defect in criterion A

| K | arm | agreement | hard flips | mean KL | cell |
|---|---|---|---|---|---|
| **256** | exact | 88.28% | 11/640 | 0.08473 | — |
| | L2 only | **88.28%** (identical) | 13 | **0.08259** | DOES NOT SHIP (critA) |
| | L3 only | 87.03% | 13 | **0.08412** | DOES NOT SHIP |
| | both | 87.19% | 15 | 0.08613 | DOES NOT SHIP |
| **1024** | exact | 86.41% | 13/640 | 0.08660 | — |
| | L2 only | **86.41%** (identical) | 17 | 0.08676 | DOES NOT SHIP (critA) |
| | L3 only | **86.41%** (identical) | 19 | **0.08445** | DOES NOT SHIP (critA) |
| | **both** | **88.28%** | **11** | **0.08211** | **SHIPS** |

**D7 decision-set verdict: DOES NOT SHIP, on every arm.**

### 2.1 critA is noise-dominated at these counts, and the numbers say so

Read the K=1024 hard-flip column in order: **exact 13, L2-only 17, L3-only 19, both 11.** The
*combination of two levers* has FEWER flips than either lever alone and fewer than exact. No
additive or causal model produces that ordering. What produces it is sampling.

Hard flips are a small count: ~13 out of 640 positions. Treated as Poisson, sd ≈ **±3.6**, so the
entire 11–19 spread lies within ±1 sd of a common mean. **Criterion A ("fast's hard-flip count may
not exceed the exact arm's") is a STRICT inequality on that count with no tolerance band**, so at
these rates it resolves close to a coin flip. Three D7 cells fail on critA alone while their
agreement is *numerically identical to exact* to the digit (86.41% vs 86.41%, 88.28% vs 88.28%) and
their KL is *lower* than exact.

This is not run-to-run variance — every arm is deterministic and re-running reproduces the counts
exactly. It is **sample** variance: which specific near-ties happen to sit in these 640 positions.
More prompts would shrink it; the pre-registered set has 10.

**This does NOT overturn any verdict, and is not used to.** D7 fails on every arm and S's
combination fails, as recorded. It is filed as a defect in §3's rule to fix *before the gate is next
run*, in the same spirit §3.1 corrected the oracle: state the flaw, keep the result. The concrete
repairs, for whoever revises §3:

- give critA a tolerance proportional to its counting noise (e.g. `fast <= exact + k*sqrt(exact)`)
  rather than a strict inequality, or gate on the flip RATE with a confidence interval;
- add the **ambiguous → parked** band §0 requires and §3 omits;
- state a minimum sample: 640 positions at a ~2% flip rate cannot resolve differences of a few
  counts, which is the size of every difference observed here.

### 2.2 What survives the noise

The one D7 signal larger than its noise band is the same one S showed: **at K=1024 the combined
fast path beats exact on all three criteria** (agreement 88.28% vs 86.41%, flips 11 vs 13, KL
0.08211 vs 0.08660), and at K=256 it is worse. The direction matches S at the same depths.

## 2.5 The two-model pattern, stated plainly

| | K=256 | K=1024 | K=3900 |
|---|---|---|---|
| S, combination | worse | **better than exact** | **better than exact** |
| D7, combination | worse | **better than exact** | not measured |

Both models, same direction at the same depths. That is harder to attribute to sampling than either
alone — a coincidence does not reproduce with the same sign on a different model at the same depth —
but it is two models, not a sweep, and the single-lever arms do NOT reproduce across models (they
ship everywhere on S and fail everywhere on D7, the latter on the noise-dominated criterion above).

## 2.6 ADDENDUM — the K=512 cell, and the floor decision (same day, after the above)

The §3 gate's standing cells are 256 and 1024. The operator's decision was to apply §3's own Floor
clause and set CUDA's floor where the fast path is both faster and clean — but **a floor placed
between two measured cells would be interpolating a fidelity result nobody took**, so a K=512
reference was generated (S 3m9s, D7 16m21s, 10 prose prompts each, same CPU f32 reference) and the
gate run there.

| model | arm | agreement | hard flips | mean KL | cell |
|---|---|---|---|---|---|
| S | exact | 93.75% | 5/640 | 0.03321 | — |
| S | **both** | **93.91%** | **5** | **0.03208** | **SHIPS** |
| S | attn only | 93.12% | 7 | 0.03204 | fails critA by 2 flips |
| S | gemm only | 93.28% | 6 | 0.03313 | fails critA by 1 flip |
| D7 | exact | 86.56% | 16/640 | 0.07629 | — |
| D7 | **both** | 86.41% | **16** | 0.07979 | **SHIPS** |

**The configuration production runs — both levers — SHIPS at K=512 on BOTH models.** With the floor
at 512 the decision set becomes {512, 1024}, and every cell in it ships on both models.

**§2.1's critA finding gets a third and fourth instance here, and they sharpen it.** At S K=512 the
combination has FEWER hard flips (5) than either lever alone (7 and 6) — again an ordering no
causal model produces. Across all six (model, K) cells now measured, "both" is best 3 times, tied
twice, and worst once, which is what noise looks like and is NOT evidence of a combination
advantage. Two cells here fail critA by one and two flips against a Poisson sd of ±2.2 on counts of
5. The criterion still needs the repair §2.1 lists before the gate is next run.

**What the floor is, and what would move it.** 512, because that is the shallowest depth with a
PASSING cell; 256 fails and is not withdrawn. Moving it DOWN requires a passing gate cell at the
new depth. The end-to-end behaviour was then confirmed at the caller with no env set: K=256 default
and `=0` measure 169.31 ms and 168.65 ms — the same number, the floor holding — while K=512 and
K=3900 measure 4.08× and 3.90× over the exact arm.

## 3. What this does and does not license

- **The default is unchanged.** Both levers remain opt-in behind `GOINFER_CUDA_FAST_PREFILL`.
  `attn_batched` + `gemv_w4a8_rn` remain the exact path, what spec-decode verify and the parity
  gates run, and what serves every declined shape.
- **The bar was not relaxed and no cell was dropped.** The K=256 cell is in the pre-registered
  decision set and it failed; Phase 1's *earlier, independent* performance data put a length floor
  between 128 and 512, which would exclude K=256 from ever running the fast path — but applying
  that floor to the gate would change the decision set after seeing the result, which is the
  re-baselining this repo warns against. It is listed as an option for a human decision, not taken.
- **10 prompts, one card, one quant, greedy, dense models.** The interaction in §1.2 is measured,
  not explained; no mechanism for it is claimed.
