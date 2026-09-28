# Noise registry: the A/A spread each gate's instrument already has on record (TE3, 2026-09-28)

> Item TE3 of [`task-test-efficiency-2026-09.md`](../tasks/task-test-efficiency-2026-09.md). The data is
> [`noise-registry.json`](noise-registry.json); [`scripts/power.py`](../../scripts/power.py) reads it and says how many
> runs a gate needs **before** the run. **Nothing was run or timed for this.** The registry is seeded from records
> already on disk. Its computed rows come from
> [`te3_registry_seed.py`](test-efficiency-2026-09/te3_registry_seed.py), which reuses `te2b_variance.py`'s loader and
> curation. Every entry carries its source, and `python3 scripts/power.py verify` recomputes each sd from the values the
> entry records (0 mismatches at this writing).

## 1. How to read an entry

- **Cell** is `machine/backend/model/depth`, and `*` matches anything.
  - The machines: `mac` is the MacBook Pro M1 Pro, 16 GB, darwin/arm64. `nobara` is the Ryzen 7 3700X + RTX 2070 SUPER.
  - The backend `cpu` means arm64 on the Mac and amd64 on nobara.
- **sd** is in natural-log units of the ratio (0.01 ≈ 1%), per the entry's **unit**, which is one paired block of that
  instrument. It is always a stated statistic:
  - `rms_log_about_0`: the RMS of ln(A/A ratio) about 0. In an A/A the true log ratio is 0, so no degree of freedom is
    spent on a mean. TE1's pre-registration uses the same statistic.
  - `sd_log`: the sample sd of ln(value) over repeats of one arm (`arm_log_level`) or of one A/B ratio.
  - `pair_diff`: the same quantity measured twice, repeated over several pairs; the sd of one reading is
    √(mean(Δln²)/2).
  - `quoted`: a record's own figure that cannot be recomputed (a max, a range). Such an entry has no sd, and power.py
    will not size N from it.
- **floor** is a bias that more blocks of the same design do not re-draw: an order effect, or a build effect. power.py
  charges it against the margin, not the sd.
- **Provenance**, from best to weakest:
  - `cross-build-aa`: two builds whose code for this cell the record says is identical. This is what a new ÷ old gate
    faces.
  - `direct-aa`: the same binary in both arms.
  - `effective-aa`: the same binary, with a switch this cell does not take.
  - `replicate`: the same A/B or arm measured twice.
  - `model-inference`: derived from other entries or from variance components.
  - `single-observation`: n = 1.
  - `floor-only`: no sd.

## 2. What the registry holds

**83 entries**, drawn from 109 record files: 43 nobara, 37 Mac, 3 both. **62 carry an sd**, and 16 gaps are listed.

| instrument | entries | provenance |
|---|---:|---|
| served decode, goinfer ÷ goinfer (`served-decode`) | 24 | 2 cross-build, 14 direct, 4 env-arm, 2 replicate, 2 floor-only |
| served goinfer ÷ peer (derived), a peer against itself, the same build across sessions | 4 + 6 + 5 | direct and derived |
| served W7 aggregate / served TTFT | 2 / 1 | cross-build |
| in-process kernel / whole-token / prefill | 9 / 14 / 1 | direct, replicate, floor-only |
| logits fidelity floors (cosine, KL, p10, argmax, bit-identity) | 17 | direct nulls and perturbation controls |

**What a served new ÷ old gate would use today.** This is the entry `power.py served-decode <cell> --bar 0.97 --paired`
picks, with a true ratio of 1.00:

| cell | entry | sd | floor | N (blocks) |
|---|---|---:|---:|---:|
| nobara CUDA 0.5B | cross-build A/A, fused_rms sweeps (n=6) | 0.0164 | 0.0144 | 11 |
| nobara CUDA 1.5B / 7B | same-binary restart pairs, pooled (n=20, 19 of them 0.5B) | 0.0081 | 0 | 3 |
| nobara CPU (any) | same-binary restart pairs (n=3) | 0.0022 | 0 | 2 |
| Mac Metal 1.5B / 7B @128 | cross-build A/A, R17 depth-128 control (n=2) | 0.0129 | 0 | 4 |
| Mac CPU 0.5B | same-binary pairs across the 09-28 passes (n=4) | 0.0128 | 0.0165 | 9 |
| Mac CPU 7B | same, n=4 | 0.0524 | 0.0165 | **cannot resolve at N ≤ 12** |

Three things stand out.

- **A same-binary A/A understates what a new ÷ old gate faces.** On nobara CUDA the same binary restarted in one session
  gives an RMS of 0.81% (20 pairs). Two builds whose 0.5B code is identical give 1.64%, and **all 6 readings sit below
  1** (0.978–0.996). That is a build effect, the Mytkowicz layout bias the campaign doc names. It is also exactly
  TE5(b)'s "the served harness at n=2 'resolves' −2% on identical code"
  (`test-efficiency-2026-09/te5b-concordance-2026-09-28.md:298`). The Mac's only cross-build A/A, R17's depth-128
  control, shows the same shape: the 1.5B reads 0.982 with the runs not overlapping.
- **The Mac CPU by day cannot resolve a 3% bar on the 7B** at any feasible N. The same 3cd62e6d binary read 16.96 /
  17.67 / 17.54 tok/s across the three 09-28 passes, and the 7B cells sag within a lifetime (TE2(b) §4.4). Every Mac
  CPU entry is daytime at `BENCH_MAX_LOADAVG=2.5`. TE1's queued A/A night is the first quiet-Mac measurement.
- **In process, the 0.5B is ~10× noisier than the 1.5B / 7B on nobara's CPU.** The smallshape A′/A RMS is 5.0% for
  the 0.5B against 0.35% (1.5B) and 0.20% (7B), and the S0 / S0b / R-06 ABBA gates' reported 0.5B pairs span
  0.899–1.040. That is why those gates graded only the 1.5B and 7B.

## 3. The ~3.5% between-session drift, reconciled

`CLAUDE.md:37` says "drift between sessions is ~3.5% on this box". TE2(b) found a median of 0.15% (p90 0.99%) for
nobara's same-build cross-session pairs (`test-efficiency-2026-09/te2b-variance-2026-09-28.md:390`). **Both are true,
because they measure different things. As a typical figure the 3.5% is an over-generalisation of one observation; it is
not a distribution.**

**Where the 3.5% comes from.** It rests on one cell, measured once in each of two sessions, on 2026-08-09 (commit
414bb364):
- goinfer, phi3-mini, temperature-only sampling, CUDA, driver 595.58.03;
- the same binary, 686c9f8, read 112.4 tok/s in one session and 116.5 in another (ln ratio 0.036);
- the two readings also came from different scripts: the original row was an add-on script's
  (`docs/legacy-benchmarks.md:908`, `:910`; restated at `docs/benchmarks.md:2415`).

**In the same gap, the peer moved 0.16% and 0.33%** (Ollama 125.8 → 125.6 and 121.8 → 121.4,
`docs/legacy-benchmarks.md:913`). That is the size TE2(b) measures for peers.

**What TE2(b)'s 0.15% measures.** It is the median |Δ| over 466 same-build restart pairs in different sessions, from
Sep 4–28.
- **Every one of those pairs is a peer:** 327 Ollama and 99 llama-server on CUDA, the rest on CPU. 419 of the 426 CUDA
  pairs are greedy.
- The RMS of those pairs is 0.74% (CUDA, n=426) and 0.73% (CPU, n=40).
- **Only 3 of the 426 CUDA pairs reach 3.5%.** The p99 is 3.12% and the max 5.03%. So 3.5% is a ~p99 event for a
  peer, not a typical one.
- **goinfer's own session term is unmeasured.** Every goinfer build on record lived in a single session (TE2(b) §4.2).
  The August observation is the only goinfer cross-session pair there is, and it is a sampled cell.

**So neither figure is simply stale:**
- 3.5% remains the only goinfer cross-session reading.
- 0.15% / 0.74% is what the peers do.
- The typical-drift claim does not hold for peers, and is n = 1 for goinfer.

**The 3.5% has also travelled to the Mac as "this box's" figure**, in Mac records: `r12-mlx-row-2026-09-18.md:22`,
`metal-depth-r2-2026-09-18.md:86`, `w4f16-decode-speed-2026-09-21.md:60`. TE5(b) uses it as its no-spread resolution
threshold on both machines (`te5b-concordance-2026-09-28.md:87`). The Mac's own cross-session figure is heavy-tailed:
Metal peer pairs have a median of 0.52% and p90 2.67%, but an RMS of 5.2% (n=35), because one Ollama 7B d2048 cell
moved 24.5%.

**Suggested wording** for the main session's `CLAUDE.md` edit: "same-build cross-session drift is ~0.7% RMS for peers
on nobara (tail to ~5%); goinfer's is measured once, 3.6%, in a sampled cell. A cross-session ratio is not a ratio."
The registry carries all of these as `served-decode-xsession` entries.

## 4. The Mac's 0.022 pass-to-pass figure

The owner's 09:05 amendment added 0.022 to the 0.97 bar (`docs/tasks/task-cpu-decode-peer-gap-2026-09.md:491`), and
TE4-SEQ-v1 used δ = 0.022 in log units. **The 0.022 is an absolute difference between ratios near 0.5** (0.42–0.57, the
L1 regression), not a log ratio.
- **In log terms the pass-to-pass differences are 0.015, 0.039 and 0.045** (0.5B / 1.5B / 7B; `te3_registry_seed.py`
  §2).
- **Pass 2 reads higher in all three models.** Pass 2 ran old first, so in both passes the arm that ran first read
  relatively lower: a consistent order effect.
- With 3 pairs, noise and order cannot be separated:
  - all-noise reading: one pass's sd is 0.0251 (3 df);
  - all-order reading: an order effect of 0.0165 (mean half-difference), 0.0225 at most.
- **The registry charges both** (entry `seed-mac-cpu-l1gate6-replicate`, and the floor on every Mac CPU served entry).
  This is conservative on purpose. A counterbalanced (ABBA) design may pass `--floor 0`.
- **The coincidence to know about:** 0.0225, the largest half-difference, happens to equal the 0.022 that was used.
  The amendment's arithmetic was therefore right if 0.022 is read as a per-arm order effect in log units, and it is 2×
  too small if read as the pass-to-pass difference, which is what the amendment called it.

## 5. Other findings the seed turned up

- **TE4's killer is in the registry.** Peer-claim cell i's Ollama fell 125.2 → 93.7 → 93.1 inside one cell, a sample
  sd of 16.9% over run means (`mac-cpu-peerclaim-i-ollama-shift`). A between-run allowance for a TE4-SEQ-v2 on the
  daytime Mac CPU has to be of that order, not 2%.
- **The in-process kernel bench on the Mac CPU by day** gives 5.3% sd for one reading of a kernel-bench ratio.
  - The same arm was read ~1 min apart: scalar-widen 1.5B 1.402 → 1.193, fused 7B 0.870 → 0.949.
  - The f32 arm itself moved only 0.4–1.3%.
- **Three records overstate their own tightness.** Their raw logs contradict the summary.
  - `attn-fused-tile-2026-09-21.md:7` says "every arm's round-to-round spread < 0.5%". The rounds reach 0.85% for the
    A0 arm and 1.2% for the others.
  - `attn-fused-tile128-default-2026-09-21.md:25` says "gemv control within +-0.6%". Per round it reaches 1.03%, and
    its 64x64 attention rounds reach 1.8% (1.5B) and 2.2% (0.5B) at K=512. The registry uses the per-round values
    (`cuda-prefill-gemv-control`).
  - `splitkv-8000-reanchor-2026-09-12.md:76` credits "§B6.3's measured floors ran 0.03–0.82%". §B6.3 holds no A/A
    floors. 0.03–0.82% is exactly the range of `aikit-kv-regather-strided-p1.md:52-55`'s CPU attention floors, so the
    citation looks misattributed.

  None of the three changes a verdict here. Each record's effect is far outside its noise.
- **Position effects in process.** In `concurrency-mc3c-s1-2026-09-27.md:96`, the same config placed after a serial
  arm read up to 1.41× apart; the record discards it. A′/A with a warm block read up to 1.071 on identical code (M=1
  q‖k‖v). The registry keeps the first as a floor-only warning.
- **Logits fidelity floors (17 entries).** The per-family numbers a fidelity bar has to sit above:
  - Metal W4A8: a 1e-6 residual nudge alone gives cosine 0.99945–0.99961.
  - Metal decode-attention KL nulls: ratios 0.954–1.027. That is why the owner's bar is ≤ 1.05.
  - CUDA V-sum trees: 0.98–1.00.
  - The CPU p10-cosine response to a 2^-11 scale perturbation is ±0.02, against Track B's 0.005 allowance.
  - CPU phi3 self-noise under ±1-ULP norms: min cosine 0.953.
  - qwen3.5's cross-machine cosine difference: 0.0015.
  - CPU runs are bit-identical session to session. Metal is bit-identical run to run on one machine.
  - Only opt-in CUDA lanes have a logits A/A.

## 6. `power.py`: the model, and the two evaluation calculators

**Speed gates.** Each paired block gives d = ln(new) − ln(old) ~ N(ln(expect) + b, sd²), with an unknown bias
|b| ≤ floor. The decision interval is TE4-SEQ-v1's: mean(d) ∓ t·SE ∓ floor. N is the smallest block count whose exact
noncentral-t power reaches `--power` at margin = |ln(expect/bar)| − floor, two-sided α by default.
- If margin ≤ 0, it prints **"cannot resolve at any N"**.
- If N exceeds the instrument's `max_n` in the registry (a policy default of 12 for served and 60 for in-process, not
  a measurement; `--max-n` overrides it), it prints **"cannot resolve at a feasible N (max N …) — change the
  instrument"**, with the smallest expected ratio that N could resolve.
- `--worst-case` charges the floor twice. A registry miss refuses rather than guesses, so pass `--sd`.
- **Checked against textbook values:** the normal formula exactly, and exact-t N against G*Power: paired d = 0.5 → 34
  (power 0.8078); two-sample d = 0.5 → 64, 0.8 → 26, 0.2 → 394.

**Binomial: the D6a amendment reproduced.** 2852eaa1 (`docs/tasks/task-constrained-confidence.md:549`) says the noul
top-1 "estimated here from 400 rows. Its standard error near p ≈ 0.9 is about ±1.5 points":

```
$ python3 scripts/power.py binomial --p 0.9 --half-width 0.015
p 0.900, half-width 0.0150 (1.50 points):
  N = 400 for it to be one SE (a 68% interval), which is how the D6a amendment (2852eaa1) read +/-1.5 points
  N = 1537 for it to be the 95% half-width (z = 1.960)
$ python3 scripts/power.py binomial --p 0.9 --n 400
p 0.900, n 400: SE = 0.0150 (1.50 points); 95% Wald half-width = 0.0294 (2.94 points)
$ python3 scripts/power.py binomial --p 0.9 --stratum 9767:400 --stratum 3219:400 --stratum 72:72
SE of the weighted proportion = 0.0118 (1.18 points); 95% half-width 0.0232 (2.32 points)
```

The amendment's figure is reproduced exactly, and two readings follow from it.
- **The ±1.5 points is one SE.** A 95% interval at 400 noul rows is ±2.9 points, nearly the whole 3-point band.
- **The kind-weighted top-1 the decision reads is a little tighter than the amendment's figure,** because noul's
  weight is 0.748: ±1.18 SE and ±2.32 at 95%, if every kind sits near 0.9.

The amendment's "within ~1.5 points of an edge is the owner's call" is a 1-SE band.

**ECE: an approximation, and what it says about D6a's ECE bar.**
- **The derivation.** Take a perfectly calibrated model. Each occupied bin's |accuracy − confidence| is |N(0,
  p(1−p)/n_b)|, and a half-normal's mean is σ√(2/π). Weight by n_b/N over B equally filled bins, and the plug-in ECE
  averages √(2/π)·√(B·p(1−p)/N).
- **B equally filled bins is the worst case for B bins** (Cauchy–Schwarz), so the formula is an upper bound for B
  occupied bins. The sd of the estimate lies between √((1−2/π)p(1−p)/N) and √(p(1−p)/N).
- **What it does not come from.** The closed form is only the half-normal mean. The upward bias of the binned plug-in
  estimator it quantifies is documented by Kumar, Liang & Ma (NeurIPS 2019, "Verified Uncertainty Calibration") and
  Roelofs et al. (AISTATS 2022, "Mitigating Bias in Calibration Error Estimation").
- **Checked** by a Monte Carlo test in `test_power.py`.
- **For D6a** (top-label ECE over 15 equal-width bins, combined over the kinds, bar ECE ≤ 0.05), a perfectly calibrated
  arm A would read:

| sample | 2 occupied bins | 4 | 15 (the upper bound) |
|---|---:|---:|---:|
| the amended 400 / 400 / 72 | 0.017 | 0.024 | 0.047 |
| the original 1000 / 1000 / 72 | 0.011 | 0.015 | 0.030 |
| JEV-9B's full 13,058-row split | 0.003 | 0.004 | 0.008 |

(`power.py ece --acc 0.9 --bins B --stratum 9767:400 --stratum 3219:400 --stratum 72:72`.) So at the amended sample, a
perfectly calibrated arm's ECE carries an upward bias of about 0.02–0.05 against a 0.05 bar. It is compared with a
published 0.0396 whose own bias is under 0.01. This is an approximation, and the number of occupied bins is unknown
until the run, but it is the ECE counterpart of the top-1 point above. It is for the owner before D6a runs, not a
verdict.

## 7. Gaps: cells a gate would need that the registry cannot answer

These are listed in the JSON's `gaps`, 16 of them. The ones that bite first:
1. **nobara CUDA 1.5B / 7B served:** no same-binary in-session goinfer replicate, and no cross-build A/A. A new ÷ old
   gate there borrows the pooled 0.5B-dominated 0.81%, which cannot see the build effect the 0.5B shows (1.6% + a 1.4%
   floor).
2. **goinfer across sessions:** unmeasured apart from the n = 1 behind the 3.5%.
3. **The Mac CPU at night:** everything is daytime at cap 2.5. TE1's A/A night fills this.
4. **Served TTFT:** only Mac Metal 1.5B through `bench_prefill_stall.py`. Nothing for `bench_peer_prefill.py`, nobara,
   or the Mac CPU.
5. **WebGPU:** no served A/A, and one in-process repeat (8%).
6. **MoE goinfer served cells** (M26 / M35): no A/A, and they trend inside a lifetime.
7. **In-process on the Mac CPU:** no whole-token A/A; only the drifting kernel bench.
8. **ncu kernel A/Bs on CUDA:** no same-kernel repeat recorded.
9. **Logits:**
   - no per-family run-to-run or cross-arch floor behind the 0.9999 / 0.99 / 0.98 / 0.95 bars;
   - no do-nothing arm for the p10 bars;
   - no logits A/A of default CUDA decode;
   - no basis for the 3% near-tie rule.

TE2(b)'s pre-registered night (1.5B / 7B CUDA goinfer, one build, 4 restarts × 6 runs) closes gap 1's same-binary half.
**A second build of identical code in the same night closes the cross-build half.**

## 8. Pre-registration template (two lines, per TE3)

```
Noise: the registry's noise for the cell is sd <sd> / floor <floor> (<entry id>, <provenance>, n=<n>), source <record:line or commit>.
power.py N: `power.py <instrument> <cell> --bar <bar> [--expect <r>] --paired` -> N = <N> blocks (power <p>), or "cannot resolve" -> <the instrument chosen instead>.
```

## 9. Refresh (monthly, per TE3)

- Re-run `te3_registry_seed.py --json`. Its entries (`origin: te3_registry_seed.py`) replace the same-id entries in the
  JSON.
- Curated entries (`origin: curated`) are added by hand from new records, with source, statistic and values.
- Run `power.py verify` and `scripts/test_power.py` afterwards.
- When a new record's A/A lands, the entry goes in with the record, not a month later.
