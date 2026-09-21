# Reduction-tree accuracy: a BLOCKED V-sum fold beats the sequential fold — measured 2026-09-12

**Pre-registered before the run** (`PREREG.md`, reproduced in §5). Verdict: **HOLDS**, the bar met
in **every** cell at nKeys ≥ 2048, worst mean-ratio 1.76 against a pre-registered 1.5.

## 1. What this decides, and what it does not

It tests the single premise the "third option" for the decode-attention fork rests on — the option
being: instead of abandoning bit-identity, **re-canonicalise** the reduction as the blocked tree and
move CPU / `attn_batched` / a split decode kernel all onto it, which preserves cross-M and
cross-backend identity AND unlocks the key-axis split that
`docs/task-decode-splitkv-attention.md`'s V-sum ceiling needs.

The premise: **for `acc = Σ_s w[s]·v[s]` in f32 at decode shapes, a blocked fold is closer to the
f64 truth than the strict sequential left fold `splitkv_vsum` runs.** If true, re-canonicalising is
an accuracy *improvement* that happens to buy occupancy. If false, it is a trade.

**It does not decide the fork.** It is one V-sum in isolation, on synthetic inputs, in Go. §4 lists
every limitation; the end-to-end question has separate, corroborating evidence in the L2 fidelity
gate (`cuda/prefill.go:694`).

## 2. Result

Error normalised by the conditioning denominator Σ|w·v|, **not** by the result (§4). Ratios are
seq/blk, so >1 means blocked is more accurate. 400 trials/cell, paired — both arms see identical
inputs, and the median is taken over per-trial ratios rather than over pooled means (CLAUDE.md
rule 7).

**The advantage grows with the split count** — the same S that buys occupancy (nKeys=8000, flat softmax):

| splits S | mean-ratio | paired median | blocked wins |
|---|--:|--:|--:|
| 4 | 1.93 | 1.72 | 67% |
| 8 | 2.51 | 2.28 | 73% |
| 16 | 3.28 | 3.27 | 76% |
| 32 | **4.93** | 4.61 | 84% |

**And it grows with depth** — the region P24 says the deficit lives in (flat softmax, S=32):

| nKeys | mean-ratio | paired median | blocked wins |
|---|--:|--:|--:|
| 2048 | 4.01 | 3.55 | 80% |
| 3900 | 4.66 | 4.40 | 83% |
| 8000 | **4.93** | 4.61 | 84% |

Across all 36 cells at nKeys ≥ 2048 (nKeys × S × softmax-peakedness): **0 failed** the
pre-registered 1.5 bar; worst 1.76, best 4.93.

## 3. The finding that matters

**Occupancy and accuracy are the SAME move here, not a trade.** The split count S that creates
threads — the only thing that can lift `splitkv_vsum` off its floor — is the same S that shortens
each partial fold and reduces error. Both trends are monotone in S and in nKeys, which is the
textbook O(n)-vs-O(log n) error-growth signature of sequential against blocked summation.

That reframes the fork. The V-sum's 3.8% occupancy is *arithmetically* pinned: nH·hd = 12·128 =
1536 threads = 48 warps against 40 SMs × 32 warp slots = 1280, giving 3.75% against 3.8% measured —
no block-size choice moves it, because it creates no threads. So the key split is the only exit, and
this says taking it costs nothing in accuracy. What it costs is **bit-identity to history**, which is
a different thing from fidelity and is the question re-canonicalisation is designed to answer.

## 4. Limitations — each one is a reason this is not sufficient on its own

1. **Synthetic inputs.** `w` = softmax over N(0,1)·scale, `v` ~ N(0,1). Realistic in shape
   (non-negative weights summing to 1; signed values, so cancellation is possible) but NOT captured
   from a real model. Real V distributions are not Gaussian and real attention is more peaked than
   even the scale=8 cells.
2. **One op, not the system.** 28 layers with feedback can compound or cancel this. Absolute
   magnitudes are 5e-9..4e-7 of Σ|w·v| — sub-ULP per op. Whether it reaches a logit or an argmax is
   not tested here.
3. **Go f32, no FMA contraction.** The CUDA kernel may contract `a*b+c`. That applies to both arms
   equally so it does not bias the ratio, but the absolute magnitudes are not the kernel's.
4. **Win rate is 49–85%, not 100%.** On an individual token the sequential fold sometimes wins. This
   is a distributional result about the mean and median, not a guarantee per token.
5. **The metric had to be fixed mid-run, and the first version was wrong.** Normalising error by the
   RESULT made the mean heavy-tailed — the result is a weighted mean of signed values and lands near
   zero often enough that single trials dominated, producing cells that read as blocked being 14x
   WORSE. Normalising by Σ|w·v| fixed it. `cuda/attn_fused_test.go:18` records this repo hitting
   the identical trap ("that bar failed widely — worst cosine 0.9642 — and the failure was NOT the
   kernel") and fixing it the same way, by normalising to |V|. The first-version table is kept, and
   is REPRODUCIBLE rather than merely described: `FIRST_METRIC=1` on the harness regenerates it, and
   it is saved beside this file as `…-raw-firstmetric.txt`.

## 4b. An unplanned result: the PAIRED statistic was immune to the bad metric

Worth recording because it is a cheap, concrete demonstration of CLAUDE.md's rule 7 ("difference
matched observations; do not pool"), on this repo's own data.

The worst cell under the bad normaliser, nKeys=2048 / scale=1 / S=4:

| metric | seq mean | blk mean | **pooled mean-ratio** | paired median | blocked wins |
|---|--:|--:|--:|--:|--:|
| result-relative (wrong) | 3.641e-06 | 5.280e-05 | **0.07** | 1.63 | 61% |
| Σ\|w·v\|-relative (right) | 2.746e-08 | 1.437e-08 | **1.91** | 1.63 | 61% |

The pooled mean pointed **14x in the wrong direction**. The paired median and win rate were
**correct and unchanged** — and not approximately: the paired columns are bit-identical across both
metrics for all 48 cells, because a per-trial normaliser cancels inside a per-trial ratio. Had this
been analysed only by pooled means, it would have produced a confident, published, backwards verdict
on the fork. The pairing is what caught it, before the normaliser was even diagnosed.

## 5. Provenance

nobara-pc · Go 1.27 · 400 trials/cell · 48 cells (nKeys {512,2048,3900,8000} × S {4,8,16,32} ×
logit-scale {1,4,8}) · seeded per cell (`n*1000 + S*10 + scale`), so every cell is reproducible ·
harness and pre-registration in the session scratchpad, copied to `treeexp/` beside this file if
kept. No GPU involved; no model loaded; nothing measured from `/srv/models`.

Decision rule as pre-registered: HOLDS if blocked's mean ≤ 2/3 of sequential's at nKeys ≥ 2048;
AMBIGUOUS→PARKED within ±20%; REFUTED if blocked worse by >20%. Two things that can disagree were
registered (mean and the paired median); they agree in all 36 cells.
