# D6b — Route B parity on JEV-9B, graded (2026-10-01)

> **Status: graded against the pre-registration. Owner decision recorded below (2026-10-02): the default is int8int8.** f32 passes exactly. All three quantized
> arms keep calibration but miss the 98% top-1 bar. The registered rule has no branch for that outcome.

Pre-registration: `docs/tasks/task-constrained-confidence.md`, D6b and its two amendments of 2026-09-30; grader
`grade.py` here, unchanged since `c5b74e50` (2026-09-30 15:32, before any arm ran). The arms ran on nobara-pc's night
queue, 2026-09-30 22:36 to 2026-10-01 00:14 (`run-d6b.sh`). Raw rows are in `results/` (one JSON line per item) and
the grader's output is reproduced below. Reference: `testdata/decisions/jev9b_ref_f32.jsonl` (transformers f32) on
`testdata/decisions/items.jsonl`'s 150 items, 84 of them with gold; the reference's own ECE on those rows is 0.1222.

## Pre-registered verdicts

| arm | rows | mean KL (bar) | top-1 (bar 0.98) | ECE − reference, 95% paired bootstrap | verdict |
|---|---|---|---|---|---|
| f32 | 150 | 0.00000 (≤ 0.01) | 1.0000 | +0.0000 [−0.0000, +0.0000]: unresolved | **PASS** |
| int8int8 | 150 | 0.00884 (≤ 0.03) | 0.9267 | −0.0046 [−0.0265, +0.0227]: unresolved | **FAIL** on top-1 |
| int4 | 150 | 0.02978 (≤ 0.03) | 0.9400 | −0.0488 [−0.0775, +0.0044]: unresolved | **FAIL** on top-1 |
| int4-cuda | 150, 2.9 ms per prompt token (valid) | 0.03797 (≤ 0.03) | 0.9067 | −0.0377 [−0.0644, +0.0103]: unresolved | **FAIL** on KL and top-1 |

Every arm answered all 150 items by route `head`, with the reference's prompt token counts, so none is void. By kind,
the top-1 misses fall mostly on `score` items (0.857 / 0.898 / 0.837) and then `choice` (0.920 / 0.940 / 0.900);
`noul` stays at 0.98 to 1.00.

**What the rule says follows.** It names one consequence: "if int4 fails calibration but passes top-1, the default for
decision models is int8". That did not happen. int4 failed top-1, not calibration, and int8int8 failed top-1 too. No
arm is resolvably worse calibrated than the reference. So the registered outcome is: Route B is exact at f32, and no
quantized arm meets the top-1 bar. What that means for the shipped default is not decided by the rule.

## After the fact, not part of the verdict: where the top-1 misses are

For each item a quantized arm got "wrong", this is the reference's own margin between its top two answers.

| arm | flips | reference margin on the flipped items | flips with margin < 0.05 / < 0.10 / < 0.20 |
|---|---|---|---|
| int8int8 | 11 | min 0.002, median 0.037, max 0.100 | 8 / 10 / 11 |
| int4 | 9 | min 0.002, median 0.082, max 0.255 | 2 / 6 / 7 |
| int4-cuda | 14 | min 0.002, median 0.082, **max 0.917** | 4 / 9 / 11 |

The reference itself has 12 of its 150 items within 0.05, 26 within 0.10 and 37 within 0.20, and the bar allows 3 flips
in all. int8int8's misses are all near-ties. Most of int4's are, with two at margins up to 0.255.

**One miss is not a tie, and it is on the GPU path.** On `painting-geometry-v1:...:palette:x6_y7` (a 16-option
`choice` item, 435 prompt tokens) the reference puts 0.956 on option 7 and 0.040 on option 6. The three CPU arms
agree: f32 0.956, int8int8 0.980, int4 0.951 on option 7. int4-cuda gives 0.559 to option 6, 0.362 to option 7, and
spreads about 0.01 over the others. It is the same quantization as the CPU int4 arm with a different answer, so it
points at the CUDA `PromptHidden` path, not at int4.

## The decision this leaves for the owner

1. **The default precision for decision models.** f32 is the only arm that meets the bar. It is exact, but JEV-9B at
   f32 is about 36 GB, so it fits nobara-pc and not a 16 GB Mac. The alternative is to keep a quantized default with its
   measured disagreement stated (int8int8: 92.7% top-1 against the f32 reference, KL 0.009, calibration no worse).
2. **Whether a later run should use a margin-aware agreement measure**, registered before it runs. Under CLAUDE.md's
   rule a bar moves only with a mechanism. The mechanism offered here is the near-tie table above, and only for the
   CPU arms.
3. **The CUDA int4 discrepancy** wants its own investigation either way (the confident flip above, and KL 0.038 against
   CPU int4's 0.030), before decisions are served from a CUDA-resident model at int4.

## Grader output

```
reference: 150 items, 84 gold rows; reference ECE on them 0.1222
f32       PASS (KL -0.00000 vs 0.01, top-1 1.0000 vs 0.98); calibration unresolved (interval reaches 0)
int8int8  FAIL (KL 0.00884 vs 0.03, top-1 0.9267 vs 0.98); calibration unresolved (interval reaches 0)
int4      FAIL (KL 0.02978 vs 0.03, top-1 0.9400 vs 0.98); calibration unresolved (interval reaches 0)
int4-cuda FAIL (KL 0.03797 vs 0.03, top-1 0.9067 vs 0.98); calibration unresolved (interval reaches 0)
```

(Full per-kind output: run `python3 grade.py results/` here.)

## Owner decision, 2026-10-02: decision models load at `int8int8` by default

Decided by the owner from the table above. (The code already loaded decision models at int8int8 from `455ae2a3`, 2026-10-01; this records the decision, and what it does and does not claim.)

- **The bar is not moved.** int8int8 FAILS the registered top-1 bar (0.9267 against 0.98) and that stays the record. The default is a product choice with the disagreement stated, not a pass.
- **Why int8int8.** f32 is the only arm that meets the bar but is about 36 GB, which excludes a 16 GB Mac and every consumer GPU; int8int8 is about half that, is the fastest of the CPU arms (6.7 s median per item against 8.9 s at f32), reads mean KL 0.009 (inside the 0.03 band), no resolvably worse calibration than the reference, and all 11 of its top-1 flips are near-ties (reference margin 0.10 or less).
- **After the fact, not part of the verdict:** against GOLD on the 84 gold rows the quantized arms lose nothing the f32 reference gets right (`gold_accuracy.py`, `gold_accuracy.txt`): reference 65/84, int8int8 66/84 (+1, lost 0), int4 70/84 (+5, lost 0), int4-cuda 71/84 (+6, lost 0). The gains are all in the `score` kind, where the reference itself is 6/16: this is small-n noise around items the head finds hard, **not** evidence that quantization improves anything,
  and it is post hoc. It is the reason the disagreement with f32 is not read as lost accuracy; it is not a re-grading.
- **Not covered by the default, stated:** (1) the int8int8 arm graded here is the CPU one; **int8int8 on a CUDA or Metal resident is not graded** (Metal falls back to int4 for int8int8; CUDA's int8int8 resident was not run on these items). (2) The CUDA int4 path has one confident flip (a 0.956 option read as 0.362) that is not a tie and is not explained: do not rely on decisions from a CUDA-resident int4 model until it is.
  (3) A margin-aware or gold-accuracy re-grade, registered before it runs, is open; until one exists no document says the int8int8 default "meets" the bar.
- **f32 stays available** (`quant=` or `--quant f32`) as the exact option.

