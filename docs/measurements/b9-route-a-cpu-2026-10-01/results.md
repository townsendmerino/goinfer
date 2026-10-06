# B9 split: how much of D6a arm B's divergence is the CUDA path (graded 2026-10-06)

The two arms ran on nobara-pc on 2026-10-01 22:50 to 2026-10-02 02:29 PDT (`run-b9.sh`; log `~/goinfer-logs/b9-route-a-cpu-2026-10-01/run.log`):
goinfer Route A, bare-v1, raw, `--ctx 4096`, `Qwen3.5-9B-Q4_K_M.gguf`, CPU at int4 (100 rows in 1h23m, 0 failed validation) and CPU at
int8int8 (100 rows in 2h12m, 0 failed validation), binary `goinfer-chat-cpu-4fa285ab`. They sat ungraded for five days; `grade.py` was
written and its bands registered before either arm ran, and it was run unchanged on 2026-10-06 against the arms' own output
(`~/goinfer-bench/b9-split-2026-10-01/results/`). Its output, verbatim:

```
reference top-1 0.540, arm B (CUDA int4) top-1 0.380, arm B agreement with the reference 0.660

## cpu-int4: 100 rows
  agreement with the reference 0.650   s = -0.03   top-1 against gold 0.430   mean P(true) shift +0.076   top-1 differs from arm B on 29 of 100
  READING (pre-registered): the CUDA path is a minor part

## cpu-int8int8: 100 rows
  agreement with the reference 0.740   s = +0.24   top-1 against gold 0.480   mean P(true) shift +0.026   top-1 differs from arm B on 28 of 100
```

## Reading

**By the pre-registered rule: the CUDA path is a minor part of arm B's divergence from the f32 reference** (CPU int4 s = -0.03, against the
band s <= 0.25). On top-1 agreement with the reference the CPU int4 arm is where the CUDA int4 arm is, 0.650 against 0.660: moving to the
CPU did not recover agreement. The int8int8 arm's s = +0.24 is also inside the minor band, so what remains is mostly the Q4_K_M file
and the int4 weights, not the route to the GPU.

## What this does not settle

- **n = 100.** One row is 0.01 of agreement, so 0.650 against 0.660 is one row, and s = +0.24 is 8 rows. The pre-registered rule was
  applied as written; the sample cannot resolve differences much below 0.1.
- **It is a top-1 reading.** B9 was filed on a KL gap (0.038 CUDA int4 against 0.030 CPU int4) and one confident flip (a 16-option `choice`
  item, CUDA int4 putting 0.559 on option 6 where the reference and all three CPU arms put 0.95 to 0.98 on option 7). Neither is tested here.
  The per-layer CUDA-against-CPU-int4 difference on that item is still the only thing that would explain the flip; this result says
  it is not where the aggregate gap comes from, which lowers its priority and does not close it.
- Arm B (D6a) ran on a different date and binary than these two arms, so "differs from arm B on 29 of 100" mixes the route with the build.
