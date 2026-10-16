# OP1 on CUDA: the day gates (nobara-pc, RTX 2070 SUPER, driver 595.91.07, 2026-10-16)

Registered in `docs/tasks/task-forward-architecture-2026-10.md` §4.6a. Logs are in `~/goinfer-bench/op1-cuda-2026-10/` on nobara.

## Step 1: the ten score sites through `cuda/attnscore.cuh` (commit 607b9daa; base 78e4faf2)

| gate | result |
|---|---|
| PTX | all 15 attention modules rebuilt from the edited sources are byte for byte the committed PTX. Control first: each module rebuilt unchanged at its own toolchain (`glue`, `glue_kv16`, `mla` at 12.6.85 from a pinned venv; the rest at the ambient 12.9.86) and `git status` showed nothing |
| G1-K | PASS: 1,864 of 1,864 digests equal base's (recorded three times at base, two processes, identical) |
| G1-K planted defect | a one-ulp change in `attn_score` and `attn_score_rn`, all 15 modules rebuilt: **41 of 41** kernel families red (`g1k-plant-ulp.log`). Not every case goes red: cases with one key (softmax of one element is 1.0 at any score) and a few short ones stay equal, which is why the test drives key counts either side of the tile bounds |
| G1-M, tiny | PASS: 27 families IDENTICAL, 0 different, 16 NOT RUN (the backend declines them or their asset is not in the tree), 3m47s, `g1m-tiny-step1.log` |
| G1-S, by name, `GOINFER_HEAVY_TESTS=1` | PASS: 34 tests by name; base 31 PASS 3 SKIP, step 1 the identical list. The three skips (`TestFlashDecodeOracleRealKV`, `TestFlashDecodeOracleRealKVGemma`, `TestFlashDecodeGateVsReference`) skip at base too. Without the heavy flag the same run is 11 PASS 23 SKIP, which proves nothing: that is why the heavy run is the gate |

The first G1-S run without the heavy flag was kept and not graded (`g1s-step1.log`): 23 of its 34 tests skipped.
