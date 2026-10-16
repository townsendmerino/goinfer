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

## Step 2: the softcap as a compile-time variant (commit 2cafcbae), and its declaration (branch `op1-cuda-declare-2026-10`)

| gate | result |
|---|---|
| G2-I, PTX | the 15 plain PTX files are byte for byte step 1's (`git diff` shows only the six new `_cap` / `_kv16_cap` files added) |
| G2-I, G1-K | PASS: 1,864 of 1,864 digests equal base's (`g2k-g1k-step2.log`) |
| G2-I, G1-M tiny | PASS: 27 IDENTICAL, 0 different, 16 NOT RUN, base against step 2 (`g2i-tiny-step2.log`) |
| G2-K | PASS: 320 of 320 kernel cases inside the bar (cosine >= 0.9999, max abs <= 1e-2) against float64: `attention`, `attn_batched` (each row) and the split-KV chain, f32 and f16 KV, caps 50 and 2, head dimensions 64 and 256, windowed or not, sink or not, with a peaked case. Each case asserts from the reference that scores lie past ±cap (at least 5% of them, and the largest at 1.5 caps or more) before it runs (`g2k-capped.log`) |
| G2-D | PASS: with the cap's application removed from the variant source (the argument read and ignored) and the six variants rebuilt, all 320 cases go red: `attention` 64 of 64, split-KV 128 of 128, `attn_batched` 128 of 128 cases (1,264 rows) (`g2d-planted.log`) |
| G2-L | PASS (`cuda/gemma2_softcap_test.go:TestGemma2Softcap_lanesCUDA`). The tiny fixtures have head dimension 32, which flash-decode and the fused kernels never serve, so each selector is asked at head dimension 64 with the resident's state set up as a served prefill would: cap off, flash-decode and fused-prefill say yes (the control); cap on, both say no. The image-block and drafter refusals name the cap. Planted: the flash-decode selector ignoring the cap, and the fused selector ignoring it, each turn the test red |
| G2-T, diagnostic | worst logit cosine over 16 positions, resident against the CPU, cap on both sides / cap removed on both sides: `gemma2-tiny` int4 0.999650 / 0.999827, int8int8 1.000000 / 1.000000; `gemma2-hd-tiny` int4 0.999756 / 1.000000, int8int8 1.000000 / 1.000000. The batched prefill (`attn_batched` with the cap) and the token-by-token decode write the same K and V rows and the same last-token logits bit for bit on both fixtures. The 0.999 bar is not applied (registered: a diagnostic) |
| G2-R, model-level | the real `gemma-2-2b-it` Q8_0 (`~/models`), one process, `g2r-cuda.log`, 82 s. Hook check: removing the cap changes the CPU's logits. B (cap removed both sides): int8int8 0.982058 (23/24), int4 0.980240 (22/24). Control (dense 1.5B, int8int8): 0.997803 (23/24). Gemma 3 4B: int8int8 0.975099, int4 0.984910 (23/24 both). **G (cap on both sides): int8int8 0.992910 (23/24), int4 0.980619 (21/24): pass, inside the 0.03 band of Gemma 3's at the same quant.** P (cap dropped from the CUDA build only): int8int8 0.992485 (21/24), int4 0.982282 (23/24): **passes the rule**, as on Metal. The cap on the CPU over P's run: 62,400 scores, 0.0032% / 0.0064% past 15, largest |score| 15.72 / 15.91, the same figures Metal recorded. So G2-R is a model-level check and not a softcap test, as registered; the declaration rests on G2-K, G2-D and G2-L |
| G2-M | the matrix regenerated: Gemma 2 reads resident on CUDA (`docs/hardware-matrix.md`) |
| fit check (by day, before any queueing) | `TestG2R_fitCheck`: the five CUDA loads (2B at int8int8 and int4, Gemma 3 4B at both, the dense 1.5B at int8int8) are all admitted, 1.5 to 10.9 s each (`g2r-fitcheck.log`) |

Also: `gate ptx` (check 4) did not rebuild the six `_cap` files, because it derived a source only for the `_kv16` suffix; it counted 42 of 48
committed artifacts. Fixed in `f5098c12` and the test's count. The heavy-group table gained a `gemma2-2b` row.

**Not run by day, queued for the night after the merge:** G1-M and G2-I on real checkpoints (`gate identity -backend cuda -assets real`), the
device gate, G2-S (speed, no model got slower), G2-P (Gemma 2 against our CPU and Ollama).

## The device gate (`gate gpu`), 2026-10-16

**First run, `b04b0273`, night job `op1-gate-gpu-cuda`, 10:36 to 12:52 PDT: FAIL, with three causes, none in OP1's code.** Passed: PTX 48/48
byte-identical at their recorded NVRTC (the six new `_cap` twins included), graphs replayed against live launches, the drain group, the resident
parity gates, every heavy group but two, hygiene. Failed:
1. **Check 2a** did not build: `cuda/kv2_optionoff_test.go` used helpers defined under `goinfer_testhooks` with only `cuda` as its tag, which CI
   never builds. Fixed by #66 (the file's tag, and a CI step that vets the cuda tests with `-tags cuda` alone).
2. **Ten heavy-tier tests** (the 26B and 35B groups and two fresh-process tests) read v16 `.giw` sidecars the v17 build refuses
   ("format v16, this build reads v17"). The v16 sidecars were deleted and the two rebuilt at v17 (26B in 80 s, 35B in 153 s,
   `cmd/prequant -quant int4 -target canonical`, sources named by `docs/measurements/giw-v16-transcode-2026-10/README.md`).
3. The other sidecar-dependent tests ride on 2.

**Re-run of the failed parts, night job `op1-gate-rerun-cuda` at `c3a81a9f` (main with #58, #63 and #66), 13:42 to 14:08 PDT:**

| part | result |
|---|---|
| check 2a, `-tags cuda` alone, `-short` | PASS |
| `TestP20ExpertLocality`, fresh process | PASS |
| `TestSpecPagerInteraction`, fresh process | PASS |
| heavy group `gemma4-26b` | PASS, 0 skips |
| heavy group `qwen3.6-35b` | **VOID** (below) |

**The 35B group is void, not failed.** Its `TestQwen36_35B_cache` died at about 14:05 to 14:08 with `CUDA_ERROR_OUT_OF_MEMORY` allocating 508 MB
during the resident build, after which the 35B ran on the CPU. The owner's reading, recorded here as his and not as measured by this run: the Mac's
`tack` session loaded a 7B server (6.3 GB of VRAM) on nobara from 14:08:03, during the test's last seconds, and that was likely also the 35B's first
load after the sidecar rebuild (a 48 GB peak into deep swap). A server of 5,958 MiB was seen holding the card at 14:27. The rule for a timed or
GPU-exclusive run is one holder at a time (`timing_lock.py`); this window had two. The device gate is **not recorded green** until
`TestQwen36_35B_cache` passes alone, in a fresh process, with the v17 sidecar already built.
