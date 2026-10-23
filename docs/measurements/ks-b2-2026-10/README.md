# KS-B2 build: vector-load staging in the 128-row fused prefill attention, 2026-10-23

The build for `docs/tasks/task-kernel-survey-2026-10.md` KS-B2. Its first measurement
(`docs/measurements/kernel-survey-2026-10/README.md` § KS-B2) read `attn_fused_bm128_hd128` on the 1.5B at K=3,900:
- warps stalled on global loads for 59.7% of their stall cycles (`long_scoreboard` + `lg_throttle`);
- the tensor pipe active 8.8% of the time;
- every staging load a scalar 32-bit word (4.03 bytes a load).

**The change** (`cuda/attn_fused_bm.cu`, both PTX twins regenerated):
- A full key tile (`nk == BN`) whose K and V caches are 16-byte aligned stages through `stage_tile_vec`, with 16-byte
  loads (4 f32 or 8 f16 elements). Each batch of up to 8 K loads, or 4 V load pairs, is issued before any of them is
  converted or stored.
- K lanes run along a key row. V lanes run along key pairs, storing `half2(key, key+1)` per column, so a warp's V^T stores
  land on distinct banks (the scalar store was 4-way conflicted).
- Partial tiles (the last of a block, a window's first) and misaligned caches keep the scalar loops.
- The values written to shared memory are the same halves, from the same f32→f16 rounding, so the output should be
  bit-identical.
- Scope: the six `attn_fused_bm*` entry points. `attn_fused.cu` (head dim 64 and windowed layers) is unchanged.
- Today's source regenerates the committed PTX byte for byte (NVRTC 12.9.86), so the PTX diff is this change alone.

Machine: nobara-pc, RTX 2070 SUPER 8 GB (Turing sm_75, 40 SMs), Ryzen 7 3700X, NVIDIA driver 595.91.07, Go 1.27.3.
Checkpoints from `~/models` on NVMe.

## Pre-registration

Registered 2026-10-23 before any GPU run. Instrument: `docs/measurements/ks-b2-2026-10/instruments/ks_b2_vecload_test.go.txt`,
copied into `cuda/` as `ks_b2_scratch_test.go` for the run and removed afterwards.
- **Arms:** the old arm is origin/main's committed `attn_fused_bm.ptx` / `attn_fused_bm_kv16.ptx`, loaded beside the new
  one on the same resident.
- **Loads:** Qwen2.5-Coder-1.5B Q4_K_M as `serve -backend cuda -quant int4` loads it, once with `-kv f32` and once with
  `-kv f16` (serve's default `-kv auto` loads it at f32 today). One process each.
- **Lock:** timed runs hold the timing lock (`timing_lock.py run --label cuda-builds-ks-b2`).

- **G1a, kernel bit-identity (tier 0, on device):** every one of the six entry points, old against new, over 1,080 shapes a
  KV type:
  - head dim 64 and 128 (the entry point's own);
  - M 1, 17, 64, 129, 517;
  - start 0, 5, 384, 1,024;
  - window 0, 100, 1,000;
  - (nH, nKV) (4, 4), (12, 2), and (8, 2) with sinks.

  `Float32bits` equality of every output element. One differing shape stops the item.
- **G1b, whole-prefill bit-identity (tier 1):** a 3,900-token `PrefillLast` on the 1.5B with the old and the new
  `attn_fused_bm128_hd128` swapped in on the same resident: the last row's logits byte-equal, at both KV types.
- **Also before merge:**
  - `TestAttnFusedTile_32x64BitIdentical`: the 456-shape sweep against the unchanged `attn_fused.cu`, both arms.
  - `TestKernelLocalMemoryCensus`.
  - The CUDA device suite.
- **G2, speed (tier 1, in process):** 10 ABBA quads of the 3,900-token prefill, old arm A against new arm B. Readout: the
  prefill speedup per quad, (A1 + A2) / (B1 + B2), its median, and how many quads are above 1.
  - **f32 decides:**
    - **ship:** at least 9 of 10 quads above 1 (CLAUDE.md's small-gains rule: bit-identical, 9 of 10 pairs, no memory
      increase);
    - **parked, to the owner:** 6–8 of 10;
    - **killed:** 5 or fewer.
  - **f16 is reported beside it.** If its median is below 1.00, the item parks for the owner even if f32 ships.
  - The survey's estimate, for scale: 2.5–4× on attention, 1.27–1.36× on the whole 3,900-token prefill.
- **Memory:** shared memory a block is unchanged (same tiles, same `SharedMemBytes`); no device buffer is added. Recorded:
  registers and local bytes per entry point, old and new. A new entry point with local memory above 0 fails the census
  test, and so fails the item. A register count that drops the 128-row kernel below one block an SM is not possible (it is
  already one block an SM at 180).
- **Recorded, no bar:**
  - attention-only time of `attn_fused_bm128_hd128` at the deepest full pass (M 512, start 3,072, 12 heads, 2 KV heads),
    old and new interleaved;
  - after the gate, the survey's ncu read (`ks.sh` phase B2) on the new build: the `long_scoreboard` + `lg_throttle`
    share and bytes a load.
- **Load rule:** every quad records the 1-minute load and `MemAvailable` at its start. A quad started above 8.0 is void and
  re-run in order, and the record lists it.
- **Split, amended 2026-10-23 13:05 before any reading:** G1a and G1b may run alone (`GOINFER_KS_B2_G1_ONLY=1`), holding the
  lock shared, since they time nothing. G2 runs afterwards with the lock exclusive, in the same instrument. Bars unchanged.
- **Stopping rule:** one run a KV type. **Cost:**
  - two 1.5B loads (~15 s each);
  - G1a, 2,160 small launches with uploads (under a minute a KV type);
  - G1b and G2, 42 prefills of ~0.9 s a KV type;
  - about 4 minutes in all, plus the ncu read (~3 minutes).

## Readings

At 4171ac2e8, the new arm (the old arm being origin/main's committed PTX), 2026-10-23. Logs: `~/goinfer-logs/ks-b2-2026-10/`.

**G1, run 13:06–13:09 PDT, holding the lock shared** (kv4b-grade's CR7B CPU reference held it shared too; G1 times nothing):

| | f32 KV | f16 KV | verdict |
|---|---|---|---|
| G1a, six entry points × 180 shapes | 1,080 shapes, **0 differ** | 1,080 shapes, **0 differ** | **PASS** |
| G1b, 3,900-token prefill, last-row logits | **0 of 151,936 differ** | **0 of 151,936 differ** | **PASS** |

The first f32 G1 attempt stopped at shape `bm128_hd128 M129 sp384` with `CUDA_ERROR_OUT_OF_MEMORY`. The instrument allocated
every shape's buffers and freed none. It now releases them per shape, and the f32 pass above is the re-run of the same sweep.
The f16 pass, smaller, ran clean the first time.

**Registers and local memory** (f32 twin; JIT on driver 595.91.07):

| entry point | old | new | local bytes, old / new |
|---|---:|---:|---|
| `attn_fused_bm32_hd64` | 134 | 142 | 0 / 0 |
| `attn_fused_bm32_hd128` | 180 | 186 | 0 / 0 |
| `attn_fused_bm32n32_hd64` | 140 | 174 | 0 / 0 |
| `attn_fused_bm32n32_hd128` | 164 | 168 | 0 / 0 |
| `attn_fused_bm128_hd64` | 134 | 165 | 0 / 0 |
| `attn_fused_bm128_hd128` | 180 | 224 | 0 / 0 |

Occupancy is unchanged. The 128-row kernels run 256 threads, and 256 × 224 = 57,344 of the SM's 65,536 registers, still one
block an SM as at 180. The 32-row kernels are bounded by shared memory (one block of about 35 KB at head dim 128, three of
about 18 KB at 64) before registers.

**G2, run 13:57–13:59 PDT, lock exclusive** (the first attempt, 13:50, could not load the 1.5B: another session's 26B test
held 6,970 MiB of the card outside the lock; nothing was timed, and the run waited for that process to exit):

| | f32 KV (decides) | f16 KV |
|---|---|---|
| prefill 3,900 tokens, old → new | 937.1 → 741.9 ms | 920.2 → 721.6 ms |
| new speedup, median (min–max) | **1.2632×** (1.2608–1.2658) | **1.2748×** (1.2742–1.2785) |
| quads above 1 | **10 of 10**, 0 void | **10 of 10**, 0 void |
| attention only, (M 512, start 3,072), old → new | 2.664 / 2.716 → 1.099 / 1.104 ms (**2.44×**) | 2.583 / 2.577 → 0.928 / 0.922 ms (**2.79×**) |
| load1 at quad starts | 4.97–6.94 | 2.62–3.68 |

**Verdict: SHIP** at both KV types (f32 decides; f16 above 1.00). G1 held again inside both G2 runs (1,080 shapes, and the
3,900-token prefill byte-equal). The whole-prefill gain matches the survey's 1.27–1.36× estimate at its low end, and attention's
2.4–2.8× sits inside the 2.5–4× it estimated, a little under at f32.

**CUDA device suite** (`scripts/cuda_device_suite.py`, the whole `cuda` module), 13:50–14:01 PDT at 834c37ec5, under the timing
lock: **GREEN, 769 passed, 0 failed, 0 device skips, 236 other skips**. Among them: `TestKernelLocalMemoryCensus`,
`TestAttnKernelDigests` (the attention output digests are unchanged), and `TestKV16_exactInputIdentity_prefillReaders` /
`_decodeReaders` and `TestKV16_everyAttentionPathStaysCloseToF32`. Log: `~/goinfer-logs/ks-b2-2026-10/device-suite/`.
`TestAttnFusedTile_32x64BitIdentical` reads PASS there but **did not run**: both of its arms skip without
`GOINFER_HEAVY_TESTS=1`, and the parent passes over two skips. It runs on its own with the flag (below).
