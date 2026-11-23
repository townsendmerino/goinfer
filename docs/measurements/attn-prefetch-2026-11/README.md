# Attention prefetch: overlapping the next K/V tile's loads with the current tile's math, 2026-11

The question: does `attn_fused_bm128_hd128` (`backend/cuda/attn_fused_bm.cu`) still wait on its staging after KS-B2
(`docs/measurements/ks-b2-2026-10/README.md`), and is attention a large enough share of a long prefill for overlapping the
next tile's loads with the current tile's MMAs to matter end to end? KS-B2 built 16-byte staging and left the survey's other
three proposals unbuilt; this item is the first of them, register prefetch of the next tile. The loop still stages a tile,
waits at a barrier, computes and waits again, and at one 256-thread block an SM no other block hides the staging. Turing has
no `cp.async`, so dedicated loader warps are out of scope.

Machine: nobara-pc, RTX 2070 SUPER 8 GB (Turing sm_75, 40 SMs), Ryzen 7 3700X, NVIDIA driver 595.91.07. Checkpoint:
`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` on NVMe, loaded as `serve -backend cuda -quant int4` loads it. ncu is
Nsight Compute 2026.1.1.

## Phase 0 pre-registration (measure only, no product code)

Registered 2026-11-23, before any run, at the tree's `origin/main` (317c0aaea).

**Instrument:** `instruments/ap0_test.go.txt`, copied into `backend/cuda/` as `ap0_scratch_test.go` for the build and removed
afterwards (never committed there), run by `instruments/ap0.sh` under the timing lock, exclusive. It holds:
- `TestKS_B2_prefillUnderNcu`, the kernel survey's instrument
  (`docs/measurements/kernel-survey-2026-10/instruments/ks_survey_test.go.txt`) with one change: the load takes serve's `-kv`
  flag from `GOINFER_AP_KV`, so both twins are read. The survey's read was the f32 twin (the test binary pins `-kv auto` to
  f32); serve's `auto` on CUDA gives this family f16, so the f16 twin is the served default today.
- `TestAP_prefillDecompKV`, `TestPrefillDecomp`'s loop (warm pass, then the best of 3 profiled passes by category sum) on the
  same load and `-kv` flag, with `-ctx 8192` for the 8,192-token depth.

**Reads:**
1. ncu, the survey's pass and sections unchanged: one 3,900-token `PrefillLast`, `-k regex:attn_fused_bm128_hd128
   --launch-skip 168 -c 4` (layers 0–3 of the seventh 512-row pass, rows 3,072–3,583, grid 48), sections WarpStateStats,
   SchedulerStats, SpeedOfLight, Occupancy, LaunchStats, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, plus
   `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`, `smsp__sass_inst_executed_op_global_ld.sum` and
   `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`. Reported: `long_scoreboard` + `lg_throttle` share of stall cycles, the
   barrier share, tensor pipe active %, bytes per thread per global load, duration. f32 and f16.
2. The decomposition at K = 3,900 (default context) and at 8,192 (`-ctx 8192`), f32 and f16: attention's share of the
   profiled category sum, and beside it attention over the unprofiled wall.

**Amended 2026-11-23 09:02 PST, before any run** (the first launch was stopped while it waited for the lock; nothing had
built or run): KS-B8 made 2,048-row passes the default where scratch allows, so the survey's `--launch-skip 168` (the
seventh 512-row pass) names a pass that no longer exists at the default width. The instrument now takes the pass width
from `GOINFER_AP_CHUNK` (the `GOINFER_PREFILL_CHUNK` knob on the loaded model), and ncu reads two passes a KV type:
- **the survey's pass:** 512-row passes, `--launch-skip 168 -c 4` (rows 3,072–3,583, grid 48), as on 2026-10-23;
- **the served width:** 2,048-row passes, `--launch-skip 28 -c 4` (layers 0–3 of the second pass, rows 2,048–3,899, grid
  12 × 15 = 180).

The decision reads the survey's pass (the brief's instrument); the served pass is reported beside it, and if the two land
in different bands the readings go to the coordinator. The decomposition runs at the build's default width, which it logs.
Cost: four ncu passes, about 18 minutes in all.

**Decision rule** (the brief's, read on the f32 twin at K = 3,900, the instrument the survey and KS-B2 read; f16 reported
beside it, and a disagreement between the twins is reported):
- **PROCEED** to Phase 1 if load stalls (`long_scoreboard` + `lg_throttle`) ≥ 40% of stall cycles **and** attention ≥ 20%
  of the 3,900-token prefill.
- **STOP** if load stalls ≤ 15%, **or** attention < 20%: recorded here, and KS-B2's prefetch leftover is marked closed with
  the numbers.
- **Anything between** (load stalls in 15–40% with attention ≥ 20%): stop and send the readings to the coordinator.

**Tier and cost:** tier 0 kernel read plus a tier 1 decomposition; one run each, no sampling. A build (~2 min), two ncu
passes (~3 min each), four 1.5B loads (~15 s each) with 4 prefills each: about 12 minutes.

**Load rule:** every phase stamps the 1-minute load, `MemAvailable`, GPU memory used, temperature and SM clock at its start.
ncu's counters are replayed inside the kernel and the lock keeps other locked GPU jobs off the card; the decomposition is a
share, not a speed, so a start above 8.0 is reported, not voided.
