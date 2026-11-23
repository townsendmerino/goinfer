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

## Phase 0 readings

Run 2026-11-23 09:37–09:40 PST at 299ddfdde (origin/main 317c0aaea plus this record), timing lock exclusive, after the
lock's previous holder released it. Logs: `~/goinfer-logs/attn-prefetch-2026-11/p0/` (the four `.ncu-rep` reports and
`run.log`). The per-launch tables are `p0/stalls-*.txt` (from `instruments/ncu_stalls.py`), and the stamps and
decomposition lines are `p0/run-summary.txt`. Every load delivered the KV type it asked for (`KVPrecisionReports`), and each
pass ran at the width it named. 1-minute load at phase starts 2.0–8.1: the first ncu pass started at 8.10, right after the
build, which the load rule above reports and does not void. GPU 461 MiB in use before each load, 48–62 °C.

**ncu, `attn_fused_bm128_hd128`, layers 0–3 of the pass** (the four launches agree within 0.6 points; the f16 row's `long_scoreboard` / `lg_throttle` split
alternates between about 10/7 and 13/5, while their sum holds):

| | f32, 512-row (decides) | f16, 512-row | f32, 2,048-row (served width) | f16, 2,048-row |
|---|---:|---:|---:|---:|
| grid | 48 | 48 | 180 | 180 |
| duration, ms | 1.134–1.153 | 0.934–0.944 | 2.711–2.753 | 2.207–2.258 |
| `long_scoreboard` + `lg_throttle`, % of stall cycles | **23.6** | **17.6–18.3** | **23.6–23.7** | **16.8–17.2** |
| of which `long_scoreboard` / `lg_throttle` | 6.6 / 17.0 | 10–13 / 5–8 | 7.0 / 16.6 | 10–13 / 4–7 |
| barrier, % | 2.8 | 4.0–4.3 | 2.6–2.7 | 3.9–4.1 |
| tensor pipe active, % of peak | 20.7 | 25.5 | 20.8 | 25.5 |
| global-load bytes per thread per instruction | 21.6 | 19.8 | 21.5 | 19.6 |
| global-load warp instructions | 370,560 | 207,744 | 1,266,720 | 715,296 |
| registers, warps active % | 224, 25.0 | 203, 25.0 | 224, 25.0 | 203, 25.0 |
| largest other stall reasons, % | wait 17.4, math pipe throttle 16.4, selected 12.7, branch resolving 8.8 | math pipe throttle 19.0, wait 18.3, selected 14.3, branch resolving 10.9 | as f32 512 | as f16 512 |

Bytes per load are the survey's formula (`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` over 32 × load instructions). That
counter is at sector granularity, so it reads above the 16 bytes a vector load asks for. The survey's 2026-10-23 read on the
same pass, before KS-B2, was 59.7% load stalls, 0.7% barrier, 8.8% tensor pipe, 4.03 bytes a load, 1,347,456 load
instructions, 2.75 ms a launch.

**Decomposition, `TestPrefillDecomp`'s loop, 2,048-row passes (the build's default width at both contexts):**

| | f32, K = 3,900 (decides) | f16, K = 3,900 | f32, K = 8,192 | f16, K = 8,192 |
|---|---:|---:|---:|---:|
| unprofiled wall | 659 ms | 647 ms | 1,535 ms | 1,465 ms |
| GEMV / attention / glue, ms | 464.1 / 104.4 / 61.6 | 465.0 / 88.5 / 61.5 | 986.7 / 405.5 / 128.9 | 987.7 / 338.7 / 127.9 |
| **attention, % of category sum** | **16.6** | **14.4** | **26.7** | **23.3** |
| attention, % of unprofiled wall | 15.8 | 13.7 | 26.4 | 23.1 |

The 3,900-token cell read the same, to 0.2 ms of attention, in its `-ctx 8192` load.

### Verdict: STOP

- **Attention is 16.6% of the 3,900-token prefill at f32, under the 20% bar, so the rule stops the item.** At f16 (the
  served default) it is 14.4%.
- **The load-stall share does not reach PROCEED either.** It is 23.6% at f32 (17.6–18.3% at f16), which is in the band that
  would have gone to the coordinator on its own. Both passes land in the same band at each KV type, so the served width
  does not change the reading.
- **KS-B2's register-prefetch leftover is closed** with these numbers. The other two leftovers (`ldmatrix.trans` for V^T,
  no mask arithmetic on interior tiles) were not in this item and are not closed by it.

**For scale, as arithmetic and not a measurement:** if a prefetch removed every load-stall cycle and none of it came back as
another stall, attention would shrink by at most about 24%. That is about 25 ms, or 3.8%, of the 659 ms prefill at f32 and
3,900 tokens. At 8,192 tokens the same bound is about 96 ms, or 6.3%, of 1,535 ms. Even the 3,900-token ceiling is below
the brief's 1.05× ship bar.

**What moved since the survey:** KS-B2's vector loads cut load instructions 3.6× and load stalls from 59.7% to 23.6%. KS-B8's
2,048-row passes and KS-B2 together took attention at 3,900 tokens from 340.9 ms (512-row passes, before KS-B2: KS-B8's own
read) to 104.4 ms, and the GEMVs, at 464 ms, are now 74% of the prefill. The kernel's stalls are now led by `wait` (fixed-latency
dependencies) and `math_pipe_throttle`, with the tensor pipe at 21–26% of peak. Those are issue-side and MMA-side costs, which
a load prefetch does not touch.
