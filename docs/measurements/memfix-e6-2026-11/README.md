# E6: the host guard's CUDA expert-cache build term, re-read (memory audit F7, 2026-11-18)

One box: nobara, 16 cores, 62 GB RAM, RTX 2070 SUPER 8 GB, driver 595.112.07, desktop session up. The server is `main` at
`cb6ad797f` built with `go build -tags cuda ./backend/cuda/cmd/serve` (CGO_ENABLED=0). The checkpoints are
`~/models/gemma4-26b-q4_k_m.gguf` and `~/models/gpt-oss-20b-MXFP4.gguf`, on NVMe. Each was served with `-backend cuda -quant int4
-moe-cache-experts -direct-load`: a direct `.gguf` load with a C' expert cache, the case the term prices. The run held the exclusive
timing lock from 14:11 to 14:12 PST (load average 3.8 at the start, 58.7 GB available, system swap 1.17 GB in use and unchanged
by the run). Tier 1, one load each. This is a memory read, not a timing.

## The question

`decoder/fitguard.go:fitCheck` prices a C' cache build from a `.gguf` at `BuildDemand.HostBuildPeak`, and that was
`2 × weights + experts`: the canonical weights, a packed copy of them and a pinned copy of the experts. The audit (F7) read a 26B
test process at `VmHWM` 30,160 MiB against a price of 38.4 GB. The comment on `packBytes` in the same struct already said
`BuildResident` no longer holds a packed copy of the model (`docs/measurements/cuda-build-heap-2026-10-18/README.md`).

## Readings (`e6_cell.py`; `/proc/<pid>/status` every 0.2 s until the server has answered one request)

| | the guard prices (its own message) | `VmHWM` | peak `VmSwap` | peak RssAnon / RssFile / RssShmem |
|---|---|---|---|---|
| Gemma-4-26B Q4_K_M | 38.4 GB = 13.2 weights + 13.2 packed copy + 12.0 pinned experts | **29.09 GiB** | 0 | 13.59 / 15.49 / 12.06 GiB |
| gpt-oss-20b MXFP4 | 32.0 GB = 11.0 + 11.0 + 10.0 | **22.38 GiB** | 0 | 11.22 / 11.22 / 10.10 GiB |

The guard's GB is 2^30. Both loads were ready in about 23 s and answered (`serve-*.log`, `e6-*.json`). The first attempt
(`run-void-404.log`) is void: the request named the wrong model, and the script stopped before writing a record.

- **No packed copy.** The anonymous peak is the weights (13.59 GiB against 13.2, the rest Go heap), never twice them.
- **The pinned copy is there.** RssShmem is the experts (12.06 against 12.0; 10.10 against 10.0).
- **The peak is the load, not the build.** Weights plus the mapped source file are 13.2 + 15.6 = 28.8 GiB (26B) and 11.0 + 11.3 =
  22.3 (gpt-oss), against 29.09 and 22.38 read. The guard already prices that as weights + `srcFileBytes`. The phases do not overlap
  (the component peaks sum to 41 GiB, the high-water mark is 29), as the term's comment says.
- **The build term over-prices by 9.3 GiB (26B) and 9.6 GiB (gpt-oss)**: one copy of the weights.

## The correction (this PR)

`HostBuildPeak` is now the weights, the experts' pinned copy (not with `-moe-experts-unpinned`, unchanged) and the packing: one packed
layer, once the config gives the layer count, and the staging blocks (`packStageHostBytes`), the figure `HostPackBytes` already
prices. The guard asks again after it reads the config, so the packed layer is in the figure it uses. Both copies of the formula
(`backend/cuda/builddemand.go:buildDemand` and `decoder/builddemand.go:defaultCUDABuildDemand`) move together.

### The margin, in one unit (GiB = 2^30, the guard's own "GB"; exact bytes from `decoder.FitQuery` and `/proc`)

| | need before | need after (the max of the guard's terms) | measured `VmHWM` | margin before | **margin after** |
|---|---|---|---|---|---|
| Gemma-4-26B | 38.406 | 28.864 (weights 13.221 + source 15.643) | 29.094 | +9.312 | **−0.231 (0.79% under the peak)** |
| gpt-oss-20b | 31.969 | 22.257 (weights 10.979 + source 11.278) | 22.383 | +9.586 | **−0.126 (0.56% under)** |

After the correction the build term (weights + pinned experts + one packed layer + staging: 25.7 GiB on the 26B) is below the weights
+ source term, which binds. **The priced need is then below the measured high-water mark, by 0.231 and 0.126 GiB.** The gap is
anonymous memory beyond the priced weights: RssAnon peaked at 13.589 GiB against 13.221 priced (+0.368, 2.8% of the weights) and 11.223
against 10.979 (+0.244, 2.2%), while the mapped source's resident pages read slightly under the file (15.485 against 15.643; 11.223
against 11.278). That is Go heap and runtime slack, which this guard has never priced, and it is smaller than the "about 9%" the old
comment carried, which was measured when the build still held a packed copy. The shortfall belongs to the weights + source term,
which priced these loads the same way before; the over-priced build term hid it. The guard compares the need against 70% of available
memory, so these loads still sit 30% inside what is free, but the need itself is not an upper bound on the process.

**This loosens what the host guard admits** by about 9.5 GB on these two loads (the 26B's need goes 38.4 → 28.9 GB, gpt-oss's
32.0 → 22.3). No threshold, fraction or margin moves; it is a priced term corrected to what the process holds. A host that
refused these loads before may now admit them. Whether to take it is Francis's call.

## Not read

A cold page cache (the source pages were cached, so the file term's pages were touched quickly; `VmHWM` counts them either way),
a `-moe-experts-unpinned` load, a 35B-class MoE, and the Mac.
