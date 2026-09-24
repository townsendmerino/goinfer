# Linux CPU decode: `.giw` sidecar vs direct `.gguf` load (2026-09-24)

**Question.** darwin loads a `.gguf` through its sidecar `.giw` by default (S1, `task-never-swap-2026-09.md`); linux loads
it directly. On linux the resident weights would then be zero-copy aliases of a page-cache mapping instead of fresh heap
copies. CPU decode here is bandwidth-bound (17–23 GB/s against a ~30 GB/s ceiling,
[`cpu-decode-roofline-2026-09-23.md`](cpu-decode-roofline-2026-09-23.md)), and anonymous heap can be backed by transparent
huge pages (this box: THP `always`, `/home` is btrfs) where a file mapping may not be. Does aliasing cost decode speed?
Nobody has measured it: S1's darwin work checked output identity, not speed.

## Pre-registration (written and committed BEFORE any measurement)

**Arms.** Qwen2.5-Coder-1.5B q4_k_m and Qwen2.5-7B-Instruct q4_k_m from `~/models`, CPU backend, quant int4.
- `direct` — `decoder.Load` on the `.gguf` (today's linux default).
- `giw` — `decoder.Load` on `<base>.int4.cpu-amd64.giw` beside it, transcoded by `cmd/prequant` at HEAD (weights
  format v12, the aligned-scales one) with `-target cpu`, i.e. the file `-stream-weights`/S1 would create and reuse.
- `direct'` — a SECOND direct load of the same `.gguf`: the do-nothing arm. `direct` vs `direct'` is the noise floor.

**Design.** One process loads all three (`TestCPURoofline_giwVsDirect`, `decoder/cpu_roofline_ab_test.go`). Same
prompt (128 tokens), 24 greedy decode tokens per generation, forward ms/token from DECODE TIMING. Paired ABBA, 7 pairs
per comparison, the arm order alternating per pair; statistic = median of per-pair ratios `giw_ms / direct_ms` (> 1 means
the sidecar is slower). Run alone on an idle box (load < 1.0), after the running GPU work finishes.

**Hard gate.** The greedy token stream must be identical across all three arms on every generation (the same quantized
weights are read either way). A difference is a defect to report, and the speed result is not read.

**Decision rule, per model.**
- **SLOWER** — median ratio ≥ 1.02 AND the ratio's min over pairs > the A/A arm's max: `.giw` costs decode speed on
  linux. Then the sidecar must NOT become the linux default; the only use left is as a fallback when the fit guard would
  refuse a direct load (where the alternative is not running at all).
- **NO COST** — median within 0.98–1.02 on BOTH models: a default flip is not blocked by speed. It is still blocked by
  disk (a sidecar per model; `/home` is at 100%), which this measurement does not address.
- **FASTER** — median ≤ 0.98 AND the ratio's max over pairs < the A/A arm's min: reported, and a mechanism is needed
  before it is believed.
- Anything else (a median inside 0.98–1.02 on one model and outside on the other, or an effect that does not clear
  the A/A arm) → **AMBIGUOUS**, parked with the numbers; no default change either way.

**Reported, not decided on.** Load wall time per arm; Go heap in use (`runtime.MemStats.HeapInuse`) and `RssAnon` /
`RssFile` from `/proc/self/status`, sampled after each load; `.giw` size on disk; transcode time. Note that one process
holding all three arms makes the RSS deltas the useful quantity, not the absolutes.

**Not measured.** CUDA (weights end up in VRAM either way); a cold page cache (the warm-up generation runs first); any
model above 7B; darwin.

## Results

Run 2026-09-24 08:52–09:02 on `nobara-pc` (Ryzen 7 3700X), idle-gated (load < 0.6), nothing else running; harness at
`3792b6e9` plus a harness fix (a `.giw` carries its own tokenizer, so the `.giw` arm now decodes the direct arm's token ids —
which also guarantees identical prompts). A first launch failed at that tokenizer load before timing anything; its load lines
are the same as below. `.giw` files: v12 weights / v3 bundle, 1.29 GB and 5.18 GB, transcoded in 20.9 s and 102.1 s. Log:
[`cpu-giw-vs-direct-2026-09-24.log`](cpu-giw-vs-direct-2026-09-24.log).

**Hard gate: PASS** — every generation's greedy stream was identical across `direct`, `direct'` and `giw`, on both models.

| model | `giw` / `direct` median (7 pairs) | min–max | A/A `direct'` / `direct` median | A/A min–max | verdict |
|---|---|---|---|---|---|
| Qwen2.5-Coder 1.5B | **1.0007** | 0.9996–1.0051 | 0.9995 | 0.9960–1.0073 | NO COST |
| Qwen2.5 7B | **1.0016** | 0.9995–1.0048 | 1.0035 | 0.9879–1.0057 | NO COST |

**Decision (pre-registered rule): NO COST on both models** — decode speed does not block a `.giw`-sidecar default on linux.
The effect, if any, is inside the do-nothing arm's own spread. The page-cache-mapping vs THP-heap mechanism that motivated
the question did not show up at this resolution. Absolute ms/token here (≈51–54 / ≈265) is the in-process harness's
forward time with three copies of the model resident, as in the 2026-09-23 roofline record, not a served tok/s.

**Reported, not decided on (one process, deltas per load):**

| model | arm | load wall | Go heap in use | RssAnon | RssFile |
|---|---|---|---|---|---|
| 1.5B | direct | 5.61 s | +1262 MB | +1339 MB | +0 |
| 1.5B | giw | 0.00 s | +0 MB | +0 MB | +24 MB |
| 7B | direct | 16.49 s | +4961 MB | +2545 MB | +0 |
| 7B | giw | 0.01 s | +1 MB | ≈0 | +26 MB |

A `.giw` load maps and does no work up front (the weight pages fault in on first use, from the page cache — the files were
written minutes earlier, so this is a WARM cache; a cold first load reads from NVMe and is not measured). Heap after load goes
from the whole weight set to ~0, as on darwin.

**What still stands between this and a linux default** — not measured here, and not decided by this record: disk (a sidecar
per model beside it in `~/models`; `/home` ran at 98–100% during this session), the one-time transcode on first load
(21 s / 102 s here), and that a pending weights format bump (v13, in progress elsewhere) would invalidate every sidecar
already written. CUDA was not measured (weights end up in VRAM either way).

## CUDA, and the linux default flip (2026-09-24)

Owner decision after the CPU result: make the sidecar the linux default (`internal/prequant.DefaultToSidecar`, darwin + linux).
CUDA was not covered above, so it was checked before the flip was committed: `serve` built from the flip's tree
(`cuda/cmd/serve`), sidecar default vs `-direct-load`, a fresh server per arm, ABBA ×2 per model, 3 greedy 256-token decodes per
server (`scripts/bench_sidecar_cuda.py`; raw rows: `cpu-giw-vs-direct-2026-09-24-cuda.json`). Not pre-registered as a separate
rule; read against the CPU rule's ±2% band.

| model | output, every arm | decode tok/s, sidecar / direct per ABBA block | load wall, sidecar | load wall, direct | first start (one-time transcode) |
|---|---|---|---|---|---|
| Qwen2.5-Coder 1.5B | byte-identical | 0.9969, 0.9993, 0.9984, 1.0045 | 2.6–2.8 s | 8.0–8.2 s | 29.7 s (transcode 20 s) |
| Qwen2.5 7B | byte-identical | 1.0006, 0.9918, 0.9997, 0.9912 | 10.0–11.0 s | 26.2–26.8 s | 110.4 s (transcode 86 s) |

No decode cost on CUDA either (all blocks inside ±2%; the two 7B blocks near 0.991 each contain one first-after-load reading of
79.7 tok/s, the rest 81.7), and a ~3× faster load after the first. The cuda-target sidecar (`<base>.int4.cuda.giw`) is a
separate file from the CPU one (`.cpu-amd64.giw`): the cache key carries the target.

**Behaviour that comes with the default, unchanged from darwin, stated so it is not a surprise:** the first start per
(model, quant, target) transcodes and writes ~model-size beside the `.gguf`; if free disk is below the source's size, `serve`
refuses to start and names `-direct-load` rather than falling back; a sidecar that a newer build can still read is NOT rebuilt
when the format version moves (a pending v13 will leave v12 sidecars in place, readable, until deleted). `-direct-load` /
`GOINFER_GGUF_DIRECT=1` restores the old load. Not measured: WebGPU, cold page cache, models above 7B.
