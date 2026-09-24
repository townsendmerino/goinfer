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

_(appended after the run)_
