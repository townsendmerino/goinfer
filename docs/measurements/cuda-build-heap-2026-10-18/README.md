# The host memory of a resident CUDA build, against what the fit guard prices (2026-10-18)

Scope, not a build (owner, 2026-10-18): the Mac's swap work found that Metal's resident build holds staging copies neither guard prices
(`docs/tasks/task-never-swap-2026-09.md`, addendum 2026-10-18). Question for CUDA: how high does the process's host memory go while a dense
model is built resident, how does that compare with what the fit guard priced, and can the packing stream into the device buffer or should it be priced?
Tier 1, by day, through the timing lock: three loads, 43 seconds in all. **Nothing here is graded.**

## Provenance

| | |
|---|---|
| build | main `5310f789`, `-tags cuda`, **sha256 `28fd105f81884715d213297560cd2129fd2df6fbc22606e4f21713b71c18825f`** (the MEM0 build) |
| machine | nobara-pc, RTX 2070 SUPER, driver 595.91.07, 64 GB RAM; each server's `VmSwap` peaked at 1 MiB or less |
| cells | `serve -backend cuda -ctx 4096` on `qwen2.5-coder-7b-instruct-q4_k_m.gguf` (a) through its `.giw` sidecar (the default) and (b) with `-direct-load`; and (c) `Phi-3-mini-4k-instruct-q4.gguf` with `-quant q4k` (no sidecar form exists for q4k) |
| instrument | `build_peak.py`: `/proc/<pid>/status` every 50 ms **from the moment the process is spawned**, through the load and 3 s past readiness; `GODEBUG=gctrace=1` for the collector's own heap figures. One run per cell, no repeats |
| priced | `testdata/fitdescribe.go` calls `decoder.FitDescribe` with the same path, quant and context: the arithmetic the host fit guard quotes |
| raw | `raw/cell-*.json` (every sample), `raw/serve-*.log`, `run1.log` |

## Readings (MiB unless GiB is written)

| cell | peak anonymous (heap) | at | peak VmRSS (anon + mapped file) | anonymous once ready | collector: largest heap before a cycle / largest live after marking |
|---|---|---|---|---|---|
| (a) 7B, sidecar | **4,121** | 9.9 s of a 10.9 s load | 8,021 | 93 | 4,018 / 2,999 |
| (b) 7B, `-direct-load` | **8,677** | 14.1 s of 15.4 s | 8,791 | **4,191** | 8,634 / 4,846 |
| (c) phi3, q4k (a direct build) | **5,314** | 2.4 s of 2.5 s | 5,421 | **2,725** | 5,283 / 4,160 |

What the fit guard priced for the same loads (`FitDescribe`, GiB):

| cell | priced | its terms | measured peak VmRSS |
|---|---|---|---|
| (a) 7B, sidecar | **nothing on the host**: a `.giw` load never reaches `fitCheckFor` (`decoder/fitguard.go:fitCheck`, `srcFileBytes`); the banner's `fit:` line is the VRAM plan | | 7.8 GiB (4.0 anonymous + 3.8 mapped) |
| (b) 7B, `-direct-load`, ctx 4096 | 8.8 GiB | 4.0 weights + 0.4 KV + 4.4 reading the checkpoint | 8.6 GiB |
| (b) at ctx 32,768 | 11.9 GiB | KV 3.5 | 8.6 GiB (not measured at that context; KV is in VRAM) |
| (c) phi3, q4k, ctx 4096 | 7.9 GiB | 2.6 weights + 3.0 KV + 2.2 reading the checkpoint | 5.3 GiB |

## What this shows

- **The host-packed copy is about one times the weights, and it is all live at once.** The anonymous peak minus what remains after the build is 4,486 MiB for the 7B (b) and 2,589 MiB for phi3: the packed int4/q4k words and scales of every layer, built in a first loop and uploaded in a second (`cuda/backend.go`, the `hls` slice; nothing is released between the two). For the 7B through the sidecar the whole anonymous peak, 4,121 MiB, is that copy (the canonical weights are mapped file pages).
- **The direct `.gguf` build is priced about right, by a different sum.** It peaks at twice the weights (canonical on the heap, plus the packed copy): 8.6 GiB against 8.8 priced. The guard reaches that total as weights + KV + the source file; at the 7B's small KV that is within 3%, and phi3's priced 7.9 against 5.3 measured is over by 2.6 GiB, about the 3.0 GiB of KV it prices as host although on CUDA the KV lives in VRAM. Priced as host, a long context over-prices the load (11.9 GiB for the 7B at 32K); it never under-prices a dense CUDA build here, because the source file stands in for the packed copy.
- **The sidecar path has a host term that nothing prices or watches.** 4.0 GiB of anonymous memory for a 7B (about 1 times the weights), gone 1 s after the build; the load-time swap watch is also not armed for a `.giw` load (the serve banner says so), and the serving half arms after the server is up. On this 64 GB box it is harmless; on a machine where the app path is the default (a 16 GB Linux box) it is the term that decides whether a 7B loads quietly.
- **The direct build keeps its canonical weights after the build** (4.2 GiB anonymous for the 7B, 2.7 for phi3 once ready; 93 MiB through the sidecar). That is the memory campaign's MEM1 (`docs/tasks/task-memory-campaign-2026-10.md`), not part of this question.

## Can the packing stream into the device buffer? Yes, for the dense projections; and that is better than pricing it

- **Why it can.** `packWeight` (`cuda/resident.go`) is row-local: an int4 row's permuted words (`permuteFast`) and its f16 scales depend only on that row's canonical nibbles and scales; int8, q4k and f32→int8 are the same. `aikit/gpu` already uploads into a range of a buffer (`Buffer.At(offset)`, then `gpu.Upload`, or `gpu.UploadBatch` over `HostCopy` ranges, which synchronizes once per batch). So a projection can allocate its device buffer first, then pack row blocks into one reused staging slice (some tens of MiB) and upload each block at its offset. The bytes are the bytes `packWeight` makes today; only when they exist changes.
- **What it needs.** The two-phase shape is the obstacle: `BuildResident` packs every layer in a first loop (where it also declines a shape it cannot take, before any device allocation) and uploads in a setup job. Interleaving pack and upload per layer means (1) a validation pass that makes the same accept/decline decision `packWeight` makes without packing (kind, `K % 4`, `K % 32` for int4, q4k size, canonical bytes present), so a decline still happens before any VRAM is taken; (2) the upload call moving into the per-layer loop on the executor thread; (3) the families that stack experts (`packWeightStack`, the C′ pinned host copy) staying as they are, because the pinned host copy is the design there. The dense q/k/v/o/gate/up/down and the LM head are where the bytes are.
- **What it buys, by this record.** The sidecar build's anonymous peak from about one times the weights (4.1 GiB) to about one layer plus the staging block (a 7B layer is about 146 MiB packed); the direct build's from twice the weights to the canonical weights plus a layer, until MEM1 removes the canonical copy.
- **Why not price it instead.** A price would have to be added to a path the guard does not run on (`.giw`), and would refuse loads that run fine on a big machine, to protect against a transient that streaming removes. Pricing is the fallback if streaming proves too invasive, and the cheap part of it is true already for the direct path.
- **Gates, sketched (registered by whoever builds it).** Tier 0: on each tiny CUDA fixture the device bytes after the build equal today's, read back, for int4, int8, q4k and f32→int8 projections; the build's peak heap, sampled at layer boundaries through a test hook, falls from O(model) to O(layer); a declined shape declines with no device buffer allocated (plant: validate after allocating, red). Tier 1: this record's `build_peak.py` on the 7B sidecar and phi3, before and after. No timed gate: loading speed is the only speed term, one more cell of this script.
- **Size.** S–M, in `cuda/backend.go` and `cuda/resident.go` only, with the accounting note in `decoder/residentneed.go` unchanged for the host (the term disappears rather than moves).

## Limits

One run per cell; samples every 50 ms, so a spike shorter than that is not seen (the collector's own figures agree with the sampled peaks within 3%). The CUDA families other than dense (MLA, DeltaNet, Mamba, Gemma 4 MoE) were not measured. The 7B at a long context was priced, not built. The drafter and the C′ expert-cache paths are out of scope.
