# The two cold-user dead ends, re-run on this box (2026-10-06, exploratory)

The cold-user run on v0.20.0 (`docs/measurements/cold-user-2026-10-05-nobara-pc.md`) ended in two dead ends: **B** (opencode on Qwen2.5-Coder-7B made no edits) and **D** (the 26B on the 8 GB card was stopped by the tester's swap-safety rule before it produced anything). Fixes for both, and for the slow image turn, landed afterwards; this re-runs B and D against a build of `main` (v0.22.1-0.20261007014722-85ec8e9fb7ba) and against the v0.22.0 release binary (053bc7110a1e),
side by side, to see what a user of each would meet.

**This is not a cold-user pass.** It ran from the repo, by the person who wrote the fixes, with one model pulled by the real `pull` command and the rest local. A cold-user run is a fresh window with only the published assets, and RELEASING.md pre-flight item 5 still owes one on the next tag. One run per cell, sampling defaults left as they are, so differences of one run are noise.

nobara-pc: Ryzen 7 3700X, 62 GB, RTX 2070 SUPER 8 GB, driver 595.91.07, `-backend cuda`, opencode 1.18.29 (the cold-user run's version), the cold-user run's 12-line `calc.go` bug. Raw outputs, the three scripts (`scenB.sh`, `scenD.sh`, `swapwatch.py`) and every log are in this directory.

## Scenario B: opencode on the README's recommended model (Qwen2.5-7B-Instruct q4_k_m, `-quant int4 -ctx 16384`)

| | v0.22.0 release binary | `main` |
|---|---|---|
| `serve check` | **7 pass, `tools, OpenAI` FAIL** ("turn two asked for the tool again", the livelock), 1 skip | **8 pass**, 1 skip (vision) |
| explicit prompt ("read calc.go, then use the edit tool"), 8 runs, file fixed | 7 of 8 (B5 added `return a + b` and left the old `return a - b` after it: correct value, dead code, counted as not fixed) | 7 of 8 (B2 left the file unchanged) |
| vague prompt ("fix the bug in calc.go"), 3 runs, file fixed | 1 of 3 | 0 of 3 |

**Reading.** The README now steers to the model that works, and under real opencode it does, on both binaries, about equally (7/8 each; the vague prompt is unreliable on a 7B and 1/3 against 0/3 is noise). The single-tool livelock the cold-user run's `serve check` pointed at did NOT hurt opencode, because opencode sends a dozen tools and the lock needs exactly one; what it breaks is `serve check` itself and any client with one tool, and `main` fixes that. So the cold-user dead end B is resolved by the README's model advice, and the fix that is not in a release is for the one-tool case. (The Coder-7B under opencode is in `docs/measurements/lenient-tool-calls-opencode-2026-10-06/`.)

## Scenario D: the README's `-moe-cache-experts` command on the 8 GB card (`pull gemma-4-26b-a4b`, 14.4 GB at ~70 MB/s, then `goinfer-serve -backend cuda -moe-cache-experts -model <it>`)

| | v0.22.0 | `main` |
|---|---|---|
| ready | 42 s (sidecar already built by the other run) | **118 s** cold, of which the one-time transcode 1 min 6 s |
| decode path / VRAM | `cuda-resident (int4)`; "C′ cache: 64 slots/layer would need 6.5 GB but only 3.9 GB free, capping to 33 (3.4 GB)" | the same |
| chat | TTFT 1.05 s, ~26 chunks/s then 34 | TTFT 1.02 s, the same |
| `serve check` | 7 pass, 1 skip (vision), 1 FAIL (`stop sequences`) | 7 pass, 1 skip (vision), 1 FAIL (`stop sequences`) |
| **the server's own swap** (`VmSwap` of its process, 0.5 s samples) | peak **35 MB** | peak **0 MB** |
| system-wide swap-used growth / pswpout during the run | +157 MB / 171 MB | +646 MB / 0.9 MB |
| MemAvailable minimum | 42.6 GB | 42.5 GB |

**Reading.** D is not a dead end any more on this box: the command works as written, the 26B answers at about a second to first token. The swap numbers show why the cold-user run stopped: the system-wide swap counter rose by hundreds of MB while the server's own swap was 0 to 35 MB (other processes, the desktop; zram on this box), which is exactly what the old rule ("stop on any rise in swap-used, or the first pageouts") would have stopped on. The amended rule (docs/tasks/task-first-hour.md rule 2) keys on growth past the guard's threshold and, in this watcher, on the server's own swap, so neither run was stopped. Both `serve` binaries print the swap-guard "not armed" notice for the transcoded `.giw` load. Friction a first-time user would still meet: **the `long prompt (~2000 words)` row takes 35 s to first token** on this model (the experts stream through VRAM for a long prefill), and **`stop sequences` FAILS here ("the reply never reached the stop sequence")**.

## Two things this surfaced

1. **FIXED (exploratory validation, 2026-10-06): the `stop sequences` row failed for models that are fine.** It asked for `Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.` with a stop at `5`. Qwen2.5-Coder-7B answered "The count is now 10." and the 26B "The total count is…"; Qwen2.5-7B-Instruct counted. The row was reporting that the model did not follow an ambiguous prompt, not that stop sequences are broken. The prompt is now "Count from 1 to 10, separated by commas, and write nothing else."; the leaked-"5" and reached-"4" checks are unchanged (`TestStop_leakFails`, `TestStop_neverFiresFails`, `TestStop_firesPasses`). Re-run with a CUDA build of the working tree (`cuda/cmd/serve`, `-tags cuda`): the 26B (`-moe-cache-experts`) now passes the row, `8 of 9 checks passed (1 skipped)`, and Qwen2.5-7B-Instruct still passes (`8 of 9`). **Not re-run: Qwen2.5-Coder-7B** (its GGUF is no longer cached here), so the 26B stands in for the "fine model that failed" case. An earlier attempt built the root `cmd/serve`, which has no backend, and ran on the CPU; it also passed the stop row on the 26B but is not the configuration of record.

2. **A cold first load of the 26B is ~2 minutes and, with the 35 s long-prompt TTFT, can look hung.** The banner prints the sidecar build ("stream-weights: transcoding…") but nothing says a long prompt will be slow on this card. The re-run above reproduced the long-prompt row at TTFT 35.09 s (35.53 s before), so that figure is stable; it is the C′ expert streaming on an 8 GB card, already batched (queue P-series M26), not a regression. Server VmSwap peak in the re-run was 405 MB (0 and 35 MB before), under the 512 MB guard.

## What this does not cover

Scenarios A, C, E and F were not re-run here (F, the image turn, was measured separately: `docs/measurements/p26b-cuda-hybrid-image-prefill-2026-10-06/`, `p26c-cpu-deltanet-fanout-2026-10-06/`). Nothing here ran on the Mac. The release binary was run from this repo's checkout, not a fresh directory.
