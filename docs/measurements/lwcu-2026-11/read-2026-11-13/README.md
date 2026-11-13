# LW-cu decode and TTFT read, nobara, 2026-11-13

The read registered in `docs/measurements/lwcu-2026-11/registration.md` (PR #897), run as registered with `docs/measurements/lwcu-2026-11/run-lwcu.sh`: `serve-cuda-old` (built from 51ed2ae27, sha256 `465227c3...`) against `serve-cuda-new`
(built from 4edbe93e9, sha256 `1d671c1e...`, main's code), A B B A, KV4's G2 instrument, RF2's bands. The run script checked both sha256 values before starting (`binaries.sha256`). Graphs off (the default path; no MPS variable
exported, no MPS daemon). Models from `~/models` on NVMe. 08:41 to 09:12 PST, 31 minutes (the registration estimated 45 to 60). Machine nobara, RTX 2070 SUPER, driver 595.112.07, 16 cores. Each invocation's start load and free memory
are in `cell-starts.log` (load1 0.38 to 0.88, MemAvailable about 61.4 GB, against the void rule's 8.0 and the fit need plus 2 GB). Exclusive timing lock: `bench_peer.py` held it for the decode phases and the script wrapped `ttft_ab.py`.
The `.log` files are gzipped.

## Result

Decode, new/old, median of the paired blocks (`score.txt`; bar 0.98 or better at every depth on every model; below 0.98 is FAIL, no parked band): **13 of 13 cells PASS, 0.999 to 1.001.**

| model | 128 | 2,048 | 3,900 | 8,000 |
|---|---|---|---|---|
| qwen2.5-coder 1.5B | 0.999 (ctx 4,096), 0.999 (ctx 12,288) | 1.001 | 1.000 | 1.000 |
| qwen2.5 7B | 1.000, 1.000 | 1.001 | 0.999 | 0.999 |
| gemma3-1b | 1.000 | 1.000 | 0.999 | not registered |

Time to first token, A/B per pair, median of three pairs (`ttft-1.5B.log`, `ttft-7B.log`; bar 0.97 or better): **PASS at all four cells.**

| model | K = 512 | K = 3,900 |
|---|---|---|
| 1.5B | 1.013 | 0.998 |
| 7B | 0.998 | 0.999 |

No cell was void (none flagged in the JSON, none printed VOID by the scorer). Under the registered outcomes table: every decode cell PASS and TTFT PASS, so LW-cu's speed half closes: main holds the 51ed2ae27 numbers on the cells read.

**Never-swap (G4): not read.** Peak `VmSwap` was 0.0 MB on every sampled server on both arms and system swap read 0 MB throughout, but `score_speed.py` scores G4 VOID: it pairs servers by cell group and the sampler logged 32 `serve-cuda-old`
servers against 31 `serve-cuda-new` (`procmem.jsonl`), a count gap I did not chase (a short-lived server the sampler's poll missed is the likely cause, not checked). The registered PASS condition (each new peak within the old's, `VmHWM` within 64 MB) was
not graded, so the record says "no swap seen", not "G4 PASS". `VmHWM` medians were 1,638 MB (old) and 1,432 MB (new), maxima 5,638 and 5,457 MB.

## What this does and does not say

- A to B is every commit from `0df3294dc` to `4edbe93e9` that reaches the CUDA serve binary, not the walk alone (registration). Nothing here needs a bisect.
- Read on this build only: graphs off. The graphs-on device read (under MPS, 2026-11-12) is in `docs/measurements/cuda-read-2026-11-12/README.md`; a graphs-on speed cell is not read.
- Not read, by the registration: MC3's aggregate and verify.
