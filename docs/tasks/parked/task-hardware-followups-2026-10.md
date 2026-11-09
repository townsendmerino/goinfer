# Task (parked): what was left when the hardware-coverage task closed (2026-10)

> **Filed 2026-10-05**, when [`task-hardware-coverage-2026-10.md`](../../completed/task-hardware-coverage-2026-10.md) (H0-H6) was archived. Everything it scoped is done, measured, or listed here with a trigger.
> What it built is live code and live generated docs, not this file: the census (`docs/hardware-coverage.json`, checked by `internal/diag/hwcensus`, moved there 2026-11-04), the "Verified on" section of `docs/hardware-matrix.md`, the startup self-tests (`docs/server.md`),
> `check --hardware`, and the CI jobs `root-forced-fallbacks`, `root-forced-fallbacks-arm64`, `root-windows` and `root-windows-arm64`.
>
> **2026-11-09:** item 1's re-run ran on 2026-10-05 (decoder 99 passed, 0 failed, no timeout; the stall did not reproduce:
> `docs/measurements/sde-2026-10-04/README.md`, "Re-run result, 2026-10-05"), but the census record still says a re-run is
> queued. Item 2's date (about 2026-11-04) has passed and nothing records a decision. The triggers of items 3 to 6 still hold.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| 1 | open | update the `-spr` census record to the clean 2026-10-05 re-run, then regenerate the matrix | `docs/hardware-coverage.json` still says "re-run is queued" |
| 2 | decision | Francis picks the SDE cadence (this doc recommends the RELEASING pre-flight item); it was due about 2026-11-04 | RELEASING.md has no SDE item |
| 3 | blocked | the owner decides to rent, borrow or be given hardware | the owner's park of 2026-10-04 stands |
| 4 | blocked | after two H4 sweeps (item 3) | item 3 |
| 5 | blocked | start-up latency complaints, or the probe growing past a few seconds | no persistent cache: `decoder/selftest_gpu.go:resetSelfTestCaches` resets only in-process results |
| 6 | blocked | item 3, or a user's `check --hardware` in a bug report | `docs/hardware-matrix.md`, "Verified on" |
| 7 | open | read before every tag; it shrinks only when a path runs | RELEASING.md pre-flight item 7; the matrix says 10 of 31 |

| # | item | state | trigger or owner |
|---|---|---|---|
| 1 | The SDE `-spr` leg of `sde-goldens` | **Partial:** 0 failures, 85 decoder tests passed, then a 25-minute cap inside one test that passes alone in 102 s; cause unexplained. A `-spr`-only re-run (`sde-spr-rerun`, 45-minute cap, output in `docs/measurements/sde-2026-10-04/rerun-spr-1/`) is queued and starts when the owner starts the night queue | Record its result in `docs/measurements/sde-2026-10-04/README.md` and the census. Until it runs the only `-spr` claim is "85 passed, 0 failed, incomplete" |
| 2 | How often SDE runs | Deferred by the owner on 2026-10-04 to **about 2026-11-04**. Options: an item in the `RELEASING.md` pre-flight and the aikit-bump checklist (queue `sde-goldens`, read `summary.txt` before tagging; the recommendation, because SDE results only move when kernels or the decoder do); a weekly `night.py add` from a nobara crontab (the queue is still started by hand); both. Not a GitHub workflow: SDE is a click-through-license download | The owner decides; the reminder is a note in the assistant's own memory, so **nothing pings anyone** |
| 3 | A rented sweep before a public claim (H4) | **Parked by the owner, 2026-10-04: no rented machines.** The design is in the archived task doc: one NVIDIA card per generation, one current Apple Silicon Mac, one native-AVX-512 x86 box, one Windows machine, correctness only, judged against the CPU reference with each backend's parity tolerance. As far as is known a Claude cloud session is a CPU container with no GPU and no Mac, so it could cover at most the AVX-512 row | The owner decides to rent, borrow or be given the hardware |
| 4 | Buy a Mac mini (H5) | Blocked on item 3 (its trigger is two H4 sweeps) | After two sweeps |
| 5 | The self-test pass cache | Parked. A cached pass would hide a driver regression unless the driver version is in the key, and WebGPU does not expose it; about 1 s per launch is what it would save | Start-up latency complaints, or the probe growing past a few seconds. Safe shape: key on build, device, driver and OS build; passes only; expire in about 30 days; `check --hardware` always fresh; show "cached" |
| 6 | Hardware nobody here owns | The self-test bars were measured on NVIDIA over Vulkan (CUDA and WebGPU), an Apple M1 Pro (Metal and WebGPU) only; AMD, Intel and DirectX 12 adapters, other CUDA generations and Apple GPU families are held to them unmeasured. The DotProd kernel speed was measured on Cobalt 100 only (Snapdragon X is a different core); its whole-token effect is unmeasured everywhere | Item 3, or a user's `check --hardware` output in a bug report (the form asks for it): add a census record and, for a GPU, its probe margins |
| 7 | The never-executed list | 10 of 31 hardware-selected paths at 2026-10-04, generated at the end of `docs/hardware-matrix.md` | Read before every tag (`RELEASING.md` pre-flight item 7); shrinks only when something runs |

**Not scoped, stated in the archived doc and unchanged:** telemetry of any kind; certifying every GPU; new native backends (`gpu-vendor-coverage.md` owns that); tuning for rented hardware.

<!-- doc-reviewed: 2026-11-09 -->
