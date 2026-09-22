# S3 load-time consumer — gpt-oss-20b positive control: detects on time, does not hold the line

**Result: the in-process swap guard genuinely trips inside its own stated budget — `OnTrip`
fired at swap-used baseline+0.72–0.80 GB, under the registered "+1 GB" bound — but the process's
real swap PEAK reached baseline+1.69 GB before an independent external kill-switch had to
`SIGKILL` it. The mechanism reduces the historical exposure by roughly an order of magnitude
(this run never approached the prior 22.9 GB incident) but did not hold the machine under +1 GB
on its own; the external backstop was load-bearing here, not a redundant margin.** Recorded with
the same care as a clean result, per this repo's own measurement discipline.

Box: MacBook (arm64, CPU backend — no Metal/CUDA involved), goinfer `84c70c38` (the S3 load-time
consumer commit). Model: `~/models/gpt-oss-20b-MXFP4.gguf` (12,109,566,624 bytes, MXFP4 GGUF),
loaded via `goinfer-serve --model ... --backend cpu` with `GOINFER_NO_FIT_GUARD=1` (the static fit
guard's own real, working escape hatch — `-fit=off` does NOT bypass it; that flag only turns off
auto-sizing/auto-retry conveniences, confirmed the hard way on the first attempt below).

## Why the fit guard had to be bypassed to reach this test at all

The first attempt, without `GOINFER_NO_FIT_GUARD=1`, never reached the swap tripwire: the static
fit guard refused cleanly before any weight-building began —

```
decoder: gpt-oss-20b-MXFP4.gguf needs ~24.3 GB resident at quant int4 + 6.0 GB KV + 11.3 GB
reading the checkpoint (the .gguf stays mapped resident for the whole load) = 41.6 GB; this
machine currently has 5.2 GB of memory available (budget 3.7 GB = 70% of that).
  Loading it would page to swap rather than run, so it was NOT loaded.
```

Zero swap growth, zero risk — exactly the layered-defense design working as intended. But it also
means the swap tripwire is only ever the LAST line of defense on this machine for this model: a
normal `goinfer-serve` invocation never reaches it. To exercise the tripwire itself (the actual
positive control this brief calls for), the static guard has to be deliberately bypassed —
`GOINFER_NO_FIT_GUARD=1`, the guard's own documented escape hatch — which is a genuinely riskier
mode than any default invocation, and the scenario recorded below.

## What ran

An independent external kill-switch (`sysctl vm.swapusage` polled every 1s, outside the server
process) was armed at the same instant as the server launch, ceiling set to baseline+1.25 GB —
250 MB past the in-process guard's own "+1 GB" success bound, intended as a pure backstop margin,
not expected to be the thing that actually fires.

Baseline swap-used immediately before launch: 1,748 MB (of a 3,072 MB swapfile; macOS grew it to
4,096 MB during the run, staying well inside the 12 GB of free disk headroom this box had at the
time).

| time | swap used | delta over baseline |
|---|---|---|
| 14:40:23–31 | 1,748 MB | +0 (mmap open, tokenizer load) |
| — in-process `OnTrip` fires — | 2,550 MB | **+0.72–0.80 GB** |
| 14:40:32 | 2,033 MB | +285 MB |
| 14:40:33 | 2,474 MB | +441 MB |
| 14:40:34 | 2,888 MB | +414 MB |
| 14:40:35 | 3,438 MB | +550 MB in one second — external kill-switch fires (ceiling 3,028 MB) |

The external kill-switch's own 1s poll could not catch the crossing until it had already
overshot its own ceiling by ~410 MB, for the same structural reason the in-process guard's 2s
poll overshot its: swap grew at a sustained ~400–550 MB/s burst once real dequant work started,
faster than either poll interval alone can bound.

The machine recovered immediately and fully once the process was killed: free pages went from
~6,000 (99 MB) to ~159,000 (2.6 GB) within seconds, no orphaned process, disk untouched (still
12 GB free). No hang, no panic, no lost session — a materially different outcome than the
`benchmarks.md` "M35/M26 on the Mac" incident this whole task exists to prevent, even though the
+1 GB bound was not held.

## Why the peak overshoots the trip

`parallelLayers`' abort check fires at grab time only — a layer already dispatched to a worker
when `OnTrip` closes the channel is deliberately allowed to finish rather than being cancelled
mid-write (`decoder/weights.go`'s own doc comment on `parallelLayers`). With `GOMAXPROCS` workers
dispatching in parallel and gpt-oss-20b's per-layer MoE expert tensors being large, several
workers were most likely still mid-dequant when the trip fired, and their continued allocation —
not a bug in the trip detection itself — is the most likely source of the overshoot. This was not
instrumented directly in this run (no per-worker allocation trace was taken), so it is a reasoned
inference from the design and the timing, not a confirmed root cause; see "Not done here" below.

## Decision

As registered: this is a genuine, documented LIMIT of the current mechanism, not a design
failure to silently paper over. The guard measurably narrows the historical exposure (an order of
magnitude below the prior incident, machine recovers cleanly) but does not meet its own stated
"+1 GB" bound unassisted on this box against this model. Per user instruction 2026-09-22: record
the limit, do not chase a fix under further real-hardware risk in this pass. `Options.LoadAbort`
and the serving-side wiring stay as built and unit-tested; this measurement is the honest ceiling
on what they currently guarantee, not a retraction of the unit-tested behavior itself (which is
correct on its own terms — see `decoder/parallellayers_abort_test.go`'s
`TestParallelLayers_midBuildAbort_inFlightFinishesNoNewStarts` for the exact, still-true
in-flight-finishes / no-new-starts contract this run's overshoot comes from).

## Not done here (owed if this is picked up again)

- The two negative controls the S3 brief also calls for (sidecar 1.5B and a 7B Metal load, 100
  completions each, never tripping) — not attempted this pass.
- No per-worker/per-layer allocation trace to confirm the in-flight-drain theory directly, rather
  than inferring it from timing alone.
- No attempt at a narrower poll interval or a lower default threshold — either would only buy a
  smaller window, not close the structural gap (in-flight workers can still outrun any poll-based
  check under a fast-enough burst); a real fix would need either bounding worker concurrency once
  armed, or a finer-grained abort check inside a layer's own build for huge-tensor families —
  neither attempted or designed here.
