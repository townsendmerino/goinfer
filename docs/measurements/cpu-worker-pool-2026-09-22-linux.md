# R9 step 2, Linux: a persistent worker pool for the CPU matmul fan-out — pre-registration

Follows `cpu-decode-attribution-2026-09-22-linux.md`, which left the 1.5B token at 54.5 ms with its MLP matmuls
(gate+up 19.9 + down 12.0 = 32 ms) streaming 19.5 of the box's 30.5 GB/s read ceiling in isolation and less
inside the token — and named goroutine wake stagger as the mechanism behind every fan-out loss it found. aikit
has no persistent pool: `Workspace.SetWorkers` only caps width; every matmul call spawns fresh goroutines
(`parallelSpawnCols`). This is the "per-token worker pool that does not re-fork per matmul" R9 step 2 lists.

## Design (aikit `linalg/pool.go`)

`WorkerPool`: N persistent goroutines. A fan-out publishes one job (fn, N, grain, an atomic chunk cursor) and
bumps a generation counter; workers spin on the counter for a bounded window (default 200 µs — longer than the
gaps between a decode token's matmuls, shorter than a request's idle) and then park on a channel; the caller
works too; completion is an in-flight-chunk counter, so a worker that wakes late simply finds no chunks. The
chunk/grain arithmetic is `parallelSpawnCols`'s own (shared helper), so which worker takes a chunk never enters
the arithmetic — the same numerically-inert contract as the width. One fan-out at a time per pool (mutex).

## Pre-registered rule

- Bit-identity gates: a new aikit `==` test (pool vs spawn on W4A8/W8A8/f32 matmuls, several shapes, including
  ragged N), goinfer's `TestForwardN_matchesSequential` + goldens, `TestParityManifest_fresh` after.
- Idle gate: workers must be parked within ~1 ms of the last fan-out (a test), so a serving process does not
  burn cores between requests.
- Speed: paired in-process ABBA on the 1.5B at depth 128, ≥ 3 pairs. **Ship ≥ 5% token speedup; park 2–5%;
  kill < 2%.** 7B and 0.5B must not regress > 2%. Ceiling from the attribution: ~1.24× (32 → ~19 ms of MLP
  matmuls if they reached the read ceiling); a result near it says the fan-out was the wall, one near 1.0 says
  DRAM was.
- Ships as a goinfer default only on the architecture it was measured on (this box); the Mac measures its own.

## Result — KILLED, negative, code not shipped (log `cpu-worker-pool-2026-09-22-linux-ab.log`)

Built as designed (aikit `linalg.WorkerPool` + `Workspace.SetPool`, 3 tests: pool == spawn bit-identical on 6
shapes W4A8/W8A8, idle parks within 50 ms, concurrent Runs exactly-once; goinfer: one process-wide pool attached
to the decode Workspace). Both readings said no:

- **Token-shaped isolated sweep** (84 W4A8 matmuls over distinct 1.5B matrices, DRAM-streamed, aikit benchmark):
  spawn 29.9 ms vs pool 30.2 ms — a null. The per-matmul spawn cost the pool removes was already small inside a
  back-to-back matmul sequence; those matmuls are DRAM/kernel-bound at ~39 GMAC/s aggregate, not fan-out-bound.
- **In the token** (`TestR9_cpuTuningAB` knob, 1.5B depth 128, ABBA, 3 pairs): pool ON **59.7** vs OFF **54.7**
  ms/token — **OFF faster in 3/3 pairs, 1.088–1.095×**. The matmul component moved 33.5 → 32.2 ms; the other
  ~6 ms appeared in the serial stretches between matmuls (activation, attention core, norms), which now share
  the cores with 15 spinning workers. The 200 µs spin window that keeps workers hot for the next matmul is the
  same window that steals from the work between matmuls.

Verdict by the pre-registered rule: **KILL** (< 2%, and negative). The pre-registration's ceiling (32 → ~19 ms
"if the matmuls reached the read ceiling") assumed the fan-out was what held them at 19.5 GB/s; the isolated
sweep shows it was not, which is the finding worth keeping: **on this box the remaining MLP matmul cost is not a
dispatch problem.** The pool code is not kept — a primitive with no consumer and a measured loss at its one
call site would be dead weight in aikit's API. A parking (non-spinning) pool would give back the serial-stretch
cycles but also the wake latency it exists to hide; not pursued.

What is left of R9 step 2 on Linux after this: nothing dispatch-shaped. The 1.5B token is 54.5 ms = matmuls
~42 (MLP 32 + attention projections 7.4 + LM head 8.7 of which the head is bandwidth) + attention core 2.5 +
activation 3.7 + norms/residual ~1; the matmul term at ~39 GMAC/s / 19.5 GB/s against a 30.5 GB/s ceiling is a
kernel/memory-system question for aikit (prefetch, page size, per-core stream count), not a goinfer one.
