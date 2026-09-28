# MC3c step 2 S1 — fused projections in the batched CPU step: 1.058× the step on a 7B, 1.051× end to end (2026-09-27)

A follow-on to MC3c step 2 ([`concurrency-mc3c-step2-2026-09-27.md`](concurrency-mc3c-step2-2026-09-27.md)), per
[`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md) "Step 2 follow-on S1". The batched CPU decode step
now runs an int4 layer's q‖k‖v as one batched W4A8 call over the B rows, and gate‖up as another. The kernel already
existed (`linalg.MatmulBTW4A8Batch`); the change is decoder-only, in `000efe2e`. It was pre-registered in `5f7480e0`
before any code.

**Result: every gate passes, and it ships on by default.**
- On the 7B the batched step at B = 4 is **1.058×** the per-projection build at both depth 128 and 512 (bar 1.03×);
  B = 2 is 1.04× and B = 8 is 1.06×.
- Every step's logits were bit-identical between the arms.
- End to end, W7 at 4 clients reads **1.051×** (11.93 → 12.54 tok/s), and the p99 turn **0.96×** (46.2 → 44.3 s), with
  every reply identical.
- The 1.5B's batched step gains more, **1.142×**. That applies only under `-cpu-batch on`, because `auto` leaves the
  1.5B on the workers.

## Why this lever (S0, exploratory)

`TestCPUBatchS0_matmulScaling` ([`s0-matmul-7b.log`](concurrency-mc3c-s1-2026-09-27/s0-matmul-7b.log)) times the 7B's
layer-0 shapes (canonical int4, amd64) at M rows against one row:

| shape | one row | M = 2 | M = 4 | M = 8 |
|---|---:|---:|---:|---:|
| gate, up (18944 × 3584) | 1.45 ms | 1.03× | 1.08–1.09× | 1.92–1.94× |
| down (3584 × 18944) | 1.55 ms | 1.11× | 1.34× | 2.35× |
| q, o (3584 × 3584) | 0.23 ms | 1.58–1.74× | 1.69–1.87× | 3.23–3.34× |
| k, v (512 × 3584) | 0.12 ms | 1.92–2.00× | 2.98–2.99× | 5.60–5.68× |
| LM head (152064 × 3584) | 20.0 ms | 1.00× | 1.02× | 1.02× |

The large shapes already read their weights once for all M rows. The small ones do not: at those sizes the time is
fork/join and compute, not bandwidth. Perfect amortisation everywhere would save ~45 ms of a ~290 ms B = 4 step, a
≈ +17–19% ceiling. Fusing the projections removes three of a layer's five per-projection fork/joins at no numeric cost.

## Measurement (`TestCPUBatchS1_fusedVsUnfused`)

- Machine: `nobara`, Ryzen 7 3700X (16 threads), CPU backend.
- Model: qwen2.5-7b-instruct q4_k_m from `~/models`, int4, at `000efe2e`.
- 8 caches are prefilled to the depth once. For each rep and each B in {2, 4, 8}, both arms (fused on or off, flipped
  in-process through the `cpuBatchFusedW4A8` test hook) run 16 steps from the same rewound caches, with the order
  alternating by rep.
- 5 reps; one `go test` per depth; idle-gated (load1 ≤ 1.0, no other go, test or serve process).
- Run 2026-09-27 16:45–16:56 PDT.
- Raw: [`s1-d128.log`](concurrency-mc3c-s1-2026-09-27/s1-d128.log), [`s1-d512.log`](concurrency-mc3c-s1-2026-09-27/s1-d512.log),
  [`run-s1.sh`](concurrency-mc3c-s1-2026-09-27/run-s1.sh).

Fused ÷ unfused, paired per rep, median of 5 (per-rep range in brackets):

| depth | B = 2 | **B = 4 (gate)** | B = 8 | B = 4 tok/s, fused / unfused |
|---|---:|---:|---:|---|
| 128 | 1.039× (1.036–1.063) | **1.058×** (1.055–1.062) | 1.062× (1.060–1.064) | 14.77 / 13.93 |
| 512 | 1.040× (1.037–1.041) | **1.058×** (1.056–1.060) | 1.059× (1.047–1.059) | 13.53 / 12.80 |

Reported: the 1.5B (qwen2.5-coder-1.5b, depth 128) reads B = 2 1.091×, **B = 4 1.142×** and B = 8 1.148×
([`s1-1.5b.log`](concurrency-mc3c-s1-2026-09-27/s1-1.5b.log)). The fork/join the fusion removes is a larger share of a
smaller model's step.

## Gates (pre-registered in `5f7480e0`)

| gate | result |
|---|---|
| 1 identity (hard): the fixture sweep, concurrent == alone (tiny; 1.5B on; 7B auto), the 7B's MC2 identity (int4, int8int8), all with fusion on | **pass** ([`identity.log`](concurrency-mc3c-s1-2026-09-27/identity.log)); and every S1 step's logits bit-identical between the arms |
| 2 speed: B = 4 fused ÷ unfused, the lower depth's median ≥ 1.03× ships; the higher < 1.01× parks | **1.058× at both depths → ships** |
| 3 no regression (hard for ship): B = 2 and B = 8 medians ≥ 0.99× at both depths | **pass** (1.039–1.062×) |

## W7 confirmation (reported; identity hard)

- Builds: step 2 (`serve` @ `2c1d89ec`) against S1 (`serve` @ `000efe2e`), both clean builds.
- Setup: the 7B at 4 clients, serve defaults, 6 turns × 128 greedy tokens, `--fixed-nonce`, 3 pairs in the order old
  new new old old new, idle-gated.
- Run 2026-09-27 17:02–17:40 PDT.
- Raw: [`w7.json`](concurrency-mc3c-s1-2026-09-27/w7.json), [`w7-run.log`](concurrency-mc3c-s1-2026-09-27/w7-run.log),
  [`w7-servers.log`](concurrency-mc3c-s1-2026-09-27/w7-servers.log), [`gates-output.txt`](concurrency-mc3c-s1-2026-09-27/gates-output.txt).

| pair | step 2 tok/s | S1 tok/s | ratio | p99 turn |
|---|---:|---:|---:|---:|
| 1 | 11.90 | 12.53 | 1.053× | 46.23 → 44.35 s |
| 2 | 11.94 | 12.55 | 1.051× | 46.15 → 44.26 s |
| 3 | 11.94 | 12.55 | 1.051× | 46.11 → 44.35 s |

- Identity and reuse pass: every turn is identical in content and prefill between the builds.
- The end-to-end gain (1.051×) is nearly the step's (1.058×), because at 4 clients almost every decode token runs in a
  4-wide step. The server stats show 768–770 batched steps per cell, 742–750 of them 4 wide.
- `gates-output.txt` labels the aggregate "owner band": that script carries step 2's bands. For S1 the W7 aggregate is
  reported, not gated.

## Suites

The full decoder suite at `000efe2e`, run after timing, passes: 707 pass, 105 skip, 0 fail
([`decoder-suite.log`](concurrency-mc3c-s1-2026-09-27/decoder-suite.log)). The sampler throughput gate that sat near
its bar in step 2's run reads 4.09× here, under its 5.0× bar.

## Exploratory probes after S1 (not gates)

**S0b** ([`s0b-7b.log`](concurrency-mc3c-s1-2026-09-27/s0b-7b.log)) times the shapes the step now issues at M rows
against one row: q‖k‖v fused 1.83× at M = 4, o 1.95×, down 1.36×, gate‖up 1.10×. Its width sweep (E2) is **discarded**.
Its "default" and "w16" arms are the same configuration (the default width is GOMAXPROCS = 16 here), and they read
1.41× apart. The arms' position after a long serial run, which leaves the workers parked, mattered more than the
width.

**S0c** ([`s0c-7b.log`](concurrency-mc3c-s1-2026-09-27/s0c-7b.log)) re-runs the width sweep without a serial arm, with
an untimed warm block before each timed one and an A/A control:

| shape | M = 4: A' (control) | M = 4: 8 workers | M = 1: A' (control) | M = 1: 8 workers | M = 1: 12 workers |
|---|---:|---:|---:|---:|---:|
| q‖k‖v | 1.008× | 1.020× | 1.071× | **1.429×** | 1.254× |
| o | 1.013× | 1.017× | 1.026× | **1.527×** | 1.229× |
| down | 1.003× | 1.035× | 0.999× | 1.034× | 1.011× |

(× = speed-up over the default width.)

- **Width is not a lever for the batched step:** at M = 4 it is ≤ 1.035×, at the noise.
- **For single-token decode it may be.** At M = 1, q‖k‖v and o run 1.43–1.53× faster on 8 workers, one per physical
  core on this 8-core, 16-thread CPU, than on 16. That is well outside the A/A spread.
  - This is a lead for every CPU request on amd64, not only batched ones. It is numerically inert (parallel matmuls
    partition output columns).
  - It is unmeasured end to end. Production decode's own Workspace settings have not been checked, and nothing
    changes on this probe alone.

**The single-token lead does not survive end to end: a negative result.**
- *Prior art first.* R9 (`cpu-decode-attribution-2026-09-22-linux.md`) swept the fan-out width **globally** and found
  16 fastest per token, because the large MLP matmuls want every thread. Only a **per-shape** width was untested.
- *The probe.* An uncommitted test hook in `matmulInto` gave int4 projections of at most 16M MACs (q, k, v, o on the
  7B), at M = 1, a narrower fan-out. Production `m.forward` then decoded 32 greedy tokens per block on one cache
  rewound to depth 128, with arms A (off), A' (off: the control), 8 workers and 12 workers. There were 7 rotated reps,
  and the logits were bit-identical across all arms.
- *Result, tok/s ÷ A:*

  | model | A' (control) | 8 workers | 12 workers |
  |---|---:|---:|---:|
  | 7B | 1.001× | 1.005× | 1.001× |
  | 1.5B | 1.000× | 1.009× | 1.002× |
  | 0.5B | 1.017× | **0.961×** | 0.975× |

  Logs: `smallshape-7b.log`, `smallshape-coder-1.5b.log`, `smallshape-coder-0.5b.log`.
- *Why S0c's 1.43–1.53× did not transfer.* That probe drove one fused shape back to back, with warm workers and hot
  caches. In a real token q, k and v run as three separate calls between other work.
  - It is the same shape of failure as the CUDA-graphs result (1.4–1.7× on a tiny model, 1.01× real): a
    microbenchmark measuring the operation in a regime the system never runs it in.
- *Outcome.* The hook and its probe were removed rather than committed. The logs stand as the record, and per-shape
  width is closed for single-token decode on this box.

## What remains

S1 takes ~5.8% of the ~17–19% ceiling. The rest is in the kernels themselves: the small shapes still cost more per row
at M = 4 than the big ones. Lever B, an aikit small-shape kernel for k/v and q/o at small M, would go after that. It
needs an aikit release, and it starts only on its own measurement.
