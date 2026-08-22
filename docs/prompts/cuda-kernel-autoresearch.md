# Linux-box task: autoresearch kernel-optimization loop over cuda/

## What this is

The same harness pattern from `docs/task-autoresearch-loop.md` (E9) — edit one kernel file, gate
correctness first, measure a fresh baseline, keep or revert — just run by hand, one target at a time,
rather than as an unattended loop. It was run this way against `metal/` for the last several rounds
(28 rounds: 15 real wins landed, 2 genuine unresolved correctness mysteries characterized and
documented, several honest negatives with new test/benchmark coverage). This prompt hands the same
method to `cuda/`, with the CUDA-specific traps already scouted so you don't lose an hour to each.

## The loop, concretely

Per target:

1. **Scope it**: confirm the kernel is actually dispatched in production (grep `cuda/*.go` for the
   kernel name — a `.cu` file with no Go-side pipeline lookup is dead weight, not a target), and check
   its real dispatch geometry (thread/block count at production sizes, not a tiny test fixture) before
   building anything.
2. **Branch**: `git checkout -b autoresearch/<target-name>` off an up-to-date `main`.
3. **Baseline**: run the kernel's existing correctness test (or write one if none exists, compiled
   from the REAL shipped `.cu`/`.ptx`, never a private inline copy) and its benchmark, both BEFORE
   touching anything.
4. **Edit → rebuild → gate**: change the `.cu` source, then **`./cuda/build_ptx.sh <kernel-name>`** —
   the embedded PTX (`go:embed`, `cuda/kernels.go`) is what the driver's JIT actually loads at
   runtime, so a `.cu` edit alone does nothing until this runs. Then the isolated correctness test,
   then immediately the closest whole-model gate you have (`go run ./cmd/gate gpu`'s `parity`/`heavy`
   groups, or `TestInt4_forwardParity`-style real-checkpoint tests) — don't defer this to "after
   measuring performance."
5. **Measure**: fresh, same-session baseline via `git stash` (stash the kernel edit, measure, pop,
   measure again) — never trust a number against a baseline recalled from an earlier round or a
   different machine-state. Order-alternated best-of-N, not best-of-min (clock-ramp noise is real).
6. **Keep or revert**: land only a measured, reproducible win; revert a regression or a null result
   cleanly (`git checkout -- <file>`) and record what you tried in a `results.tsv` on the scratch
   branch either way — a documented negative is worth something, a silently-abandoned attempt isn't.
7. **Land**: full `cuda` test suite green, then merge to main and push (checking `origin/main` hasn't
   moved, per the usual discipline) — only for genuine keeps. Kept-for-the-record-but-unwired kernels
   (correctness-proven, no measured win) are a legitimate outcome; say so plainly, don't force a null
   result into looking like a win.

## Two traps that will otherwise cost you the first hour

- **PTX is a build artifact, not source.** Editing a `.cu` file and running `go test` against it
  changes nothing — the binary embeds whatever `.ptx` is already in `cuda/testdata/`. Always
  `./cuda/build_ptx.sh <name>` (or the bare script with no argument to rebuild everything) between
  editing and testing. `build_ptx.sh`'s own header explains why NVRTC and not `nvcc` (gcc 15 on this
  box can't drive `nvcc`'s host compiler).
- **The bit-identical decode long-context attention lever is already closed.** Per
  `docs/task-autoresearch-loop.md` §3: Campaign A (hand-tuned) topped out at ~1.17×, the V-sum unroll
  was refuted, and the walls there are structural (dispatch-/occupancy-bound). A loop pointed at THAT
  specific lever will mostly re-derive a known ceiling — acceptable as a cheap, logged refutation, not
  worth spending a night on. This is a narrower, more specific caveat than "don't touch attention
  kernels at all" — see the concrete leads below, which are reduction-restructuring inside attention-
  adjacent kernels, not a rewrite of the decode attention throughput strategy.

## Concrete leads already scouted (not exhaustive — go find more)

**Warp-shuffle reduction is already applied broadly on the GEMV family** (`gemv_w4a8*.cu`,
`gemv_w8a8*.cu`, `fused_qkv.cu`, `moe.cu` all use `__shfl_down_sync` somewhere) — that lever is largely
mined there already; re-checking it is a fine "cheap refutation" round but don't expect much.

**Genuinely untouched**: `attn_block.cu` and `decode_splitkv.cu` both still carry the classic
shared-memory tree reduction —

```
for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }
```

— for BOTH the softmax max and the softmax denominator sum, at (grep for the exact lines first, they
may have moved). `router_f32.cu` has the same pattern too. Both `.cu` files are live: `attn_block_full`
is bound as a `Pipeline` in `cuda/drafter.go` (the DFlash block drafter's non-causal attention), and
`decode_splitkv` fills the SMs for decode's split-KV path (`cuda/kernels.go`'s own comment: "in-order →
byte-identical to attn_batched(M=1)").

**The exact lever, if it applies**: max is exact and order-independent regardless of reduction
structure — safe to replace the `__syncthreads()` tree with `__shfl_down_sync`-based warp reduction
(CUDA's equivalent of what this session called "SIMD-shuffle" on Metal: reduce within a warp via
shuffle, one cross-warp combine via shared memory, done). The **sum** reduction (softmax denominator)
is float-non-associative — do NOT touch its structure, only the max tree, exactly the isolation this
session's Metal work used repeatedly. If `nt` (the reduction width) is small enough to fit in one warp
already, there may be no lever at all — check the real dispatch width before assuming there's headroom,
the same geometry-check discipline that saved wasted effort on Metal's `qk_norm`.

**Not yet checked at all — `glue.cu` (18 `__syncthreads()`/tree-reduce hits) and `prefill_batched.cu`
(20)** are the two largest concentrations of untouched reduction-adjacent code in the kernel set. Worth
a first look before anything else, since nobody's scouted them at all this pass.

**Files with zero warp-shuffle AND zero heavy sync usage** (`argmax.cu`, `gemv_fwd.cu`,
`megakernel.cu`) are either already lean or simply don't have a reduction to restructure — confirm
before spending time, don't assume.

## What "safe to vectorize" means here (same rule as Metal, translated)

- **Exact-integer accumulation** (int8×int8 dot products, a single scale applied once at the end) is
  always safe to reorder or batch — this is how the GEMV family's existing warp-shuffle wins were
  presumably justified; the same argument applies to any batched (`float4`/`int4`-width) memory
  access you add.
- **max() is exact and order-independent** — safe to restructure the reduction tree itself, freely.
- **Sum reductions are float-non-associative** — NEVER restructure their reduction tree. This is the
  single rule that mattered most on the Metal side; two kernels there (`rmsnorm_quant`,
  `swiglu_quant`'s load side) drifted from vectorizing FLOAT LOADS near Gemma's massive-activation
  values, for reasons never fully root-caused (would need real disassembly). If something analogous
  exists on the CUDA side (a fused norm+quant kernel with its own massive-activation-sensitive path),
  expect and budget for a possible drift-and-revert outcome — that's a valid, informative result, not
  a wasted round.

## Gates available to you

- `go run ./cmd/gate gpu` — the CUDA half of this was already verified against the shell script it
  replaced (`docs/task-gate-runner.md` §12 has the Mac-side verification write-up for comparison; the
  CUDA half was done first, on this box). Its `suite`/`parity`/`heavy`/`cgofree` groups are your
  fastest whole-model confidence check.
- `go run ./cmd/gate census` — a cheap sanity check across the whole matrix; run it before and after a
  campaign to make sure nothing outside `cuda/` regressed.
- Known, pre-existing gap (unrelated to anything you'll touch): `TestInt4_forwardParity/gpt2` fails on
  the CPU-only decoder path — confirmed pre-existing on unmodified `main`, not something to chase.

## Known flakiness — check before you chase a ghost

The Metal side has a well-characterized, pre-existing, probabilistic native crash (bare fault, no Go
stack, crash SITE migrates between runs) that inflated apparent regression counts more than once this
session until it was recognized and isolated via direct retries + `vm_stat`/`uptime` checks. **Check
whether the CUDA suite has an equivalent** (VRAM contention from a stray process is the closest analog
— `nvidia-smi` before a run, per the gate's own "clean GPU" check) before concluding a measured
regression or crash is real. Two or three retries plus a system-state check is cheap; a false-positive
revert of a real win is not.

## Reporting back

Same standard as the Metal work: a running tally (table form) of target / technique / result / kept-
or-reverted, CI confirmed green (not assumed) after every push, and any genuine correctness finding
(a drift, a latent bug in something being replaced or compared against) written up with enough detail
that it can be picked up cold later — see this repo's own `metal-gate-gpu-verify.md` for the level of
detail expected in a cross-machine handoff.
