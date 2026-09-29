# Why `down` streams 17% slower per byte than gate+up — explained (2026-09-28)

**2026-09-28 · `linux` (nobara-pc) · goinfer `a93100ed`, aikit `v1.50.1`.** Answers lever 2's open question in
[`cpu-decode-peer-gap-2026-09-27.md`](cpu-decode-peer-gap-2026-09-27.md) §4: *"Why `down` streams 17% slower per
byte than gate+up is still unexplained. The per-worker timestamps inside a real token (S-02) are still not taken;
they are the next instrument."*

**Short answer: two compounding, structural causes, both already understood elsewhere in this repo — no new defect,
no mystery kernel inefficiency.**

1. `down` is issued as its own, unfused fork/join (`decoder/mlp.go`'s `matmulInto(..., &lw.DownProj, ...)`), so it
   pays a fixed per-barrier scheduling cost ALONE — while `gate+up` amortizes the SAME fixed cost over roughly
   **twice** the weight bytes, because `cpuFusedGateUp` (default on amd64 since L2/R-06's own work) fuses gate, up
   and the SwiGLU activation into one combined fork/join.
2. `down`'s activation quantizer runs on a vector of length `intermediate_size` (3.4–6.4× longer than `hidden_size`,
   the vector `gate+up`'s shared quantizer runs on) — and this quantizer is **serial, scalar, single-threaded, and
   runs on the calling goroutine before the fork/join even starts** (aikit's own S-03 item,
   `docs/task-simd-audit.md`). Its cost is not just proportional to length; it is **super-linear**, because at
   `down`'s longer lengths the two-pass (abs/max scan, then round+clamp) algorithm stops fitting in L1/L2.

This was NOT asked for as a literal instrument build (true per-goroutine `(start,end)` timestamps inside a live
token, aikit-side, the way S-02's arm64 probe works) — that still doesn't exist for amd64. It was answered instead
with an already-existing, cheaper real-token instrument (`GOINFER_DECODE_TIMING=1`), cross-checked three
independent ways.

## Method 1: the built-in real-token phase split, across all three model sizes

`GOINFER_DECODE_TIMING=1` (`decoder/model.go`) already prints `gate+up matmuls` / `down matmul` ms/token — nobody
had pointed it at this question. One short greedy generation per model (`demo/chat`, CPU backend, 40 tokens):

| model | H (hidden) | I (intermediate) | layers | gate+up ms/tok | down ms/tok | down÷gate+up |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 896 | 4864 | 24 | 6.92 | 4.70 | 0.679 |
| 1.5B | 1536 | 8960 | 28 | 18.72 | 11.19 | 0.598 |
| 7B | 3584 | 18944 | 28 | 84.65 | 46.35 | 0.548 |

Raw logs: `decode-timing-qwen2.5-coder-0.5b-instruct-q4_k_m.log`,
`decode-timing-qwen2.5-coder-1.5b-instruct-q4_k_m.log`, `decode-timing-qwen2.5-7b-instruct-q4_k_m.log`.

**If `down` were purely bandwidth-bound at the same rate as gate+up, this ratio would be a flat 0.500** (down
streams exactly the elements of ONE of gate/up, since it's the same `hidden×intermediate` element count run the
other direction). It isn't flat — it climbs from 0.548 (7B) to 0.679 (0.5B), monotonically, as the model shrinks.
That shape — a per-token, per-model-size, MONOTONIC excess over the byte-ratio floor — is the signature of a fixed
or slowly-growing cost being amortized over shrinking absolute work, not a fixed shape inefficiency (a pure
kernel/cache-shape effect would not care about absolute model size, only the K:N ratio, which is roughly constant
across these three checkpoints, 5.3–5.8).

## Method 2: a global fit isolates two real, physically distinct terms

Model: `time_ms = L·τ + L·c·n + bytes/B`, where `τ` is a fixed per-fork/join-barrier cost, `c` is the serial
scalar quantizer's cost per element, `n` is that call's own quantizer length (`H` for the shared gate+up
quantization, `I` for down's own), and `B` is the true DRAM streaming bandwidth. 3 unknowns, 6 equations (2 phases
× 3 model sizes) — genuinely overdetermined, not a per-model curve-fit with zero degrees of freedom
(`fit.py`, `fit-result.txt`):

```
tau = 71.1 us/barrier
c   = 5.706 ns/element (serial scalar quantizer)
B   = 26.03 GB/s (true streaming bandwidth)
R^2 = 0.9998
```

Residuals are 0.1–8.2% across all six numbers (worst on the 0.5B, the smallest and noisiest absolute times).
`B ≈ 26 GB/s` lands right where it should: below the ~30 GB/s read ceiling `readbw.c` measured on this box
(`cpu-decode-peer-gap-2026-09-27.md` §3), and close to Ollama's own corrected 22.5–26.0 GB/s.

## Method 3: a direct microbenchmark of the quantizer confirms the super-linear shape independently

`quantizer-microbench.go.txt` times `linalg.QuantizeRowInt8` directly, warm-cache, at exactly the six `H`/`I`
values these three checkpoints use (`quantizer-microbench.log`):

| K | ns/element | note |
|---:|---:|---|
| 896 (0.5B H) | 3.572 | |
| 1536 (1.5B H) | 3.473 | |
| 3584 (7B H) | 3.623 | flat ~3.5 ns/elem while the row fits comfortably in L1 |
| 4864 (0.5B I) | 3.779 | |
| 8960 (1.5B I) | 5.371 | **row ≈ 35.8 KB, past typical L1D — cost jumps** |
| 18944 (7B I) | 6.768 | row ≈ 75.8 KB — jumps further |

This is a genuinely independent measurement (a standalone Go program calling the real quantizer, no goinfer
process, no fork/join, no DRAM-cold weight stream) and it reproduces the SAME shape the global fit inferred: flat
around hidden-size lengths, rising past ~9K elements. The fit's single average `c` (5.7 ns/element) sits almost
exactly between the two `I`-scale direct measurements (5.37, 6.77) that dominate the 1.5B and 7B data points — as
it should, since the fit assumed one constant `c` and the truth is mildly super-linear.

**Code confirms the mechanism, not just the numbers.** `quantizeRowInt8Core` (aikit `linalg/quant.go`) does two full
scalar passes over the row — an abs/max scan, then round+clamp+store — with **no SIMD dispatch on amd64**
(`docs/task-simd-audit.md` S-03: "amd64 stays scalar... is open"; the NEON version already shipped for arm64,
2026-09-03). Two full passes over a row too large for L1 means the second pass re-fetches from L2/L3, which is
exactly what a cache-size-dependent per-element cost predicts.

## Fix built (same day): an AVX2 amd64 quantizer, mirroring the shipped arm64 one

aikit branch `amd64-quant-avx2` ([PR #2](https://github.com/townsendmerino/aikit/pull/2)), not released.
`linalg/quant_act_amd64.go`/`.s`/`_test.go`. Bit-identical to the scalar oracle (the existing
arch-independent `TestQuantizeRowInt8_bitIdenticalToScalar`/`_corners`, plus new kernel-level tests
mirroring arm64's own), mutation-checked, full `linalg` suite green, gofmt/vet/pinned-staticcheck clean.

**Two x86 ISA differences from ARM64 found and fixed, neither assumed by analogy:**

1. `VMAXPS`'s NaN rule is asymmetric (unlike ARM's symmetric `FMAXNM`). The first attempt got the operand
   order backwards — `TestQuantizeRowInt8_corners`'s `all-nan` case caught it immediately ("scale differs:
   scalar 0, dispatched NaN"). Verified empirically (a throwaway probe, every operand order) and fixed: the
   accumulator is always the first Go-listed operand, so a NaN row value can never poison it.
2. `VCVTTPS2DQ` (truncate to int32) returns one fixed "integer indefinite" value for ANY invalid conversion
   — NaN or overflow alike — unlike ARM's `FCVTAS`, which saturates toward the input's own sign. A naive
   port would turn `+Inf` into `-127` instead of `+127`. Fixed by clamping the FLOAT into `[-127,127]`
   *before* truncating, so the indefinite-value path is never taken for a genuine overflow; NaN is handled
   separately (an ordered-predicate compare + AND, zeroing exactly the NaN lanes).

**Isolated quantizer microbenchmark, same K values as Method 3, after the fix:**

| K | before (ns/elem) | after (ns/elem) | speedup |
|---:|---:|---:|---:|
| 896 | 3.572 | 0.430 | 8.3× |
| 1536 | 3.473 | 0.320 | 10.9× |
| 3584 | 3.623 | 0.246 | 14.7× |
| 4864 | 3.779 | 0.241 | 15.7× |
| 8960 | 5.371 | 0.213 | 25.2× |
| 18944 | 6.768 | 0.201 | 33.7× |

The super-linear degradation is gone — the win *grows* with K, exactly where `down` lives.

**Real-token decode-timing, all three models, before/after (exploratory — `GOINFER_DECODE_TIMING=1`
samples, not the paired `bench_peer` gate; goinfer built against this aikit branch via `go.work`):**

| model | down ms/tok before→after | down/gate+up ratio before→after | ~forward speedup |
|---|---|---|---:|
| 0.5B | 4.70 → 2.80 | 0.679 → 0.479 | 1.25× |
| 1.5B | 11.19 → 8.73 | 0.598 → 0.481 | 1.08× |
| 7B | 46.35 → 41.10 | 0.548 → 0.494 | 1.05× |

The ratio lands within 0.02 of the pure byte-ratio floor (0.500) on all three models — essentially closing
the gap this record set out to explain. A real greedy generation (1.5B, 65 tokens, temp=0) is text-identical
old vs new.

**Formal gate result (2026-09-29): SHIP, every model, both pass directions.** `bench_peer.py`, CPU, 3 runs,
`serve-cpu-new-avx2quant` vs `serve-cpu-old` (aikit branch `amd64-quant-avx2` vs origin/main pre-branch), rule
≥ 1.03× on every model both directions / park below 1.00× on any (`~/goinfer-bench/cpu-avx2quant-2026-09-28/
run-avx2quant-speed.sh`).

| model | pass 1 (new, old) | pass 2 (old, new) |
|---|---:|---:|
| 0.5B | 52.6 / 45.0 = **1.169×** | 53.4 / 47.1 = **1.134×** |
| 1.5B | 22.4 / 20.8 = **1.077×** | 22.3 / 20.8 = **1.072×** |
| 7B | 5.9 / 5.6 = **1.054×** | 5.9 / 5.7 = **1.035×** |

All six cells clear the 1.03× bar (smallest: 7B pass 2 at 1.035×); both directions agree per model (no order
effect), consistent with the exploratory `GOINFER_DECODE_TIMING` samples above. Logs:
`~/goinfer-bench/cpu-avx2quant-2026-09-28/avx2quant-speed-pass{1,2}.json`. **Not yet decided: whether to cut an
aikit release for this branch** — owner call.

## What this does and does not close

- **Closes the "why" question** lever 2 left open, with three independently-converging measurements.
- **Does not build the literal S-02 instrument** (true per-goroutine timestamps inside a live token) — that answer
  came from a cheaper, already-existing tool instead. If a future question needs true per-worker granularity (not
  just per-phase), that instrument still doesn't exist on amd64.
- **Does not ship a fix.** Two independent candidate levers, neither built:
  1. **An AVX2 `quantizeRowInt8` for amd64**, mirroring the arm64 NEON one already shipped (S-03). This is the
     more targeted fix — it shrinks `c`, which is disproportionately `down`'s problem (longer `n`) but also helps
     every other unfused caller (q/k/v, o) a little. Real new SIMD assembly on the decode critical path belongs in
     aikit, with its own bit-identical gate and mutation check (this repo's own memory: a legacy-SSE encoding once
     cost the Q4_K kernel 16× here) — pre-register before writing it.
  2. **Fusing `down` into a bigger fork/join** (batching consecutive tokens' or adjacent layers' down calls, or
     folding it into the same barrier as the NEXT layer's gate+up dispatch prep) — shrinks `τ`'s relative share.
     Structurally harder: `down`'s output IS the residual the next layer's gate+up consumes, so batching across
     layers changes the dependency shape, unlike R-06's q/k/v fusion (independent, same-layer calls).
  - Lever (1) is the smaller, lower-risk, and more broadly useful change; it is the natural next step, not (2).

## Not established

- Whether `c`'s cache-boundary jump is genuinely at L1 (32–48 KB) specifically, vs. a TLB or prefetcher effect —
  the microbenchmark warms the SAME row repeatedly (steady-state cache behavior for THAT row), which is not
  identical to decode's cold-per-token access pattern; it isolates the compute+cache-locality shape, not the real
  cold-DRAM cost of the first touch.
- Whether an AVX2 quantizer would actually close the projected gap — **built and confirmed, see the formal
  gate result above**: SHIP at ≥1.03× on every model, both pass directions, 2026-09-29.
