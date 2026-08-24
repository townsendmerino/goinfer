# Mac CPU decode vs ollama — diagnosis (2026-08-22)

> Answers `docs/prompts/mac-cpu-decode-vs-ollama.md`, run against goinfer `0e00d7d` on
> `MacBook Pro, Apple M1 Pro, 6 performance + 2 efficiency cores (8 logical), 8099 unavailable —
> ran on 127.0.0.1:8199/11599`. Method matches `scripts/bench_peer.py`: both engines over their own
> HTTP server, decode-only (client-timed from the first streamed token), warm-up completion
> discarded, greedy, `NGEN=64`, `NCOMP=8`, `NRUNS=2`, server restarted between cells. Harness:
> `bench_mac_cpu.py` (Mac-local paths/ports/quant/thread sweep; not committed — a scratch adaptation
> of `bench_peer.py`, kept in the session scratchpad).
>
> Weights verified identical (not by md5 — see `scripts/gguf_same_weights.py`'s own note on why):
> **291/291** tensors (0.5B), **339/339** tensors (1.5B), both ollama models built from goinfer's own
> GGUFs via `FROM`.

## The Mac baseline — worse than Linux, not better or absent

| model | goinfer CPU (int4) | ollama CPU | ratio | Linux/amd64 ratio (for comparison) |
|---|---|---|---|---|
| qwen2.5-coder-0.5b | 34.5 tok/s | 109.0 tok/s | **0.32x** | 0.54x |
| qwen2.5-coder-1.5b | 17.0 tok/s | 68.3 tok/s | **0.25x** | 0.44x |

**The gap does not shrink on arm64 — it roughly doubles.** The scoping note in the task doc left
open the possibility that the Mac shows a smaller gap or none; it shows a bigger one. Ollama itself
is far faster in absolute terms here than on the Linux box (109 vs 50.7 tok/s at 0.5B) — llama.cpp's
build on this Mac links Apple's **Accelerate** framework (`ACCELERATE = 1` in its own logged
`system_info`); goinfer's CPU path has no equivalent (see "residual gap" below).

## §1 — the quant confound, answered: it explains PART of the gap, not all of it

| model | int4 (W4A8, default) | int8int8 (W8A8) | f32 (native) | int8int8/int4 |
|---|---|---|---|---|
| 0.5B | 34.5 tok/s | **56.3 tok/s** | 11.8 tok/s | **1.63x** |
| 1.5B | 17.0 tok/s | **26.7 tok/s** | 3.8 tok/s | **1.57x** |

Switching goinfer off its default int4/W4A8 kernel to int8int8/W8A8 buys **~60% more throughput** —
a real, substantial effect, confirming the confound is genuine. But it does not close the gap:
even at int8int8, goinfer is still **56.3 / 109.0 = 0.52x** (0.5B) and **26.7 / 68.3 = 0.39x** (1.5B)
of ollama's native Q4_K_M. **So this is not purely "int4 vs Q4_K_M"** — there is a real backend-level
deficit underneath the quant choice, on the order of ~2x, that survives matching to the nearest
comparable int format goinfer has.

f32/native is dramatically slower than either quantized mode (as expected — no int8 SIMD path, much
more memory traffic) and is not informative for the confound question beyond confirming quantization
itself is working as intended on this backend.

**This exactly reproduces prior art, not a new phenomenon.** `docs/ideas-weight-memory.md` (2026-06-14,
aikit W4A8 NEON spike) already measured **W4A8 running 1.58x slower than W8A8** despite reading 1.6x
fewer bytes — "the nibble-unpack ALU is the limiter," i.e. compute-bound, not bandwidth-bound. Today's
fresh numbers land at **1.63x / 1.57x** — the same finding, independently reproduced two months later
on different hardware. The W4A8 NEON kernel (`aikit/linalg/dot_w4a8_arm64.s`, `dotW4A8FoldSDOT`) is
already a hand-written, DotProd-fused assembly kernel — not a naive per-nibble loop — so this is not
"someone forgot to optimize int4"; the June spike already characterized this as compute-bound and
shelved a 3-bit follow-on for exactly that reason (§ "Speed" in `docs/ideas-weight-memory.md`).

## §4 item 1 — thread count / E-core inclusion: RULED OUT for goinfer, confirmed real for ollama

Measured, not assumed, per the task doc's instruction:

- **ollama's actual default: 6 threads** (`n_threads = 6 (n_threads_batch = 6) / 8` in its own
  logged `system_info` — llama.cpp is avoiding this Mac's 2 efficiency cores by default).
- **goinfer's actual default: 8** (`GOMAXPROCS(0)` on this machine; `aikit/linalg`'s
  `resolveWidth` fans a parallel matmul out to `min(width-or-GOMAXPROCS, GOMAXPROCS, columns)`,
  and nothing in `internal/serveapp` overrides `parWidth`, so it always includes both E-cores).

| threads | goinfer int4 0.5B | goinfer int4 1.5B | ollama 0.5B | ollama 1.5B |
|---|---|---|---|---|
| 6 (P-cores only) | 35.9 tok/s | 17.6 tok/s | — | — |
| 8 (default / E-core-inclusive) | 34.5 tok/s | 17.0 tok/s | 109.0 tok/s | 68.3 tok/s |
| 8 forced on ollama (vs its own default 6) | — | — | 107.6 tok/s (flat) | **39.2 tok/s (-43%)** |

**Capping goinfer to 6 threads makes no real difference** (within run-to-run spread, if anything
marginally faster at 6 across every cell tested) — this rules out E-core inclusion as the
explanation for *goinfer's* deficit. `aikit/linalg.SetParallelWidth` exists precisely for this
scenario but isn't wired to a CLI flag; based on this measurement, wiring one would not move the
needle on the goinfer side.

**The mechanism is real, though — just not goinfer's bottleneck.** Forcing ollama's own thread count
up to 8 (matching goinfer's default) reproduced a large, one-sided regression at 1.5B (68.3 → 39.2
tok/s, a real -43%, not noise — the 0.5B cell stayed flat, so this is size-dependent). llama.cpp's
default of 6 is doing real work to avoid exactly this. That goinfer's own fork-join matmul shows no
equivalent hit is itself worth a note (finer-grained, per-matmul barriers vs. ggml's coarser
per-token thread-pool dispatch is the likely reason, not measured further here — out of scope for
this diagnosis). Either way: **not the cause of goinfer's gap**, so §4 does not become the priority.

## Cause, and what's still open

**Two stacked causes, not one:**

1. **The default int4/W4A8 kernel is ~1.6x slower than int8int8/W8A8 on this NEON path** —
   compute-bound on the nibble-unpack ALU, matching and reproducing the 2026-06-14 finding exactly.
   Bounded, already-characterized, already shelved as a research direction (rotation/3-bit) for
   being either lossy or not actually faster.
2. **A residual ~2x gap even at the closest comparable format (int8int8 vs Q4_K_M)** that is not
   explained by thread count/E-cores (ruled out above) and was not fully profiled to ops-per-byte
   here (§3's specific asks — unpack cost per weight byte vs llama.cpp's superblock-amortized
   kernels, bandwidth- vs issue-bound GEMV, `QuantizeActivationsInto`'s per-token cost — were not run
   individually; the quant/thread sweep already answered the higher-priority §1/§4 questions this
   session covered). The most likely single structural factor: **ollama's build links Apple's
   Accelerate framework** (confirmed in its own log); `aikit/linalg` has no Accelerate/vecLib/AMX
   call anywhere (`grep -rl "Accelerate\|vecLib\|cblas\|vDSP"` : no hits) — it is hand-written NEON
   assembly throughout. Llama.cpp's Q4_K_M kernel and superblock layout amortizing dequant
   differently than W4A8/W8A8's per-32-group layout is the other standing hypothesis from §2/§3 of
   the task doc. Neither is measured to a specific number here — this is the honest boundary of
   what this session confirmed vs. what remains a hypothesis.

## Recommendation

- **Don't chase int4-vs-int8int8 as new work** — it's the same compute-bound ALU limit the June
  spike already found and shelved; there's no unexplored lever there, just the known trade.
  **Do** consider surfacing it in user-facing guidance: on Apple Silicon CPU decode, `-quant
  int8int8` is ~60% faster than the int4 default, at roughly double the RAM — a fact a CPU-only Mac
  user benefits from knowing today, at zero engineering cost.
- **Don't wire `SetParallelWidth` to a CLI flag on the strength of this alone** — measured neutral
  for goinfer; the E-core mechanism is real on this hardware but doesn't explain goinfer's number.
- **The residual ~2x backend gap (int8int8 vs Q4_K_M) is the only piece with real remaining size**,
  and it is not cheap to close: it likely requires either an Accelerate-backed kernel path (a real
  build/dependency change, CPU-only-Mac-specific) or a Q4_K_M-style superblock kernel redesign —
  both are multi-day efforts with uncertain payoff, not a quick win. **Recommend: hold, disclosed,
  rather than start now** — this session's evidence narrows *where* the gap lives (not thread count,
  not the int4-specific ALU cost alone) but doesn't yet size the fix. A follow-up ops-per-byte
  profile (§3 of the task doc, not run this session) is the right next step *if and when* CPU-decode
  parity on Apple Silicon becomes a priority — not before.

---

## Follow-up (same day, 2026-08-22) — the ops-per-byte profile, a real bandwidth ceiling, and the
## per-token overhead check, all run

The recommendation above held off on §3's ops-per-byte profile pending priority. It turned out to
be cheap enough to just run, plus two more checks a second pass through this doc (and prior art in
aikit's own gitignored internal notes — **uncommitted local work from the same 2026-08-19/20 amd64
investigation this doc's §2 already cited**, unlinked here since a committed document cannot cite an
uncommitted path) surfaced as worth doing. Net effect: **the Accelerate hypothesis above is walked
back — probably not it** — and
the "hold, no clear next step" verdict is replaced by a concrete, evidenced lever that's specific to
arm64 and does NOT inherit amd64's dead end.

### Zero-cost, shipped

`-quant int8int8` is now called out as faster than the `int4` default on Apple Silicon CPU in three
places: `README.md` (the server quickstart), `internal/serveapp/main.go`'s `--quant -h` help text,
and this doc's own numbers folded into `docs/benchmarks.md`'s Apple Silicon CPU table (which
previously had `—` in its peer column).

### The Accelerate hypothesis, walked back

llama.cpp/ggml only routes through Accelerate/BLAS for large dense f32 GEMMs (batched, M>1 —
prefill-shaped). Batch-1 quantized decode GEMV goes through ggml's own hand-written NEON `vec_dot_*`
kernels regardless of the `ACCELERATE=1` build flag — Accelerate's BLAS doesn't know how to multiply
a block-quantized k-quant format without a full dequant-to-f32 first, which nobody does per-token.
So the `ACCELERATE = 1` this doc's headline saw in ollama's own logged `system_info` is real but
almost certainly doesn't touch what was actually measured (decode-only). Two OTHER flags in that
same log line — `LLAMAFILE = 1` (tinyBLAS/hand-tuned GEMM kernels, no Apple framework involved) and
`REPACK = 1` (aarch64 weight-layout interleaving for friendlier NEON access) — are the more
plausible levers, and neither requires depending on a macOS-only framework the way Accelerate would.
**The ops-per-byte profile below settles this more directly: the measured 1.97x unpack+scale-fold
tax on the W4A8 kernel alone accounts for most of int4's gap to a comparable int8 kernel, with no
Accelerate/AMX story needed to explain it.**

### The NEON ops-per-byte profile — ported from amd64, run on the M1 Pro

`aikit/linalg/w4a8_opsperbyte_bench_arm64_test.go` (new, arm64-gated, **uncommitted** — matching the
existing amd64 sibling `w4a8_opsperbyte_bench_test.go`, itself still uncommitted in `aikit` from the
2026-08-19 investigation this doc's §2 cites). Same method and shape as the amd64 original
([17408,5120], the FFN gate/up/down matmul): `dotW4A8FoldSDOT` hot (L1-resident) vs cold (streaming
55.7 MB) vs `dotI8SDOT` (the W8A8 reference kernel, no nibble-unpack, no per-group scale fold), plus
the marginal-FMA issue-width probe.

**A first run measured "hot" 12x SLOWER than "cold" — physically impossible for L1-resident vs
DRAM-streaming, and it did not reproduce** under a targeted re-check (order swapped, different rows,
explicit warm-up: all gave ~205ns regardless). Root cause not fully pinned down (this box had two
other Claude Code sessions running concurrently, and the first run also carried `-bench .`, pulling
in unrelated benchmarks) but almost certainly contention, not a real effect. Fixed by adopting this
repo's own standing convention (this repo's gitignored internal notes' P6a clock-ramp lesson):
three repeats, rotating which measurement runs first, min-of-N per measurement. **Report only the
corrected numbers below** — this is exactly the kind of single-fixed-order trap the convention exists
to catch, and it caught it.

| | ns/call | GMAC/s | GB/s |
|---|---|---|---|
| W4A8 hot (L1-resident) | 205 | 24.98 | — |
| W4A8 cold (streaming 55.7 MB) | 208 | 24.62 | 15.38 |
| I8SDOT hot (reference, no unpack/fold) | 104 | 49.23 | — |

- **Cold is only 1.01x slower than hot** — same qualitative finding as amd64 (1.05x there): this
  kernel is nowhere near memory-bound in isolation, at any thread count. 15.38 GB/s is a small
  fraction of even a single core's DRAM bandwidth (measured below), let alone the multi-core ceiling.
- **Unpack+scale-fold tax: 1.97x** (W4A8 hot vs I8SDOT hot) — smaller than amd64's measured 3.08x.
  Plausible reason: SDOT already cuts the base dot-product's own instruction count vs plain AVX2, so
  the same fixed unpack+fold overhead is a smaller relative multiplier on a cheaper base. This
  closely matches the real end-to-end decode ratio (int8int8/int4 = 1.6-1.63x measured at both model
  sizes above) — the isolated kernel tax essentially IS the real-world gap; there isn't much
  additional overhead layered on top in the real decode path.
- **Issue-width probe: ratio 1.11 → IS issue-limited.** This is the one that matters most for what
  to do next, and it's the **opposite** of amd64's finding (ratio 0.91, NOT issue-limited — latency-
  bound on a single serial accumulator chain instead). Being issue-limited means fewer
  instructions/MAC in `dotW4A8FoldSDOT`'s nibble-unpack + per-group-scale-fold sequence would
  **directly** translate to decode speed here. On amd64, the equivalent avenue was tried and closed:
  a 4-independent-accumulator restructuring (mirroring the reference kernel exactly) measured ~0%
  real gain (`perf-dead-ends.md` §8.9) because the bottleneck there wasn't accumulator-chain latency
  after all. **That amd64 dead end does not transfer to arm64** — this is a genuinely open, evidenced
  lever here, not a re-tread of a closed question.

### The bandwidth ceiling — measured, not assumed

A parallel STREAM-triad microbenchmark (`stream_triad.c`, pthread, 480 MB total — well past this
chip's shared cache, sized down from an initial 4.8 GB attempt that was accidentally measuring swap:
this machine had 2.27 GB of swap in use and ~1.7 GB truly free at the time, from other concurrent
sessions):

| threads | GB/s |
|---|---|
| 1 | 71.90 |
| 6 (P-cores) | **120.97** |
| 8 (P+E) | 110.78 |

**The real ceiling is ~121 GB/s at 6 threads, not the ~67-76 GB/s ollama itself achieves at 1.5B.**
So "ollama is bandwidth-saturated" (a plausible-sounding back-of-envelope from tok/s × bytes/token
alone) does not hold up against a real measurement — ollama is running at roughly 55-63% of the
6-thread ceiling, and goinfer's int8int8 (~42-43 GB/s implied) and int4 (~13-14 GB/s implied) are
further still. **None of the three configurations are bandwidth-saturated at the whole-model-decode
level.** One nuance worth carrying forward: ollama's achieved rate sits much closer to this
benchmark's *single-thread* number (71.9) than its 6-thread aggregate (121.0) — a per-token GEMV's
memory-access pattern (many scattered small-row reads) likely doesn't parallelize across the memory
controller as cleanly as STREAM's purely sequential access does, so treat the 6-thread ceiling as an
optimistic upper bound for this workload shape, not a number a real GEMV kernel should be expected to
reach. (The same P/E-core asymmetry shows up here too, independently of any LLM code: 8 threads is
*slower* than 6 — 110.78 vs 120.97 GB/s — matching the earlier finding that forcing ollama to 8
threads regressed it.)

### Per-token overhead — real, and proportional, not fixed

A throwaway diagnostic (`decoder/weightmat.go`'s `matmulInto`, env-gated on `GOINFER_STUB_MATMUL`,
zeroing the output instead of calling the real kernel — reverted immediately after measuring, same
disposition as aikit's own "diagnostic kernel... removed after recording the number" precedent) skips
the matmul compute to isolate everything else per token: attention scoring, norms, KV-cache writes,
sampling, HTTP/SSE streaming. Int4 only (int8int8 has a separate batched qkv/gate-up fast path,
`matmulW8A8Batch`, that this single-point stub doesn't cover — skipped rather than gotten wrong).

| model | stubbed (non-matmul only) | real (int4) | non-matmul share |
|---|---|---|---|
| 0.5B | 130.8 tok/s (7.65 ms/tok) | 34.5 tok/s (28.99 ms/tok) | **26.4%** |
| 1.5B | 64.1 tok/s (15.60 ms/tok) | 17.0 tok/s (58.82 ms/tok) | **26.5%** |

**Essentially identical proportion at both sizes.** This refutes a hypothesis this doc's own
follow-up prompt raised — that a fixed per-token cost matters more at 0.5B and is "a red herring at
1.5B+." It isn't fixed at all: the non-matmul cost itself scales with model size (7.65ms → 15.60ms,
~2.04x, roughly tracking hidden-dimension-scaled attention/norm work), landing at the same ~26%
share either way. **A real, substantial, previously-unmeasured cost — about a quarter of decode time
on this backend is not the weight matmuls at all.** Not decomposed further this session (how much is
attention/norm compute vs. KV-cache bookkeeping vs. sampling vs. HTTP/JSON streaming chunking is an
open question) — worth its own investigation, independent of and orthogonal to the whole
quant/kernel discussion above.

### Revised recommendation

1. **Zero-cost guidance: shipped** (README, flag help, benchmarks.md).
2. **Accelerate/AMX: walked back, probably not it.** Don't chase a build-dependency change on this
   basis.
3. **The int4 NEON kernel has a real, evidenced, arm64-specific lever now: it's issue-limited, and
   amd64's dead end on this exact question does not transfer.** Worth a bounded next step (reduce
   `dotW4A8FoldSDOT`'s unpack+fold instruction count) if CPU-decode speed on Apple Silicon becomes a
   priority — this is no longer a guess, it's a measured, differentiated-from-amd64 result. Still not
   started; the amd64 investigation's own experience (a plausible-sounding restructuring landing at
   ~0% real gain) is a reason to measure any attempt carefully rather than assume the win before
   building it.
4. **The ~26% non-matmul per-token overhead is a new, separate, sizable target** that nothing in the
   original task doc anticipated. It doesn't require touching the quant kernels at all. Recommend
   scoping a follow-up specifically for this — it's a quarter of decode time, on both model sizes.
5. **Bandwidth is not the binding constraint for any configuration measured** — real ceiling ~121
   GB/s (6 threads), all three configurations run at 11-63% of it. Don't reach for a bandwidth-side
   fix (there isn't one to reach for); the remaining gap is compute/instruction-count (§3 above) and
   non-matmul overhead (this section), not memory bandwidth.
