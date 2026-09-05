# Task: the prefill gap — one contract, four levers (2026-09-05)

> **BLUF.** Decode is a ~30% problem everywhere (0.71–1.24× of Ollama, `benchmarks.md` §B8).
> Prefill is not: on CUDA the overhead-free marginal cost per prompt token is **5.8× behind at
> K≈512 and 12.7–14.8× at K≈3900**, and it *grows* with K while Ollama's is flat
> (`measurements/cuda-prefill-reanchor-2026-09-01.md`); on Mac Metal the **shipped default is
> sequential prefill through the decode path** (~40–60 tok/s on 1.5B) because the batched kernel is
> declined for not being bit-identical (`metal/backend.go:236`); on Mac CPU the gap has shrunk to
> ~1.8× at K=512 and about parity at depth since the S-01 tile (`897fb18`), which §A of
> `benchmarks.md` does not yet say. The workload that matters most — W4, the agent turn
> (`task-peer-benchmarks.md:71`) — is prefill of each turn's new 500–3000 tokens, which sits exactly
> in the band where the deficit is 1–6× on TTFT.
>
> **The cause is one decision, not several kernels.** Every batched prefill kernel is "the M=1 decode
> kernel with an M dimension", chosen so batched prefill is bit-identical to sequential decode. That
> forecloses tensor cores on the weight term (`cuda/gemv_w4a8_batched.cu:19`) and a fused schedule
> on the attention term (`cuda/prefill_batched.cu:156`). The CPU backend already split this contract
> — `--cpu-fast-attention` is default ON and `--cpu-exact-prefill` buys identity back
> (`internal/serveapp/main.go:373`, `:318`) — and CUDA decode is held to the 3% near-tie parity rule
> rather than to bytes (`benchmarks.md` §B2). This doc extends that contract to the GPU backends and
> sequences the four levers it unlocks, cheapest first: **L1** flip Metal's batched prefill on
> (measured 3.7–4.6×, already in the tree); **L2** a fused attention kernel on CUDA; **L3** a
> tensor-core int4 GEMM on CUDA that keeps group scales; **L4** the remaining CPU items, which now
> live in aikit's SIMD audit and are the smallest prize.
>
> **Status: SCOPED 2026-09-05, nothing built.** goinfer `3b20f74`, aikit v1.34.0. Path:line
> citations were taken at that commit; `scripts/queue_citation_lint.py --update` re-indexes them.
> Not filed in `queue-performance.md` yet — another session was editing it while this was written.

## 0. What must not change

- **`benchmarks.md` § Methodology is the authority.** Same box, same checkpoint+quant, greedy,
  pinned versions, thermal note, `~/models` only, interleaved arms, spreads reported. A prefill
  number is a **fresh-prefix** number: Ollama caches prompts (repeat/fresh 0.53 on CUDA, 0.00–0.05
  on CPU), so every request in a cell carries a unique prefix. `scripts/bench_peer_prefill.py`
  already does this; it is the instrument for every peer row below.
- **TTFT rate is not prefill throughput.** Ollama carries a 340–356 ms per-request floor that
  goinfer does not; the comparable quantity is the marginal cost between adjacent depths
  (`measurements/cuda-prefill-reanchor-2026-09-01.md:75`–`:78`). Every row here reports both.
- **The exact path is retained, not replaced.** Whatever ships as the default, a bit-identical
  prefill stays selectable (`--exact-prefill`, below) and remains the reference the parity gate,
  the goldens, and the speculative verify oracle run against. Spec-decode verify keeps the exact
  kernels regardless of this doc — `spec/00-core`'s lossless contract has no opt-out, and nothing
  here reopens it.
- **Projection bands before measuring; results held to them.** Each lever pre-registers a ship /
  ambiguous / park band on a named cell. The 2026-09-01 P19 result (1.7× at the kernel, 1.08×
  end to end) is the reason: a kernel win is not an end-to-end win until Amdahl has been paid.
- **Realistic prompts for anything content-dependent.** `scripts/prompts.json` is four unique words
  and is fine for throughput; the fidelity gate (§3) uses prose.

## 1. The gap, per backend, with provenance

| backend | cell | goinfer | peer (Ollama v0.32.5) | ratio | source |
|---|---|---|---|---|---|
| CUDA, dense 1.5B int4, TTFT rate | K=512 | 1368 tok/s | 1245 | **0.91×** (goinfer ahead) | reanchor `:56` |
| CUDA, same | K=3900 | 702 | 4300 | **6.13×** | reanchor `:59` |
| CUDA, **marginal ms/token** | 136→519 | 0.760 | 0.132 | **5.8×** | reanchor `:77`–`:78` |
| CUDA, marginal | 2067→3919 | 1.847 | 0.145 | **12.7×** | same |
| CUDA, dense 0.5B, marginal | 2067→3919 | 0.932 | 0.063 | **14.8×** | reanchor `:75` |
| CUDA, D7 7B int4, batched vs its own sequential fallback | M=8012 | 6.358 ms/tok | (seq. 19.211) | 3.02× | `prefill-chunking-d7-2026-09-04.md:34` |
| Mac CPU, 1.5B int4, TTFT rate — **pre-tile** | K=512 | 68.4 | 203.7 | 2.98× | `cpu-peer-prefill-2026-09-01.md:28` |
| Mac CPU, same — pre-tile | K=3900 | 61.9 | 111.2 | 1.80× | `:31` |
| Mac CPU, 1.5B int4 — **post-tile, cross-session, INDICATIVE** | K=512 / 3900 | 110.8 / 102.9 | (Sep 1 peer) | ~1.8× / ~1.08× | `897fb18` commit message |
| Mac Metal, 1.5B W4A8, **shipped default** (sequential) | P=2048 | 24.87 ms/tok (~40 tok/s) | **unmeasured** | — | `ollama-chase.md:1583` |
| Mac Metal, `--metal-fast-prefill` | P=2048 | 6.33 ms/tok (~158 tok/s) | **unmeasured** | — | same |

Three things the table says that the benchmarks page currently does not:

1. **The CUDA deficit has two terms, and they are separately sized.** At K=512 attention is 14.3%
   of prefill (`measurements/cuda-prefill-attention-share-2026-09-01.md:23`), so the 5.8× marginal
   gap at the shallow interval is almost entirely the **weight term** — the batched GEMV. By K=3900
   attention is 55.0% (`:25`) and the marginal gap has grown to 12.7× — the **attention term**, whose
   cost per token rises with K. Two mechanisms, two levers (L3 and L2 respectively), and neither
   closes the other's half.
2. **The Mac CPU row is stale in goinfer's favour.** §A's 2.98× was measured at `9d8e382` on aikit
   v1.31.0; the S-01 register-blocked tile landed in v1.32.0 the next day (`897fb18`) and its
   end-to-end measurement put int4 prefill at 110.8 tok/s (K=512, n=9: 113.2) and 102.9 (K=3900),
   1.6–1.8× over the pre-tile arm, with the int4/int8int8 inversion gone. Against the Sep 1 peer
   figures that is ~1.8× behind at K=512 and ~1.08× at K=3900 — but those are two sessions, so
   it is a prediction for the interleaved re-run, not a row. **§A must be re-measured before any
   CPU work is funded** (L4 step 0).
3. **Mac Metal has no peer prefill number at all**, and the default it would be measured at is
   the sequential path. W2's Mac cells (`benchmarks.md:1570`) are the missing instrument.

**Out of scope, owned elsewhere:** M26/M35 prefill on the 8 GB card is DMA-bound (59.5% of M26's
prefill is host→VRAM expert traffic, `queue-performance.md` P20) — no lever in this doc touches it,
and P20 already names expert-major as its item. WebGPU prefill is not measured against a peer and
has no counterpart to measure against.

## 2. Mechanism — where the time goes, in the code

### 2.1 The weight term: a GEMV that loops M, not a GEMM

`cuda/gemv_w4a8_batched.cu` is a warp-per-output-row dp4a GEMV with `MT=32` register accumulators
(`:54`), copied from `gemv_w4a8_fwd` verbatim so that "every output element equals its
sequential-GEMV value to the bit" (`:18`–`:19`). Its own header states the consequence — "This is
WHY there is no IMMA/MMA path" (`:19`) — and its profile: L1TEX-latency-bound at 46% compute, not
DRAM, not issue (`:22`–`:26`). No shared-memory tiling, no `ldmatrix`, no tensor cores. On Turing
dp4a is roughly a third of IMMA int8 throughput and the kernel sits at ~half of dp4a, which is the
arithmetic behind the ~5× shallow-depth gap.

The IMMA question was scoped and deferred on 2026-08-04 (`a276201`;
`completed/task-rotation-perrow-imma.md` §1, §11) on two grounds. The first — that group scales
force float accumulation and therefore per-row scales (and a rotation to make them quality-neutral)
are a prerequisite for tensor cores — is **not correct as a hardware constraint**. A group-scaled
GEMM on `mma.sync` accumulates each 32-wide group in int32 (two `m16n8k16.s8` instructions), then
folds `float(acc) * groupScale` into an fp32 accumulator, exactly the per-group math the dp4a kernel
does over 8-value words today. Same int8 products, same group scale, same precision; the only thing
that changes is the association of the cross-group float sum — which is bit-identity with the M=1
kernel, not quality. llama.cpp's MMQ kernels run Q4_K on Turing this way. The second ground —
"prefill-only ~3× on an already-past-threshold TTFT" — predates both the 2026-09-01 re-anchor and
W4 being named the headline workload. Both premises moved; §11's reopen trigger is met.

### 2.2 The attention term: a decode kernel launched once per query row

`attn_batched` (`cuda/prefill_batched.cu:163`) is gridded `(head, query row)` (`:156`). Each block
streams the **entire** K prefix from global memory for its one query (`:180`), materialises the
full score row in shared memory, and then runs a serial `nKeys`-long FMA chain per output lane for
the AV product (`:215`). Nothing is shared across query rows: K and V are re-read M times per layer,
and the AV chain's length grows with K. That is an O(M·K) memory pattern with a serial O(K) tail —
the O(K²) signature the re-anchor measured as marginal cost rising 0.377 → 0.932 ms/token. A
FlashAttention schedule loads each K/V tile once per block of 64–128 query rows, keeps the running
max/denominator in registers, and does both products on tensor cores; that is what a flat marginal
cost looks like on the peer.

Metal's batched path has the same shape: its GEMM is already `simdgroup_matrix` f16 MMA
(`metal/prefill.go:10`–`:20`), but `attention_prefill` is "one threadgroup per (row m, query head)"
reusing the decode math, with the note "an MMA/flash version is a later throughput lever"
(`metal/prefill.go:210`–`:214`).

### 2.3 Why it is one decision

The CPU backend already faced this and split the contract: f32 prefill attention became the default
above 512 tokens, P19's fused schedule ships under the same flag, and `--cpu-exact-prefill` is the
documented way back to `decode == prefill` byte-identity (`internal/serveapp/main.go:318`). Metal
built the fast path and then declined it by default on a *stream-divergence* number (54%,
`metal/backend.go:236`–`:241`) — a measure of whether any token ever differs, which is the wrong
gate for quality: CUDA decode is not held to it either (`benchmarks.md` §B2, "3% near-tie parity
rule"). The GPU prefill paths are the only place in the tree where bit-identity still governs the
*default*, and it costs 5–15× on the workload W4 measures.

## 3. The contract — one flag, one gate, every backend

**Flag.** `--exact-prefill` (server) / `Options.ExactPrefill` (library): force the bit-identical
prompt-ingestion path on whichever backend is running. Subsumes `--cpu-exact-prefill` (kept as an
alias for one release) and inverts `--metal-fast-prefill` (kept as a no-op alias with a deprecation
line). Default off. The fast path is the default on every backend **once it passes the gate**;
until then that backend's default is unchanged.

**Gate — reuse the repo's, do not invent one.** A backend's fast prefill may become the default
when, on ≥10 realistic prose prompts per model (not `prompts.json`) spanning K ∈ {256, 1024, 3900},
against the exact path on the same backend:

| check | bar | why this one |
|---|---|---|
| seed-logit argmax agreement | the 3% near-tie parity rule, as CUDA decode is already held to CPU (`benchmarks.md:611`) | the same bar the tree already trusts for a non-identical backend |
| teacher-forced top-1 agreement over 64 continuation positions | reported, and ≥ the CUDA-decode-vs-CPU figure on the same model | the fidelity column `task-peer-benchmarks.md` §4 is building anyway |
| seed-distribution KL vs exact | reported, not gating | the number a reader will ask for |
| greedy stream divergence rate | **reported, not gating** | it is what gated Metal; it measures reproducibility, not quality |

A backend that fails the gate keeps the exact default and the fast path stays opt-in with the
failure recorded beside it. The gate runs as a heavy test per backend (`GOINFER_HEAVY_TESTS=1`),
with the exact path as the oracle, so it is a regression gate afterwards, not a one-off.

**What the contract does not touch.** Decode (unchanged on every backend), speculative verify
(exact kernels, lossless contract intact), and the parity gates (which run the exact path). The
change is confined to prompt ingestion, which is why `--exact-prefill` is a complete undo.

## 4. Levers, cheapest first

### L1 · Metal: make the batched prefill the default (gate-conditional)

- **What exists:** `PrefillLast` on Metal is a working f16-MMA batched prefill, measured 3.93–4.56×
  over sequential at P=128…2048 and 3.74× end to end on a real 1450-token prompt through the
  server (`ollama-chase.md:1578`–`:1611`). It is declined unless `GOINFER_METAL_BATCHED_PREFILL=1`
  (`metal/backend.go:248`), which `--metal-fast-prefill` sets (`internal/serveapp/main.go:372`).
- **Work:** run the §3 gate on Metal for S and D7. If it passes, flip the default and route the
  flag through `--exact-prefill`. If it fails, record the cell and stop — L1 is then "not
  quality-neutral as built", which is a finding about the f16 activation path, not about the
  contract. Either way, **run W2's Mac cells** (`scripts/bench_peer_prefill.py` needs a Metal
  backend option; the harness is CPU/CUDA today) so `benchmarks.md` carries a Metal prefill row in
  both arms, default and fast.
- **Band (TTFT, S at K=2048, vs the shipped default):** ≥3× ships (the measured figure is 3.9×);
  <2× means the serve path is eating it and the item reopens as a serving investigation.
- **Then, not now:** a `simdgroup_matrix` flash attention for `attention_prefill` — the Metal twin
  of L2. Scoped after L1's peer row exists, because that row decides how much of the remaining gap
  is attention.

### L2 · CUDA: fused (FlashAttention-style) prefill attention

- **Kernel shape:** one block per (head, 64-row query tile); K/V streamed in 64-key tiles through
  shared memory, converted f32→f16 on load (the resident KV stays f32 — this is not a KV-quant
  item, and KV-quant was refuted as a decode lever on this card); `mma.sync.m16n8k16.f16` with f32
  accumulation for QKᵀ and PV; online softmax with the running max/denominator in registers; causal
  mask per row, sliding window per row, GQA head grouping, and the gpt-oss `sinks` term — every
  seam `attn_batched` handles today (`cuda/prefill_batched.cu:156`–`:162`) must be handled here,
  with a test per seam. `attn_batched` stays as the exact path.
- **Not bit-identical**, by the rescale in the online softmax; this is the P19 category, on the
  backend P19 explicitly left "still not tested".
- **Band (dense 1.5B int4, K=3900, end to end TTFT, `TestPrefillTTFT` extended to 3900):**
  attention is 55.0% there, so a perfect kernel caps at 2.22×; a 4× kernel gives ~1.7×.
  **Ships at ≥1.4×; ambiguous 1.15–1.4×; parks below 1.15×.** At K=512 the same kernel is worth at
  most 1.17× and is not the cell that decides it. Kernel-level ratio recorded separately via
  `TestPrefillDecomp`'s attention category so the two numbers cannot be confused.
- **Depends on:** nothing. Can start first. The chunked pass (`prefillChunked`, ≤512 rows) is
  unaffected: the kernel reads the positional KV the same way.

### L3 · CUDA: tensor-core int4 GEMM with group scales

- **Kernel shape:** `mma.sync.m16n8k16.s8.s8.s32` (Turing IMMA). Activations are already per-row
  int8 with `aScale[m]`, so the A operand is ready; W nibbles are unpacked to int8 into shared
  memory in fragment order (the pack-time nibble permutation `permuteFast` can be chosen to make
  this a shift-and-mask, not a gather), loaded with `ldmatrix`. Per 32-wide group: two MMAs into
  int32, then `facc += float(acc) * gs[n][g]` in fp32; per row: `* aScale[m] + bias`. Weight-
  stationary over an N-tile, streaming M in 64–128-row tiles. `gemv_w4a8_batched` stays as the exact
  path and as the M<16 path (tensor cores lose below a warp's worth of rows).
- **Same products, same per-group scale, same precision as today's kernel**; the cross-group float
  order differs. Quality is expected to be indistinguishable from the exact path under the §3 gate
  — but "expected" is the claim, and the gate is the evidence.
- **Band (dense 1.5B int4, `TestPrefillDecomp` gemv category at K=512, where that category is
  82% of prefill):** the kernel is at ~54% of dp4a and dp4a is ~⅓ of IMMA, so ~5× is the counted
  ceiling on the category. **Ships at ≥2.5× on the category; ambiguous 1.5–2.5×; parks below
  1.5×.** End-to-end at K=512: ships at ≥2×.
- **Bit-identity gate at M=1:** not required — this kernel is never selected at M=1.
- **Depends on:** the `moe.ptx` rule (audited artifact, untouched — new `.cu` beside it, as
  `decode_splitkv.cu` was added). NVRTC is on the box. Also reopen
  `completed/task-rotation-perrow-imma.md` §11 with a one-line entry: "premise corrected — group
  scales do not preclude IMMA; funded as task-prefill-gap L3".

### L4 · CPU (aikit): the remainder, after re-measuring

- **Step 0, measure:** re-run `scripts/bench_peer_prefill.py --backend cpu` on the Mac, interleaved,
  1.5B int4 vs Ollama `num_gpu:0`, K ∈ {512, 1024, 2048, 3900}, on the current aikit. Replace §A's
  2.98×/1.80× row with the result. **Decision rule:** marginal ≤1.15× behind → close L4 with the
  row; ≥1.5× → the aikit brief's steps 1–2; between → step 1 only.
- **What is left inside the kernel is small and named.** The arm64 tile runs at 99% of a
  four-pipe issue ceiling for its µop count (aikit `docs/task-simd-audit.md`, S-01 read-back); the
  levers are S-05 (fold the −8 centering into the SDOT accumulator, 96→72 µops, counted 1.33×,
  bit-identical) and an M-block outer loop so the activation panel is not re-streamed from L2 per
  quad at large M (measure the panel traffic first). S-06 step 1 (parallelise the elementwise
  loops, goinfer side) is bit-identical and independent. The i8mm/SMMLA GEMM is **not available on
  the M1 Pro** — the CPU reports `asimddp` and no `i8mm` — and stays gated on an M2+/Graviton box
  as the audit already has it.
- **The parity-class option** (a lane-per-weight-row interleave with a vectorised 4-output fold)
  exists and is cheaper per MAC, but it changes the M=1 kernel too and is a goldens re-baseline;
  it is only on the table under the §3 contract and is not started without that decision.
- The brief for the aikit session is separate (issued 2026-09-05 with this doc).

## 5. Sequencing

| order | item | why here |
|---|---|---|
| 1 | §3 gate harness, CUDA + Metal, exact path as oracle | every default flip below needs it; it is also the fidelity column W2/W4 need |
| 2 | L1 Metal flip + W2 Mac cells | cheapest, largest, on the machine most used; produces the first Metal prefill peer row |
| 3 | L4 step 0 (re-measure Mac CPU) | one afternoon; decides whether L4 exists |
| 4 | L2 CUDA fused attention | the growing term; independent of L3; can run in parallel with 2–3 |
| 5 | L3 CUDA tensor-core GEMM | the constant term; the larger kernel project |
| 6 | W2 CUDA re-run with both landed, then the W4 replay | the row this doc is for |

L2 and L3 are independent kernels on independent categories; measure each against the exact
path alone, then together, so the end-to-end number has an attribution.

## 6. Projection — counted, not measured, and labelled so

Dense 1.5B int4, RTX 2070 SUPER, from the `TestPrefillDecomp` categories
(`measurements/cuda-prefill-attention-share-2026-09-01.md:23`, `:25`) and the re-anchor's TTFT
cells. **Do not quote these as results.**

| K | today (catSum) | L2 alone (attn ÷4) | L3 alone (gemv ÷4) | L2+L3 | Ollama TTFT − floor |
|---|---|---|---|---|---|
| 512 | 374 ms | ~334 ms (1.12×) | ~143 ms (2.6×) | ~103 ms (3.6×) | ~78 ms |
| 3900 | 5466 ms | ~3211 ms (1.70×) | ~3669 ms (1.49×) | ~1414 ms (3.9×) | ~560 ms |

Reading: with both landed, goinfer is inside ~1.3× of Ollama's overhead-free prefill at K=512 and
~2.5× at K=3900 — and on TTFT, ahead below ~2k tokens because of Ollama's floor. The crossover
moves from ~600 tokens to past the W4 band. Closing the last ~2× at depth is the f16-KV and
kernel-tuning tail, priced after these two exist. On Metal, L1 alone is the measured 3.9×; the
peer row decides the rest.

## 7. What this doc does not claim

- No number above for L2/L3 is measured; §6 is arithmetic on measured shares.
- The §3 gate may fail a backend. That is a result, and the default stays exact there.
- M26/M35 prefill is not moved by anything here (P20).
- The 7B cell is in the harness table and unswept for prefill; D7 is the second model for L2/L3's
  end-to-end rows, not a projection cell here.
- The Mac CPU post-tile figures in §1 are cross-session and are a prediction for L4 step 0.
