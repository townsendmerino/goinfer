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
> sequences the four levers it unlocks, cheapest first: **L1** flip Metal's batched prefill on —
> **re-run 2026-09-05 against a real f32-activation reference: neither model ships on the decision
> set, on two DIFFERENT narrow criteria, not a shared cause — see §3.1 and §4 L1**; **L2** a fused
> attention kernel on CUDA; **L3** a tensor-core int4 GEMM on CUDA that keeps group scales; **L4**
> the remaining CPU items, which now live in aikit's SIMD audit and are the smallest prize.
>
> **Status: L1 gate RE-RUN 2026-09-05 against a reference (§3.1) — CLOSED, does not ship, but not
> for the reason the first-form gate suggested.** `measurements/prefill-gate-l1-ref-2026-09-05.md`:
> both Metal arms (exact sequential, fast batched) scored against a CPU f32-activation reference,
> teacher-forced on the reference's own tokens, paired per prompt. **S fails its decision set at
> K=1024** (fast's agreement trails exact's by 1.4pt, just past the 1.0pt tolerance); **D7 fails at
> K=256** (fast's hard-flip count, 17/640, exceeds exact's 14/640) — opposite ends of the K range,
> opposite models, different criteria, both misses narrow. The one consistent finding across all
> five cells measured: **fast's mean KL divergence from the reference is lower than exact's every
> time** — the continuous measure never favors exact, only the two discrete pass/fail thresholds
> occasionally do, by small margins. This is not the "fast is worse" the first-form gate implied,
> and not the "fast is closer, more so on D7" §3.1 predicted — it's real, narrow, cell-level noise
> around a boundary where the two arms are genuinely close. Metal's default stays sequential;
> `--metal-fast-prefill` stays the disclosed opt-in; Phase 2 (`--exact-prefill`, the default flip)
> does not proceed from this doc's plan. L2/L3/L4 remain SCOPED, nothing built. goinfer `3b20f74`
> (scoped) / `6022b29` (first gate) / `3ab5230` (first reading, withdrawn by §3.1) / `556523a`
> (re-run code) / this revision (re-run result). Path:line citations were taken at `3b20f74`;
> `scripts/queue_citation_lint.py --update` re-indexes them.
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
GEMM on `mma.sync` accumulates each 32-wide group in int32 (two `m8n8k16.s8` instructions on sm_75), then
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
**both** the fast path and the exact path are scored against a **reference with f32 activations**
(§3.1 says which), and the fast path is not further from it than the exact path is:

| check | bar | why this one |
|---|---|---|
| seed-logit argmax agreement, each arm vs the reference | the 3% near-tie parity rule (`decoder.NearTieArgmaxForTest`, the same rule CUDA decode is held to CPU by — `benchmarks.md:611`); **the fast arm's hard-flip count may not exceed the exact arm's** | the bar the tree already trusts, applied to both arms so the exact path's own cost is visible |
| teacher-forced top-1 agreement over 64 continuation positions, each arm vs the reference's own greedy continuation | paired per prompt: fast ≥ exact − 1.0 pt on the cell mean, and fast ≥ exact on ≥ half the prompts | the fidelity column `task-peer-benchmarks.md` §4 is building anyway; the exact arm sets the bar because it is what ships today |
| mean continuation KL(reference ‖ arm) | fast ≤ 1.1 × exact on the cell mean | the number a reader will ask for, with the exact arm's value beside it |
| greedy stream divergence rate | **reported, not gating** | it is what gated Metal; it measures reproducibility, not quality |

**Floor.** As `--cpu-fast-attention` already does (exact below 512 prompt tokens, because the win
scales with length and the divergence does not — `internal/serveapp/main.go:320`), each backend's
fast path engages only above a prompt-length floor set where its measured win starts; short prompts
stay exact at no cost.

**Floor vs decision set — an inconsistency in this section, resolved 2026-09-05 by explicit
decision, and recorded rather than quietly applied.** The clause above mandates a floor. The gate
below names a decision set spanning **K ∈ {256, 1024, 3900}** — and 256 is below any floor either
backend would plausibly set, since the CPU precedent this clause cites is 512. So §3 as written
asks a backend to pass a fidelity cell at a depth its own Floor clause says the fast path will
never run at. On CUDA that is exactly what happened: the combined fast path FAILED at K=256 and
SHIPPED at 512, 1024 and 3900 (`measurements/prefill-l2l3-phase3-2026-09-05.md`).

The resolution taken, on the operator's decision: **the decision set is the cells at or above that
backend's floor.** CUDA's floor is **512**, and the honest part is that 512 was not interpolated —
§3's standing cells are 256 and 1024, so a K=512 reference was generated and the gate run there
specifically to avoid resting a floor on a fidelity result nobody took.

**What this does not license, stated so a later reader can check it.** The K=256 measurement is not
withdrawn; it stands on the record as a real failure of the combination at that depth. No bar was
relaxed and no criterion was reinterpreted. Moving the floor DOWN requires a passing gate cell at
the new depth, not an argument that the curve looks smooth. And this was a change made AFTER seeing
results — defensible because the Floor clause predates them and the CPU precedent lands on the same
number, but a reader should weigh it knowing the order.

A backend that fails the gate keeps the exact default and the fast path stays opt-in with the
failure recorded beside it. The gate runs as a heavy test per backend (`GOINFER_HEAVY_TESTS=1`),
with the reference logits computed once per (model, K) and stored, so it is a regression gate
afterwards, not a one-off — and the exact-vs-reference half of it is the first fidelity number
the tree has for that backend's shipped path, which the peer matrix needs regardless.

### 3.1 Correction, 2026-09-05 (later the same day) — the oracle was wrong, and it was this doc's error

The first version of the table above scored the fast path **against the exact path** and called any
non-near-tie disagreement a failure, on the argument that this is "the same bar the tree already
trusts for a non-identical backend" (CUDA decode vs CPU). That argument does not transfer, and the
gate it produced cannot answer the question it was built for:

- **CUDA-decode-vs-CPU compares two implementations of the same numerics** — W4A8 on both sides —
  so a disagreement there is a defect signal. **Fast-vs-exact prefill on Metal compares two
  different numerics of the same model.** The exact (decode) path quantises activations to int8 per
  row before every GEMV (`rmsnorm_quant` → `gemv_w4a8_*`, `metal/model.go:460`); the batched path
  keeps them in f16 and dequantises the int4 weights to f16 in-kernel (`metal/prefill.go:12`–`:13`).
  They are guaranteed to disagree. The measurement in `measurements/prefill-gate-l1-2026-09-05.md`
  is a correct measurement of *how much* — it is not evidence about *which arm is wrong*.
- **The prior is that the exact arm is the lossier one.** Per-row int8 activation quantisation is the
  known-lossy step in W4A8 (the LM head's move to W8A8 was precision-gated at 1.5% argmax flips on
  its own, `docs/task-w4a8-neon-bandwidth.md`), its error grows with model size as activation
  outliers grow, and the run shows exactly that shape: D7's fast-vs-exact KL is 6–7× S's at every
  depth, with no depth dependence. A defect in the batched path would not be expected to scale with
  model size and not with K; an activation-precision difference would.
- **A zero-hard-flip bar over 640 positions is a bar the exact path itself would likely fail**
  against an f32-activation reference. The CUDA decode gate applies "0 hard fails" over ~10 argmax
  checks of one forward, not 640 continuation positions.

**What replaces it.** The reference is the CPU backend on the same GGUF with f32 activations —
`Options{Backend:"cpu", Quant:""}` (f32 weights) where the model fits, else `Quant:"int8"`
(weight-only per-row int8, f32 activations; `decoder/model.go:153`). Both Metal arms share identical
int4 weights, so the weight-requantisation error is common-mode and the comparison isolates the
activation path. The reference's own greedy 64-token continuation is the teacher-forced token
stream for both arms. Reference logits are computed once per (model, K) in a separate process and
written to disk, then the two Metal arms run in one process and are scored against them — this also
keeps a 7B CPU reference and a 7B Metal resident from sharing 16 GB at the same moment. The decision
is paired and relative (table above): the fast path ships as the default if it is at least as close
to the reference as the path that ships today. If it is *closer* — which is the prediction — the
run is also the first evidence the tree has of what W4A8's activation quantisation costs on the
Metal decode path itself, which is a separate finding to file, not to act on here.

**Pre-registered for the re-run:** the decision set is K ∈ {256, 1024} on both models — S with the
f32 reference (1.5B f32 is ~6 GB, fits), D7 with the int8 weight-only reference (the f32 7B does
not fit 16 GB; the 2026-09-04 kernel panic on this machine came from a model that did not fit).
The 2026-09-05 run showed no K dependence in either arm's distance, so depth is a confirmation
axis, not a decision axis: S at K=3900 runs after the decision, as one confirmation cell. The
reference is the sequential CPU per-token forward (`ForwardForTest`, the seam the GPU parity gates
already use), so it carries the exact f64-accumulating attention as well as f32 activations. Prediction, written before
running: fast closer than exact on every cell, by a margin that grows from S to D7. If the
prediction fails on D7 but not S, the batched attention or the f16 weight dequant is the suspect,
and the next probe is per-layer, not per-model.

**What the contract does not touch.** Decode (unchanged on every backend), speculative verify
(exact kernels, lossless contract intact), and the parity gates (which run the exact path). The
change is confined to prompt ingestion, which is why `--exact-prefill` is a complete undo.

## 4. Levers, cheapest first

### L1 · Metal: make the batched prefill the default — RE-RUN AGAINST A REFERENCE, CLOSED, does not ship

- **What exists:** `PrefillLast` on Metal is a working f16-MMA batched prefill, measured 3.93–4.56×
  over sequential at P=128…2048 and 3.74× end to end on a real 1450-token prompt through the
  server (`ollama-chase.md:1578`–`:1611`). It is declined unless `GOINFER_METAL_BATCHED_PREFILL=1`
  (`metal/backend.go:248`), which `--metal-fast-prefill` sets (`internal/serveapp/main.go:372`).
- **First form, WITHDRAWN (2026-09-05, `measurements/prefill-gate-l1-2026-09-05.md`):** scored
  fast against Metal's own exact path as the oracle. §3.1 found that comparison cannot tell a
  defect in the fast path from the exact path's own int8-activation loss — both are quantisations
  of the same model. The numbers stand as a measurement of fast-vs-exact distance; the "gate
  FAILED" verdict first attached to them is withdrawn.
- **Re-run against a reference (2026-09-05, `measurements/prefill-gate-l1-ref-2026-09-05.md`):**
  both Metal arms scored against a CPU f32-activation reference (S: f32 weights; D7: int8
  weight-only — the pre-registered fallback, since D7's f32 weights don't fit 16GB — worked),
  teacher-forced on the reference's own tokens, paired per prompt, under §3's amended gate.
  **Neither model ships its decision set (K ∈ {256, 1024}), and the two failures don't share a
  cause:** S fails at K=1024 (fast's agreement trails exact's by 1.4pt, just past the 1.0pt
  tolerance); D7 fails at K=256 (fast's hard-flip count, 17/640, exceeds exact's, 14/640) — opposite
  ends of the K range, opposite models, different criteria. S's K=3900 confirmation cell and D7's
  K=1024 both ship cleanly. **Across all five cells measured, fast's mean KL divergence from the
  reference is lower than exact's every time** (4–15% lower) — the continuous measure never favors
  exact; only the two discrete thresholds occasionally do, by narrow margins. This is neither the
  first form's implied "fast is worse" nor §3.1's predicted "fast is closer, more so on D7" — it
  reads as real cell-level noise around a boundary where the two arms are genuinely close, not a
  clean directional result either way. §3.1's own fallback (isolate per-layer whether attention or
  weight dequant is the cause, "if the prediction fails on D7 but not S") doesn't apply: the
  prediction failed once on EACH model, not uniformly on one, so there is no clean model-level
  split for a per-layer probe to explain.
- **Band (TTFT, S at K=2048, vs the shipped default):** ≥3× ships (the measured figure is 3.9×);
  <2× means the serve path is eating it and the item reopens as a serving investigation. The
  first-form run measured **3.93× at K=256, 3.12× at K=1024, 2.02× at K=3900** — clearing the
  ship line at K≤1024, ambiguous at the band's own cell (K=2048 interpolates to ~2.7×), and never
  reaching the <2× reopen trigger. Not re-measured in the reference re-run (that run scores
  fidelity, not speed, and both arms pay extra teacher-forced Forward calls that would make its
  timings incomparable to a clean TTFT measurement). The decay itself is not a serving effect: per-
  token batched cost rises from 3.4 ms at P=256 to 9.4 ms at P=3900 because `attention_prefill` is
  one threadgroup per (row, head) reusing the decode math (`metal/prefill.go:210`–`:214`) — the
  same O(K²) term §2.2 names on CUDA — while the sequential arm pays it too but hidden under a
  ~19 ms/token GEMV. So the speedup L1 delivers is the GEMM's and it shrinks as the attention share
  grows; that is the *next* item's evidence, below.
- **Disposition:** Metal's default stays the sequential path and `--metal-fast-prefill` /
  `GOINFER_METAL_BATCHED_PREFILL=1` remains the disclosed opt-in. Phase 2 (flip the default, build
  `--exact-prefill`) does not proceed from this doc's plan — not because fast was shown to be
  worse (it wasn't, on the one measure that never flipped), but because the pre-registered gate, as
  amended, doesn't clear on either model's full decision set. Re-opening would want either a larger
  prompt sample (the two narrow misses may not survive n>10) or accepting the KL evidence over the
  discrete thresholds — neither decided here. **W2's Mac cells are still not run**
  (`scripts/bench_peer_prefill.py` still has no Metal backend option) — that remains open
  regardless, since `benchmarks.md` still has no Metal prefill peer row at all.
- **Then, and now sized:** a `simdgroup_matrix` flash attention for `attention_prefill` — the Metal
  twin of L2. The 2026-09-05 TTFT curve prices it: at K=3900 the batched arm spends ~6 of its 9.4
  ms/token above the flat ~3.4 ms the GEMM costs, so a fused attention that held per-token cost
  near flat would take the S/K=3900 speedup from 2.02× to ~5× and is worth more than L1 itself at
  the depths W4 cares about. Independent of the fidelity question — it changes speed, and its own
  gate is the same §3 gate. Scoped after the L1 re-run and W2's Mac row exist, because that row
  says how much of the remaining Metal gap is attention.

### L2 · CUDA: fused (FlashAttention-style) prefill attention

- **Kernel shape:** one block per (head, 64-row query tile); K/V streamed in 64-key tiles through
  shared memory, converted f32→f16 on load (the resident KV stays f32 — this is not a KV-quant
  item, and KV-quant was refuted as a decode lever on this card); `mma.sync.aligned.m16n8k8`
  f16×f16→f32 for QKᵀ and PV — **the sm_75 shape**; `m16n8k16` needs sm_80, and llama.cpp's
  `mma.cuh` makes the same Turing/Ampere split; online softmax with the running max/denominator in registers; causal
  mask per row, sliding window per row, GQA head grouping, and the gpt-oss `sinks` term — every
  seam `attn_batched` handles today (`cuda/prefill_batched.cu:156`–`:162`) must be handled here,
  with a test per seam. `attn_batched` stays as the exact path.
- **Not bit-identical**, by the rescale in the online softmax; this is the P19 category, on the
  backend P19 explicitly left "still not tested".
- **Band (dense 1.5B int4, K=3900, end to end TTFT, `TestPrefillTTFT` extended to 3900):**
  attention is 55.0% there, so a perfect kernel caps at 2.22×; a 4× kernel gives ~1.7×. The
  kernel's own ceiling is far above 4×: the causal QKᵀ+PV work for this model at K=3900 is ~0.66
  TMAC (28 layers × 12 heads × 128 × K²/2 × 2), i.e. ~65 ms at 20 TFLOPS of a ~70 TFLOPS f16
  tensor peak, against the 3.0 s measured — the current kernel is latency-shaped, not
  compute-shaped, so a fused kernel at even 5% of tensor peak is >10× on the category.
  **Ships at ≥1.4×; ambiguous 1.15–1.4×; parks below 1.15×.** At K=512 the same kernel is worth at
  most 1.17× and is not the cell that decides it. Kernel-level ratio recorded separately via
  `TestPrefillDecomp`'s attention category so the two numbers cannot be confused.
- **Depends on:** nothing. Can start first. The chunked pass (`prefillChunked`, ≤512 rows) is
  unaffected: the kernel reads the positional KV the same way.

### L3 · CUDA: tensor-core int4 GEMM with group scales

- **Kernel shape:** `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` — **the sm_75 int8 shape**
  (`m16n8k16`/`k32` need sm_80; llama.cpp's `mma.cuh` issues two `m8n8k16` on Turing for the same
  reason). Activations are already per-row int8 packed four per int word with `aScale[m]`, which is
  the A-fragment register format as it stands. The B fragment wants four consecutive k of one
  weight row per register — and the pack-time nibble permutation (`permuteFast`, `cuda/kernels.go`)
  already puts elements 8j…8j+3 in the low nibbles of word j and 8j+4…8j+7 in the high nibbles (it
  is what lets the dp4a kernel pair `lo`/`hi` with two consecutive activation words), so the same
  two-instruction unpack yields B fragments straight from the packed word with no shared-memory
  transpose; verify against `kernels.go` before relying on it. Per 32-wide group: two MMAs into a
  zeroed int32 C fragment, then `facc += float(c) * gs[n][g]` in fp32 per element (the C fragment's
  two elements per thread are two adjacent n, so two scale loads per thread per group); per row at
  the end: `* aScale[m] + bias`. Stage the activation tile in shared memory for reuse across the
  block's N-slices; read weights once per block. `gemv_w4a8_rn` (what `bGemvB` launches today)
  stays as the exact path and as the M<16 path (tensor cores lose below a warp's worth of rows).
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

## 6. Projection vs measurement — the counted numbers, and what they got right

**L2 and L3 are now BUILT AND MEASURED** (`measurements/prefill-l2l3-phase{1,2,4}-2026-09-05.md`),
so this section is no longer a projection. The original counted table is kept because comparing it
against the measurement says something useful about the method the bands were set with.

### 6.1 What was projected, and what happened

Dense 1.5B int4, RTX 2070 SUPER. Projection from measured category shares; measurement is
end-to-end `TestPrefillTTFT`, same box, same session as its baseline.

| cell | §6 projected | **measured** | error |
|---|---|---|---|
| K=512, L2 alone | ~334 ms (1.12×) | **329.9 ms (1.13×)** | +1.2% |
| K=512, L3 alone | ~143 ms (2.6×) | **130.4 ms (2.85×)** | +9.7% |
| K=512, L2+L3 | ~103 ms (3.6×) | **90.5 ms (4.10×)** | +13.8% |
| K=3900, L2 alone | ~3211 ms (1.70×) | **3213 ms (1.70×)** | **−0.1%** |
| K=3900, L3 alone | ~3669 ms (1.49×) | **3617 ms (1.51×)** | +1.4% |
| K=3900, L2+L3 | ~1414 ms (3.9×) | **1393 ms (3.91×)** | +1.5% |

**Amdahl on measured category shares held, and held conservatively.** Every projection
over-estimated the time — it under-promised — and at K=3900 all three arms landed within 1.5%,
L2-alone within 0.1%. The loosest cell is K=512 L2+L3 at 13.8%, because L3 beat the 4× the
projection assumed for it.

### 6.2 The peer extrapolation on top of it did NOT hold, and that is a separable claim

§6 also read: *"with both landed, goinfer is inside ~1.3× of Ollama's overhead-free prefill at
K=512 and ~2.5× at K=3900."* Measured at the deepest interval (`prefill-l2l3-phase4-peer`):
**3.16× behind** on the 1.5B, **1.89×** on the 0.5B.

So the internal arithmetic was accurate and the peer claim built on it was **optimistic by ~25% at
depth**. The two are separable and only one failed: the internal projection extrapolated nothing,
while the peer claim assumed goinfer's marginal cost would flatten to meet Ollama's flat curve. It
did not — goinfer's marginal still RISES with K (1.5B fast: 0.0402 → 0.1173 ms/token across the
ladder). That residual curvature is the remaining O(K²) attention term, and Phase 1's `ncu` capture
prices the headroom: the fused kernel runs at **1.72% of tensor peak and 12.6% occupancy**, so it
is not a floor.

### 6.3 Where CUDA prefill actually stands

| cell | before | after (both levers) |
|---|---|---|
| e2e prefill, 1.5B K=3900 | 5.451 s | **1.393 s (3.91×)** |
| e2e prefill, 1.5B K=512 | 371.5 ms | **90.5 ms (4.10×)** |
| marginal vs Ollama, 1.5B, deepest | 12.1× behind | **3.16× behind** |
| marginal vs Ollama, 0.5B, deepest | 14.5× behind | **1.89× behind** |
| TTFT crossover vs Ollama, 1.5B | ~K=600 | **past K=2048** |
| TTFT crossover vs Ollama, 0.5B | ~K=1024 | **off the measured ladder — ahead everywhere** |

On Metal, L1's re-run against a real reference is CLOSED and does not ship (narrow, mixed —
`measurements/prefill-gate-l1-ref-2026-09-05.md`); the Metal fused attention is what would move it,
and the Metal peer row is still unmeasured.

## 7. What this doc does not claim

- No number above for L2/L3 is measured; §6 is arithmetic on measured shares.
- The §3 gate may fail a backend. That is a result, and the default stays exact there — **this
  happened for real on Metal, 2026-09-05, against a real reference** (§4 L1,
  `measurements/prefill-gate-l1-ref-2026-09-05.md`), on both models. The first-form run's own
  verdict is withdrawn (it was scored against an oracle that cannot distinguish a defect from the
  exact path's own loss, §3.1) and its numbers are kept as the fast-vs-exact distance they measure;
  the re-run against a real reference is what actually decided this, and the decision is "does not
  ship," not "fails cleanly" — see the KL-vs-discrete-threshold split above.
- Nothing here claims the batched Metal path is *better* than the sequential one, and nothing here
  claims it is *worse* either. §3.1 predicted "fast closer than exact on every cell, margin larger
  on D7" and that prediction was wrong as stated — written down before the re-run specifically so
  it could be wrong in public, and it was. What the re-run actually found (KL favors fast
  everywhere; two discrete criteria each fail once, on different models, different reasons) is not
  the same claim as either "better" or "worse," and this doc does not round it to either.
- M26/M35 prefill is not moved by anything here (P20).
- The 7B cell is in the harness table and unswept for prefill; D7 is the second model for L2/L3's
  end-to-end rows, not a projection cell here.
- The Mac CPU post-tile figures in §1 are cross-session and are a prediction for L4 step 0.
