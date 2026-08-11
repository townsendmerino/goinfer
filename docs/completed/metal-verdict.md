# The Metal verdict — where the Ollama gap actually is, and the decision that closes it

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


**Status:** consolidated verdict, 2026-08-04. Everything here is measured, bounded, or explicitly
marked unmeasured; nothing becomes a plan of record until a campaign is opened. Consolidates the
Metal threads of `task-metal-cgofree-spike.md` (the 4 → 73.6 tok/s arc), `metal-decode-review-package.md`
+ `metal-decode-headroom-fable.md` (the diagnosis), `ollama-chase.md` §A1-Metal/§A2-Metal/§7/§D4
(the campaign and the fork), `benchmarks.md` §B3, and `gemma4-resident-scope.md` §9c (the 26B).
Nothing below is new measurement — this doc exists so the standing question is answered once, in
writing, instead of re-derived per session.

**The standing question:** why doesn't the Metal backend get close to Ollama on the Mac, and does
it need a rewrite?

**The short form:**

- "The gap" is **three different problems**, not one. Short-context dense decode is at ~0.9× the
  peer and at its measured ceiling. Long-context decode is the real gap (~2×+ behind, growing with
  depth) and it is **structural**: the peer's fix is the one mechanism the bit-identity contract
  forbids. Prefill and the 26B are different problems again (peer-class batched GEMM; capacity).
- The binding constraints are the **hardware** (no DP4A/int8-dot, no grid-sync, ~14 cores, a
  per-dispatch floor) and the **bit-identity contract** (pinned reduction widths, whole-fold
  accumulation order). A rewrite changes neither. **Verdict: no rewrite** (§7).
- The lever is a **decision** — the §7 fork of `ollama-chase.md`, Metal edition (§6) — plus one
  evening of peer measurement that is currently missing (§5 M0).

**Rig for every on-box number:** Apple M1 Pro, 14-core GPU, 16 GB, ~200 GB/s SoC fabric (~150–170
GPU-reachable), macOS 26.5.2, MSL 3.1, `metal/` cgo-free via purego-objc. **Peer:** Ollama-Metal —
**0.32.0 at every existing measurement**; current is **v0.32.5** (engine = upstream `llama-server`
since Ollama PR #16031, 2026-05-29 — so Ollama-Metal-on-GGUF *is* llama.cpp's Metal backend).

> **⚠ Staleness, stated up front (this is itself a finding).** Every goinfer-vs-Ollama Metal
> number in the repo predates BOTH the peer's current version AND most of the August decode work
> (f16 scales, encode-ahead, half4 attention coalescing `994539c`). And there is **no Metal
> depth row for the peer at all** — the long-context gap, the one that matters, has never been
> measured on this platform. Per the standing rule ("peer version is part of the measurement";
> the §B2 re-anchor caught exactly this on CUDA), **no competitive Metal claim should be made or
> funded before M0 runs.**
>
> **✅ M0 RAN (2026-08-04) — see §5.** Peer re-anchored at v0.32.5 on this M1 Pro; FA confirmed
> on-box; the missing depth rows are filled. Result: FA-on peer is nearly flat (86→77 tok/s over
> 128→3663), the depth gap is **larger than estimated** (~1.34× @128 growing to **~4.2× @4000**), and
> the verdict's structural thesis holds. The depth-row claims below are now measured, not absent.

---

## 1. Where we actually stand (every number with its harness)

Four measurement sets exist; they use different methods and must not be conflated.

| # | what | goinfer | Ollama-Metal | provenance | stale? |
|---|---|---|---|---|---|
| 1 | 1.5B q4_K_M, **server-to-server wall clock**, short prompts | **~54 tok/s** | **~73** (v0.32.5) | `benchmarks.md` §B3, **re-anchored 2026-08-04** | **no — both current** (goinfer `38e5cd7` W4A8, peer v0.32.5 FA-on) |
| 2 | 0.5B q4_K_M, same method | **~116** | ~121 (v0.32.5) | §B3, **re-anchored 2026-08-04** | no — both current |
| 3 | 1.5B **decode-only, best-of-40 warm**, W4A8 lane | **73.6** | 83.3 (0.32.0, best-of-3 warm, spike Step 0) | `task-metal-cgofree-spike.md` (f16 scales 71.4 → encode-ahead 73.6) | goinfer side current; **peer side stale** |
| 4 | 1.5B decode **vs KV depth** (resident, depth-A/B harness, best-of-40, post-`994539c`) | 63.8 / 39.8 / **28.4** / 18.5 at 128/1024/2048/4000 | **85.2 / 79.1 / ~80 / 77.5** (M0, 2026-08-04, v0.32.5, FA-on) | goinfer `ollama-chase.md` §A1-Metal; peer = **M0** (§5) | both current |

Supporting facts that are current and not in dispute:

- **GPU-busy effective bandwidth 73.6 GB/s** (cgo-free GPU timestamps, Step 0: wall 14.0 ms =
  GPU 13.0 + host bubble 0.9), against the incumbent's **~75–83 GB/s** band on this rig (83.3
  tok/s × ~0.9–1.0 GB/token). Nobody demonstrably exceeds ~85 GB/s at batch-1 on this rig for
  this model class.
- Traffic 868 MB/token after f16 scales (was 965); the −10% traffic cut left GPU-busy **flat**
  — the issue-bound confirmation.
- Depth slope: the shallow decode floor is **dispatch-count-bound** (~337 dispatches/token ×
  ~3.8 µs GPU-side floor ≈ 1.3 ms irreducible; `tax_test.go`); the depth term is the attention
  kernel's distinct K/V DRAM reads (§3).

**The honest one-line position:** at ~0.9× the (stale) native peer on short-context dense decode —
effectively at this chip's measured ceiling for the current architecture — and ~2×+ behind at real
context depth, where the peer has never been measured on this box and the deficit's cause is
structural, not implementational.

---

## 2. The three gaps are three different problems

**2a. Short-context dense decode — CLOSED, at ceiling.** 73.6 vs 83.3 (0.88×, stale peer). The
whole tuning arc ran: 4 → 30 (coalescing) → 56 (attention rewrite) → 68.9 (Stage A) → ~70 (encode
trims) → 71.4 (f16 scales) → 73.6 (encode-ahead). The residual ~12% is fully attributed: (a) no
DP4A ⇒ the int4 nibble-unpack + int-MAC costs ~10–12 issue slots/weight (ALU-issue ceiling
~120–145 GB/s, observed 84–105 — and the lm head streams at 84 GB/s **with 19,000 threadgroups**,
falsifying every occupancy/latency story); (b) the per-dispatch floor. The GEMVs already run in
the peer's own effective-bandwidth band — **there is no large unclaimed dense-decode pool on an
M1 Pro for anyone.** Do not fund work here; do not let an agent "optimize the GEMV" again.

**2b. Long-context decode — THE gap, and structural.** Depth curve above: 64 → 28.4 tok/s by
2048. Cause measured on-box (§A2-Metal): the decode attention's depth term runs at **~17 GB/s
(8.6% of nominal) with ~0.7% ALU** — DRAM-latency-bound — and the collapse probe (pin every K/V
read to key 0; same loop/ALU/threadgroup traffic, zero distinct DRAM) took all-28-layer attention
**21.5 → 5.3 ms**: 75% of attention at depth is the **6× GQA-redundant K/V read** (12 query heads
re-reading 2 KV heads). The peer does not pay this the way we do: current llama.cpp runs **flash
attention by default, on Metal included** (D4 confirmed at source on CUDA — `resolve_fused_ops:
Flash Attention enabled`; Metal default-on externally corroborated, on-box verification is an M0
item). FA fills the machine by splitting the KV axis inside one kernel and merging partial
softmaxes by online rescale — **a reordering of the float sum, which the bit-identity contract
forbids** (reduction width and fold order are pinned: `tgReduce*`, `ee4e01f`). Every bit-identical
approximation of that shape was built and measured on this box and lost (§4). This is not a
kernel-skill gap; it is the §7 asymmetry — **Ollama has no bit-exactness contract at all.**

**2c. Prefill and the 26B — different problems again.** Prefill: the f16 `simdgroup_matrix` MMA
path (P1–P5) took 140-token TTFT 2034 → 543 ms (**3.7×**) and ships; but the peer's prefill is a
years-tuned batched GEMM stack + FA, `metal/`'s `PrefillLast` is dense-uniform only (declines MoE
and Gemma-4 geometry — `ollama-chase.md` §C3), and **no Metal prefill-vs-peer number exists**
(unmeasured — M0 candidate, second priority). The 26B-A4B: **capacity, not kernels** — ~14.3 GB
resident vs ~12.5 GB practically usable (`recommendedMaxWorkingSetSize` ≈ 10.6 GB); the scoped
path is the no-copy file-backed paging probe (`newBufferWithBytesNoCopy` over the mmap'd `.giw`),
expected **15–25 tok/s** via page-cache re-faults (~131 MB/token), written down before the run in
`gemma4-resident-scope.md` §9c. Nobody is fast in that regime (fieldfare: 5.1–6.3 tok/s on its
own constrained 8 GB rig — different silicon, a floor not a comparison). No kernel work points at
this; the fix is memory or a model that fits.

---

## 3. The three walls (why the remaining gap is not an engineering search)

**Wall 1 — hardware (M1 Pro).** No DP4A/int8-dot ⇒ integer decode is issue-bound (proven twice:
int4 halved bytes for zero gain; f16 scales cut traffic 10% with GPU-busy flat) — and the same
property **killed speculation on Metal**: the batch-k verify does not amortize a MAC-bound kernel
(`batchk_test.go`: T_k ≈ k·T_1; a win requires accept-rate a > k, impossible — so CUDA's D1
[1.53×] does **not** port). No cooperative launch / grid-sync ⇒ the true megakernel is
architecturally impossible; the sync-free redundant-recompute variant measured **70 → 52** and was
reverted. ~14 cores + serial hazard-tracked encoding ⇒ a ~3.8 µs/dispatch GPU-side floor across
~337 dispatches — the ~70 tok/s shallow "dispatch floor" — and ICB is ruled out (only ~0.5 ms of
the ~2.4 ms overhead was Go-side encode). UMA ⇒ readback fusion is throughput-neutral (fused
argmax: bit-exact, kept for the API, zero speed).

**Wall 2 — the bit-identity contract.** Cross-thread float sums are wired to a pinned reduction
width (`tgReduceNorm=256`, `tgReduceAttn=128`; ground rule since the staged-kernel divergence at
nKeys>256) and to whole-fold accumulation order. Consequences, all measured: the decode attention
cannot go wide (12 threadgroups × 128 threads ≈ 1.5k threads on a machine that wants tens of
thousands — most of the GPU idles while DRAM latency is exposed serially); the KV-split needs a
kernel-launch barrier (+56 dispatches/token) instead of FA's in-kernel online merge; K/V dedup
needs group co-location (occupancy death). FA's online-softmax rescale, KV quantization, and
f16 accumulation are all outside the contract by construction. **The contract is a stated trade,
not a deficiency — but at long context on this hardware it prices out the only mechanism that
works.**

**Wall 3 — the peer asymmetry.** Ollama-Metal is llama.cpp's Metal backend: years of tuned MSL,
FA default-on, q8/q4 KV available, no bit-exactness claim between any two paths (§7, confirmed at
source in D4). It lands at 75–83 GB/s effective on this rig — the same band goinfer already
reaches. The peer's edge at depth is **contractual freedom, not kernel quality.**

Method rules carried forward unchanged: profile the unit before designing the fix (the
five-attribution record); total is the verdict line; a synthetic must reproduce the pressure;
reduction width is part of the bit-identity contract; thermal control (interleaved repeats,
session-start run dropped); **peer version is part of the measurement.**

---

## 4. Refuted on Metal — do not re-propose (the consolidated ledger)

Every entry was built (or probed) and measured on this box. An agent session that proposes one of
these has not read this doc.

| lever | result | why it died |
|---|---|---|
| GEMV Stage B (tile-repack, lane-per-row, broadcast) | gate/up 164→118 µs isolated, decode **flat**, −1 parity | not GEMV-compute-bound in the cold-weight regime; reverted |
| int4 over int8 (½ weight bytes) | ~0 | not bandwidth-bound (issue-bound) |
| Dispatch fusion 19→12/layer | ~0 | not (that kind of) dispatch-bound |
| Megakernel, redundant-recompute (12→9) | **70 → 52** | recompute of full-vector reductions dwarfs each tg's thin GEMV slice |
| Megakernel, true 1-launch/layer | not attemptable | **no grid-sync on Metal** |
| ICB | ~0 (setBuffers proxy recovered the ~0.5 ms Go side) | remaining overhead is GPU-side per-dispatch latency |
| `commandBufferWithUnretainedReferences` | no change | bubble is raw encode msgSends, not retain/release |
| Fused argmax lm head (as a *speed* lever) | neutral | UMA readback is a ~30 µs zero-copy view; kept as API |
| Speculative decode (batch-k verify) | T_k ≈ k·T_1 (`gemv_w4a8_sa_bk`, k=4 ties, k=8 regresses) | MAC-bound verify ⇒ win requires a > k, impossible; CUDA's D1 does not port |
| Split-KV decode attention (bit-identical, 3-dispatch port of the CUDA win) | 0.94–0.98× at every depth; reverted | Metal decode attention is **not occupancy-starved** (32×/4× threadgroup increases moved nothing); +2 dispatches/layer on a dispatch-bound path is the whole regression |
| Grouped K-dedup, 3-dispatch | 0.45→0.92× | scores pass won (10 → 4.66 ms) but the V-fold + dispatch tax lost it |
| Grouped K-dedup, 2-dispatch | 0.92× | same |
| Threadgroup-staged K/V, 1 dispatch (the cleanest dedup) | **0.23×** | dedup requires 1 tg per KV head = 2 threadgroups on ~14 cores; **dedup and occupancy are in direct opposition on Metal** |
| WILLNEED readahead / F_NOCACHE / concurrent preads / per-encoder residency scoping / N-knob (26B paging line) | all null or negative | see `ollama-chase.md` §10 |
| WebGPU-on-Metal as the substrate | 31.9 tok/s ≈ CPU (32.66) | measured degenerate — a wash with the pure-Go floor |
| MPS / MPSGraph | not pursued | no clean purego-objc surface; doesn't speak the W4A8 packing; the "cuBLAS would not save you" logic |

Four independent confirmations (split-KV, two grouped variants, staged) that the DRAM-dedup prize
at depth is **real (~16 ms/token @2048) but uncapturable inside the contract on this hardware.**

---

## 5. Live levers (what is NOT refuted), in order

**M0 — re-anchor the peer on this box. ✅ DONE (2026-08-04).** Ollama **v0.32.5** (upgraded off the
stale 0.32.0 that every prior number used), local `qwen2.5-coder-1.5b-instruct-q4_k_m` GGUF, warm,
greedy, decode-only via `eval_count/eval_duration`, nonce-prefixed prompts to defeat Ollama's KV
prefix cache (the first cut mis-read depth until this was fixed).

- **FA verified on-box (the D4-Metal check):** llama-server is launched `--flash-attn auto` and the
  load log prints `resolve_fused_ops: Flash Attention enabled` for this qwen2 — *even though*
  `OLLAMA_FLASH_ATTENTION:false` (auto lets llama.cpp turn it on per-model, exactly as D4 found on
  CUDA). **The verdict's central assumption holds on Metal.** Peer ctx capped `-c 4096` (matches
  goinfer's `metalCtxCap`).
- **Peer depth curve (decode tok/s, best-of-2 warm):** 85.2 @128 · 79.1 @1024 · 80.7 @1953 · 77.5
  @3663 — **nearly flat (86→77), the FA signature.** goinfer (§A1-Metal): 63.8 / 39.8 / 28.4 / 18.5.
- **The gap GROWS with depth, and is STEEPER than the pre-run estimate:** ~1.34× @128 → 1.99× @1024
  → ~2.8× @2048 → **~4.2× @4000** (estimate was 1.15–1.25× shallow / 2.5–3× @4000). Shallow best-vs-
  best is milder — goinfer's decode-only spike harness (row 3, 73.6) vs peer 85.6 ≈ **1.16×** — so
  the shallow gap is 1.16–1.34× depending on goinfer harness (a ~15% goinfer harness discrepancy,
  row 3 vs row 4, worth its own note); the *depth* gap is unambiguous and larger than written.

**M0+ FA-off isolation (2026-08-04) — FA is causal, but only PART of the gap.** Re-ran with
`OLLAMA_FLASH_ATTENTION=0` (→ llama-server `--flash-attn off`, `flash_attn = disabled` in the log):
peer FA-off depth curve **82.4 / 85.5 / 69.3 / 56.2** (128/1024/1953/3663) vs FA-on **85.2 / 79.1 /
80.7 / 77.5**. So FA is causally confirmed — it flattens the curve (86→77 with, 86→56 without, ≈1.4×
at depth). **But the decomposition matters:** peer-FA-off (56 @3663) is still **~3× goinfer (18.5)**.
The ~4.2× depth gap is therefore **~1.4× FA × ~3× the peer's contract-free *non-FA* attention
parallelization** — and that second, larger factor is the contract itself: llama.cpp's plain (non-FA)
attention already parallelizes far better than goinfer's contract-bound kernel, which is exactly why
all four bit-identical attempts lost (§4). **The FA-off peer is the external witness that "the contract
prices out the mechanism" is bigger than FA alone** — it prices out the whole attention parallelization,
FA or not.

**Re-ranking:** the deficit is bigger than the doc assumed (4.2× not 3× @4000) *and* it isn't purely
FA. Consequences for M1: (a) **M-A concedes 4.2× at 4000**; (b) **M-B** — breaking the contract for an
FA-style throughput mode — captures the depth term up to goinfer's *own* shallow floor (~50–64, §6.1's
estimate stands; the residual peer 86-vs-goinfer-64 shallow is the §2a dense-decode ceiling, not
fundable); (c) a purely *bit-identical* depth win is now doubly refuted — §4's four attempts plus the
FA-off witness that even non-FA parallelization is beyond the contract's reach. **§B3 wall-clock re-anchor
DONE (2026-08-04):** goinfer's server re-measured vs Ollama v0.32.5, both from the identical local q4_K_M
GGUF — 0.5B **0.96×** (116 vs 121), 1.5B **0.74×** (54 vs 73); ratios held (was 1.03×/0.77× @0.32.0). The
short-prompt method isolates decode+serving; goinfer's batched-prefill decline (bit-identity) widens the
gap on long prompts (0.66× 1.5B @70-tok) — a serving trade, not a decode deficit. §B3 table refreshed.

**M1 — the fork decision (§6). ✅ DECIDED (2026-08-04): M-A — stay bit-identical, defer M-B.** Keep
the contract everywhere; accept the measured depth floor (4.2× @4000); make no competitive long-context
claim. **M-B is deferred, not rejected** — the FA-style throughput-mode design in §6 stands scoped and
available to fund if/when "usable at real context on a MacBook" becomes a goal; nothing about it is
retracted, it is simply not built now.

**M2 — inside the current contract, what little remains:** (a) the concurrent-dispatch encoder
with hand-placed barriers (headroom L6) — bounded small because the layer graph is a chain;
counter-gate it; (b) lm head at tg=128 Stage-B shape (review §6.3, the one clean unexplored
single-kernel win, ~+1.5–2.5 tok/s); (c) prefill continuation — the MMA prefill's own levers
(dequant lane width, flash-style MMA prefill attention) are **not** contract-blocked (the prefill
gate is last-token argmax + cosine, not byte-identity) and are where Stage-B-shaped work genuinely
pays. None of these move the depth gap.

**M3 — the 26B paging probe** (`gemma4-resident-scope.md` §9c): one run, expected 15–25 tok/s,
settles the fieldfare-comparison platform question either way. Independent of the fork.

**Watch item — Metal 4 tensor ops / `MTLDispatchTypeConcurrent` + untracked hazards** (macOS 26):
the forward-looking encoder design per the headroom doc; unassessed. Does not change Wall 2.

---

## 6. The fork, Metal edition (the decision this doc exists to force)

> **✅ DECIDED 2026-08-04 (maintainer): Option M-A — stay bit-identical for now; M-B deferred, not
> rejected.** The bit-identity contract holds across the whole Metal backend; the depth floor (4.2×
> @4000, M0) is accepted; no competitive long-context claim is made. M-B below remains a scoped,
> available lever — revisit it if long-context-on-a-MacBook becomes a goal. The rest of §6 is retained
> as the record of what M-B would be and what it would cost, so the deferral is informed, not amnesiac.

`ollama-chase.md` §7 states the CUDA fork (group scales vs tensor cores). The Metal fork is the
same shape with different content: **on Metal the contract prices out the depth mechanism, and
only the depth mechanism** — the GEMVs are already at peer band, so unlike CUDA there is no
"prefill 3×" on the other side of the fork; there is a long-context decode ~2×.

**Option M-A — keep the contract everywhere; accept the depth floor.** Ship the identity story
(`benchmarks.md` §B3 already states it: "Metal's story is not raw speed — cgo-free / no-Xcode").
Defensible claims: portability, correctness parity, ~0.9× the native peer at short context,
first `CGO_ENABLED=0` GPU path on the platform. Cost: the depth curve stands; no competitive
long-context claim, ever, and the doc saying so is this one.

**Option M-B — an opt-in throughput mode with no bit-identity claim** (the honest version of what
the peer does). Scope, deliberately minimal:

1. **FA-style decode attention** — one dispatch/layer, KV axis split *inside* the kernel across
   many simdgroups (fat threadgroups: fill the ~14 cores from 12 heads), partial `(m, l, acc)`
   merged by online rescale. This is exactly the D4-confirmed peer mechanism and exactly what the
   pinned-width contract forbids today. Bound on the prize (from the collapse probe, attention
   →5.3 ms at current parallelism): token @2048 ~35 → ~19 ms ⇒ **~28 → ~45–55 tok/s**, more if
   the wider launch also cuts the 5.3 ms — capped by the ~64 shallow floor. Unmeasured; step 1
   of M-B is measuring it.
2. **KV q8** (the §A3 lever) — halves the depth-term bytes on top of (1). **Family-gated:** Gemma
   is measured KV-precision-sensitive (f16 KV was catastrophic on the sandwich path — 0.64 vs
   0.92 cosine — which is why Gemma runs f32 KV and pays 2× depth bytes today); qwen is
   insensitive. Expect q8 to be a qwen-class lever and possibly a Gemma non-starter.
3. **Explicitly NOT in scope:** an f16/FMA GEMV rewrite (the GEMVs are already at the peer's
   effective band — no prize), tensor-core-style decode GEMM (no batch dim at M=1), speculation
   (Wall 1), anything on the 26B (capacity).

**Gates for M-B** (this is what keeps it honest where the peer is silent): default **off**; a
named flag (`GOINFER_METAL_FAST_ATTN=1` or a `--mode=throughput` umbrella); its own parity suite
at the **token level** (N-token greedy stream agreement with the contract path on real
checkpoints, near-tie rule applied, drift curve recorded per family) — a *quality* gate, stated
as one, never presented as the byte gate; `benchmarks.md` rows labeled with the mode. **The
middle-path trap carries over verbatim from §7:** do NOT relax the default path's byte gate to
tolerance — discrete decisions flip on ~0.001 margins (measured, MoE routing); a sometimes-red
gate is worse than either strict alternative. M-B is a *mode beside* the contract, never a
softening *of* it.

**What M-B buys, stated plainly:** long-context decode moves from ~2.5–3× behind (pending M0) to
plausibly ~1.2–1.5×, and the qualitative claim "usable at real context lengths on a MacBook"
becomes available. What it costs: a second attention kernel to maintain, a second gate class, and
the sentence "in throughput mode, goinfer's output is not bit-identical to its own contract path"
in the README. Whether that sentence is acceptable is the maintainer's call — it is the entire
decision.

---

## 7. The rewrite question, answered

**A from-scratch Metal rewrite is the one option with known-negative expected value.** It
preserves both binding walls (hardware, contract) and discards the assets:

- the purego-objc binding — the "80% risk" that is now retired engineering: the ~20-selector
  compute surface, the `LC_BUILD_VERSION`/MSL-2.4 landmine defusal (golang/go#77917; explicit
  MSL 3.1 + read-back assertion + bfloat canary), `MTLSize`-by-reference ABI, the
  LockOSThread/NSAutoreleasePool discipline (the SIGSEGV class), the encode-ahead executor;
- 17+ bit-exact-validated MSL kernels, the fused-argmax head, the f16-MMA prefill path, the
  Gemma f32-KV twins, the pinned `tgReduce*` width constants;
- and, worth more than the code: **the measured map** — §4's ledger. A fresh implementation
  re-walks it at multi-session cost and lands at the same ~73/±, because the walls are not in
  the code.

The arc itself is the evidence: attention was rewritten (1-thread-per-head → threadgroup-per-head
→ split-KV → grouped ×2 → staged), the GEMV went four generations (naive → coal → Stage A →
Stage B), the dispatch model was rebuilt twice (bind-batching, encode-ahead) — every structure a
rewrite would try has been tried, measured, and either shipped or reverted with the reason
recorded. **What was never tried is not a structure — it is a policy** (§6 M-B), plus one
measurement (M0).

The only "start from scratch" with real content is a **substrate change** — MLX / MPSGraph /
Metal-4 tensor ops instead of hand MSL. That is Wall 2 wearing different clothes: Apple's stack
buys ANE/AMX-class matmuls (`gpu-assessment.md`: MLX/CoreML-only, unreachable from hand MSL) at
the price of the cgo-free identity and the bit-identity contract — the two properties that are
the point of the backend (`CGO_ENABLED=0` on the platform where pure-Go matters most). If those
properties ever stop being the point, the honest move is not a goinfer-Metal rewrite; it is
depending on the ecosystem that already exists (yzma-style llama.cpp bindings, MLX) and saying
so. Short of that reversal, the current backend is the right chassis.

---

## 8. Suggested order

1. ~~**M0**~~ **✅ DONE (2026-08-04, §5)** — peer re-anchored v0.32.5, FA confirmed on-box, depth rows
   filled. Gap steeper than estimated (~1.34× @128 → ~4.2× @4000). Structural thesis holds.
2. ~~**M1**~~ **✅ DECIDED (2026-08-04): M-A** — stay bit-identical, accept the 4.2×-@4000 depth floor,
   make no competitive long-context claim. M-B deferred, not rejected (§6).
3. **M-B — DEFERRED (available lever, not funded).** If revived: FA-style decode attention first (the
   whole depth prize, ~28→45–55 tok/s bounded by the shallow floor), KV-q8 second (family-gated).
   Token-level gate suite lands with the first kernel, default off. Trigger: long-context-on-a-MacBook
   becomes a goal.
4. **M3** — the 26B no-copy paging probe when the fieldfare comparison is wanted. One run.
5. **M2/prefill** — lm-head tg=128, concurrent-encoder (counter-gated), MMA-prefill levers —
   only after the above; none of it moves the depth gap.
6. Re-generate §B3 with mode-labeled rows once any of 3–5 lands.

## 9. Boundary (what this doc does not do)

- **No code, no campaign.** It consolidates measurements and forces one decision; M-B's design
  above is a scope sketch, not a spec — the spec gets its own doc if M1 chooses it.
- **No new numbers.** Every figure is sourced from the docs named in the header; the two
  estimates (M0 expected outcome, M-B prize bound) are labeled as estimates and exist to be
  falsified by their measurements, per the write-the-expectation-first discipline.
- **No release claims.** §B3 stays the only citable Metal comparison until M0 re-anchors it.
- **Peer mechanism caveat:** "FA default-on, Metal included" is confirmed at source on CUDA (D4)
  and externally corroborated for Metal; the on-box Metal verification is M0's job, and this doc
  should be corrected if M0 contradicts it.
