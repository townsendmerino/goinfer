# Task: FA-class decode attention — the fast-mode fork, scoped as a campaign

> **Status: SCOPED — campaign of record (source-verified).** Drafted 2026-08-10 from the
> plan-still-slow relay record; the load-bearing claims were verified against the working tree (state
> through the Metal leg, b3ac838) — test names, the WebGPU precedent, the softcap/sink
> inventory, and the Metal megakernel record are cited from source below. **Both §8 residual
> items are now discharged on the Mac at origin/main HEAD** (geometry matrix enumerated; §1 numbers
> re-confirmed). Not funded; M1 and M3 are go/no-go, not assumed-pass (§3, §4). (See
> `docs/completed/plan-still-slow.md`). This is the campaign that document's close-out named
> as "the next real decision — outside this plan": the one remaining decode-speed lever on
> both GPU backends, and the first real constituent of the deferred `--mode exact|fast`
> capstone (P7). Nothing here is funded; this doc exists so the decision is made against
> the measured record instead of re-derived from intuition.

## 1. The measured case (why this is the biggest item on the board)

Every number below is from anchored campaigns in `docs/benchmarks.md` §B6–§B7 (CUDA,
RTX 2070 SUPER, Ollama v0.32.6) and §B3 (Metal, M1 Pro):

- **Depth coefficient, CUDA:** goinfer decode costs a flat ~**0.74 µs/KV-position** (0.5B)
  / ~**1.0** (1.5B) once past the shallow regime; the peer holds **~0.03–0.09**. The
  penalty is linear (it plateaus — measured to 32k), so it integrates: **1.4× behind at
  3900, 2.0× at 8k, 3.2–3.4× at 16k, 5.54× at 32k** (39.0 vs 215.9 tok/s).
- **Depth coefficient, Metal:** ~**9–12 µs/position** — worse than CUDA by ~10×; decode
  falls 3.4× over 128→4000.
- **The cause is named, not assumed.** goinfer moves KV bytes at **6–10% of peak DRAM
  bandwidth** while the peer reads identical bytes 3.2× faster; the GQA-re-read model was
  falsified by arithmetic (it would put the peer >100% of peak). Metal: the half-width KV
  probe moved attention only **12%** (full 16.62 → q8-width 14.66 ms; pin0 floor 5.22 ms).
  **Both backends are latency/occupancy-bound, not byte-bound** — which is why KV
  quantization was refuted as a speed lever on both (reachability-only).
- **A second, separable cost with the same cure:** split-KV's fixed per-token overhead
  (~0.7–1.0 ms, small-kernel execution across 3 dispatches × nLayers; §B6.2, P6a) — now
  gated by measured per-geometry crossovers, but still paid wherever split-KV wins. A
  fused one-kernel attention deletes it structurally.
- **What's already exhausted:** Campaign A took every bit-identical lever — float4
  coalescing (1.34×), split-KV (1.20×), the V-sum unroll (refuted) — and **closed at the
  bit-identity ceiling (~1.17× behind at 2048 on the geometry it measured)**. The ncu
  record shows nothing throughput-saturated: the exact-order reduction pins the kernel
  shape, and the kernel shape pins occupancy.

D4 (ollama-chase §8) confirmed what the peer does at depth: **flash attention**. That is
the technique this campaign builds — and the reason it was never built here is a contract,
not an oversight.

## 2. Why bit-identity blocks it, precisely

FA-style decode fuses attention into one kernel that tiles over the KV cache with an
**online softmax**: a running maximum and denominator, rescaled per tile. The rescale
accumulates in tile order, so its float rounding differs from the shipped 128-wide
partition+tree reduction — same distribution to within ULPs, different bytes. goinfer's
CUDA decode attention is gated **byte-identical** (`TestSplitKV_bitIdentical` +
`_gemma3`, `cuda/splitkv_bitident_test.go`; `TestAttnBatched_bitIdentical`,
`cuda/attn_batched_test.go:28` — names verified against source), and reduction *width* is
explicitly part of the contract (the Metal 128-vs-256 finding). Campaign A's own record
says the quiet part: the fix "is split-KV parallelism, **not FA's online rescale, which is
not bit-exact**." Crossing that line is a decision, not a patch — which is what this doc
scopes.

**The §7 trap, restated so it is not re-litigated:** the forbidden move is
tolerance-gating the *default* path — it converts clean gates into intermittently-red ones
that fail at near-ties months later on a family nobody watches. The sanctioned shape is an
**opt-in path with its own quality lane**, existing alongside an untouched exact default.
This repo already operates two such lanes:

1. the 26B pager: lossy but **argmax-faithful, cosine ≥ 0.99** vs the f32 cache;
2. **WebGPU's MLA attention is already an online-softmax kernel** — `gpu/mla.go:27`:
   "One workgroup per query head, online (FlashAttention-style) softmax over keys" —
   **gated on cos ≥ 0.9999 / maxAbs ≤ 1e-3** (`gpu/mla_test.go:366,430,471`; verified).
   The byte-identity contract is *per-backend in practice*: CUDA and Metal
   defended it; WebGPU quietly did not. An FA lane on CUDA would not be the repo's first
   tolerance-gated attention — it would be its first *deliberate* one.

## 3. Design sketch — CUDA (the primary target)

One fused M=1 kernel (or a two-pass with a **fixed, deterministic combine order** — see
the determinism gate): tile the key axis across blocks/SMs, online max+denominator per
tile, rescale, single pass over KV, output the weighted V sum. Coverage must match what
the exact path serves today: GQA at hd 64/128, gemma3 hd=256 + sliding window (windowed
layers interacting with the per-layer split-KV gate). Two simplifications, verified from
source: **no attention softcap in the served set** — only Gemma-2 ever had
`AttnLogitSoftcap` and it is not in the arch set; Gemma-4's `FinalLogitSoftcap=30` is
logit-stage, outside the attention kernel, and gemma4 is not resident anyway
(`docs/completed/c8-softcap-residency.md`). And **no attention sinks**: the sink users
(gpt-oss class; DeepSeek-V4's `attn_sink`) are not resident families, so the FA kernel
needs neither. **Geometry matrix (§8.1, enumerated): the splitkv bit-identity gate covers hd=128
(qwen 1.5B, full-causal) and hd=256-windowed (gemma3) — the FA gate mirrors those two. hd=64 (the
0.5B) is NOT a bit-identity cell today, so it is a NEW gate cell to add if the FA lane covers 0.5B,
not a mirror.**

**Determinism is a design requirement, not an aspiration:** fixed tile size, fixed
traversal order, fixed combine tree, no atomics-order dependence, no dynamic scheduling
that changes summation order run-to-run. The lane's promise is "deterministic on given
hardware, not byte-equal to the exact path" — the same promise the Metal snapshot golden
already enforces for a different purpose (machine-pinned golden, env-branch on
hardware/OS). Reuse that harness shape.

**Payoff arithmetic (0.5B, from the §B7 anchors):** base ≈ 3.14 ms/tok; current
coefficient 0.735 µs/pos → 26.6 ms/tok at 32k (≈ the measured 39 tok/s). At 3× the peer's
coefficient (~0.09): ≈ 6.0 ms → **~165 tok/s at 32k** (peer: 215.9). At the peer's ~0.03:
≈ 4.1 ms → ~240. **Milestone target: within 3× of the peer's per-position coefficient**
— that alone converts 5.54× at 32k into ~1.3×. Parity is the campaign goal, not the first
milestone: the peer's kernel has years of tuning; expecting to match it out of the gate is
the §7 lesson about cuBLAS transplanted.

**⚠ M1 may not be reachable on the 2070S at all — state it up front, don't discover it at M2.** The
milestone arithmetic assumes a fused FA kernel can lift occupancy enough to reach ≤3× the peer's
coefficient. Campaign A's ncu record says the exact-order reduction pins the kernel shape and the
shape pins occupancy; FA removes the *exact-order* constraint, but whether that buys enough occupancy
on this specific card is the open M0→M1 question, not a given. So **the campaign can honestly END at
M1 with a written refutation** ("FA fuses cleanly but the 2070S occupancy wall holds, coefficient
still N× peer") — exactly the M3 posture, applied to CUDA. M1 is a go/no-go, not a milestone assumed
to pass; do not fund M2 coverage until M1's number is in hand.

## 4. Design sketch — Metal (second, profile-first)

**The honest Metal picture is harder than the CUDA one, and the record says why.** The
shipped Metal decode attention is *already FA-decode-shaped in the way that matters*: one
dispatch, one threadgroup per head, a single serial pass over the keys — there is no
3-dispatch split to fuse away and no cross-threadgroup sync to eliminate (that was
split-KV's regression, already reverted). An online-softmax rescale does not change its
memory behavior, and its measured walls are elsewhere: DRAM-latency-bound serial K/V
reads (75% of attention time; the half-width probe moved it only 12%), the dedup-vs-
occupancy opposition (four structures tried, all lost), and **megakernel-style fusion
tested and CLOSED on Metal** — no grid-sync, redundant-recompute net-negative
(`docs/completed/task-metal-cgofree-spike.md`, "megakernel tested and CLOSED"). So M3 is
a genuine go/no-go, not a port: profile what an FA-style rewrite could even change on M1
(candidate levers: deeper in-flight read pipelining within the serial pass, wider
per-threadgroup key tiling — both must beat the latency wall the record describes). The
null hypothesis is the A2-Metal conclusion — "the lever is elsewhere, or accept the
floor" — and a written refutation joining that record is a legitimate M3 outcome.
Metal's ~9–12 µs/pos remains the largest single backend deficit in the repo; that is the
prize *if* a lever exists, not evidence that one does.

**M3 STARTED — first data point, and it lands on the refutation (`TestZZ_attnM3ThreadWidth`,
2026-08-10, M1 Pro).** On M=1 decode, FA's "don't materialize the score vector" benefit does not
apply (the score vector is only nKeys floats), so FA's only lever over the shipped one-threadgroup-
per-head serial pass is more in-flight parallelism to hide the latency wall. Sweeping the per-head
threadgroup width **128 → 256 → 512** (4× the concurrent K/V loads per head), all-28-layer attention
@2048, min GPU-busy/20 reps: **17.05 / 17.5 / 17.5 ms — the wider tiles are 102–103% of shipped, i.e.
slightly worse, never better.** More parallelism *within* a head does not touch the wall. And the
other direction — more threadgroups *across* the key axis (split-KV) — was already measured as a
**regression** on Metal and reverted (§A2-Metal). So **both forms of added parallelism fail on M1**,
and FA's remaining theoretical lever (smaller threadgroup memory → higher occupancy) does not bind
here: an attention dispatch launches only nH≈12 threadgroups, already under the threadgroup-memory
occupancy limit, so the ceiling is DRAM-latency serialization per access, not occupancy FA could lift.
**M3 verdict: NO-GO on Metal** — this is the written refutation joining the A2-Metal record. Caveat:
this is the go/no-go probe plus the split-KV/half-width record, not a full FA-prototype disproof, but
all three point the same way. The Metal ~9–12 µs/pos deficit stands as accepted floor unless a
fundamentally different memory system (not this M1 Pro) changes the latency picture.

## 5. The quality lane (the actual spec)

The exact path stays default; its byte gates are **untouched** — not one threshold moves.
The FA path is opt-in (env/flag during development; `--mode fast` / P7 is the eventual
user surface, per the P7 design constraints: printed expansion, explicit-flag override,
its own gate lane before it ships). Gates, all on the opt-in path:

1. **Determinism golden** (machine-pinned): fixed token sequence through the FA path on
   committed tiny models, sha-compared, byte-identical run-to-run and across code
   versions on the same hardware; env-differs branch on GPU/OS per the Metal golden
   protocol (`metal/snapshot_golden_test.go` is the pattern — note CUDA's existing
   goldens are forward-parity artifacts, not decode-stream shas, so the CUDA
   determinism golden is a new twin built on that pattern, not a reuse). Red = the
   kernel's order became data- or schedule-dependent.
2. **Accuracy floor vs the exact path:** on real checkpoints across depths
   {128, 2048, 8192, 32000} and every served family shape (GQA, hd=256+window, softcap):
   argmax-faithfulness rate + logit cosine floor. **Thresholds set from the measured
   distribution at build time** — run the exact-vs-FA comparison first, then pin floors
   with margin, so the gate is born from data (the throughput-gate re-bound lesson, P2b).
3. **Token-divergence characterization vs exact greedy:** nonzero by nature (near-tie
   flips); measure the rate at depth, publish it in the lane's docs, and gate on a ceiling
   — never pretend it is zero, never let it be "intermittently red" (it is *expected* red
   at a bounded rate, asserted as such).
4. **The vacuous-gate check** (P6a lesson): every gate must be shown to *fire* — flip the
   tile width, watch the determinism golden go red; perturb a logit, watch the floor
   trip. A gate that cannot fail is not a gate.

## 6. Milestones

| # | milestone | exit criterion |
|---|---|---|
| M0 | CUDA prototype, one geometry (qwen 1.5B class) | exact-vs-FA accuracy distribution measured at 4 depths; determinism holds; coefficient measured |
| M1 | coefficient ≤ 3× peer on that geometry — **go/no-go** | anchored A/B vs exact path and vs peer (§B-style table), OR a written refutation ("FA fuses but the 2070S occupancy wall holds") — §3 |
| M2 | coverage: GQA/hd=256/window/softcap + gate lane complete | all §5 gates live and demonstrated-firing; opt-in flag |
| M3 | Metal profile → go/no-go → port if go | **NO-GO (2026-08-10)** — thread-width probe refuted the parallelism lever (§4); split-KV already refuted; occupancy doesn't bind at nH≈12. Written refutation, joins A2-Metal |
| M4 | P7 wiring | `--mode fast` aggregates it per the P7 constraints |

Class: **weeks per backend** (M0–M2 is the campaign core). Not in scope: prefill's path 3
(same technique, separate decision — decode is where the measured money is), CPU, any
change to defaults, any change to exact-path gates.

## 7. Queue position

Post-v1.0, and it should be **the** next performance campaign when one is funded — it is
the only lever left that addresses a measured every-model deficit (the depth coefficient)
rather than a coverage or capacity item. Pairs with, but does not wait on: MLA-on-CUDA
(orthogonal; that task's attention reuse is exact-path), K3 (orthogonal). Supersedes
nothing in §10 of ollama-chase: CUDA-graphs-as-speed-lever stays refuted (the fixed
dispatch cost dies here by fusion, not by graphs); KV-quant stays reachability-only.

## 8. Verification record + residual items

**Verified 2026-08-10 against the working tree (state through b3ac838):** gate names
(`TestSplitKV_bitIdentical`/`_gemma3`, `TestAttnBatched_bitIdentical`) from source; the
WebGPU online-softmax precedent and its tolerances (`gpu/mla.go:27`,
`gpu/mla_test.go:366,430,471`); attention softcap absent from the served set and sinks
confined to non-resident families (`c8-softcap-residency.md` + Phase 0 record); Metal
megakernel tested-and-CLOSED (`task-metal-cgofree-spike.md`) — folded into §4's revised
Metal posture; snapshot-golden harness confirmed as pattern-not-reuse for CUDA.

**Residual items — both DISCHARGED on the Mac at origin/main HEAD (2026-08-10):**

1. ✓ **Splitkv bit-identity geometry matrix enumerated.** The gate covers **two** cells:
   `TestSplitKV_bitIdentical` (qwen2.5-1.5b: **hd=128, GQA nKV=2, full-causal, no window**) and
   `TestSplitKV_bitIdentical_gemma3` (gemma-3-4b: **hd=256, sliding-window, winStart>0** at depth
   1536 > window 1024), each forcing the split path past the per-geometry depth gate so neither is
   vacuous; plus `TestAttnBatched_bitIdentical` for the batched path. So the FA gate matrix mirrors
   **{hd=128 full-causal, hd=256 windowed}**. ⚠ **hd=64 (the 0.5B) is NOT a bit-identity cell today**
   — see the §3 correction; if the FA lane wants 0.5B coverage that is a NEW cell, not a mirror.
2. ✓ **§1's numbers confirmed against `benchmarks.md` §B7 at HEAD** (the draft was checked against
   the Mac copy pre-pull; on tip they hold): 32k **39.0 vs 215.9 tok/s = 5.54×** (§B7 line 916);
   plateau coefficients **+0.713 / +0.748 / +0.735 µs/pos (0.5B)**, **+0.979 / +1.015 (1.5B)** (§B7
   lines 931–936). Metal 9–12 µs/pos and the half-width probe (16.62→14.66, pin0 5.22) are the
   measured `TestZZ_attnKVWidthProbe` / `TestZZ_metalDepthBench` numbers.
