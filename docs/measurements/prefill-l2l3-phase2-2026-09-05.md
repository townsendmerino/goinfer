# CUDA prefill L3 — tensor-core int4 GEMM: SHIPS. With L2, prefill is 3.91× at K=3900 (2026-09-05)

**`gemm_w4a8_mma` clears both its pre-registered bars: 4.52× on the gemv category at K=512
(bar ≥2.5×) and 2.85× end to end there (bar ≥2×). With L2 also on, end-to-end prefill is
4.10× at K=512 and 3.91× at K=3900.** Neither is the default; both stay behind
`GOINFER_CUDA_FAST_PREFILL` until §3's reference gate runs on CUDA.

## Provenance

Identical to `prefill-l2l3-phase1-2026-09-05.md` (same box, driver 595.91.07, same session, same
`~/models` local-NVMe checkpoints, GPU 51–55 °C, box idle) except: goinfer at `46d5b3a8` + this
change. Logs: `~/goinfer-logs/prefill-l2l3/p2-*.log`.

**The env gate is now per-lever**, which §5 requires ("measure each against the exact path alone,
then together, so the end-to-end number has an attribution") and §3.1 needs for a different reason
(if the combined path ever loses the fidelity gate, the first question is *which kernel*):

    GOINFER_CUDA_FAST_PREFILL=1|true   both      =attn  L2 only      =gemm  L3 only      unset  neither

## 1. The result

### Category (`TestPrefillDecomp`), with the cross-lever controls

| K | gemv base | gemv L3 | **gemv ratio** | attn base | attn in the L3-ONLY arm |
|---|---|---|---|---|---|
| 128 | 74.68 ms | 35.92 ms | 2.08× | 4.176 ms | 4.115 ms |
| **512** | **307.2 ms** | **67.94 ms** | **4.52×** | 53.43 ms | 53.61 ms |
| 2048 | 1.238 s | 273.3 ms | 4.53× | 831.6 ms | 832.0 ms |
| 3900 | 2.353 s | 530.9 ms | 4.43× | 3.025 s | 3.028 s |

**The right-hand column is the control and it holds:** L3 does not touch attention, and attention
does not move. In the combined arm each category likewise matches its own single-lever arm (gemv
67.59 vs 67.94 ms; attn 12.70 vs 12.95 ms at K=512), so **the two levers compose independently**
rather than interacting — which is what lets the end-to-end number be attributed at all.

### End to end (`TestPrefillTTFT`, batched arm), all four arms

| K | baseline | L2 only | L3 only | **both** |
|---|---|---|---|---|
| 128 | 80.91 ms | 85.98 ms (0.94×) | 43.80 ms (1.85×) | 47.05 ms (1.72×) |
| **512** | 371.5 ms | 329.9 ms (1.13×) | **130.4 ms (2.85×)** | **90.52 ms (4.10×)** |
| 2048 | 2.099 s | 1.471 s (1.43×) | 1.139 s (1.84×) | **517.3 ms (4.06×)** |
| 3900 | 5.451 s | 3.213 s (1.70×) | 3.617 s (1.51×) | **1.393 s (3.91×)** |

**Bands, pre-registered in §4 L3:** category ≥2.5× ships → **4.52×**. End to end at K=512 ≥2×
→ **2.85×**. **L3 SHIPS.**

### The projection in §6 was very nearly right, and that is worth recording

§6 pre-computed L2+L3 from measured category shares and labelled it "counted, not measured":

| K | §6 projected (L2+L3) | measured catSum | measured e2e |
|---|---|---|---|
| 512 | ~103 ms (3.6×) | 90.87 ms | 90.52 ms (4.10×) |
| 3900 | ~1414 ms (3.9×) | 1.421 s | 1.393 s (**3.91×**) |

At K=3900 the projection lands within 1.5% of the measurement. That is evidence for the *method* —
Amdahl on measured category shares — not just for this result, and it is the same method §4 used to
set the bands. At K=512 the measurement beats the projection because L3 exceeded the 4× the
projection assumed.

## 2. Correctness

`TestGemmMMA_vsExact` over the real production shapes of both models — S gate/up (N=8960 K=1536),
S down (N=1536 K=8960), S fused q/k/v (N=2048), D7 gate/up (N=18944 K=3584), D7 down (N=3584
K=18944), plus a deliberately odd N=1543 for the write guards — at M ∈ {16, 64, 100, 512}:

| shape | worst relative delta (bar 1e-5) |
|---|---|
| S/gate-up | 2.7e-7 – 4.4e-7 |
| S/down | 5.5e-7 – 8.1e-7 |
| S/fused-qkv | 2.0e-7 – 3.3e-7 |
| D7/gate-up | 3.9e-7 – 5.7e-7 |
| **D7/down** | 8.7e-7 – **1.107e-6** (worst overall) |
| odd-N 1543 | 2.4e-7 – 3.3e-7 |

**The pre-registration held this time**, and the derivation predicted it: f32 eps = 6.0e-8, the
gap between two association orders of G terms is ~eps·√G·|result|, and the largest reference case
(K=18944 → 2368 word-terms) gives ~2.9e-6 worst case against a measured 1.1e-6. The bar is stated
against **max |dst| across the output**, not per element — a deliberate correction from Phase 1
§2.1, where a per-element bar scaled by a quantity that can cancel to near zero proved unmeetable
for reasons unrelated to the kernel.

**Why the agreement is this tight.** The two kernels compute the *same* int8 products against the
*same* per-group f16 scales; only the cross-group float association differs. In fact
`gemm_w4a8_mma` performs **4× fewer float roundings** than `gemv_w4a8_rn`: the reference folds one
float FMA per WORD (K/8 terms, because dp4a accumulates only 8 elements exactly), while the two
`m8n8k16` MMAs accumulate all 32 elements of a group in int32 with **no rounding at all**, leaving
one fold per GROUP (K/32). The new kernel is, if anything, the more accurate of the two; the test
does not assume that, it bounds the gap.

`TestGemmMMA_selectorKeepsExactBelowFloor` pins the M<16 floor **at the selector**, not at the
kernel. `TestGemmMMA_vsExact` launches the GEMM directly, so it would pass just as happily if the
selector had been wired to use it at M=1 — the G27 rule from CLAUDE.md: a component whose contract
depends on WHEN it is called must be tested through its caller.

## 3. The floor is a PER-LEVER question, and only the attribution shows it

At K=128: L3 alone is **1.85×**, and adding L2 **drops it to 1.72×**. L2 costs ~3.2 ms there
because attention is 5% of prefill at that depth while a 64-row query tile still stages 64 keys per
block (Phase 1 §3).

So the floor §3 requires is **not one number for "fast prefill"**:

- **L2 needs a length floor**, somewhere between 128 and 512 on this evidence.
- **L3 does not.** It wins at every depth measured, 2.08× on the category at K=128 and 1.85×
  end to end.

An all-or-nothing flag would have averaged these into a single worse answer and hidden the fact.
Phase 3 sets both floors from data.

## 4. Kernel facts, for the record

`ptxas -arch=sm_75`: **80 registers, 0 bytes spill stores, 0 bytes spill loads**, 9 KB shared. The
PTX carries **no `.local`**. The SASS carries **32 × `IMMA.8816.S8.S8`** — Turing's int8 tensor-core
op at exactly the 8×8×16 shape, so the tensor path is real and not an `IDP4A`/`FFMA` fallback.

The activation tile's row stride is padded to 36 int words (`GAPAD=4`) so lane's bank is
`(m*36 + kw) % 32 = (m*4 + kw) % 32`, distinct for all 32 lanes — conflict-free. A stride of 32 puts
all eight m of a k-column in ONE bank (8-way conflict) and 33 still collides ~3-way.

## 5. What this does not claim

- **The default has not changed.** Both levers are opt-in. `gemv_w4a8_rn` remains the exact path,
  is bit-identical to the M=1 decode GEMV, is what the parity gates run, and serves M<16, int8
  bundles, K not a multiple of 32, and any shape where the module did not load.
- **No fidelity claim.** §3's reference gate has not been run on CUDA. These numbers say the kernel
  is arithmetically within 1.1e-6 of the exact path and 4.5× faster. Whether the *combined* fast
  prefill is as good as the exact path on real prompts is Phase 3's question, unanswered here.
- **One card, one quant, greedy, dense models.** MoE prefill declines batched prefill statically and
  is untouched (P20). D7 end-to-end with L3 is not yet measured; only its GEMM shapes are gated.
- The §6 agreement is two cells. It is evidence for the method, not a general claim about Amdahl
  projections.
