# R1 — the layer-26 "catastrophe" was the oracle, not the kernel: W4A8-as-ground-truth at the attention-sink position

**Result: there is no defect in the W4F16 decode lane. Both new instruments say the same thing
independently. (X1) Against an f64 reference built from the SAME resident int4 weight bits and the
SAME post-attention residual, the f16 gate/up GEMV is exact on real data at every layer including
26 (cosine 1.000000000 vs the f64 dot of its own half-rounded input, max relative error ≤ 6.6e-5),
and it sits on the f64 reference (cosine ≥ 0.99999997) while the SHIPPED W4A8 arm is the coarse one
(cosine 0.9976 at layer 26, worst rows 5–26× off) — because W4A8's single per-tensor int8
activation scale (amax/127 = 1.154 at layer 26, position 0) quantizes 96.9% of the 1536 FFN-input
channels to exactly zero. The parked record's flagged "row 2908, 31% off" had its labels backwards:
f64 reference 71.62, f16 71.62, W4A8 54.70. (X2) Teacher-forced against an external CPU reference
(int8 weights, f32 activations, exact attention) the f16 lane is at least as faithful as W4A8 on
every pooled criterion (mean KL 0.060 vs 0.334, top-1 45/48 vs 43/48, hard flips 1 vs 2), the whole
margin coming from position 0 — the BOS attention-sink token — where the shipped W4A8 lane is badly
wrong (cosine 0.587, KL 14.4, a hard argmax flip at 80% of the logit range) and f16 is not. R1 is
UN-PARKED: the "bug" the 2026-09-19 investigation localized and could not explain was the
instrument.**

## Why the parked record reached the wrong conclusion

[`w4f16-decode-investigation-2026-09-19.md`](w4f16-decode-investigation-2026-09-19.md) scored the
f16 lane against the shipped W4A8 lane and read every disagreement as an f16 defect. It contains no
measurement against anything that is not one of the two arms. Three things compounded:

1. **The oracle was the lossier arm, and the repo already knew it.**
   [`../completed/task-prefill-gap.md`](../completed/task-prefill-gap.md) §3.1 ("Correction,
   2026-09-05 — the oracle was wrong") withdrew exactly this exact-as-oracle scoring for the prefill
   lane: "the prior is that the exact arm is the lossier one" — per-tensor int8 activation
   quantization loses more than f16 activations do. The decode record re-instated the withdrawn
   oracle, per layer and per row, and its own registered prediction (the f16 arm is *closer* to the
   CPU reference than W4A8 on most prompts) was never tested.
2. **Cosine on the residual is blind where the record looked.** The residual stream at position 0
   carries one "massive activation" channel (408, ≈ −3627 through layers 2–25). With ‖x‖ dominated by
   that channel, an error of norm E in the other 1535 channels gives cosine ≈ 1 − E²/(2‖x‖²): the
   record's layers 1–25 cosines (0.999998–0.9999995) are consistent with percent-level disagreement
   everywhere else, and their maxAbs 14.7–28.7 *was* that disagreement, dismissed as quantization
   noise. Layer 26 — where the FFN partially cancels channel 408 — is merely where the residual's
   norm drops enough for the disagreement to show in the cosine. A three-orders-of-magnitude
   "improvement" from layer 0 (0.99967) to layer 1 (0.999999) in the record's table is the massive
   channel arriving, not fidelity improving.
3. **The whole trace ran at position 0** — the attention-sink token, where the normed FFN input's
   amax (146.6 at layer 26) makes W4A8's per-tensor step coarsest. The shipped lane's only
   logit-level record on this checkpoint ([`prefill-gate-l1-ref-b-2026-09-09.md`](prefill-gate-l1-ref-b-2026-09-09.md))
   scores positions ≥ 255 and never position 0.

Two of the record's stated facts are also wrong on re-measurement: row 2908's weight nibble at
channel 408 is 5 at layer 26, not 8 (it is 8 at layers 8, 24 and 25 — a different layer's row was
read); and W4A8's post-layer-26 value of channel 408 is −1037, not +1037 (magnitude right, sign
wrong; the FFN *adds* to the channel in both arms, f16 adds ~1.6× more).

## Method

Box `apple-m1pro` (M1 Pro, 16 GB), Darwin 25.6.0, goinfer `f9c3dd31`, tree clean at the time of
the runs. Model: `qwen2.5-coder-1.5b-instruct` GGUF Q4_K_M (the record's checkpoint), Metal
resident at `Quant:"int4"` (int4-g32 requant, f16 group scales — the same bits both arms read).
Orchestrated as an 8-agent workflow: three read-only hypothesis lenses (ground-truth validity,
kernel/dispatch defects, producer/downstream), two experiments run one-at-a-time on the GPU, three
adversarial refuters (instrument/statistics, code correctness, does-it-reproduce) — none refuted;
both experiments re-ran byte-identical on every measurement line. Logs archived under the
goinfer-logs/r1 directory in the home directory (x1/x2 originals and re-runs); the tests themselves
are kept in the tree.

**X1 — `metal/r1_gu_reference_test.go` (`TestR1_guReference`).** One model load, lane OFF; the
lane is toggled at runtime through `r.decodeLaneW4F16` (which `canUseF16Lane` reads on every
call). Token 785 at position 0, exactly as the record. For each of layers {0, 2, 8, 16, 24, 25, 26,
27}: replay the W4A8 lane through the preceding layers, run the attention block of the target layer,
capture the post-attention residual, then dispatch each arm's FFN norm + gate/up **verbatim from the
production call sites** (`metal/model.go` L2186–2195) into separate command buffers and read `r.gu`,
`r.mxF16`, `r.mq`/`r.mSc` and `r.dSc` (after `swiglu_quant`) back. CPU side in float64: the normed
activation from the same residual (eps and addOne read from the resident's own uniform buffers);
the 17920×1536 weight dequantized row by row from `L.guW`/`L.guS`; three dots per row — W·a_f64,
W·a_f16half (the exact half activation the f16 kernel received), W·(mq·mSc) (the exact int8 activation
the W4A8 kernel received). Then the record's repro, same model: W4A8 through layers 0–25 → x1; W4A8
26–27 vs f16 26–27 from the same x1, decomposed by channel.

**X2 — `metal/r1_lane_vs_cpu_test.go` (`TestR1_laneVsCPU`).** Teacher-forced, no generation: a
48-token Go snippet (this is a coder model) tokenized with the model's own tokenizer, fed one token
per position to (A) the CPU backend at `Options{Backend:"cpu", Quant:"int8"}` with
`GOINFER_CPU_FAST_ATTENTION=0` — weight-only per-row int8, f32 activations, exact f64-accumulating
attention; the same reference shape `decoder/prefill_ref_gen_test.go` documents for D7 and for the
same reason (the S reference at `Quant:""` is f32 weights, ~6 GB, and does not fit beside anything on
this machine today) — closed before (B) one Metal load run twice from position 0, W4A8 then f16 by
runtime toggle (restarting at position 0 is clean: `Forward` calls `setPos`, which rewrites the KV
slot and `curNKeys`). Per position: cosine, KL(softmax(cpu)‖softmax(arm)) in float64 over the full
151936-vocab, top-1 agreement, and hard flips via `decoder.NearTieArgmaxForTest`. The reference is
*external* (its weight requant differs from Metal's int4-g32; its LM head and embedding are W8A8 on
both sides), so both Metal arms carry the same weight-quant noise floor against it and the PAIRED
comparison between arms is what carries information.

## Data — X1: gate/up per layer, both arms against the f64 reference

| layer | normed amax (ch) | W4A8 `mq==0` | f16 vs f64 cos | f16 kernel vs own input, maxRel | W4A8 vs f64 cos | W4A8 worst row (got / ref) | f16 closer on |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 2.74 (553) | 2.3% | 0.999999991 | 1.4e-5 | 0.999972891 | −0.0105 / 0.0012 | 98.9% |
| 2 | 49.9 (408) | 50.4% | 0.999999990 | 5.3e-6 | 0.999027040 | −0.934 / −0.071 | 99.6% |
| 8 | 21.4 (408) | 65.1% | 0.999999975 | 6.6e-5 | 0.998526257 | 0.0835 / 0.0004 | 99.6% |
| 16 | 22.7 (408) | 68.7% | 0.999999994 | 1.7e-5 | 0.997553273 | 0.0911 / −0.00003 | 99.9% |
| 24 | 39.9 (408) | 82.2% | 0.999999983 | 9.3e-6 | 0.998447418 | −0.373 / 0.0025 | 99.6% |
| 25 | 77.6 (408) | 92.3% | 0.999999984 | 3.5e-5 | 0.998322762 | (rel 19.9) | 99.7% |
| **26** | **146.6 (408)** | **96.9%** | **0.999999985** | **7.3e-6** | **0.997605443** | 0.592 / −0.005 | **99.7%** |
| 27 | 69.8 (408) | 27.0% | 0.999999994 | 4.8e-6 | 0.999320909 | (rel 6.2) | 99.7% |

The W4A8 kernel is likewise exact against the f64 dot of its own int8 input (maxRel ≤ 2e-7) — both
kernels compute exactly what they are given; only the inputs differ. `rmsnorm_f16_act` matches the
f64 normed activation to ≤ 8e-4 relative (half round-to-nearest is 4.9e-4). The "0.9976" the record
measured for `r.gu` at layer 26 is W4A8's own distance from truth: f16-vs-W4A8 equals W4A8-vs-f64
because f16 sits on the reference.

**Row 2908 at layer 26** (the record's headline anomaly): f64 71.6212, f16 71.6236, W4A8 54.7000.
It is the layer's *super-neuron*: h = silu(gate)·up peaks at neuron 2908 in both the f64 reference
(1190) and the f16 arm (1190); W4A8 puts it at 627 — 47% low. At layer 24 W4A8 mis-estimates the
super-neuron the other way (53.3 vs 38.6, 38% high) — the shipped arm's error is not layer-26-specific;
layer 26 is where a cancellation makes it visible. The f16 lane's down-proj int8 step at layer 26
is 1.9× coarser (9.37 vs 4.94) because amax(h) doubles — a real downstream difference, but the
driving neuron passes through at code 127 in both lanes, so the step cannot be what moves channel 408.

**Repro decomposition** (W4A8 through 0–25, both lanes 26–27 from the same x1; x1[408] = −3627):
after layer 26 cosine −0.696, maxAbs 1629.67; after layer 27 cosine **−0.585, maxAbs 1467.18 — the
record's figure reproduced exactly**. Channel 408 carries 46% of ‖diff‖² (W4A8 −1036.9 vs f16
+592.8); channels 520, 940, 609 with it carry 79%; four channels change sign between arms. With the
five |x| > 200 channels removed the remaining 1531 channels correlate at cosine 0.13 — the whole
layer-26 FFN output differs between arms, not one scalar. The record's "cosine goes negative" carried
no information about which arm was right; X2 does.

## Data — X2: teacher-forced logits, both Metal arms vs the CPU reference (N = 48)

| arm | mean KL(cpu‖arm) | top-1 agreement | hard flips | soft flips | mean cos | min cos |
|---|---:|---:|---:|---:|---:|---:|
| W4A8 (shipped) | 0.3341 | 43/48 | 2 | 3 | 0.9783 | 0.5874 |
| **f16 lane** | **0.0603** | **45/48** | **1** | 2 | 0.9844 | 0.8481 |

Lane-vs-lane top-1 agreement 44/48. McNemar d = 4 positions where exactly one arm matches the CPU
argmax: only W4A8 at position 15; only f16 at positions 0, 10, 20. f16's KL is lower on 25/48
positions. All three pooled rules from `metal/prefill_gate_ref_test.go`'s header hold with f16 as
the candidate and W4A8 as the shipped arm: (a) hard flips 1 ≤ 2 + 2√2; (b) top-1 0.9375 ≥ 0.8958 −
2√4/48; (c) mean KL 0.060 ≤ 0.334.

**Position 0 is the whole margin.** BOS (token 151643): W4A8 cosine 0.587, KL 14.37, a HARD flip
whose gap is 79.7% of the CPU logit range; f16 cosine 0.848, KL 1.37, argmax correct. Excluding
position 0 the arms are statistically indistinguishable on this prompt (f16 KL lower on 24/47, one
hard flip at position 44 shared by both arms at the same 3.48% gap). f16 at position 0 is itself far
from the CPU reference because the parts the lane does *not* change — `swiglu_quant`'s per-tensor
int8 of the down-proj input, and the int8 LM head — are common to both arms.

## What this establishes, and what it does not

**Established.** No f16 defect exists anywhere the record looked or anywhere these instruments
looked: the kernels are exact on real data at production shapes, the producer is exact to half
rounding, and the lane is at least as faithful as the shipped path against an external reference.
The parked record's localization ("gate/up at layer 26") was correct as a localization of *where
the arms differ*; its attribution was inverted. Separately, the **shipped W4A8 decode lane has a
real, previously unmeasured defect at position 0** on this checkpoint (a hard flip at 80% of the
logit range): the per-tensor int8 activation scale collapses at the attention-sink token.

**Not established.** (i) This is NOT a §3.2 pooled fidelity-gate pass for the lane — one prompt,
one cell, N = 48 against the gate's 3 K × 10 prompts × 65 positions, and rule (c)'s paired sign-test
clause is 25/48, a coin flip; the supportable claim is *no defect found, not-worse is underpowered,
better at exactly one position*. (ii) Position 0's logits are never consumed by production
serving (prompts are longer than one token and the batched prefill path is f16 already), so the
W4A8 finding is about the shipped lane's sink behaviour under sequential prefill, not a
user-visible loss — whether it damages positions > 0 through the KV written at position 0 is
untested here (positions 1–47 were unaffected on this prompt). (iii) The CPU reference is
int8-weight, not the repo's f32-weight S reference (does not fit on this box today); its LM head
and embedding are W8A8. (iv) X1's repro toggles the full lane (QKV and o-proj in f16 too), so the
attribution of the post-26 difference to the FFN half rests on the 2026-09-19 record's
attention-only figure (0.99999995), not re-measured here; the verdict rests on X2 either way.
(v) Uncentered logit cosine is reported but should not be read as a headline — KL and top-1 carry
the decision. (vi) 1.5B only; 0.5B and 7B unmeasured.

## Decision

**R1's Build phase is UN-PARKED.** The kernel and wiring are correct; what remains is what the
brief always required and the parked attempt never reached: gate (3), the real teacher-forced
pooled fidelity gate with the lane on (against the f32-weight CPU reference on a box that fits it,
or the int8-weight reference with that stated), and the served-tok/s measurement against the
registered band (three arms interleaved, `scripts/bench_peer.py`, `BENCH_BACKENDS=metal`, the lane
selected by `GOINFER_METAL_DECODE_LANE=w4f16`). Neither is run here.

The deliberately-failing keeper test `metal/gemv_w4f16_layer26_repro_test.go` is **deleted**, not
loosened: it asserted the wrong arm as truth, which its own comment forbade "papering over" — the
mechanism it asked for is this record. `metal/r1_gu_reference_test.go` reproduces its numbers
(the −0.585) and supplies the reading; it and `metal/r1_lane_vs_cpu_test.go` are the new keepers,
both `GOINFER_HEAVY_TESTS`-gated and cheap (3 s and 8 s after load).

**Follow-up worth its own brief, not done here:** the shipped Metal W4A8 decode lane's per-tensor
activation scale at the sink position. The CPU int8 path already uses per-row activation scales and
the Metal prefill fast lane is f16; the decode lane is the odd one out. R1 shipping would make the
question moot on the plain dense path; if R1 is instead killed on speed, this defect still stands.
