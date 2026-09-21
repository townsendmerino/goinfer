# R2 — `attention_fa`'s "divergence at the 3rd decode token" was one int8 rounding crossing, not a kernel defect

**Result: the kernel is correct. Measured on identical inputs, `attention_fa`'s output matches
the shipped `attention` kernel's to ≤ 1e-5 absolute at every one of the 28 layers, at every decode
step, including the ones the parked record calls "bad". What the record saw was the end-to-end
LOGITS of a kernel that is non-bit-identical *by design* (its own doc comment says so), scored
through goinfer's per-tensor int8 activation quantization on Gaussian-noise embeddings — a
discontinuous map. The accumulated trajectory difference stays at f32 noise (≤ 3e-5) for 16
layers and then, at layer 16 of decode position 1602, crosses one int8 rounding boundary: 1334 of
1536 residual elements move by > 1e-3 in a single layer whose own kernel discrepancy is 3e-6, and
that one-code step grows to 0.78 in the residual by layer 27 — the record's 0.588 logit
divergence, to the digit. A control that never touches `attention_fa` (shipped kernels only,
plus a 1e-6 Gaussian nudge on the residual after layer 0) reproduces the same 0.57–0.74 logit
jump at every step. The "position-linked" pattern the 2026-09-20 follow-up found is real — and it
is a property of the input: position 1602's computation has an element within ~1e-6 of a boundary,
which is why the same position gives bit-identical divergence numbers with or without prior
`attention_fa` steps. R2's Build phase was parked on an instrument, the same family of error as
R1's (`r1-layer26-rootcause-2026-09-20.md`), by a different mechanism.**

## What the two prior records had, and what they lacked

[`r2-attn-fa-2026-09-19.md`](r2-attn-fa-2026-09-19.md) proved the kernel correct in isolation
(16/16 adversarial cases, cosine 1.0000000) and found the end-to-end divergence.
[`r2-attn-fa-followup-2026-09-20.md`](r2-attn-fa-followup-2026-09-20.md) found and fixed a real
encode-ahead race (not the cause), retracted its own call-count conclusion after an outside review
caught a mod-4 confound, audited every dispatch and buffer site (clean), and established with a
position sweep that the divergence tracks an absolute position (1602) rather than a call count.
Every one of those comparisons was on **final logits**, 28 layers and three int8 quantization
points per layer downstream of the kernel under test. Nobody had compared the kernel's own output
against the shipped kernel's on the same production inputs at the failing step — the one
measurement that distinguishes "the kernel computes something wrong" from "a correct kernel's
rounding-order noise gets amplified downstream". The follow-up's own ULP control (a 1-ULP change
to the attention scale flipping logits by ~0.9 *immediately*) was the tell that the pipeline is
hypersensitive on this input, and was read the other way: "immediate, therefore a different
shape from attention_fa's delayed divergence". The shape difference has a simple explanation
(§3): a scale nudge is a dense perturbation of every score at every layer and crosses a boundary
at once; `attention_fa`'s noise is sparse — most output elements are bit-identical — and crosses
one only when the input happens to place an element close enough.

## Method

Box `apple-m1pro` (M1 Pro, 16 GB), Darwin 25.6.0, goinfer `ebe50556` plus this record's tests.
`qwen2.5-coder-1.5b` GGUF Q4_K_M, `Quant:"int4"`, the e2e repro's own construction reproduced
exactly: `rand.NewSource(7)` Gaussian embeddings ×0.05, batched prefill (`ForwardBatch`) to
`prefillLen`, then per-token decode. `metal/r2_ctx_diff_test.go` (`TestR2_perLayerCtxDiff`), one
resident, the kernel toggled at runtime through `r.decodeAttnFA` (read by `canUseAttnFA` on every
dispatch; pipelines and the partial buffer are always built) — the technique R1's root-cause
tests validated. Three trajectories per run:

1. **Shipped, end to end** via `ForwardEmb` — the record's reference arm, unchanged.
2. **Manual per-layer harness, kernel off** — each layer encoded in its own command buffer
   (`encodeLayer`, then the final norm + int8 LM head verbatim from `forwardLogits`). **Bit-identical
   to trajectory 1 at every step** (asserted, fatal otherwise) — the harness adds nothing.
3. **`attention_fa` trajectory, instrumented**: at every (step, layer), the shipped kernel is run
   from the pre-layer residual and KV state, its `ctx` and layer output captured, the residual
   restored, then `attention_fa` run on the *identical* state — so `|ctxFA − ctxShipped|` is a pure
   kernel comparison — and the trajectory continues from `attention_fa`'s own output, so its
   per-step logits reproduce the record's arm. Alongside, the **accumulated** divergence of this
   trajectory's residual from trajectory 2's at the same (step, layer).

Plus a **perturbation control** (`GOINFER_R2_MODE=perturb`): trajectory 3 replaced by shipped
kernels only, with `x += N(0,1)·1e-6` added to the residual after layer 0 of every step — a
perturbation of the same order as `attention_fa`'s measured layer-output discrepancy, injected at
the same place, through the shipped pipeline alone.

Runs: prefill 1600 (the record's) and 1602 (the sweep's "bad at step 0" case), FA and control
each; ~80 s per run; one resident; no swap growth. Logs under the r2 directory of this session's
scratchpad, copied beside the other R2 logs.

## Data

**Prefill 1600, `attention_fa` trajectory — logits vs shipped (reproduces the record exactly):**

| step | pos | nKeys | cosine | maxAbs |
|---:|---:|---:|---:|---:|
| 0 | 1600 | 1601 | 1.0000000 | 2.86e-06 |
| 1 | 1601 | 1602 | 1.0000000 | 2.86e-06 |
| 2 | 1602 | 1603 | **0.9995727** | **0.588** |
| 3 | 1603 | 1604 | 0.9995052 | 0.623 |
| 4 | 1604 | 1605 | 0.9995974 | 0.569 |

**Local kernel comparison, identical inputs, every layer, every step:** `|ctxFA − ctxShipped|`
maxAbs between 7.7e-07 and 9.5e-06 (ctx amax 0.9–9.4) at all 140 (step, layer) cells except one —
step 1, layer 0: 1.67e-04 on amax 2.97, still 5.6e-05 of the vector's scale. The layer's own output
on identical input: maxAbs ≤ 9.5e-06 everywhere, exactly 0 at many cells. **No layer computes a
materially different attention output. The isolated gate's numbers (2e-7 to 6e-6) are the
production numbers.**

**Accumulated divergence, prefill 1600, step 2 (the first "bad" step), `attention_fa` trajectory
vs the pure shipped trajectory, per layer** (`elems>1e-3` = residual elements differing by more
than 1e-3, of 1536):

| layer | local kernel diff (ctx) | accumulated residual diff | elems > 1e-3 |
|---:|---:|---:|---:|
| 0–15 | 7.7e-07 – 4.8e-06 | 0 – 8.6e-06 | 0 |
| **16** | **2.9e-06** | **5.99e-02** | **1334** |
| 17 | 1.6e-06 | 1.65e-01 | 1428 |
| 20 | 2.1e-06 | 1.06e-01 | 1497 |
| 24 | 6.0e-06 | 2.41e-01 | 1526 |
| 27 | 7.6e-06 | 7.83e-01 | 1528 |

At step 1 (clean) the accumulated diff is 9.5e-07 at layer 0 and never exceeds 3.05e-05 through
layer 27 — no crossing, logits 2.86e-06. The jump at layer 16 of step 2 is discontinuous (5.7e-06
→ 6.0e-02 in one layer, 1334 elements at once — the signature of one int8 code changing and
fanning out through a GEMV), at a layer whose own kernel discrepancy is 2.9e-06.

**Prefill 1602, step 0 (no prior `attention_fa` step at all):** logits cosine 0.9995727, maxAbs
0.588 — bit-for-bit the 1600 run's step-2 numbers, with a different KV history (positions
1600–1601 batch-prefilled here, decoded there). Accumulated table: layers 0–15 ≤ 4.8e-06, **layer
16: 5.988e-02, 1334 elements** — the identical crossing. Steps 1 and 2 (positions 1603, 1604)
likewise reproduce the 1600 run's steps 3 and 4 to the digit.

**Perturbation control — shipped kernels only, `x += N(0,1)·1e-6` after layer 0:**

| prefill | step / pos | first crossing (layer) | final residual diff (layer 27) | logits cosine / maxAbs |
|---|---|---:|---:|---|
| 1600 | 0 / 1600 | 3 | 0.65 | 0.9996119 / 0.567 |
| 1600 | 1 / 1601 | 4 | 0.83 | 0.9995973 / 0.590 |
| 1600 | 2 / 1602 | 4 | 1.60 | 0.9994473 / 0.642 |
| 1600 | 3 / 1603 | 4 | 0.77 | 0.9994749 / 0.611 |
| 1600 | 4 / 1604 | 4 | 1.10 | 0.9995969 / 0.595 |
| 1602 | 0 / 1602 | 7 | 0.88 | 0.9995184 / 0.590 |
| 1602 | 1 / 1603 | 8 | 1.33 | 0.9994522 / 0.745 |
| 1602 | 2 / 1604 | 4 | 1.89 | 0.9995978 / 0.661 |

A 1e-6 nudge — with `attention_fa` nowhere in the pipeline — produces, at every step, the same
discrete crossing within a few layers and the same 0.57–0.74 logit divergence the record
attributed to the kernel. It crosses *immediately* at every step because a dense nudge touches
all 1536 elements; `attention_fa`'s noise is sparse (the exact-0 cells in the local table), so it
finds a near-boundary element only where the input supplies one.

## Reading

1. **Not a kernel defect.** Direct measurement on identical inputs, every layer, every step,
   both prefill lengths: `attention_fa`'s output is the shipped kernel's to f32 reduction noise.
   The 2026-09-19 record's isolation gate was right; its end-to-end harness was measuring
   something else.
2. **The mechanism is one int8 activation-quantization crossing.** Per-tensor int8 rounding
   (`quant_vec` before o-proj, `rmsnorm_quant` before gate/up, `swiglu_quant` before down — three
   per layer) is a step function; an accumulated ~5e-6 difference lands within a step of a boundary
   at layer 16 of position 1602's computation and one code changes. The GEMV fans that code out to
   ~1300 elements at ~1e-2, and the remaining layers grow it to O(1). The "stable plateau" at later
   positions is the changed trajectory persisting through the KV cache (positions ≥ 1602 written
   from the diverged residual), plus each later position's own crossings.
3. **"Position-linked, not call-count" (§2b of the follow-up) was correct and is now explained.**
   Position 1602's forward pass, on this Gaussian input, has an element within the crossing distance
   of `attention_fa`'s sparse noise; positions 1600 and 1601 do not. This is a property of the input
   row at that position (and the near-identical KV state before it), not of the kernel — hence the
   identical numbers across KV histories, and hence why a mod-4 residue and a fixed absolute
   position both fit some of the data and neither fit all of it.
4. **The follow-up's ULP control was evidence FOR hypersensitivity, misread as evidence against.**
   "Immediate, uniform divergence from a 1-ULP scale change" means the pipeline turns 1e-7
   relative perturbations into 0.9 logit jumps on this input. The proper control — same magnitude,
   same sparsity class, same injection point as the real discrepancy — is the one above, and it
   reproduces the kernel's signature without the kernel.
5. **This is the R1 lesson again, one level up.** R1's oracle was a coarser arm presented as truth;
   R2's oracle is a *discontinuous instrument* — end-to-end logits through per-tensor int8 on a
   noise input — applied to a kernel that was never meant to be bit-identical. Neither record
   contained a measurement of the kernel's own output on real data. For any non-bit-identical
   kernel, that measurement (kernel output on identical inputs) and the accumulated per-layer
   divergence are the first two numbers to take, before any mechanism hunt.

**A mistake this record made and caught.** The first version of the dump/replay tool captured the
attention `q` after the whole 28-layer step had run, so the "isolated replay" recomputed attention
with layer 27's `q` over layer 0's K/V and disagreed with production by 0.95 — which briefly
looked like "the isolated kernel doesn't reproduce production". The isolated kernels agreed with
the f64 reference on that wrong input to 5e-6, which is what exposed it. Fixed (`q` captured
inside the per-layer loop); the replay tool is kept for the next kernel, unused for the verdict
here, which rests on the per-layer tables above.

## Gate (3) — the real fidelity gate on real prompts: PASSES

R2's brief names it: "the same teacher-forced fidelity gate R1 uses, S at depth 3900 as the
decision cell, W4A8 + shipped attention as the exact arm". `metal/r2_gate_test.go`
(`TestR2_decodeFidelityGate`) — `metal/r1_gate3_test.go`'s construction with the toggle swapped to
`r.decodeAttnFA` and K=3900 (above `attnFADepthFloor`=1536, so all 64 continuation positions
dispatch `attention_fa` in the candidate arm). Reference: the S-K3900 CPU f32-weight/
f32-activation files `decoder/prefill_ref_gen_test.go` built (prompt-final logits + 64
teacher-forced decode rows, exact attention), prompt set "a", 10 prompts. Both arms prefill through
`rf.PrefillLast` (the batched path — identical for both, `attention_fa` cannot engage there) and
teacher-force the reference's own continuation through the per-token path. Pooled §3.2 criteria via
`poolCells`, called directly. One resident, `ResidentContext` 3972; 6m28s; no swap growth.

| prompt | shipped: agree / HF / mean KL | `attention_fa`: agree / HF / mean KL |
|---:|---|---|
| 1 | 90.6% / 5 / 0.2374 | 90.6% / 5 / 0.2393 |
| 2 | 53.1% / 23 / 1.6347 | 51.6% / 25 / 1.6450 |
| 3 | 79.7% / 4 / 0.0734 | 79.7% / 1 / 0.0728 |
| 4 | 93.8% / 0 / 0.0351 | 96.9% / 1 / 0.0337 |
| 5 | 60.9% / 14 / 0.9419 | 60.9% / 14 / 0.9437 |
| 6 | 85.9% / 2 / 0.0687 | 85.9% / 0 / 0.0676 |
| 7 | 93.8% / 0 / 0.0382 | 95.3% / 0 / 0.0372 |
| 8 | 54.7% / 20 / 1.7304 | 54.7% / 21 / 1.7181 |
| 9 | 95.3% / 0 / 0.0377 | 96.9% / 0 / 0.0434 |
| 10 | 87.5% / 2 / 0.0633 | 90.6% / 1 / 0.0579 |

**Pooled (640 positions):** (a) hard flips `attention_fa` 68 ≤ shipped 70 + 2√70 — true (fewer,
in fact); (b) agreement 80.31% ≥ 79.53% − 2√19/640 — true (higher); (c) mean KL 0.4859 ≤ 0.4861,
lower on 6/10 prompts, no cell above 1.1× — true. **All three hold: PASSES.** The two arms are
within ±0.012 mean KL of each other on every prompt; the kernel that the e2e harness called
catastrophically divergent is, against an external reference on real text, indistinguishable from
the shipped one and marginally ahead on every pooled measure.

A side observation this gate surfaces about the *shipped* path, not about R2: prompts 2, 5 and 8
sit at 53–61% agreement and KL 0.94–1.73 against the f32 reference in **both** arms — the W4A8
decode path's own fidelity at depth 3900 on those prompts, present with `attention_fa` off. Not
this record's question; recorded because it is the kind of number a later depth-fidelity brief
should not have to rediscover.

## Decision

**R2's Build phase is UN-PARKED.** The kernel is correct (isolation gate, 2026-09-19) and faithful
(gate (3), here); the divergence that parked it was the instrument. What remains is exactly what
the brief always required: the depth-bench speed band (≥60 tok/s at 4000 AND ≥58 at 2048 AND no
regression beyond 3% at 128, min-of-batches, `GOINFER_METAL_ATTN_FA=1` vs shipped) — not run
here. The 2026-09-19 isolated numbers (0.38–0.53× at S=1 everywhere; 1.07× at K=1536 rising to
1.26× at K=3900 at a sized S) are the standing expectation for that run, and the brief's own
caution that isolated attention-term wins shrink end to end applies. `GOINFER_METAL_ATTN_FA` stays
off by default until that band is measured.

`TestAttentionFA_endToEndReproduction`, its depth-2200 twin and the position sweep are kept as
what they are — reproducers of the *instrument's* signature, with their doc comments corrected to
say so; `TestAttentionFA_ulpPerturbationControl` is kept with its reading corrected.
`metal/r2_ctx_diff_test.go` (the per-layer harness with its accumulated-divergence table,
perturbation mode and dump/replay tools) and `metal/r2_gate_test.go` are the new keepers. The
2026-09-20 race fix in `setPos` stands on its own merits, as that record already said.

## What this does not establish

- R2's speed band — untouched; the isolated 2026-09-19 numbers are expectations, not results.
- The exact quantization site (o-proj input, gate/up input, or down input) of the layer-16
  crossing — the accumulated table localizes it to layer 16 and the local table excludes the
  attention output; which of the three int8 points crossed was not isolated. It does not change
  the reading.
- Anything about the M1's dispatch/occupancy floor or R2's speed band — untouched here; the
  isolated speed numbers in the 2026-09-19 record stand as they were.
