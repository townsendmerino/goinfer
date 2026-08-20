# Gated-DeltaNet residency — the plan, and why WebGPU proves it first

**Status: DONE AND MEASURED — 2026-08-19.** The family is RESIDENT on WebGPU, gated end-to-end
against the CPU forward for both siblings AND at the released head geometry, at **11.4–12.2×
CPU decode**. Getting a real-width number surfaced a blocker that the tiny fixtures had hidden
completely — see "The head_dim 256 wall" below. Written at Phase 0 so the phases, the reuse claims and the kill criteria were on record
before any shader was.

## Why this, why now

Every Gated-DeltaNet hybrid — `qwen3_5_moe` (Qwen3.5/3.6), `qwen3_next`, `qwen3_5` (Qwen3.8) — is
**CPU-only on every backend**, and honestly labelled as such: `decodeRunnerEligible` refuses the
whole `arch.qwen35 != nil` family up front because no backend implements the mixer. That is the
correct posture for a missing capability and a bad one to keep forever.

It is now also the last big lever for this family. The projections were quantized on 2026-08-19
(1.60× decode, 7.4× TTFT — P12), which leaves the CPU path within ~2.55× of llama.cpp on a kernel
question that belongs to aikit (P14). Residency is a different axis entirely.

## Why WebGPU first, not CUDA

**CUDA has no recurrent state at all.** Its 24 `.cu` files are attention, GEMV, MoE and glue —
nothing stateful. Building DeltaNet there means inventing per-layer device state, a conv window, a
gated norm AND the delta rule, with no in-backend precedent.

**WebGPU already runs a recurrent mixer.** It is the only backend declaring `FeatSSM`, and
`gpu/mamba.go` carries the whole engine:

| DeltaNet needs | Mamba-2 engine already has |
|---|---|
| depthwise causal conv over a K-wide window, SiLU | `ensureMambaConv` — same shape, in-place `win [(K-1)*convDim]` |
| persistent per-layer state updated in place per token | the `ssm` storage buffer + `DecodeRunner` state plumbing |
| gated RMSNorm × SiLU(z) | `ensureMambaGNorm` (semantics differ slightly — see Phase 2) |
| per-head scalars (β, decay) | `headP [nHeads*3]` packing precedent (Aexp/dtBias/D) |
| **the delta-rule update** | **nothing — this is the new kernel** |

**And the threading model transfers exactly.** `mambaSSM` gives each thread one `(head, pi)` owning
a contiguous state row: dot, in-place update, dot. The delta rule has the same shape *if the state
is stored transposed relative to the CPU layout* — `[headV][hv][hk]` instead of `[headV][hk][hv]` —
so thread `(headV, vd)` owns a contiguous `S[vd][0:hk]`:

```
kv         = Σ_kd S[vd][kd]·k[kd]      // contiguous dot
S[vd][kd] += k[kd]·delta               // contiguous axpy
o          = Σ_kd S[vd][kd]·q[kd]      // contiguous dot
```

The CPU reference walks that state column-wise with stride `hv` and makes two passes; the GPU
layout fixes both, which is a reason to expect the port to be *faster than* a transliteration, not
merely equivalent.

CUDA follows once the design is settled here.

## The 8 GB problem, and the house answer to it

No hybrid model fits this card at int4 — Qwen3.8-27B ≈ 15.3 GB, Qwen3.6-35B-A3B ≈ 19.2 GB,
Qwen3-Next-80B ≈ 44 GB, against 8 GB of VRAM. Qwen3.8 is DENSE, so the C′ expert-cache streaming
that rescues big MoEs does not apply.

**This does not block the work**, because it is the same problem gemma4 residency had and the repo
already has the answer: a SCALED fixture with real head geometry at reduced depth
(`scripts/pin_gemma4_moe_scaled.py` → `TestGemma4MoEScaled_residentParity`). Real `head_dim`,
`linear_*_head_dim`, `num_*_heads` and conv kernel; fewer layers. That is what makes the kernel
provable on this box rather than on a promise.

## Phases

**Phase 1 — the fixture. REFRAMED, and NOT the blocker it was scoped as.** The plan assumed
nothing downstream could be gated without a real-geometry scaled checkpoint. That was wrong in a
useful way: the KERNELS are gated at real head geometry with no checkpoint at all (Phase 2 drives
them from synthetic weights through the CPU reference), and the WIRING is gated by the existing
tiny fixtures, because a wiring error in a recurrent model shows as cosine DRIFT rather than as a
width-dependent numeric gap. All four wiring mutations were caught on a 64-hidden fixture.

What a scaled fixture would still buy is the quantization-at-width question — whether int4/int8
DeltaNet projections hold up at hidden 5120 the way they do at 64 — which is a QUALITY question,
not a correctness one, and is the honest remaining gap. Original scope: `scripts/pin_qwen35_scaled.py`: real DeltaNet geometry (key/value head
dim 128, 16 key heads, 48 value heads, conv 4, head_dim 256), reduced layers and hidden, both mixer
kinds present (3 linear + 1 full at minimum), seeded weights, CPU golden. Nothing downstream can be
gated without it.

**Phase 2 — the kernel, unit-tested before it is wired. DONE.** `deltaRuleShaderWGSL` (thread owns
a contiguous transposed state row) + `deltaNormShaderWGSL` (per key head, one pass). Gated in
`gpu/deltanet_test.go` against the CPU `gatedDeltaNetStep` through a new `deltaCapHook` seam — the
same arrangement `mambaCapHook` has, because the recurrence output is a local that the gated norm
overwrites in place.

- **Result:** cosine 1.000000000, worst maxAbs/rms 6.2e-6 over 64 tokens at hk=hv=128, nk=16,
  nv=48. The error does not grow with step count (2.2e-6 at step 1, 3.2e-6 at step 64), which is
  the signature of reassociation noise rather than state drift.
- **Non-vacuity, by mutation:** breaking the GVA head mapping, dropping the decay, and un-transposing
  the state row each fail the gate. Dropping the l2-norm epsilon does NOT — at ordinary magnitudes
  1e-6 is below f32 resolution — so `TestDeltaNorm_cpuParity` covers that case separately with an
  all-zero head, where `inverseSqrt(0)` poisons the state with NaN and the reference does not.
- **The gated norm DOES need its own kernel** — `ensureMambaGNorm` is not reusable, and the
  first version of this line said it was. The two spell the same words and compute different
  functions: mamba normalizes the GATED PRODUCT (`g = y·silu(z); out = w·g/rms(g)`), DeltaNet
  normalizes the recurrence output and gates AFTERWARDS (`out = core/rms(core)·w·silu(z)`).
  Substituting one measures cosine 0.986 and 12× RMS error — a plausible tensor of the right shape
  with the wrong values, which is exactly the failure a shape-only reuse argument cannot see.
  `deltaGNorm` is ~20 lines and, because DeltaNet's `[hv]` weight is indexed by `vd`, needs no
  load-time tiling either — so the correction costs less than the reuse would have.
- **Four kernels, not two, and they are gated CHAINED.** `deltaGates` (β and the decay, on device —
  the alternative is a round-trip per layer per token) and `deltaGNorm` join the two above. The gate
  runs them composed, each consuming the previous one's real GPU output, per the A′ post-mortem's
  "isolation proves the primitive, never the composition"; each stage is scored separately so a
  failure names the culprit. Worst over 64 steps: gates 2.3e-7, rule 6.0e-6, gnorm 1.3e-5 maxAbs/rms,
  all at cosine 1.000000000.
- **Still reused, unchanged:** `ensureMambaConv` covers the causal conv exactly — same shape, same
  SiLU, same ring window; DeltaNet is bias-free, so it binds an all-zero `convB` (the `moeZeroBias`
  precedent).

**Phase 3 — wire it. DONE.** `lw.isDeltaNet` beside `lw.isMamba`, `dnetRunParams`, per-layer
`{mambaWin, dnState}`, `FeatDeltaNet` declared for webgpu only, and the blanket `arch.qwen35`
refusal replaced by a fall-through so the decline moves to the feature gate. CUDA and Metal decline
there, which is what the hardware matrix now records.

**The plan MISSED half the work, and it is worth naming.** It scoped the mixer and nothing else.
But the family's *softmax* layers are not ordinary GQA either: `attn_output_gate` makes `q_proj`
double width, `[query ‖ gate]` interleaved PER HEAD, with the context scaled by `sigmoid(gate)`
before `o_proj`. Nothing in any backend had that. Two extra kernels (`deltaQSplit`,
`deltaAttnGate`) — the split happens on the ACTIVATION, not the weight, because slicing rows out
of a quantized `WeightMat` with its per-group scales is real surgery while splitting a 6144-float
activation is a copy. Reading the fused weight as an ordinary `q_proj` measures cosine 0.90 with a
drifting signature: plausible logits from the wrong tensor.

**Phase 4 — end-to-end. DONE.** `TestQwen35ResidentParity`, resident vs the CPU `runLayersQwen35`
at matched weight quant, over 128 tokens:

| fixture | worst cosine | drift | argmax |
|---|---|---|---|
| `qwen3_5-tiny` (dense) | 0.999919 | 0.0001 | 127/128 |
| `qwen3_5_moe-tiny` (sparse) | 0.995883 | 0.0041 | 127/128 |

Both siblings, because the MoE one composes the DeltaNet mixer with the sparse router + stacked
experts + shared expert in the same layer and NOTHING gated that pairing — mixer+MoE is gated for
Mamba-2, the mixer alone by the dense fixture, neither for this. The MoE fixture is new
(`pin_qwen3_5_forward.py --moe`) and its CPU forward is gated against transformers 5.12 at cosine
1.000000, so the chain is resident ≡ CPU ≡ HF for both.

Checked by mutation, all caught: the GVA head map inverted, the `v` slice offset dropped, the
attention output gate not applied, the q/gate split read as two blocks instead of per-head, and the
recurrent state never reset. The last one needed three attempts to catch — comparing generation 2
against the CPU does NOT see it (the decay gate shrinks stale state faster than 16 tokens of
comparison notices), so the check became a REPLAY: same tokens after `Reset`, the resident must
reproduce its own logits to f32 determinism.

Two things the drift check earned separately from the absolute floor: a recurrence that is merely
coarse sits flat and low, one that is wired wrong decays. Every wiring mutation above was caught by
the DRIFT bound, several of them while still above the cosine floor.

**Phase 4 (original scope) — `residentParity` on the scaled fixture against the CPU forward, then
the admission tests and the hardware matrix regenerate themselves.** The admission tests and both
matrices did regenerate themselves, as scoped.

## The head_dim 256 wall — what asking for tok/s found

Chasing the speed number produced a 4-layer fixture at fully released width, and it declined
residency on the first run:

    gpu: resident decode declines head_dim=256 > 128 (attention kernel workgroup is 128-wide)

**Every released model in this family is head_dim 256** — Qwen3.8-27B, Qwen3.6-35B-A3B and
Qwen3-Next-80B, read from their configs. The single-query attention kernels put one lane on each
head dim at `@workgroup_size(128)`, and `attnHeadDimSupported` correctly refuses anything wider
rather than dot only the first half. So the residency landed above was real, gated, and unable to
run a single released checkpoint — for a reason with nothing to do with DeltaNet.

The tiny fixtures use head_dim 32. Nothing in the correctness work could have found this; only
asking a performance question at real width did. **That is the argument for the real-width fixture,
and it is a different argument than the one Phase 1 originally made** (which was about numerics).

**Fixed properly, not raised one notch.** The first cut widened the workgroup to 256, which would
have covered this family and hit the same wall again at Gemma 4's head_dim-512 global layers —
and 256 is WebGPU's guaranteed `maxComputeInvocationsPerWorkgroup`, so one-lane-per-dim has
nowhere further to go. The shipped version instead gives each lane a **stride** of dims (`d0`,
`d0+WG`, `d0+2·WG`, …), so a fixed 256-wide workgroup covers any head_dim up to
`attnWGWide × attnMaxPerLane` = 2048. head_dim stopped being a reason to decline.

The narrow kernels are untouched and still serve head_dim ≤ 128: striding costs a per-lane array
and two extra loops, and every ordinary model would pay that for nothing. The wide variants are
built from ONE template (`attnWideTemplateWGSL`) with three small per-precision fetch expressions
(f32 / f16-unpack / int8-unpack-and-scale) — the online-softmax algorithm is the part that is easy
to get subtly wrong and hard to notice, and three copies of it is three places for a fix to land
in two.

Gated against `refAttn`, the same Go reference `TestAttnBlock_parity` already uses, at head_dim
64, 128, 256, **257**, 512 and 1024 — cosine 1.000000000 at every one. The list is chosen for the
striding boundaries, not for the model: 256 is `nper=1`, the degenerate case where a stride bug
hides completely, and 257 is the ragged case where lane 0 owns two dims and the rest own one. A
test that ran only 256 would pass with the stride loop entirely broken. It also asserts no output
element is exactly 0, because a lane that stops early leaves whole dim ranges untouched and a
cosine over a mostly-correct vector will not notice.

## Measured: 11.4–12.2× CPU decode at released width

`TestQwen35ResidentDecodeRate`, same box, matched weight quant, warm:

| fixture | quant | resident | CPU | ratio |
|---|---|---|---|---|
| 4 layers @ released width (hidden 5120, hd 256, inter 17408) | int8int8 | 177.3 tok/s (1.410 ms/layer) | 14.6 tok/s (17.172 ms/layer) | **12.18×** |
| same | int4 | 235.1 tok/s (1.06 ms/layer) | 20.6 tok/s (12.12 ms/layer) | **11.40×** |
| `qwen3_5-tiny` (hidden 64) | int8int8 | 2529 tok/s | 5746 tok/s | 0.44× |
| `qwen3_5_moe-tiny` (hidden 64) | int8int8 | 1283 tok/s | 3764 tok/s | 0.34× |

The tiny rows are in the table on purpose. They are dispatch-bound — roughly 20 GPU dispatches per
DeltaNet layer against microseconds of arithmetic — and they are what a casual "run the benchmark"
would have reported. Publishing only the flattering row is how a 0.34× becomes a footnote.

**The kill criterion is answered.** This is not the CUDA-graphs 1.01× that had to be relabelled a
safety improvement; residency is a real decode win for this family.

**Quote the RATIO, not an extrapolated absolute.** ms/layer × 64 gives ~15 tok/s resident and
~1.3 tok/s CPU for the 27B, but the CPU half of that is about 2× faster than the 0.656 tok/s
actually measured on the real 27B — the fixture omits the 248320-row LM head (~5% of real
per-token MACs) and runs a short context. The ratio is robust because both arms omit the same
things; the absolute is not.

## What is NOT done

- **The released 27B has not been run resident, and cannot be on this box.** It needs 15.3 GB at
  int4 against 8 GB of VRAM. Everything above is real geometry at reduced DEPTH.
- **Quantization at width is measured for speed, not quality.** The parity gate at released width
  (4 layers, int8int8) holds cosine 0.995 with drift 0.0015, which is reassuring but is 4 layers,
  not 64 — and error in a recurrent stack compounds with depth. Separately the resident path
  quantizes `in_proj_b`/`in_proj_a` to W8A8 where the CPU deliberately keeps them f32 (they feed
  the write/decay gates, where the recurrence is most precision-sensitive).
- **CUDA and Metal.** Both decline at the feature gate, correctly. CUDA is the next step and now
  has a settled design to port rather than one to invent — including the head_dim 256 question,
  which CUDA will meet too.

## Kill criteria, stated up front

- **The delta-rule kernel cannot match the CPU reference to the resident-path tolerance** (the other
  resident gates use 0.95 cosine against int4/int8 noise; f32-vs-f32 should be far tighter). A
  recurrence that drifts is worse than no kernel — stop and report rather than loosening a floor.
- **Residency is admitted but not faster.** The CUDA-graphs precedent applies: a capability that
  measures 1.01× is a safety improvement, not a speed one, and should be labelled that way rather
  than shipped as a win. Measure decode on the scaled fixture resident-vs-CPU before declaring
  anything.
- **The blanket `arch.qwen35` refusal cannot be made conditional safely** — e.g. if `qwen3_next`'s
  fused-projection variant or the MoE siblings need paths this work does not cover. Then the feature
  is declared for the dense family only, or held.
