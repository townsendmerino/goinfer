# Gated-DeltaNet residency — the plan, and why WebGPU proves it first

**Status: PHASE 2 DONE — 2026-08-19.** The delta-rule and norm kernels exist and are gated against
the CPU reference at real head geometry (`gpu/deltanet.go`, `gpu/deltanet_test.go`). Phases 1/3/4
are open. Written at Phase 0 so the phases, the reuse claims and the kill criteria were on record
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

**Phase 1 — the fixture.** `scripts/pin_qwen35_scaled.py`: real DeltaNet geometry (key/value head
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
- **The gated norm needs no shader variant.** Mamba's `ensureMambaGNorm` is already the right
  computation with `nGroups=nv, groupSize=hv`; DeltaNet's `[hv]` weight is shared across heads, so
  Phase 3 tiles it to `[nv*hv]` ONCE at load. Zero per-token cost, one fewer kernel to maintain.

**Phase 3 — wire it.** `lw.isDeltaNet` beside `lw.isMamba` in `encodeLayer`, per-layer state
allocation, the conv window, and `FeatDeltaNet` declared for webgpu only. `decodeRunnerEligible`'s
blanket `arch.qwen35 != nil` refusal becomes conditional — carefully, because that refusal currently
protects three families.

**Phase 4 — end-to-end.** `residentParity` on the scaled fixture against the CPU forward, then the
admission tests and the hardware matrix regenerate themselves.

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
