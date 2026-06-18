# Resident SSM/Mamba decode engine — scope (Granite-4.0-H / Nemotron-H)

MEASURE + CHARACTERIZE + DESIGN + ESTIMATE, read-only (a throwaway measurement harness +
this doc; no engine code). On the box, `-tags gpu`, granite-4.0-h-tiny-Q8_0 (6.9 GB, fits).

## Headline verdicts

- **Recurrence vs scan: it's a RECURRENCE (resident-able).** The earlier triage's
  "intractable, sequential scan" call was about *prefill*. Mamba-2 **decode** is
  `mamba2Step` — a bounded per-token update of persistent conv-state + SSM-state, exactly
  KV-cache-shaped. No scan, no position loop, no host round-trip. **This corrects
  `residency-port-triage.md` / the eligibility analysis, which wrongly excluded it.**
- **Prize: LARGE and REALIZABLE on this box — but it's CPU→GPU, not fence-tax.** The hybrid
  own-forward runs on **CPU** today; webgpu-staged is *slower*. granite-tiny: 300 ms/token
  CPU → estimated ~20–50 ms/token resident = **~6–15×**. granite-tiny FITS 8 GB (unlike
  Llama4), so this prize is actually collectable here.
- **Go/No-Go: GO.** Decode is a bounded recurrence; state slots in like a KV cache; the new
  work is ~3 contained kernels; the per-layer-kind branch has MoE/MLA precedent. Effort
  M–L. This is a *better realizable-prize* first port than Llama4 (which doesn't fit 8 GB).

## Phase 0 — measured prize (granite-4.0-h-tiny, best inter-token)

| path | ms/token | tok/s | resident? |
|---|---|---|---|
| CPU | 300 | 3.3 | no |
| webgpu (staged) | 498 | 2.0 | no |

The webgpu staged path is **slower than CPU** — the hybrid forward (`runLayersGranite`) is
custom Go running on CPU (`matvec`, `attendQuery`, the conv/SSM loops); webgpu only stages the
MoE matmuls, so it adds CPU↔GPU ping-pong per token and loses. **There is no fence-tax §2
prize here** (the path isn't staged-GPU-fenced; it's CPU). Instead the prize is moving the
entire forward onto the resident GPU runner. Estimate: a resident int8 ~7B with sparse
MoE (only active experts stream) lands ~20–50 ms/token (cf. resident dense 1.5B ≈ 10 ms;
Mamba layers carry small weights, MoE streams top-k), i.e. **~6–15× over 300 ms CPU**.
Unlike the transformer §2 prize, this one is collectable on the 8 GB card because the model fits.

## Phase 1 — characterize the hybrid decode

**Layer composition** (config-driven, mostly-Mamba):
- **Granite-4.0-H** (`runLayersGranite`): every layer is mixer = Mamba-2 (`isMambaLayer(i)`) OR
  plain GQA softmax attention (`graniteAttention`), and **MoE-on-every-layer** FFN, plus 4
  Granite scalar multipliers (embedding/residual/attention/logit).
- **Nemotron-H** (`runLayersNemotron`): single-op blocks, `blockKind[i] ∈ {mamba, NoPE-attn,
  relu²-MLP}`. Mostly mamba, periodic attention, dedicated MLP blocks.

**One Mamba-2 DECODE step (`mamba2Step`, mamba2.go) — the bounded recurrence:**
1. `in_proj` GEMV: h → `[z ‖ xBC ‖ dt]` (projDim = 2·dInner + 2·nGroups·N + nHeads).
2. **Depthwise causal conv** over xBC: per channel, K taps = K-1 from the conv ring window +
   current token, + bias + SiLU. Then push xBC into the ring (drop oldest). Bounded: convDim
   channels × K taps.
3. Split conv → x (dInner), B, C (nGroups·N each).
4. **Selective SSM state update** per head: for each `(head, pi∈P)`, loop `n∈N`:
   `S[pi,n] = S[pi,n]·dA + (dt·x[pi])·B[n]; y[pi] += S[pi,n]·C[n]`, where `dA =
   exp(softplus(dt+dtBias)·(−exp(aLog)))`. **Reads+writes `st.ssm` in place; NO position
   loop** — work is heads×P×N, all parallel.
5. **Gated grouped RMSNorm**: `y·SiLU(z)`, RMS-normed per `NormGroups`, ×weight.
6. `out_proj` GEMV → hidden.

**Persistent state (per Mamba layer), KV-cache-like:**
- `convWin`: ring of last K-1 xBC vectors → `[K-1, convDim]` (K≈4, convDim ≈ dInner+2·nGroups·N) — a few KB.
- `ssm`: `[nHeads·P·N]` f32 = dInner·N floats. For dInner≈4 k, N=128 → ~2 MB/layer; ×~40 mamba layers ≈ ~80 MB. Fits 8 GB trivially.

**What the resident runner lacks:** (a) the SSM state buffers as resident build-once tensors;
(b) three per-token dispatches — conv1d-state-update, selective-SSM-state-update,
gated-grouped-RMSNorm; (c) a per-layer *mixer-kind* branch (Mamba vs attention) in the plan.
The attention layers are plain GQA (Granite) / NoPE-GQA (Nemotron) → existing resident attention
(+ C5/C6/C7); the FFN is MoE (Granite, C3) / relu²-MLP (Nemotron, a bounded non-gated kernel).
**Only the SSM step is genuinely new.**

## Phase 2 — design the resident SSM path

**State buffers** (build-once, in `residentDecoder`, updated in place per token — the KV-cache
pattern): per mamba layer a `convWin` buffer `[K-1, convDim]` with a per-token rotating write
index (a uniform, like KV pos) and an `ssm` buffer `[nHeads·P·N]`. Reset on new generation.

**Kernels** (all per-token, fold into the alloc-free single compute pass; bind group + dims
built once):
1. **`mambaConv`** — one thread per conv channel: read K-1 from the ring + current xBC, FIR +
   bias + SiLU, write conv[c], and (one thread) advance the ring index. Geometry: convDim/64 WGs.
2. **`mambaSSM`** — one thread per `(head, pi)`: load `dA`/`dt` (precompute softplus+exp inline
   or a tiny pre-kernel), loop N updating `ssm[head,pi,n]` in place and accumulating `y[pi]`,
   add `D·x`. Each thread owns its `[pi,·]` row → no intra-dispatch hazard. Geometry: nHeads·P
   threads. This is the one nontrivial kernel; arithmetic is small (N≈128 MACs/thread).
3. **`mambaGatedNorm`** — `y·SiLU(z)` then grouped RMSNorm ×weight: a workgroup per group (≈ the
   existing rmsnorm kernel + a SiLU-gate input + per-group reduction). Close to existing norm.
4. `in_proj`/`out_proj` are the existing resident W8A8 GEMV (reuse).

**Interleave:** the runner's per-layer plan branches on mixer kind — Mamba layer dispatches
{in_proj, mambaConv, mambaSSM, mambaGatedNorm, out_proj}; attention layer dispatches the existing
resident attention; FFN is the existing MoE (C3) / relu²-MLP. All in one command buffer, one
Submit, one Poll. **Fits the alloc-free / one-compute-pass / one-Submit model** — the SSM state
read-modify-write is local per thread, the only new structural need is the mixer-kind branch
(precedent: C3 MoE-kind, C4 MLA-kind already branch per layer).

**Doesn't-fit risks:** none fundamental. The conv ring is a fixed small buffer (not a growing
cache); the SSM update is in-place per-thread (no atomics). The only subtlety is f32 state
precision (below).

## Phase 3 — feasibility, effort, go/no-go

**Feasible in the current runner** with bounded additions; no structural rewrite. The one
runner change (per-layer mixer-kind + SSM state buffers) mirrors the MoE (C3) and MLA (C4) ports.

**Parity is the spine.** The SSM recurrence accumulates state across tokens, so the GPU f32
update must track the CPU f32 `mamba2Step` within tolerance over a long sequence (state drift is
the failure mode). Reference = the staged/CPU path (mamba2.go), already the parity oracle for the
existing forward. Gate cosine ~1.0 / bounded maxAbs over 100+ tokens, mamba layers included.

**Effort: M–L.** Three new kernels (one nontrivial: `mambaSSM`), KV-cache-like state buffers, the
mixer-kind branch, plus wiring the already-resident attention/MoE/relu²-MLP per layer and the
trivial Granite multipliers. Bigger than a single C-lever, smaller than the MLA port.

**Phase sketch (the eventual build, parity-gated):**
- **P0** baseline + oracle: staged/CPU granite-tiny decode + per-position logits + per-layer SSM
  state snapshots as the recurrence oracle.
- **P1** resident SSM state buffers (convWin ring + ssm), build-once + reset, KV-cache-style.
- **P2** `mambaConv` kernel → parity vs CPU conv (single layer, multi-token).
- **P3** `mambaSSM` kernel → parity vs CPU selective update (the spine; long-sequence state drift).
- **P4** `mambaGatedNorm` + in/out_proj wiring → full Mamba-layer parity.
- **P5** per-layer mixer-kind branch; interleave resident attention (Granite GQA / Nemotron NoPE)
  + MoE (C3) / relu²-MLP; whole-model resident-vs-CPU parity (cosine ~1.0).
- **P6** flip `decodeRunnerEligible()` to admit granite/nemotron (guarded; other own-forward
  families stay gated); confirm existing resident models bit-unchanged.
- **P7** measure realized ms/token + tok/s vs 300 ms CPU; full `-tags gpu` suite.

## Go / No-Go

**GO — fund the build.** The reframe holds: Mamba *decode* is a resident-able bounded recurrence,
not a scan. The prize is large (~6–15× CPU→GPU) and, crucially, **collectable on the 8 GB box**
because granite-tiny / Nemotron-Nano-9B fit — making this a *better realized-prize* first port than
Llama4. Effort M–L, the one genuinely new piece is the `mambaSSM` kernel, and parity against the
existing f32 mamba2Step is the gate throughout. Recommend the gemma own-forward and SSM ports both
re-enter the queue (this scope removes SSM from the "intractable" bucket).
