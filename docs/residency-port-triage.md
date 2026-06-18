# Residency-port ROI triage — Llama4 vs Granite vs Nemotron

Read-only scoping (no production change). Prize per family ≈ **3× / ~19 ms→~10 ms per token**
(the architecture-independent §2 fence-tax recovery measured in `decode-twe-split.md`). Cost is
what differs. Verdict up front:

**Recommended FIRST port: `llama4_text`** — the only one of the three that is a *residency
port* rather than a *new execution engine*. **Granite-4.0-H and Nemotron-H are INTRACTABLE**
(Mamba-2 selective-SSM recurrence; the resident DecodeRunner has no recurrent-state path), and
goinfer carries **only the hybrid variant** of each — there is no dense fallback to serve. One
caveat on Llama4 (below): no Llama4 checkpoint fits this 8 GB box resident, so the *realized*
throughput prize is VRAM-gated; the port's on-box value is correctness + reusable infra.

## Variants in goinfer's registry (one each — this is the crux)

| family | registry arch | variant | Mamba? | serve-relevant |
|---|---|---|---|---|
| Llama4 | `llama4_text` | MoE transformer (Scout/Maverick) | **no** | yes (only one) |
| Granite | `granitemoehybrid` | Granite-4.0-H: Mamba-2 ⊕ attn ⊕ MoE/every-layer | **yes** | yes (only one) |
| Nemotron | `nemotron_h` | Nemotron-H: single-op block {mamba\|NoPE-attn\|relu²-MLP} | **yes** | yes (only one) |

There is no `granite3`/`nemotron4` dense entry — the served variant **is** the intractable one
for Granite and Nemotron. (Generic dense Llama/Granite-3-style checkpoints that load via the
plain `llama` path are already resident; they are not these archs.)

## Gate tables

### `llama4_text` — trips exactly ONE gate

| `decodeRunnerEligible()` gate | value | blocks? |
|---|---|---|
| **`llama4 != nil`** | **true** | **YES** (own-forward) |
| gemma4 / qwen35 / granite / nemotron / mla | nil | no |
| MoE set & `!moeResidentEligible()` | MoE set, eligible **true** | no (C3 covers) |
| NonGatedMLP / LearnedPosEmbed / OutBias | false | no |
| NormPlacement != NormPre2 | **NormPre2** | no |
| FinalLogitSoftcap / AttnLogitSoftcap | 0 / 0 | no |

Llama4 is **NormPre2 + NormRMS, no QKV bias, no sandwich norm, no softcap** — unlike Gemma-4 it
trips no structural norm/softcap gate. All divergence lives inside `runLayersLlama4`, and each
piece maps onto an existing C-lever:

| Llama4 forward divergence | difficulty | reuse |
|---|---|---|
| dense/MoE layer interleave (`isMoE[i]`) | **TRIVIAL** | C3 `runLayer.isMoE` is already per-layer |
| MoE top-1 sigmoid + ungated shared | **TRIVIAL** | C3 route kernel covers sigmoid/top-k/shared-ungated |
| parameter-free L2 QK-norm (RMS-over-headDim, no weight) | **TRIVIAL** | C1 `qkNorm` kernel with a unit-weight buffer == L2 |
| per-layer NoPE (skip RoPE on some layers) | **BOUNDED** | per-layer rope-enable flag (C5/C7 per-layer rope infra; NoPE = no-op the dispatch) |
| attn-temperature on NoPE query (`log1p(floor((pos+1)/floor)·scale)+1`) | **BOUNDED** | per-token query scalar → build-once-shaped uniform written per token (CPU-computed) |
| MoE **input**-scaling (sigmoid·h INTO the SwiGLU expert, ≠ C3's output-scale) | **BOUNDED KERNEL** | one `moeExpert` variant that scales the int8 activation pre-GEMV; the only genuinely new MoE bit |
| chunked/local attention | **none** | goinfer already attends full-causal (parity gates accept it) |

No STRUCTURAL blocker beyond standing up the own-forward path itself, and no INTRACTABLE one.

### `nemotron_h` — INTRACTABLE (Mamba)

| gate | value | blocks? |
|---|---|---|
| **`nemotron != nil`** | **true** | **YES** (own-forward) |
| per-layer Mamba-2 mixer (DState/NGroups/DConv, persistent SSM state) | yes | **INTRACTABLE** |
| **NonGatedMLP** (relu² FFN, `Act=ActReLU2`) | true | YES — *but* BOUNDED-kernel, moot behind Mamba |
| NoPE attention layers | yes | BOUNDED, moot |

### `granitemoehybrid` (Granite-4.0-H) — INTRACTABLE (Mamba)

| gate | value | blocks? |
|---|---|---|
| **`granite != nil`** | **true** | **YES** (own-forward) |
| `layerIsMamba(i)` per-layer Mamba-2 + persistent SSM state | yes | **INTRACTABLE** |
| MoE on every layer | eligible | C3, moot behind Mamba |

**Why intractable:** `mamba2.go` is a *selective state-space recurrence* — sequential per-head
SSM state (`st.ssm`) carried token-to-token, plus a causal-conv window. The resident
DecodeRunner is a fixed, alloc-free, one-command-buffer matmul graph with **no recurrent-state
slot and no sequential scan**. Residency-porting Mamba is a different execution engine (its own
funded lever), not a residency port. A per-layer mix of resident-attention + staged-mamba also
breaks the single-command-buffer model. **Exclude both — they are not cheap wins.**

## Port-cost estimate (serve-relevant variant)

| family | cost | why |
|---|---|---|
| **Llama4 (`llama4_text`)** | **M (medium)** | own-forward path standing up; 3 TRIVIAL + 2 BOUNDED + 1 BOUNDED-KERNEL deltas, all reusing C1/C3/C5/C6/C7; parity gates already exist (tiny golden + real Scout staged) |
| Granite-4.0-H | **INTRACTABLE** | needs a resident Mamba-2 SSM engine first (≫ a port) |
| Nemotron-H | **INTRACTABLE** | same; plus relu² MLP (bounded, moot) |

**VRAM caveat (Llama4):** Scout (109B) / Maverick (400B) don't fit 8 GB resident even at int4
(~55 GB+). On *this* box the model stages regardless of eligibility, so the realized §2 prize is
~0 here; the port delivers **correctness + reusable transformer-residency infra** now, and the
throughput 3× on a card that fits a Llama4 checkpoint. Granite-tiny / Nemotron-Nano-9B *do* fit —
but are Mamba-blocked, so they can't realize it either. **No one of the three yields an on-box
realized prize today**; Llama4 is the right first port for cost + infra, not for an 8 GB tok/s win.

## Ranking + recommendation

1. **`llama4_text` — PORT FIRST.** Only tractable family; **M** cost; reuses every shipped
   C-lever; one gate (`llama4 != nil`) with bounded divergences behind it. Builds reusable
   transformer-residency infra (per-layer NoPE flag, parameter-free L2 QK-norm, per-token attn
   temperature uniform, MoE input-scaling). Realized tok/s prize is VRAM-gated on this box.
2. **Granite-4.0-H / Nemotron-H — DO NOT PORT as residency.** INTRACTABLE (Mamba-2 SSM). Fund a
   resident SSM engine separately if ever; do not mistake their small-model fit for a cheap win.

## Phase sketch for the recommended first port (`llama4_text`), parity-gated

- **Phase 0 — baseline + oracle.** Confirm `llama4_text` forces staged via `decodeRunnerEligible()`;
  capture the staged/CPU reference (token sequence + per-position logits) on the tiny-Llama4 golden
  (and real Scout if a fitting card is available) as the parity oracle.
- **Phase 1 — blockers.** (This triage.) `llama4 != nil` + the 6 forward deltas above.
- **Phase 2a (TRIVIAL)** — per-layer `isMoE`/dense already in C3; wire L2 QK-norm by binding a
  unit-weight buffer into the C1 `qkNorm` kernel (no kernel change).
- **Phase 2b (BOUNDED)** — per-layer NoPE: a `runLayer` rope-enable flag; NoPE layers no-op the
  rope/qkv-finalize rope half.
- **Phase 2c (BOUNDED)** — attn-temperature: CPU-compute the per-token NoPE query scalar and write
  it into a build-once uniform (no per-token alloc); fold into the attention scale.
- **Phase 2d (BOUNDED KERNEL)** — MoE input-scaling: a `moeExpert` variant that multiplies the
  int8 activation by `sigmoid(logit_e)` before the expert GEMV; shared expert runs on the unscaled
  activation, added ungated (C3d shared path).
- **Phase 3 — PARITY GATE (must pass first).** Resident logits/sampled tokens match the Phase-0
  oracle within tolerance over a multi-token sequence (incl. NoPE + MoE layers); **confirm every
  existing resident model is bit-unchanged** (new flags default off / no-op).
- **Phase 4 — flip eligibility (guarded).** Admit `llama4 != nil` only when the implemented deltas
  cover the config; other own-forward families (gemma4/qwen35/granite/nemotron) stay gated.
- **Phase 5 — measure.** Confirm the resident path is taken (log/gate); report ms/token + tok/s vs
  staged. On 8 GB this is gated by fit (note it explicitly); the 3× lands on a card that holds a
  Llama4 checkpoint. Run the FULL `-tags gpu` suite to prove no resident-model regression.
