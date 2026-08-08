# C8 softcap residency — Phase 1 STOP: target is not softcap-dense

**Outcome: stopped at Phase 1 per the funding rule** ("if it trips a gate other than
softcap, STOP and report — we only fund softcap-dense here"). The target Gemma-4-E2B is
blocked from the resident DecodeRunner by **own-forward + sandwich-norm gates, not softcap**.
Softcap-dense alone — the funded scope — does not move it onto the resident path. No
production code was changed.

## Phase 1 — measured eligibility gates (Gemma-4-E2B, `gemma-4-E2B_q4_0-it.gguf`)

Loaded the target and enumerated every `decodeRunnerEligible()` gate (diagnostic; reverted):

| gate | value | softcap? | blocks? |
|---|---|---|---|
| `arch.Name` | `"gemma4"` | — | — |
| **`gemma4 != nil`** | **true** | **no** (own non-uniform forward) | **YES** |
| **`NormPlacement`** | **`sandwich`** (need `NormPre2`) | **no** (4 norms/layer vs 2) | **YES** |
| `FinalLogitSoftcap` | `30` | **yes** | yes |
| `AttnLogitSoftcap` | `0` | yes | no (not present) |
| qwen35 / granite / nemotron / llama4 / mla / MoE | nil/false | — | no |
| NonGatedMLP / LearnedPosEmbed / OutBias | false | — | no |
| `DecodeRunnerEligible()` | **false** | | |

## Why softcap-dense is necessary-but-insufficient here

The task's premise was that softcap is Gemma's gate. Measurement says otherwise — softcap
is the *least* of it:

1. **`gemma4 != nil` (own forward, `forward_gemma4.go`).** Gemma-4 runs `runLayersGemma4`:
   alternating local (256) / global (512) attention, **different local vs global HeadDim and
   rotary width**, per-layer KV-head counts, scale-less v-norm, matformer E2B layout. The
   resident runner assumes one uniform GQA geometry per model; this is a full own-forward
   port (its own C-lever), explicitly out of scope here.
2. **`NormPlacement = sandwich`.** Gemma applies four norms per layer (input + post-attention
   + pre-FFN + post-FFN); the resident runner's plan does two (Llama-style pre-norm: attn_norm,
   mlp_norm). A separate resident-runner change (extra norm dispatches + build-once weights).
3. **Differing local/global rope tables** also trip `ropeResidentCompatible()` (the runner's
   single `rotaryDim/2` dispatch can't represent two head-dim widths) — same root as (1).
4. **`FinalLogitSoftcap = 30` is the only softcap gate that fires; `AttnLogitSoftcap = 0`** —
   Gemma-4 has no attention soft-cap (only Gemma-2 did). So even the attention-softcap half of
   the funded work doesn't apply to this target.

If softcap were implemented on the resident path and `decodeRunnerEligible()` flipped to admit
softcap-dense, Gemma-4-E2B would still trip (1)–(3) and be routed to a runner that cannot
represent its forward → **it would fail the parity gate** (wrong logits). Admitting it would be
a correctness regression, not a residency win.

## No pure softcap-dense model exists to fund

The only softcap-dense architecture is Gemma-2 — and it is **not in goinfer's arch set**
(`registry.go` has `gemma3`, `gemma4` only; no `gemma2`). Gemma-3 sets `FinalLogitSoftcap = 0`
(softcap dropped) and is itself sandwich-norm + own-forward-class. So:

- There is **no model on this box, and no arch in goinfer, that trips softcap *and nothing
  else***. A softcap-on-resident kernel could not even be parity-validated against a real model
  here (resident-eligible models are all non-softcap; all softcap models are blocked for
  other reasons). That alone makes a standalone softcap-dense pass unfundable as specified.

## What actually unlocks resident Gemma (out of this scope)

Resident Gemma is a **gemma-own-forward residency lever**, not a softcap lever. It needs, as one
bundle: sandwich-norm (4 norms/layer), per-layer alternating local/global attention with
**two head-dim/rotary widths**, per-layer KV-head counts, scale-less v-norm, AND final-logit
softcap. That is the Gemma analogue of the MLA (C4) / hybrid own-forward ports — a multi-phase
family port, parity-gated, not the one-kernel softcap change this task scoped.

## Correction to a prior note

`docs/decode-twe-split.md` listed "softcap residency (C8)" as the Gemma unlock. That
oversimplified: softcap is one of ≥3 blockers, and the dominant one is Gemma-4's own forward.
The §2 prize (~19 ms/token, 3×) for moving Gemma off the 61.5 ms staged path is real, but the
lever is a **gemma own-forward residency port**, not softcap-dense.

## Recommendation

Do not implement softcap-dense as a standalone pass — it unlocks nothing on this box and can't
be parity-validated. If resident Gemma is worth the §2 prize, fund the **gemma4 own-forward
residency port** (sandwich-norm + dual-width alternating attention + per-layer KV + v-norm +
final softcap) as one parity-gated family lever, the way MLA/hybrid families were done.
