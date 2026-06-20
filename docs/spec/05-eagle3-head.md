# 05 — EAGLE-3 feature head

> Status: **proposal**. Depends on [00-core](./00-core.md). Highest effort, but it is
> the general-purpose SOTA and the strongest moat. Tackle after the cheap wins
> ([01](./01-grammar-fused.md), [02](./02-cache-ngram.md), [04](./04-adaptive-depth.md)).

## Idea

The cheap drafters cover structure (01) and copied text (02) but not **novel
prose/reasoning**, where the target distribution is the model's own and nothing
external matches it. The proven way to draft there is a small **feature-level**
head that autoregresses over the target model's **hidden states** and predicts the
next token directly — EAGLE / EAGLE-2 / EAGLE-3. As of early 2026 EAGLE-3 is the
production standard (reported ~0.8+ acceptance, 3–4× throughput) and is merged into
vLLM, SGLang, and TensorRT-LLM.

By [00-core](./00-core.md) §1, the head wins by directly minimizing `TV(q, p)`: it is
*trained* to match the target, so it pushes acceptance up exactly in the region the
other sources cannot reach.

## Why goinfer is well-placed

- It **owns the full forward pass in pure Go**, so exposing the target's hidden
  states (the head's input) is a clean internal seam, not a hack — unlike wrapping a
  closed runtime.
- "Pure-Go EAGLE-3 in a single static binary" is a position no other maintained
  runtime occupies; it pairs with the [03 router](./03-router-tree.md) and tree
  verification already specified.
- The head is tiny (roughly one transformer layer) — feasible to run on the existing
  SIMD / WebGPU matmul paths without new heavy dependencies.

## What it requires (the honest cost)

- **A trained head per model family** (and often per checkpoint). This is the real
  expense — either train heads (needs data + a training pipeline, currently outside
  goinfer's pure-inference scope) or import/convert existing open EAGLE-3 heads where
  license permits. Decide import-vs-train early; see IP note below.
- A clean **hidden-state seam** in `decoder` to feed the head (read-only export of
  the residual/feature stream at the chosen layer(s)).
- Head **autoregression + tree drafting** (EAGLE-2/3 build a draft tree from the head;
  this is the natural input to [03](./03-router-tree.md)).
- **Numerics parity** held to goinfer's existing bar: the head is part of the draft
  path only (losslessness is still enforced by verification), but its outputs should
  be parity-checked against the reference head implementation to ensure acceptance
  matches the published numbers.

## Licensing / IP note

EAGLE's reference implementation (SafeAILab/EAGLE) and Medusa are released as open
source — last checked Apache-2.0, which carries an **express patent grant** for the
contributors' contribution; **verify the LICENSE on the specific repo/weights before
importing**. Reimplementing the *method* from the papers is the cleaner path for an
MIT project; importing *weights* pulls in their license. Either way this is the spoke
most worth a deliberate license decision — note it in the PR. (See the project-level
patent discussion; the foundational draft/verify loop has separate Google patent
activity independent of EAGLE's grant.)

## Risks / open questions

- **Per-family head availability** is the gating problem; without a head a family
  falls back to 01/02/plain decode (graceful, but no novel-text speedup).
- Hidden-state seam must not perturb the target forward numerics (read-only).
- Training pipeline, if we train, is a new capability area for the project — scope it
  separately or rely on imports initially.
- Acceptance is workload- and family-dependent; reproduce published numbers on our
  harness before claiming them.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); output ≡ baseline.
- Acceptance parity: reproduce reference EAGLE-3 acceptance on a matching
  model/workload before integrating, to confirm the head/seam is faithful.
- Speed/acceptance: `chat` / `reasoning` suites in the [00-core](./00-core.md)
  harness, plus combined runs with [03](./03-router-tree.md) (head + grammar + n-gram
  in one tree) to measure the full-stack number.
