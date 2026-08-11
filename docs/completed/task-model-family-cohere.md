# Model family: Cohere / Command-R (+ Aya)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **Why this one:** the **largest unsupported cluster** in the popular-model catalog — 8
> entries (`command-r`, `command-r-plus`, `command-r7b`, `command-a`, `command-r7b-arabic`,
> `aya`, `aya-expanse`, `north-mini-code-1.0`) — and **Aya is the only strong multilingual
> family** in that set, which nothing in the registry currently serves. Two genuinely new
> primitives; everything else rides shipped features. Both new primitives are **reusable**
> (see *Leverage*).
>
> **Verify before building.** The deltas below are from the HF `CohereForCausalLM` /
> `Cohere2ForCausalLM` shape as understood; **read the actual `config.json` + modeling file
> first** and correct this table. Per the playbook: the hypothesis is a target to test.

## Architecture deltas

| delta | difficulty | note |
|---|---|---|
| **LayerNorm, not RMSNorm** | **NEW primitive**, bounded | Cohere norms are LayerNorm (no bias): **mean-subtract + variance**, vs RMSNorm's no-mean. goinfer's norm stack is RMSNorm-centric (+ Gemma's `(1+w)` offset). Needs a norm-kind in the descriptor + one kernel. |
| **Parallel attention + MLP block** | **NEW forward shape**, bounded | `x + attn(norm(x)) + mlp(norm(x))` — both sublayers read the **same** normed input and sum into the residual, vs the sequential attn→resid→norm→mlp→resid. A third `NormPlacement` alongside `NormPre2` and sandwich. |
| `logit_scale` on output logits | **✅ have it** | `FeatLogitScale` (Granite's `logits_scaling`). |
| No biases anywhere | ✅ | bias is already optional. |
| Tied embeddings, GQA, 256k vocab | ✅ | all shipped. |
| **cohere2 only** (R7B / Command-A / -arabic): interleaved **sliding-window** + **NoPE on the full-attention layers** | bounded | sliding window ✅ (`FeatSlidingWindow`); per-layer NoPE is the same per-layer rope-enable flag **Llama 4 also wants** — build it once. |
| QK-norm in cohere2? | verify | if present it's ✅ (`FeatQKNorm`). |

No **intractable** blocker — no recurrent state, no new cache type, no own execution engine.
This is a descriptor + loader + two bounded primitives.

## Taxonomy additions (so nothing ships silently wrong)

Two new arch-derivable `ResidentFeature`s in `decoder/features.go`:

- `FeatLayerNorm` — norm kind ≠ RMS
- `FeatParallelBlock` — `NormPlacement == parallel`

**Declared by no backend initially.** Consequence, by construction: Cohere runs **correct on
CPU** on day one, and **every GPU backend declines it to CPU automatically** via
`MissingResidentFeatures` — no silent-wrong risk, no hand-coded decline. GPU residency lands
later, per-backend, when the two kernels do, and the generated `hardware-matrix.md` cells
flip themselves. This is the taxonomy working exactly as designed.

## Phasing

- **Phase 1 — `cohere`** (Command-R, Command-R+, Aya, Aya-Expanse): descriptor + loader +
  the two primitives (LayerNorm kernel, parallel-block forward) + T1 tiny golden + T3
  real-checkpoint parity. **Ships the family CPU-correct.**
- **Phase 2 — `cohere2`** (Command-R7B, Command-A, -arabic): + interleaved sliding-window
  (have it) + per-layer NoPE (build once; Llama 4 reuses it).
- **Phase 3 — GPU residency (demand-gated).** Add LayerNorm + parallel-block kernels per
  backend, declare the two features, let the matrix cells flip. Not required to ship the
  family.

## Test targets

**Aya-Expanse-8B** or **Command-R7B** — small enough for a real-checkpoint T3 on either box,
and R7B exercises the cohere2 path. Command-A (111B) / Command-R+ (104B) are **RAM-gated,
not arch-gated** — support them by construction, qualify them when hardware allows (proxy per
the qualification sweep).

## Parity + gates

- **T1:** generated tiny-cohere golden, argmax-exact + logit cosine.
- **T3:** real-checkpoint vs the HF bf16 reference (argmax + cosine), recorded in
  `parity_manifest.json` with `validated_at`.
- **Break-it-first** on both new primitives — perturb LayerNorm's mean-subtraction and the
  parallel-sum ordering, confirm the gate goes **RED**, revert. A new primitive whose gate
  can't fail is not gated.
- **Known side-effect:** the two primitives touch **core** forward files, so they re-stale
  every `uses: [core]` family's `deps_hash`. That's the scripted, goldens-gated path
  (`scripts/refresh_parity_hashes.sh`) — not a re-validation.

## Leverage (why the cost is lower than it looks)

Both new primitives unlock more than Cohere:

- **LayerNorm** is the last major norm variant goinfer lacks — it opens the whole
  pre-RMSNorm generation of models.
- **Parallel attention+MLP** is the GPT-J / Falcon block shape — it's the structural
  blocker behind `falcon`/`falcon2`/`falcon3` and several others in the same catalog.
- **Per-layer NoPE** (Phase 2) is a prerequisite Llama 4 already needs.

So an 8-entry family add also retires the primitives blocking a second cluster.
