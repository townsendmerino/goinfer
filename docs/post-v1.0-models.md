# Post-v1.0 model families (the breadth backlog, after the axes are covered)

> **Audience:** internal planning. The capability-driven family phase is **done**:
> goinfer now covers all three efficient-attention axes — **gated-linear**
> (DeltaNet, `qwen3_5_moe`), **state-space** (Mamba-2: Granite, Nemotron-H), and
> **latent-KV** (MLA: DeepSeek-V2/V3) — plus softmax-GQA and learned-pos. So every
> family below adds **breadth/popularity, not a new capability axis**. They're
> backlog, not milestones. **Kimi is being done now** (it rides the just-shipped
> MLA path — see `prompts/kimi.md`); everything else waits behind v1.0.

## The framing: families are no longer the high-leverage work

With the axes covered, the marginal family is a descriptor + loader + golden on an
*existing* path. That's cheap, but it's no longer where the leverage is. Two things
matter more than a sixth/seventh family and should land **before** this backlog:

1. **Validate what just shipped.** GLM, Granite, Nemotron-H, DeepSeek-V2/V3 arrived
   fast. Each needs its `parity_manifest.json` row + T1 golden + capability-matrix
   entry (`task-parity-coverage.md`, `task-capability-matrix.md`), or the support
   claims are unbacked and green CI is hiding asset-skips. Backfilling those five is
   worth more than any new family.
2. **Cut v1.0.** The roadmap's v1.0 trigger was "a second hybrid family survives the
   hybrid-cache abstraction." We now have *two* Mamba-2 hybrids **and** MLA — the
   abstraction absorbed far more than the bar. The evidence is overwhelming; freeze
   v1.0 and ship the capability matrix. **This backlog is explicitly post-v1.0.**

## The backlog (ordered by popularity-per-effort, all post-v1.0)

### Tier A — ride a just-built path for nearly free

- **DeepSeek-V4** — now the open-weight frontier. An evolution of the MLA + MoE path
  we just built; most likely a **config-delta on `deepseekArchitecture`** plus
  possibly a routing tweak. Verify the routing/expert-bias changes vs V3; gate with
  `weightDiff` + layer-slice (full oracle infeasible at V4 scale), same as V3/Kimi.
  Highest prestige in the backlog, probably the cheapest.

- **Other DeepSeek-V3-shaped MoEs** — the MLA + DeepSeekMoE shape is now a *de facto
  standard* (Kimi already rides it). As more 2026 frontier MoEs adopt MLA, each is a
  near-free alias on `deepseekArchitecture` with scalar deltas + the
  `scoring_func`-keying check (the Kimi gotcha). Treat new MLA models as alias-first.

### Tier B — cheap dense/MoE on existing rails (popularity filler)

- **Phi (Phi-4 / Phi-MoE)** — dense + MoE, no new primitive; popular for the
  on-device / efficient-deployment crowd. Pure descriptor + golden.
- **Newer Qwen dense/MoE** — the Qwen ecosystem keeps shipping; new dense or
  standard-MoE Qwen variants are descriptor-close to what we already run. (Qwen
  *hybrid* variants would ride DeltaNet/Mamba — also cheap.)
- **Yi / InternLM / other Llama-shaped families** — mostly `llama`-shaped; add only
  on concrete demand.

### Tier C — more Mamba hybrids, with a real caveat

These compound the Granite/Nemotron machinery (hybrid cache + per-layer-kind + MoE),
**but most use Mamba-1, not Mamba-2**, so they are *not* automatically free:

- **Jamba** (AI21, Apache-2.0, Mamba + attention + MoE, 256K ctx) and **Falcon
  Mamba** (pure SSM) — both **Mamba-1**. Riding them needs a **Mamba-1 scan variant**
  alongside `mamba2.go` (a small new primitive, not a descriptor add). Worth it only
  if a Mamba-1 family is actually in demand.
- **Hunyuan-TurboS** and other Mamba-Transformer hybrids — check the Mamba version
  first; Mamba-2 ones are near-free (Nemotron-class), Mamba-1 ones carry the variant
  cost above.

Decision rule for Tier C: **confirm Mamba-1 vs Mamba-2 before estimating** — it's
the difference between "alias + config" and "a new scan kernel + parity."

## Explicitly out of scope (not "families")

- **Diffusion LMs** (LLaDA/Mercury/Gemini-Diffusion class) — a different *decode
  loop* (parallel iterative refinement, bidirectional attention), not a registry
  adapter. It'd be a second decode engine and a GPU/compute-bound play — squarely
  against goinfer's CPU single-binary lane. Not a family; revisit only as a separate
  initiative if ever.
- **Full multimodal early-fusion / vision-out / audio** — the text-decoder-only call
  (drop the vision tower) stays the policy; full multimodal is its own track.
- **A third linear primitive (RWKV / RetNet / lightning-attention)** — we already
  cover gated-linear + state-space; a third recurrent mixer is diminishing returns on
  the axis. Only on strong, specific community demand.

## Selection rule going forward

Popularity is now the tiebreaker, not the lead — and "rides a shipped path" beats
"prestigious but new primitive." Concretely: **MLA-shaped MoEs (Tier A) > dense
fillers (Tier B) > Mamba-1 hybrids (Tier C, gated on the variant cost) >
out-of-scope.** And every one of them goes through the `parity-coverage-policy.md`
definition of done before it earns a README support row — no exceptions, especially
for the cheap ones.
