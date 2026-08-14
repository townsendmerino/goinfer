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

- ~~**DeepSeek-V4**~~ — **MOVED OUT OF TIER A (2026-08-09). Config-verified: NOT an alias.**
  The assumption below ("a config-delta on `deepseekArchitecture` plus possibly a routing
  tweak") was tested against the real `config.json` + `inference/model.py` and **refuted**:
  `kv_lora_rank` is absent, attention is `sparse_attn` over a learned `Indexer` with a KV
  `Compressor` and sliding window, and the router has no `n_group`/`topk_group` and a third
  `scoring_func` (`sqrtsoftplus`). Eight new primitives. Full findings and the re-scope:
  `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`. **Kimi-K3 is PARTIAL** — its MLA and router
  *are* ours, but MLA runs on only 24 of 93 layers behind a linear-attention (KDA) mixer.

- **Other DeepSeek-V3-shaped MoEs** — ⚠ **the "de facto standard" premise did not survive its
  first two test cases.** Both 2026 flagships moved off the V3 shape within one generation
  (V4 replaced the KV latent outright; K3 demoted MLA to a quarter of its layers). The
  corrected rule: **frontier MoE attention is diverging, not converging.** "Alias-first" is
  still the right *starting hypothesis* — it is cheap and sometimes right (Kimi-K2 was) — but
  it is a hypothesis to config-verify every time, never an estimate to fund. The
  `scoring_func`-keying check (the Kimi gotcha) generalized in an unexpected direction too: K3
  renames the key entirely (`moe_router_activation_func`), and V4 introduces a third value.

- **Watch — Kimi K3** (reportedly in development, ~3–4T params, billed as "the next
  major architecture jump"). *Not* a Kimi K2.x point release — K2 through K2.7-Code
  are the same DeepSeek-V3 arch and already supported via `kimi_k2` (no work). K3 is
  the one that may bring a genuinely new shape (or new config quirks), so unlike the
  K2.x line it could need real adapter work. **Default assumption: DeepSeek-V3-shaped
  (alias-first) until proven otherwise — re-check the arch when weights drop**; only
  if attention/MoE/routing diverged from V3 does it leave alias-first territory.

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
