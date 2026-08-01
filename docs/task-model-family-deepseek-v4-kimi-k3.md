# Model family: DeepSeek-V4 / Kimi-K3 (the 2026 trillion-scale MLA-MoE wave)

> **Why this one, now.** The frontier just shipped a wave of open-weight trillion-scale MoEs —
> **DeepSeek-V4** (Apr 2026: 1.6T total / 49B active, MIT, HF day-one; plus **V4-Flash** 284B/13B)
> and **Kimi-K3** (open weights Jul 27, 2026: 2.8T / 16-of-896 experts, native vision, modified
> MIT). Per the standing rule — *prioritize a family that (a) cashes in a primitive we just built
> for ~free, or (b) adds an axis we lack; popularity breaks ties, it doesn't lead* — these are the
> strongest **(a)** case in the backlog: MLA + group-limited DeepSeekMoE are already shipped
> (`deepseekArchitecture` serves V2/V3/Kimi-K2), so **each is alias-first, not a new adapter**.
>
> **The Gemma-4 tie-in is the timing.** Gemma-4's just-landed **streamed quantize-at-load**
> (`streamExperts`, per-expert int4 — no whole-tensor f32 transient) + **frequency-aware
> expert-pager eviction** is exactly the infrastructure that makes a 1.6T/2.8T checkpoint
> *loadable on a normal box*. Finishing Gemma-4 is what makes this family actionable rather than
> aspirational — the spillover is the loader, not an arch primitive.
>
> **Verify before you build (this is a hard gate, not a formality).** The alias assumption is a
> *hypothesis to test*, not a truth to trust (`docs/parity-hunt-playbook.md`). DeepSeek has been
> shipping attention changes (the V3.2 sparse-attention line); if **V4 or K3 replaced MLA with a
> sparse-attention variant, or changed the router shape, this stops being an alias and becomes a
> primitive add** — re-scope before writing a loader. Read the real `config.json` + modeling file
> first and correct every table below.

## Architecture deltas — expected (must be confirmed against real configs)

`deepseekArchitecture` (`decoder/registry.go:1112`) already provides: **MLA latent-KV**
(kv_lora_rank cache, decoupled qk_rope, asymmetric qk/v head dims), **DeepSeekMoE** (V3 sigmoid
routing + `e_score_correction_bias` + group-limit `n_group`/`topk_group`, ungated shared expert,
`first_k_dense_replace` prefix), and **YaRN** dual-mscale. If V4/K3 are V3-shaped, the delta is a
descriptor + loader only.

| trait | expected vs shipped V3 path | difficulty | action |
|---|---|---|---|
| **Attention = MLA** | assume yes | ✅ alias **iff true** | **confirm first** — if V4 ships sparse attention (DSA-line) it is a NEW primitive; K3 likewise |
| **Routing = group-limited DeepSeekMoE** | assume yes | ✅ have it | confirm `n_group`/`topk_group`/`e_score_correction_bias` present |
| **Expert count over the resident cap** | K3 = 896, V4 = large; cap = 256 (`features.go:204-207`) | bounded | **bump the router-capacity cap** (eligibility number, not correctness) — see below |
| **Trillion-scale weights** | new regime for the loader | bounded | **wire the MoE loader to the gemma4 streamed path** (`streamExperts`) |
| **Kimi-K3 native vision** | multimodal | bounded | **skip the vision tower**, text decoder only (the `qwen2_5_vl` precedent) |
| **`first_k_dense` prefix, tied/untied head, RoPE base** | config-driven | ✅ | read from config, don't assume |
| new `model_type` strings (`deepseek_v4` / `kimi_k3`?) | registration | trivial | alias into `deepseekArchitecture` after the config check |

No **intractable** blocker *if* the attention is still MLA. If it isn't, the whole estimate
changes — which is why Phase 0 is a gate.

## The two things that are genuinely new work (not "just an alias")

1. **Router-capacity cap.** `residentMoECapacityOK` (`features.go:211-223`) caps resident MoE at
   `{experts:256, groups:32}` on cuda/webgpu (Metal has no fixed cap). K3's 896 experts and V4's
   count **exceed 256**, so today they'd correctly **decline to CPU** on those backends. Raising
   the cap is a resident-*eligibility* change, not a correctness change — but do it deliberately,
   and re-run the capacity break-it-first (a model just over the raised cap must still decline
   cleanly, never crash). GPU residency for these is separate (see *Not in scope*).
2. **Streamed loading at trillion scale.** A 1.6T/2.8T checkpoint cannot land as whole-tensor f32.
   The gemma4 work generalized `streamExperts` / per-expert quantize-at-load to *any*
   fused-stacked-expert MoE (agent-confirmed) and the freq-aware pager eviction restores hit-rate
   on large routers — **wire the DeepSeekMoE loader onto that path**. This is the concrete cash-in
   of the Gemma-4 infra, and the difference between "runs on paper" and "runs on the dev box."

## Taxonomy (so nothing ships silently wrong)

If confirmed MLA, no new `ResidentFeature` is needed — `FeatMLA` + `FeatMoE` already gate it, and
MLA/DeepSeekMoE are webgpu-resident but **not** cuda/metal-resident, so on the native GPU backends
these decline to CPU by construction (correct, just unaccelerated). **If** V4/K3 introduce sparse
attention, add a `FeatSparseAttn` declared by no backend — CPU-correct day one, GPU-auto-decline,
matrix cells flip themselves later. Either way the taxonomy keeps it honest.

## Parity + gates (the checkpoints are too big to download whole — plan for it)

Real-checkpoint parity hits the same wall gemma4 did (a 50 GB+ download is infeasible on the dev
box; V4 full is ~TB-scale). Tiered, per the qualification-sweep proxy discipline:

- **T1 — tiny synthetic golden.** A random V4/K3-shaped tiny checkpoint: argmax-exact + logit
  cosine vs the CPU reference, byte-identical greedy continuation. Proves the forward.
- **T3-real — use the tractable member.** **DeepSeek-V4-Flash (284B/13B)** is the realistic
  real-checkpoint target — big but not TB-scale, and it exercises the exact V4 path. **Kimi-K3
  (2.8T) is proxy-validated, never real-e2e** (assembly-equivalence + per-kernel parity), and the
  cell is **labeled "proxy"** — a cell you could only prove indirectly is not claimed directly.
- **Break-it-first** on the two real changes: (i) perturb the MLA latent projection / router
  group-limit, confirm the T1 gate goes **RED**, revert; (ii) set expert count just over the
  raised cap, confirm it declines to CPU (doesn't crash). A gate that can't fail isn't a gate.
- **Known side-effect:** touching the shared MoE loader re-stales `uses:[core]` families'
  `deps_hash` — that's the scripted goldens-gated refresh (`scripts/refresh_parity_hashes.sh`),
  not a re-validation.

## Phasing

- **Phase 0 — CONFIRM THE SHAPE (gate).** Read V4 + K3 `config.json` + modeling files. Assert:
  attention == MLA, routing == group-limited DeepSeekMoE. **If either fails, STOP and re-scope as
  a primitive add** (sparse attention / new router) — do not write a loader against a wrong shape.
- **Phase 1 — DeepSeek-V4 alias.** Register `deepseek_v4` → `deepseekArchitecture`; wire the
  streamed expert loader; T1 tiny golden; **V4-Flash T3-real**. Ships V4 CPU-correct.
- **Phase 2 — Kimi-K3 alias.** Register `kimi_k3`; **skip the vision tower** (text decoder only);
  **bump the router cap** for 896 experts; T1 tiny golden; **proxy-validate** the 2.8T (labeled).
- **Phase 3 — GPU residency (separate, demand-gated).** Only if MLA lands on CUDA/Metal (its own
  task) — until then these are CPU on native backends by design. Not required to ship the family.

## Leverage (why the cost is lower than it looks)

- **The MLA + DeepSeekMoE shape is now the de-facto frontier standard.** V4, K3 (and likely the
  next MiniMax/others) ride it — each is a *near-free alias*, so the marginal cost per new frontier
  MoE trends toward "register + config-verify." Treat new MLA models as **alias-first**.
- **The streamed trillion-scale loader path** proven here serves *every* future large MoE, not just
  these two — it's the reusable half of the gemma4 investment.
- **The raised router cap** unblocks Kimi-K2 (384 experts, already shipped but currently
  cap-declined) at the same time.

## Not in scope (recorded, so it isn't re-litigated)

- **MLA-on-CUDA/Metal residency** — the real "make it fast" work, and arguably higher-return than
  more families (these run CPU-only on native GPU today). It's a **separate kernel task**, not part
  of this alias add. Flag it as the depth follow-on.
- **Kimi-K3 vision** — the tower is skipped; native multimodal is its own track.
- If Phase 0 finds sparse attention, the sparse-attention *primitive* is a different task; this doc
  covers only the alias case.
