# Plan: model families after GLM + Granite — go for coverage axes, not model count

> **Audience:** internal planning, the sequel to
> `task-model-families-glm-granite.md` (GLM-4.5/4.6 **shipped** — safetensors +
> GGUF + real GLM-4.5-Air; Granite-4.0-H **shipped** — Phases 0–3, Mamba-2 scan +
> hybrid loader/forward + GGUF, real granite-4.0-h-tiny generating coherently).
> This doc covers what comes *after* those two, and reframes the
> selection rule: **stop counting models, start covering attention/​mixing
> axes.** Same recurring muscle — arch adapter + tensor schema + loader + parity
> golden, with the v0.2–v0.5 surface inheriting — and the per-family "definition
> of done" from `parity-coverage-policy.md` applies to every item here.

## The thesis: coverage axes, not model count

After GLM + Granite, goinfer will run two *kinds* of efficient sequence mixing —
**gated-linear** (Gated DeltaNet, qwen3_5_moe) and **state-space** (Mamba-2,
Granite). The strategic prize is the **third axis, latent-KV attention (MLA)** —
at which point goinfer covers the three major efficient-attention families the
field has converged on. "Runs every modern attention variant, in one pure-Go
static binary" is a far stronger claim than "supports N models," and it's a claim
no other pure-Go runtime can make. So the selection rule for this phase is:
**prioritize a family when it (a) cashes in a primitive we just built for ~free,
or (b) adds a coverage axis we lack — popularity breaks ties, it doesn't lead.**

This reframing is itself a deliverable (Step 0 below), not just framing for this
doc — it should live where the roadmap and positioning are read.

---

## Step 0 — document the coverage-axis positioning ✅ DONE

Recorded "coverage axes, not model count" where it steers decisions:
`docs/roadmap.md` now has a "Positioning: attention-coverage axes" section + the
Model-freshness lines reframed axis-first (pointing here); `README.md` states the
axis claim (latent-KV gated on MLA landing); `docs/ARCHITECTURE.md` documents the
two recurrent mixers. Original step text below.

**Explicit, required step:** record "coverage axes, not model count" where it
will actually steer decisions, not only here:

1. **`docs/roadmap.md`** — add a short "Positioning: attention-coverage axes"
   note: the three axes (gated-linear / state-space / latent-KV), which goinfer
   has, and that family selection is axis-driven with popularity as tiebreak.
   Replace the flat "New model families (DeepSeek V4, GLM-4 MoE, …) · on demand"
   line with the axis framing + a pointer to this doc.
2. **`README.md`** — one line in the positioning/capability section: goinfer runs
   the major modern attention families (softmax/GQA, gated-linear, state-space,
   and — once MLA lands — latent-KV) in one cgo-free static binary. Gate the
   "latent-KV" claim on MLA actually landing (per the claim discipline in
   `parity-coverage-policy.md` — no claim without a passing gate).
3. Cross-link both to this doc and to `parity-coverage-policy.md`.

Rationale: the value of the GLM/Granite/MLA sequencing is only legible if the
axis framing is written down; otherwise the next person reads it as "more models"
and re-litigates priority by popularity alone.

---

## Part 1 — Nemotron-H / Nemotron Nano 2 (the free win that compounds Granite)

### Why first

Nemotron-H is a hybrid **Mamba-2 + Transformer** (≈8% attention layers, the rest
Mamba-2) — the *exact* primitive Granite forces us to build. So once Granite
lands, Nemotron is **descriptor + tensor schema + parity golden, zero new
forward-pass work.** It's the "moat compounds" payoff: the second Mamba-2 family
costs almost nothing.

### Two bonuses over a plain filler

- **It's small and fully parity-testable.** Nemotron-Nano-2 is ~9–12B and is
  designed to run 128k context on a single 22 GiB A10G (bf16). Unlike GLM-4.5-Air
  (whose 212 GB bf16 oracle won't fit the 65 GB box → weightDiff-only), a **full
  bf16 forward oracle fits** — so Nemotron is the *cleanly validated* proof the
  Mamba-2 work generalizes (third hybrid family, second Mamba-2).
- **Popular on its own** — NVIDIA-backed, reasoning-focused, long-context.

### Work + gates

Granite's `mamba2.go` + the Mamba-2 cache-state kind + per-layer-kind dispatch are
reused as-is; the new work is `nemotronArchitecture` + `nemotronTensorSchema` +
the layer-kind pattern (its attention layers are evenly dispersed, ~1-in-12) +
loaders. **Gate (per policy):** model-free scan-equivalence already exists (from
Granite); full argmax + cosine vs HF bf16 (it fits!); GGUF `weightDiff`; T1 tiny
golden; manifest row. **Effort: low** — no new primitive.

---

## Part 2 — MLA / DeepSeek (the strategic primitive — the third coverage axis)

> **Supersedes** the "DeepSeek V4 — deferred" line in
> `task-model-families-glm-granite.md`. Two corrections: (a) the deferral's
> scale objection is now solved by the `parity-coverage-policy.md` weightDiff +
> layer-slice approach (the GLM-4.6 precedent), and (b) **"DeepSeek V4" is
> unconfirmed as of 2026-06** — the real target is **V3 / V3.2 (MLA)**.

### Why it's the prize, not just another model

**Multi-head Latent Attention** projects K/V into a shared low-rank latent
(compression dim ~512) and caches only that latent, reconstructing per-head K/V on
the fly ("absorb" mode). Two reasons it's the right next *primitive*:

- **It's the missing coverage axis** — latent-KV attention, the third major
  efficient-attention family, now spreading beyond DeepSeek.
- **It's synergistic with our distinctive strength, the KV-memory program.** MLA
  is *compressed KV by construction* — the natural next chapter after the int8 /
  f16 / paging KV work. This isn't "support a model," it's "extend the KV-memory
  story with a new mechanism," which is exactly goinfer's lane.

### The work (this one is a real primitive)

- **MLA attention** as a new attention kind: the down-projection to the latent,
  the per-head up-projection, decoupled RoPE on the separate rope-carrying dims,
  and the **"absorb" caching mode** (cache the latent, not full K/V). New
  forward-pass + a new cache representation (a latent KV cache alongside the
  existing full-KV and recurrent-state kinds — the per-layer-kind machinery from
  the hybrid work generalizes to "attention-kind" here).
- **MoE FFN** reuses the shipped path (DeepSeekMoE = fine-grained + shared
  experts, which the GLM shared-expert work already exercises).

### Target + gates

- **Scale handled, not deferred:** the 671B V3 is `weightDiff` + layer-slice
  oracle (the GLM-4.6 call). For a *full* bf16 oracle, target the **smallest
  MLA checkpoint available** (a DeepSeek distill / small MLA model) so the new
  attention math gets one fully-validated end-to-end gate.
- **Gate:** layer-slice argmax/cosine vs HF bf16 (catches the MLA projections +
  decoupled-RoPE wiring), `weightDiff`, T1 tiny golden, manifest row, coherent
  generation. **Effort: medium–high** — the only real new primitive in this doc;
  sequence it after Nemotron (which is free) so the cheap axis-win lands first.

---

## Part 3 — popularity momentum on rails we already have (tiebreak picks)

Cheap descriptor adds, no new axis — the freshness race, interleaved around
Parts 1–2 as demand dictates:

- **Llama 4 (Scout/Maverick), text decoder.** MoE (have it) + **iRoPE**
  (interleaved attention with **NoPE** layers — a moderate descriptor wrinkle: a
  "no-positional" layer kind + the interleave pattern, not a new primitive). Take
  the text decoder, skip the early-fusion multimodal (the qwen3.6-VL call). Huge
  install base. **Caveat:** Meta shipped closed-source *Muse Spark* (2026-04) and
  ended the open Llama line — Llama 4 is last-of-its-line, valuable for the
  existing user base but not a growing family.
- **Phi (Phi-4 / Phi-3.5-MoE).** Dense / MoE on the shipped path, very popular in
  the small-model / on-device crowd. Near-free filler.

Both are pure descriptor+loader+golden, gated identically, and add **no coverage
axis** — so they're explicitly tiebreak/momentum, not milestones.

---

## Sequencing, deferrals, and the per-family contract

**Order:** Step 0 (document the axis framing — cheap, do immediately) → **Nemotron-H**
(free, compounds Granite, fully parity-testable) → **MLA/DeepSeek** (the third
axis; the one real primitive) → **Llama 4 / Phi** as momentum interleaved by
demand. The logic: harvest the just-built Mamba-2 adjacency before building a new
primitive; then build the primitive that *also* extends the KV-memory moat.

**Still deferred (unchanged):** **Gemma 3n** (MatFormer + AltUp + LAuReL — a stack
of new mechanisms for no new axis; revisit only on on-device demand) and **full
multimodal early-fusion** (Llama 4 vision) unless image demand pulls.

**Per-family contract:** every item here ships under the
`parity-coverage-policy.md` definition of done — arch adapter + loader, a **T1
committed tiny golden**, a **T3 manifest row** (full oracle where it fits — Nemotron
and the small MLA model — else `weightDiff`), a **T2 small-real-model sweep entry**,
and a README support row **only after** those exist. With GLM/Granite already
landed (their `pin_*`/`weightDiff` tests committed), the parity-coverage harness
(`task-parity-coverage.md`) is being **retrofitted now** — stand it up first and
backfill GLM/Granite as its first rows, so Nemotron and MLA land *onto* a working
checklist rather than extending the gap.
