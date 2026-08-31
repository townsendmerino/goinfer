# Plan: benchmark refresh — what's worth (re-)measuring, and what isn't

> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


> **Audience:** internal planning. Catalogs every benchmark worth running given
> where goinfer is now (v0.7.0; 15+ families; three attention axes covered). The
> standing 60–70%-of-CUDA headline (`benchmarks.md §B`) has a **narrow scope**, and
> most of the coding since it was measured (new families + CPU/footprint work)
> does *not* touch the path that produced it. This doc says which numbers to
> refresh, which are pinned and pointless to re-run, and — crucially — which
> *new* benchmarks fit the new families' actual lane.
>
> **Provenance rule (inherited from `benchmarks.md` / `gpu-assessment.md`):** no
> number ships without a traceable run — the exact commit, model+quant, card/box,
> peer version + build flags, and config (warm/greedy/single-stream). A stale ratio
> is as misleading as a guess.

## The scope trap (read first)

The 60–70% figure is **GPU-residency decode, dense Qwen2/Llama, equal quant**:
Qwen2.5-1.5B int8 = 89.7 tok/s = **61%** of Ollama-CUDA 147; Qwen2.5-7B int4 =
51.7 = **71%** of llama.cpp-CUDA 72.8 (`benchmarks.md §B`, peer figure from
CHANGELOG v0.5.0). **Both rows were RETIRED 2026-08-27** — withdrawn rather than
corrected, and archived in `benchmarks-archive.md`; this doc's premise is the refresh
that retirement now requires. Two facts constrain any "re-benchmark with new families":

1. **GPU residency is dense-Qwen2/Llama-only.** `decodeRunnerEligible` =
   `MoE==nil && gemma4==nil && qwen35==nil && SlidingWindow==0 && …`. So **every
   new family (Granite/Mamba-2, Nemotron, DeepSeek/MLA, GLM, Kimi, Gemma 4) is
   residency-INELIGIBLE** and never runs on the path the headline measures.
2. **Dense decode is at the WGSL wall** (~90 tok/s int8; megakernel inexpressible
   in WGSL — `gpu-next-levers-assessment.md`). Re-running the *same* dense decode
   yields ~the same number; nothing coded since moves it.

⇒ This does **not** mean new families can't run on the GPU — they can, via the
**staged path** (per-matmul dispatch; see the GPU-coverage levers below). It means
a staged-path number is **not comparable to the residency 60–70% headline** and
must never be presented as if it were. So: *measure it if useful, but label it
"staged path, residency-ineligible" and keep it out of the headline row.* The
mistake to avoid is the **framing** (equating staged with the residency claim, or
calling it goinfer's competitive GPU lane), not the measurement itself. A blind
"Granite vs Ollama-CUDA" with no path label is the category-confused number — don't
produce *that*.

---

## Tier 1 — GPU-vs-CUDA, dense path: refresh (justified by new work)

Same scope as the headline (dense Qwen2/Llama, equal quant, 2070 SUPER, warm,
greedy, single-stream), re-run because two inputs changed:

- **B1 — dense decode *with GPU speculative decode ON*.** The one thing that
  postdates the 60–70% number (the resident `ForwardN` / `GenerateSpeculative` /
  Lever-2 commits). Spec changes wall-clock-per-output-token even though raw
  per-token decode is walled, so this is the only lever that can move the ratio.
  Qwen2.5-1.5B (int8) + 7B (int4). **Fairness:** enable the peer's draft/spec too,
  or report both spec-on and spec-off configs for both sides. Flagged in
  `gpu-next-levers-assessment.md` as "biggest decode upside, real uncertainty
  against the glue wall" — i.e. measure it, don't assume.
- **B2 — refresh the peer baseline.** The 72.8 / 147 CUDA figures are v0.5.0-era;
  Ollama + llama.cpp have moved, so the *ratio* is stale on both sides even if
  goinfer is unchanged. Re-run both fresh on the same card; pin both versions +
  CUDA version + build flags.
- **B3 — Vulkan same-API number (honesty/context).** From-source
  `llama.cpp -DGGML_VULKAN=ON`: WebGPU-vs-Vulkan, both non-CUDA, separates "WGSL
  dispatch overhead" from "CUDA megakernel advantage." The roadmap's open
  measurement gap; contextualizes the 60–70% honestly.

**Do NOT re-run:** dense decode *without* spec (pinned at the WGSL wall — the
next-levers doc says stop optimizing single-stream decode; you'd relearn 61/71%).

---

## Tier 2 — the new families, in *their* lane (CPU + footprint, not CUDA)

MoE/hybrid/MLA compete on pure-Go CPU single-binary + memory, so the fair peer is
**llama.cpp-CPU**, and the headline metrics are footprint/capability, not tok/s-vs-
CUDA. These are *new* numbers (most don't exist yet), genuinely measurable now:

- **B4 — CPU decode + prefill vs llama.cpp-CPU, per family, equal quant.** Run on
  the same box (the RTX-box CPU / macbook-arm64). Covers the families that have no
  GPU story: Granite, Nemotron, DeepSeek-V2/V3, GLM, Kimi, Mixtral, Qwen-MoE,
  Mellum. Single-stream, stated context length.
- **B5 — MoE expert-paging footprint (the "35B on 16 GB" capability claim).** Peak
  RSS running a 35B-A3B-class model under a `--weight-cache` budget, + the cold-miss
  tok/s cost. You already spiked the LRU hit-rate (`moepaging_spike_test.go`) — turn
  it into a reported peak-RSS-vs-budget curve. This is a *capability* number (runs
  at all on a small box), not a speed race.
- **B6 — MLA KV-cache footprint (context-per-byte).** DeepSeek/Kimi MLA caches a
  shared latent vs full per-head KV — measure KV bytes/token (and max context on a
  fixed RAM budget) vs a standard-GQA family of similar size. The latent-KV axis's
  selling point, quantified.
- **B7 — Mamba-2 long-context memory flatness.** Granite/Nemotron recurrent state is
  O(1) in context vs attention's O(context) KV — plot resident KV/state bytes vs
  context length, Mamba-hybrid vs a pure-attention family. Pair with CPU decode rate
  at long context.

---

## Tier 3 — already-shipped levers worth a headline number (in-lane, no peer needed)

These quantify goinfer's own features against *itself* (the honest, peer-free
claims), several already partly measured:

- **B8 — weight-memory program, end-to-end.** int8/int4 + the KV program (rings +
  int8 KV) + `.giw` zero-copy: peak RSS and tok/s for a fixed model across the knobs,
  showing the footprint reduction the program bought (the docs cite ~20× KV on
  Gemma-class; surface it as a reproducible table).
- **B9 — cold-start / single-binary story.** Boot-to-first-token wall, binary size,
  zero-install — the positioning claims in `benchmarks.md` (~0.5 s boot, <100 MB
  heap). Re-confirm at v0.7.0; these are the "LLM in one file" headline.
- **B10 — speculative decode on CPU** (if/where it applies) + the GPU-vs-CPU spec
  gate already in the tree — report the measured speedup, not just that it exists.

---

## Tier 4 — measure only after specific work lands (gated, not now)

- **B11 — batched GPU prefill**, *after* `dot4I8Packed` (dp4a) is wired (the
  next-levers doc notes it's now merged upstream). Prefill TTFT is wash until then;
  measuring before is pointless.
- **B12 — wasm/browser inference** throughput, *after* the client-side `demo/gemma-web`
  scaffold is real — marketing number, bounded effort, low perf risk.
- **B13 — expanded GPU residency** numbers for more families — only if residency
  eligibility is ever widened (open-ended per-arch work, deferred; decode is walled
  regardless, so this is low value as a benchmark driver). See the coverage levers
  below for *how* eligibility would widen.

---

## GPU coverage levers — running non-dense families on GPU at all

This is the engineering context behind the benchmarks above: there are **two** GPU
paths, and "residency-ineligible" ≠ "no GPU."

- **Residency** — full-token on-GPU forward (`DecodeRunner`); dense-Qwen2/Llama only
  (`decodeRunnerEligible`). The path the 60–70% headline measures.
- **Staged** — per-matmul dispatch to the GPU backend, CPU glue between. From
  `weightmat.go`'s `matmul`: **f32 and int8/W8A8 weights route to the GPU backend**
  (`be.MatmulW8A8` / `be.MatmulBT`) for *any* arch. So every family already runs its
  int8 matmuls on the GPU when built `-tags gpu` — just glue/dispatch-bound, not a
  megakernel.

The ladder, cheap → hard:

- **Lever A — staged-path benchmark, today, no code (run int8).** Every ineligible
  family (Granite, DeepSeek, GLM, Kimi, …) already does its int8 matmuls on the GPU.
  So a real staged-GPU number exists now — see **B14** below. **Caveat from the
  dispatch code:** **int4 (W4A8) stays on CPU** — the int4 branch calls
  `linalg.MatmulBTW4A8` directly with no backend routing, so `-tags gpu` does nothing
  for an int4 model. For a staged-GPU number, run **int8**.
- **Lever B — route W4A8 to the GPU backend (small, real gap).** The int8 branch
  checks `be.(QuantBackend)`; the int4 branch doesn't. Adding a GPU W4A8 path would
  let int4 families use the staged GPU path too. Modest; bounded gain (staged is
  still glue-bound).
- **Lever C — extend residency eligibility, per-arch** (real engineering), ranked by
  unlock: sliding-window residency → Mistral/Gemma3 (moderate); QK-norm / partial-RoPE
  in the DecodeRunner → Qwen3/GLM-attn (small, per-feature); **MoE residency** →
  DeepSeek/GLM/Kimi/Mixtral/Qwen-MoE (biggest unlock, hardest — GPU expert
  routing/indirect dispatch + expert-VRAM); hybrid (Mamba-2 scan / MLA latent-KV /
  Gemma4 PLE) ported to the resident runner → Granite/Nemotron/DeepSeek/Gemma4
  (substantial, family-specific).

**The wall caveat — decide capability vs ratio before investing in C.** Decode is at
the WGSL wall (~90 tok/s int8; megakernel inexpressible in WGSL). So extending
residency to a family makes it **run at the same ~60–70%-of-CUDA ceiling** — it buys
*GPU coverage for that family*, **not a better ratio**. A higher ratio needs native
CUDA/Metal, which is off-strategy. Pursue C for *capability* (e.g. sliding-window and
the cheap QK-norm/partial-RoPE wins), never to beat the headline number.

### B14 — staged-path GPU numbers for non-dense families (Lever A)

Run the new families (int8) with `-tags gpu` and report decode tok/s **labeled
"staged path, residency-ineligible"** — separate table, never the headline row, with
the path stated. Useful as an honest "here's where goinfer's GPU staged path stands
for MoE/hybrid/MLA," and as the before/after baseline if Lever B or C lands. Pair
with the Tier-2 CPU numbers so the staged-vs-CPU delta per family is visible (on
some families/quant the glue-bound staged path may not beat tuned CPU SIMD — that's
a finding worth having).

**MEASURED (2026-06-17, RTX 2070 SUPER, `gpu.TestStagedGPU_nonDense`):** the
"finding worth having" landed — **the staged GPU path LOSES to CPU at decode.**

| family | axis | quant | cpu tok/s | staged-gpu tok/s | gpu/cpu |
|---|---|---|---|---|---|
| deepseek-v2-lite | MoE + MLA | W8A8 (int8) | 2.87 | 1.63 | **0.57×** |

- **Why it loses:** at M=1 decode the staged path is glue/dispatch-bound (each of the
  hundreds of tiny per-token matmuls is a dispatch + the result round-trips host↔device),
  and aikit's tuned CPU int8 SIMD beats it. Same wall as the resident path, minus the
  fusion — so the staged path is *strictly worse* than CPU here.
- **Coverage caveat — most families can't even use staged GPU:** only **W8A8** (full
  int8×int8) matmuls route to the backend. **Q8_0 GGUF** (weight-only int8, e.g. the
  granite/nemotron checkpoints on hand) hits `linalg.MatmulBTQ8` with no backend routing →
  CPU regardless; **int4/W4A8** likewise (Lever B). So a staged-GPU number only exists for
  a family loaded from safetensors at `Quant:"int8int8"`.
- **Consequence for the ladder:** Lever B (route W4A8 to the same staged path) is
  **contradicted** — if int8 staged already loses 0.57×, int4 staged (more unpack work)
  cannot win at decode, and its prefill case is the DP4A-blocked wash. **The only path that
  beats CPU for these families is the *resident* fused path (Lever C)** — staged GPU is not
  a decode win on this hardware, at any quant. Recorded so it isn't re-litigated.

---

## Methodology (applies to every tier)

- **Equal quant, same box, warm, greedy, single-stream** — single-stream is
  goinfer's lane; never benchmark against a peer's *batched* serving (unfair the
  other way, and goinfer has no continuous batching by design).
- **Pin everything:** goinfer commit; peer name + version + CUDA/Vulkan version +
  build flags; model + quant; card/CPU; prompt + context length; warm vs cold.
- **Spec-decode fairness:** if goinfer spec is on, let the peer use draft/spec too,
  or report both configs side by side.
- **Same checkpoint both sides** for parity-of-quality; note where quant formats
  differ (int8 vs q8_0, int4 vs q4_K) so "equal quant" is honest.
- **Record to a cited home** (`benchmarks.md` for the public table, `gpu-assessment.md`
  / a new `bench/` run-log for the raw runs) — no floating numbers.

## Priorities

1. **B1 + B2** (dense GPU spec-on + fresh peer) — the only thing that can move the
   famous 60–70% headline; do together so the ratio is current on both sides.
2. **B4 + B5 + B6** (new families' CPU + footprint) — the new families have *no*
   honest number today, and these are their actual selling points; highest
   information-per-effort.
3. **B8 + B9** (weight-memory + cold-start) — cheap, reinforces the v1.0 positioning.
4. **B3, B7, B10, B14** — context/honesty additions (B14 = staged-path GPU, labeled,
   no code — cheap to fold in alongside the Tier-2 CPU runs).
5. **B11–B13** — gated on their prerequisite work; don't pull forward.

## What this is not

- Not a re-litigation of single-stream dense GPU decode — it's walled (`gpu-next-
  levers-assessment.md`); B1's spec-on is the only dense-GPU re-run worth doing.
- Not a *headline* GPU-vs-CUDA number for MoE/hybrid/MLA families — they run on the
  **staged** path, not residency, so any GPU number for them is reported separately
  and **labeled "staged"** (B14); Tier 2 (CPU + footprint vs llama.cpp-CPU) remains
  their primary, in-lane comparison. ("Impossible" was wrong — staged works today;
  what's off-limits is presenting a staged number as the residency claim.)
- Not new perf engineering — every Tier 1–3 + B14 item measures code that already
  exists; Levers B/C are engineering, scoped separately and gated on the wall caveat.
