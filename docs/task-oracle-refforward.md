# Task: a pure-Go reference-forward oracle (`oracle/`) — the torch-replacement research for E7 item 7

> **Status: PARKED, 2026-08-18 (Francis).** The v0.13.0 freeze this plan was gated behind has cleared,
> but Francis reviewed the plan and explicitly declined to fund it now — not rejected, just not a
> priority: may be picked up someday as a low-priority item, no active trigger. Do not restart Phase 0
> or any part of the build without a fresh go-ahead; this status line is the record of that decision,
> not a stale gate waiting to open. The design below (scoping, phases, open decisions) stays as
> reference for whoever picks it up later — nothing about the analysis changed, only the funding call.
>
> Also folded into this decision: aikit carries its own, separate ~23-file torch-oracle cluster
> (`aikit/scripts/oracle/*.py`, the BERT/SPLADE/cross-encoder/SigLIP/Qwen2.5-VL-vision parity
> generators) — conceptually the same problem, one repo over. Any future revisit should treat the two
> as one joint Phase-0 clustering exercise (goinfer's 57 decoder scripts + aikit's 23 encoder/vision
> ones) rather than two separate plans, since the independence argument and the phase structure above
> apply to both without change. See `aikit/docs/internal/roadmap.md` §2 for aikit's side of this note.
>
> Original scoping below, unchanged from when it was drafted 2026-08-12 from the E7 inventory (67
> tracked `.py`; the 57 torch/HF scripts are the surface this replaces):

## 1. What this replaces, and what it does not

E7's inventory split the repo's Python on one axis: 10 stdlib scripts (migratable now, E7's own work)
vs **57 that import torch / transformers / safetensors / numpy** — the *reference-tensor surface*,
~5000 lines, the whole `scripts/pin_*` family plus the torch oracles and golden generators. Those 57
are what an `oracle/` module addresses.

But the 57 are **not monolithic**, and Phase 0's first job is to separate three kinds inside them:

- **Forward-oracle golden generators** (the bulk — `pin_qwen3_real.py`, `pin_gemma4_forward.py`,
  `pin_deepseek_moonlight.py`, …): run the model's forward in HF and emit reference logits/tensors.
  **These are what `refforward` replaces.**
- **Weight-space checks** (e.g. the `glm4_moe` weightDiff path): compare weights directly, no forward
  oracle needed. Already language-agnostic in shape — may need no `refforward` at all.
- **One-off analysis probes** (`diff_gemma4_12b`, `gemma4_scale_probe`, `gemma4_quant_recon`,
  `dump_gemma4_12b_trace`, `mxfp4 extract`, …): debugging artifacts, not standing gates. These are
  **retired or ported individually on need**, not part of the oracle surface. Do not build oracle
  machinery to preserve a probe that has already served its purpose.

## 2. The one principle: independence is about shared *code*, not shared *language*

The whole value of a parity oracle is that an **independent** implementation agrees with goinfer. A Go
reference that reuses goinfer's own forward (`decoder/forwardn.go`), its kernels, or its loaders is a
**tautological gate** — it proves goinfer equals goinfer, not that goinfer is correct. This is the
self-consistent-gate blind spot the repo has already been bitten by and built guards against (the
`paged ≡ non-paged` trap; the reduction-width change that moved both arms of the Metal golden until an
*absolute* stored reference was added). The relay's recurring rule applies verbatim: **a gate that
compares the code under test against itself cannot fail in the one way that matters.**

Python's independence today is partly an accident of language — nobody worries `pin_qwen3_real.py` is
coupled to `forwardn.go`, because it *can't* be. Moving to Go forfeits that free boundary. So the move
is only sound if the accidental boundary is replaced by a **deliberate, structural** one. The rest of
this doc is how.

The pattern already exists in-tree at sub-component scale: the sampler's `refTopFilter` / `refLazyDraw`
and the `w4a8`/`w8a8` bit-identity references are pure-Go, independent, test-only references that share
nothing with the fast path. `refforward` is that pattern extended from a kernel to a whole forward.

## 3. Placement — a dedicated `oracle/` module inside goinfer

**Decision: `goinfer/oracle/` with its own `go.mod`.** Not aikit, not a separate repo.

- **Not aikit.** aikit is code-search/RAG primitives (embed, ann, bm25, linalg, mmap), heading to its
  own v1.0. Its extraction bar — generic mechanism *with a second consumer* — is met by neither: a
  reference LLM forward is architecture-specific, must track goinfer's 23 families as they change, and
  has exactly one consumer (goinfer's parity gates). Putting it in aikit couples aikit's clean scope to
  per-architecture LLM churn — the "carry the duplication rather than force the abstraction" caution
  from the mmap lift, inverted. `oracle/` may *consume* an aikit primitive; it is not part of aikit.
- **Not a separate repo.** Buys no independence a submodule doesn't, and adds version friction plus a
  circular-dependency risk if the oracle ever wants goinfer's arch configs.
- **A submodule fits.** goinfer is already multi-module (root + gpu/cuda/metal + demo). An `oracle/`
  module gives a **structural import boundary enforced by the module graph** — the same mechanism M-19
  uses to keep the pure-Go root from resolving the GPU dependency set. The oracle module simply cannot
  import `decoder/forwardn`; the boundary is a compile error, not a lint someone can wave through. It
  also satisfies E7's "no tooling dependency in the main `go.mod`" constraint by construction.
- **Neither side imports the other.** `refforward` emits committed golden JSON in the **existing
  schema**; the root's `*_parity_test.go` read it exactly as they read the Python-generated goldens
  today. The forward never imports the oracle; the oracle never imports the forward. This mirrors how
  the Python scripts already work (standalone, emit JSON) — which is why the cutover is a swap, not a
  rewire.

## 4. The four independence rules (load-bearing regardless of placement)

1. **Own weight reader.** Read HF safetensors (f32/bf16) with the oracle's *own* reader — **not**
   goinfer's GGUF loader. If both arms dequantize through the same loader, a loader bug moves them
   together and parity passes while both are wrong. The HF oracle avoids this today precisely because
   it loads independently; `refforward` must too.
2. **Own trivial math.** f64 textbook attention / MLP / norm in plain loops. **No aikit linalg
   kernels, no goinfer kernels.** Naïveté is the point — it doubles as executable spec documentation
   (a `docs/spec` cousin), and it guarantees no shared arithmetic with production.
3. **Anchor against HF once per architecture.** `refforward` is trusted only after it reproduces the
   HF oracle for a family within tolerance. This is what demotes Python — see §5.
4. **Guard the boundary.** A test in the module graph (M-19 shape) that fails if `oracle/` imports the
   forward path — so the independence can't erode in a later "helpful" refactor.

An f64 reference is legitimately a **better** oracle than bf16 HF: parity is cosine/argmax-tolerant, not
bit-exact against HF, so the higher precision sits inside the tolerance and the bf16→f64 gap is what the
tolerance already absorbs.

## 5. What stays Python — the honest limit

This shrinks Python; it does not delete it. The only fully-independent ground truth for "does this
reproduce Qwen/Gemma/DeepSeek as the authors intended" is the reference implementation the model authors
shipped — PyTorch. So:

- Python drops from **~50 per-model golden generators** to a **handful of per-architecture anchor
  runs** that validate `refforward` itself (rule 3). Cadence: once per architecture family, re-run only
  when that family's forward math changes.
- Everything else — routine golden generation, pinning a new checkpoint within a validated family,
  layer-by-layer bisection — becomes pure Go, no torch in the loop.
- "Zero Python ever" would mean trusting a hand-written Go transcription of each architecture's math
  with **no external anchor**, reintroducing exactly the risk the parity system exists to catch. The
  research conclusion is therefore explicit: **keep the root of trust, stop invoking it per-golden.**

This is the same logic as the manifest's existing `shared-path` provenance tier (a family inherits a
validated parent's oracle) — generalized so a Go reference, once anchored, becomes the oracle for its
cluster.

## 6. Phases

**Phase 0 — cluster the surface (the real first step; the E7 inventory counted but did not cluster).**
Read the forward-oracle subset of the 57 and group the 23 families by *shared forward-math*: attention
variant (MHA / GQA / MLA / sliding-window / KDA-class) × norm placement (pre / sandwich / qk-norm) ×
RoPE flavor (standard / NoPE / YaRN) × MoE routing (dense / sigmoid-group-limited / hash). **The count
of distinct math kernels is the whole estimate** — five kernels reused across families is a small build;
twenty near-unique ones is a campaign. Deliver a cluster table + a per-cluster effort class. Also tag
each of the 57 as forward-oracle / weight-space / retire-able probe (§1).

**Phase 1 — the f64 core + one anchor family.** Build the `oracle/` module (own go.mod, boundary
guard, own safetensors reader, f64 primitives) and one family end-to-end (suggest a dense GQA family —
`qwen2`/`llama` shaped — as the simplest cluster). Anchor it against the existing HF golden for that
family: the `refforward` output must match the current committed golden within the same tolerance the
Python oracle established. That equality **is** Phase 1's proof.

**Phase 2 — expand by cluster.** One cluster at a time, cheapest first, each anchored against its HF
golden before its Python generators are retired. MLA and KDA-class (DeepSeek / Kimi) and MoE routing
are the hard clusters; sandwich-norm (gemma) and windowed attention are mid.

**Phase 3 — cut over and delete.** Per validated family, apply E7's acceptance criteria verbatim
(they were written for exactly this): (a) Go and Python **agree** on the current tree before any swap;
(b) mutation-check the Go both ways (introduce the defect it catches → RED, remove → GREEN); (c)
**delete the Python generator in the same commit** that lands its Go replacement — never two sources of
truth; (d) the **scope line survives** — whatever the Python printed about what it did and did *not*
validate, the Go prints too. The per-arch HF anchor scripts (§5) are kept, clearly relabeled as
anchors, not per-model generators.

## 7. Sequencing, and what is not in scope

- **After v0.13.0**, behind §C1 and the CUDA gate — same freeze discipline as E7. Docs (this file) are
  exempt and land now; code does not.
- **Not in scope:** the 10 stdlib-only scripts (that is E7 proper); the `nvrtc_compile.py` build helper
  (a separate FFI question E7 flagged for Francis); the retire-able analysis probes (§1) beyond tagging
  them; any change to the parity tolerances or the golden schema (the oracle emits the schema that
  exists).

## 8. Open decisions for Francis

1. **Reference precision source:** anchor `refforward` against HF **bf16** (matches what the model
   authors run) or HF **f32** (cleaner, further from goinfer's quantized path)? Recommend f32 where the
   checkpoint offers it — maximal independence, and the tolerance absorbs it.
2. **Anchor cadence:** "once per architecture" — pinned to a family's forward-math `deps_hash`, so a
   core change to that family's math re-triggers exactly one anchor run? (Recommended — it makes the
   Python dependency self-documenting and minimal.)
3. **The `nvrtc` build helper — RESOLVED 2026-08-12 (Francis): "no Python" covers it; do it LAST,
   low priority.** Replace `cuda/nvrtc_compile.py` with a purego NVRTC binding (~6 C-ABI functions;
   purego is already in the cuda/metal dep set, so no new ecosystem) in `cuda/cmd/ptxgen`, dlopening
   the **pinned** `libnvrtc.so` with the byte-identical-rebuild control as acceptance. Removes the
   Python, not the (irreducible) NVRTC dependency. Tracked in QUEUE.md E7 as the build-tooling
   capstone, sequenced after `queue_citation_lint`. **Independent of this plan** — recorded here only
   so the two decisions stay unconflated, as §8 promised.
