# Changelog

All notable changes to goinfer are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The forward-pass and quantization numerics are parity-gated against HuggingFace
and are the stable contract. **Which API surfaces v1.0 will semver-bind is now
declared in [`docs/api-tiers.md`](docs/api-tiers.md)** (signed off 2026-08-18):
the load/generate, tokenizer, chat, constrain and `serve` surfaces are the Hard
tier; the backend/residency seam, family descriptors, drafters, multimodal and
serialization plumbing are named Experimental — explicitly, not by omission.
**That split takes effect at the v1.0 tag.** Until then goinfer is pre-1.0 and
any surface may still change.

## [Unreleased]

### Added

- **Olmo 3 as a new family** (`olmo3`; Ai2, 7B/32B): two real departures from every other family
  here, both verified against the real `modeling_olmo3.py`/`configuration_olmo3.py` rather than
  assumed. `NormPlacement` gains a fourth value, `NormPostOnly` — no pre-norm at all; only the
  attention/MLP sublayer OUTPUT is normalized before the residual add. QK-norm applies to the WHOLE
  projected q/k vector, not per head — a new `QKNormWhole` flag reuses the existing `rmsNorm` kernel
  with `rows=1, dim=numHeads*headDim` instead of `rows=numHeads, dim=headDim`, so no new math (and
  fixed a latent bug the addition surfaced: the existing `FeatQKNorm` derivation would have also
  incorrectly required the per-head kernel for a whole-vector family). YaRN scaling applies only to
  full-attention layers, sliding layers stay at plain RoPE — the same local/global RoPE split
  Mellum's own gate already implements, reused with the sliding-layer scaling left unset. Parity-
  gated against a tiny oracle whose sliding/full split and RoPE tables are actually exercised
  (prompt length exceeds the fixture's sliding window): 100.0% / 0.9999999999997883.

- **Olmo Hybrid as a new family** (`olmo_hybrid`; Ai2, 7B): the sibling MoE-free DeltaNet+softmax
  hybrid — qwen3_5's Gated DeltaNet on 3-of-4 layers, olmo3's own full-attention shape on the rest.
  Its norm placement differs BY LAYER KIND within one model (full-attention layers: `NormPostOnly`;
  DeltaNet layers: plain `NormPre2`) — the first family here where that varies, so `Architecture`
  gains `NormPlacementLinear`, a per-layer override keyed on the same `layerIsLinear` hook that
  already selects the mixer (nil for every other family, unaffected). Every other departure is a
  parameterization of the shared DeltaNet code, not new math: `linear_allow_neg_eigval` doubles the
  write-gate beta after the sigmoid; q/k/v projections AND the depthwise conv are separate tensors
  rather than qwen3_5's pre-concatenated ones (the conv split's true q/k/v-boundary layout was only
  confirmed by fetching the real checkpoint's safetensors header directly — both the source and a
  local re-save produced different, wrong splits); the output gated-RMSNorm is named `o_norm`/
  `o_proj` with a hardcoded 1e-5 epsilon independent of the model's own `rms_norm_eps`. The released
  checkpoint's `rope_parameters` is `{"rope_theta": null}` — no RoPE anywhere, on any layer — handled
  by a new `NoPositionEncoding` flag naming this as a fourth legitimate "no RoPE table" case in the
  existing position-information guard (alongside GPT-2/Nemotron-H/MLA). Parity-gated against a tiny
  oracle whose fixture reproduces the release's actual tensor layout: 100.0% / 0.9999999999998704.

- **Bailing Hybrid as a new family** (`bailing_hybrid`; inclusionAI, Ling 3.0): DeepSeek-style
  Multi-head Latent Attention alternating with Kimi Delta Attention (KDA) every `layer_group_size`
  layers being MLA, over a DeepSeekMoE FFN. MLA and the MoE router compose `deepseekArchitecture`'s
  existing code, parameterized for two real naming departures (both mixers are `self.attention`,
  not `self.self_attn`; MLA's output projection is `self.dense`, not `o_proj`) plus an optional
  per-head sigmoid attention-output gate riding the same mechanism Laguna's own gate already ships
  (sigmoid where Laguna's is softplus — a real, checked difference, not an assumption). KDA is the
  one genuinely new primitive: a delta-rule recurrence structurally identical to Gated DeltaNet but
  with a PER-CHANNEL decay (one value per row of the state matrix, where Gated DeltaNet's is a
  single scalar per head) — proven against `fla-org/flash-linear-attention`'s actual reference
  implementation, not the HF modeling file's opaque Triton-kernel call (this was proven ahead of
  time as a standalone rehearsal; see the "Fixed" entry below for what shipping it as a real family
  found). `layer_types` is not a config.json field for this family at all — the MLA/KDA pattern is
  computed from `layer_group_size`, replicated exactly from the real decoder layer's own layer-type
  formula. The real `BailingMoeV3ForCausalLM` can't be instantiated on this Mac (its modeling file
  imports Triton transitively, unavailable on this platform), so the tiny fixture is a hand-assembled
  reference verified piece by piece against the real source, with KDA's own recurrence calling
  `fla-org`'s real reference functions directly. Parity-gated: argmax exact, cosine
  0.9999999999999437; a deliberate negative control (reverting the decay to a per-head scalar)
  dropped cosine to 0.93984, confirming the gate actually discriminates the new primitive.

- **SmolLM3-3B as a new family** (`smollm3`): a plain llama-shaped dense GQA model with per-layer
  NoPE on every 4th layer via `no_rope_layers` — a field whose VALUES are the opposite of what its
  name suggests (1 = has RoPE, 0 = NoPE), verified against the real `modeling_smollm3.py` rather
  than assumed from the name; getting this backwards would silently flip which 9 of 36 layers are
  NoPE with correct shapes and plausible-but-wrong logits, no crash. Reuses the `Config` field and
  boolean convention `llama4_text` already established for the same JSON key, and the same
  `layerNoPE` `Architecture` hook `cohere2` already populates — no new mechanism, just composed
  onto a third family. Tensor names byte-identical to llama (`llamaTensorSchema` reused verbatim).
  CPU-only (`FeatNoPE` is undeclared on every resident backend, same as `cohere2`). Parity-gated
  against a tiny oracle at the release's own every-4th-layer pattern (100.0% / 0.9999999999999544).

- **Ministral 3 as a new family** (`mistral3`/`ministral3`; 3B/8B/14B): Mistral's GQA skeleton
  (tensor names byte-identical, reused verbatim) plus two real deltas found by checking the
  release rather than assuming a config alias: no sliding window at all on any released size, and
  YaRN RoPE with an extra field, `llama_4_scaling_beta`, that scales the query by
  `1 + beta·ln(1 + floor(pos/original_max_position_embeddings))` after RoPE, on every layer —
  Llama 4's own attention-temperature-tuning formula, generalized here into two new generic
  `Architecture` fields (`AttnTempBeta`/`AttnTempOrigMaxPos`, 0 = off) and wired into both generic
  forward paths (sequential decode and batched prefill/verify), proven to agree bit-for-bit. A new
  `FeatAttnTemp` resident-admission flag keeps this CPU-only until a GPU backend implements it —
  no backend declares it, so cuda/metal/webgpu all correctly decline rather than silently dropping
  the scale. Parity-gated against a tiny oracle whose prompt is deliberately longer than its
  `original_max_position_embeddings`, so the new mechanism is actually exercised, not identity
  (100.0% / 0.9999999999999605).

- **Qwen3-MoE as a new family** (`qwen3_moe`; Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct): qwen3's
  QK-norm dense attention with the FFN replaced on every layer by a sparse MoE, and — unlike its
  `qwen2_moe` sibling — no always-on shared expert, confirmed against the real released config.json
  and a real GGUF's tensor list. Pure composition of two already-shipped adapters: no new forward
  path, so the family rides the generic uniform-layer dispatch (`canBatchN`/`specRollbackSafe` answer
  correctly for free) and is resident-admitted on cuda/metal/webgpu from day one, same backends as
  `qwen2_moe` and `qwen3`. GGUF support included (`general.architecture == "qwen3moe"`, verified
  against a real file's header via HTTP Range, not assumed). Parity-gated against a tiny oracle
  (100.0% / 1.00000); real-checkpoint T3 is a follow-up (`docs/task-families-2026-09.md`).

- **Dense Granite 4.2 (3B/8B/30B) as a new family** (`granite`; distinct from the existing
  `granitemoehybrid` Granite-4.0-H). A plain llama skeleton — confirmed byte-identical tensor names
  to llama by instantiating `GraniteForCausalLM` directly — plus three of Granite's four scalar
  multipliers, all already generic on the shared `Architecture` descriptor (embedding/attention/
  logits scale); `residual_multiplier` is rejected unless 1.0, the only value any released 4.2 size
  ships. Reuses `llamaTensorSchema` verbatim — no new tensor schema. GGUF support included
  (`general.architecture == "granite"`, verified against a real file). Resident-admitted on
  cuda/metal/webgpu from day one (empty feature profile — every scalar that varies from identity on
  a real checkpoint is either baked into the generic attention scale or checked to be 1.0).
  Parity-gated against a tiny oracle with non-trivial multipliers (100.0% / 0.9999999999999).

### Fixed

- **`docs/capability-matrix.md`'s "GPU-resident" column read the wrong gate for six families**
  (`cohere`, `cohere2`, `mistral3`, `smollm3`, `olmo3`, `olmo_hybrid`): it derived from
  `decodeRunnerEligible()` alone — an arch-SHAPE predicate, "can the generic resident forward
  represent this at all" — instead of the full admission truth (`decoder/features.go`'s own
  `ResidentEligible`, which additionally checks whether any real backend's declared feature set
  covers what the arch needs), which `docs/hardware-matrix.md`'s generator already used. All six
  read "GPU-resident: yes" despite their own `admissionGolden` rows being empty (CPU-only) — the
  same failure mode the 2026-08-31 LFM2 fix patched for one family via a shape-level special case,
  recurring here through undeclared FEATURES instead. Fixed by pointing the column at the same
  `ResidentEligible` gate the other doc uses, so the two generated docs can't disagree about the
  same family again.

- **`decoder/serialize.go` (`.giw`) had no idea Bailing Hybrid's KDA mixer, or MLA's optional
  attention-output gate, existed** — the exact failure class `audit-2026-09-02`'s R3/C-03 already
  named for LFM2 (a field added to `LayerWeights` with no corresponding round-trip code, nil-
  dereferencing in the decode goroutine on the first token). Caught by
  `TestSerializeCensus_noSilentFieldDrop` panicking on a round-tripped `bailing_hybrid` fixture.
  Fixed by bumping the format to v9 (`kda.go`'s nine tensors, plus `mlaWeights`' `gProj`) rather
  than retrofitting v6Layer's fixed byte layout, which would have corrupted every already-shipped
  v6/v7/v8 file's read.

## [v0.16.0] — 2026-09-05

This release is two things at once. The headline is prefill: CUDA prompt ingestion moves to tensor
cores and is on by default above 512 tokens, MoE prefill runs expert-major, and the CPU path gets
head fan-out, a fused schedule and aikit's register-blocked int4 tile — so the prefill deficit
against Ollama, the repo's largest open gap, narrows on every backend. Underneath that is a
whole-repo audit (`docs/audit-2026-09-02.md`) and its review (`docs/review-2026-09-04.md`) worked
through in a fortnight, plus the first pieces of the onboarding work: a `pull` command, a browser
UI, a `serve check` doctor and a startup banner. Three defaults in this release change output at
temperature 0 on long prompts; all three are called out below with their opt-outs.

### Added

- **LFM2 / LFM2.5 as an experimental family** (`lfm2`; gated short convolution on most layers, GQA +
  QK-norm on the rest). CPU-only — no backend implements the short-conv feature yet — and
  parity-gated against a tiny oracle (100.0% / 1.00000). The bring-up found two silent bugs the
  forward hid (a zeroed `NormEps` read from the wrong JSON key, an absent `AttnScale`), which is why
  `resolveArchitecture` now runs a `validateResolved()` chokepoint for every family, written or not
  yet. The first cut also fell through seven family-dispatch lists (any prompt of two or more tokens
  panicked) and serialized without its conv weights; both were caught by the audit before the tag
  and are fixed here (`audit-2026-09-02.md` C-01/C-02/C-03).

- **`pull` — fetch a model without leaving the tool.** `goinfer-chat pull <owner/repo>[:quant]` and
  `serve pull` fetch a GGUF from HuggingFace, sha256-verified from the tree API, resumable; `demo:`
  shortcuts carry pinned digests, and a cached `demo:` ref resolves offline before any network call.
  `--model` accepts `hf:` and `demo:` refs directly. `pull -embed` bakes a pulled model into a single
  static binary. The package is exported as `pull` for embedders. HF repo names are validated
  against an allow-list rather than a slash count.

- **`serve -web`** — a local browser UI for chat and model pulls, served from the same binary. The
  root page is reachable without the API key so a browser can load it; the pull/list routes behind
  it are same-origin-checked and authenticated like the rest of the API.

- **`serve check`** — drives a running server the way an agent harness would (chat, streaming,
  tool call, structured output) and reports each surface; the structured-output check verifies the
  returned value, not only that JSON parsed, and the summary names what it skipped.

- **The startup banner now reports the state a harness otherwise has to discover** — backend,
  residency and prefill path (with its chunk width), and the declines that change what the server
  can do, including a pre-tokenizer shape this build cannot walk (`PreTokenizerDecline`).

- **gpt-oss decodes GPU-resident on CUDA.** The resident gate (G7) ran for the first time and
  failed: CUDA never applied gpt-oss's per-expert down-projection bias, and under expert caching the
  bias table was indexed by slot id rather than expert id. With `gemv_w4a8_moe_wacc_bias` the parity
  cosine went 0.750 → 0.9993 and the gate is green.

- **Release binaries are attached to the GitHub Release** (the README's download link previously
  pointed at nothing), the embedded 1.5B tier ships, and `NOTICE` now lists the Qwen2.5-Coder
  weights those binaries embed, with their licence (`audit-2026-09-02.md` M-33).

- **Peer benchmarking:** `scripts/bench_peer.py` gains llama.cpp as a third engine (with `--fit`
  left to place MoE layers rather than a forced `-ngl 99`) and MLX on the Mac; a new
  `scripts/bench_peer_prefill.py` measures TTFT-derived prefill rate with unique prefixes per
  request so a peer's prompt cache cannot stand in for its prefill. The redone matrix is scoped in
  `docs/task-peer-benchmarks.md`; its first pass is in `docs/benchmarks.md`.

- **MTP / NextN self-draft head loader** (`decoder/mtp.go`) — reads the head that every existing
  load path skips, by two detection routes because the formats disagree (GGUF declares a count in
  arch-prefixed metadata; the safetensors Qwen checkpoints declare nothing and are discoverable only
  by tensor presence). **Measurement adapter only**: nothing is wired into serving, the router or any
  generation path.

### Changed

- **CUDA prompt prefill runs on tensor cores, and is ON BY DEFAULT for prompts of 512 tokens or
  more.** Two new kernels — `attn_fused` (FlashAttention-style attention, `mma.sync m16n8k8`) and
  `gemm_w4a8_mma` (int4×int8 GEMM with group scales, `mma.sync m8n8k16`) — replace the batched
  decode-shaped kernels for prompt ingestion. **End-to-end prefill 3.91× faster** at a 3900-token
  prompt on a 1.5B int4 (5.451 s → 1.393 s) and **4.10×** at 512; against Ollama v0.32.5 the
  overhead-free marginal gap at depth narrows from **12.1× to 3.16×** (1.5B) and **14.5× to 1.89×**
  (0.5B), and goinfer is now faster to first token at every swept depth on the 0.5B.

  **These kernels are NOT bit-identical to decode** — the fused attention uses f16 K/V with an
  online-rescaled softmax, and the GEMM re-associates the cross-group float sum. They ship as a
  default only because `docs/task-prefill-gap.md` §3's fidelity gate passed: both arms scored
  against a CPU reference with f32 weights *and* f32 activations, teacher-forced on the reference's
  own continuation, over 10 prose prompts per cell on two models. **At depth the fast path is
  measurably CLOSER to that reference than the exact path it replaces** (hard flips 7 vs 10 at
  K=1024, 8 vs 12 at K=3900). It **fails at K=256**, which is why the 512-token floor exists and
  why it is 512 — a K=512 reference was generated specifically so the floor rests on a measured
  cell rather than an interpolation.

  `GOINFER_CUDA_FAST_PREFILL=0` restores the previous behaviour completely; `=attn` / `=gemm`
  select one lever. The exact path stays bit-identical to the M=1 decode kernels and is what
  spec-decode verify and the parity gates run regardless of this setting.

  **If you have prefill numbers from before this change, they measured the exact path.** In
  particular `TestPrefillTTFT`'s "batched" column now times the fast path at K ≥ 512 without the
  test having changed; set `=0` to reproduce the older rows.

- **Prefill attention now uses a FUSED (FlashAttention-style) schedule under `--cpu-fast-attention`,
  and THIS CHANGES OUTPUT.** The score block is kept resident and folded into the output
  accumulator with a running max and running sum, instead of materialising a `kt × nKeys` matrix
  and making three passes over it. **A long prompt can now produce a different response than the
  same build produced before, at temperature 0** — the committed long-prompt goldens are
  regenerated on both architectures.
  - **What it buys: +8.0% end-to-end** (dense 1.5B, K=4096, paired and interleaved). The kernel
    win is much larger — 1.69–1.73× over a whole prefill's tiles — but A3's head fan-out already
    took most of what attention had to give, leaving it ~18% of this prefill.
  - **This is a deliberate trade and it is a close one.** Eight percent for a user-visible output
    change is a worse ratio than the flip that introduced this flag (1.43–2.28×). It ships because
    the operator chose it after the measurement was presented, not because the number argued for
    itself.
  - **The divergence is small relative to what the flag already accepts**: measured on one
    checkpoint at one depth, acc64 vs f32-materialized is cosine 0.998283 and acc64 vs f32-fused is
    0.998262 — an increment of ~2e-5. It rides `--cpu-fast-attention` rather than adding a second
    user-facing flag for that reason. `GOINFER_FUSED_ATTENTION=0` restores the materialized path.
  - Declines, rather than approximating, for acc64 (whose bit-identity it would break) and for tree
    attention. See `docs/measurements/p19-fused-attention-2026-09-01.md`.

- **`--cpu-fast-attention` is now ON BY DEFAULT, floored at 512 prompt tokens, with
  `--cpu-exact-prefill` as the opt-out.** Prompt attention on the CPU backend runs in f32 unless
  you ask otherwise; the opt-out wins if both flags are passed. **This changes output.** Above the
  floor, prefill is no longer bit-identical to decode, and a long-prompt response can differ from
  what the same build produced before, at temperature 0.
  - **The floor is the part that makes it safe to default.** Attention cost grows with the square
    of the prompt, so the saving grows with length while the divergence does not — an eight-token
    prompt was measured diverging at the third generated token while buying nothing measurable
    (1.15× at 512, 1.43× at 2048, 2.28× at 8192). Below 512 tokens the exact kernel runs
    regardless, which is why every pre-existing forward golden still passes untouched.
  - **It is not reproducible across architectures.** The same prompt on the same checkout produces
    a different first token on arm64 than on amd64 — the compiler fuses multiply-add on one and not
    the other, and f32 has no wider accumulator to absorb it. `--cpu-exact-prefill` is the way to
    get a byte-comparable transcript between machines, and exists for that reason rather than as a
    courtesy.
  - Decode is unaffected either way, and speculative verify is structurally excluded (it passes the
    exact kernel as a parameter, not a runtime check).

- **`--cpu-fast-attention` now covers Mixture-of-Experts architectures.** The flag previously
  refused MoE outright, on the argument that an f32 QKᵀ reassociation flips a top-k expert at a
  near-tie and cascades. That argument had **never been measured on a MoE** — no MoE appears in the
  A3 kernel-ratio record, and both G24 divergence tests load the dense bench checkpoint, including
  the one whose doc comment claimed it pinned the exclusion and whose body asserted nothing about
  MoE at all.

  Measured, the argument is **half right**. The mechanism is real: at 28 layers, 14.5% of `moeMLP`
  calls select a different expert set, and replaying the acc64 routing recovers 70.1% of the
  divergence. The magnitude does not carry a categorical refusal — matched on depth *and* prompt
  length (28 layers, K=2048 both sides), MoE diverges **2.126e-3** against dense's **2.352e-3**,
  and a 48-token greedy continuation is **identical**. Both sit ~4× inside the ≥0.99 bar the flag
  already ships behind.

  **What it buys, on the full model rather than a slice: 1.52×** (Ryzen 3700X, int8int8, 28-layer
  Mellum2, K=8192, 8411.6 s → 5540.1 s), corroborated at 1.59× on an M1 Pro at int4. Earlier
  figures of 3.11× came from a **4-layer slice** and were quoted as if they were model-level
  numbers; they are not. The flag remains **off by default** — this removes a refusal, it does not
  turn anything on. Record: `docs/measurements/mellum2-moe-prefill-split-RESULT.md`.

- **Optimistic forward is gated at T ≤ 0.2** (previously enabled for all sampled decode). It is a
  measured loss of **5.5–6.8%** at the T = 0.7–1.0 range typical of chat, against a best case of
  1.1% at T = 0.2 — an asymmetry that settles the question independently of exactly where the
  crossover sits.

- `scripts/bench_peer.py` now takes its token count from each engine's own reported `usage` where
  the engine provides one, falling back to frame counting only where it does not, and records a
  `tokens_per_chunk` diagnostic per cell so the assumption is a recorded number rather than an
  inherited belief.

### Performance

- **Resident prefix reuse — an agent's third turn goes 9.13 s → 0.42 s (21.7×).** A GPU-resident
  model used to re-prefill the whole conversation on every request; it now reuses the resident KV
  for the longest matching prefix and prefills only the suffix, so a turn costs its new tokens, not
  its history. In `docs/benchmarks.md` §B10 that is a warm TTFT of 7 ms against a cold 1.7–1.9 s on
  the same 256-token prompt. The reused prefix is held to the same bit-identity contract as a cold
  prefill.
  - Recurrent families (Gated-DeltaNet, Mamba-2 hybrids) are handled rather than assumed: reuse
    first excluded them outright after it was found returning wrong output on Qwen3.5 (the
    positional bookkeeping has no notion of recurrent state), and now reuses an *exact extension*
    only — the same exactness rule the CPU `Session` already applied (`docs/task-recompute-audit.md`
    R-01 phase 0). A rewind still costs a cold prefill on those families.
  - The cache is committed wherever it is provably consistent — natural completion, both cancel
    exits, and speculative completion — and cleared by every path that writes it without completing
    (R-00, R-02, R-03; review V-04, V-09, V-11). Agent harnesses cancel constantly, and each of
    those used to cost the next turn a full re-prefill.

- **MoE prefill runs the experts EXPERT-MAJOR — 4.36× end-to-end, bit-identical.** At K=4096 on a
  full 28-layer Mellum2, batched prefill went 1206.9 s → 276.6 s (second pair 4.50×). Previously
  the FFN ran one row at a time, issuing its three expert matmuls at M=1, so an expert's weights
  were re-read for every token routed to it — ~10³ times per expert per layer at K=8192.
  - **Bit-identical, so nothing about the output changes** — no golden updates, no documented
    divergence, no flag to reason about. Achieved by computing every (row, rank) expert output
    first and folding each row in *routing-rank* order, since float addition is not associative and
    an expert-major visit order would otherwise change results.
  - It declines, rather than approximating, for a live expert pager, a shared expert, and the
    routing test seams; those fall through to the per-row path unchanged.
  - `GOINFER_MOE_EXPERT_MAJOR=0` restores the old path.
  - **The obvious explanation is wrong and was measured, not assumed:** per-row allocation churn
    (339,293 GCs / 20.9 GB in the profile) accounts for *none* of it — reusing one scratch across
    the row loop is worth 0.99×. See `docs/measurements/p18-expert-major-e2e-2026-09-01.md`.

- **f32 prefill attention now fans out over query heads — 1.58× at K=2048 and 1.92× at K=4096
  end-to-end, with byte-identical output.** CPU only; CUDA and Metal attention are separate
  implementations and are untouched. Applies to the f32 prefill path, i.e. prompts above the
  512-token floor with the default `--cpu-fast-attention`.
  - **It is bit-identical, so nothing about the output changes** — this is the same computation on
    more cores. The committed long-prompt golden, which pins real token ids through this exact
    path, passes unchanged, and the A/B asserts the two arms' logits match bitwise before timing
    anything.
  - **The item was costed at ~13% and the estimate was wrong for an instructive reason.** It rested
    on "the f32 branch is single-threaded by construction". That is true of the head loop but not
    of the work: `MatmulBT` already fanned out internally over output columns, so the path was
    running at 1.67× utilization, not 1.0×. The ~13% came from feeding a *serial-vs-serial* kernel
    ratio into an Amdahl model built on a *parallel-path* profile share. What was actually left on
    the table was the ~58% of the arm outside any matmul — the K/V gather, the per-row softmax and
    the scatter — which only head-level fan-out reaches.
  - Every worker slot already owned a full-size `kh`/`vt` pair (`prefillAttnWorkers` had budgeted
    them on every prefill all along, including the acc64 path that never touches them), so the
    fan-out needed no new allocation. `GOINFER_PREFILL_ATTN_WORKERS=1` restores the previous
    single-worker shape.
  - **SCOPE, measured after the fact and worth reading before quoting the number:** the 1.92× is a
    0.5B *dense* model at K=4096. On the full 28-layer Mellum2 **MoE** at K=8192 the fan-out adds
    nothing resolvable — 1.597× against the 1.59× that f32 alone already delivered — because at
    that size attention is ~70% of the work under acc64 and the f32 kernel swap alone collapses
    most of it. Do not quote these figures for large-MoE prefill.
  - Measurement, method and the retraction: `docs/measurements/a3-f32-attention-fanout-2026-09-01.md`,
    `docs/measurements/mellum2-fullmodel-profile-RESULT.md`.

- **CUDA C′ expert cache batches its uploads** through `gpu.UploadBatch` (aikit `gpu/v0.32.0`),
  submitting one batch per layer instead of two synchronizes per admitted expert:
  **20,916 copies → 2,038 synchronizes (10.3× fewer)**, worth **+9.3% tok/s** on
  gemma-4-26b-a4b-it int4 at 30 slots/layer (14.55 → 15.90 tok/s, arms interleaved in one session,
  two passes each, `-count=1`). H2D time falls 10.2% = **2.98 ms/token**, against the ask's
  predicted "~3.0 ms". Bit-exact: batching changes *when* copies are issued, never what is copied.

- **Metal `pread` expert staging** for the generic `qwen3_5_moe`-class MoE path, replacing the
  mmap byte-copy: **3.23×** (0.595/0.641 → 1.967/2.022 tok/s) on a 35B-A3B, M1 Pro 16 GB, by
  eliminating major page faults (98.5 per staging operation → essentially zero). Note the honest
  denominator: against the **CPU pager** measured in the same session the Metal path's advantage is
  **1.23×**, not 3.23× — the byte-copy arm was not the shippable alternative.

- **aikit v1.32.0 → v1.34.0, across all five modules.** The register-blocked W4A8 int4 tile (S-01)
  roughly doubled Apple Silicon CPU prefill (67.6 → 141.7 tok/s at K=512 on the M1 Pro, pre/post on
  one box): the Mac CPU-prefill peer row went from **2.98× behind Ollama at K=512 and 1.80× at 3900**
  to **1.54× behind at 512 and 0.91× — ahead — at 3900**, marginal ratio 0.86× goinfer-faster
  (`docs/measurements/cpu-peer-prefill-2026-09-05.md`); int8int8 no longer beats int4 at M>1.
  v1.33.0 replaced three Go fallback blocks with NEON — the int8 quantiser (13–19×), and the f64
  attention AV/QK accumulators (2.3–2.5× / 1.7–2.0×), worth **2.14× on depth-8192 attention** on the
  M1 — all bit-identical to the blocks they replace. v1.34.0 adds `MatmulBTW4A8Batch`.
  `aikit/gpu` v0.31.0 brings the device-to-device copy that spec/09's state-snapshot design was
  waiting on.

- **WebGPU decode attention splits over keys, not head dims — 2.4× at 1k context**; and the
  quantize kernel no longer computes its row max-abs serially on lane 0 (**104.8 → 118.4 tok/s**,
  bit-identical). §B10 measures the resident WebGPU paths **~1.25–1.35× up** on the June table.

- **Constrained decoding: an exact plain-string bitmap** replaces the per-token automaton walk for
  literal strings — mask **6.03 → 0.35 ms**, constrained decode **1.21×** end-to-end; the EOS map
  became a slice (mask −15%). Measured under `docs/audit-2026-09-02.md` P-20.

- **Loading and CPU MoE:** `loadFusedExperts` no longer materialises whole 3-D f32 expert tensors per
  layer × GOMAXPROCS (P-11); the Gated-DeltaNet state update walks its `[hk,hv]` state row-major
  instead of stride-`hv` (P-07); the fused-attention scratch that the P19 default allocated per
  decoded token and never read — 11.9 MB/token at context 256, 42.6 MB at 1024 — is allocated only
  where it is used (M-03); the safetensors source mapping is released at end of load, not at `Close`
  (P13); the layer pager's resident set is sized at window+ahead, which is what it peaks at (P-12).

- **Serving hot paths:** `/v1/embeddings` tokenized each input twice and prefilled one token at a
  time — it now tokenizes once and prefills batched (R-07); the streaming path re-decoded and
  rescanned the whole generated sequence on every token (R-08); `top_logprobs` no longer full-sorts
  the vocabulary per token (P-13); sampled n-gram speculation reuses its scratch instead of two
  full-vocab vectors per position (P-14); the repetition-penalty map rebuild was O(n²) over a
  generation (P-15). CUDA expert-cache uploads fold the slot index into the batch and the generic
  MoE pool is pinned like the Gemma-4 one (P-21, P-22).

- **CUDA batched prefill now covers MoE and non-uniform-geometry models, and no longer OOM-declines
  on long prompts — bit-identical throughout.** Three things kept M26 (Gemma-4-26B-A4B) and
  similar families on the one-forward-per-token sequential path:
  - **Per-layer geometry and K=V.** `prefillCore` hoisted layer 0's head/KV dims into every launch
    (Gemma-4's local/global layers differ), and its global layers have no `v_proj` (V is
    `v_norm(k_proj)`, handled here as a second k-projection GEMV into the V buffer). Neither needed
    a new kernel. `TestPrefillNonUniform_bitIdentical`: 0/256 logits differ vs the sequential path
    on a fixture with both shapes at once; mutation-proven (dropping the v_norm reddens 255/256).
  - **The MoE FFN guard.** `r.moe || r.gemma4Moe` is gone — a MoE layer's FFN now runs row-by-row
    off the batched residual (attention batches, routed experts keep decode's exact per-token
    order). **+8.0–8.5% end-to-end on real M26** (43.7→43.4 ms/tok at M=512/2048), against a 2.3×
    parameter-count projection that was wrong for a measured reason: M26's expert stack exceeds
    the card, so the host→VRAM expert DMA is **59.5% of prefill wall-clock and identical in both
    arms** (`GOINFER_MOE_CACHE_PROF=1`) — batching removed VRAM weight-reads that were never the
    constraint. End to end at depth 8000 the *cell wall clock* moved **384.3 s → 368.4 s (4.1%)** with
    the decode rate unchanged at 15.5 tok/s — the 8.0–8.5% above is per-token prefill, and the cell
    is prefill plus four decodes under prefix reuse, so the two numbers are consistent and measure
    different things. `TestPrefillMoE_real26B`: 0/262,144 logits differ, equality not tolerance
    (routing is a discrete argmax, so a small numeric drift runs a different expert, not a
    slightly-off one); mutation-proven (binding row 0 for every row reddens 4095/4096).
  - **Chunking.** Batched prefill was all-or-nothing on `M`, with `O(M×inter)` device scratch — an
    8012-token prompt on an 8 GB card (D7 int4 at `-ctx 8192`, the peer-benchmark harness's own
    deep cell) OOM'd on its first allocation and silently fell back to sequential, **4× slower,
    with `PrefillPath()` still reporting "batched"** because it reads static properties, not the
    per-call decline. Now runs in passes of ≤512 rows with a KV-only tail on non-final passes; an
    OOM retries at half width from the same position, and the surviving width is remembered on the
    resident. **8012-token D7: 153.9s → 50.9s, 3.02×.** `TestPrefillChunked_bitIdentical`:
    real 1.5B, 0/151,936 seed logits differ and 8 greedy decode steps match id-for-id;
    mutation-proven (an off-by-one position reddens all 151,936 logits).
  - **`Prefiller.PrefillLast` now takes a `context.Context`** (Experimental tier per
    `docs/api-tiers.md`, so the signature change is in-contract). The sequential fallback checks
    `ctx.Err()` **per token** — G18 added that so "an abandoned client leaves the whole prompt
    streaming through the device" could not happen — and a batched pass had no equivalent, so
    chunking would have taken a cancelled MoE request from ~46 ms to ~22 s of uninterruptible GPU
    work. Checked at entry, at each chunk boundary, and between rows of the MoE FFN loop: measured,
    a cancel at 237 ms returns at **239 ms** against 1.183 s uncancelled. A cancelled prefill
    returns `ctx.Err()` rather than falling through to the sequential loop the caller was also
    cancelling. Each of the three checks is separately mutation-proven — two of them were
    initially untested, because the MoE fixture's per-row check masked the chunk-boundary one and
    a prompt fitting in a single pass skipped both.
    **Still open:** `PrefillSeedArgmax` (block-spec seed) ingests a whole prompt and remains
    uncancellable; `decoder.ResidentSeedArgmax` carries no context.
  - `PrefillPath()` now states chunk width instead of claiming "one pass"; a call-time decline
    warns once via `warnPrefillDeclined`.
  - **Does not touch M35** (Qwen3.6-35B-A3B): 30 of its 40 layers are Gated-DeltaNet with in-place
    recurrent state, which needs a chunked delta-rule scan, not this batching — see the Fixed
    entry below for why that boundary matters. See `docs/measurements/prefill-moe-m26-2026-09-04.md`,
    `docs/measurements/prefill-chunking-d7-2026-09-04.md`.

### Fixed

- **`LoadSession` rejected valid session snapshots.** The guard bounding a blob's `pos` by
  `len(body)/(numLayers·kvDim)` assumed every position stores at least one byte of KV — which
  `kvsnapshot.go`'s own writer contradicts: a never-written ring and a KV-shared layer each
  serialise zero KV bytes, and a ring layer stores only `min(count, W)` rows. In production that
  rejected any session longer than ~8× the window on an all-sliding-window model. Replaced with two
  checks that do not assume a body layout: `pos == len(tokens)` moved *before* the allocation (a
  tighter bound than the old ratio for the OOM it was written for, and an invariant the format
  actually guarantees), plus an explicit ceiling on the bytes the cache would occupy. The original
  hostile-header protections are unchanged and still tested.

- **Speculative decode's `Theta` could not express Metal, so Metal drafted when it should not
  draft at all.** `AdaptiveDepth.Theta` was documented and enforced in `[0,1)` and reset anything
  outside that to 0.5. Metal measures **1.006–1.048** (two models, two depths), so every value
  Metal actually has was rejected and replaced by 0.5 — and because depth is
  `floor(ln Theta / ln alpha)`, a *smaller* Theta drafts *deeper*, so the substitute was the most
  over-drafting setting available. Measured effect on Metal: speculation was **1.07× slower than
  not speculating**; wiring the measured value recovers it (**~6%**) by declining to draft.
  - `Theta >= 1` is now legal and means what it measures — a verify node costing at least a whole
    target step, which no acceptance rate repays.
  - The periodic probe that forces `D>=1` to refresh a stale estimate is skipped under
    `Theta >= 1`, where it can never change the answer and is pure wasted draft work.
  - **Theta is now wired per backend from the probes** (CPU 0.5, CUDA 0.251, Metal 1.02) instead of
    every backend running the CPU constant. It keys on *resident vs staged*, not on the requested
    backend, because a declined-residency model verifies on the CPU path.
  - **Validated end-to-end on the RTX 2070 SUPER** (driver 595.91.07): on two of four cells the
    shipped 0.5 was *slower than not speculating at all* (0.91× and 0.92× of no-drafter), and the
    wired value returns both to parity (+9.3%, +6.0%) while leaving the already-winning cells
    unchanged. Zero lossless violations. The conservative 0.251 lands within ~1% of the true
    measured 0.155/0.235 on every cell, so under-claiming the win cost nothing here — a result,
    not a guarantee.
  - **Read the size honestly: this buys back a regression rather than adding a speedup**, on both
    Metal and CUDA. See `docs/measurements/theta-per-backend-2026-09-01.md` and
    `docs/measurements/theta-cuda-ab-2026-09-01.md`.

- **`stream_options: {"include_usage": true}` is now honoured on the OpenAI-compatible streaming
  surface**, emitting the spec's final usage-only chunk (empty `choices`, populated `usage`) after
  the finish chunk. Previously the field was accepted and silently dropped, so a streaming client had
  **no way to obtain a token count from goinfer at all** — it either guessed or counted SSE frames.
  This is an OpenAI spec conformance point, not a convenience: clients use it for cost accounting,
  rate display and context-budget tracking, and its absence is the kind of thing that reads as
  "goinfer does not support usage" rather than as a bug worth reporting.

  **The practical consequence for anyone who worked around it: counting SSE chunks under-reads
  goinfer's token rate, and only ever downward.** A chunk is not a token — the stream holds bytes
  back for UTF-8 continuation and for stop-string matching, so several tokens can arrive in one
  frame. Measured across 92 goinfer cells on this fix: chunks equal tokens exactly at temperature 0,
  and diverge with sampling to as much as **1.046 tokens per chunk (a 4.4% under-read)** on
  phi3-mini at T=0.6. A client measuring that way sees goinfer as slower than it is, silently and
  with no error to notice.

- **From the audit and its review** (`docs/audit-2026-09-02.md`, 2026-09-02/03;
  `docs/review-2026-09-04.md`, 2026-09-04) — the fixes a user could have hit, each with a gate that
  can fail:
  - **One writer per SSE response.** Two goroutines wrote the same `http.ResponseWriter` on the
    incremental tool-call stream; every frame in both protocols now goes through one `sseWriter`
    (C-06).
  - **Paged gpt-oss on Metal applied expert 0's biases to every routed expert** — finite, plausible,
    wrong (C-09).
  - **Sampling:** a rounding miss in `drawChunked`/`drawFull` could return a token top-k/top-p had
    removed (N-02); the chunked scans' decay underflowed f32 at real checkpoint rates (N-01).
  - **Vision:** user text could forge a turn boundary through the image path (M-22); the demo
    agent's image turn was encoded as ordinary text after that fix (V-03) and the image block is
    spliced at its last occurrence, not its first (V-19).
  - **Constrained output:** four ways the grammar constrained the wrong thing (M-27–M-30); embedded
    structs are promoted the way `encoding/json` promotes them, unexported-typed embeds included
    (M-28, V-14); `TokenText` reports an added token's surface verbatim in byte-level mode, so the
    grammar mask sees the right bytes (V-13).
  - **`serve`:** `-web` was unreachable when `-api-key` was set (V-02); `/v1/responses` now stores
    `tool_calls` so `previous_response_id` continuity survives a tool turn (V-18); the split-KV
    guard covers prefill, verify and the drafter and names its decline at the source (M-16, V-05 —
    the decline is still swallowed by `residentPrefillSeed`; filed as B20).
  - **Gemma-4 PLE `validateShapes` checked the wrong axis**; Metal's `DecodeTokenFusedBatched`
    deadlocked on the command-buffer cap; `LoadSession` validated after allocating (M-04); a
    header-only `.giw` bundle and a poisoned prequant cache are refused (M-09, M-12).
  - **Gates that could not fail:** metal-parity never selected `TestBatchedVerifyKernelParity`
    and now must list or explicitly exclude every gate-shaped test; webgpu-parity and metal-prefill
    could print PASS having run nothing (V-06, V-21); WebGPU has its own `gate gpu` group instead of
    a private env var (G-09); the suite frees the resident models that were emptying the card
    mid-run (peak VRAM 8034 → 274 MiB) and says "out of VRAM" instead of "no GPU".

- **Compute-time LoRA now refuses every own-forward family** (derived from `arch.ownForward()`,
  LFM2 included) instead of loading cleanly and applying nothing (review V-12).

- **A batched-prefill change above accidentally removed the only thing keeping Gated-DeltaNet
  models off it — caught and fixed the same night, before any shipped row was actually wrong.**
  Removing the categorical `r.moe` refusal also removed the sole guard keeping M35 (`qwen3_5_moe`)
  off the batched path, which has no notion of recurrent state: a Gated-DeltaNet layer's conv ring
  and matrix state must advance strictly one token at a time, in order, and a batched pass would
  advance them `M` rows at a time and return plausible-but-wrong logits. `ForwardN` already
  excluded this deliberately (`r.prefillReady && r.dnet == nil`); the new `PrefillLast` path never
  had to, because `r.moe` happened to be refusing M35 for an unrelated reason.
  - **No shipped commit was actually unsafe — checked, not assumed.** A DeltaNet layer loads no
    q/k/o (it builds `dnQKV`/`dnZ`/`dnOut` instead), so the same commit's `nonBatchableKind` fix
    (an absent weight had been reported as kind `""`, read by its caller as "no problem") still
    caught M35 by coincidence and declined it. **That was an accident of this family's weight
    layout, not a guard**: a hybrid whose recurrent layers also carried q/k/o would have sailed
    past the weight-kind check into the dense attention stack — the same silent-wrong-computation
    bug class as the LFM2 incidents (`docs/audit-2026-09-02.md` C-01), which has reached `main`
    twice before.
  - `prefillStaticDecline` now refuses `r.dnet != nil` directly, for the true reason. New fixture,
    `TestPrefillPath_recurrentDeclines`: a synthetic DeltaNet model **with valid int4 q/k/o** — the
    one shape the accidental guard could not have caught, and which no real checkpoint here
    produces, so the real guard had never been exercised by anything before this. Carries a
    no-`dnet` control and is mutation-proven (disabling the check turns it red).
    `TestPrefillPath_matchesPrefillCore` could not have caught this on its own: it asserts the
    guard and the report agree, so deleting the check just makes both sides say "batched" together.

### Measured and NOT shipped

- **Expert-major MoE prefill batching is parked.** Its ceiling was measured at **under 5%** at
  K=1–2k and was **not resolvable above run-to-run spread** at K ≥ 4096 — the two passes disagreed
  in sign at K=8192. Attention runs **77.3% → 97.1%** of MoE prefill work from K=1024 to K=8192
  while all weight matmul falls to 2.9%, so the two levers scale in opposite directions. The case
  for it has to be made on streaming I/O, not compute, and measured there.

- **MTP acceptance (spec/09 Gate 1) passes on all three suites** — 2.024 / 2.905 / 2.476 tokens per
  verify on code / math / chat against a 1.60 cross-target reference, on a 0.8B target. Read as
  pre-registered: a pass on **mechanism**, not economics. Gate 2 was not evaluated and is not
  evaluable at that scale, one prompt per suite makes the third digit noise, and the MTP-bearing
  families are exactly the ones `specRollbackSafe` refuses — so shipping would require the
  state-checkpoint track first.

## [v0.15.0] — 2026-08-27

**CPU decode roughly doubled on Apple Silicon, the Gated-DeltaNet family went GPU-resident on two
backends, and a 35B runs on an 8 GB card.** 183 commits since v0.14.0, in six days. The one users
of v0.14.0 should act on is smaller than any of that: **`.giw` was silently dropping gpt-oss
attention sinks**, producing a CRC-valid, sink-free bundle that generated confidently wrong text
with no error at write, load, or run time. If you built a `.giw` for gpt-oss on v0.14.0 — via
`prequant` or `serve --stream-weights` — rebuild it.

### Added

#### Residency

- **Gated-DeltaNet is GPU-resident on WebGPU *and* CUDA** — `qwen3_5` dense, its MoE sibling and
  `qwen3_next` all resolve to a real runner instead of a feature-gate decline. **15.9× CPU decode**
  on CUDA at released width, measured 2026-08-20 on the pre-2026-08-25 driver stack and NOT
  re-anchored since (driver `595.91.07` / Nobara 44); the correctness gates were re-verified across
  that bump, the throughput figure was not. Resident-vs-CPU worst cosine 0.999919,
  replay-after-Reset self-cosine 1.000000000.
- **Qwen3.6-35B-A3B decodes RESIDENT on an 8 GB card** — ~20 GB of int4 experts against 8 GB of
  VRAM, via C′ host→VRAM expert streaming. Raising the C′ slot default to `8*topK` is worth
  **1.59×** on the 35B and **2.6×** on the 26B.
- **Metal MoE expert paging is generic**, no longer gemma4-only.
- **`gpu`: strided-lane wide attention** — `head_dim` is no longer a reason for the WebGPU backend
  to decline a model.

#### Loaders and formats

- **Qwen3.8 GGUF loader (arch `qwen35`)** — **55.6 GB bf16 → 16.5 GB** on disk (3.4×). A 1.69×
  decode difference against the safetensors path was also measured, but only with 46.8 GB of 62 GB
  already resident, and its own record says not to assume it transfers — so it is recorded, not
  claimed. Real 16.5 GB Q4_K_M gate passes: 64 layers, correct geometry, distinct-trigram 0.847.
- **`.giw` v6 — every registered family is representable and `canSerialize` is now EMPTY.** The
  tail writes `GProj` (Laguna), `AttnSinks` and per-expert biases (gpt-oss), the MLA sub-struct
  (DeepSeek/Kimi) and the Mamba-2 sub-struct (Granite/Nemotron). The tail is UNCONDITIONAL rather
  than arch-gated — arch-gating is precisely how gpt-oss's sinks went missing.
- **arm64 W4A8 row4 repack wired into GGUF and safetensors loading**, at the two streaming choke
  points every family's int4 path funnels through. A no-op off arm64 and on any shape the repack
  rejects, so it needs no per-family changes.

#### Elsewhere

- **Optimistic next-token forward for sampled decode** — **+21.4% at T=0.2** on CUDA.
- **`cmd/gate`** — seven shell/Python gate scripts collapsed into one Go runner over
  `go test -json` (`census`, `heavy`, `parity`, `composition`, `selector`, `gpu`, `mutation`).
- **DeltaNet Gate D0**, an env-gated component timing split (`GOINFER_DELTANET_TIMING=1`): on real
  35B-A3B decode, projections 37.6% / recurrence 42.1% / other 20.2%, reconciling to 100.0%.

### Changed

- **`--cpu-fast-attention`** — opt-in f32 prompt attention on the CPU backend. **2.28× faster
  prefill on an 8k prompt** (dense 1.5B, M1 Pro: 602.9 s → 264.6 s): attention is ~70% of a long
  prefill and the f64-accumulating kernel is ~8× slower than f32 at those shapes. **Not
  bit-identical** — measured cosine 0.9976 against the default, stable across 256/1024/2048-token
  prompts — so a long-prompt response can differ from the default even at temperature 0. Off by
  default; decode is unaffected. **Speculative decoding is never affected** (its verify pass always
  uses the exact kernel, structurally, or verify would stop matching greedy), and **MoE is refused
  at any setting** because an f32 reassociation can flip a top-k expert at a near-tie and cascade.
- **Streaming tool calls stream their prose incrementally** on the ChatML/Qwen, Mellum2 and Gemma 4
  families, instead of arriving in one delta at the end. Only families whose tool-call parser
  derives its prose as a prefix of the output qualify; Mistral and Llama-3 normalize theirs, so
  they keep the buffered behavior. Nothing about tool-call parsing changes — the full output is
  still parsed as before, so what a client finally receives is identical; only *when* the prose
  arrives is different.
- **Streaming tool calls now emit SSE keep-alives instead of going silent — and a streaming
  generation error is now an SSE error event, not a 500.** The tool paths must buffer the whole
  generation before a tool call can be parsed, so `stream: true` with `tools` sent **zero bytes
  until the generation finished** — measured at **1682.6 s to first byte** against a harness whose
  idle timeout was 300 s, which no output survives however correct it is. They now emit SSE
  **comment frames** (`: ping`) while the buffer fills; comments carry no data and every SSE parser
  drops them, so tool-call parsing and the emitted content are unchanged.
  **Compatibility note:** keep-alives require the SSE headers to be sent *before* the generation, so
  on `/v1/chat/completions` and `/v1/responses` **with `tools` and `stream: true`**, a generation
  failure now arrives as a mid-stream `error` event on a 200 response rather than as a 500. This
  matches what the non-tool streaming paths already did (a status code is no longer available once
  headers are flushed). **Non-streaming requests are unchanged and still return 500.**
- **`/v1/messages` now validates message roles** — the array accepts only `user` and `assistant`,
  as the upstream Anthropic API does, and anything else is a `400 invalid_request_error` naming the
  offending role. Previously **any** unrecognized role — a typo, an invented name, or `system`
  (which is a top-level field on this API, not a message role) — was silently folded into the
  conversation as a *user* turn, restructuring what the model saw with no error anywhere. Real
  Anthropic-shape clients only ever send legal roles, so this costs them nothing; everyone else now
  gets a loud failure instead of a quiet mangling. Applies to `/v1/messages/count_tokens` and the
  vision path too, which share the same conversion.
- **`role: "developer"` is accepted as an alias for `role: "system"`** on the OpenAI-compatible
  routes — same position, same last-one-wins precedence two `system` messages already have.
  OpenAI's newer APIs send the system prompt under that role for reasoning-class models and agent
  harnesses have followed; notably, at least one harness sends it to *any* endpoint it cannot
  identify, which is every goinfer deployment. Previously the role matched no case and fell through
  to a **user** turn — silently demoting a whole agent scaffold into the user's first message.
  `/v1/messages` is unaffected (the Anthropic API carries `system` as a top-level field and has no
  developer role).
- **`/health` and `/v1/models` no longer describe CPU prefill as plain `batched`.** The word named a
  weight-streaming *shape* and was being read as a throughput claim; the measured best-case CPU
  prefill rate is the same order as the model's decode rate, on one thread. The field now says so.
- **`aikit` v1.21.0 → v1.28.0**, **`aikit/gpu` v0.28.0 → v0.30.0**, all five modules aligned.
- **Go 1.26.6 → 1.27.0** across all five modules.
- **Attention restructured (A1), bit-identical, 2.4–2.8×** — `attendBatchedHeads` now uses aikit's
  `MatmulQKAcc64`/`MatmulAVAcc64`. Real 1.5B shape on M1 Pro: depth 128 **10.84 → 4.54 ms**,
  depth 8192 **828.77 → 298.36 ms**. Zero logit difference on the parity tests.
- **`int4` is the right CPU default again on Apple Silicon.** An earlier measurement this cycle
  found `int8int8` ~60% faster and the guidance was changed to say so; with the W4A8 kernel and
  the int4-mode LM head both shipped it is now backwards — int4 matches or beats int8int8 at both
  real sizes, at half the RAM. **Both measurements were correct when taken**; the guidance moved
  twice because the code did.
- **The qwen35 projections honour `Options.Quant`** instead of staying f32 — **1.60× decode,
  7.4× TTFT**, and it is a deliberate bandwidth trade with a stated cost: min cosine on the GGUF
  gate moved 0.99298 → 0.98740 and coherent prompts 8/10 → 7/10. If you run qwen3_5 and care more
  about the last fraction of accuracy than about decode rate, that is the trade you are now taking.

  > **CPU prefill now runs attention's heads in parallel** (v0.15.0, G16), bit-identical to the
  > serial path: measured M1 Pro, dense 1.5B `int8int8`, **1520 tok 89.7 s → 33.8 s (2.65x)** and
  > **3020 tok 333.3 s → 101.6 s (3.28x)**. The fan-out is budgeted against per-slot scratch that
  > grows with the square of prompt length, so it steps down toward serial on very long prompts —
  > attention is computed in query-row tiles (G20), so per-slot scratch stays flat — ~16 MB at an
  > 8k prompt rather than 272 MB — and the fan-out survives long prompts instead of collapsing to
  > one worker. Measured at 8192 tokens: **2381.8 s → 607.6 s (3.92x)**.
  > `GOINFER_PREFILL_ATTN_WORKERS=1` restores the old serial behavior.
  >
  > **Prefill remains superlinear in prompt length**
  > — roughly n^1.85 for *both* quants (measured M1 Pro, dense 1.5B: 170 tok 4.1 s · 620 tok 22.5 s ·
  > 1520 tok 100.3 s · 3020 tok 355.5 s at int4; int8int8 within ~5% at every point). That is
  > attention's own cost curve, not a quantization effect, and it is why a multi-thousand-token
  > prompt is minutes on CPU. It does **not** change which quant to pick: int4 and int8int8 scale
  > alike. See `docs/queue-performance.md` G16 for the single-threading, which is the lever.

### Fixed

- **Prefill ignored cancellation — an abandoned client left a core running.** The serve layer passed
  `r.Context()` into the generation correctly, so it reviewed as wired, but the context never
  reached the work: `prefillLogits` / `forwardLayersN` / `runLayersFromEmbedN` took no context at
  all and the resident prefill loop never checked one. A client that gave up left the whole prefill
  running to completion, and a harness that retries stacked one generation per retry — measured at
  **47:38 of CPU with nothing attached**. Context is now threaded through the prefill chain and
  checked per layer (per token in the sequential fallback and the resident loop). **The bound is one
  layer's work, not instant:** ~12.3 s at a 3072-token prompt, against a full prefill of ~335 s
  (`int8int8`) or ~1587 s (`int4`). Found by a real agent harness driving the server, not by a gate.
- **`.giw` silently dropped gpt-oss attention sinks** (see the note at the top). `canSerialize`
  ACCEPTED gpt-oss; per-head sinks were per-layer state the writer had no field for. Refused first,
  then properly represented in v6. Laguna was accepted too and failed loud at load instead of
  silent — still a broken artifact produced without a warning.
- **The `realckpt` test build was broken on `main`** — two incompatible `envOr` in one package, so
  the release parity sweep could not compile, let alone run. Untagged `go vet` was clean, which is
  why CI stayed green. **CI now vets the `realckpt` build**, the configuration the sweep uses and
  the one nothing built.
- **Optimistic-forward raced the resident's logits buffer on CUDA.**
- **`matmul`/`matmulInto` never actually called `w.MatmulBTW4A8Into`** — the W4A8 kernel was
  shipped but not reached from the dispatch path.
- **The kind-4 pager double-`WILLNEED`'d**, registering more than the span the kernel reads.
- **qwen35 GGUF transcode streams** instead of building the model resident first.
- **`TestQwen35GGUF_weightDiff` died on a stale precondition and named the wrong tensor.** Its
  helper was f32-only, which held until the projections started honouring `Options.Quant`; the
  failure message hardcoded "router" while the tensor that was not f32 was `in_proj_qkv`. It
  measures rather than refuses now, and the router is confirmed bit-identical.
- **Two GPT-2 int4 golden floors** raised/exempted on measured grounds.

### Measured and NOT shipped

Recorded because a negative result costs the same to obtain and is worth as much:

- **`.giw` kind 4 — correctness fully green, throughput regresses 25–30%.** Kind-3 vs kind-4
  byte-identical at real scale on both bundles, paged full-vs-paged decode byte-identical with real
  non-vacuous eviction, T3 green — and it is a **loss**, not the projected ~1.3× win, confirmed at
  a fixed budget and at one matched to kind-3's residency fraction. The dispatch-inertness trap and
  a paging-budget artifact were both ruled out.
- **DeltaNet D1 parked** — a 6–7% projected ceiling, below the ~10% floor this repo already used to
  reject another retry, and at *more* implementation cost. Parked shovel-ready, not abandoned.
- **Metal `quant_vec`+o-proj fusion: net loss.** **CUDA `attn_block_full` max-shuffle: refuted**
  (0.41%, ranges overlap) — reverted, benchmark kept.
- **W4A8 is compute-bound on NEON, and this is the second independent confirmation.** It runs
  ~1.6× slower than W8A8 while reading 1.6× fewer bytes; the nibble-unpack ALU is the limiter. A
  June spike measured 1.58× and shelved a 3-bit follow-on for the same reason; this cycle measured
  1.63×/1.57× on different hardware two months later.

### Parity and evidence

- **The peer benchmark was re-measured on both boxes, and its fairness guarantee had to be
  replaced.** `bench_peer.py` promised "the same GGUF file on both sides (verify by md5)". That is
  **unsatisfiable through Ollama's import path for any model**: `ollama create` repacks the
  container in a different tensor ORDER, so the file hash always differs even when every tensor is
  bit-identical. New `scripts/gguf_same_weights.py` compares per tensor at each file's own offsets
  — **339/339, 339/339, 291/291 tensors identical** across the three models used.
- **The CPU deficit was found, diagnosed, and largely closed.** goinfer decoded 0.32–0.54× of
  Ollama on CPU at the start of this cycle (worse on arm64, not better). The quant confound was
  real but partial; thread count and E-core inclusion were **ruled out by measurement** for
  goinfer, while confirming llama.cpp's 6-thread default is doing real work. With the W4A8 row4
  repack, the int4-mode LM head and the A1 attention restructure shipped, M1 Pro int4 decode went
  **34.5 → ~82 tok/s (0.5B)** and **17.0 → ~40 tok/s (1.5B)**.
- **Two op-level goldens stopped pinning identity weights.** `granite_mamba` and `qwen35_deltanet`
  pinned HF's default init, leaving `D=1`, `dt_bias=1`, `norm.weight=1`, `conv1d.bias=0` — so a bug
  in how goinfer *applies* those (×1 or +0) could not move the reference and the test passed
  regardless. Regenerated from HF with seeded non-trivial values; both pass, so those paths are now
  genuinely pinned. A scan of all 89 goldens found no other all-identical weight array.
- **`TestQwen35GGUF_gate` floor re-baselined on the bisect, not on "it changed".** The 0.992 min
  floor was set long ago and never revisited; `6d4fc79`'s deliberate bandwidth trade crossed it.
  Min → 0.985 **plus a new mean floor at 0.995**, so the re-baseline does not simply lower a bar:
  the mean now carries the systematic-drift duty because min is box-sensitive (~0.0015 across CPUs).
  Both sit below the measured value with margin rather than at it.

## [v0.14.0] — 2026-08-19

**Six new model families, gpt-oss on Metal, and the v1.0 API tiers declared.** 188 commits since
v0.13.0. The headline for consumers is smaller and more boring than any of that: **the tagged
submodules are `replace`-free**, so `go install github.com/townsendmerino/goinfer/...@v0.14.0`
works from outside a checkout for the first time (v0.13.0's `gpu`/`cuda`/`metal` tags each carried
`replace … => ../`, which `go install pkg@version` applies because that module becomes the main
module — and `../` does not exist).

### Added

#### Model families

- **Qwen3-Next (`qwen3_next`)** — the 80B-A3B Gated-DeltaNet/softmax hybrid, gated by a real-weight
  layer-slice oracle at **cosine 1.00000000**.
- **Nemotron 3 Nano (`nemotron_h` MoE)** — T3 real oracle **cosine 0.997668**, continuation exact,
  plus a real Q4_K_M GGUF gate (coherent, 0.843 distinct-trigram).
- **Laguna (poolside) XS-2.1 / XS.2 / M.1** — safetensors **and** GGUF (llama.cpp's own arch),
  tiny goldens at cosine 1.000000 for all three generations, a real 33B-A3B gate, and a
  real-weight slice oracle at cosine 1.00000000. Softplus attention output gating and **per-layer
  query-head counts** are new axes; the real gate found two per-layer-head bugs a tiny fixture
  could not.
- **InternLM2** (adapter: renamed tensors + a GROUPED fused `wqkv` de-interleave) and **InternLM3**
  (a llama ALIAS — its dynamic-NTK rope is in-window identity).
- **Qwen3.8 (`qwen3_5`)** — the dense member of the DeltaNet/softmax hybrid; see below.

#### gpt-oss

- **safetensors/MXFP4 loader** for `gpt-oss-20b` and `-120b`, gated against the GGUF path
  (argmax identical, cosine 0.999121).
- **The `harmony` chat template** — without it the family was unreachable through `chat.Detect`.
- **GPU residency on Metal (G10): SHIPPED and admitted end-to-end** — attention sinks, the
  clamped-SwiGLU MoE expert kernel, and a custom router. The CUDA kernels landed too, **without
  touching the audited PTX**.

#### Elsewhere

- **GPT-2 GPU residency on Metal (G9)** — shipped, admitted end-to-end.
- **Block-drafting speculation as a production API**: `serve --drafter`, `BlockSpec.GenerateStream`,
  a resident block trunk, batched hidden-state capture, and `PrefillLastNArgmax` (batching the
  verify's LM head).
- **Metal batched prefill behind `--metal-fast-prefill`** (P11) — a 3.9–4.6× TTFT lever, opt-in
  because it is not bit-exact.
- **`serve`: optional native TLS** (`-tls-cert`/`-tls-key`), and a **hard failure when `-addr` is
  non-loopback with no `-api-key`** — the unauthenticated-by-default posture is now enforced, not
  merely documented.
- **The v1.0 API tier declaration** (`docs/api-tiers.md`, signed off 2026-08-18) and the
  **apidiff gate** that enforces it (`scripts/apidiff_check.sh`, wired into CI). v0.13.0 → this
  release is **clean: zero incompatible changes** to any Hard-tier name.
- **`docs/release-1.0-gate.md`** — v1.0 as a decision against criteria, each line naming its
  evidence.

### Changed

- **`aikit` v1.17.1 → v1.21.0** (root module).
- **GPT-2's `"gelu"` now runs the exact-erf form.** It had been silently running the tanh
  approximation — a different function, not a different rounding.

### Fixed

- **Qwen3.8's two `pos>0` bugs**, the Laguna per-layer-head pair, and the batched-prefill path that
  never applied the attention gate (it read as a plausible cosine 0.957, not as a failure).
- **`multimodal`: oversized Qwen vision inputs are rejected before the pixels are decoded**, not
  after.
- **The w4a8 decode-parity gate now RUNS** — its matched int8 `.giw` half had never been built, so
  it had skipped at every tag since 2026-08-12. Built from the same source GGUF as the int4 half;
  **16/16 greedy-token agreement** on first invocation.
- **The parity-manifest emitter no longer promotes on evidence it does not have.** `EMIT_MANIFEST=1`
  used to write `status: validated` beside `method: tiny-golden` for four families, and mangled
  `real-model-oracle` into a name no tier rule recognises. Status is now DERIVED from the method,
  the method is a closed vocabulary, and a source census over all 18 call sites runs in plain CI —
  where the emitter itself never does, which is why the typo survived for months.
- **Two tautological CUDA gates**: the expert-cache bit-exactness tests never checked the cache was
  engaged (and `allocSlots` really does clear it silently), and the greedy fast-path test never
  checked `ResidentGreedy`. Both would have compared a path to itself and reported success. Guarded
  and mutation-verified on the hardware.

### Measured and NOT shipped

Recorded because a negative result costs the same to obtain and is worth as much:

- **CPU block drafting is a loss** — 0.75× on dense, and break-even (8.89 tok/round) exceeds the
  ceiling (8.00). Implemented, measured, unwired.
- **DFlash pairing on CPU MoE: 0.82×** — works, lossless, and still a loss. Do not ship.
- **Metal's batched small-M verify kernel: bit-identity holds, the ceiling does not** (NO-GO).
- **`--drafter` is a loss on the served default.** The published 1.60×/1.50× figures rested on a
  wrong baseline; re-measured, code is 1.44× and math 1.58× while **chat is 0.61× unguarded**.
  Corrected in place rather than quietly dropped.

### Parity and evidence

- **Mellum2 real-weight slice oracle (G11)** — the CPU half at **cosine 1.00000000** on both the
  sequential and batched-prefill paths, and the Metal resident half at 9/12 argmax-exact,
  cosine 0.972006. This closes the one open correctness gap G10 opened by admitting Mellum to
  Metal's resident path with no end-to-end validation there.
- **B13's standing reds are closed.** `TestSerializeWeightsTo_matchesBuffer` was already green.
  `TestQwen35GGUF_vsSafetensors` was **reclassified on measured mechanism, not on a moved floor**:
  the router is bit-identical between containers, every transform-bearing tensor sits at a uniform
  Q8_0 noise floor, the per-layer divergence curve has no step, and the two containers pick
  different top-8 experts in 779 of 3200 decisions — quant noise at a routing decision boundary,
  not a loader defect. The gate now floors the MEAN with a measured min.

### Qwen3.8 (`qwen3_5`), in full

- **The dense member of the Gated-DeltaNet/softmax hybrid family.**
  Alibaba's Qwen3.8-27B (2026-08-14, Apache 2.0) is the same 3:1 hybrid goinfer already runs
  as `qwen3_5_moe` and `qwen3_next`, with a plain SwiGLU where they have a router. The
  checkpoint is multimodal; **this is the text decoder only** — the vision tower
  (`model.visual.*`, 333 tensors) and the MTP head (`mtp.*`) are never requested.

  The structural change in the forward is one FFN branch; everything else — the DeltaNet
  step, the gated softmax attention, the hybrid cache, the sequential prefill — is the
  existing path untouched. Registered as both `qwen3_5` and `qwen3_5_text` (the released
  config nests the text dims under `text_config` and states each spelling at a different
  level).

  Three things were read off the released checkpoint rather than inherited by resemblance,
  and each would have been a silent wrong answer:

  - `head_dim` is **256** at `hidden_size` 5120 with 24 heads, so **nH·hd = 6144 ≠ hidden**.
    Deriving head_dim (or the query projection width) from hidden is wrong for this family.
  - `attn_output_gate` is true, so `q_proj` is **double width** (query ‖ gate, 12288 rows).
  - The DeltaNet projections ship as `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b`
    — qkv fused, z separate — which is **neither** `qwen3_next`'s fused pair
    (`in_proj_qkvz` + `in_proj_ba`) **nor** four fully-separate tensors. The existing split
    reader is the right one; the index is what says so.

  The config carries `mrope_section [11, 11, 10]` with `mrope_interleaved: true`. For
  **text** input this reduces exactly to standard partial RoPE — `position_ids` arrive 2-D
  and are expanded to three identical components, so the interleaved overwrite is a no-op —
  which is why no m-RoPE code was added. Image input is a follow-on, not a silent gap.

  **Parity: tiny-oracle.** HF f32 tiny golden (tiny-random `Qwen3_5ForCausalLM`, text path):
  argmax exact, logit cosine 1.000000, greedy continuation exact. The fixture keeps the
  released model's *shape character* rather than its size — head_dim independent of
  hidden/heads, 3:1 `layer_types` so both mixers run, GVA value/key head ratio, and
  `mrope_section` present.

  **And the real 27.8B runs.** `TestQwen38Real_gate` loads Qwen/Qwen3.8-27B (18 bf16 shards,
  55.6 GB) at int4 on a 62 GB linux/amd64 box, asserts the geometry and BOTH mixers' tensor
  sets against the released index, and generates 96 greedy tokens: distinct-trigram 0.770,
  three correct Paris landmarks with correct detail (Champ de Mars, the 1889 World's Fair,
  the *Mona Lisa*). That is **coherence, not an oracle** — no bf16 reference forward was run
  against it — so the family stays `experimental` rather than claiming a validated tier.

  **Admission: CPU-only**, the same posture as every DeltaNet hybrid (no backend implements
  the mixer). GGUF and vision are follow-ons.

## [v0.13.0] — 2026-08-14

### Security

- **Go 1.26.5 → 1.26.6 across all four modules**, closing three standard-library
  vulnerabilities that are **reachable from this project's own code** — not dormant
  transitive imports:

  | | | reached via |
  |---|---|---|
  | [GO-2026-6090](https://pkg.go.dev/vuln/GO-2026-6090) | `crypto/tls` — limit post-handshake messages | `serve` → `http.Server.ListenAndServe` → `tls.Conn.HandshakeContext`; also `prequant.readHead` → `io.ReadFull` |
  | [GO-2026-6089](https://pkg.go.dev/vuln/GO-2026-6089) | `net/http` — apply `ReadHeaderTimeout` on the unencrypted HTTP/2 check | `serve` → `http.Server.ListenAndServe` |
  | [GO-2026-5972](https://pkg.go.dev/vuln/GO-2026-5972) | `encoding/asn1` — enforce maximum recursion depth | `serve` → `signal.Notify` → `asn1.Unmarshal` |

  `govulncheck ./...` reports **no vulnerabilities** after the bump. If you build
  `goinfer serve` yourself, use Go 1.26.6 or later; the `go` directive now requires it.

  The CUDA gate was re-run in full on 1.26.6 before tagging — a toolchain change is a
  compiler change, and the resident parity gates exist to catch a forward that moved.

### Performance

- **Prefill is ~4.5% faster on one measured shape**, from `aikit`'s f32 blocked-matmul rework
  (arriving with the v1.17.1 bump below). **Benchmark-level +4.49%** (median; bootstrap 95% CI
  +4.24% to +5.26%).

  **Measured, and the method is part of the claim:** `BenchmarkPrefillLong`, Qwen2.5-Coder-0.5B
  (dense), **f32**, **512-token** prompt, batch prefill, on one box (Ryzen 7 3700X, linux/amd64).
  Both arms interleaved in a single session, warm-up discard and a 0.6% significance floor fixed in
  advance from the instrument's own characterization, 12 retained samples per arm. The arms do not
  overlap and per-visit medians are consistently ordered across three rounds. Full record, including
  every raw sample and the pre-registration: `docs/measurements/aikit-v1.17.1-prefill-ab.md`.

  **Scope — what this does not say.** One model, one prompt length, one quantization, one box,
  **prefill only**. It says nothing about decode, about other prompt lengths, about MoE or Gemma4
  architectures (which route down a sequential per-token path and never reach this shape), or about
  any other machine. A *derived* figure of ~8.6% within the reworked kernel itself follows from
  dividing by that path's profiled share of runtime — **derived, not measured**, and quoted second
  for that reason.

  *Recorded because it was measured to the standard this project demands of a regression. A record
  that discloses losses and withholds equally-measured wins is biased, not cautious.*

### Changed

- **`aikit` v1.16.0 → v1.17.0, `aikit/gpu` v0.27.0 → v0.28.0** across all five modules. A
  dependency update. The quantized GEMV is untouched — `gpu/testdata/gemv_quant.ptx` is
  byte-identical across `gpu/v0.27.0..gpu/v0.28.0`, and the `gemv_quant.cu` diff is three comment
  lines — as is the vision tower (`vit.ptx` byte-identical, `ViTBlock`/`LNBLOCK` still 256). What
  reaches goinfer is in the root module: a new AVX2 int8 kernel behind `w8a8Span` on the W8A8 decode
  path, and a reworked inner loop in the blocked f32 matmul. Both are argued bit-identical upstream;
  what demonstrates it here is the forward goldens — **33 passed, 0 failed, 9 skipped, of which 14
  drive a quantized path and 19 are f32**, all against recorded values.

  **The only performance figure carried for this bump is goinfer's own prefill measurement above.**
  Upstream reports its own numbers; those are not reproduced here. Decode was also measured and is
  **flat** — no benchmark-level change against v1.16.0 — recorded in
  `docs/measurements/aikit-v1.17.1-decode-ab.md` rather than claimed here, because "no measurable
  change" is not a release-note item.

## [v0.12.0] — 2026-08-12

**MINOR** (0.11.0 → 0.12.0): a correctness release for the CUDA MoE expert cache, three
bit-identical performance items, and behaviour changes that are disclosed above. No public API was
removed. The core-numerics surface is unchanged since `6edd1ca`; every change to a path covered by
`testdata/parity_manifest.json` carried a goldens run whose axis composition is printed with the
result.

> **⚠ Gemma 4 output changes on CUDA and Metal.** Gemma 4 previously fell back to the CPU path
> unless you set `GOINFER_GEMMA4_RESIDENT`; it now runs GPU-resident by default. The resident path
> is W4A8 — it quantizes activations to int8, which the CPU path does not — so **logits differ and
> a token can flip at a near-tie.** Both paths are parity-gated — argmax-exact with a calibrated
> cosine, which is the contract; GPU and CPU are not byte-identical to each other — and
> argmax agreed at every position on the new real-width gate, but this is a real output change for
> anyone running Gemma 4 on a GPU. It is opt-out: `--backend cpu` keeps the previous numerics.
>
> **This is the third consecutive release to change output for some users** — v0.10.3 moved the
> sampler tie-break, v0.11.0 shifted the seed→token mapping for the temperature and filtered
> sampling paths, and this one moves Gemma 4 from CPU to GPU numerics. Each was individually
> justified and each was disclosed, but three in a row is worth stating plainly in one place
> rather than leaving a user to assemble it from three release notes. If you depend on
> reproducible output across upgrades, pin a version and read this section before moving.

### Changed
- **Gemma 4 is resident by default on CUDA and Metal.** `GOINFER_GEMMA4_RESIDENT` is now inert and
  can be dropped from any script that sets it. The flag was a bring-up gate that outlived its
  purpose in an instructive way: because every real Gemma-4 checkpoint reaching the resident path
  was only ever compared against *itself* (expert-cache on/off, CUDA-graphs on/off), the flag had
  become the only thing standing between users and a forward whose numerics no gate asserted at
  real width. It comes off because that gate now exists, not because it looked like residue.
  WebGPU still declines (no Gemma kernels) and E-models (E2B/E4B, PLE) decline everywhere.

### Added
- **A real-width parity gate for the Gemma-4 resident forward** (`TestGemma4MoEScaled_residentParity`)
  and the composition gate the expert cache never had (`TestGemma4MoE_cacheExpertsBitExact_scaled`,
  `..._cacheReuse_scaled`), on a new fixture that keeps the real per-expert row geometry
  (hidden 2816, `moe_intermediate` 704) and transplants per-group weight scales from the real 26B.
  The scaled cache gate had never executed since it was written: it named a fixture that did not
  exist. Generate with `scripts/pin_gemma4_moe_scaled.py`.

### Performance

All three are **bit-identical** — same operands, same order, no tolerance involved.

- **Gemma's final-logit softcap runs in parallel: 1.43 ms → 640 µs** per sampled token at a 262,144
  vocabulary. Elementwise, so each output depends only on the input at the same index and the split
  cannot change a bit. Sampling path only — greedy reduces the argmax on-device and never paid it.
  The threshold is measured rather than chosen: below ~32k elements the split is a *loss* (8,192
  elements parallelise at 0.95×), so small vocabularies keep the serial path.
- **MoE experts share one gate/up buffer pair per token instead of one per expert** — at top-k 8,
  **16 allocations per token become 2**. The experts run sequentially, so the extra pairs were never
  simultaneously live. Applied to both MoE forwards.
- **W4A8 reuses the per-stream `Workspace`** instead of allocating a fresh one per projection per
  token. It was excluded by a dispatch that tested "is this W8A8" rather than "does this weight have
  a form that takes a workspace"; it now asks the second question.

### Fixed
- **The MoE expert cache sizes itself correctly on an 8 GB card, and the 26B decodes.** The cap was
  a division over a raw byte sum; the CUDA driver charges each of four buffers per MoE layer its own
  whole **2 MiB quantum**, so the requirement is a *step function* of the slot count. At 34 slots all
  four tip at once — putting the requirement **203,816,960 B over free** — and the granted cap
  allocated successfully and then could not launch. The forward produced **zero tokens**.
  `capSlots` is now a **search** over the granularity form rather than a division (a division plus a
  correction term is wrong at exactly the boundaries the failure lives on), and it is the single
  implementation `allocSlots` calls. On this card the auto-cap moves **34 → 31** and the 26B decodes
  coherently.
- **The deferred first-launch reservation is paid before the cache is sized.** The on-GPU router
  kernel declares per-thread scratch, and the driver backs it for the device's occupancy the *first
  time that kernel runs* — **138,412,032 B retained, 289,013,760 B demanded at the launch itself**,
  none of it visible to the free-VRAM reading the cap was computed from. Forcing that launch before
  the reading makes the cap correct **by construction** instead of covered by a margin. Measured
  after: nothing at all is consumed between the sizing decision and the end of the token.
- **A launch that runs out of memory now names the kernel and both slot counts.** Previously a bare
  `cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY` with nothing tying it to the setting that caused it. It
  now names the kernel, the **requested** count and the **effective** count after capping — they
  differ once the cap fires, and naming only the effective one sends someone who set 48 to lower it
  to 40, which caps to the same value and fails identically.
- **A resident decline now always says why.** The reason was gated behind `GOINFER_RESIDENT_DEBUG`
  on all four backends, so a model silently moving its entire forward to CPU looked identical to
  one running resident. Unconditional now — one line at load.

### Docs
- **The 26B-A4B section is retracted and rewritten.** It told users the slot count was "a manual
  workaround for a safety net that is not holding". That was accurate and is no longer: the cap
  holds. The section now records what the old behaviour was (34 slots allocates, then cannot launch),
  names both costs the cap was missing with their measured figures, and gives a version test —
  `capping to 33` has the fix, `34` does not. The example sets the slot count high and lets the cap
  choose, because it can now be trusted to.

### Verification

- **int4 now has forward goldens** — 23 fixtures across 16 architectures, comparing int4 output
  against *recorded int4 output*. int4 is the documented default quantization and nothing gated it:
  every golden that ran was f32, so a change that was correct in f32 and wrong in int4 passed. This
  is a note about what the gates cover, not a feature; nothing in the runtime changed because of it.
  The goldens run that gates a core change went from 19 to 33 tests, 14 of them on a quantized path,
  and now prints that composition alongside the count.

### Known unfixed

- **~150 MiB of reported-free VRAM is not allocatable**, on this driver and card. `cuMemGetInfo`
  reports it as free and `cuMemAlloc` refuses it at *any* request size down to 1 MiB —
  **151,191,552 B**, measured, cause unattributed. It is not fragmentation (a request 2.71× smaller
  than free was refused) and not exhaustion.

  This is why the 384 MiB slot margin cannot simply be lowered to recover the two slots the sizing
  fix costs: **151,191,552 B of that margin is this floor, not slack.** A 128 MiB margin was measured
  working on this card and is *below* the floor — it worked because the cap it produced happened to
  leave enough leftover, which is luck rather than safety. The margin holds the correct value for the
  wrong-looking reason, and lowering it needs the floor understood first.

## [v0.11.0] — 2026-08-10

**MINOR** (0.10.3 → 0.11.0): backward-compatible additions and behavior changes — new flags
(`-ctx`, per-model `ctx=N`, `GOINFER_SPLITKV_MIN_KEYS`, `-unload-drain-wait`), eligibility changes
(`top_k=1` greedy routing, MoE router cap 256→512), and performance/gating fixes. No public API was
removed. The core-numerics surface is unchanged since `6edd1ca`; the delta from there to the tag is
docs-only.

> **⚠ Sampled output for a given seed changed in this release** (the temperature-only path AND the
> filtered `top_p`/`top_k`/`min_p` paths). **The distribution is unchanged** — same sampler semantics,
> same probabilities — only the seed→token mapping shifts, once, for both paths together. If you pin a
> seed for reproducible sampling, expect a different token stream; see the sampling entry below.

### Docs
- **The post-v0.10.3 "still slow" campaign is closed and filed** (`docs/completed/plan-still-slow.md`;
  a pointer remains at `docs/plan-still-slow.md`). Relay executed 2026-08-09. Outcomes: the sampling
  cliff is largely closed — temperature-only decode +126–137%, temp+top_p +98–108% (P2b), `top_k=1`
  +18–22% (P1); the published sampled deficit fell from 2.1–2.9× to 1.08–1.40×. `top_k=1` closed
  outright (C-14 was already fixed; stale docs corrected, P0). Long-context was measured to 32k —
  linear with a plateau coefficient (~0.74/1.0 µs/pos CUDA, ~9–12 Metal vs peer ~0.03–0.09; 5.54× at
  32k) — split-KV re-gated (+19% @0.5B@512) and a live regression fixed (P6a); KV-quant was refuted
  as a speed lever on both backends (P4, reachability-only, build-deferred) and prefill (P5) stays
  banked. The remaining decode lever is a non-bit-identical FA-class rewrite (unfunded).

### Changed
- **Sampled output for a given seed changed; the distribution is unchanged.** The full-vocabulary
  softmax denominator on both the temperature-only and `top_p` paths is now summed in parallel over
  a fixed 64-chunk split rather than in one sequential pass. Float addition is not associative, so
  the regrouping moves the denominator by a few ULPs, which can flip a draw that sits exactly on a
  token boundary. Same distribution, same sampler semantics — only the seed→token mapping shifts,
  once, for both paths together.
- **Temperature sampling is 1.6–4.7× faster on the host, and up to 2.4× end-to-end.** The
  normalization is parallel (fixed chunk count, reduction ordered by chunk index — so output stays
  identical across machines and `GOMAXPROCS`, gated by `TestChunkedSoftmax_MachineIndependent`), and
  the separate normalize-divide pass is gone: the draw walks unnormalized weights with a two-level
  chunk walk. Measured decode-only on an RTX 2070 SUPER, q4_K_M int4, 128-token prompt:
  temperature-only **97.6 → 220.2 tok/s** (qwen2.5-coder-0.5b), **56.6 → 134.2** (gemma3-1b),
  **88.8 → 113.7** (phi3-mini); `temperature+top_p` **93.4 → 184.7**, **56.3 → 117.0**, **81.5 → 94.7**.
- **`top_k=1` requests now take the on-device greedy fast path.** A request with `top_k=1` and no
  `logprobs` / `logit_bias` / repetition-presence-frequency penalties is deterministic — temperature
  scaling is monotone so it preserves ordering, and a one-token distribution has nothing to sample —
  so it now skips the full-logits readback (594 KB/token at a 151936 vocab) exactly as
  `temperature=0` does. **Emitted tokens are unchanged**; this is a routing change, gated
  byte-for-byte by `decoder.TestTopK1_MatchesGreedy`. Measured on an RTX 2070 SUPER (decode-only,
  prefill excluded, q4_K_M int4, 128-token prompt, 8 completions × 2 runs): qwen2.5-coder-0.5b
  **271.4 → 319.2 tok/s (+17.6%)**, gemma3-1b **144.3 → 175.9 (+21.9%)** — landing on the greedy
  figure in both cases. Implemented as a new `Sampler.GreedyEquivalent()` predicate rather than by
  widening `ArgmaxEquivalent`, so each call site chooses deliberately. **Speculative-decode
  eligibility is unchanged** (those paths gate on `temperature <= 0` directly): `top_k=1` with a
  temperature is still not speculative-eligible. Metal inherits the routing but has no e2e
  measurement yet.
- **Parity `status: validated` now MEANS a T3 run, enforced — and a new `experimental` tier lets a
  weak row stop lying.** The manifest parsed `Method` as an opaque blob and never checked it against
  the T3 method list, so a row could sit at `validated` with a T1 artifact in a T3 slot and be counted
  as *supported* (the staleness gate keys on `deps_hash`, which says nothing about HOW a row was
  validated). `TestParityManifest_methodTier` now requires a T3 method (`full-forward-oracle` /
  `real-model-oracle` / `weightDiff` / `shared-path`) for `validated`; a genuine but sub-T3 row takes
  the new `status: experimental`, keeps its method and metrics, renders as `experimental: <method>` in
  the generated capability matrix, and is **excluded from the supported count**. Validation-semantics
  and docs only — no numerics.

### Documentation
- **Deep-context decode is measured for the first time (8k/16k/32k) and it is linear, not
  superlinear.** With the cap raised, `docs/benchmarks.md` §B7 records decode by KV depth out to
  32000 against Ollama v0.32.6. goinfer's per-position cost rises from +0.330 µs/pos mid-range to a
  **plateau of ~+0.74 (0.5B) / ~+1.0 (1.5B)** reached by ~8k and confirmed flat by a 16384→32000 probe
  (+0.748 → +0.735) — so deep context is an optimization target with a predictable cost, not
  unreachable in principle. Against the peer that is a flat ~25× per-position penalty (5.54× end-to-end
  at 32k on the 0.5B). The bound is **not DRAM bandwidth**: goinfer moves KV at 6–10% of peak at depth
  while Ollama is 3.2× faster reading identical bytes. This decides `docs/plan-still-slow.md` **P4** —
  KV quantization is **no-go as a CUDA speed lever, go as a reachability lever** (4× VRAM cut from f32).
  The 1.5B/32000 cells were deliberately skipped and the omission is stated in §B7's header.

### Changed
- **Kimi K2 is now GPU-resident-eligible on WebGPU: the MoE router capacity cap is raised 256 → 512.**
  This is an **eligibility** change, not a correctness change — nothing about how a model computes
  moved. K2 ships 384 routed experts, and the router scored experts into a fixed-size scratch array
  of 256, so it correctly declined to CPU rather than route on only the first 256 (plausible-looking
  wrong output). Raised to **512**, which also covers DeepSeek-V4-Pro's 384; it deliberately stops
  short of Kimi-K3's 896, since an unbuilt family should not set a validated limit. Models past the
  new cap still decline cleanly, gated by `TestResidentMoECapacity_routerCap`. **`Kimi K2` flips
  `CPU → ✅ resident` on WebGPU in the generated hardware matrix.**

  "The cap" was three shader constants plus one Go map, not one number: `gpu/moe.go` (WGSL, 512),
  `cuda/moe.cu` `MOE_MAX_E` (512), `metal/moe.go` (**still 256** — raising it is Mac-validation work),
  and `residentBackendMoECap`. **Metal is now DECLARED in that map for the first time** at its real
  `{256, 64}`: the map claimed "absent entry = no fixed-size router cap", which was false for Metal
  and made the generated hardware matrix derive a different answer than the runtime gives.

  Note the CUDA raise does **not** by itself make K2 resident there — `FeatMLA` is declared by WebGPU
  only, so on CUDA/Metal K2 declines on features regardless of capacity. CUDA's raise admits non-MLA
  MoEs at 257–512 experts and clears one of two blockers ahead of any MLA-on-CUDA work.

  Raising CUDA's constant required regenerating the **audited** `cuda/testdata/moe.ptx`, which the
  repo otherwise avoids. Done at the **pinned, identical** NVRTC 12.6.85 with a byte-identical
  rebuild-unchanged control run first; the resulting PTX diff is confined to `moe_route`'s stack depot
  (2368 → 4416 B) with `gemv_f32_a8` / `gemv_w4a8_moe` / `gemv_w4a8_moe_wacc` / `shared_gate_combine`
  byte-identical. Procedure, shas and the perf control (router 1.038× at nE=8, within the 5% budget):
  `cuda/testdata/REGEN.md`. The policy comment now states the actual rule — *never regen at a
  different toolchain*, a pinned same-version regen is the sanctioned edit path — so the next reader
  does not clone the kernel into a second file to dodge it.

### Added
- **`-ctx N` raises the CUDA resident KV capacity, which was a hard-coded 4096.** The effective cap is
  `min(model context window, -ctx)`; per model via `--model name=path,ctx=N`. **The default is
  unchanged** — `-ctx` unset still means 4096, so nobody who did not ask allocates deep-KV VRAM, and a
  request past the cap still fails cleanly to the staged path. The KV a configured cap implies is
  computed and **VRAM-checked at load**, after the weights are resident and before the caches are
  allocated: a configured cap that does not fit is a **hard startup failure naming the cost**
  (`resident context 32768 positions needs 1.88 GB of KV (56.0 KB/position across 28 layers) but only
  0.37 GB is free …`), surfaced in the startup `decode path:` line and on `/health`, and a non-zero
  exit under `-require-backend`. A *default*-cap miss keeps the historical decline, so existing
  deployments cannot start failing to boot. This is an allocation-size change only — no kernel and no
  numerics; `checkCap` guards the same invariant at the new cap. Measured: at `-ctx 8192` the 1.5B's
  VRAM rose exactly +224 MiB over the default and at `-ctx 32768` +1570 MiB, against a predicted
  +1568 MiB. It unblocks the 8k/16k/32k depth cells, which previously fell to the staged path.

### Fixed
- **Split-KV decode attention is re-gated per geometry; up to 1.19× faster decode on CUDA, with
  byte-identical output.** This is a **kernel-selection change only** — no kernel, numerics, or weight
  path was touched, and both arms remain bit-identical (`TestSplitKV_bitIdentical`), so any given
  request produces exactly the same tokens as before. The previous gate turned the split-KV path on at
  `nKeys ≥ 256` for every model, a constant characterized on one geometry (qwen2.5-1.5b) by an
  in-process best-of-3-minimum microbenchmark. End-to-end measurement across four geometries × six KV
  depths shows it fired **3–12× too early** and was a live regression of up to **18–25%** — and that it
  was wrong on its own geometry too (the 1.5B loses at both 256 and 512; its real crossover is
  (512, 1024]). phi3-mini (MHA, nH=32) **never** benefits at any depth: its ratio declines
  monotonically to 0.754 at 3900. Because a threshold rule always predicts "wins eventually", no
  formula can express that, so the gate is now a measured per-geometry table with a "never" class and a
  conservative default, documented with its derivation beside the constants and pinned by
  `TestSplitKVGate_measuredGeometries`. The gate also now tests the **effective attended span** per
  layer rather than the raw position, so sliding-window layers are judged by their window: gemma3-1b at
  3900 ctx now beats both uniform arms (163.0 vs 154.5 all-split and 142.4 all-single). Measured
  decode-only, RTX 2070 SUPER, q4_K_M int4: qwen2.5-coder-0.5b at 512 ctx **240.0 → 286.6 tok/s**,
  gemma3-1b at 512 **152.7 → 167.7**. `GOINFER_SPLITKV_ATTN=0` still force-disables; new
  `GOINFER_SPLITKV_MIN_KEYS=<n>` overrides the threshold without a rebuild. Full table and method:
  `docs/benchmarks.md` §B6. Note these thresholds are **not device-portable** — the occupancy term
  scales with SM count and all cells are one 40-SM part.
- **`POST /admin/models/unload` now frees the model's native memory instead of leaking it.** Unload
  deleted the entry from the registry but never called `Close()`, and with no ARC or finalizers under
  purego, GC reclaimed only the Go wrappers — so each unload/reload cycle leaked a full model
  (measured ~450 MB on Metal; on an 8 GB CUDA card the reload then OOMs). Adding a bare `Close()` was
  a use-after-free: a request between resolving the model and taking its mutex touches the model
  unlocked, so `Close()` could free weights mid-request (a driver SIGSEGV on CUDA). The fix is a
  drain: a per-`*decoder.Model` liveness `RWMutex` is held by every request from resolution through
  completion (via a single `withModel` wrapper; the old `pick` is gone so no handler can skip it),
  and unload unpublishes the model, then on a detached goroutine waits that lock out before closing —
  freeing the shared model only when it is the last owner (a base and its adapters share one). Design
  and rationale: `docs/completed/task-admin-unload-drain.md`.

### Changed
- **`POST /admin/models/unload` no longer returns `409 busy` for a model that is generating; it
  drains.** The model is unpublished immediately (unroutable at once), then the response is a bounded
  wait: `200 {"freed":true}` when the native memory is released within `-unload-drain-wait` (default
  5s), else `202 {"freed":false}` with the release continuing in the background. `GET /health` gains a
  `draining` array listing models whose memory is not yet freed; `?wait=false` returns `202`
  immediately. The old `409` was only ever safe because unload never actually freed anything.
- **New flag `-unload-drain-wait` (default `5s`)** — how long an unload waits for in-flight requests
  to drain before answering `202` (the release always completes regardless).

## [v0.10.3] — 2026-08-08

### Security
- **Request bodies are bounded before they are buffered or tokenized (G1/G2/G3).** An oversized
  body was read and tokenized in full before the context-window check ran, so a body under the
  32 MiB chat cap (e.g. 20 MB) drove RSS up by gigabytes and spent ~27 s tokenizing only to be
  rejected 400, and a ≥40 MB body tripped the reader mid-upload and closed the connection with no
  HTTP response. This ran before the request queue, so `-max-queue` did not bound it. Now, in
  increasing cost order: a **Content-Length pre-check** rejects an over-cap declared body with a 413
  before a byte is read (O(1) in body size — measured 6–145 µs, ~5 KB heap at 1/20/40 MB); the
  existing `MaxBytesReader` backstop's 413 now **names the limit and received size**; a
  **pre-tokenization guard** rejects an input whose text cannot fit the context window before the
  O(n) tokenize (the ~27 s → µs fix), as a conservative upper bound that never rejects a servable
  prompt; and the body cap is **derived per route** (`-max-body-bytes` to override every route) and
  reported on the startup line. Gated in `internal/serveapp/bodylimit_test.go`.
  - **The cap is derived per route, not globally** (pre-tag review). The text cap comes from the
    largest served *decoder's* context window; the vision routes (`/v1/chat/completions`,
    `/v1/messages`) add 32 MiB for base64 image data; and **`/v1/embeddings` gets its own 64 MiB
    cap, independent of any decoder**. That last one was a real false rejection: `/v1/embeddings` is
    served by the encoder, which is not in the decoder registry, so on an embed-only server the cap
    collapsed to the 4 MiB text floor and rejected a batch the route's own bounds
    (`maxEmbedInputs=2048`, `maxEmbedInputBytes=1 MiB`) accept — a legal 2048×4 KiB batch is ~8 MiB.
    Multimodal was checked and is safe: the pre-tokenization guard measures only `type:"text"`
    content parts, so base64 image bytes are never charged against a context window they do not
    consume. Both pinned in `internal/serveapp/bodycaps_routes_test.go`.

### Fixed
- **`temperature` is validated consistently with `top_p` (G4).** A negative `temperature` returned
  200 while `top_p=-1` returned 400. It was not silently-wrong output — the sampler treats
  `temperature <= 0` as greedy argmax before any scaling — but the inconsistency is fixed: negative
  `temperature` is now a 400; `0` stays valid as greedy/deterministic.
- **A chat request with no `messages` is now a 400 (G5)** naming the field, instead of a 200
  generated from a BOS-only prompt.
- **An unknown `model` name is rejected on a single-model server (G6).** `pick` fell back to the
  only loaded model for any name, so a wrong id produced confident output from a different model; it
  now serves an omitted name but rejects a non-empty unknown one, naming what is served (the
  multi-model behavior was already correct).
- **`POST /v1/embeddings` is registered even when no embedding model is loaded (G7).** It returned
  Go's text/plain `404 page not found` (SDKs surface it as a wrong-URL `NotFoundError`); it now
  returns a JSON 501 naming the `-embed-model` flag that enables it.
- **Sampling with `top_p`/`top_k` no longer full-sorts the whole vocabulary on the host every
  token.** The filtered-sampling path softmaxed all V logits and then ran a `sort.Slice` over all V
  — an O(V·log V) sort whose per-comparison reflection cost made it *present* as linear in V, single-
  threaded on the decode critical path. It is replaced with bounded selection in logit space
  (`decoder/sampler.go`): a k-bounded min-heap for `top_k` (O(V·log k)), an adaptive partial
  selection for the `top_p` nucleus (grown until the retained mass provably reaches `p`), and a
  `min_p` logit-space threshold — with the softmax applied **only to the retained set**, not all V.
  Measured on synthetic logits at `temperature=0.8, top_p=0.95`: selection cost **122.3 ms→1.80
  ms/tok at 152 064-vocab (68×)** and **233.0 ms→4.37 ms/tok at 262 144-vocab (53×)**; the full
  sampling path at `temperature+top_p` now runs within **1.1–1.5×** of temperature-only sampling
  (was ~7×). `top_k` no longer funnels through the general `top_p` routine. A test-only reference
  implementation gates the bounded path bit-for-bit across a wide seed sweep and both vocab sizes
  (`decoder/sampler_selection_test.go`), and a throughput gate asserts the ratio to the
  temperature-only baseline. Speculative decoding's distribution vector uses the same selection, so
  it stays lossless. **This does not address the separate `temperature==0`→nonzero cliff** (loss of
  on-device argmax → full logit readback per token); that needs device-side kernels and is scoped in
  `docs/ollama-chase.md` §8 D6.

### Changed
- **Tied-probability selection order is now specified.** The old `sort.Slice` filter was not stable,
  so tokens with equal probability were ordered arbitrarily — and since that order feeds the
  cumulative-CDF draw, it was an unspecified part of the sampling result. Ties now resolve by
  **ascending token id**, and the renormalization/`top_p` sums run in a fixed **descending-
  probability** order (a load-bearing contract: a different summation order can move the denominator
  by ULPs and flip a draw near a `top_p` boundary). Greedy (`temperature==0`) argmax is unchanged.
  Sampled output for a given seed may therefore differ from prior releases at tie/boundary points;
  the distribution is unchanged. (Same defect class as the CUDA argmax-reduce index tie-break.
  **Correction (2026-08-09):** this entry said that tie-break was "open ... not fixed here". It was
  already fixed — audit C-14, `c6600fc`, 2026-08-05 — which predates this release, so the statement
  was wrong at press time, not merely overtaken. Gate: `cuda.TestArgmaxTieBreak`.)
- **`docs/benchmarks.md` now requires sampling configuration as a per-number metadata field**, on
  the same footing as machine/driver/peer-version/date, for goinfer and the peer. Rows whose
  sampling config was not recorded are marked `sampling: unrecorded` and must not be assumed greedy
  (the one such row is the §B2 v0.32.5 re-measure box).

## [v0.10.2] — 2026-08-08

### Changed
- **The `.giw` bundle format (v5) now records the resolved quant label in its header** instead of
  leaving the reader to re-infer it (T1-6 follow-up, structural). A freshly baked bundle stores
  `int4` / `int4mix` / `int8int8` / `int8` / `native`, and the loader **prefers** that field; a
  pre-v5 bundle (or a streamed bundle, which writes the header before its layers load) has the field
  absent and falls back to the corrected tensor-kind inference — so **existing bundles keep working
  unchanged**. The reader still accepts v3/v4. Also single-sourced the "which tensors define the
  quant" fact: `quantLabel` and the recorded field now both classify over one `bodyMatmulWeights`
  (matmuls minus the int8-pinned logit tables), so the T1-6 class of bug — two lists of one fact
  drifting apart — cannot reopen within the decoder. (The cuda batched-prefill gate inspects the
  resident per-layer projections, a different type in a different module; it agrees on excluding the
  logit tables but cannot share the host-side list — documented at both sites.)

### Fixed
- **`--quant` on a prequantized `.giw` bundle is now a startup error when it conflicts, instead of
  being silently ignored** (T1-7). A `.giw` is serialized at a fixed precision, so `--quant` cannot
  re-quantize it — but it was accepted and dropped, so a user who passed `--quant int8int8` at a
  `.giw` baked at int4 got int4 with no signal. `serve` and `chat` now fail before binding a port:
  `--quant "int8int8" cannot apply to the prequantized .giw bundle <file> — it is baked at "int4",
  and a .giw carries its own quant; pass --quant int4 or omit --quant` (same shape as the
  safetensors int4mix decline). The comparison uses the corrected `Quant()` label, not the raw
  header field. A **matching** quant proceeds silently, and a bare default never conflicts — only an
  *explicit* `--quant` (global flag actually passed, or a per-model `quant=` override) is checked, so
  scripting one command across GGUF and `.giw` targets still works.
- **The submodule build command is now correct in every active doc** (field-report F2). The README
  build section already carried the verified per-backend entrypoints (v0.10.1); this sweeps the
  rest: `docs/cuda-backend.md` used the pre-split `go build -tags cuda ./cmd/serve` (the root, now a
  no-op) — replaced with the verified `CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`.
  The historical `docs/releases/v0.9.0.md` keeps its original command (correct at v0.9.0) but gains
  a forward-pointer note; `docs/completed/audit-2026-08-05.md` is left verbatim (it documents the
  finding). Every command written was run in-tree first. Note: the *published* v0.10.0 GitHub
  release notes still show the pre-split command and a literal `<exact command — fill from the repo
  before publishing>` placeholder — flagged for a maintainer decision (not editable from the repo).
- **`go build -tags cuda|gpu|metal ./cmd/serve` on the ROOT module now fails the build instead of
  silently producing a CPU binary** (audit D-B / field-report F1). Since the M-19 submodule split
  the root `cmd/serve` imports no backend, so those tags were accepted, exited 0, and yielded a
  CPU binary 48 bytes larger than the untagged one — the trap anyone upgrading from a cached
  README, blog post, or the v0.9.0 release page falls into. The v0.10.1 guard already failed the
  build, but with one generic message; it is now **one guard file per tag**, and the compiler
  error names the *exact* replacement command (`… go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`,
  and the gpu/metal equivalents). A new `TestBackendTagGuardFailsBuild` runs in the default
  `go test ./...` (no `-short`, no build tag): it shells out `go build -tags <backend> ./cmd/serve`
  for all three backends and asserts both a non-zero exit and the exact replacement command in
  stderr. The submodule entrypoints (`./cuda/cmd/serve`, `./gpu/cmd/serve`, `./metal/cmd/serve`)
  are a different module and build unaffected — verified after the change.
- **A prequantized int4 `.giw` bundle no longer mislabels itself as `int4mix`** on `/health`,
  `/v1/models`, and the startup line. `-quant int4` pins the embedding / LM head to int8 by
  default (logit-critical; `--embed-int4` opts out), and the label inference counted those tables
  — so an all-int4-projection bundle read as `int4mix` while the batched-prefill gate (which
  inspects only the seven projections) correctly batched it. The label now scans just the body
  matmuls the quant actually selects, matching the gate; the int8-pinned logit tables are
  excluded. Generation was always correct (the bundle is genuinely int4 — bit-identical logprobs
  to the GGUF int4 load); only the reported label and the KV-snapshot fingerprint were wrong.
  Direct (non-`.giw`) loads were unaffected — they report the requested quant string verbatim.
  (Reported as T1-6.)

## [v0.10.1] — 2026-08-07

### Added
- **Batched CUDA prefill for int8 weights (`int8`/`int8int8`), closing the "int4-only" TTFT gap.**
  A new `gemv_w8a8_batched.cu` (the per-row W8A8 GEMV plus a tiled multi-column loop) lets the
  dense resident prefill path batch int8 and int4mix bundles, not just int4. Because int8 is
  per-row symmetric — the dot accumulates in exact int32 and the scales apply once at the end —
  the batched result is **bit-identical** to the sequential per-token GEMV *by construction* (no
  reduction-order sensitivity, unlike the int4 lane's group-scaled float sum). Gated at three
  levels on both int4 and int8int8: kernel bit-identity at M ∈ {1,8,45,100}, prefill KV +
  logits bit-identical across all layers × all rows past a sliding window, and a 64-token greedy
  decode byte-identical. Measured TTFT (qwen2.5-coder-1.5b, RTX 2070 SUPER), sequential →
  batched: **128 tok 642→312 ms (2.06×), 512 tok 2.73→1.47 s (1.86×), 2048 tok 12.3→6.69 s
  (1.84×)**. The win is smaller than int4's (~4.4–5.7×) because int8 doubles the weight bytes per
  row on a bandwidth-bound kernel — but the ~9× sequential fallback the old int8int8 default hit
  is gone, and **every quantized mode now batches** (only native f32 stays sequential).

### Changed
- **BREAKING (behaviour): the default `--quant` moved from `int8int8` to `int4`.** int4 is
  smaller in RAM and, being the lightest quant on a bandwidth-bound kernel, the fastest to
  prefill and decode. **Output will differ from prior versions for anyone who relied on the
  default** — int4 is lossier than int8 (a decode/quality change, not just speed). Pass
  `--quant int8int8` (or `int8`/`int4mix`/`""`) to keep the old behaviour. `demo/chat` aligned
  to the same default; `demo/gemma` stays native-f32 (a faithful inspection CLI) and
  `cmd/prequant` stays `int8int8` (it bakes a `.giw` bundle that carries its own quant). The
  flag help now enumerates all five values with their accuracy/speed/RAM tradeoffs — including
  that **`--backend metal` requires `int8int8`** (int4 declines to CPU on the dense Metal
  resident path), and that all quantized modes get batched prefill while native f32 falls back to
  the sequential path. (At v0.10.0 batched prefill was int4-only, so this default flip also moved
  the out-of-box path off the ~9× slower sequential prefill; the int8 batched-prefill work above
  now removes that restriction entirely.)

### Fixed
- **The pre-v0.10.0 GPU build command no longer silently produces a CPU binary.** After M-19
  moved the backend imports to the submodule entrypoints, `go build -tags cuda|gpu|metal
  …/cmd/serve` (the *root* command in every ≤v0.9.x doc/note) kept exiting 0 while building an
  inert CPU binary — you only found out from a warning line at startup. The root `cmd/serve`,
  `demo/chat`, and `demo/gemma` now carry a `//go:build cuda || gpu || metal` guard that **fails
  the build** with a message naming the submodule entrypoint to use instead. (Reported by an
  external v0.10.0 trial, F1.)
- **The runtime "backend not registered" note pointed at the command that no-ops.** It said
  "build `-tags cuda`" — exactly what the user did — instead of naming the submodule entrypoint.
  Now: "build the submodule entrypoint (… `…/cuda/cmd/serve`) — not `-tags cuda` on the root".
- **README documents the GPU `serve` entrypoints, including the out-of-tree module-path form**
  (`go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve`) and an explicit
  upgrade note. Previously the only GPU example was the in-tree `cd cuda && … ./cmd/chat` (the
  REPL, and `cd cuda` presumes a checkout an out-of-tree consumer doesn't have). (F2.)

## [v0.10.0] — 2026-08-07

### Added
- **Resolved compute paths are reported, not inferred.** Both GPU fast paths — the resident
  decode runner and the batched prefill — decline *per call* and fall back correctly but
  silently. The sharp case: `--backend cuda --quant int8int8` on a dense model builds a full
  resident decode path (looks healthy, decodes at ~0.7× int4) and then takes the sequential
  per-token prefill on every prompt, because the batched GEMV is int4-only — **TTFT 1.73 s vs
  0.19 s on a 300-token prompt (9×), 4.56 vs 0.22 CPU-seconds (20×)**, with nothing logged.
  Now:
  - serve prints the **resolved** `decode path` / `prefill path` per model at load;
  - **`GET /health`** (new) carries `decode_path`, `prefill_batched`, `prefill_path`; the same
    three fields ride on each `GET /v1/models` entry as a **vendor extension** (unknown keys
    are ignored by the Go/Python/JS OpenAI clients, but `/health` has no schema contract to
    break for strictly-typed decoders);
  - **`--require-backend`** (new, opt-in) exits non-zero at startup on either decline, so a
    batch client fails at second zero instead of discovering a 9× under load;
  - `Model.ResidentDecline` names *which* residency decline applies — module not built in, no
    usable device, or ineligible arch — since the three need different fixes and are otherwise
    indistinguishable from outside.

  The prefill report shares `prefillStaticDecline` with `prefillCore` rather than restating
  its conditions, so the startup line cannot drift from the decline it describes.

### Changed
- **BREAKING (build): the opt-in GPU/CUDA/Metal builds moved to submodule entrypoints** (audit
  M-19). The pure-Go root module no longer imports any backend, so `go install
  …/goinfer/cmd/serve@latest` — and any SBOM/vuln scan of the root — no longer resolves
  `cogentcore/webgpu`, `ebitengine/purego`, `eitamring/gocudrv`, or `aikit/gpu`. The accelerated
  binaries now live in the submodules:
  - was `go run -tags cuda ./demo/chat` → now `cd cuda && go run -tags cuda ./cmd/chat`
    (likewise `./cmd/serve`); `gpu/cmd/*` under `-tags gpu`.
  - was `go run -tags metal ./demo/gemma` → now `cd metal && go run ./cmd/gemma` (the metal
    module is `darwin`-gated, so no `-tags metal`); `metal/cmd/{serve,chat}` too.
  The root `cmd/serve`, `demo/chat`, `demo/gemma` are unchanged as **pure-Go CPU** binaries. Serve
  logic moved to importable `internal/serveapp` / `internal/chatapp` / `internal/gemmaapp`.
- **BREAKING (API): `decoder.StreamTranscodeGGUF` gained a leading `ctx context.Context`** (audit
  M-21), so a minutes-long, multi-GB `.gguf`→`.giw` transcode is cancellable — a cancelled context
  aborts at the next layer boundary. `cmd/prequant` now aborts on Ctrl-C/SIGTERM; serve's admin
  model-load cancels on client disconnect.
- **BREAKING (API): `cuda.SpecStats` renamed to `cuda.GPUSpecStats`** (audit M-22) so the GPU
  batched-verify counters no longer share a name with `decoder.SpecStats` (CPU-spec counters); the
  two have distinct, non-1:1 field sets and now carry reciprocal doc comments.

### Fixed
- **The entire 2026-08-05 code audit is closed** ([`docs/completed/audit-2026-08-05.md`](docs/completed/audit-2026-08-05.md)):
  14/14 blockers, 31/31 criticals, 6/6 gates, 23/23 majors (+ the later latent M-25), 24/24 minors.
  Behaviour-affecting highlights:
  - **No silently-wrong output:** gemma4 session KV snapshots refuse a per-layer-KV geometry the
    format can't restore instead of mis-slicing on reuse (C-05); aborted Metal command buffers surface
    as an error instead of the host reading stale logits (C-09); the resident build declines odd
    (non-%8 / non-%32) GEMV/pack widths → CPU fallback rather than corrupt logits (C-10/C-11); the cuda
    f16 scale conversion is byte-canonical (C-15); group-limited MoE masks trailing experts to −inf
    (N-02); the vision projector errors when `mm_tokens_per_image` exceeds the patch grid, instead of
    emitting silent NaNs (N-19).
  - **API / serving:** `/v1/responses` reports `status:"incomplete"` (+ `incomplete_details`) when cut
    off by `max_output_tokens` (N-15); a chat request carrying both `tools` and an image now 400s
    instead of silently dropping the tools (N-16); the backend-fallback note goes to stderr, never
    contaminating a piped token stream (N-03).
  - **Hostile-input robustness:** the resident build declines a threadgroup-memory budget over the
    device limit (M-11); the GGUF metadata reader bounds a hostile `n_dims` / nested-array header into
    a typed error instead of a ~4e9-iteration spin or stack overflow (N-17/N-18); a corrupt tokenizer
    merges array is a typed error, not a misleading "decode-only vocab" (N-20).
  - **Concurrency:** resident `GenerateSpeculative` claims (C-03) and recurrent + multimodal
    cross-sequence state resets (C-01 / M-25) are regression-gated; the adapter registry and the webgpu
    fallback counter are race-clean (C-29 / N-06).
  - **Cancellation & perf:** multi-GB `.gguf`→`.giw` transcodes are cancellable (M-21, above); the
    ~152k-entry constraint token→bytes table is built once per model, not per request (N-14); the
    fused-decode rmsnorm dispatches one workgroup, not 64 (N-07).
  - Plus test-gate hardening (env/fixture independence — G-03/G-05, `GOINFER_MODELS_DIR` — G-06) and
    dependency hygiene (pure-Go root module graph — M-19, above; `go 1.26.5` in demo/agent — N-23).
- **Two independent post-audit reviews closed** ([`docs/completed/goinfer-post-audit-review.md`](docs/completed/goinfer-post-audit-review.md),
  [`docs/completed/goinfer-fresh-review-2026-08-07b.md`](docs/completed/goinfer-fresh-review-2026-08-07b.md)):
  30 findings (R-01…R-30) + 7 follow-ups (F-01…F-07) — residuals and regressions of the audit fixes,
  all fixed and device-verified (the CUDA leg on the RTX box). Behaviour-affecting highlights:
  - an adapter (LoRA) request against a resident base returned **base-model output at HTTP 200** — now
    routed down the adapter-applying session path (R-01);
  - a transient `.giw` read error on the Metal paged 26B decode could **crash the whole server mid-token**
    — now a failed request via the command-buffer-abort path (R-02);
  - a CUDA gemma-4 MoE accumulator zero-fill raced the previous layer's kernels (R-03);
  - the webgpu **MLA (DeepSeek/Kimi) residency path was silently disabled** by an over-broad head-dim
    guard introduced with the M-12 fix (R-05);
  - the resident context cap over-rejected vision and adapter prompts the CPU path serves (F-01);
  - plus crafted-`.giw` shape validation (R-07/F-02), JSON-error Go-type-name leaks (R-11/F-03), a
    defective FMA lint that couldn't see its own kernel (R-04), and several bounded GPU resource leaks
    (R-06/R-17).
- **Docs: batched-prefill coverage was stated per family without the int4-only caveat**, in
  both `CHANGELOG` and `docs/releases/v0.9.0.md`. A family inside the covered seven, loaded at
  `--quant int8int8`, also falls back to sequential prefill. Corrected in place.
- **`TestQwen2Moe_forwardParity` `Fatalf`'d on a partial fixture** — it stat'd
  `model.safetensors` but not `config.json`, so a half-present checkpoint (interrupted HF
  download) failed as if it were a numeric parity regression, which made
  `scripts/refresh_parity_hashes.sh` refuse a provably non-numeric refresh. It now skips
  unless every file `Load` needs is present.

### Known
- The int8 prefill fallback itself is **not yet fixed** — only made visible. The fix is scoped
  in `docs/ollama-chase.md` §C6: int8 is per-row symmetric, so an int32-accumulated batched
  W8A8 GEMM is bit-identical *by construction* (unlike the int4 lane, where group scales force
  a non-associative float sum). Speed is unmeasured and deliberately not predicted.
## [v0.9.2] — 2026-08-05

Docs only; no code, `go.mod`, or numerics change from v0.9.1.

### Fixed
- **README benchmark figures corrected.** The 1.5B 2048-context row was measured before
  split-KV attention landed and understated the result (~133 tok/s / 0.71× → **160.1 tok/s
  / 0.86×**); the table now carries the full four-depth curve (128 / 512 / 2048 / 3900) from
  `46829cc` and states the decode crossover at roughly 1000 tokens of context.
- **Two overstated claims corrected** — the feature taxonomy declines an architecture a
  backend can't fully run rather than making wrong output "structurally impossible", and the
  CUDA speed claim is stated against the measured curve. Docs only; no behaviour change.

## [v0.9.1] — 2026-08-05

Release plumbing only, no code or numerics change from v0.9.0: the final step of the
tri-module tag — the root module now requires the tagged `gpu` / `cuda` / `metal` v0.9.0
(previously Aug-2 pseudo-versions), closing the module requirement cycle so the opt-in GPU
backends are `go get`-able at a real version.

## [v0.9.0] — 2026-08-04

Theme: **two cgo-free GPU decode backends land as opt-in — CUDA (Linux/NVIDIA) and Metal
(macOS/Apple Silicon) — joined by speculative decoding, batched prefill, and long-context
decode, all under a bit-identity contract that makes every GPU execution strategy
byte-reproducible against every other.** *(Corrected 2026-08-11: this originally read
"byte-reproducible against the CPU reference", which overstates it. Bit-identity holds
GPU-vs-GPU — batched vs sequential prefill, graph replay vs live, cache on vs off, split-KV vs
full. Agreement with the CPU path is argmax-exact with a calibrated cosine, not byte-identical.
See the correction note in `docs/releases/v0.9.0.md`.)* Both backends are `CGO_ENABLED=0` (pure Go via purego/dlopen),
behind build tags + `--backend`, with graceful CPU fallback — so the default pure-Go build
is unchanged. **Pre-1.0:** the full per-family parity backfill remains the 1.0 gate;
families without both a T1 committed golden and a current T3 manifest row ship
*experimental* (`docs/parity-coverage-policy.md`). See `docs/completed/task-cuda-cgofree-spike.md`,
`docs/completed/task-metal-cgofree-spike.md`, `docs/completed/metal-verdict.md`, `docs/ollama-chase.md`, and
`docs/spec/` for the full arcs, scorecards, and dead ends.

> **Peer numbers re-anchored.** Every goinfer-vs-Ollama comparison in `docs/benchmarks.md`
> was re-measured against **current Ollama v0.32.5** (the prior figures used 0.5.7 / 0.32.0,
> 12–18 months stale). Lead with absolute tok/s + the cgo-free property, never a peer
> multiple — the ratio is size-, depth-, and platform-dependent (below).

### Added
- **cgo-free CUDA backend** (`cuda/`, `-tags cuda`, `--backend cuda`) — resident decode +
  prefill via **gocudrv** (dlopen `libcuda`) + **NVRTC** (PTX JIT), no cgo. Measured on an
  RTX 2070 SUPER (real q4_K_M): **1.5B 218.6 tok/s, 0.5B 507.5 tok/s** (goinfer absolute).
  Optimization arc: coalesced W4A8 GEMV + 2× ILP unroll (a dp4a-latency frontier, ~46% of
  peak), launch diet (18→13 dispatches/layer), f16 group scales, on-device greedy argmax
  (`ResidentGreedy`), pinned logits readback, and super-kernels **K1/K3a** (fold rmsnorm
  into the QKV / gate-up GEMV, +21.3% on 0.5B; K2 measured null and reverted). Qwen3 runs
  resident (QK-norm kernel). CI builds/vets/tests `-tags cuda`; the executor tax is 15.3
  µs/token (0.34%); the forward is guarded by a mutation-proven parity gate
  (`TestRealForwardParity`, `t.Fatalf` on any position past a 3% logit-range flip).
- **cgo-free Metal backend** (`metal/`, `-tags metal`, darwin-only, `--backend metal`) —
  resident decode + prefill via **purego-objc** (dlopen Metal.framework, `objc_msgSend`,
  MSL 3.1 compiled at runtime). Defused the `LC_BUILD_VERSION`/MSL-2.4 landmine
  (golang/go#77917; explicit MSL 3.1 + read-back assertion). Decode arc **~20 → 73.6 tok/s**
  (W4A8, best-of-40 warm); the practical batch-1 ceiling on M1 without DP4A. Server-to-server
  wall-clock vs Ollama-Metal v0.32.5 (`benchmarks.md` §B3): 0.5B **0.96×**, 1.5B **0.74×** —
  Metal's story is cgo-free / no-Xcode, not raw speed.
- **Speculative decoding** — opt-in, lossless. **CPU n-gram** (`--spec ngram`, prompt-lookup
  drafter over batched verify+rollback) wins on copy-heavy traffic (rag-copy 1.55×, agent-json
  1.59× on 0.5B; code-edit ~parity) — greedy bit-exact vs sequential (`TestNgramSpeculativeGreedyParity`,
  `TestResidentSpecServe`). **Grammar-fused** drafting for constrained/tool requests (forced
  DFA bytes drafted free, lossless, modest). **Resident CUDA (D1)** — batched M=k verify,
  **1.23×@128 / 1.86×@512 / 1.18×@2048**, byte-identical to plain greedy
  (`TestSpecDecodeCurve`). Recurrent families (Mamba-2 SSM, Gated DeltaNet) and the staged
  sliding-window ring are **guarded out** → plain decode (`TestSpecRollbackSafetyGuard`).
- **Batched prefill (CUDA), default-on, bit-identical** — weight-stationary batched W4A8
  GEMV ingests the whole prompt in one M=len pass. TTFT vs sequential: 128 **5.78×**, 2048
  **6.17×** (13.1 s → 2.1 s). Families joined bit-identically: llama, mistral, phi3, qwen2,
  qwen2_5_vl, qwen3 (batched qk-norm), gemma3 (batched sandwich norms) — 7 of 23
  (`TestPrefillCoverageAudit`). **W4A8 means int4 only**: a covered family loaded at
  `--quant int8int8` still takes the sequential prefill (9× TTFT) while building a full
  resident decode path. Reported at load — see [Unreleased]. The Metal MMA prefill path (3.7× TTFT) exists but **declines
  by default** (not bit-identical — 54% stream divergence; opt-in `GOINFER_METAL_BATCHED_PREFILL=1`).
- **Split-KV long-context decode (CUDA), default-on ≥256 ctx, bit-identical** — splits the
  independent key/value axes without touching the reduction (FA's online rescale is *not*
  bit-exact, so it is deliberately not used). 2048 decode **133→160 tok/s (1.20×)**; with the
  preceding attention coalescing, **99.5→160 = 1.61× over glue** (Campaign A, closed at the
  bit-identity ceiling). Gates `TestSplitKV_bitIdentical{,_gemma3}`; `GOINFER_SPLITKV_ATTN=0`
  disables.
- **Attention coalescing** — half4/float4-coalesced decode+prefill attention K-read.
  **Metal**: bit-identical, isolated kernel 1.79× @2048, end-to-end decode **1.37×@2048 /
  1.40×@4000** (the first Metal decode speed win, long-context). **CUDA prefill**: float4 QK,
  3.1× isolated, bit-identical.
- **Model-family coverage expanded.** Resident on **both** backends: Qwen2/Llama, **Qwen3**
  (QK-norm), **Gemma3** (sandwich norm / gated-GELU / embed-scale / per-layer RoPE), **MoE**
  (router + stacked int4 experts + shared expert; Mixtral / Qwen2-MoE / Qwen3-MoE / GLM-4.5/4.6).
  Metal-only: **Mistral** (sliding-window), **Phi-3** (partial rotary). **Cohere / Command-R
  v1 + v2** land CPU-correct (cosine 1.0 vs HF; new NormParallel / interleaved-RoPE / bias-free
  LayerNorm / per-layer-NoPE / reciprocal-LogitScale primitives; committed tiny fixtures —
  below). Every GPU backend auto-declines what it cannot run.
- **Shared feature taxonomy** (`decoder/features.go`) — admission is one subset check over
  arch-flag-derived requirements vs each backend's declared implemented set; a registry-driven
  gate (`TestResidentAdmission_registryCovered`) fails CI on any unclassified arch. Eliminates
  the silent-wrong-output class (a family the backend can't fully run now declines to CPU
  instead of running with a feature dropped). Metal + CUDA + WebGPU all use it.
- **Gemma-4 26B-A4B fully GPU-resident via host↔VRAM expert paging (CUDA)** — decodes a
  ~15 GB MoE on an 8 GB card at **16.98 tok/s** by streaming routed experts host→VRAM into
  an LRU slot cache (81.6% hit at 30% residency; auto-caps slots to free VRAM, never OOMs).
  The bottleneck is **capacity, not kernels** — a model that fits would decode *faster* than a
  dense 7B. Metal MoE expert paging primitives (slot-pool LRU, pread staging default-on,
  MTLResidencySet-pinned pool) land for the same path on Apple Silicon.
- **Bit-identity gate suite** — the **Metal snapshot golden** (`TestMetalSnapshotGolden`):
  a machine+OS-pinned sha256 of Metal's own logits at fixed inputs, decoded past both
  reduction widths through two committed tiny models — the absolute stored reference that
  catches changes a self-consistent gate (paged≡non-paged, GPU-vs-CPU tolerance) is blind to
  (width sweeps, fused rewrites, accumulation-order moves). Reduction widths pinned as
  bit-identity constants (`tgReduce*`). CUDA carries `TestKernelFMALint` (build fails on any
  bare float MAC).
- **Committed tiny parity fixtures** — `testdata/cohere-tiny/` + `cohere2-tiny/` (656 KB each,
  deterministic random-weight, `scripts/pin_cohere*.py`) are committed via targeted `.gitignore`
  exceptions so the Cohere/Command-R forward-parity gates run in CI on every push, not only
  where someone regenerated the asset (same rationale as `mixtral-tiny`). The committed set is
  now enumerated as chosen policy in `docs/parity-coverage-policy.md`.
- **Test census tool** (`scripts/skip_census.py`) — the release ritual: runs `go test -json`,
  reports PASS/SKIP/FAIL with every skip bucketed by *why* (missing-fixture / missing-golden /
  no-gpu-device / heavy-model / integration-env), so a run that skipped 200 asset-gated tests
  can't be mistaken for one that exercised them. `GOINFER_REQUIRE_FIXTURES=1` turns any
  committed-fixture skip into a hard failure. It also flags native GPU-contention crashes
  (`fault 0x10`) as spurious rather than real.
- **MXFP4 / gpt-oss (CPU)** — bit-exact MXFP4 unpacker vs GGUF on real gpt-oss:20b, CPU
  forward + loader + parity; CUDA/Metal decline gpt-oss at load → CPU (never mis-run).

### Changed
- **Ollama peer re-anchored to v0.32.5** across `benchmarks.md` §B2/§B3/§B4. §B2 (CUDA): 0.5B
  still ~1.7× ahead, 1.5B short-ctx parity (~1.19×), goinfer behind at 2048 ctx and on prefill.
  §B3 (Metal): 0.96× / 0.74× (0.5B / 1.5B). "Peer version is part of the measurement" is now a
  standing method rule.
- **§B4 retracted** — the "peers fail to load the 26B" claim was false and is withdrawn: Ollama
  v0.32.5 loads and runs Gemma-4 26B-A4B via a 42%-GPU/58%-CPU split at ~24.5 tok/s (faster
  than goinfer here). The honest, narrower claim: goinfer runs it *fully GPU-resident* — an
  architecture difference, not a capability peers lack.
- **Batched prefill default flipped to ON (CUDA)** after the contraction fix made it
  bit-identical; **declined by default on Metal** (not bit-identical). Split-KV decode
  attention default-on at ctx ≥ 256.
- **Fast-math stays the Metal default**; `GOINFER_PRECISE_MATH=1` is an opt-in (measured 4–7%
  slower, no CPU-parity gain — the int8→int4 requant is the parity gap, not fast-math). Robustness
  against compiler/OS drift is carried by the snapshot golden, not by strict math.
- **aikit `gpu` module bumped to v0.25.2** — `CompileLibraryPrecise` now self-verifies
  (`setMathMode:MTLMathModeSafe` → `setFastMathEnabled:NO` fallback, reads state back and errors
  if neither selector responds), ViT reduction width pinned.

### Fixed
- **Batched-prefill / spec-decode were not bit-identical to sequential decode (CUDA) — 84%
  token-stream divergence on real weights** (invisible on uniform fixtures). Root cause:
  compiler **FMA contraction** (`facc += p*s` = mul+add / 2 roundings in decode vs fma / 1
  rounding in the batched kernel), ~1 ULP, data-dependent. Fix: explicit **`__fmaf_rn`** in
  both paths → `TestPrefillDivergenceRate` 0/50, decode speed unchanged; this is what unblocked
  lossless spec-decode. Durable guard: `TestKernelFMALint`.
- **As-cap overflow (Qwen3/Mistral):** `head_dim` independent of `hidden` (`nH·hd ≠ H`)
  overflowed a hardcoded activation-staging buffer, which would have silently broken every real
  Qwen3/Mistral size. Fixed on Metal with dynamic threadgroup memory (bit-exact to K=4096);
  CUDA audited immune.
- **Phi-3 dropped `sliding_window` on every backend (CPU included)** — `phi3Architecture` never
  set it, so Phi-3 ran full attention and diverged from HF past 2047 tokens. Now honored;
  Metal windowing works as a result.
- **`lmHeadN` dropped `LogitScale` on the batched-forward path** — surfaced by Cohere (the first
  forwardN-eligible logit-scale family); also closed 3 gemma4_text merge gaps.
- **Metal Gemma correctness:** GELU-tanh argument clamp (tanh overflow → NaN on the sink's large
  gate); prefill LM head pinned to the int8 path (was NaN on the f16-MMA path).
- **Security hardening (all shipped):** resident-KV context-cap DoS / OOB KV writes (C3, CUDA +
  Metal), speculative-rollback sliding-window leak (C1), PrefillLast unified-memory leak (C5),
  discarded CUDA launch errors + mixed-quant fuseQKV guard (M23), prompt-injection via
  special-token surface forms (M25), GGUF/VL untrusted-input hardening.

### Findings (no API change)
- **Metal decode is dispatch-/issue-bound, and the long-context gap is structural.** The
  short-context GEMV is already at the peer's effective-bandwidth band (no DP4A ⇒ integer decode
  is issue-bound). The long-context deficit (~4.2×@4000 vs Ollama's flash attention) is priced
  out by the bit-identity contract — four independent bit-identical dedup attempts (split-KV,
  two grouped, staged) were built and lost. **Decision (M1 = M-A):** stay bit-identical, accept
  the depth floor; an opt-in FA-style throughput mode (M-B) is scoped and deferred, not rejected
  (`docs/completed/metal-verdict.md`).
- **CUDA graphs are a measured null on real models** (1.01×; the ~1.4–1.7× was a
  tiny-model/dispatch-dominated artifact). Only a safe-gate shipped
  (`GOINFER_CUDA_GRAPHS=1`, promotes to live under EXCLUSIVE_PROCESS/MPS + a startup
  bit-exactness self-test); default declines.
- **Rotation + per-row scales + IMMA (tensor cores) deferred** — the cheap path (per-row scale
  search) is measured dead (per-row perplexity 108 vs per-group 28.5; the 1.24× weight-space
  error compounds ~4× at the output), and the expensive path buys only a prefill-only ~3× on an
  already-past-threshold TTFT. Format stays group-scaled int4 (`docs/completed/task-rotation-perrow-imma.md`,
  `ollama-chase.md` §7).
- **EAGLE-3 and Stage-B GPU verify are built-and-parked** — EAGLE-3 is lossless but a CPU
  wall-clock loss (needs GPU); Stage-B M=k GEMM verify is a NO-GO on small models, conditional-GO
  only for ~70B-class + short linear drafts. Both wait on GPU throughput they don't yet have
  (`docs/spec/`).

## [v0.8.0] — 2026-06-20

Theme: **the GPU resident-decode path expands from "dense Qwen2/Llama only" to most
families served, and gains a Mamba-2 SSM engine for hybrids.** (See
`docs/completed/decode-residency-campaign.md` for the full arc, scorecard, and dead ends.)

### Added
- **Resident decode for most mainstream families** (the C-lever ladder — bounded
  eligibility widenings, existing kernels reused). MoE: Mixtral / qwen2_moe / **GLM-4.5/4.6**
  (partial-RoPE) / **DeepSeek-V2/V3 + Kimi-K2** via a new **MLA latent-attention** residency
  bridge (rank-space attention + latent KV store + per-head absorb/lift kernels). **Mistral**
  (sliding-window residency) and **Mellum** (per-layer-RoPE residency). Most served models now
  decode pure-GPU instead of staged (~3× the per-token speed where it applies).
- **Resident Mamba-2 SSM decode engine** for hybrid families — the reframe that *decode is a
  bounded per-token recurrence, not the prefill scan*, so Mamba state slots onto the
  `DecodeRunner` like a KV cache (`mambaConv`/`mambaSSM`/`mambaGatedNorm` kernels, build-once
  persistent {conv-ring, ssm} state, drift-gated to 2k tokens).
- **Nemotron-H resident, DEFAULT-on at int4** — the dense squared-ReLU hybrid (Mamba-2 /
  NoPE-GQA / non-gated relu² MLP, single-op-per-block + a `relu2Quant` kernel). Near-lossless
  (perplexity 1.677 vs f32 1.695, KL 0.058; the ~7.5% greedy disagreements are 100% benign —
  99.6% top-2 agreement, every divergence at a near-tied position), ~13× the f32 CPU path.
  Guarded: default-on only at int4; int8 opt-in behind `GOINFER_SSM_RESIDENT`.
- **Granite-4.0-H resident** (Mamba-2 + attention + MoE-every-layer + the 4 Granite multipliers),
  **opt-in** — a 10× greedy speedup, but int8-quality-limited (below).
- **Decode kernel fusion** — fused q-rope + k-rope-store + v-store into one dispatch (+1.5%);
  q/k/v bias folded into the GEMV epilogue for bias models (+2.3% real Qwen2.5).
- **Mellum2 fast-load (and any safetensors MoE): prequant → int4 `.giw` + direct upload.**
  `cmd/prequant` now accepts a safetensors **directory** (not just a GGUF), producing a
  resident-loadable int4 bundle (`decoder.Load` + `SerializeWeightsTo`; tokenizer carried as the
  dir's `tokenizer.json`, loaded via `tokenizer.LoadJSONBytes`). And the resident int4 upload
  skips the per-element unpack + `packNibbles` repack: the decoder's int4 storage is
  byte-identical to the GPU packed layout for K%32==0 (`TestInt4LayoutMatch`), so it
  `CreateBufferInit`s the decoder bytes straight. Mellum2's 12B int4 resident load **~66 s →
  ~13 s warm** (the bundle skips the requant; the direct upload skips the repack); benefits every
  int4 resident load, `.giw` and direct alike. Gate: `gpu.TestGIWInt4_resident`.

### Changed
- The mmap/madvise/span-residency weight substrate moved to `aikit/mmap` (shared primitive).
- **Parity manifest: `serialize.go` split into its own `serialize` deps set** — serialize-only
  edits (a new `.giw` quant layout, the safetensors dir-input path) no longer re-stale every
  forward-parity row; the one place serialize affects numerics (the int4 deserialize→resident
  seam) keeps its own gate (`TestGIWInt4_resident`). Policy: a bare `deps_hash` re-hash of a
  validated family that changed a forward/core file is now forbidden (`parity-coverage-policy.md`).
- **`Model.Quant()` reports `int4mix` for a mixed bundle** (int4 experts + an int8-kept
  embedding/LM head) instead of mislabeling it `int8` — it scans all weight kinds. The `.giw`
  format tag (`quantMode`) is untouched, so existing bundles still load. (Heads-up: this changes
  the `.giw-kv` warm-snapshot fingerprint for mixed bundles → one-time cold prefill on upgrade.)
- **wgpu-native's benign warnings silenced** — `gpu.New()` sets `LogLevelError`, so "No
  windowing system present / No config found" no longer spams stderr on headless GPU runs.

### Fixed
- **GGUF→`.giw` dropped `rope_parameters` for MLA/YaRN families.** A nil `json.RawMessage` config
  field marshals to the literal `null` (4 bytes), which on the `.giw` round-trip
  (`json.Marshal(Cfg)` → reload) re-fired an "is it present?" check on an absent field — so a
  GGUF DeepSeek bundle failed self-check with "rope_parameters: rope_theta must be >0". `,omitempty`
  on the optional Config RawMessage fields keeps nil truly absent (`TestConfig_giwRoundTrip_nilRawMessage`).

### Findings (no API change)
- **qwen3_5_moe numbers shifted slightly under the forward refactors — re-validated, still
  gate-backed.** The v0.8.0 §1 re-run at HEAD (vs the `e3eb033` baseline) moved its int8-vs-bf16
  Gate-2 from argmax 74/80, cosine_min 0.99466 → **66/80, 0.99333**. Benign: the mean barely
  moved (0.99837), and *every* argmax divergence is a proven near-tie (worst gap 0.0045 of
  range), so the forward is sound — the flips are coin-flips on tied tokens, amplified by the
  `forward_qwen35.go` refactors (MoE demand-paging / WeightMat migration / scratch-reuse). The
  bf16 Gate-2's enforced bar is **cosine_min ≥ 0.98 + all-near-tie**, both cleared; 0.99466 was
  the *achieved* number that run, not a floor (the 66/80 + 0.9943 figure is the *separate* Q8_0
  GGUF gate's relaxed bar). deepseek_v2/v3 re-ran clean to argmax 100% / cosine 0.999, confirming
  the re-validation discriminates.
- **Granite int8 resident is quality-limited and stays opt-in/greedy-only** — characterized as a
  *fundamental* cliff (not a bug): its 64-expert top-6 MoE router turns chaotic f32-reduction-order
  perturbations into discrete expert-selection flips. Proven precision-invariant (int8 ≈ f16 ≈
  W8A16) and NOT a GPU-kernel bug (the SSM kernels are bit-correct). Nemotron-H, having no router,
  does not hit this — which is why it's default-on. Full write-up: `docs/ssm-int8-quality.md`.
- **No "wgpu-native v29 decode penalty"** — measured ≈ cogentcore/v22 (gemv + per-dispatch record);
  the real binding blocker is the go-webgpu *goffi* (zero-CGO) Go-1.26 crash, not v29. Staying on
  `cogentcore/webgpu`. (`docs/completed/gpu-gowebgpu-migration-assessment.md`.)

## [v0.7.0] — 2026-06-15

### Added
- **Qwen2.5-VL — a second vision-language family (image→text, pure Go).** goinfer's
  vision path now generalizes beyond Gemma 3 / SigLIP. The whole pipeline is gated
  against HuggingFace: HF-exact preprocessing (smart-resize + a PIL-bicubic port +
  the spatial-merge patchify), the aikit Qwen2.5-VL ViT + patch merger (dynamic
  resolution), **m-RoPE** (3D temporal/height/width positions — `applyMRoPE` + a port
  of `get_rope_index`, prefill *and* decode), and the decoder image path (merged-
  feature injection, causal attention). `serve` accepts image requests for a
  Qwen2.5-VL `--model` (OpenAI + Anthropic), validated end-to-end on the real
  Qwen2.5-VL-3B. Text decode for every other model stays byte-identical (m-RoPE
  reduces to scalar RoPE when the three position components are equal). New API:
  `decoder.GenerateQwenVL`; `cmd/serve` family auto-detection + routing.
- **Compute-time multi-adapter LoRA (`serve --adapter <name>=<base>=<dir>`) — N
  fine-tunes on one resident base.** Instead of merging a LoRA into the weights (a
  full base copy per adapter), the low-rank `A`/`B` are applied in the forward
  (`Y = W·x + s·B(A·x)`), so N adapters of one base cost ≈ base + N small deltas — the
  multi-tenant footprint win. Each adapter is a served model that shares the base's
  resident weights but keeps its own KV sessions; requests route via the OpenAI
  `model` field. Merged `--lora` stays the faster single-fine-tune default. Safetensors
  base, dense gated-MLP archs; an active adapter takes the sequential prefill path.
- **`--quant int4mix`: per-tensor mixed precision (idea #5).** A calibration spike
  found the int4→int8 quality loss is concentrated in **attention** (promoting
  `attn_output` alone recovered >half the gap), while the **FFN bulk is int4-tolerant**
  — and attention is the *cheaper* tensors. So int4mix keeps attention (q/k/v/o) +
  embed/head at int8 and the FFN (gate/up/down/experts) at int4: **near-int8 quality
  below int8 RAM** (≈0.5–0.8× int8, model-dependent on the FFN ratio). It's a
  load-time policy keyed on llama.cpp tensor names (`matmulQuant`); the resident
  weights and `.giw` carry the resolved per-tensor int8/int4 kinds (the format already
  stores per-`weightMat` kind, so a mixed model round-trips). No new kernels, no format
  change, zero decode cost. GGUF load path only. `Model.Quant()` now reports the
  requested quant for direct loads so int4/int4mix/int8 don't collide in the KV
  fingerprint. Gated by `TestInt4MixMode`.
- **Per-model `serve` overrides — a heterogeneous model zoo.** `--model` now takes
  comma-separated per-model overrides of the server-global defaults:
  `--model big=moe.giw,stream,weight-cache=16,quant=int4 --model fast=small.giw`
  streams the big MoE while the small one stays resident — the case the old flat
  global config (the `// per-model overrides are a follow-on` TODO) couldn't serve.
  Keys: `quant,lora,kv,kv-quant,stream,weight-cache,embed-int4`; overrides are
  pointer-typed so "inherit the default" stays distinct from a real `""` (f32).
  Backward compatible — no comma suffix inherits the globals. Paths may not contain
  commas.
- **`decoder.Options.Validate()`** checks the stringly-typed knobs (`Quant`,
  `Backend`, `KVPrecision`, `KVQuant`) against their allowed values, so an invalid
  enum (e.g. `-kv-quant=int8` instead of `i8`) is a clear load-time error instead of
  a silent fall-through to the default. serve calls it once per resolved model.
- **`--embed-int4`: relax the int8 embed/head pin to int4 (idea #3, opt-in lossy).**
  In int4 mode the token-embedding / LM-head table is pinned to int8 because it's
  logit-critical; for a big-vocab small model that pinned table is the single largest
  resident tensor. `--embed-int4` (decoder `Options.EmbedInt4`) stores it at int4 too,
  halving it, for ~2.3 pts top-1 (a 1.5B Q4_K_M spike — ≈0 on frequent tokens, ~3 on
  rare). Default off keeps the bit-exact int8 pin. GGUF direct-load path only (not the
  `--stream-weights` `.giw` cache — prequant with the knob to bake it). The doc's
  stronger "row-blocking" variant (b) was spiked and shelved: the int4 damage is
  entirely on tail rows, which tiering keeps at int4, so it's dominated by full-int4.
- **`serve --stream-weights` now works on a plain `.gguf` (transparent `.giw` cache,
  idea #1 "D").** Streaming needs the read-only mmap that only `.giw` provides; rather
  than make users run `prequant` by hand, serve now transcodes a `.gguf` to a sidecar
  `<model>.<quant>.giw` once (streamed, ~model-size peak RAM — no OOM) and loads that,
  reusing it on later runs and rebuilding it if the `.gguf` changes. The one-time
  transcode is logged. No per-token cost (resident bytes stay the dequant-once
  int8/int4); it's the convenience floor of #1. The transcode core moved to
  `internal/prequant` (shared by `cmd/prequant` and serve).
- **Dense weight streaming (`serve --stream-weights` on a dense `.giw`) — run a
  model bigger than RAM.** The companion to MoE expert paging for dense models
  (Llama / Qwen2 / Qwen3 / Mistral): because the transformer layer loop is
  sequential and known in advance, a sliding-window pager prefetches the next
  layer's weights (`MADV_WILLNEED`) while the current layer computes — overlapping
  the fault — and releases the layer that slides out the back (`MADV_DONTNEED`).
  Resident weight RAM is bounded to **floor + window** (sized by `--weight-cache`)
  instead of the whole model, so a model too big for RAM still runs (floored by
  NVMe bandwidth). The floor is non-zero: only the per-layer projections stream;
  embed / final-norm / LM-head stay resident (multi-GB for big-vocab models — the
  complementary lever is sub-int8 embed/head). Bit-exact by the same read-only-re-fault property as expert
  paging — validated byte-identical over a decode with a 3-layer window evicting
  and re-faulting most layers every token. Same `--stream-weights` flag, which now
  picks expert paging for MoE and layer streaming for dense; no-op when the model
  fits the budget.
- **`.giw` now round-trips the `qwen3_5_moe` DeltaNet-hybrid family (format v2).**
  The prequant `.giw` serializer dropped the hybrid's per-layer `delta` (Gated
  DeltaNet) and `qattn` (gated-softmax) tensor sets, so a `.giw`-loaded 35B-A3B
  segfaulted on the first forward (nil delta). v2 appends a one-byte-per-layer
  hybrid tail carrying those f32 tensors; v1 blobs are rejected by the version
  guard and rebuilt from the GGUF (a `.giw` is a regenerable cache). This is what
  lets MoE expert demand-paging (below) actually run its headline 35B-A3B — now
  validated end-to-end: a 512 MB expert cache against 16 GB of experts evicted and
  re-faulted 5k+ in-use experts over a decode with byte-identical output.
- **Streaming `.giw` serialization (`prequant` no longer OOMs on big models).**
  `decoder.SerializeWeightsTo(io.Writer)` + `giw.WriteStream` write the bundle
  straight to disk with a running CRC and a seek-back length patch, so peak RAM is
  ~the resident weight size instead of 2×+ (resident + full blob + bundle copy). A
  35B int4 now prequantizes at ~20 GB peak instead of thrashing into swap; the
  streamed bytes are byte-identical to the in-memory path.
- **MoE expert demand-paging (`serve --stream-weights` + `--weight-cache <GB>`) —
  run a big MoE on less RAM.** A sparse MoE (e.g. Qwen3-A3B class) resides at tens
  of GB of experts but activates only K·L per token. With `--stream-weights`, a
  `.giw` MoE model keeps its experts in the (now mmap'd, idea-Inc-1) read-only
  mapping and pages them on demand: the router's top-k selection drives an LRU
  bounded by `--weight-cache` GB (0 = auto, ~½ available RAM), releasing the tail
  with `MADV_DONTNEED` and faulting misses with `MADV_WILLNEED`. **Bit-exact** —
  the mapping is read-only and file-backed, so an evicted-then-reused expert simply
  re-faults from disk (identical bytes); the cost is the cold-miss fault, ~+24 ms/
  token at a 16 GB budget on a measured 35B-A3B (≈2× RAM reduction; see the spike
  in `decoder/moepaging_spike_test.go`). Opt-in, CPU `.giw` MoE only; a no-op for
  non-MoE / non-`.giw` / sub-page-expert models. Page-granular eviction, so it
  caps RAM on Linux firmly and best-effort on other unixes.
- **`.giw` weights are now mmap'd (pageable residency).** The prequant `.giw` fast
  path already aliased its int8/int4 weights with no per-tensor copy, but read the
  bundle with `os.ReadFile` (heap, not pageable). It now maps the file read-only
  (`MAP_PRIVATE`), so the aliased weights are views into the OS page cache —
  faulted in lazily, evictable, and shared across processes mapping the same file.
  Bit-exact and lower load time; released by `Model.Close`. The substrate the
  expert paging above (and future weight streaming) builds on. Windows falls back
  to the prior heap read.
- **Tiered KV cache (`serve --kv-idle-demote`) — demote idle warm sessions RAM →
  NVMe.** `--kv-sessions` pins RAM for every warm conversation; tiered KV adds a
  policy over the existing `--session-dir` `.giw-kv` persistence so a small-RAM box
  can hold many intermittent chats. A session untouched for `--kv-idle-demote`
  (e.g. `10m`) is snapshotted to disk and its RAM freed by a background sweep;
  capacity evictions tier the coldest session to disk instead of discarding it; the
  next request whose prompt extends a demoted session faults it back transparently.
  The fault-back continuation is **byte-identical** to a cold prefill (exact
  `.giw-kv` restore). The on-disk tier is bounded by `--kv-demoted-max` (default 64)
  and is in-process scratch (the resident tier is what survives a restart). Off by
  default; needs `--session-dir` and `--kv-sessions > 0`. Pure serve-layer policy —
  no decoder changes.

### Changed
- **aikit `v1.7.3` → `v1.8.1`.** Adds the pure-Go Qwen2.5-VL vision tower
  (`vision.LoadQwenVisionEncoder` / `Forward` / `ForwardViT`) — RMSNorm ViT, windowed
  + full attention, 2D rotary, the spatial-merge patch merger — parity-gated against
  HuggingFace. Bumped in both the root and `gpu` modules.

### Fixed
- **darwin (Apple-silicon) build break.** The MoE-paging madvise helper called
  `syscall.Madvise` under `//go:build unix`, but darwin has no `syscall.Madvise`
  (it lives in `golang.org/x/sys/unix`), so the root module — and everything
  downstream — failed to compile on macOS. Split per-platform: Linux/BSD keep
  `syscall` + `MADV_DONTNEED` (firm RAM cap, unchanged); darwin uses `x/sys/unix`
  for the `MADV_WILLNEED` prefetch and no-ops eviction (no macOS syscall reclaims a
  read-only file-backed mapping, so the `--weight-cache` cap is best-effort on
  darwin — the OS reclaims the clean, re-faultable pages under memory pressure).
  Adds `golang.org/x/sys` as a direct root dependency. (CI now runs darwin/arm64
  jobs to catch this class of platform gap.)

## [v0.6.0] — 2026-06-13

### Added
- **GPU int8 KV cache (`--kv i8`, opt-in `-tags gpu`) — 4× vs f32, ~64k context on
  8 GB.** The full-residency decode path (dense Qwen2/Llama) gains an int8 KV cache
  alongside f32 (16k ctx) and f16 (32k): per-(position, KV-head) symmetric int8,
  written and read by on-device kernels (RoPE-store / V-store quantize per head;
  attention unpacks ×scale), so decode and prefill stay on the GPU. ~6.9 GiB peak for
  a 7B int4 model + 64k KV. Lossy but argmax-faithful (≥0.99 cosine vs the f32 cache).
  The f32 and f16 KV paths are unchanged. Selected by `serve --kv i8`.

### Changed
- **aikit dependency consolidated and bumped `v1.3.0` → `v1.7.3`.** Three surfaces
  goinfer had open-coded moved onto aikit's shared types — pure de-duplication, no
  behaviour change: (1) the SigLIP vision encoder now lives in `aikit/vision` (goinfer
  keeps only the Gemma projector + soft-token glue, as package `multimodal`); (2) the
  shape-checked safetensors weight reads collapse onto
  `embed.SafetensorsFile.TensorF32` / `TensorI32`; (3) the decoder's three-precision
  quantized-weight matrix (f32 / per-row int8 / group-wise int4) is now
  `linalg.WeightMat`, with goinfer retaining only its matmul backend-routing policy —
  the `.giw` zero-copy load is preserved via aikit's `WrapInt8` / `WrapInt4`. 1.7.3
  also makes `linalg.MatmulBT` M-invariant (a row computed alone is bit-identical to
  the same row inside a batch) and fixes an amd64-only AVX2 reduction bug on
  odd-length-`K` shapes.

### Fixed
- **`--backend webgpu` falls back to CPU when the GPU device fails to initialize,
  instead of panicking.** The webgpu backend factory returned a typed-nil
  `*webgpuBackend` on adapter/device failure, which auto-converts to a *non-nil*
  `Backend` interface — so `Model.withResidency` saw a "real" backend, type-asserted
  it, and dereferenced a nil receiver (`BuildResident` → nil mutex). The factory now
  returns a literal `nil` interface on error, so a headless box (or GPU exhaustion)
  cleanly uses the CPU path with a note, as intended. `-tags gpu` only.
- **Same-model speculative decoding is now bit-exact with plain greedy.** Dense decode
  computed attention with a scalar kernel that was not bit-identical to the batched
  kernel the speculative *verify* pass uses (cosine ≥0.99, not exact), so the target
  rejected ~11% of its own draft tokens (acceptance ~0.89) and the streamed output
  could drift from greedy. Decode now runs the same batched attention as
  prefill/verify, with f64 accumulation, so decode == prefill == verify exactly —
  speculative output is token-identical to greedy and acceptance is ~1.0. This also
  removes the decode↔prefill numerics seam for all dense models; combined with aikit
  1.7.3's M-invariant `MatmulBT` it holds across f32/int8/int4. (int4 greedy decode
  now tracks the higher-precision reference one token further as a side effect.)

### Added
- **Multimodal vision input (Gemma 3 VL) — image→text end to end, pure Go.** An
  image now flows through `vision` (preprocess → SigLIP encoder → projector) into
  the text decoder's embed-by-vector seam (`decoder.GenerateVL`), and the serve
  surface accepts it on **both** the OpenAI `/v1/chat/completions` (`image_url`
  content parts) and Anthropic `/v1/messages` (`image` blocks) APIs —
  **base64/data-URI only** (a remote URL is never fetched; SSRF guard). Start with
  `--vision <dir>` (auto-discovered when the `--model` dir is a Gemma 3 VL
  checkpoint). Image tokens (256/image) count in `usage`. `demo/agent` (the web UI)
  also gains image input: drop, paste, or pick an image and the agent answers it
  via the same path. Numerics are HF-parity-gated (encoder/projector/end-to-end
  goldens). Loading a real `google/gemma-3-4b-it` now works directly (sharded +
  prefixed VL safetensors, nested `vision_config`, and the Gemma3-text class
  defaults the minimal `text_config` omits). **Caveat (CPU):** the SigLIP prefill
  is heavy on CPU (~171 s/image at 896², 4096 patches) — correct but slow. The GPU
  encoder below is the fix.
- **Resident GPU SigLIP encoder (`-tags gpu`, opt-in) — ~9× faster image
  prefill.** `--backend webgpu` runs the whole vision tower on the GPU: the tower
  uploads once, the `[4096, hidden]` activation stays on-device through all 27
  layers (patch-embed → LayerNorm → int8 qkv → bidirectional attention → gelu-tanh
  MLP → residuals → final LN), and there is one readback — paying WebGPU's
  submit/sync cost ~27× instead of the ~162× a per-op offload would (a measured
  dead end). On an RTX 2070 SUPER: **gemma-3-4b-it image prefill 171 s → 18.8 s**,
  parity **cosine 1.000000** vs the CPU W8A8 encoder (0.999959 vs the HF golden).
  New WGSL kernels (batched LayerNorm, gelu-tanh, row softmax, bias/residual
  add, per-head gather/scatter) join the existing tiled W8A8 GEMM. The pure-Go
  default build is untouched (`vision.Encoder` delegates to the device only when a
  resident backend is attached). `demo/agent` (the web UI) gets it too via
  `--vision-backend webgpu` — dropped/pasted image captions go ~9× faster. The
  attention matmuls still use a naive f32 kernel — a tiled GEMM there is the next
  lever toward the ~8–12 s estimate (`docs/completed/task-gpu-vision-tower.md`).
- **SigLIP attention vectorized** — QKᵀ/scores·V moved onto the SIMD A·Bᵀ kernels
  (QKᵀ f64-accumulating for parity), >2× faster vision prefill (>400 s → ~190 s),
  bit-faithful (encoder golden cosine 1.0).

## [v0.5.0] — 2026-06-11

_A **minor bump** per SemVer: `constrain` schema validation carries a
user-visible behavior change (under Changed) — shipped alongside the Anthropic
Messages API (Claude Code can point at a pure-Go runtime), the f16 GPU KV cache,
the ~2.4× sparse-MoE / ~3.4× dense prefill speedups, and the now soak-verified
untrusted-input fuzzing hardening._

### Changed
- **`constrain` rejects unenforceable / unsatisfiable JSON Schemas at compile**
  instead of silently accepting them. Schemas that previously compiled but could
  never be enforced now return an error: a `required` entry naming a property
  absent from `properties`, an array with `maxItems` < `minItems`, and
  negative / non-integer array bounds. **Behavior change** — a caller that relied
  on the old silent-accept (and the non-conforming output it produced) now gets a
  compile error instead. (`3cb27e6`)
- **aikit `v1.1.1` → `v1.3.0`** — a hardened `embed.parseGGUF` (overflow-safe
  length checks, map/array pre-sizing capped to the remaining input, so a hostile
  GGUF header errors instead of panicking / OOM-ing one import down) and the
  scores·V attention vectorization (`v1.2.1`), plus `MatmulBTAcc64` — an
  f64-accumulating A·Bᵀ matmul (bit-identical to a sequential-order f64 reduction)
  that the MoE attention path needs (`v1.3.0`). Bumped in both the root and `gpu`
  modules. (`b77922c`)
- **MoE attention now runs through the f64-accumulating matmul** (decode AND
  prefill). It is more accurate than — and may differ at routing near-ties from —
  the prior f32/scalar path, but stays parity-gated against the HF bf16 oracle
  (Mellum2 logit/window parity unchanged). The discrete top-k router amplifies any
  attention reassociation, so the precision is load-bearing, not cosmetic.

### Added
- **Anthropic Messages API** (`POST /v1/messages`, `POST /v1/messages/count_tokens`)
  alongside the OpenAI surface — point **Claude Code** (or any tool that speaks
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_MODEL`) at a pure-Go
  single-binary runtime. The second de-facto-standard chat surface, served by the
  same edge-translation trick llama.cpp/Ollama/LM Studio use: the request is
  mapped into the existing internal path (system + turns + sampling + tools) and
  the result mapped back out — `drive`/`prepare` unchanged. Honors `system`
  (string or text-block array), content blocks (`text`, plus `tool_use`/
  `tool_result` conversation replay), `tools` (`input_schema`) and `tool_choice`
  (`auto`/`any`/`tool` — `any`/`tool` reuse the constrained-decoding path, so a
  malformed tool call is physically impossible), `stop_sequences`, and the named-
  event streaming SSE protocol (`message_start` → `content_block_*` →
  `message_delta` → `message_stop`, no `[DONE]`). Compatible-not-full-spec (the
  llama.cpp bar): image blocks 400; `thinking` / `cache_control` / `metadata`
  accepted and ignored (Claude Code sends `cache_control` on every request).
  Pure stdlib `net/http`, no new deps. (`d0b0f66`)
- **`demo/agent`: a fully-local stdlib RAG coding agent** over goinfer + the
  `ken` MCP retrieval server — a CLI and a single-binary web UI, kept in its own
  module so its MCP dependency stays out of the goinfer root graph. (`36eae14`)
- **f16 GPU KV cache** (`--kv f16`, opt-in; default `f32` stays bit-exact) — halves
  per-token KV bytes on the full-residency path, unlocking **32k context for a
  7B-int4 on an 8 GB card** (32k f16 fits in the same VRAM as 16k f32 — measured
  6912 vs 6926 MiB on an RTX 2070 SUPER). Manual WGSL f16 via the core
  `pack2x16float`/`unpack2x16float` builtins (no `shader-f16` device feature, so the
  CI software adapter still compiles). Decode parity vs an f32 cache: argmax
  preserved, full-logit cosine ≥0.998 over an 8000-key context. (`138d5e0`,
  `b40d8dd`)
- **Chunked-parallel Gated DeltaNet scan kernel** (`deltanet_chunked.go`) — the
  reformulation the `qwen3_5_moe` perf rewrite needs, unrolling the gated
  delta-rule recurrence over a chunk into the matmul-friendly form. Proven
  algebraically equivalent to the sequential recurrence over random
  inputs/chunk sizes (`TestGatedDeltaNet_chunkedMatchesSequential`; self-contained,
  no checkpoint/torch). Not yet on the hot path — the forward is still
  single-token streaming — so it's a zero-regression-risk reference. (`750b4ac`)

### Security
- **Hostile model files / request bodies now error, never panic.** A fuzzing pass
  over the untrusted-input surface (Go native fuzzing, CI-enforced via committed
  seed corpora) found and fixed five panic/OOM vectors, each turned into a typed
  error with a regression test:
  - GGUF: a metadata `block_count` ≥ 2³¹ overflowed `int` to a negative
    `NumLayers` → `make([]LayerWeights, …)` panic. (`9cdf98d`)
  - `.giw` bundle: a near-`maxint64` v2 length overflowed the bound check in
    `cur.take` → slice-bounds panic. (`92f0d83`)
  - `tokenizer.json`: a hostile token id overflowed / OOM'd / negative-indexed the
    `idToPiece` allocation. (`8e0b8bd`)
  - GPTQ/AWQ: dims not a multiple of 8 slipped past the integer-division shape
    check → dequant indexed out of bounds. (`aebbd45`)
  - Serialized `.giw` weights: a corrupt blob could drive a multi-GB allocation;
    the layer/expert count gates were tightened to sane bounds. (`c6b5ff2`)
  Fuzz targets now cover `constrain`, the GGUF / serialized-weights / GPTQ-AWQ
  loaders, the `.giw` frame, `tokenizer.json`, and the serve request shapers, and a
  `-race` serve soak/chaos test exercises multi-model + admin + sessions under
  concurrency (zero goroutine leaks, byte-identical warm-KV restore). The GGUF
  interpretation fuzzer was re-enabled after the aikit hardening landed (`e3bc98d`).

### Performance
- **~3.4× prompt prefill** — vectorized the prefill attention (QKᵀ + scores·V)
  onto the SIMD path. (`7fa82c2`)
- **~1.7× Gemma 4 prefill** — the same treatment for gemma4 attention. (`88b7aaa`)
- `qwen3_5_moe`: softmax attention reuses the cache scratch for scores
  (`4fcc069`), and its M=1 projections route through the SIMD A·Bᵀ kernel
  (`443ab7a`).
- **~2.4× sparse-MoE prompt prefill** (Mellum2 12B-A2.5B: 3.36 → 8.11 tok/s at a
  1024-token context). `canBatchN` now admits standard sparse-MoE (Mellum / Mixtral),
  so their prefill batches the attention onto the SIMD A·Bᵀ kernel like the dense
  families — a CPU profile put the scalar per-token attention at ~83% of MoE prefill,
  the expert matmuls only ~17% (those stay per-token: the router picks different
  experts per row). The batched attention uses the f64-accumulating `MatmulBTAcc64`
  and decode is routed through the same kernel, so the result is **bit-identical to
  the sequential forward** (the discrete router won't tolerate an f32 matmul's ~1e-6
  reassociation). The qwen3_5_moe DeltaNet hybrid and Gemma 4 keep their own forwards.

## [v0.4.0] — 2026-06-09

### Added
- **GPU full-residency decode (W4A8 int4) — run bigger models pure-GPU.** Real
  int4 `.giw` models now decode entirely on the GPU through `decoder.Generate` on
  the `webgpu` backend (dense Qwen2/Llama): the full-token forward is the resident
  DecodeRunner, not the per-matmul staged path. The win is **footprint, not a
  speed record** — int4 halves resident weights, so a **7B int4 fits and decodes
  at ~51 tok/s on an 8 GB card** (the model class that does NOT fit at int8), at
  **~71% of llama.cpp-CUDA (q4) at equal 4-bit quant** (51.7 vs 72.8 tok/s; greedy
  output matches the CPU decode bit-for-bit on the first tokens). int8 residency
  peaks ~89.7 tok/s on the 1.5B (3.5× the staged hybrid; **61% of Ollama-q8** at
  equal int8 quant, 89.7 vs 147). *(Peer figures as measured for v0.5.0, 2026-06,
  on the WebGPU backend vs then-current llama.cpp / Ollama; these are not current —
  the cgo-free CUDA backend + the Ollama v0.32.5 re-anchor in `docs/benchmarks.md`
  §B2 supersede them.)* v1 limits: **stateless `Generate` only**
  (`Session`/prefix-reuse/`GenerateSpeculative` fall back to the staged path),
  **16k context cap** (f32 KV), **eligible archs only** (dense Qwen2/Llama;
  MoE/Gemma/hybrid → staged). See `docs/completed/gpu-assessment.md` §0.0 + the §1 decision
  matrix. (The `.giw` bundle's weights length is now u64 — v2 — so int4 models
  past 4 GiB, i.e. the 7B+ class, serialize without truncation; v1 bundles still
  load.)
- **cmd/serve: multi-model + admin + Responses API.** `--model` is now repeatable
  as `name=path` to serve a model zoo from one process; requests route on the
  OpenAI `model` field (exact match, or the sole model for single-model compat),
  unknown → an OpenAI-shaped 404, and `/v1/models` lists all. Each model has its
  own mutex (distinct models run in parallel) and warm-KV dir
  (`--session-dir/<fp>/`). **Admin API** (gated behind `--allow-admin`, default
  off — RCE-adjacent): `POST /admin/models/{load,unload}`, with unload refusing a
  busy model (409) and snapshotting its warm KV. **`/v1/responses`** (OpenAI
  Responses API): `input`/`instructions`/`text.format`(→constrain)/`tools`,
  streaming event shapes, and `store`/`previous_response_id` (an in-memory ring
  that rides the per-model sessionLRU for warm KV). **Backpressure**: a bounded
  per-model queue (`--max-queue`, default 8) returns 429 + Retry-After when full
  (single decode worker per model; not continuous batching). Internally the
  generative half is a `loadedModel` registry. Pure stdlib `net/http`, no deps.
- **Qwen3.5/3.6-MoE (`qwen3_5_moe`)** — the hybrid linear/softmax-attention MoE.
  Most layers are **Gated DeltaNet** (linear attention with a recurrent matrix
  state — short causal conv + gated delta rule + gated RMSNorm, its own forward
  primitive), the rest gated softmax attention (double-width q_proj → query‖gate,
  QK-norm, partial RoPE, output gate), over a 256-expert + shared MoE on every
  layer. A **hybrid cache** holds KV for the softmax layers + a fixed-size
  `deltaState` per linear layer; prefix-reuse / speculative fall back for the
  hybrid (recurrent state isn't position-truncatable). Loads from safetensors,
  bit-exact vs the HF oracle: the DeltaNet primitive op-for-op (cosine 1.0,
  `deltanet_test.go`) and the full model argmax + cosine 1.0
  (`qwen35_forward_test.go`). **Loads from both safetensors and GGUF** — the
  `qwen3_5_moe` loader reverses llama.cpp's fused/stacked transform back to the
  per-expert layout — with **real-checkpoint int8 parity on the 35B-A3B** (Gate 2:
  argmax **74/80**, sample cosine **0.99466** vs the banked HF bf16 golden), and the
  GGUF path proven by weight-diff against the safetensors load. **Honest scope:**
  this is the **text decoder of the Qwen3-VL 35B-A3B** model (the language tower),
  and the hybrid arch runs the **staged path — not GPU residency**. Parity-first f32
  forward. See `docs/qwen3_5_moe.md`.
- **Mellum2 chat template** — `chat.Mellum2()` (a named ChatML alias) + a `Detect`
  fingerprint (its distinctive `normalize_content` macro) so JetBrains Mellum2 is
  identified as `mellum2` by `cmd/serve` / `demo/chat` rather than falling through
  to generic ChatML. Its template is ChatML byte-for-byte (`<|im_start|>`/
  `<|im_end|>` turns, stop `<|im_end|>` = EOS id 28, Hermes `<tool_call>` tools),
  verified against HF `apply_chat_template` (`testdata/chat_goldens/mellum2.json`:
  system+user, multi-turn, no-system) — the same byte-exact gate as the other five
  families. Tools ride the existing ChatML Hermes `RenderTools`/`ParseToolCalls`.

### Changed
- **Mellum2 parity-gated** — Mellum2 is no longer the README family list's parity
  exception. `TestMellum2_logitParity` pins the forward (MoE 64/top-8, 3:1
  sliding/full interleave, YaRN-on-full RoPE, QK-norm) against the HF bf16 oracle
  on a chat-templated prompt: argmax exact (`Paris`), sample-256 cosine **0.99955**
  (int8int8 vs bf16). The sliding-window EVICTION path — untested when validation
  stayed under the 1024 window — now has a model-free unit proof
  (`TestMellum2_slidingWindowEviction`: a sliding layer's output is invariant to an
  out-of-window key, a full layer's isn't) plus a real-checkpoint past-window point
  (`TestMellum2_windowParity`: 1441-token prompt, cosine **0.99636**).
- **CPU int4 decode is now usable** — the `decoder.Generate` int4 path moved onto
  aikit v1.1.1's `MatmulBTW4A8` (int4×int8 integer decode) from the f32-activation
  `MatmulBTQ4`, which was dequant-bound at M=1. CPU int4 goes from **~20–45× slower
  than int8 to ~1.4–1.9× int8** (arm64 + amd64), so int4 is now a real CPU
  footprint option, not only a GPU one.
- **aikit → v1.1.1** — adds the `MatmulBTW4A8` CPU int4 kernel above (the
  `embed`/`linalg` deps; both modules track it).

## [v0.3.0] — 2026-06-07

### Added
- **Embeddings endpoint** — `cmd/serve` serves OpenAI `/v1/embeddings` from an
  `--embed-model` CodeRankEmbed encoder (`--embed-quant f32|q8`), alongside or
  instead of the generative `--model`; endpoints register for whatever loaded.
  Honors `input` (string|array), `encoding_format` (`float`|`base64`), and
  `dimensions` (truncate + renormalize); vectors are L2-normalized. An optional
  `input_type` (`query`|`document`, default document — the Cohere/Voyage
  convention) selects the encoder's asymmetric query instruction prefix.
  `usage.prompt_tokens` is counted via the encoder's tokenizer; the encoder is
  goroutine-safe, so embeddings serve without the decoder's lock. Rides aikit
  v1.0.0's frozen `encoder` API (`Encode`/`EncodeBatch`/`HiddenDim`).
- **Prompt-prefix KV caching / cross-call session reuse** — `decoder.Session`
  pairs a KV cache with the tokens materialized in it and reuses the KV of the
  longest token prefix a new prompt shares, prefilling only the divergent suffix
  (exact — bit-identical to a cold prefill). `cmd/serve` layers a session LRU
  (`--kv-sessions`, default 4; 0 disables) so a continuing chat — or an agent
  loop with a fixed system prompt + tool specs — skips re-encoding the whole
  history; `--session-dir` persists warm sessions to CRC+identity-guarded
  `.giw-kv` snapshots and restores them across restarts. Fixes
  `KVCache.TruncateTo` to derive each layer's stride from what it holds, so it is
  correct on Gemma 4's per-layer KV widths and KV-shared tail layers. Reuse
  parity is gated id-for-id vs a cold prefill (Qwen 2.5 + Gemma 4 E2B).
- **LoRA adapter loading** — `decoder.Options.LoRA` merges a PEFT adapter
  (`adapter_config.json` + `adapter_model.safetensors`) into the base weights at
  load: W′ = W + (α/r)·B·A (α/√r under rslora), applied to the f32 weight *before*
  quantization, so a merged model costs nothing extra at decode. Targets the
  per-layer attention/MLP projections; an unsupported target (e.g. `embed_tokens`)
  or a GGUF base is a loud error. `demo/chat` and `cmd/serve` gain `--lora`.
- **Tool calling** — composes chat templates + JSON-Schema constraint + a
  per-family parser (parse against the model's template, not a naive JSON scan).
  `chat.Template.RenderTools` declares tools in each family's syntax and
  `ParseToolCalls` recovers structured calls; supported: ChatML/Qwen (Hermes
  `<tool_call>`), Mistral (`[TOOL_CALLS]`), Llama-3 (bare `{name,parameters}`),
  and Gemma 4's bespoke `<|tool_call>` micro-language (incl. its declaration
  format, byte-exact). `constrain.ToolCallGrammar` constrains a call to a tool's
  schema (the family wrapper around `{"name":const,…:<args schema>}`). `cmd/serve`
  honors OpenAI `tools` / `tool_choice`: renders declarations, constrains the call
  when unambiguous (one tool or a forced function — Gemma 4 is parse-only, no
  logit constraint), and returns `tool_calls` with `finish_reason:"tool_calls"`.
  Tested: per-family parse + round-trip, the tool-grammar property (every
  constrained generation is a valid, schema-conforming call), and a server
  integration call (Qwen → `get_weather(...)`).
- **OpenAI-compatible HTTP server** (`cmd/serve`) — pure stdlib `net/http`, no
  deps. `/v1/chat/completions`, `/v1/completions`, `/v1/models`; streaming via
  SSE; the sampling knobs (temperature/top_p/top_k/seed/frequency_penalty/
  presence_penalty/stop/logprobs) map onto goinfer's sampler; and
  **`response_format`** (`json_object` / `json_schema`) rides the constrain
  grammar for output the model cannot violate. Chat templates auto-detected per
  model (the `chat` package). Point Open WebUI / LangChain / the OpenAI SDKs at
  `http://host:8080/v1`. (Its own release; depends on the chat-template,
  sampling, and JSON-Schema work above.)
- **K-quant coverage confirmed: Q6_K and Q4_K_S** now have dedicated logit-parity
  tests (TinyLlama, same harness as the other GGUF quant tests), alongside the
  existing Q2_K/Q3_K_M/Q4_K_M/Q5_K_M. The dequant paths (via `aikit/embed`) were
  already present — most HF repos ship Q5_K_M/Q6_K next to Q4_K_M and goinfer
  loads them all; these tests pin per-quant parity (argmax exact; cosine-floored).
- **JSON Schema constrained decoding** (the v0.2 flagship) — `constrain.JSONSchema`
  compiles a JSON Schema into the existing incremental byte-level `Grammar`, so the
  streaming logit masker drives it unchanged: the model **physically cannot** emit
  non-conforming JSON (invalid tokens are −∞, i.e. unreachable). Supported subset:
  objects (required + optional, `additionalProperties:false`), arrays
  (`items`/`minItems`/`maxItems`), `string`/`number`/`integer`/`boolean`/`null`,
  `enum`/`const`, arbitrary nesting; unsupported keywords are a loud compile error.
  **`constrain.GrammarFromStruct(v)`** derives the schema from a Go struct's json
  tags — "a struct the model cannot violate": constrain → `json.Unmarshal` always
  succeeds. `demo/chat --schema <file.json>` constrains the demo. A property-based
  test asserts every constrained generation validates against its schema; the
  unconstrained path is untouched (purely additive).
- **`chat` package** — chat templating as a library feature (no Jinja engine).
  `chat.Detect(meta)` fingerprints the GGUF/HF `tokenizer.chat_template` string
  (falling back to a vocab-marker heuristic for bare checkpoints) and returns a
  `Template`; `Template.Render(system, turns)` produces the exact prompt string
  and `Template.Stops()` the turn-stop markers. Native Go renderers for **Gemma 3,
  Gemma 4, ChatML (Qwen), Llama-3, and Mistral**, each **byte-exact against HF
  `apply_chat_template`** (golden fixtures in `testdata/chat_goldens`). An
  unrecognized template is an explicit `ErrUnknownTemplate` (caller falls back to
  raw completion). New tokenizer accessors `ChatTemplate()`, `Has()`, `TokenID()`.
- **Sampling completeness** in `SamplingParams` / `Sampler` — the standard
  controls a llama.cpp/Ollama/vLLM user expects:
  - **min-p** (`MinP`) — keep tokens with prob ≥ min-p·max-prob (a relative
    floor), composable with top-k/top-p.
  - **repetition controls** — llama.cpp-style `RepeatPenalty` (scales repeated
    tokens' logits) plus OpenAI-style `PresencePenalty` and `FrequencyPenalty`,
    over a `RepeatLastN` window (the prompt is seeded so it counts). `Generate`
    wires the history automatically.
  - **`LogitBias`** — per-token logit offsets (force or ban specific tokens).
  - **logprobs out** — `Sampler.SampleWithInfo` and `SamplingParams.Logprobs` /
    `TopLogprobs` report the chosen token's log-probability and the top
    alternatives; `Generation.Logprobs` collects them per emitted token.
  - `Seed` (already present) gives reproducible draws.
  `Sample(logits)` and the greedy parity path are unchanged. `demo/chat` gains
  `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`,
  `--repeat-last-n`.

### Changed
- Depends on **aikit v1.0.0** (was v0.5.2): its now-frozen Hard tier covers the
  embedding encoder behind `/v1/embeddings`. The `goinfer/gpu` submodule is
  bumped to match, so both modules ship against the same published aikit.
- `cmd/serve` `--model` is now optional when `--embed-model` is given (and vice
  versa); at least one is required. The single-model flag set moved to a config
  struct as the server grew a generative and an embedding half.
- `demo/chat` and `demo/gemma-web` now render prompts via the `chat` package
  (was duplicated per-demo). The Gemma 4 demo render matches HF exactly,
  including the `<|channel>thought` scaffold (so the model may emit reasoning).

### Removed
- `tokenizer.ChatStyle` (enum + method, shipped in v0.2.0) — superseded by
  `chat.Detect`, whose fallback subsumes the old vocab-marker heuristic.

## [v0.2.0] — 2026-06-06

### Added
- **Gemma 4 support** (HF `model_type` `gemma4_unified_text`; GGUF arch
  `gemma4`) — the **E2B** and **E4B** "E-models" plus the **12B dense**, all
  parity-gated against the HF bf16 oracle. Gemma 4 is a meaningfully new
  architecture on top of the Gemma 3 stack, driven entirely through the
  `Architecture` descriptor:
  - per-layer **head_dim** (256 local / 512 global) and per-layer **KV-head
    count**, scale-less **v-norm**, **proportional (partial-rotary) RoPE** on the
    global layers, dual-base RoPE, and the final-logit softcap (30);
  - the E-model additions — **Per-Layer Embeddings (PLE)** branch, **cross-layer
    KV sharing**, variable per-layer **FFN width**, and a per-layer output
    scalar (no AltUp/Laurel — those are Gemma 3n);
  - the 12B's **`attention_k_eq_v`** (V reuses K's projection on the global
    layers).

  Greedy generation is coherent ("Paris"). Gates: `TestGemma4_logitParity`
  (E2B, sample-256 cosine **0.99938** vs HF bf16) and `TestGemma4_12B_logitParity`
  (12B, argmax exact + cosine **0.990**). Out of scope: the 26B-A4B MoE and 31B
  multimodal vision towers (text-only runtime).
- **Gemma 4 chat template** in the tokenizer + `demo/chat` — Gemma 4 replaced
  Gemma 3's `<start_of_turn>`/`<end_of_turn>` with new `<|turn>`/`<turn|>`
  markers. `Tokenizer.ChatStyle` now detects them (`ChatStyleGemma4`), the GGUF
  loader resolves the turn-stop token, and the demo renders the right template
  so replies are clean and stop correctly.

## [v0.1.3] — 2026-06-05

### Added
- **`decoder.LoadGGUFBytes`** / **`tokenizer.LoadGGUFBytes`** — load a GGUF model
  and its tokenizer from an in-memory `[]byte`, touching no filesystem. The
  shared GGUF build core is reused by the path-based `Load` / `LoadGGUF` (both
  unchanged); EOS ids resolve from the GGUF's own metadata.
- **`decoder.SerializeWeights` / `LoadSerializedWeights`** + `Model.Weights()` /
  `Model.Quant()` / `NewModel` — a versioned binary format (`.giw`: magic +
  version + config + quant guard + CRC, lazy fallback like ken's index) for an
  already-quantized weight bundle. On load the big int8/int4 arrays are **aliased
  zero-copy** out of the blob (float scales are copied for alignment), so a
  prequant build skips dequant+requant *and* the resident-weight heap copy.
- **`cmd/prequant`** — build-time generator: turns a GGUF into a `.giw` bundle
  (serialized int8 weights + a metadata-only GGUF carrying the tokenizer).
- **`decoder.GenerateSpeculative`** + **`demo/chat --draft <gguf>`** — greedy
  speculative decoding: a small draft model proposes K tokens, the target
  verifies them in one batched pass (`forwardN`), keeping the matching prefix.
  Output is **token-identical to plain target greedy** (gated by
  `TestSpeculativeGreedyParity`); `cache.TruncateTo` does the KV rollback. On the
  pure-Go CPU int8 kernel it's ~break-even (decode is ~half compute, which
  doesn't amortize at M=K), so it's a forward-looking / bandwidth-bound-backend
  feature — correct and exact, not a CPU speedup.
- **`decoder.SetDecodeParallelThreshold`** / `DefaultDecodeParallelThreshold` —
  goinfer-owned tuning of aikit's matmul parallelism crossover for batch=1
  decode (the demo sets it; aikit's library default stays conservative).

### Changed
- **Faster decode (~48 → ~70 tok/s on the 0.5B, +42%; ~36 on the 1.5B; M-series
  CPU)** via aikit v0.5.x: a per-stream `Workspace` makes steady-state decode
  **zero-alloc** (4660 → 19 allocs/op), q/k/v and gate/up run as **batched**
  matmuls, and a decode-tuned parallelism threshold parallelizes the per-token
  weight matmuls. Numerics bit-identical (`TestDecodeParity`).
- **Batched prompt prefill** — the prompt now runs through one `M=len(prompt)`
  pass instead of sequential single-token forwards, **~1.9–2.9× faster
  time-to-first-token** on the 1.5B (e.g. ~2.7 s → ~1.3 s for an 80-token
  prompt). Seed token bit-identical.
- Requires **aikit ≥ v0.5.2** (the `Workspace` / batched / `Into` W8A8 matmul API
  and the column-blocked W8A8 kernel that reuses weights across M>1 rows).
- `demo/chat` (embed build) loads the baked-in model **from memory by default** —
  no temp file, so the single-file binary runs on a read-only / `FROM scratch`
  filesystem. `--model-tmp` / `GOINFER_MODEL_TMP=1` opts back into a temp-file +
  mmap load (lower peak RAM for large models; tmpfs caveat documented). Load
  progress prints to stderr.
- `demo/chat` gains a **`-tags prequant`** build (now `build-embed.sh`'s default)
  that embeds a `.giw` bundle and maps the int8 weights from the binary image.
  Measured on the 0.5B (Qwen2.5-Coder, M-series CPU): cold start **2.3 s → 0.48 s**
  and resident heap (`phys_footprint`) **772 MB → 78 MB**. The win scales with
  model size — a 4B int8 model no longer needs a ~5 GB weight heap per launch.
  Quant is fixed at bundle-build time; the runtime `--quant` flag and the GGUF
  fallback apply only to the `--model <gguf>` path.
- The embed build now bakes the GGUF **uncompressed** (dropping zstd — q4 weights
  are high-entropy, so it shaved only ~3% while costing inflate time + a
  full-size heap buffer). Removes the `klauspost/compress` dependency; the
  default module graph is back to `aikit` + `x/text`.
- `demo/chat` now ships in **two size tiers** built from the same program:
  Qwen2.5-Coder **0.5B** (~617 MB, ~57 tok/s — the headline) and **1.5B**
  (~1.7 GB, ~26 tok/s — bigger, smarter, still one file). `build-embed.sh --name`
  parameterizes the output basename so the tiers build side by side without
  clobbering. Prequant keeps the 1.5B's resident heap at ~87 MB (≈ the 0.5B), and
  the 1.5B binary stays under GitHub's 2 GiB asset cap.

## [v0.1.2] — 2026-06-04

### Added
- **`demo/chat`** — "an entire LLM in one file": an interactive REPL with a
  zstd-`//go:embed`-ed Qwen2.5-Coder-0.5B-Instruct model baked into a static,
  no-cgo, cross-compiled binary (macOS / Linux / Windows × amd64/arm64).
  Defaults to the fast int8×int8 (W8A8) kernel; ships live prompt-tuning
  slash-commands and canned `/demo` prompts.

### Changed
- Requires `aikit ≥ v0.4.1`, which platform-guards the embed loader's mmap so the
  decoder (and the demo binary) cross-compiles to `windows/amd64`.

## [v0.1.1] — 2026-06-04

### Added
- **`decoder`** — generic decoder-only forward pass with HuggingFace logit
  parity: Gemma 3, Qwen 2.5/3, Llama 2/3, Mistral, Mixtral, GPT-2, Mellum.
  Loads safetensors (single or sharded), GGUF, GPTQ, and AWQ checkpoints;
  f32/bf16/f16 plus int8 (weight-only and W8A8) and group-wise int4
  quantization; KV-cache; the standard samplers; and a pluggable matmul
  `Backend` (default pure-Go SIMD).
- **`tokenizer`** — byte-level BPE and SentencePiece byte-fallback tokenizers
  for the decoder LLMs, from `tokenizer.json` or a bare `.gguf`, id-exact
  against HuggingFace.
- **`constrain`** — constrained / structured decoding via a logit mask that
  forces output to satisfy a grammar; ships a streaming JSON grammar.
- **`gpu`** (opt-in submodule, `-tags gpu`) — WebGPU compute backend
  (Metal / Vulkan / DX12) that registers a resident-weight matmul into both
  the goinfer `decoder` and the aikit `encoder`. The cgo
  `github.com/cogentcore/webgpu` dependency is confined to this submodule;
  the default goinfer build is pure Go, no cgo.
- `demo/gemma` and `demo/gemma-web` — CLI and HTTP examples wiring tokenizer +
  decoder with optional sampling and JSON-constrained output.

### Notes
- goinfer is extracted from [`aikit`](https://github.com/townsendmerino/aikit)'s
  `decoder` / `tokenizer` / `constrain` packages so the LLM runtime can release
  on its own cadence, independent of aikit's stable retrieval core. It depends
  inward on `aikit/embed` + `aikit/linalg` (≥ v0.4.0, which promoted `linalg`
  to public and gave `encoder` a pluggable GPU `Backend`). The split is tracked
  in `aikit/docs/aikit-module-split-plan.md`. Pre-extraction history for these
  packages lives in the aikit repository.
