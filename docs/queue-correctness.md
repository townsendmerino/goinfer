# Correctness queue

Parity, numerics, goldens, quantization, model families. Anything whose success criterion is **agreement with a reference** — a cosine, an argmax match, a golden. If the question is *does it compute the right thing*, it belongs here.

> **One of four queues.** The work list is split by *success criterion*, not by component:
> [performance](queue-performance.md) · [correctness](queue-correctness.md) ·
> [engineering](queue-engineering.md) · [release](queue-release.md).
> [`QUEUE.md`](QUEUE.md) is the index over all four and holds the cross-cutting sweeps.
>
> **Task docs are NOT queues.** `docs/task-*.md` are *design records* — why a thing is built as it
> is — and they are cited from 88 code comments. A queue entry cannot carry that, so the task docs
> stay put and the queues hold only the open work.
>
> Entries keep the section they were filed under (`In flight`, `Queued`, …) and their original IDs,
> so a citation to an ID still finds it.


## Queued

> **Seven closed entries were archived 2026-08-31** to
> [`docs/completed/queue-correctness.md`](completed/queue-correctness.md) — G4, G5, G6, G9, G10,
> G11 and Q2. What is below is the open work, and it is all of it.

**G7 · gpt-oss residency upgrade (safetensors + MXFP4 loader, GPU residency)** — `any`

Full scoping and reasoning: `docs/post-v1.0-models.md` "Next up" §4. Not a new family — `gpt_oss`
already has real-oracle parity (`docs/capability-matrix.md:89`) — but it is **GGUF-only,
CPU-only** today. The MXFP4 *reader* already exists (`decoder/mxfp4.go`, verified bit-for-bit
against the reference `gguf` Python library) but was built for the GGUF path only. Two pieces: (a)
a safetensors loader for gpt-oss's native MXFP4-packed weights, (b) Metal/CUDA GPU residency.
gpt-oss-20b is one of the most-run local models in 2026 guides — an upgrade to an already-popular
family plausibly moves more real users than a new family would; weigh against `G4`-`G6` on that
basis, not just novelty.

**GPU residency progress (2026-08-18) — kernels built and gated on BOTH CUDA and Metal, neither
declared yet.** *(Superseded on the Metal half: **G10 declared Metal** later the same day — "Net
state — DECLARED", `TestGptOssResidentParity` green at 8/8 argmax-exact, min cosine 0.9989. G10 is
archived at `docs/completed/queue-correctness.md`. **CUDA is still undeclared**, and together with
the safetensors MXFP4 loader that is what keeps G7 open. The capability-matrix wording was not
re-verified as part of this correction.)* `FeatAttnSink` bundles gpt-oss's three departures from every other resident
family: the learned per-head softmax sink, the clamped interleaved-SwiGLU expert (per-expert
biases, asymmetric clamp, α-scaled sigmoid, +1 linear branch), and a router whose bias reaches the
selection WEIGHT, not just the selection itself (`moe_route`'s bias-steers-selection-only contract
is wrong for this family).

- **CUDA** (`linux-62gb`) has all three as real kernels (`cuda/gptoss_act.cu`'s `glu_quant_gptoss` +
  `route_gptoss`, plus the sink argument threaded through `decode_splitkv.cu`'s attention kernels),
  each gated against the CPU reference. Declaring `FeatAttnSink` was tried and **reverted**
  (`2224441`): CI correctly caught that kernel-level parity is not end-to-end parity — nothing has
  run a whole gpt-oss forward on the resident path — and two MORE capabilities are still missing,
  `FeatOutBias` (o_proj bias) and `FeatRopeMscale` (YaRN), neither of which any resident backend
  declares for ANY family yet.
- **Metal** (`mac`) now has the same three kernels ported and gated the same way —
  `metal/gptoss_kernels_test.go`: `attention_f32_sink` (sink term added to the max-shift and
  denominator, verified including the case where the sink DOMINATES the max — the case a
  post-hoc denominator patch gets wrong), `swiglu_quant_gptoss` (clamp coverage asserted on both
  branches, not just present), `route_gptoss` (bias asserted to actually change which experts win,
  not just that the output is finite). Each compiles its own isolated MSL source, touching neither
  `allKernels` nor `moeKernels` — zero risk to any family already resident on Metal. Full
  `go test ./metal/...` still green. Metal is at the exact same phase CUDA is: kernels exist,
  nothing wired into `model.go`'s resident dispatch, `FeatAttnSink` not declared, and `FeatOutBias`/
  `FeatRopeMscale` still missing here too.
- **Not yet attempted on either backend:** the resident-bridge wiring CUDA's own commits show is a
  separate, non-trivial phase even after the kernels exist (`cuda/resident.go` needed three
  "wiring" commits — activation dispatch + resident fields, loading the pipeline, uploading sinks
  and per-expert biases into the launches). `FeatOutBias`/`FeatRopeMscale` wiring is itself
  reusable beyond gpt-oss (GPT-2, Mellum/long-context YaRN) but touches the shared resident bridge
  for every family that would opt into it, so it's a bigger, more careful piece of work than the
  gpt-oss-specific kernels were.
- **`FeatRopeMscale` groundwork landed on Metal (2026-08-18) — kernel capability only, still not
  declared.** `decoder.Model.RopeMscaleLayer` already existed and is already used by WebGPU
  (`gpu/residency.go`), so no decoder-side work was needed — only Metal's `rope` kernel
  (`kernels.go`, used by every resident family's decode path, 8 dispatch sites in `model.go`)
  was missing the `scale` multiply `decoder/rope.go`'s `applyRoPE` applies to cos/sin. Added as a
  new trailing buffer parameter, defaulting to 1.0 (a true no-op) everywhere via
  `residLayer.mscale = m.RopeMscaleLayer(l)`. **Caught a real regression from the change itself,
  not the pre-existing flake:** two OTHER test files (`layer_test.go`, `rope_partial_test.go`)
  independently dispatch the same shared `rope` pipeline outside `model.go` and were missed on
  the first pass — the full suite correctly went red (`TestLayerB_fullLayerForward` cosine 0.229,
  `TestRopePartial` maxAbs 1.31), not the known fault-0x10 crash, a clean assertion failure from
  an unbound buffer. Fixed by adding `scale=1.0` to both; two full `go test ./metal/...` runs
  since are clean. New gate: `TestRope_mscale` (`metal/rope_test.go`), checking exact
  per-component values (not just cosine similarity) against `applyRoPE`'s cos/sin-scaling
  placement, since scaling the ROTATED OUTPUT instead is a different, wrong result that a loose
  tolerance could miss. Not declared for Metal (same discipline as `FeatAttnSink` above — no
  family exercises `scale != 1.0` end-to-end here yet).
- **`FeatOutBias`'s kernel landed too (2026-08-18) — the last of the two missing capabilities is
  now scoped and built.** Unlike mscale, no existing SA-family GEMV combines bias-add with
  residual-accumulate (`gemv_w4a8_sa_bias` overwrites, `gemv_w4a8_sa_resid` accumulates, neither
  does both — checked directly rather than assumed cheap), so this needed a genuinely new kernel:
  `gemv_w4a8_sa_bias_resid` (`kernels.go`). Purely additive — no pipeline is instantiated in
  `model.go` for it (no family resident on Metal declares `FeatOutBias` yet, same as
  `gemv_w4a8_sa_amax`'s existing "created but never dispatched" precedent, N-09), so this carries
  even less regression risk than the mscale change: nothing existing was touched, only a new
  kernel added to the shared library. Gated standalone (`TestSAGemvBiasResid`,
  `metal/gemv_w4a8_sa_bias_resid_test.go`) against a CPU dequant reference of the same packed
  int4 weights, with two explicit negative checks — the output must NOT match a bias-dropped or a
  residual-dropped reference — so a regression that silently drops one epilogue term would fail
  loudly instead of passing on a coincidental near-match. Full `go test ./metal/...` clean.
  **Both of gpt-oss's non-family-specific prerequisites now have a Metal kernel; wiring either
  into `model.go`'s real per-layer dispatch (for GPT-2's `FeatOutBias` or a resident YaRN family's
  `FeatRopeMscale`) is the next piece, and is itself independent of gpt-oss ever landing.**
- **Real-checkpoint end-to-end validation is blocked on both machines for different reasons:**
  gpt-oss-20b MXFP4 is ~13.8GB against the CUDA box's 8GB VRAM (testable there only via the
  host↔VRAM MoE-streaming path, coupling two hard things at once); this Mac has 16GB total RAM but
  only ~12GB free DISK as of 2026-08-18 — the checkpoint cannot even be downloaded here right now,
  a harder blocker than the RAM tightness itself. Neither box can currently reach the real
  end-to-end gate without freeing real resources first.

**G8 · DeepSeek V4-Flash as a new family — blocked on fp8 support, post-1.0** — `any`. **The fp8 blocker is now FILED as its own item (`Q3`, below) with an estimate, per this entry's own instruction; G8 itself is unchanged and still lowest priority.**

Scoping already done: `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`'s Phase 0 verdict.
**Not** a `deepseekArchitecture` alias — eight new primitives (DSA sparse attention over a learned
Indexer, strided KV compression, sliding-window + attention sink, grouped low-rank output
projection, hash routing, `sqrtsoftplus` router scoring, hyper-connections, clamped SwiGLU).
**Hard prerequisite, not a subtask:** V4-Flash ships fp8 e4m3 blockwise-quantized weights and
**there is no fp8 support anywhere in the tree today** — file/estimate the fp8 reader as its own
piece of work before scoping the primitive additions. MIT license, DeepSeek's brand pulls the
whole local community, and native sparse attention is where the field (V3.2, GLM-5.1, V4) is
converging — building the DSA/compressor path once plausibly buys the next several Chinese
frontier releases, which is the strategic case for filing this now even though it's not a
near-term ship. Lowest priority of the five items filed alongside this one (`G4`-`G7`).

---

**The fp8 prerequisite, filed as its own work per this entry's own instruction (2026-08-31).**

**The "no fp8 anywhere in the tree" claim is VERIFIED, not carried forward on trust.** Searched both
repos for `fp8|e4m3|e5m2|float8` outside tests: the only hits are `simdgroup_float8x8` in
`metal/prefill.go` and aikit's `gpu/metal_vit.go`, which is **Metal's 8×8 SIMD-group float TILE
type, not the fp8 numeric format**. Worth writing down, because that name collision makes a casual
grep look like fp8 support already exists.

**Q3 · fp8 e4m3 reader** — `any`, **NOT STARTED. Hard prerequisite for G8; useful independently.**

Three pieces, and only the middle one is novel:

1. **Dtype recognition (aikit).** `embed/safetensors.go` widens `F32`/`BF16`/`F16` and nothing else,
   so an fp8 checkpoint cannot currently be read at all. Add `F8_E4M3` (and `F8_E5M2` while there —
   the header parse is shared).
2. **The decode itself.** e4m3 is byte-aligned with 256 representable values, so the whole format is
   a **256-entry float32 lookup table** — strictly simpler per element than the existing
   `decoder/mxfp4.go` (137 lines), which has to unpack nibbles. The catch is the OCP e4m3 special
   cases: bias 7, **no infinities**, `S.1111.111` is NaN, max finite 448.
3. **Blockwise scales, which MXFP4 does NOT have an analogue for.** MXFP4 carries its e8m0 scale
   **inline, one per 32-element block**. DeepSeek's fp8 keeps scales in a **separate tensor**
   (`*.weight_scale_inv`) over 2-D blocks, so this needs block-index arithmetic and a second tensor
   fetched alongside each weight — new plumbing in `decoder/weights.go`, not a copy of mxfp4.go.

**Estimate: ~150–250 lines**, anchored on mxfp4.go's 137 for a comparable format decoder, plus the
2-D scale addressing that has no precedent here. Novel logic is small; the risk is all in (3).

**Verification is not optional and has a precedent to copy.** mxfp4.go's header records that its
layout was *"NOT inferred — it is transcribed from the reference `gguf` Python library … and
verified bit-for-bit against it on a real checkpoint"*. **Do the same for e4m3**: transcribe from a
reference implementation and gate it bit-for-bit, rather than deriving the table from the spec and
hoping. A quietly wrong dequant table is exactly the failure class this repo keeps naming — it
produces plausible output, not an error.

**Sequencing.** Q3 does not need G8 and should not wait for it: fp8 checkpoints are increasingly
common (DeepSeek V3/V3.2/V4, and others), so the reader has value on its own. G8 stays post-1.0 and
lowest priority; Q3 is the piece that could be picked up any time.

**G1 · LFM2.5-2.6B as an experimental family** — `linux`, **SCOPED AND ESTIMATED; ready to build,
not started.**

A fifth sequence-mixing family: interleaved gated short-convolution blocks and GQA, `layer_types`
controlling the pattern, `conv_L_cache` 3, per-head **RMSNorm** QK-norm, FFN dim stated (10752). The
conv layers carry a rolling conv state instead of a KV cache.

> **This entry was stale in four ways and is rewritten above; the old text is kept below because
> one of the four was a factual error, not just an out-of-date status.**
>
> It said the estimate "turns on two questions" plus one "unestablished" — all three were answered
> in [`docs/scoping-lfm2.md`](scoping-lfm2.md) on 2026-08-11, the same day the scoping ran:
>
> | question the entry called open | scoping doc's answer |
> |---|---|
> | is Mamba-2's causal depthwise `conv1d` factored out or inlined? | **§B: inlined** — three verbatim copies (`mamba2.go`, `mamba2_chunked.go`, `deltanet.go`); the 3-way dup argues for extracting a shared helper |
> | does the cache carry mixed per-layer state types? | **§C: yes** — `KVCache` already holds parallel per-layer arrays; LFM2's conv state is a strict subset of `mamba2State` |
> | is LFM2.5 architecturally the same as LFM2? | **§A: yes, a scaled retrain** — topology byte-identical; differences are scale only (vocab **128,000** not 65,536; rope_theta 1e7) |
>
> **And the error: "LayerNorm QK-norm (not RMSNorm)" is wrong.** It repeats the original brief,
> which §E corrected — the reference `modeling_lfm2.py` uses `Lfm2RMSNorm(head_dim)` per-head, and
> `scripts/pin_lfm2_tiny.py`'s header (written against the real config) independently says "per-head
> Q/K RMSNorm". §E flags this exact field as "the 'quiet wrong answer' risk", and the wrong answer
> was sitting in the queue. It matters concretely: RMSNorm means **zero new code** (goinfer's
> existing hardcoded QK-norm path); LayerNorm would mean ~25 lines plus `.bias` plumbing.
>
> **The freeze gate is also gone** — lifted 2026-08-18/19, thirteen days before this was read.

**Estimate (from the scoping doc):** ≈350–500 new lines, of which only ~15–40 are novel logic (the
gated short-conv step). safetensors-first, CPU-only, experimental tier; GGUF deferred (llama.cpp
supports arch `lfm2` natively and an official `LFM2.5-2.6B-GGUF` exists, so it is a straightforward
follow-on).

**What actually gates it now is cost, not permission.** The surface touches
`registry.go`/`config.go`/`arch.go`/`weights.go`/`kvcache.go` — core+loaders — so it re-stales
`deps_hash` for all 19 enforced families and forces a goldens re-validation. **Batch it with other
core-touching family work** so that re-validation is paid once rather than per family.

**One confirmation still owed, and it is cheap:** weight-level proof that no
`q_layernorm.bias`/`k_layernorm.bias` tensor exists in the real checkpoint. That is a tensor-name
check at load, not an experiment — but no LFM2 checkpoint is on either box today.

**Original entry follows.**

**G1 (original) · LFM2.5-2.6B as an experimental family** — `linux`

Scoping prompt written. A fifth sequence-mixing family: interleaved gated short-convolution blocks
and GQA, `layer_types` controlling the pattern, `conv_L_cache` 3, LayerNorm QK-norm (not RMSNorm),
FFN dim computed rather than stated. The conv layers carry a rolling conv state instead of a KV
cache.

The estimate turns on two questions: whether Mamba-2's causal depthwise `conv1d` is factored out or
inlined, and whether the cache abstraction already carries mixed per-layer state types
(Granite-4.0-H and Nemotron-H suggest it may). Also unestablished: **whether LFM2.5 is
architecturally the same as LFM2** — the transformers docs cover only LFM2.

Blast radius matters: anything touching shared `decoder/` core re-stales all 19 enforced families.
Answer that before estimating.

**Q1 · The forward goldens prove f32 ONLY — no quantized path has a golden that runs** — `linux`,
**NEW. G-01 at the largest scale it has appeared.**

> **The "14 quantized" composition figure, resolved by enumeration rather than by authority
> (2026-08-12).** Two classifiers disagreed — an ad-hoc name grep said **7**, the refresh script's
> said **14** — and 14 had already propagated into commit bodies and into the proof requirement.
> Adopting it because it was the script's would have been a tiebreak by authority, so both were
> tested instead.
>
> **7 was structurally incapable of being right**, for two independent reasons. Five of the fourteen
> carry no quantization token in their NAME at all: `TestGemma4_logitParity` and
> `TestMellum2_logitParity` set it in the test body (`Options{Quant: "int8int8"}`), and
> `TestGGUF_gemma3/qwen2/qwen3_parity` set it in the **fixture filename** the test loads. No
> name-based match can see either. (The other two misses, `Q2_K` and `Q3_K_M`, were a plain gap in
> the ad-hoc pattern, which listed `q4|q5|q6|q8` — a bug rather than a structural limit, but it lands
> in the same place.)
>
> **The script's classifier cannot double-count.** `grep -c` counts matching LINES; every top-level
> result is one line; subtest lines are indented and excluded by its `^--- PASS:` anchor. Measured on
> the captured run: 33 top-level PASS lines, **0** indented ones, no duplicate names among the 14.
>
> And it does not misclassify — all fourteen drive a genuinely quantized path:
>
> | gate | quantization | set where |
> |---|---|---|
> | `TestGemma4_logitParity` | int8×int8 | test body |
> | `TestMellum2_logitParity` | int8×int8 | helper body |
> | `TestInt4_forwardParity` | int4 group-wise | test body |
> | `TestGGUF_Q2_K_parity` | Q2_K (+Q3_K/Q4_K/Q6_K mix-ins) | fixture |
> | `TestGGUF_Q3_K_M_parity` | Q3_K (+Q4_K/Q6_K) | fixture |
> | `TestGGUF_Q4_0_parity` | Q4_0 | fixture |
> | `TestGGUF_Q4_K_M_parity` | Q4_K (+Q6_K) | fixture |
> | `TestGGUF_Q4_K_S_parity` | Q4_K_S | fixture |
> | `TestGGUF_Q5_K_M_parity` | Q5_K (+Q6_K) | fixture |
> | `TestGGUF_Q6_K_parity` | Q6_K | fixture |
> | `TestGGUF_Q8_0_parity` | Q8_0 (tinyllama) | fixture |
> | `TestGGUF_gemma3_parity` | Q8_0 (gemma-3-270m) | fixture |
> | `TestGGUF_qwen2_parity` | Q8_0 (Qwen2.5-0.5B) | fixture |
> | `TestGGUF_qwen3_parity` | Q8_0 (Qwen3-1.7B) | fixture |
>
> **So 14 stands, and every commit body citing it is correct.** The reason is now recorded, which is
> the point: the figure is load-bearing in the proof requirement, and "the script said so" is not a
> reason. Note what the table also shows — **11 of the 14 take their quantization from a fixture**,
> so any future classifier that reads test names will undercount for the same structural reason.

int4 is the documented default quantization. **Zero goldens drive it.** And the hole is wider than
that: of the 19 goldens that actually RAN in the 2026-08-12 refresh, **every one is f32**.

| quantization | golden files | did any RUN? |
|---|---|---|
| f32 (explicit or default) | 24 | **19 ran** |
| `int8int8` (W8A8) | 3 — `gemma4_parity`, `gemma4_12b_parity`, `mellum2_parity` | **all 3 SKIPPED** |
| `int8` (weight-only Q8) | 1 — `gptoss_real` | not matched by the goldens regexp at all |
| **`int4` / W4A8** | **0** | — |

So `scripts/refresh_parity_hashes.sh` — the sanctioned freeze-exception path, and the thing that makes a
core edit auditable — **proves f32 numerics and nothing else**. A change that is bit-identical in f32
and wrong in int4 passes it in 6 seconds.

**Retroactive scope, and this is the part to act on.** Any claim of the form *"the parity suite
covers X"* is scoped to **the quantizations the goldens drive**, which today is f32. Every place such
a claim is written down needs that scope added — `docs/parity-coverage-policy.md`'s tier table,
`RELEASING.md`'s §C1, the README's support matrix, and the P6 commit body (which states it already).

**And the freeze protects what the goldens check.** The `6edd1ca` numerics freeze over `decoder/` is
enforced by `deps_hash` staleness, whose release valve is this goldens run. Where the goldens are
silent — every quantized path — the freeze is a *procedural* barrier with no numeric proof behind it.
That is not an argument for lifting it; it is an argument for knowing what it is.

**WHY THIS OUTRANKS THE REST OF THE QUEUE — sequencing, not enthusiasm.**

**P1 is the v1.0 headline and lives in the frozen core.** The numeric proof available when that core
unfreezes was **f32-only**. So lifting the freeze did not buy the ability to verify the work the
freeze defers — and the shortfall **would not have announced itself**, because the goldens would pass.
An f32-green refresh over an int4 regression is a passing gate, not a silent one; nothing in the
output distinguishes them.

That makes Q1(c) a **prerequisite for the v1.0 core work**, not a parallel item, and it belongs ahead
of the E-group release gate for that reason rather than because it is interesting. **Done
2026-08-12 (`1d0d1ed`)**: 23 fixtures across 16 architectures, so the prerequisite is now met for
int4 specifically.

**RUN WHAT EXISTS FIRST — and most of it was UNPLUMBED, not missing.** Done 2026-08-12, `a6c5b57`:

- **(b) the three `int8int8` goldens** skipped for one liftable reason, the same for all three:
  `GOINFER_HEAVY_TESTS` unset. **Two of the three pass here in ~70 s** (gemma4, mellum2). The refresh
  now enables heavy by default. The third (gemma4-12B) skips on a genuinely absent GGUF — an asset
  question, not a plumbing one.
- **(a) the `int8` golden did NOT turn out to be a selector bug.** `TestGptOssReal_logitParity` **does**
  match the regexp. It is invisible because `decoder/gptoss_real_test.go` is behind `//go:build realckpt`,
  which the refresh does not pass — and with the tag it still skips for a missing GGUF. **Two gates,
  either sufficient.** A one-line regexp change would have bought nothing.

**Non-f32 rows after (a) and (b): 2** (21 passed, 2 quantized). The distinction the ordering was meant
to test comes out clearly: **int8 was unplumbed** (one env var), **int4 is genuinely missing**, and the
gpt-oss int8 row is **asset-blocked behind a build tag**.

The refresh now also prints the **quantization breakdown**, because "19 passed" and "21 passed" read
identically to a human and that is precisely how this stayed invisible through nine prior refreshes.

**(c) int4 goldens — DONE `1d0d1ed`.** Scope measured *before* authoring and stated as a target: int4
has no divisibility constraint (`nGroups` is a ceiling divide), so eligibility was never the limit —
fixture availability was. **Target: 23 fixtures / 16 architectures. Delivered: 23 / 16.**

The goldens compare **int4 output against recorded int4 output**, not int4 against f32 within a
tolerance. A tolerance band against f32 measures quantizer loss — a real question with its own gate
on the policy's quant axis — and would read as "int4 is covered" while proving nothing about whether
the W4A8 path still computes what it computed yesterday. Only the self-comparison catches a
regression in the path the freeze protects and P7 will change.

Fixtures are **enumerated** from `testdata/` rather than listed by name, so a new family is picked up
without editing the gate, and a run comparing **zero** fixtures **fails** rather than passing.
Mutation-checked by perturbing the quantizer itself (`int4GroupSize` 32 → 64 → red).

Recorded **absences**, not gaps: `gpt_oss` (MXFP4-prequant, rejects a conflicting `--quant` by
design), `siglip_vision_model` (an encoder), `gpt2` / `mellum` / `qwen2` / `qwen3` (no tiny
safetensors fixture), `qwen2_moe` and `gemma4-dense-scaled-{24,48,64}` (incomplete fixture dirs).

**Refresh now reports 22 passed / 3 quantized**, against 19 passed / 0 quantized when this began.

**Also record with P6's 6.09 s price: cheap and thorough are different properties.** 6.09 s buys 19
passes and 11 skips. The skips are not free — they are the coverage this item is about.

**`TestDecodeParityInt4` diverges from its recorded golden — REAL checkpoint, NOT the synthetic
goldens above, found 2026-08-15, unclaimed.** `decoder/parity_int4_test.go`, real
qwen2.5-coder-0.5b int4 (W4A8, safetensors-loaded gguf), greedily continuing a fixed prompt: got vs
want diverge at token index 5 (`got 1438 want 11047`) and every token after — not a subtle drift,
a different continuation entirely. **Confirmed pre-existing and unrelated to two same-day changes**
via an isolated `git worktree` bisect: fails identically on the P1 pre-change tree AND at aikit
`v1.17.1` (before the day's aikit v1.19.0 bump) — same got/want arrays, byte for byte. So this
predates both P1 (`97f824a`) and the bump (`fb8e26b`); it was sitting on `main` before either.

One live lead, not yet chased: the test's own comment records a **recent asset-resolution fix**
("this site previously skipped whenever `GOINFER_PREQUANT_GGUF` was unset... under a bare `go test
./decoder`, this gate now RUNS where it used to skip") — meaning this real-checkpoint gate may have
been silently skipping for a long stretch, during which `parityWantInt4`'s golden could have gone
stale against real drift nobody was watching for. That is a hypothesis, not a finding — needs a real
bisect (not the two-point check done here) to find which commit actually broke it, or whether the
golden itself was simply never right. ~~**Unclaimed — pick up either box.**~~

**RESOLVED 2026-08-15 (`8f63a7d`, linux). The lead above was right, and it was the whole answer.**
The real bisect ran (474 revisions, 9 steps, isolated worktree): first bad commit **`7deb368`**
(2026-06-14) — *"integrate aikit v1.8.1 Qwen2.5-VL vision encoder"*. Its only code delta is
`go.mod`; aikit 1.7.3→1.8.1 carries two `linalg` commits that are not vision at all (`36ce824`
"fold W4A8 weight scales in-register", `52890f5` wiring it on NEON, both **aikit-repo** SHAs).
Folding the scales changes W4A8 accumulation, which moves a greedy continuation. **So: red for two
months, not two days** — consistent with the two-point check finding it already red at v1.17.1,
since v1.17.1 is far downstream of the actual cause.

**Answering the entry's own either/or: the golden was right when captured, and went stale — it was
NOT "never right".** And it is stale in the direction that matters. Scored by leading ids matching
an **f32 forward** of the same 0.5B on the same prompt: the new int4 path matches **11** of 24,
the pinned golden **5**, int8int8 (unchanged) **19**. The kernel made int4 *twice as faithful to
f32*; the gate was holding the *worse* path. Re-captured on that measurement rather than on the
gate being red — the distinction `parity-coverage-policy.md` draws between promoting a first-run
result and silencing a gate — and the identical mistake that file's own 2026-06-12 note records
for its predecessor. Second time on this gate.

**The finding worth keeping is not the fix.** The dark gate hid not just the failure but **when it
started**, and therefore what caused it — a dependency bump moved a numerics path with the one gate
watching it skipping. That is filed against the selector-coverage campaign in
`queue-engineering.md`, where "red for at least two days" is now corrected as the visible floor,
off by a factor of thirty.

