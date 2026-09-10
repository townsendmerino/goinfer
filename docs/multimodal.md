# Multimodal (vision-language) for goinfer — plan

> **Status (2026-09-07): P0–P4.5 SHIPPED (June); P5 HALF-LANDED; everything below §"2026-09
> update" is the June plan kept as the design record.** Where it stands today, in one paragraph:
> images work end-to-end for **one family** (Gemma 3, SigLIP tower, fixed 256-token block, base
> resolution, single image per turn) on the OpenAI and Anthropic surfaces and in `demo/agent`;
> Qwen2.5-VL loads its **text side only** (tiny-oracle parity, safetensors only — the image path
> and the GGUF `mmproj` seam are P5's open half). The vision tower runs on the **CPU** (~171 s/image,
> a pre-re-anchor June figure never re-measured) or on **WebGPU** (18.8 s, same caveat) — and WebGPU
> needs cgo, so **the cgo-free release binaries have no GPU vision at all**; a downloaded
> `goinfer-serve` does images at CPU speed on every platform. No audio in, no video, no image out.
> Nothing multimodal is in `serve check`, the fit guard, the recommendation registry, or the
> cold-user protocol. The next program is §"2026-09 update" below.
>
> The June status text, kept as written: P0–P3 (image→logits at HF parity) landed through
> `9412e4e`; P4 (serve vision API + agent image input) is a real user-facing feature —
> `cmd/serve` accepts images on both the OpenAI (`image_url`) and Anthropic (`image`) surfaces,
> base64/data-URI only, behind `--vision <dir>`; `demo/agent` takes a dropped/pasted image;
> loading a real `google/gemma-3-4b-it` works directly. The resident WebGPU SigLIP encoder is
> parity cosine 1.000000 (`886c8fd`/`5d7c572`, `docs/completed/task-gpu-vision-tower.md`); an
> int8 CPU tower was evaluated and is a wash on AVX2 (no VNNI;
> `docs/completed/task-cpu-vision-prefill.md`). Drafted 2026-06-10, revised after external review,
> re-grounded against the code the same day (two forward-path seams, not one; the `gemma3` gate
> pinned on text-only 270m). Audio/video were out of scope for v1. (This is the doc
> `benchmarks.md` references.)

## 2026-09 update — state, gaps, and the next program

### What changed under this doc since June

Three things that make the June plan's assumptions stale, in the direction of *more* reachable:

- **The engine is faster where the tower is slow.** The CPU prefill work of 2026-09-01 (f32
  attention default, A3 head fan-out, P19 fused schedule) and the CUDA tensor-core prefill
  (`attn_fused`, `gemm_w4a8_mma`) are exactly the shape of a ViT forward — a K≈4096 prefill —
  and **none of it reaches `multimodal/`**: the tower's attention is still the naive f32 loop the
  June doc flagged ("attention QKᵀ/scores·V are still naive f32"). The 171 s / 18.8 s figures
  predate all of it and predate the 2026-08-25 re-anchor; neither is a current number.
- **The families moved.** Gemma 4 — resident on all three backends, E2B/E4B on-device, the
  26B-A4B measured in §B4 — is natively multimodal on every size, with a **custom 16-layer
  vision encoder** (patch 16, 3×3 pooling, learned 2-D positions, a *variable* soft-token budget of
  70/140/280/560/1120 per image at native aspect ratio) and, on E2B/E4B/26B-A4B, an **audio
  encoder** (12 layers, conv subsampling, 128-mel input) — both injected through the same
  placeholder-token mechanism as images, with a `use_bidirectional_attention: "vision"` option for
  the image block. goinfer loads these checkpoints text-only (`gemma4_unified_text` drops
  `vision_config`). Qwen's VL line has moved to Qwen3.5/3.6-VL (the Qwen3.6-VL text side already
  loads); LFM2.5-VL-3B (SigLIP2 + the `lfm2` decoder now supported), Ministral 3 (Pixtral tower on
  the `ministral3` decoder just added) and Cohere's North Micro Vision are the small-edge entries.
- **The product surface grew and multimodal is absent from all of it.** `serve check` has no
  vision row; the fit guard prices weights and KV but not the tower or the image-token KV (Gemma 3:
  256 positions per image; Gemma 4: up to 1120); the recommendation registry lists no VL
  checkpoint; resident prefix reuse opts out of any turn with an image, so an agent loop that
  attaches a screenshot re-prefills the whole conversation every turn — the exact cost R-02/R-03
  removed for text.

### The gaps, ranked by who hits them

0. **RESOLVED 2026-09-08 (CUDA + WebGPU; Metal deferred).** `GenerateVL`/`GenerateQwenVL` used to
   be stateless and CPU-only by design (`decoder/generate_vl.go`'s old doc comments; the premise
   was inherited, not re-derived, when V-11 fixed the race around it — `docs/review-2026-09-04.md`)
   — they never touched `m.resident` at all, so on a GPU box an image turn ran the WHOLE turn on
   CPU, decode included, not only the tower. Fixed via a hybrid design: the CPU prefill (the
   bidirectional image-block attention mask has no resident equivalent) stays on CPU exactly as
   before, but its resulting KV is pushed into the resident GPU cache (`UploadKV`, extended with a
   `base` position — a sliding-window ring's live K/V can start at a nonzero absolute position
   once wrapped) and decode continues on GPU from there
   (`decoder/generate_vl_resident.go:residentUploadPrefill`, wired into both functions behind the
   existing `resBusy` claim). Qwen2.5-VL specifically needed one real kernel change on both
   backends — `Forward(embedding, pos)`'s single position can't serve m-RoPE decode, which needs a
   DIFFERENT rotation angle (`pos+mropeDelta`) than the KV-storage/attention position (`pos`) once
   decode moves past an image block (the merge-compressed image grid makes `mropeDelta` nonzero for
   any real image turn, not an edge case) — added as a new optional interface,
   `decoder.ResidentMRoPE.ForwardMRoPE(embedding, pos, ropePos)`, alongside the existing `Forward`
   rather than widening it (CUDA: a `ropePos` kernel argument threaded through `rope_kv`,
   `cuda/gemv_fwd.cu`, PTX regenerated and diff-verified isolated to that one kernel; WebGPU: no
   shader changes — its RoPE kernel is pure rotation with no KV-write side effect — just the
   `DecodeRunner` uniform-generation plumbing widened to carry a second position). Both new
   primitives (`UploadKV`'s offset, `ForwardMRoPE`'s angle split) pass their first real
   correctness tests on real hardware at cosine 1.0 (`cuda/uploadkv_parity_test.go`,
   `cuda/forwardmrope_parity_test.go`, and their WebGPU twins). **Measured real payoff, real
   checkpoint (Qwen2.5-VL-3B, real image, int4 both arms, interleaved-paired, median of 5):
   decode 5.95 → 22.98 tok/s, 3.86× — clears the pre-registered ≥1.5× bar by a wide margin** (full
   real-checkpoint parity gate: `cuda/qwen25vl_resident_real_test.go`, cosine 0.99+/exact argmax on
   a forced-trajectory decode step past a real image block, mropeDelta confirmed nonzero so the
   m-RoPE split is genuinely exercised). **Gemma 3 (`GenerateVL`) closed 2026-09-08**: needed no
   m-RoPE work — plain `Forward` suffices — real-checkpoint gate built (`scripts/pin_gemma3_real.py`
   + `cuda/gemma3_resident_real_test.go`, a pre-sized 896×896 test image so goinfer's bilinear
   resize vs HF's bicubic is a non-issue), decode-step cosine **0.998167**, exact argmax, tighter
   than Qwen's (no m-RoPE noise to contend with). **Measured decode-only speedup (text-decoder
   prefill+decode isolated from the vision tower, which is unaffected by this change either way):
   6.96 → 92.27 tok/s, 13.26×** — larger than Qwen's, consistent with Gemma 3 having no m-RoPE
   overhead on the resident path. **Metal is out of scope**: `UploadKV` is unimplemented there
   (`metal/backend.go`), so this design doesn't reach it; a real gap for whoever picks up Metal
   residency next, not attempted here.
   **UPDATE 2026-09-08: a cold (first-time) image turn's own PREFILL, not just decode, is now
   also resident (CUDA/Gemma-3 only).** Gap 0 as shipped deliberately kept CPU prefill (no
   resident equivalent of the bidirectional image-block mask existed) and only moved decode; this
   left the CPU prefill itself — measured **8.215 s** in isolation — as the largest remaining
   cost on a cold turn (dwarfing gap-0's own decode fix). Closed via a new resident CUDA kernel
   (`cuda/attn_img_prefill.cu`, `attn_img_batched`) that runs the SAME bidirectional attention
   `decoder/kvcache.go`'s CPU reference computes — `decoder.ResidentImagePrefill`, v1 requires the
   whole prompt to fit in one weight-stationary pass (declines to the unchanged CPU-prefill+
   `UploadKV` bridge otherwise). **Measured: 8.165 s → 0.367 s, 22.27×** (prefill only, tower
   excluded — `docs/benchmarks.md` "Resident image-block PREFILL, finishing gap 0"). Real-checkpoint
   gate: cosine 0.997042, exact argmax, matched int4 precision — `cuda/gemma3_img_prefill_resident_real_test.go`.
   Combined with the decode fix above, a cold Gemma-3 image turn's non-tower cost drops from ~8.2 s
   to ~0.37 s; the vision tower (~31.3 s, unaffected) is now essentially the whole cost of a cold
   turn. Out of scope for this pass: the `attn_fused` L2 tensor-core path (its tile-level
   aggregates assume monotonic per-row key counts, which an image block breaks — low priority, it
   never engages below the 512-token `fastPrefillFloor` on either shipped VL fixture), and
   Metal/WebGPU (same reasons as gap 0's own decode fix).

   **UPDATE 2026-09-08 (same day, follow-on pass): Qwen2.5-VL's own version of this gap is also
   closed, and multi-chunk image-call declines are far rarer.** Qwen's image tokens already attend
   causally in prefill (no new mask kernel needed — confirmed, `attn_batched` serves it unmodified),
   but its m-RoPE 3-component rotation wasn't servable by the existing batched-prefill rope kernel —
   and, a real subtlety, even ORDINARY text rows after an image block needed the fix too, since
   `mropePositions` compresses their position by the merged image grid rather than counting
   sequentially. New kernel `cuda/rope_mrope_prefill.cu` (`rope_kv_mrope_batched`) +
   `decoder.ResidentMRoPEPrefill` close it — **measured 711.2 ms → 19.7 ms, 36.14×** (prefill only,
   `docs/benchmarks.md` "Resident m-RoPE PREFILL (Qwen2.5-VL)"); real-checkpoint gate cosine
   0.993995, exact argmax (`cuda/qwen25vl_mrope_prefill_resident_real_test.go`). Unlike Gemma-3's
   bidirectional image block, m-RoPE has no cross-row attention coupling, so it chunks cleanly —
   `PrefillMRoPELast` reuses the ordinary chunked-prefill loop rather than declining outright.
   Separately, `PrefillImageLast`'s own chunk floor was raised from the shared 512-row text default
   to a dedicated 2048-row budget (`prefillImageChunkRows`, justified by an existing real
   measurement on this box at that width) — most real image+chat prompts no longer hit the decline
   at all, though a bidirectional image block spanning more than one pass remains structurally
   blocked regardless of the floor (chunk N's in-block rows need chunk N+1's not-yet-computed K/V
   at every layer — no pass ordering can supply that). Remaining out of scope: `attn_fused` L2 for
   image blocks (as above), Metal/WebGPU.
1. **A downloaded binary cannot use the GPU for images** (cuda/metal have no vision tower; WebGPU
   is cgo). Every Mac and Linux user of the release gets ~minutes per image.
2. **Vision is one family.** ~~Gemma 4 — the family most of the resident work went into — is
   text-only here~~ **CLOSED for CPU serving 2026-09-09/10 (P7 Phases B+C): Gemma 4 E2B/E4B
   (Phase B) AND 26B-A4B/31B (Phase C, the bidirectional-block-attention case) now serve real
   images, CPU-only, both gated at cosine 1.0 on real end-to-end fixtures** — see §P7 below for
   scope (GPU-resident decode remains open; Phase C's real-26B-A4B end-to-end validation is
   deferred, the tiny fixture is what validates it this pass). Qwen-VL, the most-pulled VL line, is still
   text-only here.
3. **No GGUF `mmproj`.** Ollama/llama.cpp users have their VL models as GGUF + mmproj; goinfer
   reads neither half of that pair for vision.
4. **Image turns defeat prefix reuse and the fit guard.** Agent harnesses with screenshots are the
   multimodal use case that matters, and both of the pieces built for agent loops skip them.
5. **No audio in.** Gemma 4's audio encoder makes this a tower port on a family already resident,
   not a new architecture.

### The program — P6 to P11, each independently shippable, each with a gate that can fail

The rule for every phase, unchanged from June: text-only behaviour is bit-identical before and
after (the parity harness is the gate), every new stage is pinned against HF in isolation, and no
number is published without provenance.

- **P6 · The tower runs on the kernels the engine already has.**
  **P6a (CPU) — DONE, 2026-09-08 (aikit v1.38.0).** Both towers (SigLIP/`encoder.go`,
  Qwen2.5-VL/`qwen_encoder.go`, both live in `aikit/vision`, not goinfer) route attention through
  `linalg.AttendTileFused`, head-parallel via a per-worker serial Workspace — the same schedule
  and fan-out `decoder/forwardn.go`'s `attendBatchedHeads` already uses for text prefill. Gate
  held: `TestSiglipEncoder_parity`/`TestQwenVisionEncoder_parity` cosine 1.0 both, `-race` clean.
  **Measured against the pre-registered rule above, and the rule's own escape clause fired: CPU
  moved under 2× — it moved ~0% (31.38s→31.26s SigLIP, 156.9ms→157.0ms Qwen2.5-VL, interleaved
  A/B, both within noise) — so stated here and tuning stopped, per the rule.** `docs/benchmarks.md`
  §A "Vision tower CPU prefill" has the full writeup and a plausible (unconfirmed) reason: the old
  per-head loop's `MatmulBT` already parallelized internally across the whole core count, one head
  at a time; the new schedule parallelizes across heads instead, each single-threaded — at head
  count ≈ core count on this box, total work is close to unchanged, just redistributed. Kept
  anyway: parity holds, it's the proven decoder mechanism, and the removed memory materialization
  may still matter at an untested config (more heads than cores, memory pressure).
  **CUDA (SigLIP/Gemma-3) — DONE, 2026-09-08. Metal — still not started.** The doc's own earlier
  framing here overstated the gap, found by reading the actual kernels rather than assuming: a
  FULLY non-causal attention kernel turned out to need ZERO new kernel source at all —
  `attn_img_batched` (already shipped this session, for Gemma-3's TEXT-decoder image-block
  attention) called with `imgStart=0, imgEnd=M` already IS full bidirectional attention over every
  patch, reachable via parameters alone; and the "int4-only, no batched-M fallback" claim was true
  only of the specialized tensor-core MMA tier, not the plain int8 batched GEMV
  (`gemv_w8a8_batched`) already used for ordinary int8 text prefill at any M. What genuinely was
  missing — SigLIP uses LayerNorm (not RMSNorm) and a plain non-gated GELU-tanh MLP, neither of
  which exists anywhere else in this codebase — needed exactly two new, small kernels
  (`layernorm_quant.cu`, `gelu_quant.cu`), not a new attention or GEMM primitive. Plugs into
  `aikit/vision`'s ALREADY-BUILT `ResidentEncoder`/`RegisterResident` seam (the same one WebGPU's
  own resident tower uses) — `cuda/vision_encoder.go` + `cuda/vision_register.go`, ~40 lines of
  registration glue, mirroring `gpu/vision_register.go` almost verbatim.

  **Measured: 1.58× (41.3s→26.1s median, real gemma-3-4b-it tower, matched int8 precision,
  interleaved).** Real-checkpoint gate cosine 0.91-0.96 (int8-vs-int8, not int4-vs-f32) —
  confirmed genuine via a matched-precision CPU probe reproducing one layer's raw GEMV output at
  cosine 1.000000, not a wiring bug (a REAL bug — the position-embedding add broadcasting row 0 to
  every patch — was found and fixed along the way; see `docs/benchmarks.md` "Resident CUDA vision
  tower" for the full writeup, both numbers, and the specific host-round-trip residual-add
  optimization named there as the natural next lever if the modest 1.58× ever needs to close
  further toward WebGPU's own ~9×). Metal, and Qwen2.5-VL's own (differently-shaped) tower, remain
  explicitly out of scope for this pass.
- **P7 · Gemma 4 vision (all sizes) and audio (E2B/E4B/26B-A4B).** Phase 0 from the real
  `Gemma4VisionConfig`/`Gemma4AudioConfig` and modeling file, not the summary above: the encoder
  block, the 3×3 pooling to soft tokens, the position table, the variable token budget and its
  aspect-ratio rule (dimensions divisible by 48), the projector into the text hidden size, how PLE
  treats multimodal positions (context-aware projection only — no token-identity term), and the
  `use_bidirectional_attention` semantics. Then the image path on the descriptor pattern (the two
  June seams already exist and are inert-gated), then audio as the same pipeline with a mel
  front-end (128 features, conv subsampling) and its own placeholder tokens. Gates: stage-isolated
  pins (preprocess, encoder, projector, end-to-end argmax + cosine on a fixed image and a fixed
  clip) against `Gemma4ForConditionalGeneration`; the E2B/E4B cells on the Mac, 26B-A4B on the
  Linux box under expert streaming. **One family buys images on every size goinfer already runs
  resident, and audio-in, without a new decoder.** Audio is scoped to *input* (transcribe/understand);
  speech output stays out.

  **Phase 0 DONE, 2026-09-09 — read against the real `modular_gemma4.py`/`configuration_gemma4.py`
  source, not assumed.** First pass mistakenly researched `gemma3n` (a different, older, unrelated
  family) on an unverified naming assumption; caught via a checkpoint-size mismatch against
  `testdata/parity_manifest.json`'s own "google/gemma-4 E2B + 12B" citation, then corrected by
  confirming `gemma4`/`gemma4_unified` are real, separate transformers model directories. **This
  doc's own scoping above was already right** — it names `Gemma4VisionConfig`/`Gemma4AudioConfig`
  correctly; the mixup was this session's research, not a stale doc.

  Confirmed facts: real checkpoints are E2B/E4B/26B-A4B(MoE, 128 experts top-8)/31B under
  `model_type: gemma4` (`text_config.model_type: gemma4_text`) — this is a DIFFERENT, separate
  family from the 12B checkpoint (`model_type: gemma4_unified`), which has **no vision or audio
  tower at all** (`del self.vision_tower` in `Gemma4UnifiedModel.__init__`; raw patches/waveform
  through a linear embedder instead) and is out of scope for this phase's "real encoder" framing —
  goinfer's existing `gemma4Architecture` text decoder already covers both families' text side
  (spot-checked: per-type KV-sharing, `layer_scalar`, scale-less `v_norm` all match real source
  exactly), so P7 is *purely* the vision/audio build, no text-decoder work needed.

  Vision tower (E2B/E4B/26B-A4B/31B): `Gemma4VisionEncoder`, 16 layers, hidden 768, 12 heads,
  head_dim 64, intermediate 3072 — a real transformer, LayerNorm not RMSNorm (same primitive P6
  just built CUDA kernels for), patch_size 16 × pooling_kernel_size 3 = 48px confirming the
  "divisible by 48" rule exactly. Soft-token budget is one of a fixed small set `{70, 140, 280,
  560, 1120}` chosen by an aspect-ratio tiling rule, not a single constant. Position embedding is
  a learned absolute table (`position_embedding_size=10240`) — **one unresolved detail**: the
  config declares `default_rope_type="axial"` but at least one real checkpoint's `rope_parameters`
  shows `{"rope_theta":100.0,"rope_type":"default"}`; `apply_multidimensional_rope` and
  `Gemma4VisionRotaryEmbedding(Sam3ViTRotaryEmbedding)` both exist in source and this wasn't fully
  reconciled — read `modeling_gemma4.py`'s actual vision forward directly before building the
  position path, don't assume from the config alone.

  Audio (E2B/E4B/26B-A4B/31B): `Gemma4AudioModel`, 12-layer Conformer-style, hidden 1024, heavily
  left-biased chunked local attention (`attention_chunk_size=12, context_left=13, context_right=0`
  — near-causal), conv subsampling front-end (`subsampling_conv_channels=[128,32]`),
  `output_proj_dims=1536` (matches E2B text hidden). Exact mel-spectrogram parameters (sample
  rate/hop/n_mels) not yet read — `feature_extraction_gemma4.py` is the source, deferred to when
  audio work actually starts.

  Multimodal splice: `masked_scatter`, confirmed directly (3 call sites: image/video/audio) in
  `Gemma4Model.forward`. PLE's multimodal-position treatment is exactly as scoped above — HF's
  `project_per_layer_inputs` function (in the upstream `transformers` package's
  `modular_gemma4.py`, not part of this repo) returns the context projection alone when
  `per_layer_inputs is None`. `use_bidirectional_attention` is real and checkpoint-gated (`null`
  for E2B/E4B, `"vision"` for 26B-A4B/31B) — this mask is built only in the multimodal
  `ForConditionalGeneration` forward, never in the plain text-only path, so it is new decoder-side
  wiring, not something the existing text gate already exercises.
  **CORRECTED 2026-09-10** (the line above originally read differently and was wrong on two
  counts, found by reading the actual installed `transformers` source rather than trusting the
  earlier citation): `create_masks_for_vision_model` **does not exist** in that checkout at all —
  the real mechanism is `create_masks_for_generate` → `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING` →
  `create_causal_mask` / `create_sliding_window_causal_mask` (both in `masking_utils.py`). And
  **global/full-attention layers do NOT stay strictly causal** — both layer types receive the
  identical `OR(windowed-or-plain causal, blockwise)` treatment, with no layer-type gate anywhere
  in the call chain: full-attention is `causal(q,kv) OR (block[q]==block[kv] AND block[q]>=0)`;
  sliding-attention is `(within_window(q,kv) AND causal(q,kv)) OR (block[q]==block[kv] AND
  block[q]>=0)` — the blockwise OR is applied at the top level in both cases, so a same-block key
  outside the sliding window (or, on a full-attention layer, causally in the future) is still
  attended. P7 Phase C (`decoder/forward_gemma4_batched.go`) implements exactly this, gated at
  cosine 1.000000 against a real HF forward on a fixture designed to fail under either the wrong
  (old) claim or a causal-only forward — see that file's own doc comments for the correctness
  proof (a query's valid key range is always a single contiguous interval, never two disjoint
  ones) and `scripts/pin_gemma4_vl_bidir_image.py` for the discriminating fixture.

  Video: frame-sampling into the same `embed_vision(...)` image path per-frame, not a separate
  temporal mechanism (confirmed for `gemma4_unified`; the plain `gemma4` family reuses the same
  `convert_video_to_patches`/`pad_to_max_patches` helpers, strongly suggesting the same pattern,
  not independently confirmed by reading `gemma4`'s own `get_video_features` body).

  **Two corrections to the Phase 0 summary above, found during Phase A implementation — both
  matter for anyone reading this doc rather than the source:** (1) the vision tower is **RMSNorm
  sandwich-norm (four independent norms per layer) + a GATED SwiGLU-tanh MLP, NOT LayerNorm +
  plain GELU** as first written — P6's SigLIP kernels do not transfer here; this tower instead
  reuses the RMSNorm/gated-GLU primitives every text decoder in this repo already has. (2) the
  axial-rope "unresolved detail" is resolved: **both** the learned absolute position table **and**
  axial 2-D rope are real and both fire (θ=100, two independent 32-wide rotate-half chunks, one per
  axis, sharing 16 log-spaced frequencies) — the `"rope_type":"default"` seen in a raw config.json
  is a harmless BC shim (`standardize_rope_params`) rewritten to `"axial"` at config-load time, not
  a real ambiguity. (3) PLE's multimodal treatment is NOT "context projection alone" as Phase 0
  said — the real multimodal forward substitutes the **PAD token's** id/embedding at every
  image/video/audio position BEFORE computing PLE's token-identity term (HF's
  `Gemma4Model.forward`, upstream `modeling_gemma4.py`, not part of this repo), so the term is
  neither skipped nor zeroed.

  **Phase A (vision, image path, CPU) DONE, 2026-09-09 — the "gemma4" plain family
  (E2B/E4B/26B-A4B/31B), NOT `gemma4_unified`.** New `vision.Gemma4Encoder` +
  `vision.Gemma4Preprocess` in aikit (`~/mycode/aikit/aikit/vision/gemma4_encoder.go`,
  `gemma4_preprocess.go` — aikit v1.38.0+, not yet released as a tagged version; goinfer's
  `go.work` points at the local checkout for now). Every load-bearing detail was checked against
  the REAL E2B-it safetensors header, not assumed from source reading alone — this caught two
  things a source-only read would have missed: `use_clipped_linears=true` on the real checkpoint
  (genuine finite per-tensor clamp bounds like `[-6.375,6.3125]` on every attention/MLP projection,
  not the harmless ±inf default `Gemma4ClippableLinear` falls back to), and the position table's
  real on-disk shape (`[2,10240,768]`, one tensor, not two).

  Gated at cosine **1.000000000** against a tiny-random `Gemma4VisionModel`+`Gemma4MultimodalEmbedder`
  checkpoint (exercises every component including the real clamp — `scripts/pin_gemma4_vision.py`),
  and cosine **1.000000000** (max|diff| ≈6.9e-6, pure f32 rounding noise across 16 layers) against
  the REAL `google/gemma-4-E2B-it` vision-tower weights on synthetic patches, matched f32 precision
  both sides (`scripts/pin_gemma4_vision_real.py`). The aspect-ratio tiling rule
  (`Gemma4AspectRatioSize`) is cross-checked numerically against the real
  `get_aspect_ratio_preserving_size` formula on five width/height/budget combinations — exact
  match. Pixel-level resize parity (PIL bicubic vs this repo's bilinear) is a known, explicitly
  deferred gap, same as the existing SigLIP preprocessing's own documented gap — not attempted here.

  Decoder-side wiring (goinfer's own tree, not aikit): `runLayersGemma4` split into a thin wrapper
  plus `runLayersGemma4FromEmbed(h []float32, pleTokenID int, cache *KVCache)` (`decoder/
  forward_gemma4.go`) — the "embed-by-vector" seam this family lacked (the June seams,
  `runLayersFromEmbed`/`runLayersFromEmbedN`, only reach the GENERIC forward path; gemma4's own
  path never went through them). `pleTokenID` lets a caller feed the checkpoint's `pad_token_id`
  (new `Config.PadTokenID`/`gemma4Params.PadTokenID`, `json:"pad_token_id"`, picked up automatically
  from `text_config` by `loadConfig`'s existing merge) at a multimodal position's PLE
  token-identity lookup, matching HF's real substitution exactly. Verified two ways against the
  real E2B GGUF: (1) `runLayersGemma4FromEmbed` reproduces `runLayersGemma4` BIT-FOR-BIT on the
  ordinary text path (a real regression check, not just "no build error") — proven by every
  existing gemma4 forward-parity gate staying green post-refactor, including the real E2B GGUF
  gate at cosine 0.99924 (unchanged from before), plus a dedicated equivalence test; (2) PAD
  substitution genuinely changes the output relative to the real-token-id path — proving the new
  branch is actually wired, not a silent no-op (`TestGemma4RunLayersFromEmbed_matchesTokenPath`).
  One simplification worth flagging: E2B/E4B ship `use_bidirectional_attention: null`, so the
  layer-type-aware image-block bidirectional mask (`AND(sliding_window, OR(causal, blockwise))`,
  global layers stay causal) is **not** built yet — E2B doesn't need it, and there's no real
  checkpoint locally to gate it against (only 26B-A4B/31B set `"vision"`). Deferred to whenever
  that checkpoint is available.

  **What Phase A does NOT include, deliberately scoped out of this pass**: the full serving/
  generation integration (placeholder-token constants, a `multimodal.Gemma4ImageBlock`-style
  prompt splice, a `GenerateGemma4VL`-shaped driving loop, HTTP wiring in `internal/serveapp/
  vision_serve.go`) — the vision-tower primitive and its decoder-side hook are proven correct in
  isolation; wiring them into an end-to-end image-in-prompt request is real, separate work, sized
  similarly to Qwen2.5-VL's own `vision_serve.go` integration, and is the natural next slice.
  Video (Phase C in the roadmap above) and audio (Phase D/E) remain untouched.

  **Phase B (serving integration) DONE, 2026-09-09 — CPU-only, E2B/E4B-class (causal) v1.** Wired
  the Phase A tower into real image-in-prompt requests: `multimodal.Gemma4ImageBlock`/
  `Gemma4PooledTokens` (the real tokenizer sentinels — `<|image>`/`<|image|>`/`<image|>`, verified
  against a real E2B-it checkpoint's `tokenizer.json`/`tokenizer_config.json`, not guessed from
  Gemma 3's symmetric naming), `decoder.GenerateGemma4VL` + `prefillLogitsGemma4VL` (a new file,
  `decoder/generate_gemma4_vl.go`), and the `internal/serveapp` wiring (`loadedModel.gemma4Enc`,
  `visionCapable()`'s third disjunct, `driveVL`'s third dispatch arm, `main.go`'s
  `loadGemma4VisionTower`, both auto-discovery and explicit `--vision <dir>`).

  **The scoping call, and why it's not an approximation for what it covers**: Gemma 4 is an
  "own-forward" family with no batched embed-by-vector hook (unlike Qwen/Gemma 3, whose CPU image
  prefill runs one batched pass via the generic `runLayersFromEmbedN`) — only the single-token
  `runLayersGemma4FromEmbed` exists. `prefillLogitsGemma4VL` walks the prompt ONE TOKEN AT A TIME
  through it instead. That is not a shortcut for E2B/E4B specifically: those checkpoints ship
  `use_bidirectional_attention` unset, so the real HF forward already attends to the image block
  strictly causally — a sequential walk **is** that forward, not a stand-in for a bidirectional
  one. 26B-A4B/31B (`use_bidirectional_attention: "vision"`) need a genuinely different, blockwise-
  masked batched forward this does not implement; `loadGemma4VisionTower` refuses those checkpoints
  at load time with a clear error rather than silently serving the wrong mask. No GPU-resident
  bridge either — Gemma 4's resident CUDA decode has a KV layout (cross-layer sharing on its tail
  layers) the generic `UploadKV`/`residentUploadPrefill` bridge was never verified against; deferred
  as separate, GPU-verified follow-up, mirroring how Gemma3/Qwen's own resident bridge ("gap 0")
  was itself a later pass after their original CPU-only `GenerateVL`/`GenerateQwenVL` shipped first.

  **Gated at cosine 1.000000, exact argmax** against a real, non-degenerate tiny
  `Gemma4ForConditionalGeneration` fixture (`scripts/pin_gemma4_vl_tiny.py` +
  `scripts/pin_gemma4_vl_image.py`, `decoder/gemma4_vl_test.go`) — both the text-only path and the
  full image→embed→splice→logits path. The fixture deliberately carries `num_kv_shared_layers=2`
  (of 4 layers, alternating sliding/full types) and norms/layer_scalar strengthened away from HF's
  identity-init defaults, not an incidental choice — see the next paragraph.

  **Two real, pre-existing bugs found and fixed along the way, both in `decoder/weights.go`'s
  safetensors gemma4 loader — found only because this session is the first time a real gemma4
  safetensors checkpoint (rather than a tiny fixture or a GGUF) was pushed through the full decoder
  end to end.** (1) The loader unconditionally tried to load `k_proj`/`v_proj`/`k_norm` for EVERY
  layer; a real checkpoint with `num_kv_shared_layers > 0` (confirmed: `google/gemma-4-E2B-it` has
  20 of 35 layers with no such tensors at all — they reuse an earlier layer's KV, exactly the
  mechanism `forward_gemma4.go`'s `kvSrc` already implements on the FORWARD side) crashed on load
  with a missing-tensor error. GGUF's loader (`decoder/gguf.go`) already had this right; the
  safetensors path mirrors it now. (2) A real checkpoint can vary `intermediate_size` by layer (the
  same E2B checkpoint: layers 0-14 are 6144-wide, layers 15-34 — exactly the KV-shared tail — are
  12288-wide), but `config.json` carries only one scalar and, unlike GGUF, has no per-layer array to
  read; the loader now discovers each layer's real width from its own tensor's on-disk shape (a
  cheap header lookup) and seeds `arch.gemma4.FFNPerLayer`, which the forward path's `arch.ffnAt(i)`
  already expected to exist. Both fixes are proven numerically inert for every existing gemma4
  golden (`scripts/refresh_parity_hashes.sh`'s self-verifying run: 52 forward goldens green, 0
  failed) and are the exact shape the new tiny VL fixture above deliberately exercises.

  **A third, separate, pre-existing gap found the same way, deliberately NOT fixed here: safetensors
  Per-Layer-Embedding (PLE) loading was never implemented** (`decoder/weights.go`'s own comment
  already called this "Phase 4" before this session touched it) — `PerLayerTokenEmbed`/
  `PerLayerModelProj`/`PerLayerProjNorm`/per-layer `PLEGate`/`PLEProj` are loaded by GGUF's loader
  and nowhere in the safetensors one. Left alone, a checkpoint with `hidden_size_per_layer_input >
  0` (which `google/gemma-4-E2B-it` has: 256) crashed deep in `rmsNorm` on a nil norm weight during
  the FIRST generation, not at load time. Fixed only the failure mode, not the gap: the safetensors
  loader now refuses such a checkpoint loudly at load time with a clear "not implemented yet"
  error, instead of loading "successfully" and crashing the process on the first request. **Every
  real gemma4-vision-capable checkpoint checked this session (E2B, 26B-A4B) has PLE** — the E-model
  family appears to carry it by design, not as an optional feature — so this is now the blocker on
  any REAL-checkpoint validation of the vision-serving path (the tiny fixture above sidesteps it by
  construction, `hidden_size_per_layer_input=0`, and is what validates correctness instead).
  Implementing safetensors PLE loading is real, separate, undone work — whoever picks it up next
  unblocks real-checkpoint gemma4 vision validation as a side effect.

  **What Phase B does NOT include**: the real-checkpoint end-to-end gate (blocked on the PLE gap
  above), an HTTP-level integration smoke test through `vision_serve.go`'s actual splice/encode path
  (the decoder-level gate above proves the numerics; the HTTP wiring itself is exercised only by
  reading the code, not yet by a running request), and — as already covered — 26B-A4B/31B's
  bidirectional attention (**closed by Phase C, next**), GPU-resident decode, video, and audio.

  **Phase C (26B-A4B/31B bidirectional-block attention) DONE, 2026-09-10.** A query position
  INSIDE an image/audio block must see LATER block positions too — impossible in Phase B's
  sequential, one-token-at-a-time forward, since later positions' K/V don't exist yet when an
  earlier position is processed. Building this turned out simpler than expected, for three facts
  confirmed by reading the code rather than assumed: (1) gemma4 has no ring-buffer/physical KV
  eviction at all (`decoder/model.go`'s `enableRings`/`setQuant` calls both exclude it — every
  layer is append-forever), so there's no ring/`base`-offset machinery to replicate from the
  generic family's own batched path; (2) the existing single-query-row attention kernel
  (`gemma4Attend`) already accepts an arbitrary contiguous absolute key range into the full cache
  arrays, so it needed zero changes; (3) a query's valid key range is PROVABLY always a single
  contiguous interval, never two disjoint ones (see `decoder/forward_gemma4_batched.go`'s
  `gemma4AttendRange` doc comment for the proof) — no new interval-union data structure needed.
  Net: one new file (`decoder/forward_gemma4_batched.go`: `gemma4AttendRange` +
  `runLayersGemma4FromEmbedN`, the batched twin of `runLayersGemma4FromEmbed`) plus a
  `prefillLogitsGemma4VLBidirectional` entry point and a one-line dispatch in `GenerateGemma4VL`
  keyed on `UseBidirectionalAttention != ""` — **zero changes** to the shipped E2B/E4B sequential
  path, `decoder/kvcache.go`, or the loader/`Architecture` layer.

  **Gated at cosine 1.000000, exact argmax**, on a real, non-degenerate, self-verifying-non-vacuous
  fixture (`scripts/pin_gemma4_vl_bidir_tiny.py` + `pin_gemma4_vl_bidir_image.py`,
  `decoder/gemma4_vl_bidir_test.go`): `layer_types` mixing both sliding and full attention (the
  masking correction above applies to both), an image block deliberately longer than
  `sliding_window` and positioned so the block's own start falls outside the last block position's
  causal window — the exact shape that distinguishes a real blockwise-OR-causal mask from a
  causal-only forward (measured: bidirectional vs causal-only last-logit cosine 0.953, argmax 142
  vs 115 — genuinely different, so the gate is not vacuous) or from a narrower, window-bound
  approximation. Dense-only (`enable_moe_block=False`): the masking change is orthogonal to FFN
  dispatch (`gemma4MoEFFN` is position-independent, confirmed by reading it — no position argument
  anywhere), already covered by `TestGemma4MoE_forwardParity`; MoE+bidirectional joint verification
  is exactly what the still-deferred real-26B-A4B gate would add. Text-only input on a
  `"vision"`-mode checkpoint is also gated: HF degrades `block_sequence_ids` to all `-1` when there
  is no multimodal content (confirmed by reading `Gemma4Model.forward`), which makes the blockwise
  term unconditionally false — so this is a real regression check that the new batched path agrees
  with both HF and the unmodified sequential path on plain text (measured: sequential-vs-batched
  logit cosine 1.00000000 on this fixture).

  **A separate, pre-existing bug found and deliberately NOT fixed here**: the already-shipped
  generic-family batched vision path (Gemma3/Qwen, `decoder/kvcache.go`'s `SetImageBlocks`/
  `attendHi`) never applies the `min(lo, blockStart)` correction the proof above derives — it
  under-attends a sliding layer's own image block whenever the window is narrower than the query's
  depth into the block. Out of scope (risks the shipped Gemma3/Qwen path, needs separate
  verification against Gemma3's own masking semantics) — filed here as a known gap, not fixed.

  **What Phase C does NOT include**: real-checkpoint validation against `~/models/gemma-4-26b-a4b-it`
  (a real 128-expert MoE checkpoint — large and slow to validate on CPU, deliberately deferred; the
  tiny fixture is what validates correctness this pass), the `attendHi` under-attend bug above, and
  — unchanged from Phase B — GPU-resident decode, video, and audio.
- **P8 · Finish P5: Qwen3.x-VL image path + GGUF `mmproj`.** Phase 0 on the real Qwen3.6-VL config
  (the vision encoder, dynamic resolution and patch grids, m-RoPE's three position components
  which the text path already degenerates correctly, any DeepStack-style multi-level injection —
  verify, don't assume). The `mmproj` companion-file seam lands here because this is the family
  Ollama users have as GGUF+mmproj: `--model x.gguf` auto-discovers the sibling mmproj (or takes
  `--mmproj`), reads its tensors into the same tower descriptor, and the loader refuses a
  mismatched pair by tensor-shape check rather than by filename. Gate: end-to-end parity on the
  safetensors path AND bit-identity between the safetensors tower and the mmproj tower on the same
  checkpoint (they are the same weights in two containers).

  **Phase 0 DONE, 2026-09-08 (text decoder only, no vision) — verified "don't assume" against real
  HF `transformers` source, not a description.** DeepStack is REAL and changes the injection shape:
  the vision tower taps hidden states at 3 intermediate layers (`deepstack_visual_indexes`, default
  `(8,16,24)` of 27), each through its own merger, and the text decoder ADDS the result into the
  residual stream at each of the FIRST 3 decoder layers (`hidden_states[visual_pos_masks] +=
  visual_embeds`) — not the single splice-before-layer-0 every other VL family here uses. The vision
  tower also gained a learned absolute position embedding on top of its axial rotary one (confirmed
  absent in Qwen2.5-VL). m-RoPE's frequency→component layout changed from Qwen2.5-VL's contiguous
  blocks (`[TTT…HHH…WWW]`) to a per-frequency-index INTERLEAVED strided layout
  (`recomposition_frequencies`) — a different, incompatible formula, not a parameter tweak.

  Given that scope, Phase 0 shipped ONLY the text decoder — `qwen3_vlArchitecture`
  (`decoder/registry.go`, aliases Qwen3's dense attention shape: per-head q/k RMSNorm, GQA, no
  q/k/v bias — confirmed from `Qwen3VLTextAttention` source, NOT Qwen2's shape, which
  `qwen2_5_vlArchitecture` aliases) + the interleaved m-RoPE component formula
  (`mropeComponentInterleaved`, `decoder/rope.go`, pinned directly against the real HF slicing
  logic, not a re-derivation of it). Tiny-golden text-only parity: cosine 1.0, exact argmax +
  continuation (`decoder/qwen3vl_test.go`, `scripts/pin_qwen3vl_tiny.py`). Real-checkpoint gate
  written (`decoder/qwen3vl_real_test.go`, `scripts/pin_qwen3vl_real.py`,
  `GOINFER_QWEN3VL_2B`/`testdata/assets.json`) but **not yet run — `Qwen/Qwen3-VL-2B-Instruct` is
  not present on this box as of 2026-09-08**; the gate skips cleanly rather than being silently
  absent, and this is a real, tracked gap, not a forgotten one.

  **Explicitly out of scope for Phase 0, named so they don't get lost**: the vision tower + DeepStack
  injection (needs a new decoder-forward hook for additive injection at N early layers — nothing
  here does that yet — plus a new `aikit/vision` encoder: axial rotary + interpolated learned
  pos-embed + multi-tap mergers, a cross-repo undertaking); GGUF `mmproj` (llama.cpp already
  supports Qwen3-VL — `PROJECTOR_TYPE_QWEN3VL` in `clip.cpp` — but `aikit/vision`'s loading path is
  safetensors-specific by construction, needing a new tensor-source abstraction); the MoE variants
  (30B-A3B/235B-A22B) — Phase 0 only targets the dense sizes.
- **P9 · Image turns in the agent loop.** (a) Prefix reuse over image blocks: an image's embedding
  block is a pure function of its bytes and the tower, so key the resident bookkeeping on a hash of
  the image bytes standing in for a token id at each placeholder position — a reused prefix with an
  unchanged image is then exactly the text case, and a changed image invalidates from its first
  position. Gate: the reuse invariant test extended with an image turn (reused == cold, bitwise),
  plus the agent-turn TTFT cell with a screenshot attached. (b) The fit guard prices the tower
  (resident weights) and image-token KV at the family's per-image token count × images per turn;
  request-time admission (task-fit-to-hardware Phase 1, from first-hour R13) counts image tokens
  in the prompt.

  **(b) DONE, 2026-09-08 — turned out to be one fix, not two.** Investigated as two separate asks
  and found both collapsed into the SAME pre-existing, non-vision-specific gap:
  `decoder/fitguard.go`'s `fitCheckFor` priced weight/KV bytes for `.gguf` paths only — any
  safetensors directory (which is EVERY currently-working vision-language load in this project;
  Qwen2.5-VL's own GGUF path is text-only, no mmproj) returned an unconditional zero-weight,
  always-fits check, vision or not. Closed generally: `estimateSafetensorsWeightBytes` prices a
  safetensors checkpoint the same way `estimateGGUFWeightBytes` always priced a GGUF one — shape-
  only, quant-independent tensor element counts, NEVER on-disk file size (a bf16 checkpoint shrinks
  several-fold once quantized on load, so file-size pricing would refuse loads that fit
  comfortably — the exact "wrong in the refusing direction" failure this guard's own design
  principle rules out). Because the estimator sums every tensor in the checkpoint uniformly, a
  bundled vision tower (Gemma 3: confirmed 50 `vision.*`/`multi_modal_projector.*` tensors present
  and counted) gets priced automatically — no vision-specific code needed in the guard at all.
  Image-token KV needed nothing new either: unpinned loads already price KV at the model's full
  `MaxPositions` (R13), the request-time worst case regardless of whether those positions hold text
  or image placeholders; and request-time admission (`contextLengthError`/`clampMaxTokens`,
  `internal/serveapp/openai.go`) already ran on the fully placeholder-expanded prompt (`vi.ids`,
  `vision_serve.go`) before this pass touched anything — confirmed by reading the code, not
  assumed. See `docs/task-first-hour.md`'s R2/guard section for the full writeup and
  `decoder/fitguard_test.go`'s `TestFitEstimate_safetensorsAgreesWithResidentWeightBytes` /
  `TestFitCheckFor_pricesSafetensorsNotJustGGUF` for the gates.
  (c) `serve check` gains a vision row (a fixed small data-URI image, a question
  with one right answer) that reports SKIP-with-reason when no tower is loaded. (d) The
  recommendation registry gains one VL checkpoint per box class, with the tower's cost in the line.

  **(d) BLOCKED, not attempted — a real, structural gap, not a small addition.** The `pull`
  recommendation registry (`pull/registry.go`) only ever recommends a SINGLE-FILE GGUF download
  (every existing entry names one `.gguf` file), and this project's GGUF loader has zero
  vision/mmproj support (confirmed during P8's research — llama.cpp already supports Qwen3-VL's
  mmproj format, this project reads neither half of that pair). Adding a GGUF checkpoint entry for
  a "VL" family would recommend something this project cannot actually use for images — worse than
  no recommendation, not better. A real fix needs either a new safetensors-directory download mode
  in `pull`, or GGUF `mmproj` support landing first (P8b) so a GGUF-based recommendation would mean
  something. Left as a named, tracked gap rather than a hollow entry.
- **P10 · Breadth on the small end.** LFM2.5-VL-3B (SigLIP2 on `lfm2`), Ministral 3's Pixtral tower
  (the `ministral3` decoder exists; the tower is in the same checkpoint), North Micro Vision 2.4B.
  Each is a tower descriptor + projector on a decoder already at parity; do them in that order,
  after P6/P7 have made the tower path fast, and only with a real-checkpoint pin each. Multi-image
  per turn and Gemma 3 pan-and-scan (multi-crop) land here as the interleaver's N-block case the
  June API was shaped for; video is *frames through the image path* with the family's temporal
  position ids (Qwen m-RoPE's t component; Gemma 4's video token) and nothing more — no decoding
  of video containers in v1.
- **P11 · Audio beyond Gemma 4 — a decision, not a build.** Once P7's audio path exists, write the
  one-page comparison: audio-in through other native audio-LLMs (Qwen3-Omni class, LFM2-Audio,
  Voxtral) versus a standalone ASR encoder-decoder (Whisper-class, cross-attention decoder — a new
  primitive) versus nothing. The niche question is whether "pure-Go Whisper" is a product on its
  own; that is a positioning call for the roadmap, and this doc records the technical cost of each
  branch. Speech synthesis and image generation stay out of scope: different model classes, no
  overlap with the descriptor pattern.

**Order.** P6 → P7 (image, then audio) → P9 (a, b, c, d) → P8 → P10 → P11. P6 first because it is
a week of kernel plumbing that makes every later phase measurable at a usable speed; P7 before P8
because Gemma 4 is the family the resident work already paid for; P9 before P8 because the agent
loop is the use case, and screenshots are what agents send.

**Measurement.** Every phase re-measures per-image (and per-clip) time on both boxes with the
current binary, paired against the previous phase, and retires the June figures explicitly — the
171 s / 18.8 s numbers are struck from `benchmarks.md` at P6, not carried. A cold-user run
(`docs/task-first-hour.md`) gains a scenario F — "show it a screenshot" — at P9.

**Not in this program.** Image generation, speech output, video decoding, remote `image_url`
fetching (still an SSRF primitive; still a flag with the `--allow-admin` posture if ever), and any
tower that is not on a decoder already at parity.

## Framing

Add image→text (VLM) inference while preserving goinfer's invariants: pure Go /
no cgo, single static binary, family-as-descriptor, HF parity-gated. The bet is
the same moat as text: land one family end-to-end through the descriptor
pattern; the serve/chat/constrain/tooling surface inherits automatically.

## What already exists to build on

- **VL config flattening** — `decoder/config.go:1358` decodes `text_config` (the nested
  text-decoder dims of a `*ForConditionalGeneration`), so VL `config.json`s
  already parse.
- **Text decoders at parity** for the natural first targets: `gemma3`, `qwen2`,
  `qwen3_5_moe` (the Qwen3.6-VL text side is already loaded, ignoring
  `model.visual.*` / MTP — `decoder/weights.go:426`).
- **m-RoPE stubs** — `decoder/gguf_qwen35.go:77` already notes the image/video mrope
  sections (currently unused).
- **Serve content-array parsing** — `contentText`/`responseInputToMessages`
  already walk OpenAI message content parts; extend to extract `image_url`.
- The descriptor/registry + pin-script + parity-golden ritual, and the `chat`
  template package.

## The pipeline (new components)

```
image bytes → preprocess → vision encoder → projector → image embeddings
                                                              │
text tokens with <image> placeholders → embed ──────────────►│ interleave at placeholder
                                                              ▼
                          text-decoder forward (+2 seams: image-block mask, embed-inject) → logits
```

1. **Preprocessing (pure Go)** — decode (stdlib `image/png`, `image/jpeg`),
   resize, normalize (mean/std), patchify → `pixel_values`. The #1 parity risk
   (HF/PIL resampling in Go) — see Decisions §2.
2. **Vision encoder** — a ViT/SigLIP forward as a descriptor (patch-embed
   conv→linear, positional emb, N pre-LN transformer blocks, final norm).
   Reuses the existing attention/MLP/matmul primitives. f32 first; int8 quant
   for the tower is a follow-on (it's ~0.4B params — CPU-fine either way).
3. **Projector (mm connector)** — the per-family MLP mapping vision-hidden →
   text-hidden.
4. **Embedding interleaving** — substitute projected vision embeddings for
   `<image>` placeholder positions in the decoder's embed step; these are real
   KV positions. Single image per turn in v1, but the interleaver takes a
   *list* of (placeholder-run, embedding-block) pairs from day one — multi-image
   later must not need a redesign.
5. **Position handling + the two forward-path changes** — the "sacred" causal
   path needs exactly two edits, both designed in P0 (not when P3 forces them):
   - **(a) bidirectional image-block mask** — Gemma 3 uses standard RoPE plus a
     bidirectional attention mask over the image-token block. `attendQuery` /
     `attendBatchedHeads` (called by `causalAttention`) are strictly causal
     today, so the attention seam grows a mask (or an image-block range the
     kernel treats as mutually visible).
   - **(b) embed-by-vector injection** — the forward is strictly token-id based
     (`runLayers(id)`, `forward(id)`, `runLayersGemma4/Qwen35(id)`, and
     `Embed.embedRow(id, h)` in both `forward` and the batched `forwardN`). Image
     positions can't go through an id lookup; the embed step needs an override
     that writes the precomputed (projected) vision embedding into `h` at the
     placeholder positions — a small but real API change on every `runLayers*`
     entry + the `forwardN` embed loop.

   Both must be **provably inert when no image block is present** — a text-only
   prompt through the new path must be bit-identical to today's causal path, and
   the existing parity harness is the gate. Qwen-VL instead needs m-RoPE
   (temporal/H/W ids; stubs present) — deferred to the second family.
6. **Tokenizer + chat template** — image placeholder/sentinel tokens and the
   family's image chat formatting.
7. **Serve** — `/v1/chat/completions` content parts with `image_url`.
   **v1 accepts `data:` URIs (base64) only — no URL fetching.** Remote-URL
   fetch from a server is an SSRF primitive (the repo just spent a release
   hardening "attacker-supplied bytes"; don't add "server fetches
   attacker-chosen URLs" in the same breath). A `--allow-image-urls` flag can
   come later with the same gating posture as `--allow-admin`. Prefix-reuse /
   speculative opt out for multimodal turns (like the hybrid). `usage` token
   counts include image tokens (they occupy real KV positions; Gemma 3: 256
   per image), and the per-model queue cost model should know a vision turn is
   a heavy prefill.

## Security surface (new untrusted inputs — campaign Track 2 extension)

- **Decompression bombs**: check `image.DecodeConfig` dims *before* decoding;
  bound total pixels (e.g. ≤ 64 MP) and reject — typed error, never an OOM.
- **Fuzz targets**: image decode→preprocess (hostile PNG/JPEG bytes), the
  serve content-part parser with image parts. Same bar as the rest: no panic,
  no OOM, no hang.

## Parity strategy (stage-isolated, like the deltanet/gemma4 op-for-op goldens)

Pin each stage against HF, committed KB-scale goldens:

- preprocess: `pixel_values` vs the HF processor (tolerance-gated, see §2),
- vision encoder: `last_hidden_state` cosine ≈1.0 on a tiny synthetic + a real
  image,
- projector output,
- end-to-end: argmax + logit cosine vs HF `*ForConditionalGeneration` on a
  fixed image+prompt — **gated on precomputed `pixel_values`** so decoder
  parity never blocks on resize parity.

## Phasing (June 2026 — P0–P5; the 2026-09 program continues at P6)

(✅ P0–P4 done — P0–P3 through `9412e4e`, P4 + the resident GPU encoder pushed;
**P5 open**.)

- ✅ **P0 — scope + harness**: first family (see §1); HF reference + pin scripts;
  tiny synthetic VL checkpoint (mirrors the qwen35-tiny approach); **pin
  gemma-3-4b text-only logit parity** (the gemma3 gate today is gemma-3-270m, a
  *text-only* checkpoint — confirm the descriptor scales to 4B before layering
  vision); design **both** forward-path seams (mask + embed-inject, §5); decide
  preprocess lives in goinfer (not aikit — no cross-repo churn for v1).
- ✅ **P1 — preprocessing** + `pixel_values` golden (tolerance), with the
  end-to-end gate decoupled via precomputed `pixel_values`.
- ✅ **P2 — vision encoder descriptor** + `last_hidden_state` parity.
- ✅ **P3 — projector + interleaving + bidirectional mask + end-to-end logit
  parity** ← the real gate (HF parity, `9412e4e`).
- ✅ **P4 — tokenizer image tokens + chat template + serve vision API** (data-URI
  only) + the security/fuzz items; the whole surface inherits.
- ✅ **P4.5 — resident GPU SigLIP encoder** (`-tags gpu`): 171 s → 18.8 s/image,
  parity cosine 1.0 (`886c8fd`/`5d7c572`). Wired into both serve (`--backend
  webgpu`) and `demo/agent` (`--vision-backend webgpu`, `87f127d`). Follow-up: a
  tiled attention GEMM (attention QKᵀ/scores·V are still naive f32).
- **P5 — second family (Qwen2.5-VL: m-RoPE + dynamic resolution) to prove the
  descriptor generalizes**, + the GGUF `mmproj` companion-file seam.

## Decisions (recommendations from review)

1. **First family: Gemma 3 4B-it.** Smallest; SigLIP tower; **fixed 256-token
   image block at fixed 896×896 input** (vs Qwen's dynamic resolution +
   variable patch grids + m-RoPE — strictly more moving parts). The `gemma3`
   descriptor is parity-proven, so the 4B text decoder loads by construction —
   but the gate today is on **gemma-3-270m (text-only)**, so pin gemma-3-4b text
   parity in P0 before vision (catches any 4B rope/sliding/config quirk).
   Pan-and-scan (multi-crop) is **out of scope v1** — base resolution only.
   Qwen2.5-VL is P5, where it earns its keep by proving generality.
2. **Resize parity: decouple, then port.** Gate end-to-end on precomputed
   `pixel_values` (P1) so the decoder gate is never hostage to resampling.
   Separately, port PIL-style separable convolution resampling (support-scaled
   antialias) in pure Go — `golang.org/x/image/draw`'s Catmull-Rom is *not*
   PIL-exact; the PIL kernel is ~100 lines and gets within tolerance. Pin
   preprocess at a documented tolerance, tighten later if it matters.
3. **safetensors-only v1.** The GGUF `mmproj` companion ("load a second file
   alongside the model") is a new loader seam — defer to P5 with the second
   family.
4. **Vision weights external in v1** (a `--vision` path or auto-discovered
   sibling). Embedding the tower in the `.giw`/static-binary story (+0.4–1.6 GB
   depending on quant) is a follow-on once int8 tower quant exists.
5. **Scope guard: image-only, single image per turn, base resolution, v1** —
   but the interleaving API shaped for N images (see §4 above).

## Risks (ranked)

1. Resize parity — mitigated by decoupling (§2).
2. The two forward-path seams (bidirectional mask + embed-by-vector injection)
   touching the causal attention kernels and every `runLayers*`/`forwardN` embed
   step — mitigated by designing both in P0 and the existing parity harness (any
   drift in *text-only* behavior is a hard fail: both seams must be provably
   inert when no image block is present).
3. ViT prefill cost on CPU (hundreds of patches × ~27 layers) — acceptable v1
   (~171 s/image), flagged in `benchmarks.md`. **RESOLVED for GPU:** the resident
   WebGPU tower (`--backend webgpu`) brings it to 18.8 s/image (~9×); originally
   scoped out, built after the int8-CPU lever proved a wash. See
   `docs/completed/task-gpu-vision-tower.md`.
4. KV/positions/sessions accounting with image blocks — prefix-reuse opts out
   v1; revisit with a "image-block-aligned prefix" scheme later.
5. Serve security surface — bounded by data-URI-only + pixel caps + fuzzing.
