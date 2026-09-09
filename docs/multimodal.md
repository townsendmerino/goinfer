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
2. **Vision is one family.** Gemma 4 — the family most of the resident work went into — is
   text-only here; Qwen-VL, the most-pulled VL line, is text-only here.
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
  **CUDA/Metal (the vision TOWER itself) — still NOT STARTED, and now lower priority.**
  `attn_fused` (CUDA) and `attention_prefill` (Metal) are causal-only by hardcoded row-index math,
  and both fast GEMMs need int4-quantized weights with no batched-M fallback — a non-causal kernel
  variant and a new tower weight-quantization pipeline, each with its own parity gate, not a rewire
  of what exists. Gap 0 (below) turned out to be the bigger, cheaper, differently-shaped lever it
  looked like when flagged — measured 3.86× on real decode with only a Go-side prefill/decode
  bridge plus one narrow kernel change (Qwen's rope-angle split), not a new GPU attention kernel
  or quant pipeline for the tower's own weights. This CUDA/Metal tower work is still real and still
  wanted eventually (the tower's own ~31s CPU cost per SigLIP image is untouched by gap 0's fix,
  which only moved decode), but it's a smaller, lower-priority remainder now that decode is fixed,
  not a prerequisite for anything.
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
- **P8 · Finish P5: Qwen3.x-VL image path + GGUF `mmproj`.** Phase 0 on the real Qwen3.6-VL config
  (the vision encoder, dynamic resolution and patch grids, m-RoPE's three position components
  which the text path already degenerates correctly, any DeepStack-style multi-level injection —
  verify, don't assume). The `mmproj` companion-file seam lands here because this is the family
  Ollama users have as GGUF+mmproj: `--model x.gguf` auto-discovers the sibling mmproj (or takes
  `--mmproj`), reads its tensors into the same tower descriptor, and the loader refuses a
  mismatched pair by tensor-shape check rather than by filename. Gate: end-to-end parity on the
  safetensors path AND bit-identity between the safetensors tower and the mmproj tower on the same
  checkpoint (they are the same weights in two containers).
- **P9 · Image turns in the agent loop.** (a) Prefix reuse over image blocks: an image's embedding
  block is a pure function of its bytes and the tower, so key the resident bookkeeping on a hash of
  the image bytes standing in for a token id at each placeholder position — a reused prefix with an
  unchanged image is then exactly the text case, and a changed image invalidates from its first
  position. Gate: the reuse invariant test extended with an image turn (reused == cold, bitwise),
  plus the agent-turn TTFT cell with a screenshot attached. (b) The fit guard prices the tower
  (resident weights) and image-token KV at the family's per-image token count × images per turn;
  request-time admission (task-fit-to-hardware Phase 1, from first-hour R13) counts image tokens
  in the prompt. (c) `serve check` gains a vision row (a fixed small data-URI image, a question
  with one right answer) that reports SKIP-with-reason when no tower is loaded. (d) The
  recommendation registry gains one VL checkpoint per box class, with the tower's cost in the line.
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

- **VL config flattening** — `decoder/config.go:1306` decodes `text_config` (the nested
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
