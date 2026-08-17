# Multimodal (vision-language) for goinfer — plan

> Status: **P0–P4 DONE.** P0–P3 (image→logits at HF parity) landed through
> `9412e4e`; **P4 (serve vision API + agent image input) is now a real
> user-facing feature**: `cmd/serve` accepts images on both the OpenAI
> (`image_url`) and Anthropic (`image`) surfaces, base64/data-URI only, behind
> `--vision <dir>`; `demo/agent` (web UI) takes a dropped/pasted image; loading a
> real `google/gemma-3-4b-it` works directly. Image prefill is **~171 s/image on
> CPU**, or **18.8 s (~9×) on `-tags gpu` with `--backend webgpu`** — the resident
> GPU SigLIP encoder, parity cosine 1.000000 (`886c8fd`/`5d7c572`,
> `docs/completed/task-gpu-vision-tower.md`). An int8 CPU tower was evaluated and is a wash on
> AVX2 (no VNNI; `docs/completed/task-cpu-vision-prefill.md`) — GPU is the real speedup.
> Both serve (`--backend webgpu`) and `demo/agent` (`--vision-backend webgpu`)
> run on the GPU encoder. Remaining (all optional / new scope): a **tiled
> attention GEMM** to push the GPU path below 18.8 s (attention QKᵀ/scores·V are
> still naive f32); and **P5** (Qwen2.5-VL second family + m-RoPE + GGUF
> `mmproj`). (This is the doc `benchmarks.md` references.)
> Drafted 2026-06-10 (vscode session), revised after external review (Claude
> app): added the bidirectional-mask forward-path item, the serve security
> surface, usage accounting, and firm recommendations on the five open
> decisions. Re-grounded against the code (2026-06-10): the forward path takes
> **two** edits, not one (the mask **and** an embed-by-vector override — the
> forward is strictly token-id based), and the `gemma3` parity gate is pinned on
> the **text-only** gemma-3-270m, so 4B text wants its own pin in P0. Audio/video
> are out of scope for v1.

## Framing

Add image→text (VLM) inference while preserving goinfer's invariants: pure Go /
no cgo, single static binary, family-as-descriptor, HF parity-gated. The bet is
the same moat as text: land one family end-to-end through the descriptor
pattern; the serve/chat/constrain/tooling surface inherits automatically.

## What already exists to build on

- **VL config flattening** — `decoder/config.go:932` decodes `text_config` (the nested
  text-decoder dims of a `*ForConditionalGeneration`), so VL `config.json`s
  already parse.
- **Text decoders at parity** for the natural first targets: `gemma3`, `qwen2`,
  `qwen3_5_moe` (the Qwen3.6-VL text side is already loaded, ignoring
  `model.visual.*` / MTP — `decoder/weights.go:355`).
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

## Phasing

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
