# P11 — Audio beyond Gemma 4: technical cost of each branch (desk research, 2026-10-06)

Read-only desk research. Facts come from HF `config.json`/`preprocessor_config.json`/model cards, transformers `main`
modeling code, and papers, fetched 2026-10-06 (sources below). **UNVERIFIED** marks a fact not read from a primary source.
"goinfer has it" means the family is registered in `decoder/registry.go` and has a row in `docs/capability-matrix.md`.
The decision on whether pure-Go Whisper is a product belongs to the owner. This page only records what each branch costs.

**What already exists (reusable).** aikit `audio`: `gemma4_audio` conformer tower plus log-mel front end, gated at cosine 1.0
(EmbeddingGemma 2, CPU and Metal). The front end is Gemma-specific: 16 kHz, 320/160 framing, a hard-coded radix-2 FFT512,
**magnitude** spectrum, 128 **HTK** mels with no area normalization, `ln(mel+0.001)`, truncation at 30 s. The image/audio
soft-token splice (`masked_scatter` at a placeholder id) is in place for Gemma 3/4, Qwen2.5-VL, Qwen3.5 and EmbeddingGemma 2.
Decoders already at parity: `qwen3`, `qwen3_moe`, `qwen3_vl` (text), `llama`, `ministral3`/`mistral3`, `lfm2`, `gemma4`.

**The finding that matters most.** Most of the candidates do *not* use Gemma's front end. Whisper, Voxtral, Qwen3-ASR and
Qwen3-Omni all use transformers' `WhisperFeatureExtractor` (128 bins, hop 160, `n_fft` 400). That extractor differs from
aikit's at every stage:

| stage | Whisper extractor | aikit's Gemma front end |
|---|---|---|
| FFT | 400 points, not a power of two (needs mixed radix or Bluestein) | radix-2, 512 points |
| spectrum | power (`power=2.0`) | magnitude |
| mel scale | Slaney, with Slaney area norm | HTK, no norm |
| log | `log10`, with each clip clamped to its max minus 8, then `(x+4)/4` | `ln(mel+0.001)` |
| frames | drops the last frame | keeps it |

Only the framing loop and the bit-exact probe method carry over. So **a Whisper-style front end is one S build, shared by
four families**. Voxtral's encoder *is* Whisper large-v3's encoder (shapes 1280/32 layers/20 heads; paper: "based on Whisper
large-v3"), so a Whisper encoder build also serves Voxtral.

| model (released) | params / license | audio encoder | front end | injection into the LLM | text decoder: goinfer has it? | audio output (out of scope) |
|---|---|---|---|---|---|---|
| **Qwen3-ASR 0.6B / 1.7B** (2026-01-29) | 0.94B / 2.35B safetensors; Apache-2.0; `qwen3_asr` in transformers | "AuT": 3× Conv2d s2 (8× ⇒ 12.5 Hz), sinusoidal positions, windowed attention (1–8 s), pre-LN, GELU; 180M (896-d) / 300M (1024-d, 24 L) | Whisper extractor, 128 mel, 30 s | soft tokens: projector + `masked_scatter` at `audio_token_id` | `qwen3` (Qwen3-0.6B / 1.7B), **yes, full-oracle**. Plain 1-D RoPE INFERRED from `text_config.model_type: qwen3` | none. Timestamps need a separate NAR `Qwen3-ForcedAligner-0.6B` |
| **Qwen3-Omni-30B-A3B** (2025-09-22) | 35.3B safetensors; Apache-2.0 | same AuT design, 1280-d/32 L/20 heads (≈0.6B, INFERRED from shapes) | Whisper extractor, 128 mel | soft tokens, `masked_scatter`. Positions use 3-D m-RoPE (TM-RoPE); the audio time-position rule was NOT read | `qwen3_omni_moe_text` (48 L, 128 experts top-8): **not registered**. Likely composes `qwen3_moe` with an m-RoPE variant (UNVERIFIED) | Talker plus code2wav, multi-codebook |
| **Voxtral Mini / Small** (2025-07) | 4.68B / 24.3B; Apache-2.0 | Whisper large-v3 encoder (Conv1d×2, fixed positions, 32 L pre-LN GELU, 50 Hz). Clips over 30 s are chunked, with positions reset per chunk | Whisper extractor, 128 mel | MLP adapter, which stacks 4 frames (⇒ 12.5 Hz), then `masked_scatter` | `text_config.model_type: llama` (Ministral 3B / Mistral Small 3.1): **yes, `llama`** | none |
| **Voxtral Mini 4B Realtime** (2026-02) | 4.43B; Apache-2.0; `voxtral_realtime` | new ≈970M causal encoder: causal Conv1d, RoPE, SWA 750, RMSNorm, SwiGLU | 128 mel, 400/160, 12.5 Hz | **added** to the text embedding at every position (`inputs_embeds += audio_embeds`), not spliced | Ministral-3-3B, plus **ada-RMSNorm conditioned on a delay embedding**: a new decoder variant | none |
| **LFM2-Audio / LFM2.5-Audio 1.5B** (repos 2025-08 / 2025-12) | 1.47B; LFM Open License v1.0 (terms UNVERIFIED) | FastConformer, 115M, 17 L × 512, 8× subsampling (from NVIDIA canary-180m-flash) | 128 mel, 25/10 ms, Hann, 16 kHz. NeMo-style details (pre-emphasis, normalization) UNVERIFIED | continuous embeddings via an adapter. Exact splice NOT read. **The reference is the `liquid-audio` package, not transformers** | `lfm2` (LFM2-1.2B shape: 2048-d, 16 L): **yes, full-oracle** | RQ "depthformer" plus Mimi detokenizer (24 kHz, 8 codebooks) |
| **Whisper large-v3 / v3-turbo** (2023-11 / 2024-10) | 1.54B / 0.81B; Apache-2.0 / MIT | Conv1d×2 GELU, fixed sinusoid table, 32 L × 1280, pre-LN, biased linears, full attention over 1500 frames | Whisper extractor, 128 mel, padded to 30 s | **cross-attention**: the encoder states are K/V in every decoder layer | own decoder: 32 L (v3) / **4 L (turbo)**, learned positions (448), tied head, 51,866-token BPE. **Not in goinfer.** It is an encoder-decoder class | none |
| **Cohere Transcribe 03-2026** (2026-03-26) | 2.07B; Apache-2.0, gated | Parakeet/FastConformer, 48 L × 1280, 8× subsampling (transformers config defaults; the checkpoint config is gated, not read) | 128 mel. Hop and window UNVERIFIED | cross-attention, 8-layer 1024-d decoder | own decoder, 16k SentencePiece vocab. No timestamps (per the card) | none |

Also seen:
- **Gemma 3n** (`gemma3n`, USM conformer, Gemma terms). The decoder is not in goinfer, and Gemma 4's tower supersedes it.
- **Encoder-only CTC models.** NVIDIA `parakeet-ctc` (CC-BY-4.0, in transformers) and IBM Granite Speech 5.0 470M TurboCTC
  (repo 2026-08, Apache-2.0, `granite_speech5_ctc`; its "English-only, no decode loop" description is from a secondary
  source, UNVERIFIED). These have no decoder at all: greedy CTC collapse.

## Branches and their goinfer cost (sizes are desk estimates, INFERRED)

0. **Nothing beyond Gemma 4.** Cost: 0. Gemma 4 E2B/E4B audio-into-the-model is already parked item 1 (M–L): the tower exists,
   and it needs the splice, prompt layout and `input_audio`. What it leaves uncovered:
   - no timestamps;
   - a 30 s hard limit in the extractor;
   - ASR quality against dedicated models is unmeasured;
   - E2B/E4B run CPU-only for the whole model today.
1. **Native audio-LLMs: the soft-token splice into a decoder goinfer already runs.**
   - *Shared prerequisite:* the Whisper front end (**S**), a second extractor in aikit `audio` with its own bit-exact NumPy
     gate.
   - *Qwen3-ASR:* the AuT encoder (Conv2d stem, sinusoid table, windowed attention, which is the SigLIP-style block with a
     block-diagonal mask) plus a projector (**M**). The decoder is `qwen3` at full-oracle. Models are small (0.6B ≈ 1.9 GB
     bf16, which fits the Mac's current 3.4 GiB free). This is the cheapest path to ASR-grade audio-in.
   - *Voxtral Mini/Small:* the Whisper encoder (**S–M**), the 4× adapter and per-30-s chunking. The decoder is `llama`.
   - *Qwen3-Omni:* reuses the AuT encoder, but the thinker is a new MoE-plus-TM-RoPE family (**L**), and 35B is out of reach
     on the Mac.
   - *LFM2-Audio:* a FastConformer encoder (**M**). It shares macaron FFN, depthwise-conv module and conv subsampling with
     `gemma4_audio`, but needs Transformer-XL relative-position attention, BatchNorm and LayerNorm. Pinning against
     `liquid-audio` rather than transformers adds harness risk. `lfm2` is already at parity.
   - *Voxtral Realtime:* additive per-position injection, a delay-conditioned ada-RMSNorm decoder variant and a causal
     streaming encoder (**L**). It is the only streaming-native candidate.
2. **Standalone encoder-decoder ASR (Whisper class).** New pieces:
   - the Whisper front end (**S**, shared with branch 1);
   - the encoder (**S–M**, shared with Voxtral);
   - a cross-attention decoder (**M**). This is a new model class outside the decoder-only descriptor pattern: a static
     per-clip K/V cache per layer, learned positions, LayerNorm with bias. INFERRED: it lives as its own package, as
     `embeddinggemma2` does;
   - Whisper's decode policy (**M–L** to match HF `generate`): special/language/task tokens, the timestamp logits rules
     (1,501 timestamp tokens, pairing and monotonicity), long-form sequential 30 s windows, temperature fallback on
     compression-ratio and log-prob thresholds, and conditioning on the previous window.

   Totals: short-form (≤30 s), greedy, no-timestamp turbo is about **M**. Product-grade Whisper is **L**, and most of that L
   is the decode policy, not the math. The cross-attention decoder would also cover Cohere Transcribe, which additionally
   needs a FastConformer encoder.
3. *(Variant)* **Encoder-only CTC** (Parakeet-CTC, Granite 5.0 TurboCTC). This needs a FastConformer encoder (**M**) and a
   CTC collapse (**S**). There is no decoder and no cross-attention, but these models are mostly English, and the license
   varies (CC-BY-4.0 for Parakeet).

## Options (costs, not a decision)

- **(a) Stop at Gemma 4.** Costs nothing new. Audio-in rides on parked item 1.
- **(b) Whisper front end plus Qwen3-ASR** (S + M). This adds 30-language ASR-quality audio-in on a full-oracle decoder with
  no new primitive. Timestamps would need its separate aligner model.
- **(c) Whisper front end plus the Whisper encoder** (S + S–M). This buys Voxtral (via `llama`) and puts two-thirds of
  Whisper in place. Adding the cross-attention decoder and decode policy (M + M–L) gives "pure-Go Whisper", the only branch
  with native timestamps and long-form handling.
- **(d) Voxtral Realtime** (L), only if streaming is the product.

(b) and (c) share the S front end, so it is no-regret under either.

## Sources (fetched 2026-10-06)

Qwen3-ASR:
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B (card, `config.json`, `preprocessor_config.json`)
- https://arxiv.org/abs/2601.21337
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_asr

Qwen3-Omni:
- https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct (card, `config.json`, `preprocessor_config.json`)
- https://arxiv.org/abs/2509.17765
- transformers `models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`

Voxtral:
- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 (card, `config.json`)
- https://huggingface.co/mistralai/Voxtral-Small-24B-2507/raw/main/config.json
- https://arxiv.org/abs/2507.13264
- transformers `models/voxtral/modeling_voxtral.py`

Voxtral Realtime:
- https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 (card, `params.json`)
- https://arxiv.org/abs/2602.11298
- transformers `models/voxtral_realtime/modeling_voxtral_realtime.py`

LFM2-Audio:
- https://huggingface.co/LiquidAI/LFM2-Audio-1.5B
- https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B (card, `config.json`)

Whisper:
- https://huggingface.co/openai/whisper-large-v3-turbo (card, `config.json`)
- https://huggingface.co/openai/whisper-large-v3/raw/main/preprocessor_config.json
- transformers `models/whisper/feature_extraction_whisper.py`

Cohere Transcribe:
- https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 (card; `config.json` gated, HTTP 401)
- transformers `models/cohere_asr/configuration_cohere_asr.py`

HF API metadata (repo creation dates, licenses, safetensors parameter totals):
- `https://huggingface.co/api/models/<repo>` for each repo above
- the same for `google/gemma-4-E2B-it`, `google/gemma-3n-E2B-it`, `nvidia/parakeet-ctc-1.1b`,
  `ibm-granite/granite-speech-5.0-470m-turboctc`

Dates are paper or card dates where those were read. Otherwise they are HF repo-creation dates, which are not the public
release date.
