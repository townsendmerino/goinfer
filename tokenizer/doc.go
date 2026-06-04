// Package tokenizer implements the BPE tokenizers the decoder LLMs ship,
// loaded from the HF tokenizer.json. One ordered-merge core serves two
// families behind a mode flag (see tokMode):
//
//   - modeGemma (M2): Gemma 3's byte-fallback SentencePiece-style BPE —
//     normalize ASCII space → ▁, per-rune symbols, <0xNN> fallback for
//     out-of-vocab runes.
//   - modeByteLevel (G3): the GPT-2 / Llama-3 / Qwen byte-level BPE — NFC
//     normalize, a GPT-2 split-regex pretokenizer, and a byte→printable-rune
//     map (space → Ġ) so every symbol is in-vocab (no byte-fallback).
//
// Load auto-detects the family from tokenizer.json (the decoder type) and
// resolves special tokens (Gemma's are required; the byte-level families read
// bos/eos/pad from tokenizer_config.json, defaulting to "none"). It is
// separate from embed.Tokenizer (WordPiece, for the Model2Vec/CodeRankEmbed
// encoders) — the algorithms and vocab format don't transfer.
//
// LoadGGUF (see gguf.go) is the sidecar-free sibling of Load: it builds the
// same Tokenizer from a bare .gguf file's embedded metadata (vocab + merges +
// special ids), so a quantized checkpoint tokenizes with no tokenizer.json. It
// covers both GGUF tokenizer families, dispatched on tokenizer.ggml.model:
// "llama" (SentencePiece byte-fallback — reuses modeGemma + the ▁ dummy-prefix
// knob) and "gpt2" (byte-level — reuses modeByteLevel, with the pretokenizer
// knobs read from tokenizer.ggml.pre instead of tokenizer.json).
//
// Golden parity against HF `tokenizers` is the gate for every family (M2 /
// G3 / GGUF): a single-token drift silently degrades generation, so the bar
// is exact id equality, not a tolerance.
package tokenizer
