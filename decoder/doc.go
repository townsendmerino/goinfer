// Package decoder runs autoregressive, decoder-only transformer language
// models as a pure-Go forward pass. It is the generative sibling of encoder/
// (bidirectional embeddings) and embed/ (Model2Vec), sharing embed/'s
// safetensors loader.
//
// One generic forward pass, parameterized by an Architecture descriptor
// resolved from the checkpoint, serves a broad slice of the open-weights
// ecosystem with logit/argmax parity validated against HuggingFace:
//
//   - Families: Gemma, Qwen, Llama, Mistral/Mixtral, GPT-2, DeepSeek and others; docs/ARCHITECTURE.md and the
//     Architecture descriptors (arch.go) are the current list.
//   - Axes: RMSNorm/LayerNorm · RoPE (incl. llama3 scaling)/learned positions ·
//     gated/non-gated/sparse-MoE MLP · full/sliding attention · tied/untied
//     heads · optional QKV/output bias · Linear/Conv1D layouts.
//   - Weights: f32, bf16, f16, and int8/int4 quantized; from a single safetensors file, a sharded checkpoint, or a
//     self-describing GGUF needing no sidecar config or tokenizer. An optional PEFT LoRA adapter (Options.LoRA) is
//     merged into the f32 weights before quantization, so it costs nothing extra at decode.
//
// See docs/ARCHITECTURE.md for the descriptor design and the per-family adapters.
//
// # Carry-over invariants (read once)
//
// These are the knobs that silently corrupt output if mishandled; the
// Architecture descriptor captures them per family rather than hardcoding
// Gemma's values.
//
//   - Embedding scale and a tied vs. untied LM head are per-family and
//     correctness-critical (Gemma scales by sqrt(hidden_dim) and ties the
//     input embedding table to the head).
//   - RMSNorm weighting differs by family: Gemma scales by (1 + weight),
//     others by weight. The wrong one shifts every activation.
//   - RoPE base(s) are per-family: Gemma 3 uses TWO tables (local θ=10000,
//     global θ=1000000); llama3 applies a frequency scaling. Using one base,
//     or the wrong scaling, corrupts long-context positions.
//   - Gemma 3 dropped Gemma 2's logit/attention soft-capping in favor of query/key RMSNorm (QK-norm). The Gemma 3
//     loader rejects a soft-capping checkpoint rather than ignoring the field.
//   - Attention is causal with grouped-query heads, optionally with a sliding
//     window on local layers. The KV cache, not a per-call growing buffer, is
//     the memory model.
//
// All compute matmuls route through a Backend (see backend.go) so a GPU backend can replace the CPU one without
// touching the forward pass.
package decoder

import "errors"

// errNotImplemented is the sentinel for a path that cannot run: the "weights not loaded" guards in the forward paths,
// and a feature combination no family needs yet, which fails loudly rather than computing a wrong answer.
var errNotImplemented = errors.New("decoder: not implemented")
