# demo/gemma

A pure-Go CLI that loads a local decoder-only LLM through goinfer's `decoder`
package, tokenizes the prompt, and streams the completion to stdout. Despite the
name it is **multi-model**: any family the decoder supports (Gemma 3 and 4, Qwen,
Llama, Mistral, GPT-2, Mixtral) loads from a HuggingFace checkpoint dir, and a
bare quantized `.gguf` loads with no sidecar config or tokenizer.

## Get a checkpoint

HuggingFace layout (`config.json` + `model.safetensors` + tokenizer), or a
single `.gguf`:

```bash
# Gemma 4 E2B — ungated and Apache-2.0, so this works on a fresh machine with no
# HuggingFace account. (The Gemma 3 checkpoints this page used to name are
# `gated=manual`: the command fails until you request access and log in.)
huggingface-cli download google/gemma-4-E2B-it --local-dir ~/models/gemma-4-E2B-it

# a bare quantized .gguf also works, with no sidecar config or tokenizer
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_M.gguf --local-dir ~/models
```

## Run

```bash
go run ./demo/gemma --model ~/models/gemma-4-E2B-it --prompt "Hello, world"

# a bare .gguf needs no sidecar config or tokenizer
go run ./demo/gemma --model ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --prompt "The capital of France is"

# options
go run ./demo/gemma \
  --model ~/models/gemma-4-E2B-it \
  --prompt "Write a haiku about Go" \
  --max 128 --temp 0.7 --top-k 40 --top-p 0.95 --seed 42 \
  --quant int8int8 \   # "" (f32) | int8 | int8int8 | int4
  --json \             # constrain output to valid JSON (logit masking)
  --backend cpu        # cpu (default) | webgpu
```

Greedy by default (`--temp 0`); set `--temp`/`--top-k`/`--top-p` for sampling.
`--json` masks logits to emit only valid JSON (see the `constrain` package).
