# demo/gemma

A pure-Go CLI that runs a local **Gemma 3** (270M / 1B) checkpoint through
aikit's `decoder` package and streams the completion.

> **Scaffold.** The wiring compiles and runs; the forward pass, tokenizer and
> BF16 loader are stubbed (see [`docs/gemma-decoder-plan.md`](../../docs/gemma-decoder-plan.md)).
> Running it today prints which milestone is outstanding instead of tokens.
> This is the harness those milestones fill in.

## Get a checkpoint

HuggingFace layout (`config.json` + `model.safetensors` + tokenizer):

```bash
huggingface-cli download google/gemma-3-270m --local-dir ~/models/gemma-3-270m
```

## Run

```bash
go run ./demo/gemma --model ~/models/gemma-3-270m --prompt "Hello, world"

# options
go run ./demo/gemma \
  --model ~/models/gemma-3-1b \
  --prompt "Write a haiku about Go" \
  --max 128 --temp 0.7 --top-k 40 --top-p 0.95 --seed 42 \
  --backend cpu        # cpu (default) | webgpu (M9; falls back to cpu)
```

## What works today

`--help`, flag parsing, backend selection, and the load→encode→generate→stream
control flow. Everything past `tokenizer.Load` / `decoder.Load` returns an
honest `not implemented [Mx]` pointing at the milestone in the plan.
