# Prequantized weight bundles (`.giw`)

What a `.giw` bundle is, how to build one with `cmd/prequant`, and why the quant is fixed at
build time. Back to the [README](../README.md).

## Prequantized weight bundles (`.giw`)

Loading a GGUF quantizes its weights on every launch. A **`.giw` bundle** stores the
already-quantized resident weights alongside a metadata-only GGUF (the source truncated at
the tensor-data boundary, so it still carries the tokenizer). Loading one skips
dequant/requant entirely — the weights are aliased straight from the file image rather than
copied into a multi-GB heap.

Build one with `cmd/prequant`:

```bash
go run ./cmd/prequant -o model.giw ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

| Flag | Meaning |
|---|---|
| `-o PATH` | output bundle path (**required**) |
| `-quant` | quant baked into the bundle: `int8int8` (default), `int8`, or `int4` |
| `-embed-int4` | with `-quant int4`, store the token-embedding / LM-head table at int4 too instead of pinning it at int8 — roughly halves the head's per-token traffic on a big-vocab model |

The quant is **baked in at build time**: a bundle made with `-quant int8int8` is an int8int8
model, and `serve --quant` cannot change it afterwards. Build a separate bundle per quant you
intend to serve.

Then serve it like any other model:

```bash
./serve --model model.giw
```

`serve --stream-weights` also produces these on demand — a plain `.gguf` is transcoded to a
sidecar `.giw` cache on first use, so the one-time cost is paid once rather than per launch.

**Peak RAM during that one-time build** bounds to roughly one layer for most families — the
weights are read, quantized, written and freed one layer at a time rather than all held resident
at once (S2, `docs/tasks/task-never-swap-2026-09.md`; measured on a real `gpt-oss-20b` GGUF,
`docs/measurements/transcode-streaming-2026-09-23.md`: swap-used stayed flat through the whole
transcode). **gemma4 is the one exception**: its fused PLE/MoE tail cannot be written
incrementally, so building a gemma4 sidecar still needs the whole model resident first, same as
every family did before S2.

## File layout: aligned arrays (weights format v12 / bundle v3)

Until v11 a `.giw` did not align the arrays inside it, so the loader could only alias the packed
int4/int8 **codes** out of the file and had to **copy the per-group scales** to the Go heap on every
load. Measured 2026-09-24 (`docs/measurements/moe-pager-mode-darwin-2026-09-23.md`, "Finding"):
that copy was 3.75 GB of a streamed Qwen3.6-35B-A3B's 5.8 GB heap and 3.0 GB of the 4.4 GB heap of
the M26 Metal load — anonymous memory the pager can neither budget nor evict.

From v12 every weight-matrix payload array (int8 scales + codes, int4 scales + nibbles, and the
row4 pair) is preceded by zero padding so it starts on a **16-byte boundary relative to the blob**,
the header ends on one too (so a present-or-absent quant label cannot shift what follows), and the
bundle header itself is padded to 64 bytes (**bundle v3**) so the blob starts 16-aligned in the
file. A mapping of the file therefore has every such array aligned, and the loader aliases the
scales like it already did the codes. f32 *matrices*, norms and biases still copy (small, and the
mapping is read-only).

Compatibility, both ways:

- A **v12 reader loads every older bundle** unchanged (it aliases a scale array only when its address
  happens to be 4-byte aligned, and copies otherwise — a quarter of an old file's arrays qualify).
- A **pre-v12 reader refuses a v12 file** through the format's own version guard; it does not misread it.
- **Existing sidecars keep their old layout and their heap copies** — nothing rebuilds them
  automatically (a 20 GB transcode is not something to trigger silently). To get the memory win,
  delete the sidecar (`<model>.<quant>.<target>.giw`, next to the `.gguf`) and it is rebuilt on the
  next `--stream-weights` / darwin default load, or re-run `cmd/prequant`.
- The trailing CRC still covers the padding; a flipped pad byte fails the load.
