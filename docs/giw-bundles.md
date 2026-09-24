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

## File layout: fused groups for Metal (weights format v13, kind 6)

S6 (`docs/tasks/task-never-swap-2026-09.md`) wants a Metal load's dense int4 weights to be **file-backed
views of the mapping** instead of a second, anonymous MTLBuffer copy of every tensor. Metal's fused
GEMVs read **one buffer** holding q‖k‖v (or gate‖up) rows back to back, and an mmap'd file can only be
aliased into such a buffer if those rows are *adjacent in the file* — which per-tensor records (a
scales array between each pair of nibble arrays) never are.

For **`-target metal` only**, v13 writes each fused tuple — `(q,k,v)` and `(gate,up)` per layer — as a
**group**: every member as a kind-6 record (`kind 6 | rows | cols | group | w8a8 | pad | u32 nScales |
f32 scales`, *no nibbles*), then one **group block**: padding to 16 bytes, then each member's nibbles
one after another with **no length prefixes** (lengths are `rows*cols/2`; a prefix between members
would be exactly the gap this removes). Members are canonical group-32 int4 with the same K
(K%32==0 keeps every member 16-aligned back to back). A tuple with an int8 member, an absent member (a
K=V layer has no V), or a mixed K is written as ordinary consecutive records — no kind 6.

The reader takes the nibbles out of the mapping as before (`WrapInt4` keeps the alias); the Metal
build (`GOINFER_METAL_ALIAS=1`, opt-in) checks whether a fused tuple's nibbles are **one contiguous
run** in the mapping and, if so, binds one no-copy buffer over them (`metal/alias.go`). It is detected,
not assumed, so a pre-v13 file, a layer whose members are not adjacent, or a heap-backed weight all
take the old copy path unchanged. Scales are still converted f32→f16 into a small buffer (about ⅛ of
the nibble bytes); storing them as f16 is a possible later step.

Compatibility:

- **Only a metal-target file is v13.** Every other target still writes v12 (`giwWriter.emitVersion`),
  so a file that can contain no kind 6 stays readable by a pre-v13 reader. A pre-v13 reader refuses a
  metal-target v13 file through the version guard, as it did for v12.
- A v13 reader loads every older bundle unchanged.
- A metal-target sidecar is now a promise to **one backend**, the way kind 5 is to one arch: nothing
  stops a CPU load from reading it (the nibbles and scales are ordinary canonical int4), but the
  group block layout exists for Metal's benefit.
- Existing metal-target sidecars keep their layout; the Metal aliasing simply does not apply to their
  fused tuples (it logs `N fused groups not adjacent`). Rebuild to get it.
