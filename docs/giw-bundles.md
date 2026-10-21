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
| `-target` | which consumer the bundle is promised to: `cpu-arm64`, `cpu-amd64`, `metal`, `cuda`, `webgpu`, `cpu` (this box's CPU) or `canonical`; see "Targets" below |

The quant is **baked in at build time**: a bundle made with `-quant int8int8` is an int8int8
model, and `serve --quant` cannot change it afterwards. Build a separate bundle per quant you
intend to serve.

Then serve it like any other model:

```bash
./serve --model model.giw
```

`serve --stream-weights` also produces these on demand — a plain `.gguf` is transcoded to a
sidecar `.giw` cache on first use, so the one-time cost is paid once rather than per launch.

**Peak RAM during that one-time build** bounds to roughly one layer for every family — the
weights are read, quantized, written and freed one layer at a time rather than all held resident
at once (S2, `docs/completed/task-never-swap-2026-09.md`; measured on a real `gpt-oss-20b` GGUF,
`docs/measurements/transcode-streaming-2026-09-23.md`: swap-used stayed flat through the whole
transcode). gemma4 was the last family to get there (2026-09-24,
`docs/measurements/transcode-streaming-gemma4-2026-09-24.md`): the 26B-A4B's transcode went from
18.1 GB of anonymous memory (34.8 GB RSS) resident to a 1.61 GB peak streamed, byte-identical output.
A transcode from a safetensors directory (`prequant` given an HF directory) streams layer by layer too
(`docs/tasks/task-prequant-dir-streaming-2026-10.md`).

## A cold load

The loader maps a bundle and reads each matrix's header through the mapping. The kernel reads around every page it
faults in, so on a bundle of thousands of small matrices, which is what an MoE's experts are, that walk would read
nearly the whole file in scattered pieces. The load therefore decides, once the header has named the model and before
any matrix is read. On Linux:

- **An MoE bundle that is not in the page cache and fits in memory with room** (three quarters of what the kernel
  reports available) **is read front to back first.** The load prints a line when it does. The weights are needed
  either way; this reads them at the disk's sequential rate.
- **An MoE bundle that cannot stay in memory** (paged with `--stream-weights`, or larger than the installed memory) has
  the read-around turned off for the walk, so the load reads the headers and not the file.
- **A dense bundle, a file already in the page cache, and any case in between are left alone.** A dense bundle has a
  few hundred matrices and its walk is cheap.

On macOS the reason is different and so is the rule. The kernel reads little around a fault there, so the walk is
cheap; it is the first decode on the CPU that is slow, pulling the weights in a page at a time. So **before a load on
the CPU backend, any bundle, dense or MoE, that is not in memory and fits with room is prefetched** (the same line is
printed), and nothing is done about the read-around. On a 16 GB Mac in use, "fits with room" keeps this to small
bundles. **A load on Metal is left as it is:** there the prefetch cost a little more than it saved.

Other systems load as before; none is measured. The measurements and gates are in
[`completed/task-giw-cold-load-2026-10.md`](completed/task-giw-cold-load-2026-10.md).

## One version

A `.giw` is a cache of its source, so the format has **one version at a time**. This build writes weights format
**v17** in bundle frame **v3**, and reads only those. There is no compatibility window in either direction.

- **A sidecar rebuilds itself.** A sidecar from another version fails its freshness check on the first load (the
  reader refuses it on its header, before reading the rest of the file) and is transcoded again from the source
  beside it, once. A format bump therefore costs each cached model one transcode.
- **The sidecar's name carries the version**: `<model>.<quant>[.e4h].<target>.v17.giw`. Two builds at different
  formats on one box (a release binary and a dev build) each keep their own file, where one shared name made each
  switch rewrite the other's. A sidecar an earlier build left under the unversioned name is used when it already loads
  as this build's format: it is hard-linked to the versioned name (no copy, no transcode) and the old name stays. One in
  another format is left alone, and the build says so after it transcodes its own. **Nothing deletes an old-version
  sidecar**, because another build may be using it; each is as large as the model, so remove the ones for versions you no
  longer run (`ls <model>.*.giw`).
- **A hand-built or embedded bundle is refused.** A `.giw` named with `--model`, or built into a binary, has no
  source for the loader to rebuild from. It fails with `format vN, this build reads v17 — delete the sidecar (it
  rebuilds) or rebuild with go run ./cmd/prequant …`, and has to be rebuilt with `cmd/prequant`.
- **Why not read old versions:** the reader had accepted thirteen layouts, each with its own gates, tests and
  test-only writer switches, and a file in an old layout loaded at a real cost (scales copied to the heap) with
  nothing to say so. The `.giw-kv` session snapshots already work this way.

### From v17: a new encoding needs no new version

- **Every tensor record starts with its encoding.** A weight matrix starts with its kind byte (1 f32, 2 int8, 3 and 5
  int4, 6 a fused int4 group). A standalone array (a norm, a bias, an f32 Mamba projection) starts with encoding 1, f32.
- **A reader refuses only the encoding it does not read.** That tensor's encoding is named in a `*SerializeError`, and a
  sidecar is rebuilt as for a version mismatch. A bundle that never uses the new encoding still loads.
- **So a new encoding is added at v17,** without a v18. B25's q4k form is that case (`docs/queue-engineering.md`): a new
  weight-matrix kind for Q4_K super-blocks.
  - An older build refuses a sidecar that uses it and writes its own in place, under the same name.
  - The newer build reads that one, since it reads every older encoding.
- **What it does not cover:** a change to the layout itself, such as a new per-layer field or a new header field. A reader
  cannot skip a record it cannot parse, so that still needs a new version.
- **The KV snapshot (`.giw-kv`) is unchanged:** it keeps its own version and untagged arrays.

What the earlier versions were, and the measurements that motivated alignment, fused groups and stored f16 scales,
are in [`code-notes/decoder.md`](code-notes/decoder.md) under "The .giw format before v16".

## File layout

A bundle is a 64-byte frame header (`GINFB`, frame version, the weights length), the **weights blob**, then the
tokenizer GGUF. The blob starts at file offset 64, a multiple of 16, so every aligned array inside it is aligned in
a mapping of the file. `decoder/serialize.go`'s file comment is the field-by-field authority; in outline:

- **Header:** `GINFW`, the version, the model id, the config as JSON, the resolved quant label (empty in a streamed
  bundle, where the reader infers it from the tensors), then zero padding to a 16-byte boundary.
- **Tensors:** the embedding, head and norms, then each layer's weights in a fixed order.
- **A trailing CRC-32** over everything before it, padding included.

Every weight matrix is a record: a kind byte, its shape, then the kind's arrays. Each aliased array is preceded by
zero padding so its payload starts on a 16-byte boundary, which is what lets the loader alias the scales as well as
the codes instead of copying them to the heap.

| kind | holds | arrays |
|---|---|---|
| 0 | nothing (an absent tensor) | none |
| 1 | float32 | the values, copied at load |
| 2 | int8 | float32 row scales, then the codes |
| 3 | int4, canonical | binary16 group scales, then the packed nibbles |
| 5 | int4 in the arm64 row4 layout only | binary16 scales and nibbles in that layout; no canonical arrays |
| 6 | int4, member of a fused group | none in the record; see below |

The int4 group scales are binary16 in the file because that is their form in memory: the loader aliases them as the
weight's own storage.

### Targets

`cmd/prequant -target` and the sidecar's file name (`<model>.<quant>.<target>.v<N>.giw`) say which consumer a bundle is
promised to. The target changes only which int4 kind a tensor takes:

- **`cpu-arm64`:** a tensor whose shape allows it is kind 5, the split-half, four-row-interleaved layout the NEON
  kernel reads. A core that cannot run that layout refuses the file with a message naming the fix.
- **`metal`:** each layer's `(q, k, v)` and `(gate, up)` are written as a **fused group**, below.
- **Every other target:** kind 3 for every int4 tensor.

### Fused groups (Metal)

Metal's fused kernels read one buffer holding q‖k‖v (or gate‖up) rows back to back, and a mapped file can be bound
into such a buffer only if those rows are adjacent in the file. So for `-target metal` each such tuple is written as
a group: one kind-6 header per member (shape only), then two blocks with no length prefixes, each starting on a
16-byte boundary:

1. every member's nibbles, one after another;
2. every member's binary16 scales, one after another.

The lengths follow from the shapes, so a prefix between members is not needed, and would be exactly the gap this
layout removes. Members are group-32 int4 with the same K, a multiple of 32. A tuple with an int8 member, an absent
member (a K=V layer has no V) or a mixed K is written as ordinary records.

The Metal build (`metal/alias.go`; aliasing is on by default, `GOINFER_METAL_ALIAS=0` turns it off) checks whether a
tuple's nibbles and scales are each one contiguous run in the mapping and binds them in place if so. It is detected,
not assumed:

- A single int4 tensor's nibbles and scales bind in place from **any** bundle, because each is one aligned run.
- A fused tuple binds in place only from a `-target metal` bundle. From any other, it is copied, and the load banner
  says so and names the rebuild.
- A CPU load reads a `-target metal` bundle without trouble: the nibbles and scales are ordinary canonical int4.
