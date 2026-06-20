# Task: fast Mellum2 (and safetensors-MoE) load via a prequant int4 `.giw`

> **Audience:** internal task/planning. Goal: turn Mellum2's ~90 s cold start into a
> ~seconds mmap load, so the GPU-resident demo (and real users) don't pay the int4
> quantization on every launch.

## Problem

Loading Mellum2 (12B MoE, safetensors, 23 GB bf16) GPU-resident at int4 takes **~70–90 s**,
versus ~22 s at int8. The cost is **at-load int4 quantization** — every ~12B param is
re-quantized to group-wise int4 (nibble-pack + f16 group scales) on the CPU before the resident
upload. It's a one-time-per-launch cost, paid every time.

Measured (RTX 2070 SUPER, `docs/mellum2-resident.md`): int8 load 22 s, int4 load ~70–90 s; the
int4 *decode* is the win (20.9 / 13.2 tok/s vs int8's host-spilled 8.9 / 6.7), so int4 is the
quant we want — but the load tax makes it painful interactively.

## The fix already exists — for GGUF

goinfer has a prequant `.giw` bundle: the **already-quantized** resident weights serialized to
disk + a metadata-only GGUF for the tokenizer. Loading a `.giw` is an **mmap** (no requant, no
heap copy) → ~5× faster cold start, ~10× less heap (`internal/prequant`, `cmd/prequant`,
`docs/weight-memory-program.md`). The demo/chat embed build already bakes one (`build-embed.sh`).

`cmd/prequant -quant int4 -o out.int4.giw model.gguf` produces exactly the int4 bundle we want.

## The gap

`internal/prequant.Transcode` is **GGUF-only**: it reads the GGUF *head* for the tokenizer half
(`readHead` → `metadataPrefixLen` → `tokenizer.LoadGGUFBytes`) and streams weights via
`decoder.StreamTranscodeGGUF(in, …)`. **Mellum2 on the box is safetensors** (no GGUF), so there's
no clean prequant path today — hence the ~90 s every launch.

## Two ways to close it

### Path A — operational (no goinfer change): get a GGUF, then prequant

1. Convert Mellum2 safetensors → GGUF (llama.cpp `convert_hf_to_gguf.py`, arch `mellum`), once.
2. `go run ./cmd/prequant -quant int4 -o ~/models/mellum2.int4.giw mellum2.gguf` (one-time,
   streams one layer at a time → fits the box).
3. Load the bundle: `--model ~/models/mellum2.int4.giw --backend webgpu` (no `--quant` — the
   bundle is already int4).
4. **Verify it goes GPU-resident int4** — the `.giw` int4 weights must deserialize into the int4
   `linalg.WeightMat` that the new stacked-MoE int4 builder (`gpu/residency.go` `buildStacked`
   case `"int4"` → `UploadStackedExpertsInt4`) consumes. This is the one thing to actually test:
   a `.giw`-int4 → `ResidentActive()` true → decode matches the safetensors-int4 numbers.

Pros: no code. Cons: needs the GGUF conversion + assumes the GGUF `mellum` loader already round-
trips (it does — `decoder/mellum_gguf_test.go`), and the int4-resident-from-giw path is untested.

### Path B — code: let prequant/`--stream-weights` accept a safetensors checkpoint

Teach the transcode path to take a safetensors dir, not just a GGUF:
- Tokenizer half: build the metadata bytes from the dir's `tokenizer.json` (`tokenizer.Load`)
  instead of slicing a GGUF header — or store the tokenizer in the bundle in a non-GGUF form and
  have the loader pick it up.
- Weights half: `decoder.Load(dir, {Quant:"int4"})` once → `decoder.SerializeWeights(m.Weights(),
  name)` → `giw.WriteStream`. (`SerializeWeights` / `LoadSerializedWeights` already exist —
  `internal/prequant/stream_test.go` uses them.) Peak RAM is the resident size; acceptable here.

Pros: any safetensors model (Mellum2, future MoE) prequants with one command, no external tools.
Cons: a real change to `internal/prequant` + the `.giw` tokenizer assumption; needs its own gate.

## Acceptance

- `mellum2 → .giw (int4)` builds once; subsequent `--model *.giw --backend webgpu` loads in
  **seconds** (mmap), goes **GPU-resident int4**, and decodes at the `docs/mellum2-resident.md`
  numbers (~21 / ~13 tok/s).
- A test gates the `.giw`-int4 → resident path (the untested seam in Path A step 4).
- Doc the recommended flow in `docs/mellum2-resident.md` ("Reproduce" → add the fast-load recipe).

## References

- `internal/prequant/prequant.go` (`Transcode`), `cmd/prequant/main.go`
- `decoder.SerializeWeights` / `LoadSerializedWeights`, `decoder.StreamTranscodeGGUF`
- `gpu/residency.go` `buildStacked` (int4 stacked-MoE, the consumer), `docs/mellum2-resident.md`
- `docs/weight-memory-program.md` (the `.giw` substrate + `--stream-weights`)

## Outcome (shipped)

Both paths landed, plus a simpler-than-planned third win:

1. **Path B** (`feat(prequant): build int4 .giw from a safetensors dir`): `cmd/prequant` /
   `Transcode` accept a safetensors directory → resident-loadable int4 `.giw`. Mellum2 (and
   any safetensors-only MoE) now prequants with one command.
2. **GGUF→`.giw` rope round-trip fix**: nil `json.RawMessage` config fields marshalled to
   `null` and re-fired "present" checks; `,omitempty` keeps them absent. (Unblocks GGUF→`.giw`
   for MLA/YaRN families.)
3. **Direct int4 upload** — the "GPU-layout `.giw`" turned out unnecessary: the decoder int4
   storage is byte-identical to the GPU packed layout for K%32==0 (`gpu.TestInt4LayoutMatch`),
   so the resident upload `CreateBufferInit`s the decoder bytes directly instead of
   unpacking+`packNibbles`-repacking. Measured 52 s → 32 s on Mellum2's `.giw` load (1.6×),
   token-identical, and it benefits every int4 resident load — no new bundle format needed.

**Remaining lever (not done):** the ~32 s floor is `CreateBufferInit` + 7 GB PCIe + per-buffer
overhead (thousands of projections). Coalescing into fewer/bigger buffers is the path toward
seconds; deferred.
