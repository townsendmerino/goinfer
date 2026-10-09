# Task — stream a safetensors directory into its .giw one layer at a time (2026-10)

**Status:** BUILT 2026-10-09; G-DS1 to G-DS3 PASS (two amendments, with their mechanisms, below); G-DS4 in the commit.
(Scoped 2026-10-09, gates registered before any code; owner: "build this".)

## Why

- `cmd/prequant` (and the S18 directory sidecar) builds a `.giw` from a safetensors directory through `transcodeDir`.
  That calls `decoder.Load` for the whole model, then serializes it.
- So the build holds every layer at once, plus the embedding as f32 before it is quantized (`loadMat`: 4.2 GB for a
  256k-vocabulary, 4096-wide table).
- **Measured 2026-10-09:**
  - Command-R7B's Metal sidecar priced at 8.3 GB against a 6.4 GB fit-guard budget, with 9.1 GB free on the 16 GB
    Mac. It needed an owner-approved guard bypass, though the file it wrote is 5.6 GB.
  - Aya's ran out of disk.
- The GGUF path (`decoder.StreamTranscodeGGUF`) has streamed every family since S2: write the header and globals, then
  build, write and free one layer at a time. Peak is about the globals plus one layer.

## What changes

1. **`decoder.StreamTranscodeDir(ctx, dir, out, quant, embedInt4, target, id)`**, the directory twin of
   `StreamTranscodeGGUF`.
   - It resolves the config and architecture as `loadWeights` does.
   - It writes back the EOS ids `decoder.Load` resolves (`resolveEOSIDs`, the M-04 lesson on the GGUF path).
   - It calls `buildWeightsFromSafetensors` with a sink.
2. **`buildWeightsFromSafetensors` takes the sink** (nil = today's resident build, unchanged).
   - **Generic builder:** after the globals (and Gemma 4's model-level PLE inputs and per-layer FFN widths, which the
     head carries), write the head, then load, write and free each layer in order. The resident build keeps
     `parallelLayers`.
   - **The embedding (and an untied head) load a row at a time** (`streamQuantizedEmbed`) when streaming, instead of
     as one f32 tensor. Byte identity (G-DS1) decides whether this is allowed; if the bytes differ, it stays whole and
     the record says so.
   - **Dedicated builders** (gpt2, granite, nemotron, phi3, glm_ocr, internlm2, spark2_5, llama4, gpt-oss): each that
     is straightforward to give the same loop gets it.
     - Any that does not keeps the resident transcode. `transcodeDir` names it ("streams: no, <family>") rather than
       failing.
     - The record lists which stream.
3. **`transcodeDir` streams by default.** A LoRA merge keeps the resident path, since the merge needs the f32
   projection and the adapter.
   - The streamed transcode does not run `decoder.Load`'s fit guard, matching the GGUF streamer.
   - The self-check after the write is unchanged.

## Gates, registered before any code

- **G-DS1, byte identity (the correctness gate).**
  - For every fixture `TestDirSidecar_matchesDirectLoad` covers, at int4 for the metal and cpu-arm64 targets and at
    int8int8: the streamed bundle is byte-identical to today's resident transcode of the same directory.
  - A family that cannot stream is listed, and its bundle comes from the resident path, so it is identical by
    construction.
  - **Planted defect:** writing one layer out of order must turn the gate red.
- **G-DS2, peak memory** (by day, real checkpoints from `~/models`; `/usr/bin/time -l` maximum resident set size,
  old path against new):
  - **gemma-3-4b-it, int4 metal.** Ship: new peak <= 50% of old. 50-75% parked, and investigated before shipping.
    Over 75%: the change failed its purpose.
  - **glm-ocr:** reported; it ships whether or not glm_ocr streams in this task.
  - The two bundles must also be byte-identical (`cmp`).
- **G-DS3, the motivating case.** Command-R7B's Metal sidecar builds on the 16 GB Mac with no fit-guard bypass.
  - Pulled again for the second limit.
  - Peak RSS reported.
  - The bundle's own self-check passes, or its failure is recorded with its cause.
- **G-DS4, no regression.** `go test ./internal/prequant/` and `go run ./cmd/gate quick` green.
- **Wall time, recorded, not gated:** old against new on gemma-3-4b-it. A one-time build may be slower once it is
  sequential; more than 2x slower gets a note and a follow-up (parallelism within a layer), not a block.

## Amendments after the first runs (bars unchanged in intent; each with its mechanism)

- **G-DS1: "byte-identical" became "byte-identical outside the v5 quant-label field and the CRC that covers it".**
  - **Why:** a streamed bundle's header is written before any layer exists, so it records the label as "" and a reader
    infers it from the tensors. This is B11, the same as every `StreamTranscodeGGUF` bundle since that work.
  - The label cannot be computed up front: it depends on every body tensor's kind (`int4mix` when int4 sits beside f32
    attention, as in qwen35), not on the requested quant.
  - The comparison is the GGUF streamer's own (`giwSplit`): everything before the label, and everything after its
    zero pad, byte for byte, with both CRCs valid.
- **G-DS2: graded on the peak memory FOOTPRINT, not the maximum resident set size.**
  - **Why:** the transcode reads the bf16 checkpoint through mmap and the self-check maps the finished bundle, so RSS
    counts gigabytes of clean, reclaimable file pages the build does not hold. On gemma-3-4b, RSS read 5.9 GB where
    the footprint read 1.85 GB.
  - The footprint is what the fit guard is trying to protect.
  - The pre-registered model's old arm (gemma-3-4b) could not run: the fit guard refused it (5.6 GB against a 4.6 GB
    budget). Its ratio uses that estimate, and GLM-OCR and Qwen2.5-VL-3B were added as cells.

## Results, 2026-10-09 (by day, the M1 Pro, checkpoints from `~/models`; logs in `docs/measurements/prequant-dir-streaming-2026-10/`)

- **G-DS1 PASS:**
  - **Tiny fixtures:** 126 configurations (42 fixtures x int4 metal + int4e, int4 cpu, int8int8) are identical outside
    the label; only gpt2 takes the resident path.
  - **Planted defect:** with layers 0 and 1 swapped, the comparison fails 35,200 bytes in.
  - **Real GLM-OCR bundles:** old and new (520,714,911 bytes each) are identical outside the label.
- **G-DS2 PASS,** peak memory footprint of the transcode alone (`transcode-only-*.log`; the self-check adds nothing
  measurable):

  | checkpoint | old | new | new/old | wall old -> new |
  |---|---|---|---|---|
  | GLM-OCR | 1.67 GB | 0.41 GB | 25% | 3.2 -> 7.5 s |
  | gemma-3-4b-it | refused by the fit guard (its estimate 5.6 GB) | 1.41 GB | ~25% of the estimate | n/a -> 52 s |
  | Qwen2.5-VL-3B | refused (estimate 3.9 GB) | 0.95 GB | ~24% of the estimate | n/a -> 36 s |

  - **Slower:** GLM-OCR's wall time more than doubled. The cause is the sequential layer loop; the resident build
    quantizes layers in parallel, and the GGUF streamer pays the same cost. Recorded as registered, with a follow-up:
    parallelism within a layer.
  - Measured and dropped: returning memory to the OS after each layer bought nothing (same footprint, same wall).
- **G-DS3 PASS:** Command-R7B's Metal sidecar built through the `prequant` CLI with no fit-guard bypass, on a Mac with
  about 6.5 GB available.
  - Peak footprint 2.31 GB, against the 8.3 GB the old path priced and was refused for.
  - 99 s; self-check passed; 5,597 MB written, the same size as the bypass-built one that morning.
- **What changed besides the streaming itself:**
  - Embeddings and untied heads load in blocks of 1,024 rows when streaming (byte-identical).
  - The logit tables are released once the head is written.
  - The fit guard's directory remedy now names the streamed `.giw` route (`decoder/fitguard.go`).
  - Families on the resident path: gpt2 (and internlm2 and gpt-oss, whose fixtures do not reach the comparison).

## Cost

- Code: a few hours (the generic builder, `StreamTranscodeDir`, `transcodeDir`'s switch, and the dedicated builders
  that fit).
- Gates: G-DS1 a few minutes; G-DS2 two builds of a 4B model (a few minutes); G-DS3 a 15 GB pull (about 5 minutes)
  and one build.
