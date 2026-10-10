# Task — `.giw` v16 (one readable layout) and a faster transcode (2026-10)

**Status:** scoped 2026-10-10. **T0 and T6 are built (2026-10-10, branch `giw-v16-2026-10`), and G-T1 for T6 is PASS on
the Mac** ([record](../measurements/giw-v16-transcode-2026-10/README.md)). T1–T5, T7 and V are not started. The gates
below were registered before any code. Two parts, in this
order: **T, transcode speed** (byte-identical output, so it ships on identity), then **V, format v16** (a breaking
format change; every existing sidecar rebuilds once, which T makes cheap). Owner decisions: § Decisions. Branch:
`giw-v16-2026-10`, merged when the track is done.

Evidence for § Audit lives in [`docs/measurements/transcode-audit-2026-10/`](../measurements/transcode-audit-2026-10/README.md).
Every figure there is **exploratory** (a 2-vCPU amd64 cloud box and a synthetic GGUF with real shapes): it sets
direction and the projection bands, and is never quoted as a result. The results are the gates' own runs.

## Summary

- **The transcode runs on one core.** A streaming transcode builds one layer at a time, and inside a layer only the
  GGUF expert loader is parallel. Dense models and every safetensors MoE quantize single-threaded. The resident build
  this replaced ran layers in parallel; the directory-streaming record already saw GLM-OCR go from 3.2 s to 7.5 s and
  named "parallelism within a layer" as a follow-up nobody picked up. Qwen3-VL-30B-A3B's 11 min 5 s (G-S10q-d) is the
  same follow-up at scale.
- **The int4 quantizer is scalar, and it is most of the work.** `QuantizeGroupInt4Row` (aikit) was 81% of a
  1.5B int4 transcode's CPU in the exploratory profile, `math.Round` alone 28%. The same model at int8int8 took
  3.9 s against int4's 19.5 s: the int8 quantizer is vectorized, the int4 one is not. int4 is the default quant,
  and the default sidecar (`e4h`) sends the embedding and head through the int4 quantizer too.
  - **Not on the Mac (T0, 2026-10-10, exploratory):** there it is about 14% of the CPU, and the GGUF dequantizer is
    about two thirds. See F3.
- **Three smaller costs:** on arm64 every int4 tensor built during a transcode also gets an in-RAM row4 copy that
  nothing reads; the safetensors fused-expert path transposes with a loop that hits power-of-two cache aliasing
  (a tiled version is 5.2× faster, bit-identical); and it copies each expert's f32 three more times than needed.
- **Measured and not worth doing:** buffering the writer (3,218 `Write` calls for a whole 1.5B), the self-check on a
  file that fits in RAM (0.13 s), the disk write itself (~0.5 s per GB).
- **v16:** read exactly one layout. Drop all thirteen the reader accepts today, kind 4, kind 7 and the dead f32 scale arrays that a
  Metal-target file still carries (projected: about 190 MB on a default 1.5B sidecar and 950 MB on a 7B), plus the
  side map that exists only because pre-v15 weights held f32 scales.

## Why

### The format

`decoder/serialize.go:giwMinReadV` is 3 and `giwVersion` is 15: the reader accepts thirteen layouts. What that costs:

- About thirteen version gates across `giwReader.weightMat`, `giwReader.fusedGroup`, `giwReader.int4Scales`,
  `giwReader.alignArray`, `giwReader.layer` and `loadSerializedWeights` (the v4/v6/v8/v9 family tails, the v5 quant
  label, v12 alignment, the v10 Granite refusal, the f32-to-f16 scale conversion for pre-v15 files, the pre-v14
  fused-group path).
- **Kind 4**, retired but still read and still writable through `decoder/serialize.go:SerializeWeightsRow4` and
  `SerializeWeightsToRow4`, with two test files of its own.
- A test-only writer switch, `decoder/serialize.go:giwEmitVersion` with `giwWriter.emitVersion`, so four test files can
  write old layouts and prove they still load.
- The v1 and v2 bundle frames in `internal/giw/bundle.go:Read` and `WeightsVersionFile`.
- `internal/prequant/prequant.go:minInt4CacheGIWVersion` and `cacheLayoutCurrent`, which exist to force a rebuild the
  reader would otherwise never ask for.
- A new read-side shim this week: `82d53dc6` takes an `int4mix` label from the weights on read "so a bundle written
  under that rule needs no rebuild".

And the current layout is not single-representation:

- A **v15 Metal-target file stores int4 group scales twice.** Every fused-group member (kind 6) and every kind-7 tensor
  writes an f32 scale array and then an f16 copy (`giwWriter.fusedGroup`, `giwWriter.weightMatKind`). The reader uses
  the f32 array only for a length check, and for kind 6 converts it to f16 and throws the result away.
- **Kind 7 is v15's kind 3 plus that dead array.** Kind 3 has stored aligned binary16 scales since v15.
- **The f16 side map** (`giwReader.recordF16`, `Weights.int4F16`, `decoder/model.go:Model.Int4ScalesF16`) exists
  because weights loaded from pre-v15 files held f32 scales. From v15 every int4 `WeightMat`'s own scales are the f16
  bytes in the mapping. `metal/moe.go` and `metal/gemma4_moe.go` already read them from the `WeightMat`
  (`Int4ScalesF16()`); only `metal/alias.go:weightAlias.scales16` still goes through the side map.

### The transcode

By default every `.gguf` load on darwin and linux resolves to a sidecar (`internal/prequant/prequant.go:DefaultToSidecar`), so
the first start of every model pays a transcode, and so does every format bump. Recorded times:

| model | source | box | target | wall | record |
|---|---|---|---|---|---|
| Qwen2.5-Coder 1.5B int4 | GGUF | Mac | cpu-arm64 | 21 s (1,233 MB) | `docs/measurements/swap-tripwire-negative-controls-2026-09-24.md` |
| Qwen2.5-Coder 1.5B int4 | GGUF | nobara | cpu-amd64 | 18 s | `docs/measurements/sidecar-v15-2026-09-29.md` |
| Qwen2.5 7B int4 | GGUF | nobara | cpu-amd64 | 87 s | `docs/measurements/sidecar-v15-2026-09-29.md` |
| Qwen2.5 7B | GGUF | nobara | cuda | 1 min 43 s | `docs/measurements/cold-user-2026-10-05-nobara-pc.md` |
| gemma-4-26b-a4b (MoE) | GGUF | nobara | cuda | 1 min 6 s (13.6 GB) | `docs/measurements/cold-user-2026-10-05-nobara-pc.md` |
| Qwen3.5-35B-A3B (M35) | GGUF | Mac | — | 5 min 50 s | `docs/measurements/moe-pager-mode-darwin-2026-09-23.md` |
| Qwen3-VL-30B-A3B int4 | safetensors dir | nobara | cuda | **11 min 5 s** (16.7 GB) | `docs/tasks/task-multimodal-support-2026-10.md`, G-S10q-d |
| GLM-OCR | safetensors dir | Mac | — | 3.2 s → 7.5 s when dir streaming landed | `docs/tasks/task-prequant-dir-streaming-2026-10.md`, G-DS2 |

The pattern is in the bytes per second. The gemma-4 MoE wrote about 200 MB/s and the dense 7B about 50 MB/s on the
same box. The MoE's experts are the one part of a streaming transcode that runs in parallel
(`buildWeightsFromGGUF`'s `stackedExperts` closure). The 30B safetensors MoE managed about 25 MB/s. Its experts run
one at a time. G-S10q-d was voided by exactly this: the sidecar outlasted the driver's 600 s wait.

## Audit

### Where the time goes (exploratory profile)

`decoder.StreamTranscodeGGUF` on a synthetic GGUF with Qwen2.5-1.5B's shapes and Q4_K_M's block types (random
weights; dequantize, quantize and write cost depend on shapes and block formats, not values), int4, default
target, to a counting sink. 2-vCPU amd64, Go 1.27.2, aikit v1.64.0, goinfer `5839762a`.

| | CPU share |
|---|---|
| `linalg/quant.go:QuantizeGroupInt4Row` (aikit), cumulative | **81.5%** |
| — of which `math.Round` | 28.5% |
| GGUF dequant (`dequantQ4KBlock` 7.4%, `dequantQ6KBlock` 5.2%) | 12.7% |
| int8 quantize (the int8-pinned embedding, AVX2) | 1.6% |
| `.giw` writer, CRC included | ~2.5% |

Wall 18.7 s with 18.3 s of CPU: **one core busy out of two.** The same model at int8int8: **3.9 s.** To a real file:
19.2 s plus a 0.13 s self-check (the file sits in page cache).

### Findings

- **F1. Streaming transcodes build one layer at a time on one core.** Eight GGUF family loops (the `sink != nil`
  branches in `decoder/gguf.go:buildWeightsFromGGUF`) and the safetensors `decoder/weights.go:finishLayers` each run
  "load layer i, write it, free it". The resident build runs the same per-layer loaders under `parallelLayers`, and
  the loops' own comments say that concurrency "is a speed choice, not a correctness dependency". The loaders are
  already proven safe to run concurrently.
- **F2. Inside a tensor, rows are sequential.** `decoder/weightmat.go:streamQuantizedRow4` and
  `streamQuantizedRepackable` dequantize and quantize one row at a time. The embedding table (151,936 rows on Qwen)
  runs before any layer and is 3.9% of the single-core time.
- **F3. The int4 quantizer is scalar.** `QuantizeGroupInt4Row` rounds every element with
  `int(math.Round(float64(x)))`. aikit already solved this for int8: `quantizeRowInt8Core` is arch-dispatched (NEON
  `FCVTAS`, AVX2) with `quantizeRowInt8CoreScalar` kept verbatim as the bit-identity oracle. int4 never got the same
  treatment. On amd64 `math.Round` is not an intrinsic; on arm64 it compiles to `FRINTA`, so the Mac's split differs
  and T0 measures it.
  - **T0, 2026-10-10 (the Mac, exploratory; the audit folder's arm64 section):** the split differs completely.
    `QuantizeGroupInt4Row` is about 14% of a 1.5B int4 transcode's CPU, and `math.Round` is not a node in the profile.
    The GGUF dequantizer is about two thirds: `dequantQ4KBlock` 55%, of which `encoding/binary.LittleEndian.Uint16`
    17%, and `dequantQ6KBlock` 12%.
  - So on arm64 T4 can remove at most about 14%, and the dequantizer is the first cost there before T4, not after it.
- **F4. Safetensors experts are built one at a time.** `decoder/weights.go:loadFusedExperts`, `streamExperts`
  (gemma4), and the per-expert `loadMatQ` loop in the generic builder's layer loader all loop over experts
  sequentially. `buildGptOssWeights` does too, but gpt-oss directories still take the resident (layer-parallel) path.
  The GGUF side parallelizes experts when streaming (`stackedExperts`); the safetensors side never got that.
- **F5. `loadFusedExperts` spends half its time not quantizing.** One Qwen3-VL-30B-A3B layer (128 experts, the
  transposed transformers-4.57 layout), single core: 22.1 s. Quantize 52%, transpose 29%, bf16-to-f32 conversion
  10%, copies 8%.
  - `decoder/weights.go:transposeF32` writes with an 8 KB stride, so every column of a tile lands in the same L1 set.
    Plain 32×32 blocking does not help (1.0×), and neither does reusing the buffer (it is not page faults). Transposing
    through a 16×16 local tile buffer and writing contiguous runs is **5.2× faster, bit-identical**.
  - `append([]float32(nil), guE[:half]...)` copies gate, up and down into fresh slices. Those slices already are fresh:
    a BF16/F16 `SubF32` allocates, and so does the transpose. Only an F32 `SubF32` (a zero-copy view of the mapping)
    under `quantNone` needs a copy.
- **F6. On arm64 the transcode repacks every int4 tensor into row4 in RAM, and nothing reads it.** Both
  `StreamTranscodeGGUF` and `StreamTranscodeDir` build with `skipRow4=false`. So `repackW4A8IfEligible` runs
  `RepackInt4Row4` (or `repackedOnlyOrCanonical` keeps canonical plus row4) on every eligible tensor. The writer never
  reads that copy: it chooses from canonical and, for a cpu-arm64 target, recomputes row4 in `repackRow4ForEmit`. So
  every Mac transcode does one wasted repack per tensor, and a cpu-arm64 one does it twice.
  - **T0, 2026-10-10 (the Mac, exploratory):** confirmed, and small. The in-RAM repack is 0.21–0.25 s, about 2% of the
    CPU, on both targets. The writer's own repack for kind 5 is another 0.22 s on cpu-arm64, and that one is needed.
- **F7. The writer round-trips scales through f32.** `giwWriter.weightMatKind` widens the stored binary16 scales to
  f32 (`Int4F32`, an allocation), `giwWriter.int4Scales` narrows them back (`F32ToF16Scales`, another), and
  `giwWriter.scales16` copies them to bytes (a third). About 2% of the profile; fold it into V1's writer change.
- **F8. NaN rounds differently per arch in the int4 quantizer today.** `int(math.Round(NaN))` is the int64 minimum on
  amd64 (`CVTTSD2SQ`, clamped to −7) and 0 on arm64 (`FCVTZS`). The int8 path documented why both arches give 0; int4
  never pinned it. A SIMD version has to match each arch's existing output, so the oracle test must include NaN.
- **F9. The self-check re-reads the whole file.** That matters only when the file is larger than free RAM. For a file
  that fits, the read comes from page cache and compares the bytes Go wrote with the CRC Go computed over those same
  bytes. That checks the writer against itself, not the disk.
  - **T0, 2026-10-10 (the Mac, by day):** the phase line reads 3.0–5.0 s of self-check on a 1.2–1.3 GB int4 sidecar
    with int8 logit tables, and 0.2 s on the `e4h` one, where the cloud box read 0.13 s.
  - The same load in a fresh process takes 0.2 s, and the self-check's CPU is 0.5–0.6 s. So the time is not the
    check's own work. What it waits on is not established
    ([record](../measurements/giw-v16-transcode-2026-10/README.md)).

### Checked and not worth doing

- **A buffered sink.** The transcode makes 3,218 `Write` calls for the whole 1.5B. The small header writes are real
  but negligible. A large MoE makes more (about ten per matrix), still under a second.
- **The self-check for a file that fits in RAM:** 0.13 s of 19.3 s. **Not so on the Mac (T0, 2026-10-10): 3–5 s of a
  15–20 s transcode for a plain int4 sidecar. See F9; it is an open question, not a closed one.** A sidecar larger than RAM (the 16.7 GB 30B on the
  16 GB Mac) re-reads from disk. T0's phase line measures that before anything changes (F9).
- **The disk write:** about 0.5 s per GB on the cloud box. Not representative of the Mac's SSD, but not where the time
  is either.
- **GGUF dequant (Q4_K/Q6_K, scalar):** 12.7% today. It becomes a larger share once T4 lands. Re-rank it from T0's
  profile after T4, not before. **On arm64 it is already about 67% (T0, 2026-10-10; F3): on the Mac it ranks first
  now.**

## Part T — transcode speed

Every T item is **byte-identical**: the `.giw` a new build writes equals the old build's byte for byte (G-T1). They
change what runs when and how fast, never what is written.

- **T0. Instruments (first, by day).**
  - **Phase line.** `internal/prequant/prequant.go:Transcode` times the stream and the self-check separately.
    `EnsureCachedGIW`'s "cache ready" line and `cmd/prequant`'s "wrote" line print both
    ("… in 21s: stream 20.1s, self-check 0.9s"). Cheap, permanent, and it answers F9.
  - **`gate identity --giw` (two modes).** It reuses `cmd/gate/identity.go`'s two-worktree machinery. Each revision
    builds its own `cmd/prequant` and transcodes the shared source into a per-revision temp directory.
    - `--giw bytes` compares the files' sha256 (G-T1).
    - `--giw logits` dumps logits from each side's own file with the existing dumper (G-V1).
    - Today's `.giw` identity assets are fixed pre-built files, which a v16 build would refuse. This mode is what lets
      any future format change be gated the same way.
  - **A `-cpuprofile` flag on `cmd/prequant`** (a flag, not an env read), and with it a profile on the Mac of the
    1.5B int4 cpu-arm64 and Metal transcodes. Under a minute each: day tier, exploratory. That gives arm64's split for
    F3 and F6.
  - **Built 2026-10-10.** What the tree made different from the text above:
    - `Transcode` keeps its signature. `internal/prequant/prequant.go:TranscodeTimed` and `TranscodeLoRATimed` return
      the two phases, and `Transcode` wraps the first.
    - `cmd/prequant` refuses a GGUF with no embedded tokenizer, and no tiny GGUF fixture has one. So `-giw` covers the
      tiny safetensors directories and real GGUFs; the tiny GGUFs are not in its matrix.
    - `-giw bytes` hashes each bundle and removes it before the next is written, so a real cell needs one sidecar's
      worth of free disk, not three.
- **T1. Windowed layer parallelism in every streaming loop.**
  - One helper replaces the nine loops (eight in `buildWeightsFromGGUF`, plus `finishLayers`): build a window of K
    layers with `parallelLayers`, write them in order, free them, repeat. Output is byte-identical by construction,
    because each layer's bytes do not depend on another's and the write order is unchanged.
  - K is the smaller of `GOMAXPROCS` and a memory budget divided by one layer's bytes. A layer's bytes are known
    after layer 0, which is built alone (the sink's byte count across it). The budget is Decision 3.
  - **A MoE family keeps K = 1** and parallelizes its experts inside the layer (T3). Nesting both would multiply
    in-flight transients, and a MoE layer is the large one.
  - Exploratory, 2 cores: 16.2 s to 9.7 s with T4a in both arms, identical sha256.
- **T2. Row-parallel logit tables.** The embedding and an untied LM head are single tensors, so T1 cannot reach them.
  - Give `streamQuantizedRepackable` a parallel form over contiguous row ranges that takes a row-reader *factory*, so
    each worker owns its reader's state.
  - The GGUF `embMat` reader is a pure function of the mapping and can be shared. The safetensors streaming reader in
    `decoder/weights.go:loadEmbedTable` keeps a 1,024-row block cache, so it needs one per worker, with ranges aligned
    to its block.
  - Do not parallelize `rowInto` generally: a stateful reader is exactly the case that breaks.
- **T3. Parallel experts on the safetensors side.**
  - `loadFusedExperts`, `streamExperts` and the generic per-expert loop get what `stackedExperts` has: `parallelLayers`
    over experts when building for a sink, sequential otherwise. In the resident build they already sit inside
    parallel layers, so nesting would multiply transients.
  - Each expert writes only its own slot; the source reads are pure.
- **T4. A fast int4 quantizer (aikit).**
  - **T4a, scalar, now.** Replace `int(math.Round(float64(x)))` with
    `f := float64(x); if f < 0 { int(f - 0.5) } else { int(f + 0.5) }`. For any float32 x the widening is exact, and
    so is the ±0.5 for 2⁻²⁹ ≤ |x| < 2⁵¹. Below that the sum stays inside (0, 1), so truncation still gives 0. Inside
    the quantizer |x| ≤ 7.000002 whenever it is finite. NaN and ±Inf convert exactly as before.
    - Checked exhaustively over every float32 with |x| ≤ 2²⁴ (1,266,679,809 bit patterns, both signs) plus ±Inf and
      NaN: zero mismatches.
    - Exploratory: −17% on the amd64 transcode, identical sha256. Expect less on arm64 (F3).
  - **T4b, SIMD.** `QuantizeGroupInt4Row` gets the int8 treatment.
    - Arch dispatch, with the current scalar body kept verbatim as `quantizeGroupInt4RowScalar`, the oracle.
    - Per 32-element group: SIMD max-abs (exact and order-independent, already argued for int8), the scalar scale and
      inverse, a SIMD multiply with no FMA, then round-half-away, clamp to ±7 and pack nibbles.
    - NEON has `FCVTAS` for the rounding. AVX2 has no ties-away mode: truncate (`VROUNDPS` toward zero), compute the
      exact remainder, and step away from zero where |remainder| ≥ 0.5.
    - The ceiling to aim at: int8int8's whole transcode is 3.9 s against int4's 19.5 s on the same model.
    - `QuantizeGroupsInt4` and `QuantizeInt4` (the resident and safetensors paths) inherit it.
- **T5. `loadFusedExperts` without the waste.**
  - A tiled transpose that converts from bf16 on the way. It needs a small aikit API, because the raw bytes are
    unexported, for example `Tensor.SubF32Transposed(start, rows, cols)`. 16×16 tiles through a local buffer, written
    out as contiguous runs.
  - Drop the three `append` copies except where the slice aliases the mapping and the mode keeps f32 (F5).
- **T6. No in-RAM repack while building for a sink.** In a streaming transcode nothing decodes, so no layout but
  canonical is needed in RAM.
  - Route every repack decision through one predicate that is false when building for a sink. That covers
    `streamQuantized*`, `repackedOnlyOrCanonical` and the embedding's `streamQuantizedEmbed`, which hard-codes
    `skipRow4=false` today.
  - The writer already builds row4 from canonical (`repackRow4ForEmit`), so the file does not change.
  - It has to be checked on the Mac: on amd64, row4 is a no-op.
  - **Built 2026-10-10.** The predicate is `decoder/weightmat.go:repackForDecode`. The two builders that can have a sink
    (`buildWeightsFromGGUF`, `buildWeightsFromSafetensorsTo`) fold it into `skipRow4` before any tensor is built, and
    `streamQuantizedEmbed` takes it as a parameter. The amd64 split-half repack, off by default, is not covered. G-T1
    on the Mac: PASS.
- **T7. Scales stay binary16 through the writer.** Write the `WeightMat`'s own f16 scales straight out (a
  little-endian view, or an encode on a big-endian host). For kind 5, repack with aikit's
  `RepackW4A8Row4ScalesF16` instead of the f32 form. Small (F7); do it inside V1, which rewrites these functions
  anyway.

## Part V — format v16

- **V1. The writer emits v16, one layout for every target.**
  - Kinds 0–2 (empty, f32, int8) are unchanged. Int4 kinds: **3** (canonical, aligned binary16 scales), **5** (row4-only, cpu-arm64 target), **6** (fused-group
    member, Metal target).
  - **Kind 7 goes.** On a Metal target an eligible single tensor is kind 3, because one tensor's f16 scales are already
    a single aliasable run.
  - **A kind-6 member record drops its f32 scale array.** Its scale count follows from its shape, and the group's f16
    block, after the nibbles, is the only copy.
  - **Kind 4 goes**: `giwWriter.row4`, `SerializeWeightsRow4`, `SerializeWeightsToRow4`.
  - `giwEmitVersion` and `giwWriter.emitVersion` go. Alignment becomes unconditional.
  - The header's legacy `quant` tag: if nothing reads it except the validation, drop it, and the quant label stays the
    authority.
  - The label is written correctly, so the reader no longer needs `82d53dc6`'s shim.
- **V2. The reader reads exactly v16.**
  - Every version constant and gate goes. So do the f32 branch of `giwReader.int4Scales` and the kind-4 and kind-7
    cases of `giwReader.weightMat`.
  - `giwReader.fusedGroup` keeps only its v16 path, `loadSerializedWeights` loses the v5 check, the Granite v10
    refusal and the label shim, and the v4/v6/v8/v9 tails become unconditional.
  - Any other version is a `*SerializeError` naming the fix: "format v15, this build reads v16 — delete the sidecar
    (it rebuilds) or rebuild with `go run ./cmd/prequant …`".
- **V3. The side map goes.** `giwReader.f16`, `giwReader.recordF16`, `Weights.int4F16` and `Model.Int4ScalesF16` are
  removed. `weightAlias.scales16` reads `w.Int4ScalesF16()` as `metal/moe.go` already does, and keeps its contiguity
  check against the mapping. Side benefit: Metal can then bind kind-3 scales in place from any v16 file, not only a
  Metal-target one.
- **V4. The bundle frame reads v3 only.** `giw.Read` and `WeightsVersionFile` drop the v1 and v2 paths. The frame's
  own version stays 3; it does not change.
- **V5. Cache freshness simplifies.** `minInt4CacheGIWVersion` and `cacheLayoutCurrent` go: a pre-v16 sidecar fails
  `selfCheck` inside `cacheFresh` and is rebuilt once, which is the path that rule was added to force. An embedded or
  explicitly named pre-v16 bundle is refused with V2's message.
- **V6. Tests.**
  - Delete the old-layout tests: `TestGIWAligned_legacyLayoutStillLoads`,
    `TestGIWF16_otherTargetsAndOlderFilesCarryNone` (its older-file half), the emit-version cases in
    `decoder/giwfused_test.go`, `TestGIW_refusesPreFixDenseGraniteBundle`, `decoder/w4a8_row4_giwkind_test.go`,
    `decoder/w4a8_row4_giwkind_loadcost_test.go`, `TestCacheFresh_pre15Int4Rebuilds`, and the v1/v2 cases in
    `internal/giw/bundle_test.go`.
  - Add three tests.
    - A v15-stamped blob (version field and CRC patched) is refused with V2's message.
    - A pre-v16 sidecar is rebuilt once by `EnsureCachedGIW`.
    - A kind-3 tensor's scales bind in place on Metal.
  - `TestStreamTranscodeMatchesResident` and its `_metalTarget` twin stay as they are.
- **V7. Docs and policy.**
  - `docs/giw-bundles.md` is rewritten around one version and the policy below.
  - The Versions history in `decoder/serialize.go`'s file comment moves to `docs/code-notes/decoder.md` (the
    comment rule).
  - A CHANGELOG entry: a breaking change for a hand-built or embedded `.giw`; sidecars rebuild by themselves.
  - **Policy, if Decision 2 agrees:** the reader reads exactly the current version, and a bump means a rebuild. The
    `.giw-kv` snapshots already work this way ("a regenerable cache, so a version bump just triggers a cold prefill").

What v16 does not change: kind 5's one-target promise, the alignment rules, the CRC, the bundle frame, and every
number a tensor holds. A cpu-target v16 file should differ from its v15 twin only in its header and CRC. G-V2 checks
exactly that.

## Gates (registered 2026-10-10, before any code)

- **G-T1, byte identity (correctness; every T item).** `gate identity --giw bytes <base> <new>`: equal sha256 for
  every cell or the item does not ship.
  - **By day, tiny assets:** G-DS1's matrix (42 fixtures × int4 metal with and without `-embed-int4`, int4 cpu,
    int8int8), plus int4 cpu-arm64. Run on the Mac, so T6's arm64 path is covered.
    - **What that matrix is on the Mac (2026-10-10):** the walk also takes `decoder/testdata/` and finds 64 fixture
      directories, of which 55 transcode (46 under `testdata/`; 42 was G-DS1's count on 2026-10-09). `cpu` resolves to
      `cpu-arm64` there, so "int4 cpu" and "int4 cpu-arm64" are one cell written twice.
    - **One cell added, `int4:cpu-arm64:e4h`:** the int4 logit tables are a path T6 changes, and no registered cell
      writes them for a CPU target.
    - **T6: PASS, 2026-10-10.** 330 tiny cells and 3 real cells IDENTICAL, 0 DIFFERENT; 54 NOT RUN, nine directories
      that are not models, each listed in the record. The real 1.5B cells ran by day, ahead of the night this gate
      names for them: an identity check of about 20 s a transcode, nothing timed.
  - **At night, real:** 1.5B int4 cpu-arm64 and metal (Mac); 7B int4 cpu-amd64 and cuda, and Qwen3-VL-30B-A3B int4
    from its directory (nobara).
  - T4 also passes aikit's oracle test (G-T4) before goinfer sees it.
- **G-T2, speed (night, two arms: new and old binaries, no peer, TE5(a)).**
  - Measured: full transcode wall (stream plus self-check, from T0's phase line), source in `~/models`, one discarded
    warm-up per arm, then ABAB pairs. Graded on the paired new/old ratio.
  - Bands are from § Audit: the single-core split, 8 cores at 0.6–0.85 efficiency, and T4a's 0.83× on amd64 (taken
    as 1.0 on arm64).

  | cell | expected | ship | park | kill |
  |---|---|---|---|---|
  | dense GGUF int4 (1.5B Mac cpu-arm64; 1.5B and 7B nobara cpu-amd64), T1 + T2 + T4a + T6 | 0.18–0.33 | ≤ 0.45 | 0.45–0.70 | > 0.70 |
  | Qwen3-VL-30B-A3B int4 dir (nobara), T3 + T4a + T5 | 0.18–0.25 | ≤ 0.40 | 0.40–0.65 | > 0.65 |
  | T4b alone, single-thread transcode CPU time against T4a (in-process, day tier, TE5(b)) | 0.4–0.6 | ≤ 0.70 | 0.70–0.90 | > 0.90 |

  A kill means the parallelism is not happening, or the kernel is no faster. It goes back to T0's profile, not to a
  wider band. T6 ships on G-T1 alone, because it only removes work; its ratio is recorded.
- **G-T3, memory (the same night runs).** Peak footprint (`/usr/bin/time -l` on the Mac, `-v` on Linux).
  - Dense: at most the old peak plus 1.5× K layers' bytes, and never above Decision 3's budget.
  - MoE: at most the old peak plus workers × 3 × one expert's f32.
  - Mac: swap flat across every run (`vm.swapusage` before and after). The never-swap rule is not traded for speed.
- **G-T4, kernel oracle (aikit, day).** The fast `QuantizeGroupInt4Row` equals `quantizeGroupInt4RowScalar` byte for
  byte on both arches.
  - Inputs: every float32 in [−8, 8] as a single-element group; random rows at scales 10⁻⁶–10⁶; groups that are all
    zero, all equal, or have one huge outlier; ±0; subnormals; ±Inf; NaN (F8).
  - Odd `cols` with the pad-nibble contract.
- **G-V1, logits identity (V).** `gate identity --giw logits <base> <new>`: each side loads the sidecar its own build
  wrote (v15 against v16).
  - Byte-identical logits for every family on the CPU, by day on the tiny assets.
  - On Metal, the same by day on the tiny assets and at night on the 1.5B and 7B.
  - The numbers in the file are the same bits, so anything short of identity is a defect.
- **G-V2, sizes (V, by day on tiny assets; at night on real ones).**
  - A cpu-target v16 file is the same size as its v15 twin. It differs only in the header (the version field, plus the
    legacy quant tag if V1 drops it) and the CRC.
  - A Metal-target file shrinks by the dead f32 arrays, within ±5% of the projection.
    - Default `e4h` sidecar, where the int4 embedding and head are kind 7 too: 62 MB on the 0.5B, 193 MB on the 1.5B,
      952 MB on the 7B.
    - Plain int4: 45 MB, 164 MB, 816 MB.
  - **Kill V1's layout change if a Metal file does not shrink:** the dead array is not what § Why says it is.
- **G-V3, Metal binding.** The alias banner's "N MB bound in place" is at least v15's figure on the 7B. Metal decode
  is byte-identical (G-V1).
- **G-V4, refusal and rebuild.** V6's three new tests pass, plus one manual run on the Mac: a v15 sidecar is rebuilt
  on the first `serve` and loads.

## Night jobs and estimates

The first night grades T; a later night grades V.

- **T, nobara, about 45 min:**
  - 1.5B and 7B int4 cpu-amd64, three ABAB pairs each: old about 20 s and 90 s, new about 5 s and 20 s, so about 7
    minutes.
  - 7B cuda target, two pairs: about 5 minutes.
  - Qwen3-VL-30B-A3B int4 directory, one warm-up per arm plus two pairs: old 11 minutes, new about 2, so about 30
    minutes.
  - G-T1's real cells ride the same runs (the sha256 of each written file).
- **T, Mac, about 12 min:** 1.5B int4 cpu-arm64 and Metal targets, three pairs each (about 4 minutes); 7B int4 Metal,
  two pairs (about 8 minutes).
- **V, about 15 min:** G-V1 and G-V3 on the Mac's 1.5B and 7B Metal sidecars (about 8 minutes); G-V1 on nobara's 7B
  cuda and cpu-amd64 sidecars (about 5 minutes); G-V2's real sizes ride the same runs.
- Shape: `docs/measurements/giw-v16-transcode-2026-10/run-*.sh` with pre-built old and new binaries, queued with
  `scripts/night.py add`. Records are committed the next morning.

## Order and cost

1. **T0** (half a day): the phase line, `gate identity --giw`, and the Mac profile. **T6** (an hour) rides along: it
   is the smallest byte-identical change and exercises the new identity mode on arm64 first.
2. **T1 + T2** (half a day, goinfer only). **T4a** (an hour, aikit patch release).
3. **T3 + T5** (half a day plus an aikit release for the transposed read).
4. Night: G-T1 real cells, G-T2 and G-T3 on both boxes.
5. **T4b** (one to two days: NEON, AVX2, the oracle; aikit minor release). Day-tier kernel A/B, then folded into the
   next night's G-T2 cells.
6. **V1–V7** (one to one and a half days), G-V1 to G-V4. Every sidecar rebuilds once, at T's speed. The order
   matters: `docs/tasks/task-pull-to-giw-2026-09.md` counted 275 GB of sidecars on one box.

## Decisions (owner)

**Decided 2026-10-10, after T0 and T6:** Decision 1, T before V ("t before v"). Decision 3, option (a), a quarter of
available RAM at transcode start ("a quarter of ram is fine"). Decisions 2 and 4 are open.

1. **Order:** T first, then V (recommended, so the rebuild wave after v16 runs at the new speed), or V first.
2. **Policy: read exactly one version** (recommended).
   - It is safe today because every `.giw` is a sidecar next to its source, or is built into a release binary at tag
     time.
   - If `task-pull-to-giw-2026-09.md` ships with delete-the-`.gguf` by default, a format bump becomes a re-download
     for those models. Its Decision 2 (a), refuse and name the command, already handles that case. Revisit this policy
     when P2 lands.
3. **T1's memory budget:**
   - (a) a quarter of available RAM at transcode start (recommended; a 7B at K = 8 is about 1 GB);
   - (b) a fixed cap such as 2 GB;
   - (c) K = `GOMAXPROCS` with no budget.

   MoE keeps K = 1 in every option.
4. **Hand-built `.giw` files** in `/srv/models` (the archive) stop loading at v16 and are rebuilt from their sources
   with `cmd/prequant`. Say if any of them has no source left.

## Prior art read

- `docs/tasks/task-prequant-dir-streaming-2026-10.md`: G-DS2's "parallelism within a layer" follow-up, which T1–T3
  are; G-DS1's 126-configuration identity method, which G-T1 reuses.
- `docs/tasks/task-pull-to-giw-2026-09.md`: Decisions 2 and 4, and the sidecar census.
- `docs/completed/task-moe-streaming.md` and the `stackedExperts` comment: why expert parallelism is limited to the
  streaming path.
- aikit's `quantizeRowInt8Core` and `quantizeRowInt8CoreScalar`: the vectorize-with-an-oracle pattern T4b copies.
- `docs/measurements/sidecar-v15-2026-09-29.md` and `docs/giw-bundles.md`: what v12–v15 bought, which v16 keeps.

<!-- doc-reviewed: 2026-10-10 -->
