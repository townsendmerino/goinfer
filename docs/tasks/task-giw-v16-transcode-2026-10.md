# Task — `.giw` v16 (one readable layout) and a faster transcode (2026-10)

**Status:** scoped 2026-10-10. **Part T is built and graded** ([record](../measurements/giw-v16-transcode-2026-10/README.md)):
T0-T6 with T4b, on aikit v1.66.0. G-T1 PASS on tiny and real cells on both boxes. G-T2 SHIP on every graded cell: new/old
wall 0.128 on the Mac's 1.5B, 0.065 and 0.051 on nobara's 1.5B and 7B, 0.18 on the 30B directory. G-T3 within its
bounds. G-T2c: the transcode is 2.7-3.1 times its own I/O on the dense cells and 1.8 on the 30B; its registered
follow-up (a vectorized GGUF dequantizer) opened on the Mac's cell by a hair and is **built and graded as T8**, on aikit
v1.67.0: single-thread transcode CPU new/old 0.847 and 0.809 for the Q4_K kernel (Mac, nobara), then 0.931 and 0.864
for the Q6_K kernel on top; bytes identical. F3's arm64 profile is retracted. T0-T6 with T4b are on `main`; T8's pin
and records are on branch `giw-v16-dequant-2026-10`. **Next: T7 and V** (Decisions 2 and 4 are made). The gates
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
  - **WITHDRAWN 2026-10-10 (F3's retraction):** the Mac is like amd64. The system sampler puts the quantizer at 83%
    of a 1.5B GGUF transcode's CPU there; the 14% was Go's profiler misattributing it.
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
  - **RETRACTED 2026-10-10: the three bullets above and the one below are wrong. The quantizer leads on the Mac too.**
    - **How it showed:** T4b's gate read 0.304 on the Mac's 1.5B GGUF, single-thread CPU 10.5 s to 3.2 s, where "14%"
      allows no better than 0.86. The only code that differs between the two builds is the quantizer.
    - **What is true (macOS `sample`, which suspends the process to read its stacks; one thread, 1.5B GGUF to int4
      `cpu-arm64`):** before T4b `QuantizeGroupInt4Row` is 83% of the on-CPU samples, `dequantQ4KBlock` 5.8% and
      `dequantQ6KBlock` 1.5%. After T4b: `dequantQ4KBlock` about 24%, the new kernel about 21%, CRC 8%, the row4 repack
      for the file 7%. For the Qwen3 0.6B bf16 directory before T4b: the quantizer about 44%, the bf16 widening
      (`BFloat16sToF32` and `SubF32`) about 43%.
    - **What was wrong:** Go's CPU profile of the same old build, on one thread, gives `dequantQ4KBlock` 54%,
      `LittleEndian.Uint16` 19% and the quantizer 12%: the reverse of the sampler. Why is not established here. The
      audit's own run had signs of it: 11.2 s of samples in 15.8 s of wall.
    - **Reach:** every share in this doc and in the two records that was read from a Go profile on the Mac is
      suspect, and each is marked where it stands: this finding, F6's "about 2%", F9's "the self-check's CPU is
      0.5-0.6 s" and its note on LFM2.5's load, and the dequantizer's rank under "Checked and not worth doing". Go
      profiles taken on Linux (the audit's amd64 one, nobara's) are not in question: T4b's amd64 reading (0.303) agrees
      with the audit's 81%.
  - **That is the GGUF split. A bf16 directory is different (2026-10-10, the Mac, exploratory, two profiles of
    LFM2.5 2.6B to int4):** `QuantizeGroupInt4Row` is about 50–60% of the CPU and the bf16 widening
    (`Tensor.BFloat16sToF32`) 18–30%, with no dequantizer in the way. So T4b is worth more on arm64 for a directory
    source than the GGUF profile says.
    **(Go's profiler on the Mac: see the retraction above. The sampler's split for a bf16 directory is about 44% and
    43%, and T4b is worth more on a GGUF, where the quantizer is 83%.)**
- **F4. Safetensors experts are built one at a time.** `decoder/weights.go:loadFusedExperts`, `streamExperts`
  (gemma4), and the per-expert `loadMatQ` loop in the generic builder's layer loader all loop over experts
  sequentially. `buildGptOssWeights` does too, but gpt-oss directories still take the resident (layer-parallel) path.
  The GGUF side parallelizes experts when streaming (`stackedExperts`); the safetensors side never got that.
- **F5. `loadFusedExperts` spends half its time not quantizing.** One Qwen3-VL-30B-A3B layer (128 experts, the
  transposed transformers-4.57 layout), single core: 22.1 s. Quantize 52%, transpose 29%, bf16-to-f32 conversion
  10%, copies 8%.
  - The plain transpose `loadFusedExperts` used then (`transposeF32`, removed by T5 on 2026-10-10) writes with an
    8 KB stride, so every column of a tile lands in the same L1 set.
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
    **(These seconds are from Go's profiler on the Mac, withdrawn under F3, and were not re-measured. T6 shipped on
    identity, so nothing rests on them.)**
- **F7. The writer round-trips scales through f32.** `giwWriter.weightMatKind` widens the stored binary16 scales to
  f32 (`Int4F32`, an allocation), `giwWriter.int4Scales` narrows them back (`F32ToF16Scales`, another), and
  `giwWriter.scales16` copies them to bytes (a third). About 2% of the profile; fold it into V1's writer change.
- **F8. NaN rounds differently per arch in the int4 quantizer today.** `int(math.Round(NaN))` is the int64 minimum on
  amd64 (`CVTTSD2SQ`, clamped to −7) and 0 on arm64 (`FCVTZS`). The int8 path documented why both arches give 0; int4
  never pinned it. A SIMD version has to match each arch's existing output, so the oracle test must include NaN.
- **F9. The self-check re-reads the whole file.** That matters only when the file is larger than free RAM. For a file
  that fits, the read comes from page cache and compares the bytes Go wrote with the CRC Go computed over those same
  bytes. That checks the writer against itself, not the disk.
  - **T0, 2026-10-10 (the Mac, by day):** the phase line reads anywhere from 0.2 to 5.8 s of self-check on a 1.0–1.3 GB
    int4 sidecar, over seventeen transcodes, where the cloud box read 0.13 s. It has no stable relation to the target or
    to the logit tables' precision (a first reading, "slow with int8 tables", did not survive the next six runs).
  - (The "0.5–0.6 s of CPU" below is from Go's profiler on the Mac, withdrawn under F3. The reading it supports
    rests on the next line's measurement instead: the same load in a fresh process takes 0.2 s of wall.)
  - The same load in a fresh process takes 0.2 s, and the self-check's CPU is 0.5–0.6 s. So the time is not the
    check's own work. It grows with the file (0.1 s under 610 MB, 9 s at 1.7 GB), and it reproduces with no goinfer
    code: 1.7 GB written over 5 s and then read through a mapping took 8.2 s the first time and 0.3 s the second. It is
    the system's handling of just-written pages; what exactly it waits on is not established
    ([record](../measurements/giw-v16-transcode-2026-10/README.md)).
  - **After T1 and T2 it is often the larger half of a Mac transcode** (exploratory: stream 2.4–6.3 s, self-check
    0.2–9.4 s on the 1.5B). Whether to keep reading the whole file back straight after writing it is a decision this
    doc does not have yet.
  - **Decided and built 2026-10-10 (Decision 5): the transcode no longer reads the file back.** It writes the
    verified-marker for the bundle it has just written (`decoder.MarkGIWWritten`), so the self-check still loads the
    bundle through the mmap path (it deserializes, and its shapes validate against the architecture) and skips the CRC
    pass. The marker follows the bundle to its final name, so the first real load skips the pass too.
    - Kept: a `.giw` with no marker (a cache found on disk, a sidecar copied from another box) is CRC'd on its first
      load; `GOINFER_GIW_VERIFY=always` forces the pass on every load.
    - Given up: after a transcode nothing compares the bytes on disk with the CRC unless that variable is set. For a
      sidecar larger than free RAM the old read-back did read the disk; that check is gone with it.
    - Exploratory, the Mac, by day (G-T1's phase lines, 48 transcodes): 0.0–0.3 s of self-check on every file up to
      1.8 GB. LFM2.5 2.6B still reads 2.7–3.9 s: its load copies large f32 tensors out of the mapping
      (`giwReader.f32`), which is the load's own read and not a CRC pass (a profile shows no CRC under the self-check).
      **(That attribution is Go's profiler on the Mac, withdrawn under F3: where LFM2.5's self-check spends its time
      is not established. That it runs no CRC pass is established by the marker and its tests, not by the profile.)**
      That copy is V's to remove.
  - **Amended 2026-10-10, the same day (Decision 5, second part; `06f82beb`).** The owner asked whether another thread
    could compare the file with its CRC as the bytes are written, and chose both of the following.
    - **A read-back behind the write** (`internal/prequant/readback.go:readBack`): a goroutine reads the temp file
      behind the writer, keeps a CRC-32 of what it reads, and compares it with the weights' trailing CRC word when
      the blob ends. A mismatch fails the transcode before the self-check.
    - **A size rule** (`markWritten`): the transcode writes the marker itself only when the read-back passed and the
      bundle is no larger than the memory available before the write. A larger bundle, or an unknown reading, gets no
      marker, so the self-check's load runs the whole-file CRC as it did before, and for such a file part of that
      read comes from the disk. That restores what "Given up" above names.
    - **What the read-back checks, measured before building it** (a 1.7 GB scratch file written over 6 s, two rounds,
      exploratory): it adds no wall time (5.8–6.0 s with it, 6.3–6.4 s without) and stays level with the writer. It
      reads the system's copy of bytes just written, so it catches a byte that reached the file different from the
      one the writer summed, or a file changed during the write; it does not show what the disk stored. An attempt
      at a true disk read (full fsync, then a read with the cache-bypass flag) returned 1.7 GB in 0.2–0.3 s, 6–8 GB/s,
      which is not established as a disk read: pages already cached may still have served it. For a bundle that fits
      in memory the old read-back did not reach the disk either.
    - Exploratory, the Mac, by day (G-T1's phase lines, 16 transcodes, beside `gate quick`): the stream phase reads
      0.4–1.6 s for the 0.5B and 1.2–5.7 s for the 1.5B with the read-back on, where the run before it read 0.4–1.6 s
      and 1.0–5.3 s. The self-check stays at 0.0–0.1 s.
  - **Open, from the T runs of 2026-10-10: the size rule does not catch the case it was written for.** nobara's 30B
    (17.5 GB out of a 58 GB directory, on 62 GB of RAM) was marked, because Linux counts reclaimable cache as available
    and the bundle "fit". It was not in memory: the old arm's CRC read it back at 1.6 GB/s, a disk read. So its
    whole-file CRC was skipped where the old read-back did reach the disk. Its self-check also got slower (19-25 s
    against 10-12 s): no CRC runs, and the load faults the file back in piecemeal. Candidates: put the source in the
    sum (bundle plus source against total memory), or ask the system what is cached. Not built; the owner's to choose.
  - **Decided and built 2026-10-10 (owner: "yes" to comparing bundle plus source against total memory).**
    `markWritten` now marks a bundle only when it and its source together are no larger than the host's installed
    memory (`decoder.HostRAMBytes`); the source is the GGUF, or the directory's `.safetensors` files. An unknown
    memory size or an unreadable source leaves it unmarked.
    - What it changes on the cells of the T runs: nobara's 30B (17.5 GB plus 58 GB against 62 GB) is unmarked, so its
      self-check runs the CRC again, as the old arm's did. Every other cell stays marked (the largest is the Mac's 7B,
      5.6 GB plus 4.7 GB against 16 GB).
    - It is a rule about installed memory, not free memory: a box whose memory is mostly taken by other work can
      still evict a bundle the rule marks. The 30B's new self-check time has not been re-measured.
    - **Re-measured the same evening (the T runs with T4b):** the 30B is unmarked and its self-check runs the CRC
      again, at 20.8-28.0 s, about what it took before the fix without one. The old arm's, with the same CRC, reads
      9.5-13.3 s. The check is restored; the time did not move, and why the new arm's is twice the old arm's is not
      established.

### Checked and not worth doing

- **A buffered sink.** The transcode makes 3,218 `Write` calls for the whole 1.5B. The small header writes are real
  but negligible. A large MoE makes more (about ten per matrix), still under a second.
- **The self-check for a file that fits in RAM:** 0.13 s of 19.3 s. **Not so on the Mac (T0, 2026-10-10): up to 5.8 s
  of a transcode that takes 5–20 s, and as little as 0.2 s. See F9; it is an open question, not a closed one.** A sidecar larger than RAM (the 16.7 GB 30B on the
  16 GB Mac) re-reads from disk. T0's phase line measures that before anything changes (F9). **Closed 2026-10-10 by
  Decision 5: the read-back is gone (F9).**
- **The disk write:** about 0.5 s per GB on the cloud box. Not representative of the Mac's SSD, but not where the time
  is either.
- **GGUF dequant (Q4_K/Q6_K, scalar):** 12.7% today. It becomes a larger share once T4 lands. Re-rank it from T0's
  profile after T4, not before. **(WITHDRAWN 2026-10-10 under F3: on the Mac it is about 7% before T4b and about 30%
  after it, by the system sampler.)** **On arm64 it is already about 67% (T0, 2026-10-10; F3): on the Mac it ranks first
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
  - **Built 2026-10-10 (`d8e91fe3`): `decoder/stream_layers.go:streamLayers`.** What the tree made different from the
    text above:
    - The window divides the budget by the largest layer written so far, not by layer 0's bytes: a hybrid's layers
      differ in size, and layer 0 need not be the large one.
    - On a host with no available-memory probe (`HostRAMAvailableBytes` reads 0) the budget is 1 GB, not a quarter of
      nothing.
    - A writer with no budget set builds one layer at a time, so only the two transcode entry points window.
  - **G-T1 for T1: PASS on the Mac, 2026-10-10** (330 tiny cells and 3 real cells IDENTICAL, 0 DIFFERENT; no data race
    under `go test -race`).
- **T2. Row-parallel logit tables.** The embedding and an untied LM head are single tensors, so T1 cannot reach them.
  - Give `streamQuantizedRepackable` a parallel form over contiguous row ranges that takes a row-reader *factory*, so
    each worker owns its reader's state.
  - The GGUF `embMat` reader is a pure function of the mapping and can be shared. The safetensors streaming reader in
    `decoder/weights.go:loadEmbedTable` keeps a 1,024-row block cache, so it needs one per worker, with ranges aligned
    to its block.
  - Do not parallelize `rowInto` generally: a stateful reader is exactly the case that breaks.
  - **Built 2026-10-10 (`ec5958fa`): `decoder/weightmat.go:streamQuantizedEmbed` over `forRowRanges`.** What the tree
    made different from the text above:
    - The parallel form is `streamQuantizedEmbed` itself, not `streamQuantizedRepackable`: an int8-pinned table never
      went through the latter (it fell back to `streamQuantized`), and int8 is the default pin.
    - It also covers Gemma 4's per-layer token table, a third logit table with a stateless reader.
    - A table under 4M elements stays on one goroutine. So no tiny fixture reaches the parallel form on its own; tests
      force it, and G-T1's real GGUF cells carry it.
  - **G-T1 for T2: PASS on the Mac, 2026-10-10** (330 tiny cells and 8 real cells IDENTICAL, 0 DIFFERENT; no data race
    under `go test -race`).
- **T3. Parallel experts on the safetensors side.**
  - `loadFusedExperts`, `streamExperts` and the generic per-expert loop get what `stackedExperts` has: `parallelLayers`
    over experts when building for a sink, sequential otherwise. In the resident build they already sit inside
    parallel layers, so nesting would multiply transients.
  - Each expert writes only its own slot; the source reads are pure.
  - **Built 2026-10-10 (`bbb3bbeb`): `decoder/stream_layers.go:forExperts`.** What the tree made different from the
    text above:
    - There are five safetensors expert loops, not three. Nemotron's per-expert loop and Llama 4's (a whole-tensor
      read, then a transpose per expert) have the same shape and take the same helper.
    - The GGUF `stackedExperts` carried its own copy of the rule; it calls the helper now.
  - **G-T1 for T3: PASS on the Mac, 2026-10-10** (330 tiny cells and 24 real cells IDENTICAL, 0 DIFFERENT; no data
    race under `go test -race`). No real MoE directory is on the Mac: the real MoE cell is nobara's Qwen3-VL-30B at
    night, as registered. By day the MoE paths are covered by the tiny fixtures and by
    `TestStreamTranscode_expertFanOutWritesTheSameBytes`, one source per loop.
- **T4. A fast int4 quantizer (aikit).**
  - **T4a, scalar, now.** Replace `int(math.Round(float64(x)))` with
    `f := float64(x); if f < 0 { int(f - 0.5) } else { int(f + 0.5) }`. For any float32 x the widening is exact, and
    so is the ±0.5 for 2⁻²⁹ ≤ |x| < 2⁵¹. Below that the sum stays inside (0, 1), so truncation still gives 0. Inside
    the quantizer |x| ≤ 7.000002 whenever it is finite. NaN and ±Inf convert exactly as before.
    - Checked exhaustively over every float32 with |x| ≤ 2²⁴ (1,266,679,809 bit patterns, both signs) plus ±Inf and
      NaN: zero mismatches.
    - Exploratory: −17% on the amd64 transcode, identical sha256. Expect less on arm64 (F3).
    - **Measured on arm64, 2026-10-10 (the Mac, an in-process A/B interleaved in one process, 31 rounds of 9.2M
      elements, normal weights; a kernel microbenchmark, so direction only): not adopted there.** The form above takes
      1.78× the time of `math.Round` (range 1.77–1.80, slower in 31 of 31): `math.Round` is one `FRINTA` on arm64, and
      the sign test mispredicts on weights of random sign. A branch-free form, `int(f + math.Copysign(0.5, f))`, is a
      wash on arm64 (1.007×, faster in 2 of 31). So arm64 keeps `math.Round`, and T4a is an amd64 change.
    - All three forms agree on every one of the 2³² float32 bit patterns on arm64, NaN and ±Inf included (NaN gives 0
      there, +Inf the int64 maximum, −Inf the minimum, each in all three).
    - **Owed: the same A/B and the same exhaustive check on amd64**, with the branch-free form as a third arm, before
      anything ships. nobara was holding a timed run when this was written.
    - **Queued 2026-10-10 on nobara's queue** (`t4a-round-ab`, 5 min; a self-contained module, no aikit checkout): its
      log is `~/goinfer-logs/t4a-ab-<stamp>.log` on nobara. If amd64 resolves in favour of a new form, the change is
      amd64-only (an arch-split helper) and its aikit release needs perfgate, a timed run, so it is its own patch
      release at night and not part of v1.65.0.
    - **Measured on amd64, 2026-10-10, 11:34 (nobara, Ryzen 7 3700X; the owner freed the box and said to run now, so
      the job was run directly under the timing lock and dropped from the queue; the same harness, two passes):**
      - **The form above is slower on amd64 too:** 1.158× the time of `math.Round` in both passes, faster in 0 of 31.
        The audit's exploratory −17% does not reproduce on weights of random sign.
      - **The branch-free form wins:** `int(f + math.Copysign(0.5, f))` takes 0.698× in both passes (0.681–0.715),
        faster in 31 of 31. A resolved direction, so it is acted on by day (TE5(b)); the size is the night gate's.
      - All three forms agree on all 2³² float32 bit patterns on amd64 (NaN and both infinities give the int64 minimum
        there, in all three, which the quantizer clamps to −7 as before).
    - **T4a as built (aikit `395f5ba`, in v1.65.0):** `linalg.roundInt4`, the `Copysign` form on amd64 and
      `math.Round` everywhere else. No output changes on any architecture. aikit's oracle tests
      (`TestRoundInt4_matchesMathRound`, `TestQuantizeGroupInt4Row_matchesReference`) pass with their full sweeps on
      amd64 (natively on nobara, and under Rosetta on the Mac) and on arm64. It goes into v1.65.0 with perfgate run
      the same day, not into a later patch release.
    - **aikit v1.65.0 tagged 2026-10-10.** perfgate on nobara: PASS against v1.64.0 (15 of 47 shapes resolve the 5%
      class; none of its shapes is a benchmark of the quantizer, so it says the release disturbed nothing else).
      Vulnerability scan: 16 of 16 modules clean. goinfer pins it from `05e8aeeb`.
    - **G-T1 for T4a:** on the Mac the gate ran on the new pin and passed (330 tiny and 8 real cells), but arm64 keeps
      `math.Round`, so that run does not exercise the change. The amd64 cells are the night's, on nobara, as
      registered.
  - **T4b, SIMD.** `QuantizeGroupInt4Row` gets the int8 treatment.
    - Arch dispatch, with the current scalar body kept verbatim as `quantizeGroupInt4RowScalar`, the oracle.
    - Per 32-element group: SIMD max-abs (exact and order-independent, already argued for int8), the scalar scale and
      inverse, a SIMD multiply with no FMA, then round-half-away, clamp to ±7 and pack nibbles.
    - NEON has `FCVTAS` for the rounding. AVX2 has no ties-away mode: truncate (`VROUNDPS` toward zero), compute the
      exact remainder, and step away from zero where |remainder| ≥ 0.5.
    - The ceiling to aim at: int8int8's whole transcode is 3.9 s against int4's 19.5 s on the same model.
    - `QuantizeGroupsInt4` and `QuantizeInt4` (the resident and safetensors paths) inherit it.
    - **Built 2026-10-10 (aikit `244a10e`, for v1.66.0).** As designed, with these differences:
      - The dispatch is per group, not per row: a group that starts on a byte boundary and is a whole number of
        kernel blocks (16 weights on arm64, 8 on amd64) takes the kernel, and the max is the activation quantizer's
        existing `maxAbsF32`. The oracle is `quantizeGroupInt4RowScalar`.
      - AVX2 rounds as the activation quantizer already does, `trunc(y + copysign(h, y))` with
        `h = nextafter32(0.5, 0)`, not by truncating and stepping on the remainder. It also sets a finite positive
        product in [2^31, 2^63) to 7, where the int32 conversion and the scalar's int64 one disagree, so the kernel
        equals the scalar on every float32 and not only on products a real group can have.
      - G-T4: the kernel sweep over every float32 in every lane, and the row test against both scalar references,
        pass natively on arm64 and on amd64. F8's NaN cases are in both.
      - **A trap on amd64.** The first AVX2 kernel built its constants through a general register, which Go assembles
        as legacy-SSE moves; with the YMM upper halves in use each costs a state transition, and at one call per
        group the kernel was correct and 1.77x slower than the scalar on nobara's Ryzen (Rosetta showed it faster).
        The constants are read from memory now.
      - Kernel microbenchmark, in one process, interleaved (direction): 0.094 of the scalar's time on the Mac, 0.097
        on nobara.
    - **Owed:** aikit v1.66.0's tag (its perfgate and vulnerability scan are on nobara's night queue as
      `aikit-v1660-gates`), then goinfer's pin and G-T1 through `gate identity`, and the next night's cells.
    - **Released and pinned 2026-10-10** (owner: "run it now"): perfgate PASS against v1.65.0 and the scan clean on
      nobara by day, aikit v1.66.0 tagged, goinfer on it from `c3e921e6`. **G-T1 for T4b: PASS on the Mac** (330 tiny
      and 24 real cells IDENTICAL, 0 DIFFERENT). The night's cells with T4b in the new arm are queued on both boxes.
- **T5. `loadFusedExperts` without the waste.**
  - A tiled transpose that converts from bf16 on the way. It needs a small aikit API, because the raw bytes are
    unexported, for example `Tensor.SubF32Transposed(start, rows, cols)`. 16×16 tiles through a local buffer, written
    out as contiguous runs.
  - Drop the three `append` copies except where the slice aliases the mapping and the mode keeps f32 (F5).
  - **Built 2026-10-10, not yet on this branch.** aikit: `embed.Tensor.SubF32Transposed(start, rows, cols)` (aikit
    `d7f903e`, in v1.65.0's release prep `babd539`; the tag waits for the vulnerability scan, queued on nobara as
    `aikit-v1650-vulncheck`). goinfer: `loadFusedExperts` on branch `giw-v16-t5` (worktree `../goinfer-t5`), which
    builds only against the local aikit checkout until the pin is bumped.
    - The copy rule as built: a matrix that stays f32 after `quantizeWM` and was read in place from an F32 tensor gets
      a slice of its own. Every other read is already a fresh slice.
    - `transposeF32` had no other caller and is removed.
    - Exploratory, the Mac, in-process and interleaved (direction only): reading one [2048 × 1536] slice through the
      tiled call takes 0.32× (bf16) and 0.26× (f32) of the time of `SubF32` plus `transposeF32`, faster in 21 of 21
      rounds each.
    - Interim identity, not the gate: the old build and the T5 build write the same bytes for `qwen3vlmoe-tiny` (the
      transposed layout) and `granite-tiny` (the row layout) in six configurations each, 12 of 12. Both fixtures are
      F32; the bf16 conversion is pinned in aikit (`TestTensor_SubF32Transposed`, bit-equal to `SubF32`) and reaches a
      real file only on nobara's Qwen3-VL-30B.
    - **Owed, in order:** tag aikit v1.65.0 and run `gpupins --fix`; bump goinfer's aikit pin in every module; put
      `giw-v16-t5` on this branch; refresh the parity hashes; `gate quick`; G-T1 for T5 through `gate identity`.
    - **Done 2026-10-10: on this branch as `a8263c3b`, on aikit v1.65.0.**
  - **G-T1 for T5: PASS on the Mac, 2026-10-10** (330 tiny cells and 8 real cells IDENTICAL, 0 DIFFERENT; the twelve
    cells of the two fused-expert fixtures among them). A bf16 fused stack has still not gone through it: that is
    nobara's Qwen3-VL-30B, at night.
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
- **T8. A vectorized GGUF dequantizer (aikit). Registered 2026-10-10, before any code** (owner: "do it", after the
  copy floor's consequence opened on the Mac's 1.5B at 3.08).
  - **What:** NEON and AVX2 kernels for the body of `embed.dequantQ4KBlock`: per 32-weight sub-block, the nibbles
    widened to float32, times the sub-block's scale, minus its offset. The scalar body stays as the oracle and serves
    other architectures. The block's prologue (the two half-floats, the 6-bit scale and min unpack, the eight products)
    stays in Go.
  - **Each architecture matches its own scalar.** `d1*float32(q) - off1` is one fused operation in Go's arm64 code and
    two rounded ones on amd64, so the two builds already differ in the last bit, and each kernel must reproduce its
    own: fused on NEON, a multiply and a subtract on AVX2. The oracle test decides, not this note.
  - **Reach.** This function is every Q4_K GGUF load, resident loads included, not only the transcode. So the forward
    goldens that load a GGUF are part of the gate.
  - **Second step, only if the first ships:** the same for `dequantQ6KBlock`, which the bench GGUFs use for a few
    tensors (about 5% of the Mac cell's CPU after T4b, by the system sampler).
  - **G-T8a, oracle (aikit, day).** The kernel equals the scalar bit for bit on: every nibble value under random
    scales and offsets; d and dmin that are zero, negative, subnormal, infinite and NaN; random blocks; and the whole
    of the real 1.5B GGUF's Q4_K tensors (a hash of the dequantized floats). Natively on arm64 and on amd64. Planted
    kernel defects must turn it red.
  - **G-T8b, identity (goinfer, day).** `gate identity -giw bytes` on the tiny matrix and the two real GGUFs, and
    `gate quick` (its GGUF forward goldens) on the pin.
  - **G-T8c, speed (day; `t4b_cpu_ab.py`, the instrument T4b's gate used).** Single-thread transcode CPU, new over
    old, old being the v1.66.0 build, on the Mac's 1.5B GGUF to `cpu-arm64` and nobara's 1.5B GGUF to `cpu-amd64`.

    | | expected | ship | park | off |
    |---|---|---|---|---|
    | each cell | 0.75-0.85 | at most 0.98 | 0.98-1.00 | above 1.00 |

    The expectation: the Q4_K dequantizer is about 24% of the Mac cell's CPU by the sampler, and a kernel four to
    six times faster leaves about 5%. The ship line is the repo's default (1.02x), not the expectation.
  - **Recorded, not gated:** the copy floor on the next night's cells (3.08 on the Mac's 1.5B today).
  - **Amendment, 2026-10-10 (after the Q4_K kernel was measured, before any Q6_K code).**
    - **The Q4_K kernel: G-T8a PASS, G-T8c SHIP on both cells** (new/old CPU 0.847 on the Mac, 0.809 on nobara;
      [record](../measurements/giw-v16-transcode-2026-10/README.md), "T8's day gate"). G-T8b waits for the pin.
    - **The note above on the two architectures is wrong for every finite weight.** `d*scale*nibble` is exact in
      float32 (an 11-bit half-float, a 6-bit scale and a 4-bit nibble), and so is `dmin*min`, so the fused subtract
      and the two-step one round once, to the same value. Measured: the crc32 of every Q4_K weight of the 1.5B GGUF is
      the same on arm64 and on amd64. The builds can differ only in which NaN comes out when d or dmin is not finite,
      and there each kernel reproduces its own scalar, which the oracle checks.
    - **The amd64 kernel takes the CPU check** (AVX2 present, or the scalar body). The first nobara reading (0.816)
      was on a build without it; the graded one (0.809) is on the build with it.
    - **Q6_K's gate, the second step.** The instrument and the two cells are G-T8c's. **Old is the build with the Q4_K
      kernel; new adds the Q6_K kernel.** Ship at most 0.98, park 0.98-1.00, off above 1.00. Expected 0.93-0.97: Q6_K
      was about 5% of the Mac cell's CPU before the Q4_K kernel, so about 6% of this old arm, and a kernel four to
      six times faster leaves a little over 1%. The expectation sits close to the ship line, so a park is a live
      outcome, and a parked Q6_K kernel is not released. Its oracle is G-T8a's with Q6_K's fields: random blocks, every
      half-float as d, every scale byte, every `ql` and `qh` byte value in every lane, the real 1.5B GGUF's Q6_K
      tensors, and planted defects. If it ships, both kernels go in one aikit release and G-T8b runs once on that pin.
    - **Q6_K graded, 2026-10-10: SHIP on both cells** (0.931 on the Mac, 0.864 on nobara, the record's "second
      step"). nobara is below the expected range; the expectation was built from the Mac's share. Both kernels go
      in aikit v1.67.0.
    - **G-T8b, 2026-10-10: PASS.** aikit v1.67.0 is tagged and pinned. `gate identity -giw bytes`: 330 of 330 tiny
      cells that ran and 8 of 8 on the two real GGUFs are IDENTICAL; `gate quick` is GREEN on the pin. **T8 is built
      and graded.** The copy floor's re-read on a later night's cells is still owed (recorded, not gated).
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
  - **Amended 2026-10-10, before any G-T2 run (Decision 5 changed what the new arm does; no band moves).** The graded
    quantity stays the full wall, as registered. The new arm no longer reads the file back, so on the Mac part of its
    gain is that removal and not the parallelism the bands were drawn from. So the record also gives the stream phase's
    paired ratio beside the graded one, ungraded, and for that the old arm is `cf66f5b3` (T0: the phase line and none
    of T1–T6), not `eee75d83`, which prints no phases. A full-wall ship with a stream-phase ratio in the kill band is
    reported as such and goes back to T0's profile.
  - Bands are from § Audit: the single-core split, 8 cores at 0.6–0.85 efficiency, and T4a's 0.83× on amd64 (taken
    as 1.0 on arm64).

  | cell | expected | ship | park | kill |
  |---|---|---|---|---|
  | dense GGUF int4 (1.5B Mac cpu-arm64; 1.5B and 7B nobara cpu-amd64), T1 + T2 + T4a + T6 | 0.18–0.33 | ≤ 0.45 | 0.45–0.70 | > 0.70 |
  | Qwen3-VL-30B-A3B int4 dir (nobara), T3 + T4a + T5 | 0.18–0.25 | ≤ 0.40 | 0.40–0.65 | > 0.65 |
  | T4b alone, single-thread transcode CPU time against T4a (in-process, day tier, TE5(b)) | 0.4–0.6 | ≤ 0.70 | 0.70–0.90 | > 0.90 |

  A kill means the parallelism is not happening, or the kernel is no faster. It goes back to T0's profile, not to a
  wider band. T6 ships on G-T1 alone, because it only removes work; its ratio is recorded.

  **Amended 2026-10-10, before any T4b reading (the row named no cell and no instrument that exists; no band moves).**
  - **Cells.** The band (0.4-0.6 expected, ship at most 0.70) was drawn from the amd64 audit, where the quantizer is
    about 80% of a transcode's CPU. It can only be met where the quantizer leads, so the row is read on two cells:
    **nobara, the 1.5B Q4_K_M GGUF to int4 `cpu-amd64`** (the audit's shape), and **the Mac, the Qwen3 0.6B bf16
    directory to int4 `cpu-arm64`** (F3: on arm64 the quantizer leads for a bf16 directory, not for a GGUF). Recorded
    beside them, ungraded: the Mac's 1.5B GGUF, where the quantizer is about 14% of the CPU and the ratio cannot go
    under about 0.86 whatever the kernel does, and nobara's 0.6B directory. **(That ceiling was wrong, and its reading
    is what exposed it: F3's retraction.)**
  - **Instrument.** Not in one process: the kernel lives in aikit, so the two arms are two builds of `cmd/prequant`
    from one goinfer revision, old on aikit v1.65.0 (T4a) and new on the T4b commit. Each transcode runs with
    `GOMAXPROCS=1`; the quantity is its user plus system CPU time from `time`. One discarded warm-up per arm, then five
    pairs in alternating order; the ratio is paired, new over old. Both arms must write the same sha256
    (`docs/measurements/giw-v16-transcode-2026-10/t4b_cpu_ab.py`).
  - **Tier.** Day, as registered: CPU time on one thread, a few minutes a cell. The full-wall effect is the night's.

  **T4b's row read 2026-10-10: SHIP on both graded cells.** Paired new/old single-thread CPU time, five pairs each:
  nobara's 1.5B GGUF 0.303 (0.303-0.308; 12.1 s to 3.7 s); the Mac's Qwen3 0.6B bf16 directory 0.407 (0.398-0.414;
  4.2 s to 1.7 s). Recorded: the Mac's 1.5B GGUF 0.304 (10.5 s to 3.2 s), nobara's 0.6B directory 0.402. Every cell's
  two arms wrote the same sha256. The new arm was built on aikit `244a10e`, ahead of its tag.

  **Read again 2026-10-10 with T4b in the new arm: SHIP on all four cells.** Medians: Mac 1.5B `cpu-arm64` 0.128
  (was 0.278); nobara 1.5B 0.065 (0.131) and 7B 0.051 (0.117); the 30B directory 0.181 (0.178, unchanged: it is
  within a factor of two of its own I/O). G-T1's real cells identical again, all seven.

  **Read 2026-10-10: SHIP on all three graded cells.** Paired new/old wall, medians: Mac 1.5B `cpu-arm64` 0.278
  (three pairs, 0.278-0.279); nobara 1.5B `cpu-amd64` 0.131 and 7B `cpu-amd64` 0.117 (three pairs each); the 30B
  directory 0.178 (two pairs, 0.189 and 0.168). The stream phase's ratios are beside them in the record and none is in
  a kill band. G-T1's real cells: every transcode of all seven cells identical across the arms. The T4b row is not
  read yet.
- **G-T2c, the copy floor (the same night runs; recorded, not gated).** Added 2026-10-10 at the owner's request, before
  any G-T2 run and before any T1–T5 code. It answers "how close to a file copy is the transcode", which new/old cannot.
  - **Two copies per cell, timed inside the same ABAB sequence** (old, new, copy; then new, old, copy), in the directory
    the sidecar is written to:
    - **C-src, a real copy of the source:** `dd if=<source> of=<dir>/copyfloor.tmp bs=8m` (for a directory source,
      each shard in turn, summed), then delete it.
    - **C-io, the transcode's own I/O with no compute:** read the source to `/dev/null`, then `dd` the old arm's
      output `.giw` to a temp file. These are the bytes a perfect transcode reads and writes.
    - No fsync in either, matching the transcode, which closes its file without one. The page cache is warm exactly as
      for the transcode arms.
  - **Never `cp`.** On the Mac's APFS, `cp -c` and a Finder duplicate clone the file. On nobara, Fedora's default btrfs
    plus coreutils' `cp --reflink=auto` default does the same. A clone finishes in milliseconds and measures nothing.
  - **Disk space first.** C-src needs the source's size free (about 62 GB for the 30B). If free space is under the
    source plus 10%, skip C-src for that cell and record why. C-io needs only the output's size.
  - **Reported:** paired ratios old/C-src, new/C-src, old/C-io and new/C-io per cell.
  - **Projections, written before the runs.** These use spec-sheet disk speeds: about 4–5 GB/s on the Mac's internal
    SSD, an assumed 2–3.5 GB/s for nobara's NVMe (model not recorded). C-io in seconds, then transcode/C-io:

    | cell | C-io | today | after T | after T + T4b |
    |---|---|---|---|---|
    | 1.5B int4, Mac cpu-arm64 | 0.5–1 s | 20–40× | 4–12× | 2–6× |
    | 7B int4, nobara cpu-amd64 | 3–6 s | 15–30× | 3–10× | 2–6× |
    | Qwen3-VL-30B-A3B int4 dir, nobara | 20–40 s | 17–33× | 3–8× | 2–5× |

    The 30B writes 16.7 GB from a 62 GB source, so after T4b it may finish faster than C-src, its source's own copy
    (about 40–70 s). Record whether it does.
  - **Consequence (registered).** If new/C-io after T4b stays above 3× on either dense cell, open a follow-up for the two
    levers this doc leaves out, re-ranked from T0's profile at that point:
    - vectorized Q4_K/Q6_K dequantization in aikit;
    - writing one window of layers while the next is built.

    At 3× or below, the transcode is near its disk floor and no follow-up opens.
  - **Recorded 2026-10-10 after T4b, and the consequence read.** new/C-io: 3.08 on the Mac's 1.5B, 2.72 on nobara's
    7B, 1.83 on the 30B directory, each inside or better than its "after T + T4b" range. **The consequence opens, on
    the Mac's cell only and by a hair** (its three pairs read 3.09, 3.19 and 2.54; the copy varied on a box in use).
    The follow-up is therefore open and not started: its first lever, a vectorized Q4_K/Q6_K dequantizer in aikit, is
    about 24% of that cell's CPU by the system sampler, so it bounds the gain at about a quarter; the second lever is
    writing one window while the next is built. Whether it is worth doing at 3.08 is the owner's call.
  - **Recorded 2026-10-10 (after T, before T4b).** new/C-io: 8.0 on the Mac's 1.5B, 6.2 on nobara's 7B, 2.0 on the 30B
    directory; old/C-io 28.8, 53.1 and 11.2. The two dense cells are inside their "after T" ranges; the 30B is better
    than its range because its copy is slower than projected (49-69 s). The consequence above is read after T4b.
- **G-T3, memory (the same night runs).** Peak footprint (`/usr/bin/time -l` on the Mac, `-v` on Linux).
  - Dense: at most the old peak plus 1.5× K layers' bytes, and never above Decision 3's budget.
  - MoE: at most the old peak plus workers × 3 × one expert's f32.
  - Mac: swap flat across every run (`vm.swapusage` before and after). The never-swap rule is not traded for speed.
  - **Read again 2026-10-10 with the available memory recorded: within the bounds, the budget clause included.** The
    largest dense increases are 1.56 GB on the Mac's 7B (a quarter of the 9.8-10.2 GB available is 2.5 GB) and
    2.26 GB on nobara's 7B (a quarter is 15.9 GB). Mac swap flat at 1,059.0 MB over 22 transcodes.
  - **Read 2026-10-10: within the bounds.** Dense increases: 0.05-0.08 GB (1.5B), 1.56 GB (the Mac's 7B, footprint),
    2.07-2.18 GB (nobara's 7B, maximum RSS). The 30B's new peak is under its old one (maximum RSS, which is mostly the
    mapped source). Mac swap flat at 1,021.5 MB over 22 transcodes. Not shown by the record: the available memory each
    transcode started with, so the budget clause is checked only loosely; the driver should record it.
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

- **T, nobara, about 50 min** (G-T2c's copies add about 5, almost all of it the 30B's):
  - 1.5B and 7B int4 cpu-amd64, three ABAB pairs each: old about 20 s and 90 s, new about 5 s and 20 s, so about 7
    minutes.
  - 7B cuda target, two pairs: about 5 minutes.
  - Qwen3-VL-30B-A3B int4 directory, one warm-up per arm plus two pairs: old 11 minutes, new about 2, so about 30
    minutes.
  - G-T1's real cells and G-T2c's copies ride the same runs (the sha256 of each written file; the two `dd` copies).
- **T, Mac, about 13 min** (G-T2c included): 1.5B int4 cpu-arm64 and Metal targets, three pairs each (about 4 minutes); 7B int4 Metal,
  two pairs (about 8 minutes).
- **V, about 15 min:** G-V1 and G-V3 on the Mac's 1.5B and 7B Metal sidecars (about 8 minutes); G-V1 on nobara's 7B
  cuda and cpu-amd64 sidecars (about 5 minutes); G-V2's real sizes ride the same runs.
- Shape: `docs/measurements/giw-v16-transcode-2026-10/run-*.sh` with pre-built old and new binaries, queued with
  `scripts/night.py add`. Records are committed the next morning.

### As queued, 2026-10-10 (before any run; no band moves)

The T night's runs, as built: `docs/measurements/giw-v16-transcode-2026-10/gt_night.py`, driven by
`run-gt-night-mac.sh` and `run-gt-night-nobara.sh`, with binaries built on the Mac from pinned revisions
(`GOWORK=off -mod=readonly -trimpath`; nobara's are cross-compiled, because the branch is not pushed).

- **Arms.** New is `6398b0f8` (T0–T6 bar T4b, the read-back, aikit v1.65.0). Old is `cf66f5b3` for every figure, the
  graded wall included: it is `eee75d83`'s transcode with T0's phase line added and nothing else, so one old arm serves
  both the graded ratio and the stream-phase ratio the amendment above asks for.
- **What is timed.** The wall is the driver's clock around the whole process; the phases are prequant's own line.
  Peak memory is `time -l`'s peak memory footprint on the Mac and `time -v`'s maximum resident set size on nobara.
  Swap is `vm.swapusage` before and after each Mac transcode.
- **Order, one change from G-T2c's text.** The two copies are made straight after the old arm in every pair (pair 1:
  old, copies, new; pair 2: new, old, copies), not after both arms. The copies read the old arm's output, and the Mac
  has about 7 GB free: a 5 GB 7B sidecar cannot sit beside the new arm's and a third copy. Where even old plus one
  copy does not fit, C-io's write is skipped and the record says why, as the text already rules for C-src.
- **Where the files go.** Outputs are written to `~/goinfer-bench/giw-v16/out` (the same filesystem as `~/models` on
  both boxes), not beside the source, so the sidecars already there are not replaced. Each is hashed and deleted.
- **Cells.** Mac: 1.5B int4 `cpu-arm64` and `metal`, three pairs each; 7B int4 `metal`, two pairs. nobara: 1.5B and 7B
  int4 `cpu-amd64`, three pairs each; 7B int4 `cuda`, two pairs; Qwen3-VL-30B-A3B int4 `cpu-amd64` from its directory,
  two pairs. One discarded warm-up per arm in every cell. G-T1's real cells are every pair's sha256.
- **Estimates, revised from the text above:** Mac 20 min, nobara 85 min. The additions are the sha256 of every output
  (about a minute for each 17 GB file) and the 30B's three old-arm runs at about 11 minutes each.
- **Exit.** Non-zero if a transcode fails or a cell's two arms write different bytes. Nothing else is judged by the
  script; the bands above are applied by hand from the record the next morning.

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
available RAM at transcode start ("a quarter of ram is fine"). Decisions 2 and 4 are open (decided later the same
day, below).

**Decided 2026-10-10, after T2:** Decision 5, the transcode does not read its file back ("no self check makes sense to
me"). Built as: the load check stays, the CRC read-back goes (F9 has what is kept and what is given up).

**Decided 2026-10-10, after T4b's day gate:** Decision 2, the reader reads only v16 ("Only v16"). Decision 4, the
hand-built bundles are rebuilt by the session that lands V ("I rebuild them").
- What Decision 4 covers, surveyed the same day. Four bundles have no source beside them by name, and every one has
  a source elsewhere:

  | bundle | where | rebuilt from |
  |---|---|---|
  | `gemma4-26b-int4.giw` (16.1 GB) | nobara `~/models` | `gemma4-26b-q4_k_m.gguf`, nobara |
  | `gemma4-26b-int4-v14st.metal.giw` (16.2 GB) | nobara `~/models` and the archive | the same GGUF, `-target metal` |
  | `qwen3.6-35b-a3b-int4.giw` (20.1 GB; the archive holds `…-int4-v12.giw`) | nobara `~/models`, the archive | `qwen3.6-35b-a3b-q4_k_m.gguf`, nobara |
  | `mellum2.int4.giw` (7.8 GB) | nobara `~/models` | `mellum2-unq`, the archive (pulled to `~/models` first) |

  The flags each was built with are not recorded beside it; they are read from each old file's header before it is
  replaced. Every other `.giw` on either box is a sidecar and rebuilds by itself.

**Amended 2026-10-10, after T3:** Decision 5, second part. Asked "can we have another thread do this as bytes are
written", and offered the size rule alone, the size rule with a trailing reader, or the trailing reader alone, the
owner chose the second ("2"). F9 has what each part checks.

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
5. **The self-check after a transcode** (added 2026-10-10 from F9): keep reading the whole file back for its CRC, or
   stop. Decided: stop; then, the same day, read the weights back behind the write and keep the CRC pass for a bundle
   larger than available RAM; then, after the T runs, for a bundle that with its source is larger than installed RAM.

## Prior art read

- `docs/tasks/task-prequant-dir-streaming-2026-10.md`: G-DS2's "parallelism within a layer" follow-up, which T1–T3
  are; G-DS1's 126-configuration identity method, which G-T1 reuses.
- `docs/tasks/task-pull-to-giw-2026-09.md`: Decisions 2 and 4, and the sidecar census.
- `docs/completed/task-moe-streaming.md` and the `stackedExperts` comment: why expert parallelism is limited to the
  streaming path.
- aikit's `quantizeRowInt8Core` and `quantizeRowInt8CoreScalar`: the vectorize-with-an-oracle pattern T4b copies.
- `docs/measurements/sidecar-v15-2026-09-29.md` and `docs/giw-bundles.md`: what v12–v15 bought, which v16 keeps.

<!-- doc-reviewed: 2026-10-10 -->
