# `.giw` v16 and the transcode: gate records

The gates are registered in [`docs/tasks/task-giw-v16-transcode-2026-10.md`](../../tasks/task-giw-v16-transcode-2026-10.md).
The exploratory profiles that set its bands are in [`../transcode-audit-2026-10/`](../transcode-audit-2026-10/README.md).

## G-T1 for T6, 2026-10-10: PASS, every cell that ran is IDENTICAL

T6 is "no in-RAM repack while building for a sink". G-T1 is byte identity of the written `.giw`: the same sha256
from the old build and the new one, for every cell.

- **Instrument:** `go run ./cmd/gate identity -giw bytes <base> <new>`. Each revision builds its own `cmd/prequant` in
  a temporary worktree and transcodes the shared source. The new build transcodes every cell twice and must agree
  with itself before it is compared with the old one.
- **Revisions:** base `eee75d83` (the branch point, `origin/main`), new `5555d4a5` (T6).
- **Box:** the MacBook Pro (M1 Pro, arm64), macOS 26.6.2, Go 1.27.2, aikit v1.64.0, by day. An identity check, not a
  timing.

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 1 min 1 s |
| the real Qwen2.5-Coder 1.5B Q4_K_M GGUF, three configurations | 3 | 3 | 0 | 0 | 2 min 38 s |

### The tiny matrix

- **Sources:** every directory holding a `.safetensors` file directly under `testdata/` and `decoder/testdata/`: 64
  directories. 55 of them transcode (46 under `testdata/`, 9 under `decoder/testdata/`).
- **Configurations** (quant:target, `e4h` for `-embed-int4`): `int4:metal:e4h`, `int4:metal`, `int4:cpu`,
  `int8int8:cpu`, `int4:cpu-arm64`, and `int4:cpu-arm64:e4h`.
  - The first five are the registered matrix. On this box `cpu` resolves to `cpu-arm64`, so `int4:cpu` and
    `int4:cpu-arm64` are the same cell written twice.
  - `int4:cpu-arm64:e4h` is an addition: the int4 logit tables are one of the paths T6 changes, and no registered cell
    writes them for a CPU target.
- **Result:** 55 sources x 6 configurations = 330 cells, all IDENTICAL.
- **NOT RUN, 9 sources (54 cells), each for the reason both builds give:**

  | source | why |
  |---|---|
  | `decoder/mellum-mellum2-slice` | 4.3 GB: over the tiny matrix's 1 GB cap (a slice of a real checkpoint) |
  | `decoder/qwen3_5-tiny-lora` | a LoRA adapter directory, not a model: no `config.json` to read |
  | `embeddinggemma2-tiny`, `gemma4-audio-tiny`, `gemma4-vision-tiny`, `siglip-tiny`, `whisper-tiny-rand` | an encoder or a tower: `unsupported model_type` |
  | `glm-ocr-vision-tiny`, `qwen3vl-vision-tiny` | a vision tower with no text model in the directory (`missing required dim`; `no m-RoPE section`) |

- **Not in the matrix: the tiny GGUF fixtures.** `cmd/prequant` refuses a GGUF with no embedded tokenizer, and none of
  the tiny GGUFs has one. The GGUF streaming path is covered by the real 1.5B below, and by
  `TestStreamTranscode_buildsNoDecodeLayout`, which drives `StreamTranscodeGGUF` on `testdata/glm-tiny.gguf` directly.

### The real 1.5B

`~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1,117,320,768 bytes).

| configuration | verdict | sha256 (first 16) | bytes |
|---|---|---|---|
| `int4:cpu-arm64` | IDENTICAL | `655aa58553924d69` | 1,211,492,739 |
| `int4:metal` | IDENTICAL | `5b6c6956e9cbbcc0` | 1,375,265,859 |
| `int4:cpu-arm64:e4h` (an addition) | IDENTICAL | `832aff86af916316` | 1,006,075,267 |

### That the gate can fail

`TestIdentityGiw_headerChangeIsDifferentBytesAndIdenticalLogits` (opt-in, `GOINFER_HEAVY_TESTS=1`): a revision that
changes one string in the bundle header is DIFFERENT on every transcoded cell in `-giw bytes`, exit 1, and IDENTICAL in
`-giw logits`. `TestIdentityGiw_bytesCommentOnlyChangeIsIdentical` is the other direction. Both passed on this box
before the run.

### What T6 removes, as far as this run shows it

Not a speed gate: T6 ships on identity, and its ratio is for the night's G-T2. The new build's own phase lines, one
run each, by day, are in `g-t1-t6-real-lines.txt`; the old build prints none (it predates T0). The exploratory arm64
profile put the removed repack at about 2% of a 1.5B transcode's CPU. (That share is from Go's profiler on the Mac,
withdrawn 2026-10-10: see "T4b's day gate" below. It was not re-measured.)

### A side observation on the self-check (F9), not part of the gate

The new build's transcodes of the 1.5B, from their phase lines, two runs per configuration in each of the two G-T1
runs on this page (T6's, then T1's):

| configuration | self-check in T6's run | self-check in T1's run |
|---|---|---|
| `int4:cpu-arm64` | 4.4 s, 4.1 s | 5.1 s, 3.6 s |
| `int4:metal` | 4.7 s, 5.0 s | 5.8 s, 0.3 s |
| `int4:cpu-arm64:e4h` | 0.2 s, 0.2 s | 0.2 s, 2.4 s |

- Loading a kept `int4:cpu-arm64` file in a fresh process, the same call the self-check makes, took 0.23 s and 0.21 s.
  So seconds of self-check are not the cost of checking a file of this size.
- **A pattern that did not hold.** T6's six runs read as "slow with int8 logit tables, fast with int4 ones". T1's six
  runs break it both ways (0.3 s on a plain file, 2.4 s on the `e4h` one). Over the twelve runs the time ranges from 0.2
  to 5.8 s with no stable relation to the configuration.
- (The next figure is from Go's profiler on the Mac, withdrawn 2026-10-10. The fresh-process load above, 0.2 s of
  wall, is what the reading rests on.)
- The self-check's CPU is about half a second (the arm64 profile), so the rest is a wait. The box had 16 GB, the
  owner's session open and about 4 GB of disk free.
- **It grows with the file.** Adding T2's run (below): the 0.5B's files (344-609 MB) read 0.1 s in fifteen of sixteen
  runs, the 1.5B's 959 MB file 0.2-2.8 s, its 1,155 MB file 3.6-5.5 s, its 1,312 MB file 0.3-6.6 s, and its
  1,704 MB int8int8 file 8.9 and 9.4 s.
- **It reproduces with no goinfer code** (a script, the same box, 2026-10-10): 1.7 GB written to a file over about 5 s
  and then CRC'd through a read-only mapping took 8.2 s the first time and 0.3 s the second. 1.2 GB written the same
  way, fast or over 12 s, took 0.2 s. So the wait is the system's handling of pages it has just been given to write,
  above some size, and not the check's own work. What exactly it waits on is not established.

## G-T1 for T1, 2026-10-10: PASS, every cell that ran is IDENTICAL

T1 is the windowed layer parallelism in every streaming loop. Same instrument, box and matrix as T6's run above; base
`eee75d83`, new `d8e91fe3`.

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 58 s |
| the real Qwen2.5-Coder 1.5B Q4_K_M GGUF, three configurations | 3 | 3 | 0 | 0 | 1 min 52 s |

- The NOT RUN set is T6's, source for source.
- The three real files are the files T6's run wrote: the same sha256 and byte counts as the table above.
- **Also checked before the run:** `go test -race` over the streaming tests and the prequant byte-identity tests (135
  streamed configurations against their resident twins) reports no race, and `TestStreamLayers_everyWindowWritesTheSameBytes`
  streams with windows of 1, 2, 3 and 64 layers.
- **Phase lines, exploratory** (one run each, by day, the box in use; `g-t1-t1-real-lines.txt`): the new build's stream
  phase read 3.8-7.4 s over the six runs, where T6's build read 10.7-16.1 s for the two plain configurations and
  13.5-13.6 s for `e4h`. Not a result: G-T2 measures it at night, in ABAB pairs.

## G-T1 for T2, 2026-10-10: PASS, every cell that ran is IDENTICAL

T2 is the row-parallel logit tables. Same instrument and box; base `eee75d83`, new `ec5958fa` (T6, T1 and T2 together).

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 58 s |
| the real Qwen2.5-Coder 0.5B and 1.5B Q4_K_M GGUFs, four configurations each | 8 | 8 | 0 | 0 | 3 min 14 s |

- **Why the real cells carry this gate:** a table under 4M elements is quantized on one goroutine, and every tiny
  fixture's tables are under it. The real GGUFs' tables are 136M and 233M elements, int8-pinned in three configurations
  and int4 in `e4h`.
- **The paths the real GGUFs do not reach** are the safetensors table reader (one block cache per row range) and
  Gemma 4's per-layer token table. `TestStreamQuantizedEmbed_rangesGiveTheSequentialTable` and
  `TestStreamTranscode_rowRangesWriteTheSameBytes` force several row ranges on small tables for those, in every
  embedding precision, and `go test -race` over them and over a real 0.5B GGUF transcode reports no race.

| source | configuration | verdict | sha256 (first 16) | bytes |
|---|---|---|---|---|
| `coder-0.5b-q4km` | `int4:cpu-arm64` | IDENTICAL | `85de0d6013819797` | 481,009,219 |
| `coder-0.5b-q4km` | `int4:metal` | IDENTICAL | `652b0544a1d5c83c` | 525,736,387 |
| `coder-0.5b-q4km` | `int4:cpu-arm64:e4h` | IDENTICAL | `13658e82d959badf` | 360,675,907 |
| `coder-0.5b-q4km` | `int8int8:cpu` | IDENTICAL | `bac71e528ce2dfdc` | 638,774,851 |
| `coder-1.5b-q4km` | `int4:cpu-arm64` | IDENTICAL | `655aa58553924d69` | 1,211,492,739 |
| `coder-1.5b-q4km` | `int4:metal` | IDENTICAL | `5b6c6956e9cbbcc0` | 1,375,265,859 |
| `coder-1.5b-q4km` | `int4:cpu-arm64:e4h` | IDENTICAL | `832aff86af916316` | 1,006,075,267 |
| `coder-1.5b-q4km` | `int8int8:cpu` | IDENTICAL | `a36c8034c2c1e067` | 1,787,283,843 |

- **Phase lines, exploratory** (`g-t1-t2-real-lines.txt`; one run each, by day, the box in use). The 1.5B's stream phase
  read 2.4-6.3 s over eight runs, and the `e4h` one 4.1-4.5 s where T1 alone read 6.4-6.8 s. The self-check read 0.2-9.4 s
  in the same runs, more than the stream in five of the eight. Not a result: G-T2 measures the transcode at night.

## G-T1 for the self-check change and T3, 2026-10-10: PASS, every cell that ran is IDENTICAL

Two changes, one run: `f006a3dd` (the transcode marks its own bundle, so the self-check reads nothing back) and
`bbb3bbeb` (T3, experts built across cores when streaming). Same instrument and box; base `eee75d83`, new `bbb3bbeb`,
which has both.

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 1 min 4 s |
| six real sources under `~/models`, four configurations each | 24 | 24 | 0 | 0 | 10 min 56 s |

- The NOT RUN set is T6's, source for source.
- **The real run is every source in the `-assets real` table that is on this box**, not the two GGUFs of T2's run: the
  0.5B and 1.5B Q4_K_M GGUFs, and four safetensors directories (Qwen3 0.6B bf16, LFM2.5 2.6B, InternLM2 1.8B,
  Spark 2.5 1.7B). It ran beside `gate quick`, which is why it took eleven minutes; an identity check, not a timing.
- **No real MoE is among them.** T3 changes only how a MoE layer's experts are scheduled, so by day it rests on the
  tiny fixtures (the matrix's MoE directories) and on `TestStreamTranscode_expertFanOutWritesTheSameBytes`, which
  streams one source per expert loop with the fan-out on and held off and compares the bytes: per-expert tensors, the
  fused stack in its transposed layout (Qwen3-VL-MoE's) and in its row layout (Granite's), Gemma 4's two stacks,
  Nemotron's, Llama 4's, and the GGUF stacked experts. Granite's and Gemma 4's tiny fixtures are gitignored: both were
  on this box and ran here, and the test reports them as not checked where they are absent (CI).
  The real MoE cell is nobara's Qwen3-VL-30B directory, at night, as registered.
- **Also checked before the run:** `go test -race` over the streaming tests reports no race, and both planted defects
  (never fan out; fan out in a resident build) turn the tests red.
- The eight GGUF files are the files T2's run wrote: the same sha256 and byte counts as the table above.

| source | configuration | verdict | sha256 (first 16) | bytes |
|---|---|---|---|---|
| `0.6b-bf16` | `int4:cpu-arm64` | IDENTICAL | `319e2fc2c405b007` | 571,805,953 |
| `0.6b-bf16` | `int4:metal` | IDENTICAL | `d4d8e7f7d5240ed9` | 626,854,849 |
| `0.6b-bf16` | `int4:cpu-arm64:e4h` | IDENTICAL | `5d8f9f50085d2215` | 434,455,809 |
| `0.6b-bf16` | `int8int8:cpu` | IDENTICAL | `c9ae5cc135cccd5c` | 765,858,049 |
| `2.6b` (LFM2.5) | `int4:cpu-arm64` | IDENTICAL | `39ca75ba0d8f011c` | 2,919,966,605 |
| `2.6b` | `int4:metal` | IDENTICAL | `5a22007479411635` | 3,178,177,709 |
| `2.6b` | `int4:cpu-arm64:e4h` | IDENTICAL | `f3c4fc20ec2b6bda` | 2,804,766,605 |
| `2.6b` | `int8int8:cpu` | IDENTICAL | `236e78b395189fe0` | 3,826,698,125 |
| `1.8b` (InternLM2) | `int4:cpu-arm64` | IDENTICAL | `b81fed8060005e18` | 1,229,561,571 |
| `1.8b` | `int4:metal` | IDENTICAL | `9867373a04518d83` | 1,418,304,099 |
| `1.8b` | `int4:cpu-arm64:e4h` | IDENTICAL | `e832d08e0f9a8bd4` | 1,062,982,371 |
| `1.8b` | `int8int8:cpu` | IDENTICAL | `ec73311566ed00ea` | 1,892,523,747 |
| `1.7b` (Spark 2.5) | `int4:cpu-arm64` | IDENTICAL | `58849182f6998e9b` | 1,090,631,742 |
| `1.7b` | `int4:metal` | IDENTICAL | `43e11c541729f3bb` | 1,270,461,182 |
| `1.7b` | `int4:cpu-arm64:e4h` | IDENTICAL | `a1d335e1b0604121` | 972,666,942 |
| `1.7b` | `int8int8:cpu` | IDENTICAL | `62923763e14cd3dc` | 1,722,333,246 |

### The self-check after the change (exploratory: one run each, by day, the box in use)

From the 48 phase lines in `g-t1-t3-real-lines.txt` (24 cells, the new build twice):

| file | self-check before (the runs above) | self-check now |
|---|---|---|
| 0.5B GGUF, 344–609 MB | 0.1 s in fifteen of sixteen runs | 0.0 s in eight of eight |
| 1.5B GGUF, 959–1,704 MB | 0.2–9.4 s | 0.0–0.2 s |
| the four directories up to 1.8 GB | not run before | 0.0–0.3 s |
| LFM2.5 2.6B, 2.7–3.6 GB | not run before | 2.7–3.9 s |

- **What changed:** the transcode writes the verified-marker for the bundle it has just written, so the self-check's
  load skips the whole-file CRC. The load itself still runs: the bundle deserializes and its shapes validate.
- (The attribution in the next bullet is from Go's profiler on the Mac, withdrawn 2026-10-10: where LFM2.5's
  self-check spends its time is not established. That no CRC pass runs is established by the marker and its tests.)
- **Why LFM2.5 still waits:** its load copies large f32 tensors out of the mapping (`giwReader.f32`, about 0.7 s of
  CPU), so the load itself reads much of the file it has just written. A CPU profile of one such transcode shows no
  CRC under the self-check. It is the load's own read, and the format change (V) is where it goes away.
- **What is given up** is in the task doc's F9 and Decision 5: after a transcode nothing compares the bytes on disk
  with the CRC unless `GOINFER_GIW_VERIFY=always` is set.
- Not a result: G-T2 measures the transcode at night.

## G-T1 for the read-back and the size rule, 2026-10-10: PASS, every cell that ran is IDENTICAL

`06f82beb` reads the weights back behind the write and leaves a bundle larger than available RAM without a marker (the
task doc's F9 and Decision 5, as amended). Neither changes a byte of the bundle. Same instrument and box; base
`eee75d83`, new `06f82beb`.

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 1 min 10 s |
| the 0.5B and 1.5B Q4_K_M GGUFs, four configurations each | 8 | 8 | 0 | 0 | 1 min 53 s |

- The NOT RUN set is T6's, source for source. The eight real files have the sha256 of the run above.
- **What the gate does not exercise:** the size rule's other branch. Every file here fits in memory, so every one was
  marked by the transcode. The unmarked branch is pinned by `TestTranscode_largerThanRAMGetsTheCRCPass` (available
  memory forced to one byte, and to unknown) and has not run on a real larger-than-RAM bundle.
- **The read-back's cost, exploratory** (`g-t1-rb-real-lines.txt`; one run each, by day, beside `gate quick`): the
  stream phase reads 0.4–1.6 s for the 0.5B and 1.2–5.7 s for the 1.5B, where the run above read 0.4–1.6 s and
  1.0–5.3 s without it. The self-check reads 0.0–0.1 s. Not a result: G-T2 measures the transcode at night.

### What a trailing read can and cannot check (a scratch file, no goinfer code, exploratory)

Measured before building it: 1.7 GB written in 4 MB pieces over about 6 s, two rounds, the box in use.

| check | cost | what it compares with the writer's CRC |
|---|---|---|
| read the file back through a mapping after the write (the old self-check) | 2.4 s in round 1, 0.3 s in round 2 | the system's cached copy |
| a second thread reading behind the writer | none: the write took 5.8–6.0 s with it and 6.3–6.4 s without, and the reader was level with the writer when the write ended | the same cached copy |
| full fsync, then a read with the cache-bypass flag | 0.0 s and 0.2–0.3 s | not established: 6–8 GB/s, and pages already cached may still have served it |

- So a trailing read costs nothing here and checks what the old read-back checked for a file that fits in memory. It
  does not show what the disk stored, and the third row is not evidence of a disk read either.
- The old read-back reached the disk only for a bundle larger than free memory, whose early pages had been evicted.
  The size rule keeps the CRC pass for exactly those.

## T4a on arm64, 2026-10-10: not adopted there (an in-process A/B, direction only)

T4a replaces the int4 quantizer's `int(math.Round(float64(x)))`. Before changing aikit, three forms were timed in one
process on this box, interleaved (old, branch, copysign; 31 rounds; each arm quantizes the same 256 rows of 8,960
normal weights four times per round, 9.2M elements, group 32), and compared for equality over every float32.

| form | time against `math.Round`, median (range) | faster in |
|---|---|---|
| `if f < 0 { int(f - 0.5) } else { int(f + 0.5) }` (the task doc's) | 1.783 (1.765–1.798) | 0 of 31 |
| `int(f + math.Copysign(0.5, f))` | 1.007 (1.000–1.020) | 2 of 31 |

- **Equality:** the three agree on all 2³² float32 bit patterns on arm64: 0 mismatches. NaN gives 0, +Inf the int64
  maximum and −Inf the minimum, in all three.
- **Reading:** `math.Round` is a single instruction on arm64 (`FRINTA`), and the sign test in the first form
  mispredicts on weights of random sign. arm64 keeps `math.Round`.
- **Owed:** the same A/B and exhaustive check on amd64, where `math.Round` is a function call. nobara held a timed run
  (another campaign's night job) when this was measured, so it waits.
- A kernel microbenchmark gives direction, not size. The harness was a scratch test in aikit's `linalg` and is not
  committed.

## T4a on amd64, 2026-10-10: the branch-free form wins and ships; the task doc's form is slower there too

The same harness as a self-contained module, on nobara (Ryzen 7 3700X, Go 1.27.0, linux/amd64), run directly under the
timing lock at 11:34 after the owner freed the box. Two passes of 31 rounds; log `t4a-ab-amd64-nobara.log`.

| form | time against `math.Round`, median (range), pass 1 | pass 2 | faster in |
|---|---|---|---|
| `if f < 0 { int(f - 0.5) } else { int(f + 0.5) }` (the task doc's) | 1.158 (1.136–1.176) | 1.158 (1.141–1.169) | 0 of 31, twice |
| `int(f + math.Copysign(0.5, f))` | 0.698 (0.681–0.715) | 0.698 (0.689–0.713) | 31 of 31, twice |

- **Equality:** the three agree on all 2³² float32 bit patterns on amd64: 0 mismatches. NaN, +Inf and −Inf all convert
  to the int64 minimum there, in all three (on arm64: 0, the maximum, the minimum).
- **Reading:** on weights of random sign the sign test mispredicts on amd64 as on arm64, so the task doc's form loses
  on both. The audit's −17% for it was exploratory, on a synthetic file, and does not reproduce here.
- **What ships:** aikit's `roundInt4` is the `Copysign` form on amd64 and `math.Round` elsewhere (arm64: 1.007×, so no
  gain to take). It changes no output. Size is not claimed from this: the transcode's ratio is G-T2's, at night.


## T5, 2026-10-10: built, interim identity only (the gate waits for an aikit release)

T5 reads the transposed fused-expert layout through a new aikit call, `embed.Tensor.SubF32Transposed`, and stops
copying matrices that are already fresh slices. It is on branch `giw-v16-t5` and builds only against the local aikit
checkout until aikit v1.65.0 is tagged, so `gate identity`, which builds each revision with `GOWORK=off`, cannot run it
yet. What was checked instead, on this box:

- **Bytes, by hand:** `cmd/prequant` built from `ddd84996` and from the T5 worktree, each transcoding `qwen3vlmoe-tiny`
  (the transposed layout) and `granite-tiny` (the row layout) in six configurations (`int4:metal:e4h`, `int4:metal`,
  `int4:cpu-arm64`, `int4:cpu-arm64:e4h`, `int8int8:cpu`, `int8:cpu`): the same sha256 in 12 of 12.
- **In test:** `TestLoadFusedExperts_transposedLayoutGivesTheRowLayout` (the same experts in both layouts build the same
  matrices in f32, int8, int8int8 and int4, with a sink and without) and `TestLoadFusedExperts_f32ExpertsOwnTheirData`.
- **Not covered here:** a bf16 fused stack. Both fixtures are F32. The conversion is pinned in aikit, bit-equal to
  `SubF32` in every dtype; a real bf16 stack is nobara's Qwen3-VL-30B.
- **Speed, exploratory (in-process, interleaved, direction only):** one [2048 × 1536] slice, eight per round, 21 rounds.
  The tiled call against `SubF32` plus the plain transpose: 0.324 (0.312–0.334) for bf16 and 0.261 (0.250–0.265) for
  f32, faster in 21 of 21 each.

G-T1 for T5 is owed once the pin is bumped.

## G-T1 for T5 and aikit v1.65.0, 2026-10-10: PASS, every cell that ran is IDENTICAL

aikit v1.65.0 was tagged the same day (the transposing read, and T4a's rounding on amd64), goinfer's pin was bumped
(`05e8aeeb`) and T5 went onto the branch (`a8263c3b`). Same instrument and box; base `eee75d83`, new `fe8f8136`, which
has all of it.

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 384 | 330 | 0 | 54 | 1 min 18 s |
| the 0.5B and 1.5B Q4_K_M GGUFs, four configurations each | 8 | 8 | 0 | 0 | 2 min 6 s |

- The NOT RUN set is T6's, source for source. The eight real files have the sha256 of the runs above.
- **T5's cells:** `qwen3vlmoe-tiny` (the transposed layout) and `granite-tiny` (the row layout), six configurations
  each, 12 of 12 IDENTICAL. Both are F32. No bf16 fused stack is on this box.
- **T4a's cells: none here.** The rounding changes code on amd64 only; on this box the new pin runs the same
  `math.Round` as before. Its bytes rest on aikit's oracle tests, which pass with their full sweeps natively on amd64,
  and on this gate's amd64 cells, which are nobara's at night.
- Phase lines, exploratory: `g-t1-t5-real-lines.txt`.

## The T runs, 2026-10-10: G-T1's real cells PASS, G-T2 SHIP on every graded cell, G-T3 within its bounds, G-T2c recorded

Both jobs ran from the night queues, which the owner started by day: the Mac's `giw-gt-mac` at 13:31-13:38 (load
average 2.48 at its start, behind four other jobs) and nobara's `giw-gt-nobara` at 13:09-14:05 (load 0.91). Driver
`gt_night.py`; old arm `cf66f5b3`, new arm `6398b0f8`; one discarded warm-up per arm, then pairs. Records:
`gt-mac.jsonl`, `gt-nobara.jsonl` (one line per event) and the two logs.

### G-T1, real cells: PASS

Every transcode of a cell, warm-ups included, wrote the same sha256 from both arms: 7 cells, 50 transcodes.

| box | cell | bytes | sha256 (first 12) |
|---|---|---|---|
| Mac | 1.5B int4 `cpu-arm64` | 1,211,492,739 | `655aa5855392` |
| Mac | 1.5B int4 `metal` | 1,375,265,859 | `5b6c6956e9cb` |
| Mac | 7B int4 `metal` | 5,584,647,875 | `8f38cbe90e6f` |
| nobara | 1.5B int4 `cpu-amd64` | 1,211,492,739 | `41f376a9bbb4` |
| nobara | 7B int4 `cpu-amd64` | 4,768,988,163 | `1981fdecb906` |
| nobara | 7B int4 `cuda` | 4,768,988,163 | `1981fdecb906` |
| nobara | Qwen3-VL-30B-A3B int4 `cpu-amd64`, from its directory | 17,499,790,393 | `f2347ac62c93` |

- **This is the first run of three things:** T4a's rounding through goinfer on amd64 (all four nobara cells), a bf16
  fused expert stack through T5's transposing read, and T3's expert fan-out on a real MoE (the 30B).
- The 7B's `cuda` file is byte for byte its `cpu-amd64` file, in both arms: on amd64 the two targets write one
  layout, so that cell repeats the one above it.

### G-T2, speed: SHIP on the three graded cells

Graded on the paired new/old ratio of the full wall. The stream phase's ratio is beside it, ungraded, as amended.

| cell | old wall | new wall | new/old, each pair | median | stream new/old | band | verdict |
|---|---|---|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` | 10.6-10.7 s | 3.0 s | 0.278, 0.278, 0.279 | **0.278** | 0.279 | ship at most 0.45 (expected 0.18-0.33) | **SHIP** |
| nobara 1.5B `cpu-amd64` | 17.2-17.3 s | 2.3 s | 0.131, 0.132, 0.130 | **0.131** | 0.123-0.129 | the same | **SHIP** |
| nobara 7B `cpu-amd64` | 82.8-83.0 s | 9.6-9.8 s | 0.117, 0.117, 0.118 | **0.117** | 0.116-0.118 | the same | **SHIP** |
| nobara 30B directory | 657-665 s | 110-125 s | 0.189, 0.168 | **0.178** | 0.153, 0.134 | ship at most 0.40 (expected 0.18-0.25) | **SHIP** |
| Mac 1.5B `metal` (recorded) | 10.4-10.5 s | 2.7 s | 0.259, 0.259, 0.257 | 0.259 | 0.248-0.257 | not in the table | |
| Mac 7B `metal` (recorded) | 60.3-60.4 s | 13.7-14.8 s | 0.246, 0.226 | 0.236 | 0.286, 0.263 | not in the table | |
| nobara 7B `cuda` (recorded) | 82.7 s | 9.8 s | 0.118, 0.119 | 0.118 | 0.117-0.118 | not in the table | |

- **Against the expectation:** the Mac's cell lands inside the expected range. nobara's two dense cells land below it
  (0.12-0.13 against 0.18-0.33); the stream phase reads the same, and this record does not break the gain down by
  item. The 30B lands at the range's lower edge.
- No stream-phase ratio is in a kill band, so no full-wall ship rests on the self-check's removal alone. On the Mac's
  7B the self-check does account for part of it: 10.9-11.0 s in the old arm (the CRC read-back of 5.6 GB), 0.5 s in
  the new one, and the stream phase alone is 0.26-0.29.

### G-T3, memory: within the bounds, with two things the record cannot show

| cell | old peak | new peak | increase | bound |
|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` (peak footprint) | 1.24 GB | 1.32 GB | 0.08 GB | 1.5 x K layers' bytes: about 0.5 GB at K = 8 |
| Mac 1.5B `metal` | 1.23 GB | 1.28 GB | 0.05 GB | the same |
| Mac 7B `metal` | 2.93 GB | 4.49 GB | 1.56 GB | about 2.2 GB at K = 8 |
| nobara 1.5B `cpu-amd64` (max RSS) | 2.25 GB | 2.34 GB | 0.09 GB | about 1.0 GB at K = 16 |
| nobara 7B `cpu-amd64` | 7.37 GB | 9.44 GB | 2.07 GB | about 4 GB at K = 16 |
| nobara 7B `cuda` | 7.37 GB | 9.55 GB | 2.18 GB | the same |
| nobara 30B directory | 55.11 GB | 52.86 GB | none | old peak plus workers x 3 x one expert's f32 |

- **Mac swap: flat.** 1,021.5 MB used before and after every one of the 22 transcodes.
- **Not shown: Decision 3's budget.** The driver did not record the available memory each transcode started with, so
  "never above a quarter of it" is checked only loosely: nobara's quarter is about 14 GB against a 2.1 GB increase; the
  Mac's 1.56 GB increase needs 6.2 GB available, and a `vm_stat` taken six minutes earlier by another job shows about
  7.4 GB inactive. The driver should record it.
- **A weak instrument on nobara:** `time -v`'s maximum RSS counts the mapped source, so the 30B's 53-55 GB is mostly
  its 58 GB of safetensors and says little about the build's own memory. The Mac's figure is the footprint.

### G-T2c, the copy floor: recorded

Transcode wall over C-io (read the source, write the output, no compute) and over C-src (a `dd` of the source).

| cell | C-io | old/C-io | new/C-io | C-src | new/C-src | the text's projection for new/C-io |
|---|---|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` | 0.37 s | 28.8 | **8.0** | 0.31 s | 9.6 | 4-12 |
| Mac 1.5B `metal` | 0.44 s | 23.9 | 6.2 | 0.32 s | 8.5 | |
| Mac 7B `metal` | 2.26 s | 26.7 | 6.3 | 2.54 s | 5.6 | |
| nobara 1.5B `cpu-amd64` | 0.41 s | 42.0 | 5.5 | 0.30 s | 7.6 | |
| nobara 7B `cpu-amd64` | 1.56 s | 53.1 | **6.2** | 1.20 s | 8.1 | 3-10 |
| nobara 7B `cuda` | 1.57 s | 52.7 | 6.2 | 1.21 s | 8.1 | |
| nobara 30B directory | 58.9 s | 11.2 | **2.0** | 50.1 s | 2.4 | 3-8 |

- The three projected cells: the two dense ones land inside their "after T" ranges; the 30B lands better (2.0 against
  3-8), because its copy is slower than projected (49-69 s against 20-40 s; reading 58 GB took 39-51 s of it).
- "Today" was projected at 15-30x for nobara's 7B and reads 53x: the copy there is faster than the 3-6 s assumed.
- The registered consequence is read after T4b, not here. For reference, both dense cells are above its 3x line now.
- The 30B's new arm (110-125 s) does not finish faster than its source's copy (45-55 s); the text asks that question
  after T4b.

### Two findings from the 30B

- **Its self-check got slower, and it is not a CRC pass.** Old arm 9.6-11.6 s, new arm 19.2-24.9 s (warm-ups
  included). A profiled
  repeat by day (20.6 s; the profile is not kept) shows no CRC under the self-check and 11.3 s of CPU in the bundle
  reader's walk over each matrix's header. The reading that fits: with 58 GB of source mapped on a 62 GB box, the
  17.5 GB output is no longer in the page cache when the self-check loads it. The old arm's CRC then read it back in
  one sequential pass (1.6 GB/s, where the 7B's cached CRC ran at about 12 GB/s), and the new arm's load faults it
  back in piecemeal, which is slower. What the faults wait on was not measured.
- **So the size rule marked a bundle that was not in memory.** `markWritten` compares the bundle with the memory
  available before the write, and Linux counts reclaimable cache as available (57-62 GB here), so the 30B "fit" and
  its whole-file CRC was skipped, on exactly the kind of file the rule was meant to keep it for. The rule needs the
  source in the sum (bundle plus source against total memory), or a direct check of what is cached. Not changed
  here; it is an open item for Decision 5. The larger-than-RAM branch has still not run on a real bundle.

## The size rule counts the source, 2026-10-10 (`852ebd3e`): identity and `gate quick` only

After the finding above, the owner chose the first candidate: a bundle is marked only when it and its source together
are no larger than installed memory. The change is in which bundles get a marker, not in any byte written.

- G-T1, tiny matrix, base `eee75d83`, new `852ebd3e`: 330 IDENTICAL, 0 DIFFERENT, 54 NOT RUN (the same nine
  directories), 54 s. The real cells were not repeated: no code that writes a bundle changed.
- `gate quick`: green, 2,315 tests passed.
- On the T runs' cells the rule now leaves nobara's 30B unmarked (17.5 GB plus 58 GB against 62 GB) and marks the
  other six. **Not re-measured:** the 30B's self-check time with the CRC back. `gt_night.py` now records the
  available memory before each transcode, for the next run's G-T3.

## T4b's day gate, 2026-10-10: SHIP on both graded cells; and Go's profiler on the Mac was wrong about the shares

T4b vectorizes aikit's int4 quantizer (NEON and AVX2). Its row in G-T2 is the single-thread CPU time of a transcode,
new over old, registered before the reading with its cells and instrument (the task doc's amendment).

- **Instrument:** `t4b_cpu_ab.py`. Two `cmd/prequant` builds of one goinfer revision (`23a2c8c0`): old on aikit
  v1.65.0, new on aikit `244a10e`. `GOMAXPROCS=1`, user plus system seconds from `time`, one discarded warm-up per
  arm, five pairs in alternating order. Records `t4b-cpu-mac.jsonl`, `t4b-cpu-nobara.jsonl`.
- **Boxes:** the Mac at 16:10 (load average 3.3, by day, in use); nobara at 16:17 under the timing lock (load 0.13).

| cell | old CPU | new CPU | new/old, five pairs | median | band | verdict |
|---|---|---|---|---|---|---|
| nobara, 1.5B Q4_K_M GGUF to `cpu-amd64` | 12.1 s | 3.7 s | 0.303-0.308 | **0.303** | ship at most 0.70 (expected 0.4-0.6) | **SHIP** |
| Mac, Qwen3 0.6B bf16 directory to `cpu-arm64` | 4.2 s | 1.7 s | 0.398-0.414 | **0.407** | the same | **SHIP** |
| Mac, 1.5B GGUF to `cpu-arm64` (recorded) | 10.5 s | 3.2 s | 0.303-0.306 | 0.304 | | |
| nobara, 0.6B bf16 directory (recorded) | 5.0 s | 2.0 s | 0.399-0.410 | 0.402 | | |

- Every cell's old and new arms wrote the same sha256. This is G-T1 by hand for T4b on four cells; `gate identity`
  runs it once goinfer pins the release.
- The kernel alone, in one process, interleaved, 31 rounds of 9.2M normal weights (direction): 0.094 of the scalar's
  time on the Mac (6.23 to 0.59 ns a weight), 0.097 on nobara (7.2 to 0.70 ns), faster in 31 of 31 each.

### G-T1 for T4b through the gate, 2026-10-10: PASS, every cell that ran is IDENTICAL

aikit v1.66.0 was tagged the same afternoon (perfgate PASS against v1.65.0 and the vulnerability scan clean, both on
nobara) and goinfer pinned it (`c3e921e6`). `gate identity -giw bytes`, base `eee75d83`, new `f4babc67`:

| run | cells | IDENTICAL | DIFFERENT | NOT RUN | wall |
|---|---|---|---|---|---|
| tiny fixtures, six configurations | 390 | 330 | 0 | 60 | 1 min |
| six real sources under `~/models`, four configurations each | 24 | 24 | 0 | 0 | 9 min 45 s |

- NOT RUN is T6's nine directories and one that `main` has since gained, `whisper-tiny-ts` (a Whisper
  encoder-decoder, not a decoder model).
- The 24 real files have the sha256 of the run that carried T3, cell for cell.
- This is arm64's kernel. The amd64 kernel's bytes are the day gate's two nobara cells above, and the night's cells.
- `gate quick` on the pin: green, 2,337 tests passed.

### The retraction this gate forced

The amendment that named the cells said the Mac's GGUF cell "cannot go under about 0.86", from the arm64 profile's 14%
for the quantizer. It read 0.304. So the profile was checked against macOS's `sample`, which suspends the process and
reads its stacks, on the old build with one thread (1.5B GGUF to int4 `cpu-arm64`):

| | Go CPU profile | `sample`, on-CPU |
|---|---|---|
| `QuantizeGroupInt4Row` | 12% | **83%** |
| `dequantQ4KBlock` (with `LittleEndian.Uint16`) | 73% | 5.8% |
| `dequantQ6KBlock` | not in the top nine | 1.5% |

- The sampler agrees with the timing (the quantizer's 6.2 ns a weight over 1.5 billion weights is the 9.6 s the old
  build spends), and with amd64, where a Linux profile had said 81%. Go's profile on the Mac does not.
- **After T4b, by the sampler:** `dequantQ4KBlock` about 24% of the on-CPU samples, the new kernel about 21%, CRC 8%,
  the file's row4 repack 7%, `dequantQ6KBlock` and `F32ToF16` about 5% each. For the bf16 directory before T4b: the
  quantizer about 44%, the bf16 widening about 43%; after: the widening about 40%, the quantizer (kernel, wrapper and
  max) about 24%.
- **What is withdrawn:** the audit's arm64 section and the task doc's F3 (the quantizer "about 14%", the dequantizer
  "about two thirds", "T4 can remove at most about 14%"), and with them every other share on this page that came from
  a Go profile on the Mac, each marked where it stands. Why that profiler is wrong here is not established.
- **For the next profile on the Mac:** `sample <pid> <seconds>`, read under "Sort by top of stack", leaving out the
  threads parked in the kernel.

## The T runs with T4b, 2026-10-10: G-T1 PASS, G-T2 SHIP on every graded cell, the copy floor's consequence opens on one cell

The same cells and driver as the T runs above, with T4b in the new arm: old `cf66f5b3`, new `946e5ca8` (aikit
v1.66.0, and the size rule that counts the source). Both queues were started by day. The Mac's `giw-gt-mac-t4b` ran
17:29-17:35 after the settle wait gave up at a load average of 2.79 (macOS's own daemons, woken when the charger went
in). nobara's `giw-gt-nobara-t4b` ran 17:57-18:54 at a load of 0.90. Records: `gt-mac-t4b.jsonl`,
`gt-nobara-t4b.jsonl` and their logs.

- **G-T1, real cells: PASS.** All seven cells, 50 transcodes, the same sha256 from both arms, and the sha256 of the
  earlier run cell for cell.

### G-T2 with T4b: SHIP on the four graded cells

| cell | old wall | new wall | new/old, each pair | median | without T4b | band | verdict |
|---|---|---|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` | 10.5-10.6 s | 1.35-1.38 s | 0.128, 0.131, 0.128 | **0.128** | 0.278 | ship at most 0.45 | **SHIP** |
| nobara 1.5B `cpu-amd64` | 17.2-17.3 s | 1.12-1.15 s | 0.065, 0.067, 0.065 | **0.065** | 0.131 | the same | **SHIP** |
| nobara 7B `cpu-amd64` | 82.6-82.8 s | 4.21-4.26 s | 0.051, 0.052, 0.051 | **0.051** | 0.117 | the same | **SHIP** |
| nobara 30B directory | 662-667 s | 98-142 s | 0.214, 0.148 | **0.181** | 0.178 | ship at most 0.40 | **SHIP** |
| Mac 1.5B `metal` (recorded) | 10.5-10.6 s | 1.25-1.26 s | 0.119, 0.119, 0.118 | 0.119 | 0.259 | | |
| Mac 7B `metal` (recorded) | 63.4-63.9 s | 7.3-10.4 s | 0.163, 0.115 | 0.139 | 0.236 | | |
| nobara 7B `cuda` (recorded) | 83.2-83.3 s | 4.24-4.32 s | 0.052, 0.051 | 0.052 | 0.118 | | |

- **The dense cells roughly halve again.** A 7B GGUF on nobara transcodes in 4.2 s where the old build takes 83 s.
- **The 30B does not move** (0.181 against 0.178, and its two pairs differ by half: 142 s and 98 s). Its new arm is
  now within a factor of two of reading the source and writing the output (below), so the quantizer is no longer
  what it waits on.
- **The Mac's 7B pairs differ too** (10.4 s and 7.3 s for the new arm), on a box in use with background daemons
  running. That cell is recorded, not graded.

### G-T3 with the available memory recorded: within the bounds

| cell | old peak | new peak | increase | available before | a quarter of it |
|---|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` (footprint) | 1.24 GB | 1.31 GB | 0.07 GB | 10.0-11.1 GB | 2.5-2.8 GB |
| Mac 7B `metal` | 2.93 GB | 4.49 GB | 1.56 GB | 9.8-10.2 GB | 2.5-2.6 GB |
| nobara 7B `cpu-amd64` (max RSS) | 7.37 GB | 9.63 GB | 2.26 GB | 63.6 GB | 15.9 GB |
| nobara 30B directory | 55.08 GB | 52.58 GB | none | 63.6 GB | |

- Every dense increase is under a quarter of the memory available when its transcode started, which is the clause the
  first runs could not check. Mac swap: 1,059.0 MB before and after all 22 transcodes.

### G-T2c after T4b, and its registered consequence

| cell | C-io | new/C-io | without T4b | the text's "after T + T4b" | C-src | new/C-src |
|---|---|---|---|---|---|---|
| Mac 1.5B `cpu-arm64` | 0.44 s | **3.08** | 8.0 | 2-6 | 0.47 s | 2.9 |
| Mac 1.5B `metal` | 0.60 s | 2.09 | 6.2 | | 0.53 s | 2.4 |
| Mac 7B `metal` | 3.69 s | 2.40 | 6.3 | | 3.62 s | 2.5 |
| nobara 1.5B `cpu-amd64` | 0.40 s | 2.83 | 5.5 | | 0.29 s | 3.9 |
| nobara 7B `cpu-amd64` | 1.56 s | **2.72** | 6.2 | 2-6 | 1.20 s | 3.5 |
| nobara 7B `cuda` | 1.55 s | 2.76 | 6.2 | | 1.19 s | 3.6 |
| nobara 30B directory | 65.6 s | **1.83** | 2.0 | 2-5 | 53.3 s | 2.3 |

- The three projected cells land inside their ranges, the 30B a little better than its.
- **The consequence** ("above 3x on either dense cell opens a follow-up") reads on the Mac's 1.5B and nobara's 7B.
  nobara's is under the line (2.72). **The Mac's is over it by a hair (3.08), so the follow-up opens**, and it should
  be read for what it is: the three pairs' own ratios are 3.09, 3.19 and 2.54, because the copy itself varied
  (0.43-0.53 s) on a box in use. A quiet run could land either side.
- **What the follow-up would rank,** from the system sampler after T4b on that cell (above): the Q4_K dequantizer
  about 24% of the on-CPU samples, the int4 kernel about 21%, the CRC 8%, the file's row4 repack 7%. So the registered
  first lever, a vectorized Q4_K/Q6_K dequantizer in aikit, is the largest single item and bounds the gain at about a
  quarter of the CPU.
- The 30B's new arm (98-142 s) does not finish faster than a copy of its source (53 s), which the text asked to have
  recorded.

### The 30B's self-check with its CRC back

The size rule now leaves the 30B unmarked (17.5 GB plus 58 GB of source against 62 GB of memory), so its self-check
runs the whole-file CRC again. It reads 20.8-28.0 s in the new arm, against 9.5-13.3 s in the old arm, which runs the
same CRC. Before the fix the new arm read 19-25 s with no CRC. So the fix restored the check and did not change the
time, and the arms' difference is not the CRC: the old arm's check follows an eleven-minute stream and the new arm's
a ninety-second one. What the file's pages are doing in that case was not measured.

## Files

- `g-t1-t6-tiny.log`, `g-t1-t6-real.log`: the two reports (colour codes removed, the run directory shortened).
- `g-t1-t6-tiny.jsonl`, `g-t1-t6-real.jsonl`: one line per cell (verdict, both builds' sha256 and byte counts).
- `g-t1-t6-real-lines.txt`: each transcode's own "wrote" line from the real run.
- `gt_night.py`, `run-gt-night-mac.sh`, `run-gt-night-nobara.sh`: the T runs' driver and the two boxes' cells.
  `gt-mac.jsonl`, `gt-mac.log`, `gt-nobara.jsonl`, `gt-nobara.log`: their records. `gt-mac-t4b.*`, `gt-nobara-t4b.*`:
  the same runs with T4b in the new arm.
- `t4b_cpu_ab.py`, `t4b-cpu-mac.jsonl`, `t4b-cpu-nobara.jsonl`: T4b's day gate and its records. `g-t1-t4b-*`: its
  run through `gate identity`, the same five files as the earlier runs.
- `t4a-ab-amd64-nobara.log`: the rounding A/B's log from nobara, both passes.
- `g-t1-t1-*`, `g-t1-t2-*`, `g-t1-t3-*`, `g-t1-rb-*`, `g-t1-t5-*`: the same five files for T1's run, for T2's, for the
  run that carries the self-check change and T3, for the read-back's, and for T5's on aikit v1.65.0.
