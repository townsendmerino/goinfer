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
profile put the removed repack at about 2% of a 1.5B transcode's CPU.

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
- **Why LFM2.5 still waits:** its load copies large f32 tensors out of the mapping (`giwReader.f32`, about 0.7 s of
  CPU), so the load itself reads much of the file it has just written. A CPU profile of one such transcode shows no
  CRC under the self-check. It is the load's own read, and the format change (V) is where it goes away.
- **What is given up** is in the task doc's F9 and Decision 5: after a transcode nothing compares the bytes on disk
  with the CRC unless `GOINFER_GIW_VERIFY=always` is set.
- Not a result: G-T2 measures the transcode at night.

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

## Files

- `g-t1-t6-tiny.log`, `g-t1-t6-real.log`: the two reports (colour codes removed, the run directory shortened).
- `g-t1-t6-tiny.jsonl`, `g-t1-t6-real.jsonl`: one line per cell (verdict, both builds' sha256 and byte counts).
- `g-t1-t6-real-lines.txt`: each transcode's own "wrote" line from the real run.
- `g-t1-t1-*`, `g-t1-t2-*`, `g-t1-t3-*`: the same five files for T1's run, for T2's, and for the run that carries the
  self-check change and T3.
