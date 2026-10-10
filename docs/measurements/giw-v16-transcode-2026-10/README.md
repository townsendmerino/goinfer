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

## Files

- `g-t1-t6-tiny.log`, `g-t1-t6-real.log`: the two reports (colour codes removed, the run directory shortened).
- `g-t1-t6-tiny.jsonl`, `g-t1-t6-real.jsonl`: one line per cell (verdict, both builds' sha256 and byte counts).
- `g-t1-t6-real-lines.txt`: each transcode's own "wrote" line from the real run.
- `g-t1-t1-*`, `g-t1-t2-*`: the same five files for T1's run and for T2's.
