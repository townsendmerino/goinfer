# Vetted-checkpoint peer cells, MacBook (pre-registered 2026-10-07)

**Status: PRE-REGISTERED 2026-10-07, before any timed run.** Two cells are queued (night job `peer-vetted-mac`): the
26B on Metal, and gpt-oss on Metal (added by the amendment below, still before any timed run). Phi-3 at q4k on Metal is
blocked, for the reason below. Results and verdicts go below *Results*.
A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Question.** goinfer.dev/models has empty Metal speed cells for three vetted checkpoints. The Mac fills them on Metal
only. Every CPU cell is nobara's ([`peer-vetted-2026-10-07-nobara-pc.md`](peer-vetted-2026-10-07-nobara-pc.md)), and a
CPU 26B is forbidden on this Mac. Granite 4.0-H Tiny is out of scope.

## The three Mac cells

| cell | checkpoint (docs/capability-matrix.json) | state |
|---|---|---|
| M1 | Phi-3 mini 4k at `--quant q4k` on Metal | **BLOCKED: a code fix, the owner's call** (below) |
| M2 | gpt-oss 20B MXFP4 (sha256 `27cd6c43…`) on Metal | **QUEUED** (amendment below; it was blocked by disk at first) |
| M3 | Gemma 4 26B-A4B, the QAT `gemma-4-26B_q4_0-it.gguf` (sha256 `3eca3b8f…`), on Metal | **QUEUED** |

- **M1, Phi-3 at q4k on Metal: Metal declines q4k.** `decoder/residency.go`:
  `--quant q4k (native Q4_K) has no resident kernel on metal yet`. Only CUDA has the Q4_K GEMV (`gemv_q4k_g32`).
  - The by-day smoke (2026-10-07) loaded the vetted file (sha256 `8a83c7fb…`, copied to `~/models`). The fit guard
    refused it first, on that afternoon's free memory (7.7 GB needed against 6.1 GB available).
  - The residency decline is in the code whatever the memory, so a Metal q4k cell is a code change, not a measurement.
  - Per the plan, it stops here for the owner.
- **M2, gpt-oss on Metal: not enough disk on this Mac.**
  - The Metal path needs the 12.1 GB file plus a one-time paged sidecar of about its size (the 26B's was 14.8 GB).
  - The Mac has 17 GB free after M3's files.
  - Its residency on Metal is declared in code (`FeatAttnSink`, `TestGptOssResidentParity`), but it is not confirmed on
    this machine. Per the plan, it is not scheduled before a one-prompt load confirms it.
  - Unblocked by freeing about 27 GB (regenerable sidecars and the build cache, listed for the owner). Then it gets
    the same smoke, and an amendment here before it is queued.

### Amendment, 2026-10-07 18:30 PDT, before any timed run: M2 unblocked

- **Disk:** the owner freed about 27 GB of regenerable sidecars and the build cache.
- **The file and the tag:**
  - The vetted gpt-oss file was copied from nobara: sha256 `27cd6c43…`, matching the matrix.
  - Ollama's `g20` is nobara's manifest with its model blob hard-linked to that file. `gguf_same_weights.py`: SAME
    WEIGHTS, one inode.
- **The by-day smoke, untimed, swap kill-watch at +1 GB:**
  - a one-time 11.8 GB sidecar (`gpt-oss-20b-MXFP4.int4.metal.giw`);
  - decode `metal-resident (int4mix→int4, no Metal int8 GEMV kernel)`: goinfer's default int4mix becomes int4 on
    Metal, which is disclosed on the cell;
  - a correct reply, and swap +0 MB.
- **One registered difference from M3: no `-require-backend` on M2.**
  - **The reason:** with that flag the load refused, because gpt-oss's prefill is sequential on Metal (the batched
    prefill implements neither its attention sink nor its clamped SwiGLU).
  - **The precedent:** the 2026-09-29 sweep left the flag off its decode cells for the same reason (Phi-3 on CUDA): a
    sequential prefill does not touch a decode-only rate.
  - **What still guards the cell:**
    - the harness's decode-path gate (anything but `metal-resident` is refused);
    - the swap void (any growth);
    - the +1 GB kill.
- **The cell:** M2 runs as M3 does otherwise (depth 128, greedy, 3 runs, context 2048, `-moe-cache-experts`, Ollama
  `g20` at its defaults), under the same rules, bands and GOINFER-ALONE rule.

## Cell M3, as it runs

- **The cell:** depth 128, greedy, 3 runs (pairs), 8 completions × 64 tokens a run, goinfer against Ollama 0.32.5, both
  on Metal at context 2048 (`BENCH_CTX`, as cell c of [`peer-sweep-2026-09-29.md`](peer-sweep-2026-09-29.md)).
- **goinfer:** `--backend metal -moe-cache-experts -require-backend`, its default quant (int4). It runs paged from a
  one-time sidecar (`gemma-4-26B_q4_0-it.int4.metal.giw`, 14.8 GB), which is the M26 day-use rule
  (`docs/tasks/task-m26-mac-2026-10.md`) and the only way a 14.4 GB MoE runs on a 16 GB Mac.
  - **The by-day smoke, 2026-10-07, untimed:** decode `metal-resident (int4)`, prefill layer-major on decode's kernels,
    one KV slot, a correct reply, swap +104 MB at peak by day with the owner's applications open.
- **Ollama:** tag `g4-26b`, at its defaults (its own Metal offload).
  - The tag is nobara's `g4-26b` manifest, copied, with its model blob hard-linked to the same file goinfer reads.
    So both engines read one inode.
  - `gguf_same_weights.py`: SAME WEIGHTS.

## Rules, fixed before the run

- **The same weights:** one file, read by both engines.
- **The same engine path:** Metal against Metal.
- **GPU residency:** M3 is VOID unless goinfer's decode path is `metal-resident`. The harness refuses any other path,
  and `-require-backend` refuses a Metal decline.
- **Each engine at its defaults**, as in cell c. goinfer's `-moe-cache-experts` and the context pin are disclosed on the
  cell.
- **Memory:**
  - One engine is loaded at a time.
  - `BENCH_SWAP_VOID_MB=0`: any growth in system swap over an engine's cell, from before its server starts to its last
    completion, voids that arm.
  - `BENCH_SWAP_KILL_MB=1024`: a server whose swap passes +1 GB is killed on the spot (the Mac's 2026-09-04/05 lesson;
    the threshold `swap_killwatch.sh` uses).
- **If Ollama cannot run the 26B without swapping,** goinfer's arm is recorded alone: GOINFER-ALONE, `peer: null` in
  `claims.json`, the speed with no ratio. Ollama is **not** retried with other settings to make it fit.
- **Bands, voids and the grader** are nobara's record's: the 2026-09-29 rules unchanged, plus the swap void and
  GOINFER-ALONE (`peer-vetted-2026-10-07/grade.py`).
- **Provenance:**
  - goinfer `65b2c22a`, `serve-metal` built from a clean worktree at it;
  - Ollama 0.32.5 (`/opt/homebrew/bin/ollama`);
  - macOS 26 (Darwin 25.6.0), M1 Pro 16 GB;
  - checkpoints from `~/models`.

**Decision rule.** As nobara's.
- **A valid cell** becomes a `claims.json` entry (machine `mac`) from this record's printed runs.
- **A VOID cell** stays "not measured", with its reason.
- **A loss stays a loss.**

**Cost.** Night only: two cells (M3, then M2), two engines each, about 20-25 minutes of cells and loads. Queued at 40 minutes (25 before the amendment).

## Results

### Night 1, 2026-10-07 23:25-23:27 PDT

- **The run:** night job `peer-vetted-mac` at the pinned `65b2c22a`, Ollama 0.32.5.
- **Raw:** `~/goinfer-bench/peer-vetted/results-mac-2026-10-07/` on the MacBook.
- **M3 (the 26B): not run.** The harness raised `KeyError: 'G26Q:128'`: the `G26Q` key had no calibrated prompt, the
  same crash as nobara's cell 1c. Fixed 2026-10-08 (`G26Q:128`, 129 tokens, calibrated on nobara).
- **M2 (gpt-oss):**
  - **goinfer:** 28.8 / 29.0 / 29.0 tok/s, `metal-resident (int4mix→int4, no Metal int8 GEMV kernel)`, swap +0.0 MB,
    every completion at 64 tokens.
  - **Ollama:** its warm-up failed with `HTTP Error 500` (no runs).
  - **The grader** reads GOINFER-ALONE (median 29.0).
  - **Not graded as final.** The harness discarded Ollama's log, so the 500's cause was unknown, and GOINFER-ALONE is
    for an Ollama arm that cannot run the cell.
  - **Checked by day, 2026-10-08, untimed:** Ollama loads the same `g20` tag on Metal and answers. It placed all 25
    layers on the GPU at 11,097 MiB against 12,123 MiB free. With 1 GB of headroom, the likely cause is Metal memory
    not yet released by goinfer's 7.4 GB server, whose teardown preceded Ollama's warm-up by about 15 s.

### The re-run, registered 2026-10-08 before it runs (owner: "anything to reschedule here")

- **The cells:** M3 and M2 run again as a second, separate night, same pre-registration.
- **What is unchanged:** cells, rules, bands, void conditions and binaries (`65b2c22a`). Ollama is **not** given
  different settings.
- **Two changes, both instruments:**
  - The harness runs from a checkout with `G26Q:128` in `scripts/prompts.json`.
  - It keeps Ollama's own log beside goinfer's serve logs, so a failed arm has its reason.
- **The verdicts:**
  - If M2's Ollama arm fails again, the cell is GOINFER-ALONE with that reason.
  - If it runs, the paired cell is the reported one, with night 1 stated beside it.
- **Script:** `peer-vetted-2026-10-07/run-mac-2.sh`, results in `~/goinfer-bench/peer-vetted/results-mac-2/`.

### The re-run, read 2026-10-08 04:31-04:36 PDT: both cells GOINFER-ALONE

- **The run:** night job `peer-vetted-mac-2`, started by day on the owner's word.
  - Binaries at `65b2c22a` (serve sha256 `43312af3…`); the harness from `836b33ef`; Ollama 0.32.5.
  - Load 1.60 at the start, the instant idle gate (wait 3.3 s each), no thermal warning; system swap 1,355.88 MB used
    at the start.
- **Raw:** `~/goinfer-bench/peer-vetted/results-mac-2/` on the MacBook (`metal-g26q.json`, `metal-g20.json`,
  `grade.jsonl`, and `serve-logs/` with Ollama's own logs). Night log: the night runner's
  `runs/2026-10-08/peer-vetted-mac-2.log`.
- **Graded** with `peer-vetted-2026-10-07/grade.py`, the rules above unchanged.

| cell | goinfer runs (tok/s) | median | decode path | goinfer swap growth | Ollama | outcome |
|---|---|---:|---|---:|---|---|
| M3 metal G26Q (the vetted 26B) @128 greedy, ctx 2048 | 18.9 / 20.2 / 20.2 | 20.2 | `metal-resident (int4)` | 0.0 MB | no runs: its load did not finish (below) | **GOINFER-ALONE** |
| M2 metal G20 (gpt-oss 20B) @128 greedy, ctx 2048 | 30.4 / 30.4 / 30.5 | 30.4 | `metal-resident (int4mix→int4, no Metal int8 GEMV kernel)` | 0.0 MB | no runs: Metal out of memory at its first decode (below) | **GOINFER-ALONE** |

- **M2, Ollama:** its warm-up returned HTTP 500 again. Its log now says why:
  - `ggml_metal_synchronize: error: command buffer 0 failed with status 5`,
    `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`, at the first decode
    (`n_batch = 512`); then `llama-server terminated ... signal: killed`.
  - The by-day check (above) had it running with 1 GB of Metal headroom. At night it ran out of Metal memory twice.
  - **By the registered rule, the cell is GOINFER-ALONE with that reason.** Night 1's reading (29.0 tok/s, also alone)
    is stated beside it, not averaged in.
- **M3, Ollama:** its warm-up failed with `Remote end closed connection without response`, 12.2 s in.
  - **Its log stops mid-load, with no error of its own,** after placing 31/31 layers: 10,480.59 MiB on Metal,
    3,273.38 MiB CPU_REPACK and 578.89 MiB CPU, with mmap disabled. That is about 14.3 GB on a 16 GB Mac.
  - System swap over the step grew from 1,355.88 MB to 2,450.94 MB (+1,095 MB), past the registered +1,024 MB kill,
    while goinfer's arm grew 0.0 MB.
  - That is consistent with the harness's kill-watch, but the harness did not record it: a failed warm-up then wrote
    no swap figures. That gap is fixed since (nobara's re-registration, above in its record).
  - **By the registered rule** ("If Ollama cannot run the 26B without swapping, goinfer's arm is recorded alone"), the
    cell is GOINFER-ALONE. Ollama was not retried with other settings.
- **M3's spread, disclosed:** goinfer's three runs spread 6.5% ((20.2 - 18.9) / 19.8).
  - The 5% spread cap applies to a paired ratio, and there is none here, so the grade stands.
  - The mechanism is visible in the per-completion rates: run 1's first nine completions climb from 17.9 to 20.2 tok/s
    as the paged expert cache (`-moe-cache-experts`) warms, and the later runs hold 20.2.
  - The site cell carries the median, 20.2, and its runs as printed.
- **M1, Phi-3:** the code fix is built (`docs/tasks/task-metal-q4k-2026-10.md`). Its gates on the real file (G-Q2,
  G-Q3) come before an M1 cell, then the cell by an amendment here.

