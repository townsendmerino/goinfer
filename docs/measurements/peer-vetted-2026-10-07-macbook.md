# Vetted-checkpoint peer cells, MacBook (pre-registered 2026-10-07)

**Status: PRE-REGISTERED 2026-10-07, before any timed run.** One cell is queued: the 26B on Metal (night job
`peer-vetted-mac`). The other two Mac cells are blocked, for the reasons below. Results and verdicts go below *Results*.
A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Question.** goinfer.dev/models has empty Metal speed cells for three vetted checkpoints. The Mac fills them on Metal
only. Every CPU cell is nobara's ([`peer-vetted-2026-10-07-nobara-pc.md`](peer-vetted-2026-10-07-nobara-pc.md)), and a
CPU 26B is forbidden on this Mac. Granite 4.0-H Tiny is out of scope.

## The three Mac cells

| cell | checkpoint (docs/capability-matrix.json) | state |
|---|---|---|
| M1 | Phi-3 mini 4k at `--quant q4k` on Metal | **BLOCKED: a code fix, the owner's call** (below) |
| M2 | gpt-oss 20B MXFP4 on Metal | **BLOCKED: disk** (below) |
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

**Cost.** Night only: one cell, two engines, about 10 minutes of cells and loads. Queued at 25 minutes.

## Results

*(not yet run)*
