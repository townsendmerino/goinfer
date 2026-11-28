# The int8 head default (#1179) on nobara: the fit check, and the re-anchor cells it moves (2026-11-20)

**Why.** Francis's decision of 2026-11-20 (PR #1179, from issue #1160) gives models of 1e9 parameters and up the int8
output head by default. The re-anchor's goinfer arms ran the int4 head. #1179 asks for the server's own fit plan at the
new head, and for the moved cells to be re-measured once it is on main.

## 1. Fit check (2026-11-20 12:43–12:44 PST, untimed, under the timing lock held exclusive)

**How it was run:**
- `serve-cuda` built from #1179's branch at `beece1df6`, sha256 `203b9c6a…`.
- Driver 595.112.07. The desktop's `kwin_wayland` held 17 MiB of the card. The page cache was evicted first.
- Each load: `-backend cuda -quant int4`, the checkpoint from `~/models`. #1179's default head (int8 at this size) and an
  explicit `-embed-int4=true` on the same binary, at `-ctx 32768` and at serve's default context.
- Read from the server's banner. Every load was `cuda-resident (int4)`. Logs are in [`fit/`](fit/).

| model | context | int8 head (#1179 default) | int4 head (`-embed-int4=true`) | change |
|---|---|---|---|---|
| 1.5B | `-ctx 32768` | 4 conversations × 32,768 | 4 × 32,768 | none |
| 1.5B | serve default | a 65,536-position pool, 16,384 guaranteed each, one up to 32,768 | the same | none |
| 7B | `-ctx 32768` | 1 conversation × 32,768 (4 asked for) | 1 × 32,768 | none |
| 7B | serve default | **a 50,944-position pool, 12,736 guaranteed each** | a 55,040-position pool, 13,760 guaranteed each | **−4,096 positions (−7.4%)** |

- **The host fit line's weights** rise by 0.19 GiB on the 1.5B (0.93 → 1.12 GiB) and by 0.45 GiB on the 7B (3.99 → 4.44
  GiB). That line prices host memory (a budget of about 41 GiB), not the card.
- **On the card, only the 7B's default pool moves.** The 7B's resident context is VRAM-bound there: "trimmed from 13824
  at the build, so the 4 requested KV slots fit beside the weights". 4,096 f16 positions is #1179's arithmetic
  (about 4,170) to within its rounding.
- **The 1.5B's context is capped by the model (32,768),** not by VRAM, with either head.

## 2. The moved cells: registered 2026-11-20 13:00 PST, before any data

- **The cells:** #1179's list, against the re-anchor's own registration ([`../README.md`](../README.md)).
  - a: CUDA greedy, the 1.5B and 7B at 128 / 2048 / 3900, and at 8000 with `BENCH_CTX=8192`.
  - d: CUDA TTFT, the 1.5B, K = 512 / 3900.
  - e: x86 CPU greedy, the 1.5B and 7B at 128.
- **Unchanged from the re-anchor:** the bands, pairs, token and decode-path gates, the void rule with its zero swap bar,
  the harness and the grader (`../grade.py`, which now also reads this run's per-model file names).
- **The engines:** goinfer, Ollama 0.32.5 and llama.cpp `a37cc46`, all run again in the same session. A peer reading from
  the 19th/20th paired with a goinfer reading from today would be a cross-session ratio.
- **The build:** main at #1179's merge, pinned in a detached worktree. Build it with
  [`run-int8head.sh`](run-int8head.sh) `build <rev>`, which writes the rev and both sha256s.
  - goinfer runs at its new default: no `-embed-int4` flag, so the int8 head at these sizes.
  - Every goinfer fit check must report `head table: int8`, or the job stops.
- **One model per step,** so the page cache is evicted and the load settled before each model's cells (Francis's
  option 1). The harness has no per-cell hook.
- **Run:** by day, under the timing lock (the harness holds it). Expected about 65 min:
  - a: about 20 min;
  - d: about 2 min;
  - e: about 31 min, the 7B CPU cells taking 15 of it;
  - fit checks, settles and evictions: about 12 min.
- **Use:** the graded cells replace the 1.5B and 7B rows of these cells in `docs/benchmarks.md`. The int4-head rows stay,
  marked as before the default changed.
- **Not run until #1179 is on main.** Its CI was red at 12:40 PST on five root jobs.

### Results (run 2026-11-20 15:38–16:29 PST, graded 16:30)

**The run:**
- One hold of the timing lock, exclusive, covering the build, the CUDA part and the CPU part. 51 min, against the
  65 estimated.
- Built from main at `48d15271c` (#1179's merge). `serve-cuda` sha256 `3c287093…`, `serve-cpu` `a456a7a6…`.
- Driver 595.112.07, Ollama 0.32.5, llama.cpp `a37cc46`.
- Every goinfer fit check and every goinfer serve log reports `head table: int8`. Every GPU cell is `cuda-resident (int4)`.
- No cell voided: no swap void, no token-gate void, no refused start. The page cache was evicted and the load settled
  before every step.
- Graded with `../grade.py`: [`results/grade.jsonl`](results/grade.jsonl). Raw results, step logs and provenance are
  in `results/`.

**Against Ollama** (median r; the int4-head reading from 2026-11-19/20 in brackets; the outcome is decided by every pair):

| cell | int8 head (the new default) | int4 head |
|---|---|---|
| a 1.5B @ 128 / 2048 / 3900 / 8000 | **AHEAD** 1.415 / 1.31 / 1.253 / 1.178 | 1.51 / 1.382 / 1.322 / 1.228 |
| a 7B @ 128 / 2048 / 3900 | **AHEAD** 1.137 / 1.07 / 1.042 | 1.187 / 1.113 / 1.082 |
| a 7B @ 8000 | **LEVEL** 0.981 | LEVEL 1.019 |
| d TTFT 1.5B, K = 512 / 3900 | **AMBIGUOUS-HIGH** 4.691 (Ollama's spread 13.7%) / **AMBIGUOUS-HIGH** 1.435 (Ollama's 6.9%; every pair ≥ 1.371) | 4.589 / 1.43 |
| e CPU 1.5B | **BEHIND** 0.927 (22.2 vs 23.9 tok/s) | LEVEL 1.012 (24.4 vs 24.1) |
| e CPU 7B | **LEVEL** 0.991 (5.9 vs 6.0) | AHEAD 1.039 (6.2 vs 6.0) |

**The CUDA cell the site's `claims.json` cites, in detail** (tok/s: each engine's three runs, their medians and the
pairs against Ollama, as [`results/grade.jsonl`](results/grade.jsonl) records them; added 2026-11-27, no figure changed):

| cell | goinfer runs | median | Ollama runs | median | pairs r | median r | outcome |
|---|---|---:|---|---:|---|---:|---|
| cuda 1.5B @128 greedy, int8 head | 259.8 / 259.5 / 259.8 | 259.8 | 183.5 / 183.4 / 183.6 | 183.5 | 1.416 / 1.415 / 1.415 | 1.415 | AHEAD |

**Against llama.cpp `a37cc46`** (beside):
- **The 1.5B:** AHEAD 1.118 / 1.067 at 128 / 2048, AMBIGUOUS-HIGH 1.032 at 3900, **BEHIND 0.954** at 8000.
- **The 7B:** AHEAD 1.044 at 128, LEVEL 0.993 at 2048, **BEHIND** 0.969 at 3900 and 0.911 at 8000.
- **The CPU:** BEHIND 0.813 on the 1.5B and 0.929 on the 7B.

**Reading:**
- **The int8 head costs decode speed, about as #1179's byte counts predict.** goinfer's own rates against its int4-head
  rates from the 19th/20th, the same binary family on the same box:
  - CUDA 1.5B: 259.8 against 277.8 tok/s at 128 (−6.5%), and 191.8 against 200.5 at 8000.
  - CUDA 7B: 82.1 against 85.9 (−4.4%), and 65.5 against 68.1 at 8000.
  - CPU: 22.2 against 24.4 (−9%) on the 1.5B, and 5.9 against 6.2 on the 7B.
  - These are cross-session figures, not graded ratios.
- **Two labels move:**
  - CPU 1.5B, LEVEL → BEHIND.
  - CPU 7B, AHEAD → LEVEL.
  - The CUDA 7B at 8000 stays LEVEL, now under 1.0.
- **TTFT does not move.** The head is applied to the last position only.
- **The 0.5B, gemma3-1b, phi3-mini and 26B rows** are untouched by #1179 and keep their 2026-11-19/20 readings.
