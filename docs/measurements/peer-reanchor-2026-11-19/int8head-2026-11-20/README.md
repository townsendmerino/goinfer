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
