# Peer re-anchor on nobara-pc: CUDA and x86 CPU against Ollama (registered 2026-11-19)

**Status: REGISTERED 2026-11-19 07:45 PST, before any timed run.** Two night jobs on nobara's runner, the night of
2026-11-19. Results go under *Results*. A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Why.** Most rows of `docs/benchmarks.md`'s TL;DR and §B11 come from builds before KS-A2, KS-B2 and KS-O1
([`ks-a2-2026-10`](../ks-a2-2026-10/README.md), [`ks-b2-2026-10`](../ks-b2-2026-10/README.md), and KS-O1 from
the [kernel survey](../kernel-survey-2026-10/README.md)), and nobara's NVIDIA driver moved from 595.91.07 to **595.112.07**.
`CLAUDE.md` says a driver change needs a deliberate re-anchor of the CUDA rows, not a carry-forward. Owner's go: Francis,
2026-11-19 03:25 PST (nobara CUDA and CPU now; the Mac later, by day, under its own registration).

## Inherited unchanged

These come from [`peer-sweep-2026-09-29.md`](../peer-sweep-2026-09-29.md) and are not restated there:
- the ratio r: goinfer ÷ peer for decode, peer ÷ goinfer for TTFT;
- the bands: AHEAD (every pair > 1.03), LEVEL (every pair in 0.97–1.03), BEHIND (every pair < 0.97), AMBIGUOUS-HIGH or
  -LOW otherwise, every pair deciding, never the mean;
- the 5% run-to-run spread cap;
- 3 pairs per decode cell (8 completions × 64 tokens a run) and 6 per TTFT cell;
- the harness's proportional token gate and the decode-path gate (a GPU cell decoded on the CPU is VOID);
- servers restarted between engines, interleaved by the harness;
- the depths and contexts (128, 2048 and 3900 at context 4096; 8000 at `BENCH_CTX=8192`; the 26B at `BENCH_CTX=2048`);
- Ollama v0.32.5 at `~/ollama-0325`, the same install and tags (`q05`, `q15`, `q7b`, `g31b`, `p3m-local`, `m26q4km`, `g20`).

Cells v1 and v2 inherit [`peer-vetted-2026-10-07-nobara-pc.md`](../peer-vetted-2026-10-07-nobara-pc.md) cells 1a and 1b:
the same weights both sides, goinfer against Ollama with `num_gpu: 0`, no llama.cpp arm.

## Cells

| cell | what | engines | last graded (the rows this replaces) |
|---|---|---|---|
| a | CUDA greedy, 0.5B / 1.5B / 7B @ 128, 2048, 3900, 8000 | goinfer, Ollama, llama.cpp | 09-29 cell a: 12 AHEAD |
| b | CUDA greedy, gemma3-1b and phi3-mini (`q4k`) @ 128, 3900 | the same | 09-29 cell b; §B11 greedy |
| f | CUDA sampled @ 128, 0.5B / gemma3-1b / phi3-mini × the shared preset (0.8 / 40 / 0.9), no truncation (1.0 / 1 / 0), and 09-29's top-p 0.95 | the same | §B11 tables 2 and 3; 09-29 cell f (top-p 0.95, owed a re-grade since the 09-30 fix) |
| d | CUDA TTFT, 1.5B, K = 512, 3900 | goinfer, Ollama | 09-29 cell d |
| c | Gemma 4 26B-A4B on 8 GB, ctx 2048 (goinfer's int4 `.giw` with `-moe-cache-experts` against Q4_K_M: an architecture comparison) | goinfer, Ollama, llama.cpp | 09-29 cell c |
| e | x86 CPU greedy, 0.5B / 1.5B / 7B @ 128 | goinfer, Ollama, llama.cpp | 09-29 cell e (0.91 / 1.01 / 1.04) |
| v1 | x86 CPU greedy, phi3-mini at its default (`q4k`) @ 128 | goinfer, Ollama | 10-07 cell 1a (0.763) |
| v2 | x86 CPU greedy, gpt-oss 20B at its default @ 128 | goinfer, Ollama | 10-07 cell 1b (0.696) |

Phi-3 at depth 3900 on CUDA is cell b's second depth.

## What differs from 09-29, fixed now

- **Driver 595.112.07** (the run stops on any other), kernel 7.3.0-202.nobara.fc44.
- **llama.cpp is `0.7.0-dev (build 988, commit a37cc46)`,** rebuilt on 2026-11-11 for the Flint work
  ([`flint-f0-2026-10.md`](../flint-f0-2026-10.md)); 09-29 ran `427291b`. Its rows are reported beside, as before, and
  no claim names it.
- **The harness is `bench_peer.py` at the pinned commit,** with what §B11 added: the same seed and context to every arm,
  every sampling field sent explicitly (`min_p 0` to the peers), and a run refused when the arms' quant classes differ.
- **The void rule is the 2026-11-08 one** (`noise-registry.md` §8): a cell starts only with load ≤ 8.0 and
  `MemAvailable` ≥ its fit need, and is VOID if system swap grows at all while it runs (the default zero bar; no
  swap-in bar is set). A void cell re-runs in place; 3 voids in a row stop that step (`BENCH_SWAP_VOID_STOP=3`), and
  its cells are reported VOID. The harness's own idle gate (`BENCH_MAX_LOADAVG=1.0`) applies on top.
- **Cell f takes §B11's configs** so the TL;DR's CUDA short-context row can be replaced like for like.
- **gpt-oss 20B's default quant now resolves to `int4` on the CPU** (the fit check's decode path, below); 10-07
  recorded `int4mix`. Both are 4-bit against MXFP4, so the cell is matched by class.
- **TTFT's goinfer server runs at serve's own default context,** 32768 at this build (the fit banner); the harness
  passes no `-ctx`, as on 09-29.
- **The grader** is [`grade.py`](grade.py), the 09-29 grader with its bands unchanged. It reads this sweep's files, and
  it voids an arm the void rule voided or that has no runs. Run on 09-29's own files, it reproduces all 50 of 09-29's
  graded rows exactly: outcomes, medians and pairs.

## Pins

- **goinfer:** `e29da447` (origin/main, 2026-11-19 07:31 PST), from a clean detached worktree at
  `~/goinfer-bench/peer-reanchor-2026-11-19/wt`. aikit v1.76.0 (go.mod), from the module cache.
- **Binaries:** `serve-cuda-e29da447`, sha256 `703b01ed…`, and `serve-cpu-e29da447`, sha256 `27b18a32…`. The script checks both
  hashes, the worktree's HEAD and its clean state.
- **Checkpoints:** all from `~/models` on NVMe. The script refuses a path that resolves under `/srv/models` or `/Volumes/`.
  None needed a `models-pull`.

## Fit checks (2026-11-19 07:37–07:40 PST)

Run with the binaries and flags the harness passes (`run-nobara.sh`, `FIT_ONLY=1`); the desktop's `kwin_wayland` held
17 MiB of the card:
- **CUDA, all `cuda-resident`:**
  - 0.5B, 1.5B and 7B at ctx 4096 and at 8192: 4 KV slots each.
  - gemma3-1b: 4 KV slots.
  - phi3-mini (`q4k`): 2 KV slots (4 asked for).
  - TTFT 1.5B with `-require-backend`.
  - The 26B: 1 conversation × 2048, its C′ expert cache capped at 41 slots a layer (4.2 GB) by free VRAM.
- **CPU:** 0.5B, 1.5B and 7B `cpu (int4)`, phi3-mini `cpu (q4k)`, gpt-oss 20B `cpu (int4)`.

Each job re-runs its own fit checks before any cell and exits 75 on a refusal (re-queued until 04:00).

## Decision rule

Nothing is decided by this run. Each graded cell replaces the row it re-measures, citing this commit, with its outcome label:
- the TL;DR's CUDA short and deep rows, the controls, x86 CPU decode, prefill and the 26B;
- §B11's CUDA rows;
- the site's `claims.json` entries for them.

A VOID cell stays VOID and is reported; its old row stays, marked as not re-anchored at 595.112.07. A loss stays a loss.
A cell whose label moved since its last graded reading is reported as moved. No move is attributed to KS-A2, KS-B2 or
KS-O1: a cross-build, cross-session comparison cannot attribute one (TE5(a); each lever has its own record).
No early stop, and no re-run of a valid cell.

- **Tier:** night, case (b): the two jobs come to about 2.75 h. Testing tier 2, because the numbers are published, and no
  smaller tier measures a peer ratio.

## Cost

From 09-29's step walls (3054 + 518 + 501 + 427 + 524 + 55 s) and 10-07's 1a/1b, scaled to these cells:
- **`reanchor-cuda` (est 90 min):** a about 24 min, b 8, f 16, d 2, c 12, fit checks 3, idle gates the rest.
- **`reanchor-cpu` (est 75 min):** e about 34 min (the 7B is 15 of them), v1 + v2 about 18, fit checks 1, idle gates
  the rest.
- **Timeline:** not before 22:00, CUDA first. Expected finish about 23:30 PST for the CUDA job and 00:50 for the CPU job.

Results go to `~/goinfer-bench/peer-reanchor-2026-11-19/results/{cuda,cpu}/` and the night logs. The next day's session
copies them here and grades them: `python3 grade.py results/cuda results/cpu`.

## Results

### Night 1, 2026-11-19 22:00 PST: most steps refused, nothing graded

The harness's preflight refused most steps, so nothing was measured for them. The preflight takes one 1-minute load
sample on Linux and refuses above `BENCH_MAX_LOADAVG=1.0`, without waiting. The 09-29 script had waited for idle before
each step; this one did not.
- **CUDA:** a-dense, a-depth8000, b-controls, f-sampled and c-26b were REFUSED at load 1.95, left by the fit checks'
  server starts.
- **d-prefill ran** and wrote its results. It started at load 1.95, which is inside the void rule's cap of 8.0, and
  `bench_peer_prefill.py` has no load cap of its own. It is kept, with that start load in the record.
- **CPU:** e-cpu ran all 9 cells, each started under 1.0 by the harness's own per-cell gate. v-cpu was REFUSED at 7.34,
  the load e-cpu's own servers left.
- A refused step has no runs, so nothing was averaged or graded.

### Amendment, 2026-11-20 00:32 PST, before the re-run's data

- **The fix:** `run-nobara.sh` now waits before each step until the 1-minute load average is under 0.8. It waits up to
  10 minutes, logging each reading, and fails the step if the load never settles.
  - 0.8 leaves room for the harness's own sample a moment later.
  - `BENCH_MAX_LOADAVG` is unchanged at 1.0.
- **What re-runs:** only the steps that produced no results: CUDA a-dense, a-depth8000, b-controls, f-sampled, c-26b; CPU
  v-cpu. They run with the script's new step argument, with the same binaries, harness, cells and bands.
- **What stands:** d-prefill and e-cpu, as run.
