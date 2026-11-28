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

### Night 2, 2026-11-20 00:34–02:22 PST, graded 04:20 PST

**What ran:**
- `reanchor-cuda-2`: a-dense, a-depth8000, b-controls and f-sampled finished with rc=0, 41 min in all with no voids.
- c-26b: the goinfer and Ollama arms are valid (Ollama's re-ran once after a swap void). The llama.cpp arm hit 3 swap
  voids in a row (+109.5, +35.3, +5.4 MB), and `BENCH_SWAP_VOID_STOP=3` stopped the step.
- `reanchor-cpu-2` (v-cpu) ran from 01:33:42 until night.py's 50-minute timeout killed it at 02:22. Swap voids re-ran the
  cells in place 7 times: goinfer phi3-mini once, Ollama phi3-mini twice, goinfer gpt-oss twice, each listed in its record.
  phi3-mini finished both arms. gpt-oss has goinfer's arm only, so v2 is VOID for this night.

**Records:**
- [`results/grade-2026-11-20.jsonl`](results/grade-2026-11-20.jsonl): `grade.py results/cuda results/cpu`.
- `results/{cuda,cpu}/`: the results, the step logs, night 1's refused attempts and the provenance.
- `results/reanchor-*.log`: both nights' job logs.
- `results/reanchor-*-2.mem.tsv.gz`: night 2's per-second memory samples.
- **Common to every cell:** pinned build `e29da447`, driver 595.112.07, a clean tree, checkpoints from `~/models`, and
  every goinfer GPU cell `cuda-resident`.

**Against Ollama** (median r; the outcome is decided by every pair; the last graded reading is in brackets):

| cell | now | was |
|---|---|---|
| a 0.5B @ 128 / 2048 / 3900 / 8000 | **AHEAD** 1.943 / 1.669 / 1.591 / **VOID** | 1.396 / 1.23 / 1.268 / 1.335 (09-29) |
| a 1.5B @ 128 / 2048 / 3900 / 8000 | **AHEAD** ×4: 1.51 / 1.382 / 1.322 / 1.228 | 1.474 / 1.342 / 1.288 / 1.196 |
| a 7B @ 128 / 2048 / 3900 / 8000 | **AHEAD** 1.187 / 1.113 / 1.082, **LEVEL** 1.019 at 8000 (moved from AHEAD) | 1.184 / 1.13 / 1.099 / 1.045 |
| b gemma3-1b @ 128 / 3900 | **AHEAD** 1.914 / 1.528 | 1.385 / 1.364 |
| b phi3-mini @ 128 / 3900 | **BEHIND** 0.909 / 0.793 | 0.895 / 0.718 |
| f greedy, 0.5B / gemma3-1b / phi3-mini | AHEAD 1.951 / AHEAD 1.911 / BEHIND 0.908 | §B11: 1.33 / 1.33 / 0.90 |
| f preset (0.8 / 40 / 0.9) | AHEAD 1.891 / AHEAD 1.835 / BEHIND 0.900 | §B11: 1.31 / 1.27 / 0.90 |
| f no truncation (1.0 / 1 / 0) | AHEAD 2.398 / AHEAD 2.363 / BEHIND 0.928 | §B11: 1.65 / 1.64 / 0.92 |
| f top-p 0.95, 0.5B / phi3-mini | AHEAD 2.892 / BEHIND 0.954 | 0.922 / 0.889 (09-29, before the 09-30 top-p fix) |
| d TTFT 1.5B, K = 512 / 3900 | **AMBIGUOUS-HIGH** 4.589 (Ollama's spread 17.2%) / **AMBIGUOUS-HIGH** 1.43 (Ollama's 5.8%; every pair ≥ 1.378) | 4.287 AMB-HIGH / 0.974 AMB-LOW |
| c 26B on 8 GB, ctx 2048 | **AHEAD** 2.266 (50.2 vs 22.2 tok/s) | 1.763 AMB-HIGH (39.3 vs 22.3) |
| e CPU 0.5B / 1.5B / 7B | **BEHIND** 0.867 / **LEVEL** 1.012 / **AHEAD** 1.039 | 0.905 / 1.01 / 1.043 |
| v1 CPU phi3-mini | **BEHIND** 0.763 (8.2 vs 10.7 tok/s) | 0.763 (10-07) |
| v2 CPU gpt-oss 20B | **VOID** (no Ollama arm; goinfer alone 9.9 tok/s, against 7.5 on 10-07) | 0.696 |

- **0.5B @ 8000 is VOID on the token gate:** one of goinfer's greedy replies ended at 8 of 64 tokens, while both peers
  returned 64. Greedy decoding is deterministic, so a re-run returns the same reply. It is not re-run here. That
  goinfer stops early where both peers do not is a correctness question, not a speed one, and is noted for follow-up.
  The raw ratio (1.381) is not graded.
- **d started at load 1.95** (night 1, above). That is inside the void rule's cap of 8.0.
- No move is attributed to a lever (see *Decision rule*).

**The cells the site's `claims.json` cites, in detail** (tok/s: each engine's three runs, their medians and the pairs
against Ollama, as [`results/grade-2026-11-20.jsonl`](results/grade-2026-11-20.jsonl) records them; added 2026-11-27,
no figure changed):

| cell | goinfer runs | median | Ollama runs | median | pairs r | median r | outcome |
|---|---|---:|---|---:|---|---:|---|
| cuda 0.5B @128 greedy | 515.0 / 515.3 / 514.7 | 515.0 | 265.0 / 264.6 / 264.9 | 264.9 | 1.943 / 1.947 / 1.943 | 1.943 | AHEAD |
| cuda phi3-mini (`q4k`) @128 greedy | 114.5 / 114.4 / 114.3 | 114.4 | 126.0 / 125.9 / 126.0 | 126.0 | 0.909 / 0.909 / 0.907 | 0.909 | BEHIND |

**Against llama.cpp `a37cc46`** (beside; no claim names it):
- **CUDA a, AHEAD in 9 cells:** 0.5B 1.363 / 1.211 / 1.15 at 128 / 2048 / 3900; 1.5B 1.198 / 1.122 / 1.086; 7B 1.091 /
  1.032 at 128 / 2048.
- **CUDA a, the rest:** 7B LEVEL 1.009 at 3900; 1.5B LEVEL 0.999 at 8000; 7B BEHIND 0.946 at 8000; 0.5B @ 8000 VOID.
- **gemma3-1b:** AHEAD 1.354 / 1.091.
- **phi3-mini:** BEHIND at every config, 0.778–0.911.
- **f, 0.5B:** AHEAD greedy (1.374) and preset (1.316). At top-p 0.95 and with no truncation it reads AMBIGUOUS-HIGH,
  capped by llama.cpp's own 5.1% and 5.6% spreads.
- **f, gemma3-1b:** AHEAD 1.322–1.878.
- **CPU:** 0.5B BEHIND 0.77, 1.5B BEHIND 0.892, 7B LEVEL 0.978.
- **The 26B:** not measured (3 swap voids).

**Not re-run, by the registration:** the valid cells, and c's goinfer and Ollama arms. Pairing c's llama.cpp arm with
the night's goinfer arm would be a cross-session ratio, so c's llama.cpp row stays "not measured".

### Why swap grows with ~61 GB available (read 2026-11-20 04:20 PST, from night 2's samples)

The void rule voids any swap growth in a cell, and on nobara a big load can grow swap even with ~61 GB available.

- **When it grows:** in 1–3 s bursts, each at a large file read, i.e. a model load. Two examples:
  - 01:27:21–01:27:36, +109.5 MB: the 26B llama.cpp arm's first void;
  - 01:33:44–01:33:57, +376 MB: goinfer phi3-mini's first v-cpu attempt (+394.1 MB over its cell).
- **What memory looked like:** the user slice's `memory.current` was 60–66 GB, on a 62.7 GiB box. Inactive page cache
  was about 49 GB of it, mostly checkpoints the sweep had read. `MemAvailable` stayed between 50.9 and 64.9 GB.
- **Whose pages moved:** in swap afterwards are idle desktop and system daemons: `dnf-app-center` 118 MB,
  `plasmashell` 70 MB, `plasma-keyboard` 57 MB, `kwin_wayland` 40 MB, `firewalld` 28 MB, and the night runner itself
  8 MB. No server process was swapped.
  - goinfer's own per-process swap guard (the server's `VmSwap` over 512 MB) never fired.
  - Over v-cpu's cells, swap-ins were 0.004–0.008 MB against 118 MB of swap-outs.
- **No leftover servers:** `MemAvailable` never dropped by a server's size between cells, and every serve log ends in its
  own shutdown.
- **The mechanism:** a load needs free pages while RAM is full of page cache. The kernel then reclaims cold anonymous
  pages of idle processes to zram as well as dropping file pages. MGLRU is on (`lru_gen` 0x0007), swappiness is 10, and
  zram is the priority-100 swap device. Nothing being timed reads those pages back. It is the same mechanism
  peer-vetted 2026-10-08 measured beside 12–14 GB loads.
- **The trigger is a full RAM, not the size of the load.** From 00:34 to 01:20 the slice stayed at 47–58 GB and swap at
  0 MB through every 0.5B–7B load. Swap first grew at 01:21, when the 26B's reads (17 GB for each peer file) took the
  slice past ~60 GB of the box's 67 GB. After that, even phi3-mini's 2.4 GB load voided.
- **Consequence:** under the zero bar, a cell on this box is likely to void whenever its load starts with the page cache
  near full.

### Amendment, 2026-11-20 04:24 PST, before the by-day re-run's data

- **What re-runs:** v2 only, as the new step `v2-g20`. Both arms (goinfer, then Ollama) are written to a fresh
  `v2-g20.json`, in one session. The night's goinfer arm in `v-cpu.json` is not paired with a day Ollama arm, because
  that would be a cross-session ratio. It stays in the record as goinfer alone.
- **The cells, bands, void rule, binaries and harness are unchanged.** `BENCH_SWAP_*` is unchanged and the zero bar
  applies.
- **c-26b is not re-run.** Its goinfer and Ollama arms are a valid same-session pair, and this registration does not
  re-run a valid cell.
- **The box's state at the start:** the 04:22 swap diagnostic evicted the page cache of `~/models` and Ollama's model
  store with `posix_fadvise(DONTNEED)` ([`evict_cache.py`](evict_cache.py); no root, no setting changed). That left the
  user slice at 6.6 GB and 55 GiB free.
  - The diagnostic's two untimed gpt-oss CPU loads grew swap by 0 MB: one with its file already cached, one after the
    eviction.
  - Nothing else about the box was changed.
- **Run:** by day, `timing_lock.py run` (exclusive) around `timeout 60m bash run-nobara.sh cpu v2-g20`, about 20 min
  expected (fit checks 1, settle 2, goinfer's gpt-oss cell about 8 and Ollama's about 5, plus gate waits).

### By-day v2 re-run, 2026-11-20 04:25–04:38 PST, graded 04:40 PST

- **The run:** `v2-g20` rc=0 in 791 s, holding the timing lock exclusive. Records:
  [`results/cpu/v2-g20.json`](results/cpu/v2-g20.json), with its log and `v2-run.log`; the grade is in
  `results/grade-2026-11-20.jsonl`.
- **Each arm's start:** goinfer at load 0.70 with 61,418 MB available; Ollama at 0.80 with 61,443 MB.
- **Swap:** 0.0 MB growth and 0.0 MB swap-ins on both arms, and no void.
- **v2, x86 CPU gpt-oss 20B @ 128 against Ollama:** **BEHIND 0.923** (9.9 vs 10.7 tok/s; pairs 0.923 / 0.923 / 0.923;
  spreads 0.1% / 0.0%), against 0.696 on 10-07. goinfer decoded `cpu (int4)`; Ollama's tag is MXFP4. Both are 4-bit, so
  the cell is matched by class.

**What this sweep leaves graded on nobara at 595.112.07:**
- every Ollama cell in the table above, with v2 replacing its night VOID;
- 0.5B @ 8000 VOID on goinfer's token gate;
- c's llama.cpp row not measured.

### Change, 2026-11-20: drop the checkpoint page cache before each step (Francis's decision, option 1)

- **Decision:** Francis, 2026-11-20 05:00 PST, relayed by the coordinator. Option 1 of the swap question above: drop the
  checkpoint page cache before cells that load a model. No guard and no `BENCH_SWAP_*` setting changes.
- **What the script does:** `run-nobara.sh` runs [`evict_cache.py`](evict_cache.py) at the start of every step, before
  the settle wait. It applies `posix_fadvise(DONTNEED)` over `~/models` and Ollama's model store, takes about 2.5 s, needs
  no root, and logs the line `=== evict:`.
- **Why:** this box's swap voids come only once RAM is full of page cache. Starting each step's loads from free memory
  removes that trigger. The diagnostic and the by-day v2 re-run both read 0 MB of growth after an eviction.
- **Limit:** it runs per step, not per cell, because the harness has no per-cell hook. A step whose own reads approach
  the trigger can still fill RAM within itself. The 26B step reads about 48 GB across its three engines; nothing else
  here reads more than about 30 GB.
- **The 26B's llama.cpp arm stays not measured** (Francis, the same day): the registration rules out re-running a valid
  goinfer/Ollama pair.
