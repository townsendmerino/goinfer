# x86 CPU peer cells at v0.25.0 on nobara-pc (registered 2026-11-22)

**Status: REGISTERED 2026-11-22 13:43 PST (commit `21f272426`; first typed as 13:58, corrected from the commit time), before any timed run.** By day, under nobara-pc's exclusive timing lock.
Results go under *Results*. A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Why.** The three x86 CPU cells below were last read at `e29da447` ([`peer-reanchor-2026-11-19`](../peer-reanchor-2026-11-19/README.md)
cells e and v), before v0.25.0's CPU changes (#1121, #1129 and the aikit bump #1170) merged. Owner's go: Francis,
2026-11-22 ("continue goinfer ... and performance work"), relayed by the coordinator.

## Inherited unchanged

Everything in [`peer-reanchor-2026-11-19/README.md`](../peer-reanchor-2026-11-19/README.md) not named under *What differs*:
- the ratio r = goinfer ÷ peer for decode; the bands AHEAD (every pair > 1.03), LEVEL (every pair in 0.97–1.03), BEHIND
  (every pair < 0.97), AMBIGUOUS-HIGH or -LOW otherwise, every pair deciding, never the mean; the 5% spread cap;
- 3 pairs per cell (8 completions × 64 tokens a run), depth 128, greedy, ctx 4096;
- the token gate, the decode-path gate, servers restarted between engines;
- the void rule of 2026-11-08 (`noise-registry.md` §8: a cell starts only with load ≤ 8.0 and `MemAvailable` ≥ its fit need,
  and is VOID if system swap grows at all while it runs), `BENCH_SWAP_VOID_STOP=3`, the harness's idle gate
  (`BENCH_MAX_LOADAVG=1.0`), the script's settle under 0.8 and its page-cache eviction before each step;
- Ollama v0.32.5 at `~/ollama-0325` (CPU-forced, `num_gpu: 0`) with the same tags; llama.cpp `a37cc46` beside on the 0.5B;
- the grader, [`../peer-reanchor-2026-11-19/grade.py`](../peer-reanchor-2026-11-19/grade.py), run on this campaign's files.

## Cells

| cell | what | engines | last graded (e29da447) |
|---|---|---|---|
| e | x86 CPU greedy, 0.5B @ 128, `-quant int4` | goinfer, Ollama, llama.cpp | BEHIND 0.867 (49.9 vs 57.5 tok/s) |
| v1 | x86 CPU greedy, phi3-mini at its default (`q4k`) @ 128 | goinfer, Ollama | BEHIND 0.763 (8.2 vs 10.7) |
| v2 | x86 CPU greedy, gpt-oss 20B at its default (`int4`) @ 128 | goinfer, Ollama | BEHIND 0.923 (9.9 vs 10.7) |

The 1.5B and 7B are not in scope: their CPU rows were re-measured at the int8-head default on 2026-11-20 (`48d15271c`).

## What differs from 2026-11-19

- **goinfer is the v0.25.0 release asset** `goinfer-serve-linux-amd64` (`v0.25.0 (d0da204c4b7a)`, backends cpu and cuda,
  aikit v1.78.0), sha256 `e93cdf59…` as the release's `checksums.txt` gives it, run with `-backend cpu`. 11-19 used a local
  build of `e29da447`.
- **The harness is `bench_peer.py` at the tag**, from a clean detached worktree at `d0da204c`. Its only change since
  `e29da447` is the fit-banner regex accepting `GiB` as well as `GB`.
- **Cell e holds the 0.5B only.** [`run.sh`](run.sh) is the 11-19 script's cpu part with these pins; the script checks the
  binary's hash and version, the worktree's HEAD, the tag and its clean state.

## Fit checks (2026-11-22 13:42 PST, `FIT_ONLY=1`)

All three load: 0.5B `cpu (int4)`, phi3-mini `cpu (q4k)`, gpt-oss 20B `cpu (int4)`. System swap already held 519 MiB from
earlier work; the void rule reads growth during a cell, not the level.

## Decision rule

- Each graded cell's outcome is reported against its 11-19 label. A cell whose label moves replaces its row in
  `docs/benchmarks.md` (the TL;DR's x86 CPU decode row and the summary sentence that quotes it), citing this record. A cell
  whose label does not move leaves the page's label as it is.
- A VOID cell stays VOID, is reported, and its old row stands.
- No move is attributed to #1121, #1129 or #1170: this is a cross-build, cross-session comparison (TE5(a)); each lever has
  its own record.
- No early stop, and no re-run of a valid cell.

**Tier:** day. Testing tier 2: the numbers are published, and no smaller tier measures a peer ratio.

## Cost

From 11-19's cells (0.5B arms about 40 s each plus about 100 s of idle gate; phi3-mini about 5 min an arm; v2's two arms
791 s): fit checks 1 min, e about 8 min, v1 about 12 min, v2 about 14 min, evictions and settles about 5 min.
**Expected (corrected from the clock, as typed it said 14:00 to 14:45): start 13:43 PST, finish about 14:30 PST.** The run holds the lock under `timeout 90m`.

## Results

### Run, 2026-11-22 13:43–14:15 PST, graded 14:16 PST

- **The run:** `run.sh` rc=0 in 32 min (registration: about 45), under the exclusive timing lock with `timeout 90m`. Fit checks passed
  again inside the script (13:42–13:43). The e-cpu step started at 13:45:21 after a 100 s settle, v-cpu at 13:51:52.
- **The void rule:** no void cell. Every arm started at a 1-minute load of 0.64–0.94 (cap 8.0; the harness's own gate is 1.0)
  with 61,181–61,293 MB available. Swap growth was 0.0 MB on every arm. The level stayed at 519 MiB, held from before the run.
- **Records:** [`results/cpu/`](results/cpu/) holds `e-cpu.json`, `v-cpu.json`, their step logs, the fit logs and
  `provenance.txt`. [`results/run.log`](results/run.log) is the script's log. The grade is
  [`results/grade-2026-11-22.jsonl`](results/grade-2026-11-22.jsonl), written by `../peer-reanchor-2026-11-19/grade.py results/cpu`.
- **Provenance:** nobara-pc, AMD Ryzen 7 3700X (8 cores, 16 threads), 62 GiB, kernel 7.3.0-202.nobara.fc44; goinfer
  `v0.25.0 (d0da204c4b7a)` release asset (aikit v1.78.0, go1.27.4); Ollama 0.32.5; llama.cpp `0.7.0-dev (build 988, commit
  a37cc46)`; greedy (`temperature 0`, seed 1), 3 runs × 8 completions × 64 tokens per arm, the median of the 3 runs reported.
  Checkpoints from local NVMe: `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`,
  `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`, `~/models/gpt-oss-20b-MXFP4.gguf` (none resolves under the archive;
  the script checks). Thermal: no CPU temperature reading was taken; a desktop on mains power, each arm started idle (load ≤ 0.94),
  and each arm's 3 runs agree within 0.7%.
- **Decode paths:** goinfer decoded `cpu (int4)` on the 0.5B and gpt-oss 20B and `cpu (q4k)` on phi3-mini, as the fit checks
  resolved. Ollama ran Q4_K_M on the 0.5B and phi3-mini and MXFP4 on gpt-oss. Every pair is 4-bit against 4-bit.

**Against Ollama** (median r; every pair decides; the 11-19 reading at `e29da447` is in brackets):

| cell | now (v0.25.0) | pairs | median tok/s, goinfer vs Ollama | was |
|---|---|---|---|---|
| e 0.5B | **LEVEL 1.023** (moved from BEHIND) | 1.023 / 1.019 / 1.028 | 58.3 vs 56.9 | BEHIND 0.867 (49.9 vs 57.5) |
| v1 phi3-mini (`q4k`) | **BEHIND 0.833** | 0.833 / 0.833 / 0.831 | 8.9 vs 10.7 | BEHIND 0.763 (8.2 vs 10.7) |
| v2 gpt-oss 20B (`int4`) | **BEHIND 0.960** | 0.960 / 0.960 / 0.960 | 10.3 vs 10.7 | BEHIND 0.923 (9.9 vs 10.7) |

- **Runs** (tok/s, run 1 / 2 / 3): 0.5B goinfer 58.3 / 58.1 / 58.5, Ollama 56.9 / 57.0 / 56.9, llama.cpp 64.8 / 64.8 / 64.4;
  phi3-mini goinfer 8.9 / 8.9 / 8.9, Ollama 10.7 / 10.7 / 10.7; gpt-oss 20B goinfer 10.3 / 10.3 / 10.3, Ollama 10.7 / 10.7 / 10.7.
- **Spreads:** goinfer 0.7% / 0.1% / 0.1%, Ollama 0.2% / 0.1% / 0.1%, all under the 5% cap.
- **Against llama.cpp `a37cc46`** (beside; no claim names it): the 0.5B is **BEHIND 0.900** (58.3 vs 64.8; pairs 0.900 / 0.896 /
  0.909), against 0.77 on 11-19.
- **What moves on the page** (the decision rule): only the 0.5B's label. It replaces the 0.5B's entry in `docs/benchmarks.md`'s
  x86 CPU decode row, and the summary sentence's list of where goinfer is behind. phi3-mini and gpt-oss 20B keep their labels, so
  the page keeps their 11-19 entries.
- **Attribution:** none. Ollama's own rates are within 1% of 11-19's on all three models, but this is still a cross-build,
  cross-session comparison (TE5(a)). The levers' own records hold their same-session A/Bs.

### Review follow-ups, 2026-11-22 14:54 PST (after the PR merged)

From the review of PR #1229, read on 2026-11-22 on nobara-pc.

- **What ran, exactly.** Each model's arms ran one after another in one session, each on a freshly started server:
  goinfer's 3 runs, then Ollama's 3, then llama.cpp's on the 0.5B. Pair i is goinfer's run i against the peer's run i. The order was
  not A B B A; that is the 11-19 protocol, unchanged.
- **gpt-oss 20B's decode rate: the ratio stands, and both arms' absolute rates read about 5% high.**
  - The harness's rate is `(completion_tokens − 1) / (t_last − t_first)` over stream events. It assumes the first event carries one
    token.
  - gpt-oss opens its reply with harmony's channel header, `<|channel|>analysis<|message|>`, three tokens that neither engine streams
    as an event. One request each, the same payload as the harness's (`sse_probe.py`, `ollama_probe.py`; outputs in
    [`results/sse-probe/`](results/sse-probe/)):
    - goinfer streamed 55 token events for 64 completion tokens: 39 `reasoning_content` and 16 `content`, its first event the fourth
      token.
    - Ollama streamed 61 events for `eval_count` 64, all `thinking`, its first event also the fourth token.
  - On both arms, then, the timing window holds 60 tokens and the numerator counts 63. Corrected by 60/63, the rates are goinfer
    9.79 and Ollama 10.20 tok/s; the pairs and the median stay at 0.960, BEHIND. Corrected runs (tok/s, run 1 / 2 / 3): goinfer
    9.8 / 9.8 / 9.8, Ollama 10.2 / 10.2 / 10.2.
  - The record's "4.0 tokens per chunk" for goinfer's arm counts only `content` deltas as chunks. It is not four tokens an event:
    most events carry one token of `reasoning_content`. The 11-19 v2 cell (0.923) has the same structure and the same cancellation.
  - Not changed here: `bench_peer.py`'s rate. Its definition is the gate instrument's, so a change goes to Francis first. One option is
    to count the tokens before the first event per completion, from the engine's own count when it reports one. Until then, a cell
    whose two arms' first events lag by different token counts does not cancel, and its record has to say so.
- **The fit checks now match each cell's expected decode path** (`cpu (int4)`, `cpu (q4k)`, `cpu (int4)`), not any `cpu` prefix.
  All three fit logs had resolved the expected path.
- **Noted, not pursued:** goinfer's gpt-oss arm spent 258 s of its runs' wall in prefill and request, about 10.8 s a request for a
  130-token prompt, against Ollama's 23 s. The decode rate excludes it.
