# Prompt: does "speculate when alone, batch under load" pay on CUDA? The premise measurement (nobara)

For a Claude Code session on `nobara-pc` (RTX 2070 SUPER 8 GB), repo `~/mycode/goinfer`. `git pull` first.

## Why

- **The candidate.** The Mac session pre-registered an MC4 candidate on 2026-09-28 (03296ed7,
  `docs/tasks/task-concurrency-2026-09.md`, MC3 follow-ons, "MC4 candidate: speculate when alone, batch under load").
  With `-spec ngram` a model would keep MC3's concurrency: a generation drafts only while it is alone, and joins
  batched decode at a round boundary when others arrive.
  - Today `-spec` forces one generation at a time on every resident (`internal/serveapp/openai.go`,
    `setConcurrency`), so under load it gives up MC3's batching.
- **What Metal showed.** Its condition was met there: a lone request is 2.08× (copy) / 1.07× (chat) plain on the
  1.5B, and under load spec is one stream against batching's four.
- **Why CUDA is its own question.** CUDA could benefit too: MC3 CUDA shipped (7a44a58e: 4 clients 1.380× on the 1.5B,
  1.826× on the 7B), and its spec verify is cheap (θ 0.155–0.251; 2.14× on a verbatim-copy prompt,
  `docs/measurements/spec-decode-lane-2026-09-21.md`). But CUDA's S and L have never been measured against MC3, and
  Metal's numbers do not transfer.
- **Losslessness is already settled on CUDA.** Your item-30 check (`cuda/spec_twoturn_test.go`) matched plain decode
  across two turns, 14/14 with the fixes. This prompt is about speed only.

**This session measures CUDA's S and L under the rule that decided Metal. It builds nothing.** The rule (§3) is
written here and in the task doc before any timing.

## Before anything

- **This is a timed served measurement: it goes on nobara's night queue, and is not run by day** (root `CLAUDE.md`,
  "Run budget").
  - Tonight's nobara queue is already ~8 h 15 min and fits only if started by 22:15. Queue this behind it
    (`--priority 90`), and it carries to the next night if it does not fit.
  - By day, only the checks in §1 and a one-cell smoke (`BENCH_RUNS=1`, labelled exploratory, never quoted).
- **Check nobody else is timing:** `uptime`, `nvidia-smi`, `python3 scripts/timing_lock.py status`.
- **Models only from `~/models`**, never `/srv/models`. Record the NVIDIA driver version (anchor: 595.91.07).

## 1. Two checks by day, before writing the run

1. **Is the CUDA resident actually used at serve's defaults?**
   - On Metal, the 2026-09-28 default `-embed-int4` (9ccf7fb1) makes the resident decline (`weight kind "int4" is not
     int8`), and serve silently decodes on the CPU (`docs/quantization.md`, "Known issue").
   - Start `serve` on the 1.5B with `-backend cuda` and read the banner's `decode path:` line, at the default and with
     `-embed-int4=false`.
   - **If the default declines on CUDA too, that is a finding for the owner** (record it in `docs/quantization.md`'s
     known issue, which says CUDA has not been checked). Then run every arm below with `-embed-int4=false`, and say so.
   - Either way, a cell whose server log does not show a CUDA-resident decode path is void.
2. **The harness speaks CUDA:**
   - `scripts/bench_spec_copy.py` hard-codes `w7.GoinferServer("metal", …)`. Add a `--backend` flag, defaulting to
     `metal` so the Mac's records still reproduce, and pass `cuda`.
   - `scripts/bench_w7_plain.py` already takes `--backend`.
   - Commit the flag with a one-line test or a dry run showing the argv.

## 2. The run: a copy of the 2026-09-27 Metal design, on CUDA

- **One binary, two arms, a fresh server per cell:**
  - `serve-cuda` built once at the commit you pin: `CGO_ENABLED=0 go -C cuda build -tags cuda -o <dir>/serve-cuda-<hash> ./cmd/serve`;
  - *batch*: serve's defaults. MC3 is on, so a lone request runs production's own path;
  - *spec*: the same plus `-spec ngram` (its banner reads "one generation at a time").
- **Workloads, greedy:**
  - *copy*: `bench_spec_copy.py --backend cuda`. Each request hands back a ~1000-token section of `decoder/model.go`
    at `cc5f8c2c` verbatim, 256 tokens, 2 requests per client;
  - *chat*: W7 as graded for MC3, `bench_w7_plain.py --backend cuda --engines goinfer`, 6 turns × 128 tokens,
    `--fixed-nonce`.
- **Cells:** 4 clients, then 1, per workload; batch/spec × 3 pairs in the order batch spec spec batch batch spec;
  idle-gated per cell (the harness defaults at night).
- **Models:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M, graded. Qwen2.5-7B-Instruct Q4_K_M copy, at 1 and 4 clients, reported.
- **Pinned and unattended:** a `docs/measurements/spec-vs-batching-cuda-<date>/run.sh`, with the binary and harness
  under `~/goinfer-bench/`. Estimate its wall time from the MC3 CUDA and spec-lane records before queuing. **Keep it
  under 3 h; split it if not.**

## 3. The decision rule (the 2026-09-27 Metal rule, unchanged; the 1.5B, per workload)

- **S** = the 1-client aggregate, spec ÷ batch; **L** = the 4-client aggregate, spec ÷ batch. Median of the 3 pairs,
  paired per pair. The 4-client p99 request / turn latency is reported.
- **S ≤ 1.05 on both workloads:** the candidate does not apply on CUDA. Record it, and the candidate stays Metal-only.
- **S > 1.05 on a workload, with L < 1.0:** the candidate applies on CUDA. Its build is then graded on nobara too,
  with the Metal-registered gates re-run there (identity every turn; alone copy ≥ 1.25× and chat ≥ 0.97×; 4 clients
  ≥ 0.97× of batch; staggered aggregate ≥ 1.00× with p99 turn ≤ 1.10×).
- **L ≥ 1.0 on a workload:** spec beats batching even at 4 clients on CUDA. That is MC4's own spec trigger, "spec
  inside a batch", and it goes to the owner rather than into this candidate.
- **Identity** (every reply's hash, spec against batch, every turn) is reported. After item 30 it should be equal
  throughout. A difference is a finding to chase before any speed number is quoted.

## 4. Record

- `docs/measurements/spec-vs-batching-cuda-<date>.md`: setup, the S/L table like Metal's §2, the decision, the
  identity table, and logs and JSON beside it (anything over 1 MB as `.gz`).
- In `docs/tasks/task-concurrency-2026-09.md`, the MC4 candidate's **CUDA** bullet: replace "It gets its own grading
  on nobara" with the result and a link.
- `gofmt`, `go vet`, the pinned staticcheck and the citation lint before pushing. Check CI after.
