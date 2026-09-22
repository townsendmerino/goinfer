# C0 — CI timing measurement (2026-09-22)

Per-job and per-step durations for the last 28 completed `ci.yml` runs on `main` (2 of the last 30
were still in progress at pull time, excluded), via `gh run view --json jobs`. Raw data pulled
2026-09-22, box `apple-m1pro` (the puller, not the runner — all runs are GitHub-hosted).

## Per-job total wall time

| job | median | p90 | min | max |
|---|---|---|---|---|
| `root-darwin` | 1339s (22.3m) | 1494s | 1123s | 1566s |
| `root` | 1170s (19.5m) | 1242s | 70s* | 1268s |
| `gpu-darwin` | 206s (3.4m) | 237s | 156s | 260s |
| `gpu` | 108s | 120s | 78s | 148s |
| `metal-darwin` | 62s | 78s | 40s | 83s |
| `cuda` | 54s | 60s | 39s | 62s |
| `readme-smoke` | 43s | 64s | 27s | 73s |

*min=70s on `root` is an early-failure run (gofmt/build/vet caught something before the test step
ever started) — not representative of a normal pass; excluded from the interpretation below.

**`root-darwin` is the actual wall-time long pole of the workflow, not `root`** — it's the single
largest job by both median and p90, ~170s ahead of `root`. Both run the identical
`go test -race -timeout 25m -tags goinfer_testhooks ./...` (ci.yml:153, :332); `root-darwin` skips
straight to it after build+vet (no staticcheck/apidiff/citation-lint stretch), so its whole ~1290s
is the test step itself.

## `root` job internals

| step | median | p90 |
|---|---|---|
| Set up job | 1s | 2s |
| checkout | 6s | 8s |
| setup-go | 11s | 14s |
| gofmt | 1s | 1s |
| build | 6s | 7s |
| vet | 6s | 7s |
| vet (realckpt) | 4s | 4s |
| **staticcheck** | **24s** | **25s** |
| apidiff | 6s | 7s |
| populate module cache (cross-repo citations) | 3s | 4s |
| queue citation lint | 2s | 2s |
| book link lint | 0s | 0s |
| cleanliness guard | 0s | 1s |
| **test (`go test -race ./...`)** | **1051s** | **1115s** |
| sampler gates exist (zero-match guard) | 1s | 1s |
| sampler gates (no -race) | 48s | 49s |

**Pre-test lint stretch (job start → test step start): median 71s, p90 76s.**

## `gpu` job internals (relevant to C5's apt-get sub-item)

| step | median | p90 |
|---|---|---|
| setup-go | 14s | 19s |
| **install GPU build deps (apt-get)** | **19s** | **32s** |
| build -tags gpu | 10s | 11s |
| staticcheck -tags gpu | 13s | 14s |
| test -tags gpu | 29s | 32s |

## Queue wait (run created → job started)

| job | median | p90 | max |
|---|---|---|---|
| `root` | 3s | 7s | 8s |
| `root-darwin` | 8s | 25s | 149s |
| `gpu-darwin` | 8s | 13s | 61s |
| `metal-darwin` | 9s | 52s | 136s |
| `gpu` | 3s | 4s | 7s |
| `cuda` | 3s | 4s | 39s |

Trivially small everywhere except a few outliers (max 149s on `root-darwin` once, 136s once on
`metal-darwin`) — **not a systemic bottleneck.** macOS runner scarcity is not what's costing time
here.

## setup-go cache-restore step

| job | median | p90 |
|---|---|---|
| `root` | 8s | 13s |
| `root-darwin` | 16s | 31s |
| `gpu-darwin` | 18s | 35s |
| `metal-darwin` | 14s | 32s |
| `gpu` | 12s | 17s |
| `cuda` | 8s | 12s |

Already fast everywhere (8-18s median) — no sign of the Linear article's "cache restore loses to a
cold install" pattern. Nothing to gain here.

## The three C0 selector questions, answered

1. **Is `root`'s test step the wall-time long pole?** Not quite — `root-darwin`'s test step (1290s
   median, isolated: total job 1339s minus ~49s build+vet+setup) is larger than `root`'s (1051s),
   and since both run as separate parallel jobs, `root-darwin` is the true critical-path long pole.
   → **C1 still clears its own pre-registered kill-line** (71s pre-test stretch > the 60s floor),
   so it proceeds — but **C4 is the item that actually shortens the workflow's critical path**,
   since it targets the job that's genuinely longest.
2. **Is macOS queue wait the long pole?** No — trivially small (§ above). Rules out the
   runner-scarcity concern C3 would have addressed; irrelevant to the items in scope here anyway
   since C3 is out of scope for this pass.
3. **Is setup-go's cache restore slower than a cold download+build?** No — already fast (8-18s)
   everywhere. Nothing to gain; this C5 sub-item does not clear its own ≥15s-improvement bar and is
   not pursued.

## What this measurement selects, for this pass (C3 explicitly out of scope)

- **C2** (concurrency block): no measurement needed, ship as scoped.
- **C1** (split `root` into `lint`+`test`): proceeds — 71s pre-test stretch clears the 60s kill-line.
- **C1′** (shard `-race`, conditional on `./decoder` dominating): see the follow-up local timing
  check in this session's own record before deciding.
- **C4** (stop double-running the full suite on darwin): proceeds — `root-darwin` is the actual
  critical-path long pole, and `-short` is already a proven, existing pattern in this exact
  `ci.yml` (`gpu`/`cuda` jobs already use it, `root`/`root-darwin` do not) with 89 real
  `testing.Short()` call sites in the tree to exploit.
- **C5, staticcheck sub-item**: proceeds — 24s median, and it's `go run @pinned-version` (a compile
  from source on every run) rather than a downloaded release binary; real room to cut, applies to
  3 jobs (`root`, `gpu`, `cuda` all run it).
- **C5, apt-get sub-item**: proceeds to investigation — 19s median (up to 32s), clears the ≥15s
  bar; whether it's build-time or run-time-only needs checking against `cogentcore/webgpu`'s
  actual requirements before deciding to cache vs remove it.
- **C5, setup-go cache sub-item**: killed by this measurement — already fast, no room to gain.
- **C6** (resilience against a stalled full-history fetch): killed — checkout step shows no
  meaningful tail (p90 8s vs median 6s on `root`); the stall scenario this item guards against
  isn't showing up in the data.
