# Where session time goes, and which package edges make the day loop slow — 2026-10-13

Asked by the owner on 2026-10-13: what share of the vscode-claude sessions' time is spent waiting on tests and gates, and
can goinfer be modularized to cut it. Nothing was timed to produce §1; it is read from the session transcripts on both
boxes. §2 reads one `gate quick` run's per-test JSON. §3 lists what was changed on the strength of it.

## 1. Session wall-clock, both boxes, 2026-09-14 to 2026-10-13

Instrument: [`session_wall.py`](test-wait-2026-10/session_wall.py). Unlike TE0's
[`transcript_miner.py`](test-efficiency-2026-09/transcript_miner.py), which sums Bash call durations, it gives every gap
between two transcript events to one bucket (the tool still running, the model, a background job the turn ended to wait
for, a scheduled wake-up, or the owner), so the buckets add up to wall-clock. Time waiting on the owner, waits over 3 h
(overnight) and model gaps over 30 min are left out. The per-day buckets it collected are committed beside it
(`mac-2026-10-13.json`: one Mac session, 2026-09-13 onward; `nobara-2026-10-13.json`: nine nobara sessions).

| bucket | hours | share |
|---|---:|---:|
| model thinking and writing | 161.6 | 35.2% |
| **waiting on local tests and gates** (`go test`, `cmd/gate`, the pre-push hook, polls on their logs) | 99.1 | **21.6%** |
| **waiting on CI after a push** (`gh run`, wake-ups scheduled to read it) | 76.4 | **16.6%** |
| other polling (mostly test logs and probes the classifier could not name) | 41.7 | 9.1% |
| benchmarks and measurements | 35.5 | 7.7% |
| reading, editing, git | 24.3 | 5.3% |
| nobara and night-queue bookkeeping | 10.4 | 2.3% |
| build, vet, lint | 10.2 | 2.2% |
| **active total** | **459** | |

- **Tests, gates and CI: 38% of active time; every kind of waiting together: about 60%.**
- **The trend was up.** Tests+gates+CI by week (Monday start): 09-14 33.5%, 09-21 36.7%, 09-28 30.4%, 10-05 47.7%,
  10-12 (one day) 49.7%.
- **On the Mac, writing code or docs was about 8% of active time** (model time before an Edit or Write, plus the edits).
- **Pushes:** GitHub lists more than 1,000 `ci` runs on `main` in the period, 30 to 60 a day; the last 60 took a median
  8.2 min (p90 9.1). The Mac session scheduled 185 wake-ups to read CI.
- **`gate quick`:** 107 runs in the Mac session at about 10 minutes each, about 18 h of the Mac's 51 h of local
  test-and-gate waiting.
- **Limits:** a background job that was running while the owner typed counts as his time, not as waiting. "Model" includes
  API latency. Parallel calls are counted once, under the highest-priority class.

## 2. One `gate quick` run, cell by cell (Mac, 2026-10-13 04:58 PDT, before the deferral change)

| cell | tests | test time | where it goes |
|---|---:|---:|---|
| `decoder` | 747 | 160 s | 13 tests are 143 s; the three statistical sampler files (`sampler_gumbel_test.go` 56 s, `sampler_topk_test.go` 48 s, `sampler_selection_test.go` 29 s) are 133 s; 695 tests are under 0.1 s |
| `decoder [checkpoint]` | 78 | 164 s | `prefillattnpool_test.go` 80 s |
| `metal` | 289 | 251 s | 30 tests are 221 s; the kernel files `mma` 63 s, `gemm_w8` 34 s, `gemm_w4` 22 s, `batchk` 15 s, `tower_attn` 11 s, `q4k_kernels` 11 s |
| `gpu` | 159 | 85 s | `gumbel_test.go` 24 s |

**The edge.** Go caches test results per package and re-runs a package when it or anything it imports changes. The
sampler code imports only the standard library, and the six Metal kernel files reference no `decoder` identifier, but
both sit in packages that a `decoder` edit invalidates (`metal` imports `decoder`). So any `decoder` edit re-runs about
290 s of tests that cannot observe it. Compiling a test binary after a one-line edit costs 4.3 s (`decoder`) and 4.8 s
(`metal`), so the cost is the slow tests, not the build.

## 3. What was changed

- **The parity deps hash reads Go files as tokens** (`decoder/parity_manifest_test.go:goCodeTokens`). A comment or
  formatting edit no longer re-stales a validated family; a code, build-constraint or literal edit still does
  (`TestGoCodeTokens_commentsOnly` checks both directions). The refresh script had recorded 307 earlier non-numeric
  refreshes.
- **Push and CI rule in `CLAUDE.md`:** push finished work, read CI at the next natural point, never wait on it.
- **The sampler moved to its own package, and the Metal kernel tests to a package that does not import `decoder`.**
  The before and after timings are in §4.
- **The KV re-architecture lands its manager as its own package** (`docs/tasks/task-kv-architecture-2026-10.md` §5.13).

## 4. After

(Filled in when the package splits are measured.)
