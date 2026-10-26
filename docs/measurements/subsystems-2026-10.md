# Where session time goes, what the night queue refuses, and which edges force the long runs — 2026-10-26

The census behind [`docs/tasks/task-subsystems-2026-10.md`](../tasks/task-subsystems-2026-10.md) §1, §2, §6 and §7.
Nothing was timed to produce it: it is read from the Mac's Claude Code transcripts, the Mac's night-queue records, git
history on goinfer main `c4cfb4cb` and aikit main (root v1.75.0, `gpu/v0.35.0`), and GitHub's run records. The scripts
and their outputs are in [`subsystems-2026-10/`](subsystems-2026-10/).

## 1. Sources and instruments

| what | instrument | output |
|---|---|---|
| session wall clock, 2026-10-13 to 10-26 | [`test-wait-2026-10/session_wall.py`](test-wait-2026-10/session_wall.py) `collect` then `report --since 2026-10-13`: every gap between two transcript events goes to one bucket, so the buckets add up to wall clock (owner time, waits over 3 h and model stalls over 30 min left out) | `session-wall-mac-2026-10-26.json`, `session-wall-report-2026-10-26.txt` |
| which commands, 2026-10-12 to 10-26 | [`test-efficiency-2026-09/transcript_miner.py`](test-efficiency-2026-09/transcript_miner.py) and [`breakdown.py`](subsystems-2026-10/breakdown.py) over its parser: Bash call durations, which overlap | `miner-mac-2026-10-26.txt`, `breakdown-mac-2026-10-26.txt` |
| the Mac's night queue, 2026-10-06 to 10-26 | [`night_census.py`](subsystems-2026-10/night_census.py) over each night's `SUMMARY.md` and job logs | `night-census-mac-2026-10-26.txt` |
| CI runs | [`ci_anatomy.py`](subsystems-2026-10/ci_anatomy.py): the last 60 `ci` runs, and per-job start offsets for those over 20 min | `ci-anatomy-2026-10-26.txt` |
| git and source | [`repo_census.py`](subsystems-2026-10/repo_census.py), 30 days from 2026-09-26, with `--aikit` for aikit's tags, churn and import graph read at its `origin/main` | `repo-census-2026-10-26.txt` |

**Not read:** nobara's transcripts. At 09:27 PDT its timing lock was held by `night.py serve`, settling for a timed job,
and a read from the Mac would have run beside it. The both-box figures are the 2026-10-13 record's
([`test-wait-2026-10.md`](test-wait-2026-10.md)).

## 2. Session wall clock (the Mac, 25 transcripts)

| bucket | hours | share |
|---|---:|---:|
| model thinking and writing | 59.1 | 49.4% |
| waiting on CI after a push | 20.3 | 16.9% |
| waiting on local tests and gates | 13.1 | 10.9% |
| tool work (read, edit, git) | 7.1 | 5.9% |
| waiting on benchmarks and measurements | 7.0 | 5.8% |
| waiting on nobara and night jobs | 5.3 | 4.5% |
| build, vet, lint | 4.9 | 4.1% |
| other polling | 2.9 | 2.4% |
| **active total** | **120** | |

| week of (Monday) | active | tests + gates + CI | all waiting |
|---|---:|---:|---:|
| 2026-10-12 (from the 13th) | 43 h | 37.5% | 51.7% |
| 2026-10-19 | 76 h | 22.5% | 40.9% |

The 2026-10-13 record, both boxes, 2026-09-14 to 10-13: tests, gates and CI 38% of active time; every kind of waiting
about 60%.

## 3. Which commands (2026-10-12 to 10-26)

13,026 calls with a duration; 86.1 h of call time, 80.8 h with each session's overlapping calls merged.

| | calls | call time | note |
|---|---:|---:|---|
| `go test` | 1,641 | 13.9 h | 1,485 of them (11.9 h) passed `-count=1`; 935 a `-bench` flag; 128 `GOINFER_HEAVY_TESTS=1` |
| `scripts/touched.py` | 334 | 10.1 h | median 0.4 min, p90 4.8 min, p99 12.7 min; 29 runs at 5 min or more |
| watching PR checks (`gh pr checks --watch`) | 34 | 5.6 h | median 9.3 min, p90 14.2 min |
| wait/poll on nobara (remote runs, its night queue) | 47 | 15.4 h | background, beside other work: 5.3 h of wall clock in §2 |
| wait/poll on CI (`gh pr checks`, `gh run`) | 87 | 10.9 h | |
| timed served gates | 638 | 7.9 h | `run-als.sh` 2.0 h (one call), `run-g3.sh` 1.6 h (4), `run-S-served.sh` 1.3 h (9) |
| `staticcheck` | 215 | 2.5 h | |
| parity refreshes | 159 | 1.7 h | median 0.4 min, max 10.6 min |

The sections overlap (a compound command can match several); they are not meant to add up.

## 4. The Mac's night queue (2026-10-06 to 10-26)

131 job rows over 20 nights (retries included), 82 ok: 41 FAILED, 3 timed out, 3 re-queued by their own check, 2
stopped. By the first matching cause in each job's log (rows that share a log name are read from its last content):

| cause | rows |
|---|---:|
| the fit or swap guard refused the load ("was NOT loaded", "would page to swap rather than run") | 16 |
| voided (a cell started under heavy load, or a guard declined) | 9 |
| re-queued by its own check, or stopped | 5 |
| the other box or the timing lock was busy | 3 |
| timed out | 3 |
| a test failed | 4 |
| a harness or script error | 2 |
| other | 7 |

**33 of the 49 rows that were not ok are the box's state; 4 are a failing test.** Of 129 job logs, 14 carry a
refusal or a void (9 "was NOT loaded", 9 "would page to swap rather than run", 5 "VOID-refused", 1 with VOID pairs).
The longest jobs were timed gates: KV8a's M5 passes (1 h 51 min each), its M2b (1 h 47 min, failed), tack's Gemma 4
E4B and E2B Metal runs (timed out at 1 h 20 min and 55 min), KV1's speed gate (1 h 02 min), KV2's share gate (1 h 00 min).

## 5. CI (`ci.yml`, the last 60 completed runs, 2026-10-25 and 26)

Median 8.6 min, p90 12.3 min, max 44.7 min. The three slowest were queueing, not test time: in the 40.7-min run
(`40305583395`) every job, Linux ones included, started 32 minutes after the run was created; in the 43.5- and
44.7-min runs (`40313348871`, `40326560969`) everything else had finished when one Windows job (`root-windows-arm64`,
`root-windows`) started, 37 and 38 minutes in. Each job then ran its usual time. `gpu-darwin` is the longest job when
nothing queues (7–9 min).
The decoder tail shard ran 1,205 top-level tests in 249 s under `-race`; its slowest is
`TestPrefillChunked_bitIdentical` at 51 s.

## 6. Git and source (30 days from 2026-09-26)

- **Rate:** 3,260 non-merge commits on main; 113 to 164 a day from 2026-10-21 to 10-25.
- **Most-touched files** (non-merge commits): `docs/tasks/task-multimodal-support-2026-10.md` 315,
  `docs/citation-index.md` 241, `CHANGELOG.md` 229, `testdata/parity_manifest.json` 225,
  `docs/tasks/task-kernel-survey-2026-10.md` 170, `docs/queue-engineering.md` 162, `internal/serveapp/main.go` 116,
  `docs/README.md` 106, `metal/model.go` 99, `decoder/model.go` 86.
- **The parity manifest:** 43 families; the `core` set (11 files) is used by 42, `loaders` by 42, `quant` by 41.
  `decoder/model.go` (2,577 lines) took 86 commits, `decoder/registry.go` (2,996 lines, 40 family adapters) 20,
  `decoder/weights.go` 27, `decoder/gguf.go` 22.
- **aikit requirement changes in the root `go.mod`:** 37, each across every module. By subject (approximate):
  linalg and kernels 16, towers (vision, audio) 14, embed, formats and mmap 5, the GPU module 1, other 1.
- **Files importing `decoder`:**

  | module | test files | importing `decoder` | non-test importing | distinct `decoder.X` |
  |---|---:|---:|---:|---:|
  | `cuda` | 347 | 240 | 21 | 93 |
  | `metal` | 401 | 245 | 16 | 103 |
  | `gpu` | 172 | 87 | 7 | 35 |
  | `internal/serveapp` | 162 | 39 | 20 | 52 |

- **Memory-related code:** 34 decoder files (7,100 lines) touched by 141 commits, plus `internal/hostmem` (458 lines,
  17 commits), `internal/swapguard` (235, 2), `internal/fitcmd` (312, 5) and `internal/kvmgr` (1,344, 11). By group:
  residency 2,019 lines and 58 commits; the fit guard 1,022 and 28; prefill scratch 653 and 21; weight and resident
  bytes 427 and 25; the KV plan 529 and 23; fit plans and queries 656 and 11; the pagers 890 and 5; the probes (16
  files) 904 and 12.
- **aikit:** 35 root tags and 8 `gpu/*` tags. The moving packages and their edges:

  | aikit package | files, lines | tests | imports (aikit) | commits (touching `.go`) |
  |---|---|---:|---|---|
  | `vision` | 23, 6,323 | 30 | `embed`, `linalg` | 22 (22) |
  | `audio` | 8, 2,227 | 6 | `embed`, `linalg` | 12 (12) |
  | `gpu` (module) | 15, 5,125 | 38 | none | 52 (13) |
  | `gpu/qwencuda`, `gpu/qwenmetal` | 1 each, 484 and 527 | 1 and 2 | `gpu`, `vision` | 41 (2), 40 (1) |
  | `gpu/visioncuda`, `gpu/visionmetal` | 1 each, 330 and 398 | 1 and 3 | `gpu`, `vision` | 39 (0), 39 (0) |
  | `gpu/webgpu` | 1, 43 | 1 | none | 0 |
  | `gpu/anncuda`, `gpu/annmetal` (stay) | 1 each, 799 and 758 | 3 and 6 | `ann`, `gpu` | 39 (0), 40 (1) |
  | `gpu/enccuda`, `gpu/encmetal` (stay) | 1 each, 446 and 300 | 2 each | `encoder`, `gpu` | 41 (2) each |

  The `gpu/*` modules' commits are nearly all requirement bumps (`gpupins --fix` after each root tag). No RAG package
  (`chunk`, `bm25`, `ann`, `embed`, `encoder`, `fuse`, `hybrid`, `late`, `sparse`, `topk`, `mmap`, `linalg`) imports
  `vision`, `audio` or `gpu`.

## 7. How to re-run

On the Mac, from a worktree of `origin/main` (so another session's tree is not read):

```sh
D=docs/measurements/subsystems-2026-10
python3 docs/measurements/test-wait-2026-10/session_wall.py collect /tmp/sw.json ~/.claude/projects/*goinfer*/*.jsonl ~/.claude/projects/*aikit*/*.jsonl
python3 docs/measurements/test-wait-2026-10/session_wall.py report --since 2026-10-13 /tmp/sw.json
python3 docs/measurements/test-efficiency-2026-09/transcript_miner.py --since 2026-10-12 --top 40
python3 $D/breakdown.py --since 2026-10-12
python3 $D/night_census.py --since 2026-10-06
python3 $D/repo_census.py --since 2026-09-26 --aikit ~/tmcode/aikit
python3 $D/ci_anatomy.py --limit 60 --slow 20
```

`session_wall.py collect` takes the transcript paths as separate arguments; in zsh, run it under `bash -c` or the glob
reaches it as one string. On nobara, check its timing lock in the same command and stop if it is held (`CLAUDE.md`,
"Run budget").

<!-- doc-reviewed: 2026-10-26 -->
