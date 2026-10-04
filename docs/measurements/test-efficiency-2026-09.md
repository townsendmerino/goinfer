# Verification cost census (TE0), both machines — 2026-09-28

TE0 of [`task-test-efficiency-2026-09.md`](../tasks/task-test-efficiency-2026-09.md): where verification wall-clock goes,
from records and session transcripts already on disk. Nothing was run or timed to produce it. Re-run it weekly while the
campaign is open, and at its close; §5 has the commands.

## 1. Sources

| source | what it sees | Mac | nobara |
|---|---|---|---|
| [`census.py`](test-efficiency-2026-09/census.py) | archived records: served-harness JSONs, run-log START/END spans with idle-gate lines, `go test` package lines, parity churn in git | `census-2026-09-28.txt` | `census-nobara-2026-09-28.txt` (with `~/goinfer-bench`) |
| [`transcript_miner.py`](test-efficiency-2026-09/transcript_miner.py) | every Bash call a Claude Code session ran, with its duration, including daytime runs that never reached a log | `transcripts-macbook-2026-09-28.txt` | `transcripts-nobara-2026-09-28.txt` |

The census is repo-wide, since the records are committed. Its nobara run adds `~/goinfer-bench`, which the Mac cannot
see. The miner is per machine, and each covers the 30 days from 2026-08-29: Mac 206 transcripts and 43,914 calls,
nobara 85 transcripts and 18,730 calls, subagents included.

## 2. Session call time by class, 30 days

Each is the sum of Bash call durations; parallel calls overlap. Day means 07–22 local.

| class | Mac calls | Mac h (day / night) | nobara calls | nobara h (day / night) | both h |
|---|---:|---|---:|---|---:|
| **wait/poll** | 1418 | 66.5 (55.7 / 10.8) | 987 | 47.8 (39.8 / 8.0) | **114.3** |
| test suite | 3921 | 23.7 (20.0 / 3.7) | 1422 | 13.0 (10.6 / 2.4) | 36.7 |
| other | 26239 | 11.6 (9.5 / 2.2) | 11665 | 14.4 (12.1 / 2.2) | 26.0 |
| in-test measurement | 669 | 8.7 (7.5 / 1.2) | 438 | 5.2 (4.7 / 0.6) | 13.9 |
| build/lint | 5085 | 6.1 (5.2 / 0.9) | 2028 | 2.6 (2.2 / 0.4) | 8.7 |
| git/gh | 5180 | 2.0 (1.8 / 0.2) | 1371 | 1.5 (1.3 / 0.2) | 3.5 |
| parity/refresh | 313 | 1.4 (1.2 / 0.2) | 137 | 1.6 (1.5 / 0.1) | 3.0 |
| detached launch | 269 | 0.8 (0.7 / 0.1) | 348 | 2.1 (1.8 / 0.3) | 2.9 |
| gate (`cmd/gate`, `tools/*`) | 179 | 2.1 (2.0 / 0.1) | 47 | 0.3 (0.3 / 0.1) | 2.4 |
| served gate (the call itself) | 613 | 1.2 (1.1 / 0.1) | 256 | 0.3 (0.3 / 0.0) | 1.5 |
| **all** | | 124.3 | | 88.9 | **213.1** |

- **Excluded:** 29 spans totalling 375 h (Mac 7 / 145.6 h, nobara 22 / 229.7 h). These are foreground calls whose
  result came more than 610 s later, which the tool's 600 s cap means were blocked rather than running, and
  background tasks that ended "stopped" (servers and watches left up until someone stopped them). Left in, they would
  be larger than everything else put together.
- **Served gates and long measurements mostly run detached,** so their own time is in the census's spans and cells
  (§3), not in this table. Their *waiting* is here, as wait/poll.

## 3. Archived timed runs (census)

- **Served harness:** 93 outputs, 1,057 cells, 24.3 h of cell wall (Mac 7.5 h, nobara 16.8 h). Decode was being timed
  for **45%** of it. The rest is server start, load, warm-up, prompt prefill and teardown.
- **Per-cell overhead is largest for goinfer.** It is 44.6 s per cell on the 7B, 2× Ollama's and 4.7× llama.cpp's.
- **Idle gate:** 14 logged timed runs with gate lines spent 138 of 311 min waiting (**44%**). On nobara alone, the two
  logged gated runs of 2026-09-27 spent 29 of 73 min waiting (40%), at night, at the default 1.0 cap.
- **Measurement gates written as `go test`** are the longest single runs: `TestPrefillGateReference` up to 168 min,
  `TestMoEExpertMajor_endToEnd` 113, `TestA3MoEExclusionIsMeasured` 108.

## 4. Findings

1. **The largest block of session time is waiting: 114 h, 54% of all session call time, 95 h of it by day.** These are
   sessions blocked in `until … sleep` / `gh run list` / `ssh … kill -0` loops on CI, remote runs and detached
   benchmarks. It is not machine work. It is the owner's wait, because a session in a loop is a session not doing
   anything else. The campaign's machine-time items shorten what is waited on; the waiting itself is a separate lever:
   - by day, queue long work on `night.py` instead of watching it (the Run-budget rule);
   - rely on completion notifications instead of poll loops;
   - do not re-run a check that already passed at this commit.
2. **Unit-test suites are next: 36.7 h, 30.6 h of it by day.** Whole-suite re-runs are visible: the Mac's `decoder`
   suite 3–6 times on single days, at 5–10 min each. That is TE7's target, affected-only selection plus the test cache.
   TE7(b) found the cache does work for `decoder`: an unchanged re-run replays in 2 s instead of 310 s.
3. **In-test measurement gates: 13.9 h.** TE8 covers resume, cached references and right-sizing.
4. **Parity bookkeeping is small in hours** (3.0 h), matching §1.4 of the task: its cost is interruptions, not wall.
   TE10 stays conditional.

## 5. Limits, and how to re-run

- **Coverage gaps.** A session on another tool, or a command typed in a terminal, is invisible to the miner. Work a
  Mac session ran on nobara over ssh is counted on the Mac: 5.4 h of work, and nobara's own transcripts do not
  double-count it.
- **Classification is regex-based.** "other" (26 h) is mostly short analysis: `grep`, `sed`, `python3` heredocs.
- **Re-run:**
  - Mac: `python3 docs/measurements/test-efficiency-2026-09/transcript_miner.py`, and `census.py` with no arguments.
  - nobara: from a temporary worktree of `origin/main`, as done here, so another session's tree is not touched:

    ```sh
    git worktree add --detach /tmp/te0-wt origin/main
    python3 …/census.py ~/goinfer-bench
    python3 …/transcript_miner.py --projects ~/.claude/projects/*goinfer* ~/.claude/projects/*aikit*
    git worktree remove /tmp/te0-wt
    ```

## 6. The re-count at the campaign's close (TE11), 2026-10-04 — nobara only

**What was re-run:** `transcript_miner.py --since 2026-09-29` on nobara's transcripts (92 files, 4,165 Bash calls, to 2026-10-04 05:57 PDT: 5.25 days), and `census.py` on the committed records. The Mac's re-count is the Mac session's (its transcripts are not readable here).
**What it can and cannot say:** the before is 30 days before the campaign (2026-08-29 to 09-28), the after is the **5.25 days since** it, on one machine, with a different workload (the Route C confidence track: a lot of tests, graded runs and waiting on CI). It is an indication, not a verdict, and nothing in it is attributable to a single item.

| class | before h/day (30 days) | after h/day (5.25 days) | change |
|---|---:|---:|---:|
| wait/poll | 1.59 | 0.90 | -44% |
| test suite | 0.43 | 0.55 | +28% |
| other | 0.48 | 0.27 | -44% |
| in-test measurement | 0.17 | 0.06 | -67% |
| build/lint | 0.09 | 0.10 | +11% |
| git/gh | 0.05 | 0.04 | -25% |
| parity/refresh | 0.05 | 0.01 | -75% |
| detached launch | 0.07 | 0.07 | -6% |
| gate (`cmd/gate`) | 0.01 | 0.02 | +141% |
| served gate (the call itself) | 0.01 | 0.21 | up (the served D14 and D13 harness calls this week; the baseline's were detached) |
| **all** | **2.96** | **2.22** | **-25%** |

- **Waiting is down 44% (1.59 to 0.90 h a day) and carries nearly all of the drop in total call time.** The work that is not waiting is **flat: 1.37 h a day before, 1.33 after.** The campaign's items that shorten timed runs (the instant idle gate, the two-arm rule, the resumable references) act mostly on night runs and on the Mac, which this re-count does not see; and the waiting that fell is the lever §4.1 named separately (queue instead of watch, notification instead of poll), which the run-budget rule and the night queue address.
- **The test-suite class did not fall on nobara (+28%).** `gate quick` exists and the `decoder` cache works, but this week's work (the Clef pipeline, `internal/serveapp`, whose package takes about 135 s) ran those suites repeatedly. Whether `gate quick` was used for them was not measured.
- **Served harness (census, committed records):** cells 1,057 to 1,310; decode being timed 45% to 48% of cell wall overall, and **70% in the 253 cells added since the baseline** (1.84 of 2.64 h; the cell mix differs, so this is indicative); goinfer's 7B per-cell overhead 44.6 s to 39.7 s. The idle gate: the committed run logs' gate waiting reads **41%** of 597 gated minutes (246), and **38% in the 286 minutes added since the baseline** (108), against 44% before. The Linux gate is unchanged (the load gate; the instant gate is the default on darwin only, TE1 attempt 5), so no large change was expected there.
- **Not re-measured:** the per-gate projections (a served gate from about 66 minutes to about 7 to 12 with the whole stack, section 3 of the task doc). No gate has run on the full stack yet, so those bands are neither confirmed nor failed here.
- **A census artifact to avoid:** `census.py ~/goinfer-bench` counts each log once per worktree copy under that directory (231 spans became 1,800), so its START/END section is only valid without the extra root. The figures above use the repo-only run.

