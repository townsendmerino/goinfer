# MM6 night-profile replay (G-NP-4) on the Mac — 2026-10-27

G-NP-4 is registered in [`docs/tasks/task-subsystems-2026-10.md`](../tasks/task-subsystems-2026-10.md), "MM6's night
profile" (on #312's branch). It prices night jobs on a profile recorded from the box's own night readings: the 25th
percentile over the last 14 nights. Does that profile, built from the Mac's readings, refuse the 8 counted refusals of
[`mm6-replay-2026-10-27.md`](mm6-replay-2026-10-27.md)?

- **Setup.** Run by mac-vk01 on the MacBook at #312's head `21d126ed` (branch `mm6-night-profile`), with nobara-mm01's
  brief.
- **Not a timed run, no model loaded.** The timing lock was free, and the Mac's night runner was idle with nothing queued.

## Result: VOID. Step 1 built no profile.

```
$ python3 scripts/night.py profile --backfill
backfilled 0 night reading(s) into ~/goinfer-logs/night/memory.jsonl
no night profile: only 0 night readings over 0 nights in the last 14 (the night profile needs 5 over 3); night jobs are priced on the day profile
```

With no profile, step 2 has nothing to price against; it was not run. Nothing was written: `memory.jsonl` and
`profile-night.json` do not exist after the run.

### Why 0: `START_RE` needs `re.MULTILINE`

`backfill_readings` in `scripts/night.py` (#312's branch) finds a job's start time with:

```python
START_RE = re.compile(r"^=== START\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
```

It calls `START_RE.search(text)` on the whole log. Without `re.MULTILINE`, `^` matches only at the start of the text.
The runner writes `=== START` after the `=== night job`, `cmd`, `cwd`, `rev`, `est` and `by` lines, so no real log
matches. All 188 logs under `~/goinfer-logs/night/runs/` carry the line, and none is read. `scripts/test_night.py` has
no backfill test, so nothing caught it.

**The fix is the flag:** `re.compile(r"^=== START…", re.M)`. #312 is nobara-mm01's branch, so the fix is theirs. A test
that feeds `backfill_readings` a log with the runner's real header would hold it.

### With the fix, still VOID

Re-reading the logs with `re.M` and the same rules (a start between 22:00 and the 06:30 deadline; the first fit-guard
"currently has X GB" after it) finds 7 night readings in all:

| started | job | available |
|---|---|---|
| 2026-10-02 01:14 | `metal-int8-gates` | 10.9 GB |
| 2026-10-07 23:04 | `s6-e4b` | 7.5 GB |
| 2026-10-08 22:21 | `metal-q4k-gq` | 7.4 GB |
| 2026-10-12 23:18 | `m2-metal-fit-ctx-night` | 6.4 GB |
| 2026-10-23 22:01 | `ksc3-g3b-1p5b-int8` | 2.9 GB |
| 2026-10-23 22:15 | `ksc1-g4-phi3-q4k` | 4.6 GB |
| 2026-10-23 22:17 | `ksc1-g3-phi3-gq2` | 6.9 GB |

The 14-night window ending 2026-10-27 starts on the night of 10-14. Only 10-23's three readings fall inside it: 3 over
1 night, against the 5 over 3 the profile needs. G-NP-4 would be VOID with the fix too.

**The brief's expected in-window readings included day starts.** It listed 2.9, 3.9, 4.6, 6.4, 6.6, 6.9, 7.0 and
7.7 GB. Three of those came from jobs the queue started by day, and by the backfill's own rule they are not night
readings:
- `op1-g2r-metal`, 10-14 at 11:18 (3.9 and 6.6 GB);
- `op1-g2r-metal`, 10-15 at 13:10 (7.0 GB at its refused Metal load);
- `p29-one-slot-ab`, 10-17 at 19:23 (6.4 and 7.7 GB).

## What would make it answerable

The Mac has too few night readings with a fit-guard line in the last 14 nights. The backfill can only see refusals and
warnings, so the sample is small and biased low, as its docstring says. Two ways to get enough readings:
- the runner's own reading at each job start (what #312 records from now on), after 3 nights with 5 jobs between them;
- a reading taken at each night's start whether or not a job runs.

Either way, the profile is not ready for the Mac yet.
