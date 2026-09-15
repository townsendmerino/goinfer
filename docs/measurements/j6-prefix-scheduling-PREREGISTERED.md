# PRE-REGISTERED — J6 prefix-aware admission scheduling

**Written 2026-09-15 BEFORE the overnight sweep was launched. Not edited after any result is
seen.** The dated write-up goes in `j6-prefix-scheduling-2026-09-15.md` (or the actual run date);
this file stays as written.

## The question

`docs/tasks/task-work-queue-2026-09.md` J6: does preferring the waiter whose prompt extends a
session already warm in the LRU (instead of J1's strict FIFO), bounded by a starvation guard,
measurably improve throughput on a mixed agent-loop workload?

## Arms

| arm | what it is |
|---|---|
| **old_fifo** | `main` at the point this sweep was launched (`15a5df56`) — `admission.release()` always grants `waiters.Front()`, strict FIFO |
| **new_prefix_aware** | this session's working-tree change — `release(resident)` scores waiters by longest resident-session prefix match, ties broken by FIFO, bounded by `admissionStarvationBound` (default 3, **not swept tonight** — see "Reduced scope" below) |

Both binaries built from the SAME `cmd/serve`, CPU backend, `int8int8` quant, same model
(`qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`) — the only difference between them is the admission
scheduling change.

## Workload

`scripts/bench_j6_scheduling.py`: M independent simulated agent-loop conversations running
CONCURRENTLY, each its own sequential turn loop (an agent loop, not a burst of unrelated
one-shots), sharing one ~600-token system-prompt prefix (forked from `scripts/w4_transcript_base.json`'s
12 real tool schemas, rendered as prose) with divergent per-conversation bug reports as the
"divergent tails." `-kv-sessions` is set BELOW the conversation count on purpose, so there is
genuine LRU eviction pressure — without it the two policies cannot differ at all, since nothing
ever gets evicted.

## Pre-registered decision rule

- **Pass band: 1.3–2.0×** aggregate tokens/s, `new_prefix_aware` over `old_fifo`, paired mean
  across interleaved reps.
- **Kill: < 1.15×.** Below this the mechanism does not ship this pass; the negative gets written
  up at full value in the task doc's J6 section, not dropped.
- **Ambiguous zone (1.15×–1.3×): parked**, not rounded to a verdict either way — re-measure with
  more reps or a larger M before deciding.
- **Void a cell** if: either arm's spread across its reps exceeds 10%; the box was not idle at the
  start of the sweep (`loadavg` recorded in the header, must be checked, not just logged); a
  conversation reports an error (a scheduling bug should never manifest as a request failure, so
  any error voids that rep for investigation, not silent exclusion).

## Reduced scope for tonight's run (user asked to stop for the night)

- **One quant only** (`int8int8`, whichever this box's model is already in) — the task doc asks
  for both; the second is a fast follow once this run's harness is proven sound, not before.
- **`admissionStarvationBound` NOT swept** — left at its shipped default (3) for every cell. The
  task doc's own words ("N configurable and measured at the same time, since the guard is what
  caps the win") make this a real gap, but wiring a runtime-configurable bound into a built binary
  was judged more scope than tonight's session should add; the PRIMARY pass/kill question (does
  the mechanism win at all, at a reasonable default) is what tonight's compute buys. If the primary
  signal clears the band, the bound sweep is the natural next-session follow-up before final ship.
- **CPU only**, no Metal cell — an admission-ordering question is backend-independent in principle;
  confirming that is itself a possible follow-up, not assumed here.

## What already ran (harness smoke test, NOT a measurement)

A 1-rep, 4-conversation, 2-turn smoke run confirmed the harness starts both binaries, drives real
concurrent generations against a real checkpoint with zero request errors, and writes valid JSON —
that run's numbers are explicitly NOT cited as a result (single rep, box not verified idle,
`loadavg` was 3.6 — see the methodology page's own "verified-idle box" rule). It exists only to
prove the instrument works before spending overnight compute on it.

## The real run

Launched detached (`launchctl submit`, per this Mac's own convention in `CLAUDE.md`'s "Long-running
work" section — self-removing on exit so it cannot restart itself the way that section's own
cautionary incident describes), sized to a bounded rep count (not open-ended) so it completes by
morning rather than running indefinitely. Raw output:
`docs/measurements/j6-prefix-scheduling-raw-2026-09-15.json`. Command line and exact parameters are
recorded in that file's own `header` block (machine state, loadavg, commit, full argv).
