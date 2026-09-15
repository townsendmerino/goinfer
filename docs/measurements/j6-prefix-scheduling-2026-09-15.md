# J6 prefix-aware admission scheduling — KILLED 2026-09-15

Pre-registered in `j6-prefix-scheduling-PREREGISTERED.md` (written before any run). Verdict below
is from the run analyzed after that file was written; nothing in the decision rule was changed
after seeing a result.

## Verdict: KILL

**Paired ratio (aggregate tok/s, `new_prefix_aware` / `old_fifo`): mean 1.024, median 1.022,
stdev 0.017, n=24.** Against the pre-registered band (pass ≥1.3×, kill <1.15×), this is a clean
kill — not ambiguous, not parked. The mechanism gives roughly a **2.2–2.4% throughput
improvement**, nowhere near the 1.3–2.0× the task doc's own hypothesis required to be worth the
added complexity (a starvation-guard counter on every waiter, a resident-session snapshot copied
on every generation's exit).

| | old_fifo | new_prefix_aware |
|---|---|---|
| mean agg tok/s | 39.01 | 39.93 |
| spread (24 reps) | 4.4% | 5.9% |

Both arms' spread is well inside the pre-registered 10% void bound — this run is clean, not a
near-miss obscured by noise. Zero request errors across all 48 cells (24 reps × 2 arms).

Full paired data: `j6-prefix-scheduling-raw-2026-09-15-run2.json`.

## The run this replaces, and why it doesn't count

A first attempt (`j6-prefix-scheduling-raw-2026-09-15-run1-VOID.json`,
`j6-prefix-scheduling-run1-VOID.log`) hit the pre-registered void rule: both arms' spread exceeded
10% (36.8% / 59.7%), with a repeating pattern of dramatic per-rep drops roughly every 7 reps. Not
discarded quietly — investigated, because a periodic pattern that regular is a signal, not noise.

**Root cause, found while investigating, not assumed:** `git` was silently broken system-wide
during that run — every invocation failed with "You have not agreed to the Xcode license
agreements," discovered only when a later `git commit` failed with the identical message. This
lines up with a macOS/Xcode update that must have run in the background during the first attempt
(matching processes were sitting idle in `ps`, and `softwareupdated` had recent activity), which is
the same class of interference the methodology page's own "verified-idle box" rule exists to catch
(`docs/benchmarks.md`: a 3020-token prefill once read 1587.1s against a real ~350s because an
abandoned prefill was still burning a core — same shape: a number that satisfies every OTHER
methodology bullet and is still wrong, because the box quietly wasn't idle).

**Harness improvement made in response, not a retroactive edit of the pre-registered rule itself:**
`scripts/bench_j6_scheduling.py` now records `loadavg` before and after every cell (not once at the
top) and writes the output JSON incrementally after each cell, so a future interference episode is
directly attributable in the data rather than requiring after-the-fact correlation, and a run that
gets interrupted still leaves usable partial data. The pre-registered pass/kill/void thresholds
themselves are unchanged from before either run.

The re-run (this document's own result) was launched once the background update activity had
subsided (`loadavg` 1-min reading dropped from 5.66 to 1.41 between the two attempts) and stayed
smooth throughout — no per-cell loadavg spike, no rep with a wall time outside a tight band
(46.9–50.5s across all 48 cells).

## Why the effect is real but small — a hypothesis, not re-litigated here

Both runs (the void one, on the reps that weren't corrupted, and the clean one) agree on the same
small positive effect, which is itself evidence it's real and not an artifact. The mechanism likely
has genuinely little to work with in this workload: with `-kv-sessions 4` and 8 concurrent
conversations, admission-queue contention exists, but by the time a second turn arrives from any
conversation, the single decode worker has usually already drained most of the queue back down —
CPU decode at ~39 tok/s on a 48-token turn is ~1.2s, and 8 conversations arriving within that
window is the exception more than the rule at this box's speed. A larger conversation count, a
slower per-turn generation (larger model or longer `max_tokens`), or a genuinely bursty arrival
pattern (all 8 conversations' next turns landing within the same ~1s window, not just started
concurrently) would each independently increase real contention — any of these is a legitimate
follow-up if this is revisited, but is **not** run here: the pre-registered workload was measured
as specified, and the answer to "does J6 clear the bar on the workload as pre-registered" is no.

## Disposition

**Code reverted, not shipped.** `internal/serveapp/admission.go`/`sessions.go`/`openai.go`'s J6
changes (commit `81e62d32`, never pushed to `origin/main`) are reverted via `git revert` rather
than silently dropped — the attempt stays visible in history, and this document is the record of
why. J1's plain FIFO (`admission.go`'s `waiters.Front()`) remains the shipped behavior.

**Not re-attempted this pass.** Task-work-queue-2026-09.md's J6 entry is marked killed, with this
document's own hypothesis section as the pointer for anyone considering a different workload shape
later — starting from "why did the measured effect stay small," not from zero.
