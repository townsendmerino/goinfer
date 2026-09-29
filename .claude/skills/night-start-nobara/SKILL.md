---
name: night-start-nobara
description: "Kick off tonight's night queue on nobara-pc (Linux): preflight what night.py start does not check, start it, verify it detached, and report the start time and expected finish. Only when the owner invokes it."
disable-model-invocation: true
---

# Start nobara's night queue

The owner runs this at bedtime, on `nobara-pc`. `CLAUDE.md` ("Run budget: quick by day, long by night") says
only the owner starts the queue, so **run this only when the owner invokes it**. Never run it on your own
initiative, and never because a job is waiting.

Input (optional): a deadline `HH:MM`, passed to `--deadline` (default 06:30: no job *starts* if it would end
later).

Everything here is nobara's own queue (`~/goinfer-logs/night`). The Mac has a separate queue and its own
`night-start` skill; this one never touches it.

**This skill is Linux-specific.** Do not reuse its exact commands on the Mac — `pmset`/`caffeinate` don't
exist here and aren't needed (a desktop box doesn't sleep the way a laptop does; `night.py`'s own `start`
only runs the battery/keep-awake checks `sys.platform == "darwin"`), and this repo's own
`pkill-self-match.md` rule changes how process checks below must be written.

## 1. Preflight: what `night.py start` does not check

Run these from the repo root (`~/mycode/goinfer`). The shell is zsh: quote globs, and read an exit code
directly, never through a pipe.

1. **The plan:** `python3 scripts/night.py list` and `python3 scripts/night.py start --dry-run` (add
   `--deadline HH:MM` if one was given).
   - Both show each job with its estimate and projected start/end, and whether everything fits before the
     deadline; `--dry-run` runs the same refusal checks `start` itself would, without starting anything.
   - `start --dry-run` refuses outright if a runner is already going — stop and report `night.py status`
     instead of starting.
   - An empty queue: say so and stop.
2. **The timing lock:** `python3 scripts/timing_lock.py status` must say `free`.
   - If a timed run holds it, every queued job that wraps a timed harness will wait for it or refuse.
   - Name the holder. **Never delete the lock file** to get past it.
3. **Load and stray processes that would hold the idle gates all night:**
   - `cat /proc/loadavg` — the 1-minute figure should be low once this session goes quiet.
   - Check for leftovers with **exact-name or anchored** matches only — on this box, GNU `pgrep -f
     '<pattern>'` matches its OWN invoking shell (measured, `pkill-self-match.md`: the Bash tool runs each
     command as `zsh -c '<command>'`, and `<pattern>` sitting in that same command line makes `pgrep -f`
     find itself). Use `pgrep -x <exact-name>` (`goinfer-serve`, `ollama`, `<=15-char names only>`) or an
     anchored `pgrep -f '^/path/to/binary'` / `pgrep -f '^\./binary'` instead of a bare substring.
   - `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` — should read near the machine's own
     idle baseline (this session's own baseline has read ~639 MiB / ~1% recently; a stray load holding
     several GB or nonzero utilization is a live process to name, not assume away).
   - Report what you find, with PIDs. **Kill nothing without the owner's say-so** — a process may be
     another session's in-flight work.
4. **Disk:** `df -h ~` needs a few GB free. Jobs write logs and JSON under `~/goinfer-logs` and wherever
   each job's own `run-*.sh` points.
5. **Each queued job's inputs exist:**
   - For every `run-*.sh` (or bare command) in the plan, read it and check the pinned binary, worktree or
     checkpoint it names is actually there (`[ -x "$BIN" ]`, `[ -f "$CKPT" ]`). A job that dies at 23:00 on a
     missing binary wastes the night.
   - **Model paths:** per this repo's own storage rule (`CLAUDE.md`, "Models: stored in the archive,
     benchmarked from local disk"), any job whose model path starts with `/srv/models` is void — it would
     measure a 5400 rpm SMR disk, not the engine, and not error while doing it. Flag this before starting,
     don't discover it in the morning's numbers.
   - Do not execute anything in this step — read the scripts, don't run them.

If any check fails, report it and **ask** before starting. The owner may start anyway.

## 2. Start

`python3 scripts/night.py start` (with `--deadline HH:MM` if given). It detaches the runner into its own
session (`start_new_session=True`) and prints its PID and the run directory
`~/goinfer-logs/night/runs/<date>/`.

## 3. Verify it took

- `python3 scripts/night.py status` shows a runner going and the first job running.
- `ps -o pid,ppid,sid -p <runner pid>` — PPID should be 1 (detached), matching this repo's own
  "Long-running work" convention for anything backgrounded.
- After ~30 s, `tail -3 ~/goinfer-logs/night/runs/<date>/runner.log` should show the first job starting (or
  the settle wait), not an error.

## 4. Report

In a few lines:
- **The wall-clock start time and the expected finish**, from the plan, never a bare duration — e.g.
  "started 22:40, 3 jobs, ~2h 10min, expect done ~00:50; deadline 06:30".
- Which jobs will run and which will not fit before the deadline.
- Anything preflight found (stray processes, low disk, a missing job input, a `/srv/models` path).
- **The morning:** `python3 scripts/night.py morning` prints the summary. `night.py stop` finishes the
  current job and stops; `stop --now` kills it too.
- The Mac's queue is separate and out of scope here — if it also needs starting, that's the Mac's own
  session and its own `night-start` skill, only on the owner's word.
