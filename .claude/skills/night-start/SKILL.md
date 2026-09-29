---
name: night-start
description: "Kick off tonight's night queue on the MacBook: preflight what night.py start does not check, start it, verify it detached and is keeping the Mac awake, and report the start time and expected finish. Only when the owner invokes it."
disable-model-invocation: true
---

# Start the Mac's night queue

The owner runs this at bedtime. `CLAUDE.md` ("Run budget: quick by day, long by night") says only the owner starts
the queue, so **run this only when the owner invokes it**. Never run it on your own initiative, and never because a
job is waiting.

Input (optional): a deadline `HH:MM`, passed to `--deadline` (default 06:30: no job *starts* if it would end later).

Everything here is the MacBook's own queue (`~/goinfer-logs/night`). nobara has a separate queue; §4 only reports
it.

## 1. Preflight: what `night.py start` does not check

Run these from the repo root. The shell is zsh: quote globs, and read an exit code directly, never through a pipe.

1. **The plan:** `python3 scripts/night.py start --dry-run` (add `--deadline HH:MM` if one was given).
   - It lists each job with its estimate and projected start and end, and names any job that will not fit.
   - It refuses outright if a runner is already going. Then stop and report `night.py status` instead.
   - An empty queue: say so and stop.
2. **The timing lock:** `python3 scripts/timing_lock.py status` must say `free`.
   - If a timed run holds it, every queued job will wait for it or refuse.
   - Name the holder. Never delete the lock file.
3. **Load that would hold the idle gates all night:**
   - `uptime` should show a 1-min load average below about 1.5 once this session goes quiet.
   - Look for leftovers: `pgrep -lf 'vm_stat|while :'` (orphan samplers, which have blocked a gate before),
     `pgrep -lx ollama` (a stray peer server holding memory; a job starts its own), and
     `pgrep -lf 'goinfer-serve|serve-cpu|serve-metal'`.
   - Macs are safe with `pgrep -f`, because BSD pgrep skips its own ancestors. It still matches *other* shells whose
     command line holds the text.
   - Report what you find, with PIDs. **Kill nothing without the owner's say-so.** A process may be another
     session's work.
4. **Disk:** `df -h ~` needs a few GB free. The jobs write logs and JSON under `~/goinfer-logs` and
   `~/goinfer-bench`.
5. **Each queued job's inputs exist:**
   - For every `run-*.sh` in the plan, read it and check the pinned binary or worktree it names is there (e.g.
     `[ -x "$BIN" ]`). A job that dies at 23:00 on a missing binary wastes the night.
   - Check the paths the script declares. Do not execute anything.
6. **Power:** `pmset -g batt` should say AC power (start also warns). The lid must stay open; closing it sleeps the
   Mac.

If any check fails, report it and **ask** before starting. The owner may start anyway.

## 2. Start

`python3 scripts/night.py start` (with `--deadline HH:MM` if given). It detaches the runner into its own session and
prints its PID and the run directory `~/goinfer-logs/night/runs/<date>/`.

## 3. Verify it took

- `python3 scripts/night.py status` shows `runner going, pid N` and the first job running.
- `ps -o pid,ppid,command -p N` shows the runner. It started with `start_new_session`, so closing this terminal or
  session does not stop it.
- `pgrep -lx caffeinate` shows the keep-awake process. The runner starts it with `-w <runner pid>`; without it the
  Mac may sleep.
- After ~30 s, `tail -3 ~/goinfer-logs/night/runs/<date>/runner.log` should show the first job starting (or the
  settle wait), not an error.

## 4. Report

In a few lines:
- **The wall-clock start time and the expected finish**, from the plan, never a bare duration. For example:
  "started 22:40, 1 job, ~75 min, expect done ~23:55; deadline 06:30".
- Which jobs will run and which will not fit.
- Anything preflight found.
- **The morning:** `python3 scripts/night.py morning` prints the summary. `night.py stop` finishes the current job
  and stops; `stop --now` kills it too.
- **nobara**, reported but not started unless the owner asks:
  `ssh -o ConnectTimeout=8 nobara 'cd ~/mycode/goinfer && python3 scripts/night.py list | head -3'`.
  - Starting it is `ssh nobara 'cd ~/mycode/goinfer && python3 scripts/night.py start'`, and only on the owner's
    word. It is a separate machine and a separate decision.
  - Say its total and the latest start time at which everything still fits, which `list` prints.
