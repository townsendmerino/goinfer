# Task: the same verdicts for a fraction of the machine time — the test-efficiency campaign (TE0–TE12) — 2026-09

> **Status: OPENED 2026-09-28; nothing built. TE0's first cut — the census in §1 — is done, from the records already on
> disk.** Owner, 2026-09-28: *"holy shit there are so many huge runs, i think we need a campaign to figure out how to
> test things much more efficiently."* The same morning's rule (root `CLAUDE.md`, "Run budget: quick by day, long by
> night"; c9d8ea04, c4e3aec2) moves long runs to the night queue. This campaign makes them shorter and fewer. It is the
> device-bound complement of [`task-ci-speed-2026-09.md`](../completed/task-ci-speed-2026-09.md) (C0–C9), whose §4 left
> exactly this out: "`go run ./cmd/gate` on a real box stays the correctness gate."
>
> **Done 2026-09-28 (the §6 first moves):**
> - TE9, the timing lock (`scripts/timing_lock.py`), and TE5(a), the two-arm rule in `CLAUDE.md`.
> - TE0's Mac transcript miner. Waiting is the largest class, 66.5 h in 30 days.
> - The analysis trio:
>   - TE2(b): a null, so restarts stay the unit;
>   - TE4: SEQ-v1 killed by its replay, on order sensitivity;
>   - TE5(b): killed as written, survives if only resolved rows count; the owner decides.
> - TE7(b): `decoder` does cache, and C7's claim is retracted.
> - TE1: built and queued for tonight (`te1-aa-mutation`, ~75 min).
>
> TE0 is complete on both machines ([record](../measurements/test-efficiency-2026-09.md)). See each item for its proof.

## BLUF

- **About a quarter of a served gate's wall-clock is measurement.** In the 14 logged timed runs that have an idle gate,
  **44% of the wall was the gate waiting** (138 of 311 min), and inside the served cells only **45% was decode being
  timed** (923 cells; the rest is server start, model load, warm-up, prompt prefill, teardown). Even on a quiet box at
  night, much of the waiting is the harness waiting out *its own previous cell*, because the gate reads a one-minute
  moving average (§1.2).
- **The longest single runs are measurement gates written as `go test`,** all-or-nothing and without resume:
  `TestPrefillGateReference` up to 168 min (it builds the slow f32-activation reference the Metal gates are scored
  against, with its staleness judged by hand), `TestMoEExpertMajor_endToEnd` 113, `TestA3MoEExclusionIsMeasured` 108,
  `TestPrefillGate` 76.
- **Decisions use more instrument than they need.** The L1 served gates ran Ollama in every pass while the graded rule
  read new ÷ old (Ollama was a quarter to a third of the cell time). A 35-second in-process interleaved kernel bench,
  run at loadavg 2.8 during the owner's day, pointed the same way as the 33-minute served gate that followed it. Passes
  run to a fixed N plus a second order-reversed pass, even when the first has already cleared the bar — the owner
  amended one at 09:05 today to skip pass 2 when pass 1 clears by more than the passes' own spread.
- **The prompt-heavy fidelity gates run 10 prompts per cell whatever each criterion needs.** At their own margins KL
  needs a median of 10 (20 of 42 logged cells need more), agreement 32, and two criteria have no margin, so no count
  decides them — and the gates' failures come from exactly those criteria. Stopping early saves little and is
  order-sensitive (TE12).
- **Bookkeeping re-validates the world for local edits.** 36 of 37 parity families depend on `core` (11 files); 144
  September commits touched a `core`/`loaders`/`quant` file, and 90 `Deps-Hash-Refresh` runs followed.
- **Some runs are simply lost.** Of 168 September measurement records, 39 carry a void / withdrawn / discarded /
  contaminated verdict somewhere, and 4 say outright that a gate could not resolve its own bar.

**Projection for the campaign as a whole** (a band, held to by TE0's re-census at the end): the same verdicts for
**a quarter to a third of the machine time**; this morning's two-pass L1 served gate (66 min) at **7–12 min**; a
typical daytime correctness loop at **≤ 3 min**.

**The constraint over everything: a faster check that stops detecting is worse than a slow one.** Every item that
shortens, skips or replaces a check ships with proof that it still goes red — `go run ./cmd/gate mutation`, or a
replay over past data with zero misses. [`task-verification-surface-audit.md`](task-verification-surface-audit.md) is
why: six mechanisms were found in one day silently not detecting, and none was found by the mechanism itself.

---

## 0. How this campaign runs

- **TE0 first, and again at the end.** The same script (§1) is the before and the after, as C0 was for CI.
- **Each item carries:** the mechanism, a projection band written before anything is measured, the equivalence proof
  (it still detects what it detects today), and a kill line.
- **The campaign follows its own rule.** Analysis items run by day; every timed arm goes on the night queue
  (`scripts/night.py`), pre-registered first.
- **The measurement standards in `CLAUDE.md` do not change** — paired differencing, the do-nothing arm,
  pre-registration, retractions in place. What changes is the instruments and designs, so those standards cost less.

---

## 1. The census — TE0's first cut, from the records on disk

Script: [`measurements/test-efficiency-2026-09/census.py`](../measurements/test-efficiency-2026-09/census.py)
(read-only; it runs and times nothing). Its full output for this cut is `census-2026-09-28.txt` beside it. It sees only
what was archived — a daytime `go test` that printed to a terminal is invisible to it — and most logs do not record their
host, so machine attribution is given only where a record carries one. TE0 closes both gaps.

### 1.1 The served harness (`bench_peer.py` and its siblings)

93 outputs, 2026-08-27 to 2026-09-28: **1,057 cells, 24.3 h of cell wall** (nobara 16.8 h over 69 outputs, the Mac 7.5 h
over 24). Of the 923 cells whose records carry token counts, **decode was being timed for 45%** of the cell wall.

| model / engine | cells | cell wall | timed | overhead per cell |
|---|---|---|---|---|
| 7B goinfer | 86 | 152 min | 58% | 44.6 s |
| 7B goinfer_old | 37 | 72 min | 69% | 35.9 s |
| 7B Ollama | 65 | 76 min | 69% | 21.5 s |
| 7B llama.cpp | 31 | 18 min | 73% | 9.5 s |
| 1.5B goinfer | 124 | 87 min | 45% | 23.2 s |
| 1.5B Ollama | 100 | 55 min | 50% | 16.5 s |
| M35 goinfer | 10 | 167 min | 4% | 962 s |

**goinfer's per-cell overhead is the largest term:** on the same 7B model it is 2× Ollama's and 4.7× llama.cpp's. The
column is everything that is not timed decode — server start, load, warm-up, each completion's prompt prefill, teardown
— and the harness restarts the server for every cell. The split between those terms is not recorded; TE0 adds it, and
TE2 goes after goinfer's excess.

### 1.2 The idle gate

| run | wall | waiting on the idle gate |
|---|---|---|
| peer-claim attempt 3 (nobara, 09-25) | 47.3 min | 20.0 min |
| L1 gate 6 pass 1 (Mac, 09-28 07:15) | 40.5 min | **29.0 min** |
| CPU decode baseline (nobara, 09-27 19:15) | 37.4 min | 15.0 min |
| L1 gate 5 (nobara, 09-27 22:29) | 35.6 min | 14.0 min |
| L1 arm64 fix (Mac, 09-28 08:57) | 33.3 min | 25.0 min |
| L1 gate 6 pass 2 (Mac, 09-28 08:00) | 25.7 min | 11.7 min |

All 14 logged runs with gate lines: 311 min of wall, 138 of it waiting (44%).

**Why it waits even on a quiet box.** `gate_cell_idle()` reads the one-minute load average, an exponentially damped
average with a one-minute time constant. After a cell that loaded the box to ~8, it takes about 60 · ln 8 ≈ 2 minutes to
fall under a 1.0 cap with nothing else running. `baseline.log`'s first gate lines read 4.55 → 3.26 → 2.40, 20 s apart —
a decay constant of ~60 s, the signature of the harness's own previous cell draining, not another job. nobara's gate 5,
late evening at the default 1.0 cap, still waited 14 of 36 minutes. On the Mac by day the owner's own load stacks on
top: gate 6 pass 1 waited 29 of 40.5 minutes, with the cap raised to 2.5.

### 1.3 `go test` in the archived logs

384 package results, **35.9 h** — `decoder` 18.8 h (89 runs, median 137 s), `metal` 12.5 h (204 runs, median 123 s),
`cuda` 4.3 h. The medians are daytime-sized; the hours are in the tail, and the tail is measurement gates:

| test | runs | longest | median |
|---|---|---|---|
| `TestPrefillGateReference` | 4 | 168 min | 73 min |
| `TestMoEExpertMajor_endToEnd` | 2 | 113 min | 113 min |
| `TestA3MoEExclusionIsMeasured` | 2 | 108 min | 108 min |
| `TestPrefillGate` | 1 | 76 min | 76 min |
| `TestVsumMechanism` | 1 | 54 min | 54 min |
| `TestPrefillGateVsReference` | 8 | 49 min | 32 min |
| `TestMoEExpertMajor_bitIdentical` | 4 | 39 min | 8 min |
| `TestQwen35GGUF_vsSafetensors` | 1 | 31 min | 31 min |

### 1.4 Parity churn

37 families; **36 use `core`**, which is 11 files — among them `registry.go` (3,013 lines), `model.go` (2,305) and
`config.go` (1,511). Any edit anywhere in them re-stales nearly every family. Since 2026-09-01, **144 commits touched a
`core`/`loaders`/`quant` file, and 90 carry a `Deps-Hash-Refresh` trailer** — each one ~4 minutes of forward goldens
(`scripts/refresh_parity_hashes.sh`), and each one an interruption: the pre-push hook refuses a push while the manifest
is stale.

### 1.5 Waste

Of 168 September measurement records, **39 carry a void / withdrawn / discarded / contaminated verdict somewhere** (not
every one means a whole run was thrown away), including "A process collision invalidated the first launch, and its
numbers are discarded" and "Every timing in this file is VOID". **4 record a gate that could not resolve its bar**, e.g.
"Four arms each cannot resolve the registered ±3% at this noise". Two more of the same class are in task docs rather
than records: the L1 f16 sweep, where the per-cell p10 cosine moved ±0.02 under a 2^-11 perturbation against a 0.005
allowance, and the Sep 5 L1 §3 gate, later shown to have ~95% false-fail for an equal arm.

### 1.6 What is not the problem (checked, so the campaign does not invent one)

- **CI.** C0–C9 closed it; the workflow went from ~20 min to 313 s (6159c2b0).
- **The unit suites by themselves.** `decoder` median 137 s and `metal` 123 s are daytime-sized; they become long
  when heavy gates ride inside them.
- **The pre-push hook.** The citation lint plus `TestParityManifest_fresh`: seconds.

---

## 2. The items

### TE0 — The census, made complete and continuous *(by day; gates everything)*

- **Complete.** Each machine's vscode-claude transcripts (`~/.claude/projects/`, one JSONL per session) hold every
  command a session ran, with timestamps. They are the only source that sees daytime runs that never reached a log. A
  session on each machine writes a small miner: every `go test`, `bench_*`, `gate`, `run-*.sh` and `night.py` call with
  its duration, then classifies it. (Cowork cannot read that directory; a session on the machine can.)
- **Continuous.** The served harness records per-cell phase times — idle-gate wait, start→listening, warm-up, timed
  runs, teardown — so §1.1's 55% is split by measurement instead of by inference. `night.py`'s per-job durations
  already land in `SUMMARY.md`.
  - **DONE 2026-09-28.** `bench_peer.py` records `counts.phases`: `start_to_listening_s`, `warmup_s`,
    `runs_wall_s` = `decode_timed_s` + `prefill_and_request_s`, and `teardown_s`. The idle-gate wait is in
    `machine.gate.wait_s` (TE1).
  - `census.py` prints a phase split for cells that carry these fields, and now counts gate waits in seconds for both
    gates. It still reproduces §1.2's 138 of 311 min on the load-gate logs.
  - A one-cell exploratory smoke (not a result) showed the phases sum to the cell's own `secs` exactly.
  - Visible in the code: teardown includes a fixed `sleep 3` "to let VRAM settle", paid on CPU cells too. That is a
    TE2(a) candidate.
- **Deliverable:** `docs/measurements/test-efficiency-2026-09.md` — hours by class (served gate, in-test measurement
  gate, sweep, suite, parity/refresh, evaluation) × machine × day/night for the last 30 days, the top 20 runs, and the
  re-runs. Re-run weekly while the campaign is open, and at close.
- **Band / kill:** none — it is the instrument.
- **Mac transcript miner: DONE 2026-09-28.** The miner is
  [`transcript_miner.py`](../measurements/test-efficiency-2026-09/transcript_miner.py), and its output is
  `transcripts-macbook-2026-09-28.txt` beside it. It pairs each Bash `tool_use` with its `tool_result`, or with its
  `task-notification` for a backgrounded call. It covers 206 transcripts (subagents included) and 43,914 calls since
  2026-08-29.
  - **Excluded, 7 spans (145.6 h in total).** These are foreground calls whose result arrived more than 610 s later
    (the tool caps a foreground call at 600 s, so these were blocked, not running), and background tasks that ended
    "stopped" (a server or a watch left up for days until someone stopped it). Left in, they were more than half of
    the raw total, including one 100.5 h "call".
  - **The largest class is waiting:** **66.5 h of wait/poll, 55.7 h of it by day**, more than half of all call time.
    These are sessions blocked in `until … sleep` / `gh run list` / `ssh … kill -0` loops on CI, remote runs and
    detached benchmarks.
  - **Unit-test suites: 23.7 h, 20.0 h by day.** Re-runs are visible: the whole `decoder` suite 3–6 times on single
    days, at 5–10 min each. That is TE7's target (affected-only selection and the test cache).
  - **Smaller classes:** in-test measurement 8.7 h, build/lint 6.1 h, parity/refresh 1.4 h. The served harness calls
    total 1.2 h because served gates run detached, and their spans are census.py's (§1.1).
  - **nobara: DONE 2026-09-28** (at 12:31, with the box idle), from a temporary worktree of `origin/main`. It covers
    18,730 calls, and waiting is again the largest class (47.8 h). The deliverable,
    [`test-efficiency-2026-09.md`](../measurements/test-efficiency-2026-09.md), combines both machines: **114 h of
    session waiting, 54% of all session call time, 95 h of it by day.** Unit suites come next at 36.7 h.

### TE1 — The idle gate reads what is running now *(one night: A/A + mutation)*

- **Change.** Gate on instantaneous utilisation sampled over a few seconds (Linux `/proc/stat` deltas; on macOS the
  CPU line of `top -l 2 -s 1 -n 0`), plus a named-process check (no other `goinfer-*`, `ollama`, `llama-server` or
  `go test` alive). Keep recording loadavg per cell exactly as now. **The refusal design stays:** "Refusing loses a
  sweep; proceeding loses the ability to tell which rows were real."
- **Band:** idle-gate share of served wall from 44% to ≤ 10% at night; served pass wall −30–45%.
- **Equivalence (pre-registered, one night).**
  - *Mutation:* start a CPU hog mid-sweep. The new gate must hold the next cell exactly as the old one does, and
    release within one sample window after the hog stops.
  - *A/A:* the same binary in both arms, 0.5B / 1.5B / 7B, 3 runs, old gate and new gate interleaved on the same night.
    The new gate's per-cell A/A spread must be ≤ the old gate's + 25%.
- **Kill:** the A/A spread widens past that, meaning the long wait was buying real recovery (thermals on the M1 Pro are
  the suspect). Fallback: a fixed, measured cool-down plus the instantaneous check. Record `pmset -g therm` per cell on
  the Mac either way, so the thermal question has data next time.
- **Built 2026-09-28** (by day, nothing timed).
  - `bench_peer.py BENCH_IDLE_GATE=load|instant`. The default stays `load`, unchanged, until the night below decides.
  - `instant` samples the share of all CPUs busy over `BENCH_BUSY_WINDOW_S` (3 s; `/proc/stat` deltas on Linux, the
    second sample of `top -l 2` on macOS), under `BENCH_MAX_BUSY` (10%).
  - It also refuses while any timed-workload-named process (`goinfer*`, `serve*`, `ollama`, `llama-server`, `*.test`)
    burned more than 5% of a core during that window. A named process that is merely alive, like the Mac's resident
    Ollama app, does not count.
  - Every cell now records `machine.gate` (kind, `wait_s`, busy % or loadavg) under both gates, and `machine.therm`
    (`pmset -g therm`) on the Mac.
  - Checked live, not timed. The sampler read 23% busy and named the `decoder.test` binary a concurrent test run was
    burning (3.5 CPU-s in the window). It did not flag the idle resident Ollama.
- **Pre-registration (2026-09-28, before queueing; the Mac's first night).**
  - **Run:** `run-te1-aa-mutation.sh`, graded by `te1_analyze.py`, both in `measurements/test-efficiency-2026-09/`.
    - The binary is pinned: `serve-cpu-b9fcde67` in both arms (A/A). It was built at `b9fcde67`, which the push
      rebased to `0607f30b`. The commits between touch only `cuda/` tests and a log, no non-test Go file, so the
      binary is exactly `0607f30b`'s.
    - CPU, 0.5B / 1.5B / 7B, 3 runs, two arms.
    - Four sweeps interleaved (load, instant, load, instant), then a mutation per gate.
  - **Mutation:** an every-core CPU hog for 90 s, started the moment cell 1's record lands, so while the harness is in
    cell 2's gate. Each gate must hold cell 2 for the hog's life (gate wait ≥ hog time − 5 s). The instant gate must
    also release within 10 s of the hog stopping. The load gate's release lag is recorded. **A mutation failure is a
    kill.**
  - **A/A spread:** per sweep and model, the log ratio of the two identical arms' means. For each gate, the RMS over
    its 6. Bands on the ratio RMS(instant) ÷ RMS(load):
    - **≤ 1.25 PASS** (the item's own line);
    - **> 2.07 KILL** (the one-sided 5% point of F(6,6), 4.28, square-rooted);
    - **between them, PARKED** (to the owner, with more pairs).
    
    This deviates from the item's single "+25%" line. At 6 values per gate, equal spreads would exceed +25% about
    30% of the time. That is §1.5's gate that cannot resolve its bar, so the kill line sits where chance alone would
    reach it 5% of the time, and the band in between is parked, not guessed.
  - **Idle-gate share of served wall:** the instant gate's mean over its two sweeps must be ≤ 10%. The sweep-wall
    reduction is reported against the −30–45% band.
  - **Thermal:** every cell's `pmset -g therm` line is kept. Any warning is listed.
  - **On PASS:** `instant` becomes the default in a separate commit, citing the result.
- **Attempt 1, the Mac's night of 2026-09-28: it did not run as designed. Nothing is graded from it.**
  Partial output is in `te1-2026-09-28/`.
  - **Only one of the four A/A sweeps ran** (instant, tag 1). Both load-gate sweeps were REFUSED at start by
    `bench_peer.py`'s preflight (load 1.50, then 6.56). The script ran its sweeps back to back, and the preflight
    refuses a busy box rather than waiting, by design.
  - **Both mutations crashed before any hog ran.** The hog was `python3 -c` with `multiprocessing`, which cannot spawn
    workers on macOS (`Can't get attribute 'spin' on <module '__main__'>`).
- **Amendment for attempt 2, 2026-09-28, before it runs. The design, bands and analyzer are unchanged.**
  - Before each sweep and each mutation, `run-te1-aa-mutation.sh` waits until the 1-min load is ≤ 1.0 (up to 30 min).
    The wait is logged before the timeline's START line, so it is outside every span `te1_analyze.py` grades.
  - The hog is `te1_hog.py`, a file with a `__main__` guard. Checked by day for 2 s: every core busy (758% CPU).
  - Output goes to `te1-attempt2/`, because `bench_peer.py` resumes from existing files. Graded as
    `te1_analyze.py te1-attempt2`.

### TE2 — Per-cell overhead: start once, time more *(analysis by day, one night to confirm)*

- **(a) Split goinfer's excess, then cut the largest term.** With TE0's phase times, attribute it to start→listening
  (load: the sidecar default already maps a cached `.giw` after first use, so whatever remains is per-start work),
  warm-up, or per-completion prompt prefill at the cell's depth. Then cut what is overhead: keep the server up across
  consecutive runs of the same engine and model where TE2(b) allows, and remove per-start work the measurement does not
  need.
  - *Equivalence:* any change to how the harness starts goinfer must leave decode tok/s within the A/A floor. A
    measured difference would itself be a finding, and the bench keeps whatever the product does by default.
  - *Band:* goinfer per-cell overhead −40–70% on the 7B (44.6 s → 15–25 s) if the excess is start-up. If it is prompt
    prefill, it is not harness overhead — the prefill campaigns own it — and TE2(a) records a null.
- **(b) Put the repetitions where the variance is** (Kalibera & Jones). Split the recorded `runs[]` into
  between-restart and within-server variance, from existing JSONs only. If within-server variance dominates, more
  completions per restart and fewer restarts give the same interval for less wall.
  - *Band:* a documented null (restarts are the right unit), or −20–40% of cells.
- **Kill:** (a) the excess is prefill, or cutting start-up moves decode beyond A/A; (b) between-restart variance is
  ≥ half the total.
- **(b) RESULT 2026-09-28: a documented null for the −20–40% band; restarts stay the unit.**
  [`te2b_variance.py`](../measurements/test-efficiency-2026-09/te2b_variance.py) and
  `te2b-variance-2026-09-28.md`, from existing JSONs only.
  - **Data:** 820 usable cells from 45 files in 25 sessions; 108 replicated cell groups, 362 restarts, 7,072
    completions.
  - **Two corrections the automatic key needed.** The prompt set is part of a cell's identity (essay-v2 moved one
    Ollama cell 195.1 → 183.0). 30 arm files had to stay file-local, because their treatment is not in the record.
    Without both, both machines read ~90% between-restart, a false kill.
  - **Read as the item states it** (the restart share of a run mean's variance): nobara 0% (90% bootstrap 0–66),
    the Mac 67% (30–85). So it splits by machine.
  - **With session drift separated,** the restart term inside one session is small: nobara 0% (0–33), the Mac 31%
    (0–66). Most of "between restarts" is **between sessions**: 69% of a Mac run mean's variance, 20% of nobara's.
    That is `CLAUDE.md`'s same-session interleaving rule, measured.
  - **Trading restarts for completions saves nothing** at goinfer's measured in-session restart term (0.35% sd):
    2 restarts × 3 runs is already the cheapest design. Only if the term were zero would 1 × 6 save 29–35% (restarts
    priced dear) or 3–11% (priced cheap). No record has an in-session 1.5B/7B CUDA goinfer replicate to tell those
    apart.
  - **On the Mac CPU, more completions per restart is the wrong lever.** Rates fall within one server lifetime
    (goinfer 7B down to −26.7%, plausibly thermal). TE2(a)'s "keep the server up" is unsafe there, and for MoE (one
    M35 lifetime +29.6%, one M26 −22.1%).
  - **Open for TE3:** nobara's same-build cross-session pairs differ by a median 0.15% (p90 0.99%), far under
    `CLAUDE.md`'s ~3.5% session drift.
  - **Night job to settle the in-session term, not queued:** 1.5B/7B CUDA goinfer, one build, 4 restarts × 6 runs in
    one session, pre-registered first.

### TE3 — A noise registry, and a power check before any gate is registered *(by day)*

- **Why:** §1.5. A gate that cannot resolve its bar costs its full wall and decides nothing, and today that is found out
  after the run.
- **Build.** `docs/measurements/noise-registry.json`, plus a short `.md` beside it. Per machine × instrument (served
  decode, served TTFT, in-process kernel A/B, logits cosine / argmax agreement per family) × model × depth: the A/A
  spread of the paired ratio or metric, with its source record. Seed it from what the records already hold:
  - the A/A arms already recorded (fourteen September records mention one);
  - this morning's Mac pass-to-pass 0.022;
  - `CLAUDE.md`'s ~3.5% between-session drift.

  Refresh it monthly as a night job.
- **Tool.** `scripts/power.py <instrument> <cell> --bar 0.97 --alpha 0.05 --power 0.8` prints the N that resolves the
  bar, or "cannot resolve at a feasible N — change the instrument". It also covers binomial and ECE sample sizes for
  evaluation runs: the D6a amendment's ±1.5-point SE (2852eaa1), done by hand today, becomes a one-liner.
- **Template:** every pre-registration gains two lines — the registry's noise for the cell, and the tool's N.
- **Band:** "ran and could not decide" → 0. N is right-sized: smaller for most gates, larger for the few that were
  under-powered, and those are correctness wins, not costs.
- **BUILT 2026-09-28.**
  - **The registry:** [`noise-registry.json`](../measurements/noise-registry.json) and its `.md` hold 83 entries from
    109 records, 62 of them with a usable sd. Each has its statistic, n, source and provenance class, and a 16-item gap
    list comes with it. `te3_registry_seed.py` computes the served rows.
  - **The tool:** `scripts/power.py`. 22 tests pass; four hand mutations turned them red.
  - **Finding: a same-binary A/A understates a new ÷ old gate's noise.** On nobara CUDA the same binary restarted gives
    0.81% RMS, while two builds with identical 0.5B code give 1.64%, all six readings below 1. That offset is TE5(b)'s
    "−2% on identical code", so the lookup prefers cross-build A/As.
  - **What the tool then says:** at a 0.97 bar a 0.5B CUDA gate needs 11 blocks, and a Mac CPU 7B gate by day cannot
    resolve at any feasible N.
  - **`CLAUDE.md`'s ~3.5% session drift is one observation:** a sampled goinfer phi3-mini cell, 2026-08-09, commit
    `414bb364`. TE2(b)'s 0.15% is 466 peer-engine pairs. goinfer's own cross-session drift is unmeasured.
    **OWNER DECISION 2026-09-28: replaced.** `CLAUDE.md` now carries the registry's wording, as do the other living
    statements of the rule (`docs/benchmarks.md`'s method notes, `task-peer-benchmarks.md`, `spec/02`, the
    adaptive-speculation brief, `QUEUE.md`).
    - **Dated measurement records** that cite ~3.5% as their reasoning are left as history; the registry's §3 lists the
      Mac ones.
    - **The frozen gate prompt snapshots** (`testdata/prefill-gate-prose-{a,b}/`) quote it too. They must not change:
      they are the gates' hashed prompts.
  - **D6a:** `power.py binomial --p 0.9 --half-width 0.015` reproduces the amendment's N = 400, read as one SE. Its
    95% half-width at 400 is ±2.94 points.
  - **The ECE approximation suggests D6a's ≤ 0.05 ECE bar may be tight:** a perfectly calibrated model would read
    0.017–0.047 at that sample. That is for the owner before D6a runs.

### TE4 — Stop when the answer is in: sequential, counterbalanced gates *(replay by day, then night)*

- **Today:** fixed N per cell, and for decisions a second order-reversed pass (a dozen September records mention one).
  The owner's 09:05 amendment to L1 today skipped pass 2 because pass 1 had cleared the bar by more than the passes'
  own measured spread. The instinct was right, but stopping ad hoc when results look good inflates false passes. The
  stopping rule has to be designed for it.
- **Design.**
  - Counterbalanced ABBA blocks inside one pass: order effects cancel within the pass, replacing the separate
    reversed pass.
  - A pre-registered sequential rule on the paired ratio — group-sequential boundaries (O'Brien–Fleming) or an
    always-valid confidence sequence (Johari et al.) — with a hard cap on N. Stop at the first block where the interval
    clears the ship bar or the kill bar. At the cap, today's ambiguous → parked band applies unchanged.
- **Validation before any new run:** replay the recorded per-run rates of past gates (`runs[]`, both passes where they
  exist) through the rule. It must reach the recorded verdict in ≥ 95% of gates, **never flip a FAIL to a PASS**, and
  use ≥ 25% fewer runs.
- **Build:** `BENCH_SEQUENTIAL=<ship>,<kill>,<maxN>` in `bench_peer.py` (paired arms only), and the same helper for the
  in-test gates that loop over prompts.
- **Band:** −40–60% wall on gates whose answer isn't close; unchanged on close ones, which is where the time belongs.
- **Kill:** the replay disagrees on more than 5% of gates, flips any FAIL, or saves less than 25%.
- **Replay RESULT 2026-09-28: TE4-SEQ-v1 is KILLED by its pre-registered replay.**
  [`te4_replay.py`](../measurements/test-efficiency-2026-09/te4_replay.py) and `te4-replay-2026-09-28.md`. The rule
  was fixed at 10:46, before the script existed.
  - **The rule:** a completion-paired log ratio, Lan–DeMets O'Brien–Fleming spending (two-sided α 0.05, t scale),
    a look after every run pair, and a δ = 0.022 allowance on served gates.
  - **The gates:** 18 eligible — 4 served new-vs-old, 8 served peer, 3 in-process ABBA, and 3 q4k lever gates. All
    93 graded sub-gates reproduce their recorded label at the cap.
  - **Agreement 17/18 (94.4%),** under the ≥ 95% bar. With n = 18, one gate is 5.6%, so the kill is provisional on
    the count.
  - **In recorded order:** no FAIL → PASS, and 29.9% of cell-runs saved (40.7% on graded cells).
  - **The disqualifying finding is order sensitivity.** Peer-claim cell i is the Mac CPU 0.5B against Ollama, and
    Ollama moved 125.2 → 93.7 / 93.1 tok/s between runs, a shift no within-run interval sees. In 2 of its 6 run
    orders the rule stops at run 2 and calls goinfer AHEAD. Every other gate agrees in every order, all 960
    in-process orders included.
  - **Neither alternative removes the bad orders:** δ = 0 (V0, pre-registered), or a variant designed afterwards (no
    stop before run 2; recorded as not a validation).
  - **What a v2 needs:** a between-run noise allowance from TE3's registry and ABBA blocks within the pass (TE4's own
    design), validated on gates not used here.
- **TE4-SEQ-v2, 2026-09-28: pre-registered, then failed its own simulation screen; the cause is the screen.**
  Record: [`te4-seq-v2-2026-09-28.md`](../measurements/test-efficiency-2026-09/te4-seq-v2-2026-09-28.md).
  - **The rule** (5551e9c3, written before any code or data):
    - mirrored ABBA quads of restarted cells;
    - no stop before the second quad;
    - the registry's sd as a prior pooled into the interval (ν₀ ≤ 6);
    - v1's O'Brien–Fleming spending over the quads;
    - a floor only for a cross-build build effect.

    Validation is a simulation screen by day, then a held-out 8-gate nobara night corpus.
  - **Screen, as registered: FAIL.** Worst P(PASS) at a true ratio at or below the ship bar was 0.317 (bar ≤ 0.05), and
    agreement with fixed-N 94.77% (bar ≥ 95%). Savings: 49.1%.
  - **Post-hoc diagnosis.** The screen's shift model moves each arm ±5% with p = 0.05 per block, and the fixed-N gate
    the rule replaces fails it too (0.300). The rule is never more than 2.2 points above fixed-N. Without shifts it is
    calibrated: 0.031 against fixed-N's 0.027, 99.53% agreement.
  - **Owner, 2026-09-28: the screen is void; the night corpus decides.** `bench_peer.py` gained `BENCH_ABBA=<quads>`
    (25950af1), and the corpus is queued on nobara as `te4v2-validation-cuda` and `te4v2-validation-cpu` (3 h 20 min).

### TE5 — The cheapest instrument that answers the question *(a rule now; calibration by day)*

- **(a) One peer read per shipped lever, not per pass.** The L1 gates ran goinfer, goinfer_old and Ollama in every
  pass. Ollama was 7.0 of 21.6 cell-minutes in gate 5, 2.9 of 11.4 in gate 6 pass 1 and 2.8 of 8.2 in the arm64 fix,
  while the graded ratio was new ÷ old and the Ollama column was reported. The campaign's headline ratio still gets
  read, once, after a lever ships, same-session interleaved as `CLAUDE.md` requires. A pass whose decision is new ÷ old
  runs two arms.
- **(b) By day, a kernel speed question goes to the in-process interleaved A/B first.** Both variants run in one
  process through a testhooks seam, alternating, so the owner's background load hits both arms alike. The example is
  this morning's arm64 fix:
  - the kernel bench: 35 s at loadavg 2.8, fused ÷ f32 kernel time 0.898–0.949;
  - the served gate after it: 33 min, new ÷ old 1.041 / 1.053 / 1.080 (5d223da3).

  Same direction. The served gate stays the instrument for anything claimed end to end, and it runs at night.
- **Calibration (no new runs).** Collect every change the records already cover with both an in-process A/B and a
  served A/B into a concordance table: sign agreement, and served ÷ in-process. `CLAUDE.md`'s microbenchmark trap is why
  this is a table and not an assumption. If the in-process ratio predicts the served direction in every pair, and the
  magnitude within a stated factor, "in-process by day, served at night for claims" becomes the default flow.
- **Band:** −25–33% of cells on goinfer-vs-goinfer served gates from (a), immediately; a daytime kernel verdict in
  ≤ 5 min from (b).
- **Kill (b):** any sign disagreement for a kernel-local change. The in-process lane then stays exploratory.
- **(b) Calibration RESULT 2026-09-28: killed as written, and survives if only resolved rows count. Which reading
  applies is the owner's call.** [`te5b_concordance.py`](../measurements/test-efficiency-2026-09/te5b_concordance.py)
  and `te5b-concordance-2026-09-28.md`, from existing records only.
  - **Coverage:** 142 candidate changes examined, 42 in the table as 87 rows, with 100 exclusions each given a
    reason. Tier 1 (same-session served A/B, the change alone) is 65 rows over 29 changes.
  - **Sign:** point estimates agree in 84/87 rows. Where both sides resolve a direction, 77/78 agree, and 37/37 of
    the tier-1 kernel-local rows (45/45 over both tiers).
  - **The one resolved disagreement is outside the kernel:** G26's sampler microbenchmark (0.810× in-process against
    1.085× served), for the reason its own record names.
  - **Strict reading, KILL.** Three tier-1 kernel-local rows fail because one side did not resolve a direction: the
    L1 0.5B at 0.968× with 3 of 5 pairs below 1, and two ~2–3% effects with no recorded spread.
  - **Resolved-only reading, no kill.**
  - **Magnitude.** A whole-token in-process A/B gives k = ln(served)/ln(in-process) with a median of 0.97, range
    0.60–1.08, on the kernel-local rows; 21/23 are within 1.25×. A kernel-only bench gives direction, not size
    (k 0.05–1.72, median 0.55).
  - **If the owner takes the resolved-only reading,** the supported flow is: a whole-token in-process A/B by day when
    it resolves a direction; anything unresolved goes to the served gate at night.
  - **The table cannot see false negatives:** a change the in-process A/B wrongly killed never got a served run.
- **(b) OWNER DECISION 2026-09-28: the resolved-only reading.** "In-process by day when it resolves a direction,
  served at night for the unresolved and for claims" is now the rule, in root `CLAUDE.md` § Run budget.
- **(a) DONE 2026-09-28:** the rule is in root `CLAUDE.md` § Run budget ("Two arms when the decision is new ÷ old").
  There is no separate `run-*.sh` template file to change; each campaign's script sets `BENCH_ENGINES`, so the rule
  goes where those scripts are written.

### TE6 — Change-proportional correctness: identity against the last validated build, cached references *(by day + one night)*

- **(a) A reference cache keyed by content.** Key each reference on the checkpoint's sha256, the prompt set's hash, the
  hash of the files the reference path compiles from, the arch, and the quant, with one directory per key. A gate looks
  its key up and computes only on a miss. Today `TestPrefillGateReference`'s CPU f32-activation reference (up to 168 min)
  is cached to disk (3.4 GB), but staleness is judged by hand — a stale set has already had to be renamed aside once.
  With the key, staleness is a lookup.
- **(b) Inherit validation by identity.** For a change meant to be numerically neutral for a family on a backend, the
  cheapest complete proof is byte-identical logits against the last validated build, on the family's parity prompt. L1
  did exactly this today: "Metal / WebGPU / arm64 CPU logits byte-identical old vs new" (48fee2e5). Formalize it as
  `go run ./cmd/gate identity <old-rev> <new-rev>`, per machine and per backend:
  - it builds both revisions, runs each family's parity prompt, and compares bytes;
  - identical → the family's validation carries forward, recorded as a new manifest method (identity-inherited from
    `<rev>`);
  - not identical → that family, and only that family, goes to its reference gate.
- **Equivalence.**
  - *Determinism first:* the same build run twice must be byte-identical on every backend used. A GPU backend that is
    not bitwise deterministic run to run gets a tolerance comparison against its own previous build, which is still far
    cheaper than the HF reference.
  - *Mutation:* a 1-ulp change in one family's path must be flagged for exactly the families that execute it; a
    comment-only edit must come back all identical.
- **Band:** real-checkpoint verification of a neutral change from hours to minutes; each reference computed once per
  reference build.
- **Kill (b):** run-to-run nondeterminism on CPU. That should be impossible under the bit-identical discipline, and
  would be a finding in its own right.
- **(a) BUILT 2026-09-28: the prefill references are content-keyed** (`decoder/prefill_ref_cache_testhook.go`).
  - **The key.** Each (model, K, prompt) reference is keyed by:
    - the checkpoint's sha256 (cached by path + size + mtime, so a 4.7 GB file is hashed once);
    - the prompt's own ids at that K (`fidelity.PromptSetHash`'s encoding);
    - the reference path's source: the non-test Go in `decoder/`, `internal/giw/` and `constrain/` (`go list -deps
      ./decoder`), plus `go.mod` pinning aikit and x/*;
    - the arch, weight quant, continuation length and forced exact attention.
  - **The store.** Entries live in `~/goinfer-logs/prefill-ref-cache/`, written atomically. A `.bin` without its parts
    record is an interrupted store, not a hit.
  - **The consumer path.** `TestPrefillGateReference` looks each prompt up, links a hit into the historical
    `prefill-ref[-<set>]/` path, and computes only misses. Beside each file it writes `<file>.key.json`, so every
    consumer reads the path it always did.
  - **Prompt identity is exact now.** Both gates, Metal's `TestPrefillGateVsReference` and CUDA's
    `TestPrefillGateVsReferenceCUDA`, check the sidecar's prompt hash before scoring. A mismatch is VOID with the
    reason. A file from before the cache has no sidecar and falls back to the seed-logit KL > 1.0 inference, which
    `prefill-ref-identity-2026-09-26.md` needed.
  - **Tests** (`prefill_ref_cache_testhook_test.go`, tiny llama, milliseconds):
    - the key moves with the checkpoint bytes, prompt ids, prompt length, quant, continuation and fast attention, and
      not with the checkpoint's path;
    - store, lookup, the interrupted-store miss, link, and identity verified / mismatch / no sidecar;
    - the generator computes 3 prompts, recomputes 0 on a rerun (cache mtimes unchanged), and computes exactly 1 after
      one prompt changes.
    
    A lookup-never-hits mutant turned two of them red.
- **(b) BUILT 2026-09-28: `go run ./cmd/gate identity <old-rev> <new-rev>`** (`cmd/gate/identity*.go`).
  - **Mechanics:** two temporary worktrees (removed afterwards, SIGINT included), one embedded public-API dumper built
    in each with `-trimpath`, and each family in its own process in the order new#1 → old → new#2.
  - **Compared:** full prefill + greedy-step logits as bytes. Every `GOINFER_*` variable is stripped from both sides,
    which closes L1's trap of an old-side env diagnostic, and each side's decode path is printed.
  - **Verdicts:** IDENTICAL means "validation inheritable from `<old>`". DIFFERENT gives the first differing step and
    logit, max |diff|, argmax agreement, and "goes to its reference gate". NOT RUN is listed apart.
  - **Manifest:** it prints the row it would record and writes nothing. Recording a manifest method
    "identity-inherited" is the owner's decision.
  - **Determinism holds:**
    - CPU: 120 of 120 cells byte-identical run to run. The kill criterion did not fire.
    - Metal: 38 of 38, over 19 families.
    - WebGPU: 80 of 80, over 30 families.
    - A tolerance mode exists (≤ 2× a cell's own run-to-run max |diff|, argmax and tokens exact, reported WITHIN
      TOLERANCE), but no backend here needed it.
  - **Equivalence** (opt-in heavy tests, scratch clones, ~16 s):
    - a comment-only edit gives all IDENTICAL;
    - a one-ulp `Nextafter32` in the NormParallel residual flags exactly cohere and cohere2;
    - an injected nondeterministic dumper is reported NONDETERMINISTIC, exit 1.

    Each is falsified by a `gate mutation` of the comparator or the run order.
  - **Wall, tiny assets, CPU, MacBook:** `identity HEAD~1 HEAD` took 43–46 s, 30 IDENTICAL and 7 NOT RUN (checked
    twice). The 7 lack a tiny fixture: qwen2, qwen3, qwen3_vl, qwen2_moe, mistral, mellum (skipped by design) and
    deepseek_v2.
  - **Real demo, 0.5B:** L1's own change (`3cd62e6d` → `5c85f7c0`) comes back DIFFERENT from prefill logit 0, as it
    must.
  - **Found on the way:** Metal's `BuildResident` panics on qwen3_next's W4A8 concat.
    - **RETRACTED 2026-09-28: not a bug.** The panic is `W4A8 concat needs K%32==0 (group=32), got K=16 (audit M-10)`.
      It is thrown on purpose (`metal/model.go`, `int4Concat`) and recovered by `BuildResident` into a clean decline
      to the CPU.
    - The old `qwen3next-tiny` fixture had `moe_intermediate_size: 16`, so its expert down-projection had K = 16.
      Real qwen3_next widths are multiples of 32.
    - 1b698b22 re-pinned the fixture at 32. `gate identity -backend metal -families qwen3_next HEAD HEAD` then runs
      both Metal cells resident and IDENTICAL.
    - The same decline line appears in the Metal suite logs under `docs/measurements/` from before the re-pin.
  - **Limits:** no CUDA backend here, and the dumper uses a LogitProcessor, so device-side greedy-argmax paths are not
    what it compares.
  - **OWNER DECISION 2026-09-28: yes, validation may be inherited by identity.**
    - **Recording:** `gate identity … -record <file>` writes `PARITY_ROW` lines, method
      `identity-inherited (<original method> @ <oracle rev>)` with the original metrics, for the decoder's
      `TestParityManifest_merge`, the manifest's one writer.
    - **Eligibility:** real checkpoints, the CPU on the row's own arch, old = the row's `validated_at`, new = HEAD, and
      all three CPU quants IDENTICAL. The rule is in `docs/parity-coverage-policy.md`.
    - **Tier:** `isT3Method` accepts the method exactly when its inner method is T3.
    - **Tests:** eligibility (one row, a chain that keeps the original oracle, eleven refusals) and the method's tier;
      an arch-check mutant turns the refusal test red.
    - **Not yet used on a real row.** gemma4, for one, validated on linux-amd64, so only nobara can inherit it.

### TE7 — `gate quick`: affected-only, cached, parallel tests by day *(by day)*

- **(a) Selection.** Changed files → their packages → reverse dependencies across the five modules (`go list -deps
  -test`) → the packages whose tests can observe the diff. Run those one invocation per package, in parallel, each
  tee'd to its own log. That keeps B19's streaming and regains the package parallelism that one-package-at-a-time gave
  up. The summary counts PASS / SKIP / FAIL separately, because a skip is not a pass.
- **(b) Go's test cache, by day.** Day loops drop `-count=1` where a test is reproducible; gates and anything timed
  keep it. C7 found `decoder` never caches, and blamed the 21 files that call `t.TempDir()`. That does not fit how the
  cache works:
  - cmd/go re-checks only paths inside the module root (`computeTestInputsID` skips stat/open outside it — "Do not
    recheck files outside the module, GOPATH, or GOROOT root" — read in Go 1.24's source; re-check at the pinned 1.27);
  - `t.TempDir()` lives under `$TMPDIR`, outside the module;
  - so something *inside the tree* changes between runs: a test writing next to a fixture (a sidecar or other cache
    file), or a directory listing that a test stats.

  Re-diagnose with `GODEBUG=gocachetest=1` on two back-to-back runs and diff the input lists. If it is fixable, an
  untouched `decoder` replays in seconds instead of ~5 min whenever a session edits only `metal/`, `cuda/`, `serve/` or
  the like.
- **(c) One command:** `go run ./cmd/gate quick`. It runs gofmt, vet (the tagged variants for the touched modules), the
  pinned staticcheck, the affected tests, and the tiny forward goldens when a parity shared set is touched, then prints
  the verdict and the time.
- **Band:** a typical daytime loop ≤ 3 min; worst ≤ 10.
- **Equivalence:**
  - a mutation in a leaf package makes `quick` run a failing test;
  - a mutation in `decoder` core fails the goldens;
  - the selection never drops a package that `go list -deps -test` says depends on the change.
- **Kill (b) only:** if the in-tree cause cannot be removed without restructuring tests, caching stays a per-package
  bonus that nothing relies on.
- **(a)+(c) BUILT 2026-09-28: `go run ./cmd/gate quick`** (`cmd/gate/quick*.go`).
  - **What it runs:** gofmt, CI's build/vet tag matrix, the pinned staticcheck (a U1000 canary proves it can go red),
    and one `go test -json` per package, `-j 4`, with GPU-device packages one at a time. The goldens run when a
    parity shared-set file changes, and zero goldens run is red. No `-count=1` by default. It refuses while the TE9
    timing lock is held.
  - **Selection:** the diff against the merge-base, through a `go list -deps -test` graph per module, with a reason
    per package: *affected* (the binary compiles the change), *observer* (its tests read the change as data), and
    everything else *cache-checked* (the Go cache replays it, and nothing is dropped). A test that reads outside its
    module is forced to `-count=1`.
  - **Equivalence:** checked against an independent per-test-binary graph (1,416 files × 28 binaries, 0 dropped). Leaf,
    cross-module and shared-set mutations all go red, and four deliberately broken selections each fail their test.
    An overlay-only one-character RMSNorm mutation failed 17 of 38 goldens.
  - **Wall, MacBook:** warm, a serveapp edit took 27 s and a decoder-core edit 5m55s (the agent's runs). A **cold** run
    after a day of commits took 5m53s (the Mac session's check): 29 cells, 15 run, 13 replayed.
  - **Found on the way:**
    - An explicit `GOWORK` leaks into tests that spawn `go`, so test cells use the repo's own `go.work`.
    - The Mac's gitignored `vendor/` is stale (aikit v1.46.0), which breaks `GOWORK=off` root commands; quick uses
      `-mod=readonly`.
    - `cuda` does not build on darwin, so it is listed and linted for linux/amd64 and reported NOT RUN.
  - **Memory-heavy cells: DONE 2026-09-28.** In the cold check, `examples/confidence` failed under `-j 4`. Its real
    0.5B load was refused by the fit guard beside other test binaries, and it passes alone in 5.4 s.
    - **Detection.** The package scan now finds the tests that reach a real checkpoint present on this host
      (a model file of ≥ 64 MiB). It follows a path literal, a bare file name under `testdata/` or `~/models/`, a
      `testdata/<dir>` join (test files only, since in production code `"gpt2"` is an architecture name), and an
      asset-registry key resolved as `decoder/assets.go` resolves it. It also follows those through package-local
      helpers and package-level tables. On this Mac that finds 12 cells: decoder, metal and gpu among them, split off
      as `[checkpoint]` cells so the rest of each package still runs in parallel.
    - **The rule is "alone", not "one loader at a time".** The first version held back only the other loaders.
      confidence still failed: its f32 load needs 3.6 GB, and the fit guard saw 3.4 GB available beside decoder,
      metal and their compiles. Alone it saw 5.8 GB. So a checkpoint cell now starts only when nothing else is running,
      and nothing starts beside it.
    - **Proof.** Four unit tests (`cmd/gate/quick_test.go`) and eleven mutations, each red on the test meant to
      catch it. The mutations: the first version's rule; a loader starting beside running jobs; jobs starting beside a
      loader; no helper propagation; no package-level names; bare names counted in production code; no size floor
      (twice); no split; no registry resolution.
    - **Real runs.** Both are `go run ./cmd/gate quick -j 4` on the MacBook. The logs are under
      `~/goinfer-logs/quick/`, as `te7-mem-lane-2026-09-28{,-run2}.log`.
      - **First version:** RED. confidence was refused while decoder, metal and gpu re-ran. Wall 5m35s.
      - **"Alone" rule:** GREEN. 1,625 passed, 0 failed; wall 5m28s, with decoder re-running and metal/gpu
        replayed. confidence passed in 5.3 s. `decoder [checkpoint]` (101 tests) took 2m47s alone, after the rest of
        decoder's 2m23s.
      - **Cold cost:** a cold run that re-runs metal and gpu as well should cost about a minute more than before
        (their ~3.6 min runs beside decoder, then the checkpoint cells run alone). That is estimated, not measured.
- **(b) RESULT 2026-09-28: `decoder` caches, so there is nothing to fix.** Two back-to-back
  `GODEBUG=gocachetest=1 go test ./decoder/` runs on the MacBook:
  - run 1 saved the result (310 s);
  - run 2 replayed it: `ok … decoder (cached)` in 2 s, with the same test ID and the same input ID.

  34 `decoder` test files call `t.TempDir()`, and no file inside the module was written during run 1. C7's
  "permanently uncacheable" is retracted in place, in `task-ci-speed-2026-09.md`.
  - **What re-runs `decoder` in practice:** `-count=1`, any Go change reaching the package (a new test binary is a new
    test ID), and changed env vars or module files the tests read.
  - **CORRECTED the same day (TE7(a)'s build):** eight `decoder` tests walk the whole repo, `.git/objects` included.
    So until they are split off, **any** edit or commit anywhere invalidates `decoder`'s cache. Replay needs a tree
    unchanged since the last run, not merely an untouched `decoder/`. `gate quick` runs those eight in their own
    ~3 s cell, and the rest of `decoder` then replays.
  - **Found on the way:** the first attempt at these runs failed on `TestDecodeParityInt4`. Its int4 golden had been
    red on arm64 since L1's merge (`5c85f7c0`) and nobody had seen it. It is re-captured in its own commit: the new
    ids are exactly the pre-L1 build's with f16-rounded scales, and agreement with f32 is unchanged at 11/24. A
    failing package is never cached, so TE7(b) could not have been measured until it was fixed.

### TE8 — Measurement gates that live in `go test`: resumable, reference-cached, right-sized *(per gate)*

- **Why:** §1.3's tail. As tests these gates are all-or-nothing: no resume, so a timeout or a panic at minute 100
  discards the 99 minutes before it, and their output is buffered unless each package runs in its own invocation.
- **(a) Harness shape.** Cells are written as they finish, and a re-run skips the done ones. `bench_peer.py` already
  works this way ("N cells planned, M already done").
- **(b) References** come through TE6(a)'s cache.
- **(c) Size.** Gate on the smallest configuration that exhibits the effect, and keep the full configuration as the night
  confirmation. Nothing is shrunk until past data shows the smaller configuration reaching the same verdict:
  `CLAUDE.md`'s "a minimal repro can be minimal in exactly the dimension that hides the bug" applies with full force.
- **Band:** re-running a failed gate costs only its unfinished cells; a re-run against an unchanged reference skips the
  reference entirely.
- **Kill:** judged per gate.
- **Per gate, 2026-09-28:**
  - **`TestPrefillGateReference`** (the longest, 168 min): **DONE** through TE6(a).
    - (a) Each prompt is stored atomically as it finishes, and a rerun computes only the missing ones.
    - (b) The reference is a content-keyed lookup.
    - Its consumers, `TestPrefillGateVsReference[CUDA]` (49 min), no longer rebuild or hand-check references.
  - **`TestMoEExpertMajor_endToEnd`** (113 min): **not resumed, on purpose.** It is a timed A/B over interleaved pairs.
    Resuming after a restart would pool pairs from different sessions, and `CLAUDE.md`'s same-session rule exists
    because that drift corrupts ratios. Its levers are TE3 (`power.py` sizing of the pairs) and TE4, and it runs on
    the night queue.
  - **`TestA3MoEExclusionIsMeasured`** (108 min): a measurement that decides nothing, by its own doc. Its timing half
    is the same case as above. Its cosine half is deterministic and could use the cache, which is worth it only if the
    measurement is re-run.
  - **(c) Right-sizing is not done for any gate.** It needs past data showing a smaller configuration reaching the same
    verdict. TE12's `power.py fidelity` is the tool for the fidelity gates.

### TE9 — One timed run per box *(small; by day)*

- **Why.** "A process collision invalidated the first launch" (a September record); Aug 27's contaminated sweep, which
  is why `gate_cell_idle()` exists at all. `night.py` serializes its own queue, but nothing stops a daytime smoke,
  another session's harness, or a timed job in a second checkout from overlapping it.
- **Build.** Every timed harness (`bench_peer*.py`, a helper for `run-*.sh`, `night.py` jobs) takes an exclusive
  `fcntl.flock` on `~/.goinfer-timing.lock` for the length of its run. A second timed run waits or refuses, naming the
  holder (pid, command, start). The idle gate checks the lock first.
- **Band:** void-by-collision records → 0.
- **DONE 2026-09-28.** `scripts/timing_lock.py` holds an exclusive `flock` on `~/.goinfer-timing.lock`.
  - **Where it is taken:** `bench_peer.py`, `bench_peer_prefill.py` and `bench_peer_transcript.py` take it at
    `__main__`, before `preflight()`. `night.py` takes it per job; a daytime holder is waited for up to
    `NIGHT_LOCK_WAIT_MIN` (default 30), then the job is skipped with the holder named. `timing_lock.py run -- CMD`
    wraps any other timed step in a `run-*.sh`.
  - **Behaviour:** a contender refuses (exit 75 from the CLI) with the holder's pid, command, start and label, or waits
    with `BENCH_LOCK_WAIT` / `--wait`. The holder's children inherit it through `GOINFER_TIMING_LOCK`, so a night job
    that runs a `run-*.sh` that runs `bench_peer.py` takes it once. The kernel drops it when the holder dies, so it
    cannot go stale.
  - **Proof, `scripts/test_timing_lock.py` (7 real-process cases):** refusal naming the holder, a foreign token not
    bypassing, children inheriting through nested wrappers, waiting until release, SIGKILL freeing it, and `status`.
    A shared-lock mutant turned 5 of 7 red.
  - **End to end:** a `night.py` job's child inherits the lock. A job blocked by another holder is skipped after the
    wait, with the holder named and its command not run. All three harnesses refuse in ≤ 0.3 s, before `preflight()`
    and without writing output.
  - **Scope:** the lock is cooperative, so a hand-typed `go test -bench` does not take it. TE1's named-process check is
    what sees those.

### TE10 — Parity staleness keyed on what each family executes *(conditional)*

- **Mechanism.** Record, per family, the code blocks its load and forward execute: Go block coverage from the family's
  tiny golden (`go test -coverprofile` per golden, with the tags per backend). A diff that touches no executed block of
  family F leaves F fresh. File-level fallbacks stay for non-Go inputs (a `go.mod` or aikit bump stales everything, as
  now) and for init-time code. Ekstazi (ISSTA 2015) is the file-level form of this; Go's block coverage makes it finer
  at no cost.
- **Equivalence (zero-miss replay).** Replay September's 144 commits. Every commit that later proved numeric for a
  family must have been flagged stale for it: a refresh that refused, or a T3 that moved its numbers. **Any miss kills
  the item.**
- **Band:** the median stale set per `core` commit falls from 36 families to ≤ 5.
- **Conditional:** only if TE0 shows parity bookkeeping is still a material cost after TE6. A refresh itself is ~4 min;
  the cost is its 90 interruptions, and TE6(b) may absorb most of them.

### TE11 — Write the new shape down, and re-count *(closes the campaign)*

- Update `CLAUDE.md`'s Run-budget section and the pre-registration template. Every gate names:
  - its tier: quick / day / night / release;
  - its instrument, and why that is the cheapest one that resolves the bar (TE3, TE5);
  - its stopping rule (TE4);
  - its estimated cost.
- The release tier — the full T3 sweep and the peer matrix, as `RELEASING.md` has them — does not change.
- Re-run TE0 and report before/after by class.

### TE12 — Prompt budgets sized per criterion: the fidelity gates first *(analysis done 2026-09-28; applies to the next gate that uses the protocol)*

- **The protocol.** The prefill and decode fidelity gates share one design, in eleven test files (19 test functions):
  10 prose prompts per cell × a K ladder × 64 scored positions per prompt, on two models, candidate against exact
  against a reference. Each verdict combines three criteria:
  - **A, hard flips:** candidate ≤ exact + 2√exact;
  - **B, agreement:** candidate ≥ exact − 1 pt — and in the CUDA form, the candidate also wins in at least half the
    prompts;
  - **C, KL:** in the CUDA form, mean ≤ 1.1× exact; in the Metal pooled form, ≤ exact with no margin, lower in at least
    half the prompts, under a ceiling.

  The 10 is the same in every one of these gates, whatever the criterion.
- **What each criterion needs.** Script: `prompt_power.py` beside the census, over the 42 logged cells with 10 prompts
  (11 logs). Its output is `prompt-power-2026-09-28.txt`. Prompts needed at 80% power when the arms are truly equal,
  at the gates' own margins:

  | criterion | prompts needed, median / p90 | cells needing more than 10 |
  |---|---|---|
  | KL, mean ≤ 1.10× exact | 10 / 24 | 20 of 42 |
  | agreement, ≥ exact − 1 pt | 32 / 72 | 38 of 42 |
  | wins in at least half the prompts (CUDA B) | no N is enough | — |
  | KL ≤ exact, lower in at least half (Metal C) | no N is enough | — |
  | a worst-case check ("no prompt regresses") | 30 prompts to bound a 10% bad-prompt rate | 10 bounds it at 26% |

  - The KL figure is computed on the gates' own statistic, the ratio of the arms' means. Per-prompt ratios explode
    where the exact arm's KL is near zero: one logged prompt reads 0.0021 → 0.0143, a 6.9× "ratio" on a 0.012
    difference.
  - The two rows with no margin are coin flips for equal arms at any N: an equal arm wins at least half of 10 prompts
    62% of the time, tending to 50% as N grows.
  - So 10 prompts is about right for a typical KL cell, too few for half of them, and about a third of what agreement
    needs. For the two no-margin criteria, prompts buy nothing. **The imbalance is in the criteria more than in the
    count.**
- **The verdicts agree.** 66 verdict lines were printed with their flags:
  - CUDA form: 19 verdicts, 6 not passing. KL passed in all 19; every failure came from the flip-count or agreement
    criteria.
  - Metal pooled form: 47 verdicts, 19 not passing, 17 of them on the no-margin KL criterion.

  The criteria that fail are the ones 10 prompts cannot resolve.
- **Stopping early saves little, and is order-sensitive.** A conservative group-sequential rule on the KL criterion
  (looks after 4, 7 and 10 prompts, one-sided α of 0.05/3 per look) used 72% of the prompts in recorded order. Over
  2,000 random prompt orders per cell, it gives a PASS the fixed-10 reading does not in 0.99% of orders. These
  concentrate in four S K=1024 cells, at 7–13% of their orders — the class of failure that killed TE4's SEQ-v1, in
  exactly the heterogeneous cells where early stopping matters. **Not proposed** until it has TE3's registry noise floor
  and a held-out validation; expect ~25% saved at best.
- **Change.** This is pre-registered with the next gate that uses the protocol; past verdicts are not re-read.
  - **Every "no worse than" criterion gets a margin.** Otherwise it is a superiority test, and the pre-registration
    says so.
  - **Agreement and hard flips are pooled** across the gate's cells at the same margins — five K cells × 10 prompts is
    50, above the median 32 — with per-cell values reported and never a per-cell veto (`task-prefill-gap.md` §3.2).
  - **The KL criterion is sized by `scripts/power.py`** from the registry's across-prompt spread. Some cells get more
    prompts, not fewer.
  - **A worst-case claim is a separate criterion.** It states the bad-prompt rate it must exclude and gets that N.
  - **Prompts are fixed, snapshotted and hashed.** Set B already is. Set A's live-document prompts left 1 to 4 of 10
    prompts scored against logits for different text (`prefill-ref-identity-2026-09-26.md`), so TE6(a)'s reference
    key includes the prompt hash.
- **Where time can come back: positions, not prompts (measure first).** Per-prompt agreement over 64 positions moves
  in 1.56-point steps. If its noise is mostly within-prompt, longer continuations — cheap decode steps — buy agreement
  precision more cheaply than more prompts, each of which costs a K-token prefill on three arms, reference included.
  The logs keep only per-prompt aggregates, so the next run of the protocol logs per-position agreement and KL once.
  The variance split then sets positions per prompt per K by the two-stage sampling optimum: positions per prompt
  ∝ √((prompt cost ÷ position cost) × (within-prompt variance ÷ between-prompt variance)).
- **The rule for other prompt-heavy tests:**
  - **Bit-identity tests:** coverage, not count — one prompt per shape that exercises a different path (a single
    token, tile and chunk boundaries, maximum depth).
  - **Speed:** the prompt text barely matters, since decode speed depends on depth. The noise is run to run and per
    engine — TE2(b), TE4.
  - **Accuracy and evaluation runs:** a binomial sample size from the decision band (`power.py binomial`, as D6a
    did).
  - **Sweeps:** an A/A arm first, so the noise is known before the bar is set.
- **Band:** fidelity cells that end undecided or failing on an unresolvable criterion → ~0. Prompt counts move both
  ways by criterion; the time comes back from gates that decide the first time, instead of re-runs and amendments.
- **Kill:** a gate registered under these criteria still ends undecided at its computed N.
- **BUILT 2026-09-28: the criteria and the sizing, ready for the next gate.** The gates' own inline criteria are not
  touched, because past verdicts are not re-read.
  - **[`internal/fidelity`](../../internal/fidelity/fidelity.go):**
    - `NonInferior` refuses a margin ≤ 0, since margin-free is a superiority test and `Superiority` is its own named
      function.
    - Hard flips are pooled under the Poisson bound, as today.
    - Agreement passes if its one-sided lower bound, clustered by prompt, is ≥ −margin points. KL passes if the
      delta-method upper bound on the ratio of means, clustered by prompt, is ≤ 1 + margin.
    - Per-cell values are reported, never used as a veto. `WorstCasePrompts`, `PromptSetHash` and `WritePositions`
      (the per-position log) complete it.
  - **Tests, calibrated on simulated gates** (5 cells × 10 prompts pooled, correlated arms with 3% discordance):
    - equal arms pass 87%;
    - a candidate at 1.25× KL passes 0%;
    - one 4 pts worse on agreement passes 0%;
    - the old no-margin KL rule passes equal arms about half the time (a test, so the reason is executable).
    
    A no-margin mutant dropped equal arms to 4% and turned the calibration test red.
  - **`scripts/power.py fidelity`** sizes both criteria with the same paired one-sided t test, from given spreads or
    from a `WritePositions` log. `power.py worstcase` gives 29 prompts to exclude a 10% bad-prompt rate, and says 10
    prompts exclude 26%. Its 26 tests pass, and a mutant turned them red.
  - **Pre-registration template for the next gate on the protocol:**
    - **Decision set:** the (model, K) cells, pooled. Per-cell values reported, no per-cell veto.
    - **Criteria:** `fidelity.NonInferior` with `AgreeMarginPts = m_a` and `KLMargin = m_k`, one-sided α 0.05, and
      hard flips under the pooled Poisson bound. No "wins in half" criterion. `Superiority` only if superiority is the
      claim.
    - **Prompts:** N from `power.py fidelity --from-positions <the last run's positions log>` (or `--agree-sd` /
      `--kl-cv` from the registry), pooled. The prompt set is fixed, and its `fidelity.PromptSetHash` is recorded.
    - **Worst case, if claimed:** the excluded rate `r`, with N from `power.py worstcase --rate r`.
    - **Log:** `fidelity.WritePositions` output, kept, so the positions-per-prompt split can be sized next time.
  - **What sets agreement's N is the discordance between the arms,** which the positions log measures. With
    independent arms (~25% discordant positions) equal arms passed 28% at 50 prompts. With the 3% discordance the
    gates' recorded d/N suggests, they passed 87%. Sizing therefore uses a real log, not an assumption.

---

## 3. Order, and what one gate looks like after

**TE0 → TE9 → TE5(a) → the analysis trio → TE12 → TE1 → TE3 → TE7 → TE6 → TE2(a) → TE8 → TE10 (if still warranted) →
TE11.**
The analysis trio is TE4's replay, TE2(b)'s variance split and TE5(b)'s concordance: one day's work over existing data.

Worked projection on this morning's L1 gate 6 — pass 1 was 40.5 min (29 of them waiting, 11.4 in cells, 2.9 of those
Ollama), and pass 2 was another 25.7:

| step | per pass | the whole gate |
|---|---|---|
| today | 40.5 + 25.7 min | 66 min (two passes) |
| TE1 (≈20 s gate check per cell) | ~14–15 min | ~28–30 min |
| + TE5(a) (two arms instead of three) | ~10–11 min | ~21 min |
| + TE2(a) (goinfer start-up cut) | ~8–9 min | ~17 min |
| + TE4 (one counterbalanced pass that stops on a clear result) | — | **~7–12 min** |

That is the size of the prize per gate, roughly 5–9×, **if the bands hold**. TE0's re-census says whether they did.

---

## 4. Prior art

**In this repo.**
- [`task-ci-speed-2026-09.md`](../completed/task-ci-speed-2026-09.md): C0's measure-first discipline; C7's cache
  finding, which TE7(b) re-diagnoses; and §4, which scoped this campaign's subject out.
- [`task-verification-surface-audit.md`](task-verification-surface-audit.md): the constraint.
- `queue-engineering.md` B19: per-package buffering, and the one-package-per-invocation rule that TE7 keeps.
- `cmd/gate` (census, selector, composition, mutation); `GATE_RUN` (a123a6ef) scoping the parity sweep to a subset.
- `scripts/refresh_parity_hashes.sh`: the goldens-gated refresh that TE6 and TE10 build on rather than replace.
- The D6a amendment (2852eaa1): sample size from a stated standard error. TE3 generalizes it.
- The L1 owner amendment of 2026-09-28 09:05 in
  [`task-cpu-decode-peer-gap-2026-09.md`](task-cpu-decode-peer-gap-2026-09.md): pass 2 skipped at a stated margin.
  TE4 formalizes it.
- `bench_peer.py` `gate_cell_idle()`: why the per-cell gate exists. TE1 keeps its refusal.
- `CLAUDE.md` on `bench_compare.sh`, "still the right tool for goinfer-vs-goinfer work": the in-process lane TE5(b)
  builds on.
- `scripts/night.py` and the Run-budget rule (c9d8ea04, c4e3aec2).

**Outside.**
- Kalibera & Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013): how many repetitions to spend at which
  level — process restarts vs iterations — for a target precision. TE2(b).
- Georges, Buytaert & Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007): intervals over
  invocations rather than best-of-N.
- Mytkowicz, Diwan, Hauswirth & Sweeney, "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009):
  environment and layout bias. The caution under every cross-build comparison (TE5, TE6).
- Wald's SPRT (1945); O'Brien & Fleming's group-sequential boundaries (1979); Johari, Koomen, Pekelis & Walsh, "Peeking
  at A/B Tests" (KDD 2017): valid inference under continuous monitoring. TE4.
- Gligoric, Eloussi & Marinov, "Practical Regression Test Selection with Dynamic File Dependencies" (ISSTA 2015),
  i.e. Ekstazi: TE7 and TE10. Machalica et al., "Predictive Test Selection" (ICSE-SEIP 2019): needs a long failure
  history; noted, not proposed.
- Bulej et al., "Duet Benchmarking" (Empirical Software Engineering, 2020): run both variants at once so interference
  hits both. Considered for the busy daytime Mac and **not proposed** — decode is memory-bandwidth-bound, so two arms
  sharing the bus measure a contended ratio, not the solo one.
- The Go world's versions of the same problems: `perflock` (serialize benchmarks on a box; TE9), `benchstat`
  (interval statistics), and `go help test` for the cache semantics TE7 relies on.

## 5. Not in this campaign

- **The measurement standards.** Unchanged (§0).
- **The release tier.** The full T3 parity sweep and the peer matrix stay as `RELEASING.md` has them.
- **CI.** Done (C0–C9).
- **Hardware.** A second GPU or a dedicated quiet bench box would change TE1's picture. That is a purchase decision, not
  a campaign item.
- **Considered and not proposed:**
  - duet benchmarking (§4);
  - timed gates on the Mac while the owner works (the Run-budget rule already says no);
  - lowering a bar so a gate finishes sooner.

## 6. First moves — by day, nothing timed

1. **TE0:** run `census.py` on nobara (its `~/goinfer-bench` logs are not in the Mac's view), and write the transcript
   miner on each machine.
2. **TE9:** the timing lock.
3. **TE5(a):** the line in `CLAUDE.md`, and the `run-*.sh` template drops the peer arm when the decision is new ÷ old.
4. **The analysis trio:** TE4's replay, TE2(b)'s variance split and TE5(b)'s concordance table, over existing JSONs and
   records.
5. **TE7(b):** the `decoder` cache re-diagnosis, two back-to-back runs with `GODEBUG=gocachetest=1`, ~10 min.
6. **TE12:** the criteria changes go into the next pre-registration that uses the fidelity-prompt protocol, and that
   run logs per-position agreement and KL once.

**Queue for the first night:** TE1's A/A + mutation run, pre-registered before it is queued.

<!-- doc-reviewed: 2026-09-28 -->
