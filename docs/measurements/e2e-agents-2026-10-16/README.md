# End-to-end agent run on main, Mellum2.1 on the 8 GB card (queued 2026-10-16)

Closes v0.23.0's "`connect` not run end to end" caveat and gives v0.24.0's agent numbers. A measurement, not a gate: it has no bar, and
nothing here is quoted as a result until the morning read grades it against these definitions.

**Not graded against a pre-registered bar. Not a peer comparison.** Cross-session comparisons below are context.

## What runs

Claude Code, Codex and opencode, one agent per `serve` process, each configured by `serve connect <agent> -write` into its own isolated
HOME (`matrix/run_cell.py` does this; the written file is not edited, and `tell_window`, the I1-v2 measurement edit, is not used). The
sandboxed I3 harness (`matrix/`, bwrap + socat, sandbox self-test first), tasks `read`, `search`, `edit` (`matrix/tasks.py`, the set the
2026-10-15 opencode run used), one run each.

| config | agent | model and serve | ctx |
|---|---|---|---|
| `e2e-opencode` | opencode | Mellum2.1 int4mix, `--moe-cache-experts`, f32 KV, windowed KV | 16,384 |
| `e2e-codex` | Codex | the same | 16,384 |
| `e2e-claude-code` | Claude Code | the same (the I3 `mellum-32k-cc` regime) | 32,768 |

Serve is built from main at `968c50d6` (OP2b step 2a merged; no behaviour change in it) by `pin-e2e.sh`, which copies the harness beside
the binary, so the tree may move before tonight. Fit check by day (the server's own banner, under the timing lock): both contexts admit,
1 KV conversation, 0.6 GB (16k) / 1.0 GB (32k) of KV, the C′ cache capped to 57 and 52 of 64 slots with the desktop session holding its
context (the 2026-10-15 run's banner said 45; slot count follows free VRAM at load).

## Budget and stops

About 90 min: a 70-minute deadline after which no new run starts, a per-run limit of 15 min (opencode, Codex) or 20 min (Claude Code,
whose first request is about 18.5k tokens, 451 s cold in job B), worst case 90 min. One `matrix.py` invocation per agent: an agent that
cannot start writes no record and `job.log` says `AGENT FOR <config> COULD NOT START`; the others still run.
`run-e2e.sh` asserts `records.jsonl` is non-empty, so a refused start cannot read as a success.

## Figures per agent (`e2e_metrics.py`)

| figure | definition |
|---|---|
| tasks passed | the harness's own grade (`passed` in `records.jsonl`: the answer match or the mechanical check) |
| session-start TTFT | serve's `ttft=` of each run's first request that carried tools (a title request is a side request and is excluded) |
| follow-up TTFT | the same for every later request that carried tools: n, median, p90, max |
| cold restarts | follow-ups with `reused=0` and no `snap=` (nothing of the previous prompt reused) |
| compactions | the record's own count (the agent's) and the wire's |
| snapshot restores | lines with `snap=<rows>`: count and total rows |
| carried-id fallbacks | `ids_fallback=` reasons summed; `-` on routes that carry no ids (Chat Completions, Messages) |
| fast-alias requests | lines with `fast=1` |

A run's wire requests are aligned to its configuration's serve-log lines by completion tokens (and prompt tokens where the wire has them);
a run that does not align is counted as `unaligned` and left out of the split, never guessed.

## Comparisons (all cross-session)

`run-e2e.sh` appends the extractor's output on: opencode of 2026-10-15 23:05 (`2a0d11bb`, same tasks, same model, same ctx; the closest
pair), I3 job B of 2026-10-13 (before carried ids and the fast alias; five tasks per agent), and the I3 after run's Codex arm with carried
ids on (2026-10-15). Different builds, days and thermal states: a ratio between them is not a ratio
(`docs/measurements/noise-registry.md`, TE3).

## Reproduce

    REV=<commit> B=$HOME/goinfer-bench/e2e-agents-2026-10-16 bash pin-e2e.sh      # by day, no GPU
    DRY_RUN=1 OUT=<empty dir> B=... bash $B/run-e2e.sh                            # prints the three matrix commands
    python3 scripts/night.py add e2e-agents-v024 --est 90 --by "<who>" --doc docs/measurements/e2e-agents-2026-10-16/README.md -- \
        env B=$HOME/goinfer-bench/e2e-agents-2026-10-16 bash $HOME/goinfer-bench/e2e-agents-2026-10-16/run-e2e.sh

Output: `$B/e2e-out/` (`records.jsonl`, `wire-*.jsonl`, `logs/serve-*.log`, `runs/`, `e2e-metrics.md`, `job.log`, `provenance.txt`).
