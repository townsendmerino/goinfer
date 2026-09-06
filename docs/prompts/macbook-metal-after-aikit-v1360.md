# Task (macbook-arm64): re-validate the Metal side after aikit v1.36.0

> **For:** Claude Code on `macbook-arm64`, in the goinfer checkout (with sibling `aikit`).
> Written 2026-09-06 from `nobara-pc`. Three items, independently droppable, ordered by what is
> most owed. **Everything here is a measurement task; nothing needs new features.**
>
> Read first: `docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md` — the amd64 half of this
> work, already done. It carries the reasoning, the failures and their dispositions, and the
> explicit list of what it did NOT cover. **This prompt is the arm64 half of exactly that.**

## Why now

`testdata/parity_manifest.json` mixes a top-level `aikit_version` into every family's `deps_hash`.
It had read **`v1.19.0` against a `go.mod` of `v1.36.0`** — hand-typed, seventeen versions stale —
so the staleness gate could not fire on an aikit change and had been vouching for nothing. It has
now been re-armed at v1.36.0 by a full sweep **on amd64**, whose proof block reads
`arch=amd64` and says so.

**No arm64 evidence exists for that bump.** goinfer's Metal path (dense residency, int8int8) and
aikit's NEON kernels are a different code path from anything the amd64 sweep touched. Until this
runs, the Metal families' green means "the hash matches on a Linux box".

---

## Item 1 — the Metal-side parity sweep (the owed one)

```bash
mkdir -p ~/gate-logs/parity-metal-2026-09-06        # MUST exist first
cd <goinfer>
EMIT_MANIFEST=1 go run ./cmd/gate parity -logdir ~/gate-logs/parity-metal-2026-09-06
```

**Traps, all of which cost real time on the amd64 run:**

- **`-logdir` must exist before launch.** Its absence does not error — it fabricates failures
  ("65 BLOCKERS", "10 FAIL cells"). Hit twice.
- **The logdir must be OUTSIDE the worktree.** A gate log inside it turns PASS into INCONCLUSIVE.
- **Not `/tmp`.** A verdict nobody can re-read is not evidence.
- **Detach it** — `launchctl submit` on macOS, not `setsid` (which does not exist there), and
  append `; launchctl remove <label>` as the job's own last action or launchd restarts it and
  overwrites the log before you read it. That has destroyed a completed run's trace before.
- Expect **~3 hours** on amd64; arm64 is unmeasured, so report elapsed and current cell rather
  than an ETA.

**Read `MODELS`/asset preflight output before trusting any blocker count** — a skip for a missing
asset is reported as a blocker, and that count is about the assets, not the tree.

**Deliverable:** the merged manifest rows with `arch=arm64` in the proof block, plus a companion
`docs/measurements/parity-sweep-metal-2026-09-06.md` in the same shape as the amd64 one —
including a section on what it did NOT validate. **If a family emits no `PARITY_ROW`, do not let
its `deps_hash` refresh stand as validation**; the amd64 run found `kimi_k2` doing exactly that,
and naming it is the point.

**Expected outcome, stated in advance so a surprise is legible:** amd64 found seventeen versions
of aikit drift cost nothing — 15 rows, argmax 100% everywhere except `qwen3_5_moe`'s known
bandwidth trade. If arm64 disagrees, that disagreement is the finding and it is worth more than
the sweep.

## Item 2 — C3, the Metal consumer window

Owed since v0.16.0's aikit bump, and now compounded by v1.36.0. It is a **manual device gate**;
`RELEASING.md` §C1-M is the authority on what "green" means for it. Run it and record the verdict
verbatim — including the part §C1-M says a green does *not* cover.

## Item 3 — install one agent CLI (10 minutes, unblocks a future run)

The 2026-09-06 cold-user run could not test the scenario-B agent sub-test at all: aider, opencode,
cline, `llm` and codex were all absent, and `continue` matched a shell builtin — a false positive
worth knowing about.

`nobara-pc` now has **opencode 1.18.29** in a contained prefix
(`npm install --prefix ~/.local/opt/opencode opencode-ai`). Do the same on the Mac, or pick
another and **name it in `docs/task-first-hour.md` §1**, which is where the protocol records which
CLI a run used. Note `aider` does **not** install on a Python 3.14 host — pip backtracks to
`aider-chat` 0.16.0 and its pinned `multidict` fails to build.

---

## Rules that are not optional here

- **Every timed run reads its checkpoint from `~/models` on this machine.** Not `/Volumes/…` —
  that is the SMB mount of a 5400 rpm SMR disk, it does not error, and it returns a plausible
  wrong number. `models-pull <name>` first. `CLAUDE.md` is the authority.
- **A SKIP IS NOT A PASS.** `go test` prints `ok` for a package whose tests all skipped. Confirm
  with `-v` and read for `--- PASS`.
- **Negative results get committed with the same care as wins.** If the Metal sweep finds drift
  the amd64 one did not, that is the most valuable thing this prompt can produce — do not
  re-baseline anything to make it green.
- Report **elapsed / current cell / done-vs-total**, not an ETA, for anything whose cost is
  dominated by checkpoint loads rather than test count.
