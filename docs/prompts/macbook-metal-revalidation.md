# Task (macbook-arm64): re-validate the Metal side, and read the fit guard before you start

> **For:** Claude Code on `macbook-arm64`, in the goinfer checkout (sibling `aikit`).
> Written 2026-09-06 from `nobara-pc`, where the amd64 half is done. Four items, independently
> droppable, ordered by what is most owed.
>
> **Read first:** [`docs/measurements/parity-sweep-aikit-rearm-2026-09-06.md`](../measurements/parity-sweep-aikit-rearm-2026-09-06.md)
> — the amd64 half, with its failures, their dispositions, and an explicit list of what it did NOT
> cover. This prompt is the arm64 half of exactly that.

## READ THIS BEFORE RUNNING ANYTHING — a new guard will change what your box can load

`decoder/fitguard.go` landed today. Before allocating, it prices the checkpoint and **refuses** if
it needs more than **70% of physical RAM**. On a 16 GB Mac that budget is **11.2 GB**, so loads
that used to succeed by paging will now fail with a message naming `-stream-weights`.

**That is the guard working, not a regression** — but it will look like one in a gate log, so
know it before you burn an hour. It already happened here: `TestLlama4Real_gate` went red because
Scout-109B needs ~62.7 GB at int4 against this box's 62.7 GB. The fix was **not** to loosen the
guard; the test now sets `GOINFER_NO_FIT_GUARD=1` with the arithmetic in a comment. Re-run that
way it passes — while driving swap from ~4 GB to **35.4 GB**, which is what "passing by paging"
looks like when you finally measure it.

Expect more of this on 16 GB than on 62.7 GB. For each refusal, decide deliberately:

- the model genuinely does not fit → the gate opts out with `GOINFER_NO_FIT_GUARD=1` **and a
  comment carrying the numbers**, so a future green cannot be misread as "this fits"; or
- the estimate is wrong → that is a real bug in `quantBytesPerElem`/`estimateGGUFWeightBytes` and
  is worth more than the gate. **Check before assuming** — on amd64 the estimate was verified
  against the file's own tensor element count, and it was right.

**One arm64-specific fact that will surprise you:** `int4` costs **more** resident RAM than
`int8int8` on Apple Silicon — 1.2500 vs 1.0156 bytes/element — because the NEON row4 repack keeps
a second buffer beside the canonical nibbles. int4 is still *faster* there; the trade is "faster
and larger". See [`docs/quantization.md`](../quantization.md). If a fit refusal surprises you at
int4, that is why.

## Item 1 — the Metal parity sweep (the owed one)

goinfer now pins **aikit v1.37.0** in all five modules. The manifest's `aikit_version` tracks it,
and the gate was re-armed by a full sweep **on amd64** — its proof block says `arch=amd64`. No
arm64 evidence exists for that pin. Metal residency and the NEON kernels are a different code path
from anything the amd64 sweep touched, so until this runs, the Metal families' green means *"the
hash matches on a Linux box"*.

```bash
mkdir -p ~/gate-logs/parity-metal-2026-09-06        # MUST exist first
cd <goinfer>
EMIT_MANIFEST=1 go run ./cmd/gate parity -logdir ~/gate-logs/parity-metal-2026-09-06
```

**Traps, each of which cost real time here:**

- **`-logdir` must exist before launch.** Its absence does not error — it *fabricates* failures
  ("65 BLOCKERS"). Hit twice.
- **Outside the worktree**, or a gate log inside it turns PASS into INCONCLUSIVE. **Not `/tmp`** —
  a verdict nobody can re-read is not evidence.
- **macOS has no `setsid`.** Use `launchctl submit -l <label> -- /bin/bash -lc "... ; launchctl
  remove <label>"`. The self-removal is not optional: a submitted job **restarts itself on exit**
  and has silently overwritten a completed run's log before it could be read.
- **A skip is not a pass.** The runner counts a missing asset as a blocker; that count is about
  your `~/models`, not the tree. Read the preflight output first.
- **Models come from `~/models`,** never `/Volumes/…` — that is the SMB mount of a 5400 rpm SMR
  disk, it does not error, and it returns a plausible wrong number. `models-pull <name>` first.

**Expected outcome, stated in advance so a surprise is legible:** amd64 found the pin cost nothing
— 15 rows, 14 families, cosines 0.98988–1.000000, **argmax 100% everywhere** except
`qwen3_5_moe`'s 77.5%, which is a deliberate bandwidth trade bisected to `6d4fc79`, not a
regression. **If arm64 disagrees, the disagreement is the finding** and is worth more than the
sweep. Do not re-baseline anything to make it green.

**Deliverable:** merged rows with `arch=arm64` in the proof block, plus
`docs/measurements/parity-sweep-metal-2026-09-06.md` in the same shape as the amd64 one —
**including a section on what it did not validate.** Specifically: **if a family emits no
`PARITY_ROW`, do not let its `deps_hash` refresh stand as validation.** The amd64 run caught
`kimi_k2` doing exactly that, and naming it is the point.

**Two known-red gates, so you can tell them from your own findings:**

- `TestQwen38GGUF_weightDiff` — `k_proj` cosine 0.997047 against a 0.999 bar. Pre-existing, its
  first-ever execution, reproduces bit-identically at aikit v1.35.0. Not yours.
- `TestSamplingThroughputGate` — **flaky, ~1 run in 4**, on an unchanged tree (measured 3.88–5.19×
  against a 5.0 bar). See `docs/queue-performance.md` P17. If it reds, re-run once before believing
  it — and if it reds *consistently* on arm64, that is new and worth reporting.

## Item 2 — C3, the Metal consumer window

Owed since v0.16.0's aikit bump, now compounded by v1.36.0 and v1.37.0. It is a **manual device
gate**; `RELEASING.md` §C1-M is the authority on what "green" means. Record the verdict verbatim,
**including what §C1-M says a green does not cover.**

## Item 3 — install one agent CLI (10 minutes)

The 2026-09-06 cold-user run could not test the scenario-B agent sub-test at all: aider, opencode,
cline, `llm` and codex were absent, and `continue` matched a shell builtin — a false positive worth
knowing about. `nobara-pc` now has **opencode 1.18.29** via
`npm install --prefix ~/.local/opt/opencode opencode-ai`. Do the same, or pick another and **name
it in [`docs/task-first-hour.md`](../task-first-hour.md) §1**. Note `aider` does **not** install on
a Python 3.14 host — pip backtracks to `aider-chat` 0.16.0 whose pinned `multidict` fails to build.

## Item 4 — only after a goinfer tag lands: cold run 2

`RELEASING.md` pre-flight step 5 now requires a cold-user run on the previous tag. Run 2 is
specified for the **Linux** box, so this is a *nice-to-have* here — but if you do run it on the
Mac, follow `task-first-hour.md` §1 exactly: fresh window, empty directory outside the repo,
published assets only, declared contamination, and **record `--version` for every binary**, which
the first run could not do and is how a Mac asset with no Metal in it reached a release.

---

## Rules that are not optional

- **Negative results get committed with the same care as wins.** If the Metal sweep finds drift
  amd64 did not, that is the most valuable thing this prompt can produce.
- **Report elapsed / current cell / done-vs-total, not an ETA**, for anything whose cost is
  dominated by checkpoint loads. The amd64 sweep took 2h51m; that predicts nothing for arm64.
- **`gofmt -l .` and `staticcheck` are CI gates** and nothing here auto-formats. Run them across
  every module you touch.
- **Check CI after pushing** (`gh run list`), and read the RUN-LEVEL status, not the job list —
  `root-darwin` finishes 5–10 minutes after everything else, so a job list showing six greens and
  one in-progress is a run with no verdict yet.
