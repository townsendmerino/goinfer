---
title: "An upgrade that can't change your answers"
area: "Reproducibility"
order: 15
summary: "gate identity builds two revisions of goinfer, compares their full logits byte for byte, and says IDENTICAL or shows the first difference."
stand: "Before you trust a new build, you can ask whether it computes exactly what the old one did. goinfer has a command that answers that with bytes, not with a similar-looking reply."
measured: 2026-09-28
reviewed:
facts:
  - {label: "command", value: "`gate identity <old> <new>`"}
  - {label: "compares", value: "full logits, prefill and each greedy step, as raw bytes"}
  - {label: "verdicts", value: "IDENTICAL, WITHIN TOLERANCE, DIFFERENT, NOT RUN"}
  - {label: "scope", value: "one machine and one backend at a time"}
doesnt:
  - title: "It doesn't hold across machines or operating systems."
    text: "The report ends by saying identity is per machine and per backend. The CPU reference is bit-identical within one architecture, not across: the Go compiler fuses `x*y + z` into one rounding on arm64 and not on amd64's default build, so 93% of the logits differ between the two, by amounts the policy doc measured as far below what changes a token."
  - title: "It doesn't cover CUDA."
    text: "The task record lists no CUDA backend for this command; `-backend` accepts cpu, metal or webgpu. A CUDA build has to be compared on a machine that has the GPU, with a tool that supports it."
  - title: "It doesn't say the answers are right."
    text: "Identical to the old build means the same as the old build, including any mistake it had. Whether the old build matches the model's own reference is a separate check (\"Checked against the reference\")."
  - title: "It compares a small slice."
    text: "It runs each family's parity prompt for a few generated tokens (8 on the tiny fixtures, 32 on real checkpoints by default), through the public decoder API. A device-side greedy-argmax path is not what it compares, and 7 families have no tiny fixture and come back NOT RUN."
  - title: "It has not yet been used to inherit a real validation."
    text: "The owner allowed identity to stand in for a re-validation on 2026-09-28, but the record says it is not yet used on a real row. Only the CPU, on the same architecture the row was validated on, is eligible."
figures:
  - {text: "120", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "38", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "80", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "19", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "30", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "43–46 s", source: docs/tasks/task-test-efficiency-2026-09.md}
  - {text: "0.897", source: CLAUDE.md}
  - {text: "93%", source: docs/parity-coverage-policy.md}
sources:
  - cmd/gate/identity.go
  - cmd/gate/identity_compare.go
  - docs/tasks/task-test-efficiency-2026-09.md
  - docs/parity-coverage-policy.md
  - CLAUDE.md
---

## The problem

You have a release that was checked carefully against a reference model, and a change since then that was meant to change nothing: a faster loop, a reordered sum, a refactor. How do you know it changed nothing?

The usual check is that a few prompts still give the same replies. That is weak. A model can keep its argmax, the single most likely next token, while its logits, the scores behind every token, have drifted. In this project's own history, two bugs in one model port held the right argmax while the logit cosine against the reference was 0.897. A prompt that "still works" would have passed both.

## What goinfer does

`gate identity <old-rev> <new-rev>` builds both revisions, runs the same prompts through each, and compares every logit as bytes. For each model family it prints one verdict. This is the shape of the report, from the code's format strings (the revs and names here are placeholders, not a recorded run):

```sh
go run ./cmd/gate identity <old-rev> <new-rev> -backend cpu
# == determinism: the new build (<sha>) run twice, separate processes, on cpu ==
#   N cell(s) byte-identical to themselves, 0 not, 0 unchecked
# == identity: old <sha> vs new <sha>, per family ==
#   IDENTICAL   <family>   <k>/<k> cell(s) ran   validation inheritable from <sha>
#   DIFFERENT   <family>   <k>/<k> cell(s) ran   goes to its reference gate
#       first differing logit: prompt 0 step 0 (0 = prefill) index ...; max |diff| ...
# == verdict ==
#   families ...: ... IDENTICAL, ... WITHIN TOLERANCE, ... DIFFERENT, ... NOT RUN
```

The command exits 0 when every family that ran is identical, 1 when any is different (or the CPU is not deterministic), and 2 when nothing ran. On the MacBook, `identity HEAD~1 HEAD` on the CPU with the tiny test fixtures took 43–46 s, with 30 families IDENTICAL and 7 NOT RUN. A change that should matter shows up: on the real 0.5B demo checkpoint, the earlier CPU-decode change known as L1 (`3cd62e6d` to `5c85f7c0`) comes back DIFFERENT from prefill logit 0.

## How it works

Two temporary git worktrees, one per revision, hold the two builds, and a small program built into each dumps the full logits as raw 32-bit floats: the prompt's prefill, then one row for every greedy step. Each family runs in its own process in the order new, old, new again. The command then compares bytes.

Full logits, not argmax, because argmax hides drift. Byte equality is also the only test with no threshold to argue about. Two details keep it honest. Every `GOINFER_*` environment variable is removed from both sides, so the only difference between them is the code, and each side's decode path is printed and a difference flagged. And a GPU cell that fell back to the CPU is reported NOT RUN, so a CPU result is never filed under Metal.

The new build must also match itself first. If a cell is not byte-identical run to run, it is compared against the old build in a tolerance mode (twice its own run-to-run difference, with argmax and tokens exact) and reported WITHIN TOLERANCE, never IDENTICAL.

## What was measured

The kill criterion for the tool was written into the task record on 2026-09-28, before it was built: run-to-run nondeterminism on the CPU. It did not fire. No backend needed the tolerance mode.

<table>
<thead><tr><th>Backend</th><th>Cells byte-identical run to run</th><th>Families</th><th>Date</th></tr></thead>
<tbody>
<tr><td>CPU</td><td>120 of 120</td><td>not stated</td><td>2026-09-28</td></tr>
<tr><td>Metal</td><td>38 of 38</td><td>19</td><td>2026-09-28</td></tr>
<tr><td>WebGPU</td><td>80 of 80</td><td>30</td><td>2026-09-28</td></tr>
</tbody>
</table>

The figures come from the task record, `docs/tasks/task-test-efficiency-2026-09.md` (TE6(b)), and match the commit message that added the command. The record does not name the machine for each backend; only the timing run above names the MacBook. It also lists three equivalence tests, run on scratch clones: a comment-only edit gives all IDENTICAL, a one-ulp change in one residual path flags exactly the two families that use it (cohere and cohere2), and an injected nondeterministic dumper is reported and exits 1.

## Use it

- `go run ./cmd/gate identity <old-rev> <new-rev>` from the repository root. The two revisions can sit before, between or after the flags.
- `-backend cpu|metal|webgpu` (default `cpu`). Identity is per backend.
- `-assets tiny|real` (default `tiny`, the forward goldens' fixtures; `real` uses small checkpoints under `~/models`).
- `-families a,b` and `-only <asset name>` narrow the run; `-quant q,...` and `-steps N` change what is compared.
- `-keep` keeps the dumps, binaries and logs. `-timeout` bounds each dumper process (default 15 minutes).
- `-record <file>` writes candidate parity-manifest rows for families that qualify. Without it the manifest is never written. The rules are in `docs/parity-coverage-policy.md`.
