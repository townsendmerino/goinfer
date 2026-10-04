---
title: "An upgrade that can't change your answers"
area: "Reproducibility"
order: 15
summary: "One command builds two versions of goinfer, compares every score the model outputs byte for byte, and says IDENTICAL or shows the first difference."
stand: "Before you trust a new build, you can ask whether it computes exactly what the old one did. goinfer has a command that answers that with bytes, not with a similar-looking reply."
measured: 2026-09-28
reviewed: 2026-09-29
facts:
  - {label: "command", value: "`go run ./cmd/gate identity <old> <new>`"}
  - {label: "compares", value: "every output score (logit) for the prompt and each generated token, as raw bytes"}
  - {label: "verdicts", value: "IDENTICAL, WITHIN TOLERANCE, DIFFERENT, NOT RUN"}
  - {label: "scope", value: "one machine and one backend at a time"}
doesnt:
  - title: "It doesn't hold across machines or operating systems."
    text: "The report ends by saying identity is per machine and per backend. goinfer's CPU path is bit-identical within one CPU architecture, not across: the Go compiler fuses `x*y + z` into one rounding on arm64 (such as Apple Silicon) and not in amd64's default build, so 93% of the logits differ between the two, by amounts the project's parity policy measured as far below what changes a token."
  - title: "It doesn't cover CUDA."
    text: "The command has no CUDA backend: `-backend` accepts cpu, metal or webgpu. A CUDA build has to be compared on a machine that has an NVIDIA GPU, with a tool that supports it."
  - title: "It doesn't say the answers are right."
    text: "Identical to the old build means the same as the old build, including any mistake it had. Whether the old build matches the model's own reference implementation is a separate check (\"Checked against the reference\")."
  - title: "It compares a small slice."
    text: "It runs each family's standard test prompt for a few generated tokens (8 on the small test models, 32 on real checkpoints by default), through goinfer's public decoder API. It does not compare the GPU paths that pick the top token on the GPU itself, and 7 families have no small test model and come back NOT RUN."
  - title: "It has not yet stood in for a real re-check."
    text: "On 2026-09-28 the project owner allowed an IDENTICAL result to stand in for re-running a family's reference check, but the record says this has not yet been done for any recorded family. Only the CPU backend qualifies, on the same CPU architecture the family was originally checked on."
figures:
  - {text: "120", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "38", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "80", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "19", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "30", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "43–46 s", source: docs/completed/task-test-efficiency-2026-09.md}
  - {text: "0.897", source: CLAUDE.md}
  - {text: "93%", source: docs/parity-coverage-policy.md}
sources:
  - cmd/gate/identity.go
  - cmd/gate/identity_compare.go
  - docs/completed/task-test-efficiency-2026-09.md
  - docs/parity-coverage-policy.md
  - CLAUDE.md
---

## The problem

You have a release that was carefully [checked against the reference](/different/06-checked-against-the-reference/), and a change since then that was meant to change nothing: a faster loop, a reordered sum, a refactor. How do you know it changed nothing?

The usual check is that a few prompts still give the same replies. That is weak. A model can keep its argmax, the single most likely next token, while its logits, the scores behind every token, have drifted. In this project's own history, two bugs found while adding one model family kept the right argmax while the cosine between goinfer's logits and the reference's was 0.897 (1.0 is identical). A prompt that "still works" would have passed both.

## What goinfer does

`go run ./cmd/gate identity <old-rev> <new-rev>` takes two git revisions (commits, branches or tags). It builds goinfer at each, runs the same prompts through both, and compares every logit as bytes. The unit it compares is a cell: one model file at one quantization (the number format its weights are stored in). For each model family it prints one verdict. This is the shape of the report, from the code's format strings (the revisions and names here are placeholders, not a recorded run):

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

The command exits 0 when every family that ran is identical, 1 when any is different (or the CPU is not deterministic), and 2 when nothing ran. On the 16 GB MacBook, `identity HEAD~1 HEAD` on the CPU with the small test models took 43–46 s, with 30 families IDENTICAL and 7 NOT RUN.

A change that should matter shows up. One earlier CPU change stores the 4-bit weights' scale factors as 16-bit floats instead of 32-bit, so it moves the numbers on purpose. On a real 0.5B checkpoint (a model with 0.5 billion parameters), that change ([`3cd62e6d`](https://github.com/townsendmerino/goinfer/commit/3cd62e6d) to [`5c85f7c0`](https://github.com/townsendmerino/goinfer/commit/5c85f7c0)) comes back DIFFERENT from the very first logit of prefill, the pass over the prompt before any token is generated.

## How it works

Two temporary git worktrees, one per revision, hold the two builds. A small program built into each dumps the full logits as raw 32-bit floats: one row for the prompt (prefill), then one row for every generated token, each chosen greedily (always the most likely one). Each family runs in its own process in the order new, old, new again. The command then compares bytes.

Full logits, not argmax, because argmax hides drift. Byte equality is also the only test with no threshold to argue about. Two details keep it honest. First, every `GOINFER_*` environment variable is removed from both sides, so the only difference between them is the code. Each side also prints the code path it used to generate tokens, and a difference is flagged. Second, a GPU cell that fell back to the CPU is reported NOT RUN, so a CPU result is never filed under Metal.

The new build must also match itself first. If a cell is not byte-identical from run to run, it is compared against the old build in a tolerance mode and reported WITHIN TOLERANCE, never IDENTICAL. The tolerance is twice the cell's own run-to-run difference, with argmax and generated tokens exact.

## What was measured

What would kill the tool was set in writing before it was built (pre-registered), on 2026-09-28: the same build giving different bytes on two CPU runs. That did not happen. No backend needed the tolerance mode.

<table>
<thead><tr><th>Backend</th><th>Cells byte-identical run to run</th><th>Families</th><th>Date</th></tr></thead>
<tbody>
<tr><td>CPU</td><td>120 of 120</td><td>not stated</td><td>2026-09-28</td></tr>
<tr><td>Metal</td><td>38 of 38</td><td>19</td><td>2026-09-28</td></tr>
<tr><td>WebGPU</td><td>80 of 80</td><td>30</td><td>2026-09-28</td></tr>
</tbody>
</table>

The figures come from the project's [test-efficiency task record](https://github.com/townsendmerino/goinfer/blob/main/docs/completed/task-test-efficiency-2026-09.md) and match the message of the commit that added the command, [`ce97d5ef`](https://github.com/townsendmerino/goinfer/commit/ce97d5ef). The record does not name the machine for each backend; only the timing run above names the MacBook. Metal runs only on Apple hardware, so the Metal counts come from a Mac. The record also lists three checks of the tool itself, run on scratch copies of the repo:

- a comment-only edit gives all IDENTICAL;
- a one-ulp change (the smallest step a float can take) in one residual path flags exactly the two families that use it, cohere and cohere2;
- a dump program deliberately made nondeterministic is reported, and the command exits 1.

## Use it

- `go run ./cmd/gate identity HEAD~1 HEAD` from the repository root compares your last commit with the one before it, on the CPU with the small test models. On the 16 GB MacBook that is the 43–46 s run above.
- In general, `go run ./cmd/gate identity <old-rev> <new-rev>`. The two revisions can sit before, between or after the flags.
- `-backend cpu|metal|webgpu` (default `cpu`). Identity is per backend.
- `-assets tiny|real` (default `tiny`: the small test models the repo's own parity tests use; `real` uses small real checkpoints under `~/models`).
- `-families a,b` and `-only <asset name>` narrow the run; `-quant q,...` and `-steps N` change what is compared.
- `-keep` keeps the dumps, binaries and logs. `-timeout` bounds each dump process (default 15 minutes).
- `-record <file>` writes candidate rows for the parity record, for families that qualify. Without it the record is never written. The rules are in the [parity coverage policy](https://github.com/townsendmerino/goinfer/blob/main/docs/parity-coverage-policy.md).
