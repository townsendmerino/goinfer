# internal/fitcmd: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/fitcmd`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn

Moved from `internal/fitcmd/defaultctx_test.go` (the comment above `TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn`) on 2026-10-09.

```text
The dry run and the real load must agree on the default context: cuda/resident.go's fitDefaultCtx says goinfer-chat fit's -ctx default is "the same
figure, so the dry run and the real load agree" — and nothing checked it, so R19's change (8192 to 16384, docs/tasks/task-first-hour.md) had two
places to forget. The cuda module cannot be imported from here (it is its own module, behind a build tag), so the constant is read from its source.
It must also hold a coding agent's first request: opencode's was 11,137 tokens.
```

## TestFreeBytesFor_cpuUsesThePassedInValue

Moved from `internal/fitcmd/fit_test.go` (the comment above `TestFreeBytesFor_cpuUsesThePassedInValue`) on 2026-10-09.

```text
TestFreeBytesFor_cpuUsesThePassedInValue is the M-19 gate (docs/audit-2026-09-10.md):
freeBytesFor's "cpu" branch must report EXACTLY the caller-supplied hostFreeBeforeLoad, proving
it no longer re-queries HostRAMAvailableBytes() itself after Run has already loaded the model
into that same RAM budget. A non-positive value must report unknown, matching the old
HostRAMAvailableBytes()-returns-0-means-unknown convention.
```

## TestRun_measureRateExcludesPrefill

Moved from `internal/fitcmd/fit_test.go` (the comment above `TestRun_measureRateExcludesPrefill`) on 2026-10-09.

```text
TestRun_measureRateExcludesPrefill is M-20's behavioral half (docs/audit-2026-09-10.md):
selfMeasure's printed rate used to be n/(prefill+decode) — roughly a third of the true decode
rate on the CPU staged path (the audit's own measurement) — because the clock started before
Generate was even called. It now starts after the FIRST token arrives, so the printed line must
say so explicitly (not the old "includes a N-token prompt prefill" phrasing, which described
the bug rather than a fix for it) and must report decode steps counted as probeDecode-1, not
probeDecode — proving the first step was excluded from the denominator, not just relabeled.
```

## TestRun_closesFirstLoadBeforeMeasuring

Moved from `internal/fitcmd/fit_test.go` (the comment above `TestRun_closesFirstLoadBeforeMeasuring`) on 2026-10-09.

```text
TestRun_closesFirstLoadBeforeMeasuring is M-20's other half: selfMeasure loads the SAME
checkpoint again, so Run must close its own first Load BEFORE calling selfMeasure — leaving it
open the whole time means two full quantized copies resident simultaneously on the CPU backend,
which can itself trip the memory guard the probe exists to measure honestly. The interruption
that matters (a real large checkpoint tripping the guard) can't be reproduced with this repo's
tiny fixtures, so this is asserted structurally instead, the same technique
TestTranscode_writesViaTempThenRenames (internal/prequant) uses for an analogous ordering
property: parse Run's AST and confirm the statement calling m.Close() appears BEFORE the `if
*measure` block that calls selfMeasure, not after.
```

## TestFreshSidecar_findsTheOneChatAndServeBuild

Moved from `internal/fitcmd/fit_test.go` (the comment above `TestFreshSidecar_findsTheOneChatAndServeBuild`) on 2026-10-09.

```text
TestFreshSidecar_findsTheOneChatAndServeBuild: a default chat or serve load (--backend cpu) writes this
host's cpu target, <base>.int4.cpu-amd64.giw or .cpu-arm64.giw. fit used to look only for the canonical
target, so it never found that sidecar and did a full direct load of the .gguf instead (measured on the
0.5B: 2.98 s and 1.0 GB RSS, against a mapped read through the sidecar).
```

## testFreshSidecar

Moved from `internal/fitcmd/fit_test.go` (the comment above `testFreshSidecar`) on 2026-10-09.

```text
testFreshSidecar builds the sidecar a load with the given --embed-int4 writes, and freshSidecar must find it. With
embedInt4 it is <base>.int4.e4h.<target>.giw, the one a default CPU/CUDA/WebGPU load writes; fit looked only for the
plain-head name and missed it.
```
