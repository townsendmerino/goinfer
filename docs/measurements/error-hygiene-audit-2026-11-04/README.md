# Error handling and logging audit (2026-11-04)

**Question.** Across the tree, how consistent are error wrapping, sentinel errors and logging; where is a log-and-return pattern repeating
reports up the stack; and what is worth fixing? Ordinary engineering, not a registered gate.

**Method.** A type-aware census of every Go file (`errlogaudit.go.txt`, go/packages), one pass per build-tag and platform combination across all
seven modules. It self-tested on a fixture that fires every detector, and it counts what it covered: 2,945 of 2,997 tracked `.go` files type-checked
clean at 230e247b. The other 52 are other-platform files and measurement scratch programs, grepped by hand, with nothing in production code.

## What the brief assumed, and what is here

The brief expects several eras of error handling, mixed logging libraries and log-and-return at every layer. None of that describes this tree.

| Production code | Before | After | Reading |
|---|---:|---:|---|
| `fmt.Errorf` calls | 3,069 | 3,069 | |
| ... wrapping with `%w` | 959 | 1,004 | |
| Arguments formatted with another verb | 87 | 14 | 6 in library code, each deliberate (below); the other 8 in `cmd/gate` |
| `err.Error()` read to decide something | 9 | 6 | 3 in library code: one owner of the device text, a dedup key, a display trim |
| `==` on errors | 8 | 4 | all correct: an `Is` method, an identity check, two `syscall.Errno`s |
| Type assertions on errors | 6 | 0 | |
| Log-and-return | 2 | 0 | now held by a test |
| Errors dropped with `return nil` | 6 | 4 | deliberate: lenient content parsing, a documented fallback, a walk that skips a file |
| Logging libraries | none | none | no `zap`, no `logrus`, no `log/slog` before this |
| Direct writes in library code | 122 | 22 | what is left: `pull` progress, CLI subcommands under `internal/guard` and `internal/diag`, this seam |
| Direct writes in CLI code and tools | 397 | 397 | not logging, see below |

**No chain any consumer reads was broken by `%v`.** `ErrLoadAborted`, `ErrKVPoolFull`, `context.Canceled` and the decline types were traced to
their `errors.Is` and `errors.As`. The wrapping work is latent risk removed, not a bug fixed.

## Why not `log/slog` everywhere, as the brief asks

- **Stderr text is an interface here.** `scripts/bench_peer.py` grades cells from a server's `fit:` and `decode path:` lines, and
  `internal/app/serveapp/banner_test.go` pins them. Timestamps and level prefixes would break both.
- **Most `fmt.Fprint` here is a program's user interface,** not its log: a CLI's answers on stdout, its usage text and its progress.

So only library code moved, onto a seam (`internal/runtime/logs`, `decoder/logger.go:SetLogger`) that defaults to today's text. An independent
tool compared all 106 migrated messages old against new, in order: 0 differences. `TestLibraryOutput_goesThroughLogs` went red on those 106 before
the rewrite.

## Left as is, on purpose

- **Six `%v` sites in library code.** Five are `%w` for the classification with `%v` for a second error that is only detail: `modelload.Load` twice,
  CUDA's audio encoder and Qwen tower, and the prefill OOM decline. The sixth, in `buildResidentMode`, can format a nil error, and `%w` would print
  `%!w(<nil>)`. The three `cmd/gate` record loaders have the same nil case.
- **`cmd/gate/grade_records.go`:** a grader's imports are limited (`TestGraders_arePure`), and nothing reads those two errors.
- **365 test sites that read `err.Error()`.** A test asserting what a message says is doing its job.

## How the change was checked

On the Mac: `python3 scripts/touched.py` (37 of 41 checks, including the whole of `backend/metal` and `backend/webgpu`), `go run ./cmd/lint`, the citation lint, the tagged
vets, and the new tests under `-race`. The 4 red checks are all `backend/cuda` failing to build, and they reproduce on an unmodified export of `main`: see follow-up 6.

On nobara, from a scratch copy, since `backend/cuda`'s tests cannot run on a Mac: the package tests beside the changed files and everything touching the allocation-failure text,
stderr capture or the launch error, 60 tests on the card: 51 passed, 8 skipped (heavy tier), 1 failed. The failure, `TestPlanSlack_realSizeDiff`, fails the same 63 groups on `main`:
see follow-up 7. `gpudev`'s new test passed there too. The 8 heavy tests did not run, nor did the rest of the package.

## Follow-ups, none started

1. **Type the device allocation panic.** `internal/runtime/mem/device_failure.go` now owns the text, but it is still text. The design: `gpudev` panics
   with an `*AllocError` that has a `DeviceOutOfMemory() bool` method; recover sites wrap it with `%w` (`runJob`, the tower encoders); `mem` asks
   `errors.As` against that interface first, so neither `gpudev` nor the root module imports the other. The driver's `CUDA_ERROR_OUT_OF_MEMORY` stays
   text until the binding types it. Every path it touches needs a card to verify.
2. **Metal's allocation failure is not recognized.** It panics `metal: MTLBuffer allocation failed ... out of memory`, which `mem.DeviceOutOfMemory` does
   not match, so a Metal tower that runs out does not fall back to the CPU tower as a CUDA one does. Possibly right, since the memory is unified. Unverified.
3. **`server.NewHandler` still prints goinfer-serve's banner.** `internal/app/serveapp` has about 105 stderr writes. Routing them is mechanical, and the
   banner's text must stay.
4. **M9's other half:** an `Options` way to make a backend decline an error (`docs/tasks/task-library-surface-review-2026-10.md`).
5. **Prefix styles are the one real "eras" finding:** `[cuda]` and `cuda:`, `[metal]` and `metal:`, `[goinfer]` and `goinfer:`, in messages that docs quote. Left
   alone, since normalizing them changes the text this change promised not to change.
6. **`scripts/touched.py` cannot build `backend/cuda` on a Mac.** It sets `CGO_ENABLED=0` and leaves the host's GOOS, and `gpudev/copy_coalesce.go` needs darwin's cgo there. Any cuda edit on a Mac
   reads red for this reason alone; `GOOS=linux` builds it.
7. **`TestPlanSlack_realSizeDiff` fails on `main`,** seen on nobara at ab4cea76: all 63 groups differ from its golden. Not investigated: a stale golden after a plan change is one explanation, a real change in
   what the planner decides is another. The subsystems doc names it as a gate (G-GF2-3, G-MA-4).

## Review round, 2026-11-06

A review of PR #700 found these; each is fixed on the branch.

- **`loadTekken` wrapped a nil error** when `token_bytes` decoded to nothing, printing `%!w(<nil>)`. The two cases are now separate errors
  (`TestTekken_refusesABadTokenBytesEntry`, red on the old code).
- **The level functions loaded the logger twice**, once to check the level and once to write, so a `Set` between the two could hand a record
  to a logger whose level was never checked. Each now loads it once.
- **The opt-in diagnostics** (`GOINFER_DECODE_TIMING`, `GOINFER_DELTANET_TIMING`, `GOINFER_ATTNFA_DEBUG`) were at Debug, so a logger at its
  default level dropped what the operator had turned on. They are at Info, and `docs/env-vars.md` says so.
- **`TestLibraryOutput_goesThroughLogs` had blind spots.** It now finds `log` and `log/slog` under any import name, carries a report out of an
  `if` branch that falls through, clears a name that is assigned again, takes `le`-style names as errors, and counts a return as returning
  an error only when the result is the error or wraps it. Five more roots are scanned: `embeddinggemma2`, `server`,
  `internal/guard/memgate`, `internal/guard/paritykey` and `internal/wrappedupload`. A planted file was flagged on its four defects and on
  none of its three controls. It still parses without type-checking, so it knows an error by its name.
- **`gpudev`'s test held a copy of the words**, so rewording `mem`'s text failed nothing there. `gpudev` may import nothing of goinfer's,
  so the check moved to `backend/cuda/kernels` (`TestAllocFailedPanic_isWhatMemReads`, there because it imports `gpudev` and not `decoder`), which calls `mem.DeviceAllocFailed` on `gpudev`'s panic,
  and the copy is gone. It runs on Linux under `-tags cuda` only, so CI's cuda job is its first run.
