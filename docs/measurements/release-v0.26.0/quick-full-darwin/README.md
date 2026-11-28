# v0.26.0, the darwin deferred-tests row at `bfce8996`: one red, a test-helper bug

<!-- citations-at: bfce89962 -->

`RELEASING.md` step 8 on darwin: `go run ./cmd/gate quick -full -base v0.25.0` at `bfce89962`, in a fresh detached worktree, on
an Apple M1 Pro with 16 GB (macOS 26.7.3), go1.27.4, the timing lock held exclusively. Launched 2026-11-28 06:55:31 PST; it
waited for the lock behind the daily Metal device suite, ran 07:08:43 to 07:55:49 PST (47m6s) and ended **RED with one failing
cell**. The record is `docs/measurements/quick-full/darwin-arm64-20261128T155549Z-bfce8996.json` (verdict `FAIL`).

| | |
|---|---|
| tests | 4,197 passed, 666 skipped, 1 failed (top-level tests) |
| cells | 155: 148 ran, 7 not runnable on darwin (the CUDA module's), none replayed from the cache, none deferred |
| lint | 25 steps, 25 pass |
| goldens | 43 passed, 21 skipped, 0 failed |
| red | `backend/webgpu/internal/towers`: the test binary died with a SIGSEGV in the garbage collector |
| every test skipped | `apps/examples/embed` (2 tests) and `constrain [checkpoint]` (2 tests): see the last section |

None of the three reds in v0.25.0's darwin row recurred: `decoder [checkpoint]` ran inside its budget (4m45s, 133 pass),
`TestOtherGPUWindow_seesAnotherProcess` passed, and the lint's staticcheck step on `backend/metal` passed.

The launcher is `../qf-mac.sh`. It pins the run to `go.mod`'s toolchain line with `GOTOOLCHAIN`, because a workspace file made
by `scripts/make_work.sh` selects go1.27.2 from its own `go` line. A first launch that would have run on go1.27.2 was stopped
while it still waited for the lock and never started a test (`aborted-before-start/`).

## The red

**What happened.** `TestGridVisionWebGPU_matchesCPU/glm-ocr` was inside the test helper `makeWritable`, in `syscall.Mprotect`,
when a GC mark worker faulted (`cells/test_backend_webgpu_internal_towers.log`):

```
SIGSEGV: segmentation violation
PC=0x100b7ec94 m=8 sigcode=2 addr=0x14
runtime.getGCMask(...)
runtime.(*mspan).typePointersOfUnchecked(...)
runtime.scanObject(...)
...
syscall.Mprotect(...)
github.com/townsendmerino/goinfer/backend/webgpu/internal/towers.makeWritable(...)
github.com/townsendmerino/goinfer/backend/webgpu/internal/towers.gvRandomise(...)
github.com/townsendmerino/goinfer/backend/webgpu/internal/towers.gvGlm(...)
```

**Why.** The helper (`backend/webgpu/internal/towers/writable_unix_test.go:makeWritable`) rounds a weight slice's address
down to its page and hands `syscall.Mprotect` a `[]byte` that starts there. For a slice that aliases the checkpoint's mapping
that is harmless. For a heap slice it is a Go pointer to wherever the page begins, which on 16 KiB pages is often other heap
memory or none, and the collector can scan it while the call is in flight. The GLM-OCR tower's four norm weights are heap
slices, so this test passes them. The helper's own comment says heap slices "are left as they are"; the code does not do that.

**It is a known defect with a known fix.** The Metal backend's copy of the same helper had it and was fixed on 2026-11-20
(`39b0c7b94`, before v0.25.0): `backend/metal/internal/towers/writable_test.go:makeWritable` calls `mprotect` with integer
arguments and keeps the slice alive, and `TestMakeWritable_heapSlicesUnderGC` beside it holds the fix. The WebGPU copy and the
CUDA copy (`backend/cuda/internal/towers/writable_unix_test.go:makeWritable`) were not changed: they are identical to each
other after the build line, and to v0.25.0's.

**Readings** (`classify-webgpu-towers/`, the same commit and toolchain, 07:59:27 to 07:59:57 PST; `go test -overlay` supplied
the test files, so the tree was not edited):

| reading | result |
|---|---|
| the cell as the gate ran it, 10 times | **10 of 10 pass** (15 PASS lines, 13 skips each): the crash is intermittent |
| the Metal backend's `TestMakeWritable_heapSlicesUnderGC`, added to this package against its helper as it is | **dies at once**: `fatal error: found bad pointer in Go heap (incorrect use of unsafe or cgo?)`, the frame `syscall.Mprotect` under `makeWritable` |
| the same test with the helper in the Metal backend's form | passes |
| the cell's own tests with the helper in the Metal backend's form | pass (16 PASS lines) |

So the red is the test helper, not the WebGPU tower: no product code is on the faulting path, the tower's own comparison passes
in every run that completes, and the planted test reproduces the fault on demand and stops reproducing it with the fix.

**It is not a matter of page size** (`classify-webgpu-towers/pagesize/`, go1.27.4). Two standalone probes with both forms of
the helper and no goinfer import, each run in its own process, natively and as amd64 under Rosetta.

| probe | helper form | 16 KiB pages (darwin/arm64) | 4 KiB pages (darwin/amd64 under Rosetta) |
|---|---|---|---|
| fresh slices, 08:36:10 to 08:36:47 PST, 3 cells of 30 runs | a `[]byte` to `syscall.Mprotect`, as the WebGPU and CUDA copies have it | survives **2 of 90** | survives 90 of 90 |
| | integers to `mprotect`, the Metal backend's form | survives 90 of 90 | survives 90 of 90 |
| long-lived slices of 96, 400 and 6,000 bytes with half their neighbours freed, 08:51:25 to 08:53:05 PST, 10 runs each | `[]byte` | survives **0 of 30** | survives **0 of 30** |
| | integers | survives 30 of 30 | survives 30 of 30 |
| the same with 4,096-byte slices, whose slots start on a 4 KiB boundary | `[]byte` | survives 0 of 10 | survives 10 of 10 |
| | integers | survives 10 of 10 | survives 10 of 10 |

- **On 4 KiB pages the rounded-down address stays in the slice's own span** (a span starts on an 8 KiB boundary), but it can be
  another slot of that span. When that slot is free, the collector marks a free object and the sweeper aborts with
  `found pointer to free object`: all 30 deaths on 4 KiB pages are that. A slice just allocated is safe there, because every
  slot below it still counts as allocated, which is why the first probe saw nothing.
- **On 16 KiB pages the address can also leave the span.** The first probe's 88 deaths are 69 `found bad pointer in Go heap`,
  16 `found pointer to free object` and 3 SIGSEGVs at `addr=0x14` in `runtime.getGCMask` under `scanObject`, the gate's
  signature (`arm64.TestSliceForm_mixed.sigsegv.log` is one).
- **So the Linux row can hit this crash too,** on the CUDA box as on any 4 KiB-page host, wherever the helper is given a small
  heap slice that has lived through a collection. The 4 KiB arm ran under Rosetta, not on Linux.
- **A correction.** For some minutes on 2026-11-28 this section said an amd64 Linux box cannot hit the crash (`3d56425c3`),
  read off the first probe alone. An automated review of pull request #1462 named the free-slot case, and the second probe
  confirms it. The sentence before that, that the Linux row can hit it, was right.

**Class:** `test-bug`. The fix is pull request #1462, open and not set to merge: both copies of the helper in the Metal
backend's form, with a regression test beside each.

**What a fix costs before the tag.** It changes files under `backend/webgpu/` and `backend/cuda/`, so by step 8 it voids the GPU
gate's record and this row's, on every box. The other way is an entry in `docs/releases/v0.26.0-accepted.json` naming this
record, which only the maintainer adds. The Linux row is exposed to the same crash (above), so an acceptance covers darwin's
record only.

## The cells where every test skipped

A skip is not a pass. These are the same two cells as in v0.25.0's darwin row.

- **`constrain [checkpoint]`:** `TestConfidenceCost_C1` and `TestMaskCost_P20`, two cost measurements that need a real
  tokenizer and skip without `GOINFER_HEAVY_TESTS=1`. The package's 126 other tests ran in the `constrain` cell and passed.
- **`apps/examples/embed`:** `TestEmbedExample_outputIsNotARepeatedToken` needs a 0.5B checkpoint that is not on this machine.
  `TestBuildPrompt_rendersOnlyOneLeadingBOS` names its committed fixture one directory short
  (`../../tokenizer/testdata/chatml-tiny.gguf` from `apps/examples/embed/`), so it skips on every machine, CI included. With
  the path corrected through an overlay it **passes** at `bfce89962`, so the behaviour it pins holds; the test itself has
  not been running. Pull request #1462 corrects the path and makes a fixture that does not load a failure; with the defect the
  test pins planted, it fails at its count assertion.

## Files

- `gate-quick-full.log`, `provenance.txt`: the gate's whole output and the run's build, machine and times.
- `cells/`: the failing cell's log and the two all-skip cells' logs.
- `classify-webgpu-towers/`: `run-classify.sh`, its summary (`run-classify.log`), the ten re-runs, `plant.log`, `fixed-gc.log`,
  `fixed-cell.log`, and under `overlay/` the two generated test files as text.
- `classify-webgpu-towers/pagesize/`: the two probes (`probe_test.go.txt` with `run-probe.sh`, `probe_settled_test.go.txt`
  with `run-probe-settled.sh`), their summaries (`run-probe.log`, `run-probe-settled.log`), each cell's first log, the first
  death's log of each cell that died, and one SIGSEGV death.
- `aborted-before-start/`: the stopped first launch.
