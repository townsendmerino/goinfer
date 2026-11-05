# Post-tag gate fixes, PR 3: build_ptx.sh compiles from a fixed directory (2026-11-05, nobara-sb01)

Asked for by the coordinator after v0.24.0: `tower_base.ptx` read DIFFERS in `gate ptx` (step 4 of the CUDA GPU gate) in some checkouts and PASS in others, and
v0.24.0's gate record carried it as an accepted red.

## The cause, already found

`sb12-cuda-gate-gpu-2026-10-29/ptx-tower-base-path/read.txt`: NVRTC V12.9.86 compiles `tower_base.cu` to one of several PTX texts depending on the absolute
directory it runs in; only `tower_attn_hd80` differs (a redundant `ld.shared.v2.f32` after a `ld.shared.v4.f32` of the same address, same arithmetic). The
source, the toolchain and the committed PTX are unchanged since 2026-10-07. `nvrtc_compile.py` passes the source's own absolute directory to NVRTC as an include path,
which is how the checkout's location reaches the output.

## Red

- `red-two-paths.log`: `go run ./cmd/gate ptx` with identical `build_ptx.sh`, `tower_base.cu` and `tower_base.ptx` (`cmp` equal) in two checkouts: RED
  (`DIFFERS: tower_base.ptx`) in `~/goinfer-bench/release-v0.24.0-wt`, GREEN (59/59) in `~/goinfer-bench/post-tag-tests-wt`.
- `red-test.log`: `TestBuildPTX_towerBaseDoesNotDependOnTheCheckoutDirectory` builds `tower_base` with the script from six differently named checkouts and
  compares each with the committed file; the checkout named `ccc` differs (250,118 bytes, want 250,036). `red.log` is the first run in this tree, GREEN.

## The change

`backend/cuda/build_ptx.sh` copies the sources (`*.cu`, `*.cuh`, `nvrtc_compile.py`) to `/tmp/goinfer-nvrtc-build/cuda`, compiles there, and copies the PTX back.
That path was picked by trying candidates against the committed `tower_base.ptx` (most reproduce it; a few do not, e.g. `/tmp/a`): this one reproduces every committed PTX, so **no PTX file
changes in this PR**. A `flock` on `/tmp/goinfer-nvrtc-build/lock` lets one build use the directory at a time. The comment above `STAGE` says not to change the path without
regenerating the PTX.

## Green

- `green-test.log`: the new test passes (it skips without NVRTC V12.9.86).
- `green-two-paths.log`: `go run ./cmd/gate ptx` with the new script: GREEN, 59/59, in this checkout and in the release worktree whose path read RED before (the script copied in
  temporarily and restored: 0 changed paths afterwards).
- `green-touched.log`: `python3 scripts/touched.py`, 7 checks passed.

## Not done

A full `gate gpu` was not run; the PTX group is the part this changes (`gate ptx` runs it alone). The regenerated-PTX route (rebuilding `tower_base.ptx` at the fixed path) was not needed.
