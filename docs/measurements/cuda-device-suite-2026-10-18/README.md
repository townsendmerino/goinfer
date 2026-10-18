# cuda-device-suite: the CUDA module's tests with the device present, nightly (2026-10-18)

**Why.** CI's cuda job has no GPU, so most of the module's tests skip there and print `ok`. `TestKernelLocalMemoryCensus` was red on every box with a device from
`367bf014f` (the commit that added `mamba_conv`, 2026-10-16) until it was pinned, and CI never saw it. A self-hosted runner is not an option: the repo is public.

**What.** `run-cuda-device-suite.sh` checks out `origin/main` in a scratch worktree, places the gitignored fixtures, and runs `go test -json -tags 'cuda goinfer_testhooks' ./...`
in `cuda/` through `scripts/cuda_device_suite.py`, under `night.py` (which holds the timing lock). No heavy tier.

**Verdict** (`scripts/cuda_device_suite.py`; the last lines of the log, which `SUMMARY.md` quotes): RED on a failed test or package, on any non-zero exit, on no passing test, and on a
**skip whose reason says there is no device** (a skip counts as a failure for a test that needs the device). Skips for an asset or `GOINFER_HEAVY_TESTS` are counted and named, not failures.
`FAILED TESTS (n):` and `DEVICE SKIPS (n):` list the names.

**Expected wall time.** About 5 min warm, 8 cold; queue it with `--est 10`. Basis: the same suite measured 12 s + 159 s in `docs/measurements/gate-gpu-noheavy-2026-10-06/gate-gpu-noheavy.log`.

**Overlap.** `gate gpu` (pre-tag, about 2 h 20 min with its heavy tier) runs this suite as its step 2a/2b; this job is the cheap daily slice of it. `night.py` has no recurring jobs: it is queued
each night it should run (`night.py add cuda-device-suite ...`, line in the script header).

**Tier 0.** `python3 -B scripts/test_cuda_device_suite.py` (CI: "cuda-device-suite verdict"): a fake run with one failing test is red and its name is in the last 25 lines; each device-skip
wording is red even when `go test` exits 0; an asset skip is green; nothing passing, a build failure and a non-zero exit with no failed test are red. The plant (`DEVICE_SKIP` made to match
nothing, i.e. skips treated as passes) reads GREEN on the device-skip case, which is what proves the red assertion can tell the two apart.
