# cuda-device-suite: the CUDA module's tests with the device present, nightly (2026-10-18)

**Why.** CI's cuda job has no GPU, so most of the module's tests skip there and print `ok`. `TestKernelLocalMemoryCensus` was red on every box with a device from
`367bf014f` (the commit that added `mamba_conv`, 2026-10-16) until it was pinned, and CI never saw it. A self-hosted runner is not an option: the repo is public.

**What.** `run-cuda-device-suite.sh` checks out `origin/main` in a scratch worktree, places the gitignored fixtures, and runs `go test -json -tags 'cuda goinfer_testhooks' ./...`
in `cuda/` through `scripts/cuda_device_suite.py`, under `night.py` (which holds the timing lock). No heavy tier.

**Verdict** (`scripts/cuda_device_suite.py`; the last lines of the log, which `SUMMARY.md` quotes): RED on a failed test or package, on any non-zero exit, on no passing test, and on a
**skip whose reason says there is no device** (a skip counts as a failure for a test that needs the device). Skips for an asset or `GOINFER_HEAVY_TESTS` are counted and named, not failures.
`FAILED TESTS (n):` and `DEVICE SKIPS (n):` list the names.

**Expected wall time.** About 5 min warm, 8 cold; queue it with `--est 10`. Basis: the same suite measured 12 s + 159 s in `docs/measurements/gate-gpu-noheavy-2026-10-06/gate-gpu-noheavy.log`.

**Recurrence.** `goinfer-cuda-device-suite-queue.timer` (a systemd user timer, installed in `~/.config/systemd/user/`, linger on) fires at 18:00 PDT each evening and runs `queue-cuda-device-suite.sh`, which runs
`night.py add cuda-device-suite --est 10` in `~/mycode/goinfer` unless a job of that name is already queued. It adds the job and does not start the queue: the owner starts it at bedtime. The main checkout must hold this
directory (`git pull` there): a checkout without it makes the unit fail loudly (`systemctl --user status goinfer-cuda-device-suite-queue.service`).

**The verdict as an issue.** After the run the job calls `scripts/cuda_device_suite_issue.py`: red opens "cuda-device-suite red on <short sha>" with the summary, the failing names and the last 40 lines of the log (home paths shortened),
or comments the same on the newest open issue of that title; green closes every open one with a comment. A log with no verdict line (the job died in setup) opens and closes nothing; `SUMMARY.md` has it. The issues are public, as the repo is.

**Other revisions.** `REV=<sha-or-ref>` runs the same suite on a branch (queue it as `env REV=<sha> bash …/run-cuda-device-suite.sh` with its own job name); only `origin/main` files or closes the issue.

**Overlap.** `gate gpu` (pre-tag, about 2 h 20 min with its heavy tier) runs this suite as its step 2a/2b; this job is the cheap daily slice of it. `night.py` has no recurring jobs: it is queued
each night it should run (`night.py add cuda-device-suite ...`, line in the script header).

**Tier 0.** `python3 -B scripts/test_cuda_device_suite.py` (CI: "cuda-device-suite verdict"); `test_cuda_device_suite_issue.py` and `test_cuda_device_suite_queue.py` do the same for the issue logic (a fake `gh`; the plant reads every verdict as green and opens nothing for a red run) and the timer's command (a fake `night.py`: adds once, skips when queued): a fake run with one failing test is red and its name is in the last 25 lines; each device-skip
wording is red even when `go test` exits 0; an asset skip is green; nothing passing, a build failure and a non-zero exit with no failed test are red. The plant (`DEVICE_SKIP` made to match
nothing, i.e. skips treated as passes) reads GREEN on the device-skip case, which is what proves the red assertion can tell the two apart.
