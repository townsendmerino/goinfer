# metal-device-suite: the Metal module's tests with the device present, daily on the Mac (2026-11-17)

**Why.** CI's `metal-darwin` job runs 5 of the module's 882 tests (the runner's paravirtual Metal layer SIGSEGVs); the rest ran only in `gate gpu` and `gate quick -full`, by hand before a tag.
The newest record under `docs/measurements/gpu-gate/` was 13 days old when this was written. `scripts/touched.py` skips `backend/metal` on a box that is not darwin.

**What.** `run-metal-device-suite.sh` checks out `origin/main` in a scratch worktree, places the gitignored fixtures and runs
`go test -json -count=1 -p 1 -tags goinfer_testhooks ./...` in `backend/metal` through `scripts/cuda_device_suite.py --backend metal`, **under the timing lock** (exclusive, waits up to an hour).
No heavy tier. Expected about 5 to 10 min (4 min 12 s on 2026-10-13, timed-run skill, gate-tiers table). It is a day job, not a night job: `com.goinfer.metal-device-suite.plist` is a
LaunchAgent calendar job at 06:30 local, before the apps open; a Mac asleep at that time runs it on wake. A fixed hour is a placeholder for Francis to move (`Hour`/`Minute` in the plist).

**Verdict.** The CUDA suite's rules, with the Metal wording: RED on a failed test or package, a non-zero exit, no passing test, or a skip whose reason says there is no Metal device
(`no metal device`, `no GPU device`, `device: <err>`, `not resident on Metal`, `EnableResident: <err>`). An asset or `GOINFER_HEAVY_TESTS` skip, a memory-guard decline and a capability skip are counted, not red.

**Issues.** A run of `origin/main` files "metal-device-suite red on <sha>" (or comments on the open one; green closes it) and rewrites the standing heartbeat issue "device-suite status: mac".
`scripts/device_suite_heartbeat.py` (daily workflow `device-heartbeat.yml`) opens "device-suite heartbeat stale: mac" when no completed run is newer than 36 h. A `REV=` run files nothing.

**New skips.** The runner keeps the names that passed last run in `~/goinfer-logs/<backend>-device-suite.state.json` (origin/main runs only). A test that passed then and skips now is listed under `NEW SKIPS`, and is red unless its reason is an asset or opt-in wording (`GOINFER_HEAVY_TESTS`, a missing checkpoint or fixture); it stays red until it passes again. The first run has nothing to compare with. Delete the file to reset.

**Install (the owner's; a session does not load a launchd job).** The Mac's checkout is `~/tmcode/goinfer` (the path the plist `cd`s to) and must hold this directory (`git pull` there). A failed `cd` or script start is written to `~/goinfer-logs/metal-device-suite-launchd.log`, and it leaves no completed run, so the heartbeat check opens "device-suite heartbeat stale: mac" within 36 h.
```
cp docs/measurements/metal-device-suite-2026-11-17/com.goinfer.metal-device-suite.plist ~/Library/LaunchAgents/ && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.goinfer.metal-device-suite.plist
```
First run now, to see the plumbing: `launchctl kickstart gui/$(id -u)/com.goinfer.metal-device-suite`. Remove: `launchctl bootout gui/$(id -u)/com.goinfer.metal-device-suite`.
The job's log is `~/goinfer-logs/metal-device-suite-<date>/verdict.log` (the agent's own stdout and stderr go to `~/goinfer-logs/metal-device-suite-launchd.log`).

**Tier 0.** `python3 -B scripts/test_cuda_device_suite.py` (the Metal wording), `python3 -B scripts/test_metal_device_suite.py` (the plist parses and names this script; the script refuses off macOS).
