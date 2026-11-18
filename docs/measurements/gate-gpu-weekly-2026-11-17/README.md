# gate-gpu-weekly: the CUDA gate with its heavy tier, once a week on nobara's night queue (2026-11-17)

**Why.** The heavy tier ran only before a tag, so it rotted in between: at v0.24.0 the CUDA `gate gpu`, both `quick -full` runs and the parity sweep were red, every red test-only and accepted
by name (`docs/releases/v0.24.0-checklist.md` §6). A weekly run makes release night a confirmation of a run that was green a week earlier. Francis approved it on 2026-11-17 (about 2 h 20 min of nobara a week).

**What.** `goinfer-gate-gpu-weekly.timer` fires `queue-gate-gpu-weekly.sh` on Saturday at 18:00 and 20:30. It runs `night.py add gate-gpu-weekly --est 150 --priority 80 --not-before 22:00 --retry-until 06:00`
unless that job is already queued, so it runs after the nightly `cuda-device-suite` (priority 50) and after any ad hoc booking; the runner serializes the queue. At 18:00 it first looks at nobara
(as the C12 launcher's `nobarafree` does) and exits 75 without queueing when another job (not `cuda-device-suite`) is queued or running, or the timing lock is held by something that is not the night
runner; the 20:30 firing queues regardless. The job is `run-gate-gpu-weekly.sh`: it exits 75 (night.py re-queues it) if another run holds the timing lock, otherwise it is
`docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh` (a detached worktree of `origin/main`, `go run ./cmd/gate gpu`, the record under `~/goinfer-logs/gate-gpu-night-<date>/records`).
The box's `night.py serve` runs the queue. The weekly record does not replace the release record, which is at the freeze commit.

**Install (nobara, once this is merged and `~/mycode/goinfer` is pulled).** `mkdir -p ~/.config/systemd/user && cp docs/measurements/gate-gpu-weekly-2026-11-17/goinfer-gate-gpu-weekly.{service,timer} ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now goinfer-gate-gpu-weekly.timer`.

**Swap.** The v0.24.0 CUDA record shows `vmswap_peak_mb` 531 for the `gemma4-e4b` heavy group and 296 for `small`, which is at the default 512 MB guard. Do not move the threshold for this: if a group
cannot run inside the default guard, leave that group out of the weekly run and say so here.

**Mac twin.** `gate quick -full` on the Mac the same day (about 10 min) is the Mac's, by hand or its own queue; not set up here.

**Tier 0.** `python3 -B scripts/test_gate_gpu_weekly_queue.py` (a fake night.py and timing_lock.py: adds once at priority 80, skips when queued, defers at 18:00 when booked, queues at 20:30).
