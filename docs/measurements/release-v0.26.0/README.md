# v0.26.0: the release gate's records at the candidate `bfce89962`

What has been read so far for `RELEASING.md` step 8, on one Apple M1 Pro with 16 GB (macOS 26.7.3), 2026-11-28. The Linux
CUDA box's rows (its GPU gate, its deferred-tests row, the parity sweep and the SDE goldens) and the cold-user run are not here
yet. Nothing is tagged.

| row | at | result | record | where |
|---|---|---|---|---|
| Metal GPU gate (`go run ./cmd/gate gpu`, the heavy tier on) | 06:21:40 to 06:28:26 PST | **PASS**: 10 pass, 2 skip, 0 fail | `docs/measurements/gpu-gate/metal-20261128T142826Z-bfce8996.json` | `gate-gpu-metal-bfce8996.log` |
| darwin deferred tests (`go run ./cmd/gate quick -full -base v0.25.0`) | 07:08:43 to 07:55:49 PST | **RED**: 4,197 pass, 666 skip, 1 fail; the fail is a test-helper bug | `docs/measurements/quick-full/darwin-arm64-20261128T155549Z-bfce8996.json` | `quick-full-darwin/README.md` |
| tool-call smoke (not a gate row) | 06:22 to 08:03 PST, three runs | named and `required` calls hold the new keywords; an unwrapped call under `auto` is not constrained, as on v0.25.0 | none | `tool-call-smoke/README.md` |

- **The Metal GPU gate's two skips** are its own: the clean-GPU check (no `nvidia-smi` on this host) and 31 CI hygiene steps
  that run only on Linux. A skip is not a pass; the Linux box's run covers the second.
- **An earlier Metal GPU-gate record at the same commit is a FAIL**
  (`docs/measurements/gpu-gate/metal-20261128T142048Z-bfce8996.json`, `gate-gpu-metal-bfce8996-run1.log`): the launcher's
  `python3` had no PyYAML, so the gate could not derive CI's check set and failed that one step (9 pass, 1 skip, 1 fail). No
  test failed. The launcher now puts the Python that has it first and checks before it starts
  (`run-relgate-gpu-mac.sh`); the run above is the re-run. The newest record at the same code decides.
- **The deferred-tests row is red until its one failing cell is fixed or accepted.** A fix changes test files under
  `backend/webgpu/` and `backend/cuda/`, which voids both records above; an acceptance is an entry in the release's accepted
  file, which only the maintainer adds. The fix is pull request #1462, open and not set to merge. The Linux CUDA
  box's rows are exposed to the same crash (`quick-full-darwin/README.md`, "It is not a matter of page size").

`qf-mac.sh` and `run-relgate-gpu-mac.sh` are the two launchers.
