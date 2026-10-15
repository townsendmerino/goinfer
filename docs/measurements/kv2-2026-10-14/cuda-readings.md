# KV2's CUDA arm: readings (nobara)

nobara-pc, RTX 2070 SUPER 8 GB, driver 595.91.07, 2026-10-15. Registration: `docs/measurements/kv2-cuda-2026-10-15.md`,
with its §6 amendments. Branch `kv2-cuda-2026-10`.
- Day readings are at `26105913`: `kv2-2026-10` at `a1141868` merged in, plus the CUDA half.
- Logs are under `~/goinfer-logs/kv2-cuda-2026-10-15/`.
- Plant output is `planted-cuda.txt`, beside this file.

## Day, 2026-10-15

| gate | reading | log |
|---|---|---|
| G0-C | KV0's arms PASS at f32 and f16, tiny and 1.5B, on the merged tree. The new arm (exact prefill equals decode) PASS, same cells | `merged-kv2-all-1.log` |
| G1-C | `cuda`, the identity patterns: 142 pass, 38 skip (heavy and named opt-ins, at night), 0 fail. `decoder` patterns: 497 / 12 / 0. `internal/kvmgr` and `internal/serveapp`: 662 / 55 / 0. Only the Mac's named exception differs from `main` among the test files | `g1-*-merged.log` |
| G2-C | PASS: tiny and 1.5B at f32 and f16; MLA (`deepseek-tiny`); shared KV (`gemma4-emodel-tiny`); the three windowed fixtures, including the decline. A copied prefix equals the prefix computed, byte for byte | `merged-kv2-all-1.log` |
| G3-C | PASS: schedules (a) to (f) on tiny (greedy f32, greedy f16, sampled f16) and on the 1.5B (greedy, sampled). Every turn's tokens and K/V rows equal its reference's; B copied 960 rows from A's busy slot under MC3. The windowed decline schedule PASS | same |
| G4-C | (a) PASS on the merged tree. At the pre-merge base it was red, as KV1 predicted: a 1,030-token prompt's last 6 rows ran exact and were recorded fast. (b) PASS: 896 rows copied, equal to cold. (e) PASS: 640 rows copied, none of the unaligned pass | same |
| G7-C | exploratory: 18,176 rows of the 1.5B in 2.77 ms at f16 (521 MB) and 5.41 ms at f32 (1,042 MB), against an estimate of 2.6 ms | same; `g7-gvb-day.log` |
| GV-C | (a) SAME banners on `e77c721f` and the branch for the 1.5B (`-ctx 24576 -kv-sessions 2`), the 7B and Mellum2.1. (b) 0 bytes allocated in 20 copies. (d) PASS: expert caching keeps one slot, and nothing is copied | `gv-plan/gv-plan-day2.txt` |
| plants | all 12 red on the merged tree | `planted-cuda.txt` |

**Finding for the decoder, reported to the Mac session (not graded).** With three slots, `Model.residentShare` takes the
first slot with the longest hit. If that slot's window has moved on, the copy is declined, even when another slot still
holds the rows: A1 B1 A2 C1 left B's slot based at 241 and C was served cold. Choosing among sources by
`ReusableKVSlot` would recover the copy. This costs time, never a wrong row.

## Night

Not yet queued (`run-kv2-cuda-night.sh`, beside the registration).
