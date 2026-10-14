# nobara: before tomorrow's v0.23.0 release rows (written 2026-10-14 on the Mac)

The five owed rows of `go run ./cmd/gate release v0.23.0` run tomorrow, 2026-10-15, on the merged release-candidate
commit, not tonight (owner, 2026-10-14). Two of them run on nobara: the CUDA GPU gate and the §C1 parity sweep. Each
needs one thing fixed first. The plan is `docs/releases/v0.23.0-checklist.md`.

## 1. The CUDA GPU gate no longer fits its 120-minute cell

The main tier of `cmd/gate/gpu.go:gpuGate.cudaHeavy` (cell `cuda-heavy`, all of `./cuda/` under
`GOINFER_HEAVY_TESTS=1` in one process) runs under `Timeout: "120m"`. The owner reports that the tier's wall is now
past that. A process that dies at its timeout reports a hang and names whichever test was running, so the record says
nothing useful.

Before tomorrow's run, do one of the following:

- **Split the tier.** Move another group into its own process, as the drain group already is: a marker read by
  `drainsDevice()`-style derivation, with the reconciliation counting the new group in both directions.
- **Re-register the timeout.** Measure the tier's current wall first, then set the new value with the headroom rule in
  `docs/code-notes/cmd-gate.md` § "cudaHeavy: main-tier timeout". Do not change the number without that measurement.

Either way, the change touches `cmd/gate/`. It lands before the release-candidate commit is chosen, because the
gate's record must be read at that commit.

## 2. The parity sweep's ten blockers (`parity-20261012T091615Z-6e50c609.json`), read 2026-10-14

| blocker | still holds? | blocks v0.23.0? |
|---|---|---|
| 7 × asset missing under the `~/models`-only rule: mellum2 ×3, deepseek ×2, glm4moe-air, qwen2moe-oracle | `ceeaa5a0`'s message says the assets were restored since; not re-read today (nobara's timing lock was held) | only if they are still missing tomorrow: `models-pull` each before the sweep |
| `TestW4A8DecodeParity` FAIL | probably not a code defect: its two assets (`qwen15-w4a8-int4.giw`, `qwen15-w4a8-int8.giw`, `testdata/assets.json`) were built 2026-08-18, before `.giw` v16, which is the only format read now. Rebuilt at v16 from the 1.5B GGUF on the Mac today, the test PASSES 16/16 at `e42db592` (arm64). | yes, until it reads green on nobara: rebuild **both** halves together with `cmd/prequant` (the recipe is in the asset's note), then run the test by name |
| `TestSamplingThroughputGate`, a red not on the sweep's list | its bar moved 5.0× → 6.5× in `783255ad` with a stated mechanism, and the test moved to `internal/sampling`, which the sweep's `./decoder/ ./tokenizer/` cell no longer runs | no |
| `TestGemma4EModel_realDeclinesResident` | fixed in `1ca10426` (the test still asserted that WebGPU declines the E-model) | no |

Any `decoder/` change makes the sweep owed again, so it is the last run before the tag.
