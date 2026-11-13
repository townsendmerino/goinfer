# CUDA device read of main after the direct commits, nobara, 2026-11-12

What landed on the card from the ten commits that went to main without a PR on 2026-11-11 and 2026-11-12, read on nobara (RTX 2070 SUPER 8 GB, driver 595.112.07,
16 cores, load 0.37 at the start, the desktop's compositor holding 425 MiB) at main `4edbe93e9`, 16:19 to 16:45 PST. Tier 0 and tier 1 reads only: no speed was measured.
Every GPU invocation held the exclusive timing lock (`scripts/timing_lock.py run`), one invocation per test, `-v`, from a test binary built once under the shared lock
(`go test -c -tags 'cuda goinfer_testhooks'`). Rows: LW-cu and MM9 (`docs/tasks/task-forward-and-memory-2026-11.md`), SEQ (`docs/tasks/task-kv-architecture-2026-10.md`), G5
(`docs/tasks/task-kv-precision-everywhere-2026-10.md`). Logs are in `logs/`; `run-step2.sh` and `run-step4.sh` are the scripts that ran them.

**Fixtures.** A new worktree has none of the gitignored tiny fixtures, and a test that finds none skips (`TestCUDAStepRagged_rowsMatchTheirShapes` printed `SKIP` for
`gemma4-dense-twogeom-tiny` on the first pass, kept in nobara's log directory, not here). The ones under 120 MB were copied from the main checkout's `testdata/` into the worktree, plus
`gemma4-dense-scaled` and `gpt2` for the identity re-run. Not copied (the 28 GB of the real-size ones): `gemma-3-270m`, `gemma3-gguf`, `gemma4-dense-scaled-{24,48,64}`, `gemma4-moe-scaled`,
`gemma4-vl-bidir-scaled`, `llama3.2-1b`, `mellum-gguf`, `qwen2-gguf`, `qwen3-gguf`, `tinyllama-*`. Every reading below was taken after the copy, with no `SKIP` except where named.

## Step 1: build

`go vet -tags 'cuda goinfer_testhooks' ./backend/cuda/`: clean (no output).

## Step 2: device tests at main (all PASS, no skips)

| test | result | log |
|---|---|---|
| `TestCheckCUDALayerWalk_TopologiesAndRejections` | PASS (builds operation lists by hand, runs no forward) | `02-layerwalk.log` |
| `TestCUDAStepRagged_rowsMatchTheirShapes` | PASS, 10 PASS lines, 0 skips | `02-ragged.log` |
| `TestKV4_scrambledTables` (G1a) | PASS, 33 PASS lines (before the flint rows below) | `02-kv4scrambled.log` |
| `TestKV4_sharedPreambleStoredOnce` (G3b, tier 0) | PASS, f32 and i8 | `02-kv4shared.log` |
| `TestKV7_slotWriteTable` | PASS, 61 PASS lines | `02-kv7slot.log` |
| `TestWindowedKV_i3Replay` (the window replay) | PASS, arithmetic only, no device | `02-kvwindow-replay.log` |
| `TestFlintResidentParityCUDA` | PASS (resident, int4, the sliding window, sinks, the expert cache at 3 slots) | `02-flintparity.log` |

Read beside them, not asked for: every `TestWindowedKV_*` (57 PASS, one SKIP: `TestWindowedKV_offsetsFastClass_real1p5B`, which needs the real checkpoint; `02x-windowedkv.log`), and
the two edits LW-cu names as more than a substitution (`05c-graphs-ple-extra.log`): the Gemma 4 E-model (PLE and layer scale) resident parity, batched-prefill bit identity, image
prefill and its planted defects, all PASS. **Graph capture is not read on this box:** `TestGemma4Graphs_bitExact_tiny`, `_tinyCache` and `_tinyCacheReuse` SKIP with "CUDA graphs not admitted on this box
(DEFAULT compute mode, no MPS)", because a graphs-on arm would run live and compare live to live. `captureGraphs` choosing its layers differently (LW-cu) had no device reading at that point; `TestGemma4Graphs_bitExact_emodel` and `TestCUDA_graph*` passed but are not evidence for it under that decline. Read afterwards, under MPS: next section.

## Graph capture under MPS (added 18:00 PST, after the coordinator started the MPS control daemon)

Francis asked for graph capture on nobara (via the coordinator, 17:55). The CUDA MPS control daemon runs as francis (pid 973928, `docs/cuda-graphs-investigation.md` §5.1); the compute mode stays Default, and a process admits graphs only with
`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps` (and `CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log`) exported. MPS was live for the test process: `get_server_list` printed a server pid (974139) at every sample during the run
(`logs/08-mps-serverlist-main.log`), and each load printed "CUDA graphs enabled: CUDA MPS active". Exclusive timing lock, `-v`.
- **At main (`08a-graphs-on-main.log`):** `TestGemma4Graphs_bitExact_tiny`, `_tinyCache`, `_tinyCacheReuse` (the three that SKIPPED), `_bitExact_emodel`, `_sameModelUnderLoad`, `_offVsOffControl`, `TestCUDA_graph*` and `TestCUDAKVSlots_graphs` all **PASS**, no skips.
  `TestGraphsSafeGate` FAILS under MPS by construction: it asserts the tenancy is UNSAFE on a DEFAULT box and finds MPS active (`graphs_safe_test.go:40`, "expected UNSAFE tenancy on a DEFAULT box, got safe: CUDA MPS active"). It passed without the variable (`logs/05c-graphs-ple-extra.log`). Not a regression.
- **The same tests on a build at 51ed2ae27 (`08b-graphs-on-old-51ed2ae27.log`):** all PASS, the same set.
- **Which layers capture, old against new, and graphs-on against live** (a throwaway probe, `lwcu_capture_probe.go.txt`, built into both trees and not committed: it loads each tiny fixture at int4 with `GOINFER_CUDA_GRAPHS` unset and set, lists each layer's captured segments
  (A, B, C), and compares 14 decode steps' logits bit for bit; `08c-capturemap-{old,new}.log`, `08c-map-{old,new}.txt`). **The 13 capture maps are identical between 51ed2ae27 and main, and graphs-on equals graphs-off bit for bit on all 13 on both builds.**
  Attention layers capture A and B (`AB`), MoE layers add C (`ABC`), DeltaNet layers capture one mixer-plus-FFN segment (`AC`, `ABC` where an attention layer sits among them), Flint `AB AB ABC ABC ABC ABC`. `nemotron-tiny` and `deepseek-tiny` do not enable graphs on either build (the flag is false), so their layers
  are uncaptured on both and the new `switch` that skips mixers other than attention and DeltaNet changes nothing there. `lfm2-tiny` and `granite-tiny` are not CUDA-resident (declined on both builds) and were not read.
- **Not read:** a graphs-on speed cell. Tomorrow's registered decode read is the default path (graphs off); a graphs-on cell would be an addendum, registered first. The daemon dies with the power-off.

## Step 3: `TestMemGate_cuda`

**Red at main, as predicted, and by more than the prediction** (`03a-memgate-red.log`, 119 s, 71 fixtures, 484 builds held, 160 options declined by CUDA's own build; the exact error lines are in
`03a-memgate-red-exact.txt`):
- 10 rows built resident and are not in `backend/cuda/testdata/memgate_rows.txt`: the nine `testdata/flint-tiny int4` rows (plain, `/ssm`, `/kv-f16`, `/kv-i8`, `/windowed`, `/expert-cache`, `/slots4`,
  `/windowed+expert-cache`, `/slots2`) **and `testdata/nemotron-tiny int4/kv-f16`**, which no Flint change explains: its nine siblings are in the file, and the row is a nemotron_h load with `--kv f16` that now builds
  resident (nemotron_h is in `decoder/kvresolve.go:cudaKVF16Delivered`).
- 20 unforgiven gap errors, ten rows at two lines (built, generated) each: `flint-tiny` line "dense", leased 113,056 B against a plan of 112,736: **320 B over** on all nine rows, and `nemotron-tiny int4/kv-f16`
  line "dense", 78,144 against 78,080: **64 B over**. Both are MM8's "dense vectors" class (the same 64 B the other nemotron-tiny rows carry as a known gap, and the `qwen25vl-vision-tiny` rows).

`-memgate.rows.update` (`03b-memgate-update.log`) added exactly those ten rows to the file (the diff is the ten lines) and still exited red; the second run (`03c-memgate-rerun.log`) is red on the 20 gap errors
only, no missing row. **Not done, by the brief:** no gap row added, no pin moved. Flint's 320 B and nemotron kv-f16's 64 B are MM8's to price or to table; Francis decides whether a gap row is added.
The three non-error lines `kv leased ... against a plan of ...` (mistral-tiny-window@window32768, llama-tiny@window32768, glm-tiny@experts16384) are in both runs.

## Step 4: `go run ./cmd/gate identity -backend cuda` (default sets, tiny assets, int4 and int8int8)

Old `51ed2ae27` (the parent of the subsystem contracts), new `4edbe93e9`; one build each in temporary worktrees, `GOINFER_*` removed from both dumpers. 5 min 53 s (`04-identity-cuda.log`).
**30 families IDENTICAL, 0 WITHIN TOLERANCE, 0 DIFFERENT, 14 NOT RUN. GREEN.** A NOT RUN is not a pass; every outcome that is not IDENTICAL is in `04-identity-not-identical.txt`:
- 42 cells where both builds decline CUDA (the model runs on the CPU in both), among them: int8int8 MoE experts (CUDA's MoE GEMVs are int4-only), MoE shapes int4 cannot take (moeInter and hidden not multiples), phi3
  (activation quantization), llama4_text, spark2_5, lfm2, granitemoehybrid, laguna, bailing_hybrid and Nemotron-H's MoE block; the exact reason is on each line. These cells stay NOT RUN until a backend admits them; the
  identity read says nothing about them.
- `mellum2-slice` (a 4.2 GB slice, not a tiny checkpoint), `deepseek_v2` (no tiny fixture), `qwen3_asr` and `voxtral` (their goldens are `golden.zip`).
- Two cells whose fixture I had not copied (`gemma4 dense-scaled`, `gpt2 124m`): re-run for those two families only (`04b-identity-cuda-gemma4-gpt2.log`): gemma4 now 6 of 10 cells IDENTICAL (the other four decline on both
  builds), gpt2 declined on both builds.
- `flint` ran 1 of 2 cells (int4; int8int8 declines), IDENTICAL. The tool's own note: 30 families' `validated_at` is not `51ed2ae27`, so nothing here is written to the parity manifest.

The run record (`identity_run.json.gz`) replays with `gate identity -replay`.

## Step 5: G5 for Flint (tier 0, on the card)

`cudaKVF16Delivered` lists `flint` (aea0ee556). What a family joins with (`docs/measurements/kv4-cuda-2026-10-21.md` §2, G5): G1 exact-input identity, G3 delivered bytes, G1a, then the range survey.
- `kv4bGroups` gained three rows on `flint-tiny`: `flint`, `flint/windowed` and `flint/paged-experts` (`backend/cuda/kv4b_groups_test.go`). `TestKV4b_deliveredBytes` PASS on all three (f16 arm allocates half the
  f32 arm's K/V bytes, layers hold f16, element counts equal) and `TestKV4b_layer0StoreIsTheRoundedF32Store` PASS on all three (layer 0's f16 cache equals the f32 arm's rounded to nearest even, bit for bit, decode
  and batched prefill); zero skips (`05a-kv4b-groups-all.log`).
- `TestKV4_scrambledTables` (G1a) gained `flint-tiny` at f32 and f16, windowed: PASS, 93 steps, 2 slots, windowed true, f16 true (`05b-kv4-scrambled-flint.log`).
- Not done: the max |K| and max |V| range survey on a real Flint checkpoint (`~/models/Flint-Q4_K_M.gguf`, with its `.giw` files). It needs a fit check first. Whether flint's `auto` stays f32 is unchanged (D7 keeps it
  f32 until a G4 passes), and `docs/kv-precision-matrix.md` has no flint-tiny row yet.

## What this does not settle

- LW-cu's speed half is registered, not read (`docs/measurements/lwcu-2026-11/registration.md`, PR #897).
- A graphs-on speed cell, the floor test through the acquire path, and the CPU-side SEQ checks are not read here. SEQ's resident adapter and `ScoreSequence` still have test callers only.
- The unforgiven MM8 gaps and the WebGPU and Metal halves of MM9 are the coordinator's and the Mac's.
