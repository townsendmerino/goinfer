# MEM11-int8, MLA: int8 projections on a quantized load, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-int8 gates, MLA first" (I8M-G), registered before the code (c65fe44c).
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. **Status: merged on D10 (b), owner, 2026-10-22: the three DeepSeek-family
manifest rows are pending until nobara re-runs their real-model gates at int8 (ledger row MEM11-int8-MLA-T3).**

**The build (B27's form for Mamba, applied to MLA).**
- **What goes int8:** on an integer-quantized load, q (or the q-LoRA pair), kv-A and o are held as per-row int8, weight-only:
  `linalg.QuantizeInt8` of the f32, the codes every GPU backend makes from them. Metal, CUDA and WebGPU now upload those codes
  as they are.
- **What stays f32:** kv-B, which the backends slice into W_UK and W_UV as f32. An f32, bf16 or f16 load keeps all six f32.
- **The `.giw`** carries the int8 as weight records, read by encoding byte, so an older bundle still loads.

| gate | reading | result |
|---|---|---|
| I8M-G1 holding | `TestMEM11_mlaInt8Holding`, deepseek-tiny: 4 MLA layers int8, kv-B f32; `ResidentWeightBytes` at int4 422,144 → 311,808 B and at int8int8 413,440 → 303,104 (both exactly the projections' saving, 110,336 B); the f32 load's logits hash `0xbd704d52…`, main's | **PASS** |
| I8M-G2 the upload | the codes and row scales, on load and after a bundle round trip, equal `linalg.QuantizeInt8` of main's f32 projections; Metal resident at int4 on deepseek-tiny: logits hash `0x7e63e089…`, unchanged from main (`TestD5_mlaOnMetal`); the Metal MLA tests and WebGPU's MLA tests pass; CUDA vetted and staticchecked from the Mac | **PASS** |
| I8M-G3 CPU fidelity, tiny | deepseek-tiny at int4, int8-MLA CPU against f32-MLA CPU, 32 steps of greedy lockstep: **min cosine 0.999887, argmax 31/32** (bar 0.999, 30) | **PASS** |
| I8M-G4 CPU fidelity, real slice | DeepSeek-V2-Lite-Chat Q4_K_M, layers 0–1, int4: **min cosine 0.997303, argmax 31/32** (bar 0.99, 28); resident weight bytes 1,248,837,632 → 1,178,890,752 for two layers, about 35 MB a layer (~0.94 GB over the model's 27) | **PASS** |
| I8M-G5 the family gate | `TestDeepseekV2LiteReal_gate` and `TestDeepseekMoonlightReal_gate` at int8int8, on nobara, 2026-10-23: V2-Lite argmax and continuation exact, cosine 0.999297; **Moonlight's continuation diverges at its second token** (13 for 11), argmax exact, cosine 0.999542. Below | **FAIL** (Moonlight) |

**Plant** (kv-B quantized too): red. kv-B differs on every layer, and Metal's logits hash moves (`0x621d19fb…`).

**The fit estimate, after CI's first run.** CI went red on `TestFitEstimate_f32PinnedMixersAgreeWithResident`'s Bailing case
(estimate over accounted 1.37 at int4, band 0.85–1.25). Two causes, both fixed here, no band moved:
- the estimators still priced MLA's q-LoRA pair and kv-A at f32; they now price them at int8 on an integer quant
  (`mlaInt8TensorName`), and kv-B stays f32;
- `ResidentWeightBytes` never counted a Kimi Delta Attention layer: Bailing's three KDA layers (six quantized projections, three
  conv kernels, the f32 gates) and the MLA output gate were missing. The ratio on main was 1.16 only because that missing part
  and the f32 MLA projections offset each other. Bailing now reads 0.96 at int4 and 0.92 at int8int8, deepseek-tiny 0.99 and 0.94.

The f32 logits hash in `TestMEM11_mlaInt8Holding` is arm64's (the amd64 jobs failed on it). The CPU reference is bit-identical
within an architecture only, so on amd64 the f32 path is held by the tiny forward golden's tolerance. The test's other checks pass
there, and under Rosetta on the Mac.

**Goldens moved, by this change only.** `TestInt4_forwardParity`'s `deepseek-tiny` and `kimi-tiny` entries are re-pinned (both
MLA). Their argmax is unchanged (120, 73), and their logit sums move 4.247686 → 4.241313 and 2.085561 → 1.977227. Every other
family's entry is unchanged.

**The parity manifest (D10 (b), owner, 2026-10-22).** `deepseek_v2`, `deepseek_v3` and `kimi_k2` were validated rows, a real-model
oracle at int8 residency, the load this change makes numeric. They are now `pending`, with `validated_at`, `date` and metrics
cleared, and their reference and method kept. That is the manifest's own mechanism: `TestParityManifest_fresh` does not
enforce a pending row's hash, and nothing in it was weakened. Their last validated readings stay on record here for the re-run
to compare:
- `deepseek_v2`: argmax 100.0%, cosine 0.99924 (7b50947a, 2026-10-05).
- `deepseek_v3`: argmax 100.0%, cosine 0.99951 (7b50947a).
- `kimi_k2`: shared-path via `deepseek_v3` (b067007).

The capability matrix (`docs/capability-matrix.md`, `.json`, `pull/capability-matrix.json`) now shows the three as pending; the
site module's tests and build pass with it. The other 39 rows' hashes are refreshed with forward goldens green (43 passed).

## I8M-G5 on nobara, 2026-10-23

nobara-pc (Ryzen 7 3700X, 62 GB, Go 1.27.3), the CPU forward at int8int8 against the HF bf16 goldens, checkpoints from `~/models`.
Run at 08:40 with the timing lock held shared; the gates are numeric, not timed. Log: `~/goinfer-logs/kernel-survey-2026-10/mem11-t3.log`.

| gate | at d6214db6b (int8 MLA) | at a570f8dfc (main before #162, f32 MLA) |
|---|---|---|
| `TestDeepseekV2LiteReal_gate` | argmax 8913 = golden; continuation `[8913 13 185 549 6077 280]` = golden; cosine **0.999297**: **PASS** | not re-run (passes at int8) |
| `TestDeepseekMoonlightReal_gate` | argmax 17374 = golden; cosine **0.999542**; continuation `[17374 13 1344 387 7081 306]` against `[17374 11 1155 387 7081 306]`: **FAIL** at `continuation[1]` | argmax 17374; cosine 0.999508; continuation exact: **PASS** |

The second column is a diagnosis, one run, not a re-run of the gate: the same test on the tree just before MLA's int8 merged
(log: `~/goinfer-logs/kernel-survey-2026-10/mem11-diag-pre162.log`). The int8 projections move Moonlight's greedy continuation
at its second token. The last-token logits agree with the golden more closely than f32 MLA's (0.999542 against 0.999508), so
this is a ranking flip at one position, not a drifted forward; the bar is the continuation all the same. Moonlight returns to
the golden's tokens from the fourth position on.

**The manifest is unchanged.** `deepseek_v2`, `deepseek_v3` and `kimi_k2` stay pending; V2-Lite's passing row was not refreshed
on its own, since the three share one forward and one decision. The question is the memory campaign's D14.

## D14 (b): the per-projection search, registered 2026-10-23 before running

Owner decision D14 (b), 2026-10-23: find the int8 MLA projection that carries Moonlight's flip and keep only that one f32;
(a), all MLA projections f32, if no single projection accounts for it.

**Arms.** Moonlight and V2-Lite are both direct-q (`q_lora_rank` null), so the candidates are three, each held f32 on every
layer while the other two stay int8: **q** (`q_proj`), **kv-A** (`kv_a_proj_with_mqa`) and **o** (`o_proj`). kv-B is f32 in
every arm. The cost of keeping each f32, by arithmetic over both checkpoints' shapes (27 layers, hidden 2048, 16 heads, f32
bytes less the per-row int8 codes and scales): kv-A **95.5 MB**, o **339.5 MB**, q **509.3 MB**; all three are the 944 MB
that MEM11-int8 saved.

**Instrument.** The family gate's own body (`deepseekRealGate`: int8int8 load, argmax, 6-token greedy continuation, cosine
≥ 0.99 against the HF bf16 golden), run under a test-only seam that keeps the named projection f32. Each run also logs, at
every continuation step, the logit gap between the golden's token and the token taken: diagnosis, not gated. One run per arm
(the CPU forward is deterministic within a build); not timed, so the timing lock is held shared.

**Order and stopping rule.**
1. Moonlight with each single arm: q, kv-A, o.
2. Each single arm whose Moonlight continuation is exact runs V2-Lite. **An arm clears iff both gates pass** (argmax and
   continuation exact, cosine ≥ 0.99, the bars unchanged).
3. **If more than one single arm clears, ship the cheapest by the bytes above** (kv-A, then o, then q). It is the smallest
   change that clears; the record names every arm that cleared, since a near-tie flip can clear under more than one.
4. If no single arm clears, the three pairs run (diagnosis). A clearing pair is not "a single projection", so it comes back as a
   decision rather than shipping; with no clearing pair, (a) ships.

**Then, for the shipped arm only:**
- **memory:** `ResidentWeightBytes` on Moonlight at int8int8, all-int8 against the shipped arm (it should rise by that arm's
  bytes above);
- **speed:** an in-process whole-token decode A/B on Moonlight at int8int8, the two loads interleaved A B B A in one process
  under `timing_lock.py run`, 4 pairs of 32 decode tokens; the cell's load average and free memory recorded at its start, and
  void under the run-budget rule (load above 8.0, or free memory under the two loads plus 2 GB). Read as a ratio with its pair
  signs; not gated: the decision is the parity bar's, and the speed cost is recorded beside it.

**The ship:** `quantizeProjections` leaves that projection f32 on every DeepSeek-family load (one forward, so `deepseek_v2`,
`deepseek_v3` and `kimi_k2` move together), the fit estimators price it f32, and the three manifest rows refresh from the
passing gate run on the shipped build.
