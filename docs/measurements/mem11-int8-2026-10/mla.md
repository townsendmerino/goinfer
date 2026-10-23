# MEM11-int8, MLA: int8 projections on a quantized load, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-int8 gates, MLA first" (I8M-G), registered before the code (c65fe44c).
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. **Status: merged on D10 (b), owner, 2026-10-22, then switched off for
every production load on D14, owner, 2026-10-23: no int8 MLA projection, single or paired, clears both DeepSeek family gates, so MLA
is f32 again (option (a)) and the three manifest rows are validated from that build. The int8 form stays, armed by tests only. See
"D14 (b): the search, read" at the end.**

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

## D14 (b): the search, read 2026-10-23

nobara-pc (Ryzen 7 3700X, 62 GB, Go 1.27.3), the CPU forward at int8int8 against the HF bf16 goldens, checkpoints from `~/models`,
`TestD14_mlaProjSearch` (`decoder/d14_mla_search_test.go`) at bb9e96a0b with the harness that 4511a861c then committed unchanged
but for the margin print. Timing lock held shared; numeric, not timed. Logs: `~/goinfer-logs/d14-mla/phase1-moonlight.log`,
`phase2-o-both.log`, `phase3-pairs.log`.

"Margin" is the golden token's logit less the best other token's, teacher-forced on the golden continuation: positive when
greedy takes the golden's token. Moonlight's deciding position is [1] (golden 11, rival 13); V2-Lite's is [3] (golden 549, rival
40619). Phase 1 printed a gap against the top token, so a passing arm there reads 0; its margins come from phases 2 and 3.

| arm (held f32; the rest int8) | f32 cost | Moonlight | margin [1] | V2-Lite | margin [3] |
|---|---|---|---|---|---|
| none (int8 MLA, as #162 shipped) | 0 | FAIL at [1], cosine 0.999542 | −0.0619 | PASS, cosine 0.999297 | +0.1134 |
| kv-A | 95.5 MB | FAIL at [1], 0.999306 | −0.0349 | not run (rule 2) | — |
| o | 339.5 MB | PASS, 0.999342 | +0.0374 | **FAIL** at [3] (40619), 0.997737 | −0.1600 |
| q | 509.3 MB | FAIL at [1], 0.999354 | −0.1774 | not run (rule 2) | — |
| kv-A + o | 435.0 MB | FAIL at [1], 0.999419 | −0.0862 | FAIL at [3], 0.998249 | −0.0904 |
| q + o | 848.8 MB | PASS, 0.999613 | +0.0517 | FAIL at [3], 0.999117 | −0.2825 |
| q + kv-A | 604.8 MB | PASS, 0.999442 | +0.0768 | FAIL at [3], 0.998734 | −0.4194 |
| q + kv-A + o (all f32: option (a)) | 944.3 MB | **PASS**, 0.999508 | +0.0653 | **PASS**, 0.999243 | +0.2667 |

Resident weight bytes (`ResidentWeightBytes`, int8int8) rose by each arm's arithmetic cost to the byte: Moonlight 16,175,272,704
all-int8, 16,514,790,144 with o f32 (+339,517,440), 17,119,555,584 all-f32 (+944,282,880); V2-Lite 15,921,154,304 →
16,865,437,184 (+944,282,880).

**Verdict, by the registered rules.** Only o clears Moonlight alone (rule 1), and with o f32 V2-Lite fails (rule 2), so no single
arm clears. No pair clears either (rule 4): every pair that passes Moonlight fails V2-Lite at [3]. So **(a) ships**: MLA f32 on
every load, giving up the 944 MB.

**What it shows.** No projection carries the flip. Both deciding positions are near-ties: Moonlight's margin is +0.065 even at
f32 MLA, and int8 MLA's whole effect on it is about a tenth of a logit. Each arm moves each margin by a few hundredths to a few
tenths in a direction that differs by checkpoint, and V2-Lite's [3] holds only at the two ends (all int8, all f32). The rerun of
the o arm in phase 2 reproduced phase 1's reading exactly (same resident bytes, same cosine 0.999342, same continuation): the
CPU forward is deterministic, so these are readings of the arms, not noise.

**The ship (d5c5a1d53).** `mlaProjInt8` is false for every production load; `mlaInt8ForTest` arms the int8 form for I8M-G1 to G3
and the harness. The fit estimators follow it (they ask `mlaProjInt8`). GPU backends already quantize their own copy from the f32
(the I8M-G2 codes are the same), so a GPU load is unchanged. `deepseek-tiny` and `kimi-tiny`'s int4 forward goldens are back to
their pre-#162 values byte for byte (`TestInt4_forwardParity` passes on them). A `.giw` that carries int8 MLA records still loads
them as int8: one built between #162's merge (2026-10-22 18:36 PDT) and this change does. nobara's only DeepSeek bundle
(`deepseek-v2-lite.int4.e4h.cuda.v17.giw`, 13:53 that day) predates it.

**The family gate on the shipped build** (d5c5a1d53, log `~/goinfer-logs/d14-mla/gates-shipped.log`, two runs, identical):
`TestDeepseekV2LiteReal_gate` argmax 8913, continuation exact, cosine **0.999243**: **PASS**. `TestDeepseekMoonlightReal_gate`
argmax 17374, continuation exact, cosine **0.999508**: **PASS**. Both equal the last validated readings (7b50947a: 0.99924,
0.99951). The manifest rows `deepseek_v2` and `deepseek_v3` are merged from these `PARITY_ROW` lines (validated at d5c5a1d53);
`kimi_k2` is validated as shared-path via `deepseek_v3` at the same commit. The capability matrix shows the three validated again.

**The cost of (a), speed** (registered above; `TestD14_mlaSpeedAB` at e925bf9cb, the shipped build plus the harness; log
`~/goinfer-logs/d14-mla/speed-ab.log`). Moonlight at int8int8 on the CPU, int8 MLA (A) against f32 MLA (B), both loaded in one
process, interleaved A B B A, 4 pairs of 32 whole decode tokens after the golden prompt, under `timing_lock.py run` (12:14–12:31
PDT). Free memory at every cell 25.4–26.0 GB (the two loads hold 33.3 GB). Two cells were VOID at the start (load 13.63, then 8.52,
from another session's suite finishing) and re-ran in place; every counted cell started at load 3.08–6.58. A warm-up pair was
discarded.

| pair | A tok/s | B tok/s | f32 ÷ int8 |
|---|---|---|---|
| 1 | 0.794, 0.796 | 0.669, 0.674 | 0.8444 |
| 2 | 0.797, 0.840 | 0.666, 0.676 | 0.8197 |
| 3 | 0.808, 0.819 | 0.681, 0.687 | 0.8410 |
| 4 | 0.812, 0.814 | 0.676, 0.671 | 0.8288 |

**f32 MLA decodes at 0.820–0.844× of int8 MLA, 4 of 4 pairs below 1** (midpoint of the middle two 0.835): about 16% slower per
token on the CPU, which is the extra 0.94 GB of f32 projections read per token. This is the CPU reference path only; a GPU load
makes its own int8 from the f32 either way, so GPU decode is unchanged. It is the speed main had before #162, given back, not a
new loss. The absolute rate (0.8 tok/s) is this harness's whole-token CPU forward on the 3700X, not a served figure.
