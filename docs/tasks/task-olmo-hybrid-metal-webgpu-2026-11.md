# Olmo Hybrid on Metal and WebGPU: NegEigval and ONormEps in their DeltaNet (2026-11)

**Registered 2026-11-02, before any code (mac-sb11m; the commit's time is the registration's).** Francis's direction of 09:52 PST,
relayed by the coordinator: do on Metal, then on WebGPU, what #604 does on CUDA (`docs/tasks/task-olmo-hybrid-cuda-2026-11.md`).
Implement `linear_allow_neg_eigval` (NegEigval, the write gate doubled to [0, 2)) and the output norm's own epsilon (ONormEps) in the
backend's Gated-DeltaNet, gate it against the CPU, then claim both in the backend's table, so Olmo Hybrid runs resident again.
One PR per backend.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| OM | active | Metal: review | this doc, § Metal, readings |
| OW | active | WebGPU: review; its Linux goldens are in | this doc, § WebGPU, readings |

## Metal (OM)

**Why.** OP4m-c1 made `program.Metal` the build's admission table. Metal's DeltaNet reads neither setting, so Olmo Hybrid declines to
the CPU (`TestOlmoHybridDeclinesOnMetal`). The CPU computes `beta = 2·sigmoid(b)` and the gated output RMSNorm at 1e-5, beside a
`rms_norm_eps` of 1e-6.

**What changes.**
- `delta_gates` and `delta_gates_rows` take a `betaScale`, 1 or 2, and multiply the sigmoid by it. Both multiplications are exact, so
  every other DeltaNet family is unchanged bit for bit.
- The `delta_gnorm` and `delta_gnorm_rows` dispatches (decode, the batched prefill, and the step's per-row DeltaNet, which is decode's)
  read the DeltaNet's own epsilon instead of the model's.
- Both come from the program's `OpDeltaNet` parameters at build. A model whose DeltaNet layers disagree on them declines.
- `program.Metal` sets `DeltaNegEigval` and `DeltaONormEps`. The step tables do not: Olmo Hybrid stays off the multi-sequence step.
- **Named outcome change:** `olmo_hybrid` goes resident on Metal, at int4. Metal runs no recurrent family at int8 (G44).

**Gates.**
- **OM-1, whole model, tiny (Metal GPU, seconds).** `olmo_hybrid-tiny`, Metal resident against the CPU at int4 on both sides, 128
  tokens, at the bar of `TestQwen35ResidentParityMetal` and #604's O-2: worst cosine at least 0.95, drift (token 1 minus worst) at most
  0.02, and the replay after `Reset` at a self-cosine of at least 0.9999999. The build must be resident (non-vacuity).
  - **Plants, each setting dropped at the build:** both must be red at this bar. #604's O-2 read both red on this fixture on CUDA
    (NegEigval 0.889, ONormEps 0.620). If the ONormEps plant is not red here, OM-2 must carry it before the claim.
  - **Red first:** before the change, the test fails on the decline.
- **OM-2, the prompt route.** ~~A prompt at or above the pass's floor equals the sequential decode's last-row logits within the bar
  the DeltaNet prefill test holds, on `olmo_hybrid-tiny`, with both plants red. If no such test admits this fixture today, it does after.~~
  **Amended before any code or reading:** Metal's batched DeltaNet prefill does not take a post-only model, so Olmo Hybrid's prompts
  run the sequential loop on decode's kernels, which OM-1 measures. Extending the pass to post-only layers is not this item.
  OM-2 asserts that `olmo_hybrid-tiny`'s prefill is not batched, so OM-1 covers its prompts. The rows kernels take the new arguments
  for every family, and OM-4 holds the other DeltaNet families identical through them.
- **OM-3, a real checkpoint.** None is on the Mac (`~/models` has no Olmo Hybrid). **NOT RUN, named.** That is not a pass.
- **OM-4, identity.** `gate identity -backend metal <main base> <head>`: every family that ran stays IDENTICAL, with `qwen3_5`,
  `qwen3_5_moe` and `qwen3_next` among them. `olmo_hybrid`'s cells may change, since their backend changes. They are listed.
- **OM-5, the censuses.** The admission and decision censuses are regenerated, and their diffs touch only `olmo_hybrid` lines.
  Then `touched.py`, `go run ./cmd/lint` and the citation lint.

**Void if:** a GPU run outside the timing lock; a NOT RUN counted as a pass; a bar moved after a reading.
**Stopping rule:** one run each. **GPU time:** a few minutes, under the shared lock.

### Readings, 2026-11-02 (mac-sb11m, branch `olmo-metal-dn-2026-11-02`; shared Mac lock, 13:50 to 14:11 PST)

Tests 0939fa47, code 92fabc7b. Logs are in `docs/measurements/olmo-hybrid-metal-2026-11/`.

- **OM-1: PASS** (`om1-om2-tiny.log`). `olmo_hybrid-tiny`, Metal resident against the CPU at int4 on both sides, 128 tokens:
  - worst cosine 0.999527 (token 1 0.999848, drift 0.0003), argmax 123/128, replay 1.000000000.
  - **Red first:** at the tests' commit, all four subtests fail on the decline naming NegEigval.
  - **Plants, both outside the bar:** NegEigval dropped, worst 0.923217 with drift 0.0756. ONormEps dropped, worst 0.540932 with drift 0.4428. So OM-2 did not need to carry ONormEps.
- **OM-2: PASS.** `olmo_hybrid-tiny`'s prefill is not batched on Metal, so OM-1 covers its prompts.
- **OM-3: NOT RUN.** No Olmo Hybrid checkpoint is on the Mac. So the family has no real-checkpoint reading on Metal.
- **OM-4: PASS** (`om4-identity-metal.log`). `gate identity -backend metal 69fead12 92fabc7b`: 28 IDENTICAL, 1 DIFFERENT, 14 NOT RUN.
  - The DIFFERENT family is `olmo_hybrid`, as named: its int4 cell moves from the CPU to Metal resident.
  - Its int8int8 cell is NOT RUN, since Metal declines a recurrent family at int8.
  - `qwen3_5`, `qwen3_5_moe` and `qwen3_next` are IDENTICAL, and so is every other family that ran.
  - The neighbours pass: the DeltaNet parity, KS-D3, MC3 recurrent, slot, KV8a and kernel tests.
- **OM-5: PASS** (`om5-censuses.log`).
  - **Admission, regenerated:** `sb2_metal_census.arm64` gains 18 lines, and `sb2_admission_census.arm64` and `.amd64` change 7 cells each. Every changed line names `olmo_hybrid`. The amd64 golden was regenerated on this Mac under Rosetta, where every other line already matched it.
  - **Decision (`testdata/mm1_census.json.gz`), regenerated on Linux amd64.** It holds one class, amd64, and its total hashes every group's full text, so it could not be patched from the Mac. Under Rosetta, 4,555 groups differ from it without this change, which is that box and not this change, so regenerating there would have written those in.
    - **The expected diff, measured on the Mac before the regeneration, in one process with the table flipped: exactly 86 groups, all `olmo_hybrid-tiny`** (D1 12, D2 6, D3 56, D4 12).
    - **The reading.** The coordinator regenerated it on nobara (linux/amd64, go1.27.3) at fc9375d1 with `go test ./decoder -run '^TestDecisionCensus$' -census.update`. Against the previous golden, of 29,637 groups: 86 changed, 0 gone, 0 added, and the changed set equals the expected list exactly. CI's `TestDecisionCensus` is the check that it holds on Linux.
    - On the Mac the test is a capture-only SKIP, which is not a pass: the golden has no arm64 class.
    - On this worktree the census also needs `testdata/gpt2` and `testdata/tiny-qwen2-moe` moved aside: gitignored, not in the golden, and holding subdirectories the census cannot read. A local condition, not this change.
  - **`touched.py` 17/17, the lint 7/7 and the citation lint pass** (`om5-touched-lint.log`). The lint and touched ran before the hardware-matrix regeneration below; it changes one generated doc.
- **After merging main (a9a88b1b, 2026-11-02 15:30 PST):** the generated files conflicted, so each was rebuilt on main's version.
  - Against main: both admission censuses change 7 `olmo_hybrid` lines each, and Metal's gains 18. Every changed line names `olmo_hybrid`.
  - `docs/capability-matrix.md`, its JSON and `pull/capability-matrix.json` move one cell, Olmo Hybrid's GPU column from `no` to `yes`. The site module's tests and real build pass.
  - The merge widened `touched.py`'s selection, which found two hand-kept pins OM had not updated: `admissionGolden` (`TestResidentAdmission_matrix`) and `TestMetalTable_noOverclaim`. Both now hold Metal's claims. They would have been red in CI at 1ac53f84.
  - `testdata/mm1_census.json.gz` went back to main's version here, since main changed it too; see the later merge below for its regeneration. The same flip read the same 86 `olmo_hybrid-tiny` groups, key for key.
- **After merging main again (4cf23ccf, 2026-11-02 19:50 PST), with #604's CUDA claims in:**
  - `admissionGolden`'s row is `{"cuda", "metal"}`, and the generated files are rebuilt on main's versions. Against main: 7 `olmo_hybrid` lines in each admission census, 18 in Metal's, and `docs/hardware-matrix.md`'s one row; the capability matrix no longer moves, since #604 already set that cell.
  - The flip on this tree reads the same 86 `olmo_hybrid-tiny` groups, key for key.
  - `testdata/mm1_census.json.gz` was regenerated at this merge, 798b21d2, on nobara (linux/amd64, go1.27.3). Against main's golden, of 29,637 groups: 86 changed, 0 gone, 0 added, equal to that list. This supersedes the regeneration at fc9375d1 above.
  - `touched.py`'s decoder selection has one red, not from this branch: `TestPlan_tableDriven` (its `moe/cuda` cases) and `TestPlan_extraBytesReservedAheadOfExperts` fail the same way at `origin/main` 4cf23ccf on this Mac, in a scratch worktree. Reported, not changed here.
- **Gemini's review of #611** (log: `docs/measurements/olmo-hybrid-metal-2026-11/om-review.log`):
  - **[risk] the census rows at f32 and int8int8: no resident path exists at either quant, so there is nothing for OM-1 to hold.**
    - Those rows flipped to `resident=true`, and 40 of the 86 decision groups are `q=f32`. Running OM-1 at f32 showed the build refuses it ("weights loaded at f32"), as `buildDeltaNet` refuses int8int8 (G44). Both loads run on the CPU, and their decode path says so.
    - `TestOlmoHybridBuildDeclinesOtherQuants` now holds that.
    - The census rows record admission, not the build. All 45 Metal rows that load at f32 read `resident=true`, as do `qwen35-tiny` and `qwen3next-tiny` at int8int8. A static decline for this family alone would be arbitrary, and a general one moves every family's rows, so it is filed as G51 (`docs/queue-correctness.md`), and #611's census diff stays as measured.
  - **[test] the rows kernels.** `TestDeltaNetSeqKernels_matchDecodeBitwise` runs Olmo's geometry at betaScale 2 and an epsilon of 1e-5, and requires some write gate above 1. `TestDeltaNetPrefill_readsBothParameters` shows the prefill's mixer binds both: on qwen35-tiny, each changes all 256 last-row logits. Plant, `delta_gates_rows` ignoring the scale: red on the new case.
  - **[test] layers that disagree.** `TestDeltaNetOpParams_layersMustAgree` rewrites one layer's parameter, each in turn: `deltaNetOpParams` and the build both decline. Plant, the guard removed: red.
  - **[nit]** The Olmo gate skips as a whole when the weights are absent. The PR's memory line now names the two uniforms.
- **Found on the way, and fixed:**
  - `docs/hardware-matrix.md` still listed the decline, and `touched.py` did not run its freshness test. It is regenerated: Olmo Hybrid now reads resident on Metal.
  - `TestSupport_eachParameterCheckDeclines` and `TestSupport_gapsOnOneOperationAllNamed` used `program.Metal` as their table without these settings. They now use a copy with both off, and `TestSupport_metalClaimsOlmoHybridsDeltaNet` holds the claims, and the step tables' lack of them.

## WebGPU (OW)

Olmo Hybrid has never been resident on WebGPU: the census reads `webgpu does not implement [nope post-only-norm qk-norm-whole]`, and NegEigval and ONormEps come on top, a five-feature port. Francis chose (2026-11-02 17:14) to pilot it as a Gemini draft (`gemini/ow-olmo-webgpu`, 09d05b3f) that mac-sb11m verifies and fixes.

**Registered 2026-11-03, before any reading of the draft (mac-sb11m; the commit's time is the registration's).**
- **Named outcome changes:** claiming the three features admits `olmo_hybrid`, `olmo3` and `smollm3` on WebGPU. The draft gates only the first, so all three are gated here.
- **OW-1, Olmo Hybrid, tiny (Mac WebGPU, seconds).** `olmo_hybrid-tiny`, WebGPU resident against the CPU at the same quant, 128 tokens, at OM-1's bar: worst cosine at least 0.95, drift at most 0.02, replay after `Reset` at a self-cosine of at least 0.9999999. The build must be resident. At int4, and at int8int8 where WebGPU admits it.
  - **Plants at int4, each setting dropped at the build:** NegEigval and ONormEps must each be red at this bar. If one is not, a kernel test must carry it before the claim.
- **OW-2, the other two families, tiny.** `olmo3-tiny` and `smollm3-tiny`, the same comparison and bar, at int4 and int8int8. Each claimed feature needs a plant red on a fixture that uses it: whole-vector QK-norm and post-only norm on `olmo3-tiny`, NoPE on `smollm3-tiny`.
- **OW-3, the prompt route.** A prompt through the path each family takes equals the sequential decode's last-row logits at the bar the batched prefill tests hold. `olmo_hybrid-tiny` and `olmo3-tiny` must not be batched, so OW-1 and OW-2 cover their prompts. `smollm3-tiny`, if batched, must match with its NoPE layers in the pass.
- **OW-4, a real checkpoint.** None is on the Mac (`~/models` has no Olmo or SmolLM). **NOT RUN, named.** That is not a pass.
- **OW-5, identity.** `gate identity -backend webgpu <main base> <head>`: every family that ran stays IDENTICAL, except the three named, whose cells move from the CPU to WebGPU. They are listed.
- **OW-6, the censuses.** Admission (arm64 here, amd64 under Rosetta), WebGPU's own census and the hardware and capability matrices regenerated, and their diffs touch only the three families' lines. The decision census's expected groups are measured here by the flip; the coordinator regenerates it on Linux.
- **What each outcome means.** A family or quant under the bar does not ship on WebGPU: it stays declined, or the fix lands and it is read once more. A plant not red means that gate cannot see that feature.

**Void if:** a GPU run outside the timing lock; a NOT RUN counted as a pass; a bar moved after a reading.
**Stopping rule:** one reading per fix. **GPU time:** minutes, under the shared lock.

**Amended after the first reading, the bar unchanged.**
- **The NoPE plant was not red at the bar:** worst 0.999935 on `smollm3-tiny`, where the clean run reads 1.000000. As registered, that bar cannot see NoPE on this fixture. So the plant is held to the clean int4 run: its distance from the CPU must be at least ten times the clean run's. OW-3's plant is the second check: NoPE dropped, the pass's last row leaves the clean decode's bar of cosine 0.999999.
- **Cells added, none read yet:** the claim also admits f32, two KV slots, and `--kv f16` and `--kv i8` on `olmo3` and `smollm3`, so each is read at the same bar.

**Amended after Gemini's review of #627, the bar unchanged.**
- **The NoPE plant's relative check was wrong:** at a clean run of exactly 1.0 it passed a no-op plant, it could fail on another adapter, and it read a zero when run alone. The plant now reads a clean run beside it and its worst cosine must fall by at least 5e-5. That figure is set after the reading of 6.5e-5, so it says the plant moves the logits, not how far they must move on another adapter.
- **Two cells added, not read:** a GQA `olmo3` (OW-2b), since admission claims the feature for any KV heads and the build refused it, and `olmo_hybrid` at `--kv f16`, `--kv i8` and i8 on two slots, since its attention layers now take the generic branch that honors the flag.

### Readings, 2026-11-03 (mac-sb11m, branch `ow-olmo-webgpu-2026-11-03`; shared Mac lock, WebGPU on the Metal adapter)

Logs are in `docs/measurements/olmo-hybrid-webgpu-2026-11/`.

- **OW-1: PASS** (`ow-gates.log`). `olmo_hybrid-tiny` against the CPU, 128 tokens: int4 worst 0.999503, int8int8 0.999527, f32 0.999325, int4 on two slots 0.999503, `--kv f16` 0.999527, `--kv i8` 0.999382 (the same on two slots), each with drift at most 0.0005 and replay 1.000000000.
  - **Plants, both outside the bar:** NegEigval dropped, worst 0.923217 with drift 0.0756. ONormEps dropped, worst 0.542499 with drift 0.4413.
- **OW-2: PASS** (`ow-gates.log`). `olmo3-tiny`: int4, int8int8 and two slots 1.000000, f32 0.997739, `--kv f16` 0.998135, `--kv i8` 0.997052. `smollm3-tiny`: int4 and two slots 1.000000, int8int8 0.999987, f32 0.999833, `--kv f16` 0.999908, `--kv i8` 0.999864.
  - **Plants:** whole-vector QK-norm dropped, worst 0.655777; post-only norm dropped, worst -0.315619: both outside the bar. **NoPE dropped, worst 0.999935, inside the bar,** so it is held to a clean run beside it (the amendments above): a fall of 6.5e-5 against the required 5e-5.
- **OW-2b, GQA: PASS** (`ow-gates.log`). `olmo3-tiny` cut to half its KV heads, generated in the test so no census sees a new fixture: int4 worst 0.999537, int8int8 0.999993, `--kv i8` 0.996685. Whole-vector QK-norm dropped: 0.739258, outside the bar. Before the change the build refused it. That is one tiny generated case, not a real GQA checkpoint.
- **OW-3: PASS** (`ow-gates.log`). `olmo_hybrid-tiny` and `olmo3-tiny`: the pass declines, and each generates through the sequential loop. `smollm3-tiny` at int4 and int8int8: the pass's last row equals the sequential decode's (cosine 1.00000000, maxAbs 0), and with NoPE dropped it leaves that bar, so the pass reads the NoPE layers' table.
- **OW-4: NOT RUN.** No real checkpoint of these families is on the Mac. So none has a real-checkpoint reading on WebGPU.
- **Not read: a Vulkan adapter.** Every reading here is WebGPU on the Mac's Metal adapter. nobara has not run them.
- **OW-5: PASS** (`ow-identity.log`). `gate identity -backend webgpu c1671c24 be23e42b`: 34 IDENTICAL, 3 DIFFERENT, 6 NOT RUN. The one code change after it is `Model.planIneligible`, which decides a plan and no logit. The three DIFFERENT are the named `olmo3`, `olmo_hybrid` and `smollm3`, whose int4 cell moves from the CPU to WebGPU. Every other family that ran is IDENTICAL, `qwen3_5`, `qwen3_5_moe`, `qwen3_next` and Gemma among them. NOT RUN: `qwen2` and `qwen3` (asset missing), `mellum` (its 4.2 GB slice), `deepseek_v2`, `qwen3_asr` and `voxtral` (no tiny golden on this path).
- **OW-6: PASS** (`ow-censuses.log`).
  - **Admission, regenerated:** both arches of `sb2_admission_census` change 28 lines, the three families' and `cohere2`'s. `cohere2` stays declined, and its list of missing features loses `nope`. `sb2_webgpu_census.arm64` gains 18 lines and the slot census 90, all the three families'.
  - **`docs/hardware-matrix.md`** moves the three families' WebGPU cells to resident, and Command-R7B's missing list loses `nope`. The capability matrix does not move.
  - **From Linux, regenerated by the coordinator on nobara at 88ee1f50:** `gpu/testdata/sb2_webgpu_census.amd64`, `sb2_webgpu_slot_census.amd64` and `testdata/mm1_census.json.gz`. The WebGPU census gains the same 18 lines as arm64. The slot census gains the same 90 cells, whose weights differ by arch, as 665 of main's own 955 lines do.
  - **The decision census, read against the flip here:** of 29,637 groups, 156 changed, 0 gone, 0 added, and the changed set equals the flip's list: all D3, all WebGPU cells, 52 each on `olmo3-tiny`, `smollm3-tiny` and `olmo_hybrid-tiny`. The list is in the log.
  - **Merging main past the goldens' commit (#624, #626, #628) moves none of them:** the decision census's dump is byte-identical before and after on arm64 and under Rosetta, and every admission census passes unregenerated.
- **Found on the way:**
  - `Model.PrefillPath` reports "batched" for every WebGPU model, including those the pass declines. Filed as G52 (`docs/queue-correctness.md`).
  - Metal's and CUDA's builds refuse whole-vector QK-norm at `nH != nKV` while admission claims the feature, the mismatch fixed here on WebGPU. A GQA Olmo model therefore goes resident on WebGPU only. Filed as G53.

