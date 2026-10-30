# Task: subsystems with contracts — validations that carry over, tests that take seconds, one memory manager (VK0–VK6, MM0–MM6, AX0–AX7, SB1–SB14, PW1–PW6) — 2026-10

> **Status 2026-10-26: SCOPED.** Every owner decision (SD1–SD8) was made on 2026-10-26, and SD4's settings are applied. **SD4:** auto-merge,
> with the checks required on `main` (owner, 2026-10-26: "yes to SD4"; PW1). **SD1:** the towers and the GPU backends
> move from aikit into goinfer, and `linalg` stays in aikit, worked on through `go.work` and tagged weekly (owner,
> 2026-10-26: "yes to this - move the towers and the gpu backends into goinfer."). That is Track AX (§7). Every gate
> below is a sketch: whoever starts a phase registers its gates first, with the rule, void conditions, stopping rule and
> cost basis (`CLAUDE.md`, "Testing tiers").
>
> Written on the owner's ask, in a Cowork conversation the same day: "long runs/gates/tests/? of more than half an
> hour, sometimes multiple hours, are killing development velocity. In my experience that is a bad smell, meaning we
> need to modularize so only modules need testing and e2e tests happen only once a day or so. what are the major
> components that we could refactor into modules that rarely change but still support the needs of the product, with a
> focus on quick testing when the code does change." Then: "also wondering if we could extract a clean
> memory-management subsystem with a clean contract", "what other subsystems should you figure out", and "yes, detailed
> doc(s) please".
>
> **Facts about the code were read at goinfer main `c4cfb4cb` and at aikit main (root v1.75.0, `gpu/v0.35.0`) on
> 2026-10-26.** The census in §1 is [`../measurements/subsystems-2026-10.md`](../measurements/subsystems-2026-10.md),
> with the scripts and outputs that produced it.
>
> **This builds on, and does not replace:** the test-efficiency campaign
> (`docs/completed/task-test-efficiency-2026-09.md`, TE0–TE12), the 2026-10-13 test-wait record and its moves
> (`docs/measurements/test-wait-2026-10.md`), the forward architecture (`docs/tasks/task-forward-architecture-2026-10.md`,
> OP and RF), the KV architecture (`docs/tasks/task-kv-architecture-2026-10.md`, KV0–KV11) and the memory campaign
> (`docs/tasks/task-memory-campaign-2026-10.md`, MEM0–MEM17). Where an item here is the same work as an item there, the
> row says so, and the other doc keeps it.

**Ledger, 2026-10-26.** One row per item, the current state; it supersedes the text below where they disagree, and is
updated in the same commit as the work. `python3 scripts/task_docs.py check` reads it.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| VK0 | done | 2026-10-26 (mac-vk01): 234 of 4,052 commits re-staled 8,895 family validations in 30 days; 8,775 (98.7%) came with no moved golden and no changed recorded metric, an upper bound on what a behavior key saves | §5.3; `docs/measurements/vk0-2026-10.md` |
| VK1 | done | 2026-10-26 (mac-vk01): `internal/kernelprint`, 65 kernels; K1, K2 and K3 PASS on arm64, arm64 `aikit_nodotprod`, amd64 AVX2, `aikit_noavx2`, `aikit_nopopcnt` (`aikit_noavx512` VOID here); all 8 aikit bumps the suite builds against (v1.70.0 to v1.75.0) left every fingerprint unchanged, 385 family re-stalings a fingerprint key would not have made; found: embed's AVX2 Q4_K/Q6_K dequant fallback is forced by no tag | §5.4; `docs/measurements/vk1-2026-10.md` |
| VK2 | done | 2026-10-26 (mac-vk01): part 1, one file per family adapter (#277, A1–A4 PASS); part 2, `core` covers the forward path (Francis 15:31, option 1; B1–B4 PASS, the census roots amended before grading): `model_forward.go` replaces `model.go` in `core`, 26 forward-path files gain a home, `TestParityManifest_coreClosed` holds it; VK0's window: 52 plumbing-only commits (2,089 re-stalings) no longer re-stale, at most 24 forward-path commits newly do | §5.5; `docs/measurements/vk2-2026-10.md` |
| VK3 | done | 2026-10-26 (mac-vk01, Francis 17:37): one record per family, `testdata/parity/<family>.json` and `shared_sets.json`; every reader through `paritykey.ManifestJSON` (the old single file still read at older commits, so `gate shadow` compares across the change); R1–R4 PASS, two families refreshed in parallel now change two files | §5.6; `docs/measurements/vk3-2026-10.md` |
| VK4 | done | 2026-10-26 (mac-vk01): `programFingerprint` per fixture config (37, `testdata/program_fingerprints.json`) and per real-checkpoint config present; F1–F3 PASS, F2 found and fixed the head tie (`tie_word_embeddings`); 33 of 43 families have a fixture program; in no key yet: it enters VK5's shadow key in week 2 (Francis 16:27) | §5.7; `docs/measurements/vk4-2026-10.md` |
| VK5 | active | week 1 graded 2026-10-27 (31 runs, all green; the aikit v1.76.0 bump re-staled 43 families by today's key and none by the fingerprint key, each explained; caveat: no kernel changed); the switch to the kernel-fingerprint key, S1–S5 all PASS, merged (#308); week 2 (option A, Francis 05:52: the program builders out, programs in): W1–W5 all PASS (#310); amended after Gemini's review of #310 (§5.8), and the week starts at the first `vk-shadow` run after that fix merges (mac-vk01, branch `vk5-review310-2026-10`), graded a week later; read early 2026-10-28 (mac-vk5grade, Francis's 16:07 approval): 81 runs, all green, no disagreement, but no push touched a program builder, so it does not qualify to end early and runs to 2026-11-03 08:27 PDT (vk5 record §9); a possible week 3 (option C, decoder's own ops fingerprinted) noted, not started; M | §5.8; `docs/measurements/vk5-2026-10.md` |
| VK6 | blocked | VK5; the daily canary and the coverage-of-reality audit; M | §5.9 |
| MM0 | done | 2026-10-26 (nobara-mm01): #287, `internal/mem/probe` the one reader of the OS's memory account, a fake for tests, and the scan that finds no OS memory read outside it; S | §6.5; MM0's gates |
| MM1 | done | 2026-10-26/27 (nobara-mm01, Francis approved all of MM1 at 17:35 on 10-26): the decision census #293 (and #302), the host guards #295, `Plan` and Metal's context #296, `ResolveKV` and I6's scan #300; Gemini follow-ups #316 and #320; the MoE reserve #321 (Francis's option A, 09:09; G-MA-1 re-registered under option C, 10:36); the census held byte for byte throughout; M | §6.5; MM1's sections |
| MM2 | done | 2026-10-27 (nobara-mm01): MM2a #307, each backend reports its own build demand and `cudaWillBeResident` and `metalWillBeResident` go (Francis's answers of 22:40, 10-26); MM2b #311, `Plan("cuda")` prices the driver's 2 MiB rounding (Francis's option A, 05:50); census unchanged; M | §6.5; MM2a's and MM2b's gates |
| MM3 | done | 2026-10-27 (nobara-mm01, Francis's go of 11:08): a ledger and every allocation a lease naming its plan line, CUDA #328 (MM3a, exact lower bound restored in #372), WebGPU #334 and its follow-up #339 (MM3b), Metal #335 (MM3c). The Mac's MM3c reads (#374): G-MM3c-2 passes after #384 (four tests release their buffers); G-MM3c-4's 3,692 B gap was Metal packing buffers of 4 KB and under into 128 KB chunks, not a ledger over-count (#383). G-MM3c-4 now reconciles the ledger's own charge per buffer to the byte for every buffer of 16 KB and up, grades untraced memory, and plants an over-charge, an under-charge and an out-of-band buffer (Francis 20:39 and 20:51; #390, corrected by #391; mac-vk01's device pass on the M1 Pro, all five cells); M | §6.5; MM3a–MM3c's gates |
| MM4 | blocked | KV4 and KV7; the arbiter, which is KV11's one budget for KV blocks, expert slots and the prefix cache; L | §6.5; KV11 |
| MM5 | blocked | OP4; demand derived from the program, each operation declaring its buffers; M | §6.5 |
| MM6 | active | queue-time pricing #306 (Francis's answers of 22:37, 10-26) and the night profile #312, #322 (Francis's option B, 05:56) merged; G-MM6-5 read FAIL by its rule (#309), which the night profile answers; nobara's G-NP and G-NP2 gates pass. Owed: G-NP-4, the Mac's replay on its night profile, VOID twice (#313: no profile; re-run 2026-10-27 19:00, mac-vk01: the backfill fixed, too few readings); it can answer once the Mac has run night jobs on 3 nights with 5 readings between them; S | §6.5; MM6's sections |
| AX0 | done | aikit PR #21 merged 2026-10-26 (`e262e12`): the README freezes the moving paths at root v1.75.0 and `gpu/v0.35.0`, the newest tags, and records `gpu/webgpu` as retired; S | §7.3 |
| AX1 | done | PR #274 merged 2026-10-26 (the device package split into `cuda/internal/gpudev` and `metal/internal/gpudev`; supersedes §7.1 and §9's `internal/gpudev`); Mac gates §7.6; nobara's CUDA device run (#279) 0 regressions against aikit `gpu/v0.35.0`, and `TestCUDA_launchDirect` segfaults on both sides alike (open, aikit's at the tag; it joins goinfer's CUDA suite at AX2d); `gpu/webgpu` retired, not copied (owner, 2026-10-26); M | §7.3, §7.6; `docs/measurements/ax1-2026-10/README.md` |
| AX2 | done | AX2a #278, AX2b #282, AX2c #283, AX2d #284; every gate PASS (§7.8–§7.10), the CUDA legs on nobara 2026-10-26 (0 differ in every tower-identity pair; the CUDA suite identical by name). Found on the way: `cuda/s4siglip`'s silent registry split (fixed in AX2b), `TestCUDA_launchDirect`'s segfault (aikit's at the tag), and four CUDA tests red on main (`TestCUDAKVSlots_pricedAgainstWhatIsLeft`, `TestResidentDenseBytes_matchesCUDADevice`), none caused by AX; M | §7.7–§7.10; `docs/measurements/ax2-2026-10/README.md` |
| AX3 | done | #289 merged 2026-10-27: no goinfer `go.mod` requires `aikit/gpu`; `internal/movedpaths` refuses the moved paths; gates §7.11 PASS except AX3-G, which `standalone-build.yml` grades at the next release tag; S | §7.3, §7.11 |
| AX4 | done | aikit #22 and #23 merged 2026-10-27, tagged v1.76.0: `vision` and `audio` deprecated toward goinfer, the four tower backends retired, `gpupins` trimmed to the four that stay; gates and results in aikit's `docs/task-ax4-retire-moved-2026-10.md`; S–M | §7.3 |
| AX5 | done | SD2: `gpupins` stopped bumping the four with AX4 (aikit #22); aikit #24 makes a bump fail (the root require held at `v1.75.0`); gates and results §7.12–§7.13; S | §7.3, §7.12, §14 |
| AX6 | done | SD3: `main` may require an aikit pseudo-version; `gate release`'s pins row refuses one, and RELEASING.md says both halves (#350, and its review follow-up); AX6-B passed 2026-10-27 17:44; gates and results §7.13; S | §7.3, §7.13, §14 |
| AX7 | blocked | AX4; re-count aikit bumps, tags and pin-only commits 30 days later; S | §7.4 |
| SB1 | active | step 1 (kernel tests out of packages that import `decoder`, and the ratchet): CUDA #286, #288, #291 (weight packers), #301 (launch geometry), batch 4 #360 (cuda reach 315 → 314, gpu 87/99 → 86/96), batch 5 #425 (`cuda/deltanet_test.go` to `cuda/kernels` against `internal/hybridattn`, cuda 246/314 → 245/313), batch 6 #429 (`kquantGemvRows` to `launchgeom`; the activation-group kernel tests `actgroup_kernel`, `ks_a1` and `kso1c_kernels` to `cuda/kernels`, cuda 245/313 → 244/310), batch 7 #431 (`gemm_w4a8_mma`'s tile geometry to `launchgeom`, `TestGemmMMA_vsExact` to `cuda/kernels`, `TestF32ToF16RNE_ties` to `wpack`, cuda 244/310 → 244/309); Metal #298 (62 test files to `metal/kernels`, Francis 17:36), batch 5 #410 (reach 333 → 330), batch 6 (`internal/hybridattn` holds the DeltaNet and q-gate CPU references, `deltanet_test` and `qgate_kernels_test` to `metal/kernels`, metal 245/330 → 243/328), batch 7 #426 (`internal/mamba2` holds the Mamba-2 mixer; two tests to `metal/kernels`, ratchet unchanged). Metal's test-only batches are done (Francis 08:22 on 10-29): every remaining `decoder` importer there loads a model or tests `metal`'s own production code; gpu's kernel layer #317 (Francis 05:46 and 06:34) after `decoder`'s adapter types #314. gpu batch 5 #433 (`gpu/deltanet_test.go` to `gpu/internal/kernels` against `internal/hybridattn`, gpu 86/96 → 85/95). `cuda/moe_route_demand_test.go` stays in `cuda` (Francis 15:39 on 10-29): it tests `cuda`'s fit policy (`slotMarginBytes`), not a kernel. Left: the op-level backend interface with OP4 and RF1; the kit also covers the towers' operations once it lands (SB9 step M, Francis 17:45 on 10-28); L | §8 SB1 |
| SB2 | active | census first (Francis 11:08): layer D #326; `Admit` #333 and its review #336; step 2's layers B-cuda and B-webgpu #344 and B-metal #357 (on the M1 Pro profiles, after Francis's ratchet answer 1); `Plan` asks `Admit` #348 (decision (a)). Next: the builds' memory and device reads, fed from the machine profiles: scoped 2026-10-29 by mbp2-sb10 (§8 SB2, "the build's memory reads"), CUDA first; M | §8 SB2 |
| SB3 | moved | VK2 and VK4 (the family adapters) | §8 |
| SB4 | blocked | VK5 week 2's grade: Francis chose option A (2026-10-27), the loader identity enters the key as a week-3 shadow on or after 2026-11-03. Built 2026-10-27 (mbp2-sb4, #342): `loaderIdentity`, `testdata/loader_identity.json` (amd64: 39 fixtures, 702 fixture-mode pairs over three backend flag sets; the arm64 entry #359), `paritykey.LoaderDigest`, no key reads it; LI1–LI6 PASS, each test shown red once; LI7 4 of 12 measured, 8 VOID, the post-hoc supplement 486 → 331 re-stalings, all fixture coverage; found: `SerializeWeights` writes an unreadable blob for bf16/f16 weights, and no fixture holds a k-quant tensor | §8 SB4; `docs/measurements/sb4-2026-10.md` |
| SB5 | done | 2026-10-27 (mbp2-sb4, Francis 18:47; the last step nobara-ci01, Francis 21:02): P1–P5 PASS. The census of 38 policy streams, `Forward` and the scripted fake, grammar, draft-model and n-gram speculation (#375); the session policy (#378); sampled n-gram speculation lossless in distribution on the fake, total variation ≤0.014 over 30,000 seeds (#382); the plain decode loop `generateInto` over a target of hooks, with no token stream changed: CPU census unchanged, CUDA 35 and WebGPU 32 resident streams byte for byte (#392) | §8 SB5; `docs/measurements/sb5-2026-10.md` |
| SB6 | done | the scheduling policy moved to `internal/scheduler` with no behavior change: census (#351), M1 (#352), M2 (#353), the §4.1 documents (#355); Francis chose to close at the move (option A, 18:05); J7's priorities and deadlines stay FIFO, a possible future item; M | §8 |
| SB7 | active | the serving shell on an `Engine` interface: wire census (#376, extended #386, #389), engine seam E1–E4 (#376, #379–#381, #386), model handle (#387, #393), ratchet 40 → 31; SB7b, an exported `engine` package (Francis's option B, 2026-10-28 05:01), gates X-census…X-callers registered (mbp2-sb6); its API shrunk (Francis, 2026-10-29 11:41): S1–S4 (mbp2-sb6, #443, #444, and S4's PR), the surface 21 → 8 names plus `enginetest.Scripted`; next, the vision and audio generators behind the engine; L | §8 |
| SB8 | done | 2026-10-27 (mac-ax01): the codec census #338, its review #341, nobara's IDs #345; the move to `internal/codec` (Francis's option 1) C1 #346, C2 #347, C3 #349, every census golden byte for byte; `demo/agent` and `examples/` stay on the public `chat` and `tokenizer` by design; the open contract question (two or more tools under `tool_choice` auto) stays with `docs/tasks/task-tool-grammar-union-2026-09.md`; S–M | §8 SB8 |
| SB9 | active | nobara-mm01 (Francis 20:28): step S, per-tower goldens on the CPU, CUDA and WebGPU, merged (#388); the Mac's Metal and darwin CPU goldens, 2026-10-28 (mac-vk5grade, coordinator's job of 09:47), G-SB9-1M to 4M PASS; the darwin WebGPU golden: Francis chose (a), CI's adapter skips (2026-10-28 12:03); recorded, G-SB9-1W to 5 PASS; step M (the towers as programs) scoped 2026-10-28 by nobara-mm01 (§8 SB9 step M); Francis 17:45: the program type in `internal/vision`, one PR per block shape, SigLIP first; goal: one description per block, walked by every backend, graded by the goldens; S, then M | §8 |
| SB10 | done | 2026-10-29 (mbp2-sb10): scope #422; the schema `internal/config.Settings` and `TestInventory` (tree to schema) #424; the loading flags' names, defaults and help from it #432; `loadflags.Resolve`, pure, #437, with serve's per-model overrides #441; `docs/env-vars.md`'s knob list #438 and tables #442 generated (`TestEnvVarsDoc`, schema to page), `docs/flags.md` already generated (`TestFlagsDoc`); the env-read scan through constants and `GOINFER_GIW_VERIFY` listed #447 (Francis 11:51); the three hidden rows moved #448; G-SB10-1 to 18 PASS; M | §8 SB10 |
| SB11 | done | 2026-10-29 (nobara-mm01; closed by the coordinator, 17:27): the cost-counter schema `internal/cost` and the CPU's counters #358; CUDA #370; WebGPU #377 (Vulkan on nobara); the arm64 WebGPU decode golden on the M1 Pro #385; Metal's counters (mac-vk5grade) with review fixes #411 and #412; G-SB11-18 on AC #423 (mac-sb11m: off cost 0.043% of a 0.5B decode token, on cost 1.012x). G-SB11-1 to 20 PASS (§8 SB11). Left over, each its own row: SB11a, SB11b, SB11c; M | §8 SB11 |
| SB11a | blocked | OP4: its walker emits SB11's operation records on CUDA, WebGPU and Metal, which count per kernel or pipeline until then, and the backends keep only the device counts below an operation; S | §8 SB11 ("How the counters are emitted until OP4"); OP4 |
| SB11b | blocked | Francis's call: SB11's report on the wire as X1's per-turn block. The server sends nothing new until he decides; S | §8 SB11; `docs/tasks/task-turn-telemetry-2026-09.md` |
| SB11c | done | 2026-10-29 (mac-sb11m): `costToken`, `pipelineNames` and `dispatchUnmeasured` moved to `metal/cost_testhooks.go` (`darwin && goinfer_testhooks`), the name cache keyed by resident there; the untagged staticcheck no longer reports them; G-SB11c-1 to 3 PASS; S | §8 SB11, "SB11c" |
| SB12 | active | mbp2-sb4 (Francis 17:41; one PR, #373, by Francis's choice of 18:42, merged): steps 1–7, the census, goldens on recorded inputs, the record-reading graders, run records with byte-identical `-replay` for census, heavy, parity, quick and identity, and the gpu verdict as a function of its tally; Gemini's findings on every step answered; `heavy -stream` refuses (Francis 18:17, A). Left: the gpu gate's check groups, which need recorded `gate gpu` runs: nobara's CUDA run is recorded (2026-10-28, `655f25dd`, FAIL; golden `gpu-verdict`, `docs/measurements/sb12-2026-10.md` §14), and the Mac's Metal run (2026-10-29, mac-sb11m, `e4dd3ed5`, PASS, §15); next: the check groups' split into runner and grader, regraded from those streams; step 8, the Metal cell groups (suite, parity, lifecycle, prefill), C2–C6 PASS (mac-sb11m, 2026-10-29; `docs/measurements/sb12-2026-10.md` §16); step 9, cgofree and webgpu, C2–C6 PASS (mac-sb11m, 2026-10-29, §17); step 10, repo, cleangpu and seam, and a full Metal record replays to the gate's verdict, C2–C6 PASS (mac-sb11m, 2026-10-29, §18); step 10 (CUDA), CUDA's cell groups (suite, parity, graphsforced) and cgofree, C2–C6 PASS (nobara-ci01, 2026-10-29, §19; #460); step 11, the parity sweep reads GATE_RUN once in its runner, C2–C6 PASS (mac-sb11m, 2026-10-29, §20); step 12, the release pre-flight's rows grade a recorded run, C2–C6 PASS (mac-sb11m, 2026-10-29, §21); step 13, comments-only's overall step grades a recorded run, C2–C6 PASS (mac-sb11m, 2026-10-29, §22); next: CUDA's heavy and ptx (nobara); `TestNemotronResidentCUDA_tier1` defaults `NEMOTRON_TIER1` to `"4b"` when unset and skips naming the path when absent (Francis 2026-10-29 07:25, option (a); SB12-N1 to N3 PASS, agy-sb12, #454) | §8 SB12; `docs/measurements/sb12-2026-10.md` |
| SB12p | open | the gpu gate's cuda-parity cell timeout is 20m (Francis 2026-10-29 16:31, option (a); #473, nobara-ci01). Open: find what slowed `./cuda`'s parity package past 10m on a quiet card: 672 tests by 600 s, with `TestLaunchGeometry_identity` and `TestPipelineLint_boundKernelsAreLaunched` in flight at the timeout, against 587 s under a 20m timeout earlier that day; S | `docs/measurements/sb12-2026-10.md` §19 |
| SB13 | done | 2026-10-26 (mac-vk01, Francis 19:52): `internal/layout/imports.json`, 167 arrows, 132 test arrows and five protected rules (the towers, the gpudev halves, the kernel packages, the memory probe, the root's backends); `TestLayout_arrows` fails on a new arrow, a stale one or a rule break; L1–L3 PASS | §9; `docs/measurements/sb13-2026-10.md` |
| SB14 | done | 2026-10-26 (mac-vk01, Francis 21:42): `TestPackageDocs_namedTestsExist` (77 package docs, 21 named tests, invariant lines) and `TestArchitecture_importAppendix` (`docs/ARCHITECTURE.md`'s import appendix generated from SB13's data); D1–D3 PASS; fixed on the way: the hand-kept imports column (stale since AX3) and two stale parity-manifest mentions | §4.1; `docs/measurements/sb14-2026-10.md` |
| PW1 | done | SD4 decided 2026-10-26: branch protection on `main` (19 required checks, not strict, admins not enforced) and "Allow auto-merge" turned on that day; `scripts/main_ci_guard.py`, its tests, hook step and CI step removed and the `CLAUDE.md` rule rewritten (DECISION PW1-D1, Francis 2026-10-26 11:47; PW1-G1' to G5' PASS); sessions set `gh pr merge --auto` themselves (Francis 13:04 and 21:42, every PR from #274 on); closed 2026-10-29 (mbp2-sb10), nothing left; S | §12, §14 |
| PW2 | done | decided 2026-10-27: OS jobs stay required; revisit if one job becomes the long pole by several minutes (Francis 13:19; after PW2a the slowest PR job was about 14 minutes and spread across Windows, `test-rest`, `test-decoder-d` and `gpu-darwin`, so moving Windows to nightly would save 0–2 minutes); S | §12; `docs/measurements/ci-tail-2026-10-27.md` §4a |
| PW2a | done | 2026-10-27 (nobara-ci01, Francis 11:14): MM1's two census tests moved from the decoder tail to their own -race steps in shards a and d; no test dropped, none renamed; G1–G6 PASS; the first PR run 21.9 → 10.1 min to its last job (the tail 19.9 → 8.0), `test-rest` now the slowest; S | §12.2; `docs/measurements/ci-tail-2026-10-27.md` |
| PW2b | done | 2026-10-27 (nobara-ci01; a PW2 follow-up, coordinator 17:30): every `go test` summary step in `ci.yml` runs only when its test step ran, so a failed earlier step no longer adds a missing-file error; no test, job or check changes; B1–B4 PASS, merged as #354; S | §12.4 |
| PW2c | done | 2026-10-27 (nobara-ci01): `test-rest` measured and not changed: its median is 7.7 minutes against `root-windows`'s 10.1, so it is not the long pole and PW2's revisit condition does not hold; the levers for either job are in the record | `docs/measurements/ci-tail-2026-10-27.md` §8 |
| PW3 | done | `docs/README.md`'s counts no longer stored (#324); `CHANGELOG.md`'s unreleased entries are one file each in `changelog.d/`, collected at release by `scripts/changelog.py` (Francis 12:14, option 1; PW3-C1..C4 PASS); `docs/citation-index.md` left as is (2 commits since CC0); the parity manifest split by VK3; S–M | §12 |
| PW4 | done | 2026-10-26 (mac-vk01, Francis 21:42): `night.py wait NAME [--timeout MIN]` follows one queued job through its re-queues and exits 0 / 1 / 2 / 124; W1–W3 PASS in a temporary `NIGHT_DIR`, three mutations shown red | §12; `docs/measurements/pw4-2026-10.md` |
| PW5 | done | retired 2026-10-27 (Francis: option A): `touched.py` passes `-count=1` everywhere again, and the test-cache scanner, its data and wiring are removed; the saving was about 9 s on a 133 s re-run and three review rounds each found a new hole | §12.3; `docs/measurements/pw5-2026-10.md` |
| PW6 | blocked | the first VK, MM and PW items; re-run §1's census 30 days after they land; S | §12 |
| SD1 | done | decided 2026-10-26: the towers and the GPU backends move into goinfer; linalg stays in aikit | §14 |
| SD4 | done | decided 2026-10-26: auto-merge, with the checks required on `main` | §14 |
| SD2, SD3, SD5–SD8 | done | decided 2026-10-26, one at a time (§14) | §14 |

## 0. Summary

**Where the time goes** (§1; the Mac's sessions, 2026-10-13 to 10-26; nobara's were not read, its timing lock was held):
- **By wall clock, waiting on CI after a push is the largest wait: 20.3 h, 16.9% of 120 active hours.** Local tests
  and gates are 10.9%, timed runs 5.8%, nobara and the night queue 4.5%. The share of tests, gates and CI fell from
  37.5% (week of 2026-10-12) to 22.5% (week of 10-19), against 38% in the 30 days before 2026-10-13. The October moves
  worked.
- **The runs of half an hour to hours are timed gates and real-checkpoint gates, not unit tests.** KV8a's M5 passes
  ran 1 h 51 min each, KV1's speed gate 1 h 02 min, served gates up to 2 h.
- **They fail on the box's state far more than on code.** Of the Mac's 131 night jobs since 2026-10-06, 49 did not end
  ok: 33 of those were the box's state (16 memory-guard refusals, 9 voids, 8 re-queues or a busy lock), 4 a failing test.
- **The day loop is minutes, not seconds.** `scripts/touched.py`'s p90 is 4.8 min, and 1,485 of 1,641 session
  `go test` calls passed `-count=1`, so the test cache never replayed a result.
- **CI is 8.6 min at the median and up to 45 min when sessions push together,** from runner queueing, not test time.

**Five structural causes** (§2):
1. **Validations are keyed on source files and a version string, not on behavior.** One 2,996-line file holds every
   family's adapter and sits in the hash set 42 of 43 families use. `decoder/model.go` is in the same set and took 86
   commits in 30 days. aikit's version is mixed into every family's hash, and goinfer changed it 37 times in 30 days.
2. **Integration tests stand where contract tests should.** 240 of cuda's 347 test files, 245 of metal's 401 and 87 of
   gpu's 172 import `decoder`; their production code uses about 100 decoder identifiers in 44 files. Every decoder edit
   re-runs them.
3. **Machine state is a hidden input.** The memory decisions read the live machine, so their tests do; Go's test
   cache cannot see that state, so the day tools disable the cache; and a busy desktop refuses a night job at 2 AM.
4. **A release train runs inside the inner loop.** aikit cut 35 root tags and 8 `gpu/*` tags in 30 days; each bump
   touches every goinfer module and re-stales every family; about 40 pin-only commits a month keep aikit's GPU modules
   on the newest root tag.
5. **Shared hot files and waiting.** At about 110 commits a day on main, a file every change touches (the citation
   index, the changelog, the parity manifest, the docs map) is a queue, and a product push waits for the previous
   push's CI.

**What this doc proposes.** Subsystems whose contracts can be checked in seconds. Each publishes a fingerprint that
the validations of what is built on it key on, so a change re-validates only what its contract says it can change.
Real checkpoints, timed runs and the peer matrix move to a daily run that nobody waits on.

| track | what | first step | size |
|---|---|---|---|
| VK (§5) | validations keyed on behavior: kernel and program fingerprints, per-family adapters and records, a daily canary | VK0–VK3 | S–M each |
| MM (§6) | the memory-management subsystem: a pure planner, a ledger with leases, one arbiter | MM0, MM1 | M |
| AX (§7) | the towers and GPU backends move into goinfer (decided) | AX0–AX2 | M |
| SB (§8) | the other subsystems: backend interface, admission, formats, engine policies, scheduler, serving shell, codec, towers, config, counters, gate infrastructure, layout | SB1's first step, SB13 | S–L |
| PW (§12) | the waiting: auto-merge, the CI matrix, hot files, cross-box completions, the test cache | PW1 (SD4), PW2 | S |

## 1. Where the time goes

The full record, instruments and re-run commands: [`../measurements/subsystems-2026-10.md`](../measurements/subsystems-2026-10.md).

### 1.1 Wall clock (the Mac, 2026-10-13 to 10-26)

`docs/measurements/test-wait-2026-10/session_wall.py` gives every gap between two transcript events to one bucket, so the
buckets add up to wall clock (the owner's time, waits over 3 h and model stalls over 30 min are left out).

| bucket | hours | share |
|---|---:|---:|
| model thinking and writing | 59.1 | 49.4% |
| **waiting on CI after a push** | **20.3** | **16.9%** |
| waiting on local tests and gates | 13.1 | 10.9% |
| tool work (read, edit, git) | 7.1 | 5.9% |
| waiting on benchmarks and measurements | 7.0 | 5.8% |
| waiting on nobara and night jobs | 5.3 | 4.5% |
| build, vet, lint | 4.9 | 4.1% |
| other polling | 2.9 | 2.4% |
| active total | 120 | |

Week of 2026-10-12: tests, gates and CI 37.5% of active time, all waiting 51.7%. Week of 2026-10-19: 22.5% and 40.9%.

**A long run costs latency more than session time.** A 1 h 51 min pass holds a decision for two hours even while the
session does other work, which is why its share of wall clock understates what it costs.

### 1.2 Which commands (2026-10-12 to 10-26; call time, which overlaps)

- `go test`: 1,641 calls, 13.9 h. **1,485 of them (11.9 h) passed `-count=1`.** `scripts/touched.py` passes it for
  every package it tests.
- `touched.py`: 334 runs; median 0.4 min, **p90 4.8 min**, p99 12.7 min; 29 runs of 5 min or more.
- Watching PR checks (`gh pr checks --watch`): 34 calls, 5.6 h, median 9.3 min.
- Polling nobara for remote and night runs: 47 calls, 15.4 h of call time, nearly all of it in the background beside
  other work (the 5.3 h of wall clock above).
- Timed served gates: 7.9 h. `staticcheck`: 2.5 h. Parity refreshes: 159 calls, 1.7 h.

### 1.3 The night queue (the Mac, 2026-10-06 to 10-26)

131 job rows (retries included), 82 ok. Of the 49 that were not:

| cause, from each job's log | rows |
|---|---:|
| the fit or swap guard refused the load | 16 |
| voided (a cell started under heavy load, or a guard declined) | 9 |
| re-queued by its own check, or stopped | 5 |
| the other box or the timing lock was busy | 3 |
| timed out | 3 |
| a test failed | 4 |
| a harness or script error | 2 |
| other | 7 |

The longest jobs were timed gates: KV8a's M5 passes (1 h 51 min each), its M2b (1 h 47 min, failed), tack's Gemma 4 E4B
and E2B Metal runs (timed out at 1 h 20 min and 55 min), KV1's speed gate (1 h 02 min), KV2's share gate (1 h 00 min).

### 1.4 CI

The last 60 `ci` runs: median 8.6 min, p90 12.3 min, max 44.7 min. The three slowest were queueing: in one, every job
started 32 minutes after the run was created; in the other two, one Windows job started 37 and 38 minutes in, after
everything else had finished. When nothing queues, `gpu-darwin` is the longest job (7–9 min). The decoder tail shard
runs 1,205 top-level tests in 249 s under `-race`; its slowest is `TestPrefillChunked_bitIdentical` at 51 s.

### 1.5 The repository's rate

3,260 non-merge commits on main in the 30 days to 2026-10-26; 113 to 164 a day from 2026-10-21 to 10-25. The most
touched files: `docs/citation-index.md` 241, `CHANGELOG.md` 229, `testdata/parity_manifest.json` 225,
`docs/queue-engineering.md` 162, `internal/serveapp/main.go` 116, `docs/README.md` 106, `metal/model.go` 99,
`decoder/model.go` 86.

### 1.6 Limits

- nobara's transcripts were not read. At 09:27 PDT its timing lock was held by a settling night job, and a read from
  the Mac would have run beside it. The both-box figures are the 2026-10-13 record's.
- Call time (§1.2) overlaps; only §1.1 adds up to wall clock.
- Classes and causes are regexes over commands and logs; the record lists them.

## 2. Why: five structural causes

### 2.1 Validations keyed on source files and a version string

- `testdata/parity_manifest.json` validates 43 families. A family's validation goes stale when
  `decoder/parity_manifest_test.go:freshDepsHash` changes: the Go tokens (`goCodeTokens`) of every file in the family's
  sets, then `aikit_version` (`rootAikitVersion`). The `core` set is 11 files; 42 families use it, 42 use `loaders`,
  41 use `quant`.
- **One family's adapter re-stales 42.** `decoder/registry.go` (2,996 lines) holds all 40 `…Architecture(cfg *Config)`
  adapters and is in `core`.
- **Plumbing re-stales 42.** `decoder/model.go` (2,577 lines, 60 functions, 86 commits in 30 days) is in `core`. Most of
  it is load, options and backend plumbing (`Load`, `modelFromOptions`, `parseQuant`, `checkHalf16Load`, `closeBackend`,
  `warnPrefillDeclined`), beside a few numeric helpers (`addResidual`, `normalize`, `normalizeInto`,
  `softcapParallel`).
- **Every aikit bump re-stales every family,** including bumps whose own subject says the output did not change
  ("output unchanged", "bit-identical" and "weights unchanged" appear in several of the 37).
- **The manifest is one shared file** (225 commits in 30 days).
- What exists makes re-staling cheaper, not rarer: token hashing (2026-10-13), the goldens-gated refresh
  (`scripts/refresh_parity_hashes.sh`, about 4 min) and `go run ./cmd/gate identity`, which carries a validation over on
  byte-identical full logits.

### 2.2 Integration tests where contract tests should be

| module | test files | of which import `decoder` | non-test files importing `decoder` | distinct `decoder.X` used |
|---|---:|---:|---:|---:|
| `cuda` | 347 | 240 | 21 | 93 |
| `metal` | 401 | 245 | 16 | 103 |
| `gpu` | 172 | 87 | 7 | 35 |
| `internal/serveapp` | 162 | 39 | 20 | 52 |

- The production seam is narrow (about 100 identifiers per backend); the test seam is the whole decoder, because the
  tests drive a backend through `decoder.Load` and a model fixture. Go re-runs a package when anything it imports
  changes, so every decoder edit re-runs these suites. The 2026-10-13 record named this "the next edge".
- Where the seam is used, it is wide: CUDA's resident has 18 methods that run the model, Metal's 18 and WebGPU's 7, and
  the decoder type-asserts a resident against 34 optional interfaces (the forward-architecture doc, §0, read
  2026-10-13).
- `internal/serveapp`'s tests call `decoder.Load` 43 times; the package takes about two minutes.

### 2.3 Machine state as a hidden input

- Fit, residency and pager decisions read the live machine (free memory, swap, free VRAM) when they decide, and the fit
  guard predicted what each backend would build (`metalWillBeResident` and `cudaWillBeResident`, which MM2a replaced with
  each backend's own report, `decoder/builddemand.go:RegisterBuildDemand`). Their tests read the machine too, so a verdict
  depends on what else is open.
- Go's test cache keys on the test binary, the cacheable flags, the environment variables a test reads and the files it
  opens inside the module (§16). It cannot see a sysctl, a device or a checkpoint under `~/models`. The day tools pass
  `-count=1`, reasonably, and pay for it on every run.
- The night queue shows the cost: 33 of 49 jobs that did not end ok (§1.3).

### 2.4 A release train inside the inner loop

- aikit cut 35 root tags (v1.48.0 to v1.75.0) and 8 `gpu/*` tags in 30 days. goinfer's root `go.mod` changed its aikit
  requirement 37 times, each time in every module. By subject (approximate): 16 for linalg kernels, 14 for the vision
  and audio towers, 5 for embed, formats or mmap, 1 for the GPU module.
- Each root tag runs aikit's ritual (its `RELEASING.md`: releasegate with golangci-lint, apidiff and a CHANGELOG line
  for every new export; perfgate on a real box; vulncheck; CI; the tag; `gpupins --fix`), then goinfer's bump and a
  parity re-arm.
- aikit's eight `gpu/*` modules took 39 to 41 commits each in 30 days, nearly all requirement bumps; their Go code
  changed 0 to 2 times each.
- One bump pinned a pre-release pseudo-version rather than wait for a tag (2026-10-24, KS-O1).
- aikit's own line puts the towers on goinfer's side: kernels in aikit, model semantics in goinfer (the premise of
  aikit's goinfer-kernel-moves task doc). SigLIP, Qwen2.5-VL, Pixtral and Gemma 4's conformer are model semantics.

### 2.5 Shared hot files and waiting

- At this rate the files every change touches are queues (§1.5), and each merge of main is a reason to re-run the
  parity refresh and the citation lint (`CLAUDE.md`, "Day loop").
- A product push to main waits for main's previous CI run (`scripts/main_ci_guard.py`, owner rule 2026-10-18). A PR
  merges only after a session watches its checks: main has no branch protection, and none of the last 30 merged PRs
  used auto-merge.
- Several sessions pushing at once queue the runners (§1.4).

## 3. What must not change

1. **Bit-identity at every migration step,** per backend, on every family fixture and real checkpoint the backend runs
   (the forward-architecture doc, §2 item 1).
2. **The CPU stays the reference;** hand-written kernels stay; this is not a graph compiler.
3. **`decoder`'s public API stays as it is.** A public type is defined in `decoder` with an internal twin, never
   aliased to an internal package: aliasing hid the sampler types' fields from `go doc` on 2026-10-13
   (`docs/measurements/test-wait-2026-10.md` §4).
4. **No new production environment reads** (`testdata/env_reads.txt` only shrinks).
5. **A change to any guard threshold (swap, fit, memory) needs Francis's word.**
6. **The measurement standards and the run budget** (`CLAUDE.md`, "Measurement discipline" and "Run budget").
7. **`RELEASING.md`'s two-step tag** for any change to a seam between modules.

## 4. Design rules for a subsystem

1. **Its contract is narrower than its code, and written down** in its package doc: inputs, outputs, what it may
   assume, what it promises.
2. **It owns a fast oracle:** a reference implementation, a fake, a fixture set or a simulator, in its own package, so
   its tests import nothing above it.
3. **It publishes a fingerprint:** a hash of its outputs on a canonical input set that reaches every dispatch path. A
   validation of something built on it keys on that fingerprint, not on its source.
4. **Packages, not modules.** A new `go.mod` only for a separate toolchain or a separate tag train. Every module is a
   release train; §2.4 is the cautionary tale.
5. **Import direction is a test** (SB13), with ratchets that only fall.
6. **Probes live at the edge.** One package reads the OS or a device; everything else takes a snapshot, as data.
7. **A contract test is checked against how the system calls it.** `CLAUDE.md`'s rule that a unit test supplying its
   own calling convention proves nothing about its callers is enforced by the daily coverage audit (VK6), not by
   reading test names.
8. **The day loop checks contracts; the daily run audits them.** A change that cannot be judged without the daily run
   is a modularity bug (`CLAUDE.md`, "Day loop").

### 4.1 How each subsystem is documented (owner, 2026-10-26)

Four homes, one kind of fact each, and each one checked so it cannot drift:

| where | holds | checked by |
|---|---|---|
| the package's `doc.go` | the contract: what it owns, inputs and outputs, the invariants it promises (each naming its test), what it may import, the one command that tests it in seconds; plus `Example` functions | `go doc`; SB14's test (each named test exists); the import gate (SB13) |
| `docs/code-notes/<package>.md` | the design: why this shape, what was considered and rejected, known limits, links to the task docs and measurements | the citation lint (code cited by declaration) |
| `docs/ARCHITECTURE.md` | the one system map: a section per subsystem, its allowed imports, links to its `doc.go` and code-notes | SB14's test: its import table equals the SB13 gate's data |
| a short `CLAUDE.md` in the module or package folder | working rules for the sessions: the day-loop command, what not to import, which fixtures to use, the gate before a push | review; it stays short, and the root `CLAUDE.md` keeps only repo-wide rules |

- **No `README.md` per package.** `go doc` does not show it, so it becomes a second copy of `doc.go`. READMEs stay at module roots
  (`cuda/`, `metal/`, `gpu/`), where GitHub shows them.
- **No ADR files.** Decisions live in the task docs, dated and in the owner's words (SD1–SD8 here); code-notes link to them.
- **History stays out of all four**, per `CLAUDE.md`'s comment rule: it goes in commit messages, task docs and measurement records.

**The template, for `internal/mem` (MM0 starts from it):**

```go
// Package mem plans, accounts for and arbitrates every byte goinfer holds: weights, KV blocks, recurrent state,
// expert slots, the prefix cache and scratch, on every tier (VRAM, unified memory, host, pinned, file-mapped).
//
// Contract. Solve(Demand, Machine, Policy) is pure and returns a Plan or a named Decline. At run time every
// allocation is a Lease from the Ledger that names its plan line; the Arbiter is the one budget for the
// evictable classes. Machine state arrives as data: mem/probe is the only package that reads the OS or a device.
//
// Invariants, each with the test that holds it:
//
//	I1 nothing is allocated without a plan line        TestLedger_noUnplannedBytes
//	I2 every lifetime closes to zero                   TestLedger_lifetimesClose
//	I3 the ledger agrees with the measured process     TestLedger_closesAgainstProbe (per backend)
//	I4 Solve is pure                                   TestSolve_pure
//	I5 Solve is monotone                               TestSolve_monotone
//	I6 one decider                                     TestNoOtherAllocationEstimate (source scan)
//	I7 thresholds carry the owner's approval           TestPolicy_approvalStamp
//	I8 budgets key on computed bytes                   TestSolve_ignoresOSAccount
//
// Imports: the standard library only (mem/probe adds golang.org/x/sys).
//
// Test it: go test ./internal/mem/... (seconds; no model, no device). The machine zoo is testdata/machines/*.json.
//
// Design: docs/code-notes/mem.md. Plan and phases: docs/tasks/task-subsystems-2026-10.md, Track MM.
package mem
```

And its folder's `CLAUDE.md`, in full:

```markdown
# internal/mem — working rules
- Day loop: `go test ./internal/mem/...` (seconds). Never load a model or touch a device here.
- Import nothing from goinfer. Read the OS or a device only in `mem/probe`.
- A guard threshold changes only in the committed Policy, with Francis's approval stamp.
- A new invariant goes in `doc.go` with its test name, or SB14's test fails.
```

## 5. Track VK — validations keyed on behavior

### 5.1 Today's key

| trigger | re-stales | how often, 30 days |
|---|---|---|
| a code edit to any of the 11 `core` files | 42 families | `decoder/model.go` alone 86 commits |
| an adapter edit in `decoder/registry.go` | 42 families | 20 commits |
| an aikit bump | 43 families | 37 |
| a loader edit (`decoder/weights.go`, `decoder/gguf.go`) | 42 families | 27 and 22 commits |

VK0 turns this into an exact before number.

### 5.2 The target

A validation of family F on backend B carries over while its key is unchanged:

    key(F, B) = H( program fingerprint of F's configs,
                   numerics fingerprints of the kernels F's operations use on B's architecture,
                   loader fingerprint of F's fixtures,
                   walker fingerprint of B )

Each fingerprint is a hash of outputs on canonical inputs and takes seconds. When the key changes, the change is
re-validated at the lowest tier that can answer it, a tiny fixture first, and the real checkpoint runs in the daily run
or before a tag. `gate identity` stays the instrument when a key changes and a real checkpoint is needed.

### 5.3 VK0 — the before number

Replay `freshDepsHash` over the last 30 days of main: for each commit, how many families went stale, which set
triggered it, and whether any golden moved. The output is the count of re-stalings a behavior key would not have made.
Tier 0; minutes. S.

### 5.4 VK1 — a numerics fingerprint for the aikit kernels goinfer calls

- For each `linalg` and `embed` kernel goinfer calls, a canonical vector suite that reaches every dispatch path: M=1
  and M>1 tiles, tails, thresholds, and each forced fallback (CI's `noavx2`, `noavx512`, `nopopcnt` and `nodotprod`
  cells already exist).
- The fingerprint is the hash of the output bits, per kernel and architecture. `freshDepsHash` mixes in the
  fingerprints for the architecture the validation ran on, in place of `aikit_version`.
- First computed in goinfer by a test (seconds); later published by aikit with each tag, so a bump can be judged
  without running anything.
- **Gates (sketch):** a planted one-ULP change in one kernel changes that kernel's fingerprint and only that one
  (mutation, both directions); a release that changes no kernel leaves every fingerprint unchanged; the suite reaches
  every path the dispatch census lists. S–M.

#### VK1's gates, registered 2026-10-26 before the code (mac-vk01)

All tier 0: no model, no device, no timed run. The instrument is the package `internal/kernelprint`: `Compute` runs each
aikit `linalg` and `embed` kernel goinfer's production code calls over a fixed, generated input suite and returns one
SHA-256 per kernel, keyed by the dispatch configuration (GOARCH, `linalg.ActiveKernels().Active`, `ForcedFallbacks()`).
"The dispatch census" is the `cpu-*` entries of `docs/hardware-coverage.json` and the predicates they name.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| K1 stable | two `Compute` calls in one process give identical fingerprints for every kernel | never | one run per configuration | one `Compute`: seconds (estimate, measured in the record) |
| K2 mutation, both directions | on a copy of aikit v1.75.0 with a planted one-ULP change in one kernel's output (+1 ULP and −1 ULP, in two kernels: `MatmulAVAcc64` and GGUF Q5_0 dequantization), exactly the planted kernel's fingerprint changes and every other kernel's stays; the same copy with no plant reproduces the module cache's fingerprints for every kernel (removing a plant restores) | a plant whose text replacement does not apply (a no-op mutation): fix the plant, not the rule | one run: the unplanted copy and four plants | five builds of `linalg`, `embed` and the test, about 1–2 min on the Mac (estimate) |
| K3 census reach | in each configuration run, every `if` in aikit's `linalg` and `embed` (the files built for that configuration) whose condition names a census predicate has its active side executed by the suite (read from a coverage profile of the suite), unless it is on a content-keyed exclusion list that says why goinfer's production code never reaches it; a predicate site the condition does not decide needs both sides | a configuration run under a forcing tag that forced nothing (`ForcedFallbacks()` empty); it is re-run on hardware where the tag forces | one run per configuration | one coverage build and run: under a minute (estimate) |

Configurations run on the Mac: arm64 as built, arm64 `aikit_nodotprod`, and amd64 under Rosetta (`ROSETTA_ADVERTISE_AVX=1`)
as built, `aikit_noavx2` and `aikit_nopopcnt`. No machine here has AVX-512 VNNI, so `aikit_noavx512` forces nothing on
it and the AVX-512 side of K3 is not run here; the record says so, and CI's mixed runner pool or nobara under SDE is
where it can run. `freshDepsHash` and the parity manifest do not change in VK1 (VK5 is the shadow mode).

### 5.5 VK2 — one file per family adapter, and only numeric code in `core`

- Move each `…Architecture` adapter out of `decoder/registry.go` into a file of its family's own. The registry map,
  `resolveArchitecture` and `Architecture.validateResolved` stay where they are. Each family's `own` list names its
  adapter file; `core` keeps the registry mechanism only.
- Move `decoder/model.go`'s numeric helpers into a numeric file in `core`; `decoder/model.go` leaves `core`.
- **Gates (sketch):** no family's state changes at the commit that narrows the sets (every validation carried by
  identity); a planted edit in one adapter stales exactly one family. S–M, mechanical.

#### VK2 part 1's gates (the adapters), registered 2026-10-26 before the code (mac-vk01)

Part 1 moves each `…Architecture` adapter, and each helper only one adapter uses, from `decoder/registry.go` into
`decoder/arch_<family>.go`, unchanged. It is done by a tool that cuts each declaration with its doc comment, then
`goimports`. Each manifest family's `own` list gains its adapter's file and the files of the adapters it calls
(voxtral→llama; qwen3_asr and qwen3_vl→qwen3; qwen2_5_vl→qwen2; qwen3_vl_moe→qwen3_moe and qwen3_vl's m-RoPE helper).
`registry.go` keeps the map, `resolveArchitecture`, `validateResolved`, the bounds check and the two helpers several
adapters share, and stays in `core`. Part 2 (`model.go` and what `core` covers) waits on a decision. Measured while
planning it: 19 files holding forward-path code are in no hash set, among them `program.go` and `program_cpu.go`.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| A1 unchanged code | every top-level declaration in `decoder` has the same Go token stream before and after the move; only the file holding it changes | never | one run | seconds |
| A2 carried by identity | `scripts/refresh_parity_hashes.sh` (goldens-gated) refreshes `deps_hash`; the manifest diff changes `own` lists and `deps_hash` and nothing else: no status, `validated_at`, date or metrics | a golden red before the move (then it is not this change's) | one run | about 4 min (the script's own figure) |
| A3 a planted adapter edit | on a throwaway commit after the move, `gate shadow` stales exactly: `{phi3}` for an edit in `arch_phi3.go`; `{llama, voxtral}` for one in `arch_llama.go`; every family that uses `core` for one in `registry.go` | — | one run each | about a minute in all |
| A4 the census holds | a new test, `TestParityManifest_adapterFilesOwned`, derives each family's adapter files from the registry and the adapters' calls and fails when a family's `own` list lacks one; `gate mutation` turns it red by dropping one entry | — | one run | seconds |

#### VK2 part 2's gates (`core` covers the forward path), registered 2026-10-26 before the code (mac-vk01)

Francis (15:31, relayed): option 1. Move `model.go`'s forward functions into a forward file in `core`, drop its load,
option and generation plumbing from `core`, give every uncovered forward-path file to `core` or to its one family's
`own` list, and add the test that fails when forward-path code sits in a file no set covers.

The rule the test holds: **`core` is closed under calls.** From every function in a `core` file, a type-checked call
graph of `decoder` (the files this GOOS and GOARCH build, calls resolved by `go/types`, interface and func-value calls not
followed) reaches only functions in files some shared set or some family's `own` list covers. Rooting at `model.go`'s
forward reaches 19 uncovered files today. Rooting at all of `core` also reaches the batched prefill, the VL prefill and
config loading: 31 in all. The 12 beyond the 19 follow the same rule. A file only one family subset reaches goes to
that subset's set or `own` lists: KDA → bailing_hybrid, Gemma 4's MoE FFN → gemma4, MLA int8 → deepseek_v2, deepseek_v3
and kimi_k2, the DeltaNet batched path → the `deltanet` set, Qwen3.5's VL prefill → qwen3_5 and qwen3_5_moe. Everything
else goes to `core`.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
**Amended 2026-10-26, before B2–B4 were graded (mac-vk01):** the rule's roots. Rooting at every function in a `core` file
does not converge: each file it adds brings its plumbing in as new roots (`blockspec.go`'s `GenerateStream`,
`lora.go`'s adapter loading), and the closure passed 65 files and was still growing, most of `decoder`. The test roots at
the forward's entry points instead: every function in `model_forward.go` (the per-token forward) and `forwardn.go` (the
batched one), both in `core`. A file the forward reaches is covered, but its other functions are not roots. That is
Francis's wording, "forward-path code". Under it, 26 files were uncovered (`config_nonfinite.go` and
`recommended_sampling.go` drop out, `lifecycle.go` comes in). The test is `TestParityManifest_coreClosed`.

| B1 unchanged code | as A1: every top-level declaration in `decoder` keeps its token stream; only `model.go`'s forward functions change file | never | one run | seconds |
| B2 the forward census | `TestParityManifest_coreClosed` passes after the change and fails on the tree before it (it names the uncovered files); `gate mutation` taking `program_cpu.go` out of `core` turns it red | — | one run each | seconds (one type-check of `decoder`) |
| B3 carried by identity | the goldens-gated refresh; the manifest diff changes the sets, `own` lists and `deps_hash` and nothing else | a golden red before the change | one run | about 4 min |
| B4 planted edits | `gate shadow` on throwaway commits stales: every `core` family for a token in `program_cpu.go` (none before) and in the new forward file; none for a token in `model.go`'s `Load`; exactly `{bailing_hybrid}` for one in `kda.go` | — | one run each | about a minute in all |

### 5.6 VK3 — one parity record per family

`testdata/parity/<family>.json`, plus one file for the shared sets; `TestParityManifest_fresh` reads the directory. Two
sessions working on different families stop colliding, and a refresh touches only the records it changes. S.

#### VK3's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (17:37, relayed): VK3 while VK5's shadow week runs. The layout:
- `testdata/parity/shared_sets.json` holds the shared sets.
- `testdata/parity/<family>.json` holds one family's row, with the same fields and values as today.
- `testdata/parity_manifest.json` is removed.

Every reader goes through one function, `paritykey.ManifestJSON`. It assembles the manifest's shape from the records,
or from the old single file at a commit that predates them. That keeps `gate shadow` and the VK0 replay working across
the change, and it errors where the day-loop selector would silently find no manifest. The decoder writes only the
records whose content changed.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| R1 carried unchanged | the records, assembled, equal the old manifest: every family's fields (`uses`, `own`, `deps_hash`, `status`, `validated_at`, `date`, `reference`, `machine`, `method`, `metrics`) and the shared sets, as JSON values; `TestParityManifest_fresh` passes with no refresh | never | one run | seconds |
| R2 the readers read the records | no Go, script or workflow outside history tools names `testdata/parity_manifest.json`; the decoder's parity tests, `gate quick`'s golden-file map, `gate parity`'s qualifying paths, `gate identity`'s families and `scripts/refresh_parity_hashes.sh` all run on the records; the refresh, run on a planted stale family, rewrites that family's record and nothing else | — | one run | the refresh's goldens, about 20 s warm |
| R3 the same families re-stale | `gate shadow` across the layout change itself (before: the single file; after: the records) classes every family agree-fresh; on a throwaway commit after it, a token planted in `arch_phi3.go` stales exactly `{phi3}`, as VK2's A3 did on the single file | — | one run each | about a minute |
| R4 parallel refreshes touch different files | two families made stale on two branches and refreshed separately change disjoint files (`phi3.json`, `llama4_text.json`), and the branches merge with no conflict | — | one run | two refreshes, under a minute |

### 5.7 VK4 — a program fingerprint

`decoder/program.go:buildProgram` takes an `*Architecture` and an experts predicate, not weights, so a program can be
built from a config alone. Serialize the op kinds and their parameters and hash them: one fingerprint per family fixture
config and per real checkpoint config in `testdata/assets.json`. A refactor that leaves every program unchanged leaves
every family's key unchanged. M.

#### VK4's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (15:55, relayed): VK4 next. Nothing in `program.go` or `program_cpu*.go` changes; the code is new files, so the
nobara sessions working near the program walker see no collision. `programFingerprint` hashes a program by a
reflection walk of its operations and their parameters: kinds, registers, every parameter field by name and value, and
floats by their bits. It writes no type names, so renaming a type is not a program change. `programForConfig` builds
the program from a `config.json` alone. Its experts predicate is the config's: layers `[FirstKDense, NumLayers)` of an
MoE architecture, or Llama 4's own per-layer MoE flags. Every committed fixture `config.json` that resolves to an
architecture gets a fingerprint in `testdata/program_fingerprints.json`; each real-checkpoint directory in
`testdata/assets.json` gets one in the record, computed where its `config.json` is present.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| F1 the table | `TestProgramFingerprint_table` recomputes every committed fixture's fingerprint and fails on any difference from the table (`-update` rewrites it); two computations in one process agree | never | one run | seconds |
| F2 config equals weights | for every fixture whose weights are present on the machine, the program built from its config has the same fingerprint as the program `Load` builds from its weights | — | one run (fixtures absent are listed, not counted) | a few seconds per tiny fixture |
| F3 refactor-blind, change-sensitive | (a) VK2's moves left every program unchanged: the table computed at `f8e248b7` (before part 2) equals the one at this branch; (b) `gate mutation` changing one parameter in one family's layer ops (`gemma4LayerOps`' v-norm epsilon) turns F1 red, and the diff names only Gemma 4 fixtures | — | one run each | seconds |
 — the key, in shadow mode first

On every push to main, compute both today's staleness and the new key, and log every disagreement with its reason.
Switch after one week (SD6, decided 2026-10-26) with no disagreement left unexplained. M.

#### VK5's gates, registered 2026-10-26 before the code (mac-vk01)

Order: VK5 shadow mode first, then VK2 (owner, 2026-10-26 13:06, relayed). Until VK4 exists, the shadow key is today's
key with one input replaced: `aikit_version=<pin>` becomes `kernelprint=<digest>`, the SHA-256 of the dispatch
configuration and every kernel's VK1 fingerprint. The source files are hashed as today. The instrument is
`internal/paritykey` (today's key as a library, and the shadow key), `go run ./cmd/gate shadow <before> <after>`, and a
workflow, `.github/workflows/vk-shadow.yml`, that runs it on every push to main. It is not a required check and has no
concurrency group, so collapsed pushes are not skipped. It writes the comparison to the job summary and to an artifact.
Both fingerprints of a push come from one runner and one suite: the after side builds the suite against the after
pin, the before side the same suite against the before pin.

Across a push, each family is one of: agree-stale, agree-fresh, old-only (today's key re-stales it, the shadow key does
not), new-only (the reverse). Old-only is **explained** when the aikit pin is the only input of today's key that moved
and every fingerprint is unchanged. New-only cannot happen with one suite and one toolchain unless something the key
does not see moved, so every new-only is **unexplained** until someone explains it in the shadow record.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| P1 equivalence | `paritykey`'s today key equals `freshDepsHash` for every family in the tree (a decoder test calling both) | never | one run | seconds |
| P2 classifier | a table test covers every class and reason; removing the "every fingerprint unchanged" condition turns it red (`gate mutation`) | never | one run | seconds |
| P3 history | `gate shadow` on three real pushes from VK0's window: an aikit-only bump the suite builds both sides of (all 43 families old-only, explained), a commit that only edits `decoder/model.go` (all agree-stale), and one that touches no dependency (all agree-fresh) | the suite does not build against a side's pin | one run each | one child build per pin change, about 5 s; under a minute in all |
| P4 the job | on a push to main it finishes and writes its summary; a push whose before commit is missing (a new branch, a force push) or whose before pin the suite cannot build is logged VOID, never as agreement | — | read at the next natural point, never watched (`CLAUDE.md`, "Push finished work") | under 3 min a push (estimate, checked on the first runs) |

The shadow period's stopping rule is SD6's: one week of pushes to main from the workflow's first run, with no
unexplained disagreement. The switch is its own PR and its own decision.

**Two shadow weeks (Francis, 2026-10-26 16:27, option 1).** Week 1, the one running, grades only the swap of the aikit
pin for the kernel fingerprints. When it ends, week 2 grades the key built from VK4's program fingerprints plus the
kernel fingerprints, with the forward's source files out of it. Each week then explains its disagreements by one
change. Week 2's gates are registered before its code, like week 1's.

**Week 1 graded (Francis, 22:40: end it early once a real aikit bump has gone through).**
- **The runs.** 31 `vk-shadow` runs on main, from the workflow's first at 15:01 PDT (2026-10-26) to 23:20. All green,
  none VOID. 17 ran on AVX2 runners and 14 on AVX-512 runners.
- **The one aikit bump** (#305, v1.75.0 → v1.76.0, comments, docs and tools only, on an AVX-512 runner): today's key
  re-staled all 43 families and the kernel-fingerprint key none. All 43 verdicts were old-only, each explained "aikit
  v1.75.0 -> v1.76.0 changed no kernel fingerprint".
- **No other disagreement in the week.** Two pushes (VK2's two parts) re-staled all 43 under both keys; the other 28
  left every family fresh under both.
- **Caveat.** This bump changed no kernel, so the week shows the key carrying a validation over, not catching a change.
  That the fingerprint catches a one-ULP kernel change rests on VK1's K2 mutation test.

Run by run: `docs/measurements/vk5-2026-10/week1-runs-2026-10-27.txt`.

#### The switch's gates, registered 2026-10-27 before the code (mac-vk01)

The switch puts the kernel fingerprints in today's key in place of `aikit_version`. The digest has to be the same on
every runner, so the key mixes in a committed table, not the runner's own fingerprints.
- **The table.** `testdata/kernel_fingerprints.json` holds every kernel's fingerprint per dispatch configuration, for
  the three CI runs on:
  - `arm64[dotprod]`, generated on this Mac;
  - `amd64[avx2,f16c,popcnt]`, generated under Rosetta. Its digest equals the one 17 week-1 runs on AVX2 runners
    recorded;
  - `amd64[avx2,avx512vnni,avx512vnni+vl,f16c,popcnt]`, #303's SDE table. Its digest equals the one 14 week-1 runs on
    AVX-512 runners recorded, so the emulation matched the hardware.
- **The key.** `freshDepsHash` and `paritykey`'s today key end in `kernel_fingerprints=<digest of the table>`.
- **The table's own check.** `internal/kernelprint`'s `TestCompute_table` compares the runner's own fingerprints with
  its configuration's entry, and reports a configuration with no entry as not checked. `-update` writes the entry.
- **The shadow job.** `gate shadow` gains one explained reason: the key's definition changed at this commit.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| S1 the table | `TestCompute_table` passes here on arm64 and amd64-AVX2; the AVX-512 entry's digest equals the real AVX-512 runners' | never | one run each | seconds |
| S2 a one-time re-stale | the goldens-gated refresh changes the 43 `deps_hash` lines and nothing else; P1 (`TestParityKey_matchesFreshDepsHash`) still holds | a golden red before the change | one run | about a minute |
| S3 the pin is out of the key | on a throwaway commit that changes only the aikit requirement, `TestParityManifest_fresh` stays green and `gate shadow` classes every family agree-fresh | — | one run | about a minute |
| S4 a kernel change re-stales | `gate mutation` changing one fingerprint in the table turns `TestCompute_table` red (this config's entry) and `TestParityManifest_fresh` red (every family; VK4's per-family narrowing is week 2's) | — | one run each | seconds |
| S5 the shadow job across the switch | `gate shadow` on the switch commit classes all 43 old-only, each explained by the key's change of definition, with none unexplained | — | one run | about a minute |

**The switch, graded 2026-10-27: S1–S5 all PASS** (`docs/measurements/vk5-2026-10.md` §5). The table's three digests
equal week 1's per configuration; the one-time refresh moved only the 43 `deps_hash` lines, with goldens 33/31/0; an
aikit-only commit leaves every family fresh; one changed fingerprint re-stales all 43; the shadow job explains all 43
across the switch by the change of definition. Added beyond the registration: the shadow job also explains an old-only
family when the table moved only for configurations other than the runner's own.

#### Week 2's gates, registered 2026-10-27 before the code (mac-vk01; Francis, 05:52: option A)

Week 2 shadows a key in which the program builders leave and VK4's program fingerprints take their place.
- **Builders, out:** `decoder/program.go`, `blockspec.go`, `arch.go`, `config.go`, `registry.go` and every
  `decoder/arch_*.go` adapter.
- **Kept in:** the executor, loader/quant/serialize and supporting files, and the kernel table's digest (today's tail).
  The program fingerprint cannot see an executor, so executors stay (the brief's option B is rejected for that reason).
- **The shadow key** for a family with programs is its dependency files less the builders, then
  `programs=<digest>`, then today's tail. The digest is over the family's fixtures' entries in
  `testdata/program_fingerprints.json`, for every GOARCH in it, so every runner computes the same key.
- **A family with no program keeps today's key, unchanged.**
- **The family's programs are a new `programs` field in its parity record:** the fixture directories whose config's
  adapter is defined in that family's own adapter file. Where one adapter file serves several families (`arch_deepseek.go`,
  `gemma12.go`, `arch_llama.go`, `arch_qwen3_moe.go`), the family the fixture's `model_type` names takes it. That gives
  31 families with programs and 12 without: `deepseek_v2`, `kimi_k2`, `gpt-oss`, `gpt2`, `granitemoehybrid`,
  `internlm2`, `laguna`, `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_vl`, `spark2_5`.
- **The shadow job** compares the two keys from the committed tables, so it no longer builds the kernel suite. The
  kernel table moves both keys alike.
- **Explained disagreements:**
  - an old-only family whose every moved input is a builder file;
  - a new-only family whose only moved input is its program list (fixtures added to its key, as at this PR).
- **Unexplained:** every other disagreement. In particular, a family whose programs move while today's key does not
  means a file outside its dependencies changed a program.
- **The limit, stated before the week.** A builder change on a path no fixture exercises leaves every program
  unchanged, so it reads as an explained old-only verdict, exactly like a refactor. Week 2's grade therefore reads
  each such push's builder diff.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| W1 equivalence | a decoder test computes the week-2 key itself (`familyDepFiles` less the builders, the program table, the kernel table) and finds it equal to `paritykey`'s for all 43 families; for the 12 without programs it equals today's key | never | one run | seconds |
| W2 the mapping | every fixture in the program table is in exactly one family's `programs`; each listed fixture has an entry for every GOARCH; its config's adapter is defined in a file of that family's `own` list. Mutation: moving `testdata/llama-tiny` to `voxtral`'s list turns it red | never | one run each | seconds |
| W3a refactor-blind | a throwaway commit adding an unused constant to `decoder/arch.go`: today's key re-stales all 43; `gate shadow` gives 31 old-only, each explained by `decoder/arch.go`, and 12 agree-stale | — | one run | under a minute |
| W3b change-sensitive | a throwaway commit changing one parameter in `arch_gemma3.go` (with the program table's `-update`): `gemma3` agree-stale, the other 42 agree-fresh | — | one run | under a minute |
| W3c a program moved from outside | a throwaway commit changing only `testdata/llama-tiny`'s entry in the program table: `llama` new-only, unexplained, exit 1 | — | one run | seconds |
| W4 classifier | `TestCompare_classes` covers every week-2 case; `gate mutation` dropping the "every moved input is a builder" condition turns it red | never | one run | seconds |
| W5 the job across this PR | `gate shadow origin/main <this PR's head>`: 31 new-only, each explained by its program list; 12 agree-fresh; none unexplained | — | one run | under a minute |

**Week 2's gates, graded 2026-10-27: W1–W5 all PASS** (`docs/measurements/vk5-2026-10.md` §6).
- 31 families have programs, from 37 fixtures; 12 keep today's key.
- W3a: a builder-only edit gives 30 old-only, all explained, and 12 agree-stale. `bailing_hybrid`, which uses no shared
  set, stays agree-fresh, a departure from the registered count that is recorded there.
- W3b: a Gemma 3 parameter change re-stales `gemma3` alone.
- W3c: a program moved without a dependency is new-only and unexplained.
- W5: 31 new-only, each explained by the program list, across this PR.
- W2's test gained the registered tie-break before its plant was run; the record says why.

The week starts at the first `vk-shadow` run on main after this PR merges.

**Amended 2026-10-27 after Gemini's review of #310, before any week-2 run is graded** (`docs/measurements/vk5-2026-10.md` §8):
- `decoder/gemma12.go`, the `gemma` and `gemma2` adapters, is a builder.
- A fixture listed on both sides whose fingerprints moved is a moved input, even when the program list also changed.
- An empty program table is an error.
- `bailing_hybrid`'s record now names the core, loader, quant and MoE sets and deepseek's MLA forward, which its forward
  runs through.
- **Week 2 starts at the first `vk-shadow` run on main after that PR merges;** earlier runs are not graded.

**Week 2's stopping rule** is SD6's: one week of pushes to main from the week-2 job's first run, with no unexplained
disagreement. In addition, every push with an explained old-only verdict has its builder diff read in the record and
classed either as a refactor or as a behavior change on a path no fixture exercises. A behavior change of the second
kind is a finding: the switch waits until a fixture covers that path. The switch is its own PR and its own decision.

**Possible week 3 (option C), not started.** A numerics fingerprint of decoder's own ops (a VK1-style suite over the
executor's op kinds on fixed inputs) would let the executor files leave the key the way the kernel table let the aikit pin
leave it. The hand-written forwards are whole-model paths, so they would need family-level fixtures. It needs its own
gates (reach, mutation). Size L.

### 5.9 VK6 — the daily canary and the coverage audit

- **The canary.** Each day, re-validate a rotating sample of carried-over (family, backend) pairs on real checkpoints,
  so every pair is re-checked within two weeks. A blind spot in a fingerprint then shows up as a canary failure, at
  most two weeks late.
- **The coverage audit.** The daily runs record which (operation, shape class, dispatch path) each real model reaches,
  from SB11's counters, and fail when VK1's suite or SB1's kit does not reach one. That holds the fast tier to the way
  the system actually calls it. M.

### 5.10 What it buys

Most commits stop re-staling anything; an aikit bump with unchanged numerics costs nothing; the manifest stops being a
merge point; real-checkpoint runs move to the daily run and to tags.

## 6. Track MM — the memory-management subsystem

### 6.1 Today

Memory decisions are spread over 34 decoder files (7,100 lines, touched by 141 commits in 30 days) and four internal
packages:

| group | files | lines | commits, 30 days | role |
|---|---:|---:|---:|---|
| `decoder/residency.go` | 1 | 2,019 | 58 | admission and resident builds, with memory checks inside the chain |
| `decoder/fitguard.go` | 1 | 1,022 | 28 | fit arithmetic, weight and KV pricing, the backend predictions |
| `decoder/fitplan.go`, `decoder/fitquery.go`, `decoder/metal_fitctx.go` | 3 | 656 | 11 | fit plans, queries, the Metal context cap |
| `decoder/prefill_budget.go`, `decoder/scratch.go`, `decoder/prefill_chunk_scratch.go` | 3 | 653 | 21 | prefill and step scratch |
| `decoder/weightbytes.go`, `decoder/residentneed.go` | 2 | 427 | 25 | weight and resident byte counts |
| `decoder/kvresolve.go`, `decoder/kvgeometry.go`, `decoder/kvpolicy.go`, `decoder/kvscope.go` | 4 | 529 | 23 | the KV plan |
| `decoder/moepool.go`, `decoder/moepaging.go`, `decoder/moeworkingset.go`, `decoder/layerpaging.go` | 4 | 890 | 5 | expert and layer pager policies |
| probes: `decoder/hostram_*.go`, `decoder/memwatch_*.go`, `decoder/swapwatch.go`, `decoder/faultcount_*.go`, `decoder/coldload*.go`, `decoder/giwmap_*.go` | 16 | 904 | 12 | reading the OS |
| `internal/hostmem` | 6 | 458 | 17 | the available reading (`cmd/hostmem`) |
| `internal/swapguard` | 2 | 235 | 2 | the swap guard |
| `internal/fitcmd` | 1 | 312 | 5 | `goinfer fit` |
| `internal/kvmgr` | 5 | 1,344 | 11 | KV blocks and the prefix index |

- `decoder/fitguard.go:fitCheckFor` reads the checkpoint's tensor headers and the machine while it decides;
  `metalWillBeResident` and `cudaWillBeResident` predict a backend's build instead of asking the backend.
- Because decisions read the live machine, their tests do. The same load passes by night and is refused by day with the
  owner's apps open (2026-10-20: a 5.1 GB budget against a 5.6 GB need), and the night queue spends rows on it (§1.3).
- **`internal/kvmgr` is the model to copy:** it imports only the standard library (`crypto/sha256`, `encoding/binary`,
  `errors`, `slices`), its tests are oracle tests, and it took 15 commits in 60 days while `decoder/residency.go` took
  58 in 30.

### 6.2 The contract (sketch)

```go
// Package mem plans, accounts for and arbitrates every byte goinfer holds. It imports nothing from goinfer.
// mem/probe is the only package that reads the OS or a device; everything here takes its readings as data.
package mem

type Tier uint8  // VRAM, Unified (Apple), HostAnon, HostPinned, FileMapped
type Life uint8  // Process, Model, Session, Request, Step
type Class uint8 // Weights, Experts, KV, RecurrentState, PrefixCache, Scratch, Staging, Tower, Aux

// A Need is one thing someone will allocate, declared before it is allocated.
type Need struct {
	Owner     string // "layer 12 kv", "experts", "prefill scratch", "vision tower"
	Class     Class
	Life      Life
	Bytes     int64  // as requested; a tier's rounding is the allocator's (CUDA rounds 2 MiB and over up to 2 MiB)
	Tiers     []Tier // acceptable placements, best first
	Evictable bool   // may be shed under pressure: experts, cached KV blocks, the prefix cache
}

// Demand is what one load asks for.
type Demand struct {
	Fixed, PerSession []Need
	PerPosition       int64 // KV and recurrent state per token, summed over layers
}

// Machine is a snapshot of one box, taken by mem/probe or read from a recorded profile.
type Machine struct {
	Name       string
	Free       map[Tier]int64 // goinfer's available reading, per tier
	SwapUsed   int64
	Compressed int64
}

// Policy holds the guard thresholds as data. A threshold change needs the owner's word, recorded here.
type Policy struct {
	Margin      map[Tier]int64
	SwapGuardMB int64
	ApprovedBy  string // only Francis writes this
	ApprovedOn  string
}

// Solve is pure: the same inputs give the same plan. A Decline names the need that does not fit, and why.
func Solve(d Demand, m Machine, p Policy) (Plan, *Decline)

// Plan lists the reservations per tier and line, the context cap, the KV blocks, the expert slots, the prefix-cache
// budget, and the arithmetic line the load banner prints.
type Plan struct{ /* … */ }

// Ledger is the runtime account: every allocation is a lease that names its plan line.
type Ledger interface {
	Reserve(n Need, line LineID) (Lease, error)
	Usage() Usage // by tier, class and life; Usage.Unplanned counts bytes with no plan line
}

type Lease interface{ Release() }

// Arbiter is the one budget for the evictable classes (KV11): it grants bytes and sheds leases under pressure.
type Arbiter interface {
	Grant(c Class, bytes int64) int64
	Shed(p Pressure) []Lease
}
```

### 6.3 Invariants, each a fast test

| # | invariant | how it is checked |
|---|---|---|
| I1 | nothing is allocated without a lease that names a plan line | every test asserts `Usage().Unplanned == 0`; tier 0 |
| I2 | every lifetime closes to zero: after a request, a session, a model's `Close` | a lifecycle test per backend on a tiny fixture; tier 0 (MEM2's class of leak) |
| I3 | the ledger agrees with the measured process, within a stated per-backend tolerance | one tiny-model test per backend: `VmHWM` on Linux, the task's footprint on macOS, the device's allocated bytes on CUDA and Metal; seconds |
| I4 | `Solve` is pure | property tests: same inputs, same plan, across runs and goroutines; microseconds |
| I5 | `Solve` is monotone | more free memory never gives a smaller plan; a larger demand never gives a larger context cap |
| I6 | there is one decider | every decline is a `*Decline` from `Solve`; a source scan finds no other estimate of a backend's allocation |
| I7 | thresholds are data with an approval stamp | a test fails when `Policy` differs from the committed one without `ApprovedBy` and `ApprovedOn` (the `night.py add` rule, carried by a type) |
| I8 | budgets key on computed bytes, never on the OS's account of them | darwin's UBC makes RSS report what survived rather than what was asked for (`CLAUDE.md`, "A guard that INVERTS"); the OS reading enters only as `Machine.Free` and in I3 |

### 6.4 How it is tested

- **The machine zoo.** Recorded snapshots, as data: the M1 Pro with the owner's apps open (7.0 GB available and a 4.9 GB
  budget in the 2026-10-15 G2-R log; a 5.1 GB budget against a 5.6 GB need on 2026-10-20), the M1 Pro idle, nobara's
  RTX 2070 SUPER with and without `kwin_wayland`'s 16 MiB context, and a CI runner with no GPU. `cmd/hostmem` already
  makes the available reading; MM0 adds a form that writes a snapshot. `Solve` runs over the zoo × every family config
  × the option grid in seconds.
- **The trace simulator.** Replay recorded request traces (Claude Code's 19,692–20,199-token first requests and its
  8,000–17,000-token thread switches, `docs/measurements/kv0-traffic-2026-10-13.md`) and recorded expert routing
  (`decoder/routercapture.go:routerCapture` captures it) through the arbiter, and read hit rates, evictions and peak
  bytes per class. This answers KV11's open question, a fixed split against one budget, without a GPU.
- **Ledger-closes, per backend** (I3): one tiny model, seconds.
- **Daily calibration.** Each box loads at the plan's edge (the largest model `Solve` says fits) and records plan
  against measured. Drift past a stated bound is a finding, not anyone's red.

### 6.5 Phases

| phase | what moves | gate (sketch) | size |
|---|---|---|---|
| MM0 | the probes (`hostram_*`, `memwatch_*`, `swapwatch`, `faultcount_*`, the cold-load probes) and `internal/hostmem` and `internal/swapguard`'s readings behind `mem/probe`; a fake probe for tests | identical readings through the interface; a source scan finds no OS memory read outside `mem/probe` | S |
| MM1 | `Solve`, out of `fitguard.go`, `fitplan.go`, `fitquery.go`, `prefill_budget.go`, `weightbytes.go`, `residentneed.go`, `kvresolve.go` and `metal_fitctx.go`; tensor-header reads move to the caller, which hands `Solve` numbers | a decision census: for every family config × zoo machine × option cell, the same decision and the same arithmetic line as today's code, byte for byte; tier 0 | M |
| MM2 | backends report their own `Demand` for a build, their rounding included; `metalWillBeResident` and `cudaWillBeResident` go | the census unchanged; I3 per backend | M |
| MM3 | a ledger and leases in the allocators: CUDA (`cuMemAlloc`'s 2 MiB rounding priced), Metal (buffers; no-copy aliasing as `FileMapped`), WebGPU (`gpu/bufaccount.go`) | I1 and I2 per backend at tier 0; I3 | M |
| MM4 | the arbiter, KV11: one budget for KV blocks, expert slots and the prefix cache; `internal/kvmgr` asks it for blocks | the simulator's policy registered and graded first, then device runs | L; after KV4 and KV7 |
| MM5 | demand from the program: each operation declares its buffers (weights, per-sequence state, per-step scratch), and `Demand` is their sum plus the backend's deltas | the census unchanged; a planted new family with no hand-written memory line is priced correctly | M; after OP4 |
| MM6 | `night.py add` and `goinfer fit` call `Solve` against the target box's recorded profile at queue time | replay the last three weeks' refused night jobs: each would have been refused at queue time | S; after MM1 |

#### MM0's gates, registered 2026-10-26 before the code (nobara-mm01)

What moves: the OS memory readings in `decoder/hostram_*`, `decoder/memwatch_*`, `decoder/faultcount_*`, the cold-load
residency sample (`giwResidentFraction`), `decoder/swapwatch.go`'s sampler, all of `internal/hostmem` (which `internal/mem/probe`
absorbs; `cmd/hostmem` prints the same lines), `internal/swapguard`'s readings, the Metal module's direct `hw.memsize` reads and
`cmd/gate`'s `/proc/<pid>/status` read. `decoder`'s exported readers (`HostRAMBytes`, `HostRAMAvailableBytes`, `SwapUsedBytes`,
`SelfSwapReader`, `SelfSwapBytes`, `StartSwapWatch` and its types) stay, as delegates (§3 item 3). Out of scope, named so the
gap is visible: the device readings (`cuMemGetInfo`, Metal's `recommendedMaxWorkingSetSize` and `currentAllocatedSize`) need a
device handle and the device library, which §9 keeps out of the probe's imports; MM2 has the backends report them.

All four are tier 0: no model, no device, no timed run, so the timing lock is not taken beyond `touched.py`'s shared hold.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM0-1 identical readings | (a) every parser test that existed moves with its parser and passes with its expected values unchanged; (b) a frozen copy of each pre-move reader (from `origin/main` at bc365a837), kept in a test file, agrees with the probe on the live box: `Physical`, `SelfSwap`'s availability and `FaultCounts`' ok exactly; `Available`, `SwapUsed` and `SelfSwap` equal, or between two reference reads that bracket the probe's, within 5 tries; (c) every decoder seam (`hostRAM`, `hostRAMAvailable`, `coldAvailBytes`, `coldRAMBytes`, `coldResidentFraction`, `sidecarHostRAM`) and the exported readers return the probe's value | the box has no swap configured and (b)'s swap leg compares 0 with 0 (reported, not passed) | all run to completion | seconds; Linux natively, darwin and windows in CI's root jobs |
| G-MM0-2 a fake drives the fit guard | with `probe.Fake` installed through one test helper, each outcome is reached and named: `guardFit` (through `fitCheckFor` on a tracked tiny fixture) proceeds on an unknown reading, proceeds when it fits, auto-pins, refuses a pinned context, refuses at the floor; `decideGIWFit` reaches the same five; `planColdLoad` through `prepareColdLoad` reaches read-first, no-read-around and nothing | an outcome is reached only by setting a non-probe input (each case changes the fake's readings alone, plus the pin) | all cases run | seconds |
| G-MM0-3 one reader | a source scan (go/ast, not text, so comments do not count) over every non-test `.go` file in all seven modules, outside `testdata/`, finds none of the OS memory reads (the literals `/proc/meminfo`, `MemTotal:`, `MemAvailable:`, `SwapTotal:`, `SwapFree:`, `VmSwap`, `VmHWM`, `VmRSS`, `statm`, `smaps`, `hw.memsize`, `vm.swapusage`, `kern.memorystatus_level`, `vm.page_`, `GlobalMemoryStatusEx`, `host_statistics64`, `task_info`; the selectors `Getrusage`, `Sysinfo`, `SYS_MINCORE`, `Mincore`) outside `internal/mem/probe`. It must go red first: a planted source per pattern is flagged. And `internal/mem/probe` imports the standard library and `golang.org/x/sys` only | the scan reads zero files, or fewer than the tree's count of non-test `.go` files less the excluded ones (a scan that read nothing passes vacuously) | runs to completion | seconds |
| G-MM0-4 builds | `go build` and `go vet` of the root for GOOS linux, darwin and windows; `metal` for darwin; `gpu` and `cuda` for linux with their tags (`cuda goinfer_testhooks`, `gpu goinfer_testhooks`); CI's pinned `staticcheck` on the touched modules after its U1000 canary goes red; `gofmt -l` empty on every touched module | the staticcheck canary does not go red (then nothing was checked) | all run | about a minute |

**MM0 read, 2026-10-26 (nobara, linux/amd64): all four pass.** G-MM0-1: `TestHost_matchesFrozenReaders` and
`TestMemProbe_seamsReadTheProbe` pass on nobara (swap in use was nonzero, so the swap leg was a real comparison); the darwin and
windows legs run in CI's root jobs. G-MM0-2: `TestFitGuard_fakeProbeReachesEveryOutcome` reaches all five `guardFit` outcomes (on
internlm2-tiny with its context raised to 131,072 in a temp copy), all five `decideGIWFit` outcomes and the three cold-load
actions. G-MM0-3: `TestOnlyProbeReadsOSMemory` scans 720 non-test files, flags every planted pattern and passes;
its first run flagged `cmd/gate`'s `/proc/<pid>/status` field names, which now come typed from `probe.ReadProcMemory`.
G-MM0-4: builds and vets clean on all three systems and the tagged modules; the pinned staticcheck's canary went red, then
the touched packages were clean on all three systems (Metal's untagged run lists unused test hooks in files this change does
not touch; CI does not run staticcheck on Metal). The parity manifest stayed fresh: no moved file is in a family's deps.

#### MM1's decision census, registered 2026-10-26 before any code (nobara-mm01)

MM1 moves the fit arithmetic into `mem.Solve`. Its bar: every decision today's code makes, on every cell below, comes out the
same from `Solve` and its callers, byte for byte, the printed and returned lines included. This step records the census
from today's code (`origin/main` at the commit this branch starts from) and commits it. It changes no product code.

**The decisions** (each with every line it prints or returns):

| id | decision | outputs recorded |
|---|---|---|
| D1 | the load-time resident guard: `fitCheckFor` and `decide`, as `guardFit` and `FitQuery` use them | the pin, the refusal and its text, `DenseStreamable`, need, budget, weight, KV and source bytes, `known`, `ratio`, the line `guardFit` would print (cap note or warning), `arithmetic()` |
| D2 | the `.giw` guard: `decideGIWFit` | every `giwFit` field, the refusal text, the note |
| D3 | `Model.Plan`, for cpu, cuda, metal and webgpu | every `Plan` field, `Reason`, `NeedBytes()` |
| D4 | `Model.MetalFitCtx` | every `MetalFit` field |
| D5 | request-time admission: `AdmitPrefillMemoryShare`, and `FitBudgetSummary` | the error text (or none), the summary's five values |
| D6 | `resolveWeightCacheBudget` | the budget |
| D7 | `ResolveKV` for cpu, cuda, metal and webgpu, and `WebGPUCtxCeiling` | delivered precision and reason; the ceiling |

**The cells.**
- **Families:** every tracked tiny checkpoint that loads on the CPU in the plain test binary (directories with a `config.json`
  and tracked weights, and tracked `.gguf` files), each at its own window and, for directories, at a long variant (a temp
  copy with `max_position_embeddings` 131,072) so the auto-pin and floor branches are reachable. D2 and D7 also take the tracked
  real-shape configs (`decoder/testdata/laguna_configs/`, `testdata/nemotron-h-configs/`, `testdata/glm_ocr_real_config.json`),
  which need no weights.
- **Options:** for D1 and D2, backend cpu, cuda or metal, with the GPU backend linked or not (a stub in the registry, as
  `withFakeCUDACompiled` does), × quant int4, int8, f32, int4mix (D1) × context unpinned, 4,096 or 32,768 × KV f32, f16 or i8
  × stream-weights off or on (D1) × the CUDA expert cache off or on (cuda only) × the no-fit-guard knob off or on (one cell
  each). D3–D5 run on each family loaded at int4 and at f32, over a request grid named in the test.
- **Machines:** a zoo of recorded snapshots in `testdata/machines/*.json`. Each names its source, and an entry not read from
  a log is marked as an assumption: the M1 Pro with Francis's apps open (7.0 GB available, the 2026-10-15 G2-R log); the M1 Pro
  under the 2026-10-20 load (a 5.1 GB budget); the M1 Pro near idle (`kern.memorystatus_level` 79%, the B41 record); nobara
  with and without `kwin_wayland`'s CUDA context (host and VRAM read on 2026-10-26); a GitHub runner with no GPU; and a
  machine that reads unknown. Every cell also runs at its own edges: for each decision that changes with free memory, the
  lowest reading at which each outcome holds, and one byte under it, found by search and stored as plain numbers.
- **The host class.** Weight bytes depend on the CPU, because the loader repacks int4 differently on AVX2, AVX-512 VNNI and
  arm64. So the census is keyed by host class (GOARCH plus the measured bytes-per-element table), as `parityWantInt4ByArch` is
  keyed by architecture. This step records nobara's class natively and the VNNI class under Intel SDE, as the int4 parity
  golden was (the read below: they are one class). A class with no entry, arm64 included, is a capture run: the test logs the census hash and skips the comparison.
  The Mac records arm64 when a session there takes it.

**Gates.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1-C1 coverage | every outcome of every decision is reached by at least one cell. D1 and D2: proceed unpriced, fits, warns, auto-pins, refuses a pin, refuses at the floor (D1 also a refusal with `DenseStreamable`, and a CUDA build-peak need). D3: each `Placement` the code can return for each backend. D4: each `Why`. D5: admitted and refused, at least once each. A missing outcome fails the test, naming it | fewer families load than the tracked fixture count less the ones named as not loadable on the CPU (a census that loaded nothing covers nothing) | runs to completion | seconds |
| G-MM1-C2 determinism | two runs in one process, and a run in a second process, give the same census hash | the probe fake is not installed (a live reading leaked in) | runs to completion | seconds |
| G-MM1-C3 the golden | the census equals `testdata/mm1_census.json.gz` for this host class, cell for cell and byte for byte. It is written once by `-update` in this PR and then frozen: MM1's `Solve` PRs reproduce it without `-update`. A cell added, removed or changed is a red run, and the PR that changes it says why | the host class has no entry (a capture run, reported as such, not a pass) | runs to completion | seconds |
| G-MM1-C4 tier | the census test runs in under 30 s on nobara and under a minute in CI, with no model asset, no device and no network | — | — | measured on the first run and recorded here |

**MM1 census read, 2026-10-26 (nobara, Ryzen 7 3700X; timing lock free, shared hold): all four pass.** The census is 29,637
groups. Per decision: D1 8,435, D2 3,948, D3 11,760, D4 2,310, D5 2,730, D6 2 and D7 451, plus the loads group. It covers 105
family variants, and 210 family loads succeeded. The loads that fail are the towers, encoders and ASR models, which the loads
group lists (a family that stops loading is a red run). G-MM1-C1: every listed outcome is reached. G-MM1-C2: two in-process runs
and a second process agree (`cce976c3…`). The first run did not agree: the long variants' temp-dir names reached the fit guard's
line, and they now keep the fixture's name. G-MM1-C3: the golden is written for the amd64 class (bytes per element f32 4,
int8 1.015625, int4 0.5625). Under SDE `-hsw` the census matches it, and under `-icx` the class and the census hash are the
same, so VNNI and AVX2 hosts share it. arm64 is a capture run until the Mac records it. G-MM1-C4: `TestDecisionCensus` takes 16.3 s
and the second-process test 16.6 s. One build of the census took 37 s on the first run and 22 s after the cuts below, and takes 8 s with D1, D3 and D4 in parallel
(D2 stays serial at 5 s, because `decideGIWFit` reads the probe seam on every evaluation; `-race` is clean). Two cuts to the registered grid came first, and they change no
decision's reach: stream-weights only for a `.gguf` (it moves nothing for a directory), and the CPU's KV f16 cell dropped (the
CPU's precision is `KVQuant`'s). The golden holds a 16-hex SHA-256 prefix per group (199 KB); `-census.dump` writes the full text
(92 MB) to diff a red run against its base.

**CI on the PR, root-windows (run 40433980269): red, then fixed on the branch.** Every group differed because Windows built the
keys from OS paths (`..\internal\vision\…`), which the golden holds as `internal/vision/…`. The host class matched the
golden. The keys now use forward slashes on every OS, and the long variants hard-link their weights, falling back to a copy
(a symlink needs a privilege on Windows that the runner happened to have). **G-MM1-C4 is not met on Windows:**
`TestDecisionCensus` took 166 s there, against a minute registered for CI. D2, serial, took 58 to 66 s of each build there,
against 5 s on nobara. The fix belongs to the next MM1 step: once `Solve` takes the available reading as data, D2 no longer
goes through the shared probe seam and runs in parallel like D1, D3 and D4. **D2's part done 2026-10-26:** `decideGIWFit` delegates to `decideGIWFitAt`,
which takes the reading as an argument, so D2 runs in parallel (1 s against 5 s on nobara; each census test 8 s against 16 s).
The golden did not move, `-race` is clean, and a serial check holds `decideGIWFit` through the probe fake equal to
`decideGIWFitAt` at the zoo's readings. G-MM1-C4 on Windows is still unmeasured: CI runs `go test` there without `-v`, so it
reports only the decoder package's time (410 s on 2026-10-27, run 40520519837), not the census's.

#### MM1's `Solve`, registered 2026-10-26 before the code (nobara-mm01; Francis approved all of MM1 at 17:35)

**What moves, in three PRs, each held to the census.**
- **MM1a, this PR: the host guards.** D1, D2, D5 and D6 move into a new `internal/mem`: the resident guard's need, budget,
  ratio, auto-pin search, refusal, warning and cap lines, the `.giw` guard, request admission, the load banner's budget, and
  the weight-cache budget. `mem` takes numbers and returns decisions with their lines. A load's KV comes in as a curve, the
  bytes at a context, because KV is not linear in the context (sliding windows flatten it). Machine state comes in as a
  `mem.Machine`. The thresholds come in as a `mem.Policy` whose default holds today's values (0.70, 0.75, the 2,048 floor, the
  1 GB `.giw` margin, the 256 MiB attention scratch); no value changes. The decoder keeps the reads: tensor headers, configs,
  the probe. It also keeps the types its tests and callers use (`fitCheck`, `giwFit`, `FitDeclineError`), as adapters over
  `mem`. I7's approval stamp is not added here: only Francis writes it, and no threshold moves.
- **MM1b: device planning.** D3 (`Plan`) and D4 (`MetalFitCtx`), with the byte counts they read (`weightbytes.go`,
  `residentneed.go`) handed in as numbers.
- **MM1c: the KV precision rules.** D7 (`ResolveKV`), and I6's source scan: no estimate of a load's memory outside `mem`.

**MM1a's gates.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1-S1 the census | `TestDecisionCensus` passes without `-census.update`, on nobara and on CI's amd64 jobs (linux and windows): every group byte for byte | the census reports a capture run on an amd64 host (the golden's class not found) | runs to completion | 16 s on nobara |
| G-MM1-S2 imports | `internal/mem` imports the standard library only (an import test, as `internal/mem/probe` has) | the test reads no source file | runs to completion | seconds |
| G-MM1-S3 pure (I4) | each `mem` decision, on 10,000 generated inputs, gives the same answer twice in one goroutine and once each in 16 concurrent ones; `mem` holds no package-level variable | — | runs to completion | seconds |
| G-MM1-S4 monotone (I5) | on the same inputs, more available memory never makes the resident or `.giw` guard's outcome worse (refusal, then auto-pin, then fits) or its pinned context smaller, and never turns an admitted request into a refusal | — | runs to completion | seconds |
| G-MM1-S5 the existing tests | the decoder's fit tests pass unchanged: `fitguard*_test.go`, `giwfitguard_test.go`, `fitguard_giw_pinned_test.go`, `prefill_budget_test.go`, `memprobe_test.go`, `fitquery*_test.go`; the `cuda`, `metal` and `gpu` modules build and vet with their tags | — | runs to completion | about a minute |
| G-MM1-S6 builds | build and vet for linux, darwin and windows; CI's pinned staticcheck after its canary goes red; gofmt; `touched.py`; the parity manifest stays fresh, or is refreshed once with its goldens before the push | the staticcheck canary does not go red | runs to completion | about two minutes |

**MM1a read, 2026-10-26 (nobara): all six pass.** G-MM1-S1: `TestDecisionCensus` passes without `-census.update`, every group
byte for byte, and so does the second-process test (16.5 and 16.8 s). G-MM1-S2: `internal/mem` imports the standard library only.
G-MM1-S3: 10,000 generated inputs give the same answers twice in one goroutine and in each of 16 at once, and the package
declares no variable. G-MM1-S4: over a 13-reading ladder, no decision gets worse with more memory. It goes red on a planted
defect (a refusal above 64 GiB), on its first input. G-MM1-S5: the 59 decoder fit tests named in the gate pass unchanged, as do
`internal/fitcmd` and `internal/swapguard`; `cuda`, `metal` and `gpu` vet with their tags. G-MM1-S6: vet on linux, darwin and
windows; the pinned staticcheck after its canary went red (it found `fitCheck.remedy` left unused, now removed); gofmt;
`touched.py`; the citation lint. The parity manifest stayed fresh. The decoder keeps `fitCheck`, `giwFit` and
`FitDeclineError` as adapters: a `fitCheck` becomes a `mem.HostCheck` (its KV as a curve over `estimateKVBytes`), and
`guardFit`, `decideGIWFit`, `AdmitPrefillMemoryShare`, `FitBudgetSummary` and `resolveWeightCacheBudget` call `mem`.


#### MM1b's gates (`Plan` and `MetalFitCtx`), registered 2026-10-26 before the code (nobara-mm01)

**What moves.** `Model.Plan`'s placement arithmetic (the context search toward the 4,096 floor, the resident and expert-cached
fits, the slot count, every reason line, and the CPU's weight-paged fallback) becomes `mem.SolvePlacement`. `MetalFitCtx`'s
search becomes `mem.SolveMetalCtx`, which calls `SolvePlacement` itself. The decoder keeps the eligibility checks: missing
features, `ResidentEligible`, and WebGPU's decline of a KV flag on the families that do not implement it. Those read the
model's architecture, not its bytes. The decoder also keeps the byte counts it hands in: dense, experts, the host copies, and
KV as a curve per backend. `mem` gets its own `Placement`, which the decoder maps to its public one (§3 item 3: never an
alias). `Policy` gains the plan floor (4,096) and Metal's fit margin (384 MiB) at today's values.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1b-1 the census | `TestDecisionCensus` passes without `-census.update`, D3 and D4 included, byte for byte | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-MM1b-2 pure and monotone | `TestSolve_pure` and `TestSolve_monotone` cover `SolvePlacement` and `SolveMetalCtx` on generated inputs. More free memory never moves a placement down the order decline, weight-paged, expert-cached, resident; never gives fewer slots at the same placement or a smaller context at the same placement; and never gives `MetalFitCtx` a smaller context. The monotone test goes red on a planted defect | — | runs to completion | seconds |
| G-MM1b-3 the existing tests | the decoder's `Plan` and `MetalFitCtx` tests (`fitplan*_test.go`, `metal_fitctx*_test.go`, every test calling `.Plan(` or `MetalFitCtx(`) and `metal/`'s fit tests (vet on darwin) pass unchanged; `cuda`, `metal` and `gpu` vet with their tags | — | runs to completion | about a minute |
| G-MM1b-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**G-MM1b-2 amended before its pass, 2026-10-26 (nobara-mm01), with the evidence.** The registered rule ("never fewer slots at
the same placement") fails on today's planner, by design: it takes context before experts. Measured on a generated MoE (8
experts, top-2; first written as top-8, which is impossible with two slots) at 16 GiB free, it was expert-cached at context 6,080 with 8 slots; at 32 GiB, expert-cached at context
83,328 with 2. The census shows the same arithmetic before and after the move, so the rule, not the move, was wrong. Holding
slots monotone would change placements, which is Francis's call; it is not done here. The amended rule: placement never moves
down the order; at the same resident or expert-cached placement, the context never shrinks, nor the slots at the same context;
`MetalFitCtx`'s context never shrinks. A refusal's context only echoes the attempt and is not compared. It goes red on a
planted defect (a refusal above 32 GiB free).

**MM1b read, 2026-10-26 (nobara): all four pass, G-MM1b-2 as amended.** G-MM1b-1: `TestDecisionCensus` passes without
`-census.update`, D3 and D4 included, every group byte for byte. G-MM1b-2: `TestSolvePlacement_pure` (10,000 inputs, twice and
in 16 goroutines) and `TestSolvePlacement_monotone` pass. G-MM1b-3: the 18 decoder `Plan` and `MetalFitCtx` tests pass
unchanged. `TestPlan_extraBytesReservedAheadOfExperts` skips for want of `testdata/gemma4-moe-tiny`, as it does on main. `cuda`,
`metal` and `gpu` vet with their tags. G-MM1b-4: vet on three systems, the pinned staticcheck (it found `gb` left unused, now
removed), gofmt, the layout test, the citation lint; the parity manifest stayed fresh.

#### MM1c's gates (`ResolveKV`, and I6's scan), registered 2026-10-26 before the code (nobara-mm01)

**What moves.** `ResolveKV`'s resolution order becomes `mem.ResolveKV`: no KV for a model without attention; the CPU cache's
rules (batched decode reads f32 only; f16 and i8 where the family allows it); Metal's (MLA keeps f32; auto is f16); CUDA's
(int8 not yet; the drafter keeps f32; a family the policy declines keeps f32; auto per the policy); and the rest (auto is
f32). The decoder keeps the policy registry and its public types (`KVFacts`, `KVF16Policy`, `RegisterKVF16Policy`,
`CUDAKVF16Decline`, `CUDAKVF16AutoDecline`), because the cuda module registers against them. The policy reaches `mem` as
closures bound to the load's options. The decoder also keeps `resolveKVHook`, a test seam, and the architecture predicates,
which `mem` takes as booleans.

**I6, as far as MM1 reaches.** A source scan over every non-test Go file outside `internal/mem` finds each reference to a
guard threshold: `WeightsMemFraction`, `fitMemFraction`, `prefillAvailFraction`, `giwMemMargin`, `residentMemFraction`,
`MetalFitMarginBytes`, `ctxPlanFloor`, `fitWarnRatio`, `ctxFloor`, and `mem`'s own constant names. Each must be a constant
declaration that aliases `mem` (directly or through one decoder alias), or a site on a ratchet list whose per-file count may
only fall. The list today: `gpu/kv_slots.go` 2 and `metal/backend.go` 3, the backends' own guards, which MM2 removes. The
scan must go red on a planted reference.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM1c-1 the census | `TestDecisionCensus` passes without `-census.update`, D7 and every decision that reads `ResolveKV` (D1, D2 through `kvPricingFor`) included | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-MM1c-2 the existing tests | every decoder test that calls `ResolveKV`, `kvRefusal`, `KVPrecisionReports` or `KVDelivered`, and the `cuda` module's KV policy tests (vet with tags; the device ones run on nobara), pass unchanged | — | runs to completion | about a minute |
| G-MM1c-3 I6's scan | `TestNoGuardThresholdOutsideMem` passes on the tree, reads every non-test file, and goes red on a planted reference | it reads fewer files than the tree holds | runs to completion | seconds |
| G-MM1c-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM1c read, 2026-10-26 (nobara): all four pass.** G-MM1c-1: `TestDecisionCensus` passes without `-census.update`. G-MM1c-2: the
decoder's six KV-resolution tests pass unchanged, the generated precision matrix among them. The `cuda` module's KV16 and KV4
identity and default tests pass on the RTX 2070 SUPER under the timing lock (47 cases with subtests), and so does the `gpu`
module's `ResolveKV` test. G-MM1c-3: `TestNoGuardThresholdOutsideMem` reads 726 non-test files. It finds exactly the ratchet
(`gpu/kv_slots.go` 2, `metal/backend.go` 3) and counts a planted reference. G-MM1c-4: vet on three systems and with the
module tags, the pinned staticcheck, gofmt, `test_touched.py`; the parity manifest stayed fresh. MM1 is done with this:
D1–D7 run through `internal/mem`, and the census holds byte for byte.

#### MM6's gates, registered 2026-10-26 before the code (nobara-mm01; Francis's answers of 22:37)

**Francis's decision (22:37):**
- 1a: refuse at queue time, with an override flag.
- 1b: the idle profile prices night jobs, the apps-open profile day jobs.
- 1c: new repeatable `--fit-model PATH --fit-opts "..."` flags, priced through `goinfer fit`. A job queued without them is not
  checked and gets a warning.

**What is built.**
- **`fit -profile FILE`** prices a checkpoint as `Load`'s own memory check would on that recorded machine, without loading it.
  It is `decoder.FitQueryAvailable`, which is `FitQuery` with the available reading taken from the profile instead of the
  probe. It prints one line (or `-json`) and exits 0 when the load proceeds (fits, auto-pins, or cannot be priced), or 4 when
  it would be refused.
- **`night.py add --fit-model PATH [--fit-opts "..."]`**, repeatable, runs that check for each declared load:
  - the profile is the box's night profile for a job with `--not-before`, else its day profile;
  - the binary is the chat build for the job's `-backend` (`cuda/cmd/chat`, `metal/cmd/chat`, else `demo/chat`), because
    what `Load` prices depends on the backends linked in;
  - a refusal stops the queueing unless `--fit-override REASON`, which the job records;
  - a job with no `--fit-model`, or a box with no recorded profile, is queued with a warning.
- The box-to-profile table: `nobara-pc` uses `nobara-no-desktop` at night and `nobara-desktop` by day; `macbookpro.lan` uses
  `m1pro-near-idle` and `m1pro-apps-open`.
- **Out of scope here:** Metal's own resident guard (`metal/backend.go`, against physical RAM) and CUDA's device-side
  context fit, which run in the backend at build time.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM6-1 the profile check | on tracked tiny fixtures, `fit -profile` against generated profiles answers as `decoder.FitQueryAvailable` does: exit 0 and `fits` at ample memory, exit 0 and `capped` at the auto-pin edge (long-context copy), exit 4 with `Load`'s own refusal text below it, exit 0 and `unpriced` at an unknown reading; and `FitQueryAvailable` at the probe's reading equals `FitQuery` | — | runs to completion | seconds |
| G-MM6-2 `night.py add` | with a scripted checker: a refused load is not queued and the message names the model, the profile and the refusal; `--fit-override` queues it and records the reason; a job without `--fit-model` queues with a warning; a `--not-before` job is priced on the night profile, any other on the day profile; an unknown host queues with a warning; the backend picks the binary | the tests do not reach `cmd_add` (they must queue into a temp queue dir) | runs to completion | seconds |
| G-MM6-3 end to end | `night.py add` with its real checker on nobara: a tracked tiny checkpoint against a profile too small for it is refused, and against `nobara-no-desktop` is queued (into a temp queue dir), through `go run ./demo/chat fit` | the build fails (that is a red run, not a void) | runs to completion | about a minute (the build) |
| G-MM6-4 the census | `TestDecisionCensus` passes without `-census.update` (`FitQuery` is refactored, not changed) | as G-MM1-S1 | runs to completion | 8 s |
| G-MM6-5 the replay (the Mac) | every night job on the Mac refused by the fit or swap guard from 2026-10-06 to 10-26, re-checked with `fit -profile` on the night profile for its model and options as queued, is refused (exit 4). A job whose refusal was the swap guard's (not priced by `FitQuery`) is listed apart, not counted against the rule. The commands are in the PR; the result follows the PR. | the job's model is no longer on the Mac's local disk (listed, not counted) | every refused job | the Mac session's estimate, minutes |
| G-MM6-6 builds | as G-MM1-S6, and `python3 scripts/test_night.py` | as G-MM1-S6 | runs to completion | about two minutes |

**MM6 read, 2026-10-26 (nobara): G-MM6-1, 2, 3, 4 and 6 pass; G-MM6-5, the Mac's replay, follows the PR.**
- G-MM6-1: `TestProfile_answersAsFitQueryAvailable` reaches all four verdicts, each with its exit code and with
  `FitQueryAvailable`'s answer. `TestFitQueryAvailable_isFitQueryAtThatReading` holds it equal to `FitQuery` at the probe's
  reading, for a directory and a `.gguf`.
- G-MM6-2: eight `FitCheckAtQueueTime` tests pass, and the whole `test_night.py` (62).
- G-MM6-3: through the real `go run ./demo/chat fit`, llama-tiny is refused on a 64 KiB profile and fits on
  `nobara-no-desktop`. The first try found that `go run` exits 1 for the program's exit 4, so `night.py` reads the JSON
  verdict, not the exit code. The scripted checker in G-MM6-2 now exits 1 as `go run` does.
- G-MM6-4: `TestDecisionCensus` passes without `-census.update`.
- G-MM6-6: vet on three systems, the chat builds' vet with their tags, the pinned staticcheck (untagged and `realckpt`),
  gofmt, the parity manifest fresh.
- The replay's commands and its pass rule are in the PR. Because the night profile is the idle Mac, the replay also
  measures whether the Mac's nights are idle: a refusal that printed about 7 GB available would not be refused on a 12.2 GB
  profile. That would be a finding about which profile prices night jobs (Francis's 1b), and it is reported as one.

**G-MM6-5 read, 2026-10-27 (mac-vk01, the Mac, `origin/main` `014cbd54`): FAIL by its rule. The finding is about the
night profile, not the code** (`docs/measurements/mm6-replay-2026-10-27.md`).
- Nine fit-guard refusals in the window. One model is no longer on local disk, so 8 are counted. No `.giw` or swap-guard
  refusals.
- `m1pro-near-idle` refuses 1 of the 8 (the 7B at 12.6 GB). It fits the other 7, which need 2.7–6.7 GB against its 8.5 GB
  budget.
- `fit -profile` prices each load within 0.1 GB of the need the real refusal logged (rows 2–8). The real refusals printed
  2.9–7.7 GB available, median 6.6, against the profile's 12.2: the Mac's nights were not idle.
- `m1pro-apps-open` refuses 7 of the 8. The 1.5B fits there; its night had 2.9 GB.
- **DECISION for Francis (1b):** which profile prices the Mac's night jobs, given these readings.

#### MM2a's gates (backends report their build demand), registered 2026-10-26 before the code (nobara-mm01; Francis's answers of 22:40)

**Francis's decision (22:40):**
- 2a: decoder keeps a default copy of each backend's demand report, and each backend registers its own at init, replacing
  it (the `ResolveKV` pattern), with a tier-0 test in each backend module that its report equals decoder's default.
- 2b: CUDA's 2 MiB allocation rounding stays out of MM2. MM2 is a pure move held to "census unchanged"; the rounding is
  proposed afterward as its own census-changing step, with a DECISION NEEDED and the census diff.

**What moves.** `cudaWillBeResident` and `metalWillBeResident` go. In their place is a registered report:
`RegisterBuildDemand(backend, func(BuildQuery) BuildDemand)`, with `DefaultBuildDemand(backend, q)` as decoder's copy. A
`BuildDemand` says:
- whether the build goes resident;
- whether its KV is device memory (the host fit then prices none);
- the host the build adds while it packs (CUDA: one packed layer and the staging blocks);
- the host peak of building an expert cache from a `.gguf` (CUDA C': weights, a packed copy, a pinned expert copy);
- the pinned expert copy a `.giw` load holds.

The fit guard (`kvPricingFor`, `fitCheckFor`, `priceCtxAndKV`, `giwPinnedExpertBytes`) and `FitQuery`'s decode path read the
report for the load's backend. The `cuda` module registers its report with its own staging constant and expert packing; the
`metal` module registers Metal's. Which architectures a backend runs resident stays decoder's feature table (SB1's to move),
read through the `FeaturesCovered` field that `BuildQueryFor` fills.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM2a-1 the census | `TestDecisionCensus` passes without `-census.update` | as G-MM1-S1 | runs to completion | 8 s |
| G-MM2a-2 the reports agree | in `cuda` (tag `cuda`) and in `metal` (darwin), a tier-0 test loads tracked tiny fixtures on the CPU, and for each config × the option cells (backend, expert cache, stream-weights) × a weight/expert ladder, the registered report equals `DefaultBuildDemand`. It goes red when a planted change moves one term. Metal's test runs in CI's darwin job (no Mac here); nobara vets it for darwin | the test compares fewer than 10 configs | runs to completion | seconds |
| G-MM2a-3 the existing tests | the decoder's fit tests pass, with the two predicate calls in `fitguard_cuda_test.go` and `fitguard_metal_test.go` read through the report (mechanical); `cuda`, `metal` and `gpu` vet with their tags | — | runs to completion | about a minute |
| G-MM2a-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM2a read, 2026-10-26 (nobara): G-MM2a-1, 3 and 4 pass; G-MM2a-2 passes for CUDA and is checked for Metal in CI's darwin job.**
- G-MM2a-1: `TestDecisionCensus` passes without `-census.update`.
- G-MM2a-2: CUDA's registered report equals `DefaultBuildDemand` on 36 tracked configs and 6,912 queries, and a planted
  change is seen. Metal's test is the same; nobara vets it for darwin, and CI's darwin job now runs it, with its own zero-match
  guard. Both tests live in existing test files that import decoder (`cuda/resident_cap_test.go`,
  `metal/resident_memguard_test.go`), because SB13's ratchet on backend test files importing decoder only falls.
- G-MM2a-3: the decoder's 60 fit tests pass, with the two predicate calls read through the report. `cuda`, `metal` and `gpu`
  vet with their tags.
- G-MM2a-4: vet on three systems and with `realckpt`, the pinned staticcheck (untagged, `realckpt`, and `cuda` with
  testhooks), gofmt, the layout tests, the citation lint.
- The PR also regenerates `docs/ARCHITECTURE.md`'s import appendix. `TestArchitecture_importAppendix` is red on main since
  #301 added `cuda/internal/launchgeom`.

#### MM6's night profile, registered 2026-10-27 before the code (nobara-mm01; Francis's option B of 05:56)

**Francis's decision (05:56), replacing 1b after the replay:** option B. Each box gets a recorded "at night" profile, priced at
the 25th percentile of real nights. On the Mac the memory in use at night is the setup itself: the Claude app, background
sessions, their builds and tests. It persists. The replay's real refusals saw 2.9–7.7 GB available (median 6.6) against the
idle profile's 12.2.

**What is built.**
- **Every job start records the fit guard's available reading.** It goes in the `=== START` line and in the box's
  `memory.jsonl` under `~/goinfer-logs/night`, marked night or day by the job's `--not-before` (as registered; since the follow-up below, by the start's time alone). Today the runner logs only the
  load average at a start, so this record starts now.
- **The night profile is computed from those readings.** It is the 25th percentile (linear interpolation) of the night-job
  readings in the last 14 nights. A reading belongs to the night that began at or before it, so 02:00 counts with the evening
  before. The profile records its percentile, window, reading count, night count and dates. It is written to
  `~/goinfer-logs/night/profile-night.json`, a per-box file, not committed. A committed profile would change the census's
  machine zoo (`testdata/machines/*.json` is census input) and its golden every night, and each box's file is that box's own
  data.
- **Night jobs (`--not-before`) price against it; day jobs keep the apps-open profile.** With fewer than 5 readings or 3 nights
  in the window, a night job falls back to the apps-open profile with a warning.
- **`night.py profile`** prints the profile. `--backfill` seeds the history once from existing night logs: the fit guard's own
  "currently has X GB of memory available" lines, in logs whose job started in a night window. That sample is biased low,
  because those lines come from refusals and warnings, and it is marked `backfill` in the record.
- **nobara is the same,** from its own readings.

**The expected replay count, stated from the data before running.** The Mac's recorded night readings are the refusals'
printed figures: 2.9, 3.9, 4.6, 6.4, 6.4, 6.6, 6.9, 7.0, 7.4, 7.5 and 7.7 GB. Their 25th percentile is about 5.5 GB, a budget
of about 3.85 GB. The 8 counted jobs need 5.1, 12.6, 6.7, 5.5, 5.6, 5.1, 5.0 and 2.7 GB. So the night profile refuses rows
1–7 and fits row 8: **expected 7 of 8**. Row 7 (5.0 GB) fits only if the 25th percentile reaches 7.2 GB. Refined before any run, from
the window rule: the 14 nights ending 2026-10-27 start on 10-14, so the 10-08 and 10-12 readings fall outside it. The
in-window readings (2.9, 3.9, 4.6, 6.4, 6.6, 6.9, 7.0, 7.7) give about 4.4 GB, a 3.1 GB budget: still 7 of 8.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-NP-1 the percentile and window | a fake-history test: the profile's figure is the 25th percentile of exactly the night readings in the last 14 nights (older readings and day readings excluded), with linear interpolation checked against a hand computation; the profile records percentile, window, count, nights and dates | — | runs to completion | seconds |
| G-NP-2 the fallback | with fewer than 5 readings or 3 nights, a night job is priced on the apps-open profile and `add` prints a warning naming the count | — | runs to completion | seconds |
| G-NP-3 the record | a job start through the real runner writes the available reading to the START line and to `memory.jsonl`, with the night flag | — | runs to completion | seconds |
| G-NP-4 the replay (the Mac) | with the profile computed from the Mac's backfilled readings, the 8 counted jobs re-priced on it: **at least 7 refused** (the expectation above); the profile's figure and count are reported beside it | the profile falls back (too few readings), which is reported as such | every counted job | minutes |
| G-NP-5 builds | `python3 scripts/test_night.py`; `touched.py`; the citation lint | — | runs to completion | about a minute |

**Night-profile read, 2026-10-27 (nobara): G-NP-1, 2, 3 and 5 pass; G-NP-4, the Mac's replay, follows the PR**
(the commands are in the PR).
- G-NP-1: the profile equals a hand computation of the 25th percentile over exactly the in-window night readings, and records
  its percentile, window, readings and nights.
- G-NP-2: 6 readings over 2 nights fall back to `m1pro-apps-open` with a warning naming the counts.
- G-NP-3: a job through the real runner writes `available 4321 MB` to its START line and the reading to `memory.jsonl`.
- G-NP-5: `test_night.py` (74 tests at this read; 78 after the fixes before merge), `touched.py`, the citation lint.
- nobara: a dry run of `--backfill` on its night logs finds no fit-guard availability lines (this was the `START_RE` bug, fixed below, not an empty history). Its night
  profile starts from the START readings, and its night jobs are priced on `nobara-desktop`, with the warning, until it has 5
  readings over 3 nights.

**Fixed before merge, 2026-10-27 (nobara-mm01), from Gemini's review of #312 and the Mac's G-NP-4 (VOID: no profile).**
- `--backfill` found nothing: `START_RE` anchored `^` without `re.M`, and every real log has header lines before `=== START`.
  nobara's "no fit-guard lines" above was this bug, not an empty history. A test now backfills a log in the real shape.
- A box is priced only on its own readings: `night_profile` keeps readings whose `host` is the box priced. A night job
  queued for another box falls back to that box's day profile with the warning, and this box's `profile-night.json` is not
  overwritten.
- A 0 MB reading counts (it was dropped as false).
- One definition of night: a window from 22:00 to the deadline. A job is a night job when its `--not-before` start falls in
  it (a 14:00 start is a day job), and a start reading is a night reading only inside it, as backfill already required.
- The window is the 14 nights that have begun. Before, between noon and 22:00 it counted tonight, which had not begun, and
  so held 13. So the replay's "14 nights ending 2026-10-27, from 10-14" reads, before 22:00 on the 27th, as 10-13 to 10-26.
- `avail_mb` lived on `Server` while `run_job` is the base `Runner`'s, so `night.py start` would have failed at every job
  start; it is now on `Runner`, and it returns None when `hostmem` fails. A test runs the real one.
- Not changed: Gemini's 1000-against-1024 point (the fit guard prints GiB, so backfill's `* 1024` is right; the test's
  values are MB labels), and its import-time binding of `MEMORY_LOG` (the tests set `NIGHT_DIR` before importing `night`,
  so they never touch the live log).
- `test_night.py`: 78 tests pass. G-NP-4 is to be re-run on the Mac after the merge.

**G-NP-4 re-run on the Mac, 2026-10-27 19:00 (mac-vk01), at `origin/main` `4c7cd4a0`, with the backfill fixed: VOID again, for too
few readings, not for the bug** (`docs/measurements/mm6-night-profile-replay-2026-10-27.md`, "Re-run").
- `night.py profile --backfill` now reads the logs: "backfilled 7 night reading(s)". The 14-night window holds only 10-23's
  three, over one night, against the 5 over 3 the profile needs. So there is no profile, and the 8 jobs were not re-priced.
- `~/goinfer-logs/night/memory.jsonl` now holds the 7, and the runner adds a reading at each job start. The replay can
  answer once the Mac has run night jobs on 3 nights with 5 readings between them.

#### MM2b's gates (CUDA's 2 MiB allocation rounding in the plan), registered 2026-10-27 before the code (nobara-mm01; Francis's option A of 05:50)

**Francis's decision (05:50):** option A, "one set of numbers that is accurate". `Plan("cuda")` prices the driver's 2 MiB
rounding through CUDA's report. The CUDA build reads the slack from the plan instead of adding its own copy. `goinfer fit`,
MM6's queue-time check and the build all use that one figure.

**The design: one copy.**
- **The computation.** It is the CUDA build's own planner. `BuildResident` becomes `buildResident(m, planOnly)`. Plan-only
  stops right after `packedAllocSlack` over the planned buffers, before any device allocation, and validates experts instead
  of packing them, so it holds no host copy.
- **The report.** `RegisterPlanSlack` is a per-model report, memoized on the model. Through it, CUDA hands that figure to
  decoder's `Plan`, which prices it as a fixed term beside the dense weights: `Plan.AllocSlackBytes`, in `NeedBytes()` and in
  every fit check.
- **The build.** `resolveCtxCapFitSlack` stops adding the slack to the extra bytes it hands `Plan`; `Plan` already holds it.
- **No mirror.** Decoder keeps no mirror of CUDA's buffer layout, which would be a second copy. Its default slack is 0, which
  is the true figure on every tracked tiny fixture (measured 2026-10-26: 60 resident builds, all 0).
- **Out of scope.** The KV pools' rounding is priced by the build already (`kvRoundingBound`); it is not part of this step.

**Amended from the proposal, before the code.** Gate 1's "shape-only configs" assumed a path into `Plan` that decoder does not
have: `Plan` reads a loaded model's matrices, and decoder has no shape-only model. Building one would re-implement `Plan`'s
inputs, which is not today's code path. So the real-size cells use the real checkpoints on nobara's local disk, planned through
CUDA's report. The test that does it loads each model on the CUDA device, so it needs one. That is a tier-1 test in the `cuda` module, which skips as a capture run where
the models are absent.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM2b-1 the real-size diff | for Qwen2.5-Coder 0.5B and 1.5B and Qwen2.5-7B (int4, from `~/models`), D3's `Plan("cuda")` cells (the request grid and a free ladder with each cell's change points) are computed with the slack and with it forced to 0. The diff is reviewed group by group and listed in the PR body (groups changed, placements moved, every cell that moves to a decline). Then their golden is written. The decoder census passes without `-census.update`, all of its cells' slack being 0 | a model path under `/srv/models` or `/Volumes/` (CLAUDE.md), or a model absent | runs to completion | about a minute on nobara |
| G-MM2b-2 the device check | on nobara's RTX 2070 SUPER, for 0.5B and 1.5B: `Plan("cuda").AllocSlackBytes` equals the real build's `packedAllocSlack` to the byte, and plan dense plus slack equals the build's dense plus slack. `TestResidentDenseBytes_matchesCUDADevice` still holds. The CUDA KV-slot and margin tests (`TestCUDAKVSlots_*`, `TestKSH6C_marginAtTheReturnedContext`, `TestKSB8_widthChooser`) still pass | a timed run holds the card (the run takes the timing lock) | runs to completion | about two minutes |
| G-MM2b-3 one copy | the build no longer adds the slack beside the plan: a source test finds `packedAllocSlack` called only inside `buildResident`, and `resolveCtxCapFitSlack` takes no slack argument. The MM2a agreement test (`TestBuildDemand_matchesDecoderDefault`) gains the slack report: CUDA's equals decoder's default (0) on every tracked tiny fixture | — | runs to completion | seconds |
| G-MM2b-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**MM2b read, 2026-10-27 (nobara): all four pass.**
- G-MM2b-1, the real-size diff (`docs/measurements/mm2b-rounding-2026-10-27/README.md`): rounding 138.5 MB (0.5B), 122.2 MB
  (1.5B) and 405.8 MB (7B). All 63 groups change. At nobara's two real readings, no cell moves to a decline, and only the 7B's
  unpinned 32,768 cells shrink (29,696 to 25,984 at 7.63 GB). 462 cells move to a decline, all inside the band one rounding
  wide below the old fit edge; the README lists them by group. The real-size golden (`cuda/testdata/mm2b_realsize.json.gz`)
  was written after that review. The decoder census passes without `-census.update`.
- G-MM2b-2: the plan's slack equals the real build's `packedAllocSlack` to the byte (145,276,928 and 128,122,880). (The
  read first said "as does dense plus slack": that check compared the plan's figures with themselves and was removed in the
  MM1 follow-up; the dense figure against the device is `TestResidentDenseBytes_matchesCUDADevice`'s.) `TestResidentDenseBytes_matchesCUDADevice`, the `TestCUDAKVSlots_` tests,
  `TestKSH6C_marginAtTheReturnedContext` and `TestKSB8_widthChooser` pass, and so do `TestBuildScratchAccounting` (six models)
  and its planted defect with `GOINFER_HEAVY_TESTS=1`.
- G-MM2b-3: `TestPlanSlack_oneCopy`; the agreement test's slack report is 0 against decoder's 0 on every tracked family. The
  new tests live in existing test files, because SB13's ratchet counts every test file of a package that reaches decoder.
- G-MM2b-4: vet on three systems and with the module tags, the pinned staticcheck (untagged, `realckpt`, cuda), gofmt,
  `touched.py`, the citation lint.
#### MM1 follow-up: the Gemini review's findings, registered 2026-10-27 (nobara-mm01)

Gemini reviewed #296, #300, #302, #307 and #311 (relayed by the coordinator, 2026-10-27 07:24). Each finding was checked
against origin/main (a22bd928d). The guard fixes below (the divide, the nil policy, the enum sync, the scan, the lookup order)
were written before this section; the gates were registered before the rest (the generators, the agreement tests, #311's
items, the census build) and before any of it ran as a gate.

**What changes, none of it a placement or a default.**
- `SolvePlacement` no longer divides by an unpriced expert size (`NExperts` with `ExpertBytesFull` under a byte each): such
  experts cost nothing, so every slot fits. Before, that load panicked; no caller hands it in today (the census holds).
- `ResolveKV`'s CUDA branch treats a nil `Decline` as no decline and a nil `Auto` as false (before: a panic).
- `decoder.Placement` and `mem.Placement` are held equal at compile time, all five values both ways.
- I6's scan: `RequestFraction` joins the thresholds, and a constant that aliases a different threshold counts as a use.
- `buildDemandFor` looks up the backend's report before building its query.
- `doc.go`'s contract says what is true: the guards proceed on unknowns; the planner declines on a 0 reading. I5 is stated
  as far as it holds (below).
- Tests only: the generators, the monotone test, `ResolveKV`'s branch and purity tests, the agreement tests, #311's tests.

**I5 does not hold for an MoE plan's placement, measured before any change.** `chooseCtx` grows the context before it prices
the experts, so one more KV step can leave less than top-k experts' room: on a generated CUDA load (8 experts, top-2), the
plan was expert-cached at context 4,480 and declined at 4,608, at 16,160 MiB. Changing that (reserving top-k experts in
`chooseCtx`) moves placements, so it is Francis's call and is not in this PR. The task-doc example in G-MM1b-2's amendment
(top-8, two slots) was misreported: the load was top-2.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-GF-1 the census | `TestDecisionCensus` passes without `-census.update`, every group byte for byte | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-GF-2 the planner tests | `TestSolvePlacement_pure`, `TestResolveKV_eachBranch`, `TestResolveKV_pure` and `TestSolvePlacement_monotone` pass. The monotone test walks free memory in 16 MiB steps to 64 GiB on generators that keep `DeviceLoad`'s invariants (tier per backend, a host copy only on Metal, Metal's linear KV), and asserts: a dense placement never worsens; a placed context never shrinks; slots never fall at the same context; the Metal context never shrinks (MoE included, as registered; withdrawn in the second follow-up below: an MoE Metal context does fall). At least a third of Metal draws reach a context past the floor. A planted defect (a refusal above 32 GiB free) turns it red | — | runs to completion | under a minute |
| G-GF-3 the guard fixes | an unpriced-expert load and a zero `CUDAPolicy` return without a panic; the scan reads every non-test file and counts a planted `.RequestFraction` read and a planted cross-alias | the scan reads fewer files than the tree holds | runs to completion | seconds |
| G-GF-4 the agreement tests | `TestBuildDemand_matchesDecoderDefault` (cuda, metal) has no duplicate backend; it asserts that a nonzero `HostPackBytes`, `HostBuildPeak` and (cuda) `PinnedExpertBytes` each occur in the corpus, or names the one that does not; a planted change per field turns it red; and decoder's registry returns the backend's own function | — | runs to completion | seconds |
| G-GF-5 #311's items | the tautological dense-plus-slack check is removed with its claim; the archive guard resolves symlinks; `TestPlanSlack_oneCopy` checks the call site's argument; `planSlackOffForTest` is atomic; a plan-only build that stops early is not memoized | — | runs to completion | seconds; the device tests run once on nobara under the timing lock |
| G-GF-6 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all six pass.**
- G-GF-1: `TestDecisionCensus` and its second-process twin pass without `-census.update`. The serial probe check now covers
  every config, not the first eight, and holds.
- G-GF-2: the planner tests pass. 919 of the 1,500 Metal draws reach a context past the floor, and the planted refusal is
  found.
- G-GF-3: the zero-policy rows of `TestResolveKV_eachBranch` return; the scan reads 734 files and counts the planted
  cross-alias and `.RequestFraction` read (3). The unpriced-expert half was not tested: no test drew such a load (the
  generators price every expert at a byte or more), so this line first claimed a check that did not exist. The second
  follow-up adds it.
- G-GF-4: 36 configs and 6,912 queries, now with no duplicate backend. Every field's planted change is seen, so a nonzero
  `HostPackBytes`, `HostBuildPeak` and `PinnedExpertBytes` each occur. `TestBuildDemand_registered` passes on cuda. Its
  metal twin is vetted for darwin but runs only on a Mac with `goinfer_testhooks`, because CI's metal job runs untagged.
- G-GF-5: `TestPlanSlack_oneCopy` goes red on a planted `+allocSlack` at the call site. On the RTX 2070 SUPER (timing lock
  held) these pass: `TestPlanSlack_matchesTheBuild` (145,276,928 and 128,122,880 again), `TestPlanSlack_realSizeDiff`,
  `TestResidentDenseBytes_matchesCUDADevice` and the `TestCUDAKVSlots_` tests (log: `~/goinfer-logs/mm1/` on nobara,
  gf-cuda-device).
- G-GF-6: vet untagged and tagged (testhooks; cuda with testhooks; darwin metal both ways); the pinned staticcheck with its
  canary red (U1000) on the same variants, clean; gofmt.

#### MM1 follow-up 2: Gemini's review of #316, registered 2026-10-27 (nobara-mm01)

Relayed by the coordinator at 08:47. The code below was written before this section; the gates were registered before they
ran.

**Withdrawn from #316: "an MoE Metal context never falls".** #316 listed Gemini's Metal case as not holding, on a probe that
stepped 16 MiB. The window it lives in is under one 8-position KV step (a few MiB), so the probe stepped over it. At a 64 KiB
step a constructed Metal MoE load (8 experts, top-2, 512 KiB a KV step) falls from 4,104 to the 4,096 floor 1,535 times between
5.6 and 9 GiB. The cause is the MoE planner's (context before experts), so it joins that decision
(the decision note for Francis); the monotone test now checks the Metal context for dense loads only.

**What changes.**
- An expert cache whose experts could not be priced (`ExpertBytesFull` under a byte each) declines with a reason. #316
  sized every slot on 0 bytes there; before #316 that line panicked, so no caller depended on either. Not changed: an unpriced
  MoE load whose dense weights fit goes resident with 0 expert bytes, as `Model.Plan` did before MM1. Changing that moves a
  placement, so it is in the decision note.
- `PlanWithoutSlackForTest` passes "no slack" to one call (`Model.plan`'s argument) instead of setting a global, so a
  concurrent `Plan` never sees it and two overlapping calls cannot restore it early.
- The monotone test also holds an MoE plan's placement at an unchanged context; Metal's agreement comment says what it plants.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-GF2-1 the census | `TestDecisionCensus` passes without `-census.update` | the golden's class not found on an amd64 host | runs to completion | 16 s |
| G-GF2-2 the planner | `go test ./internal/mem/` passes, with `TestSolvePlacement_unpricedExperts` (0 and 7 bytes over 8 experts decline naming the unpriced experts); reverting the guard turns it red | — | runs to completion | under a minute |
| G-GF2-3 the slack hook | the rounding diff (`TestPlanSlack_realSizeDiff`) holds its golden on nobara's card; vet with the testhooks and cuda tags | a timed run holds the card | runs to completion | about a minute |
| G-GF2-4 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all four pass.** G-GF2-1: the census and its second-process twin pass. G-GF2-2: `internal/mem`
passes; the unpriced-expert test goes red with the guard reverted. G-GF2-3: `TestPlanSlack_realSizeDiff` holds its golden and
`TestPlanSlack_matchesTheBuild` passes on the RTX 2070 SUPER (exclusive timing lock). G-GF2-4: vet (testhooks, cuda, darwin
metal), the pinned staticcheck with its canary red, gofmt, `touched.py`, the parity refresh, the citation lint.

#### MM1's MoE reserve (option A), registered 2026-10-27 before the code (nobara-mm01; Francis's option A of 09:09)

**Francis's decision (09:09):** "option A for the MoE planner. Reserve the top-k experts' bytes before growing the context in
chooseCtx, so more memory never gives a worse MoE placement. Gate it as MM2b was." The question and the measured census effect
are in the decision note (`MM1 follow-up 2` above names the cause).

**The change.** `SolvePlacement`'s `chooseCtx` prices an MoE's top-k experts (top-k × the per-expert bytes) beside the dense
weights, the slack and the host copy before it grows the context, on both its paths (the per-position rate and CUDA's block
search). The expert-cache and resident fits after it are unchanged. A load whose experts could not be priced reserves nothing
(it declines at the expert cache, as #320 made it).

**Expected before the run.** The census moves in D3 only, on MoE fixtures: 984 of 29,637 groups, measured on a scratch copy at
e934b8ce plus #320 (10,336 readings declined → expert-cached, 5,028 weight-paged → expert-cached, 80 declined → resident, 24
resident → expert-cached). D1, D2 and D4–D7 do not move.

**Pre-registered contingency.** Reserving top-k alone may leave a fall between resident and expert-cached, where every
expert's extra bytes are under one KV step, and Metal's context fit needs a resident answer. If the monotone test finds either,
the planner tries the resident placement first with every expert (and Metal's host copy of them) reserved, and the expert cache
second with top-k reserved, each choosing its own context. That is the same rule (reserve what the placement needs before
the context grows) one placement at a time. The census diff is then re-measured and listed the same way; the context invariant
holds within a placement (G-MM1b-2 as amended), not across an upgrade from expert-cached to resident.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MA-1 the census diff | the census is computed with and without the change; the diff is listed in the PR body group by group (each changed group, its backend and family, and every reading whose placement moves, by direction), before `-census.update`. Any group outside D3, or any non-MoE group, that moves is a failure of the change, not a diff to bless | the golden's class not found on an amd64 host | runs to completion | about a minute |
| G-MA-2 monotone, MoE included | `TestSolvePlacement_monotone` asserts, for every load, MoE included: the placement never moves down the order; within a placement the context never shrinks, nor the slots at the same context; the Metal context never shrinks. The walk adds a fine ladder (64 KiB steps across each draw's sawtooth window) to the 16 MiB one. A planted defect (the reserve removed) turns it red, and so does the planted refusal | — | runs to completion | under a minute |
| G-MA-3 the probe | the constructed Metal MoE load of the follow-up (1,535 falls at 64 KiB steps) has none | — | runs to completion | seconds |
| G-MA-4 real sizes and the device | `TestPlanSlack_realSizeDiff` holds its golden (dense models: no move expected), and the CUDA KV-slot tests pass on nobara's card | a timed run holds the card | runs to completion | about a minute |
| G-MA-5 builds | as G-MM1-S6 | as G-MM1-S6 | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): G-MA-2 to G-MA-5 pass. G-MA-1 fails its registered rule's letter, and the golden is updated with
that stated; Francis decides whether it stands.**
- **The contingency was taken.** The top-k reserve alone passed the generated walk, but on a constructed CUDA load whose experts
  are under one KV block it still fell from resident to expert-cached 449 times at 64 KiB steps. So the planner tries resident
  first (every expert reserved), then the expert cache (top-k reserved), as registered.
- **G-MA-1, the census diff** (`docs/measurements/mm1-moe-reserve-2026-10-27/`, groups listed one by one in `groups.tsv.gz`):
  - 3,534 of 29,637 groups change; 1,068 of them move a placement or class (D3 984, D4 84).
  - By reading: 9,676 declined → expert-cached, 5,028 weight-paged → expert-cached, 732 declined → resident, 92 expert-cached →
    resident, and 1,232 Metal fits leave the floor. **None moves down.** Every moved group is an MoE fixture.
  - **The registered rule called any move outside D3 a failure, and D4 moves.** Those are Metal's context fit, which the
    contingency exists to fix (it needs a resident answer). The expectation and the rule were written for the top-k reserve
    alone and were not amended when the contingency was registered. The golden is updated because the diff was listed and every
    move is upward on an MoE fixture, but the PR is not set to merge by itself: that is Francis's call.
  - The registered expectation (984 groups, 24 resident → expert-cached) was the top-k-only measurement; the contingency removed
    the 24 downward readings.
- G-MA-2: `TestSolvePlacement_monotone` asserts I5 for every load, MoE included, with 64 KiB windows at each draw's first change
  points. It finds the planner without the reserve (input 364: resident, then declined) and the planted refusal.
- G-MA-3: `TestSolvePlacement_moeSawtooth`. The CUDA load has no fall (the planner without the reserve falls), and the Metal
  load climbs from the floor without a fall.
- G-MA-4: `TestPlanSlack_realSizeDiff` holds its golden (dense models). The CUDA KV-slot, margin, dense-bytes and rounding
  tests pass on the RTX 2070 SUPER (exclusive timing lock; log `~/goinfer-logs/mm1/` on nobara, ma-cuda-device). Real MoE
  checkpoints, which the census does not hold: `goinfer fit` on CUDA said the 26B (gguf and .giw) and Qwen1.5-MoE declined;
  it now says expert-cached (8/128 at context 11,264; 4/60 at 15,104), as the CUDA build runs them.
- G-MA-5: listed in the PR.

**G-MA-1 re-registered, 2026-10-27 (Francis's option C of 10:36), before the re-run.** The rule above is replaced by this
one. The census is computed with and without the change: "without" is this branch with `internal/mem/placement.go` taken from
main. The diff is graded as before: placements and classes at every reading either run records, contexts only at readings
both runs record.
- **Pass:**
  - D1, D2, D5, D6 and D7 are byte-identical;
  - every changed group is in D3 or D4 and on an MoE fixture (one whose D3 groups price experts above 0 in the base census);
  - **no move is downward anywhere**: no placement moves down the order (decline < weight-paged < expert-cached < resident),
    no Metal fit falls to the floor from above it, and no context is smaller at a reading both runs record within the same
    placement or class. A decline's context is excluded, because it only echoes the attempt.
- **A D4 move is allowed only when it is upward on an MoE fixture.** Any downward move, or any move on a non-MoE fixture, is
  still a failure.
- **Expected, from the mechanism (not from the earlier run):**
  - D1, D2, D5–D7: 0 groups change. They do not call `SolvePlacement`'s MoE path.
  - Non-MoE groups: 0. The reserve is 0 without experts.
  - Downward: 0. The reserve only moves the context choice toward where the placement's experts fit. Where the old planner
    placed an MoE at context c, the experts already fit at c, so the new context at the same placement is at least c, and a
    placement it reached it still reaches.
  - D3: at most the 4,592 MoE groups, all upward.
  - D4: at most the 168 Metal-fit groups that can leave the floor in the base census (an MoE fixture, unpaged, known memory,
    a context above the floor at some reading; 14 fixtures, `lower` 0 and 8192, 84 each). A group with `lower=2048`, paged
    experts or unknown memory never leaves the floor, so it cannot move.
- **Then:** the device tests (G-MA-4) re-run on nobara's card. On a pass, auto-merge is set on #321. On a failure of this rule:
  stop, DECISION NEEDED.

**Re-run read, 2026-10-27 (nobara), against the re-registered rule: PASS.**
- The census was dumped without the change (this branch, `placement.go` from main at efa85df15's second parent) and with it,
  and graded by `docs/measurements/mm1-moe-reserve-2026-10-27/grade.py`: **0 failures.**
- D1, D2 and D5–D7 are byte-identical. Every changed group is on an MoE fixture: 3,366 D3 and 168 D4.
- The D4 changes are exactly the 168 groups the mechanism allowed: 14 fixtures, `lower` 0 and 8192, 84 groups each.
- 960 D3 groups and 84 D4 groups move a placement or class. By reading: 9,676 declined → expert-cached, 5,028 weight-paged →
  expert-cached, 732 declined → resident, 92 expert-cached → resident, and 1,232 Metal fits leave the floor.
- No placement moves down, no Metal fit falls to the floor, and no context is smaller within a placement at a reading both runs
  record.
- The committed golden holds (`TestDecisionCensus` passes) and `go test ./internal/mem/` passes.
- G-MA-4 re-run on the RTX 2070 SUPER under the exclusive timing lock: 15 of 15 pass (the rounding diff's golden, the
  KV-slot, margin, dense-bytes and agreement tests; log `~/goinfer-logs/mm1/` on nobara, ma-cuda-device-rerun).

#### MM6 night profile and the unpriced MoE: Gemini on #312 and #320, registered 2026-10-27 before the code (nobara-mm01)

Relayed by the coordinator at 10:02. #320's two test findings (an MoE decline unchecked at the same context; MoE Metal draws
counted as searched) are on the monotone test that #321 rewrites: it asserts every load, MoE included, so they are fixed there.

**What changes.**
- **The night profile is weighted by night, not by reading:** each night's lowest reading, then the 25th percentile over the
  nights in the window (still 14 nights, at least 3 of them, and at least 5 readings). One night with many short jobs no longer
  outweighs the others. **Amended before merge (coordinator, 10:27, from Gemini on #322):** per night alone can raise the
  figure (a night with many low readings becomes one), so the figure is the lower of the per-night and the per-reading 25th
  percentile. The change can then only lower the figure or keep it, so the queue-time check can refuse more night jobs, never
  fewer; that is listed for Francis. The registration first claimed that of per night alone, which was wrong.
- **A reading is a night reading by its time alone** (inside 22:00 to the deadline), in the runner as in backfill, because a
  backfilled log does not record its job's `--not-before`. The profile describes the box at night, whichever job read it.
  Pricing still goes by `--not-before`.
- **Backfill skips a job that already has a reading** of any source at the same start, so a refused night job is not counted
  twice.
- **A failed reading is logged** in the runner's log, instead of only `available ?` in the job's.
- **An MoE whose experts could not be priced is declined on a device** (paged on the CPU), resident included, before any
  context is chosen. Before, with no slot count asked, it went resident with 0 expert bytes. That changes what the planner
  admits; the census diff is listed in the PR.
- Tests: a job run through the real runner at 23:00 records a night reading; the queue-time test seeds past nights only;
  the hostmem test asserts the reading is a number where hostmem works.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-NP2-1 per night | a fake history with one night of ten readings at 1 GB and four nights of one reading at 6 GB: the profile is the 25th percentile of the five nights' minimums, checked against a hand computation | — | runs to completion | seconds |
| G-NP2-2 the record | through the real runner at 23:00 a job records `night: true`; at 20:00 `night: false`; a failed reading logs a line naming it; backfill adds nothing for a job that has a start reading | — | runs to completion | seconds |
| G-NP2-3 the unpriced MoE | on loads that keep `DeviceLoad`'s invariants (CUDA with no host copy; Metal whose host copy is its experts'), an unpriced MoE declines on CUDA and Metal and pages on the CPU, at slot counts 0 and 2; `go test ./internal/mem/` passes | — | runs to completion | seconds |
| G-NP2-4 the census | `TestDecisionCensus`: the diff with and without the change is listed in the PR body group by group before `-census.update`. Expected: no group moves, since no fixture's experts are unpriced. Any move is listed for Francis | the golden's class not found on an amd64 host | runs to completion | about a minute |
| G-NP2-5 builds | `test_night.py`, `touched.py`, vet, the pinned staticcheck, the citation lint | — | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara): all five pass.**
- G-NP2-1, as amended: `test_a_busy_night_counts_once` (per reading is the lower figure there, and is taken) and
  `test_per_night_weighting_never_raises_the_figure` (per night is the lower figure, and is taken).
- G-NP2-2: the real runner records `night: false` at 20:00 and `night: true` at 23:00. A failed reading logs
  "the available reading failed" and records nothing. Backfill adds nothing for a job with a start reading. The hostmem test
  now requires a number: it failed in a shell without `go` on its PATH. nobara's `goinfer-night.service` has
  `/usr/local/go/bin` on its PATH, so its runner reads.
- G-NP2-3: `TestSolvePlacement_unpricedExperts` passes on CUDA, Metal and the CPU at 0 and 2 slots. With the guard
  disabled, it goes red (resident with 0 expert bytes).
- G-NP2-4: the census dump is byte-identical with and without the change. No fixture's experts are unpriced, so nothing moves.
- G-NP2-5: `test_night.py` (81 tests), `go test ./internal/mem/`; the rest is in the PR.

#### MM3a's gates (the ledger, and leases in CUDA's allocator), registered 2026-10-27 before the code (nobara-mm01; Francis's go of 11:08)

**Francis's go (11:08):** "a ledger and leases in each backend's allocator, so every allocation is a lease naming its plan
line. CUDA first (cuMemAlloc's 2 MiB rounding priced, now one figure after MM2b), then WebGPU, then Metal. Gates before code:
I1 … and I2 … per backend at tier 0; I3 … with one tiny model per backend, seconds. No change to what any guard admits."

**The design.**
- **`internal/mem` gets the ledger**, its one stateful type. It records reservations; it reads nothing and decides nothing.
  - `Ledger.Reserve(tier, line, requested, charged)` returns a `*Lease`, and `Lease.Release` is idempotent.
  - `Usage()` reports, per tier and per line, the open leases and their requested and charged bytes, plus the peak charged.
    `Unplanned` is the charged bytes of open leases whose line is empty or not a known plan line.
  - The plan lines name the `DevicePlan` terms: `dense`, `experts`, `kv`, `state` (recurrent and conv state), `slack`
    (rounding the plan prices), `companion` (a drafter's or tower's bytes, `ExtraBytes`), `scratch` (per-pass activations, the
    margin's share), `staging` (the build's host staging), `pinned-experts`, and `test` (a test's own buffers).
- **CUDA's `gpudev.Device` keeps a `Ledger`.**
  - Every allocation leases its bytes: requested, and charged by the driver's rule (a buffer of 2 MiB or more rounds up to
    2 MiB). That rule is one function, which `allocRoundSlack` (the plan's slack) also calls.
  - `ReleaseBuf`, `ReleaseAll` and `ReleaseObjects` release the leases.
  - Pinned host buffers lease on `HostAnon` (the open question in §6.7, "CUDA pinned host memory as its own tier", stays open;
    the line names them).
- **Every production allocation names its line.**
  - `gpudev` gains lined constructors. The unlined ones remain for tests and lease on the line `gpudev.UnlinedLine`: "" (so
    unplanned) unless a test package sets it to `test`.
  - The resident's helpers (`af`, `akv`, `ai`, `au32`) take the line from the build phase in progress (`r.onLine`). The build
    and request paths run on the resident's executor goroutine, so the phase needs no lock.
- **Nothing reads the ledger to decide.** No guard, plan or default changes.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3a-1 I1, static | a source scan over every non-test file of the `cuda` module (`internal/qwencuda` and `internal/visioncuda` included) finds no call to an unlined `gpudev` constructor (`NewBufferLenOf`, `NewBufferOf`, `NewBufferLen`, `NewBufferBytes`, `MustBuf`, `NewHostBuffer`, `NewMappedHostBuffer`, `RegisterMappedHostBuffer`); it reads every file and goes red on a planted call | it reads fewer files than the module holds | runs to completion | seconds |
| G-MM3a-2 I1, dynamic | the `cuda` package's `TestMain` installs a hook on `ReleaseObjects` and fails the run if any released device's `Usage().Unplanned` was above 0; run with the device suite once on nobara (tags `cuda goinfer_testhooks`, the tests MM2b and MM1 ran: KV slots, dense bytes, plan slack, build scratch, KS-B8, KS-H6C) and `-v`. A planted allocation on the empty line is reported | a timed run holds the card | one run | about two minutes on the card |
| G-MM3a-3 I2 | tier 0: `internal/mem`'s ledger tests (reserve and release in any order, idempotent release, peak, by line and tier, concurrent use under `-race`); on the device: every released device's ledger has 0 open leases and 0 charged bytes after `ReleaseObjects`, and `ReleaseBuf` closes its one lease (the same hook as G-MM3a-2) | as G-MM3a-2 | as G-MM3a-2 | seconds, and the same run |
| G-MM3a-4 I3 | on the RTX 2070 SUPER under the exclusive timing lock, for llama-tiny and Qwen2.5-Coder-0.5B q4_k_m (`~/models`): the device's free VRAM just after the device is created, against free VRAM after the build and one 8-token generate, gives the measured bytes. **Pass: charged ≤ measured ≤ charged + 64 MiB.** Small buffers are charged at their size, while the driver hands them out of shared 2 MiB pages; the 64 MiB also covers the modules the build loads and its streams. The record reports both numbers and the gap | another process's allocation on the card changes during the cell (the desktop's context is constant and noted) | one run per model | under a minute |
| G-MM3a-5 one rounding rule | `TestPlanSlack_oneCopy` still passes; `allocRoundSlack` calls `gpudev`'s rule (a scan finds the 2 MiB constant once); `TestPlanSlack_matchesTheBuild` still matches to the byte | as G-MM3a-2 | runs to completion | seconds |
| G-MM3a-6 no guard change | `TestDecisionCensus` passes unchanged; the CUDA fit, KV-slot and margin tests pass unchanged; a scan finds no non-test read of `Usage()` outside tests and the ledger's own package | — | runs to completion | about a minute |
| G-MM3a-7 builds | vet with the module tags, the pinned staticcheck with its canary, gofmt, `touched.py`, the citation lint | — | runs to completion | about two minutes |

If I3 misses its band, the finding is reported first and the band is not widened.

**Two changes to the design, made before any gate ran.**
- `gpudev` may not import goinfer (SB13's rule "gpudev-standalone"). So `gpudev` defines a `Leaser` interface and takes plan
  lines as strings, and the cuda package attaches a `mem.Ledger` adapter when it creates a device (`createDevice`). The plan
  lines are untyped constants, so they pass as strings unconverted.
- The I1 hook reads the ledger through that adapter.

**MM3a read, 2026-10-27 (nobara, RTX 2070 SUPER, driver 595.91.07, kwin_wayland's 16 MiB context present): all seven pass.**
- **G-MM3a-1:** `TestLedger_everyAllocationNamesALine` reads all 58 non-test files outside `gpudev` and finds no unlined call. It
  counts the planted file's 2.
- **G-MM3a-2 and G-MM3a-3:** in the device run (tags `cuda goinfer_testhooks`, `GOINFER_HEAVY_TESTS=1`, exclusive timing lock,
  `-v`), 96 devices are released, none with an unplanned byte and none with an open lease. 27 tests pass; 7 skip on their own
  gates (KS-B8's timing and KS-H6C's breakdown and margin, which need their variables). `TestLedger_unplannedIsReported` sees
  the planted 3 MiB, charged as 4, and the planted open lease. `internal/mem`'s ledger tests pass under `-race`.
  - `touched.py`'s wider selection first found 4 devices with unplanned bytes: three pack-stream tests and
    `TestWindowedKV_view`, which drive the build's helpers outside a build. A stack per unplanned lease showed every one came
    from those tests and none from production. They now name the test line, and the selection passes.
- **G-MM3a-4, I3 (inside the band):**

  | model | measured | charged | gap |
  |---|---:|---:|---:|
  | llama-tiny | 8,388,608 | 4,479,240 | 3.9 MB |
  | Qwen2.5-Coder-0.5B | 991,952,896 | 988,562,064 | 3.4 MB |

  By line, the 0.5B is dense 483.6 MB, kv 503.3 MB and scratch 2.3 MB. Both ledgers hold no lease after `Close`.
  - **Instrument amendment, 2026-10-27 (nobara-mm01), after the CUDA device suite read the test red that evening.** The
    band and the pass rule are unchanged. The measured bytes are now this process's own device memory (the driver's
    per-process figure, read through nvidia-smi) instead of the device's free VRAM. Free VRAM is still logged beside it.
    - The suite runs `./...` in `cuda/`, so the packages run side by side. Free VRAM is device-wide, so this registration's
      void condition (another process's allocation during the cell) held, but the suite cannot void a cell. It read
      llama-tiny at 4,194,304 bytes against the ledger's 4,479,240, and so a FAIL.
    - Alone, the two instruments agree to the byte: 8,388,608 and 991,952,896, the figures above.
    - With three copies of the test running at once (18 readings), free VRAM moved by anything from −1.51 GB to +1.99 GB.
      The per-process reading was 8,388,608 and 991,952,896 every time, and all 18 pass.
    - A planted band of 0 turns both models red.
    - Where nvidia-smi cannot name the process, at the device's creation or after the requests, the test skips with the
      reason rather than falling back to free VRAM.
    - The bound is the registered one, `[charged, charged + 64 MiB]`. A first push of this amendment let the measured
      bytes fall 1 MiB short of the charge, for nvidia-smi's whole-MiB reads; that widened the band without Francis's
      word and was reverted before it merged. No reading needed it: every one sat at least 3.39 MB above the charge,
      including all 9 with three copies running at once under the exact bound.
- **G-MM3a-5:** `allocRoundSlack` is `gpu.ChargedBytes(n) - n`, and the 2 MiB constant is defined once
  (`gpu.AllocQuantumBytes`). `TestPlanSlack_oneCopy` and `TestPlanSlack_matchesTheBuild` pass (145,276,928 and 128,122,880
  again).
- **G-MM3a-6:** `TestDecisionCensus` passes unchanged. The only non-test reader of the ledger is the test seam's call site.
- **G-MM3a-7:** vet (cuda with and without testhooks, gpu, darwin metal); the pinned staticcheck with its canary red, clean;
  gofmt; the layout tests, with `imports.json` and the ARCHITECTURE appendix regenerated for the new `→ internal/mem`
  arrows; `touched.py`; the citation lint.
- Logs: `~/goinfer-logs/mm1/` on nobara (mm3a-device2).

**Fixed before merge, 2026-10-27 (coordinator, 12:01; CI's lint and Gemini's review of #328).**
- **CI lint:** the new source-scan test was not in `scripts/touched_scans.py`. It is listed now.
- **Leases where none were made (Gemini 1, real).** A device created other than through the cuda package's wrapper (the
  `qwencuda` and `visioncuda` towers call `gpu.CreateSystemDefaultDevice` directly) had no leaser, and the release check skipped
  it. `gpudev.NewLeaser` is now set by the cuda package at init, so every device a process with cuda linked creates is leased.
  A released device with no ledger is now a break. Only a test (`cuda/s4siglip`) imports those two towers today.
- **I2 after release was nearly tautological (Gemini 2, real).** `ReleaseObjects` frees and releases everything it tracked
  before the check, so only an untracked pin could show there. I2 is now also held within a model's life: the charged bytes
  after a second and a third identical request equal those after the first, which a buffer leaked per request would grow.
  Passes on llama-tiny and the 0.5B.
- **The plants did not exercise the suite's check (Gemini 3, real).** `TestLedger_unplannedIsReported` now runs the suite's
  own `ledgerBreak` on an unplanned allocation, an open lease, a clean device and a device with no ledger. The static scan
  now reads `gpudev` too (68 files), counting any unlined call outside the unlined constructors' own chain. `UnlinedLine =
  test` in the tests stays: production's unlined calls are excluded statically, which is what the static gate is for.
- **`onLine` off the executor (Gemini 4, real).** `prefillCore`, `SetAdapter`, `AttachDrafter` and the drafter's four
  methods set the line on the caller's goroutine before their `r.do`. They now pass it to `r.doOn`, which sets it inside the
  job. Every remaining `onLine` is in a function that allocates directly, so it already runs on the executor (allocation needs
  the context current there).
- **llama-tiny was skipped without the 0.5B (Gemini 5, real).** It runs alone now.
- **Gemini 6:**
  - A `NewBufferLenOn` twin is added.
  - `slack` and `staging` stay as lines: they name plan terms, though CUDA leases neither. Its rounding is in each buffer's
    charged bytes, and its pack staging is Go memory the allocator never sees.
  - `ByLine` summing tiers is by design; `ByTier` splits them.
- **Re-run:** the device suite under the exclusive lock (34 pass, 102 devices released, no break; I3 unchanged at 3.9 MB and
  3.4 MB), `touched.py`, the pinned staticcheck, `test_touched.py` and the citation lint. Log: mm3a-device3.

#### MM3b's gates (leases in WebGPU's allocator), registered 2026-10-27 before the code (nobara-mm01; Francis's go of 11:08)

**What is different from CUDA.**
- WebGPU has no allocation chokepoint. The kernels package calls `wgpu.Device.(Try)CreateBuffer(Init)` directly at 133 sites,
  and the gpu package at 3 more. Buffers are freed by `Release` or `Destroy` at 168 sites, mixed in with releases of
  pipelines, bind groups and encoders.
- The existing `liveBufferBytes` counts only the caller-owned wrapper types, process-wide.

**The design.**
- **Every creation goes through a lined Context wrapper** (`c.createBuffer(line, …)` and its siblings, exported as
  `kernels.CreateBufferOn` for the gpu package). It leases the buffer's `GetSize()` bytes on the Context's `mem.Ledger`
  (tier VRAM) and records the lease against the buffer.
- **Every buffer release goes through `releaseBuf` (or `destroyBuf`)**, which closes the lease.
- **The rewrite is mechanical.** A typed scan (go/types) found the release sites, so only `*wgpu.Buffer` releases are
  rewritten. Pipelines and bind groups are untouched.
- **Lines by enclosing function:**
  - `kv`: the KV caches and MLA's latent store;
  - `dense`: the weight uploads;
  - `experts`: the stacked experts;
  - `companion`: vision, towers and LoRA;
  - `state`: Mamba's state;
  - `test`: the test hooks;
  - `scratch`: the per-call ops and the runners' buffers.
- **`liveBufferBytes` stays as it is**, a separate gauge.
- **Nothing reads the ledger to decide.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3b-1 I1, static | a typed scan over the gpu module's non-test code (`gpu` and `gpu/internal/kernels`, tags `gpu goinfer_testhooks`) finds no `CreateBuffer`, `CreateBufferInit`, `TryCreateBuffer` or `TryCreateBufferInit` call outside the lined wrappers, and no `Release` or `Destroy` of a `*wgpu.Buffer` outside `releaseBuf` and `destroyBuf`. It goes red on a planted call of each kind | it reads fewer files than the packages hold | runs to completion | seconds |
| G-MM3b-2 I1 and I2, dynamic | a release check on every Context the gpu module's tests close: no unplanned byte ever, and no open lease after `Close`. Run once on nobara's card (Vulkan; tags `gpu goinfer_testhooks`; the KV-slot, residency, leak and parity tests), with `-v` | a timed run holds the card | one run | minutes |
| G-MM3b-3 I2 per request | on llama-tiny, the charged bytes after a second and a third identical request equal those after the first | as G-MM3b-2 | one run | seconds |
| G-MM3b-4 I3 | after the build and one request on llama-tiny (resident on WebGPU), the ledger's open VRAM leases equal wgpu-native's registry count of buffers the user holds (`GenerateReport().Hub.Buffers.NumKeptFromUser`), exactly. The bytes are reported beside the process's VRAM from `nvidia-smi`; that comparison is context only, not graded, because the Vulkan allocator sub-allocates buffers from large blocks | the registry's count is shown to include buffers the runtime holds for itself: reported, not re-banded | one run | seconds |
| G-MM3b-5 no guard change | `TestDecisionCensus` passes unchanged; the WebGPU KV-slot and fit tests pass unchanged; nothing outside tests reads the ledger | — | runs to completion | about a minute |
| G-MM3b-6 builds | vet with the gpu tags, the pinned staticcheck with its canary, gofmt, `touched.py`, the layout tests (`imports.json` updated for any new arrow), the citation lint | — | runs to completion | about two minutes |

**Found while building, before the gates' read.**
- **Freeing by method value.** The runners keep buffer releases as method values (`r.keep = append(r.keep, b.Release)`), and
  so do 16 other sites, which free a buffer later without a call at the site. The typed scan counts a method value as a free,
  and those sites take `releaser(b)`, which binds `b` when it is evaluated, as the method value did.
- **The ledger must not keep a buffer alive.** Some buffers are never released explicitly; the binding's finalizer frees them
  when the GC collects them. A first version keyed leases by the buffer itself. That kept those buffers, and the Vulkan devices
  they pin, alive, and the kernels suite ran out of devices ("Not enough memory left"; main passes all 110). Leases are now
  keyed by a weak pointer, and a `runtime.AddCleanup` closes a collected buffer's lease.
- **The check must not force collection inside `Close`.** A first version of the check ran the GC inside `Close` and so started
  finalizer-driven device releases beside the caller's. `TestContextClose_lateWrapperClose` deadlocked. The hooks now keep each
  closed Context's ledger, and `LedgerBreaks` judges them once after the run, letting the GC collect first.

**MM3b read, 2026-10-27 (nobara, RTX 2070 SUPER over Vulkan; exclusive timing lock): all six pass.**
- **G-MM3b-1:** `TestLedger_everyBufferIsLeased` type-checks the 61 non-test files of `gpu` and `gpu/internal/kernels` and
  finds no create, release or destroy outside the wrappers. The plant gives 4: a create, a release, a destroy and a method
  value; a pipeline's `Release` is not counted.
- **G-MM3b-2:**
  - The kernels suite passes all 111 tests (main's 110 and the scan) in 227 s, against main's 216 s. 703 Contexts closed, and
    none broke the ledger.
  - The gpu package's selection (KV slots, residency, parity, ledger) passes: 24 pass, and 30 skip on their own gates (heavy
    checkpoints, opt-in variables, missing fixtures, darwin-only). 81 Contexts closed with no break.
- **G-MM3b-3:** the charged bytes after a second and a third identical request equal the first.
- **G-MM3b-4, I3 (exact):** 196 open leases against wgpu-native's 196 held buffers. The ledger charges 246,568 bytes (dense
  102,704, kv 131,072, scratch 12,792). The process's VRAM from `nvidia-smi` is 202–212 MiB, for context only.
- **G-MM3b-5:** `TestDecisionCensus` passes unchanged. Nothing outside tests reads the ledger, except `BuffersHeld`, a
  diagnostic.
- **G-MM3b-6:**
  - vet passes with the gpu and testhooks tags.
  - The pinned staticcheck is clean with those tags (CI's variant), and its canary goes red. It flagged two wrappers nothing
    called, which are removed.
  - gofmt, `touched.py`, `test_touched.py` (the scan listed in `touched_scans.py`) and the citation lint pass.
  - The layout tests pass, with `imports.json` and the ARCHITECTURE appendix regenerated for the `→ internal/mem` arrows.
- Logs: `~/goinfer-logs/mm1/` on nobara (mm3b-kernels3, mm3b-gpu2, and main's baseline mm3b-baseline-main).

#### MM3b follow-up: Gemini's review of #334, registered 2026-10-27 before the code (nobara-mm01; coordinator, 14:18)

**Findings checked against main.**
- **Lines (real).**
  - `UploadF32` charged `dense`, but its callers upload per-call activations (scratch) and vision inputs (companion); only the
    resident's `up32` uploads weights.
  - The `kvslot-headroom` probe is created and freed at once, but leased on `kv`.
  - `MLALatentStore` is a synchronous op that frees its buffers before returning, but leased them on `kv`.
- **The dynamic I2 check can pass vacuously (real).** A collection cleanup closes any lease whose buffer the GC frees, and the
  check runs after the tests, when every test's locals are unreachable. A buffer that production code forgets to release is
  then counted as closed.
- **The typed scan ignores `CgoFiles` (real in principle).** No kernels file has `import "C"` today.
- **`destroyBuf` and `ReleaseDestroyBuffer` are dead (real).** Nothing destroys a buffer, and a destroy before a release would
  break I3's count. The line lists in the PR and the doc omit `state` and `test`.

**What changes.**
- `UploadF32` leases `scratch`. A lined `UploadF32On` is used by the vision paths (`companion`) and the resident's weights
  (`dense`). The probe and `MLALatentStore` lease `scratch`.
- The collection cleanup counts what it closes. A lease closed by the GC, rather than by `releaseBuf`, is a buffer production
  code never released, and the I2 check reports it with its count and bytes. Before any change, the cleanup's count on the kernels
  suite and the gpu selection is measured, with the creation sites of what it closed. That measurement decides the rule:
  - if every one comes from test code, the check fails on a count above 0 from production code;
  - otherwise, the production sites are listed, and each is fixed (an explicit release) or, if it is by design, named in an
    allow-list with why. No such entry is added without saying so in the PR.
- The scan reads `CgoFiles` too. `destroyBuf` and `ReleaseDestroyBuffer` go. The doc's line list gets `state` and `test`.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3bf-1 lines | the three sites lease as above; the census unchanged | — | runs to completion | seconds |
| G-MM3bf-2 I2 not vacuous | the kernels suite and the gpu selection pass with the collection count checked by the rule above. A planted buffer that production code leaves unreleased is reported (a unit test creates one, drops the reference, and sees the count) | a timed run holds the card | one run | about four minutes |
| G-MM3bf-3 the scan | `TestLedger_everyBufferIsLeased` reads `GoFiles` and `CgoFiles`; its plant still gives 4 | — | runs to completion | seconds |
| G-MM3bf-4 builds | as G-MM3b-6 | — | runs to completion | about two minutes |

**Read, 2026-10-27 (nobara, RTX 2070 SUPER over Vulkan; exclusive timing lock): all four pass, after the measurement found three
real faults.**
- **The measurement.** My first count of GC-closed leases ran without `-v`. `go test` discards a passing package's output, so
  it reported none, and that reading was wrong. Run with `-v`, it found 625 in the kernels suite and 448 in the gpu package:
  - **Production, `FusedMLP`.** It kept three raw buffers in a `[]interface{ Release() }`. Their `Release` reached wgpu by
    interface dispatch, bypassing `releaseBuf`, which the typed scan did not see. They are now `leasedBuf`, whose `Release`
    goes through `releaseBuf`. The scan also flags any `*wgpu.Buffer` converted to an interface with a `Release` method (an
    append, a slice literal or an argument); on the old `FusedMLP` it reports all three.
  - **Production, `DecodeRunner.release`.** It freed the runner's buffers but not a LoRA adapter still bound (four buffers and
    two bind groups per projection). They were left to the GC. The runner now frees them with itself.
  - **Tests.** The two Context-close reproducers leak a matrix per round on purpose, and they already set `leakExempt`. A
    Context closed while it is set is not judged; my plant sets it too.
- **G-MM3bf-1:** `UploadF32` leases `scratch`. `UploadF32On` takes `companion` in the vision paths and `dense` for the
  resident's weights. The headroom probe and `MLALatentStore` lease `scratch`. The census is unchanged.
- **G-MM3bf-2:** `LedgerBreaks` counts a buffer the GC freed with its lease open as a break. Both suites pass with none:
  - the kernels suite passes all 112 tests, judging 103 Contexts (the reproducers' 600 and the plant are exempt);
  - the whole gpu package passes, with 251 Contexts.
  - `TestLedger_anUnreleasedBufferIsCounted` sees its one dropped buffer, 512 bytes.
  - I3 stays exact: 196 against 196.
- **G-MM3bf-3:** the scan reads `GoFiles` and `CgoFiles`, and its plant now gives 5.
- **G-MM3bf-4:**
  - The pinned staticcheck passes with the gpu and testhooks tags, as do `touched.py` (the whole gpu package) and the
    citation lint.
  - The layout tests pass; the arrow data drops a test import the tree no longer makes.
  - `destroyBuf` and `ReleaseDestroyBuffer` are removed.
  - The doc's line list is now `kv`, `dense`, `experts`, `companion`, `state`, `test` and `scratch`.

#### MM3c's gates (leases in Metal's allocator), registered 2026-10-27 (nobara-mm01; Francis's go of 11:08)

The lease code in `metal/internal/gpudev` (the `Leaser`, `MustBufOn`, the lined constructors) was written before this
section, while surveying, and is not yet committed. Everything else is written after it, and every gate is graded after it.
The device gates run on a Mac: their commands are in `/tmp/mm3-metal.md`, handed to a Mac session by the coordinator.

**The design (CUDA's, with Metal's differences).**
- **gpudev leases every MTLBuffer** through a `Leaser` that the metal package installs at init (`gpudev.NewLeaser`). gpudev
  imports nothing of goinfer's ("gpudev-standalone").
  - A buffer's requested bytes are its `length`, its charged bytes its `allocatedSize`.
  - It leases on `Unified`; a `NewBufferNoCopy` alias leases on `FileMapped` (Francis's go).
- **The metal package has no allocation chokepoint.** Model code calls gpudev's constructors and the package's own helpers
  (`NewBufferU32`, `NewBufferFloats`, `byteBuf`, the int4, int8 and q4k packers, the alias helpers) at 546 sites.
  - Each helper gets a lined twin (`…On(line, …)`); the unlined name stays as a wrapper on `gpudev.UnlinedLine` for the 826
    test call sites.
  - Every production call site names its line, chosen by its enclosing function (with per-line overrides where a function
    mixes lines), by a typed rewrite (go/types, darwin).
  - Lines: `dense` (weights, norms, RoPE tables), `experts` (expert stacks and their biases, the expert pool), `kv`, `state`
    (recurrent and conv state, checkpoints), `companion` (towers, encoders, LoRA), `test` (test hooks), `scratch` (the
    rest: activations and the per-op uniforms).
- **Nothing reads the ledger to decide.**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-MM3c-1 I1, static | a scan over every non-test file of the metal module finds no call to an unlined gpudev constructor (`NewBufferLen`, `NewBufferBytes`, `NewBufferNoCopy`, `MustBuf`, `NewBufferOf`, `NewBufferLenOf`) or an unlined metal helper outside its own wrapper; it reads every file and goes red on a planted call. It runs on darwin; the metal-darwin CI job runs it, and it is also run from Linux with a darwin type-check before the push | it reads fewer files than the module holds | runs to completion | seconds |
| G-MM3c-2 I1 and I2, dynamic (Mac) | a release hook in the metal package's `TestMain` fails the run if any released device had an unplanned byte, or a lease open after release; run with the Mac's device suite (the commands name the tests) | the Mac's apps hold the GPU (the run notes `CurrentAllocatedSize` at its start) | one run | the Mac session's estimate |
| G-MM3c-3 I2 per request (Mac) | on llama-tiny resident on Metal, the charged bytes after a second and a third identical request equal those after the first | as G-MM3c-2 | one run | seconds |
| G-MM3c-4 I3 (Mac) | `CurrentAllocatedSize` just after the resident's device is created, against after the build and one request on llama-tiny: **charged (Unified) ≤ measured ≤ charged + 64 MiB** (pipelines, libraries, the command queue). The FileMapped leases are reported apart, with whether the measured delta includes them | another Metal allocation in the process during the cell | one run | seconds |
| G-MM3c-5 no guard change | `TestDecisionCensus` passes unchanged; the Mac's Metal fit tests (`MetalFitCtx`, the memory guard) pass unchanged; nothing outside tests reads the ledger | — | runs to completion | about a minute |
| G-MM3c-6 builds | vet for darwin (untagged and `goinfer_testhooks`), the pinned staticcheck for darwin with its canary, gofmt, `touched.py`, the layout tests, the citation lint | — | runs to completion | about two minutes |

If I3 misses its band on the Mac, the finding is reported first and the band is not widened.

**MM3c read on nobara, 2026-10-27 (a darwin type-check; the device gates follow from the Mac, `/tmp/mm3-metal.md`).**
- **G-MM3c-1:**
  - The typed rewrite found 546 calls: 147 to gpudev's methods, 36 to its generic constructors and 363 to the package's
    helpers. 38 of them are inside the helpers themselves.
  - Each call now names its line. Each helper has a lined twin, and an unlined wrapper where tests call it.
  - staticcheck flagged the ten wrappers no test calls, and they were removed.
  - A replay of the scan's logic on Linux reads all 71 non-test files and finds no unlined call.
  - The scan joins CI's device-free metal-darwin step, behind its zero-match guard.
- **G-MM3c-5 (the part that runs here):** `TestDecisionCensus` passes unchanged. Nothing outside tests reads the ledger.
- **G-MM3c-6:** `go vet` for darwin passes, untagged and with `goinfer_testhooks`. The pinned staticcheck for darwin with the
  test hooks is clean. Also pass: gofmt, `touched.py`, `test_touched.py` (the scan listed), the layout tests (`imports.json` and
  the ARCHITECTURE appendix updated) and the citation lint.
- **G-MM3c-2, 3 and 4:** to be run on the Mac.

**Amended before merge and before the Mac run (coordinator, 14:28, Gemini's review of #335).**
- **G-MM3c-1's scan is tightened.** It now also flags a lined call handed `UnlinedLine` or `""` as its line, and it skips only
  true wrappers: a function whose body hands `UnlinedLine` to its own `…On` twin. Before, it skipped any function with a listed
  name. `newBufferLenUntracked`, a probe allocator with no production caller, takes its line from its caller.
- **G-MM3c-4 gets an aliasing cell, because llama-tiny never aliases.** `gemma4-dense-scaled` is transcoded to a Metal `.giw`
  and loaded with `GOINFER_METAL_ALIAS=1`, about 53 MB of no-copy windows. Pass:
  - its FileMapped leases are above 0 (aliasing was exercised); and
  - measured is within [U, U+64 MiB], or within [U+F, U+F+64 MiB], where U is charged Unified and F is charged FileMapped.
  - The record says which of the two held, which answers whether `CurrentAllocatedSize` counts no-copy buffers.
  - A FileMapped charge counts each window's pages, so a page shared by two adjacent windows is counted twice (at most a page
    per tensor). That is stated, not changed.
  - Each I3 cell runs in its own process (`-run` per cell), because the system device is shared and earlier tests' frees can
    land during a cell.
- **#2 holds as designed.** `ReleaseAll` at `Close` is how a Metal resident frees its buffers, so a lease closed there is not a
  leak. The leak check within a model's life is G-MM3c-3: charged bytes do not grow across identical requests.
- **#1 holds.** `UnlinedLine = test` makes a test's own unlined calls planned. Production's unlined calls are excluded by the
  static scan, which is what I1's static half is for.
- **Lines (#5):** MLA's RoPE table leases `dense`; the q4k segment counts and `uHasSink` (uniforms) lease `scratch`. The counts
  in this section read 546 calls throughout.

**Mac read, 2026-10-27 (mac-vk01), at `origin/main` `4c7cd4a0`, after #335 merged: G-MM3c-2 and the llama-tiny cell of G-MM3c-4
FAIL by their rules; the aliasing cell PASSES; DECISION NEEDED (nobara-mm01 / Francis).**
- **How it ran.** The full `./metal/` package with `-tags goinfer_testhooks -v`, under the timing lock shared (18:55 PDT,
  load 2.62), then each I3 cell in its own process. Tiny fixtures were cloned into the worktree. Logs:
  `~/goinfer-logs/mm3c-metal-device-main.log` and `mm3c-i3-cells-main.log` on the Mac. The figures equal the run at #335's
  head `2bbd0d33` exactly; four non-test decoder files had changed since, so it was re-run.
- **G-MM3c-1: PASS.** The scan reads 71 non-test files.
- **G-MM3c-2: FAIL.**
  - 760 devices released. 4 have leases open after release, all on the `test` line: 87 leases (9,669,704 B), 288
    (100,448,288 B), 412 (596,200 B) and 2 (144 B). I1 holds: every unplanned peak is 0.
  - Neither I3 cell trips it in its own process ("5 devices released"). So the leases are other tests' own buffers, open
    when their device is released.
  - `TestLedger_metalUnplannedIsReported` passes.
- **G-MM3c-3: PASS.** No "charged bytes grow".
- **G-MM3c-4, llama-tiny: FAIL.**
  - Measured 2,834,432; U 2,838,124; F 0; measured − U = −3,692.
  - By line: dense 122,368, kv 2,097,152, scratch 618,604.
  - The ledger charges 3.6 KB more than the device reports, in its own process too.
- **G-MM3c-4, aliasing: PASS.**
  - F = 64,307,200 > 0. Measured 187,006,976; U 122,665,312; measured − (U + F) = +34,464.
  - Measured − U = 64,341,664 is also within [U, U + 64 MiB], only because F is just under 64 MiB. The 34 KB fit to U + F
    shows that `CurrentAllocatedSize` counts no-copy buffers.
- **G-MM3c-5:** 366 pass, 213 skip, 3 fail. Besides the llama-tiny cell, `TestMEM6_prefillHostBytes` and
  `TestI8_exactKernelModelIsSplit` fail, as they do on `origin/main` before #335 (checked at `92a64e92`).
- **The consequence on main.** `go test ./metal/` on a Mac with a Metal device is red: the cell fails, and `TestMain`'s
  release hook fails the package. CI's `metal-darwin` job does not catch it.
- **The decision:**
  - G-MM3c-2: are the `test`-line leases test cleanup to fix, or should the hook exclude the `test` line?
  - G-MM3c-4: is the 3.6 KB over-charge a ledger over-count to fix? The band is not widened without Francis.

**Francis (20:13, relayed):** (a) 1 and (b) 1. The gates stay strict and the band is not widened. (a), the leases tests leave
open after release, goes to mac-vk01. (b), the 3,692 B over-count, goes to nobara-mm01.

**(b) found, 2026-10-27 (nobara-mm01, with mac-vk01's runs at `mm3c-ledger-overcount` `43c9dd94`, M1 Pro): the ledger does not
over-count; the cell's baseline missed small buffers.**
- **No buffer is charged above what Metal allocated.**
  - Every Unified buffer comes from `newBufferWithLength` or `newBufferWithBytes`, and its charge is Metal's own
    `allocatedSize`.
  - On llama-tiny all 311 buffers have `allocatedSize` equal to their length, and requested equals U (2,838,124).
- **Metal counts small buffers by chunk** (`TestMetal_smallBufferSuballocation`).
  - The first buffer of 256 B moved `CurrentAllocatedSize` by 131,072. The next six, from 256 B to 4,096 B, moved it by 0.
  - Buffers of 16,384 B and 65,536 B moved it by exactly their size.
  - So buffers of 4 KB and under are packed into shared 128 KB chunks, and only a chunk moves the count.
- **The cause.** The baseline was read when the resident's device was created, after earlier allocations in the process
  had opened a chunk. The resident's small buffers that fit in that chunk's free space (3,692 B of them) were charged by the
  ledger but never moved `CurrentAllocatedSize`.
- **The fix is the instrument's, and the band is unchanged.** The cell reads its baseline before the load, ahead of any
  goinfer allocation in the process, so every chunk the build opens is counted. The rule is still
  `[U, U + 64 MiB]`, or that band above U + F. The cell logs both baselines and, per buffer, how far `allocatedSize` runs
  past the length.
- **Confirming run, mac-vk01 at `4174445c`, M1 Pro, each cell in its own process: both PASS.**
  - `CurrentAllocatedSize` read 65,536 before the load and 1,703,936 when the resident's device was created. The new
    baseline therefore also counts the device's own creation (+1,638,400 B), which the 64 MiB band covers.
  - **llama-tiny:** measured 4,472,832; U 2,838,124; F 0; measured − U = +1,634,708. Inside `[U, U + 64 MiB]`. All 311
    buffers have `allocatedSize` equal to their length.
  - **aliasing:** measured 188,645,376; U 122,665,312; F 64,307,200; measured − (U + F) = +1,672,864, and measured − U =
    65,980,064. Both bands hold, the second about 1.1 MB inside 64 MiB.
- **What the aliasing cell no longer shows.** With the device's creation inside the delta, the cell cannot tell whether
  `CurrentAllocatedSize` counts no-copy buffers; it logs "both bands hold". The evidence that it does stays the earlier read
  above: measured = U + F + 34,464 with the baseline at the device's creation. The pass rule is unaffected, because either
  band passes. A larger aliasing fixture would fail the U band and pass U + F.

**Francis (20:39, relayed), on #383: make the slack exact (option 1). This replaces #383's baseline move.** Gemini's point:
reading the baseline before the load folded Metal's own device creation (+1,638,400 B on llama-tiny) into the measured
bytes. The lower bound `measured ≥ U` then had about 1.64 MB of slack, so the ledger could over-charge by that much and
still pass.
- **The baseline is back at the resident's device creation.**
- **The lower bound is `measured ≥ U − S`.** S is the requested bytes of the buffers of 4 KB or less, summed from the
  ledger's own buffers (`Device.BufferSizes`). The reason is the probe above: Metal packs buffers of 4 KB and under into
  shared 128 KB chunks and counts the chunk, so a small buffer placed in a chunk counted before the baseline moves
  nothing. Buffers of 16 KB and over are counted exactly. With F, the bound is `measured ≥ U + F − S`.
- **The upper bound is unchanged:** U + 64 MiB, or U + F + 64 MiB.
- **The plant** (`TestLedger_metalAgreesWithTheDevice_overcharge`) charges one Unified buffer of 16 KB or more 4 KB above
  Metal's allocation (`gpudev.OverchargeForTest`). The cell must fall outside the bounds.
- **What it rests on.** The plant is red only if the slack `measured − (U − S)` on llama-tiny is under 4 KB. #383's read
  implies the slack is S − 3,692. If S is over 7,788 B, small buffers also opened chunks after the baseline. The plant would
  then pass, and that is a finding for Francis, not a reason to move the bound.
- **Confirming run, mac-vk01 at `f4bf9499`, M1 Pro, each cell in its own process.**
  - **llama-tiny: PASS.** Measured 2,834,432; U 2,838,124; S 85,612 over 292 buffers; measured − (U − S) = 81,920.
  - **aliasing: PASS.** Measured 187,006,976; U 122,665,312; F 64,307,200; S 268,640; measured − (U + F − S) = 303,104.
  - **The plant: FAIL**, meaning a 4 KB over-charge passes. There are two causes.
    - **The plant missed.** It over-charged the first Unified buffer of 16 KB or more, which was transient and released
      before the measurement: the planted run logged the unplanted U. It now targets a KV-line buffer, which lives until
      the cell measures (the branch, after `f4bf9499`).
    - **The bound has 81,920 B of slack on llama-tiny,** so even a plant that lands passes. S counts every small buffer.
      Only those placed in chunk space counted before the baseline (3,692 B of the 85,612) are unseen. The rest were
      counted by chunks the build opened, so `U − S` sits 81,920 B below what the device counted. The slack can be
      anywhere from 0 to S, and a plant smaller than it cannot go red.
- **The question put to Francis:** how exact G-MM3c-4's lower bound must be. The options were (a) a plant above S,
  (b) a per-buffer reconciliation, or (c) #383's baseline before the load.

**Francis (20:51, relayed): (b), reconcile per buffer, in this cell only. It replaces the `U − S` bound above.**
- **The seam.** `gpudev.AllocTraceForTest` sees every buffer the device's constructors allocate: its line, length and charge,
  and how far `CurrentAllocatedSize` moved across that allocation alone. The constructors wrap each Objective-C allocation in
  `allocID`. That reads the count before and after the call only while the trace is set; otherwise it adds a nil check and no
  behaviour change.
- **The rule.**
  - Every traced buffer of 16 KB or more must move the count by exactly its charge, to the byte, whether it is Unified or
    no-copy. A buffer allocated outside the constructors fails, and so does a cell that traced no buffer of that size.
  - The buffers under 16 KB are packed into shared chunks; their growth is measured and reported, not graded.
  - What the delta holds outside the buffer allocations (pipelines, the queue, net of frees) is reported, and the
    delta must stay at or under U + F + 64 MiB.
- **The plant** is unchanged in size: 4 KB on one KV-line buffer of 16 KB or more. It must now fail the cell.
- **mac-vk01's run at `7153c5bf`, M1 Pro, each cell in its own process.**
  - **llama-tiny: PASS.** 13 buffers of 16 KB or more, each exact to the byte. 298 smaller ones grew the count by
    131,072, one chunk. Nothing outside the buffer allocations: measured 2,834,432 = 2,703,360 + 131,072.
  - **aliasing: PASS.** 127 buffers of 16 KB or more, each exact to the byte, 93 of them no-copy. So Metal counts no-copy
    memory exactly, the question the earlier baseline left open. 263 smaller ones grew the count by 393,216 (three
    chunks). Nothing outside the buffer allocations.
  - **The plant missed again.** It was one-shot for the whole process, so a device that the load creates and drops before
    the resident's consumed it, on that device's first KV buffer. The measured device logged the unplanted U.
  - **Fix:** the plant now applies once per device, so the measured device's first KV buffer of 16 KB or more always
    carries it. The cell requires the reconciliation to name that KV buffer as charged more than it moved.
- **The confirming run, mac-vk01 at `7f398461`, M1 Pro, each cell in its own process: all three PASS.**
  - **llama-tiny and aliasing:** the same numbers as at `7153c5bf`. Every large buffer is exact to the byte (13, and 127
    with 93 no-copy). Small-buffer growth is 131,072 and 393,216, and nothing sits outside the buffer allocations.
  - **The plant:** U 2,842,220 (+4,096) and kv 2,101,248 (+4,096) on the measured device. The reconciliation names the
    buffer: "a 262144-byte buffer on line "kv" (no-copy false) is charged 266240 but moved CurrentAllocatedSize by
    262144".

**#390 corrected, per Francis's 20:51 decision (relayed after Gemini's review of #390).** #390 merged at 21:12, but
it did not reconcile the ledger. It compared the device's movement with the charge `mustBuf` computed, which is Metal's
allocatedSize, never with what the ledger recorded. It also dropped the lower bound, and on llama-tiny it could not fail,
even with an empty ledger. The correction:
- **Every allocation is traced, leased or not, and every `ReleaseBuf` too,** each with how far `CurrentAllocatedSize` moved
  across it. The trace carries Metal's allocatedSize and the leased charge separately.
- **Per buffer:**
  - each buffer was leased, through the constructors, at Metal's allocatedSize;
  - each buffer of 16 KB or more moved the count by exactly that size.
- **The ledger against the buffers:** the ledger's own record by line and by tier, and its lease count, must equal the charges
  of the traced buffers still alive. Allocations and frees are replayed in order, since Metal reuses ids. This catches a ledger
  that records more or less than it was asked, a lease no buffer made, and a freed buffer whose lease stayed open.
- **The device against the trace:** the measured delta must equal the traced allocations' and frees' movement exactly. The
  remainder was 0 in both cells on both of mac-vk01's runs, so a buffer allocated around the constructors fails. The large
  buffers' charges plus the small buffers' net growth must be at most the measured delta. The upper band is unchanged.
- **Plants, each its own cell, each required to be reported:**
  - the ledger records one KV lease 4 KB over (`ledgerOvercharge`) or 4 KB under (`ledgerUndercharge`) its charge, through a
    seam in the metal package's ledger, not in the trace;
  - a 64 KB buffer is allocated around the constructors (`untraced`, `gpudev.Device.AllocUntracedForTest`).
- **Limit:** a buffer under 16 KB allocated around the constructors into open chunk space moves nothing, so the device cannot
  show it. The ledger check still fails if anything leased it.
- **Confirming run, mac-vk01 at `525ca2d9`, M1 Pro, each cell in its own process: all five PASS** (posted on #391).
  - **llama-tiny and aliasing:** every buffer of 16 KB or more is exact, 93 of them no-copy. The ledger by line and tier
    matches the 311 and 390 live traced buffers. Nothing was freed, and 0 bytes sit outside the traced buffers.
  - **ledgerOvercharge:** `line "kv": the ledger records 2101248, the live buffers leased on it were charged 2097152`.
  - **ledgerUndercharge:** `line "kv": the ledger records 2093056, the live buffers leased on it were charged 2097152`.
  - **untraced:** `the device counted 65536 bytes outside the traced buffers' allocations and frees`.

**(a) resolved, 2026-10-27 (mac-vk01): G-MM3c-2 PASS after a test-only fix.**
- **The leaking tests.** Each of the package's 585 tests ran in its own process (317 s), and four trip the release hook,
  one per device in the full run:

  | test | leases open after release |
  |---|---|
  | `TestArgmaxRowsPart_matchesHost` | 87 (9.67 MB) |
  | `TestP3_attentionF32MatchesF16OnExactInputs` | 288 (100.4 MB) |
  | `TestP3b2_storeMatchesKvStoreI8` | 412 (0.6 MB) |
  | `TestInt4Pack_declinesNonMultipleOf32K` | 2 (144 B) |

- **The cause.** Three of them released their device with `defer d.ReleaseObjects()` alone. The fourth called
  `ReleaseObjects` before `ReleaseAll`. Either way, the hook ran while the test's buffers were still leased.
- **The fix, in test code only.** Each now calls `ReleaseAll` before `ReleaseObjects`, as `Close` does.
- **After the fix: G-MM3c-2 PASS.** The four pass alone. The full package releases 760 devices with no break of I1 or I2.
  Its only failures are the llama-tiny I3 cell (nobara-mm01's baseline fix, #383) and the two that fail on main before
  #335. Log: `~/goinfer-logs/mm3c-metal-device-afterfix.log` on the Mac.

### 6.6 What it absorbs, and what stays outside

- **Absorbs:** KV11 (MM4); the fit guard's byte and KV accounting, which the owner put on Track OP on 2026-10-21 (MM5);
  MEM12-scratch's tier-0 counters (I1); MEM2's real `Close` and MEM6's returning memory after each chunk (I2); MEM7's
  sizing to the machine (`Solve`); the guard-threshold rule (I7).
- **Stays outside:** the paging mechanics (the pagers keep their I/O and ask the arbiter), the allocators' own code
  (they report to the ledger), and the kernels.

### 6.7 Open questions

- macOS: which reading the ledger closes against (the task's footprint, not RSS), and how Metal residency sets and
  no-copy buffers count.
- CUDA pinned host memory as its own tier, or `HostAnon` with a flag.
- Whether the per-process `VmSwap` reading (owner, 2026-10-17) stays a guard or becomes I3's calibration input on Linux.

#### MM1's first item: the two CUDA device tests that fail on main, registered 2026-10-26 before the fix (nobara-mm01)

Francis put these in MM1's scope (2026-10-26 17:35): `TestCUDAKVSlots_pricedAgainstWhatIsLeft` (the free-VRAM probe runs twice,
the test wants once) and `TestResidentDenseBytes_matchesCUDADevice` on the 0.5B and the 1.5B (479 MB and 998 MB more on the card
than Plan expects). Both fail the same way at origin/main (the AX2d logs on nobara).

**The cause, as found in the code before any run:** KS-B8's `choosePrefillWidth` (commit 1622062be) reads `cudaFreeVRAM` a second
time, as the build's last step, after the KV pools are allocated. Both tests stub `cudaFreeVRAM` and treat every call as
`checkKVFits`' read, which comes before the KV. The probe-count test fails on the count. The dense-bytes test keeps the last
call's reading, so its "device before KV" figure includes the KV pools. That would explain its surplus: Qwen2.5's KV at the
default context comes to roughly what is reported. Production is unaffected: `checkKVFits` still reads first and alone prices the KV,
and no fit decision or default moves. So the fix is to the tests: each records the build's first read.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-CU-1 the cause | in the dense-bytes test on the 0.5B and the 1.5B, the second read's drop exceeds the first's by at least the KV pools' charged bytes (`kvChargedBytes` at the build's context and slots), and the first read is within Plan + slack + margin | a timed job holds the card (the run takes the timing lock), or the build reads the probe other than twice | one run | under a minute of GPU on nobara, both models from `~/models` |
| G-CU-2 the fix | both tests pass at the fix on llama-tiny, the 0.5B and the 1.5B, with `-v` showing `--- PASS` for each case; the rest of `cuda`'s `cudaFreeVRAM`-stubbing tests (`TestCUDAKVSlots_`, `TestKSB8`, `TestKSH6`, `TestBuildScratch`) still pass | as G-CU-1 | one run | under two minutes |

If G-CU-1 fails (the second read does not account for the surplus), the cause is something else and the fix is not made: the
finding is reported first.

**Read, 2026-10-26 (nobara, RTX 2070 SUPER, driver 595.91.07; kwin_wayland's 17 MiB context present; timing lock held): both pass.**
G-CU-1: the build reads the probe twice, and the second read sits 480 MB (0.5B) and 1,008 MB (1.5B) above the first, exactly
the KV pools' charged bytes (one slot at context 16,384). That accounts for main's 479 MB and 998 MB surplus. At the first read
the device holds 466 MB and 1,054 MB, 5 and 6 MB above Plan + slack, inside the 214 and 248 MB margins. G-CU-2: both tests pass
on llama-tiny, the 0.5B and the 1.5B, with every `TestCUDAKVSlots_` test. With `GOINFER_HEAVY_TESTS=1`, `TestBuildScratchAccounting`
(six models), its planted-defect test and the three KS-B8 build tests pass too. Logs: `~/goinfer-logs/mm1/` on nobara
(cuda-probe-tests and cuda-probe-heavy, 2026-10-26).

## 7. Track AX — the towers and GPU backends move into goinfer (SD1, decided 2026-10-26)

### 7.1 What moves (aikit main on 2026-10-26)

| aikit package (module) | files, lines | tests | goinfer non-test importers today | lands in goinfer |
|---|---|---:|---|---|
| `github.com/townsendmerino/aikit/vision` (root module; Hard tier) | 23, 6,323 | 30 | 33 files: 18 in the root module, 1 in `demo/agent`, cuda 6, metal 3, gpu 5 | `internal/vision` (SD5) |
| `github.com/townsendmerino/aikit/audio` (root module) | 8, 2,227 | 6 | root 10, cuda 1, metal 1 | `internal/audio`, beside `internal/whisper` |
| `github.com/townsendmerino/aikit/gpu` (its own module, v0.35.0): the device substrate (Metal and CUDA behind one `Device`/`Buffer`/`Queue`/`Pipeline`/`Encoder` vocabulary), copies and uploads, streams, residency sets, the ViT kernels | 15, 5,125 | 38 | cuda 31, metal 8, root 1 | `internal/gpudev` |
| `…/gpu/qwencuda`, `…/gpu/visioncuda` (modules) | 1 each, 484 and 330 | 1 each | `visioncuda` in one cuda test | the `cuda` module |
| `…/gpu/qwenmetal`, `…/gpu/visionmetal` (modules) | 1 each, 527 and 398 | 2 and 3 | none | the `metal` module |
| `…/gpu/webgpu` (module) | 1, 43 | 1 | none | the `gpu` module, or retired |

- **Stays in aikit:** `linalg`, `embed`, `encoder`, `mmap` and the RAG packages (`chunk`, `bm25`, `ann`, `fuse`,
  `hybrid`, `late`, `sparse`, `topk`). The edges run one way: `vision` and `audio` import `embed` and `linalg`, and no
  RAG package imports `vision`, `audio` or `gpu`.
- **The edge cases (SD2):** `…/gpu/anncuda` and `…/gpu/annmetal` (ANN on the GPU, used by aikit's own `examples/gpu-ann`)
  and `…/gpu/enccuda` and `…/gpu/encmetal` (the text encoder's matmuls on the GPU; no goinfer, ken or tack import)
  depend on the device substrate.

### 7.2 Where it lands, and why internal

- **Internal packages add no public API,** so no apidiff and no tier promise (SD5). The `internal` rule is by import
  path, so the `cuda`, `metal` and `gpu` modules, whose paths sit under `github.com/townsendmerino/goinfer`, can import
  the root module's internal packages; `golang.org/x/tools/gopls` does the same with `golang.org/x/tools/internal`.
- **`multimodal` (public) keeps the goinfer side of each tower** (block sentinels, preprocessing configs, splicing) and
  exposes what the examples and the demo need, so no example imports an internal package.
- **Version skew follows the existing rule:** a backend module's tag pairs with the root tag it was built with
  (`RELEASING.md`'s two-step tag).

### 7.3 Steps

| step | what | gate (sketch) |
|---|---|---|
| AX0 | freeze the moving paths in aikit at the tags the move starts from (root v1.75.0 or later, `gpu/v0.35.0` or later); a note in aikit's README | no aikit commit touches a moving path after its freeze tag |
| AX1 | copy into goinfer with provenance: the commit message names the aikit tags; NOTICE and THIRD_PARTY_LICENSES lines move with the code; tests, goldens and fixtures move; build tags kept | the moved tests pass in goinfer, by name, with `-v` (a skip is not a pass) |
| AX2 | switch the imports one module at a time: root, then gpu, metal, cuda | **tower identity:** for every tower family's fixture, the tower's output bytes are identical before and after (raw bits, mutation-checked once); every family's image and audio parity tests unchanged; build, vet and staticcheck under every tag |
| AX3 | drop `github.com/townsendmerino/aikit/gpu` and `…/gpu/visioncuda` from every goinfer `go.mod`; a source scan in `scripts/touched_scans.py` refuses an import of a moved aikit path | `GOWORK=off` builds of each module after the two-step tag |
| AX4 | aikit side. `vision` is in the Hard tier: a `Deprecated:` package comment pointing to goinfer, frozen, removed only at a v2. `audio` is not: deprecated for one minor, then removed. The GPU module and the four tower backends: deprecation comments in their `go.mod` files and no further tags. `gpupins` and the GPU gate tools trimmed to what stays; aikit's README tiers updated | aikit's releasegate and consumer gate (ken, tack) green |
| AX5 | SD2: the ANN and text-encoder GPU backends | per SD2 |
| AX6 | SD3: `linalg` and `embed` worked on through `go.work` (goinfer's gitignored `go.work` lists `../aikit` for cross-repo work); aikit tagged weekly; with VK1, a bump with unchanged numerics re-stales nothing | — |
| AX7 | 30 days after AX4: aikit bumps in goinfer, aikit tags, pin-only commits and parity re-stales caused by aikit, against §2.4 | — |

### 7.4 What it should buy

- About 15 of the last 30 days' 37 bumps (the towers and the GPU module) stop being bumps; the linalg and embed changes
  batch into about four tags a month.
- The pin-only commits for the moved modules stop.
- Tower and device work becomes an ordinary goinfer edit: the test cache, `touched.py`, one PR.

### 7.5 Risks

- aikit keeps a frozen copy of the device substrate for the ANN and encoder backends (SD2); the two copies diverge by
  design.
- The history stays in aikit; goinfer's copy cites the freeze tags.
- Cross-repo citations in goinfer's docs that point at moved aikit files must be re-pointed in the same commits, or the
  citation lint goes red.
- The CUDA half of the substrate builds on Linux only; the move keeps its build tags.

### 7.6 AX1 gates (registered 2026-10-26 11:40 PDT by mac-ax01, before the copy was made or any test ran; committed here once this doc reached the branch)

Tier 0 throughout: no model, no timed run. The Metal device tests touch the Mac's GPU, so every test command holds the
timing lock shared (`timing_lock.py run --shared`).

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX1-T (tests carry over) | Baseline: each moved package's tests at its freeze tag (a `git archive` of root `v1.75.0` and of `gpu/v0.35.0`, tracked files only), on this Mac, `go test -v -count=1`. The copy: the same packages in goinfer, same flags. **Pass:** every test that reads `--- PASS` in the baseline reads `--- PASS` in goinfer, by name, and no test reads `--- FAIL` in goinfer. A test that skips in the baseline and skips in goinfer for the same reason (a real checkpoint, an untracked bench fixture, a CUDA device) is listed, not counted as a pass. | a baseline test fails at the tag (it is listed and excluded, not fixed here); the timing lock is held exclusive by another session (wait, do not overlap) | each side runs once, to completion; a red is re-run by name only | not yet measured; estimated under 10 min a side on the Mac (vision and audio CPU tests plus the Metal ViT tests); the baseline's wall time becomes the record |
| AX1-S (source identity) | Every moved `.go` file equals its file at the tag after the scripted rewrite (import paths, the package clause and doc line of `gpudev`, the fixture paths), and nothing else differs; every non-Go file (`.cu`, `.ptx`, `testdata/`) is byte-identical. A script diffs the two trees. | — | complete | seconds |
| AX1-B (builds under every tag) | `go build` and `go vet` of: the root module (darwin), `internal/gpudev` with `GOOS=linux`, the `metal` module (`metal`, `metal goinfer_testhooks`), the `cuda` module with `GOOS=linux GOARCH=amd64 CGO_ENABLED=0` (`cuda`, `cuda goinfer_testhooks`), the `gpu` module; `gofmt -l` empty; CI's pinned staticcheck on the new packages, after its canary prints U1000 | — | complete | about a minute |
| AX1-N (adds only) | `git diff origin/main --name-status` shows only added files, plus `go.mod`/`go.sum` lines, `.gitignore` exceptions for the moved fixtures, NOTICE and THIRD_PARTY_LICENSES lines, and docs; no existing `.go` file changes | — | complete | seconds |

What this gate cannot see, and who runs it: the CUDA device tests (`internal/gpudev`'s `cuda_*_test.go`, `cuda/internal/visioncuda`,
`cuda/internal/qwencuda`) build here with `GOOS=linux` and run only on nobara; they are listed for nobara in the PR.

**Result, 2026-10-26 (mac-ax01):** AX1-S, AX1-N, AX1-B and AX1-T PASS on the Mac; the measured cost was 11 s for the aikit baseline and 8 s for goinfer. The record, the logs and the two scripts are in [`../measurements/ax1-2026-10/`](../measurements/ax1-2026-10/README.md). Open: the CUDA device tests on nobara.

**Amended after CI, 2026-10-26:** the device package does not land at `internal/gpudev`. In the root module it put purego and
gocudrv into the root's packages and module graph, which `ci.yml`'s cleanliness guard refuses; a nested module would
have been a sixth module against rule 4 (§4). It splits by backend instead: `cuda/internal/gpudev` (the Linux files, the
`.cu` sources and PTX) and `metal/internal/gpudev` (the darwin files), shared files in both. That supersedes the
`internal/gpudev` landing spot in §7.1 and §9. Gates re-run on the split, all PASS; the record has the detail.

### 7.7 AX2 plan (proposed by mac-ax01 at AX1's stop point; AX2 registers these gates in full, with void, stopping and cost terms, before its code)

**`vision` and `audio` switch in every module in one commit; the device package switches one module at a time after**
(each backend to its own half).
The towers cannot go one module at a time: the backends register their resident encoders into `vision`'s package-level
hooks (`vision.RegisterResident`, `vision.RegisterQwenResident`) and receive the root's `*vision.Encoder` values, so a
root on `internal/vision` beside a backend on aikit's `vision` either fails to compile (the types) or compiles and
silently runs the CPU tower (the hooks). Importers on origin/main `4a91080e`, non-test + test files:
`vision` 18 + 22 root, 6 + 23 cuda, 3 + 16 metal, 5 + 8 gpu, 1 `demo/agent`; `audio` 10 + 18 root, 1 + 3 cuda, 1 + 3
metal; aikit's `gpu` 31 + 52 cuda, 8 + 10 metal (the root imports none: `cmd/gate`'s release check and
`internal/hwcensus`'s census name it as a module string, and AX3 points both at the two `gpudev` halves).

| step | what | gate |
|---|---|---|
| AX2a | a tower-identity test, landed before any import moves: for every tower family's committed tiny fixture (SigLIP, Gemma 4, Qwen2.5-VL, Qwen3.5-VL, Qwen3-VL DeepStack, GLM-OCR, Pixtral, SigLIP2 NaFlex, Whisper, Gemma 4 audio, Qwen3-ASR, Voxtral) and every backend present (CPU; Metal on the Mac; CUDA on nobara), it writes the tower's output bytes' sha256 and whether the resident (device) path ran, to a JSON record | records at the base commit; tier 0 |
| AX2b | `vision` and `audio`: rewrite every importer in root, `gpu`, `metal`, `cuda` and `demo/agent` to `internal/vision` and `internal/audio`, one mechanical commit (a script, as AX1's); `examples/invoice` moves to what `multimodal` exposes (§7.2) | **tower identity:** every (family, backend) hash equal to AX2a's record, and the resident flag unchanged; mutation-checked once by leaving one backend on aikit's `vision` (its resident flag must flip and the gate go red); every family's image and audio parity test unchanged; build, vet and staticcheck under every tag; apidiff clean |
| AX2c | aikit's `gpu` to `metal/internal/gpudev` in the `metal` module (with `metal/kernels`; `metal/internal/*` already use it) | the same identity record on Metal; the metal suite's device tests by name |
| AX2d | aikit's `gpu` to `cuda/internal/gpudev` in the `cuda` module, and its one test's `visioncuda` import to `cuda/internal/visioncuda` | the same identity record on CUDA, on nobara; the CUDA suite's device tests by name |

Then AX3 drops `aikit/gpu` and `aikit/gpu/visioncuda` from every `go.mod`. Docs citing a moved aikit file stay valid
while aikit keeps the frozen copy (AX4 removes `audio` after one minor); each is re-pointed in the AX2 commit that moves
its importer, or the citation lint goes red at AX4.

### 7.8 AX2a and AX2b gates (registered 2026-10-26 15:22 PDT by mac-ax01, before AX2a's code)

Tier 0: tiny committed fixtures, f32 towers, no model, no timed run. Every Mac test command holds the timing lock shared.
The test-only environment variable below is read by tests only, so `testdata/env_reads.txt` does not change.

**The instrument (AX2a).** `internal/towerid` runs each tower family on its committed tiny fixture with fixed synthetic
inputs: SigLIP, Qwen2.5-VL, Qwen3.5-VL, Qwen3-VL with DeepStack, GLM-OCR, Gemma 4 vision, Pixtral, SigLIP2 NaFlex,
Gemma 4 audio, Whisper, Voxtral and Qwen3-ASR. On a device backend it attaches each tower through the registry production
uses: the unnamed slots (`EnableResident`) for SigLIP and Qwen2.5-VL, and the named ones (`multimodal.NewQwen3Tower`,
`NewGlmOcrTower`, `NewGemma4Tower`, `embeddinggemma2.NewAudioAccelerator`, `multimodal.UseWhisperDevice`) for the rest. It
covers every family the backend serves. A record holds, per (backend, family, case), the SHA-256 of the output's float32
bits and whether the device path attached. One test per backend calls it: the root module (CPU), `metal` (Metal), `gpu`
(WebGPU, `-tags gpu`) and `cuda` (CUDA, on nobara). `GOINFER_TOWER_ID_OUT=<file>` writes the record.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2a-T (the instrument holds) | Each backend's test passes. Every case runs twice with equal hashes, or the identity gate below could not be read. Every family the backend serves attaches its device path; an attach error is a FAIL, not a skip. | the backend's device is absent (a named skip, VOID for that backend, never a pass) | once per backend, to completion | not measured; estimated under a minute a backend on the Mac. The first run's wall time becomes the record |
| AX2b-ID (tower identity) | A record from the AX2a test at AX2b's base commit and one at AX2b's head, same machine and flags. **Pass:** every (backend, family, case) hash and device flag is equal. Any difference is a FAIL, with no tolerance band: the switch changes no numeric code, because `internal/vision` and `internal/audio`'s production files are aikit v1.75.0's, byte for byte. Backends: CPU, Metal and WebGPU on the Mac; CPU and CUDA on nobara (relayed, by day). | as AX2a-T | each record once; a difference is diagnosed by name, never re-run until it agrees | AX2a-T's measured time, twice per backend |
| AX2b-M (the identity gate can go red) | Once, at AX2b's head, plant the failure it exists for. One backend's registration for an unnamed slot is left on aikit's `vision`. If that compiles, its family's device flag must flip and AX2b-ID read FAIL. If no such plant compiles, the record says so: the split cannot ship silently there. The gate is then planted at the test instead, with the registration name changed, and must read FAIL. | — | once | minutes |
| AX2b-N (the mechanical commit is the script) | The import switch is one commit made by a script. Re-running the script on that commit's parent reproduces it byte for byte. The commit contains nothing else. The script also runs gofmt. | — | complete | seconds |
| AX2b-S (nothing left on aikit's towers) | `go list -deps` of every module, under every build tag it uses (root with `realckpt` and `goinfer_testhooks`; `metal`; `gpu`; `cuda` with `GOOS=linux`; `demo/agent`), names neither `github.com/townsendmerino/aikit/vision` nor `.../aikit/audio`, test packages included | — | complete | seconds |
| AX2b-B (builds) | Build and vet under every tag. The cleanliness guard is run from the root as CI runs it. gofmt, and CI's staticcheck after its canary. `touched.py`. CI's full suite on the PR, apidiff included. The image and audio parity tests are in that suite, and none of their goldens may change. | — | complete | minutes; CI as usual |

**AX2a result, 2026-10-26 (mac-ax01):** AX2a-T PASS on the CPU (12 families, 22 cases), Metal (6 families, 17 cases) and
WebGPU (5 families, 15 cases), each under a second; CUDA builds and waits for nobara with AX2d. A device backend with no
registration linked in errors on its first declared tower instead of returning CPU bits. The record is
[`../measurements/ax2-2026-10/`](../measurements/ax2-2026-10/README.md).

**AX2b result, 2026-10-26 (mac-ax01):** AX2b-N, -S, -ID (CPU, Metal, WebGPU: 54 cases, 0 differ) and -M PASS on the Mac;
AX2b-S caught `cuda/s4siglip`'s blank import of aikit's `gpu/visioncuda`, which compiles and would have registered into a
slot nothing reads, and the script now maps it to `cuda/internal/visioncuda`. AX2b-M's fallback form: no single
registration file left on aikit compiles, and two plants at the registration each turned the Metal identity test red.
CUDA's and nobara's CPU records wait for nobara. Detail in the record.

**What AX2b changes outside the import lines.** First, `multimodal` and `embeddinggemma2`'s exported functions that take
tower types now take `internal/vision` and `internal/audio` types. Both packages are Experimental tier
(`docs/api-tiers.md`; apidiff checks `decoder`, `tokenizer`, `chat` and `constrain`), so this is allowed, and it gets a
CHANGELOG line. Second, `examples/invoice` and `demo/agent` reach the towers through new `multimodal` functions instead
of importing an internal package (§7.2). That is a separate commit after the mechanical one, gated by AX2b-B, with each
example's own test.

### 7.9 AX2c gates (registered 2026-10-26 15:45 PDT by mac-ax01, before AX2c's code)

AX2c switches the `metal` module (with `metal/kernels`) from aikit's `gpu` to `metal/internal/gpudev`, the device
package's Metal half from AX1. It is one scripted commit (`ax2cd_rewrite.py metal`): the import path becomes
`gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"`, so no body changes. Tier 0 and tier 1 (the Mac's GPU,
no model), by day, under the shared timing lock.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2c-N | Re-running the script on the commit's parent reproduces the commit byte for byte. The commit contains nothing else. | — | complete | seconds |
| AX2c-S | `go list -deps -test -tags goinfer_testhooks ./...` in `metal` names no `github.com/townsendmerino/aikit/gpu`. | — | complete | seconds |
| AX2c-ID | `TestTowerIdentity_metal`'s record at AX2c's head equals AX2b's head record, case by case. | no Metal device | once | a second |
| AX2c-T (the Metal suite carries over) | The `metal` module's tests, one package per `go test -v -count=1 -tags goinfer_testhooks` call, at AX2b's head (the base) and at AX2c's head, compared by name with `ax1_compare.py`. **Pass:** every base PASS is a PASS at the head, and no FAIL at the head. A test that fails at the base is listed and excluded, not fixed here. | another session holds the timing lock exclusive (wait); the Mac's free memory is under 4 GB at a run's start (a device suite under memory pressure fails for reasons no code owns; the run is re-made) | each side once, to completion; a red is re-run by name only | the Metal suite measured 4 min 12 s in a quick-full run (`CLAUDE.md`, "Gate tiers"); estimate 5 min a side, 10 min in all |
| AX2c-B | Build, vet and staticcheck of `metal` under its tags (plain, `goinfer_testhooks`); gofmt; the root cleanliness guard | — | complete | a minute |

### 7.10 AX2d gates (registered 2026-10-26 15:45 PDT by mac-ax01, before AX2d's code)

AX2d switches the `cuda` module from aikit's `gpu` to `cuda/internal/gpudev`, the device package's CUDA half from AX1,
in one scripted commit (`ax2cd_rewrite.py cuda`). The Mac compiles it (`GOOS=linux`); the device legs run on nobara,
by day, through nobara's own session (relayed), never over ssh from the Mac.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX2d-N | Re-running the script on the commit's parent reproduces the commit byte for byte. The commit contains nothing else. | — | complete | seconds |
| AX2d-S | `GOOS=linux go list -deps -test -tags 'cuda goinfer_testhooks' ./...` in `cuda` names no `github.com/townsendmerino/aikit/gpu`. | — | complete | seconds |
| AX2d-B | Build, vet and staticcheck of `cuda` with `GOOS=linux GOARCH=amd64 CGO_ENABLED=0`, tags `cuda` and `cuda goinfer_testhooks`; gofmt; the root cleanliness guard | — | complete | a minute |
| AX2d-ID (nobara) | `TestTowerIdentity_cuda` and `TestTowerIdentity_cpu` records at AX2b's base, AX2b's head and AX2d's head, equal case by case (one nobara session covers AX2b-ID's CUDA leg too). | no CUDA device answers | once each | seconds each |
| AX2d-T (nobara) | The `cuda` module's tests (`-tags 'cuda goinfer_testhooks'`, one package per `go test -v -count=1` call) at AX2c's head (the base) and at AX2d's head, compared by name with `ax1_compare.py`. **Pass:** every base PASS is a PASS at the head, and no FAIL at the head. `TestCUDA_launchDirect` segfaults on both sides at AX1 (#279), so it runs alone and the rest with `-skip '^TestCUDA_launchDirect$'`; a test failing at the base is listed and excluded, not fixed here. | a desktop session holds a CUDA context (`kwin_wayland`), which shifts plans and skips the drain-group tests (`CLAUDE.md`, "Testing tiers"): the record notes it, and drain-group skips on both sides are not counted as passes | each side once, to completion | nobara's session states the estimate from its last CUDA suite record before starting |

**AX2c and AX2d results, 2026-10-26 (mac-ax01):** AX2c-N, -S, -ID, -T (775 base passes, 0 regressions) and -B PASS on
the Mac. AX2d-N, -S and -B PASS on the Mac (compiled for linux/amd64); AX2d-ID and -T wait for nobara. Detail in the record.

### 7.11 AX3 gates (registered 2026-10-26 16:54 PDT by mac-ax01, before AX3's code)

AX3 drops `github.com/townsendmerino/aikit/gpu` and `…/gpu/visioncuda` from every goinfer `go.mod`. A test then keeps the
moved paths out: it fails on a `.go` file importing aikit's `vision`, `audio` or `gpu` (or anything under it), and on a
`go.mod` requiring the `gpu` module or one under it. Three things outside the imports move with it.

- **`internal/hwcensus`'s census** scans goinfer's two `gpudev` halves in place of the aikit module.
- **`cmd/gate release`'s row 2 and `decoder/aikit_pin_test.go`** stop pinning `aikit/gpu`; no module requires it.
- **The citation lint.** It resolves aikit `gpu/...` citations through `cuda/go.mod`'s require. With the require gone,
  it falls back to the frozen tag `gpu/v0.35.0`, which aikit keeps (SD2, AX0), so those citations stay valid. §7.5
  expected every citation to be re-pointed instead. CI's cache step downloads that tag.

One mechanical commit first: `cuda/kernels/ptx_test.go` (SB1's batch 1, merged after AX2d's rewrite) still imports aikit's
`gpu`, and `ax2cd_rewrite.py cuda` switches it.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| AX3-S (the scan) | The new scan passes on the tree. Planted once, it reads red twice: an aikit `gpu` import added to a test file, then an `aikit/gpu` require added to `metal/go.mod`. It reads green again after each plant is removed. It is listed in `scripts/touched_scans.py`. | — | complete | seconds |
| AX3-M (the module graphs) | No goinfer `go.mod` or `go.sum` names `aikit/gpu` or `aikit/gpu/visioncuda`. `go list -m all` names neither in `metal` (workspace), nor in `cuda` (workspace, `GOOS=linux`). | — | complete | seconds |
| AX3-B (builds and the moved checks) | Build, vet and staticcheck of every module under its tags. The root cleanliness guard. `TestHardwareCensus`, `cmd/gate`'s release tests and `decoder`'s pin test, by name with `-v`. `test_touched.py` and the citation lint, exit code read directly, with every aikit `gpu/...` citation resolving. `TestTowerIdentity_metal` against AX2c's record, as a smoke check. | — | complete | minutes |
| AX3-G (standalone builds) | §7.3's `GOWORK=off` build of each module. It needs the two-step tag, so it is deferred to the next release, where `standalone-build.yml` runs it; nothing here grades it. | — | — | — |

### 7.12 AX5 gates (registered 2026-10-27 16:35 PDT by mac-ax01, before AX5's code)

SD2: the ANN and text-encoder GPU backends (`gpu/anncuda`, `gpu/annmetal`, `gpu/enccuda`, `gpu/encmetal`) stay in aikit
on the frozen `gpu/v0.35.x`. AX4 (aikit #22, released in v1.76.0) already took them off the newest root tag: `gpupins`
judges only their `aikit/gpu` require, against the latest `gpu/v0.35.x` tag, and no longer judges or rewrites their root
`aikit` require, which stays at `v1.75.0`, the freeze tag. Left: a check that fails if a bump comes back. Today a root
bump of the four would pass `gpupins` silently, and so would a `gpu` require moved off the `v0.35` series by hand.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| AX5-F (the freeze is checked) | `gpupins` judges each of the four on both requires: `aikit` must equal the freeze tag `v1.75.0` (a constant beside the backends list, with SD2 named), and `aikit/gpu` the latest `gpu/v0.35.x` tag, as now. `--fix` still never writes the root require. CI's existing `gpupins` step (aikit `ci.yml`) is where it fails. | — | complete | seconds |
| AX5-R (it goes red) | Planted on a scratch copy of the four go.mod files, each reads FAIL: a root require bumped to `v1.76.0`, and a `gpu` require moved to `v0.36.0`. As unit tests in `tools/gpupins` (no git tags needed), plus one run of the real command against a planted tree, then reverted. | — | complete | seconds |
| AX5-B | `go vet` and `go test` of `tools/gpupins`; `go run -C tools ./gpupins` reads OK on aikit's main; aikit's lint on the tool. | — | complete | seconds |

### 7.13 AX6 gates (registered 2026-10-27 16:35 PDT by mac-ax01, before AX6's code)

SD3: goinfer's main may require an aikit pseudo-version between weekly aikit tags; a goinfer release requires a tagged
aikit. Nothing on main refuses a pseudo-version today, so the main half holds already. The release half is enforced where
a release is cut: `go run ./cmd/gate release vX.Y.Z` (RELEASING.md's pre-flight), whose module-pins row reads every
module's `aikit` require.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| AX6-P (the release row refuses a pseudo-version) | `gate release`'s pins row is OWED when any of the five modules requires an `aikit` pseudo-version (`vX.Y.Z-[0.]yyyymmddhhmmss-<12 hex>`, the three forms Go writes), naming the module and version and SD3. A tagged version, a pre-release tag (`v1.80.0-rc.1`) and agreement are unchanged. | — | complete | seconds |
| AX6-R (it goes red) | `TestRowPins` gains the three pseudo-version forms (agreeing across all five modules, so only the new rule can fail them) and a pre-release tag that must stay done. Shown red once by disabling the check, then reverted. | — | complete | seconds |
| AX6-B | `touched.py` green; `cmd/gate`'s release tests by name with `-v`; RELEASING.md's invariants say both halves. | — | complete | seconds |

**AX5 and AX6 results (2026-10-27, the Mac).**

- **AX5** (aikit #24): `gpupins` judges the four backends' root `aikit` require against `frozenRoot` (`v1.75.0`), and
  a moved one FAILs with a message that `--fix` does not move it; the `gpu` require is held to the latest `gpu/v0.35.x`
  tag as before, and `--fix` writes only that. **AX5-F:** PASS on aikit's main (`all 4 backends pin aikit v1.75.0 and
  gpu v0.35.0`). **AX5-R:** `TestCheckModule_theFreezeHolds` (frozen passes; a root bump, a root pseudo-version and a
  `gpu` require of `v0.36.0` fail) and `TestFix_neverMovesTheFrozenRoot` pass; the real command, run on a tree with
  `anncuda`'s root bumped to `v1.76.0` and then with `encmetal`'s `gpu` at `v0.36.0`, FAILs each time, reverted.
  **AX5-B:** vet, the tool's tests and staticcheck clean; staticcheck also found `orNone` unused on main, removed.
- **AX6** (this repository): `gate release`'s pins row is OWED on an `aikit` pseudo-version (`rePseudo`, Go's three
  forms), naming the module and SD3; RELEASING.md's version-alignment invariant states both halves. **AX6-P/R:**
  `TestRowPins` covers the three forms (each agreed across all five modules, so only the new rule fails them) and a
  pre-release tag that stays done; with the check disabled the three forms read releasable (red), reverted. Today all
  five modules require the tag `v1.76.0`, so the row reads done. **AX6-B:** gofmt and `go vet ./cmd/gate` clean;
  `TestRowPins` and `TestModRequire` pass with `-v`; citation lint exit 0. `touched.py` and staticcheck were refused:
  a timed run (`tack eval pilot`, driving nobara) has held the Mac's timing lock since 16:49, and the scans `touched.py`
  would add for this diff are the tree-wide ones any Go change selects, which CI runs. The tool tests above ran without
  the shared lock for a few minutes inside that run; it measures nobara, so the Mac's load does not void its cells.
- **AX6 follow-up: Gemini's review of #350, and AX6-B passed (2026-10-27 17:42–17:44 PDT, under the shared lock).**
  `TestRowPins` now asserts that the row's detail names each module with its version and cites SD3, and covers a
  pseudo-version with build metadata (`…+incompatible`), which `rePseudo` now matches. With the build-metadata group
  removed from `rePseudo`, the `+incompatible` case reads releasable and the test fails; reverted. **AX6-B:**
  `touched.py` green (`cmd/gate` in full, 84 s, and the tree-wide scans), staticcheck on `cmd/gate` clean. The row was
  marked done in #350 before AX6-B ran; it now has its evidence.

## 8. The other subsystems (SB1–SB13)

Each: what it is today, its contract, its fast check, what stops re-running, its first step and size.

### SB1 — the backend interface, at the level of operations

- **Today:** §2.2.
- **Contract:** a backend implements operations, not families. It receives a program, a view of the weights and a
  memory plan, and returns a resident with one call shape (RF1) that runs ragged rows. It declines at build time, by
  operation and shape, never at run time.

  ```go
  // package backendapi (internal): imported by decoder and by the cuda, metal and gpu modules. A sketch; OP4 and RF1 own the design.
  type Backend interface {
      Name() string
      Ops() OpSet                                                // the operation kinds and shape classes it implements
      Build(p Program, w Weights, plan mem.Plan) (Resident, []Decline, error)
  }

  type Resident interface {
      Forward(ctx context.Context, b Batch, out Outputs) error // decode, verify, prefill and image rows
      Demand() mem.Demand                                      // MM2
      Close() error
  }
  ```
- **Fast check:** an op conformance kit per backend, in the shape of llama.cpp's `test-backend-ops` and PyTorch's OpInfo
  (§16). Every operation kind runs on random small tensors across shape classes (head dims 64, 128, 256 and 512, GQA
  ratios, tails, thresholds) against the CPU reference: bit-identical where the operation is declared exact, within a
  declared tolerance where it is not. Synthetic two-layer programs exercise the walker. Nothing imports `decoder`.
- **Stops re-running:** backend suites on decoder edits, and family × backend runs (a pair carries over by VK's key).
- **First step, before OP4:** move the kernel-level tests in `cuda/` and `gpu/` into packages that do not import
  `decoder`, the way `metal/kernels` did on 2026-10-13. A ratchet test: the count of backend test files importing
  `decoder` (240, 245 and 87 today) may only fall.
- **Then** OP4 (residents walk the program) and RF1 (one call shape) land with the kit as their gate. L in all; the
  first step S–M.

#### SB1 step 1 — gates, registered 2026-10-26 before the code (nobara-sb01)

- **What a file-level count misses.** Package `cuda`'s own non-test files import `decoder` (21 of them), so every test in
  package `cuda` is rebuilt and re-run on a decoder edit whether or not its own file imports `decoder`. The same holds
  for `metal` and `gpu`. So the ratchet counts two things per backend module: **direct**, the test files whose own
  imports name `decoder` (the §6 census figure: 240, 245, 87), and **reach**, the test files in a directory whose test
  build reaches `decoder` through any chain of goinfer imports. Reach is the one a decoder edit invalidates; a kernel
  test leaves it only by leaving the package, as `metal/kernels` did.
- **Ratchet (tier 0).** `internal/layout`'s `TestBackendTests_decoderImportsOnlyFall` parses the imports of every Go
  file in the tree (build tags ignored, so a count is the union over tags; `//go:build ignore` files skipped) and fails
  when either count of any of the three modules differs from the committed figure: above it is a regression; below it,
  the figure is lowered in the same commit. It lands at the counts of `origin/main` at `bc365a837`, in its own commit:
  direct 246, 245 and 87, reach 352, 395 and 172 for `cuda`, `metal` and `gpu`.
  `-v` prints the census. Cost: milliseconds; no device, no model. Void: none (it is deterministic over the tree).
- **Each batch of moved tests (tier 0 on the device, seconds).** The batch's files move from `cuda/` to `cuda/kernels/`,
  a package that imports nothing from goinfer (aikit `gpu` and gocudrv only); it reads the PTX the moved tests load
  from `cuda/testdata/*.ptx`, the same files `cuda/kernels.go` embeds. A moved file's `goinfer_testhooks` constraint is
  dropped: it gates package `cuda`'s test seams, which a moved test does not use, so the moved test runs under `-tags
  cuda` alone, in `gate gpu`'s 2a suite.
  - **Rule:** every moved `Test`/`Benchmark` function is listed before the move and found after it (no test deleted);
    each moved test is run by name with `-v` on nobara's RTX 2070 SUPER before the move (in `./cuda/`, under `cuda
    goinfer_testhooks`) and after (in `./cuda/kernels/`, under `cuda`), and its outcome (PASS, FAIL or SKIP with the same reason) is the same in both;
    `go build` and `go vet` of the cuda module pass under each of no tag, `cuda`, and `cuda goinfer_testhooks`;
    `go list -deps -test ./cuda/kernels/` contains no `goinfer/decoder`; the ratchet's reach count for `cuda` falls by
    the number of moved files, and is lowered.
  - **Void:** a run while `timing_lock.py status` shows a timed holder (the runs hold the lock shared, so they refuse
    instead); a run on a card with another process's context (`nvidia-smi` names it), which can fail a test on memory
    for reasons not in the test.
  - **Stopping rule:** one before-run and one after-run per batch; a mismatch stops the batch and is reported, not
    re-run until it agrees.
  - **Cost basis:** the kernel tests in `gate gpu`'s 2a suite take about 12 s together
    (`docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh`'s header); a batch is a fraction of it.
    Heavy-gated (`GOINFER_HEAVY_TESTS`) and env-gated (`GOINFER_A13_*`) tests skip in both runs, and the record says so.
- `gate gpu`'s 2a kernel-level suite lists `./cuda/kernels/` beside `./cuda/`, so the moved tests stay in it; the night
  `cuda-device-suite` and CI's cuda job run `./cuda/...` and need no change.
- **Batch 1 (2026-10-26):** 25 files (35 tests, 16 benchmarks) moved to `cuda/kernels/`, every outcome the same before
  and after; cuda reach 352 → 327. `gate gpu`'s heavy tier also runs `./cuda/kernels/` as one more process, so its two
  heavy-gated bandwidth tests stay in it. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`.
- **Batch 2 (2026-10-26):** 5 files (6 tests) moved, outcomes the same; cuda reach 327 → 322. Same record. The next dozen
  kernel tests check kernels against package `cuda`'s production packers, so they wait for the packers to move.

#### SB1 step 1 — metal's batches: gates, registered 2026-10-26 21:41 PDT before the code (mac-ax01)

Francis's 17:36 decision: metal and gpu, the test-only method. gpu has no test-only move (nobara-sb01's record, "metal and gpu").
metal has 62 candidate files (67 `Test` and 20 `Benchmark` functions; 39 tagged `darwin`, 23 `darwin && goinfer_testhooks`).
Each file's only package-`metal` dependencies are `metal/kernels` aliases, `device.go`'s aliases and the pure helpers in
`testshared_test.go`. None imports `decoder`.

- **The move.** The files go from `metal/` to `metal/kernels/` (`package kernels`), in batches.
  - Each file keeps its build line. `goinfer_testhooks` is dropped only where the file uses none of package `metal`'s test
    seams; that is always so for these, since none uses `metal`'s test hooks.
  - `metal/kernels` gains test-only files: lowercase aliases for the MSL names and pack wrappers `metal/msl.go` and
    `metal/pack.go` give package `metal`, and copies of `argmaxF`, `cosF`, `parityStats` and `assertParity` (it already has
    `mustFinite`, `f16ToF32` and the device aliases).
  - No `decoder` file changes. If one would have to, the batch stops with DECISION NEEDED.
- **Rule, per batch:**
  - every moved `Test`/`Benchmark` function is listed before and found after (none deleted);
  - every moved `Test` runs by name with `-v` on the Mac's Metal device before the move (`./metal/`, tags
    `goinfer_testhooks`) and after (`./metal/kernels/`, the same tags), with the same outcome (PASS, FAIL, or SKIP for the
    same reason), compared with `ax1_compare.py`;
  - `go build` and `go vet` of the metal module pass with and without `goinfer_testhooks`;
  - `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`;
  - `internal/layout`'s metal reach figure falls by the files moved (Direct stays 245) and is lowered in the same commit.
- **Void:** a run while another session holds the timing lock exclusive (the runs hold it shared and wait); a run with
  under 4 GB available (Metal tests fail on memory for reasons not in the test).
- **Stopping rule:** one before-run of all 67 and one after-run per batch; a mismatch stops the batch and is reported,
  never re-run until it agrees.
- **Cost basis:** kernel tests on tiny synthetic inputs, seconds each; the before-run is the measurement. Benchmarks are
  not run (`-run` only), and nothing is timed.

#### SB1 step 1 — moving cuda's weight packers: the bit-identity gate, registered 2026-10-26 before the code (nobara-sb01)

Francis's decision (17:36): move `packWeight` and its packers into a small decoder-free package, with a bit-identity gate
on the packed output registered first; stop with DECISION NEEDED if the move needs a `decoder` change or changes a byte.

- **The move.** `cuda/internal/wpack` (imports aikit `linalg` only) takes the packing bodies: `packWeight`'s per-kind
  packing, `packWeightStack`'s concatenation and its kind and K checks, `packI8`, `packI8Into`, `packInt4Into`,
  `permuteFast`, `nibblePosFast`, `f32tof16`. `decoder.Int4F32` is not imported: its body (`Int4F16`, then
  `linalg.F16ToF32Slice`) is the same two calls, made in `wpack`. Package `cuda` keeps every name as a thin wrapper, and
  keeps `planWeight`'s admission ahead of each pack, member by member, so its errors come in the same order. `hostW`
  stays in package `cuda`: it is the resident's plan (planned sizes, row readers, the int8 alternative, the stack's
  sources), and only its packed fields come from `wpack`.
- **Gate (tier 0, no device).** `cuda`'s `TestWeightPackers_bitIdentity` hashes (SHA-256) the output of every entry
  point above, through package `cuda`'s names, on seeded inputs: `packWeight` on every kind `planWeight` admits (int4 from
  `QuantizeInt4` and from `WrapInt4F16` with arbitrary f16 scales, subnormal and negative included; int8 with and without
  W8A8; q4k from raw blocks; f32; bf16), at more than one shape; `packWeightStack` on stacks of each; `packI8`,
  `packI8Into`, `packInt4Into` on random data; `nibblePosFast` on 0–7; `permuteFast` on 2^22 seeded words plus every
  single-nibble word; `f32tof16` on every one of the 2^32 float32 bit patterns. Each hash covers the kind, N, K, the
  three packed slices, and the error text where one is expected (a mixed-kind and a ragged-K stack, an unsupported
  kind). The hashes are written from the tree before any code moves and committed; after the move the same test, with no
  edit, must reproduce every one.
- **Void:** none; it is deterministic. **Stopping rule:** one run before (writes the hashes), one after; any differing
  hash stops the move with DECISION NEEDED. **Cost basis:** seconds, most of it the 2^32 sweep; measured: 28 s writing the
  32 hashes at `91fe36e7b`, 30 s reproducing all 32 on an unchanged tree.
- **Then** the kernel tests that needed only the packers move to `cuda/kernels` by batch 1's gate (each by name with
  `-v` on the device, before and after), using `wpack` directly, and the ratchet falls.
- **Done (2026-10-26):** all 32 hashes match after the move, and the gate went red on two planted changes (reverted). Six
  test files left package `cuda` (four to `cuda/kernels`, two to `cuda/internal/wpack`), every outcome the same; cuda
  reach 322 → 317. gpu has no test-only move (its kernels are package `gpu`'s production code); metal has 62 candidates
  that need a Mac run. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`.

#### SB1 — launch geometry out of package cuda: the identity gate, registered 2026-10-26 before the code (nobara-sb01)

Francis's decision (21:42), item 1. The relayed text names `decoder/prefill.go` and `decoder/flash_decode.go`; neither file
exists, and no `decoder` file declares these constants. They are `cuda/prefill.go`'s and `cuda/flash_decode.go`'s, which the
STOP POINT it answers named. This work reads it that way and touches no `decoder` file; its PR waits for Francis to confirm
the reading before auto-merge is set.

- **The move.** `cuda/internal/launchgeom` (standard library only) takes the launch geometry the attention-digest tests
  need: `faWarps`, `faMaxRows`, `faRowsPerCTA`, `faRun` and `faRowRuns` (`flash_decode.go`); `attnFusedBM`,
  `attnFusedThreads`, `attnFusedKPAD`, `attnFusedMinRows`, `imgBlockMaxNWin`, `imgBlocksMaxNWin` (`prefill.go`). The KV
  cache's f16 converters `f32ToF16RNE` and `f32SliceToF16RNE` (`kvhalf.go`) go to `cuda/internal/wpack`. Package `cuda`
  keeps every name, as a constant or a thin wrapper.
- **Gate 1, values (tier 0).** `cuda`'s `TestLaunchGeometry_identity` hashes every moved constant's value and every moved
  function's output over a grid (`faRowsPerCTA` over head dims 32-512 and GQA ratios 1-16; `faRowRuns` over n0, m,
  window and S; `imgBlockMaxNWin` and `imgBlocksMaxNWin` over windows and blocks; `f32ToF16RNE` on all 2^32 float32 bit
  patterns; `f32SliceToF16RNE` on a seeded slice), through package `cuda`'s names, into
  `cuda/testdata/launchgeom_hashes.txt`, written before any code moves and committed; after the move the same test, with
  no edit, must reproduce every hash.
- **Gate 2, outputs (device, short, under the lock shared).** Before and after, by name with `-v`: the moved tests
  (`TestAttnKernelDigests` compares every attention kernel's output digest with `cuda/testdata/attn_digests.json`), and
  `gate gpu`'s 2b parity group (`go test -tags 'cuda goinfer_testhooks' ./cuda/`, the resident parity tests). Same outcome
  for every test. Void: a run beside a timed holder (the shared lock refuses). Stopping rule: one run each side; a
  difference stops the move with DECISION NEEDED.
- **Then** the attention-digest family (`attn_digest_rig`, `attn_digest`, `attn_softcap_kernel`, `mamba_kernels`) and
  `prefill_splitkv_img` move to `cuda/kernels`, and the ratchet falls.

#### SB1 — gpu's kernel layer: decided, and its gates, registered 2026-10-27 before the code (nobara-sb01)

Francis's decisions: the alias split (05:46), contained in `gpu/internal/kernels` (06:34), after `decoder`'s adapter types
moved to `internal/adapter` (06:57, #314). The cut, from a type-checked declaration graph of package `gpu` with its tests
(tags `gpu goinfer_testhooks`): a declaration stays in `gpu` if it uses `decoder`, uses a declaration that stays, or is a
method of a type that stays; an exported method that stays keeps its type in `gpu`, and an unexported one becomes a
function. That leaves in `gpu` the residency glue (`backend.go` with the `decoder.RegisterBackend` call, so WebGPU still
registers through the same import, `hwinfo.go`, `kv_slots.go`, `kvcopy.go`, `prefill_scratch.go`, `residency.go`,
`resident_sample.go`, `testhooks_kvscale.go`, `deprecated_release.go`'s backend check) and 75 test files; 48 non-test and
78 test files move. Two in-package pre-steps first: `Context.uploadProj` becomes the function `uploadProj(c, w)`, and
`deprecated_release.go`'s `Release` aliases move to `release_aliases.go`.

- **Mechanics.** `gpu` keeps an alias for each of its 12 exported types and a wrapper for each of its 3 exported
  functions, so callers compile unchanged. A name the staying code uses from the moved code is exported inside the internal
  package, except a field or method of one of the 12 public types: that is reached through a package-level accessor in
  `gpu/internal/kernels` (a field as `*kernels.ContextDevice(c)`, a method as `kernels.ContextEnsureAttn(c, ...)`), so no
  name becomes public. A test helper used across the line is copied. A type-checked rewriting tool makes every edit; its
  rename table is committed with the record.
- **G1, declaration identity (tier 0, no device).** A standard-library tool compares the package at the base with `gpu`
  plus `gpu/internal/kernels` at the head: every base declaration (functions, methods, types, vars, consts, `init`s and
  `_` checks) appears exactly once, and its code, printed without comments, is identical after mapping back the rename
  table, the `kernels.` qualifiers, the accessor calls and the two pre-step conversions. The only other declarations
  allowed are the generated aliases, wrappers and accessors, each in its generated shape, and the copied test helpers,
  each equal to its source. Any other difference fails it.
- **G2, the gpu suite by name (WebGPU on Vulkan, under the lock shared).** `go test -count=1 -p 1 -v -tags 'gpu
  goinfer_testhooks' ./gpu/...` at the base and at the head: every test the base reports has the same outcome (PASS,
  FAIL, or SKIP with the same reason) at the head, wherever it now lives.
- **G3, logits (`go run ./cmd/gate identity <base> <head> -backend webgpu -assets tiny`).** Every family the base runs
  is byte-IDENTICAL; a family NOT RUN on both sides is listed with its reason.
- **G4, towers.** `TestTowerIdentity_webgpu` with `GOINFER_TOWER_ID_OUT` at the base and the head: every case's hash
  equal. The comparison with AX2a's Mac record (`docs/measurements/ax2-2026-10/records-ax2a/webgpu-darwin-arm64.json`)
  is a Mac run; its commands go to mac-ax01.
- **G5, API.** apidiff on package `gpu`: the only incompatible changes are the 12 types' identity (and the signatures
  that name them); no name removed, none added. CHANGELOG names the moved types, the `%T` and reflection change, and no new
  public package.
- **Void:** a device run beside a timed holder (the shared lock refuses it). **Stopping rule:** one run each side; any
  difference stops the move with DECISION NEEDED. **Cost basis:** G1 seconds; G2 measured at the base run; G3 families ×
  steps, stated before it runs; G4 3 s (measured 2026-10-26).
- **Then** the freed tests are already in `gpu/internal/kernels` (they move with the code they test, which keeps their
  bodies unedited), and `internal/layout`'s gpu reach falls by the moved test files.
- **Done (2026-10-27):** G1 PASS (1141 declarations), G2 PASS (281 tests, every outcome and skip reason the same), G3
  PASS (39 families byte-identical on WebGPU, 4 with no tiny checkpoint of their kind), G4 PASS on nobara (15 tower cases;
  the Mac comparison with AX2a's record is mac-ax01's), G5 PASS (the 12 types' identity only). 48 non-test and 75 test
  files moved; gpu reach 172 → 99. Record: `docs/measurements/sb1-gpu-split-2026-10-27/README.md`.

#### SB1 — gpu's kernel layer (item 2 of 21:42): parked, DECISION NEEDED (nobara-sb01, 2026-10-26)

A type-checked declaration graph of package `gpu` (tags `gpu`) shows the shape the split would take. 15 declarations
use `decoder` directly and 37 in all reach it; the other 557 of 594, the runners included, do not. So the cut that frees tests
is everything clean into `gpu/kernels`, with package `gpu` keeping the residency glue (`residency.go`, `kv_slots.go`,
`kvcopy.go`, `lora_resident.go`, `resident_sample.go`, `prefill_scratch.go`, `testhooks_kvscale.go`) and the backend
registration. It would free 80 of the 152 in-package test files. Its cost: 129 files move; 111 unexported names the glue
and the 72 remaining tests use must be exported (32 methods, and test seams such as `g4DropVNormForTest`); `gpu`'s public
API needs a wrapper or alias per moved exported name, and a mutable exported var cannot be aliased without splitting its
state; the four vision-tower files (AX's area) move too. A `metal/kernels`-style move (the WGSL sources only) is small but
frees no test, because every gpu kernel test dispatches through `Context`. Both readings are in the DECISION below; the
item is parked until Francis picks one.

#### SB1 step 1 — the last test-only moves on cuda and gpu, registered 2026-10-27 before the code (nobara-sb01)

A type-checked scan of every test file in `cuda/` and `gpu/` lists the package-level declarations each file uses and
where they are declared. Only four files are kernel tests whose dependencies all live outside the backend package or in
test helpers:
- `cuda/gemv_rn_test.go` moves to `cuda/kernels`. It reads its PTX from `../testdata` and packs with `cuda/internal/wpack`.
  `rnCfg`, which `gemm_mma_test.go` also uses and which stays, is copied there with its "must match the .cu" note.
- `gpu/moe_expert_test.go`, `gpu/moe_test.go` and `gpu/gumbel_test.go` move to `gpu/internal/kernels`. The gumbel test's
  host reference, `decoder.GumbelDrawForTest` and `decoder.GumbelKeyForTest`, is three calls into `internal/sampling`; the
  moved test makes those calls itself, so it is held to the same code.

Everything else that does not import decoder is one of these:
- a test of package `cuda`'s or `gpu`'s own production code: packers' wrappers, slot and split-KV policy, the resident,
  the backend;
- a source scan of that package's files;
- a helper that tests which load models use;
- a tower test, which belongs to AX.

Those stay. Moving them needs the production code to move first: `actgroup`/`ks_a1`'s helpers feed `kso1c_kernels_test.go`,
which reads a real GGUF through decoder, and `gemm_mma` uses the resident's launch constants.

- **Gate, as batches 1–3.**
  - Every moved `Test`/`Benchmark` is listed before and found after the move, so no test is deleted.
  - Each moved test runs by name with `-v` before (in `./cuda/` or `./gpu/`, tags `cuda goinfer_testhooks` or
    `gpu goinfer_testhooks`) and after (in `./cuda/kernels/` under `cuda`, or `./gpu/internal/kernels/` under
    `gpu goinfer_testhooks`). Each outcome (PASS, FAIL, or SKIP with the same reason) must be the same.
  - The cuda and gpu modules build and vet under each tag set CI uses.
  - `go list -deps -test` of each destination contains no `goinfer/decoder`.
  - The ratchet falls by what moved: cuda reach −1; gpu direct −1 (the gumbel test) and reach −3.
- **Void:** a run while a timed holder has the lock, or a card with another process's context that fails a test on memory.
- **Stopping rule:** one before-run and one after-run; a mismatch stops the batch and is reported.
- **Cost basis:** seconds on nobara's RTX 2070 SUPER (Vulkan for gpu).
- **Read (2026-10-27).** Every moved test's outcome is the same before and after: 5 PASS, and `TestGemvRNBandwidth`
  skipped on both sides for its heavy gate.
  - Ratchet: cuda reach 315 → 314; gpu direct 87 → 86, reach 99 → 96.
  - Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "Batch 4".
  - With this batch, SB1 step 1's test-only moves are done on cuda and gpu. What remains needs production code to move
    first, and metal's batch is the Mac's.

#### SB1 step 1 — metal batch 5, registered 2026-10-28 10:05 PDT before the code (mac-ax01; Francis's go of 09:47)

The Mac is on battery: correctness tests only, no timed run, `go test -p 2` (another session runs Metal tests). A scan of
`metal/`'s 428 test files (`go/ast`, each file's references to package `metal`'s top-level declarations and to `decoder`):

- **245 import `decoder`.** 231 call `decoder.Load` (they load a model). Of the 15 that do not, none moves test-only:
  - `deltanet_test.go` and `qgate_kernels_test.go` are kernel tests whose CPU references are `decoder`'s own production code
    (`gatedDeltaNetStep`, `splitQGate`, `qGateContext`, through `decoder/testhooks.go`). A copy of a reference is a second
    unvalidated implementation (`docs/parity-coverage-policy.md`), so they wait for that code to leave `decoder`, as batch 4
    found for cuda.
  - The rest test package `metal`'s own resident or backend code, or reach a model through another file's helper.
- **88 do not import `decoder`**, and reach it only by being in package `metal`. #298 moved 62 such files. Of these 88, the
  ones whose every package-`metal` reference is a name `metal/kernels` declares (its own, or `moved_helpers_test.go`'s
  aliases): four kernel tests, `moe_test.go`, `prefill_gemm_s2_test.go`, `profile_test.go`, `sa_qv_fusion_test.go`.
  The others are helpers that tests which stay use (`heavytest`, `modelsdir`, `writable`, `voidrule`, `ksc1_old`), tests that
  need package `metal`'s backend or tower registration (`embeddinggemma2_real`, `tower_identity`), or tests of package
  `metal`'s production code.

Batch 5 moves those four to `metal/kernels`, if they compile there unchanged apart from the package clause:

- **Gate, as batches 1–4.** Every moved `Test`/`Benchmark` is listed before and found after the move. Each moved test runs by
  name with `-v`, before in `./metal/` and after in `./metal/kernels/` (tags `goinfer_testhooks`, `-p 2`), with the same
  outcome (PASS, FAIL, or SKIP with the same reason). The metal module builds and vets with and without
  `goinfer_testhooks`. `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`. The ratchet's metal reach falls by
  the files moved (333 → 329; direct stays 245), lowered in the same commit; the SB2 census exemptions are unchanged.
- **Void:** a run under 4 GB available (Metal tests fail on memory for reasons not in the test), or a battery under 25%
  (the work stops and pushes what it has).
- **Stopping rule:** one before-run and one after-run; a mismatch stops the batch and is reported. A file that needs more
  than its package clause changed to compile in `metal/kernels` is left in `metal/` and listed.
- **Cost basis:** seconds of device time (batch 1–4's before-run took 14.5 s for 67 tests).
- **Read (2026-10-28, on battery).** Three files moved (`moe_test.go`, `profile_test.go`, `sa_qv_fusion_test.go`): 7 PASS
  before and after. `prefill_gemm_s2_test.go` stays by the stopping rule: `prefill_decomp_test.go`, which stays, compiles its
  429-line MSL source. `clampI` is copied next to `gemma_test.go`. Ratchet: metal reach 333 → 330, direct 245. A first
  before-run at 3.996 GB reclaimable is void and kept. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`,
  "metal batch 5". With this batch, metal's test-only moves are done too: what remains needs `decoder`'s production code
  (the DeltaNet and q-gate references) to leave `decoder` first.

#### SB1 step 1 — metal batch 6, registered 2026-10-28 17:47 PDT before the code (agy-sb1)

The Mac is on battery: correctness tests only, no timed run, `go test -p 2`. Batch 5 left two kernel test files in `metal/`
whose only `decoder` dependency is CPU reference functions reached through `decoder/testhooks.go`:

- `metal/deltanet_test.go` (`TestDeltaNetKernels_cpuParity`, `TestDeltaNorm_zeroHead`, `TestDeltaNetKernels_mutations`) calls
  `gatedDeltaNetStep` and `l2normScaled` (with `NewDeltaNetForTest` and `SetDeltaCapHook`).
- `metal/qgate_kernels_test.go` (`TestQGateKernels_cpuParity`, `TestQGateKernels_mutations`) calls `splitQGate` and
  `qGateContext`.

Copying a reference is a second unvalidated implementation (`docs/parity-coverage-policy.md`). Batch 6 moves those CPU
reference functions out of `decoder` into `internal/hybridattn` (holding the hybrid-attention layers' operations shared
by `decoder` and the kernel tests, importing only the standard library and `aikit/linalg`), with `decoder` calling
`internal/hybridattn` so there is still one implementation and `decoder/testhooks.go` forwarding its `*ForTest` helpers for
`cuda` and `gpu`. Then `metal/deltanet_test.go` and `metal/qgate_kernels_test.go` move to `metal/kernels` against
`internal/hybridattn`, dropping their `decoder` import and `goinfer_testhooks` build constraint:

- **Gate, as batches 1–5.** Every moved `Test` is listed before and found after the move (`TestDeltaNetKernels_cpuParity`,
  `TestDeltaNorm_zeroHead`, `TestDeltaNetKernels_mutations`, `TestQGateKernels_cpuParity`, `TestQGateKernels_mutations`).
  Each moved test runs by name with `-v` (`-p 2`, under the shared timing lock), before in `./metal/` (`-tags
  goinfer_testhooks`) and after in `./metal/kernels/` (both untagged and `-tags goinfer_testhooks`), with the same outcome
  (PASS, FAIL, or SKIP with the same reason). `decoder`'s outputs are bit-identical on its existing tests covering these
  paths (`TestGatedDeltaNet_parity`, `TestDeltaNetCoreN_bitIdentical`, and the tiny-fixture forward parity tests for
  `qwen3_5`, `qwen3_5_moe`, `qwen3_next`, and `olmo_hybrid`). Both modules build, vet, and pass `staticcheck` with and
  without `goinfer_testhooks`. `go list -deps -test ./metal/kernels/` names no `goinfer/decoder`. The ratchet's metal
  counts fall by the two files moved (`Direct: 245 → 243`, `Reach: 330 → 328`), lowered in the same commit; `imports.json`
  and `docs/ARCHITECTURE.md` record the new `internal/hybridattn` arrows; the SB2 census exemptions are unchanged.
- **Void:** a run under 4 GB reclaimable (`cmd/hostmem -reclaimable`), a battery under 25% (the work stops and pushes what
  it has), or a timed holder on the timing lock.
- **Stopping rule:** one before-run and one after-run; a mismatch, a non-bit-identical `decoder` output, or a need for a
  public API change stops the batch (`STATE: decision`).
- **Cost basis:** seconds of device and CPU time.
- **Read (2026-10-28, agy-sb1).** `internal/hybridattn/deltanet.go` holds the Gated DeltaNet and q-gate CPU reference
  functions (`GatedDeltaNetStep`, `DeltaNetCore`, `DeltaNetRecurrence`, `L2NormScaledInto`, `L2NormScaled`,
  `NegExpAFromLog`, `SetDeltaCapHook`, `HasDeltaCapHook`, `SplitQGate`, `SplitQGateAlloc`, `QGateContext`), with `decoder`
  and `decoder/testhooks.go` calling them so there is one implementation. `decoder`'s 6 tests covering these paths
  (`TestGatedDeltaNet_parity`, `TestDeltaNetCoreN_bitIdentical`, `TestQwen35_forwardParity`, `TestQwen3_5_textParity`,
  `TestQwen3Next_textParity`, `TestOlmoHybrid_forwardParity`) are bit-identical. `deltanet_test.go` and
  `qgate_kernels_test.go` moved to `metal/kernels`: 5 PASS before in `./metal` (`batch6-before.log`, 4.52 GB reclaimable)
  and 5 PASS after in `./metal/kernels`, both with `-tags goinfer_testhooks` (`batch6-after.log`, 4.03 GB reclaimable) and
  untagged (`batch6-after-untagged.log`, 5.62 GB reclaimable). `uploadF32` and `cosF32` are copied to
  `metal/mamba_kernels_test.go`. Ratchet: metal direct 245 → 243, reach 330 → 328. Record:
  `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "metal batch 6".

#### SB1 step 1 — metal batch 7, registered 2026-10-29 before the code (mac-ax01)

The Mac is on AC; correctness tests only, no timed run, `go test -p 2` under the shared timing lock. `metal/mamba_kernels_test.go`
reaches `decoder` for two things: the Mamba-2 CPU reference (`decoder/mamba2.go:mamba2Step`'s conv, selective scan and gated
norm, and `gatedRMSNormGrouped` through `GatedRMSNormGroupedForTest`), and, for `TestMambaKernels_cpuParity` and
`TestMambaKernels_plantsAreRed`, a tiny-fixture forward through `decoder.Load` whose real per-token inputs it captures.
Batch 7 moves the reference, the way batch 6 did:

- **What moves:** the mixer between the two projections (the causal conv, the split, the selective-scan recurrence, the gate
  and the grouped gated RMSNorm), with its state and geometry, into `internal/mamba2` (standard library, `aikit/linalg` and
  `internal/hybridattn`'s scalar activations only). `decoder`'s `mamba2Step` keeps the projections, the int8 seams and the
  capture hook, and calls `internal/mamba2` for the rest, so there is one implementation; `decoder/testhooks.go` keeps its
  exports for `cuda` and `gpu`.
- **Tests that move:** `TestMambaGNorm_groups` (against `internal/mamba2`'s grouped norm) and `TestActQuant_relu2` (no
  `decoder` use) to `metal/kernels`. The two chain tests stay in `metal/`: their inputs are a real forward's, and only
  `decoder` produces those. Moving them needs those inputs from elsewhere (a committed capture, or a decoder-free loader), a
  change to what the gate measures, which is not this batch's to make.
- **Gate.** (1) `decoder` is bit-identical: a scratch test, not committed, hashes `mamba2Seq`'s outputs on seeded random
  weights (both fixtures' geometries, `ssmForceF32` off and on) and the full-forward logits of `nemotron-tiny` and
  `granite-tiny` over a seeded 24-token prompt, before and after; every hash equal. `TestMamba2_parity`,
  `TestGranite_textParity`, `TestNemotron_textParity`, the `TestNemotron3NanoMoE_*` tests, `TestB27_*` and
  `TestTruncateTo_resetsRecurrent` pass before and after. (2) The four tests in `metal/mamba_kernels_test.go` run by name with
  `-v`, before in `./metal/` (`-tags goinfer_testhooks`), after in `./metal/` for the two that stay and `./metal/kernels/`
  (untagged and tagged) for the two that move, with the same outcome and the same logged figures. (3) Both modules build,
  vet and pass `staticcheck` with and without `goinfer_testhooks`; `go list -deps -test ./metal/kernels/` names no
  `goinfer/decoder`; `internal/layout`'s tests pass with `imports.json` and `docs/ARCHITECTURE.md` recording the new arrows.
  The ratchet's metal counts are unchanged (no file leaves `decoder`'s reach); the SB2 census exemptions are unchanged.
- **Void:** a run under 4 GB reclaimable (`cmd/hostmem -reclaimable`), or a timed holder on the timing lock.
- **Stopping rule:** one before-run and one after-run; a hash mismatch, a changed test outcome or figure, or a need for a
  public API change stops the batch (`STATE: decision`).
- **Cost basis:** seconds of CPU and device time.
- **Read (2026-10-29, mac-ax01).** `internal/mamba2` holds the mixer (`Params`, `Weights`, `State`, `NewState`, `Mix`,
  `GatedRMSNormGrouped`); `decoder/mamba2.go:mamba2Step` calls `mamba2.Mix`, so there is one implementation. Gate 1: all 16
  scratch hashes equal before and after, and `decoder`'s mamba tests pass with the same figures. Gate 2: 4 PASS before in
  `./metal/`, the same 4 PASS after (2 in `./metal/`, 2 in `./metal/kernels/` tagged and untagged), every figure equal. Three
  cells started under 4 GB reclaimable were void and re-run (listed in the record). Build, vet, pinned `staticcheck` and
  `internal/layout` green. Ratchet unchanged (metal direct 243, reach 328). Record:
  `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "metal batch 7".
- **After batch 7 (Francis, 2026-10-29 08:22): no further test-only batch on metal.** Every `decoder` importer left among
  `metal`'s tests (243 direct, 328 reach) either loads a model or tests `metal`'s own production code (`decoder.Knobs`,
  `AutoBackend`, `KVCopyDecline`, and the like), so moving tests can no longer lower metal's ratchet; lowering it further means
  changing what a test measures (the Mamba chain tests' inputs from a committed capture, say), which is a decision of its own.
  The activation-group helpers are cuda's (`actgroup_kernel` and `ks_a1`, which feed `kso1c_kernels_test.go`), not metal's;
  that move goes to nobara-sb01 after #425.

#### SB1 step 1 — cuda batch 5, registered 2026-10-29 before the code (nobara-sb01)

Metal batch 6 moved the Gated DeltaNet and q-gate CPU references out of `decoder` (into
`internal/hybridattn`). In `cuda/` and `gpu/`, only `cuda/deltanet_test.go` and `gpu/deltanet_test.go` still reach
them through `decoder/testhooks.go`, and no cuda test uses the q-gate references. This batch is the cuda one:
`cuda/deltanet_test.go` (`TestDeltaNetKernels_cpuParity`, `TestDeltaNorm_zeroHead`) moves to `cuda/kernels`, calling the
references directly (`NewDeltaNet`, `SetDeltaCapHook`, `DeltaNetStep`, `L2NormScaled` for `decoder`'s `NewDeltaNetForTest`,
`SetDeltaCapHook`, `DeltaNetStepForTest(nil, …)`, `L2NormScaledForTest`), and drops its `decoder` import and its
`goinfer_testhooks` constraint. Its PTX is already read from `../testdata` by `cuda/kernels`. `cosF32`, declared there and
used by tests that stay in `cuda/`, is copied to `cuda/mamba_kernels_test.go` (same build constraint, as metal did).
`decoder`'s `*ForTest` hooks stay for `gpu/deltanet_test.go`, the gpu module's batch.

- **Gate, as batches 1–4.**
  - Both moved `Test`s are listed before and found after the move.
  - Each runs by name with `-v`: before in `./cuda/` (`-tags 'cuda goinfer_testhooks'`), after in `./cuda/kernels/` under
    `-tags cuda` and under `-tags 'cuda goinfer_testhooks'`. Each outcome (PASS, FAIL, or SKIP with the same reason) must be
    the same, and the per-stage worst cosines are recorded on both sides (the projections now go through the reference's own
    f32 matvec rather than `decoder`'s CPU backend; the kernels consume the captured intermediates, so the stage figures
    are expected to agree closely, and any stage that falls under the test's own bar is a mismatch).
  - The cuda module builds, vets and passes `staticcheck` under no tag, `cuda`, and `cuda goinfer_testhooks`.
  - `go list -deps -test -tags cuda ./cuda/kernels/` names no `goinfer/decoder`.
  - The ratchet's cuda counts fall by the one file (`Direct: 246 → 245`, `Reach: 314 → 313`), lowered in the same commit;
    `imports.json` and `docs/ARCHITECTURE.md` record the new `cuda/kernels` arrow; the SB2 census exemptions are unchanged.
- **Void:** a run while a timed holder has the timing lock, or with another process's CUDA context on the card besides the
  desktop compositor's (`nvidia-smi --query-compute-apps`, recorded before each run).
- **Stopping rule:** one before-run and one after-run per tag set; a mismatch stops the batch and is reported.
- **Cost basis:** seconds on nobara's RTX 2070 SUPER (batch 4's runs).
- **Read (2026-10-29, nobara-sb01).** Both moved tests are listed before and after. 2 PASS before in `./cuda/`, and 2 PASS
  after in `./cuda/kernels/` under `cuda` and under `cuda goinfer_testhooks` (against `internal/hybridattn`), each on a card
  holding only the desktop compositor. The per-stage worst figures are equal on every run (`delta_conv` `2.37e-06`,
  `delta_gates` `2.42e-07`, `delta_rule` `7.07e-06`, `delta_gnorm` `2.22e-05`, `delta_norm` clean). The first pair of
  after-runs started with another session's `cuda.test` on the card: void, re-run, and kept in the record. Build, vet,
  staticcheck and `internal/layout` green; `cuda/kernels`' test build reaches no `decoder`. Ratchet: cuda 246/314 → 245/313.
  Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "cuda batch 5".

#### SB1 step 1 — cuda batch 6, the activation-group kernel tests, registered 2026-10-29 before the code (nobara-sb01; Francis's option B, 08:22)

The item as relayed was to move "the activation-group helpers" out of `decoder`. Read against the tree, none are there:
`cuda/actgroup_kernel_test.go` and `cuda/ks_a1_test.go` import no `decoder`, and their helpers (`actGroupHarness`,
`quantG32Go`, `packActI8`, `outlierVec`, `q4kHostRows`, the `ksA1*` Q5_K/Q6_K row dequantizers) are test code in those
files. What holds the three activation-group test files in package `cuda` is package `cuda`'s own production code: the device
wrapper (`CreateSystemDefaultDevice`), the embedded `actGroupPTX`, `f32tof16`/`packI8`/`permuteFast` (one-line forwards to
`cuda/internal/wpack`), and, for `cuda/kso1c_kernels_test.go`, the launch constant `kquantGemvRows` (`resident.go`, which must
equal `KQ_ROWS` in `actgroup.cu`). The only `decoder` use among them is `TestKSO1C_referenceMatchesAikit`'s
`decoder.OpenGGUFMmap`, which is aikit's `embed.OpenGGUFSplitMmap` over the file's shards. So the batch is:

- **Production:** `kquantGemvRows` moves to `cuda/internal/launchgeom` as `KQuantGemvRows` (as the `gemv_w4a8_rn` constants
  did), and `resident.go` reads it there. A new `launchgeom` test holds it to `#define KQ_ROWS` in `actgroup.cu` (no test
  does today); it is shown red once by a planted mismatch. `TestLaunchGeometry_identity` gains a `kquantGemv` hash (the
  constant and the grid over a row ladder), written from the tree before the move and held after it.
- **Tests that move to `cuda/kernels`:** `actgroup_kernel_test.go` (`TestActGroupKernels_quantizers`, `_gemv`, `_gemvQ4K`,
  `_fusedG32BitIdentical`), `ks_a1_test.go` (`TestKSA1_gemvTiming`) and `kso1c_kernels_test.go` (`TestKSO1C_nativeKernels`,
  `_fusedEqualsUnfused`, `_untouchedKernelsBitIdentical`, `_referenceMatchesAikit`). They take the primary context through
  `gc` as the other kernel tests do, read `actgroup.ptx` and `ks_a1_proto.ptx` from `../testdata`, pack with `wpack`, read
  `launchgeom.KQuantGemvRows`, and open the GGUF through aikit's `embed` directly. `kso1cPlanes` and `kso1cModels`, also used
  by `kso1c_resident_test.go`, which loads models through `decoder` and stays, are copied (a layout twin and a file list);
  `kso1cLoad1` and `ms` are copied too.
- **Gate.**
  - All nine `Test`s are listed before and found after the move.
  - Each runs by name with `-v`: before in `./cuda/` (`-tags 'cuda goinfer_testhooks'`), after in `./cuda/kernels/` under
    `-tags cuda` and `-tags 'cuda goinfer_testhooks'`, with `GOINFER_KSO1C_OLD_ACTGROUP_PTX` set to `cuda/testdata/actgroup.ptx`
    on both sides so the untouched-kernels test runs rather than skips. Each outcome (PASS, FAIL, or SKIP with the same reason)
    must be the same, and the logged figures equal. `TestKSA1_gemvTiming` is a timed reading behind `GOINFER_KSA1_TIMING` and
    is expected to skip on both sides; it is not run timed here.
  - `TestLaunchGeometry_identity` passes before (after writing the new hash) and after; the `launchgeom` define test passes
    and is red once under the plant.
  - The cuda module builds, vets and passes `staticcheck` under no tag, `cuda`, and `cuda goinfer_testhooks`;
    `go list -deps -test -tags cuda ./cuda/kernels/` names no `goinfer/decoder`.
  - The ratchet's cuda counts fall by the three files (`Direct` −1 for `kso1c_kernels_test.go`, `Reach` −3), lowered in the
    same commit; `imports.json` and `docs/ARCHITECTURE.md` record any new arrow; the SB2 census exemptions are unchanged.
- **Void:** a run while a timed holder has the timing lock, or with another process's CUDA context on the card besides the
  desktop compositor's (`nvidia-smi --query-compute-apps`, recorded before each run).
- **Stopping rule:** one before-run and one after-run per tag set; a mismatch stops the batch and is reported.
- **Cost basis:** seconds to a minute on nobara's RTX 2070 SUPER; the GGUF reference test reads three rows of every Q5_K and
  Q6_K tensor of two local files (`~/models`).
- **Read (2026-10-29, nobara-sb01).** All nine tests are listed before and after, with the same outcome and the same logged
  lines on all three runs, each on a clear card: 8 PASS, and `TestKSA1_gemvTiming` skipped for its timing flag.
  `TestKQuantGemv_matchesCuDefine` is red under its plant and green after. `TestLaunchGeometry_identity` holds the
  `kquantGemv` hash written before the move. Build, vet, staticcheck and `internal/layout` are green, and `cuda/kernels`
  reaches no `decoder`. Ratchet: cuda direct −1, reach −3 (245/313 → 244/310 after cuda batch 5). Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`,
  "cuda batch 6".

#### SB1 step 1 — cuda batch 7, `gemm_mma`'s kernel test and the f16 KV converter's test, registered 2026-10-29 before the code (nobara-sb01; coordinator 11:05)

Two of the scan's candidates, each held in package `cuda` by one production edge:

- `cuda/gemm_mma_test.go` has three tests. `TestGemmMMA_vsExact` is a kernel test (gemm_w4a8_mma against gemv_w4a8_rn on
  the real prefill shapes); it needs `gemmMMABM`, `gemmMMABN`, `gemmMMAThreads` and `gemmMMAShmem` (`prefill.go`, which must
  equal `GBM`, `GBN`, `GWARPS*32`, `GKSTEP` and `GAPAD` in `gemm_w4a8_mma.cu`) and `packWeight`/`packI8` (forwards to
  `wpack`). The other two, `TestGemmMMA_selectorKeepsExactBelowFloor` and `TestFastPrefillFloor_selectorHonoursIt`, test
  `cudaResident`'s selectors and stay, so the file stays too.
- `cuda/kvhalf_test.go`'s one test, `TestF32ToF16RNE_ties`, is a host test of `f32ToF16RNE`, a one-line forward to
  `wpack.F32ToF16RNE`. Its home is `cuda/internal/wpack`, beside the function it tests.

What changes:

- **Production:** the gemm_w4a8_mma tile constants and its shared-memory size move to `cuda/internal/launchgeom`
  (`GemmMMABM`, `GemmMMABN`, `GemmMMAThreads`, `GemmMMAKStep`, `GemmMMAAPad`, `GemmMMAShmem`, and `GemmMMAGrid` for the
  launch grid), and `prefill.go` reads them there. `gemmMMAMinRows` is the selector's floor, not the kernel's geometry, and
  stays. A new `launchgeom` test holds the five to `gemm_w4a8_mma.cu`'s defines, is listed in `scripts/touched_scans.py`, and
  is shown red once under a plant. `TestLaunchGeometry_identity` gains a `gemmMMA` hash (the constants, the shared-memory
  size and the grid over a shape ladder), written from the tree before the move and held after it.
- **Tests:** `TestGemmMMA_vsExact` (with its pre-registered bound `gemmMMAMaxRelDelta`) moves to `cuda/kernels/gemm_mma_test.go`,
  using `cuda/kernels`' `rnCfg` and `itoa`, `wpack.Pack` and `wpack.PackI8`. `TestF32ToF16RNE_ties` moves to
  `cuda/internal/wpack`, calling `F32ToF16RNE`.
- **Gate.**
  - Both moved tests are listed before and found after.
  - `TestGemmMMA_vsExact` runs by name with `-v`: before in `./cuda/` (`-tags cuda`), after in `./cuda/kernels/` (`-tags cuda`),
    with the same outcome and the same per-shape, per-M logged deltas. `TestF32ToF16RNE_ties`: before in `./cuda/` (`-tags
    cuda`), after in `./cuda/internal/wpack/` untagged, the same outcome.
  - `TestLaunchGeometry_identity` passes before (after writing the new hash) and after; the define test passes and is red
    once under the plant. The two selector tests still pass in `./cuda/`.
  - The cuda module builds, vets and passes `staticcheck` under no tag, `cuda`, and `cuda goinfer_testhooks`; the
    destinations' test builds name no `goinfer/decoder`; `scripts/test_touched.py` passes.
  - The ratchet's cuda counts fall by the one file that leaves (`kvhalf_test.go`: `Reach` −1; `Direct` unchanged, since it
    never imported `decoder`). `gemm_mma_test.go` stays, so it does not count.
- **Void:** a GPU run while a timed holder has the timing lock, or with another process's CUDA context on the card besides
  the desktop compositor's (recorded before each run).
- **Stopping rule:** one before-run and one after-run; a mismatch stops the batch and is reported.
- **Cost basis:** seconds on nobara's RTX 2070 SUPER for `TestGemmMMA_vsExact` (its largest shape is the 7B's 18944x3584);
  seconds on the CPU for the rest.
- **Read (2026-10-29, nobara-sb01).** Both moved tests and the two that stay have the same outcome before and after, each GPU
  run on a clear card: `TestGemmMMA_vsExact`'s 24 per-shape, per-M lines are identical (worst `1.107e-06` against `1e-5`).
  The define test is red under a `GKSTEP` plant and under a 129-thread plant; `TestLaunchGeometry_identity` holds the
  `gemmMMA` hash. Build, vet, staticcheck, `scripts/test_touched.py` and `internal/layout` green. Ratchet: cuda 244/310 →
  244/309. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "cuda batch 7".

#### SB1 step 1 — gpu batch 5, the DeltaNet kernel test against `internal/hybridattn`, registered 2026-10-29 before the code (nobara-sb01; coordinator 11:05)

`gpu/deltanet_test.go` (`TestDeltaRule_cpuParity`, `TestDeltaNorm_cpuParity`) is the gpu twin of cuda batch 5: its only
`decoder` use is the Gated DeltaNet CPU reference through `decoder/testhooks.go` (`NewDeltaNetForTest`, `SetDeltaCapHook`,
`DeltaNetStepForTest(nil, …)`, `L2NormScaledForTest`), and everything else it touches (the context, the DeltaNet pipelines
and layouts, `newOrSkipHW`, `cosSim`) lives in `gpu/internal/kernels`. It moves there, calling `internal/hybridattn`'s
`NewDeltaNet`, `SetDeltaCapHook`, `DeltaNetStep` and `L2NormScaled` directly, and drops its `decoder` import and its
`goinfer_testhooks` constraint. No production code moves. `decoder`'s DeltaNet `*ForTest` hooks then have no caller outside
`decoder`; removing them is left to a later change.

- **Gate, as batches 1–4.**
  - Both moved `Test`s are listed before and found after.
  - Each runs by name with `-v`: before in `./gpu/` (`-tags 'gpu goinfer_testhooks'`), after in `./gpu/internal/kernels/`
    under `-tags gpu` and `-tags 'gpu goinfer_testhooks'`, on nobara's RTX 2070 SUPER through Vulkan, with the same outcome
    and the same per-stage figures.
  - The gpu module builds, vets and passes `staticcheck` under no tag, `gpu`, and `gpu goinfer_testhooks`;
    `go list -deps -test -tags gpu ./gpu/internal/kernels/` names no `goinfer/decoder`.
  - The ratchet's gpu counts fall by the one file (`Direct` −1, `Reach` −1), lowered in the same commit; `imports.json` and
    `docs/ARCHITECTURE.md` record the new `gpu/internal/kernels` test arrow to `internal/hybridattn`.
- **Void:** a run while a timed holder has the timing lock, or with another process's GPU context on the card besides the
  desktop compositor's (recorded before each run).
- **Stopping rule:** one before-run and one after-run per tag set; a mismatch stops the batch and is reported.
- **Cost basis:** seconds on the card.
- **Read (2026-10-29, nobara-sb01).** Both moved tests: 2 PASS before in `./gpu/` and 2 PASS after in `./gpu/internal/kernels/`
  under `gpu` and under `gpu goinfer_testhooks`, each on a clear card, with every logged figure equal (worst `deltaRule`
  `6.03e-06`, `deltaGNorm` `1.32e-05`). Build, vet, staticcheck and `internal/layout` green; the destination reaches no
  `decoder`. Ratchet: gpu 86/96 → 85/95. Record: `docs/measurements/sb1-kernel-tests-2026-10-26/README.md`, "gpu batch 5".

#### SB1 step 1 — cuda tests that stay in package `cuda`, decided (Francis 15:39 on 2026-10-29)

A scan for the next kernel-test batch skips these:

- `cuda/moe_route_demand_test.go` (`TestMoERouteDemandThreshold`, `TestMoERouteDemandThresholdChild`). It balloons the card
  in child processes, bisects the free VRAM at which `moe_route`'s first launch fails, and asserts that `slotMarginBytes`
  covers that demand. That makes it a test of `cuda`'s fit policy, not a kernel test. Moving it would need
  `slotMarginBytes`, a fit-guard threshold that `TestAllocFloor` and `slotcap_test.go` also check, to leave `cuda` or be
  copied, and it would split the drain-marker ordering (`drainsDevice`) between two packages. Francis chose to leave it.

### SB2 — admission and routing

- **Today:** `decoder/admit.go:Admit`, pure over `AdmitFacts`: the backend and precision checks, the runner shape
  (`admitRunner`), the activation-group gate, then `decoder/features.go:residentGateReasons` against `ResidentBackendFeatures`;
  then the backend's own build through `DeclineResident`; the option-path admission grid of 2026-10-08 (`docs/completed/task-option-path-admission-2026-10.md`).
  The owner's note of 2026-10-21 in the forward-architecture doc: 20 of the 71 registration and bound findings counted on
  2026-10-08 were one family that one path did not know about, six of them Critical.
- **Contract:** `Admit(program, request shape, options, backend op set, plan) → Route or Decline`, pure.
- **Fast check:** the whole table, every family × backend × shape × option, in milliseconds, as golden output;
  `docs/capability-matrix.json` and `docs/hardware-matrix.md` generated from it, so they cannot drift from the code.
- **Stops re-running:** finding out in a real run which path a request took. M.

#### SB2 — the admission and routing census, registered 2026-10-27 before the code (nobara-sb01)

Francis's decision (11:08): census first; the pure function must reproduce it exactly; the matrices generated from it must
equal today's committed ones byte for byte; memory questions go through `internal/mem`; no change to what is admitted or
routed (one that appears stops the work with DECISION NEEDED).

Where today's decisions live (surveyed 2026-10-27): the decoder's load path (backend choice, the fit guards, which MM1's
`TestDecisionCensus` already records, the resident gates, KV resolution, the path reports, CPU batching and speculative
eligibility); each backend's `BuildResident` (CUDA alone has about 60 declines, several reading free VRAM or device
attributes); and request-time routing. The census is layered to match, each layer a golden in the module that owns the
decisions:

- **Layer D (decoder; this step, tier 0).** `decoder`'s `TestAdmissionCensus` records, deterministically and without a
  device:
  - **model-free, per registered family (`representativeConfig`, as the hardware matrix does) × webgpu, cuda, metal:**
    the runner-shape decline and every resident gate, for the default activations and for per-32 activations;
  - **per tiny fixture × option cell × backend:** the outcome of a real `Load` with that backend name, a fake resident
    backend standing in for cuda, metal and webgpu that builds whatever it is asked to, so the record is the decoder's
    own decision: the load's error or success, whether the resident build was reached (admitted) or the decline's
    reason, `DecodePath` and the KV precision reports; on cpu also `PrefillPath`, `canBatchN(2)`, the CPU-batch model
    eligibility, `CPUConcurrentSafe` and `specRollbackSafe`. Option cells: Quant int4, int8int8 and f32, each with
    ActQuantGroup 0 and 32; int4 with the `GOINFER_SSM_RESIDENT` knob set; int4 with KVPrecision f16; int4 with KVQuant
    i8. The startup self-test is off for these loads (it runs a backend's kernels on its device: layer B);
  - **backend choice:** `autoBackend` for each `testdata/machines` profile (its own goarch and free bytes) × each
    linked set (cpu; cpu+cuda; cpu+metal; cpu+cuda+metal; cpu+webgpu).
  The golden is the full text, gzipped, one per architecture (`testdata/sb2_admission_census.<goarch>.txt.gz`: the
  CPU's int4 layout is a repack on arm64, a decision that differs by GOARCH), so a diff names the cell; the run is
  repeated in-process and must agree with itself; an architecture with no golden fails and prints its census for
  recording. No reading of the machine enters a line: backend choice reads the profile's goarch and free bytes, and the
  loads read a fixed fake probe (64 GiB, 48 GiB available), so the fit guards proceed.
- **Layers B (later steps, in the backend modules):** each backend's `BuildResident` decisions and device routes, with
  every live read replaced by a profile input through the backend's existing seams: CUDA and WebGPU on nobara, Metal on
  the Mac. Registered here; each gets its own registration of cells before its code.
- **Then the pure function.** `Admit` must reproduce layer D with the census unchanged, and `docs/hardware-matrix.md` and
  `docs/capability-matrix.json` generated from it must equal the committed files byte for byte. **Void:** none
  (deterministic). **Stopping rule:** any change to a recorded decision stops with DECISION NEEDED. **Cost basis:**
  measured: about 9 s for the census built twice (2,711 lines).
- **Layer D landed (2026-10-27).** `decoder`'s `TestAdmissionCensus`, goldens
  `testdata/sb2_admission_census.<goarch>.txt.gz` (amd64 and arm64): 156 model-free lines (52 registered families ×
  webgpu, cuda, metal), 35 backend-choice lines (7 machine profiles × 5 linked sets), and 2,520 load lines (70 committed
  fixtures × 9 option cells × cpu, cuda, metal, webgpu; the 15 that are not decoder checkpoints, towers and encoders,
  record their load error). Only fixtures whose weights git tracks are read, so a checkout with local-only fixtures
  censuses what CI does. Shown red: dropping the q4k condition from the ActQuantGroup gate changes 145 cells (reverted).
  The arm64 golden was recorded from CI's own run of the census at `343630740` (the test prints an unrecorded
  architecture's census and fails): root-linux-arm64, root-darwin and root-windows-arm64 printed the same 2,711 lines,
  and they differ from amd64's in 212 cells, every one only in the CPU's int4 layout note (a repack on arm64).
- **Step 1 landed (2026-10-27): `Admit`.** `decoder/admit.go:Admit` is the decoder's resident admission as a pure function
  of `AdmitFacts` (the arch, the backend's name and whether it can build residents, the quant, -backend auto, the
  activation group, whether the projections are int4, the SSM knob): the residency, precision, runner-shape,
  activation-group and capability gates in the order the load applied them. `Model.residencyGate` gathers the facts and
  calls it; `DecodeRunnerEligible` and `ResidentEligible` read its pieces, and the hardware matrix prints its runner shape
  and capability gates. The census is unchanged on both architectures' goldens, and both matrices regenerate byte for
  byte (`-update` leaves no diff).

#### SB2 step 2 — the backend layers, registered 2026-10-27 before the code (nobara-sb01)

Each backend's own `BuildResident` decides after the decoder admits. Its static gates (shapes, weight kinds, activation
groups, KV layouts) read the model alone; its memory and device decisions read free VRAM, host RAM or device attributes.
The static gates are censused now, on the same fixtures and option cells as layer D, with the machine out of the record:

- **Layer B-cuda (`cuda` module, `-tags 'cuda goinfer_testhooks'`, no device; without `goinfer_testhooks` the file is not built).** `TestBackendAdmissionCensus_cuda`: per committed fixture ×
  layer D's 9 option cells, a `Load` under a fake backend named `cuda` (the decoder's admission is the real one); where
  the decoder admits, the fake runs CUDA's own build in plan-only mode (`cudaBackend.buildResident(m, true)`, which runs
  every static decline and returns before it allocates) and records the decline or the planned allocation slack, then
  declines so nothing is uploaded. Golden `cuda/testdata/sb2_cuda_census.<goarch>.txt.gz`. Runs on CI's deviceless cuda
  job.
- **Layer B-webgpu (`gpu` module, `-tags 'gpu goinfer_testhooks'`, no device; likewise).** WebGPU's static gates move, in their order, into one function
  `BuildResident` calls first (a refactor: same checks, same reasons); `TestBackendAdmissionCensus_webgpu` records it the
  same way. Golden `gpu/testdata/sb2_webgpu_census.<goarch>.txt.gz`.
- **Layer B-metal (the Mac).** Commands in `/tmp/sb2-metal.md`: `residentPreBuild`'s static gates and its memory decisions
  with `metalLiveAvailable` and `metalWorkingSetCap` set from each `testdata/machines` profile.
- **Memory and device decisions on cuda and webgpu.** The context and slot plan is `Model.Plan`, which MM1's census already
  records for cuda and webgpu across the machine zoo. The build's own reads (CUDA's `checkWeightsFit` and `checkKVFits`,
  expert slots, prefill width, SM shape; WebGPU's slot probe) come after allocation and are the next layer: each has a pure
  helper over a free-memory figure, to be fed the profiles' readings.
- **Gate.** Each layer is deterministic (built twice, equal), shown red by a planted change, and changes no decision: the
  WebGPU refactor must leave layer D unchanged and its own census equal to the same census run on the code before it.
  **Cost basis:** measured on the first run; tiny loads, seconds.

**Step 2 read (2026-10-27, nobara-sb01).** Record: `docs/measurements/sb2-backend-census-2026-10-27/README.md`.

- **B-cuda.** 412 cells reach the build, including every cell layer D builds resident. 279 plan and 133 decline. Both builds agree,
  in 1.7 s for the two. The planted MoE per-32 decline turns exactly its 42 cells red.
- **B-webgpu.** The static gates are now `gpu/residency.go:staticResidentDecline`. 218 cells reach it: 207 pass and 11 take the
  quantized-KV decline. The plant turns exactly those 11 red. The census before the refactor (the real build on nobara's device)
  agrees with the one after it on all 218 cells, and the real build gives byte-identical answers before and after.
- **B-metal.** The Mac session's commands are in `/tmp/sb2-metal.md`.
- **The ratchet.** `internal/layout:TestBackendTests_decoderImportsOnlyFall` exempts the three census files by exact path
  (`decoderRatchetExempt`), and every backend's limits stay where they were. This was Francis's decision of 2026-10-27, option 1
  of three, the others being to raise each limit by one or to offset each file by moving out a kernel test. Under any other
  name the same file turns the ratchet red.
- **Next layer.** The builds' memory and device reads, fed from the machine profiles.

#### SB2 — the build's memory reads, scoped 2026-10-29 (mbp2-sb10), CUDA first

**Inventory at `origin/main` 2026-10-29.** CUDA's non-test code reads free VRAM or a device attribute at 14 sites:
- **Already censused.** The memory probe registered at init (MM1's `TestDecisionCensus`, through `Model.Plan`) and the context
  plan, `cuda/resident.go:resolveCtxCapFitSlack`, which asks that probe.
- **Before any allocation, decisions the census can reach.** `checkWeightsFit` (fixed weights plus a companion attach against
  free less the context margin; pure core `fitsWeightsBudget`) and `checkKVFits` (the one-slot fit, the trim of an unpinned
  context to what one slot holds, and the slot count; pure cores `kvChargedBytes`, `kvChargedCapForBytes`, `kvChargedSlotsFit`).
- **After allocation, read from the device in the state the build left it.** `allocSlots` (expert-cache slots, twice),
  `choosePrefillWidth` (`prefillWidthFor`), the recurrent-checkpoint and windowed-KV grants (`ckptFreeVRAM`,
  `kvGrantFreeVRAM`).
- **Device attributes, not memory.** `readSMShape` (SM count, threads and shared memory per SM) and `graphsTenancySafe`
  (compute mode).
- **Not load decisions.** `ResizeExpertCache` (a running model's reserve) and two diagnostic reads in `launch`.

**What blocks a census today.** Plan-only mode (`buildResident`'s `planOnly`) returns at the weight plan, before the layer
geometry (`planLayerGeometry`) and the context margin (`ctxMarginFor`) that every later decision reads exist, and the code
between that return and the first device read starts the executor and, for a MoE model, opens the device (`openForPlan`).
So the census needs the build's planning, not a stand-in for it: a second stop, after the context plan and before the device
opens, with the free-VRAM figure supplied.

**Steps.** Each its own PR, each changing no decision.
1. **The memory-plan seam.** `buildResident` gains a plan mode that runs through the geometry and the context plan with the
   free-VRAM figure injected and returns the memory decisions as a record (weights fit, one-slot fit, trimmed context, slot
   count), allocating nothing; the decisions themselves move into pure functions over that figure, called by both the real
   build and the census. The real build's answers must not move: shown on nobara's device by the same check B-webgpu used
   (the real build before and after, byte-identical answers), which this box cannot run.
2. **The census.** B-cuda's planning cells × the profiles in `testdata/machines` that carry `cuda_free_bytes` (today
   `nobara-desktop`, `nobara-no-desktop`), golden `cuda/testdata/sb2_cuda_mem_census.<goarch>.txt.gz`, built twice, shown red by
   a planted change to one decision.
3. **After allocation.** The post-allocation reads take the free figure the build itself would leave (the profile's reading less
   what the plan says is on the device by then): a derived figure, labelled so in the census, never presented as a reading.
   Whether that model is good enough to census at all is decided after step 2, on its numbers.
4. **WebGPU's slot probe**, the same way, then Metal's (B-metal already feeds `metalLiveAvailable` from the profiles).

**Who runs what.** Steps 1 and 2 are CPU work here (the CUDA module is cgo-free; B-cuda's census runs on this box in 1.3 s), except
step 1's device leg, which needs a nobara session's run on the GPU. Gates are registered per step before its code.

#### SB2 memory reads, step 1 — the memory-plan seam in cuda's build: gates, registered 2026-10-29 before the code (nobara-sb01; Francis 15:40)

**What changes.** `buildResident` gains a memory-plan mode, beside plan-only. It runs past plan-only's stop:
- the cuda resident's fields;
- the planned layer geometry (`planLayerGeometry`);
- the context margin;
- the context plan (`resolveCtxCapFitSlack`);
- the weights check (`checkWeightsFit`);
- the KV fit (`checkKVFits`: the one-slot fit, the trim of an unpinned context, the slot count).

It does this with the free-VRAM figures supplied instead of read, and returns those decisions as a record (`memPlan`). It allocates nothing, opens no device and starts no executor. The decisions become pure functions of a free figure:
- `weightsFitDecision`, the weights check over `fitsWeightsBudget`;
- `kvFitDecision`, the KV fit over `kvChargedBytes`, `kvChargedCapForBytes` and `kvChargedSlotsFit`, returning the context, the slot count, the log lines and the error;
- `resolveCtxCapFitSlack`, which reads free VRAM through a function it is given.

The real build calls the same functions with its live readings, so there is one implementation. The record has two readings because the real build reads twice:
- `FreeAtPlan`: what the plan and the weights check read. For a MoE build, that is after `openForPlan` has paid `moe_route`'s local-memory pool.
- `FreeAtKV`: what `checkKVFits` reads, after the weights and scratch are on the device.

Which figure a census feeds for each (step 2) is not decided here.

**Gates.**
- **G-MP1, tier 0, no device:**
  - Layer B-cuda's census (`TestBackendAdmissionCensus_cuda`, plan-only) is byte-identical to its golden.
  - MM1's decision census (`decoder`'s `TestDecisionCensus`) is unchanged.
  - The pure functions have table tests at their boundaries: fit exactly at the budget, one byte over, an unknown reading, the trim floor at `cudaCtxCapDefault`, and the slot clamp.
  - The cuda module builds, vets and passes `staticcheck` under no tag, `cuda`, and `cuda goinfer_testhooks`.
- **G-MP2, the device leg, nobara's RTX 2070 SUPER.** A probe test, kept in the record as a `.go.txt` and not committed as a test, runs the real `BuildResident` on every B-cuda cell that reaches the build. For each cell it records:
  - the answer (built, or the decline text);
  - the memory decisions: the planned context, the context after the KV fit, the KV slot count, the margin, and the expert-cache slot request;
  - the readings each decision used.

  It runs twice before the change (an A/A, which names any field that varies run to run) and once after. Every field the A/A shows stable must be byte-identical after. A field the A/A shows varying is listed with its spread and excluded, with the reason.
- **G-MP3, the seam agrees with the build, on the card.** On the after tree, the probe also runs memory-plan mode on each cell, fed the readings the real build took on that cell. The record must equal the real build's decisions on every cell: the weights verdict, the planned context, the context after the KV fit, the slot count and the margin. A planted change to `kvFitDecision` (the one-slot trim turned off) must turn it red on at least one cell, and the record lists which.
- **G-MP4, the MoE pool on the card.** For each MoE cell, the probe also records free VRAM just before `openForPlan` and the reading the plan took after it. Their difference is what the census must subtract from a profile's `cuda_free_bytes` for a MoE build. It is reported per cell beside the figure the KS-H6 notes give (about 138 MB retained on this card), for step 2 to use.

**Amendment, 2026-10-29, before any after-run (nobara-sb01).** The A/A pair showed that, on this card, real readings exercise
almost none of the KV arm. Every dense tiny cell plans at the 4096 floor without a plan reading, and every cell's KV fits with
gigabytes to spare. So no real cell trims or refuses, and the registered plant cannot go red. G-MP3 therefore gets a second
pass, **G-MP3b**:
- `cudaFreeVRAM`, the test seam `checkKVFits` reads through, is forced to 190 MiB in the real build. That is a few MiB over
  the tiny cells' margins, so the KV arm trims, clamps or refuses.
- The memory plan is fed that same figure as its KV reading.
- The plan must agree with the real build on every cell, as in G-MP3.
- The plant, the one-slot trim turned off on the plan's side only (`kvTrimOffForTest` set around the plan's call), must turn
  at least one cell red. The record lists which.

G-MP3 on real readings still runs as registered.

**Second amendment, 2026-10-29, after the 190 MiB pass and before any other (nobara-sb01).** At 190 MiB the plan and the build
agreed on all 412 cells, with and without the plant. The plant could not show, because no cell can trim at that figure:
- Only an unpinned context planned above 4096 can trim, and in this census that is qwen3next-tiny's 16384.
- Its margin is 250,333,184 bytes (238.7 MiB), above 190 MiB, so even the 4096 floor does not fit there. Both sides refuse it
  the same way.

G-MP3b's forced figure is therefore 245 MiB. That is inside qwen3next-tiny's trim window: its margin plus one slot's KV is
about 241 MiB at 4096 and about 249 MiB at 16384. Every other rule of G-MP3b is unchanged, including the plant, which must turn
at least one cell red. The 190 MiB pass stays in the record.

**Void:** a run while a timed holder has the timing lock, or with another process's CUDA context on the card besides the desktop compositor's (recorded before each run).

**Stopping rule:** the A/A pair, then one after-run. A mismatch on a stable field, or a G-MP3 disagreement, stops the step and is reported (`STATE: decision`).

**Cost basis:** layer B-cuda reaches the build on 412 cells, all tiny fixtures. A real build is about a second each, so a pass is minutes, under ten.

**Read (2026-10-29, nobara-sb01).** All gates pass. Record: `docs/measurements/sb2-memreads-2026-10-29/README.md`.
- **G-MP1:** the table tests and a deviceless plan of `testdata/llama-tiny` pass. Layer B-cuda's census matches its golden,
  and its plant still turns 42 cells red. `decoder/` is untouched.
- **G-MP2:** on the card, the real build's decisions on all 412 cells are equal across the A/A pair and byte-identical after
  the change. Only the readings vary.
- **G-MP3:**
  - Fed the build's own readings, the plan agrees with the build on all 412 cells.
  - At a forced 245 MiB KV reading, the build trims qwen3next-tiny from 16384 to 16320 on three cells, and the plan agrees on
    all 412.
  - The plan-side plant turns exactly those three red.
  - The 190 MiB pass agreed everywhere but could not show the plant. No cell can trim at that figure, which is why the second
    amendment moved it.
- **G-MP4:** a MoE build's plan reading is 138,412,032 bytes below its pre-pool reading, on every qwen3next-tiny cell. A
  census must subtract that from a profile's figure for a MoE plan.
- **For step 2:** only qwen3next-tiny's window (32768) reaches the context plan in this census. The other fixtures' windows are
  under the 4096 floor, so the plan and trim arms need wider cells.
- **After Gemini's review of #468:** four items fixed red-first, and the probe checks more (static decline text, companion
  bytes, slot request, log lines). Re-run on the card: G-MP2 still byte-identical on all 412 cells; G-MP3 agrees on all 412 on
  every field; the plant is red on exactly the same three cells. Record: same README, "After Gemini's review of #468".

#### SB2 memory reads, step 2 — the census over the machine profiles: gates, registered 2026-10-29 before the code (nobara-sb01; Francis 15:40)

Built on step 1's seam (`cudaBackend.planMemory`, #468); this branch starts from it.

**What the census feeds.** For each layer B-cuda cell, and each `testdata/machines` profile that carries `cuda_free_bytes` (today
`nobara-desktop` and `nobara-no-desktop`), with P the profile's `cuda_free_bytes`:
- **WeightsOpen** = P. A MoE build checks its weights after opening its device, before the pool.
- **Plan** = P less the profile's `cuda_moe_route_pool_bytes` for a MoE build whose context is planned, else P. The plan of
  such a build reads after `moe_route`'s pool is paid. Step 1's G-MP4 measured the pool at 138,412,032 B on this card and
  driver. The figure is added to both nobara profiles with that source, and a profile without it is not censused for a MoE
  plan: its line says so.
- **Weights** = the Plan figure. The setup job's check reads after the build's own device and context exist.
- **KV:** not fed. It is read after allocation, which is step 3's model. The census prints `kv: step 3` and records no KV
  decision.

The figures are written in each line, labelled as the profile's reading or as derived from it.

**The census.** `TestBackendMemoryCensus_cuda`, in `cuda/sb2_census_test.go`, the file the ratchet already exempts. One line
per cell and profile:
- the profile;
- the figures fed;
- the planned context;
- the margin;
- the weights verdict, or the refusal and its site.

A cell the build declines statically is layer B-cuda's line and is not repeated. The golden is
`cuda/testdata/sb2_cuda_mem_census.<goarch>.txt.gz`.

**Gates.**
- **G-MC1, tier 0, no device, on CI's deviceless cuda job.** The census is built twice in one process, equal; matches its
  golden; and covers every cell layer B-cuda plans (279 today) for each of the two profiles.
- **G-MC2, a plant.** `weightsFitDecision`'s margin is multiplied a thousandfold under a test flag. Every line must turn red
  (each becomes a weights refusal), and the census must name the count.
- **G-MC3, the census's figures against the card.** A device probe, kept in the record as a `.go.txt`, reads the registered
  memory probe (`decoder.FreeBytesFor("cuda")`, which is how a profile's `cuda_free_bytes` was taken) immediately before each
  load. It then records the real build's readings at each site (`memReadSeenForTest`):
  - On the four qwen3next-tiny MoE cells that plan, the plan reading must equal the probe reading less the profile's pool
    within 2 MiB (one allocation quantum).
  - For the dense cells, the weights-site reading less the probe reading is reported per cell. The census feeds the Plan
    figure there; the record states the spread.
  - A MoE cell outside 2 MiB stops the step (`STATE: decision`): it would mean the census's MoE figure is wrong.
- **Checks:** build, vet and `staticcheck` under no tag, `cuda`, and `cuda goinfer_testhooks`; `internal/layout` green, with
  no new test file in package `cuda`.

**Void (G-MC3 only):** a run while a timed holder has the timing lock, or with another process's CUDA context on the card
besides the desktop compositor's.

**Stopping rule:** one G-MC3 pass; a MoE disagreement stops the step.

**Cost basis:** the census is about two seconds with no device, as layer B-cuda's is; the device probe is about a minute, as
step 1's was.

**Expected informativeness, stated before the run.** Both profiles report about 7.6 GB free, and every tiny fixture's weights
are kilobytes. So the weights verdicts will all pass and the two profiles' lines will differ only in the figures. What the census
pins is the decision path and the figures fed. Profiles of smaller cards, or real-checkpoint cells, would make it discriminate.

**Read (2026-10-29, nobara-sb01).** All gates pass. Record: `docs/measurements/sb2-memcensus-2026-10-29/README.md`.
- **G-MC1:** 558 lines (279 planned cells × 2 profiles), built twice and equal, golden written.
- **G-MC2:** the planted margin turns all 558 into weights refusals.
- **G-MC3:** on the card, the four qwen3next-tiny MoE cells that plan read the probe less exactly the profile's pool,
  with zero deviation. The other 275 cells' weights reading equals the probe to the byte.
- **After Gemini's review of #476:** `PoolPaid` is set only after the pre-pool check passes, and the plant reaches the setup
  job's check, matched exactly. Its MoE lines name the post-pool figure, and with the pool left unsubtracted it goes red. G-MC3's
  probe asserts its tolerance in code and was re-run: 0 B deviation on all four cells.
- As stated before the run, every verdict passes on these profiles.

#### SB2 — Plan asks Admit, registered 2026-10-27 before the code (nobara-sb01)

Francis's decision (a) of 2026-10-27, option 1. `Model.Plan`'s eligibility check (`planIneligible`) asked `ResidentEligible`,
which is the arch's answer. It reads neither the precision policy nor WebGPU's Nemotron-MoE override, so Plan called Nemotron 3
Nano "not eligible" on webgpu while the load builds it resident. Plan will ask `Admit` with the model's own facts, and the
backend it is asked about.

- **What may change.** Plan's eligibility verdict, only where it disagreed with the decoder's admission. Its pricing does not
  change. WebGPU's lossy-KV mirror (Nemotron, Qwen3.5 and MLA at `--kv f16` or `--kv i8`) is unchanged. The `fit:` message for a
  declined backend names `Admit`'s refusal.
- **Gate 1 (decoder, tier 0).** `TestPlanAgreesWithAdmission` runs on layer D's fixtures and option cells, on cuda, metal and
  webgpu:
  - Plan's verdict comes from a CPU load with the cell's options and the KV request dropped (admission does not read the KV).
  - It must equal whether the decoder's admission let the load reach the backend's build, under layer D's fake backend.
  - Every census line is compared, and a disagreement is red.
- **Gate 2 (gpu, tier 0).** For each cell layer B-webgpu records, Plan's webgpu verdict with the cell's KV request must equal
  `staticResidentDecline`'s.
- **Plant.** The same comparison, fed the old verdict (`MissingResidentFeatures` plus `ResidentEligible`), must go red. It must
  name the Nemotron 3 Nano webgpu cells among its disagreements.
- **MM1's census.** It records Plan, so its golden moves. Every changed line must be a backend and cell where the old Plan
  disagreed with the decoder's admission, and the record lists them. A changed line outside that set stops the work with
  DECISION NEEDED.
- **Out of scope, reported.** CUDA's own build-time declines (`buildResident` in plan-only mode) are not the decoder's admission,
  and Plan does not read them. Layer B-cuda counts the cells where they decline. Closing that is a separate decision.
- **Cost basis:** tiny loads, seconds; the MM1 census is about a minute.

**Read (2026-10-27, nobara-sb01).**
- **Gate 1.** `decoder/plan_admit_test.go:TestPlanAgreesWithAdmission` passes: Plan and the load's admission agree on all 1,890
  cells (70 fixtures × 9 cells × 3 backends). 405 of those cells have no CPU model. They are fixtures no backend loads, and none of
  them reached a build.
- **Plant.** The old arch-only verdict disagrees with the load on 260 cells:
  - 216: per-32 activations on metal and webgpu, where Plan said eligible and the load declines;
  - 8: dense Nemotron-H at int8 or f32 without the knob, where Plan said eligible and the load declines;
  - 4: Nemotron 3 Nano on webgpu, where Plan said not eligible and the load builds it;
  - 2: Phi-3 at per-32 W8A8 on cuda, where the activation hazard is cleared, Plan said not eligible and the load builds it.
- **MM1's census.** It was dumped on main and on this branch and compared group by group.
  - 2,520 lines are wording only: a decline that stays a decline and now quotes `Admit`'s sentence.
  - The groups whose verdict changed are all `testdata/nemotron-tiny` at f32 (D3 on cuda, metal and webgpu, and D4, Metal's
    context fit, which reads the same eligibility) or `testdata/nemotron3nano-tiny` at int4 on webgpu (D3).
  - Every one of them is in the plant's disagreement set, so nothing outside it moved. The amd64 golden is regenerated; it is the
    only class the golden records.
- **Gate 2.** `gpu/sb2_census_test.go:TestPlanAgreesWithLoad_webgpu` passes: Plan's webgpu verdict, with each cell's KV
  request, agrees with the load (the decoder's admission, then `staticResidentDecline`) on all 630 cells. 135 of those cells have
  no CPU model, and none of them reached the build.
  - With main's old `planIneligible` in place, the same test is red on 113 cells.
  - Plan's KV request is the serving flag's precision (`KVPrecision`). The census's `kv-i8` cell sets `KVQuant` alone, which
    webgpu does not deliver.
- **Amendment after Gemini's review of #348 (2026-10-27, the coordinator's relay).** These are within Francis's decision that
  Plan agrees with the load.
  - WebGPU's routed-expert kind check (int8 or int4 only) read the weights alone but ran mid-upload. 16 f32 MoE cells passed
    `staticResidentDecline`, so Plan placed them while the build failed.
    - The check moves into `staticResidentDecline` as a typed decline, reading the new `decoder.Model.MoEExpertKind`. Plan's
      webgpu check mirrors it.
    - Layer B-webgpu's golden changes on exactly those 16 cells.
    - The real build on nobara's device changes on exactly those 16 cells too, from `gpu: MoE residency expert kind "f32"` to
      the decline. Both fall back to the CPU, and every other answer is byte-identical.
    - MM1's census changes in 297 groups, all D3 webgpu at f32 on those 16 fixtures, and all now the expert-kind decline.
  - Gate 1 now holds Plan's admission step (`Model.planAdmitDecline`) to the admission on all three backends. On cuda and metal
    it also checks that Plan's whole eligibility check equals that step. webgpu's mirrors are gate 2's.
  - Gate 2 adds the serving flag `--kv=i8` (both KV options) as a tenth cell, and passes on all 700 cells. Its in-test plant, a
    Plan asked without the KV request, turns 22 cells red, all lossy-KV cells of the families webgpu declines.
  - Both census builders and gate 2 clear `GOINFER_GEMMA4_RESIDENT` too, and unregister their fake backends
    (`decoder.UnregisterBackendForTest`).

### SB3 — family adapters

Moved to VK2 (one file each) and VK4 (program fingerprints). With SB1 and SB2, adding a family is a file, a fixture and
a program fingerprint, and nothing shared changes.

### SB4 — formats and the model store

- **Today:** the GGUF and safetensors readers, the `.giw` container (v17; the reader refuses any other version, per the
  owner's v16 decision), transcodes and `pull`. `decoder/weights.go` (3,106 lines) and `decoder/gguf.go` (3,028) are
  in 42 families' hash set.
- **Contract:** what a loader returns for a file: per tensor, its layout kind and a hash of the loaded bytes.
- **Fast check:** synthetic GGUF and safetensors files written in-test, round trips, per-tensor hashes; on real
  checkpoints, the hashes of the first tensors of each kind.
- **Stops re-running:** forward parity for loader-only edits (identical tensors give an identical forward; the loader
  fingerprint is one input to VK's key).
- **Open question for the format's owner:** whether a layout kind can carry its own version inside the single-container
  rule, so a new kind does not re-transcode every cached `.giw`. M.

#### SB4 — the loader identity: gates, registered 2026-10-27 before the code (mbp2-sb4)

Coordinator's brief (2026-10-27 15:17, Francis's go of 15:13): SB4 provides the loader identity as an input; VK decides
how it enters the key. This item changes no key. Every gate below is tier 0: tiny fixtures, CPU only, seconds.

- **The identity.** `loaderIdentity(*Weights)` walks the loaded weights by reflection and lists one entry per tensor:
  its field path (`Layers[3].QProj`, `Layers[0].Experts[2].Up`), its layout (the WeightMat's kind, a plain slice's
  element type, or `scalar`), its shape and a SHA-256 of every byte and field of it, including aikit's unexported
  WeightMat fields and any repacked copies. It leaves out the config and architecture (VK4's program fingerprint covers
  them), the source handles, the load profile and the `.giw` quant label. A field of a kind the walk does not know
  (a map, a func, an interface) panics, so a new kind is learned, never skipped.
- **The table.** `testdata/loader_identity.json`: GOARCH, then fixture, then mode, then the digest of the identity
  lines. Fixtures are every committed `testdata/*/` with a `config.json` and a `model.safetensors`, and every committed
  `testdata/*.gguf`. Modes are `parseQuant`'s eight; a fixture and mode is in the table when `loadWeights` takes it
  with the CPU backend's flags (no canonical int4 kept, row4 allowed) and `checkHalf16Load` allows it, and the test
  fails both on a pair that loads with no entry and on an entry that no longer loads. A GOARCH with no entry is a named
  skip, not a pass; this box makes the amd64 entry, and the arm64 entry is a Mac's `-update`.
- **The shadow input.** `internal/paritykey` gains the table's path, its reader and a per-family digest over a fixture
  list, which no key reads. How it enters the key is a decision for VK (DECISION NEEDED in the record).

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| LI1 stable | every fixture and mode, loaded twice in one process, gives equal identities; `TestLoaderIdentity_table` in a second process equals the committed table | never | one run | 0.7 s for 33 fixtures × 3 modes in the probe |
| LI2 one byte, one tensor | in a temp copy of `llama-tiny` and of `glm-tiny.gguf`, one byte inside one tensor's data is flipped; in every mode that loads, the identity diff is exactly the paths built from that tensor (one path), nothing else. Mutation: hashing a slice by its length alone turns it red | never | one run | seconds |
| LI3 refactor-blind | (a) every fixture and mode: the identity after a `.giw` round trip (`SerializeWeights`, `LoadSerializedWeights`: a second reader sharing no read path) equals the direct load's; (b) a throwaway commit that refactors `decoder/weights.go` without changing any tensor (a helper renamed, two independent loads reordered) leaves `TestLoaderIdentity_table` green, while today's key re-stales all 43 families | (b): the refactor does not build | one run each | seconds |
| LI4 change-sensitive, narrow | a throwaway commit scaling Phi-3's final norm in `buildPhi3Weights` by 1.001 turns `TestLoaderIdentity_table` red, and its diff names `testdata/phi3-tiny` only, in every mode | — | one run | seconds |
| LI5 the walk is total | a struct with a map field makes the walk panic; every name in `loaderSkip` is a `Weights` field. Mutation: the walk's default case returning silently turns it red | never | one run | seconds |
| LI6 no key moves | `TestParityKey_matchesFreshDepsHash`, `TestParityKey_shadowMatchesDecoder` and `TestParityManifest_fresh` stay green; `gate shadow origin/main <head>` classes all 43 families agree-fresh | — | one run | under a minute |
| LI7 the counterfactual (record only) | for each commit in VK0's 30-day window whose dependency changes touch only the `loaders` set, the table at the commit and at its parent: the number of families whose fixtures' identities moved, against the 43 today's key re-staled | a commit whose tree cannot build the identity: VOID, listed | the window's loader-only commits, all, or stop at 08:00 PDT 2026-10-28 with the count reached stated | about a minute a commit; state the count before starting |

**The limit, stated before the code.** Like week 2's program fingerprints, the identity sees only what the fixtures
load: a loader change on a path no fixture takes (GPTQ, AWQ, a sharded set, a GGUF quant type no tiny GGUF holds)
leaves every identity unchanged and reads as a refactor. `decoder/gguf.go` is covered only for the families with a GGUF
fixture.

**Graded 2026-10-27: LI1–LI6 PASS** (`docs/measurements/sb4-2026-10.md`).
- LI1: 234 fixture-mode pairs on amd64, each stable across two loads and two processes.
- LI2: one flipped sign bit moves exactly `Layers[0].KProj`, in 6 modes for `llama-tiny` and 6 for `glm-tiny.gguf`.
- LI3: 162 `.giw` round trips leave every identity unchanged (bf16 and f16 left out: the `.giw` has no form for them,
  and `SerializeWeights` writes a blob its reader refuses, a finding); a `weights.go` refactor leaves the table green
  while today's key re-stales all 43 families.
- LI4: a Phi-3 final-norm change moves `phi3-tiny` alone.
- LI5, LI6: the walk panics on an unknown kind; no key moved (43 agree-fresh).
- Every test was shown red once by a mutation, listed in the record.
- LI7: 12 loader-only commits in VK0's window, not 30. As registered, 4 were measured and 8 were VOID: trees before giw
  v16 hold a map field the walk refuses. A supplement decided after seeing that, not graded, measured all 12. No
  fixture identity moved, and option A's rule takes the window's 486 re-stalings to 331, every one from a family
  without a loader or GGUF fixture.
- How the identity enters the key: Francis chose option A on 2026-10-27, a week-3 shadow after week 2 is graded (on or
  after 2026-11-03); the record's §4.

### SB5 — engine policies (sessions, prefix reuse, speculative decoding, sampling, constraints)

- **Today:** 26 `spec_*` test files and the session and reuse tests in `decoder` run through loaded fixtures;
  `internal/sampling` and `internal/kvmgr` are already packages of their own.
- **Contract:** each policy runs over a `Forward` interface; an exact policy produces the same token stream as plain
  decoding.
- **Fast check:** a fake model with scripted logits: acceptance patterns, rollbacks, branch points. M.

#### SB5 — gates, registered 2026-10-27 before the code (mbp2-sb4)

Francis picked SB5 at 18:47 (relayed by the coordinator): behavior first, one PR by morning, CPU only on mbp2 (`go
test -p 4`), tier 0. **No token stream may change; if one does, that step stops with DECISION NEEDED.**

**What the code is today** (a survey of `decoder` at `4c7cd4a0`):
- Every policy is a method on `*Model` over a concrete `*KVCache`: draft-model speculation (`GenerateSpeculative`),
  n-gram speculation (`genNgramInto`, fixed and adaptive), grammar speculation (`genGrammarInto`), sessions (resume,
  rewind, prefix copy) and prefix reuse.
- The model calls inside the speculative loops are already local closures (`targetVerify`, `targetTruncate`).
- The only model interface is the resident one (`ResidentForward`), so no CPU-path fake exists, and the parity tests
  that assert "speculative equals plain greedy" mostly load a bench checkpoint (`GOINFER_PREQUANT_GGUF`) that is not
  committed: on a box without it they skip.
- Neither MTP nor a production tree verify exists, so they are out of scope.

**The contract.**
- **`Forward`** is the model as a policy drives it: `Prefill(ids)` (the last row's logits), `Verify(ids)` (one row per
  id) and `TruncateTo(pos)`.
- `cpuForward` implements it over `*Model` and `*KVCache`.
- `scriptedForward` is a fake whose logits are a pure function of the token history.
- A policy moved onto `Forward` is one function used by the real path and the fake alike, not a second path.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB5-P1 the census | `TestPolicyStreams_census` writes, and then holds, `testdata/sb5/policy_streams.json`: the token streams of every policy on committed tiny fixtures (llama-tiny, with llama-attnfa-tiny as a draft, a byte-level vocabulary for grammar), with each exact policy's stream compared to plain decoding's in the test. The cases are plain greedy; n-gram K 1, 4, 8 and adaptive; a draft model, including the target as its own draft; grammar speculation against constrained greedy; session resume, rewind and prefix copy against cold decoding; and seeded sampling, plain and n-gram, recorded but not compared, since sampled speculation is lossless in distribution, not in stream | a fixture absent: listed, never counted | one run | seconds |
| SB5-P2 the census can go red | for each policy, one planted change (an acceptance rule off by one, a rollback position off by one, the grammar mask skipped on a draft, a session rewind off by one) turns P1 red | — | one run each | seconds |
| SB5-P3 exact on the fake | each policy moved onto `Forward` reproduces plain decoding (constrained, for grammar) on `scriptedForward` for scripted patterns: every draft accepted, every draft rejected, a rejection at each depth, a stop token inside a draft, maxTokens reached inside an accepted run, and scripted branch points where the target's choice changes after a rollback | — | one run | seconds |
| SB5-P4 the census holds through the move | after each move, P1 reproduces with no stream changed and the golden unedited | a golden edited in the same commit: void | every move | seconds |
| SB5-P5 the ratchet | `testdata/sb5/loaded_policy_tests.txt` lists the decoder tests in `spec_*_test.go` and `session*_test.go` that load a model to check a policy, and a test fails on one not listed. Each move removes its tests from the list | — | every move | seconds |

**This PR's first move** is grammar speculation (`genGrammarInto`, CPU only): its loop becomes a function over
`Forward`, and `TestGrammarSpecParity`'s claim moves to the fake. Draft-model, n-gram, session and prefix-reuse moves
follow in batches.

#### SB5 — the plain decode loop, registered 2026-10-27 21:15 PDT before the code (nobara-ci01; Francis's assignment of 21:02)

The step left after the census and the four moves (`docs/measurements/sb5-2026-10.md` §8): `generateInto`'s decode loop.
Its skeleton (the cancel check, the host sampler with the processor and its gate, the stop check, logprobs, the send, the
generated list, the KV-pool-full and error exits, the commit callback) moves into `plainDecodeLoop`, over a
`decodeTarget` of hooks. `generateInto` sets each hook to a closure over the code it already has: `pick` (a token the
device already chose: the greedy fast path or chain, the device draw, a device top-K row, optimistic forward), `logits`
(the row for the host sampler), `advance` (the forward: resident, MC3, the CPU batcher or the CPU), `send` (MC3's hold
grace) and `finish` (the resident commit). The prefill and the path selection stay in `generateInto`. No token stream may
change; if one does, the step stops with DECISION NEEDED.

The device record: `ResidentStreamsForTest` (a `goinfer_testhooks` hook in `decoder`) runs 16 cases on llama-tiny
resident, each a path chosen by its knob and sampling parameters (greedy and its chain or fast path, the logits path,
`top_k` 1, a stop id, logprobs, a processor, a gated processor, device and host sampling, device and host top-K, top-p, a
penalty, optimistic forward on and off), alone and with MC3 on (plus three concurrent generations where the resident
batches). `TestSB5ResidentStreams` in `cuda/` and `gpu/` writes the streams and the path counters (device draws, top-K
served and fallen back, optimistic-forward guesses and hits, the budget, logprobs) with `-sb5.out` and holds them with
`-sb5.want`. Recorded at `7b90d2829` before any code changed: 35 streams on CUDA, 32 on WebGPU (no MC3 there), each
reproduced by a second run (`docs/measurements/sb5-generateinto-2026-10-27/`).

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB5-P4 (CPU) the census holds | `TestPolicyStreams_census` reproduces `testdata/sb5/policy_streams.json` byte for byte, the golden unedited | a golden edited in the same commit | one run | seconds |
| SB5-P6 the device record holds | `TestSB5ResidentStreams -sb5.want` passes on CUDA and on WebGPU on nobara (RTX 2070 SUPER, driver 595.91.07) against the baseline at `7b90d2829`: every stream and every path counter equal | the baseline's own second run differed (it did not); a timed run holds the card (wait) | one run per backend | seconds each |
| SB5-P3 exact on the fake | `plainDecodeLoop` on a `scriptedForward` target emits what a direct reference loop over the same fake emits, for: greedy and seeded sampling; maxTokens 0, 1 and past a stop; a stop id inside the stream (not emitted, not forwarded); logprobs; a processor and a gated processor (each called with the tokens generated so far); a device-chosen token (`pick`) that bypasses the sampler; a cancel before and after a send (`finish` called once); a KV-pool-full refusal on a step (no commit for it, `finish` not called); and the commit callback once per forwarded token, in order. Red once each on four plants: the commit before the forward, the stop token emitted, the gate given the list with the new token, `finish` skipped on a cancel | — | one run | seconds |
| SB5-P5 the ratchet | `TestPolicyTests_loadedRatchet` passes, its list unchanged or shorter | — | one run | seconds |
| SB5-P7 the decode tests that ran | the decoder tests beside the touched files, the CUDA KV-slot, plan-slack and ledger tests and the WebGPU KV-slot tests pass (as `touched.py` and the device suite name them) | a timed run holds the card | one run | minutes |

**Results, 2026-10-27 (nobara)** (`docs/measurements/sb5-2026-10.md` §9). P6 PASS: CUDA 35 and WebGPU 32 streams equal the
`7b90d2829` baseline byte for byte, with two plants red. P4 PASS: the census is unchanged. P3 PASS: 153 cases, with four
plants red. P5 PASS: 40 listed. P7 PASS on CUDA, WebGPU and CPU, plus the three real-checkpoint identity tests on CUDA. No
token stream changed.

 — the scheduler

- **Today:** queueing, fair admission, jobs and batches live in the HTTP package (`internal/serveapp/batches.go`,
  `internal/serveapp/sessions.go` and the job files; `docs/tasks/task-work-queue-2026-09.md`, J1–J9).
- **Contract:** a policy over a queue and an engine's capacity: admission order, fairness, priorities, deadlines,
  prefix-aware placement (J6), and later RF's batch building.
- **Fast check:** a simulator on a virtual clock over recorded arrivals. M.

#### SB6 — the scheduling census, registered 2026-10-27 17:17 PDT before the code (mbp2-sb6)

Francis chose SB6 for mbp2's second session (17:13). The way SB8 worked: census first, recorded at `origin/main`; the move
to `internal/scheduler` must reproduce every golden byte for byte; nothing about scheduling behavior may change (a change
stops the work with DECISION NEEDED).

**What schedules today, read before registering.** Three decisions, all in `internal/serveapp`:
- **Admission order** (`internal/serveapp/admission.go:admission`): a per-model FIFO turn granter, `-max-concurrent` wide (`setCap`), that a
  waiter leaves the moment its context ends, with a direct hand-off on `release` and `position` for `GET /v1/jobs/{id}`.
- **The queue bound** (`internal/serveapp/openai.go:loadedModel.tryEnter`): a slot of `lm.queue`, `concurrency + -max-queue` deep, claimed
  before the turn and refused at once when full (the 429); a halt found after the grant gives the turn back.
- **Prefix-aware placement** (`internal/serveapp/sessions.go:sessionLRU.acquire`): which resident session a prompt extends
  (`pickSessionFor`, `bestExtend`, `keepsFirst`), which idle session gives rows by copy (`donorFor`), and which is
  evicted (`coldestIdle`, `fresh`).

There is no fairness across sessions, no priority, no class and no deadline: J6's reordering was measured and killed
(`docs/measurements/j6-prefix-scheduling-2026-09-15.md`) and J7 was never built (`docs/tasks/task-work-queue-2026-09.md`
§J6, §J7). Admission never reads a clock. So the census records those cases as they are, FIFO, with each trace's
session, priority and deadline kept beside the order, so the first policy that uses them shows its whole diff against
today's.

**The census** is two tests in `serveapp` that add test files and goldens under `internal/serveapp/testdata/sched/` and
change no other file:
- `TestSchedCensus_admission`: a simulator on a virtual clock that replays a trace of events (arrive, finish, cancel,
  set capacity, halt, resume) through the real `tryEnter`, `exit`, `position` and `setCap` of a bare `loadedModel`, one
  goroutine per request. After each event it waits until every live request has either returned or stands in the queue,
  so the record is deterministic. Recorded per event: who was granted, refused (queue full, cancelled, halted) and when,
  in virtual time; the running set; the queue in order with each waiter's `position`; `load`. Traces, synthetic: FIFO at
  one and at N turns; two sessions, one flooding; mixed interactive and batch annotations; deadlines that pass while
  waiting; cancellation at the front, middle and back of the queue and of a running request; a burst into a bounded
  queue and an unbounded one; capacity raised and lowered with waiters; a halt with waiters, then resume; J6's agent-loop
  mix (a shared preamble, divergent tails).
- `TestSchedCensus_placement`: the real `sessionLRU` over `testdata/llama-attnfa-tiny` (committed, CPU), replaying
  conversations (new, continue, roll back, edit the last message, a shared preamble, more conversations than slots,
  all slots busy, sharing on and off). A conversation's next prompt is its last prompt plus the model's reply plus new
  tokens, as a client sends it, so the record depends on the reply's length and not its token values. Recorded per
  acquire: the session handed out (a stable label), its tokens' common prefix with the prompt, fresh, evicted, transient
  or taken in place, the donor and rows copied, and the LRU order after.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| SC1 recorded | both census tests pass at `origin/main` with the goldens written from it; recording twice gives identical files; `-count=20 -race` on the admission census passes | none: deterministic | a nondeterministic record is fixed in the driver before anything is recorded | seconds |
| SC2 shown red | each plant fails the census and is reverted: `release` handing the turn to the back waiter (LIFO); the queue bound one deeper; a waiter granted although its context ended; `bestExtend`'s floor removed; `donorFor`'s `>=` made `>`; `coldestIdle` taking the most recent | none | one run each | seconds |
| SC3 the move | `internal/scheduler` imports the standard library only (a rule in `internal/layout/imports.json`, so `TestLayout_arrows` fails on any goinfer import there, shown red with a planted `decoder` import); both census tests pass with no file under `internal/serveapp/testdata/sched/` in the diff | none | any golden change stops the move with DECISION NEEDED; there is no tolerance | seconds |
| SC4 callers | `serveapp`'s admission, session, concurrency, KV-share, job and batch tests named for the moved code pass unchanged, with no test or golden of theirs in the diff; `touched.py` green | none | a red test is fixed in the move or the move is reverted, never a golden re-recorded | `touched.py` seconds; the rest is CI's |

- **Census recorded (2026-10-27, mbp2, Linux amd64, Go 1.27.3, at `origin/main` `2405f834`).**
  `internal/serveapp/sched_census_test.go`: `TestSchedCensus_admission`, 11 traces under
  `internal/serveapp/testdata/sched/admission_traces/` with their records under `testdata/sched/admission/`, and
  `TestSchedCensus_placement`, 10 traces under `testdata/sched/placement_traces/` with records under
  `testdata/sched/placement/` (252 KB in all, plain JSON). The admission driver sizes the turns and the queue with the real
  `setConcurrency` (no model at one turn; `testdata/llama-attnfa-tiny` on the CPU above it). Each placement reply is
  forced to one token per conversation with `LogitBias`, so no record depends on a logit. Both tests take about 4 s
  together. **SC1:** pass; recording twice gives identical files; `-race -count=20` on the admission census and
  `-count=5` on both pass. The first driver read a cancelled waiter as still queued at its cancel event (a race between
  the cancel and the waiter's own exit), which recording twice would not have caught; `settle` now waits for that waiter
  to return. **SC2:** each plant failed and was reverted: LIFO `release` fails 10 of 11 admission traces (all but the
  halt); the queue one deeper fails `burst_bounded_queue`; a waiter that ignores its context fails `cancel_queued`; no
  halt check after the grant fails `halt_with_waiters`; no `bestExtend` floor and `donorFor`'s `>` each fail
  `more_conversations_than_slots`; `coldestIdle` taking the newest fails four placement traces.
  What the census shows today, recorded as it is, unchanged:
  - a raised turn count wakes no waiter, and a newcomer still queues behind the waiters; a lowered one takes hold only
    as the queue empties, since `release` hands a turn straight to the next waiter (production calls `setCap` only at
    load);
  - a halt refuses each waiter in queue order as the running request's turn is handed on, including one that queued
    after the halt (in production `haltGate` refuses that one before admission);
  - two conversations that share a 130-token preamble and hold little of their own trade one session every turn,
    although the LRU has room: the room rule holds a candidate back only when the request would keep less of it than it
    overwrites;
  - with every slot full, conversations sharing a preamble take the most recent session that holds it, cutting each
    other back to the preamble, while a session sharing nothing stays resident.

The move's package shape is registered after the census lands, as SB8's was.

#### SB6 — the move to `internal/scheduler`, registered 2026-10-27 before the code (mbp2-sb6)

`internal/scheduler` owns the decisions, over plain values; `serveapp` keeps what touches a `decoder.Session`, a halt, a
job or HTTP, and calls it. Standard library only. Two pull requests, each green on its own:

- **M1, admission and placement.** `scheduler.Admission` is today's `admission` with its methods exported (`SetCap`,
  `Capacity`, `Load`, `Enter`, `Position`, `Release`) and `scheduler.Record` is `admissionRecord`. `serveapp`'s
  `admission` embeds it and keeps its lowercase methods as one-line delegates, so its zero value and its tests are
  unchanged. Placement: `CommonPrefix`, `BestExtend`, `PickSession` (today's `pickSessionFor`), `KeepsFirst`,
  `ColdestIdle` (over busy flags, most recent first) and `PickDonor` (over each idle session's shared hit, the `>=` rule
  kept) move; `sessionLRU` computes the inputs (tokens, first prompts, busy flags, `SharedHit`) and calls them, and
  `serveapp`'s `commonPrefix`, `bestExtend`, `pickSession`, `pickSessionFor` and `keepsFirst` become delegates for the
  tests that call them. `internal/layout/imports.json` gains a rule, `scheduler-standard-library`: `internal/scheduler`,
  tests included, imports no goinfer package and nothing outside the standard library (a `std_only` field the layout
  test gains for it).
- **M2, the queue bound.** The bounded queue (`lm.queue`, `concurrency + -max-queue` slots claimed before the turn,
  the 429 when full, released after the turn) and `tryEnter`'s order (claim a slot, wait for the turn, give both back
  on a halt found after the grant, through a halt function `serveapp` passes) move into `scheduler`; `serveapp` keeps
  the refusal's error shapes.
- **Not moved:** `sessionLRU` itself, tiering and its disk I/O (they hold `decoder.Session`s), jobs, batches and their
  stores (records, not policy), and `haltGate`.

**SC4, amended before the code:** `jobs_w27_test.go` waits for its waiters by reading `admission`'s lock and list, which a
type in another package cannot expose; that loop reads the same count from `position` instead, and none of its
assertions changes. No other test file under `internal/serveapp/` is in either diff. SC1 to SC3 stand as registered.

- **M1 landed on its branch (2026-10-27, mbp2).** `internal/scheduler/admission.go` (`Admission`, `Record`) and
  `internal/scheduler/placement.go` (`CommonPrefix`, `BestExtend`, `PickSession`, `KeepsFirst`, `ColdestIdle`,
  `PickDonor`). `serveapp`'s `admission` embeds `scheduler.Admission` behind one-line delegates (its `capacity` delegate
  is gone: nothing but the old `enter` called it); `sessionLRU.donorFor` and `coldestIdle` pass the busy flags and shared
  hits to the scheduler; the five placement helpers are delegates. **SC3:** both census tests pass, 21 traces, with no
  file under `internal/serveapp/testdata/sched/` in the diff; the same plants made inside `internal/scheduler` (LIFO
  `Release`, `PickDonor`'s `>`) fail the same traces as in `serveapp` before the move (10 admission traces;
  `more_conversations_than_slots`), so the census runs through the moved code. The layout rule
  `scheduler-standard-library` is in `imports.json`, with a `std_only` field in the layout test, which now knows every
  package directory, not only those with a non-standard import (a standard-library-only package had looked absent);
  a planted `decoder` import in a scheduler test file and a planted `golang.org/x/sys/unix` import each turn
  `TestLayout_arrows` red, naming the rule. **SC4:** `jobs_w27_test.go`'s wait loop reads `position`'s count, as
  amended; 54 tests in the admission, session, concurrency, KV-share, job, batch, cancel and backpressure files run
  under `-race`: 45 pass, 9 skip (they need a `.gguf` in `GOINFER_SERVE_MODEL` or `GOINFER_MC3C_MODEL`, not run here:
  tier 0 on a CPU-only box). `touched.py` green; staticcheck clean.
- **M2 landed on its branch (2026-10-27, mbp2).** `internal/scheduler/queue.go`: `NewQueue`, `QueueFull`, `MaxQueued`,
  `Admit` (claim a slot, wait for the turn, then the halt check, which gives both back) and `Exit` (the turn, then the
  slot). The queue stays a `chan struct{}` on `loadedModel`, so the census reads it unchanged; `tryEnter` is a call to
  `Admit`, `exit` to `Exit`, and the four places that sized or read the queue call the scheduler. The halt callback
  returns `(reason, halted)`, so a halt with an empty reason still refuses, as `tryEnter`'s `hi != nil` did;
  `TestAdmit_haltWithNoReasonRefuses` in `internal/scheduler` pins it (shown red by keying the refusal on a non-empty
  reason, reverted). **SC3:** both census tests pass with no golden in the diff; plants in `queue.go` fail the census and
  are reverted: one slot deeper fails `burst_bounded_queue`, no halt check fails `halt_with_waiters`, a slot kept after
  a cancelled wait fails `cancel_queued`. **SC4:** the same 54 tests under `-race`: 45 pass, 9 skip, as in M1; no test
  file under `internal/serveapp/` in M2's diff. `touched.py` green; staticcheck clean; `go vet` with `realckpt` and
  `goinfer_testhooks` clean.
- **The subsystem's documents, per §4.1 (2026-10-27, mbp2).** `internal/scheduler/doc.go` (the contract; six invariants,
  each naming its test, which `TestPackageDocs_namedTestsExist` checks), `internal/scheduler/CLAUDE.md` (the day loop,
  the import rule, a census move needs Francis's word), `docs/code-notes/internal-scheduler.md` (why decisions over
  plain values; admission's hand-off and post-grant halt check; placement's floor, room rule and donor tie-break; what
  it does not decide, with J6's kill and J7), and `docs/ARCHITECTURE.md` (the request path and the packages table). The
  design reasons that sat on `serveapp`'s `admission` comment moved to the code-notes.
- **Gemini's review of #352, answered (2026-10-27, mbp2).** The exported `Admission.Capacity` read the cap without the
  lock `SetCap` writes it under. It now takes the lock, and `Enter`, which already holds it, reads `capLocked`.
  `TestAdmission_capacityUnderConcurrentSetCap` reads `Capacity` while `SetCap`, `Enter` and `Release` run; `-race`
  passes, and with the old unlocked read it reports a data race (shown, reverted). No caller in `serveapp` reads
  `Capacity`, so the census is unchanged; it was re-run after merging `origin/main` into #352, #353 and #355, and every
  golden matches.
- **Gemini's review of #355, answered (2026-10-27, mbp2).** `NewQueue` sized the queue `turns + maxQueue` while
  `Admission` and `MaxQueued` count turns as at least one; it now clamps the same way. Every caller passes at least one
  turn (`setConcurrency` clamps n to 1 or more; the load sites pass 1), so no queue changes size and the census is
  unchanged; `TestNewQueue_unboundedAtZero` checks 0 and -2 turns and fails without the clamp. `PickDonor` indexed
  `busy` unguarded; a session past its end now counts as idle (a nil `busy`: none busy), and `TestPickDonor_shortBusy`
  panics without the guard. `serveapp` passes a flag for every session, so the census is unchanged.
- **Closed (2026-10-27).** Francis chose option A (18:05, relayed): SB6 closes at the move. The scheduling policy is in
  `internal/scheduler` with no behavior change: the census (#351) reproduces byte for byte through M1 (#352) and M2
  (#353), and the §4.1 documents are #355. Priorities, classes and deadlines (J7, `docs/tasks/task-work-queue-2026-09.md`)
  stay FIFO, as the census records them, and remain a possible future item on this package; the census traces for them
  (`priorities_classes`, `deadlines_pass_while_waiting`, `fairness_flooding_session`) are the baseline a J7 policy would
  diff against, with Francis's word before any golden moves.

### SB7 — the serving shell

- **Today:** `internal/serveapp`, 76 source files and 19.9k lines; `main.go` took 116 commits in 30 days; 39 of its
  162 test files import `decoder`.
- **Contract:** an `Engine` interface (generate, stream events, cancel, stats, models).
- **Fast check:** a fake engine; the OpenAI and Anthropic shapes and SSE checked with no model. The `sdktest` module
  stays the end-to-end check. M.

#### SB7 — the wire census, registered 2026-10-27 18:56 PDT before the code (mbp2-sb6)

Francis picked SB7 for mbp2's session (18:06). The way SB6 worked: census first; the shell moves onto an `Engine`
interface with every golden reproducing byte for byte; nothing a client sees may change (a change stops the work with
DECISION NEEDED). The census is recorded on a branch carrying SB6's move (#352, #353, #355), which changes no wire
behavior (SB6's SC4).

**The census** is a test in `serveapp`, `TestWireCensus`, that adds test files and goldens under
`internal/serveapp/testdata/wire/` and changes no other file. It drives the in-process handler (`NewHandler`, as
`TestNewHandler_servesTheAPIInProcess` does) over a scripted list of requests and records, per case, the status, the
headers a client reads (`Content-Type`, `Retry-After`, `Cache-Control`) and the body byte for byte, with an SSE stream
kept whole (event names, data lines, separators, `[DONE]`).
- **Two kinds of case.** Real-engine cases run the tiny chat fixture (`tinyChatCheckpoint`, CPU) and force every
  generated token with `logit_bias` (accepted on the OpenAI routes and, as a goinfer extension, on `/v1/messages`), so
  no record depends on a logit and the goldens hold on every CI architecture. Tool-call and reasoning cases, which a
  forced single token cannot produce, run the server's scripted-generation seam (`loadedModel.scripted`): the whole
  serving path, with the model's token ids fixed.
- **Requests:** `/v1/chat/completions` (plain, streamed, with `stream_options.include_usage`, `n`-less, a stop string, a
  tool call and its streamed form, a tool result turn, reasoning split out); `/v1/completions` (plain, streamed);
  `/v1/responses` (plain, streamed); `/v1/messages` (plain, streamed, a tool use, thinking); `/v1/messages/count_tokens`;
  `/v1/models`; `/health`; errors (unknown model, malformed JSON, missing messages, `max_tokens` over the ceiling, an
  unsupported field, the wrong method); a cancel mid-stream (the client closes after the first event: the record is the
  events received before it).
- **Normalized,** and named in each golden's header: response and call ids (`chatcmpl-…`, `msg_…`, `resp_…`, `call_…`,
  `toolu_…`) as placeholders numbered by first appearance, `created`/`created_at` as 0, and any duration or timestamp
  field the record turns up, each listed. Nothing else.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| W1 recorded | `TestWireCensus` passes with the goldens written from the tree; recording twice gives identical files; `-count=3` passes | none: deterministic | a nondeterministic field is normalized (and named) or the case is fixed before anything is recorded | seconds |
| W2 shown red | each plant fails the census and is reverted: the SSE event separator; a `finish_reason` value; an Anthropic event name; an error's status code; a usage field dropped; a scripted tool call's argument encoding | none | one run each | seconds |
| W3 portable | no real-engine case's record depends on a logit: each forces its tokens, which the census asserts (every generated token is the forced one) | none | a case that cannot be forced is moved to the scripted seam | seconds; CI's arm64, darwin and windows jobs run it |
| W4 the ratchet | `serveapp` test files importing `decoder` may only fall: a test holds the count at today's 40 (39 in the brief, plus SB6's scheduling census, whose placement half needs real sessions) and fails on 41; shown red with a planted import | none | one run | seconds |
| W5 the engine (later steps) | with the shell on the `Engine` interface, the census passes through the real engine with no golden in the diff, and the scripted cases pass through the fake engine | none | any golden change stops with DECISION NEEDED | seconds |

- **Census recorded (2026-10-27, mbp2, Linux amd64, Go 1.27.3, on `sb7-serving-shell-2026-10` off #355's head).**
  `internal/serveapp/wire_census_test.go`: `TestWireCensus`, 38 cases in `internal/serveapp/testdata/wire/cases.json`,
  one golden each under `testdata/wire/golden/` (status, client-read headers, body), 25 on the tiny model and 13 on the
  scripted model. Each case gets a fresh in-process server, so no record depends on the cases before it; two cases send
  their request twice (`warm`) to record prefix reuse. The census takes about 0.3 s. Normalized, and named in each
  golden: ids (28 goldens), `created`/`created_at` (19), the one logprob (its value is a logit) and `/health`'s
  `selftest` list, a process-wide registry that other tests in the package write (a placeholder; see below). **W1:** pass; recording three times gives identical files; `-count=5` and `-race -count=2`
  pass. **W2:** each plant failed and was reverted: the SSE data separator (7 streamed cases), `finish_reason`
  `length` renamed (13), an Anthropic event name (2), the unknown-model status (1), `completion_tokens` renamed (9),
  tool-call arguments re-encoded (3 chat cases). **W3:** every generated string in a tiny-model case is the forced
  token, asserted per case; a case whose bias is zeroed fails it (`"[y"`, unforced). **W4:**
  `internal/layout`'s `TestServeappTests_decoderImportsOnlyFall` holds 40; a planted import makes 41 and fails.
  What the census shows today, recorded as it is, unchanged: with tools offered, a streamed chat reply has no opening role
  chunk (a plain stream opens with `{"role":"assistant"}`): prose before a call streams with no role, and the role first
  arrives on the tool-call chunk; the scripted model, which has no checkpoint, is listed by
  `/v1/models` and `/health` with its `-fast` alias and no context length.
- **CI red on #376, fixed (2026-10-27, mbp2).** `TestWireCensus/health` failed on test-rest, root-windows and
  root-linux-arm64, and nothing else did. Not machine-dependent: `/health`'s `selftest` lists every self-test recorded in
  the process, and `TestServe_healthReportsSelfTests` records a fake `healthtest-gpu` backend into the same registry, so
  the body depended on which tests ran first; every local run had run the census alone. Reproduced by running the two
  together. The census now replaces the `selftest` value with `"<process-wide>"` (bracket-matched, named in the golden's
  header); that test checks the field itself, and the server is unchanged. The golden's one change is that value. With
  the two tests together under five `-shuffle` seeds the census passes.
- **The census extended to jobs, batches, files and admin (2026-10-27, mbp2; `sb7-census2-2026-10`).** A case may now be
  a list of steps, each recorded, a later step naming an earlier response's field (`{{1.id}}`), polling until a field
  reaches a terminal state, or uploading a multipart file (a fixed boundary). Five cases, recorded on the tree as it
  stood (no non-test change): a job submitted, polled to its end and its event stream replayed; an unknown job's GET and
  DELETE; an OpenAI batch of two lines (upload, create, poll, the output file); an Anthropic message batch with one
  good and one failing request; the admin routes with `--allow-admin` off (403; `NewHandler` refuses that flag, so the
  enabled case is not reachable in process). Normalized as well, and named per golden: `job_`, `batch_`, `batch_req_`
  and `file-` ids, and RFC 3339 or epoch times on `created`, `started`, `finished` and `*_at`. **Racing, recorded as
  `"<racing>"`:** a batch's create response reads the line results its own goroutines are writing (the lines start
  before the response is built), so its `status` and `request_counts` (OpenAI) and `processing_status` and
  `request_counts` (Anthropic) race the lines today; the census masks them on the create step only, and each poll's final
  response is recorded whole. Recording fifteen times gives identical files; `-race -count=5` passes. *Flaky on arm64 after
  merging, fixed (2026-10-27):* `batches_openai`'s two lines sent one prompt and run at once, so whichever ran second
  could reuse the first's session (`prefill_reused_tokens` 19 instead of 0 on root-linux-arm64, failing other PRs).
  Shown by sending the prompt twice in sequence (19), and two different prompts in either order (0 each): each line
  now asks something different (`hi a`, `hi b`), so the record holds whichever finishes first. Plants: a job's
  `status` key renamed fails the job flow, Anthropic's `ended` renamed fails its batch flow, both reverted. 44 cases.

#### SB7 — the engine seam, registered 2026-10-27 19:07 PDT before the code (mbp2-sb6)

Read before registering: every text generation enters `loadedModel.drive`, which picks a token stream (the scripted
seam, the GPU-resident paths, or a CPU session with its speculative variants), runs `streamTokens` over it, and reads
the generation's report (`*decoder.Generation`: budget clamp, logprobs, prefix reuse, KV-share and checkpoint counts,
the error). The report also reaches the request log (`reqTrace.generated`) and `effectiveBudget`. `genRequest` carries
`decoder.SamplingParams` and a `constrain.Masker`. So the interface comes out of the decoder in steps, each its own
commit, each with the census unchanged:

- **E1, the seam.** An unexported `engine` interface in `serveapp`, `start(ctx, gr) (tokens <-chan int, report
  *decoder.Generation, done func())`; `decoderEngine` is `drive`'s stream selection moved verbatim (resident, block
  drafter, n-gram, grammar speculation, sessions and their check-in in `done`), and `scriptedEngine` is the scripted seam
  (a nil report). `drive` keeps registration, jobs, `streamTokens` and the tail, with the scripted tail's
  differences kept exactly (no `cancelled` finish, no logprobs, no reuse, no error). `driveVL` is not touched.
- **E2, the report.** `serveapp`'s own report type, holding the fields the shell reads, in place of
  `*decoder.Generation` on the interface.
  *Amended before E2's code (2026-10-27 19:12 PDT):* the engine returns the report as a function the shell calls at
  each place it read `*decoder.Generation` (the budget check once the stream has closed, the request log, `drive`'s
  tail), so each read happens when it did before. `effectiveBudget` becomes the report's `budgetFor`, and
  `openai_test.go`'s `TestEffectiveBudget_M04` table moves onto it, the same five rows: the one test file in E2's diff,
  which then no longer imports `decoder` (the ratchet falls to 39).
- **E3, the request.** The engine request carries the sampling, constraint and think state as the shell's own types,
  so `engine` names no `decoder` or `constrain` type; the interface is exported for SB7's contract (generate, stream
  events, cancel, stats, models).
  *Amended before E3's code (2026-10-27 19:18 PDT):* `genRequest.sp` becomes `internal/sampling.SamplingParams`, which is
  field for field `decoder.SamplingParams` (decoder converts to it itself), so the decoder calls convert it in place and
  nothing about sampling changes. `genRequest.masker` stays `*constrain.Masker`: `constrain` is not `decoder`, and it is
  the codec's (SB8). Exporting the interface waits for E4, when the fake engine is its second real user. Measured
  while scoping: the shell calls 45 distinct `lm.model` methods at 86 sites in 15 files and names 53 `decoder`
  identifiers, so an `Engine` that owns all of them (models, stats, the vision and audio paths, admin) is several steps
  past E4, and SB7 is nearer L than M.
- **E3b, the logprobs** *(registered before its code, 2026-10-27, mbp2):* `genReport.logprobs`, and what `drive`,
  `driveVL` and `loadedModel.logprobs` pass, become `internal/sampling.SampleInfo`. The two `SampleInfo` structs match
  field for field, but their `Top` element types are distinct named types, so the report copies them (`logprobsOf`),
  keeping nil as nil and empty as empty at both levels. After it `engine` names no `decoder` type. Gate: W5 (the
  census's logprobs case reproduces, and a copy that drops `Top` or turns a nil into an empty slice fails it or the
  table test that pins nil-for-nil) and W6.
  **Landed on its branch (`sb7-e3b-2026-10`, stacked on E4 batch 1).** `logprobsOf` copies the decoder's logprobs;
  `engine` and `genReport` now name no `decoder` type. Read while building: nil and empty never reach a client (both
  renderers build fresh arrays whatever they are given), so the nil-for-nil table test registered above would pin an
  internal property only and was not written. What the census lacked was a `Top` entry (its logprobs case asks for
  none), so a case was added before the code, `chat_logprobs_top1` (one alternative, the forced token), recorded on the
  pre-E3b tree, and W3 now checks logprob `token` values too. **W5:** both logprobs cases reproduce; a copy that drops
  `Top` fails `chat_logprobs_top1`, and one that shifts the id fails both, reverted. **W6:** 96 pass, 9 skip, `-race`
  clean. `touched.py` green; staticcheck clean; `go vet` with `realckpt` and `goinfer_testhooks` clean.
- **E4, the fake engine.** A test helper serving any route on a fake engine, and `serveapp` tests that need no model
  moved onto it in batches; each batch lowers `serveappDecoderTests`.
  *Batch 1, registered before its code (2026-10-27 19:21 PDT):* `b35_keep_test.go` and `kv8aq_gate1_test.go` already
  run on `scriptedEngine` and name `decoder` only for `decoder.KVScope`, the type of the scripted engine's scope hook.
  The request's KV scope becomes the shell's own `kvScope` (the same two fields), `decoderEngine` converts it where it
  hands the decoder `WithKVScope`, and the two tests read the shell's type: the ratchet falls to 37. Their assertions do
  not change.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| W5 per step | `TestWireCensus` passes with no file under `internal/serveapp/testdata/wire/` in the diff; the tiny-model cases run through `decoderEngine` and the scripted cases through `scriptedEngine` (E1 on), shown by a plant in each engine failing its own cases | none | any golden change stops the step with DECISION NEEDED | seconds |
| W6 callers | the `serveapp` tests named for the moved code (scripted, empty-answer, think, tool, KV-boundary, concurrency, session, cancel, job) pass unchanged, with no test file in the step's diff except E4's moves; `touched.py` green | none | a red test is fixed in the step or the step is reverted | `touched.py` seconds; the rest is CI's |
#### SB7 — the model handle, registered 2026-10-27 before the code (mbp2-sb6; the coordinator's option A)

Measured before registering, by retyping `loadedModel.model` to a one-method interface on a scratch tree: 98 compile
errors, 87 of them methods of `*decoder.Model` the shell calls (about 50 distinct, through `lm.model`, `base.model`
and the session LRU's model) and 11 sites that need the concrete type (the liveness map and `retainLocked` /
`releaseLocked`, `newSessionLRU`, `decide.ModelHidden`/`ModelPrefill` and their batch forms, one assignment in `main.go`).
A narrower interface cannot type the field, so the steps are groups of tests, not of methods:

- **H1, the handle.** `modelHandle` in `internal/serveapp/model_handle.go`: every method the shell calls on a model,
  with `var _ modelHandle = (*decoder.Model)(nil)`. `loadedModel.model` and the liveness keys take it; the session LRU keeps its `*decoder.Model`
  (`asDecoder`), so sessions are untouched (*narrowed before the code*). Every assignment goes through `handleOf`, which keeps a nil `*decoder.Model` a nil interface, so each of the
  27 `model == nil` / `!= nil` checks reads as it does today (an interface holding a typed nil is not nil). The sites
  that need the concrete type call `asDecoder`, the handle's `*decoder.Model` or nil. No behavior changes; no test file
  in H1's diff.
- **H2 onward, the tests.** `fakeModel` (a test file: a struct embedding `modelHandle`, implementing only what a test
  needs) replaces the `&decoder.Model{}` sentinels, first in the liveness, unload-drain, tower-close and audio-close
  tests, each batch lowering `serveappDecoderTests`.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| H-census | `TestWireCensus` and `TestSchedCensus_*` pass with no golden in the diff | none | any golden change stops the step with DECISION NEEDED | seconds |
| H-nil | every assignment into a handle field goes through `handleOf` (a source-scan test fails on a bare `*decoder.Model` assigned to one), shown red by a planted bare assignment | none | one run | seconds |
| H-callers | `touched.py` green; the W6 set and the liveness, admin, unload, tower, audio, adapter, session and concurrency tests pass unchanged (H1) or with only the sentinel swapped (H2+); `go vet` with `realckpt`, `goinfer_testhooks` and `-tags cuda`/`metal` builds of the modules that import `serveapp` | none | a red test is fixed in the step or the step is reverted | `touched.py` seconds; the rest is CI's |
- **H1 landed on its branch (2026-10-27, mbp2; `sb7-handle-2026-10`).** `internal/serveapp/model_handle.go`: `modelHandle`
  (`Close` and the 45 methods the shell calls, signatures read from `*decoder.Model` by reflection), `handleOf`,
  `asDecoder`. `loadedModel.model` and the liveness keys are handles; the load path sets the field with `handleOf`;
  `newSessionLRU`, the decision heads (`decide.ModelHidden` and the rest) and the Gemma 4 tower swap take
  `asDecoder`. *Amended in the step:* four test files built `map[*decoder.Model]*modelLiveness`, which a handle-keyed
  map cannot take, so H1 moved them onto `fakeModel` at once (H2's first batch): `audio_close_test.go`,
  `liveness_test.go`, `tower_close_test.go` and `unloaddrain_testhooks_test.go` drop `decoder`, and the ratchet is 33;
  two tagged tests (`batch_cli_parity_test.go`, `prefill_share_testhooks_test.go`) pass `asDecoder(lm.model)` where a
  helper takes the concrete model. **H-census:** the wire and scheduling censuses pass, no golden in the diff.
  **H-nil:** `TestModelHandle_assignedThroughHandleOf` checks every `loadedModel` literal and every `.model`
  assignment on `lm` or `base` (other types here have a string `model` field a scan without types cannot tell apart);
  a bare `model: model` at the load site and a bare `lm.model = m` each fail it, reverted. **H-callers:** the W6 set,
  the censuses and the moved tests: 105 pass, 9 skip; the tagged unload-drain and prefill-share tests pass on the fake;
  the test binary builds under every combination of `realckpt` and `goinfer_testhooks`; `-race` clean; `touched.py`
  green; staticcheck clean, with and without `goinfer_testhooks`.
- **Gemini on #380 and #381, answered (2026-10-27, mbp2; folded into #387, the top of the stack).** `drive` read the
  report as `report()` after `streamTokens`, which reads it through `reportNow` because the engine contract lets a
  report function be nil; an engine returning none would have panicked there. `drive` uses `reportNow` too.
  `loadedModel.fakeEngine` (a test seam: the engine every text generation runs on when set) lets
  `TestDrive_engineWithNoReportFunc` drive an engine with no report function: it finishes as the scripted seam does,
  and it panics with the old `report()` (shown, reverted). The census is unchanged.
- **H2, batch 1 (2026-10-27, mbp2; in #387).** The attached block drafter is a `blockDrafter` (its one method the shell
  calls, `GenerateStream`), set through `drafterOf`; `TestModelHandle_assignedThroughHandleOf` holds that field to the
  same rule. `blockdrafter_test.go` uses `fakeDrafter` and drops `decoder`; `serveappDecoderTests` is 32. 110 pass, 9
  skip, `-race` clean; both censuses unchanged.
- **Gemini on #387, answered (2026-10-27, mbp2).** `retainLocked` and `releaseLocked` take a handle, and a nil
  `*decoder.Model` passed as one would be a non-nil interface, which their `m == nil` test misses. Every caller passes
  `lm.model`, which only `handleOf` sets, so none reaches them today; they are made robust anyway. `nilHandle` reports
  a nil handle or one holding any nil pointer, and both functions treat it as no model (no liveness entry, nothing to
  close), as the nil `*decoder.Model` was before. `TestModelHandle_nilModelIsNoHandle` retains and releases `nil`,
  `handleOf(nil)` and a typed-nil `*fakeModel`; with the old check the typed nil gets a liveness entry and the test
  fails (shown, reverted).
- **#387 red on four platforms, fixed (2026-10-27, mbp2).** `TestServerBanner_routesMatchRegistration` panicked: an
  entry with no model (an embedding-only server) calls `LoadProfile` on its model, which a nil `*decoder.Model`
  answers and a nil handle cannot, so H1 changed behavior for model-less entries. The handle stays nil (the 27
  `model == nil` checks read as before) and every method call goes through `lm.mdl()`, which returns the handle or a nil
  `*decoder.Model`, so a model-less entry answers exactly as before (89 call sites; the one in `publishLoaded`'s
  raced-duplicate branch keeps the text `model.Close()` that `TestAdminLoad_racedDuplicateIsClosed` scans for, and the
  guarded close at unload keeps `lm.model.Close()`). The panic aborted the package's test binary, so CI's pass list
  covered nothing after it: the whole `serveapp` package was run once, untagged and with `goinfer_testhooks`, both
  green. `TestModelHandle_noModelAnswersAsDecoderDoes` pins it (it panics without `mdl`'s nil branch). Not every decoder
  method answers a nil model (`ExpertCacheStats` does not); those stay behind the shell's own nil checks, as before.
- **H2, batch 2 (2026-10-27, mbp2; `sb7-h3-2026-10`).** An image block's span is the shell's `imageSpan` (the same three
  fields as `decoder.ImageSpan`); `imageSpans` and `visionInput.spans` hold it, and `driveVL` converts once, through
  `decoderSpans`, before the vision generators. `pixtral_vision_test.go` reads `imageSpan` and drops `decoder`;
  `serveappDecoderTests` is 31. This batch's production change was covered by H2's registration ("each batch lowering
  the ratchet") and not registered separately before the code, as E4 batch 1's was. The serve-level vision tests that
  would see a lost span field need real checkpoints and skip here (12 skips among 84 vision tests run, 68 pass), so a
  plant dropping the hash in the conversion passed them; `TestDecoderSpans_keepsEverySpan` now pins every field and
  fails on that plant (shown, reverted). The whole `serveapp` package passes; `touched.py` green; staticcheck clean.
- **What the 31 still import `decoder` for (2026-10-27, mbp2), read file by file.** 22 load a real checkpoint on purpose
  (`decoder.Load`, `tinyServed`): integration tests of the shell over a model, which a fake would hollow out. Six read
  or set decoder-owned registries and test hooks the shell reports (the self-test registry, compiled backends, prefill
  fallbacks, host-RAM and weight-allocation hooks). The rest name a decoder type the shell passes through
  (`decoder.Options` built from flags, `DecisionHeadQuant`, `AutoChoice`, `RecommendedSampling`,
  `ErrWontFitResident`); a shell copy of each would move the count without decoupling anything. So the model-handle
  group ends here, and what remains of SB7's contract (an exported `Engine`: generate, stream events, cancel, stats,
  models) is a package extraction, not a test batch: a decision for Francis (`~/wt/sb6-status.md`).

#### SB7b — the exported engine package, registered 2026-10-28 before the code (mbp2-sb6; Francis's option B, 05:01)

Francis chose option B (05:01, relayed): an exported engine package holding the request, the report and the model
handle, which meets SB7's contract in full. Proposed package: `github.com/townsendmerino/goinfer/engine`, in the
Experimental tier of `docs/api-tiers.md` (it may change in any release; apidiff gates only the Hard tier, so the new
package is an addition). Its name and exported surface are Francis's to see, in the first pull request's description.

Steps, each its own pull request, each green on its own:
- **B1, the types.** `engine.Model` (the model handle), `engine.BlockDrafter`, `engine.KVScope`, `engine.Report`
  (the generation's account, with `BudgetFor`), `engine.Request` and `engine.Engine` (`Start`), with `HandleOf`,
  `DrafterOf`, `AsDecoder`, `IsNil` and `LogprobsOf`. The shell aliases its names to them (`modelHandle =
  engine.Model` and so on), so its call sites change only where a report field is read (now exported).
- **B2, the engines.** `drive` builds an `engine.Request` from its `genRequest` and the scripted and decoder engines
  implement `engine.Engine`; the decoder engine moves into the package as far as it can without the session LRU.
- **B3 on, the call sites,** in batches, the shell naming the engine package where it named the moved types.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| X-census | `TestWireCensus` and `TestSchedCensus_*` pass with no golden in the diff | none | any golden change stops the step with DECISION NEEDED | seconds |
| X-imports | what moved is named through `engine` in the shell; `engine` imports no shell package (a layout rule, `engine-below-the-shell`: `engine` imports nothing under `internal/serveapp`, `internal/chatapp` or `cmd`), shown red with a planted import | none | one run | seconds |
| X-api | `scripts/apidiff_check.sh v0.13.0` passes (no Hard-tier change); `engine` is listed in `docs/api-tiers.md`'s Experimental tier; its `doc.go` names its invariants' tests | none | a Hard-tier report stops with DECISION NEEDED | seconds |
| X-ratchet | `serveappDecoderTests` does not rise | none | a rise is fixed in the step | seconds |
| X-callers | `touched.py` green; the whole `serveapp` package passes untagged and with `goinfer_testhooks` (once per step, the shared timing lock held); `go vet` under `realckpt` and `goinfer_testhooks` | none | a red test is fixed in the step or the step is reverted | about 45 s each |
- **B1 landed on its branch (2026-10-28, mbp2; `sb7b-engine-2026-10`).** `engine/` (`doc.go`, `model.go`, `report.go`,
  `engine.go`): `Engine`, `Request`, `Report` (exported fields, `BudgetFor`), `SampleInfo`, `TokenLogprob`, `KVScope`,
  `Model`, `BlockDrafter`, and `HandleOf`, `DrafterOf`, `Call`, `IsNil`, `AsDecoder`, `ReportOf`, `Reporter`,
  `ReportNow`, `LogprobsOf`. The package defines its own `SampleInfo` rather than expose `internal/sampling`'s in a
  public signature. The shell aliases `modelHandle`, `blockDrafter`, `kvScope` and `genReport` to them, its helpers
  delegate, and its unexported engine interface is `genEngine` (the package name took `engine`); `Engine` and
  `Request` have no user yet, which B2 changes. **X-census:** both censuses pass, no golden in the diff. **X-imports:**
  the layout rule `engine-below-the-shell` is in `imports.json`; a planted `engine` → `internal/serveapp` import fails
  it, naming the rule. **X-api:** `scripts/apidiff_check.sh v0.13.0` passes; `engine` is in `docs/api-tiers.md`'s
  Experimental tier and the packages table of `docs/ARCHITECTURE.md`; its `doc.go` names four tests of its own and the
  shell's census. **X-ratchet:** 31, unchanged (`openai_test.go`'s budget table reads the exported fields).
  **X-callers:** the whole `serveapp` package passes untagged and with `goinfer_testhooks`; `go vet` under `realckpt`
  and `goinfer_testhooks`; `touched.py` green; staticcheck clean.
- **B2 landed on its branch (2026-10-28, mbp2; `sb7b-b2-2026-10`, stacked on B1).** The shell's scripted and decoder
  engines implement `engine.Engine` (`Start(ctx, engine.Request)`), and the shell's own interface is gone;
  `loadedModel.fakeEngine` is an `engine.Engine`. `drive` builds the `engine.Request` from its `genRequest`
  (the sampling converted once, where each decoder call converted it before: `gr.sp` does not change in between, so the
  values, function fields included, are the same). The decoder engine stays in `serveapp` for now: it reaches the
  session LRU, the block drafter and the speculative flags on `loadedModel`. **X-census:** both censuses pass, no golden
  in the diff; one token fewer on the decoder engine's default path fails 17 cases and a script one token shorter fails
  7, each reverted. **X-callers:** the whole `serveapp` package passes untagged and with `goinfer_testhooks`;
  `go vet` under both tags; `touched.py` green; staticcheck clean. **X-ratchet:** 31.
- **B3 landed on its branch (2026-10-28, mbp2; `sb7b-b3-2026-10`, stacked on B2).** `engine.Scripted` is a fake
  engine (fixed ids, cut to each request's `MaxTokens`, no report; `OnStart` sees each request), and the shell's
  scripted seam is an `engine.Scripted` whose `OnStart` calls its hooks; the shell's `scriptedEngine` and
  `scriptedStream` are gone. `TestScripted_streamsItsIDsWithinTheBudget` in `engine`, named in its `doc.go`.
  **X-census:** both censuses pass, no golden in the diff; a `Scripted` one id shorter fails 7 cases, reverted.
  **X-api:** apidiff passes; `Scripted` is in the Experimental tier's list. **X-callers:** the whole `serveapp`
  package passes untagged and with `goinfer_testhooks`; `go vet` under both tags; `touched.py` green.
- **B4 landed on its branch (2026-10-28, mbp2; `sb7b-b4-2026-10`, stacked on B3).** The decoder engine is
  `engine.Decoder` (`engine/decoder.go`), its stream selection moved verbatim with its comments: the model (called
  through `Call`), a `SessionPool` for the CPU sessions, the block drafter, and the speculative and adapter flags, with
  the tokenizer as an `Encode` function for the grammar drafter. The session LRU stays in `serveapp`: `sessionPool`
  is it as an `engine.SessionPool`, each operation under `sessMu` exactly as the inline lock and unlock were.
  **X-census:** both censuses pass, no golden in the diff; one token fewer on `Decoder`'s default session path fails
  17 cases, reverted. **X-api:** apidiff passes; `Decoder` and `SessionPool` are in the Experimental tier's list and
  `doc.go`. **X-callers:** the whole `serveapp` package passes untagged and with `goinfer_testhooks`; every tag
  combination builds; `touched.py` green; staticcheck clean.
- **Gemini's review of the engine package, answered in B4 (2026-10-28, mbp2).** `Call` checked only `h == nil`, so a
  typed-nil handle, which `IsNil` counts as no model, came back as itself; it now uses `IsNil` and returns a nil
  `*decoder.Model` for both. `Scripted.Start` sliced `ids[:req.MaxTokens]` and panicked on a negative budget (the HTTP
  routes reject one, but the type is exported); it now cuts to `max(req.MaxTokens, 0)`. `TestHandleOf_nilStaysNil`
  gains a typed-nil case and `TestScripted_streamsItsIDsWithinTheBudget` a budget of -1; each fails without its fix
  (shown, reverted). The censuses and the whole `serveapp` package pass.
- **Gemini on #405, answered (2026-10-28, mbp2).** `Decoder` tested `d.Drafter != nil`, which a typed-nil drafter
  passes, the pitfall `Call` had. `Decoder.drafter` returns nil for a nil drafter or one holding a nil pointer, and the
  resident path asks it. `TestDecoder_typedNilDrafterIsNone` fails with the pointer check disabled (shown, reverted);
  the whole `serveapp` package passes untagged and with `goinfer_testhooks`.
- **E1 landed on its branch (2026-10-27, mbp2).** `internal/serveapp/engine.go`: `engine`, `scriptedEngine`,
  `decoderEngine` (the stream selection, comments and all, moved from `drive`; the session check-in is its `done`).
  `drive` asks `lm.engine()` for the stream and keeps the scripted tail's differences behind `gen == nil`. **W5:** the
  census passes with no file under `testdata/wire/` in the diff; one token fewer in `decoderEngine`'s default path fails
  the 13 tiny-model generation cases, and a script one token shorter in `scriptedEngine` fails 7 scripted cases (the
  other 6 still parse the same call or text from the shorter script), each reverted. **W6:** 103 tests in the 29 test
  files that use the scripted seam or exercise `drive`'s paths (sessions, concurrency, cancel, jobs, think, tools,
  overflow, logit bias, the request log, KV sharing) run: 94 pass, 9 skip (a `.gguf` in an env var, not run on this
  CPU-only box); the same set passes under `-race`. No test file in E1's diff. `touched.py` green; staticcheck clean;
  `go vet` with `realckpt` and `goinfer_testhooks` clean.
- **E2 landed on its branch (2026-10-27, mbp2; `sb7-e2-2026-10`, stacked on the census and E1).** `genReport` in
  `internal/serveapp/engine.go`, the fields the shell reads (budget clamp, logprobs, reuse, KV copy, snapshot and
  checkpoint counts, error), and `engine.start` returns it as a function, read afresh at each place `drive`,
  `streamTokens` and the request log read `*decoder.Generation`. `effectiveBudget` is now `genReport.budgetFor`.
  `driveVL` hands `streamTokens` its generation's report the same way and keeps its own reads of the vision fields.
  **W5:** the census passes with no file under `testdata/wire/` in the diff; a report that drops `prefillReused` fails
  the two prefix-reuse cases, reverted. **W6:** the same 103 tests and `TestEffectiveBudget_M04`, its table moved onto
  `genReport` with the same five rows: 95 pass, 9 skip, `-race` clean. `openai_test.go` no longer imports `decoder`,
  and `serveappDecoderTests` is 39. `touched.py` green; staticcheck clean.
- **E3 landed on its branch (2026-10-27, mbp2; `sb7-e3-2026-10`, stacked on E2).** `genRequest.sp` is
  `internal/sampling.SamplingParams`; the 17 calls into `decoder` (11 in `decoderEngine`, 6 in `driveVL`) convert it in
  place, and the layout data records the new arrow (`internal/serveapp` to `internal/sampling`). `engine` now names
  `decoder` only through `genReport.logprobs` (`decoder.SampleInfo`). **W5:** the census passes with no file under
  `testdata/wire/` in the diff; a conversion that drops `LogitBias` on the default session path fails 16 cases (the
  forced tokens are lost), reverted. **W6:** 118 tests, the W6 set plus the budget, Harmony, logit-bias,
  sampling-default and tool tests: 109 pass, 9 skip, `-race` clean; no test file in E3's diff. `touched.py` green;
  staticcheck clean; `go vet` with `realckpt` and `goinfer_testhooks` clean.
- **E4 batch 1 landed on its branch (2026-10-27, mbp2; `sb7-e4-2026-10`, stacked on E3).** The request's KV scope is
  the shell's `kvScope`; `decoderEngine` builds the `decoder.KVScope` it hands `WithKVScope` from it, and the scripted
  engine's hook takes the shell's type. `b35_keep_test.go` and `kv8aq_gate1_test.go` read `kvScope`, assertions
  unchanged, and no longer import `decoder`; `serveappDecoderTests` is 37. The three B35 tests pass (and under
  `-race`); `TestKV8aQ_gate1Scopes` is an env-gated measurement and skips here, so it is checked by build and vet only.
  The census passes with no golden in the diff, and the W6 set (110 pass, 9 skip). `touched.py` green; staticcheck clean.

#### SB7b — shrinking the engine API, registered 2026-10-29 before the code (mbp2-sb6; Francis, 11:41)

Francis chose (11:41, relayed) to shrink the exported `engine` package before SB7b's vision and audio generators,
after a read-only review of its surface. Four changes, in three pull requests, each green on its own:
- **S1** (with S2, the first PR). `Model`, `BlockDrafter` and the helpers `HandleOf`, `DrafterOf`, `Call`, `IsNil`,
  `AsDecoder`, `ReportNow` and `LogprobsOf` leave the package: the model handle and its helpers go back into
  `internal/serveapp`, `Decoder.Model` is a `*decoder.Model` and `Decoder.Drafter` a `*decoder.BlockSpec`. `ReportOf`
  and `Reporter` stay until S3, whose stream constructors replace them (the shell's vision path reads a
  `*decoder.Generation`'s report, and `Decoder` needs the same conversion, so neither side can own it alone).
- **S2.** `engine`'s `SampleInfo`, `TokenLogprob` and `KVScope` go; `Report.Logprobs` is `[]decoder.SampleInfo` and
  `Request.Scope` a `decoder.KVScope`, which `Decoder` hands the decoder as it is.
- **S3.** `Engine.Start` returns one `*Stream` with `Tokens()`, `Report()` and `Close()`; `NewStream` and
  `GenerationStream` build one, and `ReportOf` and `Reporter` leave the package.
- **S4.** `Scripted` moves to `engine/enginetest`.

Not in scope (Francis did not choose them): renaming `Decoder`, `Spec`, `Adapter` or `PlainConstrained`, and splitting
`Report`.

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| S-census | `TestWireCensus` and `TestSchedCensus_*` pass with no golden in the diff, at each step | none | a golden change stops the step with DECISION NEEDED | seconds |
| S-surface | a test in `engine` lists its exported names and fails on any other; after each step the list is the step's target (S1+S2: `Engine`, `Request`, `Report`, `Decoder`, `SessionPool`, `Scripted`, `ReportOf`, `Reporter`; S3: `Stream`, `NewStream`, `GenerationStream` in, `ReportOf`, `Reporter` out; S4: `Scripted` out, and `enginetest` exports `Scripted` only); shown red with a planted export | none | one run per step | seconds |
| S-imports | `engine-below-the-shell` still holds; `engine` declares no copy of a `decoder` type (S-surface names none) | none | one run | seconds |
| S-api | `scripts/apidiff_check.sh v0.13.0` passes (no Hard-tier change); `docs/api-tiers.md` lists the new surface; `engine`'s `doc.go` names each invariant's test, and each named test exists | none | a Hard-tier report stops with DECISION NEEDED | seconds |
| S-ratchet | `serveappDecoderTests` does not rise | none | a rise is fixed in the step | seconds |
| S-callers | `touched.py` green; the whole `serveapp` package passes untagged and with `goinfer_testhooks` (the shared timing lock held); `go vet` under `realckpt` and `goinfer_testhooks`; staticcheck clean on `engine` and `serveapp` with its canary red | none | a red test is fixed in the step or the step is reverted | about 45 s each |
| S-stream (S3) | a `Stream`'s `Report()` is nil-safe (a nil report function, a nil result, a nil `*Stream`'s report is nil) and `Close()` runs its done function once however often it is called; a test each | none | one run | seconds |
- **S1 and S2 landed on their branch (2026-10-29, mbp2; `sb7b-api-shrink-2026-10-29`).** `engine/model.go` is gone:
  the model handle, the block drafter and their helpers are `internal/serveapp`'s again (`modelHandle`,
  `blockDrafter`, `handleOf`, `drafterOf`, `nilHandle`, `callHandle`, `asDecoder`, `asBlockSpec`; `reportNow` beside
  the engine glue). `Decoder.Model` is a `*decoder.Model`, `Decoder.Drafter` a `*decoder.BlockSpec` (no reflection
  left in `engine`), `Report.Logprobs` is `[]decoder.SampleInfo` and `Request.Scope` a `decoder.KVScope`. The shell
  hands `Decoder` `asDecoder(lm.model)` and `asBlockSpec(lm.blockSpec)`; no test serves a fake model or drafter through
  `Decoder` (a `fakeModel` embeds a nil handle, so one would panic). **S-census:** both censuses pass, no golden in the
  diff. **S-surface:** `TestSurface_exportsOnlyItsContract` lists the eight names; a planted `PlantedExport` fails it,
  reverted. **S-imports:** `engine-below-the-shell` holds; `engine`'s tests no longer import `decoder`, so that test
  arrow leaves `imports.json`. **S-api:** apidiff passes; `docs/api-tiers.md` and `docs/ARCHITECTURE.md` list the new
  surface; `doc.go` names three tests of its own and the census. **S-ratchet:** 31. **S-callers:** `touched.py` green;
  the whole `serveapp` package passes untagged and with `goinfer_testhooks` (43 s each); `go vet` under `realckpt` and
  `goinfer_testhooks`; staticcheck clean, its canary red. `TestModelHandle_noModelAnswersAsDecoderDoes` gains the
  typed-nil and fake cases the deleted engine tests held; with `callHandle` testing only `h == nil` it panics (shown,
  reverted).
- **S3 landed on its branch (2026-10-29, mbp2; `sb7b-stream-2026-10-29`, stacked on S1+S2).** `Engine.Start` returns
  a `*Stream` (`engine/stream.go`): `Tokens()`, `Report()` (nil-safe, for a nil `*Stream`, a nil report function or a
  nil result) and `Close()` (its done function once, under a `sync.Once`). `NewStream` and `GenerationStream` build
  one; `ReportOf` is unexported and `Reporter` gone. `drive` defers `st.Close()` where it deferred `done()`, and hands
  `streamTokens` `st.Tokens()` and `st.Report`; `driveVL` wraps its generation in a `GenerationStream` for the same two
  reads. **S-census:** both censuses pass, no golden in the diff. **S-surface:** nine names, `ReportOf` and `Reporter`
  out. **S-stream:** `TestStream_reportIsNilSafeAndCloseRunsOnce`; with the `sync.Once` removed it fails ("done ran 2
  times"), reverted. **S-api:** apidiff passes; `api-tiers.md` and `doc.go` updated. **S-callers:** `touched.py`
  green; the whole `serveapp` package passes untagged (50 s) and with `goinfer_testhooks` (44 s); `go vet` under both
  tags and `realckpt`; staticcheck clean. **S-ratchet:** 31.
- **S4 landed on its branch (2026-10-29, mbp2; `sb7b-enginetest-2026-10-29`, stacked on S3).** `Scripted` is
  `engine/enginetest.Scripted`, unchanged, with its test beside it; the shell's scripted seam (production code, a test
  seam on `loadedModel`) builds one from there, so `internal/serveapp` imports `engine/enginetest` as well as `engine`.
  `engine-below-the-shell` now covers `engine/enginetest`, and the layout data records its three arrows (`-update`).
  **S-census:** both censuses pass, no golden in the diff. **S-surface:** `TestSurface_exportsOnlyItsContract` reads
  both directories: `engine` exports eight names, `enginetest` only `Scripted`; a wrong `enginetest` list fails it,
  reverted. **S-api:** apidiff passes; `api-tiers.md`, `ARCHITECTURE.md` (a packages-table row) and both `doc.go`
  files updated. **S-callers:** `touched.py` green; the whole `serveapp` package passes untagged and with
  `goinfer_testhooks` (43 s each); `go vet` under `realckpt` and `goinfer_testhooks`; staticcheck clean.
  **S-ratchet:** 31. The shrink is done; SB7b's next step is the vision and audio generators behind the engine.
- **S5, registered 2026-10-29 before the code (mbp2-sb6; Francis, 13:18, relayed).** The renames, while nothing
  outside goinfer uses `engine`: `Decoder` → `Local`, its `Spec` → `Speculative` and `Adapter` → `AdapterBound`, and
  `Request.PlainConstrained` → `NoSpeculation`; one PR before the vision and audio generators. `Report`'s split stays
  out of scope. Gates are S-census, S-surface (target: `Local` in place of `Decoder`), S-api (`api-tiers.md` and every
  live doc naming the old names), S-ratchet and S-callers, as above, plus **S-names**: no Go file and no live doc
  outside `docs/completed/`, `changelog.d/` fragments already written and dated records names `engine.Decoder`,
  `PlainConstrained` or the two old fields (a grep, shown empty).
- **S5 landed on its branch (2026-10-29, mbp2; `sb7b-renames-2026-10-29`).** `engine/decoder.go` is `engine/local.go`
  and `Decoder` is `Local`, with `Speculative` and `AdapterBound`; `Request.NoSpeculation` replaces `PlainConstrained`
  (its comment says it is read on a CPU session, since the resident path never speculates a constrained request and
  goinfer_confidence always carries a masker). **S-census:** both censuses pass, no golden in the diff.
  **S-surface:** the list names `Local`. **S-api:** apidiff passes; `api-tiers.md` and `ARCHITECTURE.md` name `Local`.
  **S-names:** the grep is empty. **S-callers:** `touched.py` green; the whole `serveapp` package passes untagged
  (50 s) and with `goinfer_testhooks` (52 s); `go vet` under `realckpt` and `goinfer_testhooks`; staticcheck clean.
  **S-ratchet:** 31.

### SB8 — the conversation codec

- **Today:** `chat` (templates), `tokenizer`, `constrain`, and tool-call parsing in the serving package.
- **Contract:** messages to tokens to tool calls and back, per family template.
- **Fast check:** golden renders and round trips per template. Its open contract question is
  `docs/tasks/task-tool-grammar-union-2026-09.md`'s: two or more tools under `tool_choice` auto. S–M.

#### SB8 — the codec census, registered 2026-10-27 14:25 PDT before the code (mac-ax01)

Francis's decision (14:16): census first, recorded at `origin/main`; the move to one package must reproduce every golden
byte for byte; tool-call parsing gets its own table of model outputs and parsed calls, malformed ones included; nothing
about what a model sees may change (a change stops the work with DECISION NEEDED).

The census is a test in `chat` (it pins the clock Llama 3's and Harmony's date lines read, which only the package can) that
calls only the API the serving routes call, in the order they call it. It adds test files and goldens under
`chat/testdata/codec/` and changes no other file, so what it records is `origin/main`'s code.

- **Template sources.** Every distinct chat template the repository tracks (`testdata/chat_goldens/` and
  `testdata/chat_think_goldens/`), every chat template a local model under `~/models` carries that `Detect` accepts
  (snapshotted into the census's testdata, so CI renders them too), and each family's constructor as the bare-checkpoint
  fallback returns it. Each source is resolved through `Detect` exactly as a load does.
- **Variants per source:** each `ThinkMode` where the template has a reasoning spec, each `ToolFormat` where it declares a
  native tool form, each reasoning effort for Harmony.
- **Conversations,** fixed: system plus user; user alone; multi-turn with non-ASCII text; a replayed assistant reasoning
  turn; tools offered; a tool call and its result; two calls and two results; a finished tool loop and a follow-up; a user
  message carrying the family's own control-token text (it must stay text); an image placeholder where the family has a
  block (Gemma 3, Gemma 4, Qwen-VL ChatML, GLM-OCR, Pixtral, LFM2-VL) and an audio placeholder for Gemma 4, built and
  spliced as `serveapp` builds them.
- **Recorded per case:** the segments (text and the Special flag), the rendered string's equality with their concatenation,
  the family name, its stop strings, its tool-call opener, whether the prompt opens a think block; and, with the source's
  tokenizer, the token IDs (`EncodeSegments`, no added BOS, as the routes encode), the stop strings' IDs, and whether the IDs
  decode back to the rendered text.
- **Tokenizers.** Tracked ones (Qwen2.5-Coder's GGUF metadata, Mistral v0.3's `tokenizer.json`) are checked everywhere; a
  local one is named by its `~/models` path and SHA-256: present with that hash, its IDs are checked; present with another,
  the test fails; absent, the test logs a named skip and prints how many ID checks ran and how many were skipped. Llama 3,
  Mellum 2 and Harmony tokenizers are not on the Mac: their IDs are recorded on nobara by the same `-update`, which keeps
  recorded IDs whose tokenizer is absent, and that commit's diff must add IDs only.
- **Tool-call parse table** (its own golden): about 40 model outputs across every family's dialect (Hermes JSON, Qwen 3.5
  XML, Mistral, Llama 3, Gemma 4, Harmony, fenced, bare), well-formed and malformed (bad JSON, unterminated, unknown tool,
  arguments of the wrong type or not an object, a call inside a think block, prose that mentions the opener), each through
  `ParseToolCalls` and `ParseToolCallsFor` with a fixed tool list, for each family and tool format, lenient off and on.
- **Tier 0 throughout.** The goldens are JSON, gzipped if one exceeds 1 MB. **Void:** none (deterministic). **Shown red:** a
  planted one-byte change to a template and to a parser must fail the census, then be reverted. **Stopping rule:** a census
  that differs from its golden after the move stops with DECISION NEEDED; there is no tolerance. **Cost basis:** seconds;
  the Gemma tokenizers load in about a second each.
- **The move** (one package for the codec) is registered after the census lands, with its package shape; `chat`,
  `tokenizer` and `constrain` are in apidiff's Hard tier, so a shape that changes their exported API is the owner's call.
- **Census recorded (2026-10-27, the Mac, at `origin/main` `92a64e92`).** `chat/codec_census_test.go`:
  `TestCodecCensus`, one golden per source under `chat/testdata/codec/sources/` (gzipped JSON, 4.8 MB as text, 0.2 MB
  stored), and `TestToolCallParseCensus`, `chat/testdata/codec/tool_call_parse.json`. 55 sources: 19 tracked templates,
  5 tracked tokenizers, 20 local models, 11 constructors; 1,780 render cases and 715 token-ID checks on the Mac, of which
  40 (the tracked tokenizers) run anywhere, the rest only where the model is on disk. Without the local models every
  render is still checked: Detect and the renderers ask the vocabulary about 6 to 13 markers per source, and the answers are
  recorded with it. The vocabulary fingerprint is a SHA-256 over every token's text in ID order, not the file's hash: a
  GGUF is the weights too. It takes about 3 s with the local tokenizers and under 1 s without. Recording twice gives
  identical files. **Shown red:** a one-byte change to Gemma 3's generation prompt fails 5 sources (the tracked template,
  the constructor, 3 local models); dropping the lead's trim in the Hermes parser fails the parse table. Both reverted.
  What the census shows today, recorded as it is, unchanged: Phi-3's GGUF tokenizer does not round-trip (its IDs decode
  without the newline after each control token); `granite-dbrx-meta.gguf`'s template is declined (raw-completion
  fallback). **Owed:** the Llama 3, Mellum 2 and Harmony token IDs, recorded on nobara by
  `go test ./chat -run TestCodecCensus -codec.update`, whose diff must only add sources and IDs.
  Each source renders the nine conversations above plus its family's image and audio placeholders, 9 to 11 cases per
  variant. Token IDs are on the 24 sources whose tokenizer is there and whose template `Detect` accepts (the tracked
  tokenizers but `granite-dbrx-meta`, and the local models); the tracked templates and the constructors carry renders only.
- **Gemini's review of #338, answered (2026-10-27, mac-ax01).** The vocabulary probes are now checked and recorded on
  every return, including a template `Detect` declines. `-codec.update` runs the local-model guard (vocabulary hash and
  chat template) on a model recorded before, so a different quant or template at the same `~/models` path fails instead
  of replacing another machine's record. A tokenizer that fails to load for any reason but a missing file fails the
  census. Both guards were shown red in check and update mode, with a corrupt `tokenizer.json` and another model's
  tokenizer under a recorded name, and no golden was rewritten. The parse table adds the five tracked templates that
  declare a native tool form, each under the three tool formats (37 entries, from 22). It records `supports_tools` and
  asserts the contract directly: a family without tool support reads no call from anything (Gemma 3, GLM-OCR, both
  Phi-3s, and Ministral, whose `[TOOL_CALLS]name[ARGS]` form `Ministral()` documents as not wired); each family with it
  reads every well-formed call of its own dialect; no family reads a call from prose, a malformed call, an empty or
  missing name, a Harmony final answer, or Ministral's form; and the tool format does not change a parse. The 22 earlier
  entries are unchanged, row for row. **Shown red:** dropping `mistral` from `SupportsTools` fails the contract;
  accepting an empty name fails it for ChatML and Llama 3. That second plant passed before the review, because no output
  had an empty name, so the table gained one (43 outputs). Both reverted.
- **nobara's token IDs, recorded (2026-10-27 15:57 PDT).** On nobara (Go 1.27.3 linux/amd64), from a detached worktree
  of `16e21d7f`, under the shared timing lock (load 1.3; the lock had been free since 15:56):
  `go test ./chat -run 'TestCodecCensus$' -codec.update -codec.models '^(llama-3\.2-1b-instruct|mellum2|gpt-oss-20b)'`.
  `-codec.models` limits discovery to the owed families; nobara holds about 80 models. The run added five sources and
  changed no recorded file: `llama-3.2-1b-instruct-q4_k_m.gguf` (Llama 3), `mellum2-unq` and `mellum2.1-thinking`
  (Mellum 2, the second with its four think modes), and gpt-oss-20b as both its HF `tokenizer.json` and its MXFP4 GGUF
  (Harmony, three efforts). That makes 108 token-ID cases, every one round-tripping. The same run also checked 680 of
  the Mac's recorded IDs against nobara's copies of the 18 Mac models it also holds: their vocabulary, template and
  every ID matched. The census has 60 sources; the Mac checks 715 IDs and skips the 108 recorded on nobara, and renders
  all 60.

#### SB8 — the move to `internal/codec`, registered 2026-10-27 16:10 PDT before the code (mac-ax01; Francis's option 1)

Francis chose option 1: a new `internal/codec` owns the path from messages to token IDs and back, with `chat` and
`tokenizer` as its parts, their public APIs unchanged. Three pull requests, each green on its own:

- **C1, the encode half.** `internal/codec` gains template detection from a tokenizer (the one `Detect` call the
  routes make), a prompt's segments (with tools or without), the image and audio block splice (one block or several),
  and their encoding (`EncodeSegments` with no added BOS, as every route encodes). No caller moves. The census gets a
  seam for its render, splice and encode steps; `TestCodecCensus_throughCodec` (an external test of `chat`, so it can
  import the codec) runs the whole census through `internal/codec` against the same goldens.
- **C2, the callers' encode path.** `serveapp` (the text, tool, overflow and vision routes), `chatapp` (text, image,
  batch) and `internal/decide` call the codec where they composed `Detect`, render, splice and encode themselves.
  `serveapp`'s carried-ID regions (`encodeRegions`) and its KV-boundary encodes encode pieces of a prompt, not a prompt,
  and stay where they are.
- **C3, the reply half.** The codec gains the reply reader: reasoning splitting (`ReplySplitter`), settling a finished
  turn's tool calls (`settleCalls`: the inline parse, a Harmony router's calls, a call read from the reasoning when there
  is no answer) and reconciling streamed prose (`reconcileProse`), moved from `serveapp`; the census's reply and
  parse-table steps get the same seam, and `serveapp` and `chatapp` call the codec.
- **Out of scope, by design.** `demo/agent` (its own module: a standalone build resolves the root at its last tag, which
  has no `internal/codec`) and `examples/` (they show the public API) keep calling `chat` and `tokenizer`.

Gates, per pull request:

| gate | rule | void | stopping rule | cost basis |
|---|---|---|---|---|
| G-CX-goldens | `TestCodecCensus` and `TestToolCallParseCensus` pass with no file under `chat/testdata/codec/` in the diff | none: deterministic | any golden change stops the move with DECISION NEEDED | seconds |
| G-CX-arm | the census through the codec passes on every source, variant and case the census holds (renders, IDs, and from C3 the replies and the parse table), and a planted change in the codec (an added BOS) fails it, then is reverted | none | as above | seconds |
| G-CX-api | no non-test file under `chat/`, `tokenizer/` or `constrain/` changes; CI's `apidiff_check.sh` reports nothing new | none | a change stops with DECISION NEEDED | seconds |
| G-CX-callers (C2, C3) | each replaced call site composes the same calls in the same order as before (read in review against the codec's body); `touched.py` green; CI's tests of `serveapp`, `chatapp` and `decide` pass unchanged, no golden of theirs in the diff | none | a red test is fixed in the move or the move is reverted, never a golden re-recorded | `touched.py` seconds; the rest is CI's |

- **C1 landed on its branch (2026-10-27, the Mac).** `internal/codec/codec.go`: `Template`, `Segments`,
  `ImageSegments`, `ImagesSegments`, `Encode`, `EncodePrompt`. The census's render, splice and encode steps go through a
  `censusPath`; `directPath` makes the same calls as before, and `chat`'s `TestCodecCensus_throughCodec` (an external
  test, `chat/census_through_codec_test.go`, reaching the census through `chat/export_test.go`) runs it through the codec.
  **G-CX-goldens:** pass, no golden in the diff. **G-CX-arm:** pass on all 60 sources, with the local models (715 IDs)
  and without (renders only); an added BOS in `codec.Encode` fails 16 sources (those whose tokenizer has a BOS token),
  reverted. **G-CX-api:** no non-test file under `chat/`, `tokenizer/` or `constrain/` changed. The layout data records
  the new arrows (`internal/codec` to `chat`, `multimodal`, `tokenizer`; `chat`'s test to `internal/codec`), and
  `scripts/touched_scans.py` runs the arm when `internal/codec`, `tokenizer` or `multimodal` changes.
- **C2 landed on its branch (2026-10-27, the Mac).** Eleven sites call the codec. In `serveapp`: `promptForT`,
  `encodePrompt`'s render and whole-prompt encode, the tools route, `fixedPromptTokens`, `encodeVisionSegments`,
  `imagesPrompt` and the model load's `Detect`. In `chatapp`: the session's `Detect`, the batch encode and the image
  prompt. In `decide`: `NewPlainTokenizer` and `EncodeChat`. Each replaced expression is the codec function's body
  (`RenderToolsSegments` with no tools delegates to `RenderSegments`, so `Segments` is the same call where the tools
  route passed none). **G-CX-callers:** `touched.py` green; the `serveapp` tests named for the changed paths (vision,
  tools, overflow, carried IDs, think, prompt, image, audio, and the per-family image tests) pass, 167 run and 11
  skipped (real checkpoints or env-gated assets); no test or golden changed. **G-CX-api:** still no non-test change under
  `chat/`, `tokenizer/` or `constrain/`.
- **C3 landed on its branch (2026-10-27, the Mac).** `internal/codec/reply.go`: `NewReplySplitter`, `SettleCalls`,
  `ReconcileProse`, `ErrProseDiverged`. `serveapp`'s `settleCalls` and `reconcileProse` are one-line delegates (its tests
  call them), `errProseDiverged` is the codec's error, and the three reply-splitter constructions (`newThinkOut`,
  goinfer-chat's turn and batch) call the codec. `SettleCalls` takes the router's calls and reasoning as values: the
  getters it replaces (`thinkOut.toolCalls`, `reasoningText`) return none for a nil router, which leaves the fallbacks
  unreachable exactly as the `th != nil` test did. The census's reply and settle steps get the seam;
  `TestToolCallParseCensus_throughCodec` checks the settle column through `codec.SettleCalls` with no router.
  **G-CX-arm:** all four census tests pass, with the local models and without. Shown red, each reverted: settling without
  the tool list fails the parse arm; no reply splitter fails the census arm; an untrimmed lead on a routed (Harmony) call
  is outside the census (it has no router) and fails `serveapp`'s `TestSettleCalls_harmony`. **G-CX-callers:**
  `touched.py` green; `serveapp`'s reply, tool, think and prose tests pass, 73 run and 5 skipped. **G-CX-api:** no
  non-test change under `chat/`, `tokenizer/` or `constrain/`.

### SB9 — the towers

After AX2: each image and audio tower with per-tower goldens (output bytes on fixed inputs), and later its blocks
expressed as programs, so SB1's kit covers them on every backend. S, then M.

#### SB9's gates, step S (the goldens), registered 2026-10-27 before any code (nobara-mm01; Francis's pick of 20:28)

**The instrument is AX2's.** `internal/towerid` runs the twelve tower families on their committed tiny fixtures with fixed
synthetic inputs (seeded PCG), and fingerprints each output as the SHA-256 of its float32 bits. Run on CPU it gives 22 cases.
On a device it runs only the families that backend declares, through the production registries; a declared family that
does not attach is an error. `towerid.Check` already requires two runs to agree bit for bit. Step S commits each backend's
record as a golden, and grades every later run against it.

- **Where the goldens live:** `internal/towerid/testdata/towers.<backend>-<goos>-<goarch>.json`. Each holds the record and
  its **anchor**, what its bits depend on:
  - CPU: `kernelprint.CurrentConfig()`, which is GOARCH, linalg's active kernels and any forced fallbacks;
  - CUDA: the GPU's name and the NVIDIA driver version (nvidia-smi);
  - WebGPU: the adapter's backend, name and driver.
- **Tolerance: exact on every backend, case by case** (family, input, device attached, value count, SHA-256). Exact is
  justified because `Check` fails first on any backend whose two runs differ. AX2's records also agree to the bit across
  four revisions on CPU and CUDA.
- **A different anchor:**
  - CPU: a named skip, printing the record to commit. A CI runner's kernel set is a different configuration, not a
    regression.
  - CUDA or WebGPU: a **failure** naming both anchors. A driver or adapter change is re-anchored deliberately
    (`-towerid.update`, reviewed), never carried forward silently (CLAUDE.md, the driver rule).
- **Metal and the darwin records:** a named skip until a Mac session records `cpu-darwin-arm64`, `metal-darwin-arm64` and
  `webgpu-darwin-arm64`.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9-1 goldens | on nobara, the CPU, CUDA and WebGPU records equal their committed goldens exactly, under the golden's anchor | a timed run holds the card | runs to completion | each record under a second (AX2a) |
| G-SB9-2 each golden can go red | for every family in a golden, a planted one-ulp change to that family's first output value (a `towerid` test seam) turns exactly that family's cases red and leaves every other case equal; on each of the three backends | as G-SB9-1 | runs to completion | one run per family: seconds |
| G-SB9-3 the backends agree with the CPU | for each device case, against the CPU's output on the same input: cosine and max abs difference over the CPU output's RMS. **Expected: cosine ≥ 0.999 and the relative max difference ≤ 0.05.** The towers are float32, and a device reorders sums and may hold some activations in f16, so exact equality is not expected. A case outside the bound is a **finding**, recorded with DECISION NEEDED. The bound is not moved to fit it | as G-SB9-1 | runs to completion | seconds |
| G-SB9-4 builds | vet and the pinned staticcheck under each module's tags, gofmt, `touched.py`, the citation lint; a `changelog.d` fragment | — | runs to completion | minutes |

The towers' outputs do not change in this step. GPU runs are seconds, and they run before nobara-sb01's night window
(22:00 to about 00:30) under the timing lock held shared.

**Step S read, 2026-10-27 (nobara: Ryzen 7 3700X, RTX 2070 SUPER, NVIDIA driver 595.91.07, Go 1.27.3): all four pass.**
- **G-SB9-1:** three goldens are committed under `internal/towerid/testdata/`. Each passes on a run without `-update`.

  | golden | anchor | families | cases |
  |---|---|---:|---:|
  | `towers.cpu-linux-amd64.json` | `amd64[avx2,f16c,popcnt]` | 12 | 22 |
  | `towers.cuda-linux-amd64.json` | `cuda NVIDIA GeForce RTX 2070 SUPER, 595.91.07` | 8 | 17 |
  | `towers.webgpu-linux-amd64.json` | `webgpu vulkan NVIDIA GeForce RTX 2070 SUPER (595.91.07)` | 6 | 15 |

  - The CPU golden equals nobara's AX2d CPU records at all four revisions (0 of 22 differ), and the CUDA golden equals the
    AX2d CUDA records (0 of 17), so the towers have not moved since AX2.
  - The anchor rules are shown both ways. A CUDA golden with another driver fails and names both anchors. A CPU golden with
    another kernel set skips as an unrecorded configuration.
  - CI's runners have other kernel sets and platforms, so CI's CPU run skips there. The CPU goldens are graded on nobara and,
    once recorded, on the Mac.
- **G-SB9-2:** every family's one-ulp plant turned exactly its own cases red and left every other case equal: 12 families
  on the CPU, 8 on CUDA, 6 on WebGPU.
- **G-SB9-3:** every device case agrees with the CPU at cosine 1.000000000 (to nine places) on both backends. The largest
  max|d|/rms values are 1.76e-4 (CUDA Whisper) and 1.03e-4 (CUDA Voxtral); every vision tower is between 3.4e-7 and 5.0e-6.
  The registered bound is 0.05. No finding.
- **G-SB9-4:** vet under each module's CI tags, staticcheck, gofmt, `touched.py`, the layout data (`internal/towerid` now
  imports `internal/kernelprint` for the CPU anchor), the citation lint; a `changelog.d` fragment.
- **Metal and the darwin goldens:** a named skip. A Mac session records `cpu-darwin-arm64`, `metal-darwin-arm64` and
  `webgpu-darwin-arm64` with `-towerid.update` and reviews them. The `metal` module still runs its identity check only,
  until a Mac session adds the golden test there.

#### SB9 step S on the Mac: gates, registered 2026-10-28 before the code (mac-vk5grade; the coordinator's job, Francis 09:47)

The MacBook (M1 Pro, macOS 26.6.2) is on battery during the power cut, so these are correctness runs only: no timed run, tiny
fixtures, each run seconds. They are step S's gates, applied to Metal and to the darwin CPU.
- **The Metal golden:** `towers.metal-darwin-arm64.json`, from a new `TestTowerGolden_metal` in `metal/`.
  - **Its anchor** is the device's name and the macOS version and build (`metal Apple M1 Pro, macOS 26.6.2 (25G83)`). Metal's
    shader compiler and driver ship with the OS, so the OS build is Metal's driver version.
  - **A different anchor fails**, naming both, as on CUDA and WebGPU. A macOS update is re-anchored deliberately
    (`-towerid.update`, reviewed).
  - **Tolerance: exact**, case by case, as on every backend: `Check` fails first when two runs differ.
  - CI's `metal-darwin` job runs only its device-free tests (the runner's Metal layer crashes), so the Mac is the only place
    this golden is graded.
- **The darwin CPU golden:** `towers.cpu-darwin-arm64.json`, from `TestTowerGolden_cpu` as it is. Its anchor is the kernel
  configuration, and another configuration skips. Exact.
- **The darwin WebGPU golden is not in this step.** CI's `gpu-darwin` job runs `TestTowerGolden_webgpu` on the runner's
  virtual adapter. A golden anchored to the M1 Pro would then fail every PR under the registered rule, a different adapter
  fails. That rule is Francis's to change (DECISION NEEDED, in this step's read), so this golden stays a named skip.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9-1M goldens | on the M1 Pro, the Metal and darwin CPU records equal their committed goldens exactly, under each golden's anchor; a Metal golden with another anchor fails and names both | another session's timed run holds the Mac | runs to completion | each record under a second |
| G-SB9-2M plants | each family's one-ulp plant turns exactly its own cases red and leaves every other case equal, on Metal and on the darwin CPU | as G-SB9-1M | runs to completion | seconds |
| G-SB9-3M Metal against the CPU | every Metal device case against the CPU, at G-SB9-3's bound (cosine ≥ 0.999, max abs difference over the CPU's RMS ≤ 0.05). A case outside it is a finding, DECISION NEEDED; the bound is not moved | as G-SB9-1M | runs to completion | seconds |
| G-SB9-4M builds | vet under `metal`'s CI tags, the pinned staticcheck on `metal/` with its canary, gofmt, `touched.py`, the citation lint; a `changelog.d` fragment | — | runs to completion | minutes |

**Step S on the Mac read, 2026-10-28 (M1 Pro, macOS 26.6.2 build 25G83, on battery, Go 1.27.2): all four pass.**
- **G-SB9-1M:** two goldens are committed under `internal/towerid/testdata/`. Each passes on a run without `-update`.

  | golden | anchor | families | cases |
  |---|---|---:|---:|
  | `towers.cpu-darwin-arm64.json` | `arm64[dotprod]` | 12 | 22 |
  | `towers.metal-darwin-arm64.json` | `metal Apple M1 Pro, macOS 26.6.2 (25G83)` | 7 | 17 |

  - The darwin CPU golden has the same 22 cases and value counts as `towers.cpu-linux-amd64.json`, and none of its hashes are
    equal: the CPU reference is bit-identical within an architecture, not across them (arm64's fused multiply-adds).
  - CI's `root-darwin` runners report the same kernel configuration, so they grade this golden too. The arm64 kernel table
    recorded on this Mac already passes there (`TestCompute_table`). If a runner's bits differ anyway, that is a finding.
  - A Metal golden anchored to another macOS build (26.6.1) fails and names both anchors. A golden with one changed hash
    (`glm-ocr [[1 4 4]]`) fails on that case alone.
- **G-SB9-2M:** each family's one-ulp plant turned exactly its own cases red and left every other case equal: 12 families
  on the CPU, 7 on Metal.
- **G-SB9-3M:** every Metal case agrees with the CPU, at cosine 0.999999968 or better. The largest max|d|/rms values are
  8.0e-4 (SigLIP) and 4.9e-4 to 6.4e-4 (the Qwen3 tower). Qwen2.5-VL, GLM-OCR and the Gemma 4 towers are between 3.4e-7 and
  5.4e-6, and Qwen3's deepstack is 6.6e-5 to 7.0e-5. The registered bound is 0.05. No finding.
- **G-SB9-4M:** vet on `metal/` (untagged and `goinfer_testhooks`); the pinned staticcheck (v0.8.1) under
  `goinfer_testhooks` clean, its canary red (U1000); gofmt; `touched.py` green; the citation lint; a `changelog.d` fragment.
- **Decided (Francis, 2026-10-28 12:03, relayed): option (a), below.** The question was the darwin WebGPU golden. CI's `gpu-darwin` job runs `TestTowerGolden_webgpu` on the runner's
  virtual adapter, so a golden recorded on the M1 Pro fails every PR under the rule that a different adapter fails. The
  options:
  - (a), recommended: on CI only (`GITHUB_ACTIONS`), a WebGPU adapter with no golden of its own skips, naming both anchors.
    On a developer's machine it still fails. The M1 Pro's golden is then recorded.
  - (b) no darwin WebGPU golden: WebGPU's goldens stay graded on nobara only.
  - (c) record a golden for the runner's adapter as well. The runner image changes without notice, so this golden would go
    red whenever GitHub updates the image.


**The darwin WebGPU golden, option (a): gates, registered 2026-10-28 before the code (mac-vk5grade).**
- **The rule:** WebGPU's golden takes a third behavior on another anchor. On CI (`GITHUB_ACTIONS=true`) it skips, naming
  both anchors and the decision. Anywhere else it fails, naming both, as before. The CPU's skip and the CUDA and Metal
  failures do not change.
- **The golden:** `towers.webgpu-darwin-arm64.json`, recorded on the M1 Pro with `-towerid.update` and reviewed. Its anchor is
  the adapter's (`AdapterAnchor`). Exact.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9-1W golden | on the M1 Pro, the WebGPU record equals `towers.webgpu-darwin-arm64.json` exactly, under its anchor | another session's timed run holds the Mac | runs to completion | under a second |
| G-SB9-2W plants | each WebGPU family's one-ulp plant turns exactly its own cases red | as G-SB9-1W | runs to completion | seconds |
| G-SB9-3W against the CPU | every WebGPU case agrees with the darwin CPU within G-SB9-3's bound; a case outside it is a finding | as G-SB9-1W | runs to completion | seconds |
| G-SB9-5 the CI-only skip | a test calls `CheckGolden` on a golden anchored to another adapter: with `GITHUB_ACTIONS=true` it skips and names both anchors; unset, it fails and names both; the skip-on-other-anchor and fail-on-other-anchor modes behave as before under both settings. Mutation: skipping whatever the environment turns it red | never | one run | seconds |
| G-SB9-4W builds | vet and the pinned staticcheck under `gpu goinfer_testhooks`, gofmt, `touched.py`, the citation lint; a `changelog.d` fragment | — | runs to completion | minutes |

**Read 2026-10-28 (the M1 Pro, on battery): all five pass.**
- **G-SB9-1W:** `towers.webgpu-darwin-arm64.json`, anchored `webgpu metal Apple M1 Pro ()`, holds 15 cases over 6 families, the
  same cases as the linux WebGPU golden, and passes without `-update`. Wgpu reports no driver string on Metal, so a macOS
  update that changes the bits fails as a hash mismatch, not an anchor mismatch.
- **G-SB9-2W:** each of the 6 families' plants turned exactly its own cases red.
- **G-SB9-3W:** every case agrees with the darwin CPU at cosine 1.000000000; the largest max|d|/rms is 4.8e-6 (GLM-OCR).
- **G-SB9-5:** `TestCheckGolden_otherAnchor` passes all seven cases. A mutation that skips whatever the environment turns it
  red. On this Mac, a golden with a changed anchor fails; under `GITHUB_ACTIONS=true` it skips, naming both anchors.
- **G-SB9-4W:** vet and the pinned staticcheck under `gpu goinfer_testhooks` and on `internal/towerid`, gofmt, `touched.py`
  green, the citation lint; a `changelog.d` fragment.

#### SB9 step M (the towers as programs): scoping, 2026-10-28 before any code (nobara-mm01; Francis 17:26)

**What exists today.**
- **The CPU towers** are twelve hand-written forwards in `internal/vision` and `internal/audio`, about 5,800 lines.
- **Each backend has one tower op layer.** CUDA's `cuda/tower_base.go:towerOps`, WebGPU's `gpu/internal/kernels/tower_ops.go:towerOps`
  and Metal's `metal/embeddinggemma2.go:eg2Ops` each give LayerNorm, RMSNorm, GEMMs with bias and bias+residual epilogues,
  GELU, the RoPE variants, clamps and scaled bidirectional attention.
- **Each backend orders those ops by hand, per family:**
  - CUDA: `grid_vision.go`, `qwen25_vision.go`, `siglip_vision.go`, `gemma4_vision.go`;
  - WebGPU: `grid_vision.go`, `qwen25_vision.go`, `gemma4_vision.go` and SigLIP's `vision_encoder.go` under `gpu/internal/kernels`;
  - Metal: `grid_vision.go`, `gemma4_vision.go`, `vl_towers.go`.
  
  That is three copies of each family's block order. The grid tower is already one parameterised block on CUDA, serving
  Qwen3, GLM-OCR and Gemma 3's SigLIP (one segment, no RoPE).
- **The blocks fall into four shapes:**
  - (1) the pre-norm ViT block: SigLIP, Qwen3, GLM-OCR, Pixtral, SigLIP2 NaFlex;
  - (2) Qwen2.5-VL's window-attention block with RMSNorm and SwiGLU;
  - (3) Gemma 4 vision's sandwich-norm block with clamps and 2-D RoPE;
  - (4) the audio encoders: Whisper, Voxtral, Qwen3-ASR and Gemma 4's conformer.
- **What is missing, all of it outside SB9:**
  - SB1's op conformance kit is not built (SB1 is in step 1, moving tests).
  - OP4 (residents walk the program) has not started.
  - The decoder's `program` is unexported in `decoder`.
  - `internal/program`, §9's home for the operation list, does not exist.
  - The `towers-standalone` layout rule (SB13) forbids `internal/vision` and `internal/audio` to import any goinfer
    package, so they cannot import `internal/program` without a rule change.

**What step M would do.** It would give each tower block one description: a list of operations with their parameters,
built from the tower's config. The CPU and each backend would walk that list over their existing op layer, in place of
the per-family orderings. Each walker calls the same op-layer methods, with the same fusions (`gemmBiasAdd` and the
others), in the same order as today, so every golden holds bit for bit. A fusion appears in the program as its own
operation, never folded in by a walker.

**Order, if it goes ahead.**
- Shape (1) first, on Gemma 3's SigLIP: the simplest block. It is declared on CUDA, WebGPU and Metal, and its CPU forward
  is `internal/vision/encoder.go`. Then Qwen3 and GLM-OCR on the same block.
- Then shapes (3) and (2).
- Audio last: only Whisper (CUDA) and Gemma 4 audio (Metal) have device paths.
- Each block is one PR, as follows:
  - the program and the CPU walker, graded on the CPU goldens;
  - then CUDA's and WebGPU's walkers on nobara, graded on their goldens;
  - then Metal's on the Mac.

**Gates, per block (registered here; the per-block section names its families).**

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9M-1 goldens | every case of the block's families equals its committed golden exactly, on the CPU and on each backend whose walker changed (nobara: CPU, CUDA, WebGPU; the Mac: Metal and darwin CPU); every other family's cases are unchanged | a timed run or a memory test holds the card (the coordinator's check of 17:45: no compute app but the desktop's) | runs to completion | seconds a backend (step S) |
| G-SB9M-2 the program can go red | three plants on the block's program, each turning that family's cases red and leaving every other family equal: an operation dropped, two operations swapped, one parameter (a norm epsilon) changed by one ulp | as G-SB9M-1 | runs to completion | seconds |
| G-SB9M-3 one order | the replaced per-family orderings are deleted, not kept beside the walker: no file both builds a program for a family and calls that family's op-layer methods in sequence | — | one read of the diff | — |
| G-SB9M-4 builds | as G-SB9-4, under each changed module's tags, with staticcheck's canary; `TestLayout_arrows`; a `changelog.d` fragment | — | runs to completion | minutes |

**Choices that are Francis's, found by the scoping (DECISION NEEDED; no code until answered).**
1. **Where the tower program lives.** This sets the operation vocabulary SB1's kit and OP4 inherit.
   - (a) **Recommended:** a tower program type inside `internal/vision`, exported for the backends; the audio towers take
     it too.
     - Pros: no layout-rule change, no new package, and nothing pre-empts OP4's design of `internal/program`. Each block
       lands bit for bit in its own PR.
     - Cons: two operation vocabularies, the decoder's and the towers', until OP4 extracts `internal/program` and folds this
       one in. Until then SB1's kit covers neither, and the goldens are the only gate.
   - (b) Create `internal/program` now, with the tower operations as its first kinds, and amend `towers-standalone` to
     allow it.
     - Pros: one vocabulary from the start, where §9 puts it.
     - Cons: this decides OP4's and SB1's program shape before either is designed. It changes a protected layout rule.
       The decoder's own program stays in `decoder` until OP4 anyway, so the vocabulary is shared in name only.
   - (c) Park step M until SB1's kit and OP4 exist.
     - Pros: the towers are written once against the final interface, and the kit grades them from the start.
     - Cons: SB9 waits on two L items, and the three hand-written copies of each block stay meanwhile.
2. **Split step M.** It is L, not M: four block shapes, twelve families, three backends with walkers. Recommended:
   one PR per block shape, in the order above, each with its own registered section, starting with SigLIP's.
   Alternatively, a smaller step: shape (1) only, with the rest left to OP4.
3. **"SB1's kit covers them" is unreachable inside SB9 as written,** because the kit does not exist. Recommended: SB9's
   goal becomes "one description per block, walked by every backend, graded by the goldens". Kit coverage moves to SB1's
   own list, as a line item for when the kit lands.

**Decided (Francis, 2026-10-28 17:45, relayed): 1 (a), 2 yes, 3 yes.**
- The tower program type lives inside `internal/vision`, exported for the backends. The layout rule does not change.
- Step M is one PR per block shape. SigLIP's comes first: the CPU, then CUDA and WebGPU on nobara, then Metal on the Mac.
- SB9's goal is now "one description per block, walked by every backend, graded by the goldens". Kit coverage of the
  towers is on SB1's list.

#### SB9 step M, block 1: SigLIP's block. Gates registered 2026-10-28 before the code (nobara-mm01; Francis 17:45)

- **The program.** `vision.TowerBlock` is a list of `vision.TowerOp`. Each op names its kind, its input and output
  registers, the block weight it reads, and its parameters (norm form and epsilon, attention scale). The kinds:
  - norm;
  - linear, which includes the bias;
  - attention;
  - linear-add, the linear with the residual add as part of the op;
  - GELU-tanh.

  `vision.SiglipProgram(cfg)` builds SigLIP's block: norm, then the q, k and v linears, attention, linear-add of the
  output projection, norm, the fc1 linear, GELU-tanh, and linear-add of fc2.
- **The walkers.** Each walks the program over its existing operations, in the same order and with the same calls as
  today:
  - the CPU: `Encoder.runBlocks`, which also runs SigLIP2 NaFlex's blocks;
  - CUDA: `siglipTower.hidden`;
  - WebGPU: `VisionEncoder.ForwardPatches`, in both its float32 and its W8A8 form.

  Metal's `siglipVResident` is the Mac's, in a later PR. The CUDA and WebGPU per-family loops for SigLIP are deleted
  (G-SB9M-3).
- **The plant seam.** `vision.TowerProgramPlantForTest`, when set, rewrites every program `SiglipProgram` returns. Each
  backend reads its program from there, so one seam reaches all three.
- **The gates** are G-SB9M-1 to 4, applied as follows:
  - **G-SB9M-1:** the `siglip` and `siglip2-naflex` cases on the CPU, and `siglip` on CUDA and WebGPU, equal their
    goldens. So does every other case.
  - **G-SB9M-2:** three plants, each run on the CPU, CUDA and WebGPU:
    - the GELU dropped;
    - fc1 and the GELU swapped;
    - the first norm's epsilon moved up one ulp. **Amended 2026-10-29, before any plant ran (nobara-mm01):** scaled by
      2^20 instead. CUDA and WebGPU read the epsilon as float32, where a float64 ulp vanishes, and even a float32 ulp of
      1e-6 is lost in variance + eps, so the registered plant could not turn any backend red. 2^20 is exact in both
      widths.

    Each plant turns `siglip` red on every backend, and `siglip2-naflex` red on the CPU, since it walks the same
    program. Every other family stays equal.
  - **The WebGPU defect seams** (`visionDefectForTest`, 1 to 5) keep turning their tests red.
  - **The W8A8 WebGPU tower's** existing tests pass.
- **Void if** another compute app holds the card (`nvidia-smi --query-compute-apps=pid,process_name`, less
  `kwin_wayland`, `Xwayland` and `plasmashell`, per the coordinator's note of 17:45), or the timing lock is held for a
  timed run. GPU cases run under the lock, held shared. Cost: seconds per backend.

**Read 2026-10-29 (nobara-mm01; RTX 2070 SUPER, driver 595.91.07; the card was clear of other compute apps; load 0.64):
pass on every gate.** Logs are in `~/goinfer-logs/sb9m/` on nobara.
- **G-SB9M-1:** every case equals its golden on the CPU, CUDA and WebGPU (`TestTowerGolden_cpu`, `_cuda`, `_webgpu`):
  12, 8 and 6 families, with their per-family plants red as before.
- **G-SB9M-2:** all three plants turn exactly the `siglip` and `siglip2-naflex` cases red on each backend, at least one of
  them on the device, and leave every other case equal. The first WebGPU run found a walker defect. With fc1 and the GELU
  swapped, the walker returned an error for a register read before it was written, where the CPU and CUDA give a wrong
  output. It now reads that register as a zeroed buffer.
  **Corrected 2026-10-29 (mac-ax01, from Gemini's review of #434):** only the CPU goldens hold a `siglip2-naflex` case,
  so off the CPU the plants were shown on `siglip` alone; the naflex half of the claim above holds on the CPU only.
  `CheckProgramPlants` now logs the families it covered and fails a golden holding none of them.
- **The WebGPU defect seams:** 1 to 5 each stay under the bar, at lowest token cosine 0.774, 0.99931, 0.816, 0.702 and
  0.801. The clean tower is at 1.0000000.
- **The W8A8 WebGPU tower:** `TestVisionEncoder_parity` and `TestVisionEnableResident_parity` pass.
  `TestVisionEncoderF32_tiledSmoke` is heavy-tier, so it skipped and was not part of this gate.

#### SB9 step M, block 1 on Metal: gates registered 2026-10-29 before the code (mac-ax01)

The Mac's half of block 1: Metal's `siglipVResident` walks `vision.SiglipProgram`. Metal's grid towers share one block loop
(`metal/grid_vision.go:gridVAccel.run`) across four kinds. SigLIP leaves that loop for a walker over the program, which issues
the same dispatches in the same order through the tower's existing operations:
- norm through the LayerNorm kernel, at the op's epsilon;
- linear through `gridVAccel.linear`;
- attention as the q prescale at the op's scale (skipped under `gvDefect.noScale`), then the per-segment attention with its
  arena flushes;
- linear-add as the linear into the `o` scratch, then the residual add;
- GELU-tanh in place.

The loop keeps Qwen3, GLM-OCR and Qwen2.5-VL, whose blocks are later shapes. Box: the MacBook (M1 Pro), on AC; correctness
runs only, `go test -p 2`, holding the timing lock shared.

- **G-SB9M-1 on the Mac:** `TestTowerGolden_metal` passes before and after with `internal/towerid/testdata/towers.metal-darwin-arm64.json`
  unedited: every case of every family equals its golden, `siglip` among them, and so do the per-family plants and the
  backend comparison it already runs. `TestTowerGolden_cpu` on darwin (`towers.cpu-darwin-arm64.json`) and
  `TestTowerIdentity_metal` pass before and after. If the CPU golden's anchor is not this Mac's, its skip is recorded as a
  skip, not counted as a pass.
- **G-SB9M-2 on Metal:** `TestTowerGolden_metal` calls `towerid.CheckProgramPlants`. Each of its three plants turns exactly the
  `siglip` and `siglip2-naflex` cases red, at least one of them on the device, and leaves every other case equal.
- **G-SB9M-3:** the walker replaces the SigLIP path through the shared loop, and no Metal file both builds SigLIP's program
  and calls its op-layer methods in sequence.
- **G-SB9M-4:** build and vet of `metal` with and without `goinfer_testhooks`, CI's pinned `staticcheck` with its canary,
  `TestLayout_arrows`, `scripts/touched.py`, and a `changelog.d` fragment.
- **Also held:** the Metal SigLIP tests that exercise this path pass before and after with the same figures: the planted
  `noScale` and other `gvDefect` seams the existing tower tests use, and the int8 tower (`siglipForceInt8`).
- **Void if** a timed holder has the timing lock, or the box has under 4 GB reclaimable at a run's start. A void run is
  re-run, never counted.
- **Stopping rule:** one before-run and one after-run. A changed golden, a plant that does not turn exactly SigLIP's cases red,
  or a changed test outcome stops the work (`STATE: decision`).
- **Cost basis:** seconds per test (step S's Metal record).

**Read 2026-10-29 (mac-ax01; MacBook M1 Pro, on AC; 5.2 to 7.4 GB reclaimable at each run's start; another session held
the timing lock shared for a gate run, no timed holder): pass on every gate.** Logs: `docs/measurements/sb9m-metal-2026-10-29/`
(`sb9m-metal-before.log`, `sb9m-metal-after.log`).
- **G-SB9M-1:** `TestTowerGolden_metal` passes before and after with `towers.metal-darwin-arm64.json` unedited: all 7
  families' cases equal their golden, their per-family plants turn red as before, and the backend comparison is unchanged.
  `TestTowerGolden_cpu` on darwin compared against `towers.cpu-darwin-arm64.json` (no anchor skip) and passed, as did
  `TestTowerIdentity_metal`. Apart from timings and the program plants' own lines, all 104 lines of figures in the two logs
  are equal: per-case hashes, device-against-CPU cosines, the `TestS3Towers_tiny` and `TestS18Int8Tower_tiny` figures and
  their planted defects, and `TestMEM2_towerScratchFreed`'s buffer counts. `TestS17Profile` (heavy) skipped on both sides,
  so the profiling path the refactor threads through is covered by reading only.
- **G-SB9M-2:** on Metal, each of the three program plants turns the `siglip` cases red, at least one on the device, and
  leaves every other case equal. **Corrected 2026-10-29 (Gemini's review of #434):** the Metal golden has no
  `siglip2-naflex` case, so the read's first wording, "exactly `siglip` and `siglip2-naflex`", held for `siglip` only;
  the check passed the absent family vacuously.
- **G-SB9M-3:** SigLIP runs only through `gridVAccel.walkBlock`. The shared loop, now `gridVAccel.loopBlock` with the segment
  attention factored into `attendSegments` (one copy, used by both), runs only the kinds that have no program yet.
  `metal/vl_towers.go` builds the program and calls no op-layer method.
- **G-SB9M-4:** `metal` builds and vets with and without `goinfer_testhooks`; CI's pinned `staticcheck` is clean on
  `metal` under `goinfer_testhooks` (canary red first; the untagged run lists only test-hook symbols that predate this
  change). `TestLayout_arrows` and `scripts/touched.py` are green, and the `changelog.d` fragment is in.

**Review fixes, 2026-10-29 (Gemini on #434; mac-ax01).** Four findings, checked against the diff:
1. **Held:** the naflex claim was vacuous on Metal, as above. `towerid.programPlantVerdict` (factored out of
   `CheckProgramPlants`) now fails a golden with no case of a family that walks the program and returns the families it
   covered, which the log prints. `TestProgramPlantVerdict_vacuous` was red on the old check before the fix and is green
   after. Metal now logs `[siglip]`, the CPU `[siglip siglip2-naflex]`.
2. **Held:** `walkBlock` ignored attention's `Out` and GELU's `In`. It now refuses attention into any register but ctx,
   and a GELU whose `In` is neither unset nor its `Out`.
3. **Held:** a linear's registers were not checked against its weight's shape, so a misrouted program could overrun a
   buffer. `walkBlock` now refuses a linear whose weight's in and out widths differ from its registers', and a linear-add
   whose sum is not hidden-wide. `TestWalkBlock_refusesMisroutedOps` (in `metal/tower_identity_test.go`, so package metal's
   test-file count stays at the ratchet) covers all five cases: red before the checks (each was queued; the red run's log
   names the file it was first written in) and green after. Logs: `docs/measurements/sb9m-metal-2026-10-29/sb9m-review-*.log`.
4. **Does not hold:** `mark` returns at once when the class is unchanged (`if a.prof == nil || class == a.prof.cur`), so
   three `mark("gemm")` calls flush once, as `loopBlock`'s one call does.

After the fixes `TestTowerGolden_metal`, `TestTowerGolden_cpu` and the tower tests pass with every figure equal to the
#434 after-run, and the golden is unedited.

**Second review round, 2026-10-29 (Gemini on #440; mac-ax01).** Four findings, checked against the code:
1. **Partly held.** A linear or norm with no input register does not run: `walkBlock` resolves `In` for every kind but
   attention and GELU, and `reg(0)` is an error (`a linear with no input register` is refused as `register 0 unknown`).
   But a norm was not checked against widths, so one routed into or from mid would run. `walkBlock` now requires a norm to
   be hidden-wide in and out, and `regWidth` of no register is 0.
2. **Held.** The test's linear-add case was refused by the general width check before it reached the linear-add guard.
   A case with `Out` mid now reaches that guard. The guard has its own message, and removing it (a plant, reverted) turns
   that case red.
3. **Held.** Attention ignored a non-zero `In`; it now refuses one.
4. **Held.** Off the CPU, `programPlantVerdict` now needs a red device case for each covered family, not one across the
   whole golden. `TestProgramPlantVerdict_hostOnlyFamily` was red before the change. Every device golden (CUDA, both
   WebGPU, Metal) holds `siglip` on the device, so the stricter rule passes them: `TestTowerGolden_metal` and, on this Mac,
   `TestTowerGolden_webgpu` pass.

The new norm, attention and linear-add cases were red before the fix. Golden unedited, and every tower-test figure equal to
the #434 after-run. The CPU (`internal/vision/tower_walk.go:walkTowerBlock`), CUDA (`cuda/tower_walk.go:gridTower.walkBlock`)
and WebGPU (`gpu/internal/kernels/vision_walk.go`) walkers have the same shape: attention ignores `In`, and no op is
checked against register widths. That is for nobara-mm01 to match. **Matched in block 2 (nobara-mm01, #436):** `vision.TowerOp.CheckRegs` holds each op to the registers its kind takes, and the CPU, CUDA and WebGPU walkers call it before every op. Logs: `docs/measurements/sb9m-metal-2026-10-29/sb9m-review2-*.log`.

#### SB9 step M, block 2: the grid towers (Qwen3, GLM-OCR). Gates registered 2026-10-29 before the code (nobara-mm01; coordinator 11:05)

This finishes shape (1) on the backends that have a device path for it. Pixtral is CPU-only and SigLIP2 NaFlex already
walks SigLIP's program, so both are left for later. Qwen2.5-VL is shape (2), block 3. On WebGPU it shares
`gridVTower.block`, which keeps only Qwen2.5-VL's path until then.

- **The vocabulary grows.** These are added to `vision.TowerOp`:
  - the RMSNorm form (weight only);
  - a per-head norm, over [rows × heads, head_dim];
  - a fused-qkv linear, which writes q, k and v from one [3·hidden, hidden] weight;
  - NeoX rotate-half RoPE from the tower's own tables;
  - the SiLU-gated product, mid = silu(mid) · mid2, with a second [rows, intermediate] register.

  Attention runs over each of the forward's segments: one for SigLIP, the image frames here.
- **Each walker implements each kind the way its loop does today:**
  - the fused qkv is one matmul and a split on the CPU, and three linears over the split weights on CUDA and WebGPU;
  - each CPU tower keeps its own matmul entry point and its own attention schedule.
- **The programs:** `vision.Qwen3Program` and `vision.GlmOcrProgram`.
  - Qwen3: LayerNorm, the fused qkv, RoPE on q and k, attention, the linear-add of proj, LayerNorm, fc1, GELU-tanh, and
    the linear-add of fc2.
  - GLM-OCR: RMSNorm, the fused qkv, the per-head RMSNorm of q and k, RoPE, attention, the linear-add of proj, RMSNorm,
    the gate and up linears, the SiLU product, and the linear-add of down.
- **The walkers:**
  - the CPU: one walker for every CPU tower block, SigLIP's included (block 1's `Encoder.walkBlock` folds into it);
  - CUDA: `gridTower.walkBlock`, extended, replacing `hiddenLocked`'s block loop;
  - WebGPU: a walker over `gridVTower`'s ops, run by `gridVTower.run` for Qwen3 and GLM-OCR. Their branches leave
    `gridVTower.block`.

  Each tower's existing planted-defect seams act at the same ops.
- **The plant seam** becomes `vision.TowerProgramPlantForTest(program string, p)`. It is called with the program's name
  (`siglip`, `qwen3`, `glm-ocr`), so a plant can hit one program and leave the others alone.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9M-1 goldens | every case on the CPU, CUDA and WebGPU equals its golden exactly (`TestTowerGolden_cpu`, `_cuda`, `_webgpu`) | another compute app holds the card (`nvidia-smi --query-compute-apps`, less `kwin_wayland`, `Xwayland` and `plasmashell`), or a timed run holds the lock | runs to completion | seconds a backend |
| G-SB9M-2 plants | for each of the three programs, three plants, each applied to that program alone: the activation dropped (GELU-tanh or the SiLU product); the activation swapped with the linear before it; the first norm's epsilon scaled by 2^20 (block 1's amended form). Each turns exactly that program's families red (`siglip`: siglip, siglip2-naflex; `qwen3`: qwen3, qwen3-deepstack; `glm-ocr`: glm-ocr) and leaves every other case equal. On CUDA and WebGPU, at least one red case for each program and plant ran on the device | as G-SB9M-1 | runs to completion | seconds a backend |
| G-SB9M-2b the seams | the towers' existing planted-defect tests on CUDA and WebGPU (the grid towers', and block 1's SigLIP seams) keep passing | as G-SB9M-1 | runs to completion | seconds |
| G-SB9M-3 one order | after the change, no Qwen3 or GLM-OCR block order is left in `internal/vision`, `cuda/grid_vision.go` or `gpu/internal/kernels/grid_vision.go` beside its program | — | one read of the diff | — |
| G-SB9M-4 builds | `touched.py`; `go vet` and staticcheck, with the canary, under `cuda goinfer_testhooks` and `gpu goinfer_testhooks`; a `changelog.d` fragment | — | runs to completion | minutes |

**Amended 2026-10-29, after the CPU read and before any device run (nobara-mm01, from a review of #436):**
- **A fourth plant:** proj's residual add swapped with the second norm. Its rule and void conditions are G-SB9M-2's.
- **Why:** the registered activation swap reads a register before its linear writes it. On SigLIP and Qwen3 its output
  equals "the activation dropped", because fc1 then overwrites the GELU's result. On GLM-OCR it multiplies by whatever
  `mid2` held, which on CUDA is unzeroed scratch. It still turns its cases red, so it stays as registered. The new plant
  reads only written registers, and differs from every drop.
- **`CheckProgramPlants` tightened:** a program with no family in the golden is now an error, not a skip. On a device
  backend, each of a program's families must have a device case go red, not just one of them.

**Read 2026-10-29 13:39 PDT (nobara-mm01; RTX 2070 SUPER, driver 595.91.07; the card clear of other compute apps; load
1.24; head 27e94e943): pass on every gate.** Logs are in `~/goinfer-logs/sb9m2/` on nobara.
- **G-SB9M-1:** every case equals its golden on the CPU, CUDA and WebGPU (`TestTowerGolden_cpu`, `_cuda`, `_webgpu`): 12,
  8 and 6 families, with their per-family plants red as before.
- **G-SB9M-2:** the four plants, in each of the three programs, each turn exactly that program's families red on every
  backend, with a device case red for each family present on CUDA and WebGPU. Every other case stays equal.
- **G-SB9M-2b:** these pass:
  - CUDA: `TestGridVisionCUDA_matchesCPU`, `_scratchDeterminismClose`, `_fusedAttentionKernelDefects` and `_deepstackTiny`,
    `TestQwen25CUDA_tiny`, `TestS4Towers_tiny`, `TestSiglipCUDA_tiny` and `_int8StaysTheInt8Tower`;
  - WebGPU: `TestGridVisionWebGPU_matchesCPU` and `_deepstackTiny`, `TestQwen25VisionWebGPU_matchesCPU`, and block 1's SigLIP
    float32 and W8A8 tests.
- **G-SB9M-3:** Qwen3's and GLM-OCR's block orders are gone from `internal/vision`, `cuda/grid_vision.go` and
  `gpu/internal/kernels/grid_vision.go`. `gridVTower.block` and CUDA's `qwen25_vision.go` keep Qwen2.5-VL's order, for
  block 3.
- **G-SB9M-4:** `touched.py` is green. vet and staticcheck are green under the cuda and gpu tags (the canary went red
  first). A changelog fragment is added.
- **Re-run 13:43 PDT at 64d5160ed**, after merging main (mac-ax01's Metal SigLIP walker) and adding
  `vision.TowerOp.CheckRegs` to the three walkers: every gate above passes again on the CPU, CUDA and WebGPU.
  - `CheckProgramPlants` now takes the programs a backend walks: every program on the CPU, CUDA and WebGPU, and `siglip`
    on Metal.
  - Metal's `TestTowerGolden_metal` now runs the fourth plant on `siglip`. It needs a Mac run to be read.

#### SB9 step M, block 2 on Metal: gates registered 2026-10-29 before the code (mac-ax01)

The Mac's half of block 2: Metal's Qwen3-VL (with DeepStack) and GLM-OCR towers walk `vision.Qwen3Program` and
`vision.GlmOcrProgram`. They use block 1's `gridVAccel.walkBlock`, extended to the new kinds through the towers' existing
operations, each in its current dispatch order:
- the RMSNorm form through the RMSNorm kernel at the op's epsilon;
- the per-head norm the same way, over `rows·heads` rows of `head_dim`, skipped under `gvDefect.noQKNorm`;
- the fused qkv as the three split linears, q then k then v;
- RoPE through the tower's rope kernel on q or k;
- the SiLU product through `SiLUMul` (mid = silu(mid)·mid2).

The shared loop (`gridVAccel.loopBlock`) is left with Qwen2.5-VL alone (block 3), and its Qwen3 and GLM-OCR branches
(the per-head norm, the GELU MLP) go. Built on #436's program vocabulary; the PR opens on `main` once #436 has merged.
Box: the MacBook (M1 Pro), on AC; correctness runs only, `go test -p 2`, holding the timing lock shared.

- **G-SB9M-1 on the Mac:** `TestTowerGolden_metal` passes with `towers.metal-darwin-arm64.json` unedited: every case equals
  its golden, `qwen3`, `qwen3-deepstack` and `glm-ocr` among them, and the per-family plants and backend comparison are
  unchanged. `TestTowerGolden_cpu` (darwin) and `TestTowerIdentity_metal` pass. The before-run on #436's tree with main
  merged (`sb9m-b2-before.log`) already shows the goldens equal. It fails only #436's program plants for `qwen3` and
  `glm-ocr` on Metal, because Metal does not walk those programs yet.
- **G-SB9M-2 on Metal:** with Metal walking the programs, `CheckProgramPlants` passes on Metal: each of #436's four plants,
  in each of the three programs, turns exactly that program's families red, each family with a red case on the device.
- **G-SB9M-2b the seams:** the Metal grid-tower and SigLIP seam tests pass before and after with the same figures:
  `TestS3Towers_tiny` and `TestS18Int8Tower_tiny`, and the grid towers' `gvDefect` tests in `metal/grid_vision_test.go`
  (G-S2c's scale, RoPE-swap, position-transpose, q/k-norm and patch-bias plants, and G-S10e's DeepStack tap shift).
- **G-SB9M-3:** no Qwen3 or GLM-OCR block order is left in `metal/grid_vision.go` beside its program. `loopBlock` keeps
  only Qwen2.5-VL's.
- **G-SB9M-4:** build and vet of `metal` with and without `goinfer_testhooks`, CI's pinned `staticcheck` with its canary,
  `TestLayout_arrows`, `scripts/touched.py`, and a `changelog.d` fragment. The walker's register and width checks
  (`TestWalkBlock_refusesMisroutedOps`) extend to the new kinds, each new refusal red before its check.
- **Void, stopping rule, cost:** as block 1 on Metal.

**Read 2026-10-29 (mac-ax01; MacBook M1 Pro, on AC; 4.4 to 5.5 GB reclaimable at each run's start; no timed holder):
pass on every gate.** Logs: `docs/measurements/sb9m-metal-2026-10-29/sb9m-b2-*.log`.
- **G-SB9M-1:** `TestTowerGolden_metal` passes with the Metal golden unedited, as do `TestTowerGolden_cpu` (darwin, compared,
  not skipped) and `TestTowerIdentity_metal`. Comparing the before-runs (`sb9m-b2-before.log`, `sb9m-b2-before-grid.log`)
  with the after-run, every figure line is equal except `TestMEM2_towerScratchFreed`'s live bytes, up 128 KB in
  several towers, `embeddinggemma2` among them, which this change does not touch. The after-run also ran the two
  grid-vision tests before it in the same process. Run alone three times on the after tree, it gives exactly the
  before-run's figures (`sb9m-b2-mem2-repeat.log`): the 128 KB is Metal's small-buffer chunking carried over from the
  earlier tests, not this change.
- **G-SB9M-2:** on Metal, each of the four plants, in each of the three programs, turns exactly that program's families
  red, each with a red device case. Before the walker they did not (the `qwen3` and `glm-ocr` reds in the before-run).
- **G-SB9M-2b:** `TestGridVisionMetal_matchesCPU` (Qwen3 and GLM-OCR, with G-S2c's scale, RoPE-swap, position-transpose,
  q/k-norm and patch-bias plants), `TestGridVisionMetal_deepstackTiny` (G-S10e's tap shift), `TestS3Towers_tiny` and
  `TestS18Int8Tower_tiny` pass before and after with the same figures.
- **G-SB9M-3:** Qwen3 and GLM-OCR run only through `walkBlock`. `loopBlock` keeps Qwen2.5-VL's block alone and refuses
  any other kind, and `gvKind.rope`, unused once the loop was Qwen2.5-VL's alone, is gone.
- **G-SB9M-4:** `metal` builds and vets with and without `goinfer_testhooks`; CI's pinned `staticcheck` under
  `goinfer_testhooks` is clean (canary red first); `TestLayout_arrows` is green. `TestWalkBlock_refusesMisroutedOps`
  gained seven new-kind cases, and each new check, planted out one at a time, turns its cases red (`sb9m-b2-plants.log`;
  the first form of two plants disabled only the first term of an `||` condition and was re-run in parentheses; the log
  keeps both). `scripts/touched.py` is green on `metal`. Its one red is `cuda`'s darwin build
  (`cuda/internal/gpudev/copy_coalesce.go`, `//go:build darwin || linux`, calls the linux-only `DeviceCopy`), which this
  branch does not touch and which is the same on `main`.
- **Re-run 14:31 PDT after #436 merged** and main was merged into this branch: `golden.go` taken from main, which resolved
  #436 against #446. Metal's `walkBlock` now calls the shared register rule (`vision.TowerOp.CheckRegs`, `ReadsIn`) that
  the CPU, CUDA and WebGPU walkers call, in place of its own duplicate register checks. It keeps the checks only Metal
  needs: per-head norm and RoPE on q or k, linear and fused-qkv widths, and a linear-add's hidden-wide sum.
  `TestTowerGolden_metal` checks all three programs. Every gate passes again (`sb9m-b2-after-main.log`):
  - the golden is unedited, and every figure line equals the earlier after-run;
  - each of the four plants, in each of `siglip`, `qwen3` and `glm-ocr`, turns exactly that program's families red on Metal.

  Each check, planted out alone (`sb9m-b2-plants-main.log`), turns its cases red. CheckRegs covers nine cases. "A
  per-head norm on v" is new: CheckRegs refuses the one on mid first, so that case no longer reached Metal's q-or-k
  check. Staticcheck, `TestLayout_arrows` and `scripts/touched.py` are green.

#### SB9 step M: the walkers refuse a misrouted op. Gates registered 2026-10-29 (nobara-mm01; coordinator 15:00, from #446)

Registered after the code was written and the refusal tests had first run. The golden gate (G-SB9M-6) is registered before its run.

#446 found that the CPU, CUDA and WebGPU walkers checked no op against register widths. Block 2's
`vision.TowerOp.CheckRegs` (#436) covers the norm and attention routing it named. What is left is a linear's weight shape.
- **The rule:** `vision.TowerOp.CheckShape` says a linear's weight [out, in] must be [the Out register's width, the In
  register's width], and the fused qkv's [3·hidden, the In register's width].
- **Where it runs:** each nobara walker calls it before the op allocates or queues anything (the CPU's, CUDA's
  `gridTower.walkBlock`, WebGPU's `gridVTower.walkBlock` and `VisionEncoder.walkBlock`). WebGPU's SigLIP walker also
  refuses, up front, a kind it does not implement.
- **The shared cases:** `vision.MisroutedTowerOpsForTest` holds the misrouted ops, Metal's list from #446 plus the grid
  towers' kinds.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9M-5 refusals | every case of `MisroutedTowerOpsForTest` is refused with an error, and none reaches a matmul, allocation or dispatch, on the CPU (`TestWalkTowerBlock_refusesMisroutedOps`), CUDA (`TestGridTowerWalkBlock_refusesMisroutedOps`) and WebGPU (`TestGridVTowerWalkBlock_refusesMisroutedOps`, `TestVisionEncoderWalkBlock_refusesMisroutedOps`); with the CPU's `CheckShape` call removed, its test goes red | — | runs to completion | seconds |
| G-SB9M-6 unchanged | `TestTowerGolden_cpu`, `_cuda` and `_webgpu` pass, with their plants, at the PR's head | as G-SB9M-1 | runs to completion | seconds a backend |

**Read 2026-10-29 15:02 PDT (nobara-mm01; RTX 2070 SUPER, driver 595.91.07; the card clear; load 0.79; at 09ff156f1):
pass.** Logs are in `~/goinfer-logs/sb9chk/` on nobara.
- **G-SB9M-5:** all four refusal tests pass. With the CPU's `CheckShape` call removed, five cases ran instead of being
  refused. The first WebGPU SigLIP run allocated a register for the fused qkv before refusing that kind; it now refuses up
  front.
- **G-SB9M-6:** `TestTowerGolden_cpu`, `_cuda` and `_webgpu` pass with every plant. So do the towers' seam tests from block
  2's read.
- **The norm width check (#446, the coordinator's item of 15:20), red first:** with `CheckRegs` removed from each walker in
  turn, its refusal test turns red on exactly the norm cases (a norm into mid, a norm from mid, a per-head norm on mid). On
  the CPU, CUDA and the grid towers' WebGPU walker the queueing ones are also flagged. SigLIP's WebGPU walker has no
  per-head norm, so it shows only the first two. With the check back, all four pass. `vision.TowerRegWidth` of no
  register is now 0, as Metal's `regWidth` is.

**Follow-up 2026-10-29, from Gemini's review of #458 (nobara-mm01), red first:**
- **The bug:** WebGPU SigLIP's float32 shape check read a weight's input width from the op's `In` register, so only its
  element count was tested. fc1 [inter, hidden] and fc2 [hidden, inter] have the same count, and a transposed routing
  passed.
  - With the cases added to `MisroutedTowerOpsForTest` (fc1, fc2, gate and down transposed), its refusal test was red on
    three of them.
  - The fix takes [out, in] from the shape `NewVisionEncoderF32` required of that weight at upload.
- **The grid walker's fused qkv** did not check `op.W` before allocating. "The fused qkv of another weight" was red; it now
  refuses.
- **`TowerOp.CheckShape`** fails a register of no width on its own, without relying on `CheckRegs`.
- **Coverage:** the int8 form's refusal test is new (`TestVisionEncoderWalkBlock_refusesMisroutedOpsW8A8`). The CPU, CUDA
  and int8 walkers refused the new cases already, since they check real weight shapes. Metal's walker checks `gvProj`'s
  own [out, in] and keeps its own case list.

#### SB9 step M, block 3: Gemma 4 vision's block. Gates registered 2026-10-29 before the code (mac-ax01; Francis 14:57)

Francis chose (B) at 14:57: the Mac writes block 3's program, its CPU walker and its Metal walker. **nobara-mm01 adds the
CUDA and WebGPU walkers after this lands.** Box: the MacBook (M1 Pro), on AC; correctness runs only, `go test -p 2`,
holding the timing lock shared.

- **The block** (shape (3), `internal/vision/gemma4_encoder.go:Gemma4Encoder.Forward`; `metal/gemma4_vision.go:g4vAccel.Hidden`):
  - RMSNorm, then three clipped linears (each clamps its input and its output) to q, k and v.
  - A per-head RMSNorm on q and k (scaled) and on v (unscaled), then 2-D axial RoPE on q and k (rotate-half within each
    half of a head).
  - Attention at scale 1.0, then the o linear, an RMSNorm of its output, and its residual add.
  - RMSNorm, the gate and up linears, the GELU-tanh product (gelu(gate)·up), the down linear, an RMSNorm of its output,
    and its residual add.
- **The vocabulary gains, in `internal/vision/tower_program.go`:**
  - `TowerAdd` (Out += In);
  - `TowerGELUMul` (Out = gelu_tanh(Out)·In);
  - `TowerRoPEAxial` (2-D axial RoPE in place);
  - the unscaled RMSNorm form, for v's per-head norm, which reads no weight;
  - weights for the post-attention and post-MLP norms.

  `vision.CheckRegs` takes the new kinds. A clipped linear is a `TowerLinear` whose weight carries clamp bounds: the CPU
  walker applies them when the tower's weights report them (Gemma 4's), and Metal's `proj` already does.
  `vision.Gemma4Program(cfg)` builds the block. Its plant name is `gemma4`, and its family is `gemma4-vision`.
- **The walkers:**
  - **CPU:** `walkTowerBlock` replaces `Forward`'s layer loop. Each new op calls the function the loop called:
    `rmsNormHeadInto` for the unscaled norm, `applyRotaryVision` on each half for axial RoPE, the tower's own attention
    schedule, `geluTanh` and the elementwise product, and `addResidual`.
  - **Metal:** a walker on `g4vAccel` replaces `Hidden`'s layer loop. It uses the tower's existing dispatches:
    `rmsEps` (with the `ones` buffer for the unscaled form), `proj`, `g4v_rope`, the eg2 attention, `geluMul` and `add`.
- **G-SB9M-1:**
  - **Goldens:** `TestTowerGolden_cpu` (darwin, `towers.cpu-darwin-arm64.json`) and `TestTowerGolden_metal` pass with their
    goldens unedited, and `TestTowerIdentity_metal` passes.
  - **The linux CPU golden:** read on nobara or in CI. The CPU walker calls only functions the loop already called, plus
    copies and an elementwise float32 product, so nothing new can fuse differently on amd64.
  - **Same figures before and after:** `TestGemma4Encoder_tinyGoldenParity`, `TestGemma4Weights_reproduceForward`,
    `TestG4VMetal_matchesCPU` and `TestKSD4_towerHashes`.
- **G-SB9M-2:** on the CPU and on Metal, each of the four plants in program `gemma4` turns exactly the `gemma4-vision`
  cases red, a device case on Metal, and the other programs' plants still pass.
  - **The plant forms extend, not change.** The GELU product counts as an activation. "proj's residual add swapped with
    the second norm" also matches Gemma 4's form: the first `TowerAdd` followed by a norm, since Gemma 4's o linear has a
    norm between it and its add. The plants for `siglip`, `qwen3` and `glm-ocr` are unchanged.
  - **On CUDA and WebGPU,** `gemma4` is not checked until nobara-mm01's walkers land.
- **G-SB9M-2b:** Metal's `clampOff` seam and the eg2 tower tests keep their figures.
- **G-SB9M-3:** Gemma 4 vision's block order exists only as `Gemma4Program`. `Forward`'s and `g4vAccel.Hidden`'s layer loops
  are gone, and so are the `attentionInto` and `mlpInto` block helpers; the head-attention schedule stays as the
  walker's attention.
- **G-SB9M-4:**
  - Build, vet, CI's pinned `staticcheck` with its canary, `TestLayout_arrows`, `scripts/touched.py`, and a `changelog.d`
    fragment.
  - The new kinds' register checks (`CheckRegs`) and Metal's walker refusals each turn a planted case red when the check
    is removed.
- **Void, stopping rule, cost:** as block 1 on Metal. A changed golden or figure stops the work (`STATE: decision`).

**Read 2026-10-29 (mac-ax01; MacBook M1 Pro, on AC): pass on every gate.** Logs: `docs/measurements/sb9m-metal-2026-10-29/sb9m-b3-*.log`.
- **Void and re-run:** the first before-run started at 2.4 and 2.7 GB reclaimable, with the iOS Simulator open, and is
  void (`sb9m-b3-before-void.log`; every test in it passed). It was re-run from a detached worktree at the registration
  commit `d57951b1`, its cells starting at 4.7 and 5.2 GB (`sb9m-b3-before.log`). The after-run's cells started at 7.9
  and 8.7 GB (`sb9m-b3-after.log`).
- **G-SB9M-1:**
  - `TestTowerGolden_cpu` (darwin, compared, not skipped) and `TestTowerGolden_metal` pass with both goldens unedited, and
    `TestTowerIdentity_metal` passes.
  - All 124 figure lines of the two runs are equal: per-case hashes, `TestGemma4Encoder_tinyGoldenParity` (cosine
    1.000000000), `TestGemma4Weights_reproduceForward`, `TestG4VMetal_matchesCPU` (its 12 cells at 1.000000000, its
    clamps-off control 0.946551452), `TestKSD4_towerHashes`, and `TestMEM2_towerScratchFreed` run alone.
- **G-SB9M-2:** on the CPU and on Metal, each of the four plants in `gemma4` turns exactly the `gemma4-vision` cases red, a
  device case on Metal. The `siglip`, `qwen3` and `glm-ocr` plants still pass on both.
- **G-SB9M-2b:** the `clampOff` control in `TestG4VMetal_matchesCPU` keeps its figure.
- **G-SB9M-3:** `Forward`'s and `g4vAccel.Hidden`'s layer loops are gone. `attentionInto` keeps only the head schedule, as
  `attendInto`, and `mlpInto`, `clippedProj.matmulInto` and `rmsNormWInto` (unused once the loop went) are deleted.
- **Left in place on purpose:** `g4vAccel`'s `o` and `f` scratch buffers are still allocated, though the walker writes the
  o and down products into `xn`. Dropping them changes MEM2's buffer counts, which the registration holds equal. That is
  for a follow-up.
- **G-SB9M-4:**
  - `TestTowerOpCheckRegs` takes `gemma4` and six misroutings of the new kinds. Making each new kind's `CheckRegs` branch
    accept anything turns its cases red.
  - `TestG4VWalkBlock_refusesMisroutedOps` holds 11 refusals (`sb9m-b3-plants.log`). Each check the pattern could plant
    turns its case red. Three lines failed to build under the pattern because they use an if with an init statement.
    One of them, the per-head norm's weight lookup, was planted by hand at the log's end, as was the linear's lookup
    (after adding a case for it). The other two are the register lookups (`a.reg`), which no test can turn red, since
    `CheckRegs` refuses every op either would refuse. *Corrected in review: this line first said two lookups, and every
    check.*
  - Build and vet of the root, `metal`, `gpu` (`gpu goinfer_testhooks`) and `cuda` (`cuda goinfer_testhooks`,
    `GOOS=linux`); CI's pinned `staticcheck` on `internal/vision`, `internal/towerid` and `metal` under
    `goinfer_testhooks`; `TestLayout_arrows` and `scripts/touched.py` are green. A changelog fragment is added.
- **After merging `main` (#458, the walkers' `CheckShape`):** build and vet as above, the `internal/vision` and
  `internal/towerid` tests, and the Metal tower tests pass again. All 51 figure lines of the Metal run are in the
  after-run (`sb9m-b3-postmerge-*.log`).
- **After merging `main` again (#463, Qwen2.5-VL's program):** both programs are in each list. The CPU plants of every
  program, `gemma4` and `qwen2.5` among them, turn exactly their own families red.
- **Review (Gemini on #466; coordinator 16:10).** It found nothing output-changing. Each finding was checked against the
  code; red-first logs are `sb9m-b3-review-{red,green,plants}.log`.
  - **(1) holds; fixed.** `CheckRegs` now also refuses a norm reading a weight of the wrong width. A norm reads a hidden-wide
    norm weight, a per-head norm q's or k's, and the unscaled per-head norm none. Gemma 4's CPU and Metal lookups served
    all six norm weights to either kind, so a norm naming q's norm read past a head-wide buffer on Metal. The shared
    misrouted cases gain the four forms. Before the check, the CPU walker, the Gemma 4 Metal walker and `CheckRegs`
    accept them; Metal's grid walker already refused them. After it, every walker's refusal test holds them; WebGPU's
    was run here, and CUDA's walker calls `CheckRegs` first. The walkers' own unscaled-norm weight checks are now
    `CheckRegs`' and are removed.
  - **(2) holds; the record is corrected** (above). Metal's per-head norm form refusal had no case. It now has one, red
    when planted out.
  - **(3) holds in part.** RoPE's and axial RoPE's table checks on the CPU had no test.
    `TestWalkTowerBlock_refusesShortRoPETables` adds one, red for each when planted out. The unscaled norm's weight check
    is now `CheckRegs`'. The GELU product's length check is unreachable after `CheckRegs` (both registers
    intermediate-wide), as the SiLU product's is.
  - **(4) holds; fixed.** The Gemma 4 Metal test's duplicates of shared cases are removed. "A linear of a weight the tower
    lacks" stays: it is the case whose registers fit, so only the lookup refuses it.
  - **(5) holds; fixed.** A clamped linear copied its input on every call. It now copies into one slice reused across the
    block, and only when the weight has input bounds; clamping to ±inf changes nothing. Dropping the input clamp turns
    `TestGemma4Encoder_tinyGoldenParity` and `TestGemma4Weights_reproduceForward` red, so the tiny fixture has bounds.
    `towerClamp` of a weight with no projection returns ±inf (no clamp).
  - **The read after the fixes** (`sb9m-b3-review-after.log`; 4.1 to 4.4 GB reclaimable at each cell's start) passes.
    Its first run started at 3.94 GB and is void (`sb9m-b3-review-after-void.log`; every test in it passed). Every one
    of its 68 figure lines (per-case hashes, cosines, the clamps-off control, MEM2) equals the registered after-run.
    Build and vet of the root, `metal`, `gpu` and `cuda`, and CI's pinned `staticcheck` on `internal/vision`,
    `internal/towerid` and `metal`, are green.
- **After merging `main` a third time (#462, #467, #469):** the shared misrouted cases hold both sets (the transposed
  routings and the norm weights). Metal's plants list `gemma4` and `qwen2.5`. Build, vet and `staticcheck` are green.
  The gated re-run (`sb9m-b3-after-main469.log`; 6.4 GB reclaimable, load 2.2) passes:
  - every one of block 3's 68 figure lines is equal, and the 20 added from block 4's Metal tests equal block 4's
    after-run;
  - each program's plants, `gemma4` and `qwen2.5` included, turn exactly their own families red on the CPU and on Metal.
- **Next:** nobara-mm01 adds the CUDA and WebGPU walkers for `gemma4` and lists it in their `CheckProgramPlants` calls.
  The linux CPU golden is read there or in CI.

#### SB9 step M, block 4: Qwen2.5-VL's window-attention block. Gates registered 2026-10-29 before the code (nobara-mm01; coordinator 15:00 and 15:20)

Shape (2). The registered order puts block 3 (Gemma 4 vision) first. It is mac-ax01's, on the Mac (Francis, relayed
15:00), so block 4 goes ahead on nobara.

- **The program:** `vision.Qwen25Program`. It uses only kinds blocks 1 and 2 already have: RMSNorm (eps 1e-6), the fused
  qkv, RoPE on q and k, attention at 1/sqrt(head_dim), the linear-add of proj, RMSNorm, the gate and up linears, the SiLU
  product, and the linear-add of down.
  - Which segments a block attends over (its windows, or the whole frame for the full-attention blocks) is the forward's
    input per block, not part of the program.
  - The window permutation and the patch embed stay outside the block.
- **The walkers:**
  - the CPU: `walkTowerBlock`, with the tower's own matmul entry point and `attendPackedInto` over each block's segments;
  - CUDA: `gridTower.walkBlock`, replacing `qwen25Tower`'s block loop;
  - WebGPU: `gridVTower.walkBlock` for Qwen2.5-VL too. `gridVTower.block`, the last hand-ordered grid block there, is
    deleted.

  The towers' planted defects (`noScale`, `swapRope`, `noWindows`, `noWindowOrder`, WebGPU's `lateKey`) act where they
  do now.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9M-1 goldens | every case equals its golden exactly on the CPU, CUDA and WebGPU (`TestTowerGolden_cpu`, `_cuda`, `_webgpu`) | another compute app holds the card (less `kwin_wayland`, `Xwayland`, `plasmashell`), or a timed run holds the lock | runs to completion | seconds a backend |
| G-SB9M-2 plants | the four plants of block 2's amended list, in `qwen2.5` alone, turn exactly the `qwen2.5-vl` cases red on each backend, with a device case red on CUDA and WebGPU; every other case equal; the other programs' plants as before | as G-SB9M-1 | runs to completion | seconds a backend |
| G-SB9M-2b seams | `TestQwen25CUDA_tiny`, `TestQwen25VisionWebGPU_matchesCPU`, and the other tower seam tests of block 2's read pass | as G-SB9M-1 | runs to completion | seconds |
| G-SB9M-3 one order | no Qwen2.5-VL block order is left in `internal/vision/qwen_encoder.go`, `cuda/qwen25_vision.go` or `gpu/internal/kernels/grid_vision.go`; `gridVTower.block` is gone | — | one read of the diff | — |
| G-SB9M-4 builds | `touched.py`; vet and staticcheck (with the canary) under the cuda and gpu tags; the walkers' refusal tests; a `changelog.d` fragment | — | runs to completion | minutes |

**Read 2026-10-29 15:42 PDT (nobara-mm01; RTX 2070 SUPER, driver 595.91.07; the card clear; load 0.44; head 601846263):
pass on every gate.** Logs are in `~/goinfer-logs/sb9m4/` on nobara.
- **G-SB9M-1:** every case equals its golden on the CPU, CUDA and WebGPU.
- **G-SB9M-2:** the four plants in `qwen2.5` turn exactly `qwen2.5-vl` red on each backend, with a device case red on CUDA
  and WebGPU. The other three programs' plants are as before.
- **G-SB9M-2b:** these pass: `TestQwen25CUDA_tiny`, `TestQwen25VisionWebGPU_matchesCPU`, the grid-tower, S4 and SigLIP seam
  tests, and the walkers' refusal tests.
- **G-SB9M-3:** `gridVTower.block` and `gridTower.norm` are deleted. No Qwen2.5-VL block order is left in
  `qwen_encoder.go`, `qwen25_vision.go` or `grid_vision.go`. `qwen_bench_test.go` still times the encoder's attention
  and MLP methods.
- **G-SB9M-4:** `touched.py` is green; vet and staticcheck are green; the changelog fragment is added.

#### SB9 step M, block 4 on Metal: gates registered 2026-10-29 before the code (mac-ax01; coordinator 16:02)

The Mac's half of block 4: Metal's Qwen2.5-VL tower (`metal/vl_towers.go:qwen25VResident`) walks `vision.Qwen25Program`
through block 1's `gridVAccel.walkBlock`, which already has every kind the program uses. The window permutation, the
patch embed and each block's segments (its windows, or the whole frame in a full-attention block) stay outside the
block, as on the other backends. `gridVAccel.loopBlock`, the last hand-ordered grid block on Metal, is deleted, and `run`
refuses a call with no program. Built on #463; the PR opens on `main` once #463 has merged. Box: the MacBook (M1 Pro), on
AC; correctness runs only, `go test -p 2`, `GOMAXPROCS=4`, one Metal run at a time, holding the timing lock shared, each
run started at a 1-minute load under 12.

- **Why it is bit-identical by construction:** the walker's RMSNorm runs the same kernel at the op's epsilon (1e-6, which is
  the tower's own `eps`); the split qkv, RoPE, the q prescale at 1/sqrt(head_dim) (the same float32), the segment
  attention, the linear-adds and the SiLU product are the loop's dispatches in the loop's order.
- **The registration commit** adds `qwen2.5` to Metal's `CheckProgramPlants` list (`TestTowerGolden_metal`), so the
  before-run shows its plants green on Metal, as they must be while Metal does not walk the program.
- **G-SB9M-1 on the Mac:** `TestTowerGolden_metal` passes with `towers.metal-darwin-arm64.json` unedited, its `qwen2.5-vl`
  cases among them; `TestTowerGolden_cpu` (darwin) and `TestTowerIdentity_metal` pass. Every figure line of the before-run
  and the after-run is equal.
- **G-SB9M-2 on Metal:** each of the four plants in `qwen2.5` turns exactly the `qwen2.5-vl` cases red on Metal, with a
  device case red; the `siglip`, `qwen3`, `glm-ocr` (and `gemma4`, once #466 is on `main`) plants are as before. In the
  before-run the `qwen2.5` plants leave Metal green.
- **G-SB9M-2b the seams,** the same figures before and after: `TestS3Towers_tiny` (Qwen2.5-VL with G-S3a's scale,
  RoPE-swap, no-window and window-order plants), `TestS18Int8Tower_tiny`, `TestGridVisionMetal_matchesCPU`,
  `TestGridVisionMetal_deepstackTiny`, `TestKSD6H4_gridTowerScratch`, and `TestMEM2_towerScratchFreed` run alone.
- **G-SB9M-3:** no Qwen2.5-VL block order is left in `metal/grid_vision.go` or `metal/vl_towers.go`; `loopBlock` is gone.
- **G-SB9M-4:** build and vet of `metal` with and without `goinfer_testhooks`, CI's pinned `staticcheck` with its canary,
  `TestLayout_arrows`, `scripts/touched.py`, and a `changelog.d` fragment. `TestWalkBlock_refusesMisroutedOps` passes
  unchanged (no new kinds). `run`'s refusal of an empty program has a test, red with the check planted out.
- **Void, stopping rule, cost:** as block 1 on Metal (void under a timed holder or under 4 GB reclaimable at a run's start;
  one before-run and one after-run; a changed golden, figure or test outcome stops the work with `STATE: decision`;
  seconds per test).

**Read 2026-10-29 (mac-ax01; MacBook M1 Pro, on AC; 5.1 to 5.3 GB reclaimable and a 1-minute load under 3 at each
cell's start; no timed holder): pass on every gate.** Logs: `docs/measurements/sb9m-metal-2026-10-29/sb9m-b4-*.log`.
The before-run is on the registration commit (#463's head with Metal's list extended). Since that commit, `main` has
changed only docs.
- **G-SB9M-1:** `TestTowerGolden_metal` passes with the Metal golden unedited, its two `qwen2.5-vl` cases at cosine
  1.000000000 against the CPU. `TestTowerGolden_cpu` (darwin) and `TestTowerIdentity_metal` pass. All 157 non-timing
  lines of the before-run and the after-run are equal. The only difference is `TestTowerGolden_metal` going from FAIL
  to PASS.
- **G-SB9M-2:** in the before-run, each of the four `qwen2.5` plants leaves both `qwen2.5-vl` cases green on Metal ("no
  device walker read the program for it"), and that is the before-run's only failure. In the after-run, each turns
  exactly the `qwen2.5-vl` cases red with a device case. The `siglip`, `qwen3` and `glm-ocr` plants are as before.
- **G-SB9M-2b:** `TestS3Towers_tiny` (Qwen2.5-VL at its registered bar, G-S3a's four plants each red), `TestS18Int8Tower_tiny`,
  `TestGridVisionMetal_matchesCPU`, `TestGridVisionMetal_deepstackTiny` and `TestMEM2_towerScratchFreed` (run alone)
  give the same figures before and after. `TestKSD6H4_gridTowerScratch` is a heavy-checkpoint test and skipped in both
  runs (no `GOINFER_HEAVY_TESTS`), so it gives no figure.
- **G-SB9M-3:** `loopBlock` is deleted, with `gridVAccel.norm` and `gvKind.rmsNorm`, which only it used. No Qwen2.5-VL
  block order is left in `metal/grid_vision.go` or `metal/vl_towers.go`. Every grid block on Metal now runs through
  `walkBlock`.
- **G-SB9M-4:**
  - `metal` builds and vets with and without `goinfer_testhooks`. CI's pinned `staticcheck` under `goinfer_testhooks` is
    clean, and its canary goes red first. `TestLayout_arrows` is green.
  - `TestWalkBlock_refusesMisroutedOps` passes unchanged. `TestRun_refusesEmptyProgram` goes red with `run`'s check
    planted out (`sb9m-b4-plants.log`).
  - `scripts/touched.py`: `metal` and `gpu` are green. Its one red is `cuda`'s darwin build (`copy_coalesce.go` calls the
    linux-only `DeviceCopy`), the same as on `main` and recorded in block 2's read (`sb9m-b4-touched.log`).
  - A changelog fragment is added.

#### SB9 step M, block 3 on CUDA and WebGPU: Gemma 4 vision's walkers. Gates registered 2026-10-29 before the code (nobara-mm01; coordinator 17:01)

mac-ax01's #466 gave Gemma 4 vision its program (`vision.Gemma4Program`), the CPU walker and Metal's. This is the nobara
half.
- **CUDA:** `g4Tower.walkBlock` replaces `g4Tower.Hidden`'s layer loop.
- **WebGPU:** `g4vTower.walkBlock` replaces `g4vTower.layer`.

Each walks the program over its tower's existing operations, as Metal's `g4vAccel.walkBlock` does:
- a norm and a per-head norm through the tower's RMSNorm at the op's epsilon (the ones buffer for v's unscaled form);
- a linear through the tower's clipped projection;
- axial RoPE on q or k;
- attention over every patch, at the op's scale;
- the GELU product;
- the residual add.

Each refuses an op that does not fit its kind (`CheckRegs`) or its weight (`CheckShape`), or that this tower does not
take, before anything is queued. The towers' planted defects act at the same ops:
- CUDA's `g4Defect`: `noClamp`, `weightedVNorm`, `scaleRoot`, `noRope`; the position ones act before the blocks;
- WebGPU's `g4vDefectForTest` 1 to 5.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB9M-1 goldens | every case equals its golden exactly on CUDA and WebGPU (`TestTowerGolden_cuda`, `_webgpu`) and the CPU | another compute app holds the card (less `kwin_wayland`, `Xwayland`, `plasmashell`), or a timed run holds the lock | runs to completion | seconds a backend |
| G-SB9M-2 plants | `CheckProgramPlants` on CUDA and WebGPU names `gemma4` too: its four plants turn exactly `gemma4-vision` red, with a device case red, and leave every other case equal; the other programs as before | as G-SB9M-1 | runs to completion | seconds a backend |
| G-SB9M-2b seams | CUDA's `TestG4VCUDA_matchesCPU`, `_plantedDefects`, `_closeAndErrors`, `_scratchAcrossSizesAndDeterminism`, `_rebuildAfterCloseBitIdentical`; WebGPU's Gemma 4 vision tests (`gpu/internal/kernels/gemma4_vision_test.go`) pass | as G-SB9M-1 | runs to completion | seconds |
| G-SB9M-5 refusals | a refusal test for each new walker runs `vision.MisroutedTowerOpsForTest` (with #466's four norm-weight cases) and refuses every case before a dispatch; CUDA's grid refusal test runs on the CUDA box | — | runs to completion | seconds |
| G-SB9M-3 one order | no Gemma 4 layer order is left in `cuda/gemma4_vision.go` or `gpu/internal/kernels/gemma4_vision.go` beside the walker | — | one read of the diff | — |
| G-SB9M-4 builds | `touched.py`; vet and staticcheck (with the canary) under the cuda and gpu tags; a `changelog.d` fragment | — | runs to completion | minutes |

**Read 2026-10-29 17:04 PDT (nobara-mm01; RTX 2070 SUPER, driver 595.91.07; the card and the timing lock clear; load 2.17;
at c93f54aad plus the walkers): pass on every gate.** Logs are in `~/goinfer-logs/sb9m3dev/` on nobara.
- **G-SB9M-1:** `TestTowerGolden_cuda` and `_webgpu` equal their goldens exactly.
- **G-SB9M-2:** the four `gemma4` plants turn exactly `gemma4-vision` red on CUDA and WebGPU, with a device case red. The
  other four programs' plants are as before.
- **G-SB9M-2b:** these pass: `TestG4VCUDA_matchesCPU`, `_plantedDefects`, `_closeAndErrors`,
  `_scratchAcrossSizesAndDeterminism` and `_rebuildAfterCloseBitIdentical`, and `TestG4VWebGPU_matchesCPU`.
- **G-SB9M-5:**
  - `TestG4TowerWalkBlock_refusesMisroutedOps` (CUDA) and `TestG4VTowerWalkBlock_refusesMisroutedOps` (WebGPU) pass. With
    `CheckRegs` removed from each walker, eight cases were accepted, #466's norm-weight cases among them.
  - CUDA's grid refusal test, `TestGridTowerWalkBlock_refusesMisroutedOps`, passes with #466's four new cases on the CUDA
    box.
- **G-SB9M-3:** `g4Tower.Hidden`'s layer loop and `g4vTower.layer` are gone. `towerOps.rmsRows` lost its last callers and
  goes; `rmsRowsEps` stays.
- **G-SB9M-4:** vet and staticcheck are green under the cuda and gpu tags, and a changelog fragment is added.

#### SB9 step M, Pixtral's block (shape 1, CPU only): gates registered 2026-10-29 before the code (mac-ax01; coordinator 17:27)

Pixtral is the last shape-(1) family with a hand-ordered block. It has no device path, so this is the program and the CPU
walker only. Box: the MacBook (M1 Pro), on AC; correctness runs only, `go test -p 2`, `GOMAXPROCS=4`, holding the timing
lock shared, each run started at a 1-minute load under 12.

- **The block** (`internal/vision/pixtral_encoder.go:PixtralVisionEncoder.forward`): RMSNorm (eps 1e-5), the separate q,
  k and v linears (no bias), rotate-half RoPE over the full head on q and k (per-patch tables), attention at
  1/sqrt(head_dim) within each image (the block mask), o added to the residual, RMSNorm, the gate and up linears, the
  SiLU product, down added to the residual.
- **The program:** `vision.PixtralProgram(cfg)`, with plant name `pixtral` and family `pixtral`. It uses only kinds that
  exist. The patch conv, `ln_pre`, the RoPE tables and the image segments stay outside the block, as do Pixtral's own
  planted defects (`pixtralColFreqOffset`, `pixtralSwapRowCol`, `pixtralNoBlockMask`).
- **The walker:** `walkTowerBlock` replaces `forward`'s layer loop, with `MatmulBT` as the matmul and `attendPackedInto`
  over the images' segments as the attention.
- **Why it is bit-identical by construction:** each op calls the function the loop called. RoPE per head on q and then k,
  rather than interleaved, rotates each slice in place and independently. The SiLU product is GLM-OCR's, already shown
  bit-identical to `gatedSiLUMLPInto` in block 2. A linear-add is the loop's matmul followed by `addResidual`.
- **The registration commit** adds `pixtral` to `towerid`'s program families and to the CPU's `CheckProgramPlants` list.
  The before-run must therefore show the `pixtral` plants leaving the CPU green.
- **G-SB9M-1:** `TestTowerGolden_cpu` (darwin) passes with `towers.cpu-darwin-arm64.json` unedited, including its `pixtral`
  case. The linux CPU golden is read in CI. The same figures before and after: `TestPixtralVisionEncoder_matchesHF` and
  `TestPixtralVisionEncoder_plantedDefects`.
- **G-SB9M-2:** on the CPU, each of the four plants in `pixtral` turns exactly the `pixtral` case red. The other programs'
  plants are as before. In the before-run, the `pixtral` plants leave it green.
- **G-SB9M-3:** no Pixtral block order is left in `pixtral_encoder.go`; `forward`'s layer loop is gone.
- **G-SB9M-4:** build and vet (the root, plus `metal`, `gpu` and `cuda` under their tags, since `internal/vision` changes),
  CI's pinned `staticcheck` with its canary, `TestLayout_arrows`, `scripts/touched.py`, and a `changelog.d` fragment. The
  walkers' refusal tests pass unchanged, since no kinds are added.
- **Void, stopping rule, cost:** as block 1 on Metal. A run is void under a timed holder, or under 4 GB reclaimable at its
  start. There is one before-run and one after-run. A changed golden, figure or test outcome stops the work with
  `STATE: decision`. Each test takes seconds.

**Read 2026-10-29 (mac-ax01; MacBook M1 Pro, on AC; 4.9 and 5.3 GB reclaimable, load 2.1 and 4.0 at the runs' starts; no
timed holder): pass on every gate.** Logs are `docs/measurements/sb9m-metal-2026-10-29/sb9m-pix-*.log`.
- **G-SB9M-1:** `TestTowerGolden_cpu` (darwin, compared, not skipped) passes with the golden unedited, its `pixtral` case
  among them. All 37 non-timing lines of the before-run and the after-run are equal:
  - every per-case hash;
  - `TestPixtralVisionEncoder_matchesHF`'s per-stage and per-image figures (cosine 1.000000000);
  - `TestPixtralVisionEncoder_plantedDefects`' three figures.

  The only difference is `TestTowerGolden_cpu` going from FAIL to PASS. The linux CPU golden is read in CI.
- **G-SB9M-2:** in the before-run, each of the four `pixtral` plants leaves the `pixtral` case green, and that is the
  before-run's only failure. In the after-run, each turns exactly that case red. The other five programs' plants are as
  before.
- **G-SB9M-3:** `forward`'s layer loop is gone. The block's order exists only as `PixtralProgram`.
- **G-SB9M-4:**
  - The root builds and vets, and so do `metal`, `gpu` and `cuda` under their tags. CI's pinned `staticcheck` on
    `internal/vision` and `internal/towerid` is clean, and its canary was shown red in block 3's read.
  - `TestLayout_arrows` passes. `TestTowerOpCheckRegs` now checks `PixtralProgram`'s ops too.
  - `scripts/touched.py` is green (`sb9m-pix-touched.log`).
  - The walkers' refusal tests pass unchanged, and a changelog fragment is added.
- **After merging `main` (#460, #468, #475; their Go changes are in `gpu/` and `cuda/` only):** build and vet of every
  module are green. The gated re-run (`sb9m-pix-after-main.log`; 5.0 GB reclaimable, load 2.1) passes. Its 37 lines equal
  the after-run's, and each of the six programs' plants turns exactly its own families red.
- **Shape (1) is now done:** SigLIP, SigLIP2 NaFlex, Qwen3, GLM-OCR and Pixtral each walk a program. Shape (4), the audio
  towers, waits on Francis's call on the `towers-standalone` rule. `internal/audio` cannot import `internal/vision`'s
  program type.

### SB10 — configuration

- **Today:** `decoder.Options` (30 fields), the 42 `GOINFER_*` reads listed in `testdata/env_reads.txt`, and the flags in
  `internal/loadflags` and `docs/flags.meta.json`.
- **Contract:** one typed schema, and `Resolve(flags, env, defaults, machine)` returning the resolved settings, pure.
- **Fast check:** table tests; the docs generated; `TestEnvVars_docAndCodeAgree` generalized to flags and options. M.

#### SB10's scope and step 1's gates, registered 2026-10-28 before any code (mbp2-sb10)

**Inventory at `b2f6e81e1`.** Four surfaces describe one model load, and nothing ties them together:
- `decoder.Options`: 30 exported fields. 16 are set from a flag by `loadflags.Flags.Options`; the rest are set by serve's
  per-model overrides, by serve's app code (`KVShareOff`, `KVHostBytes`, `CPUBatchDecode`, `ResidentKVSlots`…) or only by
  callers of the library (`ExtraResidentBytes`, `LoadAbort`, `ActQuantGroup`).
- Environment: the 42 direct reads in `testdata/env_reads.txt` (startup and diagnostic), plus the 52 per-model knobs that
  `decoder/knobs.go` snapshots at Load (23 named constants, `cudaKnobs`, `metalKnobs`) and that `Options.Knobs` overrides by
  name. `docs/env-vars.md` documents both sets; `TestEnvVars_docAndCodeAgree` checks the doc against the code.
- Flags: 17 shared loading flags in `internal/loadflags` (both binaries); serve's 72, curated in `docs/flags.meta.json` and checked
  against `registerFlags` by `TestFlagsDoc`. Chat's own flags (sampling, REPL) are not load configuration and stay out of scope.
- Resolution today is spread: `Register` reads `GOINFER_GGUF_DIRECT` and `runtime.GOOS` while registering; `resolveAuto` probes
  the machine; `Options()` maps; serve layers per-model keys on top.

**Where the schema lives.** `internal/config`, standard library plus `decoder` only. One `Setting` record per knob a person can
turn: its name, kind, default, and each surface it appears on (flag name and which binaries, env var and whether it is a
startup read or a per-model knob, `decoder.Options` field, serve's per-model key), its class (operator or diagnostic) and its
doc pointer. A surface that has no setting (an `Options` field only a library caller sets, a cmd/gate variable) is listed in the
schema with its reason, so the inventory is total. Internal, so nothing here is public API.

**Steps.** Each is its own PR.
1. **The schema and the inventory test.** No production file outside `internal/config` changes; no behavior changes.
2. **Defaults from the schema.** `loadflags.Register` takes each flag's name, default and help from the schema; `--help` and
   `docs/flags.md` byte for byte unchanged.
3. **`Resolve(flags, env, defaults, machine)`, pure.** The env read in `Register`, `runtime.GOOS` and the auto-backend probe
   become inputs; `Flags.Options()` calls `Resolve`. Table tests over the inputs; the resolved `decoder.Options` for every case in
   the existing loadflags tests unchanged.
4. **Docs generated.** `docs/env-vars.md`'s rows and `docs/flags.md`'s rows from the schema plus the curated text that is there
   today; `TestEnvVars_docAndCodeAgree` and `TestFlagsDoc` become one check over flags, env vars and options.

**What is Francis's, decided before the step that needs it, not now.** Step 3 returns `decoder.Options` (no API change) unless he
chooses a new exported type; whether serve's per-model overrides go through `Resolve` in step 3 or later; any default or
meaning change found on the way is reported, never made. Step 1 needs none of these.

**Step 1's gates** (tier 0, CPU, seconds).

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-1 inventory | every exported field of `decoder.Options` (by reflection), every name in `testdata/env_reads.txt`, every per-model knob in `decoder/knobs.go`, every flag `loadflags.Register` registers for chat and for serve, and every flag in `docs/flags.meta.json` (equal to serve's flags by `TestFlagsDoc`) is in the schema exactly once; the schema names nothing that is on none of them | — | runs to completion | seconds |
| G-SB10-2 well-formed | unique setting names; no surface claimed by two settings; every setting has at least one surface or a listed reason; each flag-backed setting's default equals the registered flag's `DefValue` on a fresh `FlagSet` (a platform-dependent default is marked so and checked against `runtime.GOOS`'s) | — | runs to completion | seconds |
| G-SB10-3 plants | the checker, fed a planted extra `Options` field, env name, knob and flag, and a planted stale schema entry, reports each one; each gate above shown red once by a planted edit before the commit | — | runs to completion | seconds |
| G-SB10-4 nothing moved | the diff touches only `internal/config`, `scripts/touched_scans.py` (the knob scan is a source scan) and docs; no parity hash stales; `touched.py`, gofmt, vet, the pinned staticcheck with its canary, the layout test (`internal/layout/imports.json` gains `internal/config`'s arrows), the citation lint | — | runs to completion | about two minutes |

**Step 1's results, 2026-10-29 (mbp2-sb10, linux/amd64).** `internal/config.Settings` holds 169 settings: the 17 shared loading
flags, serve's 55 others, 40 startup and diagnostic variables (`GOINFER_GGUF_DIRECT` and `GOINFER_API_KEY` sit on their flags'
settings), 50 knobs (`GOINFER_MOE_CACHE_EXPERTS` and `_SLOTS` sit on theirs), and seven `Options` fields only code sets, each with
its reason. A knob's class is its `docs/env-vars.md` section: "Diagnostics and experiment knobs — NOT contract" is Diagnostic.
- G-SB10-1, G-SB10-2: PASS (`TestInventory`). It was red once before the commit on a default the table had wrong (`--fit` prints
  `true`, not `on`), so the default check reads. Only the shared loading flags' defaults are recorded and checked: serve's own
  flags are registered in `internal/serveapp`, which this package does not import; step 2 owns them. `--moe-pager`'s darwin
  default is recorded and checked only on a darwin run.
- G-SB10-3: PASS (`TestInventory_plants`, eight plants, each reported by name).
- G-SB10-4: PASS. The diff is `internal/config`, two `scripts/touched_scans.py` rows, `internal/layout/imports.json`'s two test
  arrows and `docs/ARCHITECTURE.md`'s generated appendix. No decoder file changed, so no parity hash stales; `touched.py`
  green, gofmt and vet clean, the pinned staticcheck (0.8.1) clean with its canary red, the layout tests green.

**Step 2's gates, registered 2026-10-29 before its code (mbp2-sb10).** Step 2 moves each shared loading flag's name, default and
help text into `internal/config.Settings` and has `loadflags.Register` read them there; serve's own flags get their defaults
recorded in the schema. Help that differs by binary is one text with serve's parts marked, not two copies. Tier 0, CPU, seconds.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-5 help and defaults unchanged | a golden of every flag `loadflags.Register` adds, for chat and for serve (name, usage text, `DefValue`, value type), recorded from `origin/main`'s code in a commit before the move, is equal byte for byte after it (`--moe-pager`'s `DefValue` is checked against `decoder.MoEPagerDefault(runtime.GOOS)`, so the golden holds on every OS); `TestFlagsDoc` green, so `docs/flags.md` is unchanged; every existing `internal/loadflags` test green | the golden commit is not the parent of the move | runs to completion | seconds |
| G-SB10-6 serve's defaults | a test beside `registerFlags` checks that every flag serve registers has a schema setting whose `Default` equals its `DefValue` with `GOINFER_GGUF_DIRECT` unset; shown red once by a planted wrong default | — | runs to completion | seconds |
| G-SB10-7 read from the schema | a test that changes one shared setting's default and help in `Settings` sees `Register`'s flag change with it, for a flag of each value kind (string, bool, int, float, and the custom `quant`, `embed-int4` and `fit` values) | — | runs to completion | seconds |
| G-SB10-8 nothing else moved | no decoder file changes, so no parity hash stales; the one new production arrow is `internal/loadflags → internal/config`; `touched.py`, gofmt, vet, the pinned staticcheck with its canary, the layout tests, the citation lint | — | runs to completion | about two minutes |

**Step 2's results, 2026-10-29 (mbp2-sb10, linux/amd64).** The shared flags' usage text moved to `internal/config/help.go`;
each setting carries `Help`, `PerModel` (serve's override key) and `ServeHelp` (`--ctx`'s serve clause), and
`Setting.Usage(app)` fills the marks. `Register` takes every name's default and usage from `config.Lookup`. Serve's 55 own
flags now carry their defaults (32 non-empty). `--direct-load` still ORs in `GOINFER_GGUF_DIRECT` at registration, and
`--moe-pager`'s platform default is now the schema's `DefaultOn`, held equal to `decoder.MoEPagerDefault` on every OS by the
golden; making both inputs to `Resolve` is step 3.
- G-SB10-5: PASS. The golden (`internal/loadflags/testdata/register.golden.json`, 17 flags for each app) was recorded from
  `origin/main`'s `Register` in the commit before the move, and `TestRegister_golden` is green after it; `TestFlagsDoc` green;
  every `internal/loadflags` test green.
- G-SB10-6: PASS (`TestConfigDefaults` in `internal/serveapp`); red once on a planted `--addr` default.
- G-SB10-7: PASS (`TestRegister_readsTheSchema`, seven planted defaults and help texts across string, bool, int, float, `quant`,
  `embed-int4` and `fit`).
- G-SB10-8: PASS. No decoder file changed. New arrows: `internal/loadflags → internal/config` (production) and
  `internal/serveapp → internal/config` (test); `internal/config`'s test is now an external test package, since `loadflags`
  imports `config`. `touched.py` green, gofmt and vet clean, the pinned staticcheck clean (its canary red earlier this session
  on the same binary), the layout tests green.

**Step 3a's gates, registered 2026-10-29 before its code (mbp2-sb10).** What the tree allows: serve's admin load sets `Flags`
fields after parsing (`Quant`, `LoRA`) and then calls `Options()`, so `Resolve`'s flags input is the parsed `Flags` value, not
the command line. Defaults, the environment (`GOINFER_GGUF_DIRECT`) and `runtime.GOOS` (`--moe-pager`) are applied when the
flags are registered, so they become inputs of registration: `Register` is `register(fs, app, goos, getenv)` with the
process's values. The one impure step left in `Options()` is the auto-backend probe; it becomes `Machine.Auto`, and
`Resolve(f Flags, m Machine) decoder.Options` reads nothing else. `Flags.Options()` probes only when the backend is `auto`, as
today, and calls `Resolve`. Serve's per-model merge is 3b, after Francis's answer. No exported `decoder` type changes. Tier 0.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-9 same options | a golden of `Options()` (every field but `LoadAbort`), recorded from the current code in a commit before the change with `autoBackend` stubbed, over a table: the defaults, each shared flag at a non-default value, `--backend` auto resolving to cuda, metal and cpu and named cpu and metal, `--embed-int4` given and not on metal, for chat and serve; `Resolve` over the same table after the change equals it byte for byte; every existing `internal/loadflags` test green | the golden commit is not before the change | runs to completion | seconds |
| G-SB10-10 pure | `Resolve` with `autoBackend` replaced by a function that fails the test returns the table's options from `Machine.Auto`; `register` given goos `darwin` and a `GOINFER_GGUF_DIRECT` of `1` through its parameters yields `--moe-pager` pool and `--direct-load` true on any host, with the process environment unset | — | runs to completion | seconds |
| G-SB10-11 nothing else moved | no decoder file changes; the golden of step 2 (`TestRegister_golden`), `TestFlagsDoc` and `TestConfigDefaults` green; `touched.py`, gofmt, vet, the pinned staticcheck, the layout tests, the citation lint | — | runs to completion | about two minutes |

**Step 3a's results, 2026-10-29 (mbp2-sb10, linux/amd64).** `loadflags.Resolve(f Flags, m Machine) decoder.Options` and
`Machine{Auto}`; `Flags.Options()` probes when the backend is `auto` and calls it. `register(fs, app, goos, ggufDirect)`
takes the OS and whether `GOINFER_GGUF_DIRECT` is set: the variable is passed as a value rather than a lookup function
because `TestEnvVars_docAndCodeAgree` finds reads by their literal `os.Getenv` call, and `Register` keeps that call.
- G-SB10-9: PASS. `testdata/resolve.golden.json` (26 cases for each app) was recorded from the pre-change `Options()`, at the
  commit before the change; `TestOptions_golden` is green after it, and every existing `internal/loadflags` test is green.
  The golden records that `--direct-load` does not reach `Options.DirectLoad`: the apps hand it to `modelload` beside the
  options, as before.
- G-SB10-10: PASS (`TestResolve_pure`); red once on a planted probe call inside `Resolve` ("Resolve probed the devices").
- Review fix (Gemini on #437, 2026-10-29): the golden rewrote any `--moe-pager` equal to the host's default, so on darwin an
  explicit `-moe-pager pool` read as the default and both tests failed there, and `pool` was not a changed value on darwin. Now
  only a case that leaves the flag unset is rewritten, and the table sets it to `pool` and to `mmap`; the golden was re-recorded
  in a worktree at the golden commit (the pre-change code), the same 50 rows plus the two new ones. `register` is checked on
  darwin and on linux, each with `GOINFER_GGUF_DIRECT` the other way round from the process's. Not changed: `Options()` still
  resolves `auto` on the `Flags` before `Resolve` (callers read `Backend` afterwards, as before), and `BackendAuto` is still
  `Auto != nil`, as it was; 3b passes `Machine` through serve's merge. `TestInventory`'s default check of the shared flags now
  compares the schema with `Register`, which reads it; their values are pinned by step 2's `TestRegister_golden`.
- G-SB10-11: PASS. No decoder file changed; `TestRegister_golden`, `TestFlagsDoc` and `TestConfigDefaults` green;
  `touched.py` green (it caught the hidden read above), gofmt, vet and the pinned staticcheck clean.

**Step 3b's gates, registered 2026-10-29 before its code (mbp2-sb10; Francis 11:36: "yes, step 3b").** `Resolve` takes
`Overrides`: one model's per-model keys from serve's `--model name=path,key=…` (quant, lora, kv, kv-quant, stream,
weight-cache, embed-int4, ctx; nil = not given) and, for serve only, its own load settings (`--kv-quant`, `--kv-sessions` and
whether it was given, `--prefill-chunk`, the KV-share answer, the snapshot budget, `--cpu-batch`), nil for chat. A decision
head's quant stays serve's rule (`decide.HeadQuant`): serve computes it and passes it as the quant override.
`modelSpec.options` becomes building `Overrides` and one call. Tier 0, CPU, seconds.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-13 serve's options unchanged | a golden of `modelSpec.options(cfg)` (every field but `LoadAbort`), recorded from the current code in a commit before the change, over a table: a bare spec, each per-model key alone (and `stream=false`, `embed-int4` both ways), a decision head with and without `quant=` and with a global `--quant`, global `--kv i8` and `--kv-quant`, each serve load setting changed, and `--backend metal` without `embed-int4`; the same table after the change equals it byte for byte; `TestOptions_golden` and `TestResolve_pure` (chat's, no overrides) unchanged | the golden commit is not before the change | runs to completion | seconds |
| G-SB10-14 one merge | after the change `internal/serveapp`'s non-test files assign no `decoder.Options` field that `Resolve` sets (a source scan for `o.<Field> =` and `opts.<Field> =` on those fields) outside a listed exception (the drafter's pricing, computed from the drafter, not configured); shown red once by a planted assignment | — | runs to completion | seconds |
| G-SB10-15 nothing else moved | no decoder file changes; `TestRegister_golden`, `TestFlagsDoc`, `TestConfigDefaults`, `TestInventory` green; `touched.py`, gofmt, vet, the pinned staticcheck, the layout tests, the citation lint | — | runs to completion | about two minutes |

**Step 3b's results, 2026-10-29 (mbp2-sb10, linux/amd64).** `loadflags.Overrides` (the per-model keys and `*ServeLoad`),
`Resolve(f, m, ov)`, and `Flags.OptionsWith(ov)`; `modelSpec.options` builds the overrides, picks a decision head's quant with
`decide.HeadQuant` as before, and makes one call. Serve's four `or*` helpers went with the merge they served.
- G-SB10-13: PASS. `internal/serveapp/testdata/options.golden.json` (28 cases) was recorded from the old `modelSpec.options`
  in the commit before the change and is equal after it; chat's `TestOptions_golden` and `TestResolve_pure` unchanged. After
  that commit the golden test was rewritten not to import `decoder` (serve's test files that import it are held at 31 by
  `TestServeappTests_decoderImportsOnlyFall`, SB7): `--moe-pager`'s default is read from the registered flag and
  `--cpu-batch` on is its value, 1; the golden data did not change.
- G-SB10-14: PASS (`TestOneMerge`). It flags 15 lines in the pre-change `main.go` and none after; its planted line is reported.
  Exceptions: `specHead`'s `o.LoRA = a` (a decision head's own adapter). Receivers `load` (a `Flags`) and `ov` (the
  `Overrides`) are skipped by name, because their fields share names with `decoder.Options`'.
- G-SB10-15: PASS. No decoder file changed; `TestRegister_golden`, `TestFlagsDoc`, `TestConfigDefaults` and `TestInventory`
  green; `touched.py` green, gofmt, vet and the pinned staticcheck clean.

**Step 4a, 2026-10-29 (mbp2-sb10).** `docs/flags.md` is already generated from serve's registered flags (`TestFlagsDoc`), and
step 2 put those flags' defaults and the shared flags' help in the schema, so what step 4 had left was `docs/env-vars.md`. Its
knob paragraph (the 52 knobs `decoder/knobs.go` snapshots) is now generated from the schema; the prose had drifted to
"fifty-three" over a list of 52. Its tables of curated purpose text stay hand-written: generating them means moving about a
hundred purpose texts into a metadata file, as `docs/flags.meta.json` does for flags, which is step 4b and Francis's call.
`TestEnvVars_docAndCodeAgree` and `TestFlagsDoc` stay two tests, because each reads a surface only its own package can
(decoder's source scan of reads, serve's unexported `registerFlags`); `TestInventory` joins them through the schema.
G-SB10-12's rule was written after a first draft of its test had run; it is an equality check with no threshold to choose.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-12 env doc | `TestEnvVarsDoc`: the block between `<!-- generated: knobs -->` and `<!-- /generated -->` in `docs/env-vars.md` is the schema's knobs exactly (count and names); every variable in the schema is named on the page (in backticks, bare or as `NAME=`); `TestEnvVars_docAndCodeAgree` green | — | runs to completion | seconds |

Result: PASS. The generated paragraph replaced the hand-written list (52 = 52 names, none added or lost; the phase notes in the
old list were history and are gone); `TestEnvVars_docAndCodeAgree` green.

**Step 4b's gates, registered 2026-10-29 before its code (mbp2-sb10; Francis 11:36: "move the env-var purpose texts into a
metadata file … and generate docs/env-vars.md's tables from the schema; keep the three cross-checks; … one pass, merge
origin/main right before pushing, … byte-identical to the current rows where the text is unchanged").** The eight tables of
`docs/env-vars.md` (71 rows) move to `docs/env-vars.meta.json`: per table its columns, per row the variables it documents, its
label when that is not the bare name, and its other cells. Each table is generated between `<!-- generated: table <id> -->`
and `<!-- /generated -->`. The page's prose and its diagnostics list stay hand-written. Tier 0, seconds.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB10-16 byte-identical | the page with its marker lines removed equals the page before the move byte for byte (checked against `origin/main`'s copy merged right before the push); the generated tables equal what the page holds | `origin/main` changed the page after the check | runs to completion | seconds |
| G-SB10-17 rows from the schema | every variable a row documents is a setting in `internal/config` with an `Env`, except in the CI and test-gate table, whose variables tests read; each variable is in at most one row; `TestEnvVarsDoc`'s every-variable-named check and the knob paragraph stay; shown red once by a planted row naming a variable the schema lacks | — | runs to completion | seconds |
| G-SB10-18 the three cross-checks | `TestInventory` (tree to schema), `TestEnvVarsDoc` (schema to page), `TestEnvVars_docAndCodeAgree` (page to tree) green; `touched.py`, the citation lint | — | runs to completion | about two minutes |

**Step 4b's results, 2026-10-29 (mbp2-sb10).** `docs/env-vars.meta.json` holds the eight tables (71 rows); `TestEnvVarsDoc`
writes each between its markers and fails when the page differs. The one prose change is the exhaustive list's opening
sentence, which said the page was not generated.
- G-SB10-16: PASS. With the table marker lines removed, the page equals its copy before the move byte for byte; `origin/main`
  had not changed the page since this branch's base when it was merged before the push. `-update` on the moved page is a no-op.
- G-SB10-17: PASS (`checkRows`; `TestEnvVarsDoc_plants`, three plants, each reported). Three rows document variables the
  schema does not hold, and each says why in an `outside` field (the check also fails if such a variable is in the schema):
  `GOINFER_DUMP_LOGITS` and `GOINFER_MEM_PROBE`, read by tests only, and `GOINFER_GIW_VERIFY` (finding 1).
- G-SB10-18: PASS. `TestInventory`, `TestEnvVarsDoc`, `TestEnvVars_docAndCodeAgree` green; `touched.py` and the citation lint green.

Findings, reported and not changed:
1. `GOINFER_GIW_VERIFY` is a production read (`decoder/giwverify.go:giwVerifyEnv`, through `os.Getenv(giwVerifyEnv)`), and
   `TestEnvVars_docAndCodeAgree` finds reads only by a literal name inside the call, so it is not in `testdata/env_reads.txt`
   and the "no new production env reads" ratchet does not hold it. Putting it on the list, or moving it to a per-model knob or
   an `Options` field, is Francis's call (`docs/env-vars.md` § "Policy — where a new setting goes"); widening the scan to
   named constants is the cheap fix that keeps the next one from slipping.
2. Three rows of the operator table (`GOINFER_CUDA_FLASH_DECODE`, `_MIN_KEYS`, `_VERIFY`) have four cells under two columns, so
   a GitHub render drops their last two cells, which hold the purpose text. They are kept byte for byte, as asked; the fix is
   to move them to the four-column "Prefill/MoE defaults" table or join their cells, a text change.

Francis (11:51) on both: (1) widen the scan and add `GOINFER_GIW_VERIFY` to `testdata/env_reads.txt` as a startup read, the one
addition he allows; (2) move the three rows into the four-column table. (1) done 2026-10-29: `TestEnvVars_docAndCodeAgree` now also
resolves a read through a `GOINFER_*` constant of the reading file's package; it went red on `GOINFER_GIW_VERIFY` alone (no
other read was hiding), which is now on the list, a schema setting (startup, operator) and a plain row of the operator table.
Francis (11:51) on finding 2: move the three rows into the four-column table. Done 2026-10-29: in `docs/env-vars.meta.json`
they moved from the operator table to the end of "Prefill/MoE defaults", whose columns (default, opt out with, what it changes)
their cells already were; no text changed.

### SB11 — cost counters and telemetry

**Closed 2026-10-29 (the coordinator's call, 17:27; written by nobara-mm01).** All twenty gates pass:
- the CPU: G-SB11-1 to 7, #358;
- CUDA: 8 to 11, #370;
- WebGPU: 12 to 15, #377, with the arm64 golden #385;
- Metal: 16, 17, 19 and 20, with #411 and #412; 18 on AC, #423.

Three things are left over, each a row of its own in the ledger:
- **SB11a:** OP4's walker emits the operation records on the devices.
- **SB11b:** the wire block, Francis's call.
- **SB11c:** Metal's tagged-only `costToken`.

- **Contract:** per operation and per phase: bytes read and written, kernel launches or command buffers, syncs,
  allocations, time. One schema for tier-0 tests, gates and the wire (X1, `docs/tasks/task-turn-telemetry-2026-09.md`).
- **Fast check:** counters on tiny fixtures, asserted exactly. The program walker (OP4) emits them for every backend. M.

#### SB11's gates, registered 2026-10-27 before any code (nobara-mm01; Francis's pick of 17:26)

**The schema (`internal/cost`, the standard library only).**
- A **report** has a `prefill` and a `decode` phase, X1's names (`docs/tasks/task-turn-telemetry-2026-09.md`). A phase that was
  not measured is absent, not zeroed (X1's rule 1).
- Each phase holds its **totals**, its **tokens**, and **counts per operation**: `calls`, `bytes_read`, `bytes_written`,
  `launches` (kernel launches or command buffers), `syncs`, `allocs`, `alloc_bytes`, `time_ns`. A field a backend does not
  measure for an operation is named in that operation's `unmeasured` list, never reported as 0.
- `Timings()` gives X1's `prefill_ms` and `decode_ms` from the phases' time.
- **Off is a nil counter:** each operation's hook is one nil check. The server sends nothing new (a wire change is Francis's call).
  The schema round-trips through JSON, the format X1's block will use.

**How the counters are emitted until OP4.**
- The CPU walks a program already (`walkToken`, `walkBatch`). Its counters are keyed by the program's operation kinds (`norm`,
  `proj_qkv`, `attention`, …), which is the unit OP4's walker will emit for every backend.
- CUDA and WebGPU emit from their dispatch points (a launch, a command buffer, a sync, an MM3 lease), keyed by the kernel's
  name. When OP4 lands, its walker emits the operation records, and the backends keep only the device counts below an operation.
  The record in this section will say how, per backend, once built.

**Step 1: the schema and the CPU.** The tiny fixture is llama-tiny at f32 (its weights' bytes are exact by shape).

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB11-1 the schema | a report round-trips through JSON unchanged; an unmeasured phase is absent from the JSON; an unmeasured field is listed, not zero; `Timings()` gives X1's `prefill_ms`/`decode_ms`; concurrent adds sum exactly under `-race` | — | runs to completion | seconds |
| G-SB11-2 CPU decode, exact | one decode token on llama-tiny at a known position. **Calls:** each operation kind's calls equal its count in the program. **Weight bytes:** the bytes read from weights, over all operations and the head, equal an independent sum over the loaded `Weights` (each matrix, norm and the head once). **Attention:** it reads exactly (position+1) rows of K and V at the cache's element size, and writes the context and this position's row. **Device counts:** launches and syncs are 0, allocations are listed unmeasured on the CPU | the fixture is absent (a skip is a failure here: llama-tiny is tracked) | runs to completion | seconds |
| G-SB11-3 CPU prefill | the batched walker's counts for an N-token prefill: calls per kind = N rows in one call each; weight bytes once per pass, not per row; attention rows read as the causal triangle (Σ i+1 for i < N) | as G-SB11-2 | runs to completion | seconds |
| G-SB11-4 plants | a planted extra operation (a duplicated norm in the program) moves its kind's calls by exactly one and turns G-SB11-2 red; a planted second count of one operation's bytes turns the weight-byte equality red | — | runs to completion | seconds |
| G-SB11-5 numerics untouched | logits with counters on equal logits with counters off, bit for bit, on llama-tiny (decode and prefill); the parity hashes stay fresh or refresh as non-numeric | — | runs to completion | seconds; the refresh about four minutes |
| G-SB11-6 off costs nothing measurable | the off hook's cost per operation (a benchmark of the nil check in the walker's shape) × operations per token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on the CPU, is **under 0.5%**. Under the exclusive timing lock; the record notes the box's power state (governor, AC) and load. The on cost is reported, not graded | the box is under heavy load (the 2026-10-19 rule) | one run | under two minutes |
| G-SB11-7 builds | vet (untagged and the module tags), the pinned staticcheck with its canary, gofmt, `touched.py`, the layout tests, the citation lint; a `changelog.d` fragment | — | runs to completion | about two minutes |

**Step 1 read, 2026-10-27 (nobara): all seven pass.**
- **G-SB11-1:** `internal/cost`'s round trip, absent phase, unmeasured list, `Timings()` and concurrent sum, under `-race`.
- **G-SB11-2:** one decode token on llama-tiny at position 3. The calls of each kind equal the program's count, and the matrix
  bytes read equal the dense enumeration (`ResidentDenseWeightBytes` less the embedding table, plus the head and one embedding
  row). Attention reads 4 rows of K and V a layer. Launches and syncs are 0, allocations unmeasured. Decode totals: 39 calls,
  674,304 bytes read, 13,312 written.
- **G-SB11-3:** a 7-token prefill through `walkBatch`. One call per operation per pass, each layer's matrices once, attention
  over the causal triangle (28 rows).
- **G-SB11-4:** the planted extra norm is reported as norm's call count; the planted second count of `proj_o` breaks the
  matrix-byte equality.
- **G-SB11-5:** decode logits and prefill rows are bit-identical with the counters on and off. The parity refresh ran 36 forward
  goldens green and refreshed `deps_hash` lines only.
  - `TestParityManifest_coreClosed` found that the forward now reaches `cost_cpu.go`, and `weightbytes.go` through `wmBytes`.
  - Rather than put the fit accounting in `core` (every fit edit would then stale parity), `wmBytes` moved to its own file,
    `wmbytes.go`, and `core` gained that file and `cost_cpu.go`.
- **G-SB11-6:** on Qwen2.5-Coder-0.5B q4_k_m on the CPU, a decode token takes 20.0 ms over 219 operations, and the off hook
  takes 3.18 ns an operation: **0.0035% of a token**, against the registered 0.5%. Conditions: the exclusive timing lock,
  governor `schedutil`, a desktop on mains power, load 1.80 at the start (1.99 at the end), 57 GB available. This reading
  replaces a first one (0.0024%, 2 ns) whose benchmark truncated to whole nanoseconds and left the hook's held start time
  out; the hook now holds it across a call, and that call is charged to the hook, so 3.18 ns is an upper bound.
- **G-SB11-7:** vet (untagged, testhooks, realckpt); the pinned staticcheck with its canary red; gofmt; the layout tests
  (`imports.json` and the appendix for `decoder → internal/cost`); the citation lint; a `changelog.d` fragment.

**Review fixes on #358, 2026-10-27 (Gemini's review and CI).** The gates did not change; four were tightened to grade what
they promised.
- **CI:** `TestKVCache_everyFieldHasAState`, which landed on `main` after the branch was cut, needs every `KVCache` field
  declared. `cost` and `costPhase` are listed as not sequence state.
- **G-SB11-3:** the batched prefill never counted its embedding, final norm or head. All three are counted now, and the test
  expects them and fails on any operation the program does not have.
- **G-SB11-1:** a sum with an unmeasured field (in one operation, or across operations into the totals) reported the
  measured part as if it were the whole. The field now reads 0 and is listed unmeasured. The names `Unmeasured` lists are
  `Counts`' JSON names (`bytes_read`, `time_ns`, …), and a test holds the two sets equal.
- **G-SB11-2:** the grader now checks every norm's and the final norm's bytes, the embedding row and the head, from
  llama-tiny's shapes.
- **G-SB11-6:** re-measured as above.

**Step 2: CUDA, registered 2026-10-27 before its code (nobara-mm01).**
- **gpudev counts device activity on the `Device`, always on:** launches (`Queue.Launch` and `LaunchDirect`) and host waits
  (`Queue.Sync`, an event's wait, and the blocking copies to and from the host). Each is an atomic add. It is the independent
  count the resident's per-kernel counts are checked against. gpudev still imports nothing of goinfer's.
- **The resident counts per kernel when a counter is attached:**
  - each `r.launch` is one launch of its kernel, named by its field (the name cached per kernel);
  - a token's host waits go to the operation `sync`, its new leases (the MM3 ledger) to `alloc`, and any launch not made
    through `r.launch` to `launch_other`.
  - Bytes and time per kernel are listed unmeasured: a launch is asynchronous.
- **What OP4 takes over:** the operation an op's kernels belong to. Until then, the CUDA operations are kernel names.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB11-8 CUDA exact | llama-tiny resident on CUDA (int4), decode tokens 2 to 4 after a warm token: each token's per-kernel launches, syncs and allocs equal a committed golden (`-sb11.update` writes it, reviewed in the PR); the sum of per-kernel launches equals the device's own launch count (so `launch_other` is 0); a steady token allocates nothing; tokens 2, 3 and 4 count the same | no CUDA device; a timed run holds the card | runs to completion | seconds |
| G-SB11-9 CUDA plants | a planted extra launch (one kernel launched twice) moves that kernel's count and the device's by one each and turns the golden red; a planted extra sync moves `sync` by one | as G-SB11-8 | runs to completion | seconds |
| G-SB11-10 CUDA off cost | the always-on device counting (two atomic adds a launch at most) and the resident's nil check, × launches per token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on the RTX 2070 SUPER, is under 0.5%. Under the exclusive timing lock; the record notes the card's power state (`nvidia-smi` performance state and power limit) and load | heavy load (the 2026-10-19 rule) | one run | under two minutes |
| G-SB11-11 CUDA unchanged | the device suite's identity and KV-slot tests (`TestCUDAKVSlots_`, `TestPlanSlack_`, `TestLedger_`) pass unchanged; logits with the counter on equal logits with it off | as G-SB11-8 | runs to completion | about two minutes |

**Step 2 read, 2026-10-27 (nobara): all four pass.**
- **G-SB11-8:** a llama-tiny int4 decode token on CUDA is 34 launches of 9 kernels and 3 syncs, with no new lease. Per kernel:
  `gemvW4` 8; `bAttn`, `fGURows`, `fQ`, `fQKV`, `fSw` and `ropeKV` 4 each; `fRms` and `gemvW8` 1 each
  (`cuda/testdata/sb11_decode_llama_tiny.json`). Tokens 2, 3 and 4 count the same, and the per-kernel launches sum to the
  device's own count, so `launch_other` is 0.
- **G-SB11-9:** the planted second launch of one kernel moves that kernel's count and the device's by one each and turns the
  golden red; the planted extra sync moves `sync` by one.
- **G-SB11-10:** on Qwen2.5-Coder-0.5B q4_k_m on the RTX 2070 SUPER, a decode token takes 1.98 ms over 194 launches. The off
  path, benchmarked at its registered bound (two atomic adds and the nil check), takes 8.63 ns a launch: **0.0847% of a
  token**, against the registered 0.5%. The device counts one add per launch, so the bound is about twice the real cost.
  Conditions: the exclusive timing lock (it waited for another session's `touched.py`), the card at P8 with a 235 W limit
  before the run, load 2.03.
- **G-SB11-11:** the KV-slot, plan-slack, ledger and cost tests pass on the tree merged with step 1's review fixes
  (19 tests); logits with the counter on equal logits with it off.
- **Builds:** vet and the pinned staticcheck under `cuda goinfer_testhooks`, staticcheck on `internal/mem`,
  `internal/cost` and `decoder`, `touched.py`, the layout tests, the citation lint; a `changelog.d` fragment.

**Review fixes on #370, 2026-10-27 (Gemini's review, relayed by the coordinator).** The gates did not change; G-SB11-8
was tightened to grade what the schema promised.
- **Time:** every CUDA operation lists its time unmeasured, so the totals' time read 0 and `Timings()` gave no `decode_ms`,
  though the token's wall time was counted. A phase now carries its own wall time (`WallNs`, from `Counter.Wall`), and
  `Timings()` reads it first. The `token` operation is gone.
- **Units:** `Calls` counts operations run. The `sync` and `alloc` operations carry their counts only in `syncs` and
  `allocs`, and command buffers have their own field, `submits`. G-SB11-8 now checks that a token's wall time reaches
  `Timings()` and that the totals' calls equal its launches.
- **No ledger:** on a device without a ledger, `alloc` lists `allocs` and `alloc_bytes` unmeasured instead of reporting 0.
- **The off path:** a buffer's host wait counts on the `Device` the buffer carries, not through a map lookup on its context.
  `Launch` and `LaunchDirect` count after their argument checks, so a refused launch is not counted.
- **Not changed:** the CPU's batched embedding counts no position rows, which is correct. The batched walker refuses a
  model with learned positions (`program.batchEligible`), and `embedNInto` adds none.
- **Re-read:** the CUDA cost, KV-slot, plan-slack and ledger tests pass (19), and so does `gpudev`'s suite. Both run with
  `CGO_ENABLED=0`, as the device suite does; with cgo on, `TestCUDA_launchDirect` segfaults on `main` as well.

**Step 3: WebGPU, registered 2026-10-27 before its code (nobara-mm01).** It runs on nobara's RTX 2070 SUPER through
Vulkan. **Metal** stays a named skip until a Mac session builds it.
- **The `Context` counts its activity, always on:**
  - the dispatches recorded into a decode runner's passes: one atomic add of the step count per recording, not one per
    dispatch;
  - the command buffers submitted, and the blocking polls (`Poll(true)`), which are the host's waits on the device.
  - It is the independent count the runner's attribution is checked against.
- **The decode runner attributes a submitted token when a counter is attached:**
  - each step of the submitted buffer is one dispatch of its pipeline, named by the `Context` field that holds the pipeline
    (cached per pipeline);
  - the token's blocking polls go to `sync`, its submits to `submit`, its new leases on the `Context`'s ledger to `alloc`,
    and its wall time to `token`.
  - Bytes and time per dispatch are listed unmeasured: a dispatch is asynchronous.
- **Encode-ahead:** a token's buffer is recorded during the previous token. Steady tokens record exactly one buffer each, so
  the recorded count over a token still equals that token's attributed dispatches.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB11-12 WebGPU exact | llama-tiny resident on WebGPU (int4), decode tokens 2 to 4 after a warm token: each token's per-pipeline dispatches, submits, syncs and allocs equal a committed golden (`-sb11.update` writes it, reviewed in the PR); the attributed dispatches equal the `Context`'s recorded dispatches over the token; a steady token allocates nothing; tokens 2, 3 and 4 count the same | no WebGPU adapter; a timed run holds the card | runs to completion | seconds |
| G-SB11-13 WebGPU plants | a planted second recording of one step moves the `Context`'s recorded count by one and not the attribution, so the equality and the golden turn red; a planted extra blocking poll moves `sync` by one | as G-SB11-12 | runs to completion | seconds |
| G-SB11-14 WebGPU off cost | the always-on counting (three atomic adds a token at most, which the benchmark charges per dispatch as a bound) and the runner's nil check, × dispatches per token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on WebGPU (Vulkan, the 2070 SUPER), is under 0.5%. Under the exclusive timing lock; the record notes the card's performance state and the load | heavy load (the 2026-10-19 rule) | one run | under two minutes |
| G-SB11-15 WebGPU unchanged | the WebGPU KV-slot tests (`TestWebGPUKVSlots_`) and the ledger tests (`TestLedger_` in `gpu` and `gpu/internal/kernels`) pass unchanged; logits with the counter on equal logits with it off | as G-SB11-12 | runs to completion | about two minutes |

**Step 3 read, 2026-10-27 (nobara, Vulkan on the RTX 2070 SUPER): all four pass.**
- **G-SB11-12:** a llama-tiny int4 decode token on WebGPU is 54 dispatches of 7 pipelines, with one submit, one sync and
  no lease. Per pipeline: `gemvW4Pipeline` 28, `rmsQuantPipeline` 9; `attnKeysPipeline`, `qkvFinPipeline`,
  `quantizePipeline` and `swigluQuantPipeline` 4 each; `gemvPipeline` 1.
  - The golden is per architecture (`gpu/testdata/sb11_decode_llama_tiny.amd64.json`), as the admission census is, since
    an adapter can select other pipelines. A box without its architecture's golden fails and prints the counts to write.
  - Tokens 2, 3 and 4 count the same. The attributed dispatches equal the `Context`'s recorded ones (no `dispatch_other`).
    The wall time reaches `Timings()`, and the totals' calls equal the dispatches.
- **G-SB11-13:** the planted second recording gives `dispatch_other` one dispatch and leaves the attribution equal to the
  golden; the planted extra poll moves `sync` by one.
- **G-SB11-14:** on Qwen2.5-Coder-0.5B q4_k_m, a WebGPU decode token takes 3.45 ms over 386 dispatches. The off path,
  benchmarked at its registered bound (three atomic adds and the nil check, charged per dispatch), takes 12.87 ns a
  dispatch: **0.1442% of a token**, against the registered 0.5%. The `Context` pays the three adds once a token, so the
  bound is far above the real cost. Conditions: the exclusive timing lock, the card at P8 with a 235 W limit before the
  run, load 0.88, 58 GB available.
- **G-SB11-15:** the WebGPU KV-slot and ledger tests pass in `gpu` and `gpu/internal/kernels`, and logits are identical
  with the counter on and off. Three KV-slot tests skip as before: two apply on darwin only, and one is heavy.
- **Builds:** vet and the pinned staticcheck under `gpu goinfer_testhooks`, `touched.py`, the layout tests (`gpu` and
  `gpu/internal/kernels` → `internal/cost`), the citation lint; a `changelog.d` fragment.

**Review fixes on #377, 2026-10-27 (CI and Gemini, relayed by the coordinator).** The gates did not change; G-SB11-12 and
G-SB11-13 were tightened to grade every path that submits.
- **CI (gpu-darwin):** there was no arm64 golden. An architecture without one now runs every other check and then skips,
  printing its counts. A Mac session records `sb11_decode_llama_tiny.arm64.json` with `-sb11.update` and reviews it; the
  numbers CI printed are not pasted in.
- **Dispatches counted at the submit, per buffer:** the `Context` counted dispatches as they were recorded, while the runner
  attributed them at the submit. With encode-ahead, a token that recorded in line (the first, or any after `RunNoLogits`,
  `SetAdapter` or a dropped buffer) counted twice and reported a whole token as `dispatch_other`. A token also counted the
  next token's buffer, so the plant showed one token early. Each buffer now carries its recorded count, added when it is
  submitted. The test counts the first token, and the plant is graded on the token that submits the planted buffer, with
  the token that recorded it unchanged.
- **Every submit is attributed:** `RunNoLogits`, `RunBatch` (on its first runner's counter) and the on-device draw
  (`RunSample`, with its two Gumbel dispatches as `gumbel_sample`) count and attribute their dispatches, submits and waits.
  The test runs each and requires no `dispatch_other`. A plant that drops `RunNoLogits`'s attribution turns it red.
- **CPU, tree mask:** a single tree-masked token's attention is listed unmeasured, as the batched walker's already was
  (`TestCost_cpuTreeMaskUnmeasured`; red on the old code).

**Step 4: Metal, registered 2026-10-28 before its code (mac-vk5grade; the coordinator's job, Francis 09:47).** It runs on the
M1 Pro, on battery during the power cut, so the gates below are correctness runs on llama-tiny; the off-cost gate waits for AC.
- **gpudev's `Device` counts its activity, always on, one atomic add each:**
  - the dispatches encoded (`Encoder.Dispatch`, `Dispatch2D`, `DispatchTG` and the `Queue.Run*` helpers);
  - the command buffers committed (`Commit`, `End`, `Run*`);
  - the host's waits on them (`WaitDone`, `End`, `Run*`).
  - It is the independent count the resident's attribution is checked against.
- **Counted when encoded.** The encode-ahead executor encodes the next token's buffer inside the current token's call, before
  the call returns. So a steady token encodes one buffer and commits one, and the token's window holds both counts. A buffer
  encoded while no counter is attached and committed while one is shows up as `dispatch_other`, never as a missing count.
- **The resident attributes the token's dispatches when a counter is attached:**
  - an encoder begun while a counter is attached reports each dispatch's pipeline, and the resident names it by the
    `resident` field that holds it (cached per pipeline; "pipeline" when no field does);
  - the token's waits go to `sync`, its command buffers to `submit`, its new leases on the device's ledger to `alloc`, its
    wall time to the phase's `WallNs`, and any counted dispatch not attributed to `dispatch_other`.
  - Bytes and time per dispatch are listed unmeasured: a dispatch is asynchronous.

| gate | rule (pass) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB11-16 Metal exact | llama-tiny resident on Metal (int4), through the production decode path (the encode-ahead executor): a first token, then tokens 1 to 3. Each of tokens 1 to 3 counts the same per-pipeline dispatches, submits, syncs and allocs, equal to a committed golden (`metal/testdata/sb11_decode_llama_tiny.arm64.json`; `-sb11.update` writes it, reviewed). Every token's attributed dispatches equal the device's (no `dispatch_other`), including a token without logits and the synchronous forward. A steady token allocates nothing. The wall time reaches `Timings()`, and the totals' calls equal the dispatches. Logits with the counter on equal logits with it off | no Metal device; another session's timed run holds the Mac | runs to completion | seconds |
| G-SB11-17 Metal plants | a planted extra command buffer (one more committed and waited on, per token) moves `submit` and `sync` by one each and turns the golden red; a planted dispatch encoded past the attribution moves `dispatch_other` by one and leaves the attribution equal to the golden | as G-SB11-16 | runs to completion | seconds |
| G-SB11-18 Metal off cost | the always-on counting (three atomic adds a dispatch, charged as a bound) and the hook's nil check, × dispatches per token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on Metal, is under 0.5%. Under the exclusive timing lock, **on AC**; the record notes the power state and the load. Not run on battery; it is open until the Mac is on AC | heavy load (the 2026-10-19 rule); on battery | one run | under two minutes |
| G-SB11-19 Metal unchanged | the Metal ledger and KV-slot tests (`TestLedger_` and `TestMetalKVSlots_` in `metal/`) and gpudev's own tests pass unchanged | as G-SB11-16 | runs to completion | about two minutes |
| G-SB11-20 builds | vet on `metal/` (untagged and `goinfer_testhooks`), the pinned staticcheck with its canary, gofmt, `touched.py`, the layout tests, the citation lint; a `changelog.d` fragment | — | runs to completion | minutes |

**Step 4 read, 2026-10-28 (the M1 Pro, macOS 26.6.2, on battery): G-SB11-16, 17, 19 and 20 pass; G-SB11-18 is open until AC.**
- **G-SB11-16:** a llama-tiny int4 decode token on Metal is 46 dispatches of 11 pipelines, one command buffer and one wait,
  with no lease. Per pipeline: `pRms` 9; `pAttn`, `pGemvResidStaged`, `pKv`, `pQv`, `pRope2`, `pSABiasRows`, `pSAResidRows`,
  `pSARows` and `pSw` 4 each; `pGemvW8` (the head) 1. The golden is `metal/testdata/sb11_decode_llama_tiny.arm64.json`.
  - Tokens 1, 2 and 3 count the same. Every pipeline is named by its field (none falls back to "pipeline"). The attributed
    dispatches equal the device's on the first token (encoded in line), on tokens 1 to 3, on a token without logits and on
    the synchronous forward: no `dispatch_other`.
  - The wall time reaches `Timings()`, the totals' calls equal the dispatches, and logits are identical with the counter on
    and off.
- **G-SB11-17:** the planted extra command buffer moves `submit` and `sync` by one each, and the counts no longer equal the
  golden. The dispatch encoded past the attribution gives one `dispatch_other` dispatch and leaves the attribution equal to
  the golden. A mutation that stops `Commit` and `End` counting their buffers turns the test red on the plant and on the
  golden.
- **G-SB11-18, not run:** a timed share needs the exclusive timing lock on AC, and the Mac is on battery. Open; the first run
  on AC reads it.
- **G-SB11-19:** the Metal ledger tests (6 pass; `_alias` skips, as before, for want of its untracked `gemma4-dense-scaled`
  fixture) and KV-slot tests (3) pass, and so do gpudev's 40 tests (4 env-gated probes skip, as before).
- **G-SB11-20:** vet on `metal/` (untagged and `goinfer_testhooks`). The pinned staticcheck (v0.8.1) with
  `goinfer_testhooks` is clean, and its canary is red (U1000). Untagged, it reports `costToken` and its helpers unused,
  beside the 13 findings it reports on `main`, since only the tagged tests call them; CI runs no staticcheck on `metal/`. Also gofmt, `touched.py` green, the
  layout data (`metal → internal/cost`, and the test arrow) and the citation lint, with a `changelog.d` fragment.
  - The test is in `metal/kv_slots_test.go`, as WebGPU's is in `gpu/kv_slots_test.go`. A new file importing `decoder` would
    raise the backend-test ratchet (`TestBackendTests_decoderImportsOnlyFall`).

**Review fixes on #411, 2026-10-28 (Gemini's review, relayed by the coordinator).** The gates did not change. G-SB11-16 and
G-SB11-17 were tightened to grade what the design above promised.
- **Dispatches counted at the commit (finding 1, holds).** The device counted a dispatch when it was encoded, at the same call
  that attributed it. So a buffer encoded with no counter attached and committed by a counted token appeared in neither
  count, and `dispatch_other` could not see it. The design above said it would. The planted unattributed dispatch only showed
  because the plant added to the device's count directly.
  - Now an encoder counts its dispatches in a plain field and keeps their pipelines when a hook was set as it began. The commit
    adds the count to the device with one atomic add, and reports the kept pipelines to the hook set then.
  - The plant seam is gone. The test runs an uncounted token, then a counted one: all 46 dispatches of the buffer the uncounted
    token encoded are `dispatch_other`, nothing is attributed, and the next token counts as a steady one.
  - A mutation that puts the device's count back at the encode turns the test red.
  - The off path is now a plain increment per dispatch and three atomic adds per command buffer.
- **Allocations (finding 2, does not hold).** `alloc` reads `Usage.Reserved`, the leases ever reserved, which only grows.
  `Leases`, the live count, is not read. So a lease taken and released inside a token counts once, and a release never
  subtracts. A test now pins it: a 4096-byte lease taken and released inside a token counts 1 alloc of 4096 bytes. A
  mutation to the live count turns it red.
- **G-SB11-18 is owed (finding 3).** The always-on counting ships with its off-cost share unmeasured, because a timed run needs
  AC and the exclusive timing lock. It is owed after the outage: the first session on the Mac on AC runs it, before any
  further Metal decode work is timed. If the share is over 0.5%, the counting moves behind the hook.
- **Re-read:** the cost test, the ledger and KV-slot tests (9), gpudev's suite, vet, the tagged staticcheck and `touched.py`
  pass. The golden did not change.

**Review fix on #412, 2026-10-28 (Gemini, relayed by the coordinator).** The gates did not change.
- **The first counted token (holds).** An encoder kept its pipelines only when a hook was set as it began. The executor
  encodes each token's buffer during the token before it, so one counted token after uncounted decode committed a buffer
  encoded uncounted: all 46 of its dispatches were `dispatch_other`, and none were attributed.
- **The fix:** when a counter is attached and the executor holds a buffer it encoded ahead uncounted, it drops that buffer and
  encodes it again under the counter, as it already does when the attention plan changes. Nothing is recorded per dispatch
  while no counter is attached; the off path gains one atomic load a token.
- **The test:** one counted token after uncounted decode counts as a steady token, equal to the golden. A buffer encoded
  uncounted and committed by a counted token is still `dispatch_other`, through any path that does not re-encode. The plant
  keeps the stale buffer and gives 46 of 46 unattributed, and the token after it is steady again. A mutation that removes the
  re-encode turns the test red, with exactly Gemini's counts.
- **Re-read:** the cost test, the ledger and KV-slot tests (9), vet, the tagged staticcheck and `touched.py` (the whole
  `metal` package) pass. G-SB11-18 is still owed on AC.

**G-SB11-18's method, written 2026-10-29 07:20 PDT before the run (mac-sb11m; the coordinator's job of 07:20).** The rule,
void conditions, stopping rule and cost basis in the table above do not change. This fixes how each term is read.
- **The instrument:** `TestCost_metalOffCost` in `metal/kv_slots_test.go`, with `GOINFER_SB11_OFFCOST_MODEL` set to the 0.5B's
  Metal int4 `.giw` in `~/models`, one process, under `timing_lock.py run` (exclusive). The run script records the power state
  (`pmset -g batt`), the 1-minute load and `hostmem -reclaimable` at its start.
- **A decode token:** `Forward` through the encode-ahead executor, uncounted, timed per call. 16 warm-up tokens are not timed;
  the median of the next 64 is the token.
- **Dispatches per token:** the device's own `Activity` over those 64 tokens, divided by 64.
- **The bound per dispatch:** three atomic adds and one atomic pointer load with its nil check, timed in a loop of 10^7 in the
  same process, five repeats, the slowest repeat taken. The off path today is cheaper than this (a plain increment a dispatch,
  three atomic adds a command buffer, one atomic load a token), so the charge stays a bound.
- **Share** = bound × dispatches per token ÷ the median token. Pass under 0.5%; at or over 0.5% it is a finding for Francis,
  and the bound does not move.
- **Context, not graded: the on cost.** Blocks of 9 tokens, uncounted and counted (each counted token through `costToken`), in
  the order A B B A, four rounds. The first token of every block is dropped from both arms: the first counted token after
  uncounted decode re-encodes the buffer encoded ahead (two CPU encodes, no overlap; Gemini on #413), which is a switch cost,
  not a counted token's. Reported as the median counted ÷ median uncounted.

**G-SB11-18 read, 2026-10-29 07:37 PDT (the M1 Pro, on AC, battery 97% and charging): PASS, 0.043% against the 0.5% bar.**
- **The run:** `TestCost_metalOffCost` at 10fdbcad (the method above), on
  `~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v17.giw`, under the exclusive timing lock. At the start the 1-minute
  load was 3.24 and `hostmem -reclaimable` read 7.97 GB, against about 0.3 GB of weights plus the 2 GB margin.
- **Two starts were refused before it, not run and not counted:** the 1-minute load was 6.69 at 07:31 and 6.03 at 07:32, both
  over 4.0. Before that, two tries found other sessions' loaders holding the lock shared (mac-tack3's `touched -all`, then
  ax01's rerun) and waited them out.
- **A decode token:** median 4661.2 µs over 64 uncounted tokens (min 4391.9, max 8592.2). Each token is 266.00 dispatches,
  1.00 command buffer and 1.00 wait, read from the device's `Activity`.
- **The bound:** 7.53 ns a dispatch (the slowest of five repeats of three atomic adds and a hook nil check). 266 dispatches
  cost 2.00 µs a token, so **the share is 0.0430%**, 11.6× under the bar. The counting stays always on.
- **Context (not graded), the on cost:** counted tokens have a median of 5031.4 µs over 64 and uncounted tokens 4971.0 µs over
  64, **1.0121×**. The first token of each block was dropped, as the method says. A counted token also attaches the hook,
  builds the pipeline-name map by reflection and fills a fresh counter, so this is the price of counting a token, not of the
  off path.
- **The record:** `docs/measurements/sb11-metal-offcost-2026-10-29/`: the passing run's log (`run-pass.log`), the two refused
  starts (`refused-heavy-load.log`) and the two scripts that ran it.
- **Checks on the change:** gofmt and `go vet -tags goinfer_testhooks ./metal/` are clean. The tests in `metal/kv_slots_test.go`
  pass (`TestCost_metalDecodeExact` included), and the new test skips without its variable. `touched.py` was red on `metal`,
  for reasons outside the change. Six tests need gitignored fixtures a fresh worktree lacks (`gpt2`, `granite-tiny`,
  `kimi-tiny`), and they pass once the fixtures are cloned in. One run crashed and a full `-v` run did not repeat it; other
  sessions' Metal work was running at the time. `TestI8_exactKernelModelIsSplit` reads `~/models/qwen3.5-0.8b` and fails on
  this Mac whatever the change: exact attention reads false, at head dim 0. That is reported to the coordinator, not
  fixed here.

#### SB11c — Metal's cost counter behind `goinfer_testhooks`: gates, registered 2026-10-29 before the code (mac-sb11m)

The ledger's SB11c, the row's second option. The first option, a production caller, would be SB11b's report on the wire,
which waits on Francis. The change:
- `costToken`, `pipelineNames` and `dispatchUnmeasured` move from `metal/cost.go` to `metal/cost_testhooks.go`
  (`//go:build darwin && goinfer_testhooks`).
- The per-resident pipeline-name cache leaves the `resident` struct (a field cannot be build-tagged) for a map in that file,
  keyed by the resident, so a counted token still reads the fields once per resident.
- `costPlantForTest` stays untagged, because the executor reads it.
- Nothing untagged changes behavior.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| G-SB11c-1 the untagged staticcheck finding is gone | the pinned staticcheck (v0.8.1, canary red) on `metal/` untagged reports none of `costToken`, `pipelineNames`, `dispatchUnmeasured`, `costNames`. Red first: before the change it reports all four. Its other findings (unused test helpers outside SB11) are listed, not fixed | never | one run | seconds |
| G-SB11c-2 the tagged build holds | `go vet` and the pinned staticcheck on `metal/` with `-tags goinfer_testhooks` report nothing in `cost_testhooks.go`; gofmt clean; `touched.py` green | never | one run | seconds |
| G-SB11c-3 the counters count as before | `TestCost_metalDecodeExact` and `TestCost_metalOffCost` (llama-tiny, `-tags goinfer_testhooks`) pass unchanged, with the golden `testdata/sb11_decode_llama_tiny.arm64.json` not edited, under the timing lock held shared, as the only Metal run on the Mac | another session's Metal test overlaps: rerun | one run | about 1 min |

**SB11c read, 2026-10-29 (the M1 Pro, on AC): G-SB11c-1 to 3 pass.**
- **G-SB11c-1:** the pinned staticcheck on `metal/` untagged reported `dispatchUnmeasured`, `costToken`, `pipelineNames` and
  the `costNames` field before the change, and none of them after. Its other untagged findings in `metal/` are outside SB11c
  and are left as they are: test-only helpers such as `forwardTrunkForTest`, `prefillEModel`, `prefillG4Paged` and
  `dequantG32`, called only from tagged tests.
- **G-SB11c-2:** vet untagged and with `goinfer_testhooks`; the tagged staticcheck reports nothing in `cost.go` or
  `cost_testhooks.go`; gofmt and `touched.py` clean.
- **G-SB11c-3:** `TestCost_metalDecodeExact` passes with its golden unedited (11 pipelines, 46 dispatches, 1 submit, 1 sync).
  `TestCost_metalOffCost` skips without its model, so it was run as G-SB11-18 was, under the exclusive timing lock (17:56,
  load 2.38, on AC). It passes: the off cost is 0.0397% of a 0.5B decode token (bar 0.5%), and the on cost, not graded, is
  1.0031x. Log: `docs/measurements/sb11-metal-offcost-2026-10-29/g-sb11c-3-offcost.log`.

### SB12 — gate and measurement infrastructure

- **Today:** `cmd/gate` (18,847 lines of Go, 13,321 of them non-test) and `scripts/` (289 Python files, touched by 426
  commits in 60 days).
- **Contract:** runners launch, lock and record; graders are pure functions of records.
- **Fast check:** graders run on recorded fixtures with plants, as the CUDA device-suite judge already does with fake
  streams. M.

#### SB12 — graders as functions of records: gates, registered 2026-10-27 before the code (mbp2-sb4)

Francis picked SB12 at 17:41 (relayed by the coordinator). CPU only on mbp2 (`go test -p 4`), tier 0, no timed run and no
timing lock. Each step is its own green PR.

**What the census found** (`cmd/gate` at `8b6e3fb8`; the full table goes in `docs/measurements/sb12-2026-10.md`):
- **Already pure.** `reportCensus`, `reportTally`, `classifyChecks` (ledger injected), `extraBlockers`, the release
  rows' `gpuCoverage`, `parityCoverage` and `quickCoverage`, `ciVerdict`, `changelogOwed`, `compareComments`,
  `judgeCell`, `summarizeFamilies` and `judgeGiwBytes` take parsed data and decide.
- **Interleaved.** The gpu gate decides inside methods that also exec (`cudaSuite`, `cudaHeavy`, `seam` and the rest) and
  in `verdictLabel`, which also execs git and reads the clock. Also `runParity`'s final sum, `runPTX`, `runMutation`,
  `shadow` and `runCommentsOnly`'s overall step.
- **No raw input is committed.** The committed records (`docs/measurements/gpu-gate` 9, `parity-sweep` 2,
  `quick-full` 4) are summaries; the `go test -json` streams exist only in `-logdir` while a run lasts. Only `census`
  and `heavy` can replay a stream (`-stream`).

**The contract.** A runner launches, holds the lock it needs and writes a **run record**: a directory with `run.json`
(the gate, its configuration, each cell's name, command, exit code and stream file, and the side inputs the verdict
reads, such as the gate ledger and the parity check list) and one `go test -json` stream per cell. A grader is a
function from a record to a verdict (exit code and report text), in a file that imports no `os/exec`, reads no
environment and no clock, and writes nothing. `gate <family> -replay DIR` regrades a record.

**Goldens.** `testdata/gate/goldens/<gate>/<case>/`: a recorded input and the verdict today's code gives on it (exit code
and report text, provenance lines excluded), captured at this registration's base, before any grader moves. The inputs
are the committed records, the gate ledger and the committed `.giw` byte-identity records for the graders that read
records, and run records captured on mbp2 for the graders that read streams (narrowed with `GATE_RUN` or a package list,
so each takes seconds to a minute to record).

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C1 the census holds | a test lists every subcommand `cmd/gate`'s dispatcher accepts and fails on one the census table in the record does not name, or a name the dispatcher no longer accepts | never | one run | seconds |
| SB12-C2 goldens reproduce | `TestGraderGoldens` regrades every case and fails on any difference in exit code or text. At the base commit it is green by construction; after each step it must stay green with no golden edited | a golden edited in the same commit as a grader: void, and the step is redone | every case, every step | seconds |
| SB12-C3 a grader is a function | per family moved: the grader's file imports none of `os/exec`, `os` (beyond types), `time` (beyond types) or `net`, and no function in it reads the environment (a source-scan test, the list of moved graders in the test) | never | one run | seconds |
| SB12-C4 each grader can go red | per family moved: a fixture test, and at least one plant (an edited record field or stream event) that turns the verdict red; the plant is shown red with the grader's check removed once (`gate mutation` or a hand edit, recorded) | never | one run each | seconds |
| SB12-C5 replay equals the run | per family moved, on a box that can run it: one live run with `-record`, then `-replay` of that record, gives the same exit code and the same verdict text. mbp2 shows it for the CPU families; the GPU families' live runs are owed by nobara and the Mac, named in the record | the live run's own precondition refuses (exit 2): rerun | one live run per family | the family's own run time, stated before it starts |

**Order.** Step 1 is the census, the record format, the goldens and C1–C2, with no grader moved. Then one family per PR,
most recorded inputs first: the record-reading graders (release rows, `gpu-records`, ledger, `.giw` bytes) need only
their I/O lifted, then the stream graders (census and heavy, parity, quick, identity), then the gpu gate, whose
graders are the most interleaved. **Any step whose goldens cannot all reproduce stops with DECISION NEEDED: no gate's
verdict changes without Francis.**

**Decided 2026-10-27 18:17 (Francis, option A):** `gate heavy -stream` refuses with exit 2 and names `-replay FILE`.
It is a verdict change he approved: a wrong RED becomes refused (`docs/measurements/sb12-2026-10.md` §8).

#### SB12 — the CUDA ledger test's open-device guard: gates, registered 2026-10-28 before the code (nobara-ci01)

Francis chose option (a) of `docs/measurements/sb12-memtests-2026-10-28.md` §3 at 12:05: keep the exact bound of
`TestLedger_closesAgainstDevice` and count open devices. The `cuda` test package records every device
`CreateSystemDefaultDevice` creates (through the leaser hook) until its release; the ledger test fails at its start,
naming the open count, when an earlier device is still open; the suite's release check fails on any device never
released by the end. The seven tests §2 left as found release their devices, and two of Gemini's findings on #406 are
answered. No bound moves.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-L1 an open device is named | a plant test creates a device and leaves it open: the open-device check names one open device and the test that created it; after its release the check is empty. Run on the card | no CUDA device (skip, not a pass) | one run | seconds |
| SB12-L2 the pool probe holds its docstring | `TestSmallAllocPool_sharedAcrossDevices` asserts the second device's small buffers move free memory by exactly 0 bytes and fails on a `MemInfo` error; its plant (the first device released before the second is created) goes red, shown once by hand | no CUDA device | one run, one plant | seconds |
| SB12-L3 the failing groups pass | `cuda-parity` and `cuda-heavy-small`, each run once as `gate gpu` runs its cell (tags, `CGO_ENABLED=0`, `-count=1`, the small group's `-run` list and env): `TestLedger_closesAgainstDevice` passes at its exact bound with no open device at its start, and the release check reports no device left open | the timing lock is held exclusively (wait, never run beside it); the parity cell's own 10 min timeout fires before the ledger test finishes (rerun with the timeout stated in the record) | one run of each group. A device left open by a test other than the seven is a finding: a missing release is fixed in the test and the group rerun; anything else stops with DECISION NEEDED | `cuda-parity` about 10 min and `cuda-heavy-small` 4,049 s, both from the 2026-10-27 night `gate gpu` run on nobara |

**Read 2026-10-29 (nobara-ci01):**
- L1 and L2 are green, and their plants go red.
- L3 is **red on one named product leak**: `cuda.VisionEncoder.Close` frees buffers and never releases the device. In both
  groups, the ledger test passes at its exact bound.
- Francis, 08:47: #415 stays test-only, and the fix is its own PR. L3's heavy-small rerun follows that fix.
- The record is `docs/measurements/sb12-memtests-2026-10-28.md` §5.
- **L3 is GREEN on the third round** (2026-10-29, after #430's fix, `8c70146d4`). Both groups pass the ledger test at its
  exact bound, and neither names a device left open. They exit 1 only on the I1/I2 breaks main already had (same record).

#### SB12 — the CUDA suite's I1 breaks: gate, written 2026-10-29 before the green run (nobara-ci01)

The coordinator asked at 11:05 for a fix of the I1/I2 breaks main shows. This gate was written after the site diagnostic and
the fixes were committed, and before the run that grades them. It was amended before that run, after Gemini's review of
#439: heavy-small in full instead of its nine named tests, and a void for another CUDA process during a step. The fix also
moved from a whole job on the test line to test-line helpers (`tf`, `ti`, `tup32`, …), so the code under test keeps the job's line.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-I1 every released device is planned | `cuda-parity` and `cuda-heavy-small` in full, and the `qwen25vl-3b`, `7b-8b` and `qwen3-4b` heavy groups, each run as `gate gpu` runs its cell: no `LEDGER BREAK` line and no I1/I2 line. The red is the site run at `bac9dc6d7` (and the 2026-10-27 night's three heavy groups on main). A break that is a test's own buffers on an empty line is fixed in the test and re-run. A product allocation on no line, or any fix that would move a bound or a price, stops with DECISION NEEDED | another CUDA process on the card at a step's start (the script waits) or while it runs (its sampler logs it); a step's own timeout | one run of each | parity about 12 min, heavy-small about 65 min (today's runs); the three groups 3, 12 and 14 min (the 2026-10-27 night) |

**Read 2026-10-29 (nobara-ci01): SB12-I1 is GREEN in all five cells.**
- Every I1 break the site report named (13, in cuda-parity and cuda-heavy-small) was a test's own buffers on no line, fixed
  in #439.
- The three heavy groups' first green runs found two more of the same shape, which the source sweep could not see. After
  that fix (#456), `qwen25vl-3b`, `7b-8b` and `qwen3-4b` ran the night's tests with no break.
- No other CUDA process was logged during any step, though heavy-small's sampler is not proven alive for its last 33 minutes.
- The record is `docs/measurements/sb12-i1i2-2026-10-29.md`.

#### SB12 step 8 (Metal) — the gpu gate's Metal cell groups grade a recorded run: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's ask of 12:42: the next unassigned step that needs this Mac. SB12's next step is the gpu gate's check groups
split into runner and grader (§11 of `docs/measurements/sb12-2026-10.md`, "What is not done"). This step takes the four Metal
groups that are one `go test` cell each: `suite`, `parity`, `lifecycle` and `prefill`. Their live recording is only possible on a
Metal box. `cgofree` (a build and `otool`), `webgpu` (a probe), `repo`, `cleangpu`, `seam` and the CUDA groups are later steps.
- **The runner.** With `-logdir`, `gate gpu` also writes a run record, `gate_gpu_run.json`, through `writeRunRecord`. The
  record holds each cell it ran, with its exit code and stream, and the Metal groups' side input: the checkpoint the
  lifecycle and prefill groups need, and whether it was present. The gate's verdict, its record (`gpu_record.go`) and its
  output do not change.
- **The grader.** A function from a group's spec, the side input and its cell's graded result and text to the group's state
  (pass, fail or skip), its line and its detail or evidence lines. It goes in its own file, with the line pickers it needs
  (`detail`'s and `evidence`'s line choice, `failureLines`, `crashExcerpt`). The live gate prints through the same function.
- **The replay.** `gate gpu -replay FILE` regrades those four groups from a run record. It prints their sections as the run
  did, then a line naming the groups it did not regrade. Its exit code is 1 when a regraded group failed and 0 otherwise; it
  gives no gate verdict, because that needs every group.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 8) goldens reproduce | `TestGraderGoldens` is green with no existing golden edited | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 8) a grader is a function | the new grader file is in the source-scan test's list and imports none of `os/exec`, `os`, `time` or `net` | never | one run | seconds |
| SB12-C4 (step 8) each group can go red | from a green recorded cell: a stream with a test turned to `fail` gives FAIL for each of the four groups; the lifecycle and prefill cells with every test turned to `skip` give FAIL (vacuous); an absent checkpoint gives SKIP with no cell; a crash in a stream gives FAIL with the crash head. The vacuous plant is shown red once with the grader's vacuous check removed (a hand edit, recorded) | never | one run each | seconds |
| SB12-C5 (step 8) replay equals the run | on the M1 Pro, one live `gate gpu -logdir DIR` at this step's commit, then `gate gpu -replay DIR/gate_gpu_run.json`: the four sections equal the live run's byte for byte, without the runner's heartbeat lines (`   ... <cell>: <time> elapsed, …`). Committed as a golden case (`gpu-metal-cells`: the run record, its four streams and the live sections) | the live run's own precondition refuses (exit 2): rerun | one live run | about 5 min (the 2026-10-29 recorded run: 4 min 45 s) |
| SB12-C6 (step 8) nothing else moved | the gate's verdict and record on that live run equal those of a run of the same groups at `origin/main` in its group counts and label (the 2026-10-29 record at `e4dd3ed5`: 9 of 9, 10 pass, 2 skip, 0 fail, PASS), and `touched.py`, vet and the pinned staticcheck on `cmd/gate` are clean | another session's Metal tests overlap a cell (the sampler shows it): rerun | one run | inside C5's run |

**Step 8 (Metal) read, 2026-10-29 (the M1 Pro, on AC): C2–C6 pass.** The record is `docs/measurements/sb12-2026-10.md` §16.
- C5: a live `gate gpu -logdir` at `f156a9f0`, then `-replay` of its record. The four sections are equal byte for byte. The
  golden is `testdata/gate/goldens/gpu-metal-cells/`.
- C6: the same verdict as §15's run (9 of 9 groups, 10 pass, 2 skip, 0 fail, PASS).
- One earlier live run at `fb2a45cb` is void: another session's Metal test overlapped its parity cell. It was rerun.

**Gemini's review of #449, 2026-10-29 (relayed by the coordinator).** The gates did not change. Findings 1 to 3 hold and are
fixed, red first; finding 4 does not hold for a run. The record is `docs/measurements/sb12-2026-10.md` §16, "Gemini's review
of #449, answered".
- **Finding 1:** the groups that fail on an all-skip cell, and those that need the checkpoint, are now pinned by name.
- **Finding 2:** the replay exits 1 on an empty or all-skip cell, as the live gate's tally does.
- **Finding 3:** the failed-test plant now reaches `failureLines`.
- An all-skip suite still passes, live as before; changing that is a verdict change, which is Francis's call.

#### SB12 — `TestNemotronResidentCUDA_tier1` unset `NEMOTRON_TIER1` default: gates, registered 2026-10-29 before the code (agy-sb12)

**Decided 2026-10-29 07:25 PDT (Francis, option (a); registered before code by agy-sb12):** `cuda/mamba_fit_test.go:TestNemotronResidentCUDA_tier1`
(the sole test in `cmd/gate/heavy_groups.go:heavyFamilies`'s `nemotron-h` heavy group) defaults `NEMOTRON_TIER1` when unset to `"4b"`
(OP2b's registered tier-1 whole-model cell, `~/models/nemotron-h-4b-instruct-128k`, `docs/tasks/task-forward-architecture-2026-10.md` §4.7).
When `NEMOTRON_TIER1` was not set explicitly by the caller and a checkpoint is absent on disk, the test calls `t.Skipf` naming the
path so `gate gpu` grades `nemotron-h` as a named all-skip group and lists `TestNemotronResidentCUDA_tier1` in the 2c skip census
(never a silent skip); when `NEMOTRON_TIER1` was set explicitly, a missing checkpoint still calls `t.Fatalf("VOID: %s is missing", c.path)`.
The default lives in the test (`cuda/mamba_fit_test.go:nemotronTier1Cells`) rather than `cmd/gate/gpu.go:gpuGate.runHeavyGroup` so
that (1) the test itself distinguishes an unset default from an explicit `NEMOTRON_TIER1=4b` without a second coordination env var,
and (2) running `go test` directly, `gate cuda-heavy -group nemotron-h` and `gate gpu` behave identically without `cmd/gate` carrying
model-specific env defaults.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-N1 unset default and explicit cell checks | `cuda/mamba_fit_test.go:TestNemotronTier1Cells` verifies that unset `NEMOTRON_TIER1` (`""`) resolves to the dense 1.5B control plus `"4b"` with `explicit=false` and skips naming the missing path when a checkpoint is absent, that an explicit setting (`"4b"`, `"9bslice, 9b"`) sets `explicit=true` and fails with `VOID: <path> is missing` when a checkpoint is absent, and that an explicit unknown cell (`"bogus"`, `"4b,"`) returns an error naming the unknown cell; `cmd/gate/heavy_groups_test.go:TestHeavyPartition_cudaTreeFitsTheTable` verifies that `nemotron-h` contains only `TestNemotronResidentCUDA_tier1` (so `TestNemotronTier1Cells` lands in `small` and `nemotron-h` still grades as an all-skip group when its checkpoints are absent) | never | one run | seconds |
| SB12-N2 live `TestNemotronResidentCUDA_tier1` on nobara with `NEMOTRON_TIER1` unset | `GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 40m -tags 'cuda goinfer_testhooks' -run '^TestNemotronResidentCUDA_tier1$' -v ./cuda/` on nobara (via `nobara-gotest`) with `NEMOTRON_TIER1` unset runs the dense Qwen2.5-Coder 1.5B control (`min cosine >= 0.95`, `NaN == 0`) and the 4B whole Nemotron-H checkpoint (`~/models/nemotron-h-4b-instruct-128k`, `min cosine >= 0.88`, `argmax >= 50%`, `NaN == 0`) and passes | timing lock held or another compute process on the GPU (`nobara-gotest` exits 3; wait and retry), or dense control min cosine < 0.95 | one run | ~25 s wall, ~16.5 GiB VmHWM |
| SB12-N3 builds and lints | `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go vet -tags 'cuda goinfer_testhooks' ./cuda/...`, `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 ~/go/bin/staticcheck -tags 'cuda goinfer_testhooks' ./cuda/...` (canary shown red), `gofmt -l`, `python3 scripts/touched.py`, `python3 scripts/queue_citation_lint.py` (exit code read directly), and `python3 scripts/changelog.py render` | never | one run | about a minute |

**Read 2026-10-29 (agy-sb12): SB12-N1 to SB12-N3 all PASS.**
- **SB12-N1:** `TestNemotronTier1Cells` passes on nobara (`0.00s` at `096cf82d`), and `TestHeavyPartition_cudaTreeFitsTheTable` passes in `./cmd/gate`.
- **SB12-N2:** `TestNemotronResidentCUDA_tier1` on nobara via `nobara-gotest` at `096cf82d` with `NEMOTRON_TIER1` unset (`58.40s` total; 8 devices released):
  - `control: dense Qwen2.5-Coder 1.5B (cuda-resident (int4))`: `min cosine 0.997580` (`>= 0.95`), `argmax 29/32`, `NaN rows 0`, first step `0.00 s`, mean decode step `4.6 ms` (`218.1 tok/s`, recorded, not graded), cell `15 s` -> `PASS`.
  - `Nemotron-H 4B whole (cuda-resident (int4))`: `min cosine 0.993996` (`>= 0.88`), `argmax 30/32` (`>= 50%`), `NaN rows 0`, recurrent checkpoint `86.5 MiB` (48 buffers, KV8a), prefill width `2048` rows (`500 MiB` scratch), first step `0.15 s`, mean decode step `13.2 ms` (`75.5 tok/s`, recorded, not graded), cell `43 s` -> `PASS`.
- **SB12-N3:** `go vet` and pinned `staticcheck` v0.8.1 under `-tags 'cuda goinfer_testhooks'` on `./cuda/...` (canary red, `U1000`), `gofmt -l`, `touched.py`, `queue_citation_lint.py` and `changelog.py render` all clean.

#### SB12 step 9 (Metal) — the gpu gate's cgofree and webgpu groups grade a recorded run: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's ask of 14:04: the next item that needs this Mac. After step 8, the Metal groups still grading inside the
runner are `cgofree`, `webgpu`, `repo`, `cleangpu` and `seam`. This step takes `cgofree` and `webgpu`, whose inputs a run
can record whole:
- **`cgofree`:** whether `metal/cmd/serve` built with `CGO_ENABLED=0`, and `otool -L`'s output on it (or that `otool`
  failed).
- **`webgpu`:** the adapter probe's state (opted out, failed, absent, or present with its backend) and its two cells.

`repo` runs CI's hygiene steps, and `cleangpu` and `seam` are later steps.
- **The runner and the replay share one sequence.** The Metal groups run in one order, with their cells and side inputs
  from a source. The live gate's source runs them and records what it got. The replay's source reads them from the run
  record, so the two cannot print the groups differently.
- **The grader.** `cgofree`'s verdict is a function of the build result and `otool`'s recorded output; the grader picks the
  linked-framework lines. `webgpu`'s is a function of the probe and its cells' graded results and text: the suite first, and
  parity only when the suite passed, as today.
- **The replay** regrades the Metal groups from step 8 plus these two, in the order the run printed them. The groups it does
  not regrade are `cleangpu`, `seam` and `repo`.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 9) goldens reproduce | `TestGraderGoldens` green; the step 8 golden `gpu-metal-cells` is not edited, and still equals its live run for the four groups it holds | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 9) a grader is a function | the new grader code is in a `grade_` file and `TestGraders_arePure` covers it | never | one run | seconds |
| SB12-C4 (step 9) each group can go red | `cgofree`: a failed build gives FAIL; `otool` output naming `Metal.framework` gives FAIL listing the line; `otool` failing gives PASS, as today. `webgpu`: opt-out and no adapter give SKIP; a failed probe gives FAIL; a failed suite gives FAIL and no parity verdict; an all-skip parity cell gives FAIL; a failed parity test gives FAIL with its assertion line. A test names, by group, the cells that must fail on an all-skip run (`webgpu-parity` with step 8's three), red with the flag or check removed | never | one run each | seconds |
| SB12-C5 (step 9) replay equals the run | on the M1 Pro, one live `gate gpu -logdir` at this step's commit, then `-replay` of its record: the regraded groups' sections equal the live run's byte for byte, without heartbeat lines. Committed as a golden case (`gpu-metal-groups`) | the live run's own precondition refuses (exit 2), or another session's Metal or WebGPU test overlaps a cell (the sampler shows it): rerun | one live run | about 5 min |
| SB12-C6 (step 9) nothing else moved | the live run's verdict and record equal §15's in group counts and label (9 of 9, 10 pass, 2 skip, 0 fail, PASS); `touched.py`, vet and the pinned staticcheck on `cmd/gate` clean, with its canary red | as C5 | inside C5's run | — |

**Step 9 (Metal) read, 2026-10-29 (the M1 Pro, on AC): C2–C5 pass; C6 passes on a second run.** The record is
`docs/measurements/sb12-2026-10.md` §17.
- C5: the live run at `67fcecce` replays equal byte for byte across six sections. The golden is
  `testdata/gate/goldens/gpu-metal-groups/`.
- C6: that run is FAIL. The fit guard refused `TestGemma2ResidentParity`'s checkpoint, with 6.9 GB available. A refusal
  inside a test is not one of the registered voids, so the run is recorded as it is. A second run at the same commit,
  with about 13 GB free, is PASS with §15's counts.
- Whether that test should skip on a refused load is named in the record for its owner and Francis.

#### SB12 step 10 (CUDA) — the gpu gate's CUDA cell groups and cgofree grade a recorded run: gates, registered 2026-10-29 before the code (nobara-ci01)

The coordinator's ask of 15:20: SB12's CUDA check groups split into runner and grader against nobara's recorded CUDA run, the
way steps 8 and 9 did Metal. This step takes the CUDA groups that are one `go test` cell each, and cgofree:
- **`suite`:** `./cuda/` and `./cuda/kernels/` under `-tags cuda -short`. On a pass it prints its evidence lines and its skip
  census by name. A census of zero prints its warning.
- **`parity`:** `./cuda/` under `cuda goinfer_testhooks`. On a failure it prints its detail. When a cosine of exactly zero
  appears, it also prints the all-zero-buffer note with the card's free VRAM. The runner reads that reading after the cell
  and records it, so the replay prints the reading the run took.
- **`graphsforced`:** `TestGemma4Graphs_` with forced capture. A cell that ran no test fails. A pass prints its counts and
  skip census, and the group always adds its forced-capture note.
- **`cgofree`:** whether `cuda/cmd/serve` built with `CGO_ENABLED=0`, with the build's output, and `ldd`'s output on the
  binary when `ldd` ran.

The runner and the replay share one sequence, as step 9 made them for Metal. `gate gpu -replay` then regrades a CUDA record:
these four groups and group W. The groups it does not regrade are `cleangpu`, `seam`, `heavy`, `ptx` and `repo`; `heavy` and
`ptx` are the next steps. A stream over 1 MB is committed gzipped (CLAUDE.md), so the replay reads a stream's `.gz` when the
plain file is absent.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 10) goldens reproduce | `TestGraderGoldens` green; `gpu-metal-cells` and `gpu-metal-groups` not edited, and each still equals its live run | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 10) a grader is a function | the new grader code is in a `grade_` file and `TestGraders_arePure` covers it | never | one run | seconds |
| SB12-C4 (step 10) each group can go red | `suite`: a failed test gives FAIL with its assertion line; a crash gives FAIL with the crash head; a pass lists its skips by name, and a zero census prints the warning. `parity`: a failed test gives FAIL; a zero-cosine line prints the note with the recorded reading. `graphsforced`: a failed test gives FAIL; a cell that ran nothing gives FAIL; a pass gives the counts and the note. `cgofree`: a failed build gives FAIL with the build's lines; `ldd` naming `libcuda`, `libnvrtc` or `libcudart` gives FAIL listing the line; `ldd` not run gives PASS. Each is shown red once by hand with its check removed: graphsforced's empty-cell check, and cgofree's linkage check | never | one run each | seconds |
| SB12-C5 (step 10) replay equals the run | on nobara, one live `gate gpu -logdir DIR` at this step's commit with `GOINFER_GATE_SKIP_HEAVY=1` (heavy is a later step), then `gate gpu -replay DIR/gate_gpu_run.json`: the regraded sections (suite, parity, graphs, cgo-free, W) equal the live run's byte for byte, without heartbeat lines. Committed as the golden case `gpu-cuda-cells` | the live run's own precondition refuses (exit 2), or another session's CUDA process is on the card during a cell (a sampler logs it): rerun | one live run | about 17 min (the 2026-10-28 WebGPU leg, heavy skipped: 04:19–04:36) |
| SB12-C6 (step 10) nothing else moved | a live run of the same command at this step's base on `origin/main`, made first, has the same verdict, the same group counts, and the same state in every group as C5's run. `touched.py`, vet and the pinned staticcheck on `cmd/gate` are clean, with its canary red | as C5 | one base run | about 17 min |

**Step 10 (CUDA) read, 2026-10-29 (nobara): C2–C6 pass.** The record is `docs/measurements/sb12-2026-10.md` §19.
- **C5:** the live run at `933733f09` replays equal across five sections, byte for byte. The golden is
  `testdata/gate/goldens/gpu-cuda-cells/`.
- **C6:** the step run and the base run at `3617a46df` agree in verdict, counts, notes and every section's state.
- **Two void runs:** another session's CUDA tests overlapped one base run and one step run, and both were rerun.
- **Parity timeout:** both valid runs FAIL only on the parity cell's 10-minute timeout. Francis's decision of 16:31 moves it
  to 20 minutes in a separate PR, with a follow-up on what slowed the package.

#### SB12 step 10 (Metal) — repo, cleangpu and seam grade a recorded run, and a Metal record replays to the gate's verdict: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's ask of 15:22: the Metal gate's remaining groups, `repo`, `cleangpu` and `seam`, split the same way. With
them, every group a Metal run declares regrades from its record. So this step also records what the verdict block reads, and
a complete Metal record then replays to the gate's own verdict and exit code. That is SB12-C5 as first registered, for the
Metal backend.
- **`cleangpu`:** the backend, whether `nvidia-smi` was found, and its process and memory lines. On Metal the group skips, as
  today. The grader also covers CUDA's branch, since one function serves both.
- **`seam`:** its cell, graded with the zero-test check it has today.
- **`repo`:**
  - the citation lint's exit and output, and the derived CI step list (`scripts/ci_checks.py`'s output, or its error);
  - whether the host is Darwin;
  - each step it ran, with its result, and its output when it failed.
  - Which steps run is a pure function of the list and the host, which the runner and the grader share.
- **The verdict's inputs:** the host (`uname -s`), the dirty tree's lines, the timestamp, and the notes the runner made before
  any group. They are recorded before the verdict is graded.
- **The replay.** When a Metal record holds every declared group's inputs and the verdict's, the replay prints the groups
  from `0. clean GPU` through the verdict block and exits with the verdict's code. A record without them (steps 8 and 9)
  replays as it does today.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 10) goldens reproduce | `TestGraderGoldens` green; `gpu-metal-cells` and `gpu-metal-groups` not edited and still equal to their runs | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 10) graders are functions | the new grader code is in `grade_` files covered by `TestGraders_arePure` | never | one run | seconds |
| SB12-C4 (step 10) each group can go red | cleangpu: Metal skips; CUDA with no `nvidia-smi` skips, with no process passes, with a stray `serve` process fails listing the processes, with only others' passes. seam: a non-zero exit fails; a run that matched no test fails; a green one passes. repo: a red lint fails with its first lines; a failed derivation fails; a failed step fails with its first lines; a `runner:` step and another platform's steps skip, counted; all green passes. Each red with its check removed once, by hand (the seam zero-test check and the stray-process check at least) | never | one run each | seconds |
| SB12-C5 (step 10) replay equals the run | on the M1 Pro, one live `gate gpu -logdir` at this step's commit, then `-replay` of its record: the output from `0. clean GPU` through the verdict block equals the live run's byte for byte, without heartbeat lines, and the exit codes are equal. Committed as a golden case (`gpu-metal-full`) | the run's precondition refuses (exit 2); another session's Metal or WebGPU test overlaps a cell (the sampler); **any cell's checkpoint load is refused by the fit guard** (the coordinator's rule of 15:22: a refusal is the box's memory, not a gate result): wait for memory and rerun. The run starts only when `memory_pressure`'s free share × 16 GB − 0.5 GB is at least 8 GB (gemma-2-2b's 5.5 GB at the guard's 70% budget) | one live run | about 5 min, plus any wait for memory |
| SB12-C6 (step 10) nothing else moved | the live run's verdict and record equal §15's counts and label (9 of 9, 10 pass, 2 skip, 0 fail, PASS); `touched.py`, vet and the pinned staticcheck on `cmd/gate` clean, its canary red | as C5 | inside C5's run | — |

**Step 10 (Metal) read, 2026-10-29 (the M1 Pro, on AC): C2–C6 pass.** The record is `docs/measurements/sb12-2026-10.md` §18.
- C5: the live run at `a241c29c` (15:53–15:58 PDT) replays equal from `0. clean GPU` through the verdict block, byte for byte,
  exit 0 as the run. The golden is `testdata/gate/goldens/gpu-metal-full/`.
- C6: PASS, 9 of 9 groups, 10 pass, 2 skip, 0 fail. The first run at the same commit (15:32–15:38) is void, by a fit-guard
  refusal and an overlapping Metal test, and is recorded beside it.

#### SB12 step 11 (CPU) — the parity sweep reads GATE_RUN once, in its runner: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's list of 16:30: the census's parity row says "mostly: the sum is inline, `gateRunFilter` reads the
environment and can exit". Step 4 moved the sum into `gradeParity`, so the row is stale on that half. What is left is
`gateRunFilter`:
- it reads `GATE_RUN` from the environment;
- three runner functions (`parityCells`, `realckptRun`, `runParity`) each call it on their own;
- on a regexp that does not compile, it calls `os.Exit(2)` from inside a helper.

The change:
- `runParity` reads `GATE_RUN` once, first, and parses it with a pure `parseGateRun`.
- A bad value returns 2 from `runParity`, with the same refusal text, before the preflight or any cell.
- `parityCells` and `realckptRun` take the parsed filter as a parameter and read no environment.
- The census row is brought up to date.

The output of a sweep is unchanged, and so is the verdict of any run.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 11) goldens reproduce | `TestGraderGoldens` green, with `parity-narrowed` and `parity-replay` not edited | a golden edited: void, redo | every case | seconds |
| SB12-C4 (step 11) a bad GATE_RUN refuses without exiting the process | an in-process test sets `GATE_RUN=(unclosed` and calls `runParity`: it returns 2, prints the refusal, and runs no cell. Red first: today the call exits the test binary | never | one run | seconds |
| SB12-C4b (step 11) the helpers read no environment | with `GATE_RUN` set to a filter that selects one gate, `realckptRun` and `parityCells` given no filter build the unfiltered pattern and the base cell's empty `-run`; given the filter, they narrow as `TestGateRunFilter_narrowsTheRealckptPattern` asserts today. Red first: today they read the variable | never | one run | seconds |
| SB12-C6 (step 11) nothing else moved | `TestGraderCensus_everySubcommand` green with the parity row updated; `touched.py`, vet and the pinned staticcheck on `cmd/gate` clean, its canary red | never | one run | seconds |

#### SB12 step 12 (CPU) — the release pre-flight's rows grade a recorded run: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's list of 16:30. The census row for `release` says "the cores yes, the rows mix git with deciding". The cores
(`gpuCoverage`, `parityCoverage`, `quickCoverage`, `ciVerdict`) are in `grade_records.go`. The rows read files, globs, git,
`go test`, `gofmt`, the citation lint, `gh` and the record directories themselves, and decide around those reads.

The change:
- **One reader.** Every read a row makes goes through one function on `relEnv`, `look(kind, args...)`. The kinds are `read`,
  `glob`, `git`, `run`, the three record loaders, the parity manifest, `changelog` and the pending list.
  - Run live, `look` does the read and records its answer (the output and the error text) in order.
  - Replayed, it answers from the record.
  - A read the record does not hold is exit 2.
- **The rows are graders.** The twelve rows, `reportRelease` and the helpers they call move to `grade_release.go`, under
  `TestGraders_arePure`. Paths in the reads are relative to the repository root, so a record does not depend on where it ran.
- **The record.** `gate release -logdir DIR` writes `gate_release_run.json` through `writeRunRecord`, with the tag, the
  previous tag, `-offline` and the reads. `gate release -replay FILE` prints the report again and exits with its code.
  `relRefusal` and `newestVersionTag` stay in the runner: a refused run (exit 2) writes no record.

The report's text, each row's state and the exit code are unchanged.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 12) goldens reproduce | `TestGraderGoldens` green with no existing golden edited; new case `release-replay` is C5's | a golden edited: void, redo | every case | seconds |
| SB12-C3 (step 12) graders are functions | the rows and their helpers are in `grade_release.go`, covered by `TestGraders_arePure` | never | one run | seconds |
| SB12-C4 (step 12) each row can go red from a record | the existing row tests (`release_test.go`, `release_pending_test.go`) pass through the live reader on their temp trees; a replay plant per row family turns that row OWED or UNKNOWN with its detail: pins that differ, a dirty `gofmt`, a red lint, a red CI run, a pending gate older than the previous release. A read the record lacks is exit 2. Red once by hand with the pins check removed | never | one run each | seconds |
| SB12-C5 (step 12) replay equals the run | a live `gate release -offline -logdir` in a local clone on branch `main` (the precondition needs `main` and a clean tree, and the main checkout is not to be touched), then `-replay` of its record: byte-identical output, equal exit codes. Committed as `testdata/gate/goldens/release-replay/` | the run is refused (exit 2): fix the clone and rerun | one live run | about 4 min (the citation lint is about 2 of it) |
| SB12-C6 (step 12) nothing else moved | in the same clone at the same tree, `gate release -offline` built from `origin/main` before the change and from this step print byte-identical reports with equal exit codes; `touched.py`, vet and the pinned staticcheck on `cmd/gate` clean, its canary red | as C5 | one run per build | about 8 min |

#### SB12 step 13 (CPU) — comments-only's overall step grades a recorded run: gates, registered 2026-10-29 before the code (mac-sb11m)

The coordinator's list of 16:30. The census row for `comments-only` says "`compareComments` per file, the sum inline in
`runCommentsOnly`". The overall step classifies each changed path, sums the per-file verdicts, writes the moved-text report and
the verdict, then runs `gofmt`, `go vet` and the citation lint and decides on them, all inline. The change:
- **The runner** keeps git, `compareComments` (it parses Go, and `go/parser` can read files, so it stays off the grader
  allowlist) and the commands. It records the changed paths, each modified `.go` file's verdict (its reds, removed runs and
  comment-line counts, or why it could not be read), the docs text the diff adds, the flags, and each follow-on check's result.
- **The grader** (`grade_comments_only.go`): the path classification, the sum, `movedTextReport` and its writer, the verdict,
  and the checks' lines and verdict. The runner asks it whether the comments check is green before running the checks.
- **The record.** `-logdir DIR` writes the run record; `-replay FILE` prints the recorded provenance and then the grader's
  report, with its exit code.

The report's text and the exit code are unchanged.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 13) goldens reproduce | `TestGraderGoldens` green with no existing golden edited; new case `comments-only-replay` is C5's | a golden edited: void, redo | every case | seconds |
| SB12-C3 (step 13) graders are functions | the new grader code is in `grade_comments_only.go`, covered by `TestGraders_arePure` | never | one run | seconds |
| SB12-C4 (step 13) each verdict can be reached from a record | plants on a recorded input: a non-docs, non-Go path is red; an added `.go` file is red; a file verdict with a red is red and runs no check; a green input with a failed `gofmt`, a failed vet and a failed lint each say so and exit 1; `-no-checks` stops at the comments verdict. Red once by hand with the non-docs-path check removed | never | one run each | seconds |
| SB12-C5 (step 13) replay equals the run | a live `gate comments-only 8ce9e115~1 -logdir` in a detached worktree at `8ce9e115` (a two-line comment change in `cmd/gate/gpu_test.go`), then `-replay` of its record: byte-identical, equal exit codes. Committed as `testdata/gate/goldens/comments-only-replay/` | the run refuses (exit 2): fix and rerun | one live run | about 3 min (the citation lint is most of it) |
| SB12-C6 (step 13) nothing else moved | in the same worktree, the reports of a build from before the change and of this step's are byte-identical apart from the provenance's date line, with equal exit codes; `touched.py`, vet and the pinned staticcheck on `cmd/gate` clean, its canary red | as C5 | one run per build | about 6 min |

#### SB12 step 11 (CUDA) — the gpu gate's ptx group grades a recorded run: gates, registered 2026-10-29 before the code (nobara-ci01)

The coordinator's ask of 18:50: CUDA's `heavy` and `ptx`, SB12's last census rows. This step takes `ptx`. How `heavy` gets
its live evidence is put to Francis (a full heavy tier is about 3.25 h); it is the next step.
- **The runner** finds the toolchains and rebuilds each PTX file with `cuda/build_ptx.sh` at the NVRTC version the file
  records. It restores the tree afterwards, as now, and records each file's outcome: rebuilt identical, differs, or not
  verified with its reason. It also records why the script could not run when it cannot, and the toolchains found.
- **The grader** turns those into the section's lines and verdicts, as the runner prints them today: the toolchain line,
  each `DIFFERS` line, the pass or fail verdict, and the counted skip naming the files not verified. The runner keeps the
  rebuilt files for `gate ptx -keep`, unchanged.
- **The replay** regrades `ptx` from a CUDA record that holds its input. `heavy` stays live-only.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 11) goldens reproduce | `TestGraderGoldens` green; no existing golden edited, and `gpu-cuda-cells` still equals its live run | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 11) a grader is a function | the new grader code is in a `grade_` file and `TestGraders_arePure` covers it | never | one run | seconds |
| SB12-C4 (step 11) the group can go red | from a green recorded input: one file that differs gives FAIL naming it; none verified gives FAIL; some not verified gives a counted SKIP naming them beside the verdict; a script that cannot run gives SKIP with its reason. Each shown red once by hand with its check removed: the differs check and the none-verified check | never | one run each | seconds |
| SB12-C5 (step 11) replay equals the run | on nobara, one live `gate gpu -logdir DIR` at this step's commit with `GOINFER_GATE_SKIP_HEAVY=1`, then `-replay`: the regraded sections (step 10's five, and now ptx) equal the live run's byte for byte, without heartbeat lines. Committed as the golden case `gpu-cuda-ptx` | the live run's own precondition refuses (exit 2), or another session's CUDA process is on the card during a cell (the sampler logs it): rerun | one live run | about 17 min (step 10's runs) |
| SB12-C6 (step 11) nothing else moved | a live run of the same command at this step's base, made first, has the same verdict, group counts, notes and state in every section as C5's run. `touched.py`, vet and the pinned staticcheck on `cmd/gate` are clean, with its canary red | as C5 | one base run | about 17 min |

#### SB12 step 12 (CUDA) — the gpu gate's heavy tier grades a recorded run: tier-0 gates registered 2026-10-29, C5 and C6 left to Francis's call (nobara-ci01)

SB12's last census row. The coordinator asked at 18:48 for the heavy grader work to go on while Francis decides how its
live evidence is gathered. So these tier-0 gates are written after a first draft of the code and before any run of it.
- **The runner** records what the tier read before any process (`heavyInput`): the skip, the two markers' tests, the
  listings, the partition and the tests that fit no group. It also records each main-group process's memory and the tier's
  wall time, and adds the heavy processes' cells to the run record.
- **The graders** (`cmd/gate/grade_gpu_heavy.go`) decide the partition block, each process's memory line, the split's
  reconciliation, the drain group, the fresh-process tests and the tier's verdict with its skip census.
- **One sequence** (`heavyTier`) prints them live and in the replay. The replay re-reads each streamed cell, printing its
  results as they arrived.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| SB12-C2 (step 12) goldens reproduce | `TestGraderGoldens` green; no existing golden edited | a golden edited with the grader: void, redo | every case | seconds |
| SB12-C3 (step 12) a grader is a function | `grade_gpu_heavy.go` is covered by `TestGraders_arePure` | never | one run | seconds |
| SB12-C4 (step 12) each part can go red | plants: a skipped tier gives SKIP; an empty drain derivation, a failed listing, partition or kernels listing, and tests that fit no group each give FAIL; a process that ran nothing gives the all-skip note; a split with a missing or doubled test gives FAIL; a drain or fresh-process count that does not reconcile gives FAIL; a failed drain or fresh-process test gives FAIL with its detail; a failed main group gives FAIL; skips are named in the census. A tier-0 replay of a fake heavy run equals the run, streamed results included. Shown red once by hand: the split's reconciliation check, and the drain reconciliation | never | one run each | seconds |
| SB12-C5, C6 (step 12) | as step 11, on a live run that includes the heavy tier: the route (a narrowed `-heavy-group` run by day, or two full runs on the night queue past the 3 h cap) is Francis's call, put to him at 18:47 | — | — | — |

### SB14 — documentation that cannot drift

§4.1's four homes, held by one test: each invariant named in a `doc.go` points at a test that exists, and the
architecture doc's import table is the import gate's data. It is `CLAUDE.md`'s rule that a doc comment claiming coverage
is not coverage, made mechanical. S.

#### SB14's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (21:42, relayed): SB14, then PW4, overnight, tier 0. The test lives beside SB13's, in `internal/layout`, and
reads every package doc comment in the tree, `doc.go` or not.
- **Every test a package doc names must exist.** A name with a package before it ("decoder's TestDecisionCensus") must
  be in that package; a bare name must be in the doc's own package.
- **Every line of an "Invariants" block must name a test.**
- **`docs/ARCHITECTURE.md` gains a generated appendix between markers,** built from `internal/layout/imports.json`: one
  row per importing package with its imports and its test-only imports, then the protected rules. The test fails when
  the appendix differs from the data; `-update` regenerates it.

Measured before the code: 8 package docs name tests today, and every named test exists.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| D1 today's tree | the doc and table tests pass on main's tree once the appendix is generated | never | one run | seconds |
| D2 drift in a doc | `gate mutation` renaming a test that a `doc.go` names turns it red, naming the doc and the test; one adding an invariant line that names no test turns it red; one naming a test in the wrong package ("decoder's TestSolve_pure") turns it red | never | one run each | seconds |
| D3 the table is the data | `gate mutation` adding an arrow to `imports.json` without regenerating turns the table test red; one editing a row of the appendix by hand turns it red | never | one run each | seconds |

### SB13 — the package layout, as a test

§9's table as a test that reads `go list -deps` for every module and fails on a forbidden edge, with ratchets (backend
test files importing `decoder` may only fall). S.

#### SB13's gates, registered 2026-10-26 before the code (mac-vk01)

Francis (19:52, relayed): SB13 while VK5's shadow week runs. The data is `internal/layout/imports.json`, which SB14
and `docs/ARCHITECTURE.md` can read. It holds two things:
- **The arrows.** Every import from one goinfer package to another, in every module, as repository-relative package
  paths. Non-test and test files are kept as two lists.
- **The protected separations, as rules:**
  - `internal/vision` and `internal/audio` import no goinfer package and, outside the standard library, only aikit's
    `linalg` and `embed`.
  - The two gpudev halves (`cuda/internal/gpudev`, `metal/internal/gpudev`) import no goinfer package.
  - `cuda/kernels` and `metal/kernels` never import `decoder`, in tests either.
  - `internal/mem/probe` imports no goinfer package; that it is the only OS memory reader stays MM0's
    `TestOnlyProbeReadsOSMemory`.
  - No root-module package imports the `cuda`, `metal` or `gpu` modules or aikit's GPU modules, and the root `go.mod`
    requires none of them.

`TestLayout_arrows` parses imports the way the existing decoder ratchet in the same package does (no `go list`, so
seconds, across all modules). It fails on:
- an arrow the data lacks: a new arrow, named with the file that makes it;
- an arrow the data has but the tree no longer makes: the data must be lowered to match;
- any arrow, made or recorded, that breaks a rule.

So an arrow, once removed from the data, comes back only through an edit to the data, and an arrow that breaks a rule
cannot be put back at all. nobara-sb01's `decoderRatchet` in the same package is not touched.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| L1 today's tree | `TestLayout_arrows` passes on origin/main's tree with the data written from it; every protected rule holds today | never | one run | seconds |
| L2 a planted forbidden import | `gate mutation` adding a `decoder` import to a `cuda/kernels` test file turns it red and the message names `cuda/kernels → decoder` and the rule; one adding an unrecorded but permitted arrow turns it red and names the arrow | never | one run each | seconds |
| L3 the ratchet only falls | `gate mutation` deleting a recorded arrow from the data while the tree still makes it turns it red (a removed arrow cannot be re-added without editing the data); one recording an arrow the tree does not make turns it red; one recording `internal/vision → decoder` in the data turns it red as a rule violation even with no code making it | never | one run each | seconds |


#### SB13: `internal/audio` may import `internal/vision`. Gates registered 2026-10-29 before the change (mac-ax01; Francis 18:20)

Francis chose (a) at 18:20 on 10-29, relayed by the coordinator: `towers-standalone` is amended so that
`internal/audio` may import `internal/vision`, and no other goinfer package. The reason is SB9 step M shape (4): the
audio towers walk `internal/vision`'s tower program type, which decision 1(a) of 10-28 put there. `internal/vision` still
imports no goinfer package, and the external allow-list is unchanged.

- **The data:** a rule gains `allow_goinfer`, a map from one of its `from` packages to the goinfer directories it may
  import despite `forbid_goinfer`. Matches are exact, so a subdirectory of an allowed one is still forbidden.
  `towers-standalone` gets `{"internal/audio": ["internal/vision"]}`. The rule check moves out of `TestLayout_arrows` into
  `layoutRule.breaks`, unchanged otherwise, so a test can call it.
- **The appendix** in `docs/ARCHITECTURE.md` renders the exception, regenerated with `-update`.

| gate | rule (pass) | void when | stopping rule | cost basis |
|---|---|---|---|---|
| L4 the exception, red first | `TestLayout_towersStandalone` reads `towers-standalone` from `imports.json` and calls `breaks`. `internal/audio` → `internal/vision` is allowed, from a test file too. Each of these is still forbidden: `internal/audio` → `internal/vision/x`, → `decoder`, → `internal/mem`, → the root package, → `multimodal`; `internal/vision` → `internal/audio`, → `decoder`; an external import outside aikit's `linalg` and `embed`. The test is committed before the data change and is red on the allowed case alone. | never | one run | seconds |
| L5 the scan still goes red | with `breaks`' `allow_goinfer` check planted to match by prefix ("internal" allows every `internal/...`), and then with the exception listed under `internal/vision` instead, `TestLayout_towersStandalone` is red each time | never | one run each | seconds |
| L6 the tree | `TestLayout_arrows` and `TestArchitecture_importAppendix` pass; the arrows are unchanged, since no code imports across yet | never | one run | seconds |
| L7 builds | `scripts/touched.py` is green | never | one run | seconds |

**Read 2026-10-29 18:20 PDT (mac-ax01; MacBook M1 Pro): pass on every gate.** Logs are in
`docs/measurements/sb13-audio-vision-2026-10-29/`.
- **L4:** with the data unchanged, `TestLayout_towersStandalone` fails on the allowed case alone (`l4-red.log`). With
  `allow_goinfer` in the data, it passes (`l4-l6-green.log`).
- **L5:** each plant turns it red (`l5-plants.log`):
  - matching by prefix lets `internal/audio` import `internal/vision/x` and `internal/mem`;
  - moving the exception's key to `internal/vision` forbids the allowed import again;
  - reversing the exception (`internal/vision` may import `internal/audio`) is red both ways.
- **L6:** `TestLayout_arrows`, `TestArchitecture_importAppendix` (the appendix regenerated) and
  `TestPackageDocs_namedTestsExist` pass, and the arrows are unchanged.
- **L7:** `scripts/touched.py` is green (`l7-touched.log`).

## 9. The target layout

| package | owns | may import |
|---|---|---|
| `decoder` (public) | the facade: `Load`, `Model`, `Options`, generation; the per-family adapter files (VK2) | everything in this table except the backend modules |
| `internal/program` (out of `decoder` when SB1 needs it) | the operation list and the CPU reference | aikit `linalg` |
| `internal/backendapi` (SB1) | the backend interface | `internal/program`, `internal/mem` |
| `internal/mem`, `internal/mem/probe` (MM) | planning, the ledger, the arbiter; the probes | the standard library (the probes also `golang.org/x/sys`) |
| `internal/kvmgr` | KV blocks and the prefix index | the standard library |
| `internal/gpudev` (AX) | the device substrate | `golang.org/x/sys`, purego |
| `internal/vision`, `internal/audio` (AX) | the towers | aikit `linalg` and `embed` |
| `multimodal` (public) | the goinfer side of each tower | `internal/vision`, `internal/audio` |
| the `cuda`, `metal` and `gpu` modules | backends | `internal/backendapi`, `internal/program`, `internal/mem`, `internal/gpudev`, aikit; **never** `decoder` once SB1 lands |
| `internal/scheduler` (SB6) | the scheduling policy | the standard library |
| `internal/serveapp`, `internal/chatapp`, `cmd/*` | the shell | `decoder` through an `Engine` interface (SB7) |

## 10. Performance: counters per change, timed runs once a day

- Most performance questions about a change are questions about counts: bytes moved, launches, syncs and allocations
  per token (SB11). They are answered at tier 0.
- A kernel's speed is an in-process interleaved A/B (TE5(b): every resolved kernel-local pair agreed in sign with its
  served A/B, 37 of 37 same-session and 45 of 45 overall,
  `docs/measurements/test-efficiency-2026-09/te5b-concordance-2026-09-28.md`). It is carried to end to end with the
  counters' category shares; those projections held within 1.5% (`docs/completed/task-prefill-gap.md` §6.1).
- Served and peer gates become the daily calibration on a quiet box: a fixed cell set, the existing rules (A B B A, the
  void rule, two arms for a new-over-old ratio), a trend over days, and a step in the trend bisected with the counters
  first.
- A change whose speed cannot be judged without a served gate is the modularity bug of `CLAUDE.md`'s "Day loop".

## 11. The daily run

- **What runs:** VK6's canary and coverage audit; the performance calibration cells; MM's plan against measured on
  each box; the product smoke (serve, the SDK tests, a tool-call turn, the web UI's fetch); tower identity spot checks.
- **Where and when:** both boxes at night, on the night queue (SD7, decided 2026-10-26), within its limits (no job over
  3 hours, none started that would end after 06:30).
- **Nobody waits on it.** It writes a record and a short summary; a failure opens a queue entry; no session holds a turn
  on it.

## 12. Track PW — the waiting

- **PW1 (SD4, decided and applied 2026-10-26).** Applied: branch protection on `main` requires 19 checks (`changes`,
  `lint`, the five decoder shards, `test-rest`, `readme-smoke`, `gpu`, `sdktest`, `cuda`, the four `root-*` jobs, the
  arm64 forced-fallback job, `gpu-darwin`, `metal-darwin`), not strict, with admins not enforced so the owner's account
  can still push docs-only changes; "Allow auto-merge" is on. The three matrix forced-fallback jobs are left out
  because a docs-only run reports them under the unexpanded name `forced fallbacks (${{ matrix.force }})`, which would
  leave a required check waiting forever. Left to do: the rule for sessions and the guard's retirement. The design as
  scoped: turn on auto-merge for goinfer (available for public repositories on GitHub Free) with a branch rule
  or ruleset on `main` that requires the fast checks (lint and the Linux test jobs), and let the owner's account push
  docs-only changes directly. A session runs `gh pr merge --auto` and moves on. Main's own CI still runs once per merge
  and a push to main does not cancel the run before it (`.github/workflows/ci.yml`'s concurrency rule), so a red run
  points at one PR, which is the purpose of `scripts/main_ci_guard.py`, served without anyone waiting. Retire the
  guard's wait when PW1 is on.
- **PW2.** Run `root-darwin`, `root-windows`, `root-windows-arm64`, `root-linux-arm64`, `gpu-darwin` and
  `metal-darwin` after merge, or only when their paths change, and keep them out of the required checks. The
  40–45-minute runs were these runners queueing (§1.4). They still gate tags through `gate release`.
  **Decided 2026-10-27 (Francis, 13:19): the OS jobs stay required; revisit if one job becomes the long pole by several
  minutes.** After PW2a, the five PR runs' last jobs finished 11.5–17.1 minutes in (median 14.6), with the last job
  varying between `root-windows`, `test-rest`, `test-decoder-d` and `gpu-darwin` according to which runners queued,
  so taking the Windows jobs out would save 0–2 minutes (`docs/measurements/ci-tail-2026-10-27.md` §4a).
- **PW2a (2026-10-27, Francis 11:14).** `test-decoder-tail` finished 19–22 minutes into a PR run, against 11–12 for the
  Windows jobs: MM1's two census tests (`TestDecisionCensus`, `TestDecisionCensus_secondProcess`) landed in it and take
  658 s of its 1,130 s under -race (`docs/measurements/ci-tail-2026-10-27.md`). They move to their own steps in
  `test-decoder-a` and `test-decoder-d`, whose sampling tests take one to three minutes. Gates in §12.2.
- **PW3.** Shard or generate the hot files. The citation index records each cited commit's subject in one file, and
  every new citation edits it: keep the check, and store the expected subjects per citing doc. CHANGELOG fragments per
  change, assembled at release. Parity records per family (VK3). The `docs/README.md` counts written by
  `scripts/task_docs.py`, which already recounts them (on 2026-10-26 it found the design-record count one short).
  **PW3, measured and gated (registered 2026-10-27 11:11 PDT by mac-ax01, before PW3's code; Francis 11:08: "next for you is PW3"):**

  | file | commits, last 30 days | last 7 days | what drives them now |
  |---|---:|---:|---|
  | `docs/citation-index.md` | 240 | 1 | since CC0 closed the door (2026-10-09) it holds only `docs/QUEUE.md`'s commit and bare-file citations: 2 commits since |
  | `CHANGELOG.md` | 280 | 12 | an entry per user-visible change |
  | `docs/README.md` | 90 | 9 | a doc's entry and the two counts `task_docs.py` rewrites |
  | `testdata/parity_manifest.json` | — | — | gone: VK3 split it into `testdata/parity/` |

  - **`docs/citation-index.md`: no change.** It stopped being hot when the path index went (CC0); sharding it would add
    machinery for two commits a fortnight.
  - **`CHANGELOG.md`: a decision for Francis.** Fragments per change, assembled at release, change how the file reads on
    `main` between releases.
  - **`docs/README.md`'s counts: removed from the file.** Two PRs that each add a task doc both rewrite the count to N+1.
    Git merges that without a conflict, because the edits are identical, and the result is one short. An add beside an
    archive conflicts. The two headings drop their numbers, and `task_docs.py check` reports the counts from the tree
    instead of comparing them with the README.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | PW3-R1 (parallel adds merge) | In a scratch clone, two branches off the same base each add a design record the way a session does (the doc, then `task_docs.py`'s recount where it applies). Merged one after the other: at the base, the README's count comes out wrong (or the merge conflicts). After the change, both merges are clean and `task_docs.py check` is clean. | — | once each side | seconds |
  | PW3-R2 (nothing else moves) | `task_docs.py check` exits 0 and prints the counts it used to compare. The README's only diff is the two headings' numbers. `archive` still runs. The citation and book-link lints pass. | — | complete | seconds |

  **PW3 results (2026-10-27, mac-ax01):**
  - **PW3-R1 PASS.** In a scratch clone, two branches off one base each added a design record and recounted the way a
    session did. At the base both merges were clean and the README said 71 design records against the tree's 72. After the
    change both merges are clean and `task_docs.py check` counts 72 from the tree; no stored count is left to go wrong.
  - **PW3-R2 PASS.** `docs/README.md` changes only in the two headings. `check` exits 0 and prints 70 (61 + 9) and 140, the
    figures the README carried. `--strict` exits 1 at origin/main too, for other sessions' uncommitted docs. The citation,
    book-link and touched-scan checks pass.
  - **Left outside this repo's code:** the doc-review skill (`.claude/skills/doc-review/SKILL.md`) still tells an
    archiving session to recount the README. That file is under `.claude`, which this session does not write.
  - **`CHANGELOG.md` waits for Francis** (DECISION NEEDED in mac-ax01's report).

  **PW3, the CHANGELOG: fragments (Francis, 12:14: option 1; gates registered 2026-10-27 12:16 PDT by mac-ax01, before the code).**
  - **Design.** Each change adds one file to `changelog.d/`, named `YYYY-MM-DD-<slug>.md`, holding its entry exactly as it
    would have appeared in `[Unreleased]`. `CHANGELOG.md`'s `[Unreleased]` becomes one line pointing there.
    - `scripts/changelog.py render` prints the entries, newest first (descending by file name), joined verbatim.
    - `scripts/changelog.py release vX.Y.Z --date D` writes `## [vX.Y.Z] — D` with that text under `[Unreleased]` and
      deletes the fragments: today's manual `[Unreleased]` → `[vX.Y.Z]` move, done by the script.
    - `gate changelog` (and so `gate release`'s CHANGELOG row) reads `[Unreleased]` as the pointer line plus every fragment.
    - RELEASING.md's step says to run the script. Today's 33 entries move into fragments whose names keep their order.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | PW3-C1 (today's text, byte for byte) | `scripts/changelog.py render` on the migrated fragments equals origin/main's `[Unreleased]` body byte for byte | — | complete | seconds |
  | PW3-C2 (a release reads as today) | On a scratch copy, `release v0.24.0 --date 2026-10-27` gives a `[v0.24.0]` section whose body is origin/main's `[Unreleased]` body, byte for byte, under the header today's move writes. It leaves no fragment behind. `gate changelog -section v0.24.0` reports what origin/main's `gate changelog` reports for `[Unreleased]`, line for line after the section name. | — | complete | seconds |
  | PW3-C3 (parallel adds merge) | Two branches off one base each add a fragment. Both merge cleanly, and `render` and `gate changelog` see both entries. At origin/main the same two adds, made in `CHANGELOG.md`, conflict. | — | once each side | seconds |
  | PW3-C4 (nothing else moves) | `gate changelog` with no flags reads the same at this branch as at origin/main. `cmd/gate`'s changelog and release tests pass, as do the script's own tests (`scripts/test_changelog.py`), `touched.py`, the citation lint and `test_touched.py`. | — | complete | minutes |

  **PW3 CHANGELOG results (2026-10-27, mac-ax01):**
  - **PW3-C1 PASS:** `render` on the 33 migrated fragments equals origin/main's `[Unreleased]` body, 17,135 bytes.
  - **PW3-C2 PASS:** in a scratch clone, `release v0.24.0 --date 2026-10-27` writes a `[v0.24.0]` section whose body is
    that same text byte for byte, under `## [v0.24.0] — 2026-10-27`. It leaves no fragment, and `gate changelog
    -section v0.24.0` reads as origin/main's `gate changelog` did (the same eight owed items).
  - **PW3-C3 PASS:** two parallel fragment adds merge cleanly and `render` lists both; the same two adds made in
    `CHANGELOG.md` conflict.
  - **PW3-C4 PASS:**
    - `gate changelog` reads identically to origin/main. With `changelog.d/` moved aside it would owe one more item
      (`POST /admin/models/fit`), so it does read the fragments.
    - `touched.py` is green, `cmd/gate`'s 79 s run included.
    - `scripts/test_changelog.py` passes 6 tests and runs in CI. The citation lint and `test_touched.py` pass.

  **PW3 review of #330 (Gemini; coordinator 12:49; registered 2026-10-27 12:50 PDT by mac-ax01, before the code).** Four
  findings, all real.
  - (1) A `changelog.d/` change voids a quick-full record: `quickRecordVoidedBy` treats every path with a `/` as voiding.
  - (2) An entry under `CHANGELOG.md`'s `[Unreleased]` passes `gate changelog` and is stranded at release.
  - (3) Only a synthetic repo exercises `render`'s name and format rules; the live `changelog.d/` never meets them before a
    release.
  - (4) `render` accepts a file holding two entries.

  Gate PW3-R330, one rule per finding, each shown red first and green after, with a test or a planted input:
  - (1) `quickRecordVoidedBy("changelog.d/x.md")` is false, and `decoder/x.go` still voids.
  - (2) `gate changelog` (and so `gate release`'s row) owes an entry left under `[Unreleased]`, and `scripts/changelog.py
    release` refuses one.
  - (3) CI runs `render` on the live tree.
  - (4) `render` refuses a two-entry file.

  The checks: `cmd/gate`'s tests, `scripts/test_changelog.py`, `touched.py` and the lints. Cost: seconds. Stopping rule: complete.

  **PW3-R330 results (2026-10-27, mac-ax01): all four PASS, each shown red first.**
  - (1) `quickRecordVoidedBy` read `changelog.d/…` as voiding; now inert, and `TestQuickRecord_whatVoidsARun` has the case. RELEASING.md
    names the path.
  - (2) A stray entry under `[Unreleased]` now owes in `gate changelog`. A plant in the real CHANGELOG read `OWED entry - **a
    planted stray entry**`, and `TestUnreleasedStray` holds the rule. `release` refuses one too
    (`test_release_refuses_an_entry_left_under_unreleased`).
  - (3) CI's step now runs `render` on the live tree.
  - (4) `render` refuses a two-entry file (`test_a_fragment_with_two_entries_is_refused`).

  `gate changelog` on the clean tree reads exactly as before. `touched.py` is green (`cmd/gate` 81 s), and the lints and
  `test_touched.py` pass.

- **PW4.** `night.py wait <job>`: block until a queued job ends and exit with its status. A session on the other box
  then runs one background `ssh` and gets one completion notification, in place of sleeping and polling over ssh.

  **PW4's gates, registered 2026-10-26 before the code (mac-vk01; Francis 21:42, relayed).** `night.py wait NAME`
  [`--timeout MIN`] follows one queued instance of the job: the one it finds queued or running, by its `seq`, through
  its re-queues (a retry or a stop) until its final record. It then prints one line with the job's outcome and log, and
  exits:
  - 0 when the outcome is `ok`;
  - 1 for any other final outcome (FAILED, timeout, VOID-refused);
  - 2 when no such job is queued, running or finished;
  - 124 when its own `--timeout` passes first.

  A job already finished when `wait` starts returns its newest record at once. Nothing else in `night.py` changes, and
  the tests run in a temporary `NIGHT_DIR`, never either box's live queue.

  | gate | rule | void when | stopping rule | cost basis |
  |---|---|---|---|---|
  | W1 backward-compatible | `scripts/test_night.py`'s existing tests pass unchanged; no other subcommand's output or file changes | never | one run | seconds |
  | W2 the semantics | new tests drive the real `Runner` in a thread under a temporary `NIGHT_DIR`: a job that ends ok → 0; one that fails → 1; one that exits 75 and is re-queued, then ends ok → `wait` waits through the retry and returns 0; one already finished → its status at once; an unknown name → 2; `--timeout` → 124 | never | one run | seconds |
  | W3 the command line | `python3 scripts/night.py wait NAME` run as a subprocess under a temporary `NIGHT_DIR` exits with the job's status while a runner in the test runs it: what a session on the other box runs as its one `ssh` | never | one run | seconds |
- **PW5 (after MM0).** `scripts/touched.py` drops `-count=1` for packages with no hidden inputs. A package whose tests
  read a probe, a device or a file outside the module is listed as such, and a source scan keeps the list true.
- **PW6.** Re-run §1's census 30 days after the first VK, MM and PW items land.

### 12.1 PW1's gates (registered 2026-10-26, before the change)

Every gate is tier 0: seconds, no model, no device, no timed run. Each runs once on the finished branch.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW1-G1 the guard's own tests | `python3 -B scripts/test_main_ci_guard.py` exits 0, all tests `ok` (the CI step of the same name) | a test is skipped | one run | stdlib, under a second (the CI step's own note) |
| PW1-G2 the guard on a docs-only range | `scripts/main_ci_guard.py 980f449b^ 980f449b` (a docs-only commit on main) exits 0 with "a docs-only push", with `MAIN_CI_GUARD_RUNS` naming a fake in-progress run, so the pass cannot come from an idle main | `ci_docs_only.sh` classifies the range as code | one run | about a second |
| PW1-G3 the guard on a product range | the same with `4f9f07aa^ 4f9f07aa` (a Go change on main) exits 1 and names the fake run | `ci_docs_only.sh` classifies the range as docs-only | one run | about a second |
| PW1-G4 the whole hook, both ranges | the hook as `install-git-hooks.sh` writes it, fed each range as a `refs/heads/main` stdin line (no push), runs the guard, the citation lint, `TestParityManifest_fresh` and the census warning, and its exit matches G2 and G3 | the citation lint or the parity test is red on the tree for a reason unrelated to the range (then G4 reports the guard's verdict only) | one run per range | the lint and the parity test, under a minute warm |
| PW1-G5 the text | `queue_citation_lint.py` exits 0 (read directly, not through a pipe) and `book_link_lint.py` exits 0 | — | one run | seconds |

If the hook itself changes after DECISION PW1-D1, G1 to G4 are re-run on the changed hook, with G3's expected exit set by the decision.

**Results, 2026-10-26, on the branch after the `CLAUDE.md` change (the hook's logic unchanged).** G1 PASS (9 tests `ok`, none
skipped). G2's first run was void by its own rule: it passed `980f449b^`, and `ci_docs_only.sh` accepts only hex SHAs, so it
failed open to "code"; the hook always passes SHAs. Re-run with both ends resolved to full SHAs: classifier `true`, guard exit 0,
"a docs-only push". G3 PASS: classifier `false`, exit 1, naming fake run 99 (against the real `gh run list` it also held,
naming #264's merge run). G4 PASS: the docs-only range exits 0 through the guard, the citation lint,
`TestParityManifest_fresh` and the census warning; the product range exits 1 at the guard. G5 PASS: both lints exit 0.

**DECISION PW1-D1, Francis, 2026-10-26 11:47 (relayed by the coordinator): remove `scripts/main_ci_guard.py` and its pre-push
step** (option (b); recommended was to turn the wait into a refusal of direct product pushes). The guard, its tests, the hook
step, its CI step and the `CLAUDE.md` sentence go. Amended gates, registered before that change, run once on the changed branch:

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW1-G1' nothing names the guard | `git grep -n -E 'main_ci_guard\|MAIN_CI_GUARD'` outside frozen records (`docs/completed/`, `docs/measurements/`) and this section's own history matches nothing that runs: no script, hook, workflow or `CLAUDE.md` line | — | one run | a second |
| PW1-G2' the hook on the docs-only range | the hook extracted from the changed `install-git-hooks.sh`, fed `980f449b`'s range as full SHAs on a `refs/heads/main` line, exits 0 | the citation lint or the parity test is red on the tree for a reason unrelated to the range | one run | under a minute |
| PW1-G3' the hook on the product range | the same for `4f9f07aa`'s range exits 0, with `MAIN_CI_GUARD_RUNS` still naming a fake in-progress run (it must now be ignored: nothing waits on main's CI) | as G2' | one run | under a minute |
| PW1-G4' the workflow and the installer parse | `ci.yml` loads as YAML with the `main CI guard` step gone and every other step of its job kept; `bash -n` on `install-git-hooks.sh` and on the extracted hook | — | one run | a second |
| PW1-G5' the text | as G5, after the deletion (a citation of the deleted file in a live doc is red and is fixed, not pinned away) | — | one run | seconds |

**Results, 2026-10-26.** G1' PASS: the only matches left are this doc's ledger, its §2.5 diagnosis, the PW1 plan text and this
section. G2' PASS: the docs-only range exits 0. G3' PASS: the product range exits 0 with the fake in-progress run named,
so nothing waits on main's CI. G4' PASS: `ci.yml` parses, and against `origin/main` the only change is the `lint` job's
`main CI guard` step, with every job kept; both `bash -n` checks pass. G5' PASS: the citation lint and the book link lint exit 0.
The `CLAUDE.md` merge step also took the coordinator's correction of 11:52: a session prints `BRANCHES READY` and the
coordinator or Francis sets auto-merge, because a background session's own `gh pr merge` is denied as a merge without
review and a session never routes around a denial.

**Amended 2026-10-26 evening (Francis, 13:04 and 21:42).** A session's own `gh pr merge --auto`, run as a bare command
while the checks are pending, passes the classifier (every PR from #274 on merged this way), so the session sets auto-merge
itself right after `gh pr create`; only a plain merge without `--auto` stays forbidden. `CLAUDE.md` says so.

### 12.2 PW2a's gates (registered 2026-10-27 11:55 PDT by nobara-ci01, before `ci.yml` changes)

The measurement that sets the target is `docs/measurements/ci-tail-2026-10-27.md`: on the slow runner class the tail's
`go test` step is 1,169 s over a 1,130 s package, of which the two census tests are 364 s and 295 s and the rest 472 s.
The change: the tail's `-skip` adds both census names; `test-decoder-a` gains a step running `^TestDecisionCensus$` on
`./decoder/` and `test-decoder-d` one running `^TestDecisionCensus_secondProcess$`, each under the same
`go test -race -timeout 25m -tags goinfer_testhooks -json` line as every other shard, each with the zero-match guard
and the summary step the other shards carry. No test's content changes, no job is added or renamed.

Purity, for the -race clause: neither census test is pure. `buildCensus` runs D1, D2 and the loads through
`parallelEach`, which starts `runtime.NumCPU()` goroutines (read in `decoder/mm1_census_build_test.go`, confirmed by
`grep -n 'go func' decoder/mm1_census*_test.go`), and the census's own mutex guards their writes. So both keep -race.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW2a-G1 every test still runs, by name | `docs/measurements/ci-tail-2026-10-27/shard_cover.py` reads every `go test` line of the five decoder shards and `test-rest` in `ci.yml` at `origin/main` and at the branch, lists the top-level tests of `./decoder` and `./internal/sampling` (`go test -list`, `-tags goinfer_testhooks`), and maps each to the steps that select it, applying `-run`, `-skip` and the package list. PASS iff, on both sides, every listed test is selected by exactly one -race step; the two sides select the same tests (counts printed, equal); the only tests whose step changed are the two census tests; and the no-race sampler-gates step selects the same tests on both sides | the two sides' test lists differ (main moved under the branch: re-run on a merged tree) | one run | `go test -list` compiles two packages, under a minute |
| PW2a-G2 -race kept | in G1's map, every test that ran under -race on `origin/main` runs under -race on the branch, the census tests included (no exception is claimed: they are not pure, above) | as G1 | one run | in G1 |
| PW2a-G3 check names | the set of job ids and of job `name:`s in `ci.yml` (loaded as YAML) is identical at `origin/main` and the branch, so the 19 required checks keep existing and branch protection needs no change | — | one run | a second |
| PW2a-G4 the census steps ran their tests | in the first PR run, each new census step's summary lists its test with an elapsed time over 60 s (a skip or a zero-match reads near 0 s), and the tail's top-level test count plus the two census steps' counts equals the count main's tail reported at the PR's merge base | a decoder test was added or removed between the merge base and the PR head (then compare against `go test -list` on the PR head) | the first PR run | read from its logs |
| PW2a-G5 the slowest required job | in the first uncontended PR run, every one of the 19 required jobs finishes under 12.0 minutes from the run's first job creation; recorded with each job's duration, the tail's and shards a and d's step times, and the runner class (`TestPrefillChunked_bitIdentical` reads about 50 s on the fast class and 85 s on the slow one). 12.0–13.0 is ambiguous: read the next run (the merge run on `main`) and record both | any required job started more than 2 minutes after the run's first job was created (runner queueing, not our jobs), or the run is a re-run attempt; then the next run is read | the first non-void run; if it is on the fast class, the next slow-class run on `main` is also read when it arrives, not triggered | read from `gh api .../jobs` |
| PW2a-G6 the scans that read `ci.yml` | `python3 scripts/touched.py` on the branch passes, which runs every source scan reading `ci.yml` (the sampler sweep-coverage check, the multimodal and kvmgr race-subset checks, the release-CI check), and `python3 -B scripts/test_touched.py` passes | the timing lock is held by a timed run (wait for it) | one run | seconds to a minute |

**Results, 2026-10-27** (`docs/measurements/ci-tail-2026-10-27.md` §3–4). G1 PASS: 1,252 tests, each in exactly one -race
step on both sides, only the two census tests moved; two plants red. G2 PASS: none lost -race. G3 PASS: 20 jobs identical,
so protection is unchanged. G4 PASS: the census steps ran 230.4 s and 228.1 s, and the tail's 1,202 plus 2 equals main's
1,204 at the merge base. G5 PASS on PR run 40560349006, slow runner class, not void: the slowest required job, `test-rest`,
finished 10.1 minutes in. G6 PASS by name only: `touched.py` ran none of the four scans that read `ci.yml`, because
`scripts/touched_scans.py` has no row for it, so they were run directly; the missing row was then added (below).

**Amended 2026-10-27, after the review of #325 (relayed 11:49).** G6's route is corrected: `scripts/touched_scans.py` lists the
four `ci.yml` readers, and `touched.py --base origin/main` runs them, each `ok`. The census guards list with
`-tags goinfer_testhooks` and report a failed listing as a build failure; their summaries skip when the test step did not
run. G1–G3 re-run on the fixed head: PASS. The review's other points and what held are in the record's §5.

### 12.3 PW5's gates (registered 2026-10-27 before the code; mac-vk01; Francis, 11:08)

**Why Go's test cache cannot be trusted everywhere.** Go replays a cached test result when the test binary, its cacheable
flags, the environment variables it read through `os.Getenv`, and the files inside its module that it opened are all
unchanged (`go help test`; `computeTestInputsID` re-checks no file outside the module root). Anything else a test
depends on is a hidden input, and a replayed pass would hide a change in it.

**The hidden inputs, each found by a source scan:**

| hidden input | how the scan finds it |
|---|---|
| OS memory | an import of `internal/mem/probe`, the only OS memory reader since MM0 |
| a device | `metal`, `cuda` or `gpu` code, or an aikit `gpu` import |
| a subprocess | `exec.Command`; its environment and its reads are untracked |
| the whole environment | `os.Environ()`, which `testlog` does not record |
| files outside the module | the home directory, the asset registry's checkpoints, absolute `/Users`, `/home`, `/proc`, `/sys`, `/tmp` or `/dev` paths, and `"../` in a nested module |
| the network | `net.Dial` or `http.Get` |
| timing | a test that reads the clock: `time.Since` or `testing.Benchmark` |

**Where a scan looks.** A package has a hidden input when one of these is found in any of:
- its own files, tests included;
- the non-test files of any goinfer package its code or tests import, transitively over `internal/layout/imports.json`'s
  arrows.

Most packages that import `decoder` are therefore not cacheable, because `Load`'s fit guard reads the probe.

**The data and how touched.py uses it.**
- The classification is data, `internal/layout/test_cache.json`: every package in the seven modules, cacheable or not,
  with its reasons.
- A `declared` section holds reasons a scan cannot see, added by hand; nothing is in it at the start.
- `TestTestCache_classified` regenerates the classification and fails on any difference (`-update` rewrites it), so a
  cacheable package that gains a hidden read goes red in the change that adds it.
- `touched.py` drops `-count=1` only for a package the data marks cacheable. An unlisted package, or a missing or
  unreadable data file, keeps it.

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| PW5-G1 complete and true | `TestTestCache_classified` passes; every directory holding a Go file in the seven modules has an entry; every non-cacheable entry names at least one reason | never | one run | seconds |
| PW5-G2 the scan bites | three `gate mutation` plants turn it red, naming the package and the reason: `exec.Command` in a test of `internal/f16`; `time.Since` in a test of `internal/giw`; an import of `decoder` in a test of `internal/towerid` ("OS memory", reached through `decoder`) | never | one run each | seconds |
| PW5-G3 touched.py honors it | `scripts/test_touched.py`: a cacheable package's `go test` has no `-count=1`; a non-cacheable one, an unlisted one, and every package when the data file is missing or malformed all keep it | never | one run | seconds |
| PW5-G4 the cache replays only what it should | in G5's second run, every package the data marks cacheable whose inputs the edit did not touch prints `(cached)`, and no non-cacheable package does | the edit touches every cacheable package | one run | from G5 |
| PW5-G5 the measurement | `touched.py`'s wall time on the diff `0158bdc0..3d01e4a3` (`decoder` 3 files, `internal/mem` 3, `metal` 1, and the scan tests they trigger), on the PW5 branch, under a private `GOCACHE` warmed once by an untimed run. Sequence: before A1, then a one-line comment edit to one `decoder` file in the diff, A2, a run with no change A3; the edit reverted; after B1, the same edit, B2, B3. Each run records its wall time, the packages that ran and replayed, and the 1-minute load average at its start. Reported as measured, with no pass bar: Francis asked for the number | a run starts with the load average above 4.0 (re-run it, and list it) | one sequence | about 6 runs × a minute or two |

**PW5 graded, 2026-10-27: G1–G3 PASS, G4 VOID in its registered run and shown by B3, G5 measured**
(`docs/measurements/pw5-2026-10.md`).
- 14 of 82 packages are cacheable (16 as first built; Gemini's two reviews of #327 widened the scan); touched.py drops
  `-count=1` for those only.
- On `0158bdc0..3d01e4a3`, a re-run with nothing changed fell from about 133 s to 124.3 s. Three cacheable tests
  replayed, saving about 9 s. Most of the time is `metal`'s device suite (about 100 s) and `decoder` (13–18 s), both
  uncached by their hidden inputs.
- G2's first plant moved to `internal/kvmgr` (`internal/f16` has no test), and timing counts only test files, as
  registered. Both are recorded there.

**Amended 2026-10-27, after Gemini's second look at #327** (`docs/measurements/pw5-2026-10.md` §5):
- Clock reads count in every file, not only in tests, failing safe; the registered "a test that reads the clock" is
  widened.
- Paths are judged on string literal values, and a climb past the module root counts in any module.

**PW5 after Gemini's third look (2026-10-27, `docs/measurements/pw5-2026-10.md` §6):**
- Climbs are counted on cleaned literals and summed per package. A declaration counts only once applied.
- The patterns are wider, and randomness is a hidden input.
- **Decided 2026-10-27: A, retire the cache** (Francis). `touched.py` passes `-count=1` for every package again; the
  scanner (`internal/layout`'s test-cache classification), `internal/layout/test_cache.json`, its `-update` and
  `touched.py`'s cache wiring are removed. Reason: about 9 s saved on a 133 s re-run, and three review rounds each found a
  new hole.

### 12.4 PW2b's gates (registered 2026-10-27 17:40 PDT by nobara-ci01, before `ci.yml` changes)

The change: each of the seven summary steps that read a `go test -json` file under `if: always()` (shards a to d, the tail,
shard c's sampling tail, `test-rest`) gets the condition the census summaries already have,
`always() && steps.<id>.outcome != 'skipped'`, and the test step that writes its file gets that `id`. Nothing else in
`ci.yml` changes. All tier 0, each run once on the finished branch.

| gate | rule (PASS iff) | void if | stopping rule | cost basis |
|---|---|---|---|---|
| PW2b-B1 only the conditions change | `ci.yml` at `origin/main` and at the branch, loaded as YAML, differ only in: each summary step's `if` (now naming its writer's `id`), and an `id` added to each writer; every summary that reads a `.json` names the step that writes it, and that step carries the `id`; every other key of every step and job is equal | — | one run | a second |
| PW2b-B2 the tests and checks are unchanged | `shard_cover.py origin/main HEAD` (no movers) prints PASS: the same 1,252 tests in the same steps, the same jobs | as PW2a-G1 | one run | under a minute |
| PW2b-B3 the scans that read `ci.yml` | `python3 scripts/touched.py` runs the four `ci.yml` readers, each `ok` | the timing lock is held by a timed run | one run | seconds |
| PW2b-B4 the normal path | in the PR's first run, every one of the seven summary steps ran (conclusion `success`, not `skipped`) and every job is green | the run is cancelled | the first completed run | read from `gh api .../jobs` |

The skipped path (a summary skipped because its test step was) rests on GitHub's documented `steps.<id>.outcome` values and
is not planted in CI: a plant costs a full PR run of shared runners. The census steps have used the same condition since #325.

**Results, 2026-10-27, on the branch** (`docs/measurements/ci-tail-2026-10-27/pw2b-gates-2026-10-27.log`). B1 PASS: 14 keys
changed, seven ids and seven conditions, each summary tied to the step that writes its file; two plants red (a test step's
`-timeout` changed; the sampling-tail summary naming shard c's other test step). B2 PASS: 1,256 tests (main had gained four
since PW2a), the same on both sides, none moved, the same 20 jobs. B3 PASS: `touched.py` ran the four `ci.yml` readers, each
`ok`. B4 PASS on #354's run 40610610391: the seven summaries and both census summaries ran, each `success`; all 22 jobs
green. Merged as `d642d060e`.

## 13. Order, sizes, and what each step is worth alone

| order | items | worth alone |
|---|---|---|
| 1 | VK0, VK1 (goinfer side), VK2, VK3 | most commits stop re-staling parity; the manifest stops being a merge point |
| 1 | PW1 (SD4), PW2, PW3 | sessions stop waiting on CI; CI stops queueing |
| 1 | AX0, AX1, AX2 | tower and device work leaves the release train |
| 2 | MM0, MM1, MM6 | night jobs stop being refused for a reason knowable at queue time; memory tests stop reading the machine |
| 2 | SB1's first step, SB13 | decoder edits stop re-running backend kernel suites |
| 3 | AX3, AX4, AX6, VK4, VK5 | aikit bumps become rare and, with unchanged numerics, free |
| 3 | MM2, MM3, SB2, SB5, SB6, SB7 | the memory account closes; admission is a table; the shell tests with no model |
| 4 | VK6, SB4, SB8–SB12, PW4, PW5 | the daily run audits the contracts |
| later | MM4 (after KV4 and KV7); MM5 and SB1's interface (after OP4 and RF1) | one budget; demand from the program; one call shape |

## 14. Owner decisions

| id | decision | recommendation |
|---|---|---|
| SD1 | move the towers and the GPU backends into goinfer; linalg stays in aikit, through `go.work`, tagged weekly | **decided 2026-10-26** |
| SD2 | the ANN and text-encoder GPU backends (`…/gpu/anncuda`, `…/gpu/annmetal`, `…/gpu/enccuda`, `…/gpu/encmetal`) | stay in aikit on the frozen `gpu/v0.35.x`; `gpupins` stops making them track the newest root tag. **Decided 2026-10-26** as recommended |
| SD3 | between weekly aikit tags, may goinfer's main require an aikit pseudo-version (a commit on aikit's main)? | yes for main; a goinfer release still requires a tagged aikit, which keeps aikit's own rule that goinfer never ships against an untagged aikit. **Decided 2026-10-26** as recommended |
| SD4 | auto-merge, and a rule on `main` requiring the checks; the pre-push wait retired | **decided 2026-10-26** (owner: "yes to SD4"); applied the same day |
| SD5 | the moved towers as internal packages, with what the examples and the demo need exposed through `multimodal` | **decided 2026-10-26:** internal, as recommended |
| SD6 | how long VK5 runs in shadow mode before it replaces today's hash | **decided 2026-10-26: one week**, with no unexplained disagreement (recommended was two weeks or 200 pushes) |
| SD7 | the daily run's boxes and hours | **decided 2026-10-26: both boxes at night** (recommended was nobara by day), within the night queue's limits |
| SD8 | guard thresholds as `mem.Policy` data with an approval stamp (I7) | **decided 2026-10-26:** yes, as recommended; it keeps "a threshold change needs Francis's word" and makes it a test |

## 15. Risks

- **A fingerprint misses a path,** and a stale validation carries over. VK6's canary bounds the delay to two weeks, and
  the coverage audit names the path.
- **Shadow-mode disagreements take time to explain.** SD6 keeps the old hash until they are explained.
- **The planner drifts from the machine.** I3 per backend and the daily calibration catch it.
- **More modules.** Rule 4: packages, not modules.
- **Public API creep.** Internal packages for everything new (SD5).
- **Mechanical moves collide with in-flight branches** at about 110 commits a day. Announce each move, land it as one
  mechanical commit per module, and keep logic changes out of it.
- **Habits outlive rules.** PW6's census checks whether the waiting stopped.

## 16. Prior art (read 2026-10-26)

- **Go's test cache** ([`go help test`](https://pkg.go.dev/cmd/go/internal/test)): "The rule for a match in the cache
  is that the run involves the same test binary and the flags on the command line come entirely from a restricted set
  of 'cacheable' test flags"; "Tests that open files within the package's module or that consult environment variables
  only match future runs in which the files and environment variables are unchanged"; "The idiomatic way to disable test
  caching explicitly is to use -count=1." **Take:** pure packages make the cache safe (PW5).
- **Bazel's test caching** ([VirtusLab's Bazel book](https://bazel.virtuslab.com/book/1~2~1/)) keys a result on the test
  binary, its runfiles, flags and environment, replays a passing result while they are unchanged, always re-runs a
  failure, and has an `external` tag for tests that depend on state outside the build graph. **Take:** declare hidden
  inputs instead of disabling the cache everywhere (PW5).
- **Meta's predictive test selection** ([Machalica et al., ICSE-SEIP 2019](https://arxiv.org/abs/1810.05286)) reports
  "over 95% of individual test failures and over 99.9% of faulty changes" while selecting fewer than a third of the tests
  build dependencies would select. **Take:** dependency-based selection over-selects badly, which is the case for
  behavior keys. **Leave:** a learned model; fingerprints and tiers are exact where it is statistical.
- **llama.cpp's `test-backend-ops`** ([its testing page](https://mintlify.wiki/ggml-org/llama.cpp/development/testing))
  "checks that different backend implementations of ggml operators produce consistent results". **Take:** the op
  conformance kit (SB1).
- **PyTorch's OpInfo** ([OpInfos FAQ](https://github.com/pytorch/pytorch/wiki/OpInfos-FAQ)): an operator's metadata and
  sample inputs; "The `ops` decorator will create a set of tests for each `device` and `OpInfo`", each compared with a
  reference. **Take:** one operation database drives every backend's tests (SB1, SB9).
- **vLLM** ([cache config](https://docs.vllm.ai/api/vllm/config/cache.html)) sizes the KV cache after a profiling run,
  inside `gpu_memory_utilization`, "the fraction of GPU memory to be used for the model executor"; its block count
  "will be set after profiling". **Take:** measure the non-KV peak, then plan, which is I3's calibration. **Leave:**
  profiling at every start; `Solve` computes and the daily run calibrates.
- **ggml's graph allocator** ([`ggml-alloc.h`, a mirror of ggml-org's](https://git.mitjafelicijan.com/r/llmnpc/blob/llama.cpp/ggml/include/ggml-alloc.h))
  reserves buffers from a worst-case measure graph and "does not allocate or modify the graph". **Take:** demand derived
  from the program (MM5).
- **llama.cpp's `--fit`** ([fit-params README, a mirror](https://huggingface.co/datasets/echodict/llama.cpp/blob/main/tools/fit-params/README.md))
  fits a model's projected memory use to the free device memory at run time (`llama_params_fit`) by changing the
  context size, the layers on the GPU and expert tensor placement, with a free-memory margin. **Take:** a decline
  that proposes the fitting change, as `Solve`'s `Decline` should.
- **MLX's memory controls** ([its `memory.h`, as vendored by go-mlx](https://forge.lthn.ai/core/go-mlx/raw/branch/main/dist/include/mlx/memory.h))
  separate a memory limit (a guideline during graph evaluation; by default 1.5 times the recommended working set on
  Metal), a cache limit, and a wired limit, "the total size in bytes of memory that will be kept resident". **Take:**
  tiers and limits as separate knobs (`Tier`, `Policy`).
- **GitHub auto-merge** ([GitHub Docs](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request))
  "is available in public repositories with GitHub Free", and merges once the required reviews and status checks pass.
  **Take:** PW1.

## 17. When this closes

Each track closes when its ledger rows are done, killed or moved. The doc closes when PW6's re-count is graded against §1.

<!-- doc-reviewed: 2026-10-26 -->
