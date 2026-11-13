# Task: the forward pass and the memory planner, what is left (2026-11)

> **Status 2026-11-12: the ledger re-read against main at 93a183f82, after ten commits that went straight to main on
> 2026-11-11 and 2026-11-12** (251184470 to 93a183f82: the Flint family, the subsystem contracts, `decoder/quant`,
> `kvmgr.SequenceState`, `program.LayerWalk`, a CI fix). None of them changed this ledger, and no row changed state by its own
> terms. Three things are in the tree without the gates their rows register: OP4m-pod2's decode half on Metal, CUDA's decode
> and prefill reading the layer walk (LW-cu, new), and flint-tiny on the memory gates (MM9, new). **Later 2026-11-12:** the Mac read
> OP4m-pod2's decode half and Metal's memory gate and they pass (rows OP4m-pod2, MM9); CUDA's and WebGPU's reads are still owed. The structures those commits
> added had no row; SUB1 and LAY1 hold them. MM5, MM5b and MM8 are corrected to what is merged. The facts are in "State after
> the direct commits" and the proposed order of work in "Plan", both near the end. One sentence in the block below is no
> longer true: the forward-shape assertions are in `decoder/resident_subsystems.go` and each backend's `subsystems.go`, not
> in `decoder/step.go` (0df3294dc).
>
> **Status 2026-11-08: open; the live remainder of two programs.** Split on 2026-11-08 by the coordinator's close-out sweep
> (Francis's yes: one combined live doc) from the forward-architecture and subsystems task docs, which are archived as the
> records at `docs/completed/task-forward-architecture-2026-10.md` (FA below) and `docs/completed/task-subsystems-2026-10.md`
> (SS below). Their designs, registrations, gates and readings stay there, with their section numbers, and each row here
> names the section it comes from. This doc holds only the open and parked work, plus MEM10 from the archived memory campaign
> (`docs/completed/task-memory-campaign-2026-10.md`). A row closes here, in the same commit as the work.
>
> **What closed in the two programs.** Forward architecture: OP0 to OP4 (OP4 on Metal; on CUDA to its d2 stop), RF0, RF1
> (#752, #757, #760, #763, #769, #770, #771: every forward-shape assertion is in `decoder/step.go`), HL on Metal (#765,
> Francis's option 1, 2026-11-08 06:50), SP0, SP2 on CUDA, OP4m's step-flag defaults (default on, Francis 2026-11-04; #748).
> FA §8's recommendations E1 to E7 are recorded as decided: E1, E2 and E7 were acted on, E4 keeps WebGPU on the RF1
> adapter, and E5's default switch is OP6-def below. Killed: OP4-W (WebGPU stays on the RF1 adapter) and OP2a-n (the whole
> V2-Lite night run; the MLA declaration rests on the slice and the tiny fixtures). Subsystems: VK0 to VK5 (VK5's builder
> switch, option B, #744/#745), MM0 to MM3, MM5a, MM6, MM7 (the gate on all four backends), AX0 to AX6, SB2 to SB14, PW1 to
> PW5, SD1 to SD8, T3-fit; MM4 is KV11 (`docs/tasks/task-kv-architecture-2026-10.md`), SB11b's remainder is X1 in
> `docs/tasks/task-turn-telemetry-2026-09.md`, AX7 and PW6 are the 2026-11-26 re-count. VK6's coverage-audit half is killed
> (it needs pieces never built: SB11a, SB1's kit).

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| RF2 | done | RF2-1 (#787), RF2-2 (#799), RF2-3a (#804), RF2-3b-1 (#811, a step heads only the rows it reads) and RF2-3b-2a (#823, the flash-decode lane's ragged twins; a step's rows split by arm per layer class) merged. G-RF2-S first-working-build read (2026-11-09, nobara): decode PASS at every depth on 1.5B, 7B and gemma3-1b (0.999-1.001), TTFT PASS (worst 0.982), MC3 PASS at 1.217 with identical replies, never-swap PASS as registered (cell medians -24.5 to +25.6 MB, no swap); verify PASS (1.003 and 1.0004). RF2-3b-2b (split-KV's ragged twins, its scores chunked inside the margin's 512-row pass) in #825; RF2-3b-3 (Forward, ForwardArgmax, ForwardSample and ForwardN as one-sequence ragged steps on the bound slot, today's calls kept beside them for the kill) in #826, identity and G-RF2-I PASS on nobara. G-RF2-S final-build read (2026-11-09, nobara, 983282891 against a2c276e3d): decode FAIL at every depth (1.5B 0.726 to 0.779, 7B 0.856 to 0.879, gemma3-1b 0.762 to 0.782), verify 1.5B:3900 AMBIGUOUS at 0.977 (7B 0.992), TTFT PASS (0.997 to 1.013), MC3 PASS at 1.223 with identical replies, never-swap not read (no host-memory samples). By the rule: the decode kill (the solo shapes back on today's calls; MC3 keeps the ragged step) and verify parked on today's ForwardN, in the kill PR (#869, merged 62649f78e). Confirmation read (2026-11-09, nobara, 62649f78e against the RF2-reverted arm): decode PASS at every depth (0.999-1.001; one swap-void cell, 7B:128, kept in the record and re-run at 1.000), never-swap PASS (peak VmSwap 0.0 MB, 36 pairs). Kill confirmed; MC3 1.22x and TTFT stand; the verify shape stays parked on today's ForwardN | FA §5.4; "RF2: registration", "RF2: addendum 1", "RF2: steps" below; `docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09/README.md`; `docs/measurements/rf2-2026-11/g-rf2-s-confirm-2026-11-09/README.md` |
| RF3 | done | RF3-1 merged (#881; the coalescer carries a newcomer's chunk in its decode run). RF3-2 PASS on nobara (2ca514def: ride passes 2, ride rows 4, replies equal alone). RF3-3 G-RF3-S read 2026-11-10 (nobara, addendum 1, 72 cells, none void, A/A pooled 1.000): PARKED, TTFT ambiguous on both models (1.5B stall 1.006, 7B stall 1.002; the registered 1.5B prediction of about 1.05 missed); every guard held (gap max 0.994 and 1.000, solo TTFT 1.004 and 0.999). Per the registered outcomes: the alternation stays the default and the reading goes to Francis (ship RF3 anyway as a lossless no-gain path, or park it and what RF4 and RF6 then rest on). **2026-11-12:** `backend/cuda/prefill.go:cudaResident.prefillCore`, the body `StepRagged` runs, changed in 90e4d44f8 after the builds G-RF2-I, G-RF3-I and G-RF3-S read; LW-cu re-read it at main on nobara the same day (gate identity 30 IDENTICAL, `TestCUDAStepRagged_rowsMatchTheirShapes` PASS). **Francis, 2026-11-12: keep it.** RF3 ships on CUDA as a lossless path with no measured gain at these cells. It has been on main since RF3-1 merged (#881), with no switch: a chunk rides wherever the resident takes ragged steps, and the alternation stays for the families that decline and for Metal and WebGPU | FA §5.4; "RF3: plan and registration" and "RF3: reading" below; `docs/measurements/rf3-2026-11/run-2026-11-10-addendum1/README.md` |
| RF4 | blocked | KV5 (RF3 shipped 2026-11-12) | FA §7a |
| RF5 | blocked | RF4; it also takes SB1's op-level interface | FA §7a; SS §8 "SB1" |
| RF6 | open | speculation inside a batch, on the ragged path, with MC4's identity gate; RF3 shipped 2026-11-12, so it can start; register it first | FA §7a |
| OP1-W | open | Gemma 2's attention-logit softcap on WebGPU: register the cells on nobara (Vulkan) before code | FA §4.6; `docs/hardware-matrix.md` |
| OP1-o1 | done | Metal `attention_fa_blk` forms its score through `attn_score` (its softcap build held to float64 per score site by `backend/metal/attn_softcap_test.go:TestAttnSoftcap_faBlk`), bit-identical to the graded text without a cap (`backend/metal/attn_fa_blk_test.go:TestAttnFABlk_routedScoreIsTheGradedKernel`, the resident's own pipeline on the 1.5B and 7B, G1-K's digests); R17's speed re-grade PASS (median routed ÷ graded attention 1.0161, bar ≤ 1.02), and the pin moved to the routed text | FA §4.6 "Owed, and not in OP1"; `docs/measurements/op1-o1-r17-regrade-2026-11-09/README.md` |
| OP1-o2 | active | the CUDA flash-decode lane with the softcap (nobara-sb01): built (`decode_fa_cap`), opt-in only (`Options.FlashDecodeSoftcap`; the exact split-KV lane stays Gemma 2's default), correctness gates pass (G2-K at both KV types, the resident path, a real Gemma 2 2B end to end). Next: the registered fidelity gate by day (G22B at K=3900), then the timed cell; a default change needs Francis | FA §4.6a "Owed, not in OP1"; `docs/measurements/op1o2-fa-softcap-2026-11/README.md` |
| OP4m-D6b5 | done | read 2026-11-09 by day under the exclusive Mac lock: PASS, median step/sequential 1.2009, 10 of 10 pairs faster, tokens equal in every pair, after the cell's recurrent refill fix; `--recurrent-step` stays off by default | FA §4.9; "OP4m-D6b5: reading" below; `docs/measurements/op4-2026-10/op4m-d6b5-speed-nemotron-h-4b.log` |
| OP4m-pod | done | Metal decode walks each mixer step (attention, DeltaNet, Mamba-2, MLA) by its operations (`backend/metal/mixer_walk.go:resident.encodeMixerWalk`): P-1 and P-2 pass on all 197 resident cells, gate identity reads 29 IDENTICAL, and P-3 passes (medians 0.9996, 1.0008, 1.0005) | "OP4m-pod: registration" below; `docs/measurements/op4m-pod-2026-11-09/README.md` |
| OP4m-pod2 | active | the FFN and own-block steps (MoE, Gemma 4's MoE block, PLE, layer scale) walked by their operations, registered as OP4m-pod was; then the batched prefill and the multi-sequence step, after which `mixerWalkBlocksForTest` goes. **2026-11-12: the decode half is in the tree without the registration or its gates (90e4d44f8).** Metal's dense, MoE, Gemma 4 MoE, PLE and layer-scale steps go through `backend/metal/ffn_walk.go:resident.encodeFFNWalk`, on by default, with the blocks kept behind `mixerWalkBlocksForTest` (which now switches the FFN steps too). The batched prefill (`backend/metal/prefill.go:resident.prefillLastImgSrc`) keeps its blocks, which now branch on the walk's fields in place of the family flags, with no second arm. The multi-sequence step (`backend/metal/batch.go`) is untouched. Tests added: `TestFFNWalk_plantsGoRed` (three plants) and `TestDecodeLayerWalk_CoversAllTopologies`. **Read on the Mac 2026-11-12 at 4baa8f78c, all PASS** (`docs/measurements/op4m-pod2-2026-11-12/README.md`): P-1 and P-2 on 202 resident cells, the plants red, `gate identity 0df3294dc` against main 30 IDENTICAL (decode and batched prefill logits; 14 not run, as before), and P-3's four cells at medians 1.0003, 1.0016, 1.0005 and 0.9991 (walk ÷ blocks). The decode half is graded; MoE and Gemma 4's MoE block are P-1 and P-2 only. Then the multi-sequence step (registered 2026-11-12, "OP4m-pod2 part 2: registration" below), after which the blocks and the switch go | "OP4m-pod: registration" below; "State after the direct commits" below; `docs/measurements/op4m-pod-2026-11-09/README.md` "Consequences" |
| LW-cu | open | CUDA's decode segments and `backend/cuda/prefill.go:cudaResident.prefillCore` read a decoded `internal/runtime/program/walk.go:LayerWalk` (checked at build by `backend/cuda/op_walk.go:cudaResident.checkCUDALayerWalk`) in place of the family flags since 90e4d44f8, with no switch back, no registration and no device run on record. Its one test, `TestCheckCUDALayerWalk_TopologiesAndRejections`, builds operation lists by hand and runs no forward. Two edits are more than a substitution: `backend/cuda/resident.go:cudaResident.captureGraphs` chooses the layers it captures differently, and the PLE and layer-scale launches left the unfused branch (effect not read). It is not OP4cu-x: CUDA's step admits what it did. **2026-11-12, nobara at main 4edbe93e9 (nobara-v1), tiers 0 and 1, no speed:** the device tests PASS with no skips (the layer-walk test, G-RF2-I's `TestCUDAStepRagged_rowsMatchTheirShapes`, G1a, G3b, KV7's slot-write table, the window replay and all `TestWindowedKV_*`, flint's resident parity, the Gemma 4 E-model PLE and layer-scale tests), and `go run ./cmd/gate identity -backend cuda` against 51ed2ae27 reads 30 families IDENTICAL, 0 DIFFERENT, 14 NOT RUN (a family or cell both builds decline). **Graph capture, read later that evening under MPS (the control daemon, Default compute mode kept):** the graphs-on tests that had skipped PASS at main and at 51ed2ae27, and a probe of 13 tiny fixtures shows `captureGraphs` capturing the same segments per layer on both builds with graphs-on equal to live bit for bit (nemotron and deepseek enable no graphs on either). Not read: a graphs-on speed cell (an addendum, registered first). Next: the decode and TTFT read, registered in `docs/measurements/lwcu-2026-11/registration.md` (main against a build at 51ed2ae27, 0df3294dc the bisect point, A B B A, KV4's G2 instrument and RF2's bands, about 45 minutes by day under the exclusive lock; both serve-cuda binaries built and pinned by sha256); RF2 is the precedent for a decode loss from a change of this kind. A failed band goes to Francis, with the revert as the do-nothing arm | "State after the direct commits" below; `docs/measurements/kv4-cuda-2026-10-21.md` §2 G2; "RF2: registration" below; `docs/measurements/cuda-read-2026-11-12/README.md` |
| OP4cu-census | done | an arm64 `TestAdmissionCensus` run passes at c2's commit (`c40e79dc6`) and on main, 2,711 lines each: the golden c2 applied per line key is what an arm64 run produces | FA §4.9a "C2-2"; `docs/measurements/op4-cuda-2026-11/op4cu-c2-c22-arm64-census-run.txt` |
| OP5-S | open | Spark-X2.5, on Metal first | FA §4.7 "OP5" |
| OP5-L | open | LFM2.5's short conv | FA §4.7; `docs/hardware-matrix.md` |
| OP5-G | open | Granite-4.0-H: the bring-up gate, plus the residual scale on CUDA and Metal | FA §4.7, §4.5 |
| OP5-La | open | Laguna, after a fit check | FA §4.7 |
| OP6-M1 | done | M1 read by day 2026-11-08 on main at 76d78d19, after #775 fixed the q4k sidecar admission: goinfer 42.4 tok/s, Ollama 41.7, median ratio 1.02, LEVEL (2026-10-20: 0.539, BEHIND); fit check RESIDENT, no swap growth | `docs/measurements/peer-vetted-2026-10-07-macbook.md` "M1 on main after #775: read"; #775, #780 |
| OP6-def | done | Francis's call, 2026-11-08 09:30, on M1's reading (LEVEL, median ratio 1.02; goinfer 42.4 tok/s, Ollama 41.7): Metal's default load for a Phi-3/4 `.gguf` is `--quant q4k`, resident on the q4k lane (`internal/runtime/modelload/modelload.go:activationSafeQuant`), and `-backend auto` admits q4k to Metal (`decoder/residency.go:autoMetalPrecision`), so the default serve takes it; an explicit `--quant` is kept, WebGPU and other families are unchanged | `docs/measurements/peer-vetted-2026-10-07-macbook.md` "M1 on main after #775: read"; FA §8 E5 |
| OP6-C | open | CUDA q4k batched prefill | FA §3.2 |
| OP6-W | open | optional: per-32 activations or a q4k lane on WebGPU | FA §7a |
| HL-C | active | registered 2026-11-08 (docs PR); code after the registration merges: HL-C1 (the gather's table, tier 0), then HL-C3 and HL-C4 on nobara by day | FA §6; `docs/measurements/hl-2026-11/registration-hlc.md`; `docs/measurements/hl-2026-11/README.md` |
| AC | done | the 13 compute, fit-guard and speculative family findings mapped to what decides each path now: 8 read the program in whole or in part (one of them the resident's own report), 5 do not, which the map lists as candidates for the owner | FA status block (owner, 2026-10-21); `docs/queue-correctness.md` G47; `docs/measurements/audit-classes-2026-10-08/compute-share-paths-2026-11-08.md` |
| F-coh | done | CUDA Cohere LayerNorm: decode runs the batched kernel (the CPU's order), so a step row is bit-identical to `Forward`; G-coh-1 red on main then green, G-coh-2 no farther from the CPU (nobara-sb01, 2026-11-08) | FA §4.5; `docs/measurements/fcoh-cohere-layernorm-2026-11/README.md` |
| MM5 | open | the memory planner's CUDA half (nobara-sb01): `cudaKVParts` priced from the declared rows and CUDA's build holding its pools to them (`kvCheckDeclared`); G-MP1 to G-MP4 and the plants pass, G-MP1 moved nothing; #739 and #741 merged 2026-11-07, and Metal prices its KV the same way. Next: the CPU and WebGPU halves, not started. The CPU's KV is priced from the architecture (`decoder/residentneed.go:Model.cpuKVBytes`) and WebGPU's from `decoder/fitplan.go:Model.kvBytesPerPositionAllLayers`, and neither build holds its buffers to declared rows | FA §4.9b; #739, #741; `docs/measurements/mm5-cuda-2026-11/README.md` |
| MM5b | open | decode scratch's program-side declaration, PR #772 (merged 2026-11-08, with #762): the operations declare their per-row activations (`program.Buffers`' per-row unit, `program.HeadBuffers`), Metal's decode scratch is priced from them (G-MS4 passes; the family-keyed price kept as the oracle, equal on 368 prices), and Metal's build holds the declared widths to its kernels'. G-MS1: only Olmo Hybrid moved (768 B of unused rows dropped), and the paged-cache boundary in a later fix. G-MS5: 29 IDENTICAL. Both Gemini reviews fixed. Next: CUDA, WebGPU and the CPU, which price no decode scratch from the program (`decoder/metal_scratch.go:Model.MetalDecodeScratchBytes` is the only reader) | FA §4.9c ("MM5b's declaration"); #762, #772 |
| MM8 | active | the memory-drift gap fixes: CUDA #721 (dense 314 rows to 10), dense-moe 18, grant 1; WebGPU's int8int8 Gemma 4 MoE dense rows on Vulkan (bound 1,064 B, read on nobara); confirm or strike the towers, host-kept embeddings and verify logits lines. Done: Metal's dense table rows and Gemma 4 MoE's 19,016 B (the plan prices the buffers Metal's build derives, `decoder/metal_dense.go:Model.metalDerivedDenseBytes`; Metal's table is empty), and WebGPU's Gemma 4 MoE router tables (`decoder/webgpu_vulkan.go:Model.webgpuDerivedTableBytes`; the 64 B int4 rows go), and the CPU's Mamba-2 window views (`internal/runtime/mamba2/mamba2.go:Mix` copies each window row, so the window holds no part of the projection's output; the CPU's gap pin 28 to 26), and the CPU's KV (`decoder/residentneed.go:Model.cpuKVBytes` prices what `NewCache` holds: a ring layer's window, twice it at f32 once a read wraps, int8's one f32 scale a KV head; the CPU's gap table is empty, pin 0; the decision census's three classes re-recorded), and the load-time host-RAM guard's KV (`decoder/arch.go:kvBytesForCtx` is the CPU cache's price, which `Model.cpuKVBytes` reads; a Metal-resident load is one slot of Metal's pools, `decoder/residentneed.go:metalKVBytesFor`; a load Metal declines is priced at the CPU cache's precision). **2026-11-12:** WebGPU's Gemma 4 MoE dense rows went on 2026-11-09 (06c735193), so that item is done. The three open PRs are neither superseded nor invalidated by the new layout: #827 and #829 (the host guard prices KV as the plan does; a `-backend cuda` load CUDA does not take) conflict with main only in generated files (the decision census and twelve parity manifests), and #721 (draft; the RoPE and router-bias tables CUDA derives, dense 314 rows to 10) also in the CUDA memory census, two test files and the archived subsystems doc. Each needs a merge of main, a parity refresh and a re-take of the census's three classes, which hold 30,143 groups each now; #721 also needs nobara's gap and row re-takes. Order: #827, then #829 on it (one census take at #829's head covers both), then #721 | SS §6.5 "MM8"; `internal/guard/memgate` gap tables; `docs/measurements/mm8-*` |
| MM9 | open | flint-tiny on the memory gates, which the Flint commits and the CI fix left uneven. **CUDA:** the admission census has flint-tiny resident on `cuda` and `backend/cuda/testdata/memgate_rows.txt` has no flint row, so `backend/cuda/sb2_census_test.go:cudaGateRowsHold` fails `TestMemGate_cuda` on a card (read from the code: CI runs no device test and no nobara run is on record). **WebGPU:** `backend/webgpu/sb2_census_test.go:webgpuMemGaps` gained a flint-tiny int4, int8-KV row and `webgpuMemGapsPin` went 27 to 28, on a table that may only shrink. **Metal:** `TestMemGate_metal` with flint-tiny PASSES on the Mac at 4baa8f78c (`docs/measurements/op4m-pod2-2026-11-12/README.md`). **2026-11-12, nobara read (nobara-v1, main 4edbe93e9):** `TestMemGate_cuda` was red as predicted and by one row more: ten rows built resident and were not in the file (the nine flint-tiny int4 rows and `nemotron-tiny int4/kv-f16`, which no Flint change explains), and ten rows carry an unforgiven "dense" gap, flint-tiny 320 B over (113,056 leased against a plan of 112,736) on all nine and nemotron kv-f16 64 B over. `-memgate.rows.update` added the ten rows (`backend/cuda/testdata/memgate_rows.txt`, done); the rerun is red on the gaps only. **The gaps are MM8's: no gap row added, no pin moved; Francis decides.** Next: the WebGPU row is fixed, or kept on Francis's word; MM8 prices or tables CUDA's flint and nemotron-kv-f16 dense vectors | SS §6.5 "MM7", "MM8"; "State after the direct commits" below; `docs/measurements/cuda-read-2026-11-12/README.md` |
| MM10 | active | the decision census golden split by host class (Francis, 2026-11-12; the goldens lane is mbp2-mem1's): `testdata/census/<arch>[-metal].txt.gz`, one sorted group-a-line text file per class (a pure re-encoding: the three classes' 30,143 group hashes and totals are equal before and after), `TestDecisionCensus` reads its own class's file and `-census.update` writes only that one; CI's root-darwin and root-linux-arm64 jobs upload the class's regenerated file as an artifact when red (never a committed rewrite). Parity manifests: not changed here, see the PR. Next: this PR's merge, then #902 and #721 re-take only the class files they move | `internal/runtime/mem/CLAUDE.md`; `decoder/mm1_census_test.go:censusGoldenPath` |
| MEM10 | done | the pinned expert copy stays the default (tier 0 and tier 1, 2026-11-08: no in-place pin on nobara; unpinned decode 0.817x / 0.750x). Francis, 13:05: ship the unpinned path as an opt-in, `--moe-experts-unpinned` (default off), which the memory refusal names when the pinned copy is what does not fit | `docs/completed/task-memory-campaign-2026-10.md` §4 "MEM10"; `docs/measurements/mem10-2026-11/README.md`; `docs/measurements/mem10-2026-11/README.md` "MEM10 tier 0"; `backend/cuda/internal/gpudev/mem10_probe_test.go:TestMEM10_fileMappingRegistration_probe`; `docs/measurements/mem10-2026-11/ab-registration.md`; `docs/measurements/mem10-2026-11/ab-registration.md` "Reading"; `docs/measurements/mem10-2026-11/README.md` "MEM10 opt-in" |
| SB2-vk | done | shipped (Francis, 2026-11-08: rebuild and ship): held-out set 6 HOLDS (worst plan 1.8 MiB from the card, two hybrids among four loads) and the fourteen-load re-run holds; the planner prices WebGPU on Vulkan as M1′ over the build's buffer list in its order | SS §8 "SB2"; `docs/measurements/sb2-vk-2026-10-31/README.md` ("Held-out set 6, read") |
| SB11a | blocked | OP4 on CUDA past d2 and OP4 on WebGPU: the walker emits SB11's operation records | SS §8 "SB11" |
| SUB1 | open | the four subsystem contracts (`decoder/resident_subsystems.go:ResidentDecodeSubsystem`, `ResidentPrefillSubsystem`, `ResidentKVSubsystem`, `ResidentRuntimeSubsystem`; 0df3294dc) stand beside the earlier `Resident*` interfaces, none of which was deleted, with hand-written forwarders in each backend's `subsystems.go` and a bridge for the test fakes. `TestResidentSubsystems_NoScatteredMicroAssertions` forbids a new assertion on the old names in `decoder/` only; it does not read type switches or other packages. Next: decide it with RF5 and SB1, whose op-level interface this overlaps: the old interfaces go behind the contracts, or the row names which stay and why; and a test that holds every resident method to a forwarder | SS §8 "SB1"; FA §7a; "State after the direct commits" below |
| LAY1 | open | desk: `docs/layout.md`'s row for `decoder/quant/` calls it a leaf over `aikit/safetensors`, and the package imports `aikit/embed`; its Tier 3 order text predates the package. `docs/ARCHITECTURE.md` describes none of the subsystem contracts, `decoder/backend_view.go:BackendModelView` or the layer walk | `internal/layout/imports.json` (rule `quant-leaf`) |
| SerWt | done | every writer entry point refuses a bundle with bf16 or f16 weights with a `*SerializeError` naming the kind, before writing a byte (`decoder/serialize.go:giwWriter.writeBundle`) | `docs/measurements/sb4-2026-10.md` §3; `TestSerializeWeights_refuses16BitWeights` (`decoder/serialize_test.go`) |
| SB4 | blocked | parked in shadow (Francis, 2026-11-08): the loader rule keeps computing its verdict without acting; the first loader-only push decides whether it switches | SS §5 "VK5"; `docs/measurements/vk5-2026-10.md` §12; `docs/measurements/sb4-2026-10.md` |
| VK6 | blocked | parked (Francis, 2026-11-08): the daily real-checkpoint canary; trigger: a carried-over validation found wrong, or an aikit bump that changes kernels | SS §5.9 |
| VK5-C | blocked | parked: VK5's option C, not started | `docs/measurements/vk5-2026-10.md` |
| OP4cu-x | blocked | parked: OP4 on CUDA past d2 (MoE, recurrent, Mamba in CUDA's step) needs Francis's word | FA §4.9a "Q4a" |
| OP2c | blocked | parked: a KDA checkpoint a box holds | FA §4.7 |
| OP2-K2 | blocked | parked: demand for Kimi K2 on Metal (its router 256 to 384) | FA §4.7 |
| OP5-L4 | blocked | parked: a Llama 4 checkpoint a box holds | FA §4.7 |
| SP1 | blocked | parked: a windowed target projected at 1.30x or more (gpt-oss read 1.07x) | FA §6a; `docs/measurements/sp0-2026-10/README.md` |
| SP2-M | blocked | parked: a Metal recurrent spec; reopens with track SP | FA §6a |
| SP2b | blocked | parked at the owner's wind-down; its registration and stop point are on `origin/sp2b-2026-10` (c3093ae0) and never reached main | FA §6a |
| SP3 | blocked | after SP2-M (optional) | FA §6a |

## RF2: registration (2026-11-08, before any RF2 code)

- **Question.** Can CUDA run decode, verify, prefill and MC3 rows, of several kinds and counts per sequence, in one `Step` pass
  whose attention reads the paged KV through the block tables (KV4), without losing identity or speed on any of today's shapes?
- **Built:** CUDA's resident `Step(batch)` and its ragged attention; RF1's `step` calls it for the shapes it takes and the
  adapter keeps the rest. A mixed batch (a prefill chunk beside other sequences' decode rows) is exercised by tests only; the
  scheduler is RF3. Each row keeps its kernel class (FA §5.1: the class is the pass's, never the batch's).
- **G-RF2-I, identity (hard; tier 0, then tier 1, nobara):**
  - per shape, against `main` at the merge base: every logit bit-identical on KV4's G1a fixture list
    (`docs/measurements/kv4-cuda-2026-10-21.md` §2 G1a), with scrambled tables, and `go run ./cmd/gate identity -backend cuda`
    IDENTICAL on its default sets;
  - batch invariance: every row of a mixed step bit-identical to the same row run in its own shape;
  - plants (each must go red, recorded, never shipped): one row's table offset off by one block; one row's kind read as another's.
- **G-RF2-F, fidelity:** none is expected, because no class changes. A shape whose class would change stops the work, and its
  gate is registered here (KV4b's G4 shape) before that code.
- **G-RF2-S, speed (tier 1, nobara, by day, exclusive timing lock):** KV4's G2 instruments, arms and void rule
  (`kv4-cuda-2026-10-21.md` §2 G2), `goinfer_old` = a `serve-cuda` from the merge base, A B B A in one session.
  - decode at depth 128, 2,048, 3,900 and 8,000, the 1.5B and the 7B, and gemma3-1b to 3,900;
  - time to first token at K = 512 and 3,900, the 1.5B and the 7B;
  - MC3's aggregate at 4 clients, the 1.5B, every turn's `content_sha` equal between arms;
  - verify: `scripts/bench_spec_lane.py`'s arm A (exact attention, `-spec ngram`) on each build, cells 1.5B:3900 and 7B:7500,
    three pairs (the harness takes the second build's binary first, as tooling).
  - **Bands,** each the median paired ratio new/old (TTFT old/new): **pass** 0.98 or better for decode, MC3 and verify, and
    0.97 or better for TTFT; **ambiguous** (parked, not shipped on this evidence) 0.97 to 0.98 (TTFT 0.96 to 0.97); **fail**
    below.
  - About 70 minutes: KV4's 35 + 6 + 6, and about 20 for verify.
- **Never-swap:** KV4's G4, from the same runs.
- **Kill (FA §5.4):** decode at depth 2,048 or more under 0.98 after two tuning rounds (each one kernel-side change and a
  re-run of only the failing cells, within one working day) keeps today's decode call beside the ragged one; the other shapes
  ship on their own bands.
- **Stopping:** each speed cell is read once on the final build, plus once on the first working build to apply the kill
  early; unchanged arms are not re-measured.
- **Outcomes:** every gate passes, RF2 ships and RF3 opens; a decode kill ships the rest with decode on today's call; an
  ambiguous or failed band on another shape parks that shape on the adapter and goes to Francis.

### RF2: addendum 1 (2026-11-08, after Gemini's review of #778; before any RF2 code or measurement)

These lines replace the registration's where they differ.
- **Sessions and pairing.** Every timed comparison runs both arms in one session on the same box, ordered A B B A with
  A = `goinfer_old` and B = this build. Each B is paired with the A next to it, and a band reads the median of the pairs. A
  baseline arm is never carried over from an earlier session (`docs/measurements/noise-registry.md`, TE3). This replaces
  "unchanged arms are not re-measured". What stands is that a cell is read once on the final build, plus once on the first
  working build for the kill.
- **Metric direction.** Every ratio reads higher-is-better, B against A:
  - decode, verify (tok/s) and MC3's aggregate (tok/s): B/A;
  - time to first token (seconds): A/B.
- **Decode at every depth, 128 included.** Decode under 0.98 at any registered depth opens the tuning budget. That budget is
  at most two rounds. Each round is one kernel-side change, then a re-run of only the failing cells, both arms A B B A in a
  new session, within one working day. A cell therefore has at most two re-runs.
  - If decode at any depth is still under 0.98 after the second round, that is the kill: today's decode call stays beside
    the ragged one, the other shapes ship on their own bands, and the reading goes to Francis.
  - Decode has no parked band of its own: 0.97 to 0.98 is under 0.98.
  - The other shapes' ambiguous bands are unchanged: a shape in its band is parked on the adapter.
- **The mixed-batch test is widened to the Question.** One step holds every kind the Question names, each at more than one
  count where the kind allows it:
  - one sequence's prefill chunk;
  - a second sequence's verify rows (k = 2 and 4);
  - one decode row each for two MC3 sequences;
  - a sequence with a decode row plus verify rows in the same step.
  Every row's logits must be bit-identical to the same row in its own shape. The Question is not narrowed.

### RF2: steps (2026-11-08)

Each step is a PR from `origin/main` that builds and passes alone. No step before RF2-3 changes the path an existing shape
takes (a decode token, a verify and MC3's step keep today's calls), so the speed gate reads once, on RF2-3's build.

| id | what | gates |
|---|---|---|
| RF2-1 | `ResidentRaggedStepper` (`decoder/residency.go`) and step's `passRagged`; CUDA's `StepRagged`: decode rows and verify rows of several sequences in one pass of prefillCore's multi-sequence mode, each row in the exact class at its own slot and position | `TestCUDAStepRagged_rowsMatchTheirShapes` on nobara (batch invariance for decode, drawn decode and verify at k = 2 and 4; both plants red); decoder's step tests; tier 0 |
| RF2-2 | a prefill chunk's rows in the same step, in the class its own prefill takes (the fast kernels over its rows, the exact ones over the others) | the full mixed batch of addendum 1, bit-identical to each row's own shape; G-RF2-I's `cmd/gate identity` |
| RF2-3 | one ragged attention launch over the block tables in place of the per-row decode gap; the solo shapes move onto `StepRagged` | G-RF2-I again; G-RF2-S and never-swap, by day on nobara; the kill |

## RF3: plan and registration (2026-11-09, docs only, before any RF3 code)

**What changes.** Today a newcomer's prefill alternates with decode (`decoder/model.go:Model.mc3Prefill`): one `-prefill-chunk`
pass (512 by default) inside `prefillExclusive`, then `tokenCoalescer.yieldToDecode`. Under RF3 the coalescer's run carries the
chunk itself: one `passRagged` step of the newcomer's chunk (first, a `ResidentRaggedPrefill` with the whole prompt's
`PromptLen`) beside every pending decode row. FA §5.1's token budget is the chunk plus the decode rows, so no new option is
added. The cuts stay mc3Prefill's (the floor raise, a tail of at least `prefillTailMinRows`). The tail, which returns the
newcomer's logits, stays a prefill pass of its own (`residentPrefillSeed`), so the worst stall is today's tail pass. Each row keeps its own shape's class (RF2-2's identity). Where the resident is not a
`ResidentRaggedStepper`, or `raggedChunkDecline` refuses the chunk (recurrent, windowed KV, MoE), the alternation stays.
Metal and WebGPU keep it until RF4.

| id | what | gate |
|---|---|---|
| RF3-1 | the coalescer folds a pending chunk into its run as a `passRagged` step, and mc3Prefill hands it the chunk instead of taking the resident; a decline falls back to the alternation | tier 0: a counting fake `ResidentRaggedStepper` driven through mc3Prefill and the coalescer (not the step alone). With 3 decoders, each of the newcomer's chunk passes holds the 3 decode rows; no exclusive prefill pass runs except on the declined path; a plant that keeps the alternation is red |
| RF3-2 | the CUDA check, in process and served | G-RF3-I |
| RF3-3 | the run script under `docs/measurements/rf3-2026-11/`, the A/A and the speed read | G-RF3-S |

**G-RF3-I (tier 1, nobara, exclusive lock).**
- A CUDA test through the coalescer: each row of a mixed run is bit-identical to the same row under today's alternation.
- The stall cell below, both arms: every reply's sha identical, the decoders' and the newcomers' (hard).

**G-RF3-S (tier 2, nobara, exclusive lock, by day).**
- *Instrument:* `scripts/bench_prefill_stall.py --serve-args=-backend cuda`. The run script runs it under `timing_lock.py run`.
  - Arms: A is the RF3 PR's merge base (the alternation), B is RF3.
  - A B B A, four pairs a cell; each B is paired with the A next to it, and a band reads the median pair.
- *Cells:* qwen2.5-coder-1.5b and qwen2.5-7b (q4_k_m, `~/models`).
  - Stall: 3 decoders, 3 newcomers, `--prompt-words 2400`.
  - Solo: `--decoders 0`.
- *A/A first, in the same session:* two pairs of A against A on the 1.5B's stall cell. nobara has no TTFT A/A
  (`docs/measurements/noise-registry.json`, the served-ttft gap). If either pair's TTFT ratio is off 1.00 by more than
  0.03, the read is void and goes to the coordinator before B runs.
- *Metrics and bands:* each is the median pair, B against A.
  - Newcomer TTFT in the stall cell (A/B, primary): pass at 1.03 or better; ambiguous 0.98 to 1.03; fail below 0.98.
  - Longest decode stall, `decoder_gap_ms` max (A/B): pass at 0.95 or better; ambiguous 0.90 to 0.95; fail below 0.90.
    It is a guard: the stall is one chunk pass either way, now with the decode rows in it.
  - Solo guard, the newcomer's TTFT with no decoders (A/B): 0.95 or better, hard.
  - The decoders' tokens inside the newcomer windows (B/A): reported, not gated.
- *Prediction:* the TTFT gain is the decode runs the alternation spends between chunks.
  - On the 1.5B (six chunks of a 3k-token prompt): about 1.05 (0.6).
  - The stall guard: 0.97 to 1.00 (0.7).
  - The solo guard: 1.00 (0.9).
- *Stopping:* fixed. Each cell is read once, with no tuning round. A cell started under heavy load is void, re-run and listed.
- *Wall:* about 45 minutes. The A/A takes about 5; the 1.5B's 16 runs about 12; the 7B's 16 about 20, with its loads.

**Outcomes.**
- Every gate passes on both models: RF3 ships on CUDA by default. The alternation stays for declined models and the other
  backends, and RF4 and RF6 can start.
- TTFT ambiguous on either model: parked. The alternation stays the default, and the reading goes to Francis.
- A failed guard or band: RF3 does not ship. The reading and a mechanism come first.

**G-RF3-S addendum 1 (2026-11-10, before any B run; the thresholds and bands above are unchanged).** The first A/A voided as the
registration says (exit 3): two pair TTFT ratios, 1.020 and 1.070, on one binary, 3 newcomers a run, per-run mean TTFT 0.607 to 0.644 s.
The instrument's run-to-run noise (about 2 to 3% a run, so about 3 to 4% on a pair's ratio) is wider than the 0.03 the check must resolve, so
the A/A is not re-rolled and the bar does not move: the sample grows instead.
- *Newcomers a run:* 3 to 6, with `--decode-tokens` 400 to 1200 so the decoders run through every newcomer's window. A cell's TTFT is the mean of
  its newcomers (the per-run median of three was the noisiest part). The solo guard takes the same 6.
- *The A/A:* two A B B A blocks (8 runs, all binary A) instead of one. It pools the four runs in the A0/A3 positions against the four in
  the B1/B2 positions; the TTFT ratio of the two means must be within 0.03 of 1.00. Four runs a side bring the ratio's noise to about 1.5%, so
  0.03 is two standard deviations. If it voids again at this n, the run stops and the noise floor goes to Francis; it is not re-rolled.
- *Cells and bands:* the four B-against-A cells, A B B A x4 each, and every band are as registered (the median pair, B against A).
- *Wall:* the A/A about 7 minutes (8 runs of about 50 s), the 1.5B stall and solo about 14, the 7B stall and solo about 24 (each stall run gains
  about 10 s from the extra newcomers), so about 55 minutes plus the builds. Start by 18:30 to end by 19:30, well before the 22:00 reservation.
- *Scoring:* `docs/measurements/rf3-2026-11/score_rf3s.py` (`--aa` for the A/A) and `run-g-rf3-s.sh`, both updated with this addendum.

## RF3: reading (2026-11-10, nobara, G-RF3-S under addendum 1): PARKED

Data and provenance: `docs/measurements/rf3-2026-11/run-2026-11-10-addendum1/` (README, `score.txt`, raw json, `cells.log`); the voided first A/A, unscored, is in `run-2026-11-10-void/`. nobara (RTX 2070 SUPER, driver 595.112.07), 11:03 to 11:54 PST, 72 cells, none void, the exclusive lock per cell, the highest start load 0.89. A is `serve-cuda-A` built from `df6393703` (sha256 `eaaa3b4dc2eb`), B is built from the RF3-1 merge `2235b2036` (sha256 `78df8496db8a`); A to B differs by exactly `decoder/mc3_batch.go` and `decoder/model.go`. A first B built from `2cb919116` carried all of main since `df6393703`, so it was not A plus RF3-1 and no scored cell used it.

- **A/A (addendum 1, pooled):** 1.000, inside 0.03. The first A/A (3 newcomers a run) read 1.020 and 1.070 and voided; the instrument's noise was wider than the check, so the sample grew and the bar did not move.
- **Stall cells:** TTFT A/B 1.006 (1.5B) and 1.002 (7B), both AMBIGUOUS (0.98 to 1.03); decoder gap max 0.994 and 1.000, PASS; decoders' tokens inside the newcomer windows B/A 1.0 and 1.007 (reported).
- **Solo guard:** newcomer TTFT A/B 1.004 and 0.999, PASS. Every reply sha identical A against B.
- **Prediction:** the 1.5B stall TTFT of about 1.05 missed at 1.006; the stall-guard (0.97 to 1.00) and solo-guard (1.00) predictions held.
- **Outcome by the registration:** TTFT ambiguous on either model is PARKED: the alternation stays the default and the reading goes to Francis. No guard failed. RF3 on CUDA is lossless and costs nothing measurable, and gains nothing measurable at this prompt length and decoder count.

## OP4m-D6b5: reading (2026-11-09, mac-fit, the exclusive Mac lock)

- **The cell:** FA §4.9's D6b-5, D4b-4's cell on Nemotron-H 4B Instruct (`backend/metal/mc3_step_test.go:TestOP4mD6b_mambaStepSpeed`):
  the int4 Metal `.giw` (v17) read from `~/models`, the step's recurrent blocks on, 4 sequences at depth 128, 64 greedy tokens a
  run, A B B A over 10 pairs. The binary was built at 5621e26b; the runner is `docs/measurements/op4-2026-10/run-d6b5.sh`.
- **The fit check first:** the Metal build's `fit` at ctx 1024, int4: metal RESIDENT, 5.14 GB at one slot (dense 3.40 GB, host
  copy 1.63 GB, recurrent state 0.08 GB, KV 0.02 GB, scratch 10.69 MB) against 11.20 GB free. The cell's own need at four slots
  was 5.44 GiB, and the step's scratch 11.4 MiB.
- **The void rule:** today's (`scripts/void_rule.py`; `docs/measurements/noise-registry.md` §8, changed 2026-11-08 by Francis), in
  place of the registration's reclaimable plus 2 GiB, which predates it. The cell started at a 1-minute load of 3.95 (cap 4.0)
  with 11,624 MB available against a need of 5,600 MB; swap, sampled through the cell, did not grow (2,119.56 MB, peak growth
  0.0); the process's major faults stayed at 20. The registration's own term held too: 7.61 GiB reclaimable against 5.44 + 2.
- **Reading: PASS.** The step ran at 32.1 to 34.0 tok/s and sequential decode at 27.4 to 28.5 tok/s (rates over the four
  sequences). Ratios 1.1264 to 1.2416, median **1.2009**, **10 of 10 pairs faster**, tokens equal in every pair, one counted
  attempt (03:45:29 to 03:55:25 PST). Against the bar (median at least 1.05, at least 9 of 10 faster): PASS. Log:
  `docs/measurements/op4-2026-10/op4m-d6b5-speed-nemotron-h-4b.log`.
- **What it decides:** the option (`--recurrent-step`, which has covered Nemotron-H since OP4m-d6b) meets its feature bar on
  this model. It stays off by default; a default change is Francis's call against its memory (the step's scratch above, and a
  slot's Mamba-2 state, 86.5 MiB on this model by its KV8a checkpoint).
- **Found first, in the cell, and fixed before the counted run.** The first attempt (03:40:50 to 03:41:57, binary at 3c4c2f47,
  start and end checks clean) stopped at pair 0 before any rate was printed: the step's token 0 was 2899, sequential decode's
  1885. Each run rewinds to position 128, which rewrites KV rows in place, but a recurrent state keeps every token it has
  absorbed, so the two arms started from Mamba-2 states 64 tokens apart. `stepCellFill` now zeroes each slot's recurrent state
  and refills it before every run on a recurrent resident. `TestStepCellFill_rewindsRecurrentState` (nemotron-tiny, two slots):
  a refill reproduces the first run's logits bit for bit and a run without one differs; with the reset planted out, 3,072
  logits differ. Log: `docs/measurements/op4-2026-10/op4m-d6b5-unrewound-stop.log`.
- **D4b-4 ran with that defect.** Its arms (Qwen3.5 0.8B, median 1.5799) also started from different DeltaNet states, and its
  tokens were equal anyway. Its rates stand, since the arms ran the same kernels over the same shapes; its "tokens equal in every
  pair" was a weaker check than it read.
- **Also changed in the cell:** a D6b-5 entry (D4b-4's requires a DeltaNet resident), and the start check
  (`stepCellStartVoid`), now the void rule's start terms (`TestStepCellStartVoid`; with the old +2 GiB margin planted back, two
  cases go red).

## OP4m-pod: registration (2026-11-09, before any OP4m-pod code)

- **Question.** Can Metal's per-token decode run each mixer step (attention, Gated DeltaNet, Mamba-2, MLA) as a walk over that
  step's operations in the program, issuing exactly today's dispatches at today's speed? Today each step is one hand-written block
  that branches on family flags (`backend/metal/model.go:resident.encodeAttentionResidualWith`,
  `backend/metal/deltanet.go:resident.encodeDeltaNetMixerOn`, `backend/metal/mamba.go:resident.encodeMambaMixerOn`,
  `backend/metal/mla.go:resident.encodeMLAMixer`).
- **Built:**
  - a table of emitters, one per operation kind, each reading its parameters from the operation (`program.Op`). For example, the
    norm reads its form, `AddOne` and bias; the QK-norm its form; the output gate its sigmoid; the residual its scale.
  - Where one dispatch serves two adjacent operations, the pair is one table entry, emitted once: the output projection with the
    residual add, and a recurrent mixer's out-projection with the residual add.
  - The decode body walks its steps' operations through the table, and an operation the table does not cover declines the build,
    naming it.
  - **Out of scope:** the FFN and own-block steps (MoE, Gemma 4's MoE block, PLE, layer scale), the batched prefill and the
    multi-sequence step (they keep calling today's blocks), and CUDA (Francis's d2 stop).
- **Do-nothing arm:** today's blocks, kept behind a test-only switch and deleted when P-3 passes (as OP4m-b2 and OP4m-d2 did).
- **P-1, the same dispatches (tier 0, hard).**
  - For every tiny fixture that builds resident at int4 and at int8int8, and at `--kv i8` and `--kv f32` where the family
    allows: one decode token's pipeline sequence (`Device.SetDispatchHook`, named by `resident.pipelineNames`) under the walk
    equals today's, dispatch for dispatch, at positions 0 and 5.
  - Plants, each red and recorded: the residual add emitted on its own after the output projection (an extra dispatch), and the
    norm emitter ignoring `AddOne` (red on the Gemma fixtures).
- **P-2, identity (tier 0, hard).** Every logit is bit-identical to today's on the same fixtures over 8 greedy steps, and
  `go run ./cmd/gate identity -backend metal` is IDENTICAL, with every family's outcome unchanged.
- **P-3, speed (tier 1, by day, the exclusive Mac timing lock, fit-checked first).**
  - The metric is decode tokens/s, the walk against today's blocks, in one process.
  - A B B A, 6 pairs; each run is 64 greedy tokens of one sequence at depth 128, and tokens must be equal in every pair.
  - Cells, each int4 from `~/models`: Qwen2.5-Coder 1.5B Q4_K_M (attention), Qwen3.5 0.8B (DeltaNet), Nemotron-H 4B (Mamba-2).
    MLA has no checkpoint that fits this Mac, so it is P-1 and P-2 only.
  - Since P-1 makes the GPU work identical, the gate reads the walk's encode cost.
  - **Pass:** every cell's median pair ratio (walk ÷ blocks) is at least 0.99.
  - **Fail:** a cell resolves slower (every pair under 1) with its median under 0.99.
  - **Parked:** any other cell under 0.99; the walk stays behind its switch, off, for Francis.
  - Void terms are the timed-run skill's, and a void cell is re-run in order.
  - About 10 minutes, start and finish stated first.
- **Stopping:** P-1 and P-2 on each build; P-3 once, on the final build.
- **Outcomes:** all pass, the blocks' family branches go and OP4m-pod closes, with the FFN and own-block steps as its own next
  row. A P-3 fail reverts the walk for the failing step kind. A P-3 park leaves the walk behind its switch, off.

## OP4m-pod2: registration (2026-11-12, after the decode half landed)

The decode half of this row went to main in 90e4d44f8 before this registration was written, so the gates below read a build
that exists; none of them has run. "Registered as OP4m-pod was" (the section above) sets the shape.

- **Question.** Does Metal's per-token decode, with the FFN and own-block steps (dense, non-gated, MoE, Gemma 4's MoE block,
  PLE, layer scale) walked by their operations (`backend/metal/ffn_walk.go:resident.encodeFFNWalk`, over
  `internal/runtime/program/walk.go:LayerWalk`), issue exactly the blocks' dispatches at the blocks' speed?
- **Do-nothing arm:** the blocks (`resident.encodeStep`), behind `mixerWalkBlocksForTest`, which switches the mixer and the FFN
  steps together. It stays until the batched prefill and the multi-sequence step walk.
- **Out of scope:** the batched prefill (`resident.prefillLastImgSrc`; its blocks branch on the walk's fields with no second
  arm, so a read, not a gate, below) and the multi-sequence step (`backend/metal/batch.go`), CUDA (LW-cu) and WebGPU.
- **P-1, the same dispatches (tier 0, hard).** `TestMixerWalk_issuesTheBlocksDispatches` on every census cell that builds
  resident: eight decode tokens issue the same pipelines, dispatch for dispatch, under the walk as under the blocks. The cells
  that walk an FFN or tail step are counted by kind, so a walk nothing reaches cannot pass. Plants, each red:
  `TestFFNWalk_plantsGoRed`'s three (an extra residual, a dropped post-MLP norm, a fresh parallel-block norm).
- **P-2, identity (tier 0, hard).** The same test's logits are bit-identical over the eight tokens, and
  `go run ./cmd/gate identity -backend metal` reads IDENTICAL for every family OP4m-pod's read did (29), with the same 14 not run.
- **Prefill read (tier 0, hard).** `gate identity` of a build at 0df3294dc against main, over the batched prefill's logits,
  per family: IDENTICAL, since the prefill blocks changed only in what they branch on.
- **P-3, speed (tier 1, by day, the exclusive Mac timing lock, fit-checked first).** `TestOP4mPod_decodeSpeed`, as OP4m-pod's
  P-3 ran it: decode tokens/s, the walk against the blocks in one process, A B B A, 6 pairs, 64 greedy tokens at depth 128,
  tokens equal in every pair, each int4 `.v17.giw` from `~/models`. Cells: Qwen2.5-Coder 1.5B (dense), Qwen3.5 0.8B (dense with
  DeltaNet), Nemotron-H 4B (non-gated), Gemma 4 E2B (PLE and layer scale). MoE and Gemma 4's MoE block have no checkpoint that
  fits this Mac: they are P-1 and P-2 only. Void terms are the timed-run skill's; a void cell is re-run in order.
  - **Pass:** every cell's median pair ratio (walk ÷ blocks) is at least 0.99.
  - **Fail:** a cell resolves slower (every pair under 1) with its median under 0.99: the walk reverts for that step kind.
  - **Parked:** any other cell under 0.99: the walk stays behind its switch, off, for Francis.
  - About 12 minutes under the lock; the start and expected finish are stated before it runs.
- **Stopping:** P-1, P-2 and the prefill read on the main build; P-3 once, after this registration merges.
- **Outcomes:** all pass, and OP4m-pod2's decode half is graded; the row stays open for the multi-sequence step and the blocks'
  removal. A red P-1, P-2 or prefill read goes to Francis with the revert of 90e4d44f8's Metal FFN walk named, and no fix
  forward without the coordinator's word.

## OP4m-pod2 part 2: registration (2026-11-12, before any code)

"Registered as OP4m-pod was", on the multi-sequence step's per-row steps; the decode half is read in
`docs/measurements/op4m-pod2-2026-11-12/README.md`.

- **Question.** Can the multi-sequence step (`backend/metal/batch.go:resident.forwardMultiOut`) walk the steps it runs per row
  (a Gemma 4 layer whole, a gated attention, a DeltaNet or Mamba-2 mixer on the row's own slot state, a MoE's norm, route and
  experts) through `resident.encodeMixerWalk` and `resident.encodeFFNWalk`, issuing today's dispatches at today's speed? Today those
  call the blocks (`resident.encodeStep` and the `...On` mixer entries).
- **Built:** the walk's entries take the row's residual and, for a recurrent step, the slot's conv window and state in place of the
  layer's own pair. The step's per-row calls go through the walk.
- **Not walked, and why:** the row-batched kernels (`attnRows`, `ffnRows`, `encodeGemma4FFNRows`, `encodeNonGatedFFNRows`) have
  no per-row operation list to walk; they stay. The batched prefill has no second arm already.
- **Do-nothing arm:** the blocks, behind `mixerWalkBlocksForTest`, which the step reads too. When P-3 passes, the blocks, `encodeStep`
  and the switch are deleted in their own PR (as OP4m-pod's note planned).
- **P-1, the same dispatches (tier 0, hard).** For every census cell that builds the step, B of 2 and 4 rows at mixed depths, one
  step's pipeline sequence (`Device.SetDispatchHook`) under the walk equals the blocks', dispatch for dispatch. Plants, each red and
  recorded: row 0's slot state for every row's recurrent step (the existing `stepDeltaSlot0PlantForTest`), and an extra residual
  after a walked output projection.
- **P-2, identity (tier 0, hard).** The step's logits are bit-identical, walk against blocks, on the same cells, and the existing
  `TestMC3Step_*` identity tests, the recurrent and MoE ones among them, pass unchanged.
- **P-3, speed (tier 1, by day, the exclusive Mac timing lock, fit-checked first).** The step's decode tokens/s, walk against blocks,
  in one process on four slots at depth 128, a sibling of `op4mStepSpeed`'s cell: A B B A, 6 pairs of 64 greedy tokens per
  sequence, tokens equal in every pair. Cells, each int4 `.v17.giw` from `~/models`: Qwen3.5 0.8B (DeltaNet), Nemotron-H 4B
  (Mamba-2), Gemma 4 E2B (a layer whole). Qwen2.5-Coder 1.5B has no walked per-row step and is not a cell; MoE has no checkpoint
  that fits this Mac (P-1 and P-2 only).
  - **Pass / Fail / Parked:** as OP4m-pod2's P-3: median walk ÷ blocks of at least 0.99 in every cell; a fail is a cell resolving
    slower (every pair under 1) with its median under 0.99; any other cell under 0.99 parks the walk, off, for Francis.
  - About 10 minutes; the start and expected finish are stated first.
- **Stopping:** P-1 and P-2 on each build; P-3 once, on the final build.
- **Outcomes:** all pass, and the blocks and the switch go. A red P-1 or P-2 stops the PR; a P-3 fail reverts the walk for the
  failing step kind; a park leaves it behind its switch.

## State after the direct commits (2026-11-12, read at 93a183f82)

Ten commits went to main without a PR on 2026-11-11 and 2026-11-12: 251184470, 8f3d314ff, d9a81ab64, 7ae20d508 and 2f6134fe2
(the Flint family and its record, `docs/completed/task-flint-2026-10.md`), aea0ee556, 0df3294dc, 90e4d44f8, 3b9729c78 (the CI
fix) and 93a183f82. This is a read of the tree and of git objects; no test was run for it. Main's CI is green at 93a183f82.
CI builds the `cuda` tag and runs no device test, and Metal's device tests cannot run there, so green says nothing about
either card.

**What landed that bears on this doc.**
- **The subsystem contracts** (0df3294dc): four interfaces of 22 to 30 methods in `decoder/resident_subsystems.go`, reached
  through `SubsystemsOf`; decoder call sites ask `XAvailable()` and call the method in place of a type assertion. The earlier
  interfaces are all still declared. Row SUB1.
- **`decoder/backend_view.go:BackendModelView`** (0df3294dc): one inspection seam over the model, read by Metal's build and by
  `decoder/fitplan.go`; the `Model.*Resident` accessors stay as forwarders. CUDA and WebGPU do not call it.
- **`decoder/quant`** (0df3294dc): the FP8, GPTQ, AWQ, NVFP4 and MLX config parsing and reconstruction, a Tier 3 leaf
  (`internal/layout/imports.json`, rule `quant-leaf`); `decoder` keeps thin forwarders. Row LAY1.
- **`internal/runtime/program/walk.go:LayerWalk`** (90e4d44f8): a layer's operation list decoded into an attention walk and an
  FFN walk. Metal's decode FFN steps walk it (OP4m-pod2), and CUDA's decode and prefill read it (LW-cu).
- **aea0ee556's five:** `decoder/registry.go:ggufRegistry` in place of the switch; `Model.ResidentSpec` with named structs;
  one safetensors reconstruct dispatch; `internal/app/fitcmd/fit.go:selfMeasureOptions`, so `fit -measure` loads under the
  plan; the citation lint reads `backend/cuda/go.mod`. `fit` now plans with `ResidentWindowedKV` on.
- **The layout ratchets did not move.** The CI fix kept `internal/layout`'s decoder-import ratchet by folding the new test
  files into existing ones. One pin outside it was raised: `webgpuMemGapsPin` (MM9).

**The regenerated decision census was read, and no decision moved.** 0df3294dc's CI run failed `TestDecisionCensus` with 2,999
groups off the golden, and 3b9729c78 rewrote the golden with no reason given
(`internal/runtime/mem/CLAUDE.md` asks for the reason first). The golden holds a hash a group, so the groups were rebuilt from
the code and matched to the stored hashes before and after:
- 29,637 groups a class became 30,143: 506 new, all flint-tiny; none missing.
- 2,940 existing `cpu` D3 groups moved by one figure, the CPU's dense bytes, +96 B, with no placement, context or slot change.
  The cause is the Flint commit, not the consolidation: `decoder/weightbytes.go:Model.cpuStructBytes` prices the size of the
  weights struct, which holds the config by value, and the config gained six fields. 196 of those groups were matched by
  hash; the rest share the mechanism and were not matched one by one.
- 59 D7 groups changed in text only (the f16 decline reason and the unsupported-type list gain `flint`), and the `loads` line
  went 210 to 214.
- The SB2 censuses gained flint lines only; the admission census also carries that one text edit on 135 `cuda` lines.

**Pricing that did change, outside the census.** `decoder/fit/weights.go:SafetensorsWeightBytes` counts quantized safetensors
tensors by new rules (blocks and scaled weights at a multiple, scale and bias tensors at zero). No census fixture has such
names; its effect on a real MXFP4 directory is not measured.

**Not verified here.** Any test outcome on a device; where the arm64 census classes were re-taken; whether the wgpu bump in
93a183f82 moves a WebGPU reading; the 2,744 `cpu` D3 groups a class not matched by hash.

## Plan (proposed 2026-11-12, for Francis)

Nothing here is decided. The order puts reading what landed ahead of building on it, because the two device backends have
no run on record for it and RF2 showed what an unread routing change can cost on CUDA.

1. **Read what landed, by day (rows LW-cu, OP4m-pod2, MM9).**
   - nobara: the CUDA device suite and gate identity at main; `TestMemGate_cuda` and the flint rows; then LW-cu's registered
     decode and TTFT read (about 45 minutes under the exclusive lock, registered first, one screen).
   - the Mac: OP4m-pod2's registration, P-1 and P-2 on the device and gate identity; `TestMemGate_metal`; P-3 against the kept
     blocks (about 10 minutes under the Mac lock).
   - A red identity or a failed band stops the row and goes to Francis with the revert named.
2. **Re-land MM8's three PRs** (#827, then #829, then #721), each after a merge of main, a parity refresh and the census's
   three classes. This is the memory work closest to done, and every day of main moving adds to its conflicts.
3. **Decide the orphans** (SUB1, LAY1): one answer on the old interfaces, then the desk fixes.
4. **Then the rows that were moving before:** OP4m-pod2's multi-sequence step and the blocks' removal; MM5's CPU and WebGPU
   halves; MM5b off Metal; HL-C; OP1-o2's fidelity gate.

**Waiting on Francis:** OP4cu-x's word; the WebGPU gap row (MM9); whether a direct
commit that lands structure must name and change its row in the same commit, as a PR's does (`CLAUDE.md`, "Task docs: the
ledger"). The KV side of the same read is in `docs/tasks/task-kv-architecture-2026-10.md`, "Plan, 2026-11-12".

## Conventions

The archived records' rules for every phase still apply to any row taken up here: gates registered and committed before the
first measurement, the tiers and timing rules in `CLAUDE.md`, and commit subjects that name the row's ID.

<!-- doc-reviewed: 2026-11-12 -->
