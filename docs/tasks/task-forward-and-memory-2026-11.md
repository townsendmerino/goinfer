# Task: the forward pass and the memory planner, what is left (2026-11)

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
| RF2 | active | RF2-1 (#787), RF2-2 (#799), RF2-3a (#804), RF2-3b-1 (#811, a step heads only the rows it reads) and RF2-3b-2a (#823, the flash-decode lane's ragged twins; a step's rows split by arm per layer class) merged. G-RF2-S first-working-build read (2026-11-09, nobara): decode PASS at every depth on 1.5B, 7B and gemma3-1b (0.999-1.001), TTFT PASS (worst 0.982), MC3 PASS at 1.217 with identical replies, never-swap PASS as registered (cell medians -24.5 to +25.6 MB, no swap); verify PASS (1.003 and 1.0004). RF2-3b-2b (split-KV's ragged twins, its scores chunked inside the margin's 512-row pass) in #825; RF2-3b-3 (Forward, ForwardArgmax, ForwardSample and ForwardN as one-sequence ragged steps on the bound slot, today's calls kept beside them for the kill) in #826, identity and G-RF2-I PASS on nobara; then the final-build read | FA §5.4; "RF2: registration", "RF2: addendum 1", "RF2: steps" below; `docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09/README.md` |
| RF3 | active | RF3-1 (the coalescer carries a newcomer's chunk in its decode run, tier 0) in its PR; its premise, the multi-sequence ragged step, stands after RF2's kill. Next: RF3-2 (G-RF3-I) and RF3-3 (G-RF3-S) on nobara | FA §5.4; "RF3: plan and registration" below; `decoder/rf3_ride_test.go:TestRF3_newcomerChunksRideTheDecodeRuns` |
| RF4 | blocked | RF3 and KV5 | FA §7a |
| RF5 | blocked | RF4; it also takes SB1's op-level interface | FA §7a; SS §8 "SB1" |
| RF6 | blocked | RF3 or RF4 | FA §7a |
| OP1-W | open | Gemma 2's attention-logit softcap on WebGPU: register the cells on nobara (Vulkan) before code | FA §4.6; `docs/hardware-matrix.md` |
| OP1-o1 | done | Metal `attention_fa_blk` forms its score through `attn_score` (its softcap build held to float64 per score site by `backend/metal/attn_softcap_test.go:TestAttnSoftcap_faBlk`), bit-identical to the graded text without a cap (`backend/metal/attn_fa_blk_test.go:TestAttnFABlk_routedScoreIsTheGradedKernel`, the resident's own pipeline on the 1.5B and 7B, G1-K's digests); R17's speed re-grade PASS (median routed ÷ graded attention 1.0161, bar ≤ 1.02), and the pin moved to the routed text | FA §4.6 "Owed, and not in OP1"; `docs/measurements/op1-o1-r17-regrade-2026-11-09/README.md` |
| OP1-o2 | active | the CUDA flash-decode lane with the softcap (nobara-sb01): built (`decode_fa_cap`), opt-in only (`Options.FlashDecodeSoftcap`; the exact split-KV lane stays Gemma 2's default), correctness gates pass (G2-K at both KV types, the resident path, a real Gemma 2 2B end to end). Next: the registered fidelity gate by day (G22B at K=3900), then the timed cell; a default change needs Francis | FA §4.6a "Owed, not in OP1"; `docs/measurements/op1o2-fa-softcap-2026-11/README.md` |
| OP4m-D6b5 | done | read 2026-11-09 by day under the exclusive Mac lock: PASS, median step/sequential 1.2009, 10 of 10 pairs faster, tokens equal in every pair, after the cell's recurrent refill fix; `--recurrent-step` stays off by default | FA §4.9; "OP4m-D6b5: reading" below; `docs/measurements/op4-2026-10/op4m-d6b5-speed-nemotron-h-4b.log` |
| OP4m-pod | done | Metal decode walks each mixer step (attention, DeltaNet, Mamba-2, MLA) by its operations (`backend/metal/mixer_walk.go:resident.encodeMixerWalk`): P-1 and P-2 pass on all 197 resident cells, gate identity reads 29 IDENTICAL, and P-3 passes (medians 0.9996, 1.0008, 1.0005) | "OP4m-pod: registration" below; `docs/measurements/op4m-pod-2026-11-09/README.md` |
| OP4m-pod2 | open | the FFN and own-block steps (MoE, Gemma 4's MoE block, PLE, layer scale) walked by their operations, registered as OP4m-pod was; then the batched prefill and the multi-sequence step, after which `mixerWalkBlocksForTest` goes | "OP4m-pod: registration" below; `docs/measurements/op4m-pod-2026-11-09/README.md` "Consequences" |
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
| MM5 | active | the memory planner's CUDA half (nobara-sb01): `cudaKVParts` priced from the declared rows and CUDA's build holding its pools to them (`kvCheckDeclared`); G-MP1 to G-MP4 and the plants pass, G-MP1 moved nothing. Next: its PR's merge; then CPU and WebGPU | FA §4.9b; #739, #741; `docs/measurements/mm5-cuda-2026-11/README.md` |
| MM5b | active | decode scratch's program-side declaration, PR #772: the operations declare their per-row activations (`program.Buffers`' per-row unit, `program.HeadBuffers`), Metal's decode scratch is priced from them (G-MS4 passes; the family-keyed price kept as the oracle, equal on 368 prices), and Metal's build holds the declared widths to its kernels'. G-MS1: only Olmo Hybrid moved (768 B of unused rows dropped), and the paged-cache boundary in a later fix. G-MS5: 29 IDENTICAL. Both Gemini reviews fixed. Next: #772's merge | FA §4.9c ("MM5b's declaration"); #762, #772 |
| MM8 | active | the memory-drift gap fixes: CUDA #721 (dense 314 rows to 10), dense-moe 18, grant 1; WebGPU's int8int8 Gemma 4 MoE dense rows on Vulkan (bound 1,064 B, read on nobara); confirm or strike the towers, host-kept embeddings and verify logits lines. Done: Metal's dense table rows and Gemma 4 MoE's 19,016 B (the plan prices the buffers Metal's build derives, `decoder/metal_dense.go:Model.metalDerivedDenseBytes`; Metal's table is empty), and WebGPU's Gemma 4 MoE router tables (`decoder/webgpu_vulkan.go:Model.webgpuDerivedTableBytes`; the 64 B int4 rows go), and the CPU's Mamba-2 window views (`internal/runtime/mamba2/mamba2.go:Mix` copies each window row, so the window holds no part of the projection's output; the CPU's gap pin 28 to 26), and the CPU's KV (`decoder/residentneed.go:Model.cpuKVBytes` prices what `NewCache` holds: a ring layer's window, twice it at f32 once a read wraps, int8's one f32 scale a KV head; the CPU's gap table is empty, pin 0; the decision census's three classes re-recorded) | SS §6.5 "MM8"; `internal/guard/memgate` gap tables; `docs/measurements/mm8-*` |
| MEM10 | done | the pinned expert copy stays the default (tier 0 and tier 1, 2026-11-08: no in-place pin on nobara; unpinned decode 0.817x / 0.750x). Francis, 13:05: ship the unpinned path as an opt-in, `--moe-experts-unpinned` (default off), which the memory refusal names when the pinned copy is what does not fit | `docs/completed/task-memory-campaign-2026-10.md` §4 "MEM10"; `docs/measurements/mem10-2026-11/README.md`; `docs/measurements/mem10-2026-11/README.md` "MEM10 tier 0"; `backend/cuda/internal/gpudev/mem10_probe_test.go:TestMEM10_fileMappingRegistration_probe`; `docs/measurements/mem10-2026-11/ab-registration.md`; `docs/measurements/mem10-2026-11/ab-registration.md` "Reading"; `docs/measurements/mem10-2026-11/README.md` "MEM10 opt-in" |
| SB2-vk | done | shipped (Francis, 2026-11-08: rebuild and ship): held-out set 6 HOLDS (worst plan 1.8 MiB from the card, two hybrids among four loads) and the fourteen-load re-run holds; the planner prices WebGPU on Vulkan as M1′ over the build's buffer list in its order | SS §8 "SB2"; `docs/measurements/sb2-vk-2026-10-31/README.md` ("Held-out set 6, read") |
| SB11a | blocked | OP4 on CUDA past d2 and OP4 on WebGPU: the walker emits SB11's operation records | SS §8 "SB11" |
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

## Conventions

The archived records' rules for every phase still apply to any row taken up here: gates registered and committed before the
first measurement, the tiers and timing rules in `CLAUDE.md`, and commit subjects that name the row's ID.

<!-- doc-reviewed: 2026-11-08 -->
