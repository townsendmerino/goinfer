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
| RF2 | open | registered below ("RF2: registration"), before any code; next: the CUDA `Step` with ragged attention through the block tables, on nobara | FA §5.4; "RF2: registration" below |
| RF3 | blocked | RF2 | FA §5.4 |
| RF4 | blocked | RF3 and KV5 | FA §7a |
| RF5 | blocked | RF4; it also takes SB1's op-level interface | FA §7a; SS §8 "SB1" |
| RF6 | blocked | RF3 or RF4 | FA §7a |
| OP1-W | open | Gemma 2's attention-logit softcap on WebGPU: register the cells on nobara (Vulkan) before code | FA §4.6; `docs/hardware-matrix.md` |
| OP1-o1 | open | Metal `attention_fa_blk` through attn_score, then the R17 re-grade | FA §4.6 "Owed, and not in OP1" |
| OP1-o2 | open | the CUDA flash-decode lane with the softcap | FA §4.6a "Owed, not in OP1" |
| OP4m-D6b5 | open | the timed cell: the int4 Metal `.giw` of Nemotron-H 4B has been in `~/models` since 2026-11-07; a few minutes by day under the exclusive lock | FA §4.9 |
| OP4m-pod | open | register per-operation dispatch inside the mixer steps | FA §4.9 "OP4m-d2", "What remains" |
| OP4cu-census | open | read an arm64 `TestAdmissionCensus` run | FA §4.9a "C2-2" |
| OP5-S | open | Spark-X2.5, on Metal first | FA §4.7 "OP5" |
| OP5-L | open | LFM2.5's short conv | FA §4.7; `docs/hardware-matrix.md` |
| OP5-G | open | Granite-4.0-H: the bring-up gate, plus the residual scale on CUDA and Metal | FA §4.7, §4.5 |
| OP5-La | open | Laguna, after a fit check | FA §4.7 |
| OP6-M1 | blocked | the M1 peer cell never ran: at 1ecbb5d6 a q4k load on Metal went to the CPU. PR the admission fix (fd146af5, branch `m1-q4k-giw-admission-2026-11-08`), re-pin the serve build in the M1 registration, run M1 by day | FA §4.7 "OP6"; `docs/measurements/peer-vetted-2026-10-07-macbook.md` |
| OP6-def | decision | Francis, once M1 is read: OP6's default switch (FA §8 E5) | FA §8 |
| OP6-C | open | CUDA q4k batched prefill | FA §3.2 |
| OP6-W | open | optional: per-32 activations or a q4k lane on WebGPU | FA §7a |
| HL-C | open | register a CUDA greedy chain under a gated processor, as HL-R did on Metal | FA §6; `docs/measurements/hl-2026-11/README.md` |
| AC | open | map the audit classes' compute share (`docs/measurements/audit-classes-2026-10-08.md` §6) to program paths | FA status block (owner, 2026-10-21); `docs/queue-correctness.md` G47 |
| F-coh | open | CUDA Cohere batched LayerNorm differs by 1 ulp; the CUDA owner's call whether to match the reduction order | FA §4.5 |
| MM5 | active | the memory planner's CUDA half (nobara-sb01): `cudaKVParts` and the pools from the shared declarations (#739); CPU and WebGPU later | FA §4.9b; #739, #741 |
| MM5b | active | decode scratch's program-side declaration (#772, review findings being fixed): `program.Buffers`' per-row unit, `program.HeadBuffers`, the plan's figure, G-MS1, G-MS4 | FA §4.9c; #762 |
| MM8 | active | the memory-drift gap fixes: CUDA #721 (dense 314 rows to 10), dense-moe 18, grant 1; Metal dense tables and Gemma 4 MoE's 19,016 B; CPU int8 per-head scale and the sliding ring; WebGPU tables; confirm or strike the towers, host-kept embeddings and verify logits lines | SS §6.5 "MM8"; `internal/guard/memgate` gap tables; `docs/measurements/mm8-*` |
| MEM10 | open | build the pre-permuted `cuda` `.giw` target so the expert cache registers from the mapping, not about 12 GiB of shared anonymous host memory beside the 13.4 GiB sidecar on the 26B (Francis, 2026-11-08: build it now); M-L, nobara; a tier-0 probe of registering a mapped file first, then a tier 1 reading by day (about 10 min) | `docs/completed/task-memory-campaign-2026-10.md` §4 "MEM10"; `docs/measurements/mem10-2026-11/README.md` |
| SB2-vk | active | rebuilt on main (branch `sb2-vk-ship-2026-11-08`; main's shared-KV pricing kept, the branch's duplicate dropped); held-out set 5 and the fourteen-load re-run registered before any load; next: `run-heldout5.sh` on nobara by day, then ship only if it holds (Francis, 2026-11-08); nobara, M | SS §8 "SB2"; `docs/measurements/sb2-vk-2026-10-31/README.md` ("Held-out set 5"); branch `sb2-vk-plan-2026-10-31` |
| SB11a | blocked | OP4 on CUDA past d2 and OP4 on WebGPU: the walker emits SB11's operation records | SS §8 "SB11" |
| SerWt | open | `SerializeWeights` does not refuse bf16 and f16 weights; give the writer a named `*SerializeError`; S, any box | `docs/measurements/sb4-2026-10.md` |
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

## Conventions

The archived records' rules for every phase still apply to any row taken up here: gates registered and committed before the
first measurement, the tiers and timing rules in `CLAUDE.md`, and commit subjects that name the row's ID.

<!-- doc-reviewed: 2026-11-08 -->
