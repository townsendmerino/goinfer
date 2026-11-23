# WG1, the Mac half: where a WebGPU token and a WebGPU prefill go now (DRAFT, 2026-11-23)

**Status: DRAFT, not registered; nothing has run.** Drafted at the coordinator's request (2026-11-23 10:45 PST) for goinfer-notes
`tasks/task-webgpu-2026-11.md`, row WG1. It becomes a registration when the owner approves it, before any reading. The nobara half is
its own registration. WG0's Mac half ([`../webgpu-baseline-2026-11/README.md`](../webgpu-baseline-2026-11/README.md)) is the baseline:
WebGPU keeps 29 / 37% of Metal at depth 128 and 16 / 20% at 2,048 (0.5B / 1.5B).

**Question.** On the MacBook Pro M1 Pro, WebGPU over Metal: which pipeline classes hold a WebGPU decode token at depths 128, 2,048,
8,192 and 16,384, and which hold a batched prefill at 256, 1,024 and 4,096 rows, on qwen2.5-coder 1.5B Q4_K_M at `int8int8`? WG2's
levers are ranked from the answer.

**Instruments, as they are; no new code.**
- **Decode:** `backend/webgpu/decode_dispatch_profile_test.go:TestDecode_dispatchProfile`, by ablation: the token's plan is re-recorded
  R times into one pass with one pipeline class omitted, and the class's cost is the difference from the full plan. Run with
  `GOINFER_DECODE_POS=128,2048,8192,16384` and `GOINFER_DECODE_GGUF` set to the 1.5B in `~/models`.
- **Prefill:** `backend/webgpu/prefill_dispatch_profile_test.go:TestPrefill_dispatchProfile` with `GOINFER_PREFILL_PROF_P=256,1024,4096`.
  This one is not an ablation. Its own comment explains why: the prefill is not a replayable step list. It accumulates per-category
  wall-clock with a sync at each boundary, so its categories sum to a little more than the pipelined pass. They are reported as
  that, not as what deleting a class would save.
- **G38 check, done for this draft:** both instruments poll blocking (`Poll(true, nil)` in the decode harness and in
  `prefill_prof.go`). No device timestamps on this stack, so every timing is host-side.

**Quant.** Both tests load `int8int8` (WG1's row says so); WG0 timed `int4`. The ranking is reported at `int8int8` only, labelled
as such. Whether it holds at `int4` is not asked here: answering it needs a quant knob in the tests, which is a separate change.

**What is reported, per box and phase:**
- a ranked table: class, share of the token or the pass, and the roofline where one can be computed (bytes moved over the M1 Pro's
  bandwidth, quoted as the nominal ~200 GB/s and labelled nominal unless a measured figure is cited);
- a large share at its roofline is not ranked a lever (WG1's rule, from G36: attention at 13.68 ms against a 0.066 ms roofline).

**Conditions.**
- By day, under the Mac's exclusive timing lock, models from `~/models`.
- Each load fit-checked first: the 1.5B at `int8int8` with f32 KV at 16,448 positions is under 2 GB of weights plus 0.94 GB of KV (28 layers × K and V × 256 floats × 16,448 positions).
- Power source and thermal state recorded per run (`pmset`); swap growth while a run is going voids it (`noise-registry.md` §8).

**Stopping rule.** Each instrument runs twice, back to back, at the test's own repetition count; no further runs. A class whose rank
differs between the two runs is reported as unranked within its neighbours, not averaged into a rank.

**Outcomes.**
- **A class above its roofline by a wide margin and a large share:** a WG2 lever, named with its share and roofline.
- **The top classes at their roofline:** no kernel lever on this box at that phase; WG2 looks elsewhere (dispatch floor, quant).
- **Unstable ranks:** reported as such; no lever is named from them.

**Cost.** Decode: four depths, about 10–20 min. Prefill: three row counts, about 10 min (the 4,096-row pass is new on this box).
About 30–45 min under the exclusive lock in total.
