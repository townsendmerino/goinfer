# SB2 memory reads, step 3, PR 2: the derived model on real checkpoints (2026-10-29, nobara-sb01)

The model, the bound and the gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, PR
2", at `352681be8`, before the code. Box: nobara, RTX 2070 SUPER, driver 595.91.07, the checkpoints from `~/models`. The pass
held the timing lock exclusively and started with only the desktop compositor on the card (`gdm2-card.txt`).

## G-DM1

`TestDeriveMemory` passes. Derive mode (`cudaBackend.deriveMemory`) is committed.

## G-DM2: the first pass is void

`void1-*` (22:36). The probe left `Options.Quant` unset, so the checkpoints loaded at f32, not int4: the quant serve's fit plans
at and the cuda gates load at. At f32:
- the host fit guard refused four of the seven loads;
- the 7B was refused at the weights check;
- only the 0.5B built, and its KV error was -1,403 MiB.

That figure is the f32 dense pricing against a device that does not hold f32. It is not counted. The probe was fixed to int4
and re-run.

## G-DM2: the pass (22:37–22:39, `gdm2.txt`, probe `sb2_dmprobe_test.go.txt`)

Error = derived − real, at each site the build read. P is the registered memory probe's reading before each load:
7,545,815,040 B on every cell.

| cell | built | sites read: error (MiB) | decisions, real vs derived |
|---|---|---|---|
| 0.5b default | yes | weights 0, kv +5.1, prefill +3.1 | equal |
| 0.5b 4-slots | yes | weights 0, kv +5.1, prefill +5.1 | context 32768 vs 16384 (see below) |
| 7b default | yes | weights 0, kv +19.4, prefill +17.4 | equal |
| 7b 2-slots | yes | weights 0, kv +17.4, prefill +19.4 | context 21120 vs 10560 (see below) |
| 26b expert-cache | refused at the KV fit (an unplanned 146,434-position context: 66 GB of KV) | weights 0, kv +82.0 | both refuse at kv |
| 26b expert-cache windowed | yes | weights 0, kv +74.0, grantPre +72.0, slots +47.0 | expert slots 28 vs 29 |
| 9b Qwen3.5 | yes | weights 0, kv +14.8, ckpt +12.8, prefill +14.6 | equal (checkpoint 52,690,944 B kept on both) |

## The rule

The largest |error| is 82.0 MiB, at the 26B's KV site. That is between 64 and 128 MiB: **ambiguous, parked**. No census
(PR 3) is built, and the decision goes to Francis.

Every error is positive: the model always thinks more is free than the card says, which is the dangerous direction. The dense
and 9B errors are 3–20 MiB. The 26B's are 47–82 MiB. On the windowed 26B, the 47 MiB at the slots site is enough to turn the
build's 28 expert slots into 29.

## Two findings beyond the bound

1. **The plan does not model `kvRaise`.** With more than one KV slot and no context asked for, the build lets one
   conversation's cap grow to the whole KV pool (`kvCapFor`): 16384 → 32768 on the 0.5B's 4 slots, and 10560 → 21120 on the
   7B's 2. `memPlan.CtxCap` reports the per-slot plan. Step 1's G-MP3 and PR 1's G-PA3 did not see it: every census cell asks
   for one slot.
2. **The 26B's larger error is where the build allocates most.** Its KV-site error (74–82 MiB) is about four times the dense
   models'. Candidates, not yet attributed:
   - the expert cache's pinned host mapping;
   - the float-activation buffers a Gemma 4 MoE build prices in `postPoolBytes` but allocates before the KV pools;
   - the decode scratch the plan does not price.
