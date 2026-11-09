# OP4m-pod: readings (2026-11-09, mac-sb11m)

These are the readings for the registration in `docs/tasks/task-forward-and-memory-2026-11.md`, "OP4m-pod: registration" (#817).
Part 1 (#820) walks the DeltaNet, Mamba-2 and MLA steps by their operations; part 2 walks the attention step.

## P-1 and P-2: PASS

- **`TestMixerWalk_issuesTheBlocksDispatches`** (`backend/metal/sb2_census_test.go`):
  - All 197 census cells that build resident walk a mixer step: int4, int8int8, `--kv i8` and `--kv f32`, and the SSM cells.
  - Eight decode tokens issue the same pipelines, dispatch for dispatch, and the same logits bit for bit, under the walk as
    under the blocks.
- **Plants, `TestMixerWalk_plantsGoRed`.** Each plant's expected red is computed from the fixture's own steps.
  - A residual added again after an output projection that already adds it is red on 7 fixtures.
  - The input norm ignoring `(1 + w)` is red on 5, the Gemma fixtures among them, and quiet elsewhere.
- **`gate identity -backend metal`:** 29 IDENTICAL, 0 DIFFERENT, 14 not run (Metal declines them, as at base), on part 1 and on
  part 2. The attention block's move into segments was also gated on its own: 29 IDENTICAL.
- **A NoPE layer** has no rope operation. The block rotates it by its all-zero table, an identity, and the walk issues the same
  dispatch. Dropping that dispatch from both arms is a one-line follow-up.

## P-3: PASS (05:35:10–05:37:38 PST, one exclusive timing-lock hold, commit fc5b65ec)

- The harness is `backend/metal/mc3_step_test.go:TestOP4mPod_decodeSpeed`, in one process. Each run refills depth 128 untimed and
  times 64 greedy tokens, and tokens were equal in every pair. The 6 pairs alternate A B and B A.
- Every checkpoint was fit-checked first, and each is an int4 `.v17.giw` from `~/models`.
- No cell is void: start loads were 0.83 to 1.60 (bar 4.0, or 12.0 per pair), available memory covered need with 6 GiB or more
  to spare, and swap did not grow. Raw: `p3.log`.

| cell | walked steps | walk tok/s (median of pairs) | median ratio walk ÷ blocks | pairs under 1 | verdict |
|---|---|---:|---:|---:|---|
| Qwen2.5-Coder 1.5B Q4_K_M | attention 28 | 92.2 | 0.9996 | 4 of 6 | PASS |
| Qwen3.5 0.8B | attention 6, DeltaNet 18 | 125.5 | 1.0008 | 2 of 6 | PASS |
| Nemotron-H 4B | attention 4, Mamba-2 24 | 28.2 | 1.0005 | 1 of 6 | PASS |

The bar is a median of at least 0.99 in every cell. The ratios sit within ±1% of 1 except Nemotron's first two pairs (1.026,
1.031), whose blocks arm ran slow in both.

## Consequences

- OP4m-pod closes, as registered. The FFN and own-block steps (MoE, Gemma 4's MoE block, PLE, layer scale) are the next row.
- `mixerWalkBlocksForTest` stays. The registration said to delete it once P-3 passed, but the blocks still run the batched
  prefill and the multi-sequence step, and P-1 is what holds the walk and those blocks equal. It goes when those paths walk.
