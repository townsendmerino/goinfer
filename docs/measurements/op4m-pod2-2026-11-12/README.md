# OP4m-pod2: readings (2026-11-12, mac-metal)

These are the readings for "OP4m-pod2: registration" in `docs/tasks/task-forward-and-memory-2026-11.md`, of the Metal decode FFN and
own-block walk that went to main in 90e4d44f8. All were taken on the MacBook (M1 Pro, 16 GB, arm64) at 4baa8f78c (main plus the
test-comment and counter fixes in this PR), through the shared or exclusive timing lock, from a worktree. Raw: the `.log` files here.

## P-1 and P-2: PASS (`p1p2-summary.log`)

- `TestMixerWalk_issuesTheBlocksDispatches`: 202 resident cells checked, 202 walk a mixer step and 202 an FFN or tail step. Eight
  decode tokens issue the same pipelines, dispatch for dispatch, and the same logits bit for bit, walk against blocks. (OP4m-pod's
  read had 197 cells; flint-tiny adds the rest.)
- Plants: `TestMixerWalk_plantsGoRed` red on 7 fixtures (extra residual) and 5 ((1 + w) ignored); `TestFFNWalk_plantsGoRed` red on
  5 (extra residual), 2 (dropped post-MLP norm) and 1 (fresh parallel norm), each as the fixtures' own steps predict.
- The test's FFN counter is new in this PR; it counts every cell, since every resident has an FFN step, so it guards only a build
  that walks none.

## Gate identity, decode and batched prefill: GREEN

`go run ./cmd/gate identity 0df3294dc origin/main -backend metal` (main 4edbe93e9), tiny assets, int4 and int8int8, 8 tokens per
prompt with the first dumped step the prefill's logits: **30 IDENTICAL, 0 WITHIN TOLERANCE, 0 DIFFERENT, 14 NOT RUN** (the same count as OP4m-pod's
read; the gate prints each one's reason); wall 2m4s. 0df3294dc is the parent of 90e4d44f8 on
Metal, so one run covers the decode walk and the prefill blocks that now branch on the walk's fields. The gate notes that 0df3294dc
is not the `validated_at` of 30 families, so nothing here is recorded into the parity manifest.

## P-3, speed: PASS (`p3-run1.log`, `p3-run2.log`)

Harness `backend/metal/mc3_step_test.go:TestOP4mPod_decodeSpeed`, one process per cell, int4 `.v17.giw` from `~/models`, depth 128,
64 greedy tokens, 6 pairs A B B A, tokens equal in every pair, exclusive timing lock. No cell void: start loads 2.46 to 3.28
(bar 4.0), available memory covered need, swap did not grow.

| cell | walked steps | median ratio walk ÷ blocks | pairs under 1 | verdict |
|---|---|---:|---:|---|
| Qwen2.5-Coder 1.5B Q4_K_M | attention 28, dense FFN 28 | 1.0003 | 3 of 6 | PASS |
| Qwen3.5 0.8B | attention 6, DeltaNet 18 | 1.0016 | 3 of 6 | PASS |
| Nemotron-H 4B | attention 4, Mamba-2 24 | 1.0005 | 1 of 6 | PASS |
| Gemma 4 E2B Q4_0 | attention, dense FFN, PLE, layer scale, 35 each | 0.9991 (run 1), 0.9981 (run 2) | 4 of 6, 4 of 6 | PASS |

- **What ran twice, and why.** Run 1 (16:33:40 to 16:53:53 PST) skipped the Coder cell: its start gate (load1 under 3.5) never
  cleared in 15 minutes (last 4.72), so nothing was timed and nothing voided. Its harness also printed only mixer steps for each
  cell, so Gemma's PLE and layer-scale steps did not show. Run 2 (16:54:24 to 16:56:03 PST, harness fixed to count the FFN and tail
  steps) timed the Coder cell and re-timed Gemma, which printed the four kinds. Gemma's run 1 reading is the one graded
  (0.9991); run 2's agrees (0.9981). Qwen3.5 0.8B and Nemotron-H 4B were timed once, by run 1's harness, whose print named mixer kinds only; their FFN steps are
  covered by P-1 and P-2.
- The bar was a median of at least 0.99 in every cell. The lowest pair is 0.9843 (Coder, pair 5, load 2.59).

## Not read

- MoE and Gemma 4's MoE block on the device beyond P-1 and P-2: no checkpoint that fits this Mac.
- A speed reading of the batched prefill: it keeps its blocks, so there is no walk to time there.
- The multi-sequence step (`backend/metal/batch.go`): still on the blocks.

## Also run, same build (`memgate-g1a-summary.log`)

- `TestMemGate_metal` with flint-tiny: PASS (10.6 s). `TestKVPages_everyResidentFixture` (G1a): PASS (42.6 s); flint-tiny runs
  (210 logit rows, 3 slots, prefill kernels none, 6 refused). The NOT RUN fixtures there are
  model types this build does not load.

## Part 2, the multi-sequence step's walk (2026-11-12, mac-metal, ef05fcb6e): P-1, P-2 and P-3 PASS

Registration: "OP4m-pod2 part 2: registration" in `docs/tasks/task-forward-and-memory-2026-11.md`.

- **Red first.** Against main's step, which calls the blocks, `TestMultiSeqWalk_plantsGoRed` was red on 3 of 3 DeltaNet fixtures
  (the plant never reached the step); `TestMultiSeqWalk_issuesTheBlocksDispatches` was green there, as it compares two arms of
  the same code.
- **P-1 and P-2:** `TestMultiSeqWalk_issuesTheBlocksDispatches` passes on 7 fixtures (dense, qwen35-tiny, qwen3next-tiny,
  qwen3_5-tiny, nemotron-tiny, three Gemma 4) at 2 and 4 rows: identical pipelines and bit-identical logits, walk against blocks.
  The extra-residual plant is red on all 3 DeltaNet fixtures. `TestMC3Step_*` (identity, recurrent, MoE, Gemma 4, plants) pass
  unchanged. `gate identity origin/main ef05fcb6e -backend metal`: 30 IDENTICAL, 0 DIFFERENT, 14 not run.
- **P-3: PASS on all three cells.** Harness `TestOP4mPod2_stepSpeed` and `TestOP4mPod2_gemma4StepSpeed`, 4 slots, depth 128, 64
  tokens per sequence, 6 pairs A B B A, tokens equal in every pair, exclusive lock, bar a median walk ÷ blocks of at least 0.99.

| cell | median step walk ÷ blocks | pairs under 1 | verdict | log |
|---|---:|---:|---|---|
| Qwen3.5 0.8B (DeltaNet) | 1.0091 | 2 of 6 | PASS | `p3-seq-run1.log` (19:29 to 19:37 PST) |
| Nemotron-H 4B (Mamba-2) | 1.0007 | 3 of 6 | PASS | `p3-seq-run3.log` (22:20 to 22:28 PST) |
| Gemma 4 E2B | 1.0022 | 2 of 6 | PASS | `p3-seq-run3.log` |

  - **Void cells, listed.** In run 1 the Nemotron-H and Gemma 4 E2B cells voided on paging (the process took 3 and 1 page-ins, with
    available memory at 6.39 GiB against a need of 5.44 on Nemotron-H, another session loading the Mac). A re-run
    (`p3-seq-run2.log`, from 19:37:34 PST) voided Nemotron's attempt 1 the same way (185 faults, swap 1400.7 to 1605.2 MiB) and was
    stopped in its second attempt on the coordinator's word. Nothing was averaged in. Run 3 read both cells once each, with no
    void: start loads 2.23 and 1.57, swap and major faults flat across every pair.
  - **Power source.** Run 3 ran on battery (`pmset`: Battery Power, 100% at the start, 95% at the end), on Francis's word relayed by
    the coordinator at 22:25; both arms of every pair ran under that one condition, and nothing in the runner, the void rule or the
    harness refuses battery, so no refusal was lifted and no threshold changed. Run 1's power source was not recorded.
  - Gemma 4 E2B's pairs 5 and 6 (1.0695, 1.1897) are the blocks arm running slow, not the walk fast; the median ignores them.
