# MC1 on WebGPU — multi-slot resident KV: interleaved conversations hold the 1-client figure (2.81× at 4 clients) (2026-09-27)

MC1 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md), on the WebGPU backend. N conversations'
KV stay resident at once, so interleaved turns reuse their own prefix instead of evicting each other. Generation is
still one at a time, because WebGPU has no batch stepper.
- Built in `3926f927`. The grading was pre-registered in the same commit, before any timing.
- The twins are the Metal record [`concurrency-mc1-2026-09-26.md`](concurrency-mc1-2026-09-26.md) and the CUDA record
  [`concurrency-mc1-cuda-2026-09-27.md`](concurrency-mc1-cuda-2026-09-27.md).

**Result: every hard gate passes, and it ships.**
- With 4 resident KV slots (`-kv-sessions` default 4), the aggregate at 2 and 4 clients is **0.998× / 1.000×** the
  1-client figure. The previous build, with one slot, reads 0.355× / 0.356×.
- **4-client aggregate: 26.00 against 9.27 tok/s, 2.805× (median of 3 interleaved pairs: 2.642, 2.805, 2.814).**
  2 clients: 2.803×.
- Every client reuses its whole history on every turn. The one-slot build reuses only the 7-token lead.
- The slowest turn at 4 clients falls from 98.9–99.7 s to 23.4–23.7 s.
- A lone request is unchanged: p50 1.001×, p99 1.008×.
- Every reply is identical. Old and new agree on all 108 turns at 2 and 4 clients, which was reported, not gated.

**Why the win is so much larger than CUDA's 1.25×.** On this backend Qwen2.5 prefills one token at a time:
`PrefillLast` declines q/k/v bias off Vulkan (`gpu/residency.go`). So a thrashing turn re-prefills its whole
conversation (176 → 786 tokens by turn 6) at decode speed, where CUDA's batched prefill makes the same re-prefill
cheap. The mechanism is the same on both backends; the thrash simply costs more here.

## Setup

- The MacBook Pro: M1 Pro, 16 GB, macOS 26.6.2. WebGPU runs through wgpu-native's Metal backend
  (`oliverbestmann/webgpu` v1.36.0).
- Model: qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`, with `-quant int4` and serve defaults. Both builds load
  the same `.int4.webgpu.giw` sidecar: the first server wrote it once, verified, before any timed cell.
  - The context is the WebGPU default, 16384 tokens, with f32 KV: ~0.94 GB per slot.
  - Decode path: `webgpu:metal-resident (int4)`.
- `scripts/bench_w7_plain.py --engines goinfer --backend webgpu --fixed-nonce`: 6 plain turns per client, 128
  greedy tokens per turn, and a fresh server per cell.
- Two builds, each built once with `go -C gpu build -tags gpu ./cmd/serve` and named by hash. Both binaries carry
  `vcs.modified=false`.
  - **old** = `serve-webgpu-68f2dbdf`, the tree just before this work, where the WebGPU resident has one KV slot. It
    was built from a clean worktree.
  - **new** = `serve-webgpu-3926f927`. It was built from `9cad1d2e`, whose one extra commit re-points a doc citation,
    so its code is `3926f927`'s. That is why the binary's VCS stamp and the harness header read `9cad1d2e`.
  - Every new server's banner reads "4 conversations kept resident", and no clamp message was logged. Every old
    server reads "one conversation".
- Cells: 1, then 2, then 4 clients, each old/new × 3 pairs in the order old new new old old new.
- Each server start waited for load1 ≤ 2.0 (`sysctl vm.loadavg`) and no other serve process.
  - The bench's own pre-server memory wait asks for 3000 MB of *free* pages. That count excludes the inactive pages
    macOS reclaims first. **Deviation from the pre-registration:** the first cell (`old1_1`) waited its full 60 s
    and proceeded with 1,490 MB free. Free plus reclaimable was ~6.5 GB at the time. Four later cells (`old1_2`, `old1_3`,
    `new1_3`, `new4_1`) waited one ~2 s check before 3000 MB were free, and the rest did not wait.
  - Nothing inside a cell is affected. `old1_1` reads 26.06 tok/s, identical to `old1_2` and `old1_3`.
- One session, 2026-09-27 20:17–20:57 PDT.
- Raw data: [`concurrency-mc1-webgpu-2026-09-27/`](concurrency-mc1-webgpu-2026-09-27/):
  - `w7.json`, `w7-run.log`, and `w7-servers.log` (every banner);
  - `gates.py` and `gates-output.txt`;
  - `run-w7.sh`;
  - `banner-check-3926f927.log` (the untimed first start that wrote the sidecar);
  - `gpu-suite.log` (the full `gpu` package);
  - `gpu-heavy-and-1.5b-identity.log`.

## Results (qwen2.5-coder-1.5b)

| build | clients | aggregate tok/s (3 cells) | ÷ its 1-client (median) | prompt − reused, turns 2–6 (every client) | slowest turn |
|---|---:|---|---:|---|---:|
| old | 1 | 26.06, 26.06, 26.06 | — | 26, 27, 24, 20, 27 | 5.81 s |
| old | 2 | 9.26, 9.26, 9.27 | 0.355× | 176, 331, 483, 631, 786 | 49.7 s |
| old | 4 | 9.27, 9.27, 9.25 | 0.356× | 176, 331, 483, 631, 786 | 99.7 s |
| **new** | 1 | 26.18, 25.88, 26.04 | — | 26, 27, 24, 20, 27 | 5.86 s |
| **new** | 2 | 26.14, 25.94, 25.98 | **0.998×** | 26, 27, 24, 20, 27 | 11.9 s |
| **new** | 4 | 24.49, 26.00, 26.03 | **1.000×** | 26, 27, 24, 20, 27 | 23.7 s |

- The new build's slowest turn grows with the client count (5.9 → 11.9 → 23.7 s). Generations are still one at a
  time, so a turn waits for the others' turns. That is admission, not prefill: each of its turns prefills only the
  new user turn.
- MC3's batched step is what removes that wait on Metal and CUDA, and WebGPU has none.
- `new4_1` (24.49) is the one cell below the others (26.00, 26.03). Load had risen to 2.16–2.40 around it, and the
  next cell's gate waited 87 s for it to fall. Its reuse is the same as every other new cell's, and the gate reads
  the median.

## Gates (pre-registered; `gates-output.txt`)

1. **Identity: PASS.** At 1 client, old == new on every turn of every pair. At 2 and 4 clients, new's client 0 ==
   new's 1-client run on every turn. Reported only: old vs new at 2 and 4 clients, 0 of 108 turns differ.
2. **Reuse: PASS.** At 2 and 4 clients, every new client's turn 1 reuses 0 (cold), and its turns 2–6 prefill exactly
   what the 1-client run's do: 26, 27, 24, 20, 27 tokens.
3. *(Expected band, not a gate.)* 0.998× at 2 clients and 1.000× at 4, both in the 0.90–1.0 band.
4. **Ship: PASS.** The 4-client aggregate new ÷ old, median of 3 pairs, is **2.805×** (bar ≥ 1.03×).
5. **Solo guard: PASS.** 1-client p50 1.001× and p99 1.008×, medians of 3 pairs (bar ≤ 1.05× each).

## Identity before timing

All in `gpu/kv_slots_test.go` on this Mac. The logs are archived.
- `TestWebGPUKVSlots_interleavedMatchesAlone` runs two conversations sharing a 3-token lead, interleaved A B A B A B
  on 2 slots, against each alone, with a one-slot control that must thrash. It passes on:
  - llama-tiny with f32, int8 (the per-head scale buffers) and f16 KV;
  - deepseek-tiny (MLA: one latent buffer per layer).
  - mistral-tiny-window **skips** here: its weights are a gitignored fixture not built on this Mac, and
    re-pinning it would change a committed golden. nobara has it.
- The same scenario on **qwen2.5-coder-1.5b** (int4, 48 tokens × 3 turns × 2 conversations): the interleaved ids
  equal the ids alone. Reuse per turn is 0 / 55 / 105 on both conversations, and the one-slot control reads
  0 / 3 / 3 and 3 / 3 / 3.
- `_adapterFollowsTheSwitch`: conversation A runs on a LoRA adapter, which is bound before the slot is acquired, and
  B on the base model. The test first checks that the adapter changes A's reply, so the scenario can see a lost
  adapter.
- `_forwardNFollowsTheSlot`: a slot's lazily grown `ForwardN` pool never serves another slot. Rows on slot 0, then
  slot 1, then slot 0 again equal fresh one-slot models' bit for bit.
- `_slotOwnsItsKV`: every per-sequence buffer is the slot's own, at slot 0's size, and every weight is shared.
- `_recurrentKeepsOne`: qwen35-tiny (DeltaNet) and nemotron-tiny (Mamba-2) asked for 4 slots allocate 1.
- `_clampedBuild`: a failure injected halfway through slot 2 of 4 grants 2, both usable, with zero live device
  bytes after Close.
- `_pricedAgainstMemory` exercises both darwin ceilings on a real build, and `TestDarwinKVSlots` is the arithmetic
  table.
- **Mutation-checked.** Removing the adapter move from `UseKVSlot` fails the adapter scenario on A *and* on B (B's
  runner kept A's delta). Letting a new slot keep slot 0's KV buffers fails identity and the `ForwardN` test.
- The full `gpu` package passes (132 pass, 0 fail, 57 skip; the skips are the heavy and absent-fixture tests). The
  heavy resident tests pass (`GOINFER_HEAVY_TESTS=1`).
  - `TestSpeculative_C03_concurrentResidentClaim` was refused by the load-time fit guard inside that batch, on
    memory the earlier tests in the same process still held.
  - It passes alone (42 s). It requests one slot, so the new code is inert there.

## What this does not cover

- **The discrete-GPU clamp is unmeasured.** Off darwin, a slot is clamped at the first failed allocation, and each
  further slot must leave a 384 MiB probe allocatable. That path ran only under the injected-failure test.
  - Whether wgpu-native on Vulkan fails such an allocation cleanly, without losing the device, has not been seen on
    real hardware.
  - The 7B on nobara's 8 GB card at the 16k f32 default (~1.9 GB per slot) is the case where it binds.
- **Context against slots.** CUDA shrinks an unpinned context so the requested slots fit (`947e06ce`). WebGPU does
  not, because it has no free-memory query to size against. Whether it should is an open owner decision.
  - **Decided after this grading, 2026-09-27:** slots first, as on CUDA.
    - On darwin it is done (`slotsBeforeContext`); see the task doc's "MC1 on WebGPU".
    - Discrete GPUs wait on the nobara measurement above.
    - The graded configuration, where 4 slots fit at 16k, is unchanged by it, so these numbers stand.
- **WebGPU off the Mac.** This grading is the M1 Pro through wgpu's Metal backend only.
