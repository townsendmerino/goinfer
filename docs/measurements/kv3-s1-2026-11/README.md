# KV3-S1: `glu_quant_batched` writes in place (2026-11-16; a 16-core amd64 box, 62 GB RAM, one 8 GB RTX 2070 SUPER)

Ledger row KV3-S1 (`docs/tasks/task-kv-architecture-2026-10.md`; KV3, "Inside a pass", item 1). Francis set it 2026-11-16. **Not a speed claim:** it removes one scratch buffer, so no timed reading is registered; the bar is bit-identical outputs and scratch bytes that do not go up.

## What changed

`backend/cuda/prefill_batched.cu:glu_quant_batched` leaves its product in the gate row it read: each thread reads `g[i]` and writes the same index, and the quantize loop reads the row after the reduction's `__syncthreads`. `g` carries no `__restrict__` (a restrict pointer written through an alias is undefined behaviour); `u` is a different buffer and keeps it. The signature loses `gOff` (the gate row starts at `row * I`, and both launches passed 0) and `dscr`. The three launches take the new arguments:
the prefill pass's MLP (`backend/cuda/prefill.go`), its Gemma 4 per-layer-embedding gate (`pleGB` is the in-place buffer), and the block drafter's (`backend/cuda/drafter.go`, whose scratch drops the same buffer). `dScrB` is no longer allocated and `backend/cuda/prefill_scratch.go:cudaResident.PrefillScratchBytes` stops pricing it, in the same commit. The five `prefill_batched` PTX variants are rebuilt with `build_ptx.sh prefill_batched` (the script regenerated the old five byte-identically first).

## Gates

| | what | reading |
|---|---|---|
| **Red first** | `TestKV3_gluQuantWritesInPlace` (`llama-tiny`, whose intermediate width 128 is no other scratch row's): the pass allocates exactly two `M x inter` float buffers | on main: **3** at M=16 and M=100 (`red-on-main.log`); with the change 2, and `TestKV3_prefillScratchBytes_matchesThePass` still holds the figure to the pass |
| **Bit-identical, the kernel** | `kernels/glu_quant_batched_test.go:TestGluQuantBatched_inPlaceIsBitIdentical`: both activations, a width that is not a multiple of the block, an up operand read at an offset; the SHA-256 of the packed rows and row scales | the digest `793db9f9...` was **recorded from main's out-of-place kernel** (twice, stable) and the in-place kernel reproduces it |
| **Bit-identical, the pass** | `gate identity -backend cuda 5a8742643 HEAD` (`identity.log`), with the gitignored fixtures placed in the worktree (`scripts/worktree_fixtures.sh`; the first run, without them, left `gemma4` unrun and is not used) | **30 IDENTICAL, 0 DIFFERENT, 14 NOT RUN**; `llama`, `qwen2`, `gemma4` (6 of 10 cells) ran |
| **The per-layer-embedding path and the drafter** | `emodel-tests.log`: `TestGemma4EModelPrefill_batchedBitIdentical`, `TestGemma4EModelImage_batchedPrefillMatchesCPU` and their plants; `gemma-drafter-tests.log`: `TestResidentDrafter_*` (block parity among them), `TestDFlashLoop_lossless`, `TestGemma4BidirSpanPrefill_*` | all pass; the real-E2B and E4B cells skip without their checkpoints. The Gemma-4 run in `gemma-drafter-tests.log` was stopped at 15 passes because its regex also matched the 26B heavy cells (long); it is a partial record, and the E-model cells were run on their own |
| **PTX** | `go run ./cmd/gate ptx` | **70/70 PTX regenerate byte-identically**: GREEN |
| **Day loop** | `touched.py` | 18 checks pass |

## Scratch bytes before and after (`PrefillScratchBytes`, `scratch-before.log`, `scratch-after.log`; qwen2.5 int4 resident, `--ctx 4096`)

| | rows | before | after | saved |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 512 | 75.26 MiB | **57.26 MiB** | 18.00 MiB |
| | 2,048 | 280.03 MiB | 210.03 MiB | 70.00 MiB |
| Qwen2.5-7B | 512 | 155.26 MiB | **117.26 MiB** | 38.00 MiB |
| | 2,048 | 598.03 MiB | 450.03 MiB | 148.00 MiB |

The saving is one `M x inter` float32 buffer a pass (`M x 8,960 x 4` and `M x 18,944 x 4` bytes). Host bytes: unchanged. The block drafter's scratch drops its own `M x inter` buffer as well (not measured here). The task doc's own estimate was 17.5 and 37 MiB at 512 rows; the measured figures include the allocator's rounding.

**Not done:** KV3-S2 (phase aliasing) stays blocked on the doc's own trigger. Metal is not the same fix (its gate and up share one fused buffer; the doc says so).
