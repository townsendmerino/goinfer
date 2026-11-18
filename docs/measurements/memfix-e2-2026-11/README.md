# E2: the CUDA weights in one device allocation (memory audit F3 a, 2026-11-18)

One box: nobara, RTX 2070 SUPER 8 GB, driver 595.112.07, desktop session up (`kwin_wayland` holds a CUDA context). `before` is
`main` at `8ba8374cd`, `after` is branch `memfix-e2` on top of it. Every checkpoint is read from `~/models` on NVMe. These are memory
reads, not timings: each ran under the exclusive timing lock, load average 0.7 to 4.9.

## What changed

The resident build used to upload each weight as its own device allocation (`wpk`, `ws`, `ws16`, `nBytes` per projection, plus every
norm and bias): 275 to 813 allocations a model. The driver rounds each allocation of 2 MiB or more up to 2 MiB and packs the smaller
ones into 2 MiB pages, and the plan priced that rounding exactly (`packedAllocSlack`, KV3-S1-residual). Now `uploadLayers`
sub-allocates every weight the plan's walk counts from one allocation sized by that walk (`weightArena`), each at 256-byte alignment,
cuMemAlloc's own. The expert stacks stay separate allocations on their own line. `packedAllocSlack` prices the arena's padding and its
one rounding, plus the expert buffers as before. The arena's weights lease on the `dense` line and its padding and rounding on the
`slack` line, where the plan prices them.

No guard threshold, margin, floor or default moved.

## Device bytes, before and after (MiB)

The card's drop over a build's allocations before its fit-check reading, less the 2 MiB a process's first allocation takes
(`seq-{before,after}-*.tsv.gz`, from `TestKV3R_allocSequence`; `seq_summary.py` computes it). The `before` column reproduces
yesterday's recorded card figures (`kv3-s1-2026-11/packing-model.md`) to the MiB on all eight models.

| model | before | after | saved | allocations before / after |
|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 4,444 | 4,026 | **418** | 572 / 11 |
| Gemma-3-4B | 2,686 | 2,366 | **320** | 726 / 11 |
| Gemma-4-E2B Q4_0 | 1,582 | 1,388 | **194** | 823 / 12 |
| Qwen2.5-VL-3B | 1,956 | 1,790 | **166** | 732 / 11 |
| Qwen2.5-Coder-0.5B | 464 | 326 | **138** | 492 / 11 |
| Qwen2.5-Coder-1.5B | 1,052 | 930 | **122** | 572 / 11 |
| Gemma-3-1B | 682 | 666 | 16 | 558 / 11 |
| Llama-3.2-1B | 778 | 778 | 0 | 284 / 11 |

The audit predicted 113 to 420 MiB on the six models it read; the savings are the slack `TestBuildScratchAccounting` read there (139,
125, 420, 320, 169, 196) less 1 to 3 MiB. Llama-3.2-1B saves nothing: at hidden 2,048 its buffers are whole quanta and eighths of one,
so they already packed with no waste. The device is lower or equal on every model, never higher.

`TestBuildScratchAccounting` (G-M1) on both builds (`scratch-accounting-before-after.txt`): device before the KV goes 466 → 328
(0.5B), 1,054 → 932 (1.5B), 4,446 → 4,028 (7B), 2,688 → 2,368 (Gemma-3-4B), 1,958 → 1,792 (VL-3B), 1,584 → 1,390 (E2B); the planned
slack goes from 125 to 420 MiB to 0 to 2 MiB; the residual stays 2 to 4 MiB. Gemma-3-4B, whose plan is limited by free VRAM at one slot,
plans **14,720 → 15,936** positions in that gate.

## The plan against the card

`TestPlanSlack_pricesThePacking` holds the plan's weight figure (dense bytes plus the planned slack) within 2 MiB of the card. Its card
table (`cardWeightMiB`) is now the drop across a build's dense-line allocations alone (the arena and the final norm), which the
contiguous upload makes readable: 324, 928, 774, 664, 1,788, 2,364, 4,024 and 1,386 MiB. The plan reads +0.00 to +0.01 MiB against each.
The old figure (whole pre-fit drop less 2 MiB) also counted the scratch allocated before the weights. On Llama-3.2-1B one of those
scratch buffers is exactly 2 MiB, which the old per-buffer packing happened to absorb; with the arena that read −3.99 against the
2 MiB bar, so the table moved to the figure the test names. The bar is unchanged.

## Goldens

- `backend/cuda/testdata/mm2b_realsize.json.gz` (63 groups): rewritten with `-planslack.update`. Compared with
  `kv3-s1-2026-11/kv3r-golden-compare.py` (`realsize-compare.txt`): 59 groups keep their outcome sequence, every threshold moving by
  the model's own slack change (7B −418 MiB, 1.5B −124, 0.5B −138). The four 7B groups at `ctx=32768` plan more context at nobara's two
  readings: **25,856 → 29,632** and 25,984 → 29,824 (23,488 → 27,328 with 256 MiB of extra).
- The two CUDA census goldens (`sb2_cuda_census`, `sb2_cuda_mem_census`) are unchanged: the tiny fixtures' weights are all under a
  quantum and filled one page before and after.

## Gates read on the card (after)

- `TestWeightArena_holdsEveryWeight` (new, in `build_scratch_accounting_test.go`): on 14 tiny fixtures covering dense, sandwich, DeltaNet, Mamba, MLA, MoE, Gemma 4 MoE,
  pairwise RoPE and m-RoPE, the arena is filled exactly (used = size) and no dense weight falls outside it.
- `TestBufferSub` (new, `internal/gpudev`): a sub-allocation's bounds, offsets and release.
- `TestResidentDenseBytes_matchesCUDADevice` (heavy, 7B included), `TestPlanSlack_matchesTheBuild`, `TestPlanSlack_oneCopy`,
  `TestPackedAllocSlack*`, `TestPagePacker`, both census tests and the plant: PASS.
- `TestKSH6C_marginPriced` and `TestKSH6C_marginPriced7B8B` (`ksh6c-after.txt`): PASS on all seven models; Gemma-3-4B now plans the
  16,384 ceiling there (15,040 on `main`).
- The CUDA parity cell (`go test -tags 'cuda goinfer_testhooks' ./backend/cuda/`, the whole package, as `cmd/gate`'s `cuda-parity`
  cell runs it, CGO_ENABLED=0): **PASS**, 460 tests passed, 0 failed, 299 skipped (heavy-gated or no asset), in 15 minutes
  (10:25 to 10:40 PST), `TestMemGate_cuda` among the passes.

Red first: `TestPackedAllocSlack_oneArena` on `main` reads 2,097,152 bytes of slack for four 1.5 MiB weights, against 0.
