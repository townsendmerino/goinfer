# The driver's page packing of sub-2 MiB allocations: a model for the plan (KV3-S1-residual design note, 2026-11-17)

<!-- Written before the plan's code changes, after the sequence reads. One box: nobara, 16 cores, one 8 GB RTX 2070 SUPER, driver 595.112.07, desktop session up. -->

Ledger row KV3-S1-residual (`docs/tasks/task-kv-architecture-2026-10.md`); the problem is `g3-residual.md`: the plan charges a buffer under 2 MiB at its own size, the card does not.

## What the 1.5B did differently

**Not frees.** `TestKV3R_allocSequence` (`backend/cuda/ksh6c_breakdown_test.go:TestKV3R_allocSequence`) records every device lease and release of a build and the card's free VRAM after each. **No build released anything** before the fit check (0 releases in all eight loads), so "frees between allocations return pages" was wrong. What differs is the placement of the *next* small buffer: a bump allocator (the earlier simulation) puts it in the newest page or opens one. The card puts it in **an older page that still has room**. In the 1.5B a 6.56 MiB buffer is followed by a 0.82 MiB one that costs no new page, three times in a row, and a bump simulation charges a page for each (68 MiB over by the end).

## The model (one rule, every family)

Sequence = the device allocations of the weight upload, in order. A buffer of 2 MiB or more takes its size rounded up to 2 MiB and leaves **no tail** a later buffer can use (a model that reuses tails is wrong by 44-138 MiB). A smaller buffer goes into the partly used page with the **least room that still holds it**, else opens a page. Cost = pages x 2 MiB. Replaying the recorded sequences against the card's reading after every allocation (`kv3r-packing-sim.py`, `kv3r-seq-*.tsv.gz`):

| rule | Gemma-3-4B | VL-3B | **1.5B** | 0.5B | Llama-3.2-1B | Gemma-3-1B | 7B | Gemma-4-E2B |
|---|---|---|---|---|---|---|---|---|
| bump (g3-residual's) | 0 | +2 | **+68** | +4 | +6 | +8 | 0 | +12 |
| first fit | 0 | 0 | 0 (2 mid-run) | 0 | 0 | 0 | 0 | 0 (2 mid-run) |
| **best fit** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| last fit | 0 | 0 | +40 | +4 | +6 | +8 | 0 | 0 |

Final error in MiB against the card; best fit is also exact at every one of the 4,700 allocation steps. It was chosen on the first three (Gemma-3-4B, VL-3B, 1.5B); the other five were read after and not used to choose it.

## What the plan can use

The plan has the buffers, not the setup job's order. Shuffling each model's recorded sequence inside windows of 6, 12 and 25 allocations moves the total by 0 MiB (2 MiB on the 1.5B, 2 on one Gemma-3-4B draw); only a whole-sequence shuffle moves it (E2B by up to 38). So `packedAllocSlack` walks in struct order (hostW's wpk, ws, ws16, nBytes, and the numeric vectors beside them) and feeds `pagePacker`.

## Predictions, and what the plan then reads (plan weights = dense + slack, MiB)

| | card | main | with the model |
|---|---|---|---|
| Gemma-3-4B | 2,686.00 | 2,620.67 (-65.33) | 2,684.01 (-1.99) |
| Qwen2.5-VL-3B | 1,956.00 | 1,922.89 (-33.11) | 1,956.01 |
| **Qwen2.5-Coder-1.5B** | 1,052.00 | 1,048.73 (-3.27) | 1,052.01 |
| Gemma-4-E2B | 1,582.00 | 1,527.03 (-54.97) | 1,582.01 |
| Qwen2.5-7B | 4,444.00 | 4,427.86 (-16.14) | 4,442.01 (-1.99) |
| Llama-3.2-1B | 778.00 | 774.75 (-3.25) | 776.01 (-1.99) |
| Qwen2.5-Coder-0.5B | 464.00 | 461.18 (-2.82) | 462.00 (-2.00) |
| Gemma-3-1B | 682.00 | 679.68 (-2.32) | 680.00 (-2.00) |

The card column is the drop over the weight upload less the 2 MiB the first allocation of a process takes. The residual 2 MiB on five models is the one page the sub-quantum scratch buffers (1,024 B and a few bytes) that the build allocates before the weights open: they are not weights, the plan does not hold them, and their page is the card's.

**Stopping rule, registered before the code:** the model passes if the plan's figure is within 2 MiB of the card on Gemma-3-4B, Qwen2.5-VL-3B and Qwen2.5-Coder-1.5B; it stands if it also does on the holdouts. **All eight pass.** `TestPlanSlack_pricesThePacking` holds the table; it was red on main on all eight (`kv3r-red-on-main.log`).

No guard threshold, margin constant, default or bypass is touched: the term is the plan's allocation slack (`packedAllocSlack`), the figure `fit`, the night queue and the build already share.

## Result (code in the same commit; one box, 2026-11-17)

**Gate.** `TestKSH6C_marginPriced` and `TestKSH6C_marginPriced7B8B` on the card (`GOINFER_HEAVY_TESTS=1 KSH6C_MARGIN=1`, exclusive timing lock, a pre-built test binary; lines in `kv3r-ksh6c-after.txt`): **PASS on all seven models, `gemma-3-4b-it` included.** The assertion is unchanged. The plan now prices Gemma-3-4B's weights at 2,684.01 MiB against the card's 2,686.00, and its planned context is built as planned.

| | planned on main (KV3-S1 merged) | planned now | built |
|---|---|---|---|
| Gemma-3-4B | 15,360, trimmed by the build to 15,296 | **15,040** | 15,040 |
| 0.5B, 1.5B, VL-3B, Gemma-4-E2B, Qwen2.5-7B, Qwen3-8B | 16,384 (the ceiling) | 16,384 | 16,384 |

**One visible effect.** Gemma-3-4B's served context on this card goes **15,296 -> 15,040** (256 positions, 1.7%): the plan now spends the 65 MiB it used to miss, and the reserve for the pools' rounding (`kvRoundingBound`, 138 MiB worst case, 0.2 MiB actual at 15,296) is the conservatism that remains, as it was designed. The gate's own allowance for that (`G3`: 256 below the build's exact figure, 584 allowed) holds. No threshold or margin constant moved; whether 256 positions of Gemma-3-4B's context is worth tightening the reserve is a separate decision for Francis.

**Goldens.** The plan's slack is in every priced line, so each moves; nothing but it and what is derived from it does (`kv3r-plan-cells-compare.txt`, `kv3r-census-compare.txt`):
- `testdata/mm2b_realsize.json.gz` (D3, 63 groups): the outcome sequence is unchanged in 59; each threshold moves by the model's own delta (7B +14.16 MiB, 1.5B +3.27, 0.5B +0.82). In four 7B groups at `ctx=32768` the context planned at nobara's reading goes 25,984 -> 25,856 (and 26,112 -> 25,984 at the second reading), the 7B's 14 MiB.
- `testdata/sb2_cuda_census.amd64.txt.gz` (421 lines): 283 move, in `plan slack=` only.
- `testdata/sb2_cuda_mem_census.amd64.txt.gz` (764 lines): every line moves in its derived readings (`kv@`, `prefill@` and the like) by the slack; no `ctxPlanned`, `kvSlots`, `ckpt`, `cacheSlots` or `prefill=` changes. The `tight` profile of `glm-tiny` is built from the plan's own figure, so its three readings move with it.

**Tests that priced the weights without the slack** now include it, since the tiny fixtures' slack went from 0 to one page: `TestResolveCtxCapFit_slotsShrinkTheContext` and `TestResolveCtxCapFit_agreesWithCheckKVFits` (their forcing term), and `TestBuildDemand_matchesDecoderDefault` (its claim that decoder's default 0 is CUDA's figure for buffers all under a quantum no longer holds: the figure is the shared pages; decoder's default of 0 is for a build with no CUDA planner linked). The floor case of the first moved its target 3000 -> 2400 positions: the build's room grew by one 2 MiB page, which put the old target within 0.3 MiB of a slot boundary.

**Not changed, noted:** a plan made without the CUDA planner linked (`decoder`'s default slack of 0, used by `fit` on a box with no CUDA) still omits the packing; it has no buffer layout to pack. The page left by the scratch buffers the build allocates before the weights (up to 2 MiB) is the plan's residual on five models.
