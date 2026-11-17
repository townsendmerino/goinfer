# Gemma-3-4B's planned-context residual (2026-11-16, found while taking KV3-S1's census refresh; a 16-core amd64 box, one 8 GB RTX 2070 SUPER)

**Resolved 2026-11-17 (`packing-model.md`):** the plan prices the term now. Two statements below are corrected there: the 1.5B's difference was not "frees between allocations" (no build frees anything before the fit check) but a small buffer taking an older page's room, which a bump allocator misses, and the pricing needed no per-family order model (one rule, best fit over partly used pages). Gemma-3-4B's served context is now 15,040, not the 15,296 option 1 said it would keep.

**What happened.** KV3-S1 removes one `M x inter` float buffer from a prefill pass, so `ctxMarginFor` (the build's context margin) falls by that buffer's charged bytes. `TestKSH6C_marginPriced` (heavy: `GOINFER_HEAVY_TESTS=1 KSH6C_MARGIN=1`, not run by CI) then fails on **Gemma-3-4B only**: the plan goes 15,296 -> 15,360 positions (margin 330.5 -> 310.5 MiB), `checkKVFits` trims it back to 15,296 at its own free-VRAM reading, and the gate's "the build kept the planned context" goes red.
`ksh6c-mine.txt` and `ksh6c-main.txt` hold the gate's lines on the change and on main. Every other model keeps its context (the 0.5B, 1.5B, Qwen2.5-VL-3B and Gemma-4-E2B at 16,384; the 7B passes). The census golden's 764 lines move only in `margin=`, and no fit decision in it changes.

**The mechanism, measured** (`g3-probe-gemma3.txt`, the plan's and the build's readings on the change):

| | |
|---|---|
| plan's free-VRAM reading | 7,177.19 MiB |
| the build's `checkKVFits` reading (after the weights and kernel modules) | 4,489.19 MiB |
| the margin | 310.52 MiB |
| KV charged at 15,296 / 15,360 positions | 4,080 / **4,216 MiB** |
| room beside the margin at the `kv` reading | 4,178.7 MiB |

The card's charge for the KV pools is a **staircase**: each pool buffer crosses a 2 MiB quantum at a block count (240 -> 241 blocks, the null block included), and Gemma-3-4B has 68 pool buffers, so the charge steps 136 MiB there. The plan prices KV linearly plus `kvRoundingBound` (the worst case, 138 MiB), which is sound while the free VRAM it reads is the free VRAM the build sees.
It is not: **the build's weight uploads take 2,686.00 MiB of the card; the plan prices 2,621.16 (the dense figure plus its packed-weight rounding): 64.84 MiB under (253 positions at this model's 273 KB a position, the "Plan's dense residual" the gate already reports).** On main that gap hides inside the rounding bound's unused part (at 15,296 the actual rounding is 0.2 of the 138 MiB). At 15,360 the actual rounding is 119 MiB, the bound's slack is 19 MiB, the residual is 65: the plan overshoots by 37 MiB and the build trims. Main's pass on Gemma-3-4B was the staircase's alignment, not margin.

**The unpriced term.** `chargedBytes` and `allocRoundSlack` charge an allocation under the 2 MiB quantum at its own size ("the driver packs it into a shared page"). The driver does pack them, but into 2 MiB pages in allocation order, opening a new page when the next buffer does not fit the remainder. Gemma-3-4B's weight upload makes **555 allocations under 2 MiB (279 MiB of them)**, and their packing wastes the pages' remainders.
Simulating a bump allocator over the build's own allocation sequence reproduces the card's drop **exactly: 2,686.00 MiB observed, 2,686.00 simulated** (plan-style charge 2,621.16). On Qwen2.5-VL-3B the simulation is within 2 MiB (observed 1,956, simulated 1,958, plan-style 1,923.91: 34 MiB under). **On Qwen2.5-Coder-1.5B it is not** (observed 1,052.0, simulated 1,120.0, plan-style 1,049.5), so the packing is not one rule across builds (frees between allocations return pages).

**Why it is not priced here.** The exact figure needs the build's allocation order (and its frees), which the plan does not have: `packedAllocSlack` walks the packed weights in struct order and charges only the per-buffer rounding. A model of the order is per-family and the 1.5B shows it is not one bump allocator. The robust bound, the sum of the sub-quantum buffers' sizes, is 279 MiB here (4x the residual) and would shrink every model's context to cover it.
No guard threshold or default is touched by any of this; the choice is how the plan should price, and the gate's assertion is not loosened.

**Options for Francis.**
1. Accept: KV3-S1 merges with the refreshed golden; Gemma-3-4B's plan is one block (64 positions) optimistic and the build's trim absorbs it, so its served context on this card is 15,296 either way; the gate's equality assertion stays red for that one model until the plan prices the term. (Nothing in CI reads that gate.)
2. Price the term: a pre-pool allocation model (order and frees) in the plan, with the 1.5B as the case that must pass; a design of its own, not an hour's fix.
3. Hold KV3-S1's margin change, keeping the third buffer priced: not a real option (the buffer is gone), listed to be refused.
