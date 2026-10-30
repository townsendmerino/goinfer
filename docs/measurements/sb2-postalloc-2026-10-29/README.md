# SB2 memory reads, step 3, PR 1: the post-allocation reads in the seam (2026-10-29, nobara-sb01)

Gates are registered in `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, PR 1", at `785f9c4f3`, before the
code. The forced figures and plant passes were recorded at `9a343e481`, before the forced pass. Francis chose option (a)
(relayed 18:22). Box: nobara, RTX 2070 SUPER, driver 595.91.07. Every device pass held the timing lock shared and started with
only the desktop compositor on the card (`*-card.txt`).

## What changed

The build's post-allocation decisions each go through a function the build and the memory plan share:

| site | function | its inputs, computed once for both |
|---|---|---|
| recurrent checkpoint | `ckptDeclineFor` | `recurrentLiveBytes` (from `winFloats`, `ssmFloats`, `stateElems`), `ckptReserve` |
| windowed-KV grant (before the expert cache, and last) | `kvGrantFor` over `kvWindowGrant` | per-position cost, reserve, maximum |
| expert-cache slots | `slotsDecision` over `capSlots` | `expertSlotGeom` / `planSlotGeomFor` (from `perExpertWords`, `cacheWQ`'s arithmetic) |
| prefill width | `prefillWidthChoice` over `prefillWidthFor` | pass prices on the layer geometry |

`planMemory` continues past the KV fit through them, each over its own reading (`memReadings.Ckpt`, `GrantPre`, `Slots`,
`Grant`, `Prefill`). `slotsFreeVRAM` is `allocSlots`' read as a variable, as `ckptFreeVRAM` and `kvGrantFreeVRAM` already were.
`memPlanPlantForTest` plants a defect on the plan's side by site.

## G-PA1 (no device)

- Table tests: `TestCkptDeclineFor`, `TestSlotsDecision`, plus step 1's.
- `TestPlanMemory_postAllocation` reaches every site: qwen3next-tiny's checkpoint, glm-tiny's and gemma4-moe-kv-tiny32's
  expert cache, gemma2-tiny's grant, and llama-tiny's width.
- `TestPlanMemory_postAllocationPlants` shows each plant changes its site.
- Both censuses are unchanged (`gpa1.log`).

## G-PA2: the real build before and after

The probe `sb2_paprobe_before_test.go.txt` covers the census's 412 cells plus 92 variants: every fixture at int4 with
`MoECacheExperts`, and at int4 with `ResidentWindowedKV`. That is 504 cells. On top of step 1's decisions it records:
- the prefill width;
- the checkpoint bytes or reason;
- the windowed slack;
- the expert-cache slots.

A1 and A2 (`before-A1.txt`, `before-A2.txt`, at `785f9c4f3`) agree on every decision. The after runs agree with A1 to the byte,
in run 1 and in run 2.

## G-PA3: the plan agrees with the build

Run 1 (`run1/`, the probe before the classifier fix below):
- On real readings, all 504 cells agreed.
- With the plants on real readings, grant was red on 10 cells and prefill on 351.
- The forced pass found one real disagreement. On gemma4-moe-kv-tiny32 with the expert cache, the build refused the cache at
  its minimum and the plan let it through. The plan counted only `isMoE` layers as cached, but the build also caches Gemma 4's
  MoE block (`g4moe`).
- With the forced plants, the checkpoint was red on 67 cells. The slots plant's 13 reds were filed by the probe as "the build
  declined later". Its classifier had no branch for a real memory refusal that the plan passes, which is also how the
  disagreement above was hidden in the unplanted pass's count.

The fix (`00d3e04e3`, red first: `g4moe-red.log`): `planSlotGeomFor` counts `g4moe` layers. The probe classifies a real memory
refusal that the plan passes as a mismatch.

Run 2, on the fixed tree:

| pass | result |
|---|---|
| `after.txt` (real readings) | 504 agree: 361 built on every field and log line, 143 the same static decline |
| `after-plant.txt` (plants: grant, prefill) | red on 10 cells at kvSlack and 351 at prefillWidth; nothing else |
| `after-forced.txt` (every `cudaFreeVRAM` 300 MiB, slots 385 MiB) | 504 agree, including the same refusal at slots on 14 expert-cache cells |
| `after-forced-plant.txt` (plants: ckpt, slots) | red on 67 cells at ckptBytes and on those 14 at slots; nothing else |

Every site's plant is red on at least one cell.

**Decision (Francis, 2026-10-29 19:05, option (a)):** run 1 tripped the registered stopping rule, a disagreement outside a plant.
Run 2 stands as G-PA3's read. The defect was in the plan side's coverage and is pinned by the committed g4moe test. The real
build was byte-identical in both runs. Both runs stay in this record.

