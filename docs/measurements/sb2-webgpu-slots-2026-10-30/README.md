# SB2 memory reads, step 4: WebGPU's slot probe (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 4" (commit 50345d655).

## Tier 0 (`tier0.log`, CPU only, no device: `CUDA_VISIBLE_DEVICES=-1`, `VK_ICD_FILENAMES=/dev/null`)

- **G-WS1: passes.**
  - `TestSlotsGranted`: `slotsGranted` equals `darwinKVSlots` on darwin over a grid of slot requests, RAM, availability, weights,
    host copies and slot sizes, and grants every slot on a discrete GPU with no figure. The headroom rule holds at its edge, a
    byte short of one slot plus 384 MiB. `slotsBeforeContextOn` leaves a discrete context alone and prices darwin's as
    `ctxWhereSlotsFit` over `darwinKVSlots`. After Gemini's review of #504 (amendment 08:57) the table also covers a pinned context
    (never shrunk), a context at the floor, unreadable and negative RAM, a non-zero `avail0`, and a context that shrinks, under tight
    RAM and under tight available memory.
  - Layer B-webgpu's census equals its golden unchanged, and its plant still turns exactly its 11 cells red.
  - The device slot tests run in G-WS3's window.
- **G-WS2: passes.**
  - 955 lines: the 191 cells layer B-webgpu passes × 5 profiles, 4 slots asked for. Built twice and equal.
  - 745 lines grant more than one slot; every discrete line grants 4 of 4. No line carries a `~`: the tiny fixtures' slots are KiB
    against GB.
  - The headroom plant turns all 298 discrete multi-slot lines to one slot and changes no unified line.
- `kvprec_decline_test.go`'s source scan now follows the `-ctx` request into `webgpuCtxBeforeSlots`, where it moved.
- **Goldens:**
  - `gpu/testdata/sb2_webgpu_slot_census.amd64.txt.gz` was written on nobara.
  - `gpu/testdata/sb2_webgpu_slot_census.arm64.txt.gz` was taken from CI's gpu-darwin runner (run 41081553865, job 124152637399, head
    f5180e646): the 955 lines the test printed, prefixed `SB2WEBGPUSLOTCENSUS`, when the golden was missing. They were written
    with `sb2census.Write`'s format, as layer B-webgpu's arm64 golden was.
  - The census reads no device and no reading of the machine it runs on (the profiles are data, the host figure fixed), so the
    runner's lines are the census.
  - The two goldens differ in the weight figures on 665 lines, as layer B-webgpu's do.

## G-WS3, the Vulkan card check (pass 1, 2026-10-30 10:52:37–10:53:24 PDT; `pass1-*`, `run-ws3.sh pass1`)

Driver 595.112.07, kernel 7.3.0-202.nobara.fc44. The card at each step held only the desktop shell's clients (kwin_wayland's
16 MiB CUDA context, plasma-keyboard, Xwayland, plasmashell, xwaylandvideobridge: 528 MiB used); the void check passed before
every measurement. Timing lock held exclusively. Host: about 61.7 GB available, load average 3.5 (`pass1-*-card.txt`).

| | qwen2.5-coder-1.5b q4_k_m | qwen2.5-7b q4_k_m |
|---|---|---|
| C, the cuda probe | 7,499,874,304 | 7,499,415,552 |
| B, the build's live device bytes (ctx 16384, one slot) | 1,911,070,576 | 6,096,462,192 |
| D, the census's deduction | 2,144,472,064 | 6,640,734,208 |
| A, what Vulkan then allocates (to 16 MiB) | 5,435,817,984 | 822,083,584 |
| **(C − D) − A, the census figure's error (graded)** | **−76.7 MiB** | **+34.9 MiB** |
| (C − B) − A | +145.9 MiB | +554.0 MiB |
| B − D | −222.6 MiB | −519.1 MiB |
| nvidia-smi's rise over the load, less B | 304 MiB | 677 MiB |
| realClamp slots granted / the plan fed C − D | 4 / 4 | 1 / 1 |

**Read: ambiguous** (1.5B at 76.7 MiB, between 64 and 128; the safe direction, the census thinking less is free than Vulkan
gives). No second pass: 76.7 is not within 8 MiB of 64 or 128.

What the numbers say:
- **The CUDA figure does not stand for what Vulkan leaves after a load, by itself.** Vulkan's device use runs 304 MiB (1.5B)
  and 677 MiB (7B) over the build's own ledger (B): nvidia-smi's rise over the load says the same as (C − B) − A. That overhead
  grows with the model (about 16% and 12% of B), so it is not one constant a profile field could carry.
- **The census's figure lands within 77 MiB only because two terms partly cancel.** The planner's deduction D over-prices the
  build's live bytes by 223 and 519 MiB, about as much as Vulkan's unledgered overhead. This is the same shape as G-DM2's 26B (a
  slack over-pricing against unledgered overhead), and no more robust.
- The realClamp grants match the plan on both models (4 of 4 and 1 of 1). Neither is near a step.

## The device slot tests: realClamp's 1.5B is not bit-identical, on `main` too

`pass1-slots.log`: every `TestWebGPUKVSlots_*` passes, except one.
- **The failure:** `TestWebGPUKVSlots_realClamp/qwen2.5-coder-1.5b` grants 4 of 4 slots, but slots 1 to 3 do not match a fresh
  one-slot load. Turn 0 is a run of token 15; turn 1 is a different text. Slot 0 matches.
- **Not this branch:** the same test fails the same way on `origin/main` (2ca872273, none of this branch's changes, a detached
  worktree, 10:54, the card idle: `pass1-realclamp-1.5b-on-main-2ca872273.log`).
- **Not previously seen:** on 2026-09-28 it passed (`../mc1-webgpu-nobara-2026-09-28/realclamp-1.5b.log`).
- **Open cause:** the driver moved to 595.112.07 this morning, and the code has moved since September. The tiny-fixture slot
  tests, including `interleavedMatchesAlone`, pass, so the defect needs a real checkpoint's size or shape.

## The decision on the read (the coordinator, received by 11:07, under Francis's 10:58 standing rule), option (a)

#504 merges with G-WS3 recorded as ambiguous. Each discrete line of the slot census now says `after-slot0=…(derived; G-WS3 band
-77/+35 MiB)`: 382 lines in each golden, with the arm64 golden changed by the same text substitution as the amd64 one the test
rewrote. A model of Vulkan's allocation overhead is the open row SB2-vk in the task doc, needed before any discrete line decides
anything. `go run ./cmd/lint`: VERDICT PASS, 7/7.
