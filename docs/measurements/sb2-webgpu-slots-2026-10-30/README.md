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
