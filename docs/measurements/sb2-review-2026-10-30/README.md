# SB2 memory reads, step 3: the follow-up to the post-merge review of #495 and #502 (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, follow-up". CPU only, every test
run with `CUDA_VISIBLE_DEVICES=-1` (nobara-ci01 held the card). `red.log` is before the fixes, `green.log` after.

| | the defect | red in `red.log` | the fix |
|---|---|---|---|
| R1 | the derive took nothing off for the split-KV score scratch's reallocation at a raised cap (`kvAfterRaise`) | `TestDeriveMemory_raiseScoreScratch`: 2 slots, 10752 → 21504 at nH 28, prefill reading 2,990,080 B too high | `postAllocPlan` takes off `chargedBytes(nH·raised·4) − chargedBytes(nH·share·4)` after the raise |
| R2 | derive mode marked `weightsOpen`, `plan` and `weights` read on every load, and `slots` on a cache with no routed bytes | `TestDeriveMemory` (llama-tiny, glm-tiny with the cache), `TestPlanMemory_degenerateExpertCache` | each site is marked read where the build reads it (`weightsCheck`, the plan's `readFree`, `allocSlots`' guard) |
| R3 | the census marked every decision of a line whose −64 MiB derive refused, and never said a refusal was 64 MiB away; a slots refusal printed slack 0 | `TestNearMarks` with the refusal shapes the derive returns; `TestDerivedTail_nearRefusal` on a real llama-tiny derive bisected to the edge of the KV fit (tail `... prefill=512~`, no refusal) | a refusal at either end is its own mark (`refuses~@kv`, `decline@kv~`); decisions are compared only where both derives reached them; a slots refusal records its slack |
| R4 | G-DC2's "unchanged before the KV fit" half could not fail; `TestDeriveMemory`'s checkpoint check passed at none kept | (not a red: a vacuous check) | every census line is derived, and G-DC1 requires the derive to refuse where the plan does before the KV fit, exercised on all 558 lines by step 2's weights plant; the vacuous half is dropped from G-DC2; qwen3next-tiny must keep its checkpoint |

`TestPlanMemory_grantsAndWidth` now sweeps llama-tiny's prefill reading from 1 GiB down in 1 MiB steps and requires the 2048,
1024 and 512 rungs. It passed on the old code too. The old four-point ladder skipped 1024, and the sweep does not.

**The census golden did not move.** All 558 lines match `cuda/testdata/sb2_cuda_mem_census.amd64.txt.gz` as committed by #502,
and no line carries a mark. No census cell has more than one KV slot (R1), an expert cache (R2's slots, R3's slots refusal), or a
derive within 64 MiB of refusing (R3). The changes are reached only by the tests above.

**Checks:** gofmt clean. vet under no tag, `cuda` and `cuda goinfer_testhooks`. staticcheck under the same three: nothing in the
changed files; under `cuda` alone, the 20 pre-existing test-hook-only U1000s. The source scans over `cuda/backend.go` and cuda's
non-test files (`TestDeclineAdvice_wiredIntoBuildResident`, `TestPTX_matchesSourcesAndBindings`) pass.
