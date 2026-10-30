# SB2-cov: the memory census reaches the raise; the slot cap cannot be reached on tiny fixtures (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2-cov" (12:34). CPU only (`CUDA_VISIBLE_DEVICES=-1`),
under the timing lock held shared, queued behind tack's timed blocks. Log: `gates.log` (the repeated decline lines left out).

| gate | read |
|---|---|
| G-DC7: built twice, equal; old lines unchanged | **pass**: 718 lines; all 712 previous lines byte-identical; 6 added, all on the `llama-tiny@window32768` variant |
| G-DC7: the raise | **pass**: both profiles' `int4/slots4` lines on the variant plan 16384 positions a slot and raise one conversation's cap to 32768, with 4 slots |
| G-DC7: a halfcache line | **fail: none can exist** (below) |
| G-DC8: the slots plant changes every halfcache line | **fail: nothing to change**, as above; the plant is not run on the census |
| G-DC5 and G-DC6, steps 2 and 3's plants | still pass |

**Why the slot cap cannot be reached on a tiny fixture.** `slotRequirement` charges each slot buffer a whole 2 MiB allocation
quantum, rounded up. Every tiny fixture's slot buffers are KiB, so one slot costs what all of them do. On glm-tiny, 2 layers × 4
buffers × 2 MiB is 16 MiB for any count from 1 to 8. A figure therefore either holds the whole cache or cannot hold its minimum,
and the build refuses at the slots. Measured by bisection: at 563,197,952 B glm-tiny's derived cache holds all 8 slots; one byte
less, it refuses at the slots with 0 fitting. No figure caps it partway. The cap is reached only by a fixture whose slot buffers
cross quanta, about 0.25 MiB a slot or more at 8 slots. `TestPlanMemory_postAllocationPlants` covers the cap on a tight reading. The
census code that searched for such a figure was taken out, since it could never produce a line.
