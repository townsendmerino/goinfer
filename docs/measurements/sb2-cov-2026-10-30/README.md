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

## The slot cap, reached on a wide-expert variant (2026-10-31, nobara-sb01)

Gates registered before the code: task doc, "SB2-cov, the slot cap" (08:11 PDT). CPU only (`CUDA_VISIBLE_DEVICES=-1`), under the
timing lock held shared, run once nobara's lock was free (12:3x PDT).
- **The variant** is `testdata/glm-tiny@experts16384` (`expertVariantFixture`), written at test time:
  - glm-tiny with 16384-wide experts and one MoE layer;
  - at int4 a slot costs a 1 MiB gate-up buffer and a 512 KiB down buffer, plus scales;
  - the requirement steps at 3 and 5 of 8 slots.
- **The `@halfcache` lines** (`memCensusHalfCacheLine`): the largest figure, to the byte, at which the derived cache holds at most
  half its profile line's slots.

| gate | read |
|---|---|
| red first (`slotcap-red-2026-10-31.log.gz`, the variant left out) | **red as registered**: G-DC7's "no @halfcache line holds fewer slots" and the slots plant's "no line makes a grant there". The golden was otherwise unchanged: no committed fixture yields a tight line. |
| G-DC7: built twice and equal; old lines unchanged | **pass**: 718 → 728 lines; all 718 byte-identical; 10 added, all on the variant |
| G-DC7: a @halfcache line | **pass**: 4 lines (expert-cache and windowed+expert-cache, on both nobara profiles) at 561,789,439 B (tight) hold 4 slots against their profile lines' 8 |
| G-DC8: the slots plant (`slotcap-green-2026-10-31.log.gz`) | **pass**: it changes exactly the 4 @halfcache lines and no other |
| G-DC1 to G-DC6, the admission census | still pass |

- **The `~` marks.** Each tight line's `cacheSlots=4` carries `~`. A figure tight to the byte sits inside the 64 MiB bound by
  construction, so the bound cannot settle it. That is expected of a synthetic tight figure, and the line says so.
- **The other five profiles** carry no moe_route pool, so a MoE line is not censused on them, which is the existing rule.
- **Cost:** `TestBackendMemoryCensus_cuda` went from 5.6 s (red, without the variant) to 6.8 s, and the census's gates together
  run in 28 s. That is well under the registered 60 s.
