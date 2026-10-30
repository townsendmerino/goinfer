# SB2 memory reads, step 3 coverage: the memory census's own cells (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, coverage". The amendment adding
`int4/windowed+expert-cache` was made before G-DC6 ran. CPU only, `CUDA_VISIBLE_DEVICES=-1`. Logs: `gates.log.gz`, the first run of G-DC5 and G-DC6; `gates-final.log.gz`, after option (i).

| gate | read |
|---|---|
| G-DC5: built twice, equal; base lines unchanged | **pass**: 712 lines; the 558 from layer B-cuda's cells are byte-identical to the previous golden |
| G-DC5: the grant site | **pass**: 40 lines (`int4/windowed` and `int4/windowed+expert-cache`), slack above the window's floor |
| G-DC5: the slot site | **pass**: 56 lines with `cacheSlots=` |
| G-DC5: the raise | **fail: 0 lines.** A raise needs more than one slot and a window above the per-slot context (4096). The only planned fixture with a window over 4096 is qwen3next-tiny (32768), and it has recurrent state, so it keeps one slot. Every other fixture's window is at most 4096. The 62 `int4/slots4` lines still census the 4-slot KV fit |
| G-DC6: `grant` plant | **pass**: 20 lines change, each reading the site; no base line |
| G-DC6: `grantPre` plant | **pass**: 20 lines change, each reading the site; no base line |
| G-DC6: `slots` plant | **fail: 0 lines.** The plant ignores the slot cap. On every fixture the cache already holds all its experts at the profiles' 7.6 GB, so ignoring the cap changes nothing. The slot decision is censused, but never at its cap |

Steps 2 and 3's plants (`TestBackendMemoryCensus_cuda_plant`, `_derivedPlant`) now compare layer B-cuda's lines only, over which they are
defined. A line of the census's own cells is kept only where it differs from its int4 line, and a plant that refuses both makes them
equal. Both still pass: 558 of 558.

Both failures are a limit of the fixtures, not of the code. The tiny fixtures have gigabytes to spare and short windows. Neither
clause can pass without a new input: a non-recurrent fixture with a window over 4096, and a figure tight enough to cap the cache.
The coordinator's call (08:45, option (i)): merge with both recorded. The test asserts what this fixture set reaches (the grant sites,
the slot site, the base lines unchanged, the grant plants) and logs the raise count. The raise and the slot cap stay covered by
`TestDeriveMemory_raiseScoreScratch` and `TestPlanMemory_postAllocationPlants`. Inputs that reach them are the open row SB2-cov. The
final run is `gates-final.log.gz`.

## The post-merge review of #505 (registered 2026-10-30 09:20 PDT; `review.log`, with the repeated "BuildResident declined" lines left out)

- **G-DC6** now compares the clean and planted censuses by line identity (fixture, cell, profile), and checks the converse: every
  line of the census's own cells that makes a grant at the planted site must change. The `grant` and `grantPre` plants each change
  exactly the 20 lines that make a grant there, and nothing else.
- **G-DC5** counts the grant (20 lines) and the grant before the cache (20) apart, and asserts each. It asserts the raise is 0, with
  a message pointing at SB2-cov, so a fixture that starts reaching it turns the test red.
- Steps 2 and 3's plants build layer B-cuda's cells only.
- **Against the stopping rule ("each gate runs once"):** G-DC5 was built twice. The first build was before the amendment that added
  `int4/windowed+expert-cache` (`gates.log.gz`), and the second after it, when G-DC6 also ran (`gates-final.log.gz` is the run
  after option (i)). G-DC6 ran once under the amended registration.
- #505's description said the registration times "had been written ahead of the clock". Those were times written and corrected
  within its own branch before it merged; nothing on `main` carried them.
- Both logs are gzipped: about 1 MB each, mostly one repeated decline line.
