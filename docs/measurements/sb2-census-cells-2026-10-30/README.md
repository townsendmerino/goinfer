# SB2 memory reads, step 3 coverage: the memory census's own cells (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, coverage". The amendment adding
`int4/windowed+expert-cache` was made before G-DC6 ran. CPU only, `CUDA_VISIBLE_DEVICES=-1`. Log: `gates.log`.

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
The branch is not pushed for merge while the census test fails on them.
