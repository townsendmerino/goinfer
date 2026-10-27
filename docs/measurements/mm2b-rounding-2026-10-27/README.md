# MM2b: CUDA's 2 MiB allocation rounding in the plan — the real-size diff (2026-10-27, nobara)

Gate G-MM2b-1 of `docs/tasks/task-subsystems-2026-10.md` ("MM2b's gates"). Instrument: the `cuda` module's
`TestPlanSlack_realSizeDiff`, on nobara (RTX 2070 SUPER, driver 595.91.07, kwin_wayland's context present, timing lock
held). Each model was loaded at int4 from `~/models` on local NVMe, with `-planslack.out` set to this directory. The test
computes D3's `Plan("cuda")` cells (contexts 4,096, 16,384 and 32,768; pinned or not; KV f32, f16 or i8; extra 0 or 256
MiB) with the rounding priced (`after.txt.gz`) and forced to 0 (`before.txt.gz`). The readings are nobara's two cuda probe
figures (7,630,553,088 bytes, and the same plus 17 MiB without the desktop) and each side's change points.

| model | rounding (the build's own `packedAllocSlack`, equal to the plan's to the byte) |
|---|---|
| Qwen2.5-Coder 0.5B | 145,276,928 bytes (138.5 MB) |
| Qwen2.5-Coder 1.5B | 128,122,880 bytes (122.2 MB) |
| Qwen2.5 7B | 425,525,248 bytes (405.8 MB) |

**Result.**
- All 63 groups change, because every plan's need now carries the rounding.
- At nobara's two real readings, no cell moves to a decline. 0.5B and 1.5B are unchanged there. The 7B's unpinned
  32,768-context cells (KV f32, f16 and i8, and f32 with 256 MiB extra) shrink from 29,696 to 25,984 positions at 7.63 GB
  free (−12.5%). With 256 MiB extra, they shrink from 27,328 to 23,616.
- 462 cells move to a decline (154 per model). Every one is at a change-point reading inside the band, one rounding wide,
  where the old plan fitted with less room than the build reserves. The build would have refused those loads, or shrunk them
  past what the plan promised. The bands, per group:

| model | cell | free range now declining (GB) | rows | was |
|---|---|---|---:|---|
| coder-0.5b | ctx=4096 pinned=false kv=f32 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=false kv=f32 extra=256MiB | 0.660–0.795 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=false kv=f16 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=false kv=i8 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=true kv=f32 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=true kv=f16 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=4096 pinned=true kv=i8 extra=0 | 0.410–0.545 | 2 | resident@4096 |
| coder-0.5b | ctx=16384 pinned=false kv=f32 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=16384 pinned=false kv=f32 extra=256MiB | 0.660–0.795 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=16384 pinned=false kv=f16 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=16384 pinned=false kv=i8 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=16384 pinned=true kv=f32 extra=0 | 0.691–0.827 | 2 | resident@16384 |
| coder-0.5b | ctx=16384 pinned=true kv=f16 extra=0 | 0.691–0.827 | 2 | resident@16384 |
| coder-0.5b | ctx=16384 pinned=true kv=i8 extra=0 | 0.691–0.827 | 2 | resident@16384 |
| coder-0.5b | ctx=32768 pinned=false kv=f32 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=32768 pinned=false kv=f32 extra=256MiB | 0.660–0.795 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=32768 pinned=false kv=f16 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=32768 pinned=false kv=i8 extra=0 | 0.410–0.545 | 16 | resident@4096…resident@9984 |
| coder-0.5b | ctx=32768 pinned=true kv=f32 extra=0 | 1.066–1.202 | 2 | resident@32768 |
| coder-0.5b | ctx=32768 pinned=true kv=f16 extra=0 | 1.066–1.202 | 2 | resident@32768 |
| coder-0.5b | ctx=32768 pinned=true kv=i8 extra=0 | 1.066–1.202 | 2 | resident@32768 |
| coder-1.5b | ctx=4096 pinned=false kv=f32 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=false kv=f32 extra=256MiB | 1.376–1.496 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=false kv=f16 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=false kv=i8 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=true kv=f32 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=true kv=f16 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=4096 pinned=true kv=i8 extra=0 | 1.126–1.246 | 2 | resident@4096 |
| coder-1.5b | ctx=16384 pinned=false kv=f32 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=16384 pinned=false kv=f32 extra=256MiB | 1.376–1.496 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=16384 pinned=false kv=f16 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=16384 pinned=false kv=i8 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=16384 pinned=true kv=f32 extra=0 | 1.783–1.902 | 2 | resident@16384 |
| coder-1.5b | ctx=16384 pinned=true kv=f16 extra=0 | 1.783–1.902 | 2 | resident@16384 |
| coder-1.5b | ctx=16384 pinned=true kv=i8 extra=0 | 1.783–1.902 | 2 | resident@16384 |
| coder-1.5b | ctx=32768 pinned=false kv=f32 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=32768 pinned=false kv=f32 extra=256MiB | 1.376–1.496 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=32768 pinned=false kv=f16 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=32768 pinned=false kv=i8 extra=0 | 1.126–1.246 | 16 | resident@4096…resident@6272 |
| coder-1.5b | ctx=32768 pinned=true kv=f32 extra=0 | 2.658–2.777 | 2 | resident@32768 |
| coder-1.5b | ctx=32768 pinned=true kv=f16 extra=0 | 2.658–2.777 | 2 | resident@32768 |
| coder-1.5b | ctx=32768 pinned=true kv=i8 extra=0 | 2.658–2.777 | 2 | resident@32768 |
| 7b | ctx=4096 pinned=false kv=f32 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=false kv=f32 extra=256MiB | 4.621–5.017 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=false kv=f16 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=false kv=i8 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=true kv=f32 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=true kv=f16 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=4096 pinned=true kv=i8 extra=0 | 4.371–4.767 | 2 | resident@4096 |
| 7b | ctx=16384 pinned=false kv=f32 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=16384 pinned=false kv=f32 extra=256MiB | 4.621–5.017 | 16 | resident@4096…resident@7744 |
| 7b | ctx=16384 pinned=false kv=f16 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=16384 pinned=false kv=i8 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=16384 pinned=true kv=f32 extra=0 | 5.683–6.080 | 2 | resident@16384 |
| 7b | ctx=16384 pinned=true kv=f16 extra=0 | 5.683–6.080 | 2 | resident@16384 |
| 7b | ctx=16384 pinned=true kv=i8 extra=0 | 5.683–6.080 | 2 | resident@16384 |
| 7b | ctx=32768 pinned=false kv=f32 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=32768 pinned=false kv=f32 extra=256MiB | 4.621–5.017 | 16 | resident@4096…resident@7744 |
| 7b | ctx=32768 pinned=false kv=f16 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=32768 pinned=false kv=i8 extra=0 | 4.371–4.767 | 16 | resident@4096…resident@7744 |
| 7b | ctx=32768 pinned=true kv=f32 extra=0 | 7.433–7.830 | 2 | resident@32768 |
| 7b | ctx=32768 pinned=true kv=f16 extra=0 | 7.433–7.830 | 2 | resident@32768 |
| 7b | ctx=32768 pinned=true kv=i8 extra=0 | 7.433–7.830 | 2 | resident@32768 |

`summary.txt.gz` lists every one of the 462 rows with its reason line.
