# The swap-guard trip in tack's C13 E0 (Qwen3-Coder-30B-A3B), re-read by day (memory audit, 2026-11-18)

One box: nobara, 16 cores, 62 GiB RAM, 77 GiB swap, swappiness 10, RTX 2070 SUPER 8 GB, driver 595.112.07, desktop session up. The
server is the E0 binary itself, `~/tack-runs/bin/goinfer-serve-cuda-9de08125c` (sha256 identical to
`~/goinfer-bench/qwen3coder/goinfer-serve`), run through a wrapper that keeps its log. Its flags are E0's (`-backend cuda
-moe-cache-experts -thinking off -ctx 16384`). The checkpoints are on `~/models` (NVMe); both load through their CUDA `.giw` sidecars.
Everything ran under the exclusive timing lock, 16:08 to 16:31 PST. A 1 Hz sampler read the server's process tree (VmSwap, VmRSS,
RssAnon, RssShmem, RssFile), system swap, MemAvailable, and `user-1000.slice`'s memory.current, memory.high, its `high` event count,
its swap and its anon/file split (`samples*.tsv.gz`). A memory read, not a timing.

## What E0 recorded (nobara night job `tack-c13-qwen3coder-e0`, 15:25 to 15:31 PST)

The fit, check and smoke phases passed, then the 4,096-token cells. Cold 8,192-token TTFT read 31,978, 34,478 and 36,169 ms, rising.
At 15:31:37, after the third, every request returned `503 halted: swap guard tripped`. The process's RSS peak was 35.95 GB. System
swap was 1,523 MB before and 1,477 MB after, and the system source trips only at +4,096 MB, so **the source that tripped was the
server's own VmSwap past +512 MB over its baseline of 0**, by elimination. E0 keeps no serve log, so the guard's own line is not
available.

## Re-reads

| run | what | server VmSwap peak | server RssAnon peak | slice peak / its `memory.high` | slice `high` events | result |
|---|---|---|---|---|---|---|
| B | the 30B, E0's command, three cold ~8.1k prompts (31.6-31.8 s each) | 0 | 2.42 GiB | 51.10 / 52.00 GiB | +0 | no trip |
| C | the 35B (Qwen3.6-35B-A3B Q4_K_M), the same flags and prompts (61.0-62.4 s) | 0 | 1.57 GiB | **52.00 / 52.00** | **+35,859** | no trip |
| A | tack's E0 unchanged (tack 08c7d30), end to end, 12 min 31 s | 0 | 0.17 GiB | **52.00 / 52.00** at 16:17:28 | +604 | **PASS**, every TTFT cell incl. 15,360 |
| D | the 30B as B, plus a 16 GiB anonymous hog in the same slice | 0 | 2.44 GiB | 51.65 / 52.00 | +1,051 | no trip |

A's cold 8,192 cells read 31,602, 32,694 and 32,820 ms (flat), its RSS peak 34.49 GB. **The trip did not reproduce** in four runs,
including tack's exact sequence.

## What the box shows

- **The night runner's jobs run under a 52 GiB soft memory limit, not the box's 62.** `goinfer-night.service` is a user service inside
  `user-1000.slice`, and that slice carries `MemoryHigh=52G` from `/etc/systemd/system/user-1000.slice.d/50-oom-protect.conf`, a drop-in
  added on Jul 31 to keep systemd-oomd off the desktop and editor. Every process francis runs shares it: the desktop, every Claude
  session, the night runner and its servers, and the page cache they charge. At 16:03 the slice had already hit it 294,050 times
  (peak 55.96 GB). C and A drove it to 52.00 GiB, and the kernel then reclaimed inside the slice while MemAvailable stayed 40-59 GiB.
- **A served MoE load holds most of its own footprint out of reclaim's reach.** The 30B's expert copy (RssShmem 14.8-15.3 GiB) is
  pinned through `cuMemHostRegister`/`cuMemAllocHost`. It is charged to the slice as shmem, so the slice's `unevictable` reads 32 KB,
  yet reclaim cannot take it. The mapped `.giw` weights (RssFile about 16 GiB) are hot on every token. What reclaim can take is the
  slice's other page cache and its anonymous memory, the server's 0.2-2.4 GiB of heap included.
- **In all four re-reads, reclaim took page cache, never the server's heap.** Even D's 16 GiB of competing anonymous memory was made
  room for by dropping about 10 GiB of cache. The server's pages are swapped only when the slice is at its limit **and** short of
  droppable cache, which needs other unreclaimable or anonymous memory resident in the slice. Nothing records what the slice held at
  15:31: night.py's `memory.jsonl` has only the job's start (60,797 MB available), and the journal has nothing in that window.
- **E0's prefill is consistent with that pressure and the re-reads' is not.** E0's cold 8,192 cells rose by about 2 s a cell (32.0,
  34.5, 36.2 s) with a 1.46 GB higher RSS peak. Weight pages reclaimed under the slice's limit and faulted back in would look like
  that; A's cells were flat (31.6, 32.7, 32.8 s).
- **Ruled out as the cause on their own:** swappiness (10, which favours dropping cache), an mlock or pin failure (RssShmem steady, the
  build pinned), and a transient anonymous peak during expert-cache refill (RssAnon never above 2.44 GiB in any run, with no spike).
- **The 35B under the same prompts did not trip either**, although it drove the slice to its limit hardest (35,859 high events in one
  cell).
- **After a trip the server stays halted for the rest of the run.** The guard refuses requests until the source is back within its
  threshold, and swapped pages stay in swap until touched (`internal/app/serveapp/swapguard.go:server.rebaselineSwapGuard`), so one
  transient swap-out voids every later cell of an E0.

## Finding

Not shown to be a goinfer bug: the guard read the server's own swap, as it is built to. The one condition under which this server
swaps with about 59 GiB available is the night runner's environment: its jobs share `user-1000.slice`'s 52 GiB `MemoryHigh` with
everything else francis runs, and a served MoE load keeps about 31 GiB of its own footprint out of reach (pinned or hot) inside that
limit. The trip itself did not reproduce, so what else filled the slice at 15:31 is not known.

**What to change (not changed here; Francis's call):**
1. **Recommended:** remove `MemoryHigh=52G` from `50-oom-protect.conf` and keep its `ManagedOOMMemoryPressure*` lines, which are the
   oomd protection the drop-in was written for. Night jobs, and the sessions beside them, then reclaim against the box's 62 GiB. Needs
   root.
2. Or give `goinfer-night.service` a `MemoryLow=` near a served load's need (about 36G for this model), so reclaim under the slice's
   limit takes from the interactive sessions first. User-level, but it squeezes the desktop and the editor when both are heavy.
3. Either way, have night jobs keep the server log and a 1 Hz VmSwap / slice sampler (this record's sampler did it), so the next trip
   names its source and what filled the slice.

The sampler, wrapper, cold-cell driver and memory hog are in `~/goinfer-bench/swaptrip/` on nobara.
