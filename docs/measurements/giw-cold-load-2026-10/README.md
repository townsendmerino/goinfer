# Cold load of a `.giw`: what the header walk reads (2026-10-11, exploratory)

A by-product of the 30B self-check investigation (`../giw-v16-transcode-2026-10/README.md`, "The 30B's self-check,
investigated"): a bundle that is not in the page cache loaded in 20 s without the CRC and in 10 s with it. This
record is the follow-up. The task it scopes is `docs/tasks/task-giw-cold-load-2026-10.md`.

Everything here is **exploratory**: two repetitions a cell, one box, by day under the timing lock, no gate. nobara
(Ryzen 7 3700X, 62 GB, NVMe, **btrfs with `compress=zstd:1`**), CPU backend, weights format v16, goinfer `92391e9a`
with the experimental advice switch of `xp-advice.patch.txt` where a mode is named. Every load starts from an evicted
file (`posix_fadvise(DONTNEED)`, residency checked with `cachestat.py`) and skips the CRC (each bundle has its marker)
unless the mode says otherwise.

## 1. The header walk reads nearly the whole file

`run1.log`, part A: the load alone.

| bundle | default | `random` during the walk | `sequential` during the walk |
|---|---|---|---|
| Qwen3.6 35B-A3B, 20.0 GB, 30,720 expert matrices | 24.1, 34.1 s; 6,140-6,551 major faults; 93.5% of the file resident after | 9.64, 9.68 s; 84,260 faults; 2.1% | 23.3, 23.4 s; 3,180 faults; 88.5% |
| Mellum2, 7.1 GB, 5,376 expert matrices | 8.5, 16.4 s; 92.2% | 1.77, 1.79 s; 14,736 faults; 1.1% | 8.0 s; 83.4% |
| Qwen2.5 7B (dense), 4.8 GB | 1.21, 1.23 s; 195 faults; 21.7% | 0.05 s; 720 faults; 0.1% | 1.0-1.1 s; 20.0% |

- The loader maps the file and reads each matrix's header, scale count and nibble count through the mapping: two or
  three touched pages a matrix. By default each fault reads a window of megabytes around it (6,140 faults for
  18.7 GB is about 3 MB a fault), so on a file of thousands of small matrices the walk is a scattered read of all of
  it. With the read-around off (`MADV_RANDOM`) the same walk is 84,260 single-page faults and 0.4 GB.
- `MADV_SEQUENTIAL` does not help: the walk skips forward, it does not read forward.
- A dense bundle has a couple of hundred matrices; its walk reads a fifth of the file in about a second.

## 2. A faster load is not a faster first reply

`run1.log`, part B: the bundle evicted, `serve` started on the CPU, one greedy 24-token reply, then a second.

| 35B | server up | first reply done | second reply |
|---|---|---|---|
| default | 24.6, 24.6 s | 38.0, 37.9 s | 10.6 s |
| `random` during the walk, default after | 9.9, 9.9 s | 44.8, 44.8 s | 11.7 s |
| `sequential` | 23.6, 23.2 s | 37.9, 37.5 s | 10.6-10.8 s |

With the read-around off the server is up 15 s sooner and the first reply finishes 7 s later: the weights have to be
read either way, and faulting them in during the decode is slower than the walk's accidental read. Where the bundle
fits in memory, the do-nothing arm beats this one.

## 3. `MADV_WILLNEED` over the whole mapping does nothing

`run2.log`: `willneed` matched the default on every figure (35B load 24.4-25.2 s, the same fault counts and residency;
first reply at 38.1 s), and `willneed,random` matched `random`. One call is honoured for one read-ahead window, not
for 20 GB.

## 4. Reading the file front to back first is what pays, on an MoE

`run3.log`, the shipped loader with no experimental advice: `crc` forces the whole-file CRC
(`GOINFER_GIW_VERIFY=always`), which reads the mapping front to back before the walk; `cat` reads the file once with
`cat` before the server starts.

| bundle | | server up | first reply done | second reply |
|---|---|---|---|---|
| 35B MoE | default | 24.4, 24.8 s | **37.7, 38.1 s** | 10.6 s |
| | `crc` | 12.6, 12.0 s | **24.7, 23.9 s** | 10.4 s |
| | `cat` | 11.4, 11.3 s | **23.3, 23.3 s** | 10.4 s |
| Mellum MoE | default | 8.7, 8.9 s | **12.6, 12.8 s** | 3.6 s |
| | `crc` | 4.5, 4.7 s | **7.9, 8.1 s** | 3.3-3.4 s |
| | `cat` | 4.2, 4.3 s | **7.7, 7.7 s** | 3.4 s |
| 7B dense | default | 1.2 s | **8.8, 9.0 s** | 5.9-6.1 s |
| | `crc` | 2.8, 3.0 s | **9.1, 9.4 s** | 5.9 s |
| | `cat` | 2.8 s | **9.0, 9.0 s** | 5.9-6.0 s |

- On the two MoE bundles a front-to-back read first takes the cold time to a first reply to 0.62-0.65 of the default
  (14 s off the 35B, 5 s off Mellum). The file is read at the disk's sequential rate instead of in scattered windows.
- On the dense 7B it is level or a few tenths of a second worse. The do-nothing arm wins there.
- The CRC costs about a second over the plain read on 20 GB.
- So a marked bundle, which skips the CRC to load faster, loads an evicted MoE file slower than an unmarked one.

## 5. After the build: one start of each build (exploratory)

The change of `docs/tasks/task-giw-cold-load-2026-10.md` as built (`2d4b9630`) against `main` before it (`3e0e084f`), the
35B evicted before each start, one start each, 08:51 under the timing lock:

| build | server up | first reply done | the load's read-first line |
|---|---|---|---|
| old | 25.8 s | 37.8 s | absent |
| new | 11.6 s | 22.5 s | present |

The replies are the same text. One run a side shows the change acting and nothing more; the graded cells are
`run-gcl-night-nobara.sh`, queued for the night.

**A run of mine that must not be read.** A first smoke of the new build ran at about 08:47 without taking the timing
lock, while another session's timed run (`mellum21-c5`) held it. It read 28.2 s to serving and 46.9 s to a first
reply, on a box doing two things at once. It is discarded here, and it may have disturbed the other session's run.

## What is not measured

- Any other filesystem. btrfs sets a large read-ahead window; a filesystem with a small one would make the default
  walk read less, and the trade would move.
- macOS.
- A bundle that does not fit in memory. Every cell here fits (62 GB). That is the case the `random` walk is for: it
  would save reading 18.7 GB that cannot stay.
- A file that is already in the page cache, where a front-to-back read is pure cost (the transcode record's reference
  times: 2.1 s with the CRC against 0.2 s without, on 17.5 GB).
- More than one reader at a time, which an NVMe disk would serve faster than one sequential pass.

## Files

- `cold-a.sh`, `cold-b.sh`, `cold-c.sh`: the load alone, the cold start to a reply by advice mode, and the same with
  the shipped loader. `run.sh`, `run2.sh`, `run3.sh`: the three runs; `run1.log`, `run2.log`, `run3.log`: their output.
- `gcl_night.py`, `run-gcl-night-nobara.sh`: the night driver and wrapper for the task's timed gates.
- `xp-advice.patch.txt`: the experimental loader's change (a scratch build, not committed). `cachestat.py`: how much
  of a file is in the page cache.
