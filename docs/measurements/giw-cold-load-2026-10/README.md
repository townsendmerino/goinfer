# Cold load of a `.giw`: what the header walk reads (2026-10-11, exploratory)

A by-product of the 30B self-check investigation (`../giw-v16-transcode-2026-10/README.md`, "The 30B's self-check,
investigated"): a bundle that is not in the page cache loaded in 20 s without the CRC and in 10 s with it. This
record is the follow-up. The task it scopes is `docs/completed/task-giw-cold-load-2026-10.md`.

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

The change of `docs/completed/task-giw-cold-load-2026-10.md` as built (`2d4b9630`) against `main` before it (`3e0e084f`), the
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

## 6. The Mac (step 0): no read-around to speak of, and a slow first reply for a different reason

`run-mac.log`, 2026-10-11 09:17, by day on the owner's word, under the timing lock with the box in use (load average
3.6). MacBook Pro M1 Pro, 16 GB, APFS, CPU backend, goinfer `360bc020` with the advice switch built for darwin. A file
is made cold with `msync(MS_INVALIDATE)` on a mapping of it (checked with `mincore`: 100% resident to 0%;
`maccache.py`). Two bundles: the Mellum MoE (7.06 GB, copied from nobara, byte count matched) and a dense 7B built
here (4.77 GB). The 35B is not run on this box. Exploratory, two repetitions.

**The load alone** (`mac-a.sh`):

| bundle | default | `random` | `sequential` | `willneed` |
|---|---|---|---|---|
| Mellum MoE | 1.45 s; 12,031 major faults; 5.6% of the file resident after | 1.17, 1.18 s; 12,031; 2.8% | 1.14, 1.19 s; 2.9% | 2.51, 2.52 s; 424,000; **100%** |
| 7B dense | 0.06, 0.07 s; 479 faults; 0.3% | 0.05 s; 0.2% | 0.05 s; 0.2% | 1.74, 1.76 s; 286,534; **100%** |

- **macOS does not read megabytes around a fault.** The default walk of the MoE bundle reads 5.6% of it in about a
  second and a half, where nobara's reads 92% in eight. The problem this task was opened for does not exist here, and
  turning the read-around off (L2) buys nothing.
- **`MADV_WILLNEED` over the whole mapping works here**, where on Linux it is honoured for one window: it reads all
  7 GB in 2.5 s.
- **The first default load of Mellum is not in the table.** It took 12.2 s and read the whole file: it ran the CRC.
  The bundle's marker did not survive the copy, because the Mac's `rsync` (openrsync, protocol 29) drops the
  nanoseconds of a modification time and the marker records them. The load rewrote the marker and every later load
  skipped the CRC.

**Cold start to a finished reply** (`mac-c.sh`):

| bundle | | server up | first reply done | second reply |
|---|---|---|---|---|
| Mellum MoE | default | 2.4, 1.7 s | **9.7, 8.4 s** | 2.4-2.7 s |
| | `random` during the walk | 1.3, 1.5 s | **7.9, 8.1 s** | 2.4 s |
| | `crc` | 11.2, 11.5 s | **13.4, 15.1 s** | 2.0-2.2 s |
| | `cat` first | 1.3 s | **3.7, 3.8 s** | 2.2 s |
| 7B dense | default | 0.2 s | **6.3, 5.6 s** | 2.2 s |
| | `random` during the walk | 0.2 s | **5.6, 6.1 s** | 2.5-2.7 s |
| | `crc` | 8.2, 9.1 s | **10.6, 11.6 s** | 2.3 s |
| | `cat` first | 0.9 s | **3.5, 3.5 s** | 2.2-2.4 s |

- **The first reply is slow on the Mac because the decode faults the weights in through the mapping**, a page at a
  time: the first reply takes 6-7 s where the second takes 2.2-2.7 s. Reading the file once with `read(2)` first
  takes the cold time to a first reply to 0.41 of the default on Mellum and 0.59 on the dense 7B. Unlike on Linux,
  the dense bundle gains too.
- **The CRC is the wrong way to read the file here.** It takes 8-11 s over 4.8-7 GB (it is bound by its own page
  faults and arithmetic, not by the disk), so forcing it makes the cold start worse, where on nobara it made it
  better.
- The `random` arm is level with the default (7.9-8.1 s against 8.4-9.7 s on Mellum, inside the spread on the 7B).
- Not measured here: the Metal backend, which is how most loads on this box run and which binds the mapping's pages
  into GPU buffers; a bundle that does not fit in 16 GB; a quiet box.

## 7. After the darwin build: a two-cell smoke (exploratory), and a cold start that was not cold

The darwin pre-read as built (`3533a383`) against `main` before it (`ac85083a`), Metal-module servers, one pair a
cell, 09:36 under the timing lock, by day (`gcl-mac-smoke.log`, the wrapper's smoke mode):

| cell | build | server up | first reply done | the bundle at serving | prefetched |
|---|---|---|---|---|---|
| Mellum MoE, CPU | old | 1.7 s | 9.7 s | 6% resident | |
| | new | 1.8 s | 8.9 s | 6% resident | **no** |
| 1.5B, Metal | old | 0.8 s | 1.6 s | 13% resident | |
| | new | 1.2 s | 1.6 s | 100% resident | yes |

- **The Mellum cell did not prefetch, correctly.** The bundle is 7.1 GB and the box had about 5 GB available by day;
  the rule reads first only what fits in three quarters of that. On a 16 GB Mac in use, the pre-read is for small
  bundles.
- **On the 1.5B on Metal the one pair is level:** the prefetch added 0.4 s to the load and the reply was 0.4 s
  shorter. One pair shows the change acting; the night's four pairs and the 7B decide the Metal cells.
- **A first smoke is discarded: its second start was not cold.** Straight after a Metal server exits, its pages are
  still wired for a moment and `msync(MS_INVALIDATE)` drops nothing: the bundle was 93% resident before the "cold"
  start, which then read 0.44 of the old build's time for no reason of its own. `gcl_night.py` now evicts, checks the
  residency, retries for up to ten seconds, and fails the start if the bundle is not cold.

## 8. The Linux gates, 2026-10-11: G-CL2 SHIP on both cells, G-CL3 and G-CL4 PASS

`run-gcl-night-nobara.sh` on nobara, 09:14 to 09:35 (20.8 min; the queue was started by day), old `3e0e084f` against
new `2d4b9630`, CPU backend, four pairs a cell in alternating order, the bundle evicted before every cold start.
Records `gcl-nobara.jsonl`, `gcl-nobara.log`. Every cell's replies are the same text in every start.

| gate | cell | new/old, first reply (pairs) | median | old to new | bar | verdict |
|---|---|---|---|---|---|---|
| G-CL2 | 35B MoE, cold | 0.597, 0.598, 0.597, 0.602 | **0.597** | 38.5 s to 23.0 s | ship at most 0.98 (expected 0.60-0.70) | **SHIP** |
| G-CL2 | Mellum MoE, cold | 0.616, 0.602, 0.610, 0.616 | **0.613** | 12.6 s to 7.7 s | the same | **SHIP** |
| G-CL3 | 7B dense, cold | 0.993, 0.998, 1.003, 1.008 | **1.001** | 8.96 s to 8.97 s | at most 1.02 | **PASS** |
| G-CL4 | 35B paged, in a 16 GB memory scope | 0.967, 0.978, 0.974, 0.979 | **0.976** | 41.3 s to 40.3 s | at most 1.00 | **PASS** |

- **G-CL2.** The new build announced its read-first in all four starts of both cells, and the bundle was 99-100%
  resident when the server came up, against 92-93% for the old build's walk. The server is up 13.8 s sooner on the
  35B and 4.6 s sooner on Mellum. The 35B's median is a hair under the expected range.
- **G-CL3, the warm starts** (the file read into the page cache first), time to serving, new minus old:

  | cell | pairs | median |
  |---|---|---|
  | 35B | +0.05, +0.05, -0.05, 0.00 s | +0.025 s |
  | Mellum | -0.05, 0.00, +0.05, 0.00 s | 0.00 s |
  | 7B | 0.00, 0.00, 0.00, -0.05 s | 0.00 s |

  All within the 50 ms bar, and **the driver polls every 50 ms, so the bar is at the instrument's resolution**: these
  rows show no slowdown of 100 ms or more, not one of 20. The warm first replies are level (0.991, 1.001, 0.999).
  Neither build read anything from the disk by the time it served.
- **G-CL4.** With the read-around off, the paged load had read **2.1%** of the file when it served (bar: at most a
  tenth) against 92.8% for the old build, and the server was up 15.3 s sooner (25.3 s to 10.0 s). The first reply
  finishes about a second earlier: inside the scope the weights cannot stay, so the old build's read of the whole file
  bought it nothing.
- **Noise, old against old:** Mellum's two pairs read 0.993 and 1.014. The 35B's read 0.925 and 1.008, and the first
  of those is not a clean pair (next item).
- **Three overlaps, all mine.** The session that queued this run also used the box while it ran, without looking at
  the timing lock.
  - 09:14:08 to 09:16:45: a 7 GB `rsync` of the Mellum bundle off nobara. It covers the first two starts of the 35B's
    old-against-old cell, which read 43.3 s and 40.0 s where the next two read 38.8 s and 38.5 s. That pair (0.925) is
    discarded. The graded 35B cell began at 09:17:56, after the copy ended.
  - About 09:27: a 22 MB upload and a test that writes and reads a 64 MB file, inside the 35B's warm cell.
  - About 09:35: a small upload and a 64 MB write, read and `sync`, inside the fourth pair of the does-not-fit cell
    (0.979, level with the other three).
  The graded cold cells of G-CL2 are clear of all three. The owner may still want the run repeated on a box left
  alone.

## 9. The Mac's gates, 2026-10-11: the CPU cells SHIP, the 7B on Metal reads 1.039 and the kill rule fires

`run-gcl-night-mac.sh` on the MacBook (16 GB), 09:53 to 09:58 (4.8 min; the queue was started by day, on the owner's
word, with his editor open), old `ac85083a` against new `3533a383`, Metal-module servers, four pairs a cell in
alternating order, the bundle evicted and checked cold before every cold start. Records `gcl-mac.jsonl`,
`gcl-mac.log`. Every cell's replies are the same text in every start, and the new build prefetched in every cold
start of every cell, so every registered cell ran.

**G-CLd2, the cold start**, time from process start to the first finished reply:

| cell | new/old (pairs) | median | old to new | expected | verdict |
|---|---|---|---|---|---|
| Mellum MoE, CPU | 0.610, 0.602, 0.608, 0.611 | **0.609** | 8.82 s to 5.37 s | 0.40-0.60 | **SHIP** |
| 7B dense, CPU | 0.723, 0.738, 0.746, 0.737 | **0.738** | 5.51 s to 4.07 s | 0.55-0.75 | **SHIP** |
| 7B, Metal | 1.037, 1.032, 1.041, 1.043 | **1.039** | 3.73 s to 3.87 s | none | **OFF** |
| 1.5B, Metal | 1.000, 0.988, 0.987, 0.981 | **0.987** | 1.59 s to 1.56 s | none | PARK |

- **The CPU cells ship.** On Mellum the server is up 1.5 s later (the prefetch) and the reply then takes 2.2 s where
  it took 7.1 s. Mellum's median is a hair above the expected range.
- **On Metal the prefetch costs what it saves, and a little more.** On the 7B the server is up 1.49 s later in every
  pair, and the reply after it is 1.36 s shorter (2.66 s to 1.30 s). The old build serves with 15% of the bundle in
  memory and Metal brings the rest in about as fast as the prefetch does. All four pairs are above 1.00, in both
  orders, within 0.011 of each other.
- **The kill rule fires, as registered:** a Metal cell is above 1.00, so the pre-read is not applied to a Metal load
  (next section's code). The 1.5B's park is moot with it.
- **Noise, old against old:** Mellum on the CPU read 0.895 and 0.996, the 7B on Metal 0.952 and 0.984. The low
  figure in each is the cell's first pair, whose first start is the slow one. The graded cells alternate the order
  and their pairs agree within 0.02.

**G-CLd3, the warm starts** (the file in the page cache), time to serving, new minus old:

| cell | pairs | median | first reply, new/old |
|---|---|---|---|
| Mellum MoE, CPU | +0.05, +0.06, 0.00, +0.05 s | **+0.05 s** | 1.022 |
| 7B dense, CPU | 0.00, 0.00, -0.01, +0.01 s | 0.00 s | 0.994 |
| 7B, Metal | -0.03, -0.03, -0.08, -0.02 s | -0.03 s | 0.984 |
| 1.5B, Metal | -0.17, -0.06, +0.04, +0.13 s | -0.01 s | 0.992 |

- **PASS on the registered measure, with Mellum at the bar and not resolved by this instrument.** The driver polls
  every 50 ms. Mellum's old starts all landed on one poll (0.11-0.12 s) and three of the new build's four on the
  next (0.16-0.17 s); one pair reads 60 ms.
- **In process the difference is under a millisecond (exploratory, after the run).** `Load` of the warm Mellum
  bundle with the darwin policy and with none, twelve interleaved pairs in one process: 21.2 ms against 21.4 ms, the
  paired difference -0.1 ms at the median (-1.8 to +2.7 ms). The residency sample, which is all the new build adds
  to a warm load, took 0.3-0.8 ms. So the sample does not explain a 50 ms step, and I have not explained why three
  of four served pairs fell on the later poll.
- No warm start announced a prefetch.

**The kill rule, built (`e6421641`).** The darwin policy now covers a load on the CPU backend only
(`decoder/coldload.go:coldPolicyDarwin`), and the planner is given the load's backend. Checked on the real server
built from that commit against `3533a383`, one cold start each of the 1.5B (exploratory, a check of the action, not
a timing): on the CPU both builds prefetched; on Metal the earlier build prefetched and the new one did not, and
served with 12.7% of the bundle in memory, which is the old build's figure in the gate. A Metal load is therefore
what it was before the pre-read, and it is not timed again.

**Beyond the rule as written:** the policy leaves alone every backend that is not the CPU, so WebGPU on macOS is not
given the prefetch either. It was never measured, and the Metal cells are the nearest evidence.

## What is not measured

- Any other filesystem. btrfs sets a large read-ahead window; a filesystem with a small one would make the default
  walk read less, and the trade would move.
- macOS on WebGPU (left as it was, section 9), and macOS with a bundle that does not fit: sections 6 and 9 are
  bundles that fit, and the does-not-fit case is held by tests of the action, not by a timing.
- A bundle that does not fit in memory. Every cell here fits (62 GB). That is the case the `random` walk is for: it
  would save reading 18.7 GB that cannot stay.
- A file that is already in the page cache, where a front-to-back read is pure cost (the transcode record's reference
  times: 2.1 s with the CRC against 0.2 s without, on 17.5 GB).
- More than one reader at a time, which an NVMe disk would serve faster than one sequential pass.

## Files

- `cold-a.sh`, `cold-b.sh`, `cold-c.sh`: the load alone, the cold start to a reply by advice mode, and the same with
  the shipped loader. `run.sh`, `run2.sh`, `run3.sh`: the three runs; `run1.log`, `run2.log`, `run3.log`: their output.
- `gcl-nobara.jsonl`, `gcl-nobara.log`: section 8, the Linux gates' record.
- `gcl-mac.jsonl`, `gcl-mac.log`: section 9, the Mac gates' record.
- `gcl_night.py`, `run-gcl-night-nobara.sh`, `run-gcl-night-mac.sh`: the night driver (Linux and macOS) and the two boxes'
  wrappers for the task's timed gates. `gcl-mac-smoke.log`: section 7's smoke.
- `mac-a.sh`, `mac-c.sh`, `run-mac.sh`, `maccache.py`, `run-mac.log`: the Mac's step 0 (section 6). Its build is the same
  advice switch through darwin's `madvise` system call.
- `xp-advice.patch.txt`: the experimental loader's change (a scratch build, not committed). `cachestat.py`: how much
  of a file is in the page cache.
