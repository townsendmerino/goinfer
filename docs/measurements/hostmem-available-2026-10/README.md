# B41: one available-memory reading that counts reclaimable memory as free — registration (2026-10-21, before any code)

**The owner's decision** (Francis, 2026-10-21 11:35 PT, verbatim): "seems like reclaimable should count as free or we'll never be able to
run anything that isn't tiny.  people use their machines in real live". It settles B40's open question (`docs/queue-engineering.md`, B40,
B41). Product code (the fit guard) plus tooling: the merge needs Francis's yes after these gates read. The rules below do not move after
data.

## The reading

- **macOS:** available = physical − (app memory + wired + compressed), where app memory = internal pages − purgeable pages. That is Activity
  Monitor's Memory split: Memory Used = App Memory + Wired Memory + Compressed, and what is left (cached files and free) is available. Read
  through `host_statistics64(mach_host_self(), HOST_VM_INFO64)` (`mach/vm_statistics.h`, `struct vm_statistics64`: `internal_page_count`,
  `purgeable_count`, `wire_count`, `compressor_page_count`), page size from `host_page_size`, physical from `hw.memsize`; called without cgo
  through `purego`, which aikit's GPU module already uses. No `vm_stat` text and no fork.
- **Linux:** `/proc/meminfo` `MemAvailable`, the kernel's own estimate, which counts reclaimable cache. `decoder/hostram_linux.go:HostRAMAvailableBytes`
  already reads it; it does not change.
- **Today, for the record:** macOS reads `vm_stat` text: free + inactive + speculative + purgeable (`decoder/hostram_darwin.go:HostRAMAvailableBytes`).
  Active file cache counts as used: after a 9B load exited the reading was 8.13 GB and 11.38 GB after a purge
  (`docs/measurements/kv8a-metal-2026-10/README.md`, "The day checks").
- **The new reading is not more generous everywhere.** It counts every file-backed page as available, but it counts every app page as used,
  inactive or compressed ones included, which the old reading partly counted as free. Read at about 11:45 PT on 2026-10-21 with Francis's apps open and the
  compressor holding 6.2 GB: old 3.76 GB, new 2.98 GB.
- **One reading, three places:** goinfer's fit guard (`decoder.HostRAMAvailableBytes` and everything that calls it), the night runner's gate
  (`scripts/night.py:free_mem_mb`), and the start checks of the job scripts and `scripts/bench_peer.py` (its heavy-load void). The scripts call
  a small command, `cmd/hostmem`, that prints the same number the guard reads; they keep no second formula.
- **Unchanged:** the 2 GB margin of the heavy-load void and of `night.py --fit-need-gb`, and the swap guards (the per-process reader, the
  512 MB growth limit and the 4,096 MB system backstop).
- **Tonight's queue is unchanged:** the Metal KV8a jobs keep the purge and the current reading. The new reading applies to jobs queued after
  this merges; the run rule records the change (`CLAUDE.md`, "Run budget"; `docs/measurements/noise-registry.md` §8).

## Gates

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| **1a. Activity Monitor agreement** (day, the Mac) | `hostmem -v` prints physical, app, wired, compressed, used and available; at the same minute Francis reads Activity Monitor's Memory tab (Physical Memory, Memory Used, App Memory, Wired Memory, Compressed, Cached Files). **Pass: Memory Used, App Memory, Wired and Compressed each within 2% of Activity Monitor's figure, and available within 2% of Physical − Memory Used.** Three readings a minute apart. | Activity Monitor's figures and the tool's are more than 30 s apart. | Three readings. | Minutes; needs Francis to read the screen. |
| **1b. Anonymous memory lowers it** (day) | A plant process allocates and touches N = 1 GiB of anonymous memory and holds it. **Pass: the reading falls by 0.8 N to 1.2 N** (median of three readings before and three while held), and returns within the same band when it exits. | Swap in use changes by more than 256 MB during the plant (the pressure moved something else). | Once; a void reruns once. | Seconds. |
| **1c. The 9B's cached file does not lower it** (day) | Read the 9B's `.giw` (5.96 GB) into the page cache through a mapping that touches every page, then unmap. **Pass: the new reading falls by less than 0.3 GB (5% of the file); the old reading, read beside it, falls by more** (the control: the plant moved what it should). | The old reading does not fall (the file was already cached: purge first, then plant). | Once; a void reruns once after a purge. | About a minute (the SSD read). |
| **2. The real-life cell** (tier 1, by day, Francis's usual apps open) | With the new reading in the guard: load the 9B resident on Metal (`-ctx 20480 --kv-sessions 1`), serve one gate-2(b) pair (opencode #1: turn 1 with its first token forbidden, then turn 2), unload. **Pass: the server process's own swap growth 0** (`vmmap -summary` swapped bytes of the serve process, before, during and after; macOS has no per-process swap counter the guard can read), **system swap growth under 4,096 MB** (the swap guard's backstop), **and memory pressure never red** (`kern.memorystatus_vm_pressure_level` below 4, sampled every 5 s). Swap, pressure and the reading are recorded before, during and after. If the guard refuses the load, that is recorded as the reading's verdict at that moment (not a fail). **On a fail the reading is too generous: report, and change nothing.** | The timing lock held by another session; a load not Metal-resident with the guard admitting it (a harness fault). | Once. | About 6 minutes (two 7k-token cold prefills at about 60 tokens/s). |
| **3. The fit guard's tests** (tier 0) | The existing fit-guard, host-RAM and Metal memory-guard tests pass (`decoder` `TestFit*`, `TestHostRAM*`, `metal` `TestResidentMemGuard`, `TestMetalMemoryCeiling*`, `TestResidentFitsMemory*`), with the `vm_stat` parser's tests replaced by the new reading's (a committed `vm_statistics64` sample, decoded and checked against the formula). **Refusals are unchanged:** a load whose need exceeds the reading still refuses, checked by a test that injects a reading below and above a fixed need. | | Every test once. | Seconds. |

PASS is every gate green. The merge then waits for Francis's yes.

## Amendment A1 (2026-10-21, after gate 1a's first data and gate 2's data)

**Francis, 2026-10-21 12:31 PT, verbatim:** "fix the formula and a re-run sounds good. however, isn't there an osx system call(s) to help
nail this down further?" With his addendum the same day: the Metal guard caps at `recommendedMaxWorkingSetSize` (gate 4), and gate 2's
own-process bar reads `task_info(TASK_VM_INFO)`.

**The first formula's readings, kept:**
- **Gate 1a, first formula** (physical − (app + wired + compressed)). Francis's first three screenshots (just before 12:15) had no times
  and no log was running, so none was matched. His timed shots at 12:22, 12:23 and 12:25 matched the 20-second log
  (`~/goinfer-bench/hostmem-gate1a/log-2026-10-21.jsonl`) at 12:22:26, 12:23:27 and 12:25:09: App, Wired and Compressed within 1.8%,
  0.9% and 2.04% at worst (App at 12:25). Activity Monitor's "GB" are GiB. The formula read 0.7 to 0.8 GiB above Activity Monitor's own
  available (physical − Memory Used): Memory Used holds about 0.6 GiB that no page counter covers (free + speculative + active +
  inactive + wired + compressor pages summed to 15.39–15.41 of 16.00 GiB) and counts purgeable pages, which Cached Files also counts.
  Not graded: the formula changed.
- **Gate 2, first formula:** 12:06 REFUSED with Francis's usual apps open (budget 5.77 GB, need 6.14 GB); 12:16 REFUSED with Docker,
  Teams and Safari closed (budget 6.14 GB at the guard's check, need 6.14 GB); 12:20 **FAIL** with Chrome closed too: admitted at 8.27
  GiB, served the pair, and the server's own swapped-out memory read 269.8 MB at the first sample (bar 0), then 219 and 35 to 40 MB.
  That figure came from `vmmap -summary`'s writable-regions `swapped_out`, which counts compressed pages as well as swapped ones; system
  swap did not grow (0 MB) and pressure stayed 1. Records: `~/goinfer-bench/hostmem-gate2/run-2026-10-21-{1206,1216,1220}/`.

**The fixed formula (macOS):** available = free + speculative + file-backed (external) + purgeable pages, all from `host_statistics64`
(HOST_VM_INFO64): it adds up what the kernel can hand back, so the memory no counter covers is never counted as available. App, wired
and compressed stay in the reading as context. Linux is unchanged (`MemAvailable`).

**Gate 1a's oracle: pending Francis.** The brief proposed `kern.memorystatus_level`. In XNU (`osfmk/vm/vm_pageout.c:vm_pressure_response`,
`osfmk/vm/vm_page.h:VM_CHECK_MEMORYSTATUS`) it is available pages × 100 / total, where available pages are pageable external + free (+
purgeable without a compressor) only under `CONFIG_JETSAM`; on macOS without jetsam they are `AVAILABLE_NON_COMPRESSED_MEMORY`, active +
inactive + free + speculative, which counts active app memory as available. Read on this Mac at about 12:35 PT: level 79–80% against 76.7–77.1%
for active + inactive + free + speculative over physical (the kernel divides by `max_mem`, which leaves out the ~0.6 GiB), and 25.7% for
the jetsam definition. So it cannot grade this reading. The proposal for Francis: the jetsam definition read from the kernel's own
sysctls, `vm.page_pageable_external_count` + `vm.page_free_count`; it read 0.28 GiB (1.7 to 1.8 points) below the fixed formula, the
difference being speculative and purgeable pages, which the formula counts and jetsam leaves out on a machine with a compressor.

**Gate 2, amended instrument (registered before the rerun):** every 5 s, `hostmem -pid <server>` reads `task_info(TASK_VM_INFO)` through
`task_name_for_pid` (no entitlement needed for the user's own process): resident, internal, `compressed` (pages in the compressor, in
memory or swapped to disk; macOS keeps no separate per-process disk-swap figure) and the footprint; with the reading, system swap and
`kern.memorystatus_vm_pressure_level`. **Pass: the server's `compressed` is 0 at every sample, system swap grows by under 4,096 MB, and
pressure never reaches 4.** Run with Francis's usual apps open; a refusal is recorded as REFUSED, and no app is quit to force admission
without his word.

**Gate 4, the Metal working-set cap (registered before building):** the Metal guard refuses, by name, a load whose Metal allocation
(the resident need less the host copy) exceeds the device's `recommendedMaxWorkingSetSize`; no cap existed (nothing in `metal/` or
aikit's GPU module read it). Tier 0: with the cap injected, an allocation one byte over refuses and one at the cap admits. Tier 1: the
9B load's allocation and the cap, read from the load.

**Candidate, not built:** serve could watch `kern.memorystatus_vm_pressure_level` while serving (B40, B41 in `docs/queue-engineering.md`).

## Readings under amendment A1

- **Gate 2, the fixed formula, Francis's usual apps open (Safari, Chrome and Teams among them): REFUSED** (2026-10-21 13:06:39–13:07:15,
  serve and `hostmem` at `02ab81ef`, record `~/goinfer-bench/hostmem-gate2/run-2026-10-21-1306/`). The reading was 4.2 to 5.1 GiB before the
  load and the guard's budget 3.74 GB at its check, against a need of 6.14 GB; the old reading read 6.06 to 6.17 GiB beside it. The load
  fell back to the CPU path and the runner stopped it before any request; pressure stayed 1, swap 2.83 GiB. Recorded as the reading's
  verdict; no app was quit. Gate 4's cap did not bind: the 9B's Metal allocation is at most the need (its `.giw` host copy is about 0),
  against `recommendedMaxWorkingSetSize` 12,713,115,648 bytes (11.84 GiB); the load prints the allocation only when the cap refuses it.
- **Gate 1a:** pending Francis's choice of oracle (above).

## Amendment A2 (2026-10-21, before gate 1a runs again)

**Francis, 2026-10-21 about 13:10 PT, verbatim:** "1 yes to your proposal, 2 working as intended, 3 yes merge". (1) gate 1a's oracle is
the kernel's jetsam definition read from its sysctls; (2) gate 2's REFUSED at his usual load is the reading working as intended; (3)
B41 merges once gate 1a reads.

**Gate 1a, registered before it runs:**
- **Oracle:** `vm.page_pageable_external_count + vm.page_free_count` (sysctl), times `hw.pagesize`: the kernel's own count of available
  pages under `CONFIG_JETSAM` (`osfmk/vm/vm_page.h:VM_CHECK_MEMORYSTATUS`), which leaves purgeable out where a compressor runs, read
  through a different interface from `host_statistics64`.
- **Ours:** `hostmem` (free + speculative + file-backed + purgeable), which keeps speculative and purgeable pages; Francis chose to keep
  them, so the expected gap is their size (0.28 GiB, 1.7 to 1.8 points, when proposed).
- **Sampling:** every 20 s for 10 minutes (30 samples), by day with Francis's usual apps open, under the timing lock.
- **Pass: at every sample, ours within 2% of physical (0.32 GiB on 16 GiB) of the oracle.** Also recorded: how many samples are within 1
  point, and, as context, the Activity Monitor equivalent (physical − Memory Used rebuilt from the counters: free + speculative +
  file-backed pages) and `kern.memorystatus_vm_pressure_level` at every sample.
- **On a fail:** report and change nothing; B41 does not merge.

**Gate 1a under A2: FAIL, 29 of 30 samples within the bar** (2026-10-21 13:15:47–13:25:31, `hostmem` at `02ab81ef`, Francis's usual
apps open, record `~/goinfer-bench/hostmem-gate1a/run-2026-10-21-1315/`). The first sample read +2.79 points (ours 5.005 GiB, oracle
4.558 GiB); the other 29 read +1.15 to +1.64 points, none within 1 point. At those 29 the difference equals the speculative and
purgeable pages to within 0.03 GiB, the gap Francis chose to keep; at the first it exceeded them by 0.14 GiB, while the reading fell 0.72
GiB in the next 20 s. The runner reads `hostmem`, then four `sysctl` processes in turn, tens of milliseconds apart, so on a machine whose
memory is moving the two readings describe different instants: the likely cause, not verified. The Activity Monitor equivalent tracked
the oracle within 0.04 GiB throughout; pressure stayed 1. B41 does not merge on this reading.

## Amendment A3 (2026-10-21, after gate 1a's run under A2)

**Francis, 2026-10-21 about 13:30 PT, verbatim:** "lets do the fix and re-attempt to get b41 finished.  and should we reschedule the
nightly jobs?" The A2 run's FAIL stays in the record above.

**Gate 1a's instrument, registered before the rerun.** The oracle and the bar are A2's; only the timing of the readings changes:
- `hostmem -oracle` reads, in one process, the kernel's jetsam figure (`vm.page_pageable_external_count + vm.page_free_count`, through
  `sysctlbyname`, times the page size), then our reading (`host_statistics64`), then the jetsam figure again, microseconds apart.
- **A sample is taken when the jetsam figure moved by at most 0.02 GiB between its two reads.** Otherwise it is retaken, up to three
  attempts, each logged; a sample still moving on the third attempt is graded on that attempt.
- **Pass: at every sample, ours within 2% of physical of the mean of the two jetsam reads.** 30 samples, every 20 s, Francis's usual apps
  open, under the timing lock; the Activity Monitor equivalent and the pressure level logged as context.
- **On a fail:** report and change nothing; B41 does not merge. On a pass B41 merges (Francis, A2).

## Amendment A4 (2026-10-21, during gate 1a's run under A3)

**Francis, 2026-10-21 about 14:42 PT, verbatim:** "1" (option 1 of three: compare like with like). The A3 run's data stays as read below
once it ends; it is not regraded under this rule.

**Why:** under A3 each sample's difference was the speculative and purgeable pages, which the formula counts and the jetsam oracle
leaves out on a machine with a compressor, plus at most 0.045 GiB; and those two terms ranged 0.09 to 0.36 GiB minute to minute, so a
fixed 2% bar (0.32 GiB) between the two definitions fails whenever they exceed it. The proposal's figure of 0.28 GiB was one reading.

**Gate 1a, registered before its next run:** A3's instrument (one `hostmem -oracle` per attempt, bracketed, retaken while memory moved),
30 samples every 20 s with Francis's usual apps open. **Pass: at every sample, ours less its speculative and purgeable pages within 2%
of physical of the mean jetsam figure** (the counters agree; the formula's two extra terms are Francis's choice, not graded). The raw
difference and both terms are logged. On a fail: report, change nothing, no merge. On a pass B41 merges (A2).

**Gate 1a under A3: FAIL, 27 of 30 samples within the bar** (2026-10-21 14:39:24–14:49:07, `hostmem` at `274b5682`, Francis's usual
apps open, record `~/goinfer-bench/hostmem-gate1a/run-2026-10-21-1439/`). The misses were the first three samples, +2.53, +2.28 and
+2.06 points (0.405, 0.364 and 0.330 GiB); the other 27 read +0.10 to +1.32 points, 22 of 30 within 1 point. Every sample was taken on its
first attempt (the jetsam figure held within 0.02 GiB across the bracket). At every sample the difference was the speculative and
purgeable pages plus 0.001 to 0.076 GiB, while those two terms ran 0.015 to 0.360 GiB: the gap A4 grades around. The run is not regraded.
Pressure stayed 1; the Activity Monitor equivalent tracked the oracle within 0.15 GiB.

**Gate 1a under A4: PASS by its rule, and it found a defect in the formula** (2026-10-21 14:51:04–15:00:49, `hostmem` at `274b5682`,
Francis's usual apps open, another session's aikit builds from about 14:57, record `~/goinfer-bench/hostmem-gate1a/run-2026-10-21-1451/`).
30 of 30 samples within the bar, 25 within 1 point, the worst +0.305 GiB (+1.91 points); every sample taken on its first attempt;
pressure 2 (warn) at five samples during the builds, 1 otherwise. **At every sample the graded residual was the speculative pages, to
within 0.5 MB** (they ran 0.003 to 0.305 GiB), so the gate passed only because they stayed under the bar. The cause:
`host_statistics64`'s free count already includes the speculative pages (`vm_stat` prints "Pages free" as the free count less
speculative; the free count read here matched `vm.page_free_count + vm.page_speculative_count`), and so does its external
(file-backed) count (anonymous + file-backed pages equal active + inactive + speculative + throttled pages, to the page, read live and in
the committed sample). The formula added them a third time, so it read twice the speculative pages too high. Recomputed from this
run's logged counters with each speculative page counted once, ours less purgeable was within 0.5 MB of the kernel's figure at every
sample. (A1's estimate of the memory no page counter covers read `vm_stat`'s fields, which do not overlap, so it stands.) B41 does not
merge on this reading.

## Amendment A5 (2026-10-21, after gate 1a's run under A4)

**Francis, 2026-10-21 about 15:08 PT, verbatim:** "1" (option 1 of four: fix the formula, then a fresh run graded under this amendment).
The A4 run's reading stays as read above; B41 did not merge on it.

**The formula, fixed:** available = free + file-backed + purgeable pages, where free is `host_statistics64`'s free count less its
speculative count (`vm_stat`'s "Pages free"), so each speculative page is counted once, inside file-backed. That is the kernel's jetsam
figure plus the purgeable pages, which Francis chose to count (A1). Built at `7cd82a0f`, with a tier-0 test: free pages that turn
speculative leave the reading unchanged (the A1 formula and a fix of the free count alone both fail it).

**Gate 1a, registered before its run:** A3's instrument (one `hostmem -oracle` per attempt, bracketed, retaken while memory moved), 30
samples every 20 s, Francis's usual apps open, under the timing lock, `hostmem` and the runner at `331b5887`. **Pass: at every sample,
ours less its purgeable pages within 0.05 GiB of the mean jetsam figure.** Recomputed from the A4 run's counters with the fix, that
residual was within 0.5 MB; a speculative page counted twice shows at any sample whose speculative pages exceed the bar. Logged as
context: the raw difference, the speculative and purgeable pages, the pressure level, and the number of samples whose speculative pages
exceed the bar (where a double count would show; the tier-0 test pins the count whatever that number is). On a fail: report, change
nothing, no merge. **On a pass, B41's merge waits for Francis's yes on the fixed code**, which changed after his yes to merge (A2).

**Gate 1a under A5: PASS** (2026-10-21 15:13:33–15:23:17, `hostmem` and the runner at `331b5887`, Francis's usual apps open, record
`~/goinfer-bench/hostmem-gate1a/run-2026-10-21-1513/`). 30 of 30 samples within the bar; ours less purgeable read −0.00004 to +0.0005 GiB
from the kernel's figure (at most 0.53 MB). Speculative pages ran 0.002 to 0.394 GiB and exceeded the bar at 12 of 30 samples, where the
A1 formula would have read 0.1 to 0.8 GiB high. Every sample was taken on its first attempt; pressure 2 (warn) at 17 samples, 1 at 13.
Gates 1b, 1c, 2 and 3 stand: 1b and 1c grade changes in the reading, and the fix lowers it, so gate 2's refusal holds. The merge waits for
Francis's yes on the fixed code (A5).

**Merged** on Francis's yes, 2026-10-21 about 15:24 PT, verbatim: "yay for pass, push". The merged `internal/hostmem` and `cmd/hostmem`
are byte-identical to the A5 run's build (`331b5887`).

## After the merge: host_statistics64 is rate-limited (2026-10-21)

main's `root-darwin` job (macos-26-arm64) failed `TestReadWithOracle_live`: ours less purgeable read 58.5 MB below the kernel's
figure, which held still across the bracket. The cause is in the instrument, not the formula. XNU rate-limits
`host_statistics64` for binaries that are not Apple's (`osfmk/kern/host.c:rate_limit_host_statistics`): once callers system-wide
pass 2 to 10 requests in a one-second window it serves the last fresh reply, and the jetsam sysctls are not rate-limited. So the
bracket could hold a reading up to a second older than itself. Measured on the Mac: after four fresh replies, 36 of 39 further
calls in the same second returned identical bytes and missed the kernel's figure by up to 15 MB. Every fresh reply matched it
to the byte.

`hostmem -oracle` now reads `host_statistics64` twice inside the bracket, with a page fault between, and prints `fresh 1` when
the replies differ: the cache changes only on a fresh reading, and the fault changes a fresh one. The live test grades only a fresh,
still sample, retaken after the window up to three times. Gate 1a's runs above took one `hostmem` process per sample, so each
process made one request and was rate-limited only if other processes crowded the window. The A5 run's residuals of at most
0.53 MB fit fresh readings. The fit guard's reading is unchanged; `Read` now documents that a sample can be up to a second old.
