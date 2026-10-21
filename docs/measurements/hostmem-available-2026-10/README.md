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

