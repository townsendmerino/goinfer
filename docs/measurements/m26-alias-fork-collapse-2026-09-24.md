# Why the M26 alias arm collapsed: fork() copies a whole mapping once Metal has wired any page of it — 2026-09-24

**Status: root-caused, reproduced without a model, FIXED and confirmed on M26 (§7a).** Companion to `s6-alias-2026-09-24.md` (the S6 build and the three
collapses) — this document supersedes that record's "mechanism unknown" and corrects several of its numbers.

## In one paragraph

A `.giw` is mapped `PROT_READ|MAP_PRIVATE` for the model's whole life. `GOINFER_METAL_ALIAS=1` points Metal at
per-tensor windows of that mapping with `newBufferWithBytesNoCopy`. When a command buffer touches such a buffer,
IOKit wires its pages by faulting each one **with write intent**, which on a private file mapping means every wired
page becomes an anonymous copy-on-write page in a kernel shadow object, and the mapping's object is marked
`true_share`. From that moment, any `fork()` of the process cannot share the mapping lazily — the kernel copies
the **entire** VM entry eagerly, page by page, on the forking thread. The serving process forks every 2 s (its swap
guard shells out to `sysctl`). On a 16 GB MacBook with a 15 GB mapping, the first such fork after the first request
reads the whole file into anonymous memory, evicts everything else, and the process is paged out until a watcher
kills it. On the 1.5B the same copy is 1.3 GB and survivable, which is why those runs looked clean. Marking the
mapping `VM_INHERIT_NONE` (`minherit(2)`) makes the copy vanish; so does never forking.

## 1. The symptom, exactly (from the archived records)

Three M26 (gemma4-26b, `gemma4-26b-int4-v13.giw`, 16,119,998,955 B) alias-arm attempts, one copy-arm run on the same
file, same machine (MacBook darwin/arm64, 16 GB), same harness (`scripts/moe_pager_ab.py`, `-backend metal -ctx 512
-moe-cache-experts -moe-cache-slots 8`, greedy 32 tokens), external kill-watch on the server pid. Data:
`s6-alias-2026-09-24/m26-v13/` and `m26-v13-after-reboot/`.

| | copy arm (copy1) | alias, attempt 1 (alias2) | alias, attempt 2 (alias1, fresh reboot) | alias, attempt 3 (diag) |
|---|---|---|---|---|
| machine at start | swap 224 MB | swap 224 MB, warm cache | **swap 0, 87% free, load 3.5** | swap 614 MB |
| load | 58.4 s (one-time CRC pass) | 10.5 s | 10.6 s | 10.4 s |
| after-load phys footprint | 3,975 MB (IOAccel 2,407 + untagged 1,550) | 3,255 MB (IOAccel 1,684 + untagged 1,553) | 3,251 | 3,255 |
| request | 96-token prefill + 32 tokens, 7.8 tok/s | reached the forward pass, 0 tokens | same | same |
| what happened next | swap flat, completed | RSS 4.4 GB → 0.18 GB in ~4 s; swap 224 → 1,334 MB | RSS 4.5 → 0.018 GB with swap still **0**, then +380 MB in one tick | RSS → 15 MB, swap +677 MB |
| task page-ins in the burst | +148 for the whole request | +189,084 (2.9 GiB) in one 2.4 s sample, faults +1,240 | +206,688 (3.2 GiB), faults **+0** | +211,253 (3.2 GiB), faults +12 |
| end | done | killed by the watcher (+1,110 MB) | killed (+380 MB single tick) | killed (+677 MB); exit −9 = the watcher's SIGKILL, process alive until then |

The after-load footprints of the three alias attempts are identical and exactly 723 MB (the bytes aliased) below the
copy arm's. The collapse begins 0–2 s after the request is sent, never during load.

## 2. What the records said before any experiment

Four things in the archived data, none of which needed a run:

1. **The burst is not the model's own I/O.** The paged expert path preads hundreds of MB per token through the same
   file, yet a full copy-arm request adds only +130 to +188 task page-ins (four runs). `pread` goes through the
   cluster path (UPL_FILE_IO) and never increments a task's page-in counter; only `vm_fault_page` on the task's own
   thread does (XNU `vm_page_internal.h`, `VM_PAGE_COUNT_AS_PAGEIN`). Three gigabytes of task page-ins with the
   user fault counter flat means the server's thread was inside one long **kernel-initiated** read.
2. **The swap guard's first probe never returned.** `internal/serveapp/swapguard.go` prints `swap guard: baseline …`
   (or `reports unknown on the first sample`) when its first read returns. Both copy-arm logs have the line; **none of
   the three alias logs does.** The read is `exec.Command("sysctl","-n","vm.swapusage")` (`decoder/memwatch_darwin.go` as of `9c43f016`, line 18 — replaced by the fix in §7),
   a real libc `fork()` on darwin (Go's own `syscall` package, `exec_libc2.go` line 86), on a 2 s ticker.
3. **The 1.5B was never "clean".** In the first 1.5B alias run the first request paged in 222 MiB; the file's
   never-read remainder after the copy arm's load was 228 MiB (97%). Those pages lie outside every aliased window.
   The copy runs paged in ~0. (`1.5b-v13-fused/*.json`.)
4. **Per-process instruments cannot see aliased pages at all.** On the 1.5B the GPU read 625 MB through no-copy
   buffers, and no `footprint` category, nor RSS, nor "mapped file" moved by it (total phys 354 → 380 MB across the
   whole request). So the S6 record's "1005 → 354 MB" is a ledger relocation, not a RAM saving.

## 3. The mechanism, step by step (XNU `main` as fetched 2026-09-24; line numbers from those copies)

Every kernel step below was read in source by two independent investigators and then re-read by the author; the one
closed component is Apple's GPU driver, whose behaviour is established by the experiment in §4, not by reading.

| step | what | where |
|---|---|---|
| 1 | `decoder.Load` maps the whole `.giw` `PROT_READ`, `MAP_PRIVATE` — one VM map entry over a copy-delayed object with `needs_copy` | aikit `mmap/mmap_unix.go:46`; `kern_mman.c:928-935`; `vm_object.c:4061-4067` |
| 2 | `weightAlias` creates one `newBufferWithBytesNoCopy` per tensor/fused group over its page-aligned window; creation wires nothing | `metal/alias.go:100-196`; aikit gpu module's `Device.NewBufferNoCopy`; TestNoCopyWiring (+40 pages at creation) |
| 3 | At the first command buffer that references such a buffer, the driver prepares it: `IOMemoryDescriptor::prepare → wireVirtual → vm_object_iopl_request`, which **faults every page with `prot \| VM_PROT_WRITE`** even for a read-only UPL, and marks the object `true_share`, `COPY_DELAY` | `IOMemoryDescriptor.cpp:4173-4245`; `vm_pageout.c:8404-8422, 8585-8586` |
| 4 | A write-intent fault on a `needs_copy` private file page **copies it** into a fresh anonymous page of the top (shadow) object — wired, and not in the process's pmap, so invisible to RSS/footprint | `vm_fault.c:2203-2246, 2289-2291` |
| 5 | `vm_map_fork` sends any entry whose object is `true_share` (or map-wired) to `slow_vm_map_fork_copy`, which copies **the whole entry** | `vm_map.c:13954-13957, 13649-13672` |
| 6 | The copy is strategised; `vm_object_copy_delayed` **refuses** an object with wired resident pages ("we can't safely take write permission away from wired pages"), so it falls through to `vm_object_copy_slowly` — every page of the entry faulted in (from disk if not cached) and copied, on the forking thread, `THREAD_ABORTSAFE` | `vm_map.c:12742-12814`; `vm_object.c:3641-3981, 4010-4074` |
| 7 | The forking thread is the server's; `task->pageins` counts each page it reads; no user fault is taken (flat `faults`); other threads block on the map lock (they stop faulting too) | `vm_fault.c:4544`; `vm_page_internal.h:1016-1022` |
| 8 | The swap guard forks every 2 s; the harness's own `top`/`footprint` run in other processes and do not fork the server | `internal/serveapp/swapguard.go:75,101`; `decoder/swapwatch.go:103-117`; `decoder/memwatch_darwin.go` as of `9c43f016` (line 18) |

Consequences that the records confirm: the copy is 15.4 GB on M26 (cannot fit: everything else is compressed and
swapped, RSS falls to nothing, the 3.2 GiB of page-ins is the part read before the machine ran out); the guard's
first fork after the first request never returns (no `baseline` line); the process dies promptly on SIGKILL
(`THREAD_ABORTSAFE`, not uninterruptible — a refuter's correction to the first draft of this chain); on the 1.5B the
copy is 1.3 GB per tick and the first one reads the file's unread remainder.

## 4. The experiment: `metal/alias_forkprobe_test.go` (`GOINFER_ALIAS_FORK_PROBE=1`)

No model, no server, 16 s. A fresh 2 GiB file of random bytes written with `F_NOCACHE` (uncached, so every page-in is
countable), mapped by aikit `mmap.MapReadOnly` exactly as `decoder.Load` maps a `.giw`; ONE no-copy buffer over a
16 MiB page-aligned window mid-file; one Metal dispatch touching one float per page of that window only; `fork+exec
/usr/bin/true` timed at each phase; an external shell sampling `vm_stat` every ~68 ms throughout. Runs 1–5 are in
`s6-alias-2026-09-24/forkprobe/` (run 1 is the version that forked *inside its own instrumentation* and thereby found
the effect; runs 3–4 reproduced runs 2's numbers and crashed in a first draft of the last arm).

| phase (run 5; run 2 in parentheses) | fork+exec | what else |
|---|---|---|
| no mapping | 2.79 ms (2.82) | |
| 2 GiB mapped, buffer created, never touched | 2.55 ms (2.59) | wired Δ ≈ 0 |
| GPU touches the 16 MiB window: 7.5 ms (8.2 ms) | | |
| **first fork right after the touch** | **1,092 ms (1,131 ms)** | system page-ins **+131,814 (+131,258) = the whole 2 GiB file** (131,072 pages); sampler: anonymous +1.23 GB mid-fork, free pages → 3,572, compressor +0.66 GB after |
| 2 s later, GPU idle | 2.56 ms (2.48) | the wire is released between command buffers |
| **during continuous dispatch** (8–17k dispatches) | **255 / 260 / 247 ms (247 / 246 / 253)** | file now cached: a 2 GiB memcpy per fork, page-ins ~0; wired only +0.25 GB |
| 2 s after the loop stopped | 3.27 ms (2.96) | |
| after `ReleaseAll` / after `munmap` | 3.78 / 2.88 ms (3.15 / 3.12) | |
| **same file mapped again with `minherit(VM_INHERIT_NONE)`: first fork after touch** | **3.81 ms** | |
| **… during continuous dispatch (8,117 dispatches)** | **3.74 / 3.98 / 3.73 ms** | the copy is gone |

Readings:

- **The wire is window-precise.** The touch wired +127 pages (the 1,024-page window would be the ceiling; the earlier
  aikit probe's whole-mapping buffer wired the whole mapping because the buffer *was* the mapping). The hypothesis that
  the driver wires the enclosing region is dead.
- **The whole-file read is inside the fork, not the touch.** A 2 GiB read cannot fit in an 8 ms touch; it fits in the
  1.1 s fork, and the sampler shows the anonymous copy growing mid-fork.
- **It is every fork made while a command buffer is in flight**, and no fork made while the GPU is idle. A decode always
  has one in flight. The serving swap guard forks every 2 s. On M26 that is a 15 GB copy per tick.
- **`VM_INHERIT_NONE` on the mapping removes it entirely** (fork stays ~3.8 ms with the window wired), and so would
  not forking. Weights are never needed by a child; the only children are exec'd helpers.
- Scale check against the records: 2 GiB from disk in ~1.1 s ≈ 1.9 GB/s; the M26 bursts read 2.9–3.2 GiB in one
  ~2.4 s sample before the machine ran out — the same rate.

## 5. What this changes about S6

- **"No-copy" over a `MAP_PRIVATE` mapping is not file-backed on macOS.** Every page the GPU touches becomes a wired
  *anonymous* copy (step 4), plus the file's own page in the cache. The S6 brief's premise — file-backed views the
  OS can reclaim — does not hold for this mapping type. The `alias.go` comment saying the pages are "file-backed and
  never anonymous" is wrong and must be corrected. Whether a `MAP_SHARED` mapping (no `needs_copy`, so no COW at wire)
  gives the file-backed behaviour the brief wants was the next design question — **answered yes, 2026-09-24**: `s6-alias-2026-09-24.md`, "MAP_SHARED"; `decoder.Load` now maps `.giw` shared on darwin.
- **The 1.5B footprint headline (1,005 → 354 MB) is an accounting artefact.** The 625 MB moved out of the process's
  ledger, not out of RAM; an honest comparison needs system-wide counters (`vm_stat` anonymous / wired / file-backed)
  around the same events, which no S6 run sampled.
- **`aliasDecision`'s "over half of RAM" guard keys on the wrong variable.** The hazard is (mapping size × fork rate ×
  free memory): the undeclined 7B (5 GB) would copy 5 GB per guard tick while aliased. The guard stays until the fix
  below is in and M26 is re-run, then it should go.
- **Gates 2 and 4 of the registered rule (depth-128/2048 tok/s, memory-hog arm) were never run**, and the tok/s the
  1.5B A/B did show (equal) is now explained: the copy runs on the guard's thread while decode threads keep running.

## 6. What was refuted (the multi-agent pass, verified against source and data)

- **H2 — the driver wires the whole containing VM region at first commit:** refuted three ways before the probe (XNU
  clips every UPL to the requested range; alias2's first phase is exactly 723 MiB followed by ordinary request faults;
  a synchronous whole-region wire on the committing thread does not fit the 1.5B timing) and directly by the probe
  (+127 pages).
- **H3 — per-window wiring plus pread churn thrashing on a ~7 GB-headroom box:** refuted on bytes (bounded at 727 MiB
  vs 3.2 GiB read), on the rebooted run (swap 0, 87% free, no shortage to trigger the retry loop), and on the copy arm
  (identical churn, clean).
- **One refuter's objection to H1** (in alias1 the first guard tick may have preceded the request by ~0.5 s and should
  then have printed `baseline`) sits inside the harness's own ±0.5 s timing; the line's absence in all three logs is the
  evidence that the first tick landed after the first command buffer. Its proposed alternative — a request-path driver
  read beyond the windows — is what the probe rules out. Its correction (`THREAD_ABORTSAFE`) is adopted above.

## 7. The fix (applied 2026-09-24)

1. **`minherit(VM_INHERIT_NONE)` on the `.giw` mapping in `decoder.Load` (darwin).** Removes the class, not just the
   guard's instance of it: any future `os/exec` in the server, and any fork by a library, stops copying weights.
   Verified: forks at ~3.8 ms with the window wired. (`SYS_MINHERIT` = 250 in the stdlib syscall table; no cgo.)
2. **Read `vm.swapusage` and `hw.memsize` with a bare `sysctl(2)`** (`unix.SysctlRaw`: `xsw_usage.xsu_used` is at byte
   offset 16 of a 32-byte struct; `unix.SysctlUint64("hw.memsize")`) instead of exec'ing `sysctl`. The comment in
   `memwatch_darwin.go` claiming there is "no bare-syscall alternative" mistakes `sysctl -n`'s formatting for the
   sysctl's type. Cheaper on every tick, and the guard's baseline no longer depends on a fork returning.
3. Then **re-run the M26 alias arm once** (kill-watch on the server pid, alias first) — done, §7a. `aliasDecision`'s
   half-of-RAM guard is removed with it.

Applied as: `decoder/forkinherit_darwin.go` (`excludeFromFork`, called on every `.giw` mapping in `decoder.Load` via
`protectGIWMapping`; tested through `Load` — `TestLoad_protectsTheWholeGIWMappingFromFork`, mutation-checked against
protecting a sub-slice and against removing the call — and `TestExcludeFromFork_syscallReachesMinherit`);
`decoder/memwatch_darwin.go` (`SwapUsedBytes` via `syscall.Sysctl("vm.swapusage")`, decoding `xsw_usage`) and
`decoder/hostram_darwin.go` (`HostRAMBytes` via `syscall.Sysctl("hw.memsize")`), each tested against the `sysctl`
command's own output on the machine. The root module stays free of `golang.org/x/sys`: `syscall.Sysctl` drops only one
trailing NUL, so both values arrive intact. `HostRAMAvailableBytes` still execs `vm_stat` (inactive pages have no
sysctl), but only at load and from the web UI — and the mapping is no longer inherited, so that fork is cheap.

## 7a. Confirmation on M26, same day

Four interleaved arms, alias first (`GOINFER_METAL_ALIAS=force` so the then-present size guard could not silently turn
the alias arms into copy arms — both alias logs carry the `weights aliased … 723 MB` banner), same file, same cell as
§1, kill-watch on the server pid, machine NOT rebooted (swap 619 MB left by the earlier collapses). Data:
`s6-alias-2026-09-24/m26-v13-after-fix/`.

| arm | tokens | decode tok/s | TTFT | swap Δ | task page-ins during the request | swap guard `baseline` line |
|---|---|---|---|---|---|---|
| alias1 | 32 | 5.98 | 18.7 s | **0** | +46,431 (725 MiB) | printed |
| copy2 | 32 | 6.19 | 17.3 s | 0 | +128 | printed |
| alias3 | 32 | 6.07 | 17.4 s | **0** | +46,395 (725 MiB) | printed |
| copy4 | 32 | 6.04 | 16.8 s | 0 | +169 | printed |

- **The collapse is gone**: 2 of 2 alias arms completed, swap flat (it had risen 380–1,110 MB and needed a SIGKILL in 3 of
  3 before the fix), and the guard's first read returned in every arm.
- **The page-ins are now exactly the windows**: 725 MiB against 723 MiB aliased — the copy-on-write copies made by the
  wire itself (step 4), paid once per page, and nothing beyond them. Before the fix the same counter showed 2.9–3.2 GiB
  and still rising.
- Greedy text identical across all four arms. Decode within noise (n=2 per arm; not a registered bench).
- Per-process footprint: alias 3,251/3,255 MB after load and 3,762 at token 32 vs copy 3,978/4,485 — **not a RAM saving**
  (§5: the aliased pages became invisible wired anonymous copies; the 725 MiB of COW page-ins are where they went).

## 8. What is still NOT established

- Driver internals beyond what the probe shows (window-precise wire; released when idle). Whether a real decode's wire
  persists between tokens was not measured separately — it does not matter for the mechanism, since a command buffer is
  in flight for the whole request.
- M26 with the fix: not run. The 7B: not run at all. S6's own gates: not run.
- ~~Whether `MAP_SHARED` avoids the COW copies~~ — it does (0 vs 15,131 COW faults for a 16,384-page window; +132 vs +40,430 on the 1.5B served path); adopted on darwin.

## 9. Method notes (so the next one is faster)

- The decisive fact was hiding in the *instrument*: run 1 of the probe called `vm_stat` (a fork) right after the GPU
  touch, saw 131k page-ins it could not explain, and timed only forks made 2 s later (fast). Any harness that forks the
  process under test participates in this mechanism; `top`, `footprint`, `vm_stat` from a *separate* process do not.
- Per-process `footprint`, RSS and "mapped file" are blind to IOPL-wired COW pages. `top`'s `pageins` counts
  kernel-initiated reads on the task's thread; `faults` does not. `pread` shows in neither.
- Reading the kernel source settled what the driver *asks for*; only the experiment settled what the driver *does*.
  Four parallel investigators (timeline forensics, code-path trace, XNU/Metal research with fetched source, instrument
  audit), a judge, nine adversarial refuters and the probe; total wall time about three hours, zero model loads.
