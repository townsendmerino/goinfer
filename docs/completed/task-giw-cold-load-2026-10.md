# Task — the cold load of a `.giw`: read an MoE bundle deliberately, or not at all (2026-10)

**Status: COMPLETE, archived 2026-10-11.** Built on Linux and on macOS, every registered gate read, the Metal kill
rule applied. Not done, and not scheduled: any Linux filesystem other than btrfs; several readers at once (Decision
3); WebGPU on macOS, which is left without the prefetch and was never measured; and a repeat of the Linux gate on a
box left alone, which the owner may still want (the record's section 8 names three overlaps).

Scoped 2026-10-11 and **built the same day on Linux** (steps 1 to 4; owner: "yes build it"). **Every Linux
gate is read: G-CL1 PASS, G-CL2 SHIP on both MoE cells (0.597 and 0.613), G-CL3 PASS, G-CL4 PASS.** Step 0 (the Mac's measurement) is read (below), and **the darwin pre-read is built** from it (§ "The darwin
pre-read"), **and its gates are read: G-CLd1 PASS; G-CLd2 SHIP on both CPU cells (0.609 and 0.738) and OFF on the 7B
on Metal (1.039), so the registered kill rule fired and the pre-read applies to a CPU load only; G-CLd3 PASS, with one
warm cell at the bar.**
The measurements that scope it are exploratory ([record](../measurements/giw-cold-load-2026-10/README.md)); the gates
below were registered before any code. Owner decisions: § Decisions.

## Why

A `.giw` load maps the file and walks every matrix's header through the mapping. On nobara (btrfs) each page fault of
that walk reads megabytes around it, so on an MoE bundle (thousands of small expert matrices) the walk reads nearly
the whole file in scattered windows:

- The Qwen3.6 35B's cold load is 24-34 s and leaves 93.5% of its 20 GB in the page cache; Mellum's is 8.5-16 s.
- Read front to back first (what a forced CRC does), the same start reaches its first reply in 24 s instead of 38 s
  on the 35B and in 8 s instead of 12.7 s on Mellum.
- With the read-around off for the walk, the load is 9.7 s and reads 0.4 GB, and the first reply is 7 s **later**:
  where the bundle fits in memory, the weights have to be read anyway.
- A dense bundle is not affected (a couple of hundred matrices): its do-nothing arm ties or wins.

So today's loader does the wrong thing in both directions. Where the bundle fits, it reads the file the slow way.
Where the bundle does not fit (the paged MoE case), it reads 93% of a file that cannot stay in memory just to learn
where the matrices are. The second case is inferred, not measured.

## What to build

One decision at load, in `decoder/model.go:Load`'s `.giw` branch, before the header walk. The config in the header says
whether the model is an MoE before any matrix is read.

- **L1, read it first.** When the model is an MoE, the file is mostly not in the page cache, and it fits in memory
  with room (file size against `decoder.HostRAMAvailableBytes`, and no expert pager budget in force): read the mapping
  front to back, then walk. Whether the file is cached is asked of the kernel on a sample of its pages (`mincore`),
  so a warm load pays a probe and nothing else.
- **L2, do not read it.** When the model is an MoE and the file does not fit, or a pager budget is in force: advise
  the kernel not to read around during the walk, and restore the default after it.
- **Dense models and warm files:** no change.
- Linux first, where it is measured. darwin has its own mapping code (`decoder/giwmap_darwin.go`) and an unmeasured
  read-ahead; it gets a measurement before it gets a change (step 0 below).
- Nothing here changes a weight or a logit: it is advice to the kernel and one extra read.

## Steps

0. **darwin, measure only.** The record's part A and part C on the Mac's MoE bundles, with a darwin build of the
   advice switch. No bar; it decides whether L1 and L2 mean anything there.
1. **The decision function**, pure and table-tested: (MoE, cached fraction, file size, available memory, pager) to
   one of {nothing, read first, no read-around}. The residency probe and the two actions behind seams.
2. **L1 on Linux.**
3. **L2 on Linux**, with a cell where the bundle does not fit (a memory-limited run of the 35B; how to limit it is
   part of the step).
4. Docs: `docs/giw-bundles.md` (what a cold load does), CHANGELOG.

## Gates (registered 2026-10-11, before any code)

Tier: day for G-CL1; night for the timed cells (each is under ten minutes, but they are graded). Instrument: the
record's `cold-c.sh` shape, old and new builds as pinned binaries, the bundle evicted before every start, arms
interleaved in alternating order. Noise: the exploratory pairs repeat within 0.4 s on 8-38 s (1-3%); no registry entry
exists for a cold start, so the first A/A of old against old (two pairs a cell) is run and recorded with the gate.
N: four pairs a cell, fixed, no early stop. Ambiguous band: parked results go to the owner.

- **G-CL1, nothing numeric moves (day).** `go run ./cmd/gate identity <base> <new>` on the tiny assets, CPU: every
  family that runs is byte-identical. The decision function's table test, and tests that a warm file and a file that
  does not fit are not read first (through the seams, asserting the action, not a time).
- **G-CL2, the MoE cold start (nobara).** Time from process start to the first finished reply, new over old, on the
  35B and on Mellum.

  | | expected | ship | park | off |
  |---|---|---|---|---|
  | each cell | 0.60-0.70 | at most 0.98 | 0.98-1.00 | above 1.00 |

- **G-CL3, no harm (nobara).** The dense 7B's cold start, new over old: at most 1.02 (the change does not apply to
  it). The warm start of all three bundles (the file in the page cache): the load is at most 50 ms slower.
- **G-CL4, the bundle that does not fit (L2).** On the memory-limited 35B cell: the load reads at most a tenth of
  the file, and the time to a first reply is not above the old build's (at most 1.00; expected below, unmeasured).
  If no honest does-not-fit cell can be built on nobara, L2 is not shipped there and waits for step 0.
- **Kill rule for L1:** if G-CL2 parks or fails on either cell, L1 is not shipped; the exploratory figures were two
  repetitions.

## Built, 2026-10-11

- `decoder/coldload.go:planColdLoad` is the rule and `prepareColdLoad` gathers the facts and acts, called from
  `decoder/model.go:Load` through a hook between the header and the matrix walk (`loadSerializedWeightsHook`). The
  Linux probes and actions are in `decoder/coldload_linux.go`; every other system reports the plan unsupported.
- **The rule as built**, a little tighter than the sketch above:
  - "An MoE" is an MoE whose separate expert matrices average at most 4 MB (file size over layers x experts x 3). An
    MoE with a few large experts walks as cheaply as a dense model and is left alone. The two bundles measured
    average 0.65 MB and 1.26 MB; the dense 7B's matrices average 24 MB.
  - "Cold" is under half of 512 evenly spaced pages resident (`mincore`).
  - "Fits with room" is the file at most three quarters of the available memory. "Does not fit" is paged
    (`Options.StreamWeights`) or larger than the installed memory. Between the two nothing is done, because the
    record's second table shows the read-around switched off losing to nothing done when the bundle does fit.
- **G-CL1: PASS.** `gate identity 3e0e084f 2d4b9630` on the tiny assets, CPU: 37 of 43 families IDENTICAL, 0
  different, 6 not run (the six with no usable tiny fixture). That run loads checkpoints directly, so it shows the
  rest of the load is untouched; the `.giw` branch is covered by `TestLoad_coldLoadPlan` (a real tiny MoE bundle and a
  dense one through `Load`, each plan, the actions asserted through the seams), `TestPlanColdLoad` (the rule's
  table) and `TestExpertMatricesAreSmall`. With the hook never called, planted, `TestLoad_coldLoadPlan` fails.
- **`TestColdLoad_linuxProbesAndActions` caught a defect before the first push.** It runs the real `mincore`, `madvise`
  and read-through on a file dropped from the page cache. The first read-through summed one byte a page into a value
  nothing used, the compiler removed the loop, and the file sampled 0% resident after it. On nobara it has to be run
  with its temporary directory on the disk: `/tmp` there is memory-backed, and the test skips where a file cannot be
  made cold.
- **The timed gates read, 2026-10-11 (the record's section 8): G-CL2 SHIP on both cells** (time to a first reply,
  new over old: 0.597 on the 35B, 38.5 s to 23.0 s; 0.613 on Mellum, 12.6 s to 7.7 s), **G-CL3 PASS** (the dense 7B
  cold 1.001; every warm start within 50 ms, which is also the driver's resolution), **G-CL4 PASS** (the paged 35B in
  a 16 GB scope reads 2.1% of the file by the time it serves and reaches its first reply at 0.976 of the old
  build's). The kill rule does not fire; L1 and L2 stay. The run was disturbed three times by the session that
  queued it; the record names the minutes and the cells, and the graded cold cells are clear of them.
- **As queued:** `docs/measurements/giw-cold-load-2026-10/run-gcl-night-nobara.sh` (driver
  `gcl_night.py`, old `3e0e084f` and new `2d4b9630` as pinned binaries). The does-not-fit cell is the 35B paged
  (`-stream-weights -weight-cache 8`) inside a 16 GB memory scope, evicted first so the scope is charged for what it
  reads.

## Step 0 read, 2026-10-11: the Mac has a different problem

The record's section 6 (exploratory, CPU backend, by day on the owner's word):

- **L2 has nothing to do on macOS.** There is no large read-around: the MoE bundle's default walk reads 5.6% of the
  file in 1.5 s. Nothing is built for darwin from L2.
- **L1's reason does not hold either, but a pre-read still pays, and for a dense bundle too.** The cold first reply
  is slow because the decode faults the weights in a page at a time. With the file read once first it is done in
  3.7 s instead of 8.4-9.7 s on the Mellum MoE and in 3.5 s instead of 5.6-6.3 s on the dense 7B.
- **How to read it matters:** `MADV_WILLNEED` over the whole mapping works on macOS (7 GB in 2.5 s) and a plain
  `read(2)` pass does; the CRC does not (8-11 s).
- **Not yet a change.** A darwin pre-read would be a different rule from Linux's (every bundle that is cold and fits,
  not only small-matrix MoEs), most loads on the Mac are on Metal, which this did not measure, and 16 GB makes "fits"
  the sharp edge. It needs its own gates, registered before its code: § Decisions, 4.

## The darwin pre-read (registered 2026-10-11, before any code)

**What:** on macOS, before the matrix walk, a bundle that is not in the page cache and fits with room (the Linux
rule's three quarters of the available memory, and not paged against a budget) is prefetched with `MADV_WILLNEED` over
its whole mapping. Any bundle, dense or MoE: step 0 measured a gain on both. Nothing is done about the read-around
(L2), which step 0 found has nothing to do there. The rule lives beside Linux's as a per-system policy; Linux's does
not change.

**Why not the Linux read-through:** touching the mapping a page at a time is what makes the CRC take 8-11 s on the
Mac. `MADV_WILLNEED` read 7 GB in 2.5 s in step 0.

Tier: day for G-CLd1; night for the timed cells, on the Mac. Instrument: `gcl_night.py` made to run on darwin (evict
with `msync(MS_INVALIDATE)`, residency with `mincore`), old and new Metal-module servers as pinned binaries (they
serve the CPU backend too), arms interleaved in alternating order. Noise: step 0's pairs repeat within 1.3 s on
8-10 s for Mellum's default and within 0.7 s on the 7B's, by day; an old-against-old pair of cells is run with the
gate. N: four pairs a cell, fixed. Ambiguous band: parked results go to the owner.

- **G-CLd1, nothing numeric moves (day).** `gate identity <base> <new>` on the tiny assets, on the CPU and on Metal:
  every family that runs is byte-identical. The rule's table for the darwin policy; `Load` through each plan with the
  darwin policy on a real tiny bundle, dense included; and on darwin a test of the real calls (a file dropped from
  memory, prefetched, sampled resident).
- **G-CLd2, the cold start (the Mac, night).** Time from process start to the first finished reply, new over old.

  | cell | expected | ship | park | off |
  |---|---|---|---|---|
  | Mellum MoE, CPU | 0.40-0.60 | at most 0.98 | 0.98-1.00 | above 1.00 |
  | 7B dense, CPU | 0.55-0.75 | the same | | |
  | 7B, Metal | none (unmeasured) | the same | | |
  | 1.5B, Metal | none (unmeasured) | the same | | |

  A cell in which the new build did not prefetch (the bundle did not fit with room that night) is not run, and a cell
  that is not run is not a pass.
- **G-CLd3, no harm.** The warm start of each of the four cells (the file in the page cache): the load is at most
  50 ms slower. The does-not-fit and paged cases are held by tests through the seams (the action asserted): nothing
  is done.
- **Kill rules.** If a Metal cell is above 1.00, the pre-read is not applied to a Metal load. If a CPU cell parks or
  fails, it is not shipped for the CPU either.

### Built, 2026-10-11

- The rule is now a per-system policy (`decoder/coldload.go:coldLoadPolicy`): `coldPolicyLinux` is the rule above,
  unchanged; `coldPolicyDarwin` covers any bundle, reads first with `MADV_WILLNEED` (`decoder/coldload_darwin.go`), and
  never touches the read-around. `prepareColdLoad` asks for the memory figures only for a cold bundle.
- **G-CLd1: PASS.** `gate identity ac85083a 3533a383` on the tiny assets: CPU 37 of 43 families IDENTICAL, Metal 25
  of 43, none different (the rest have no usable tiny fixture or do not build resident on Metal). The tests named in
  the gate pass, on the Mac and, for Linux's policy, natively on nobara; with the prefetch replaced by a no-op,
  planted, `TestColdLoad_darwinProbesAndActions` fails.
- **A smoke by day** (the record's section 7, exploratory, one pair): the 1.5B on Metal prefetched and was level
  (1.6 s against 1.6 s); the Mellum cell did not prefetch because 7 GB did not fit in the 5 GB available, as the rule
  intends.
- **Queued for the night on the Mac:** `docs/measurements/giw-cold-load-2026-10/run-gcl-night-mac.sh`. A registered
  cell whose bundle does not fit with room that night will read NOT RUN: Mellum needs about 9.4 GB available and the
  7B about 6.4 GB.
- The driver's eviction now checks itself: straight after a Metal server exits the drop does nothing for a moment,
  and the first smoke's "cold" Metal start was warm.

### Read, 2026-10-11

The Mac's queue was started by day on the owner's word; the run took 4.8 minutes. Every registered cell ran (the new
build prefetched in every cold start) and replies are the same text throughout. The record's section 9 has the pairs.

| gate | cell | median | bar | verdict |
|---|---|---|---|---|
| G-CLd2 | Mellum MoE, CPU, cold | 0.609 (8.82 s to 5.37 s) | ship at most 0.98, expected 0.40-0.60 | **SHIP** |
| G-CLd2 | 7B dense, CPU, cold | 0.738 (5.51 s to 4.07 s) | the same, expected 0.55-0.75 | **SHIP** |
| G-CLd2 | 7B, Metal, cold | 1.039 (3.73 s to 3.87 s) | off above 1.00 | **OFF** |
| G-CLd2 | 1.5B, Metal, cold | 0.987 (1.59 s to 1.56 s) | park 0.98-1.00 | PARK |
| G-CLd3 | warm starts, time to serving | +0.05 s Mellum CPU; 0.00, -0.03, -0.01 s the others | at most 50 ms slower | **PASS**, Mellum at the bar |

- **The kill rule fired and is built** (`e6421641`): a Metal cell is above 1.00, so the pre-read is not applied to a
  Metal load. `coldPolicyDarwin` covers a load on the CPU backend only. Both CPU cells ship, so the CPU keeps it.
- **Beyond the rule as written:** no backend other than the CPU gets the prefetch on macOS, so WebGPU is left as it
  was too. It was never measured. Owner's call if that should be its own cell.
- **The warm Mellum cell is at the bar and the driver cannot resolve it** (it polls every 50 ms). An in-process read
  after the run puts what the new build adds to a warm load under a millisecond; the served step of one poll is not
  explained.
- Linux's policy is not changed by any of this: its read-first is for a header walk that reads the file whatever the
  backend decodes on, and its gate cells were on the CPU backend.

## Decisions

**Decided 2026-10-11 (owner: "yes build it", on the recommendations):** a plain read, not the CRC; Linux first, the Mac
measured before it gets any change; one reader.

1. **Plain read or the CRC for L1?** The plain read is about a second cheaper on 20 GB; the CRC re-verifies a file
   whose marker already vouches for it. Recommended: the plain read.
2. **Linux first, darwin after its own measurement** (recommended), or hold both until the Mac is measured.
3. **One reader or several?** One sequential pass is what is measured (about 1.75 GB/s here). Several readers may be
   faster on NVMe; it would be its own gate, after L1.
4. **A darwin pre-read (from step 0). Decided 2026-10-11: build it** (owner: "yes fix this for macos"). Its gates are
   § "The darwin pre-read", registered before its code. As first written:
   **A darwin pre-read (open, from step 0).** Register gates and build it, or leave the Mac as it is. If built: for
   any bundle that is cold and fits with room, `MADV_WILLNEED` over the mapping before the walk; cells on the CPU and
   on Metal, a dense and an MoE bundle, with the cold time to a first reply as the measure and a warm start and a
   does-not-fit case as the no-harm cells. Recommended: yes, as its own registered step, with the Metal cells
   deciding.
