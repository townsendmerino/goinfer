# Task — the cold load of a `.giw`: read an MoE bundle deliberately, or not at all (2026-10)

**Status:** scoped 2026-10-11 and **built the same day on Linux** (steps 1 to 4; owner: "yes build it"). G-CL1 PASS.
The timed gates G-CL2, G-CL3 and G-CL4 are queued on nobara for the night and **not read**; the kill rule below still
applies to what is on `main`. Step 0 (the Mac's measurement) is owed and nothing changes on darwin until it is done.
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
- **Queued for the night on nobara:** `docs/measurements/giw-cold-load-2026-10/run-gcl-night-nobara.sh` (driver
  `gcl_night.py`, old `3e0e084f` and new `2d4b9630` as pinned binaries). The does-not-fit cell is the 35B paged
  (`-stream-weights -weight-cache 8`) inside a 16 GB memory scope, evicted first so the scope is charged for what it
  reads.

## Decisions

**Decided 2026-10-11 (owner: "yes build it", on the recommendations):** a plain read, not the CRC; Linux first, the Mac
measured before it gets any change; one reader.

1. **Plain read or the CRC for L1?** The plain read is about a second cheaper on 20 GB; the CRC re-verifies a file
   whose marker already vouches for it. Recommended: the plain read.
2. **Linux first, darwin after its own measurement** (recommended), or hold both until the Mac is measured.
3. **One reader or several?** One sequential pass is what is measured (about 1.75 GB/s here). Several readers may be
   faster on NVMe; it would be its own gate, after L1.
