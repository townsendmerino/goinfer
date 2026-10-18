# The int4 lane's staging, before and after 1cc27f17: Go heap and footprint during the resident load

Registered 2026-10-18, before the run. Asked for by the owner as "the 7B int4 resident load on Metal, on main and on the
build before it".

**What it measures.** Before 1cc27f17, Metal's int4 lane (`metal/model.go:int4Concat`, `metal/model.go:int4BufA`) built
each projection's words and scales in Go slices and copied them to the device. That garbage was not collected until the
build returned. The change writes each part straight into the device buffers
(`docs/tasks/task-never-swap-2026-09.md`, "Addendum, 2026-10-18").

**Why not the 7B as asked.**
- **On its default path,** a 7B `.gguf` on darwin loads through its sidecar `.giw`, which aliases the file's mapping and
  never runs `int4Concat`. By day on main (53267aee), at `-ctx 4096`, it went `metal-resident (int4)` with a lifetime peak
  footprint of 93 MB and a largest Go heap of 68 MB.
- **A direct load does not fit this Mac.** The fit guard prices weights (4.4 GB), KV (0.2 GB) and the mapped source
  (4.7 GB), about 9.3 GB against 70% of available memory, and is not bypassed.

So the graded case is the 1.5B q4_k_m loaded directly (`-direct-load`), the path `int4Concat` runs on. By day on main it
went `metal-resident (int4)`: weights 1.1 GB, largest Go heap 1,227 MB, lifetime peak footprint 2,176 MB, swap +0 MB. The 7B
on its default path is kept as a control.

- **Job:** `int4-staging-7b-2026-10-18/run.py`, a Mac night job on 2026-10-18 after the two already queued, about 5
  minutes.
  - Builds: `serve-metal-a2732f00` (before) and `serve-metal-53267aee` (main, with 1cc27f17).
  - Graded: the 1.5B direct, in order A B B A.
  - Control: the 7B on its sidecar, A then B.
  - Each file is read once, untimed, first. Each load is a fresh server with `GODEBUG=gctrace=1`, read once it serves and
    then stopped with no request: the largest Go heap at any collection, the lifetime peak physical footprint
    (`footprint.py`, `ri_lifetime_max_phys_footprint`) and swap before and after.
- **Bands, on the 1.5B direct (W = the weights in its banner's `fit:` line), from the two loads of each build:**
  - **Fixed:** main's largest Go heap is at most 1.25 × W in both its loads, and neither grows swap.
  - **Ambiguous:** main's largest heap is between 1.25 × W and 1.6 × W.
  - **Not fixed:** main's largest heap is at least 1.6 × W, or main grows swap.
  - **The band was set after a day reading of main** (largest Go heap 1,227 MB, about 1.12 × W), so the graded comparison is
    main against a2732f00, not main against the band alone.
  - **The contrast is reported, not graded:** the before build's largest heap, to confirm the staging is what the fix
    removed. If the before build also stays under 1.25 × W, the change had no effect at this size, and the result says
    so.
- **The 7B control:** both builds are expected to stay far under W. A load that grows swap is reported.
- **VOID:** a guard refuses a load (the job records it and never bypasses), or a load does not reach `metal-resident (int4)`.
- **No stop rule** beyond the six loads.
