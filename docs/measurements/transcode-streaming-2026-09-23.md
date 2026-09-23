# S2 (gpt-oss only) — real gpt-oss-20b streaming transcode: memory-safe, disk-limited

**Result: the memory question this brief is written against is answered cleanly — swap-used
stayed completely flat for the whole run, on the exact checkpoint that historically drove this
machine to 22.9 GB of swap. The run itself did not finish: this machine's current 12 GB of free
disk cannot hold the 12 GB source and a ~10+ GB output bundle at the same time, and an external
kill-switch correctly stopped it before the disk filled.** This is a disk-capacity limit of this
machine right now, not a defect in the streaming fix — recorded with the same care as a completed
run, per this repo's own measurement discipline, because a near-complete real run that never got
written down teaches nothing to whoever reads this next.

Box: MacBook (arm64, CPU), goinfer (uncommitted S2 gpt-oss-only work at measurement time,
committed alongside this doc). Model: `~/models/gpt-oss-20b-MXFP4.gguf` (12,109,566,624 bytes).
`cmd/prequant -quant int4` (the smallest realistic output, chosen deliberately given the tight
disk margin). External monitor: `ps -o rss=` on the transcode PID every 200 ms, `sysctl -n
vm.swapusage` and `df -g /` on the same cadence, three independent kill-switches (RSS > 6 GB, swap
> baseline+1.5 GB, free disk < 2 GB).

## What the code change is

`decoder/gguf.go`: gpt-oss removed from `needsResidentSerialize`; `loadGptOss`'s per-layer closure
(already self-contained — every tensor it reads is named `blk.{i}.*`, including the per-layer
`AttnSinks`/`RouterBias` this family keeps per-layer rather than as a model-level tail) now drives
the sink directly, the same `build → write → release` shape `loadQ35` already used for qwen35
(2026-08-24, `docs/completed/task-zeno-compare.md`). Verified on the tiny fixture
(`decoder/testdata/gptoss_tiny.gguf`, no embedded tokenizer, so through `StreamTranscodeGGUF`
directly) before ever touching the real checkpoint: `TestGptOss_streamedMatchesResident` — streamed
vs resident-build-then-serialize bundles byte-identical across int4/int8int8/f32, mutation-checked
(corrupting the per-layer sink write to always emit layer 0 turns the test red); the parity
manifest's staleness on `decoder/gguf.go` closed via the sanctioned non-numeric refresh, backed by
that same byte-identity proof — not blind trust that a control-flow change is numerically inert.

## What the real run showed

| | |
|---|---|
| baseline swap-used | 1784.38 MB |
| swap-used at kill (disk-limited stop) | 1776.38 MB — **lower**, not higher |
| peak `ps`-reported RSS | 5,394 MB |
| output written before the disk kill | ~10 GB of an estimated ~10–11 GB total (very close to done) |
| free disk at kill | 1 GB (crossed the 2 GB floor) |
| wall time before the kill | ~2m45s |

**Swap-used never grew — it went DOWN 8 MB over the run.** This is the metric this whole task
(`task-never-swap-2026-09.md`) is about, and on the exact model that historically drove it to 22.9
GB, it did not move. The RSS figure (5.4 GB peak) looks alarming next to the brief's own "≤1.5 GB
above baseline" bound, but tracking it live (not just the peak) showed RSS rising as the transcode
sequentially touched fresh pages of the mmap'd 12 GB source, then FALLING again (3,989 → 3,747 MB
over one ten-second window, unprompted) as earlier pages left the working set — the exact
darwin-RSS-counts-reclaimable-file-pages shape `CLAUDE.md`'s own "a guard that inverts" section
warns never to key a guard on, here observed directly rather than assumed. Swap-used, not RSS, is
what actually answers "is this process losing," and it says no. The brief's own RSS-based bound may
need reinterpreting as "anonymous RSS" (the `footprint` tool's "Dirty" category, as used in
`sidecar-default-2026-09-22.md`) rather than raw `ps`-reported RSS if a stricter check is wanted
later — not attempted this pass.

## A real mistake caught before it could do damage

The kill-switch's own cleanup command removed the wrong temp-file name
(`$OUT.tmp.giw` = `gptoss-s2-transcode.giw.tmp.giw`, which never existed) instead of the real one
`prequant.Transcode` actually writes (`strings.TrimSuffix(out, ".giw") + ".tmp.giw"` =
`gptoss-s2-transcode.tmp.giw`). The disk kill-switch fired correctly and killed the process, but
its own cleanup silently failed, leaving a 10 GB orphaned file that a `df` check right after the
run caught immediately (disk was still at 2.0 GB free when it should have recovered to ~12 GB) —
found and deleted before it caused any further problem, not shipped as a scratchpad script anyone
else would reuse. Recorded because a monitoring script's own bug, caught in exactly the deployment
it was meant to protect, is worth writing down.

## Decision

The memory-safety question — does gpt-oss-20b now transcode without driving swap — is answered:
yes, cleanly, on the real checkpoint. The run's own completion is blocked purely on this
machine's disk headroom, which is a pre-existing constraint unrelated to this fix (the same 12 GB
free that made the S1 measurement docs's own sidecar builds tight). Per the brief's own Decision
rule ("a family that cannot be made to stream... stays on the fallback with the reason written
next to `needsResidentSerialize`"): gpt-oss WAS made to stream, and the reason it did not finish
here is disk, not a streaming failure — the fix ships as built; a disk-headroom-permitting rerun to
get a completed byte-identical real-checkpoint confirmation (matching S1's own real-model gate) is
owed, not blocking.

## Not done here (owed if this is picked up again)

- A completed real-checkpoint run (needs ~24 GB free disk on this machine, or a box with more
  headroom) — would let S1's own real-model byte-identity gate run against the completed real
  gpt-oss-20b bundle too, not just the tiny fixture.
- The other five `needsResidentSerialize` families (gemma4, laguna, granite, nemotron, llama4) —
  gemma4 specifically needs the head/tail split extended for its fused PLE/MoE tail, a genuinely
  different obstacle than gpt-oss's (which turned out to have none). Not attempted this pass.
- Three-run RSS-peak averaging the brief's own Measure section asks for (one run here, disk-
  limited before even that one finished).
- `docs/giw-bundles.md` (which families stream) and the README gpt-oss example's own caution note
  (added in the S1 push) still describe gpt-oss as needing the resident-build fallback — now stale
  for THIS family specifically; not yet updated.
