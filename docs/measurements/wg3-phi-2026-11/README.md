# WG3-phi: results

Registered in [registration.md](registration.md) (committed before any timed run). nobara-pc, RTX 2070 SUPER, driver
595.112.07, WebGPU over Vulkan; the compositor's context present throughout (`kwin_wayland`, 19 MiB). Checkpoint
`~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`. Arms built at base `26a73a8b0` and head `813ba9b23`
(serve sha256 `f6af1bbb…` and `5b653c17…`); fit: base `webgpu-staged (int8int8)` (the CPU path), head
`webgpu:vulkan-resident (int8int8)`.

## TTFT @1024 (2026-11-23 10:26-11:30 PST)

A B B A × 2, 6 timed requests a cell, median; prompt `phi3-mini:1024` (1,031 tokens by `usage`). All eight cells valid;
three void attempts (swap grew by 4-8 kB during the cell) re-ran in place and are listed in `ttft.json`, none averaged.

| block | head ms | base ms | base ÷ head |
|---|---|---|---|
| 0 | ~~17,818~~ | ~~26,121~~ | **VOID**, re-run pending |
| 1 | 17,822 | 25,800 | 1.448 |
| 2 | 17,802 | 25,805 | 1.450 |
| 3 | 17,853 | 25,734 | 1.441 |

**Block 0 is void.** Another session ran a 2-second `go` command on nobara without the shared timing lock at
10:40:23-10:40:25 PST (reported by the coordinator, 14:00). Block 0's base cell (its second attempt, started 10:37:21;
the next cell's line puts its end near 10:42:15) spans that window; no other cell does (block 0's head cell ended
before 10:29:58, block 1's base began 10:44:30). The whole pair is voided, the head cell with it, so the re-run stays a
same-session pair; both rows are marked `"valid": false` with the reason in `ttft.json`, the file as it was is kept
beside it, and block 0 re-runs in place (head, then base) in nobara's afternoon queue.

**Block 0's first re-run is void too.** It started 15:00:43 (head cell valid at 17,824 ms, 15:04-15:08; base cell's first
attempt void on swap, +66 MB; its second attempt started 15:15:30) and nobara crashed at 15:16:45 PST (a hard stop, no
shutdown or kernel GPU error in the journal; the box rebooted at 15:18). The pair is voided again, its head row with the
reason, the file as it was kept beside it, and block 0 re-runs after cuda-small's run. That re-run also started out of
the coordinator's order: it should have waited for cuda-small's job, which the orchestrator's process-name check missed.

The three valid pairs read 1.441-1.450 (median 1.448): no regression so far against the 0.8 bar; the verdict waits for
block 0's re-run. The resident's prompt runs token by token (the batched prefill takes no per-32 scales) and is about
1.45× faster to first token than the CPU path it replaces.

## Decode @128

Not yet run: attempts 1-3 stopped in the harness before any request (the load gate after the builds, the prefill helper's
model table, bench_peer's phase-B backend). Re-queued alone against the same two arm commits.

## Harness review (goinfer #1254's, for the families harness, which repeats this one), checked 2026-11-23

This harness had the same two that hold there, both fixed before the decode cell: `-require-backend` was stripped from
both TTFT arms (now the base binary's only, `WG3_BASE_SERVE`), and the load settle proceeded on timeout and ran once (now
before each cell, stopping at 15 minutes); each cell's rc now propagates, the base fit must name the CPU path, and
`fit()` stops waiting on a dead server. The TTFT cells above are not void: each records its server's decode path, and
all four head cells read `webgpu:vulkan-resident (int8int8)`, so none fell back. The decode-prompt, nested-lock and kwin
findings are refuted in `../wg3-families-2026-11/README.md` and do not apply here either (this harness uses the shared
`scripts/prompts.json`, whose `phi3-mini:128` entry is calibrated).
