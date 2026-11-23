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
| 0 | 17,818 | 26,121 | 1.466 |
| 1 | 17,822 | 25,800 | 1.448 |
| 2 | 17,802 | 25,805 | 1.450 |
| 3 | 17,853 | 25,734 | 1.441 |

Median pair ratio 1.449, min 1.441: **no regression** (bar 0.8). The resident's prompt runs token by token (the batched
prefill takes no per-32 scales) and is still 1.45× faster to first token than the CPU path it replaces.

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
