# Kill-switch time-to-quiescence — K2's first ledger entry

**Answer: ~10ms (halt → every cancelled generation actually stopped) under saturation on this
box, ~2200× inside K2's own 2× per-token-latency gate bound.** First value for the ledger K8's
`gate kill` (docs/task-halt-2026-09.md) will eventually own and append to automatically; recorded
by hand here since K8 itself is not built yet.

**Box:** MacBook, Apple Silicon (arm64), CPU backend. **Model:** Qwen2.5-Coder-0.5B-Instruct,
Q4_K_M source quantized to int8int8 at load. **goinfer:** `4b979d0` (branch
`killswitch-k1-k2-k5`, off `main`). **Method:** `TestServe_haltUnderLoad`
(`internal/serveapp/halt_test.go`) — 32 concurrent streaming chat completions at
`-max-inflight 32 -max-queue 32` (saturated), `POST /admin/halt`, measuring from the halt call to
`generationRegistry.waitEmpty` returning.

## The number

| trigger | cancelled (was streaming) | refused 503 (was queued) | quiesced_in_ms | admin round trip |
|---|---|---|---|---|
| `POST /admin/halt`, no `-race` | 1 of 32 | 31 of 32 | 10 | 11–37ms (varies by build) |
| `POST /admin/halt`, `-race` | 1 of 32 | 31 of 32 | 10 | 37ms |

Only one of 32 requests is ever actually cancelled mid-generation — the rest are refused before
starting at all. This is expected, not a shortfall: goinfer serializes every generation for one
model behind a single decode-worker mutex (`loadedModel.mu`), so only the current mutex holder is
ever really "in flight" at once; see `docs/task-halt-2026-09.md`'s K2 status line, "found while
building" bullet 1, for the bug this measurement caught (queued requests originally bypassed the
halt gate entirely and would have run to natural completion).

## What this number is not

Not an end-to-end release gate yet — K8's `gate kill` is scoped but not built. This entry exists
so the FIRST measurement isn't lost by the time that drill exists to reproduce it automatically,
per this doc's own ground rule ("every switch has a measured time-to-quiescence in the gate
ledger"). Re-measure on the CI box(es) K8 actually runs on before treating this number as a gate
threshold rather than a single-machine sample.
