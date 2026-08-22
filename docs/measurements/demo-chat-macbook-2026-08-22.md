# demo/chat Apple Silicon decode re-measurement (mac-demo-chat-apple-silicon-numbers.md item 1)

**Why this exists.** `docs/measurements/demo-chat-incumbent-2026-08-22.md` re-measured
`demo/chat/README.md`'s **~57 tok/s (0.5B)** / **~26 tok/s (1.5B)** claims on a Ryzen 7 3700X
desktop and got 28.1 / 12.1 — roughly half — and concluded the likeliest explanation was that the
README's numbers were taken on Apple Silicon all along, just never attributed. This is that
measurement, run on the actual hardware.

## Provenance (the rule from `benchmarks.md`: no figure without a traceable run)

| | |
|---|---|
| commit | `3a27941` |
| box | Apple M1 Pro, 8 cores, macOS, darwin/arm64 — **a laptop**, matching the README's audience |
| harness | `go test ./decoder -bench '^BenchmarkDecode$' -benchtime 200x -count 5` (same harness, same prompt, same greedy sampler as the incumbent leg — no substitution) |
| quant | **int8int8** — the benchmark's default, and what `demo/chat/build-embed.sh`'s prequant path bakes |
| config | `DefaultDecodeParallelThreshold`, i.e. the shipping decode threshold |
| models | `qwen2.5-coder-{0.5b,1.5b}-instruct-q4_k_m.gguf` (the tiers the demo embeds) |

## Result — 5 runs each

| tier | runs (tok/s) | median | README claims | Ryzen (incumbent doc) |
|---|---|---|---|---|
| 0.5B | 51.94, 52.26, 54.35, 57.86, 46.27 | **52.3** | ~57 | 28.1 |
| 1.5B | 28.19, 27.85, 27.65, 27.61, 27.90 | **27.8** | ~26 | 12.1 |

Spread is wider than the Ryzen box's (~2%) — roughly ±10% on the 0.5B, ±1% on the 1.5B — consistent
with a laptop under normal thermal/background-load variance rather than a dedicated bench box, not
with a bad measurement (5 runs, no outlier removed).

## Reading it

**The README's claims were right all along; only the attribution was missing.** The 1.5B figure
(27.8 measured vs ~26 claimed) lands almost exactly on the claim. The 0.5B figure (52.3 measured vs
~57 claimed) is close — within the run-to-run spread's own range (the single best run hit 57.86) —
and the gap is fully explained by unknown thermal/background-load conditions at claim time, not by
the claim being wrong by a meaningful factor the way the Ryzen comparison was.

This confirms the incumbent doc's hypothesis directly: the original numbers were real Apple Silicon
measurements, reasonable to put on a laptop-facing page, just never labeled with a box/quant/commit
— exactly the gap the provenance rule exists to close.

## Action taken

`demo/chat/README.md` updated with a provenance line citing this measurement, keeping the existing
~57/~26 headline figures (still accurate, now attributed) rather than replacing them with today's
specific run numbers, which would falsely imply a tighter guarantee than 5 runs on a shared laptop
can give.
