# demo/chat incumbent decode re-measurement (Tier 2 gate 4, incumbent half)

**Why this exists.** `demo/chat/README.md` advertises **~57 tok/s (0.5B)** and **~26 tok/s (1.5B)**
"on a laptop CPU". Those figures carry no commit, box, quant or config, and
`docs/task-demo-refresh.md` flags them as pre-P12 and requires the incumbent to be re-measured
before any candidate is compared against it. This is the incumbent half.

## Provenance (the rule from `benchmarks.md`: no figure without a traceable run)

| | |
|---|---|
| commit | `f21324b` |
| box | AMD Ryzen 7 3700X, 8 cores / 16 threads, linux/amd64 — **a desktop, NOT a laptop** |
| harness | `go test ./decoder -bench '^BenchmarkDecode$' -benchtime 200x -count 5` |
| quant | **int8int8** — the benchmark's default, and what `demo/chat/build-embed.sh`'s prequant path bakes |
| config | `DefaultDecodeParallelThreshold`, i.e. the shipping decode threshold the benchmark measures by default |
| models | `qwen2.5-coder-{0.5b,1.5b}-instruct-q4_k_m.gguf` (the tiers the demo embeds) |

## Result — 5 runs each, tight

| tier | runs (tok/s) | median | README claims |
|---|---|---|---|
| 0.5B | 28.07, 27.61, 28.12, 28.03, 28.19 | **28.1** | ~57 |
| 1.5B | 12.19, 12.14, 12.14, 12.12, 12.00 | **12.1** | ~26 |

Spread is under 2% on both tiers, so the measurement is not the uncertain part.

## Reading it — what this does NOT establish

**It does not show the README is wrong by 2×.** It shows the README's numbers are *unattributable*.
Both claims are roughly double what this box does, and the most likely explanation is benign: they
were probably taken on Apple Silicon, whose memory bandwidth genuinely puts a 0.5B int8 decode in
that range, and a demo aimed at laptop users is a reasonable place for an Apple Silicon number. The
defect is that **nobody can tell** — no box, no quant, no commit — so the figure cannot be compared
against, reproduced, or aged out. That is precisely the gap the provenance rule exists to close.

**So these numbers are not a drop-in replacement for the README's.** Substituting a desktop Ryzen
figure into a page whose audience is laptop users would trade one misleading number for another.
What the README needs is either a measured Apple Silicon run with provenance, or both boxes named.
That call belongs with whoever owns the demo.

## What it IS good for

Gate 4 requires the incumbent and the candidate measured **on the same box, same quant, same
commit**. This is the incumbent leg of exactly that comparison, so a candidate measured here can be
compared against 28.1 / 12.1 without either number needing to be right in absolute terms.
