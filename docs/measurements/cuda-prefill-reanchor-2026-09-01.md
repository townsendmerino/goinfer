# CUDA prefill — RE-ANCHORED to the 595.91.07 stack (2026-09-01)

**The retired "~4–5× behind" was a single point on a steep curve. Measured across depth, the
ratio runs from 0.13× (goinfer nearly 8× FASTER) at K=128 to 6.13× behind at K=3900.**

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER, **driver 595.91.07**, Nobara Linux 44, kernel 7.2.0-202.fc44 |
| goinfer | **`fb43caf`**, tree clean (recorded in the results header, not attested) |
| peer | Ollama **v0.32.5** (`~/ollama-0325`) — the same peer build §B8 anchors decode against |
| models | qwen2.5-coder **0.5B** / **1.5B** instruct **q4_k_m**, int4 on goinfer, same GGUF both sides, from `~/models` on local NVMe |
| method | `scripts/bench_peer_prefill.py`, both engines over their own HTTP server, **interleaved per cell** with a server restart between, 6 distinct prompts per cell, medians |
| load | 0.06 / 0.20 / 0.41 at start; box idle |
| results | `~/goinfer-logs/prefill-reanchor-FINAL.json` (archived on the box, not in `/tmp`) |

**This stack matches §B8's decode anchor**, so for the first time the prefill and decode rows on
this page describe the same machine.

## Why this needed a harness rather than a re-run

`bench_peer.py` excludes prefill **on both sides by construction** — it times the inter-token rate
from the first streamed token onward. That is a deliberate property of the decode anchor, so
prefill was never a stale number waiting to be re-run; it was a measurement with no instrument.

And the instrument could not simply be "record TTFT in the existing loop", because that loop sends
**the same prompt for every completion in a cell**. Measured before writing anything:

| engine | repeated prompt | fresh prompt | ratio |
|---|---|---|---|
| goinfer | 2142 ms | 2151 ms | 1.00 — no caching |
| Ollama | 337 ms | 640 ms | **0.53 — caches** |

Naively extending the decode harness would have divided goinfer's real prefill by Ollama's *cache
lookup* and reported **~6.3×** where that cell actually reads ~3.4×. Every request therefore
carries a unique fixed-length prefix. Ollama's cache-check ratio falls monotonically with depth
(1.04 → 0.58 at 0.5B; 0.97 → 0.38 at 1.5B), confirming both that the cache is real and that the
fresh prompts are missing it.

## The result

**TTFT rate** — `prompt_tokens / TTFT`, what an interactive caller feels, each engine's request
overhead included:

| K | goinfer TTFT | Ollama TTFT | goinfer tok/s | Ollama tok/s | ratio |
|---|---|---|---|---|---|
| **0.5B** | | | | | |
| 128 | 39 ms | 346 ms | 3476 | 453 | **0.13×** |
| 512 | 183 ms | 371 ms | 2831 | 1456 | **0.51×** |
| 1024 | 429 ms | 414 ms | 2402 | 2542 | 1.06× |
| 2048 | 1082 ms | 469 ms | 1910 | 4456 | 2.33× |
| 3900 | 2807 ms | 586 ms | 1396 | 6728 | **4.82×** |
| **1.5B** | | | | | |
| 128 | 88 ms | 383 ms | 1540 | 410 | **0.27×** |
| 512 | 379 ms | 434 ms | 1368 | 1245 | 0.91× |
| 1024 | 857 ms | 497 ms | 1203 | 2116 | 1.76× |
| 2048 | 2163 ms | 648 ms | 956 | 3221 | 3.37× |
| 3900 | 5584 ms | 916 ms | 702 | 4300 | **6.13×** |

**The crossover is ~K=1024 at 0.5B and ~K=600 at 1.5B.** Below it goinfer wins, by a lot at the
short end.

## But TTFT rate is NOT prefill throughput, and the gap is worse than it looks

TTFT = fixed per-request overhead + prefill + one sampling step, and **the overhead is not
common-mode**: Ollama's fitted floor is **340–356 ms**, goinfer's is tens of ms. That floor is
what makes Ollama read an absurd 453 tok/s at K=128 and 6728 tok/s at K=3900 — one constant
amortised over more tokens — and it is most of why goinfer "wins" at K=128.

The overhead-free quantity is the **marginal cost per token**, measured between adjacent depths:

| ms per token | 136→519 | 519→1031 | 1031→2067 | 2067→3919 | growth |
|---|---|---|---|---|---|
| goinfer 0.5B | 0.377 | 0.480 | 0.630 | **0.932** | **2.5×** |
| Ollama 0.5B | 0.064 | 0.084 | 0.053 | **0.063** | flat |
| goinfer 1.5B | 0.760 | 0.933 | 1.261 | **1.847** | **2.4×** |
| Ollama 1.5B | 0.132 | 0.124 | 0.146 | **0.145** | flat |

**That is the actual finding.** Ollama's marginal cost per prefill token is flat across a 30×
depth range; goinfer's grows ~2.5×. At the deepest interval the marginal ratio is **14.8× (0.5B)
and 12.7× (1.5B)** — considerably worse than the 4.8×/6.1× the TTFT view suggests, because the
TTFT view is partly measuring Ollama's own overhead against us.

Flat marginal cost is the signature of tiled/flash-style attention; rising marginal cost is
O(K²) attention. This is the *prefill* twin of the decode finding already on record — §B8's
"a real win at short context, a widening loss with depth" — and it has the same mechanism.

**A method note that is itself a result:** the least-squares fit returned goinfer's fixed overhead
as **negative** (−227 ms at 0.5B, −439 ms at 1.5B), which is impossible — nothing answers before
the request arrives. That is the signature of fitting a straight line to convex data, and the
harness now flags it as `LINEAR_FIT_INVALID` rather than printing a `marginal_tok_s` that asserts
a constant which does not exist. Ollama's intercept on the same fit is +340/+356 ms, consistent
between models and matching its measured K=128 floor, so its linear fit is sound. The asymmetry
is real, not an artifact.

## What this does to the retired figure

The old table said `prefill | 0.66 ms/tok | 0.14 ms/tok | 4.7× behind`. Against this sweep:

- **It was not wrong at the depth it was taken at.** 4.82× (0.5B) and 6.13× (1.5B) at K=3900
  bracket it. The error was presenting one point on a curve that spans 0.13× → 6.13× as a
  constant, with no depth attached.
- **It understated the prefill deficit** where the deficit is a compute claim: marginal throughput
  is 12–15× behind at depth, not 4.7×.
- **It hid a real win.** goinfer is 2–8× *faster* to first token below ~600–1000 prompt tokens,
  which no version of this page has ever said.

## Not claimed

- **Two models, one card, one quant (int4 / q4_K_M), greedy.** 7B is in the harness's table but
  was not swept here.
- Depth stops at 3900 (`cudaCtxCap`), so the deep-context cells §B7 covers are untouched.
- CPU and Metal prefill are separate measurements; `docs/prompts/prefill-baseline-measure.md`
  still describes the CPU one and is unaffected by this.
- TTFT includes one sampling step. It is a small constant on both sides and is not subtracted.
