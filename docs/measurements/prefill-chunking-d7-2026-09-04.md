# CUDA batched prefill declined every prompt long enough to need it — chunking, 2026-09-04

**The batched path reported itself healthy at load and then refused every deep-context prompt.**
Its device scratch is O(M·inter); at Qwen2.5-7B's `inter=18944` an 8k prompt asks for **2.28 GB** on
top of the weights and the KV, and an RTX 2070 SUPER holding that model with `-ctx 8192` has
**1.96 GB** free. So `prefillCore` OOM'd on its first big allocation, returned a decline,
`residentPrefillSeed` discarded it, and the prompt went through the per-token loop — with nothing
anywhere saying so. Chunking the pass fixes it: **8012 tokens, 153.9 s → 50.9 s (3.02×)**,
bit-identical.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, AMD64, **NVIDIA GeForce RTX 2070 SUPER (8 GB)**, driver **595.91.07**, Linux 7.2.0-202.nobara.fc44 |
| model | **Qwen2.5-7B-Instruct**, GGUF **q4_k_m** → `int4`, from **local disk** `~/models/qwen2.5-7b-instruct-q4_k_m.gguf` (4.68 GB) — the peer matrix's D7 |
| config | `decoder.Options{Backend:"cuda", Quant:"int4", ResidentContext:8192}` — the same cap `scripts/bench_peer.py` passes as `-ctx 8192` under `BENCH_DEEP_CTX` |
| free VRAM after load | 1.96 GB of 8.16 GB |
| harness | `cuda/prefill_longprompt_test.go` (`TestPrefillLongPrompt`), `GOINFER_HEAVY_TESTS=1` |
| versions | go1.27.0, `aikit/gpu v0.32.0`, goinfer at `5fca4e43` + this change |
| thermal | idle 50 °C; 77–78 °C under the sequential arm, no clock drop observed (1905 MHz) |
| logs | `docs/measurements/prefill-chunking-2026-09-04/` — `before-d7.log`, `after-d7.log`, `bit-identity-gate.log` |
| method | one process per arm, cache `Reset()` between lengths; single timed run per cell — the effect is 3× against a between-run drift of ~3.5% on this box, so pairing was not needed to resolve it |

## Result

Per-token prefill cost, same model, same process, same lengths:

| M | batched **before** | batched **after** (chunk 512) | sequential fallback |
|---|---|---|---|
| 512 | 2.776 ms | 2.777 ms | 12.480 ms |
| 2048 | 3.543 ms | 3.552 ms | 13.878 ms |
| 4096 | 4.440 ms | 4.639 ms | 15.665 ms |
| **8012** | **DECLINED** — `cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY` on the 607 MB gate buffer | **6.358 ms** | 19.211 ms |

At the length the deep-context cells actually run, the batched path went from *not running at all*
to **3.02× the sequential fallback it was silently falling back to** (50.938 s vs 153.92 s for 8012
tokens).

## The shape of the cost, and why the chunk width is 512

Per-token cost is `a + b·(mean attended keys)`. Fitting the three *before* points gives
**a ≈ 2.52 ms/token** of weight+glue work and **b ≈ 1.0 µs/key**. Attention is charged per position
against its own prefix whatever the chunking, so **chunk width buys nothing on the `b` term** — it
only sets how often each weight is re-read, which is the `a` term, and 512 rows already amortizes
that within a hair of the M→∞ limit.

The fit predicted 6.52 ms/token for a chunked 8012; measured **6.358**. It also predicted no change
at 2048 (3.54 vs 3.543 measured before, 3.552 after) — **so this is not a trade against the lengths
that already worked.** The one cost is 4096: **4.639 vs 4.440 ms/token, +4.5%**, eight passes'
worth of per-pass fixed cost on a length that happened to still fit in one. That is the price of
never paying the 4.5× cliff, and it is the right side of that trade.

## What the failure looked like from outside, which is the part worth keeping

Nothing was broken. Every gate was green:

- `PrefillPath()` answered `true, "batched (one weight-stationary CUDA pass)"` — it reads the
  model's STATIC properties, and the decline is a function of **M**, which it never sees.
- The serve banner and `/v1/models` both printed that string.
- `TestPrefillTTFT`, the harness built for exactly this question, stops at **M=2048 on a 1.5B
  model** — a shape that fits, on a model that fits.
- `residentPrefillSeed` treats any prefill error as "use the other path", which is correct and is
  why the fallback never surfaced.

The only symptom was a benchmark cell taking 25 minutes. **A load-time report about a
call-time-dependent property is a check that cannot fail.** Three changes close it: the width is now
in the report (`"…passes of up to 512 rows"`), a call-time decline warns once
(`decoder.warnPrefillDeclined`), and `TestPrefillLongPrompt` probes the lengths a deep cell uses
rather than the lengths that fit.

## Bit-identity

`cuda/prefill_chunked_test.go` (`TestPrefillChunked_bitIdentical`), real qwen2.5-coder-1.5b at
int4, M=1024: chunk widths **256** (divides evenly) and **300** (short final pass) against a
single-pass reference — **0 of 151,936 seed logits differ**, and eight greedy decode steps taken
from the resulting cache match id-for-id. The second half is not redundant: identical seed logits
with a divergent continuation would mean the chunked run left the KV in a different state, which
the seed alone cannot see.

Chunking is bit-identical by construction — pass *k* writes its K/V at the same absolute positions
and attends the same keys passes 0…k-1 wrote, and every batched kernel is documented as the M=1
kernel with an M dimension — but "by construction" is the claim, not the evidence.

## What this does NOT fix

**M35 (Qwen3.6-35B-A3B) and M26 (Gemma-4-26B-A4B) are untouched.** They decline batched prefill
*statically*, on `r.moe` / `r.gemma4Moe`, before any of this runs — a different gate, a different
fix, and the one that owns the 25.5 min / 6.4 min W3 cells. See queue-performance **P20**.

It also does not move any tok/s figure. Prefill is TTFT and cell wall-clock; the W3 decode table
(goinfer 21.4/15.5 against llama.cpp 30.8/22.7) is a separate deficit with a separate cause.
