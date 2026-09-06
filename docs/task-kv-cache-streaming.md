# Experiment: KV cache streaming — is there a problem to solve?

Status: **measured, 2026-09-06. Recommendation: park the pager; three cheap fixes it surfaced are done.**
Filed 2026-09-02 (brief below); measured entirely on `nobara` (RTX 2070 SUPER, 8 GB, driver
595.91.07) plus arithmetic done cold. No pager code, no allocator changes, no CUDA work, per the
brief's own constraint — this doc is measurement and a verdict only.

## The lead (as filed)

`RaymondHuang210129/llama.cpp-adaptive-kv-streaming` — a research llama.cpp fork (1 star) — keeps
KV in pinned host memory with a bounded CUDA pool shared between resident pages and a transfer
ring, adapting the split as context grows and prefetching ahead. Interesting because it *moves*
the whole context rather than discarding it (no quality cost, no architectural assumption) — a
memory-placement problem, the kind `decoder/moepaging.go`'s expert-streaming design already solves
for weights. Their own numbers are not ours (single config, one card, research code) — cited for
the approach, not the figures.

## Step 0 — arithmetic: is there a problem at all?

All weight sizes are real files on disk (`~/models/*.gguf` on this box) or `.giw`; all layer/head/
kv-head/sliding-window fields read directly from each checkpoint's own GGUF metadata via
`aikit/embed.GGUFFile`, except H27 (no local file — used `docs/task-families-2026-09.md`'s own
independently re-verified header fetch). KV bytes = growing-layer-count × 2(K,V) × kv_heads ×
head_dim × ctx × 4B (f32, goinfer's default `kvqF32`).

| model | weights | growing:fixed layers | KV@8k | KV@32k | KV@128k | total@8k | total@32k(native) | total@128k |
|---|---|---|---|---|---|---|---|---|
| S Qwen2.5-Coder-1.5B (Q4_K_M) | 1.04 GiB | 28:0 (plain GQA) | 0.44 GiB | 1.75 GiB | 7.00 GiB | 1.48 GiB | 2.79 GiB | **8.04 GiB** |
| D7 Qwen2.5-7B-Instruct (Q4_K_M) | 4.36 GiB | 28:0 (plain GQA) | 0.88 GiB | 3.50 GiB | 14.00 GiB | 5.24 GiB | **7.86 GiB** | **18.36 GiB** |
| G20 gpt-oss-20b (MXFP4) | 11.28 GiB | 12:12 (1:1 SWA@128 cap) | 0.38 GiB | 1.50 GiB | 6.00 GiB | 11.66 GiB | 12.78 GiB | **17.28 GiB** |
| M26 Gemma-4-26B-A4B (Q4_0) | 13.45 GiB | 5:25 (1:6 SWA@1024 cap) | 0.31 GiB | 1.25 GiB | 5.00 GiB | 14.15 GiB | 15.09 GiB | 18.84 GiB |
| M35 Qwen3.6-35B-A3B (Q4_K_M) | 20.50 GiB | 10:30 (1:4 DeltaNet) | 0.31 GiB | 1.25 GiB | 5.00 GiB | 20.88 GiB | 21.81 GiB | 25.56 GiB |
| H27 Qwen3.8-27B (Q4_K_M) | 15.37 GiB | 16:48 (1:4 DeltaNet) | 1.00 GiB | 4.00 GiB | 16.00 GiB | 16.46 GiB | 19.46 GiB | 31.46 GiB |

Bold = exceeds the 8 GB CUDA card. Context columns are context length, not a claim every model
natively reaches 128k (D7/S declare 32k; G20/M35 declare 128k/256k natively).

**Hybrid/GQA check, as the brief asked:** M35 is exactly 10 of 40 full-attention layers (25.0%)
and H27 exactly 16 of 64 (25.0%) against each checkpoint's own `full_attention_interval: 4` —
"roughly a quarter" is exact. G20 and M26 get the same relief a different way: sliding-window
layers cap their own KV at the window size (128 / 1024 positions) regardless of total context, so
growth is confined to the 12-of-24 and 5-of-30 global layers.

**Weight-limited vs KV-limited:** M35/H27/M26 are weight-limited on both boxes before KV enters at
all (20.5/15.4/13.5 GiB). G20 is weight-limited on the 8 GB card outright, but a genuine borderline
KV case on the 16 GB Mac at its own *native* 131072-token ceiling (11.3 GiB weights + 6.0 GiB KV =
17.3 GiB). **D7 is the clean case:** weights (4.36 GiB) fit both boxes with enormous headroom, but
KV alone pushes the 8 GB card to 7.86 GiB at D7's own *native* 32k context — 98% utilized before
any matmul/activation scratch — and exceeds both boxes at 128k (YaRN-extended). S is a milder
version of the same shape, relevant only at 128k on the 8 GB card.

**Gate 0: not empty.** D7 (Qwen2.5-7B-Instruct) is the model — weights fit trivially, KV does not,
at a plausible (native) context length.

## Step 1 — measured on nobara (D7, CUDA, `int4`, driver 595.91.07)

### Backend selection gotcha (documented for whoever runs this next)

`decoder/decode_depth_bench_test.go`'s `BenchmarkDecodeAtDepth` does **not** select a backend —
`Options.Backend` defaults to `"cpu"` regardless of `-tags cuda`. A first attempt at this
measurement burned real wall-clock time running plain CPU decode of a 7B model under that
benchmark, believing the `-tags cuda` build tag alone was enough. The real CUDA path (resident or
staged) is only reachable through the actual server, `cuda/cmd/serve` (not root `cmd/serve`,
which refuses `-tags cuda` with its own compile-time guard pointing here), via `-backend cuda`.

### What the resident engine actually does (real code, not the flag text)

`cuda/resident.go`'s `checkKVFits` runs at load time against real free VRAM (`MemInfo()`). The
**default** resident cap is 4096 positions (`cudaCtxCapDefault`) — a deliberately conservative
*software* choice, not the VRAM ceiling; `-ctx` raises it, `min(model context window, request)`.

Measured decode-only tok/s (streaming, timed from first token after prefill to last — isolates
decode from prefill), D7 int4, resident, within cap:

| depth | resident tok/s |
|---|---|
| 512 | 69.89 |
| 2048 | 59.09 |
| 4096 | 48.91 |

A real, expected slope even *inside* the resident cap (more keys to attend per decode step as
depth grows) — not the failure mode in question, just the honest baseline.

### The real VRAM ceiling for D7 int4 on this 8 GB card

`-ctx 20000` **loads resident successfully** (7257/8192 MiB used, 935 MiB free — a cap the default
4096 leaves entirely on the table). `-ctx 24576` **fails** `checkKVFits`, with the exact real
message: *"resident context 24576 positions needs 2.82 GB of KV (112.0 KB/position across 28
layers) but only 2.90 GB is free on the device beside the weights (plus 384 MB reserved for driver
and decode scratch)."* True crossover for this exact model/quant/card sits between 20000 and
24576 — a clean, informative, load-time number, not an OOM crash.

### What actually happens past the cap — two different behaviors, and the flag text is wrong about one

1. **A per-request prompt beyond the *active* cap** (default 4096, or whatever `-ctx` set) is
   rejected with a clean HTTP 400 `context_length_exceeded`. Confirmed in `internal/serveapp/
   openai.go`, and confirmed BY DESIGN: a comment there cites `audit R-10` — an earlier version of
   this code *did* attempt something and produced a 500 leaking an internal "use the staged path"
   hint, and R-10 replaced it with the clean 400 specifically because **there is no staged
   fallback on the stateless resident path** at the per-request level. **The `-ctx` flag's own
   help text is stale**: it currently claims *"a request past the cap still fails cleanly and
   falls back to the staged path"* — real for the scenario below, not for this one. Small,
   free doc fix.

2. **A requested `-ctx` that itself doesn't fit VRAM** demotes the *entire model* to the CPU-staged
   path for every request, silently, unless `-require-backend` is passed (in which case the server
   refuses to start, naming the reason). Measured cost of this, D7 int4, same box, same quant,
   decode-only tok/s:

   | depth | resident | staged (fallback) | ratio |
   |---|---|---|---|
   | 512 | 69.89 | 4.69 | 14.9× |
   | 2048 | 59.09 | 3.97 | 14.9× |
   | 4096 | 48.91 | 3.20 | 15.3× |

   A strikingly consistent **~15× throughput cliff**, not a slope. Silent without
   `-require-backend`.

### A quant/CPU interaction found along the way, not chased further

The staged fallback inherits whatever `-quant` the resident config was using. `int4` on this box
(Zen 2, `lscpu` confirms no AVX-512/VNNI) gives the reasonable ~4-70 tok/s numbers above.
`int8int8` on the *same* CPU-staged path did not finish a depth-512 measurement in over 7 minutes
(killed) — consistent with `docs/task-simd-audit.md`'s canonical-int8-wants-VNNI /
split-half-int4-is-AVX2-only split (also the subject of `[[kernel-optin-must-check-best-tier]]`).
An operator who picked `int8int8` for the resident config (a real, plausible choice — it's the
`decode_bench_test.go` default) and then drifts past the VRAM ceiling gets a fallback that is not
just slow, it is closer to unusable. Not measured further — flagged for whoever picks this back
up.

## Gates

**Gate 0 — a KV-limited model exists: PASS.** D7, established above.

**Gate 1 — the bus has room: PASS, close to trivially.** D7 is dense — no MoE, no expert
streaming, nothing else competing for the PCIe bus. The brief's contention concern
(`docs/deltanet-residency-plan.md`'s ~48% expert-DMA share) is a non-issue for this model by
construction, not by measurement.

**Gate 2 — is the failure worth engineering around: does not clear, recommend park.**
Flagged honestly: the brief's own discipline calls for pre-registering the decision rule *before*
measuring, and that didn't happen here — this task arrived as a live investigation, and the
threshold below is being stated after the numbers, not before. Treat this verdict as reasoned, not
pre-registered.

The practical problem space a pager would close turns out to be narrower than the brief's framing
suggested, for two reasons found only by measuring:

- **Most of the theoretical gap is already free.** The default cap (4096) is a software choice,
  not a VRAM limit — `-ctx 20000` works today, no code change, on the exact card this was measured
  on. An operator hitting the 4096 ceiling has a zero-cost lever already available and, per the
  stale flag text above, may not know it goes that far.
- **Past the *real* ceiling (~20000–24576 here), the existing failure is not silent corruption
  or a crash — it is a named, informative wall** (a clean 400 for the request-level case, a
  named-GB refusal at load for the explicit-request case). The ~15× fallback cliff is real and
  worth fixing, but the fix already exists (`-require-backend`) and is an awareness/defaults
  question, not a missing capability.

What a pager would actually buy: turning the request-level 400 (rejecting a prompt between the
active cap and the model's true context window) into a served-but-slower response, and letting the
resident cap itself exceed the ~20-24k ceiling this card can hold today. Real, but narrow — it
only helps requests specifically in that window, which is a fraction of D7's 32k native context
and does not touch the 128k-and-beyond case (weights + full KV there already exceed both boxes
regardless of any streaming scheme, since streaming trades bandwidth for capacity, not for the
underlying byte count).

## Recommendation

**Park the pager.** Three cheap, unrelated fixes this investigation surfaced — done:

1. **Fixed** the `-ctx` flag help text (`internal/serveapp/main.go`) — it asserted a per-request
   staged fallback that `audit R-10` deliberately removed in favor of a clean 400. Also fixed the
   same stale claim in `docs/cuda-backend.md`, which had it too.
2. **Filed** the `int8int8`-on-non-VNNI finding where quant guidance actually lives:
   `docs/cuda-backend.md`'s new "CPU-staged fallback inherits the resident quant" section, next to
   the flag help's own per-backend quant caveats.
3. **Recorded** the true `-ctx` ceiling (20000 succeeds, 24576 fails, D7 int4, this card) and
   answered the question directly: 4096 is a round, conservative default, not a value ever tuned
   against real VRAM headroom — `cuda/resident.go`'s `cudaCtxCapDefault` comment now says so, with
   the measured ceiling as a concrete reference point. Same content mirrored into
   `docs/cuda-backend.md` and the flag help.

Re-open this experiment if a model surfaces that is *specifically* weight-fitting with a native
context whose KV lands squarely between the software default and the true per-card ceiling for a
real deployment someone wants today — that is the exact window a pager would serve, and it did not
show up broadly across the model set checked in Step 0.
