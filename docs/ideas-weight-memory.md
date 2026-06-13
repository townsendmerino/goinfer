# Ideas: weight-memory reduction (the next program after KV)

> **Audience:** internal brainstorming, written in the planning-doc style of
> `completed/task-memory-program.md` / `roadmap.md`. The KV-cache program (rings + int8 +
> GPU int8) is shipped/scoped; **weights are the larger, less-attacked half of
> resident RAM** and the roadmap only watches them (TurboQuant "weights angle =
> watch", `.giw` f16 scales = "lowest value of all"). This doc proposes the
> weight-side program: same house rule — default path stays bit-exact, every
> lossy step is an opt-in knob with a written gate.

## Where the RAM actually is today (the grounding)

Three facts from the current code set the opportunity:

1. **GGUF / safetensors weights are heap-resident after load.** `gguf.go` mmaps
   the file but the load is a per-tensor **dequant + re-quant into heap**
   `WeightMat`s; `weights.go` mmaps safetensors but dequants into heap too. Only
   `.giw` (`serialize.go`) aliases the int8/int4 arrays **zero-copy over the
   mmap** — which is exactly why the README says *"Resident int8 models are
   expensive — prequant `.giw` maps weights zero-copy for a cheap zoo."* So for
   the common case (a user points at their GGUF), the quantized weights live in
   heap **twice over**: once as the mmap'd source page cache, once as the heap
   `WeightMat`.
2. **MoE keeps every expert resident.** `LayerWeights.Experts []expertWeights`
   holds all experts; Qwen3.6-35B-A3B is *"~39 GB resident"* even though only
   ~3 B params activate per token. The biggest resident-RAM models in the zoo are
   the ones using the least of their weights per step.
3. **Decode is weight-stream-bound.** The whole GPU assessment + CPU prefill work
   established the token cost is dominated by *reading the weights* (91% of a GPU
   token is re-reading 1.55 GB of resident weights). The weights are the working
   set; shrinking or paging them is the lever that also helps the bottleneck.

The KV program bought ~20× on Gemma-class KV. **None of it touched the weight
footprint**, which for a small-context single-user session is the *majority* of
RSS. These ideas attack that.

---

## Tier 1 — high leverage, uses kernels we already ship

### 1. Zero-copy GGUF: score native K-quant blocks from the mmap (kill the heap requant)

**The win:** today `.giw` is the only zero-copy weight path; a plain GGUF pays
full heap for its quantized weights (fact 1). If `MatmulBT` could score directly
against the GGUF's native Q4_K / Q6_K / Q8_0 **super-block layout** read
zero-copy from the mmap, a GGUF model would cost ~0 heap for its weights — the
`.giw` RAM win, but for the format everyone already has, with no extra build step
and no second file. For a 7 B Q4_K model that's ~4 GB of heap that simply
disappears (moves to reclaimable page cache shared with the OS).

**Why it's plausible:** the dequant kernels for these block formats already exist
(aikit K-quant dequant, fuzzed). The new piece is a *scoring* kernel that
consumes a super-block (sub-scales + packed nibbles) without first expanding it —
i.e. a `MatmulBT`-shaped path over Q4_K blocks, analogous to the existing
`MatmulBTW4A8` but reading the llama.cpp block struct instead of goinfer's
per-row int4. Decode is ALU-bound, so this need not be *faster*; it just has to
hit the parity gate while reading from the alias.

**Cost / risk:** medium. K-quant super-blocks (sub-block scales, min, the Q4_K
6-bit scale packing) are fiddlier than goinfer's flat per-row int8/int4, and
you'd want it behind the same lazy-fallback discipline as `.giw` (mismatch →
dequant-to-heap path). But it's the single biggest weight-RAM win for the most
common input. **Gate:** argmax + cosine vs the current heap-dequant path, on the
Q4_K / Q6_K goldens already in the repo.

### 2. MoE inference-time expert demand-paging (the 10× for big MoE)

**The win:** a 35B-A3B activates ~3 B params/token but resides at ~39 GB (fact
2). Keep the expert weights **mmap'd and not heap-resident**, fault in only the
experts the router selects per token, and hold a small **LRU of hot experts** in
heap (quantized). Resident RAM drops from "all experts" toward "active + hot set"
— potentially **~10× for the 35B-A3B class** — turning models that need a 64 GB
box into ones that run on 16–24 GB.

**Why it's plausible:** the router already produces the per-token expert
selection before the expert matmuls run, so the demand signal exists at exactly
the right seam. Experts are independent `expertWeights` bundles — page-able
units. Expert-activation skew (a minority of experts dominate) means a modest LRU
gets a high hit rate; the cold misses are an mmap fault, bounded by NVMe
bandwidth.

**Cost / risk:** medium-high. Token latency becomes hit-rate-dependent (a cold
expert = a page-fault stall), so it's an **opt-in `--moe-page` mode** for the
RAM-bound case, not the default. Needs: an expert-residency LRU, `madvise`
hints, and a measured hit-rate/latency curve on a real 35B-A3B. **This is the
single highest-capability idea here** — it's the only thing that makes the
largest models in the zoo run on consumer RAM, and MoE is where the field is
going (Qwen3.6, DeepSeek V4, GLM-4 MoE, Mellum2 all in the roadmap).

### 3. Sub-int8 embedding/head for big-vocab small models (the *narrowed* idea — the easy version is already done)

**Premise check (done — the naive version doesn't exist):** I traced this and
**both loaders already quantize the embedding and LM head.** `weights.go`
(safetensors) calls `quantizeWM(w.Embed, quant.embedding())` and the same for the
untied head; `gguf.go` routes them through an `embMat` helper on the same policy;
Gemma-4 PLE tables go through it too. The policy `quantMode.embedding()` is
deliberate: int8 in int8 mode, **pinned to int8 even in int4 mode** because the
tied head dots every logit against this table (int4 there flips the argmax —
mirroring how GGUF Q4_K_M keeps `token_embd`/`output` at Q6_K), and f32 only when
the whole model is f32. So "the embed table is silently left f32" is **not**
happening — scratch the easy win.

**The real residual lever:** that int8 pin is exactly what makes the table the
**single largest resident tensor for a big-vocab small model at int4.** A 2 B
model at int4 has its transformer at ~⅛ f32, but a 256 k × hidden embedding held
at int8 (¼ f32) — order ~0.5 GB — can still **outweigh the entire transformer
stack**. The opportunity is a *careful* sub-int8 scheme for this one table that
doesn't flip logits: e.g. int4 on the **input-embedding read path** (a gather, far
more tolerant) while keeping the **tied-head matmul** at the higher-precision
"Q6_K output" idiom — i.e. split the precision of the two roles the tied tensor
plays, instead of pinning both to int8 for the head's sake.

**Cost / risk:** medium (up from "low" — it's no longer a free quantize). It needs
the tied-tensor split (two precisions, one storage) and rides directly on the
logit-parity gate that the `embedding()` policy was written to protect. **Gate:**
argmax + cosine on the per-model goldens, same bar the current int8 pin clears.
Highest payoff on exactly the "LLM-in-one-file" small models the README leads
with — where this table is the footprint.

---

## Tier 2 — bigger swings, real capability unlocks

### 4. Bigger-than-RAM weight streaming (run a 70B on a 16 GB laptop)

**The win:** decode touches each weight **once per token, in strict layer
order** (fact 3). That access pattern is the ideal case for streaming: mmap the
quantized weights, process layer-by-layer, and **prefetch layer L+1 while
computing layer L** (`MADV_WILLNEED` / a prefetch goroutine), letting the OS
reclaim already-consumed layers. The model never has to be RAM-resident in full —
RSS is bounded by ~a few layers' working set, and throughput is bounded by disk
bandwidth (NVMe ~5 GB/s ÷ a 20 GB int4 model ≈ **~4 tok/s for a model that
otherwise doesn't load at all**).

**Why it fits goinfer specifically:** the forward is already a clean
layer-sequential loop over independent `LayerWeights`, and the mmap loaders +
zero-copy `.giw` aliasing are the exact substrate. No other pure-Go runtime can
offer "single static binary, no install, runs a model 3× your RAM off an SSD."
That's a *positioning* win, not just a feature — it extends the README's
"LLM-in-one-file / runs offline" lane to "runs the big one offline too."

**Cost / risk:** medium. Prefill (which touches weights L² / batched) is less
streaming-friendly than decode — likely cap the streamed mode to modest prompts
or stream prefill too at a TTFT cost. Pairs naturally with #2 (MoE paging is a
special case of the same machinery). **Opt-in `--stream-weights`**, with a
measured tok/s-vs-RAM curve as the deliverable.

### 5. Calibration-driven per-tensor mixed precision

**The win:** uniform int8 (or int4) leaves quality on the table on tolerant
tensors and risks it on sensitive ones. A short **calibration pass** (a few
hundred tokens) scores each tensor's quant sensitivity, then packs each at the
lowest precision that holds the parity gate — sensitive layers (first/last,
attention out-proj) at int8, tolerant bulk (MLP up/gate) at int4/int3. Net
resident weights land **below uniform int8 at uniform-int8 quality**, tuned per
checkpoint.

**Why it's plausible:** every kernel needed (int8, int4, W4A8) already ships and
is parity-gated; the only new code is the calibration scorer and a per-tensor
precision tag in the `.giw` header (which already carries per-`weightMat` kind —
`uint8 kind 0/1/2/3`, so *mixed precision is already representable in the format*;
this is a packer + a chooser, not a format change). Offline, so **zero decode
cost**.

**Cost / risk:** medium. The research risk is the sensitivity metric (MSE-on-
calibration vs. end-to-end argmax). Timebox the scorer; the fallback is uniform
quant, which we already ship. **Gate:** the per-model goldens, plus "mixed ≤
uniform-int8 size AND ≥ uniform-int8 cosine."

### 6. Rotation/incoherence 3-bit weights at `.giw` build time (the TurboQuant weights angle, de-risked)

**The win:** the roadmap dismisses 3-bit weights ("weak ROI, no decode speed,
permanent format branch") — but that verdict predates the insight from the
TurboQuant KV spike: a **Hadamard/randomized-rotation incoherence pass** before
low-bit rounding is what makes 3-bit *near-lossless*. The same trick that the
spike points at KV applies to weights (this is the QuIP# family). Critically, the
rotation can be **folded into adjacent matmuls offline** at the `.giw` build step,
so it adds **no decode-time cost** — the objection that killed plain 3-bit. Result:
~25% smaller resident weights than int4 at int4-or-better quality, which is the
difference between a 13 B fitting where only a 7 B did.

**Why now vs. the roadmap's "watch":** the ROI flips precisely in the
RAM-constrained case — and #2/#4 above are *about* the RAM-constrained case. 3-bit
weights compound with MoE paging and weight streaming (smaller pages → more
experts/layers per RAM budget, higher hit rate, higher streamed tok/s). It stops
being "25% off a file" and becomes a capability multiplier on the other ideas.

**Cost / risk:** high — this is the one genuine research item, same tier as the
TurboQuant KV spike. **Treat it identically:** 2–3 day reading + CPU-prototype
spike, written go/no-go bar (≥ int4's measured cosine at 3 bits, or it closes).
Sequence it *after* #1–#4 ship, so the cheap wins land first.

---

## Suggested order (mirrors the KV program's "free/cheap before research")

1. **#1 zero-copy GGUF** — biggest win for the most common input; reuses dequant
   kernels + `.giw` fallback discipline.
2. **#2 MoE expert paging** — highest *capability* unlock; the only thing that
   runs 35B-A3B-class on consumer RAM.
3. **#4 weight streaming** — shares machinery with #2; the "bigger than RAM"
   positioning win.
4. **#5 mixed precision** — the `.giw` header already represents it; a packer +
   scorer.
5. **#3 sub-int8 embed/head** — narrowed (the easy quantize is already shipped);
   the tied-tensor precision split is the residual lever for big-vocab small
   models, on the logit gate.
6. **#6 rotation 3-bit weights** — the lone research-risk item; spike last,
   written go/no-go, compounds with #2 and #4.

## What this is *not* (keeping the house rules)

- Default decode stays **bit-exact**: #1 falls back to heap-dequant on any
  block-format mismatch; #2/#4 are opt-in `--moe-page` / `--stream-weights`;
  #5/#6 are build-time `.giw` precisions behind their own gates.
- Not chasing kernel throughput (the roadmap's standing filter) — every idea here
  is a **footprint/capability** play, not a tok/s record. The one that *also*
  helps speed (#1/#3, smaller working set on a weight-stream-bound decode) does so
  as a side effect, not the pitch.
- Reuses the existing seams: the layer-sequential forward, the zero-copy `.giw`
  aliasing, the per-`weightMat` kind tag, the lazy-fallback + parity-gate
  discipline. Minimal new permanent surface for the resident-RAM ceiling it lifts.
