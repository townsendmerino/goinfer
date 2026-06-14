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

## The ideas (the menu)

> These are written as a catalogue, grouped loosely by leverage. **The
> authoritative sequencing is the dependency + trigger sections at the end**, not
> the order here — in particular #1 reads first but is the *hardest*, late item
> once you account for the prerequisite structure.

### High leverage on the resident weight footprint

### 1. Zero-copy GGUF: score native K-quant blocks from the mmap (kill the heap requant)

**The win:** today `.giw` is the only zero-copy weight path; a plain GGUF pays
full heap for its quantized weights (fact 1). If `MatmulBT` could score directly
against the GGUF's native Q4_K / Q6_K / Q8_0 **super-block layout** read
zero-copy from the mmap, a GGUF model would cost ~0 heap for its weights — the
`.giw` RAM win, but for the format everyone already has, with no extra build step
and no second file. For a 7 B Q4_K model that's ~4 GB of heap that simply
disappears (moves to reclaimable page cache shared with the OS).

**What it actually takes (corrected — this is the hardest item, not a medium
one):** the existing kernels *dequant* a K-quant block → f32; this needs a
different thing, a **scoring** kernel (block · activation → output *without*
expanding the block). Those are not the same kernel, and it isn't one kernel —
it's **one per K-quant type** (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q4_0, Q8_0…), each
parity-gated, each carrying llama.cpp's fiddly super-block layout (Q4_K's 6-bit
scale packing is its own small nightmare). That's a **parallel quant
representation** standing alongside goinfer's flat per-row int4/int8 — real,
permanent maintenance surface, not a reuse of what's there.

**And the trade is real, not free.** "Need not be faster" undersells it: scoring
from the block means you **re-dequant every block every token** — you're trading
heap RAM for per-token ALU. On a stream-bound decode that's *plausibly* neutral
(fewer bytes read from the working set, more ALU per byte), but it must be stated
as the trade it is and **measured**, not waved off. It could regress decode on an
ALU-bound CPU path.

**Cost / risk:** **HIGH.** Behind the same lazy-fallback discipline as `.giw`
(any block-format mismatch → today's dequant-to-heap path). **Gate:** argmax +
cosine vs the current heap-dequant path on the Q4_K / Q6_K goldens already in the
repo, **plus a decode tok/s non-regression bar** (the ALU trade above). This is
the program's foundation-generalization step, not its opener — see sequencing.

### 2. MoE inference-time expert demand-paging (run big MoE on less RAM)  ✅ SHIPPED (2026-06-13)

**Shipped** as `serve --stream-weights` + `--weight-cache <GB>` (0 = auto). Built on
the mmap substrate (Inc 1, below): a `.giw` MoE keeps its experts in the read-only
mapping; the router's top-k drives an LRU bounded by the budget, releasing the tail
with `MADV_DONTNEED` and faulting misses with `MADV_WILLNEED`. Bit-exact by
construction — the read-only file-backed mapping re-faults an evicted-then-reused
expert with identical bytes (proven model-free by `TestMadvise_dontneedRefaultsIntact`;
LRU policy by `TestExpertPager_lruEviction`; end-to-end paged==full bit-identity by
`TestExpertPaging_bitExact`, asset-gated on `GOINFER_MOE_GIW`). Page-granular, so
sub-page experts aren't paged (irrelevant for the multi-MB real target). Hook seam:
the `topK` selection in `moeMLP` (`mlp.go`), where the spike's `moeSelTrace` sat.
**Substrate Inc 1 (mmap the `.giw` load) shipped first** — `.giw` was zero-per-tensor-
copy but `os.ReadFile`-heap (not pageable); now `MAP_PRIVATE` read-only, bit-exact.

**Validated end-to-end on the real 35B-A3B (2026-06-13).** Three enablers were
needed first: the mmap substrate (Inc 1); streaming `.giw` serialization
(`SerializeWeightsTo` + `giw.WriteStream`) so the 35B prequantizes at ~20 GB peak
instead of OOMing; and `.giw` format **v2**, which serializes the `qwen3_5_moe`
DeltaNet-hybrid's per-layer `delta`/`qattn` tensors (the 35B-A3B *is* that arch — it
couldn't round-trip through `.giw` before, segfaulting on nil delta). Result, at a
512 MB expert cache vs 16 GB of experts: `hits=4706 misses=5534 evictions=5190`,
decode **byte-identical** to fully resident over 24 tokens — i.e. `MADV_DONTNEED` on
the read-only mapping is provably lossless even when thousands of in-use experts are
evicted and re-faulted mid-decode.


**The win:** a 35B-A3B activates ~3 B params/token but resides at ~39 GB (fact
2). Keep the expert weights **mmap'd and not heap-resident**, fault in only the
experts the router selects per token, and hold a small **LRU of hot experts** in
heap (quantized). Resident RAM drops from "all experts" toward "active + hot set"
— a measured **~2× for the 35B-A3B class** at an interactive operating point (see
the spike numbers below), turning a model that needs a ~40 GB box into one that
runs on ~16–20 GB. Not the "10×" originally guessed here — the skew is real but
finite, and the floor is the active set K·L, not a handful of hot experts.

**Why it's plausible:** the router already produces the per-token expert
selection before the expert matmuls run, so the demand signal exists at exactly
the right seam. Experts are independent `expertWeights` bundles — page-able
units. Expert-activation skew (a minority of experts dominate) means a modest LRU
gets a high hit rate; the cold misses are an mmap fault, bounded by NVMe
bandwidth.

**Measured (2026-06-13 viability spike — `decoder/moepaging_spike_test.go`,
real Qwen3.6-35B-A3B, int8, 200 generated tokens, 40 layers × 256 experts ×
top-8, expert ≈ 3.1 MB, all-experts ≈ 32 GB):** the skew is real but finite —
the hottest 10% of experts absorb 72% of accesses, hottest 25% absorb 94%, and
only 50% of the universe is ever touched. The active set (the unavoidable
resident floor) is K·L = 320 experts ≈ 1.0 GB. Whole-expert LRU, hit-rate /
added-latency vs cache budget (NVMe ≈ 20 µs seek + 3 GB/s):

| cache | pages | hit % | miss/tok | +ms/tok |
|------:|------:|------:|---------:|--------:|
| 3 GB  |   960 |  51.3 |    155.8 |   166.4 |
| 8 GB  |  2560 |  89.6 |     33.2 |    35.5 |
| 16 GB |  5120 |  92.9 |     22.7 |    24.3 |

So the realistic interactive operating point is ~16 GB cache → **93% hit,
≈ +24 ms/token** (≈ ½ the experts resident, ~2× RAM reduction). Below ~8 GB the
miss tail dominates and it stops being interactive. The cold-miss cost is
**bandwidth-bound and unhideable** — the router selects just-in-time, so there's
no prefetch lead. Build to *this* point, not the dead "10×".

**Cost / risk:** medium-high. Token latency becomes hit-rate-dependent (a cold
expert = a page-fault stall), so it's an **opt-in `--moe-page` mode** for the
RAM-bound case, not the default. Needs: an expert-residency LRU, `madvise`
hints (the spike's curve already stands in for the measurement). **The
highest-capability idea here** — it's the thing that makes the largest models in
the zoo run on less RAM, and MoE is where the field is going (Qwen3.6, DeepSeek
V4, GLM-4 MoE, Mellum2 all in the roadmap). The hook seam is `mlp.go`'s router
selection (the `idx` returned by `topK` in `moeMLP`); the spike's `moeSelTrace`
sits exactly there.

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
stack**.

**The mechanism problem (must be solved before this is real).** The obvious pitch
— "int4 on the tolerant gather, Q6_K on the logit-critical head" — **does not work
for a tied model**, which is the common case here (Gemma, small Qwen). A tied
embedding is *one array*; it can't be int4-for-the-gather and higher-for-the-head
at once without either storing it twice (no RAM win) or storing it at int4 and
letting the head read int4 (the exact thing the `embedding()` policy forbids
because it flips logits). The clean two-precision split only exists for **untied**
big-vocab models (two tensors) — and those are rarer. So the honest version is one
of: (a) scope this to untied models only, accept the narrower reach; or (b) a
tied-compatible **frequency row-blocking** scheme (the strongest version, from the
external review). Store the **hot head of the vocab** (the few-thousand most-used
token rows — which carry the large majority of the token stream) at int8/f16 and
the **long tail** at int4, in one logical table; the gather indexes by token id
into whichever tier the row lives in, and the head matmul becomes two
concatenated matmuls (int8 block ‖ int4 block). One array's worth of RAM, tiered —
no double storage, and the tied head is protected where it matters.

**The nuance that decides whether (b) is real:** "top-k tokens = 90% of the
stream" is an *input/gather* frequency stat — and the gather is the tolerant path
anyway. The *head* dots the hidden state against **every** row, tail included, so
a correct-but-rare next token still gets ranked from its int4 row. The bet is that
the argmax rarely lands on a tail row and is "good enough" when it does — plausible
(it's the same class of bet as the existing `embedding()` int8 pin), but it must be
gated on **rare-token continuations specifically**, not average-case argmax, or the
tail-token regressions hide in the aggregate.

**Cost / risk:** medium for (a); medium + a frequency permutation / per-row tier
map for (b). Rides directly on the logit-parity gate the `embedding()` policy was
written to protect. **Gate:** argmax + cosine on the per-model goldens **plus a
rare-token-continuation stress set** (the (b)-specific bar above). **Trigger:** a
big-vocab small model (256 k-class vocab, ≤4 B) where the int8 embed/head table is
measured as the dominant resident tensor — i.e. the "LLM-in-one-file" footprint
actually bites.

---

### Bigger swings — capability unlocks

### 4. Bigger-than-RAM weight streaming (run a 70B on a 16 GB laptop)  ✅ SHIPPED (2026-06-13)

**Shipped** as `serve --stream-weights` on a dense `.giw` (the same flag picks expert
paging for MoE, layer streaming for dense). A `layerPager` (decoder/layerpaging.go)
slides a window over the `runLayersFromEmbed` / `runLayersFromEmbedN` layer loop:
because the order is known, it `MADV_WILLNEED`s the next layer while the current one
computes (overlapping the fault — the latency hiding MoE's just-in-time router can't
do) and `MADV_DONTNEED`s the layer sliding out the back. Resident weight RAM is
bounded to **floor + window** (sized by `--weight-cache`) — *floor* because only the
7 per-layer projections stream; embed / final-norm / LM-head aren't per-layer and
stay resident, so for a big-vocab model that floor is multi-GB on its own (the
complementary lever is #3 sub-int8 embed/head — "bigger than RAM" is not "≈ 0 RAM").
The latency floor is NVMe bandwidth (a model that doesn't fit is re-read ~once per
token). Bit-exact by the
read-only-re-fault property — gated by `TestLayerPaging_bitExact` (3-layer window,
decode byte-identical to fully resident, prefetches + evictions > 0). Built on the
mmap substrate (Inc 1); reuses the expert pager's `alignedMappedSpan` + `madviseBytes`.

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

### Serve-side density (multi-tenant footprint — from the external review)

These two attack a different RAM axis than #1–#6: not one model's resident
weights, but the **multiplier from serving many tenants** (adapters, sessions).
Both ride the same zero-copy base-residency substrate as #2/#4.

### 7. Multi-adapter weight sharing (don't merge LoRA — add it at compute time)

**The win:** today LoRA is *"merged at load"* (`lora.go`), so serving one base
model with N fine-tunes pays the full transformer RAM **N times over** — 5
adapters on a 7 B = ~20 GB. Keep the base **immutable and zero-copy mmap'd**
(the substrate), store each adapter's low-rank `A`/`B` separately, and apply the
delta in the forward: `Y = W_base·x + (B(A·x))·s`. Five adapters then cost
~base + 5 small deltas ≈ **~4.2 GB instead of ~20 GB.**

**Two corrections to the framing (vs. how it was pitched):**
- **RAM-density only, not throughput.** S-LoRA's headline is many adapters *in one
  batch* via segmented GEMM; goinfer's serve is **single-decode-worker-per-model,
  no continuous batching** (deferred, roadmap). So the win is "N adapters share one
  resident base," not concurrent high-density serving — real, but it's a footprint
  play, consistent with this doc's whole thesis.
- **Cost is medium-**_high_**, not medium.** The delta is an f32/f16 low-rank
  branch that must fuse onto goinfer's *quantized* matmul seam (int4/W4A8/int8 base
  + f32 add), on the hot path in every arch's forward. Merged-LoRA stays the
  faster, simpler default; this is the opt-in RAM-vs-speed knob (house rule).

**Cost / risk:** medium-high. **Gate:** bit-exact vs the merged-adapter path
(the add is exact); decode tok/s non-regression (the extra low-rank pass).
**Trigger:** `cmd/serve` is asked to host several adapters of one base and goes
RAM-bound on the duplicated transformer.

### 8. Tiered KV: demote idle warm sessions RAM → NVMe  ✅ SHIPPED (2026-06-13)

**Shipped** as `serve --kv-idle-demote <dur>` (+ `--kv-demoted-max`, default 64).
A background sweep demotes any resident session idle past the threshold to a
`.giw-kv` blob under `--session-dir` and frees its RAM; capacity evictions tier the
coldest to disk instead of discarding it; the next continuation faults it back
**byte-identically** to a cold prefill. Pure serve-layer policy over the existing
persistence — no decoder changes — exactly as scoped below. The cold tier is
in-process (cross-restart cold-tier persistence is a possible Inc 2). Gated by the
model-driven `TestServe_tieredKVDemoteFaultBack` (demote → fault-back continuation
== cold prefill) + the model-free `TestColdTierOverflow`.


**The win:** `--kv-sessions` pins RAM for every warm conversation. The
serialization to do something about it **already exists** — `--session-dir`
persists warm sessions to disk via `.giw-kv` and restores them exactly. The new
piece is only the **policy**: an idle-LRU that demotes a session's KV to disk after
N minutes untouched and faults it back on the next request, instead of holding all
of them resident. A small-RAM box then serves **hundreds of intermittent** chats
without OOMing.

**Cost / risk:** low — mostly policy plumbing in serve's `sessionLRU`; composes
with the shipped int8 KV (already 4× smaller on disk) and the exact `.giw-kv`
restore. The only real cost is fault-back latency (deserialize on resume), which is
exactly what an idle threshold trades against. Scope it as *"automatic tiering over
the existing persistence,"* not new machinery. **Trigger:** warm-session RAM is the
serve OOM cause (many intermittent tenants on a small box).

### Assessed and rejected (keep the record, KV-program style)

- **Cross-model tensor dedup (content-addressable weights)** — *rejected.* The
  premise that sibling fine-tunes (e.g. Coder vs Instruct) share **bit-identical**
  tensors is mostly false: instruction-tuning perturbs ~all layers. The cases where
  it holds are already covered — the *same file* loaded twice shares pages via the
  **OS page cache** (we mmap), and "one base, many adapters" is #7 done properly.
  So it pays a real load-time hashing cost (hashing GB of weights) for an unlikely
  cross-file hit rate. Only salvageable as an explicit base+delta *packaging*
  format — a niche choice, not a general win.

---

## The dependency that reorders everything (corrects the first draft)

The first draft led with #1 (zero-copy GGUF) and called #2/#4 "share machinery."
That's wrong, and the relationship is the most important thing in this doc:
**#2 (MoE paging) and #4 (streaming) require zero-copy weight residency as a hard
prerequisite.** You cannot demand-page or stream a heap-resident `WeightMat` — it
is *already in heap*; that's the thing you're trying to avoid. Paging/streaming
only mean anything when the weights stay mmap'd and fault in on demand. **Today
exactly one path provides that substrate: `.giw`** (the zero-copy aliasing in
`serialize.go`). #1 is not the opener — it's the step that *generalizes that
substrate to plain GGUF*, and it's the hardest item in the program.

So the marquee capability (#2/#4) does **not** have to wait behind the expensive
kernel work. Prove it on the substrate that already exists:

**Phase 1 — capability, on the `.giw` substrate that ships today (the headline).**
Build #2 (MoE expert demand-paging) and/or #4 (dense weight streaming) on `.giw`'s
existing zero-copy mmap. This front-loads the marquee — *"run a 35B-A3B / a 70B on
a 16–24 GB laptop, single static Go binary, no install, offline"* — and de-risks
the program **without writing a single K-quant scoring kernel.** Users prequant to
`.giw` (the path the multi-model zoo already wants).

**Phase 2 — generalize the substrate to plain GGUF (#1, HIGH cost).** Only once
the capability is proven and wanted, bring it to the format users already have, by
making GGUF zero-copy too (the per-K-quant scoring kernels). This is the right
place for the expensive, parallel-quant-representation work — *after* it has a
proven payoff pulling it, not gating the whole program behind it.

**Parallel / independent of the substrate:**
- **#5 mixed precision** — the safe incremental tuner; the `.giw` header already
  represents per-tensor kind, so it's a packer + scorer. Ships anytime.
- **#3 sub-int8 embed/head** — narrowed, and blocked on the tied-tensor mechanism
  above; do (a) untied-only or solve (b) first.
- **#6 rotation 3-bit weights** — the lone research-risk item; timeboxed spike,
  written go/no-go, **last.** Compounds with #2/#4 (smaller weights → more
  experts/layers per RAM byte → higher hit rate / streamed tok/s).

## Triggers (the KV program had these; this one needs them too)

The first draft was "the roadmap watches weights, so here's the program" with no
firing condition. Explicit triggers, so this is a *scoped program* and not a
standing menu:

- **#2 (MoE paging) fires when:** a big-MoE model in the roadmap pipeline
  (DeepSeek V4, GLM-4 MoE, and the already-landed Qwen3.6-35B-A3B / Mellum2)
  becomes something an adopter wants to run **under its all-resident footprint** —
  e.g. "35B-A3B on a 24 GB box," or the multi-model zoo going RAM-bound on stacked
  MoE.
- **#4 (streaming) fires when:** a concrete *bigger-dense-than-RAM* ask appears —
  "run a 70B/dense-class off NVMe on a 16 GB laptop." Without that ask it stays a
  positioning idea, not scheduled work.
- **#1 (zero-copy GGUF) fires when:** #2 or #4 is proven on `.giw` **and** the
  "but my model is a plain GGUF" friction is real (i.e. the prequant-to-`.giw`
  step is the adoption blocker).
- **#5 / #3 / #6:** #5 on a measured "uniform int8 leaves quality/size on the
  table" case; #3 on the big-vocab-small-model footprint biting (its own trigger
  above); #6 only as a deliberate research spike when the proven rungs are in —
  same discipline as the TurboQuant KV spike.

## Scope decision (2026-06-13): build the shared substrate

The scoping question — *MoE-on-small-RAM (#2) or bigger-dense-than-RAM (#4)?* —
was answered **both**. So the program does **not** collapse to one headline; it
collapses to **the substrate they share**, built once, with #2 and #4 as its two
specializations. That's the spine:

**The shared substrate — zero-copy, demand-resident, layer-sequential weights.**
Weights stay mmap'd (the `.giw` aliasing, generalized), never fully heap-resident;
the forward's existing layer-sequential loop drives residency on demand; RSS is
bounded by a working set, not the whole model. Everything below is one of two
faces of this:

- **#2 = the MoE face.** The demand signal is the router (runs *before* the expert
  matmuls — the seam is already in the forward): fault in selected experts, hold a
  hot-expert LRU. Headline: *35B-A3B on a 24 GB box.* **Implementation seam:** the
  residency hook belongs at `moeMLP`'s router output (`idx, wts := topK(probs, k)`,
  `mlp.go:69`), where the **whole selected set** for the layer is known — prefetch
  that set (and ideally the next layer's likely experts) in one shot, *before* the
  expert loop dereferences `&lw.Experts[e]`. **Not** a per-expert, error-returning
  call inside the loop after the matmul (that faults the expert *after* using it and
  stalls serially on each cold one); branch once on whether a pager is attached so
  the default forward stays byte-for-byte unchanged, and carry residency state on
  the pager/model, not on `LayerWeights`.
- **#4 = the dense face.** The demand signal is layer order: prefetch layer L+1
  while computing L (`MADV_WILLNEED` / a prefetch goroutine), let consumed layers
  reclaim. Headline: *a 70B off NVMe on a 16 GB laptop.*

They are the **same machinery** (mmap residency + a prefetch/eviction policy over
the layer-sequential loop) with two demand sources — which is *why* "both" is
coherent rather than double the work: build the residency core + policy interface
once, then the router-driven and order-driven policies are small specializations.

**Revised plan of record:**

1. **Substrate core on `.giw`** — mmap-resident weights + a prefetch/eviction
   policy seam over the layer loop. No new kernels. (Phase 1.)
2. **#4 dense streaming policy** (order-driven prefetch) and **#2 MoE paging
   policy** (router-driven + hot LRU) as the two policies on that seam — sequence
   by whichever adopter ask lands first; both are in scope.
3. **#1 zero-copy GGUF** — generalize the substrate off `.giw` to plain GGUF (the
   HIGH-cost per-K-quant scoring kernels), once the capability is proven and the
   "my model is a GGUF" friction is the live blocker. (Phase 2.)
4. **#5 mixed precision** in parallel (tuner, ships anytime); **#6 rotation 3-bit**
   the timeboxed research spike, last — it compounds with the substrate (smaller
   weights → higher hit rate / streamed tok/s); **#3** as its own narrow,
   trigger-gated item.
5. **Serve-side density (#7 LoRA sharing, #8 tiered KV)** — a separate axis from
   the substrate (multi-tenant multiplier, not one model's footprint). **#8 is the
   cheapest win in the whole doc** (policy over existing `.giw-kv` persistence) and
   can land independently; **#7** rides the same immutable-base residency as the
   substrate and fires when `cmd/serve` goes adapter-RAM-bound.

## What this is *not* (keeping the house rules)

- Default decode stays **bit-exact**: #1 falls back to heap-dequant on any
  block-format mismatch; #2/#4 are opt-in `--moe-page` / `--stream-weights`;
  #5/#6 are build-time `.giw` precisions behind their own gates; #7 is bit-exact
  vs. the merged-adapter path (the low-rank add is exact); #8 restores KV exactly
  from `.giw-kv`. Merged-LoRA stays the default — #7 is the opt-in density knob.
- Not chasing kernel throughput (the roadmap's standing filter) — every idea here
  is a **footprint/capability** play, not a tok/s record. The one that *also*
  helps speed (#1/#3, smaller working set on a weight-stream-bound decode) does so
  as a side effect, not the pitch.
- Reuses the existing seams: the layer-sequential forward, the zero-copy `.giw`
  aliasing, the per-`weightMat` kind tag, the lazy-fallback + parity-gate
  discipline. Minimal new permanent surface for the resident-RAM ceiling it lifts.
