# aikit task: a sub-range async upload, because profiling now shows `Buffer.upload` on a hot path

> **STATUS: OPEN and UNANSWERED as of 2026-08-21.** aikit/gpu has cut **v0.29.0** since this was
> written and `Buffer.upload` there still ends in a full device `Synchronize`; no
> sub-range async API exists in any cached version. So this has neither been built nor explicitly
> declined. **Declining is a valid outcome** — the prompt sizes the prize at 1.18× on one workload
> and says so — but the decision has not been made, and an un-refused ask is not the same as a
> refused one. (Separately: goinfer pins aikit/gpu v0.28.0 while v0.29.0 exists.)
>
> **DECIDED 2026-08-28 — DECLINED as scoped; see the DECISION section at the foot of this file.**
> The real prize is 5.6%, not 1.18× (async does not make PCIe faster), it is inside this workload's
> own demonstrated noise, and the offset+async primitive does not exist in gocudrv so the cost is a
> dependency change rather than a wrapper. A batched upload — N copies, one synchronize — captures
> the whole 3.6 ms using primitives that already exist, with no async and no caller stream contract.
> (Pin note now stale twice over: aikit/gpu is at v0.31.0.)


## Why this task (read first)

`gpu/cuda.go`'s `Buffer.upload` ends with a **full device `Synchronize`**, and its own comment
already anticipated this request:

> The synchronize makes the copy complete before we return. It is a full device sync, which is
> heavier than ideal, but uploads are per-request (a corpus, a tower, a batch of patches) rather
> than per-op, and a correct upload is not negotiable. **A cheaper fix — an async copy on the
> queue's own stream — needs pinned host memory and a queue-aware Buffer; worth doing if profiling
> ever shows this on a hot path.**

**Profiling now shows it on a hot path, and both stated preconditions are already met.**

goinfer's C′ expert streaming (`cuda/resident.go`, `loadExpertSlot`) runs a MoE model whose experts
exceed VRAM by DMA-ing the routed experts into device slots **per token, per layer**. On
Qwen3.6-35B-A3B (40 layers, top-8 of 256 experts, 75% LRU hit rate) that is:

    ~240 Upload calls per token  ->  ~240 FULL DEVICE SYNCS per token
    236 MB/token at 30.5 ms      ->  7.56 GB/s effective
    PCIe 3.0 x16 practical       ->  ~11-12 GB/s

The expert source is **already pinned** (`cuMemAllocHost`), and the caller **already owns a queue**
(`cudaResident.stream`), on which the expert GEMVs that consume the copy are launched. So the
ordering the sync exists to guarantee would come for free from stream order.

## What is actually needed, and why the existing API cannot express it

gocudrv has `Buffer[T].CopyFromHostAsync(ctx, stream, src PinnedHost[T])` — but it requires
`host.length == b.length`: a **whole-buffer** copy. An expert slot load is a sub-range on BOTH
sides:

    src: expert e's slice of a stacked pinned host buffer   [e*perExpert : (e+1)*perExpert]
    dst: slot s's region of the device slot buffer          [s*perExpert : (s+1)*perExpert]

So the ask is an **offset+length** variant, e.g.

    func (b *Buffer[T]) CopyFromHostRangeAsync(ctx, stream *Stream, dstOff int, src PinnedHost[T], srcOff, n int) error

plus whatever aikit-side wrapper exposes it on a queue-aware `Buffer` (aikit's `Buffer` already
carries a bind offset, so the aikit side may only need to thread a `Queue` through).

## Size the prize before building it — this is 1.18×, not 2×

Do not take this on expecting the 48% of token time the DMA occupies. Most of that is **real PCIe
transfer**, which async does not remove:

| | |
|---|---|
| sync overhead (240 × ~15 µs) | 3.6 ms of a 64 ms token — **5.6%** |
| DMA at the current 7.56 GB/s | 30.5 ms |
| DMA at a perfect 11 GB/s | 21.0 ms |
| **best case whole-token** | 64 → 54.5 ms = **1.18×** |

The transfer VOLUME is set by the cache hit rate, which is VRAM-bound, so it is not a lever here.
If 1.18× on one workload is not worth an API addition, the honest answer is to decline — this
prompt exists to make that judgement possible, not to presume it.

## The correctness property that must not regress

`upload`'s sync was added for a real, hard-won race, documented in that same comment: gocudrv
streams are `CU_STREAM_NON_BLOCKING` and therefore unordered against the legacy null stream, and
`cuMemcpyHtoD` from PAGEABLE memory returns once the source is staged, with the transfer still in
flight. It surfaced as an intermittently wrong GEMM that reproduced 15/15 at one shape and vanished
when another kernel was launched ahead of it.

An async variant is only safe because the destination is consumed **on the same stream**. That is a
contract on the CALLER, not a property of the copy, so it belongs in the doc comment in those
terms: *the copy is ordered with respect to this stream only; anything reading the destination from
another stream, or from the null stream, must synchronize explicitly.*

## Reproducing the goinfer side

```
GOINFER_HEAVY_TESTS=1 GOINFER_MOE_CACHE_PROF=1 \
GOINFER_QWEN36_35B=~/models/qwen3.6-35b-a3b-Q8_0.gguf \
  go test -tags "cuda goinfer_testhooks" ./cuda/ -run TestQwen36_35B_cache -v -timeout 90m
```

prints the round-trip split (`stall | host | dma`) the numbers above come from. Add
`GOINFER_CUDA_GRAPHS=1 GOINFER_CUDA_GRAPHS_UNSAFE=1` for the 15.69 tok/s configuration; without
graphs it is 10.67 and the DMA share is correspondingly smaller.

## Arch caveat

Measured on linux/amd64, RTX 2070 SUPER, PCIe 3.0 x16. On the MacBook there is no discrete-GPU
PCIe hop at all — unified memory makes the whole staging question different in kind, not degree.
The API addition is still arch-neutral; the MOTIVATING measurement is not.

---

# DECISION (2026-08-28): declined as scoped — a cheaper form gets the same prize

Reviewed from the aikit side at goinfer's request. **This prompt is declined as written**, and a
different change is recommended that captures the whole of the real prize for a fraction of the
cost. The prompt asked to be judged rather than presumed on; this is the judgement.

## 1. The 1.18× is not what this API buys. 5.6% is.

The prompt's own table says so, and the headline quietly bundles in something async cannot deliver:

| term | value | does async remove it? |
|---|---|---|
| sync overhead, 240 × ~15 µs | 3.6 ms of a 64 ms token — **5.6%** | **yes** |
| DMA at the measured 7.56 GB/s | 30.5 ms | no |
| DMA at a hypothetical perfect 11 GB/s | 21.0 ms | **no — async does not make PCIe faster** |

The 64 → 54.5 ms that yields 1.18× assumes BOTH the syncs vanish and the transfer runs at the PCIe
practical ceiling. Only the first is on offer here. **Size this at ~5.6%, not 1.18×.**

## 2. 5.6% is inside this workload's own demonstrated noise

`deltanet-residency-plan.md` records a run at 5.46 tok/s against 10.74 for byte-identical work —
an `rsync --server` competing for memory bandwidth, which lands squarely on a DMA that reads from
pinned host RAM — and warns that every single-run number in the slot sweep carries that same
uncertainty. A 5.6% effect cannot be demonstrated on a box where unrelated I/O halves the result.
Corroborated the same day on the aikit side: a 20 MiB device-to-device copy measured 24.6 / 7.4 /
133.1 GB/s across three runs on a loaded box and settled to ±2% only once it was idle.

So the change would ship on an argument, not a measurement, unless the box is quiesced first.

## 3. The cost is higher than "thread a Queue through"

The prompt suggests the aikit side "may only need to thread a `Queue` through". The primitive it
would thread to **does not exist**. gocudrv v0.3.2 has `CopyFrom`, `CopyFromHost`,
`CopyFromHostAsync` and `CopyFromAt` — offsets and async exist separately and **never together on
the H2D path**. So this is: a gocudrv change, a version bump, a pin, then the aikit wrapper. That
is the opposite of the device-to-device task, where `CopyToDeviceAt` already existed unused and the
aikit side really was a wrapper.

## 4. The safety trade is real and lands on a library with two consumers

`upload`'s sync exists for a hard-won race (non-blocking streams unordered against the null stream;
an intermittently wrong GEMM that reproduced 15/15 at one shape and vanished when another kernel
launched first). The async form is safe **only** because the destination is consumed on the same
stream — a contract on the CALLER, as the prompt itself says. Trading a proven-correct default for
a caller obligation, to buy an effect that cannot be measured on the available hardware, is a poor
trade in a package goinfer and ken both depend on.

## 5. What to do instead — and one idea checked and rejected

**REJECTED: coalescing adjacent slot loads.** Proposed first, then checked against
`cuda/resident.go`'s `loadExpertSlot`, which kills it:

    wOff := e    * w.perExpertW*4     // src offset — the ROUTED EXPERT index
    dst  := w.W.At(slot * w.perExpertW*4)  // dst offset — the LRU VICTIM slot

`e` comes from routing and `slot` from LRU victim selection. They are independent, so two
consecutive loads are essentially never adjacent in *both* src and dst, which is what merging
requires. (The two uploads inside one `loadExpertSlot` cannot merge either — they target different
buffers, `w.W` and `w.ws16`.) Coalescing works for contiguous snapshot ranges; it does not work
here, and the aikit-side `CopyDeviceBatch` coalescer would collapse nothing on this shape.

**RECOMMENDED: a batched upload — N copies, ONE synchronize.** This needs no adjacency, and it
attacks precisely the term that is actually on offer:

    // aikit/gpu, sketch
    func UploadBatch(copies []HostCopy) error   // N × CopyFromAt, then a single Synchronize

Why this is the better shape:

- **It removes 239 of the 240 syncs** — the entire 3.6 ms, i.e. all of the real prize.
- **It needs no new primitive anywhere.** `CopyFromAt(ctx, dstOffset, src)` already exists in
  gocudrv v0.3.2 and is *already what `Buffer.upload` calls today*. No dependency change, no bump.
- **It needs no async and no caller contract.** The batch still synchronizes before returning, so
  the correctness property §"The correctness property that must not regress" protects is preserved
  exactly — it is amortized, not weakened.
- **`srcW`/`srcS` are already `*gpu.MappedHostBuffer`** (pinned), so the copies issue without the
  driver staging through a bounce buffer and the single trailing sync covers all of them.
- Precedent: `CopyDeviceBatch` (gpu/v0.31.0) is the same one-sync-for-many shape on the D2D path.

**The caller-side change is the smaller half of the win.** `loadExpertSlot` currently issues two
uploads *and two syncs* per expert. Collecting a layer's misses and issuing one batch per layer
turns ~120 expert loads / 240 syncs per token into ~40 batches / 40 syncs, with the remaining
per-copy dispatch untouched. If the W and scales stacks were interleaved per expert rather than
kept as two separate stacks, it would be one copy per expert rather than two — but that is a
goinfer layout decision, out of scope here, and worth only the dispatch term.

## 6. Revisit if

- The box can be quiesced enough to resolve 5.6% (the `rsync` incident says it currently cannot).
- The DMA volume changes shape — the ceiling here is set by cache hit rate, which is VRAM-bound.
  A larger card or Metal expert streaming changes that term and this should be re-sized.
- A future workload needs genuine H2D/compute *overlap* rather than fewer syncs. That is the one
  thing only the async form gives, and no measurement currently asks for it.

**Caveat on this review:** it is a reading of the write-up and the code, not an independent
measurement. Nothing here was reproduced — 240-syncs-per-token is a CUDA-resident MoE workload and
the reviewing box is an M1 Pro. The `GOINFER_CUDA_GRAPHS_UNSAFE` caveat also means the 1.47×
baseline this sits on is unavailable on a shared box at all.
