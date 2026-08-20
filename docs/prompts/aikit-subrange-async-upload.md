# aikit task: a sub-range async upload, because profiling now shows `Buffer.upload` on a hot path

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
