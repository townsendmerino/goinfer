# Task (goinfer): adopt `gpu.UploadBatch` in the C′ expert cache

> **For:** Claude Code in `~/tmcode/goinfer`, module `cuda`.
> Written 2026-08-29 from the aikit side, as the counter-proposal to
> `docs/prompts/aikit-subrange-async-upload.md` — **which was declined**, see the DECISION
> section at the foot of that file. This is the change that captures its prize instead.
>
> **Bump first:** `aikit/gpu` is at **`gpu/v0.32.0`**. `UploadBatch` landed there (`220372b`).

## What changed upstream, and why it is this and not what was asked for

`aikit/gpu` gained one verb:

```go
func UploadBatch(copies []HostCopy) error
type HostCopy struct{ Dst Buffer; Src []byte }
```

It issues every copy and then synchronizes **once**.

The ask was a sub-range *async* upload. It was declined on four grounds — the load-bearing one
being that **the 1.18× headline is not what that API buys.** Your own table in
`deltanet-residency-plan.md` reaches 64 → 54.5 ms by assuming *both* that the 240 syncs vanish
*and* that the DMA jumps from a measured 7.56 GB/s to a hypothetical perfect 11. Async delivers
only the first. **The real prize is the sync term: 3.6 ms of a 64 ms token, 5.6%.**

`UploadBatch` gets all of that 3.6 ms, and:

- **needs no new primitive** — it uses gocudrv's `CopyFromAt`, which `Buffer.upload` already calls.
  No gocudrv change, no bump, no pin.
- **needs no caller stream contract.** It still synchronizes before returning, so the guarantee is
  identical to `Upload`'s. The race that sync was added for (non-blocking streams unordered against
  the null stream; the intermittently-wrong GEMM that reproduced 15/15 at one shape) stays covered.
  Nothing moves into your hands.

## The change, concretely

Today, [`cuda/resident.go`](../../cuda/resident.go)'s `loadExpertSlot` issues **two uploads, and
therefore two full device syncs, per expert**:

```go
gpu.Upload(w.W.At(slot*w.perExpertW*4),  srcW[wOff:wOff+wLen])   // sync
gpu.Upload(w.ws16.At(slot*w.perExpertS*2), srcS[sOff:sOff+sLen]) // sync
```

`loadRoutedExperts` calls it once per cache miss, per MoE layer, per token — ~120 misses ⇒ ~240
syncs on the 35B.

**The shape to move to:** have `loadExpertSlot` *append* to a batch rather than upload, and have
`loadRoutedExperts` submit one `UploadBatch` per layer after admitting that layer's misses. That
turns ~240 syncs/token into ~40 (one per MoE layer). Going further — one batch per *token* rather
than per layer — is not available without restructuring, because `loadRoutedExperts` reads the
router's idx back per layer and that host round trip already orders the layers.

Expected: **~40/240 of the sync term survives**, so ~3.0 ms of the 3.6 ms recovered, ≈4.7% of the
token. Do not expect more; see the sizing below.

## Size the prize before building it — this is ~5%, and it is inside your own noise

| term | ms of a 64 ms token |
|---|--:|
| sync overhead, 240 × ~15 µs | **3.6 (5.6%)** ← all this change touches |
| DMA at the measured 7.56 GB/s | 30.5 |
| GPU work, stall, LRU host | ~30 |

**The measurement hazard is the real risk here, and it is documented in your own tree.**
`deltanet-residency-plan.md` records a run at **5.46 tok/s against 10.74 for byte-identical work**
because an `rsync --server` was competing for memory bandwidth — which lands squarely on a DMA
reading from pinned host RAM. A ~5% effect cannot be demonstrated against that. Corroborated
independently on the aikit side the same week: a 20 MiB device-to-device copy measured
24.6 / 7.4 / 133.1 GB/s across three runs on a loaded box and settled to ±2% only once idle.

**So: quiesce the box before measuring, and measure interleaved.** Alternate the batched and
unbatched arms within one session and compare medians rather than running them separately — that
method turned an inconclusive aikit measurement into a ±1% answer where best-of and
cold-vs-warm comparisons had both misled.

**Better still, measure the sync term directly.** You already have the seam: `GOINFER_MOE_CACHE_PROF`
counts `profWCalls`/`profSCalls` and times them. Asserting that call count drops 240 → 40 is a
*counting* question and needs no quiet box at all; the tok/s delta is the noisy part. Land the
change on the count, and treat the throughput number as confirmation rather than the gate.

## Gates

- `profWCalls + profSCalls` per token unchanged; **sync count per token drops ~240 → ~40.**
- Decode output **bit-identical** to the unbatched path — this is a batching change, not a
  numerics change, so anything else is a bug.
- A bad pair must not leave the batch half-applied (`UploadBatch` validates every pair before
  issuing any; the aikit tests cover length overrun, bind-offset overrun, and a mixed batch).
- Tok/s: report it, with the box quiesced and the arms interleaved, and say plainly if it lands
  inside noise. **"Inside noise" is a valid result here and does not invalidate the change** — the
  sync count is the thing being fixed.

## Notes and traps

- **All destinations in one batch must share a CUDA context.** `UploadBatch` enforces this and
  refuses a mixed batch rather than half-awaiting it (the trailing `Synchronize` is per-context).
  Your slots all live on the one resident context, so this should never fire — if it does,
  something about the layer's buffers is not what you think.
- **Do not expect coalescing.** aikit's `CopyDeviceBatch` merges contiguous pairs; `UploadBatch`
  deliberately does not, because it would collapse nothing on this shape: your source offset is the
  **routed-expert index** and your destination offset is the **LRU victim slot**, which are
  independent, and the two uploads per expert target different buffers (`w.W`, `w.ws16`) anyway.
  This was checked against your code, not assumed.
- **`srcW`/`srcS` are already `*gpu.MappedHostBuffer`** (pinned), so the copies issue without the
  driver staging through a bounce buffer and the single trailing sync covers all of them. No change
  needed there.
- **The 1.47× graphs baseline this sits on requires `GOINFER_CUDA_GRAPHS_UNSAFE`** and
  EXCLUSIVE_PROCESS/MPS compute mode. On a shared box you get 10.67, not 15.69 — measure the
  batching delta within one mode, not across two.
- **If you want the interleave win too:** storing an expert's weights and scales adjacently rather
  than as two separate stacks would make it one copy per expert instead of two, halving *dispatch*
  (not sync) count. That is a goinfer layout decision, out of scope for this task, and worth
  clearly less than the sync fix.

## Explicitly NOT in scope

- Any aikit change. `UploadBatch` is shipped and tested on both real devices at `gpu/v0.32.0`.
- The async variant. It is declined and the reasoning is recorded; reopening it needs a workload
  that wants genuine H2D/compute **overlap**, which no current measurement asks for.
- Expert buffer layout (see the interleave note above).
