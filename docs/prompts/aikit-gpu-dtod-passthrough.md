# Task (aikit): expose a device-to-device copy in `aikit/gpu`

> **For:** Claude Code in `~/mycode/aikit/aikit`, module `github.com/townsendmerino/aikit/gpu`.
> The local checkout's newest tag is `gpu/v0.30.0`; goinfer pins **v0.30.1**, so `git fetch` first.
> Written 2026-08-28 from goinfer-side measurements.
>
> **This is plumbing, not design.** The primitive already exists one layer down: `gocudrv`
> (`github.com/eitamring/gocudrv`, pinned at v0.3.2 by goinfer's `cuda` module) dispatches
> `memcpyDtoD` and `memcpyDtoDAsync`, surfaced on its device buffer type as `CopyToDevice` and
> `CopyToDeviceAsync` — find them in its `cuda/buffer.go`. `aikit/gpu` exposes `Upload`
> (host→device) and `Download` (device→host) and nothing between two device buffers. That gap is
> the whole task.
>
> *(Paths into gocudrv are given in prose deliberately: goinfer's citation lint resolves only
> goinfer, aikit and aikit/gpu, so a `path:line` into a third repo is a hard error there rather
> than a convenience.)*

## Why this is worth doing — measured, not assumed

goinfer needs to snapshot ~20 MiB of recurrent decoder state that is **already on the device**, and
put it back on a rejected speculative round. With no device-to-device copy the only implementable
path is device → host → device. Measured on an RTX 2070 SUPER, real model, in situ:

| path | cost | vs one resident decode step |
|---|---|---|
| PCIe round trip (`Download` + `Upload`) | 8.1 ms | **~100% ± 13 pp** |
| device-to-device, same bytes, via `gocudrv` directly | ~446 µs | ~5.5% |

A snapshot currently costs **an entire decode step** — a whole token's work per speculative round,
which no acceptance rate recovers. With DtoD it becomes a few percent. **~18×, and it changes the
answer's category rather than improving a number.** Nothing downstream can proceed until this
exists; the feature it gates is otherwise dead.

## What to add

A device-to-device copy in `aikit/gpu`, following `Upload`/`Download`'s existing shape and
conventions (bind offsets via `Buffer.At`, the allocation ledger, closed-buffer handling, the
error style already in `cuda.go`).

**Suggested signature only** — match the package's conventions over this sketch:

```go
// CopyDevice copies n elements from src into dst, device to device, without a host round trip.
func CopyDevice[T Scalar](dst, src Buffer) error
```

`gocudrv` enforces same-context and equal length and returns `ErrContextMismatch` /
`ErrLengthMismatch`; decide whether to surface those as-is or wrap them, and whether a partial /
offset copy is worth supporting (goinfer's consumer copies whole buffers).

**Metal: implement it, but it is not the motivation and must not hold up the CUDA side.** Metal is
UMA — `Upload`/`Download` in `metal.go` are plain `copy()` over `selContents`-mapped memory, with
no transfer at all. A device-to-device copy there is a memcpy between two mapped slices: trivial,
and it buys nothing, because the PCIe cost this task exists to remove does not exist on that
backend. Add it for API symmetry so callers do not have to branch by backend.

## Traps, all of them paid for already

**1. `CopyToDevice` is ASYNC despite its name and despite dispatching the synchronous
`memcpyDtoD`.** It returns before the transfer completes. Timing it alone measures dispatch (~9 µs)
rather than transfer (~116 µs for 20 MiB). Any benchmark must time **copy + `Synchronize`
together**. This produced a reading of **8145 GB/s**, and every part of the combination pointed the
wrong way — the name, the sync-not-async dispatch, and a plausible-looking microsecond figure.

**2. Assert against the physical ceiling, derived from the device rather than hardcoded.**
`MemoryClockRate × GlobalMemoryBusWidth × 2` gives 448 GB/s on a 2070 SUPER, matching spec. 8145
against that is not a suspicious number, it is an impossible one, and that is what caught it. A
hardcoded constant would go stale on another card and check nothing. **Make the test fail on an
impossible result rather than print it.**

**3. The composition is not the primitive, and it costs 2×.** One contiguous 20 MiB copy runs at
347 GB/s (78% of peak). The real consumer issues **36 separate copies** (18 layers × two buffers,
73 KB and 1 MiB), and that runs at 174 GB/s (39% of peak) — the small copies are dispatch-dominated
at ~9 µs each. **If a batched or multi-buffer form is cheap to add, it is worth more to the caller
than the single-copy form**, and a benchmark of one big copy will overstate what callers see.
Benchmark both shapes.

## Gates

- The new call is exercised by a test that **times copy + synchronize** and asserts the resulting
  bandwidth is **below a device-derived ceiling** — a test that cannot fail on an impossible number
  is not covering the trap that motivated it.
- Both backends build; Metal's path is exercised on darwin.
- Bandwidth reported for **both** shapes: one contiguous copy and a many-small-copies batch.
- Existing `Upload`/`Download` semantics unchanged — this is additive.
- A `gpu/vX.Y.Z` tag so goinfer can bump.

## Explicitly NOT in scope

- Any goinfer change. goinfer bumps the dependency and re-measures **in situ through the real
  snapshot**, which is a different measurement from the synthetic-buffer probe that produced the
  ~446 µs figure above. Do not treat 5.5% as an observed integration cost; it was measured through
  the primitive, not through an implemented snapshot.
- The snapshot feature itself, and anything about speculative decoding.
- Optimising the caller's buffer layout. Making the conv windows contiguous is a goinfer-side
  design note, recorded there.
