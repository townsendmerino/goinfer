# Task: one memory-accounting path per backend (2026-09)

> **Status 2026-09-25: DONE.** Metal half `504a0410` (`docs/measurements/memory-accounting-metal-2026-09-25.md`);
> CUDA half measured and closed on the Linux box (`docs/measurements/memory-accounting-cuda-2026-09-25.md`): CUDA's KV
> layout differed from Plan's formula in kind — f32 whatever precision is requested, one latent buffer on MLA — so
> `ResidentKVBytes` got a `"cuda"` branch exact to `kvBytesForCap`; item 3 was already pinned.
>
> **Earlier status 2026-09-25: the METAL half is DONE (`504a0410`, `docs/measurements/memory-accounting-metal-2026-09-25.md`);
> the CUDA half (item 3, and item 1's `kvBytesForCap` agreement test) is OPEN and needs the Linux box.** `fit`'s `Plan("metal")`
> and Metal's resident guard now compute the same number, from one function (`decoder.Model.ResidentNeedBytes`); re-measured
> on the Mac before changing anything, the disagreement was larger than stated below AND had a second cause — `Plan` priced
> Metal's KV at f32 when Metal allocates f16. Item 1 was also corrected in the doing: see its note.
>
> **Original status 2026-09-24: written up, not started.** Investigated and sized on Linux; owner chose a task doc over
> starting now because the Metal half overlaps the in-flight never-swap work (`task-never-swap-2026-09.md`, S4/S6).
> Whoever picks up S4/S6 next should do this first: it is the accounting those steps change.

## The problem, verified 2026-09-24

Memory needed by a model is computed in several places that were written separately and have drifted.

**A real disagreement: `Plan("metal")` omits Metal's second copy of the dense weights.**

- Metal's pre-build guard, `residentNeedBytes` (`metal/backend.go`), is
  `ResidentWeightBytesPaged(slots) + ResidentHostCopyBytes(slots) + residentKVBytes(m)`: the weights, a host copy
  of the weights (Metal's unified memory holds the quantized host `WeightMat` *and* a re-packed device buffer), and KV.
- `decoder.Model.Plan` (`decoder/fitplan.go`) prices `ResidentDenseWeightBytes()` + experts + KV. No host-copy term.
- Both judge against the same ceiling (`metalMemoryCeiling`, unified in S4 item 3). So `fit` (`internal/fitcmd`,
  which calls `m.Plan(backend, free, req)`) can report **resident** for a model the Metal guard then **refuses**.
- Size of the missing term, measured on Linux (it is a decoder method): for a GGUF load it equals the whole dense
  term — **1.29 GB** on qwen2.5-coder-1.5b (dense 1.29 GB), **0.50 GB** on the 0.5B.
- It shrinks toward zero for an aliased `.giw` on Metal: S6 (shipped, `723fa854`) aliases `.giw` weights by default,
  and `ResidentHostCopyBytes` already exempts mmap-aliased weights
  (`TestResidentHostCopyBytes_exemptsMmapAliasedGIWWeights`). The term now depends on how the model was loaded, which
  is the argument for computing it in one place.

**Five KV-cache byte formulas:**

| where | what it covers | known gaps |
|---|---|---|
| `kvBytesForCtx` (`decoder/arch.go`) | per-layer `kvDimAt` (0 for linear/Mamba/conv, MLA latent width), sliding-window count cap | the most complete; unexported |
| `kvBytesPerPositionAllLayers` (`decoder/fitplan.go`, `Plan`) | per-layer `kvDimAt`, per-position rate | deliberately no sliding-window cap (its own comment) |
| `kvBytesPerPosition` / `estimateKVBytes` (`decoder/fitguard.go`) | load-time host-RAM guard, from `Config` | pre-model; its own per-layer handling |
| `residentKVBytes` (`metal/backend.go`) | Metal guard, at Metal's resolved ctxCap | skips DeltaNet layers only (not Mamba/conv), uses head×hd not MLA's latent width, int8 scales as f32 per head |
| `kvBytesForCap` (`cuda/resident.go`) | sums what CUDA actually allocates | allocation-exact by design — **keep**, it is a different quantity |

On a dense model they agree (1.5B at 32k, f16: `Plan` 0.940 GB = `kvBytesForCtx` 0.940 GB); they diverge at the
edges named in the table. (`DrafterKVBytesPerPosition` prices a drafter's own cache, a separate model; out of scope.)

**Two copies of the 0.70 fraction:** `fitMemFraction` (`decoder/fitguard.go`) and `residentMemFraction`
(`metal/backend.go`).

**Slot sizers:** CUDA's search-based sizing (`capSlots`/`slotRequirement`, `cuda/resident.go`), Metal's
`autoMoESlotsFor` (`metal/backend.go`, solves the guard's inequality for N), and `Plan`'s expert cap.

Not independently verified: the originating review's "12 places" total. The items above are the ones checked.

## The work

1. **One KV function.** Export `kvBytesForCtx` (or a wrapper with the model as receiver) and use it in `Plan`, both
   `fitguard` formulas where a model exists, and Metal's `residentKVBytes`. Where a caller wants a per-position rate
   (`Plan`'s ctx search), give it the same function evaluated at ctx, so the sliding-window cap comes along instead of
   being approximated. CUDA's `kvBytesForCap` stays; add a test that it agrees with the shared function on dense
   geometries (within the driver's rounding).

   > **Metal: DONE, with a correction to this item's plan.** Metal allocates the FULL padded context on every attention
   > layer — its sliding window is a mask, not a ring buffer — so reusing `kvBytesForCtx` (which caps local layers at the
   > window, modelling the CPU ring buffer) would have made Metal's guard under-count Gemma-style models. The shared
   > function is instead `decoder.Model.ResidentKVBytes(backend, ctx, f16, i8)`, carrying each backend's allocation layout:
   > on `"metal"` it is exact to `buildResident`'s buffers (f16 or int8 + per-head scales, padded to 8, none on a DeltaNet
   > layer — `TestResidentKVBytes_matchesMetalAllocation`, 5/5 exact); every other backend keeps `Plan`'s existing
   > per-position formula, unchanged. The `fitguard` pre-model formulas were not touched: they run from `Config` before
   > a backend is chosen.
   >
   > **CUDA: DONE (2026-09-25, `docs/measurements/memory-accounting-cuda-2026-09-25.md`).** Measured on ten real CUDA
   > residents (dense, sliding-window, per-layer geometry, DeltaNet hybrid, MLA) before changing anything: CUDA has **no**
   > sliding-window cap (full context on local layers, like Metal) and no cache on DeltaNet layers — there the old formula
   > was already exact — but it differs in kind twice: it allocates **f32 whatever KV precision is requested** (Plan
   > priced f16 at 0.50×, i8 at 0.28×), and an **MLA layer holds one latent buffer** (Plan doubled it at f32, which reached
   > CUDA's default context sizing). `ResidentKVBytes` got a `"cuda"` branch exact to `kvBytesForCap`:
   > `TestResidentKVBytes_matchesCUDAAllocation` (GPU, 10/10 exact, mutation-checked) and `TestResidentKVBytes_cudaLayout`
   > (decoder, CI). Per-buffer 2 MiB rounding stays out of both, as in the resident's own fit check.
2. **One "bytes needed" function per backend**, with a unified-memory host-copy term for Metal (aliasing-aware, via
   `ResidentHostCopyBytes`). `Plan`, the Metal guard (`residentNeedBytes`) and Metal's auto-slots (`autoMoESlotsFor`'s
   `needFixed`) all call it, so `fit`'s verdict and the guard's decision are the same number by construction. Pin that
   with a test: for a model where they used to disagree, `Plan("metal")` and the guard agree.

   > **DONE.** `decoder.Model.ResidentNeedBytes(backend, slots, ctx, …)` = weights + (Metal) host copy + KV; Metal's
   > `residentNeedBytes` and `residentKVBytes` call it (auto-slots' `needFixed` gets the shared KV term through
   > `residentKVBytes`), and `Plan` builds from the same pieces with a new `HostCopyBytes` field.
   > `TestPlan_metalAgreesWithResidentNeedBytes` (resident + expert-cached) and `TestPlan_metalDeclinesWhereTheGuardWould`,
   > both mutation-checked. Serve's banner also reports the KV precision that runs (`ResidentKVPrecision`: `KV f16` on Metal).
3. **Keep CUDA's search-based slot sizing.** The driver rounds allocations up in 2 MiB steps, so dividing a budget by a
   per-slot size cannot invert it. Pin the search with a test at a boundary where division would over-admit.

   > **Already pinned — nothing added (checked 2026-09-25).** `cuda/slotcap_test.go`'s `TestSlotCapArithmetic_search`
   > asserts, on the real 26B's figures, that plain division returns 34 where the search returns 33 (the 33→34 step is 4
   > quanta × 30 layers, the requirement 203,816,960 B over free), plus monotonicity and "largest count that fits";
   > `TestSlotCapArithmetic_mutation` proves it red with the rounding removed.
4. **One 0.70 constant**, owned by decoder, used by Metal.

   > **DONE.** `decoder.WeightsMemFraction`; `fitMemFraction` and Metal's `residentMemFraction` both read it.

## Where it runs, and the gate

- Decoder parts (1, the `Plan` side of 2, 4) build and test on Linux. `decoder/fitplan.go` and `decoder/arch.go` are
  parity-manifest files; run `scripts/refresh_parity_hashes.sh`.
- Metal parts (the guard and auto-slots) need a Mac run: `go test -tags goinfer_testhooks ./metal/` and the untagged
  `go test ./metal/`, plus the memory-guard tests (`resident_memguard_test.go`). Coordinate with the S6 work, which is
  editing `metal/backend.go`.
- Default behaviour changes only where the old numbers disagreed: `fit` may now say "decline" or "expert-cached"
  for GGUF-loaded models on Metal where it said "resident". That is the fix, and it should be stated in the commit.

<!-- doc-reviewed: 2026-09-25 -->
