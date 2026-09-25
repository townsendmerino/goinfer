# One memory-accounting path for Metal — `fit` and the resident guard now agree (2026-09-25)

`docs/tasks/task-memory-accounting-2026-09.md` (items 1, 2 and 4, for Metal). The task doc was written on the Linux box;
its central claim was re-measured here on the Mac before anything changed.

## What was measured first (M1 Pro, `~/models`, ctx 4096, int4)

| load | Metal resident guard | `Plan("metal")` (what `fit` reports) |
|---|---|---|
| qwen2.5-coder-1.5b, direct `.gguf` | 2.106 weights + **2.106 host copy** + 0.117 KV = **4.33 GB** | 2.106 + 0.235 KV = **2.34 GB** |
| qwen2.5-coder-1.5b, v14 metal `.giw` (aliased) | 1.287 + 0 + 0.117 = 1.40 GB | 1.287 + 0.235 = 1.52 GB |
| qwen2.5-7b, v14 metal `.giw` (aliased) | 5.170 + 0 + 0.235 = 5.40 GB | 5.170 + 0.470 = 5.64 GB |

Two disagreements, not one:

1. **No host-copy term in `Plan`** — the task doc's finding, confirmed and larger here (2.1 GB, not Linux's 1.29, because
   arm64 also holds the repacked int4 copy). Since S6 it is 0 for an aliased `.giw`, so it now bites a direct `.gguf` on
   Metal: `fit` could say *resident* for a load the guard then refused.
2. **`Plan` priced Metal's KV at f32; Metal allocates f16** (its f32 KV kernels are compiled out, `metal/model.go`'s
   `r.kvF32`), so `Plan`'s KV was twice the real one. Serve's banner printed `KV f32` on Metal for the same reason.

Also found reading the allocation: Metal gives **every attention layer the full padded context** (no sliding-window ring
buffer — the window is a mask), so the task doc's item 1 as written (reuse `kvBytesForCtx`, which caps local layers at
the window) would have made the guard under-count Metal on Gemma-style models. "One KV function" has to carry the
backend's allocation layout.

## What shipped

- `decoder.Model.ResidentKVBytes(backend, ctx, f16, i8)` — on `"metal"`, exact to the allocation: f16 (int8 + per-head f32
  scales with an int8 KV cache), full ctx padded to 8 on every attention layer, nothing on a Gated-DeltaNet layer. Other
  backends keep `Plan`'s existing per-position formula, unchanged.
- `decoder.Model.ResidentNeedBytes(backend, slots, ctx, …)` — weights + (Metal) host copy + KV. Metal's guard
  (`residentNeedBytes`) and KV term call it; `Plan` builds from the same pieces and gains `HostCopyBytes`, counted in
  `NeedBytes()`. Plan's context shrink stays on Metal's 8-position grid so its KV is exactly what Metal allocates.
- `decoder.WeightsMemFraction` (0.70) — the fit guard's `fitMemFraction` and Metal's `residentMemFraction` both read it.
- `decoder.Model.ResidentKVPrecision()` — what the resident runner allocates; serve's banner prints it (Metal: `KV f16`).

## Gates

- `TestResidentKVBytes_matchesMetalAllocation` (metal): the shared KV figure equals the bytes of the KV buffers
  `buildResident` really allocates — dense, dense + int8 KV, Gemma 4 two-geometry, Gemma 3 (sliding pattern), qwen35
  (DeltaNet hybrid), at ctx 100 (not a multiple of the padding): **5/5 exact**. Mutation: without the padding (the old
  guard formula) the three padded cases fail.
- `TestPlan_metalAgreesWithResidentNeedBytes` (decoder): `Plan("metal").NeedBytes()` equals `ResidentNeedBytes` resident
  and expert-cached (llama-tiny, mixtral-tiny). `TestPlan_metalDeclinesWhereTheGuardWould`: one byte under the guard's
  figure, `Plan("metal")` no longer says resident (the old `Plan` did); `Plan("cuda")` unchanged. Mutations: no host
  copy, and the old f32 KV formula — each fails both tests.
- `TestResidentKVPrecision_metalReportsWhatRuns`: a real Metal load reports f16 with no `-kv` and with `-kv f32`, i8 with
  `-kv i8`; banner cases added.
- Suites: decoder, metal (testhooks + untagged), internal/fitcmd, serveapp, chatapp green; `gpu`/`cuda` vet clean against
  the local decoder; parity goldens 38/0 (no manifest file touched).

## Behaviour change

`fit` on Metal can now say *decline* or *expert-cached* for a directly loaded `.gguf` where it said *resident* — the load
Metal's guard would have refused anyway. For an aliased `.giw` it asks for slightly **less** than before (the KV halves).

## Not done here

Item 3 (a test pinning CUDA's search-based slot sizing at a boundary where division would over-admit) and CUDA's
`kvBytesForCap` agreement test — both need the Linux box. The `fitguard` pre-model formulas (`estimateKVBytes`) are unchanged:
they run from `Config` before any backend is chosen.
