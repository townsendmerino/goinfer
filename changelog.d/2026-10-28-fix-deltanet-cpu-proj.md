- **`decoder`, `internal/program` (fix #418 regression)**: `gatedDeltaNetStep` now always passes its `proj` callback
  (`st.hq.prepare` / `matvecWMPre` / `matvecWM`) instead of guarding it on `be != nil`, restoring CPU projections when
  `be == nil` on quantized (`int8`, `int4`, `q4k`) or 16-bit (`bf16`, `f16`) `WeightMat`s. `GatedDeltaNetStep`'s nil-`proj`
  fallback now checks `wm.F32()`'s boolean and panics with a clear message when handed a non-`f32` `WeightMat` rather than
  computing on a nil slice (`TestGatedDeltaNetStep_nilBackendNonF32`).
