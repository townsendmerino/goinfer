// Package memprice computes pure per-backend weight, KV-cache, recurrent-state,
// decode-scratch, and allocator-rounding byte footprints from resolved
// Architecture, Weights, and layer-program declarations, without importing
// decoder or any GPU backend module.
//
// Invariants:
//   - KV and recurrent-state pricing read per-layer program buffer declarations
//     rather than family switches (TestNoFamilySwitchInPricing).
//   - Metal decode scratch pricing reads per-row program declarations and head
//     buffers rather than family switches (TestDecodeScratchNamesNoFamily).
//   - Backend KV policies, row sizes, and allocator block models hold their
//     unit contracts (TestMemprice).
package memprice
