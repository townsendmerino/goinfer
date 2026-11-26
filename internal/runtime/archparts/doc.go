// Package archparts holds the per-family parameter and weight structs referenced by
// decoder/arch.Architecture and decoder/arch.LayerWeights, together with shared internal
// helpers (k-quant source tracking, KV row padding, and family-specific shape constants).
//
// Keeping these structs internal prevents per-family layout details from expanding the
// public decoder/arch API surface while allowing decoder and sibling internal/runtime
// packages to read and populate them directly.
package archparts
