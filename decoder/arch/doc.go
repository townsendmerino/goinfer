// Package arch holds the resolved decoder Architecture descriptor, Weight and LayerWeights
// containers, feature flags, KV geometry types, and LoadProfile instrumentation.
//
// It sits below decoder (which re-exports its public types and constants as aliases in
// decoder/zz_arch_alias.go) so loader, serializer, and backend packages can depend on the
// descriptor and weight types without importing decoder. Per-family internal structs live in
// internal/runtime/archparts.
package arch
