// Package families holds the per-family architecture adapters (arch_*.go and gemma12.go),
// the model_type registry, and the safetensors tensor-name schemas.
//
// Each adapter resolves a checkpoint decoder/config.Config into a family-agnostic
// decoder/arch.Architecture descriptor and its internal/runtime/archparts.TensorSchema.
// Keeping the adapters below decoder lets family-config changes build and test without
// compiling the loader or CPU forward engine.
package families
