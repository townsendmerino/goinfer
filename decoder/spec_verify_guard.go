package decoder

import "fmt"

// DecodeVerifyDiverger is an optional interface for a resident runner whose M=1 decode and batched verify can be made
// to disagree. It is asserted rather than added to ResidentForward, so backends that cannot diverge carry no method for
// it. Exported so an implementation can pin itself to it at compile time: an optional interface fails open (rename the
// method on either side and the type assertion in SpecDecodeConflict stops matching, the guard reports "no conflict"
// forever, and nothing errors), and a `var _ decoder.DecodeVerifyDiverger = (*T)(nil)` turns that silent disarm into a
// build failure.
type DecodeVerifyDiverger interface {
	DecodeVerifyDivergence() error
}

// ExactAttentionScoper is an optional interface for a resident whose M=1 decode attention has a lane that is not
// bit-identical to the tree its batched verify uses (cuda's flash-decode lane). While a scope is held the resident runs
// its exact decode attention, so a speculative generation decodes and verifies with one tree and stays token-identical
// to plain greedy, instead of refusing to run (a lane with no scope to enter still refuses via DecodeVerifyDiverger).
//
// A scope is per generation: hold it from before the first decode step of a speculative run until its last. Nesting is
// allowed (a counter, not a flag). The resident's KV is single-tenant, so no plain generation can interleave inside a
// speculative one on the same Model. Exported so an implementation can pin itself to it at compile time, for the same
// fail-open reason as DecodeVerifyDiverger.
type ExactAttentionScoper interface {
	EnterExactAttention() (leave func())
}

// enterExactAttention holds the resident's exact-attention scope for one speculative generation and returns its
// release; a no-op for any model whose resident has no such lane. Called synchronously, BEFORE the generation's
// goroutine starts, so the scope is active from the moment the entry point returns.
func (m *Model) enterExactAttention() func() {
	if m == nil || m.resident == nil {
		return func() {}
	}
	if sc, ok := m.resident.(ExactAttentionScoper); ok {
		return sc.EnterExactAttention()
	}
	return func() {}
}

// SpecDecodeConflict reports why speculative decoding on this model would not be lossless, or nil when it would be.
// Two independent sources, checked in order:
//
//  1. The staged webgpu backend's own MatmulW4A8 (gpu/backend.go) declines any M != 1 and falls through to the CPU int4
//     kernel, so a staged-int4 model on webgpu decodes each M=1 token on the device (a WGSL f32 GEMV with f16 group
//     scales) but verifies M>1 batches on the CPU's integer kernel: two implementations of the same logical matmul with
//     no numeric tolerance between them measured or pinned. This is a staged-path risk (m.resident is nil), so it is
//     checked before, and independent of, the resident check.
//  2. The resident's own verify-vs-decode divergence (DecodeVerifyDiverger), for a resident whose M=1 decode and
//     batched verify can disagree.
//
// Every speculative loop that verifies on the resident (the n-gram path, the two-model draft path, block drafting) is
// token-identical to plain greedy only while a verified position scores exactly as a decoded one does. When either
// source says the two can diverge, those entry points refuse instead of running, the shape specRollbackSafe uses for
// state a rollback cannot restore: refuse at the source, never degrade silently.
//
// Exported for callers that must refuse at startup rather than per request: serve's --spec ngram treats a per-request
// spec error as "fall back to plain decode", which would leave an operator who asked for speculation serving at 1x with
// no signal, so serve checks this once at load.
//
// The resident half is not consulted by the staged speculative paths (grammar-fused, and any Session-driven n-gram run):
// they verify on a CPU cache, never touch the resident, and cannot see a resident divergence. The staged half applies to
// them in principle (a staged webgpu-int4 model's decode/verify split is the same whoever calls), but neither entry
// point calls this function.
func (m *Model) SpecDecodeConflict() error {
	if m == nil {
		return nil
	}
	if m.be != nil && isWebGPUBackend(m.be.Name()) {
		switch m.Quant() {
		case "int4", "int4mix":
			return fmt.Errorf("speculative decoding would not be lossless: staged webgpu int4 decodes M=1 on the device and verifies M>1 on the CPU kernel, two different kernels with no measured tolerance between them (M-09, docs/audit-2026-09-10.md); use Generate")
		}
	}
	if m.resident == nil {
		return nil
	}
	d, ok := m.resident.(DecodeVerifyDiverger)
	if !ok {
		return nil
	}
	if err := d.DecodeVerifyDivergence(); err != nil {
		return fmt.Errorf("speculative decoding would not be lossless on this resident: %w", err)
	}
	return nil
}
