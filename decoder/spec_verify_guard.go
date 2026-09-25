package decoder

import "fmt"

// DecodeVerifyDiverger is implemented by a resident runner whose M=1 decode and batched verify can
// be made to disagree (cuda's flash-decode V-sum spike is the only such case today). It is an
// OPTIONAL interface, asserted rather than added to ResidentForward, so backends that cannot diverge
// — every one of them except an explicitly opted-in CUDA resident — carry no method for it.
//
// EXPORTED SO AN IMPLEMENTATION CAN PIN ITSELF TO IT AT COMPILE TIME. An optional interface fails
// OPEN: rename the method on either side and the type assertion in SpecDecodeConflict simply stops
// matching, the guard reports "no conflict" forever, and nothing errors. The cuda package holds a
// `var _ decoder.DecodeVerifyDiverger = (*cudaResident)(nil)`, which turns that silent disarm into a
// build failure.
type DecodeVerifyDiverger interface {
	DecodeVerifyDivergence() error
}

// ExactAttentionScoper is implemented by a resident whose M=1 decode attention has an OPT-IN lane that is not
// bit-identical to the tree its batched verify uses (cuda's flash-decode lane, GOINFER_CUDA_FLASH_DECODE). While a
// scope is held the resident runs its EXACT decode attention, so a speculative generation decodes and verifies with
// one tree and stays token-identical to plain greedy on that tree — instead of refusing to run (the V-sum spike's
// treatment, which has no scope to enter and so still refuses via DecodeVerifyDiverger).
//
// A scope is per GENERATION and must be held from before the first decode step of a speculative run until its last;
// nesting is allowed (a counter, not a flag). The resident's KV is single-tenant, so no plain generation can be
// interleaved inside a speculative one on the same Model. Exported so an implementation can pin itself to it at
// compile time, for the same fail-open reason as DecodeVerifyDiverger.
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

// SpecDecodeConflict reports why speculative decoding on this model would not be lossless, or nil
// when it would be. Two independent sources, checked in order:
//
//  1. M-09 (docs/audit-2026-09-10.md): the STAGED webgpu backend's own MatmulW4A8
//     (gpu/backend.go) declines any M != 1 and falls through to the CPU int4 kernel — so a
//     staged-int4 model on webgpu decodes each M=1 token on the device (a WGSL f32 GEMV with f16
//     group scales) but verifies M>1 batches on the CPU's integer kernel: two different kernel
//     implementations of the same logical matmul, with no numeric tolerance between them ever
//     measured or pinned. This is a STAGED-path risk specifically — it exists because m.resident
//     is nil, not despite it — so it is checked before, and independent of, the resident check
//     below.
//  2. The resident's own verify-vs-decode divergence (DecodeVerifyDiverger), for backends whose
//     resident M=1 decode and batched verify can disagree (cuda's flash-decode V-sum spike is the
//     only such case today).
//
// Every speculative loop that verifies on the resident — the n-gram path, the two-model draft path,
// and block drafting — is sold as token-identical to plain greedy. That holds only while a verified
// position scores exactly as a decoded one does. When either source above says the two can diverge,
// those entry points refuse instead of running, the same shape specRollbackSafe uses for state a
// rollback cannot restore: refuse at the source, never degrade silently.
//
// Exported for callers that must refuse at STARTUP rather than per request. cmd/serve's --spec
// ngram treats a per-request spec error as "fall back to plain decode", which would leave an
// operator who asked for speculation serving at 1x with no signal; it checks this once at load.
//
// The RESIDENT half is NOT consulted by the staged speculative paths (grammar-fused, and any
// Session-driven n-gram run): those verify on a CPU cache, never touch the resident, and so
// cannot see a resident divergence. The STAGED (M-09) half above applies to them too in principle
// — a staged webgpu-int4 model's decode/verify split is the same regardless of caller — but neither
// of those entry points calls this function today, matching this function's own
// "exported for callers that must refuse at startup" scope; they are unaffected either way.
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
