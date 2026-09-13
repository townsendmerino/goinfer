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

// SpecDecodeConflict reports why speculative decoding verified on this model's resident would not
// be lossless, or nil when it would be.
//
// Every speculative loop that verifies on the resident — the n-gram path, the two-model draft path,
// and block drafting — is sold as token-identical to plain greedy. That holds only while a verified
// position scores exactly as a decoded one does. When the resident says the two can diverge, those
// entry points refuse instead of running, the same shape specRollbackSafe uses for state a rollback
// cannot restore: refuse at the source, never degrade silently.
//
// Exported for callers that must refuse at STARTUP rather than per request. cmd/serve's --spec
// ngram treats a per-request spec error as "fall back to plain decode", which would leave an
// operator who asked for speculation serving at 1x with no signal; it checks this once at load.
//
// NOT consulted by the staged speculative paths (EAGLE, grammar-fused, and any Session-driven
// n-gram run): those verify on a CPU cache, never touch the resident, and so cannot see a resident
// divergence. Refusing them would block constrained requests for nothing.
func (m *Model) SpecDecodeConflict() error {
	if m == nil || m.resident == nil {
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
