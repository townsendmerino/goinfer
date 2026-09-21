//go:build cuda

package cuda

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
)

// The link that keeps the guard armed. decoder.SpecDecodeConflict finds DecodeVerifyDivergence by
// an OPTIONAL interface assertion, which fails OPEN: rename the method on either side and the guard
// reports "no conflict" forever without a single error. This line turns that silent disarm into a
// BUILD failure — in a non-test file on purpose, so `go build -tags cuda` catches it, not only a
// test or vet run someone has to remember.
var _ decoder.DecodeVerifyDiverger = (*cudaResident)(nil)

// DecodeVerifyDivergence reports whether this resident's M=1 decode and its batched verify can
// currently produce different logits for the same position, and why. decoder.Model consults it
// (decoder.SpecDecodeConflict) before any speculative loop that verifies on this resident, because
// every one of those loops is sold as LOSSLESS — token-identical to plain greedy — and that claim
// rests on one invariant: a verified position scores exactly as a decoded one would.
//
// Today the only thing that breaks the invariant is the flash-decode V-sum spike. With
// GOINFER_SPLITKV_VSUM_SPLIT set, M=1 decode (launchToken -> splitKVAttnDecode) sums V over key
// chunks and combines them in a fixed order, while batched verify (ForwardN / PrefillLastN ->
// prefillCore) still runs attn_batched's single sequential fold. The two trees round differently.
// Within ONE speculative generation that means the prompt is seeded into the KV by one tree and
// verified by the other; across requests it means "--drafter on" and "--drafter off" return
// different text. Nothing errors and nothing looks wrong, which is why it is refused up front.
//
// It returns an error whenever the spike is ENABLED, not only once a layer crosses the split-KV
// depth threshold: a conversation that starts shallow grows past the threshold mid-stream, and a
// guard keyed on the current depth would let it start and then quietly stop being lossless.
//
// The optimistic-forward overlap (decoder/spec_optfwd.go) is deliberately NOT affected: its guess
// and its miss-redo both go through M=1 Forward — the same tree as decode — so it stays consistent
// with the spike on.
func (r *cudaResident) DecodeVerifyDivergence() error {
	if r.skVsumSplit > 1 {
		return fmt.Errorf("GOINFER_SPLITKV_VSUM_SPLIT=%d is set, so decode sums attention values in "+
			"a different order from the batched verify speculative decoding uses — the two can pick "+
			"different tokens at near-ties, and the output would no longer match plain greedy. "+
			"Unset GOINFER_SPLITKV_VSUM_SPLIT to use speculative decoding (see "+
			"docs/measurements/vsum-split-spike-2026-09-13.md)", r.skVsumSplit)
	}
	return nil
}

// The flash-decode lane (GOINFER_CUDA_FLASH_DECODE) is NOT a divergence source here, because speculative generations hold an
// exact-attention scope (decoder.ExactAttentionScoper): while one is active the lane is bypassed, so decode and verify both run
// the exact tree. Pinned at compile time for the same fail-open reason as DecodeVerifyDiverger above.
var _ decoder.ExactAttentionScoper = (*cudaResident)(nil)

// EnterExactAttention holds the exact-attention scope: while the count is non-zero the lane is bypassed. Counted, so nested or
// overlapping scopes compose; atomic, because it is entered on a caller goroutine and read on the executor.
func (r *cudaResident) EnterExactAttention() (leave func()) {
	if r.faVerify {
		// The multi-row lane serves verify rows bit-identically to M=1 lane decode, so decode and verify already share ONE tree (the lane's)
		// and speculation needs no exact scope. GOINFER_CUDA_FLASH_DECODE_VERIFY=0 restores the scope (option A).
		return func() {}
	}
	r.faExactScope.Add(1)
	var once sync.Once
	return func() { once.Do(func() { r.faExactScope.Add(-1) }) }
}
