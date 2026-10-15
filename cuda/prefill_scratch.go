package cuda

// prefillPassRows is the most rows one batched pass takes now: GOINFER_PREFILL_CHUNK or the default (512, a multiple of the block
// size 64), lowered to the width a previous prompt learned the card can hold, and to a windowed model's slack. prefillChunked cuts
// by it; PrefillChunkRows reports it to admission.
func (r *cudaResident) prefillPassRows() int {
	chunk := prefillChunkRows(r.knobValue("GOINFER_PREFILL_CHUNK"))
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned // a previous prompt already found the default too wide for this card
	}
	if r.kvWin {
		chunk = min(chunk, kvWindowSlack+1) // a windowed layer's slack bounds the rows one pass can add (kvRoomFor)
	}
	return chunk
}

// PrefillChunkRows (decoder.ResidentPrefillChunks, matched structurally: TestKV3_residentReportsItsChunks holds the wiring) is the most rows one pass of a long prompt's prefill takes.
func (r *cudaResident) PrefillChunkRows() int { return r.prefillPassRows() }

// PrefillTailMinRows (decoder.ResidentPrefillTailMin) is the fewest rows a prefill's last pass may have: a shorter one runs the
// exact kernels while the pass before it ran the fused ones (attnFusedMinRows).
func (r *cudaResident) PrefillTailMinRows() int { return attnFusedMinRows }

// prefillScratchSeenForTest, when set (tests only), is told the byte size of each scratch buffer prefillCore asks for, before the
// allocator's rounding, so a test can hold PrefillScratchBytes to the pass.
var prefillScratchSeenForTest func(bytes int64)

// scratchBufBytes is what the device charges for one scratch buffer of n 4-byte elements: the request, rounded up to the
// allocator's quantum from a quantum upward (allocRoundSlack). A request of no elements is one (prefillCore's af and ai).
func scratchBufBytes(n int) int64 {
	b := int64(max(n, 1)) * 4
	return b + allocRoundSlack(b)
}

// PrefillScratchBytes (decoder.ResidentPrefillChunks) is what one ordinary prefill pass of rows rows allocates on the device beyond
// the standing buffers: the M-row buffers prefillCore creates, each after the allocator's rounding. It prices a pass whose tail is
// the last row only (PrefillLast), text only; the all-logits tails of speculative verify add an M x vocab logits block, an image pass
// its block table, and neither is a prompt's prefill. It mirrors prefillCore's allocations by the same arithmetic, and
// TestKV3_prefillScratchBytes_matchesThePass holds it to them.
func (r *cudaResident) PrefillScratchBytes(rows int) int64 {
	M := rows
	if M <= 0 {
		return 0
	}
	maxQDim, maxKvDim := r.prefillMaxGeom()
	hidden, inter := r.hidden, r.inter
	if r.eModel {
		for l := range r.layers {
			inter = max(inter, r.layers[l].ffnI)
		}
	}
	var sum int64
	for _, n := range []int{
		M * hidden,        // xB
		M * hidden / 4, M, // aqB, aScB
		M * maxQDim, M * maxKvDim, M * maxKvDim, // qBb, kBb, vBb
		M * maxQDim,        // cctxB
		M * maxQDim / 4, M, // cqB, cScB
		M * hidden / 4, M, // mqB, mScB
		M * inter, M * inter, // gOb, uOb
		M * inter / 4, M, M * inter, // dqB, dScB, dScrB
	} {
		sum += scratchBufBytes(n)
	}
	if dp := r.dnet; dp != nil {
		for _, n := range []int{M * dp.convDim, M * dp.nv, M * dp.nv, M * dp.valueDim, M * dp.convDim, M * dp.keyDim, M * dp.keyDim,
			M * dp.nv * 2, M * dp.valueDim, M * dp.valueDim, M * dp.valueDim / 4, M, M * 2 * maxQDim, M * maxQDim} {
			sum += scratchBufBytes(n)
		}
	}
	if r.sandwich || r.postOnly || r.parallelBlock {
		sum += scratchBufBytes(M * hidden)
	}
	if r.pleP > 0 {
		sum += scratchBufBytes(r.nLayers*M*r.pleP) + scratchBufBytes(M*r.pleP)
	}
	return sum
}
