//go:build gpu

package gpu

// PrefillChunkRows (decoder.ResidentPrefillChunks, matched structurally) is the most rows one batched pass takes: the block
// PrefillLast cuts a longer prompt into.
func (rd *residentDecoder) PrefillChunkRows() int { return prefillBlockRows }

// prefillUniformBytes is what the pass is priced for each small uniform buffer it creates (they measure 16 to 64 bytes, about two
// dozen a layer).
const prefillUniformBytes = 64

// PrefillScratchBytes (decoder.ResidentPrefillChunks) is what one batched pass over rows rows allocates on the device beyond the
// standing buffers, as an upper bound: the activation buffers of ONE layer with none of them recycled (the pass recycles a dead
// layer's buffers into the next, so it holds about one layer's, R-25) plus the residual stream, and the fixed part: the head's
// vocab-sized staging buffers (two are created), the last row, and the uniform buffers. The per-row part carries a 5% margin: the
// pass's recycling is not identical from one layer to the next, and the figure must not fall under it. TestKV3_prefillScratchBytes_boundsThePass holds it to the pass:
// never under, and within 25% over, on the dense text path. A layer with its own geometry (Gemma 4) is priced at the widest.
func (rd *residentDecoder) PrefillScratchBytes(rows int) int64 {
	if rows <= 0 {
		return 0
	}
	hidden, inter := int64(rd.hidden), int64(rd.inter)
	var qN, kvN int64
	for _, ly := range rd.rm.layers {
		hd, nKV := int64(rd.hd), int64(rd.nKV)
		if ly.ghd != 0 {
			hd, nKV = int64(ly.ghd), int64(ly.gnKV)
		}
		qN, kvN = max(qN, int64(rd.nH)*hd), max(kvN, nKV*hd)
	}
	// floats a row: the residual stream; the attention-norm quantisation (packed int8 plus a scale); q, k, v; the context, its
	// quantisation and scale; the o-projection output; the MLP-norm quantisation; gate and up; the gated quantisation; the down output;
	// and the sandwich norms' two outputs for a family that has them.
	perRow := hidden + (hidden/4 + 1) + qN + 2*kvN + qN + (qN/4 + 1) + hidden + (hidden/4 + 1) + 2*inter + (inter/4 + 1) + hidden
	for _, ly := range rd.rm.layers {
		if ly.postAttnNorm != nil || ly.postMLPNorm != nil {
			perRow += 2 * hidden
			break
		}
	}
	fixed := 2*int64(rd.vocab)*4 + hidden*4 + int64(len(rd.rm.layers))*24*prefillUniformBytes
	return int64(rows)*perRow*4*21/20 + fixed
}
