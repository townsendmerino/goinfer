package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/program"

// metalDerivedDenseBytes is what a Metal resident build of m leases on the dense line beyond the tensors the dense sum counts
// (residentWeightBytesSplit): the buffers the build derives from the weights and the geometry, by the build's own rules.
//   - Per attention layer: a one-float sink where the family has none; the layer's RoPE table and YaRN scale; zero Q|K|V biases
//     where the checkpoint has no Q bias (the build then uploads zeros for all three, not the K and V biases); and on a K=V layer
//     the second k_proj copy its fused Q|K|K projection holds, at W4A8, the layout that projection takes at every quant.
//   - Per layer: zero up and down biases on a non-gated FFN; a MoE router's zero bias where the checkpoint has none; Gemma 4's
//     layer scalar, which a dense layer uploads when it is neither 0 nor 1 and a MoE block always; and on Gemma 4's MoE block its
//     zero router bias, less the router scale the build folds into the router.
//   - Per model: the RoPE table, the unit v_norm weight at the widest head, and the m-RoPE axis map at the first attention
//     layer's table width (the build's prefill geometry), wherever the family has one. A build that declines the batched prefill
//     leaves the map out, so there the price is over.
//
// TestMemGate_metal holds the plan's dense line to the build's leases, and TestMetalDensePrice_isTheBuildsLeases to the byte.
func (m *Model) metalDerivedDenseBytes() int64 {
	if m == nil || m.w == nil {
		return 0
	}
	w := m.w
	_, _, nH, _, _, _, _ := m.Dims()
	_, _, _, _, _, attnGate, dnetOK := m.Qwen35ResidentParams()
	nE, _, isMoE := m.moeGeometry()
	nonGated := false
	if p := m.Program(); p != nil {
		if s, err := program.ShapeOf(p); err == nil {
			nonGated = s.NonGated
		}
	}
	var floats, bytes int64
	maxHd, firstHalf := 0, -1
	for l := range w.Layers {
		lw := &w.Layers[l]
		mixer, hasFFN := m.ResidentLayerOps(l)
		if mixer == MixerAttention {
			hd := m.HeadDimAtResident(l)
			kvDim := m.KVHeadsAtResident(l) * hd
			maxHd = max(maxHd, hd)
			if len(lw.AttnSinks) == 0 {
				floats++
			}
			half := len(m.RopeInvFreqLayerResident(l))
			if half > 0 {
				floats += int64(half) + 1
			}
			if firstHalf < 0 {
				firstHalf = half
			}
			if qGate := dnetOK && attnGate; !qGate {
				if lw.QBias == nil {
					floats += int64(nH*hd+2*kvDim) - int64(len(lw.KBias)+len(lw.VBias))
				}
				if m.KVSrcAtResident(l) == l && m.VFromKResident(l) {
					bytes += metalW4A8Bytes(lw.KProj.Rows(), lw.KProj.Cols())
				}
			}
		}
		g4 := lw.gemma4moe
		switch {
		case !hasFFN:
		case g4 != nil && w.arch.gemma4 != nil:
			floats += int64(g4.nE) + 1 - int64(len(g4.routerScale))
		case isMoE && len(lw.Experts) > 0:
			if len(lw.RouterBias) == 0 {
				floats += int64(nE)
			}
		case nonGated:
			if len(lw.UpBias) == 0 {
				floats += int64(lw.UpProj.Rows())
			}
			if len(lw.DownBias) == 0 {
				floats += int64(lw.DownProj.Rows())
			}
		}
		if g4 == nil && w.arch.gemma4 != nil {
			if s := lw.LayerScalar; s != 0 && s != 1 {
				floats++
			}
		}
	}
	floats += int64(len(m.RopeInvFreq()))
	if maxHd == 0 {
		maxHd = 4 // no attention layer: the build sizes its attention buffers at 4
	}
	floats += int64(maxHd)
	if firstHalf > 0 {
		floats += int64(len(m.MRopeAxisResident(firstHalf)))
	}
	return 4*floats + bytes
}

// metalW4A8Bytes is a rows × cols projection in Metal's W4A8 layout: a nibble a weight and an f16 scale per 32.
func metalW4A8Bytes(rows, cols int) int64 {
	n := int64(rows) * int64(cols)
	return n/2 + 2*(n/32)
}
