//go:build gpu

package webgpu

import "github.com/townsendmerino/goinfer/decoder"

// The build unpacks decoder.ResidentSpec's family sub-structs into the positional locals its phases read; each returns zeros and ok=false
// when the model's architecture carries no such subsystem.

func graniteParams(s decoder.ResidentSpec) (nHeads, headDim, dState, nGroups, dConv int, embMul, residMul, logitScale, attnScale float32, ok bool) {
	g := s.Granite
	if g == nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, false
	}
	return g.NHeads, g.HeadDim, g.DState, g.NGroups, g.DConv, g.EmbMul, g.ResidMul, g.LogitScale, g.AttnScale, true
}

func nemotronParams(s decoder.ResidentSpec) (nHeads, headDim, dState, nGroups, dConv, normGroups int, attnScale float32, ok bool) {
	np := s.Nemotron
	if np == nil {
		return 0, 0, 0, 0, 0, 0, 0, false
	}
	return np.NHeads, np.HeadDim, np.DState, np.NGroups, np.DConv, np.NormGroups, np.AttnScale, true
}

func moeParams(s decoder.ResidentSpec) (nE, k, inter, sharedInter int, sigmoid, norm, sharedUngated bool, scale float64, nGroup, topkGroup int, ok bool) {
	mo := s.MoE
	if mo == nil {
		return 0, 0, 0, 0, false, false, false, 0, 0, 0, false
	}
	return mo.NumExperts, mo.TopK, mo.IntermediateDim, mo.SharedIntermediateDim,
		mo.RouterSigmoid, mo.NormTopKProb, mo.SharedUngated, mo.RoutedScale, mo.NGroup, mo.TopkGroup, true
}

func mlaParams(s decoder.ResidentSpec) (qLoRA, kvLoRA, qkNope, qkRope, vHead int, interleave bool, attnScale, ropeScale float64, ok bool) {
	p := s.MLA
	if p == nil {
		return 0, 0, 0, 0, 0, false, 0, 0, false
	}
	return p.QLoRARank, p.KVLoRARank, p.QKNopeHeadDim, p.QKRopeHeadDim, p.VHeadDim,
		p.RopeInterleave, p.AttnScale, p.RopeScale, true
}

func qwen35Params(s decoder.ResidentSpec) (convKernel, keyHeadDim, valueHeadDim, numKeyHeads, numValueHeads int, attnGate, ok bool) {
	g := s.Qwen35
	if g == nil {
		return 0, 0, 0, 0, 0, false, false
	}
	return g.ConvKernel, g.KeyHeadDim, g.ValueHeadDim, g.NumKeyHeads, g.NumValueHeads, g.AttnGate, true
}
