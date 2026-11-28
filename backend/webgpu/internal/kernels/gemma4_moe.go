//go:build gpu

package kernels

import (
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/wgsl"
)

// Gemma 4's parallel dense and MoE feed-forward (enable_moe_block; decoder's gemma4MoEFFN) in the decode runner. The
// router's top-k and the indexed expert GEMVs are the runner's own MoE kernels; this file adds the two steps they lack.

// G4MoELayer is one enable_moe_block layer's feed-forward on the device.
type G4MoELayer struct {
	PreFFNNorm, PostFFNNorm1, PreFFNNorm2, PostFFNNorm2, PostFFNNorm *wgpu.Buffer // [hidden] each
	RouterNorm                                                       *wgpu.Buffer // the weight that makes the runner's RMSNorm weightless
	Router                                                           *wgpu.Buffer // [nE, hidden] float32, the learned scale and hidden^-0.5 folded in
	PerExpertScale, ZeroBias                                         *wgpu.Buffer // [nE]
	Gate, Up, Down                                                   DecodeWeight // the dense branch
	ExpGateUp, ExpDown                                               *ResidentStackedW8A8
	NE, TopK, DenseInter, MoeInter                                   int
}

// G4MoEDefectForTest plants one defect in the block, for its gate's planted-defect arm (gpu/gemma4_moe_test.go): 1 the
// per-expert scale skipped, 2 post_ffn_norm_1 skipped, 3 the expert branch fed pre_ffn_norm, 4 the joint post_ffn_norm
// skipped, 5 the expert sum not cleared between tokens, 6 the gate and up halves exchanged, 7 the router fed the dense
// branch's weighted norm. 0, the only value outside that test, plants nothing. It is read when a runner is built.
var G4MoEDefectForTest int

// ensureGemma4MoE compiles the block's two kernels (lazily, once per Context).
func (c *Context) ensureGemma4MoE() error {
	if c.g4TailCopyPipeline != nil {
		return nil
	}
	var err error
	if _, c.g4RouteScalePipeline, c.g4RouteScaleLayout, err = c.mkPipeline("g4RouteScale", wgsl.G4RouteScaleWGSL); err != nil {
		return err
	}
	_, c.g4TailCopyPipeline, c.g4TailCopyLayout, err = c.mkPipeline("g4TailCopy", wgsl.G4TailCopyWGSL)
	return err
}
