//go:build gpu

package kernels

import "github.com/oliverbestmann/webgpu/wgpu"

// Gemma 4's parallel dense and MoE feed-forward (enable_moe_block; decoder's gemma4MoEFFN) in the decode runner. The
// router's top-k and the indexed expert GEMVs are the runner's own MoE kernels; this file adds the two steps they lack.

// g4RouteScaleWGSL: wgt[j] *= pes[idx[j]] over the k chosen experts, the per-expert scale on the renormalised top-k
// weights.
const g4RouteScaleWGSL = `
struct P { k: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       idx: array<u32>; // [k]
@group(0) @binding(1) var<storage, read>       pes: array<f32>; // [nE]
@group(0) @binding(2) var<storage, read_write> wgt: array<f32>; // [k]
@group(0) @binding(3) var<uniform>             p:   P;
@compute @workgroup_size(1)
fn main() {
    for (var j: u32 = 0u; j < p.k; j = j + 1u) { wgt[j] = wgt[j] * pes[idx[j]]; }
}
`

// g4TailCopyWGSL: dst[i] = src[off + i] for i < n, the up half of an expert's fused gate‖up output.
const g4TailCopyWGSL = `
struct P { n: u32, off: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { dst[i] = src[p.off + i]; }
}
`

// g4MoELayer is one enable_moe_block layer's feed-forward on the device.
type G4MoELayer struct {
	PreFFNNorm, PostFFNNorm1, PreFFNNorm2, PostFFNNorm2, PostFFNNorm *wgpu.Buffer // [hidden] each
	RouterNorm                                                       *wgpu.Buffer // the weight that makes the runner's RMSNorm weightless
	Router                                                           *wgpu.Buffer // [nE, hidden] float32, the learned scale and hidden^-0.5 folded in
	PerExpertScale, ZeroBias                                         *wgpu.Buffer // [nE]
	Gate, Up, Down                                                   DecodeWeight // the dense branch
	ExpGateUp, ExpDown                                               *ResidentStackedW8A8
	NE, TopK, DenseInter, MoeInter                                   int
}

// g4MoEDefectForTest plants one defect in the block, for its gate's planted-defect arm (gpu/gemma4_moe_test.go): 1 the
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
	if _, c.g4RouteScalePipeline, c.g4RouteScaleLayout, err = c.mkPipeline("g4RouteScale", g4RouteScaleWGSL); err != nil {
		return err
	}
	_, c.g4TailCopyPipeline, c.g4TailCopyLayout, err = c.mkPipeline("g4TailCopy", g4TailCopyWGSL)
	return err
}
