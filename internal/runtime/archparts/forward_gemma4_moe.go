package archparts

import (
	"github.com/townsendmerino/aikit/linalg"
)

// Gemma4MoEWeights is one layer's FFN sub-block, as WeightMats (matmul path) + f32
// norm/scale vectors. Built by the loader (production) or a golden (the op-test).
type Gemma4MoEWeights struct {
	PreFFNNorm   []float32 // pre_feedforward_layernorm   (dense branch pre-norm)
	PostFFNNorm1 []float32 // post_feedforward_layernorm_1 (dense branch post-norm)
	PreFFNNorm2  []float32 // pre_feedforward_layernorm_2  (moe branch pre-norm)
	PostFFNNorm2 []float32 // post_feedforward_layernorm_2 (moe branch post-norm)
	PostFFNNorm  []float32 // post_feedforward_layernorm   (joint post-norm on x1+x2)

	MlpGate, MlpUp, MlpDown linalg.WeightMat // dense MLP at denseInter

	RouterProj     linalg.WeightMat // [nE, hidden]
	RouterScale    []float32        // [hidden] learned pre-projection scale
	PerExpertScale []float32        // [nE] learned scale on the renormalized top-k weights

	ExpertsGateUp []linalg.WeightMat // per expert [2*moeInter, hidden] (gate‖up, contiguous)
	ExpertsDown   []linalg.WeightMat // per expert [hidden, moeInter]

	LayerScalar                    float32
	DenseInter, MoeInter, NE, TopK int
}
