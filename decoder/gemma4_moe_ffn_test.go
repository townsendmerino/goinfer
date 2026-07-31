package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestGemma4MoEFFN_parity gates the Gemma 4 parallel dense+MoE FFN sub-block
// (gemma4MoEFFN) against the HF oracle in isolation (scripts/pin_gemma4_moe_ffn.py:
// a real Gemma4TextDecoderLayer with attention zeroed, so input→output is purely
// the FFN sub-block + layer_scalar). f32 golden — any divergence is the router
// pre-norm/scale, per-expert scale, the three parallel-branch norms, the gelu-tanh
// experts, the join, or layer_scalar, not quant noise. The scaling params are
// strengthened in the generator so identity/no-op bugs can't hide (Phase 2).
func TestGemma4MoEFFN_parity(t *testing.T) {
	const path = "../testdata/gemma4_moe_ffn_golden.json"
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_gemma4_moe_ffn.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Dims struct {
			Hidden     int     `json:"hidden"`
			DenseInter int     `json:"dense_inter"`
			MoeInter   int     `json:"moe_inter"`
			NumExperts int     `json:"num_experts"`
			TopK       int     `json:"top_k"`
			Seq        int     `json:"seq"`
			RmsEps     float64 `json:"rms_eps"`
		} `json:"dims"`
		Weights map[string][]float32 `json:"weights"`
		Input   []float32            `json:"input"`
		Output  []float32            `json:"output"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	d := g.Dims
	W := g.Weights

	// per-expert WeightMats sliced out of the fused [E,2*moe,hidden] / [E,hidden,moe] blobs.
	gu, dn := W["experts_gate_up"], W["experts_down"]
	guStride, dnStride := 2*d.MoeInter*d.Hidden, d.Hidden*d.MoeInter
	eGU := make([]linalg.WeightMat, d.NumExperts)
	eDn := make([]linalg.WeightMat, d.NumExperts)
	for e := 0; e < d.NumExperts; e++ {
		eGU[e] = linalg.WrapF32(gu[e*guStride:(e+1)*guStride], 2*d.MoeInter, d.Hidden)
		eDn[e] = linalg.WrapF32(dn[e*dnStride:(e+1)*dnStride], d.Hidden, d.MoeInter)
	}
	w := &gemma4MoEWeights{
		preFFNNorm:     W["pre_ffn_norm"],
		postFFNNorm1:   W["post_ffn_norm_1"],
		preFFNNorm2:    W["pre_ffn_norm_2"],
		postFFNNorm2:   W["post_ffn_norm_2"],
		postFFNNorm:    W["post_ffn_norm"],
		mlpGate:        linalg.WrapF32(W["mlp_gate"], d.DenseInter, d.Hidden),
		mlpUp:          linalg.WrapF32(W["mlp_up"], d.DenseInter, d.Hidden),
		mlpDown:        linalg.WrapF32(W["mlp_down"], d.Hidden, d.DenseInter),
		routerProj:     linalg.WrapF32(W["router_proj"], d.NumExperts, d.Hidden),
		routerScale:    W["router_scale"],
		perExpertScale: W["per_expert_scale"],
		expertsGateUp:  eGU,
		expertsDown:    eDn,
		layerScalar:    W["layer_scalar"][0],
		denseInter:     d.DenseInter,
		moeInter:       d.MoeInter,
		nE:             d.NumExperts,
		topK:           d.TopK,
	}
	arch := &Architecture{HiddenDim: d.Hidden, NormEps: d.RmsEps}
	be := &cpuBackend{}

	var maxAbs, dot, na, nb float64
	for s := 0; s < d.Seq; s++ {
		h := g.Input[s*d.Hidden : (s+1)*d.Hidden]
		got := gemma4MoEFFN(be, arch, h, w)
		want := g.Output[s*d.Hidden : (s+1)*d.Hidden]
		for i := range got {
			if ad := math.Abs(float64(got[i] - want[i])); ad > maxAbs {
				maxAbs = ad
			}
			dot += float64(got[i]) * float64(want[i])
			na += float64(got[i]) * float64(got[i])
			nb += float64(want[i]) * float64(want[i])
		}
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("gemma4 MoE FFN parity: maxAbs=%.2e cosine=%.8f over %d tokens", maxAbs, cos, d.Seq)
	if maxAbs > 1e-3 {
		t.Errorf("max abs diff %.2e > 1e-3", maxAbs)
	}
	if cos < 0.99999 {
		t.Errorf("cosine %.8f < 0.99999", cos)
	}
}
