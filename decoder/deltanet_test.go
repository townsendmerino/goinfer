package decoder

import (
	"encoding/json"
	"errors"
	"github.com/townsendmerino/aikit/linalg"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestGatedDeltaNet_parity checks goinfer's Gated DeltaNet primitive op-for-op
// against a traced HF layer (scripts/pin_qwen35_deltanet.py forces the exact
// per-step recurrence). Validates the conv + gates + gated delta recurrence +
// gated RMSNorm + out_proj end to end, independent of the model weight loader.
func TestGatedDeltaNet_parity(t *testing.T) {
	const path = "../testdata/qwen35_deltanet_golden.json"
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — run scripts/pin_qwen35_deltanet.py", path)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Dims map[string]float64   `json:"dims"`
		W    map[string][]float32 `json:"weights"`
		In   []float32            `json:"input"`
		Out  []float32            `json:"output"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	d := func(k string) int { return int(g.Dims[k]) }
	hidden, seq := d("hidden"), d("seq")
	p := qwen35Params{
		ConvKernel:    d("conv_kernel"),
		KeyHeadDim:    d("head_k_dim"),
		ValueHeadDim:  d("head_v_dim"),
		NumKeyHeads:   d("num_k_heads"),
		NumValueHeads: d("num_v_heads"),
	}
	keyDim, valueDim := p.KeyHeadDim*p.NumKeyHeads, p.ValueHeadDim*p.NumValueHeads
	convDim := 2*keyDim + valueDim
	w := &deltaNetWeights{
		inProjQKV: linalg.WrapF32(g.W["in_proj_qkv"], convDim, hidden),
		inProjZ:   linalg.WrapF32(g.W["in_proj_z"], valueDim, hidden),
		inProjB:   g.W["in_proj_b"],
		inProjA:   g.W["in_proj_a"],
		convW:     g.W["conv1d_weight"],
		dtBias:    g.W["dt_bias"],
		negExpA:   negExpAFromLog(g.W["A_log"]),
		normW:     g.W["norm_weight"],
		outProj:   linalg.WrapF32(g.W["out_proj"], hidden, valueDim),
	}

	h := make([][]float32, seq)
	for t := range seq {
		h[t] = g.In[t*hidden : (t+1)*hidden]
	}
	out := gatedDeltaNet(&cpuBackend{}, h, w, p, hidden, g.Dims["rms_eps"])

	// Compare flattened output: max abs diff + cosine. (chunk vs recurrent fp
	// drift is removed by the golden, so the tolerance is tight.)
	var maxAbs, dot, na, nb float64
	for t := range seq {
		for j := range hidden {
			got, want := float64(out[t][j]), float64(g.Out[t*hidden+j])
			if ad := math.Abs(got - want); ad > maxAbs {
				maxAbs = ad
			}
			dot += got * want
			na += got * got
			nb += want * want
		}
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("Gated DeltaNet parity: maxAbs=%.2e cosine=%.8f", maxAbs, cos)
	if maxAbs > 1e-4 {
		t.Errorf("max abs diff %.3e > 1e-4", maxAbs)
	}
	if cos < 0.99999 {
		t.Errorf("cosine %.8f < 0.99999", cos)
	}
}
