package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestMamba2_parity gates goinfer's sequential Mamba-2 scan against an isolated HF
// GraniteMoeHybrid mamba mixer (scripts/pin_granite_mamba.py) — the "one genuinely
// new thing" for the Granite family, validated in isolation before any loader, the
// way deltanet's parity was first proven. Bit-exact-ish (f32 reductions reorder), so
// gate on cosine + max-abs.
func TestMamba2_parity(t *testing.T) {
	const golden = "../testdata/granite_mamba_golden.json"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_granite_mamba.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Dims struct {
			Hidden  int     `json:"hidden"`
			NHeads  int     `json:"n_heads"`
			HeadDim int     `json:"head_dim"`
			DState  int     `json:"d_state"`
			NGroups int     `json:"n_groups"`
			DConv   int     `json:"d_conv"`
			DInner  int     `json:"d_inner"`
			Seq     int     `json:"seq"`
			RMSEps  float64 `json:"rms_eps"`
		} `json:"dims"`
		Weights struct {
			InProj  []float32 `json:"in_proj"`
			ConvW   []float32 `json:"conv1d_w"`
			ConvB   []float32 `json:"conv1d_b"`
			ALog    []float32 `json:"A_log"`
			D       []float32 `json:"D"`
			DtBias  []float32 `json:"dt_bias"`
			Norm    []float32 `json:"norm"`
			OutProj []float32 `json:"out_proj"`
		} `json:"weights"`
		Input  []float32 `json:"input"`
		Output []float32 `json:"output"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	p := mamba2Params{
		NHeads: g.Dims.NHeads, HeadDim: g.Dims.HeadDim, DState: g.Dims.DState,
		NGroups: g.Dims.NGroups, DConv: g.Dims.DConv, Hidden: g.Dims.Hidden,
	}
	w := &mamba2Weights{
		inProj: g.Weights.InProj, convW: g.Weights.ConvW, convB: g.Weights.ConvB,
		aLog: g.Weights.ALog, d: g.Weights.D, dtBias: g.Weights.DtBias,
		normW: g.Weights.Norm, outProj: g.Weights.OutProj,
	}

	hidden, seq := g.Dims.Hidden, g.Dims.Seq
	in := make([][]float32, seq)
	for t := range seq {
		in[t] = g.Input[t*hidden : (t+1)*hidden]
	}
	out := mamba2Seq(in, w, p, g.Dims.RMSEps)

	var dot, na, nb, maxAbs float64
	for t := range seq {
		for i := range hidden {
			a, b := float64(out[t][i]), float64(g.Output[t*hidden+i])
			dot += a * b
			na += a * a
			nb += b * b
			if d := math.Abs(a - b); d > maxAbs {
				maxAbs = d
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	t.Logf("mamba2 parity: cosine=%.8f max|Δ|=%.2e over %d tokens", cos, maxAbs, seq)
	if cos < 0.99999 {
		t.Errorf("cosine %.8f < 0.99999", cos)
	}
	if maxAbs > 1e-3 {
		t.Errorf("max abs diff %.2e > 1e-3", maxAbs)
	}
}
