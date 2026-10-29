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

// TestGatedDeltaNetStep_nilBackendNonF32 checks that gatedDeltaNetStep with a nil Backend routes its WeightMat projections
// through matvecWMPre and matvecWM, producing bit-identical output, state and conv window for quantized and BF16 weights.
func TestGatedDeltaNetStep_nilBackendNonF32(t *testing.T) {
	const (
		hidden   = 32
		hk, hv   = 8, 8
		nk, nv   = 2, 4
		convK    = 4
		keyDim   = nk * hk
		valueDim = nv * hv
		convDim  = 2*keyDim + valueDim
		eps      = 1e-6
		steps    = 4
	)
	p := qwen35Params{
		ConvKernel:    convK,
		KeyHeadDim:    hk,
		ValueHeadDim:  hv,
		NumKeyHeads:   nk,
		NumValueHeads: nv,
	}
	fill := func(n int, seed float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(math.Sin(float64(i+1)*float64(seed))) * 0.25
		}
		return out
	}
	toBF16 := func(f []float32) []uint16 {
		u := make([]uint16, len(f))
		for i, v := range f {
			u[i] = uint16(math.Float32bits(v) >> 16)
		}
		return u
	}
	bits := func(t *testing.T, got, want []float32, what string) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: len %d != %d", what, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]: got %v (%08x), want %v (%08x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}

	qkvF32 := fill(convDim*hidden, 1.1)
	zF32 := fill(valueDim*hidden, 1.3)
	outF32 := fill(hidden*valueDim, 1.7)
	inProjB := fill(nv*hidden, 0.7)
	inProjA := fill(nv*hidden, 0.9)
	convW := fill(convDim*convK, 0.5)
	dtBias := fill(nv, 0.2)
	negExpA := negExpAFromLog(fill(nv, 0.3))
	normW := fill(hv, 1.5)

	cases := []struct {
		name string
		wrap func(f []float32, rows, cols int) linalg.WeightMat
	}{
		{
			name: "int8_w8a8",
			wrap: func(f []float32, rows, cols int) linalg.WeightMat {
				return linalg.QuantizeInt8(f, rows, cols, true)
			},
		},
		{
			name: "int4_w4a8",
			wrap: func(f []float32, rows, cols int) linalg.WeightMat {
				return linalg.QuantizeInt4(f, rows, cols, 32)
			},
		},
		{
			name: "bf16",
			wrap: func(f []float32, rows, cols int) linalg.WeightMat {
				return linalg.WrapBF16(toBF16(f), rows, cols)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := &deltaNetWeights{
				inProjQKV: tc.wrap(qkvF32, convDim, hidden),
				inProjZ:   tc.wrap(zF32, valueDim, hidden),
				inProjB:   inProjB,
				inProjA:   inProjA,
				convW:     convW,
				dtBias:    dtBias,
				negExpA:   negExpA,
				normW:     normW,
				outProj:   tc.wrap(outF32, hidden, valueDim),
			}
			gotSt := newDeltaState(p)
			wantSt := newDeltaState(p)
			for step := range steps {
				h := fill(hidden, float32(step+2)*0.37)
				got := gatedDeltaNetStep(nil, h, w, p, hidden, eps, gotSt)

				wantSt.hq.prepare(nil, &w.inProjQKV, h, 1)
				mixed := matvecWMPre(nil, &w.inProjQKV, &wantSt.hq, h)
				bt := matvec(w.inProjB, nv, hidden, h)
				at := matvec(w.inProjA, nv, hidden, h)
				z := matvecWMPre(nil, &w.inProjZ, &wantSt.hq, h)
				core := make([]float32, valueDim)
				deltaNetCore(core, mixed, bt, at, z, w, p, eps, wantSt)
				want := matvecWM(nil, &w.outProj, core)

				nonZero := false
				for _, v := range want {
					if v != 0 {
						nonZero = true
						break
					}
				}
				if !nonZero {
					t.Fatalf("step %d: reference output is all zero", step)
				}
				bits(t, got, want, "out")
				bits(t, gotSt.s, wantSt.s, "state S")
				if len(gotSt.convWin) != len(wantSt.convWin) {
					t.Fatalf("step %d: convWin len %d != %d", step, len(gotSt.convWin), len(wantSt.convWin))
				}
				for i := range wantSt.convWin {
					bits(t, gotSt.convWin[i], wantSt.convWin[i], "convWin")
				}
			}
		})
	}
}
