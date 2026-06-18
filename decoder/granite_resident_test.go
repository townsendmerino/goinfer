//go:build gpu

package decoder

import (
	"os"
	"testing"
)

// TestGraniteResidentBridge validates the P5b.1 accessor bridge: the resident SSM builder
// reads the Mamba geometry, per-layer mixer-kind, and correctly-shaped f32 mixer tensors
// through these accessors. Shape-checks the weights against the projDim/convDim/dInner the
// kernels expect, so a wiring error surfaces here before any GPU dispatch. Additive — no
// production routing change (eligibility flip is P6).
func TestGraniteResidentBridge(t *testing.T) {
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	m, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	nHeads, headDim, dState, nGroups, dConv, embMul, residMul, logitScale, attnScale, ok := m.GraniteResidentParams()
	if !ok {
		t.Fatal("GraniteResidentParams ok=false for a granite model")
	}
	dInner := nHeads * headDim
	convDim := dInner + 2*nGroups*dState
	projDim := 2*dInner + 2*nGroups*dState + nHeads
	t.Logf("granite mamba dims: nHeads=%d headDim=%d dState=%d nGroups=%d dConv=%d → dInner=%d convDim=%d projDim=%d",
		nHeads, headDim, dState, nGroups, dConv, dInner, convDim, projDim)
	t.Logf("multipliers: emb=%.4g resid=%.4g logitScale=%.4g attnScale=%.4g", embMul, residMul, logitScale, attnScale)

	hidden, _, _, _, _, _, _ := m.Dims()
	nLayers := m.w.arch.NumLayers
	mamba, attn := 0, 0
	want := map[string]int{"inProj": projDim * hidden, "convW": convDim * dConv, "convB": convDim,
		"aLog": nHeads, "d": nHeads, "dtBias": nHeads, "normW": dInner, "outProj": hidden * dInner}
	for l := 0; l < nLayers; l++ {
		if !m.GraniteMambaLayer(l) {
			attn++
			continue
		}
		mamba++
		inProj, convW, convB, aLog, dW, dtBias, normW, outProj := m.GraniteMambaWeights(l)
		got := map[string]int{"inProj": len(inProj), "convW": len(convW), "convB": len(convB),
			"aLog": len(aLog), "d": len(dW), "dtBias": len(dtBias), "normW": len(normW), "outProj": len(outProj)}
		for k, w := range want {
			if got[k] != w {
				t.Errorf("layer %d %s: len %d, want %d", l, k, got[k], w)
			}
		}
	}
	t.Logf("layer composition: %d mamba + %d attention = %d (mostly-Mamba hybrid)", mamba, attn, nLayers)
	if mamba == 0 {
		t.Error("no mamba layers found")
	}
}
