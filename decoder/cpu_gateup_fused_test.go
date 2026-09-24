package decoder

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestGatedMLPFusedGateUp_bitIdentical pins the claim cpu_gateup_fused.go rests on: the fused fork/join
// produces, bit for bit, what the unfused path produces — gate·h and up·h through the ordinary parallel
// int4 matmul, then swiglu over the whole vector. Compared with != on every element (never a
// tolerance), across fan-out widths that do and do not divide N, ragged N, and several K, because the
// chunk boundaries are the only thing this path adds and a boundary bug shows up as one wrong element.
func TestGatedMLPFusedGateUp_bitIdentical(t *testing.T) {
	arch := &Architecture{Act: ActSiLU}
	shapes := []struct{ N, K int }{
		{8960, 1536}, // 1.5B gate/up
		{4864, 896},  // 0.5B gate/up
		{517, 256},   // ragged N: no width divides it
		{64, 64},     // N < 2*16: worker count clamps
		{4, 32},      // tiny
	}
	defer linalg.SetParallelWidth(0)
	for _, sh := range shapes {
		rng := rand.New(rand.NewPCG(uint64(sh.N), uint64(sh.K)))
		fill := func(n int, scale float32) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = (rng.Float32()*2 - 1) * scale
			}
			return v
		}
		lw := &LayerWeights{
			GateProj: linalg.QuantizeInt4(fill(sh.N*sh.K, 0.5), sh.N, sh.K, 32),
			UpProj:   linalg.QuantizeInt4(fill(sh.N*sh.K, 0.5), sh.N, sh.K, 32),
		}
		h := fill(sh.K, 1.5)

		// Reference: the unfused path's own two matmuls + a serial swiglu over the full vector.
		wantGate, wantUp := make([]float32, sh.N), make([]float32, sh.N)
		ws := &linalg.Workspace{}
		ws.SetThreshold(int4ParThreshold)
		lw.GateProj.MatmulBTW4A8Into(ws, h, wantGate, 1)
		lw.UpProj.MatmulBTW4A8Into(ws, h, wantUp, 1)
		swiglu(wantGate, wantUp)

		for _, width := range []int{1, 2, 3, 5, 8, 16, 0} {
			t.Run(fmt.Sprintf("N%dK%d/width%d", sh.N, sh.K, width), func(t *testing.T) {
				linalg.SetParallelWidth(width)
				scr := &decodeScratch{gate: make([]float32, sh.N), up: make([]float32, sh.N)}
				if !gatedMLPFusedGateUp(h, lw, arch, scr) {
					t.Fatal("fused path declined a canonical W4A8 SiLU MLP")
				}
				for i := range wantGate {
					if scr.gate[i] != wantGate[i] {
						t.Fatalf("gate[%d] = %v, unfused %v — not bit-identical", i, scr.gate[i], wantGate[i])
					}
				}
			})
		}
	}
}

// TestGatedMLPFusedGateUp_declines pins the preconditions that keep it off paths it was not measured
// or built for: a non-SiLU activation, a non-int4 weight, and a scratch too small for the output.
func TestGatedMLPFusedGateUp_declines(t *testing.T) {
	const N, K = 64, 64
	w := make([]float32, N*K)
	lw := &LayerWeights{
		GateProj: linalg.QuantizeInt4(w, N, K, 32),
		UpProj:   linalg.QuantizeInt4(w, N, K, 32),
	}
	h := make([]float32, K)
	scr := &decodeScratch{gate: make([]float32, N), up: make([]float32, N)}
	if gatedMLPFusedGateUp(h, lw, &Architecture{Act: ActGeluTanh}, scr) {
		t.Error("fused path ran for a GeGLU activation")
	}
	small := &decodeScratch{gate: make([]float32, N-1), up: make([]float32, N)}
	if gatedMLPFusedGateUp(h, lw, &Architecture{Act: ActSiLU}, small) {
		t.Error("fused path ran with a gate scratch shorter than N")
	}
	f32 := &LayerWeights{GateProj: linalg.WrapF32(w, N, K), UpProj: linalg.WrapF32(w, N, K)}
	if gatedMLPFusedGateUp(h, f32, &Architecture{Act: ActSiLU}, scr) {
		t.Error("fused path ran for f32 weights")
	}
}
