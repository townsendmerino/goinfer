//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestL01_unpermuteFast_roundTrips is the correctness gate unpermuteFast's own doc comment
// claims: a fixed nibble-position permutation, content-independent, so proving it round-trips
// for varied words already proves it for every possible word (the map never looks at nibble
// VALUES). Kept as a real test alongside the one-off 2,000,000-sample check this pass already
// ran manually (docs/task-l01-hybrid-moe-cpu-gpu.md), so CI keeps re-proving it.
func TestL01_unpermuteFast_roundTrips(t *testing.T) {
	for i := 0; i < 100000; i++ {
		x := uint32(i) * 2654435761
		if got := unpermuteFast(permuteFast(x)); got != x {
			t.Fatalf("unpermuteFast(permuteFast(0x%08x)) = 0x%08x, want 0x%08x", x, got, x)
		}
	}
}

// TestL01_cpuExtraction_matchesModelOwnWeights is the correctness question
// docs/task-l01-hybrid-moe-cpu-gpu.md's own prototype note names before anything is wired into
// the decode path: does l01ComputeExpert's extraction from CUDA's C′ pinned host stack
// (fast-nibble-permuted, unpermuted back here) produce the SAME expert computation the model's
// own CPU-native weights would? Both loads read the identical pre-quantized bytes from the same
// .giw-equivalent source (no runtime requant), so a correct extraction should match closely —
// this is the gate that would have caught a wrong permutation, a wrong field (moeInter vs
// inter, gate/up half split), or a wrong scale decode BEFORE any of that touched a real decode.
func TestL01_cpuExtraction_matchesModelOwnWeights(t *testing.T) {
	// qwen35-tiny (model_type qwen3_5_moe_text): the SAME family as L-01's own audit decision
	// rule target (Qwen3.6-35B-A3B), hidden_act=silu, hidden=64/moeInter=32 (both multiples of
	// 32 — tiny-qwen2-moe's hidden=32/moeInter=44 fails the int4 CUDA kernel's group-size
	// requirement and never builds resident). NOT gemma4: gemma4's MoE uses a different weight
	// shape and gelu-tanh GeGLU activation (decoder/forward_gemma4_moe.go's own comment — "does
	// NOT reuse the SiLU moeMLP/swiGLUExpert"), so it would never populate the generic
	// LayerWeights.Experts this test compares against.
	const dir = "../testdata/qwen35-tiny"
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")

	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("cuda load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skip("not resident on cuda")
	}
	if !rf.cacheExperts {
		t.Fatal("TAUTOLOGY GUARD: cacheExperts did not engage — srcW/srcS would be nil and this test would pass having tested nothing")
	}

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("cpu load: %v", err)
	}
	defer mcpu.Close()

	layer := -1
	for i := range rf.layers {
		if rf.layers[i].expGU.srcW != nil {
			layer = i
			break
		}
	}
	if layer < 0 {
		t.Fatal("no MoE layer found with a populated pinned host stack")
	}
	experts := mcpu.Weights().Layers[layer].Experts
	if len(experts) == 0 {
		t.Fatalf("cpu model's layer %d has no Experts — fixture/layer mismatch", layer)
	}

	rng := rand.New(rand.NewSource(7))
	hidden := rf.hidden
	tryExperts := []int{0, 1, len(experts) - 1}
	for _, e := range tryExperts {
		if e < 0 || e >= len(experts) {
			continue
		}
		t.Run(fmt.Sprintf("expert%d", e), func(t *testing.T) {
			for trial := 0; trial < 3; trial++ {
				h := make([]float32, hidden)
				for i := range h {
					h[i] = rng.Float32()*2 - 1
				}

				gotDst := make([]float32, hidden)
				rf.l01ComputeExpert(&rf.layers[layer], e, h, gotDst)

				ew := experts[e]
				wantDst := make([]float32, hidden)
				gateScr := make([]float32, rf.moeInter)
				upScr := make([]float32, rf.moeInter)
				decoder.ComputeExpertMLP(decoder.CPUExpertWeights{Gate: ew.Gate, Up: ew.Up, Down: ew.Down},
					h, wantDst, rf.moeInter, gateScr, upScr)

				var dot, na, nb, maxAbs float64
				for i := range gotDst {
					a, b := float64(gotDst[i]), float64(wantDst[i])
					dot += a * b
					na += a * a
					nb += b * b
					if d := math.Abs(a - b); d > maxAbs {
						maxAbs = d
					}
				}
				cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
				t.Logf("layer=%d expert=%d trial=%d cosine=%.8f maxAbsDiff=%.6g", layer, e, trial, cos, maxAbs)
				// NOT bit-identical by construction: this fixture loads from safetensors (f32
				// source), so the cuda and cpu backends each quantize independently at load
				// time rather than aliasing the same pre-quantized bytes (unlike a .giw bundle,
				// where they would be). A tiny per-group rounding difference between the two
				// quantizers is expected; a wrong permutation/field/expert-index bug would miss
				// by orders of magnitude more (cosine well below 0.99, not 0.99998+).
				const wantCosine = 0.9999
				const wantMaxAbs = 1e-3
				if cos < wantCosine || maxAbs > wantMaxAbs {
					t.Errorf("layer=%d expert=%d trial=%d: cosine=%.8f (want >=%v) maxAbsDiff=%.6g (want <=%v) — extraction likely wrong (permutation/field/scale bug)",
						layer, e, trial, cos, wantCosine, maxAbs, wantMaxAbs)
				}
			}
		})
	}
}
