//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func cosine64(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// TestActGroup_phi3ResidentMatchesCPU: Phi-3-mini resident on CUDA at int8int8 with per-32 activation
// scales (queue-engineering.md H2's configuration that passed the quality gate) — admitted despite
// the activation-quantization hazard, it runs the actgroup.cu kernels; its logits over a ~130-token
// chat prompt (filler, the regime that broke per-vector scales) match the CPU per-32 path to kernel
// accumulation order, and hold the gate's quality bar against goinfer's own f32.
//
//	GOINFER_PHI3_MINI_GGUF=~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda -run TestActGroup_phi3Resident -v
func TestActGroup_phi3ResidentMatchesCPU(t *testing.T) {
	path := os.Getenv("GOINFER_PHI3_MINI_GGUF")
	if path == "" {
		t.Skip("set GOINFER_PHI3_MINI_GGUF")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the ", 110) + "\n\nThe text above is only filler. Now write a long, detailed essay about the history of the printing press."
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
	if err != nil {
		t.Fatal(err)
	}

	// ResidentContext 512: 4096 positions of f32 KV plus the int8 weights exceed an 8 GB card.
	gpu, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int8int8", ActQuantGroup: 32, ResidentContext: 512})
	if err != nil {
		t.Fatalf("cuda load: %v", err)
	}
	defer gpu.Close()
	rf := gpu.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("resident did not engage: %s", gpu.ResidentDecline())
	}
	if r, ok := rf.(*cudaResident); !ok || !r.actG32 {
		t.Fatal("resident engaged without the per-32 kernels")
	}
	var gl [][]float32
	for p, id := range ids {
		lg, err := rf.Forward(gpu.EmbedResidentForTest(id), p)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", p, err)
		}
		gl = append(gl, append([]float32(nil), lg...))
	}

	cpuLogits := func(opts decoder.Options) [][]float32 {
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("cpu load %+v: %v", opts, err)
		}
		defer m.Close()
		c := m.NewCache(len(ids) + 1)
		out := make([][]float32, len(ids))
		for p, id := range ids {
			lg, err := m.ForwardForTest(id, c)
			if err != nil {
				t.Fatal(err)
			}
			out[p] = append([]float32(nil), lg...)
		}
		return out
	}
	stats := func(name string, ref [][]float32) (p10, median float64) {
		cs := make([]float64, len(ids))
		for p := range ids {
			cs[p] = cosine64(gl[p], ref[p])
		}
		sort.Float64s(cs)
		t.Logf("%s: %d positions, cosine min %.6f p10 %.6f median %.6f", name, len(ids), cs[0], cs[len(cs)/10], cs[len(cs)/2])
		return cs[len(cs)/10], cs[len(cs)/2]
	}
	// The kernels are responsible for matching the CPU per-32 path, not for the prompt's own
	// difficulty: on this prompt the CPU per-32 path itself sits at p10 ~0.93 against f32 (measured
	// 2026-09-25; the gate's prompt, without the "\n\n", gave 0.973). So the bars are relative:
	// median agreement with CPU per-32, and quality no worse than CPU per-32's on the same prompt.
	// Measured: agreement median 0.99983; quality p10 0.934 (CUDA) vs 0.930 (CPU).
	cpuG, f32 := cpuLogits(decoder.Options{Quant: "int8int8", ActQuantGroup: 32}), cpuLogits(decoder.Options{})
	_, agreeMed := stats("cuda per-32 vs cpu per-32", cpuG)
	cudaQ, _ := stats("cuda per-32 vs cpu f32", f32)
	gl = cpuG
	cpuQ, _ := stats("cpu per-32 vs cpu f32 (same prompt, the baseline)", f32)
	if agreeMed < 0.9995 {
		t.Errorf("CUDA per-32 vs CPU per-32: median cosine %.6f, want >= 0.9995 (same algorithm; differs only by kernel rounding)", agreeMed)
	}
	if cudaQ < cpuQ-0.01 {
		t.Errorf("CUDA per-32 quality p10 %.4f is worse than the CPU per-32 path's %.4f on the same prompt", cudaQ, cpuQ)
	}
}
