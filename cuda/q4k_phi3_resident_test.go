//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestQ4K_phi3ResidentMatchesCPU: Phase 1b of docs/tasks/task-int4-weight-quality-2026-09.md. Phi-3
// at --quant q4k on CUDA, at the DEFAULT resident context (the capacity criterion: int8int8 does not
// fit there on the 8 GB card), with gemv_q4k_g32 actually bound, against the CPU q4k path on the H2
// resident test's 141-token filler prompt. The registered bars are relative:
//   - agreement: per-position logit cosine against CPU q4k, median ≥ 0.999;
//   - quality: p10 against f32 no more than 0.01 below CPU q4k's own on the same prompt.
func TestQ4K_phi3ResidentMatchesCPU(t *testing.T) {
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

	gm, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "q4k"})
	if err != nil {
		t.Fatalf("cuda load: %v", err)
	}
	defer gm.Close()
	rf := gm.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("q4k did not build resident at the default context: %s", gm.ResidentDecline())
	}
	r, ok := rf.(*cudaResident)
	if !ok || !r.actG32 || r.gemvQ4K == (Pipeline{}) {
		t.Fatal("resident engaged without per-32 activations and gemv_q4k_g32")
	}
	nQ4K := 0
	for _, Ly := range r.layers {
		for _, w := range []cudaWQ{Ly.q, Ly.k, Ly.v, Ly.o, Ly.g, Ly.u, Ly.d} {
			if w.kind == "q4k" {
				nQ4K++
			}
		}
	}
	if nQ4K == 0 {
		t.Fatal("no projection is on the q4k kernel: the test would measure int8 only")
	}
	t.Logf("resident: context %d positions, %d projections on gemv_q4k_g32, fused per-32: %v", r.ctxCap, nQ4K, r.fuseG32)
	var gl [][]float32
	for p, id := range ids {
		lg, err := rf.Forward(gm.EmbedResidentForTest(id), p)
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
	stats := func(name string, a, ref [][]float32) (p10, median float64) {
		cs := make([]float64, len(ids))
		for p := range ids {
			cs[p] = cosine64(a[p], ref[p])
		}
		sort.Float64s(cs)
		t.Logf("%s: %d positions, cosine min %.6f p10 %.6f median %.6f", name, len(ids), cs[0], cs[len(cs)/10], cs[len(cs)/2])
		return cs[len(cs)/10], cs[len(cs)/2]
	}
	cpuQ4K, f32 := cpuLogits(decoder.Options{Quant: "q4k"}), cpuLogits(decoder.Options{})
	_, agree := stats("cuda q4k vs cpu q4k", gl, cpuQ4K)
	cudaQ, _ := stats("cuda q4k vs cpu f32", gl, f32)
	cpuQ, _ := stats("cpu q4k vs cpu f32 (same prompt, the baseline)", cpuQ4K, f32)
	if agree < 0.999 {
		t.Errorf("CUDA q4k vs CPU q4k: median cosine %.6f, want >= 0.999", agree)
	}
	if cudaQ < cpuQ-0.01 {
		t.Errorf("CUDA q4k quality p10 %.4f is more than 0.01 below the CPU q4k path's %.4f", cudaQ, cpuQ)
	}
}
