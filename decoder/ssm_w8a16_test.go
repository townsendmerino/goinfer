package decoder

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Control A (docs/ssm-int8-quality.md): does int8-WEIGHTS + f32-ACTIVATIONS (W8A16) beat the
// W8A8 ceiling? If yes, the resident's gap (which the Phase-A capture pinned on int8 activation
// re-quantization across the deep stack, NOT the mamba kernels) is fixable with a W8A16 GPU GEMV
// at ~no decode cost (activation is negligible vs the N*K weight). Quant:"int8" loads per-row
// int8 weights and dispatches MatmulBTQ8 = f32 activation × int8 weight (W8A16). Quant:"int8int8"
// is W8A8 (the int8 activation path). Mamba in/out_proj stay f32 here (they are not WeightMat);
// GOINFER_SSM_Q8WEIGHTS additionally round-trips the mamba weights to int8 (weights-only) for a
// full W8A16. Scored teacher-forced vs the f32 reference (R1).
func TestSSMW8A16Ceiling(t *testing.T) {
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("W8A16 ceiling control — slow (granite cpu loads); set GOINFER_SSM_QUALITY=1")
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmLocHeldOut, true)
	if len(ids) > 220 {
		ids = ids[:220]
	}
	N := len(ids) - 1

	softmax := func(lg []float32) []float64 {
		mx := float64(lg[0])
		for _, v := range lg {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		out := make([]float64, len(lg))
		var s float64
		for i, v := range lg {
			e := math.Exp(float64(v) - mx)
			out[i] = e
			s += e
		}
		for i := range out {
			out[i] /= s
		}
		return out
	}
	argmax := func(v []float32) int {
		bi, bv := 0, v[0]
		for i, x := range v {
			if x > bv {
				bv, bi = x, i
			}
		}
		return bi
	}
	top5 := func(p []float64) map[int]bool {
		type kv struct {
			i int
			p float64
		}
		a := make([]kv, len(p))
		for i, v := range p {
			a[i] = kv{i, v}
		}
		sort.Slice(a, func(x, y int) bool { return a[x].p > a[y].p })
		m := map[int]bool{}
		for i := 0; i < 5 && i < len(a); i++ {
			m[a[i].i] = true
		}
		return m
	}
	run := func(m *Model) ([][]float64, []int) {
		cache := m.NewCache(len(ids) + 1)
		dist := make([][]float64, N)
		arg := make([]int, N)
		for i := range N {
			lg, e := m.ForwardForTest(ids[i], cache)
			if e != nil {
				t.Fatal(e)
			}
			dist[i] = softmax(lg)
			arg[i] = argmax(lg)
		}
		return dist, arg
	}

	// R1: f32 reference.
	r1, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load R1: %v", err)
	}
	r1Dist, r1Arg := run(r1)
	r1NLL := 0.0
	for i := range N {
		r1NLL += -math.Log(math.Max(r1Dist[i][ids[i+1]], 1e-12))
	}
	r1.Close()
	t.Logf("R1 (f32): perplexity=%.3f", math.Exp(r1NLL/float64(N)))

	score := func(name string, dist [][]float64, arg []int) {
		agree, kl, t5, nll := 0, 0.0, 0.0, 0.0
		for i := range N {
			if arg[i] == r1Arg[i] {
				agree++
			}
			r1t5, xt5 := top5(r1Dist[i]), top5(dist[i])
			inter := 0
			for kk := range r1t5 {
				if xt5[kk] {
					inter++
				}
			}
			t5 += float64(inter) / 5
			for v, p := range r1Dist[i] {
				if p > 1e-9 {
					kl += p * (math.Log(p) - math.Log(math.Max(dist[i][v], 1e-12)))
				}
			}
			nll += -math.Log(math.Max(dist[i][ids[i+1]], 1e-12))
		}
		t.Logf("%-40s agreement=%.1f%%  meanKL=%.4f  top5=%.1f%%  perplexity=%.3f (R1 %.3f)",
			name, 100*float64(agree)/float64(N), kl/float64(N), 100*t5/float64(N),
			math.Exp(nll/float64(N)), math.Exp(r1NLL/float64(N)))
	}

	// W8A16: int8 weights (Quant:"int8" → MatmulBTQ8), f32 activations. attn/MoE/lm_head only;
	// mamba in/out_proj stay f32 unless GOINFER_SSM_Q8WEIGHTS round-trips them (weights-only int8).
	ssmQ8WeightsOnly = os.Getenv("GOINFER_SSM_Q8WEIGHTS") != ""
	wm, err := Load(path, Options{Backend: "cpu", Quant: "int8"})
	if err != nil {
		t.Fatalf("load W8A16: %v", err)
	}
	if q8, _, w8a8, ok := wm.w.Layers[0].QProj.Int8(); ok {
		t.Logf("W8A16 check: QProj int8=%v w8a8(activation-quant)=%v len(q8)=%d", ok, w8a8, len(q8))
	}
	wDist, wArg := run(wm)
	wm.Close()
	ssmQ8WeightsOnly = false
	mambaNote := "f32 mamba"
	if os.Getenv("GOINFER_SSM_Q8WEIGHTS") != "" {
		mambaNote = "int8 mamba weights"
	}
	score("W8A16 (int8 weights, f32 act; "+mambaNote+")", wDist, wArg)
}
