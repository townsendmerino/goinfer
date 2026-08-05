//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// held-out factual prose for teacher-forced scoring (≈200 tokens).
const ssmHeldOut = `The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. ` +
	`It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. ` +
	`Locally nicknamed "La dame de fer", it was constructed as the centerpiece of the 1889 World's Fair. ` +
	`The tower is 330 metres tall, about the same height as an 81-storey building, and the tallest structure in Paris. ` +
	`Its base is square, measuring 125 metres on each side. During its construction, the Eiffel Tower surpassed the ` +
	`Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until ` +
	`the Chrysler Building in New York City was finished in 1930. The tower has three levels for visitors, with ` +
	`restaurants on the first and second levels. Photosynthesis is the process by which green plants convert light ` +
	`energy into chemical energy stored in glucose. Water boils at one hundred degrees Celsius at sea level pressure.`

func softmaxOf(logits []float32) []float64 {
	mx := float64(logits[0])
	for _, v := range logits {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	out := make([]float64, len(logits))
	var s float64
	for i, v := range logits {
		e := math.Exp(float64(v) - mx)
		out[i] = e
		s += e
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

func argmaxOf(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func top5(p []float64) map[int]bool {
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

// TestSSMInt8Quality scores the int8 resident SSM path for granite-4.0-h-tiny against
// (R1) f32 CPU ground truth and (R2) the staged default, on teacher-forced agreement,
// distribution distance (KL, top-5 overlap), and perplexity over a held-out sample.
// Measurement only; loads sequentially (granite f32 ≈ 28 GB RAM each). Deliverable:
// docs/ssm-int8-quality.md.
func TestSSMInt8Quality(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() || os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("ssm int8 quality eval — slow (5 granite loads); set GOINFER_SSM_QUALITY=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/granite/granite-4.0-h-tiny-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no granite model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmHeldOut, true)
	if len(ids) > 220 {
		ids = ids[:220]
	}
	N := len(ids) - 1 // positions 0..N-1 predict ids[i+1]
	t.Logf("held-out: %d tokens scored", N)

	// ---- R1: f32 CPU ground truth. Store per-position dist + argmax. ----
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	r1, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load R1: %v", err)
	}
	r1Dist := make([][]float64, N)
	r1Arg := make([]int, N)
	r1NLL := 0.0
	cache := r1.NewCache(len(ids) + 1)
	for i := 0; i < N; i++ {
		lg, e := r1.ForwardForTest(ids[i], cache)
		if e != nil {
			t.Fatal(e)
		}
		r1Dist[i] = softmaxOf(lg)
		r1Arg[i] = argmaxOf(lg)
		r1NLL += -math.Log(math.Max(r1Dist[i][ids[i+1]], 1e-12))
	}
	r1.Close()
	t.Logf("R1 (f32 CPU): perplexity=%.3f", math.Exp(r1NLL/float64(N)))

	// score: teacher-forced agreement + KL(R1||X) + top5 overlap + perplexity for path X.
	score := func(name string, dist [][]float64, arg []int, ids []int) {
		agree, klSum, t5Sum, nll := 0, 0.0, 0.0, 0.0
		for i := 0; i < N; i++ {
			if arg[i] == r1Arg[i] {
				agree++
			}
			r1t5, xt5 := top5(r1Dist[i]), top5(dist[i])
			inter := 0
			for k := range r1t5 {
				if xt5[k] {
					inter++
				}
			}
			t5Sum += float64(inter) / 5
			for v, p := range r1Dist[i] {
				if p > 1e-9 {
					klSum += p * (math.Log(p) - math.Log(math.Max(dist[i][v], 1e-12)))
				}
			}
			nll += -math.Log(math.Max(dist[i][ids[i+1]], 1e-12))
		}
		t.Logf("%-22s agreement=%.1f%%  meanKL(R1||X)=%.4f  top5overlap=%.1f%%  perplexity=%.3f (R1 %.3f)",
			name, 100*float64(agree)/float64(N), klSum/float64(N), 100*t5Sum/float64(N), math.Exp(nll/float64(N)), math.Exp(r1NLL/float64(N)))
	}

	// ---- INT8 resident ----
	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	res, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load INT8: %v", err)
	}
	if !res.ResidentActive() {
		t.Fatal("INT8 not resident")
	}
	rf := res.ResidentForwardForTest()
	resDist := make([][]float64, N)
	resArg := make([]int, N)
	rf.Reset()
	for i := 0; i < N; i++ {
		lg, e := rf.Forward(res.EmbedResidentForTest(ids[i]), i)
		if e != nil {
			t.Fatal(e)
		}
		resDist[i] = softmaxOf(lg)
		resArg[i] = argmaxOf(lg)
	}
	score("INT8-resident vs R1", resDist, resArg, ids)
	res.Close()
	os.Unsetenv("GOINFER_SSM_RESIDENT")

	// ---- R2: staged default (webgpu, no SSM residency) ----
	r2, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Logf("load R2 staged failed: %v (skipping R2)", err)
	} else {
		r2Dist := make([][]float64, N)
		r2Arg := make([]int, N)
		c2 := r2.NewCache(len(ids) + 1)
		for i := 0; i < N; i++ {
			lg, e := r2.ForwardForTest(ids[i], c2)
			if e != nil {
				t.Fatal(e)
			}
			r2Dist[i] = softmaxOf(lg)
			r2Arg[i] = argmaxOf(lg)
		}
		score("R2-staged vs R1", r2Dist, r2Arg, ids)
		r2.Close()
	}

	// ---- free-running + qualitative: resident vs f32 CPU on diverse prompts ----
	prompts := []string{
		"The capital of France is",
		"Write a Python function that returns the nth Fibonacci number:",
		"Q: What is the boiling point of water at sea level?\nA:",
		"Once upon a time, in a small village,",
		"The three primary colors are",
		"Explain photosynthesis in one sentence:",
	}
	gen := func(m *decoder.Model, p string, n int) string {
		pid, _ := tk.Encode(p, true)
		ch, _ := m.Generate(context.Background(), pid, n, decoder.SamplingParams{Temperature: 0})
		var out []int
		for tok := range ch {
			out = append(out, tok)
		}
		s, _ := tk.Decode(out)
		return s
	}
	r1b, _ := decoder.Load(path, decoder.Options{Backend: "cpu"})
	r1Gen := make([]string, len(prompts))
	for i, p := range prompts {
		r1Gen[i] = gen(r1b, p, 48)
	}
	r1b.Close()
	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	resb, _ := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	t.Logf("=== free-running: resident(int8) vs f32 CPU ===")
	for i, p := range prompts {
		rg := gen(resb, p, 48)
		match := rg == r1Gen[i]
		t.Logf("PROMPT: %q", p)
		t.Logf("  f32 : %q", r1Gen[i])
		t.Logf("  int8: %q  [identical=%v]", rg, match)
	}
	resb.Close()
	os.Unsetenv("GOINFER_SSM_RESIDENT")
}
