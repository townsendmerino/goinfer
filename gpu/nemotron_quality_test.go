//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// P5/P6: the headline. Nemotron-H is the SAME deep hybrid + recurrent-mamba architecture whose
// int8 resident gap granite proved FUNDAMENTAL and precision-invariant — but it has NO MoE router,
// so the verdict is OPEN. Measure it: resident int4 (int8 doesn't fit 8 GB — see
// docs/nemotron-resident.md; granite proved the gap is precision-invariant so int4≈int8) vs the
// f32 CPU reference (R1), teacher-forced agreement/KL/top-5/perplexity, plus free-running coherence
// and decode speed. The AGREEMENT NUMBER drives the default-vs-opt-in flip — no pre-set bar.
func TestNemotronResidentQuality(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("nemotron resident quality (slow: 9B f32 + int4 loads); set GOINFER_SSM_QUALITY=1")
	}
	if _, err := New(); err != nil {
		t.Skipf("no webgpu: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no nemotron model: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids, _ := tk.Encode(ssmHeldOut, true) // shared held-out prose
	if len(ids) > 160 {
		ids = ids[:160] // cap for the slow f32 9B reference
	}
	N := len(ids) - 1
	t.Logf("held-out: %d tokens scored", N)

	// ---- R1: f32 CPU ground truth ----
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	r1, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load R1 (cpu f32): %v", err)
	}
	r1Dist := make([][]float64, N)
	r1Arg := make([]int, N)
	r1NLL := 0.0
	cache := r1.NewCache(len(ids) + 1)
	for i := range N {
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

	score := func(name string, dist [][]float64, arg []int) {
		agree, klSum, t5Sum, nll := 0, 0.0, 0.0, 0.0
		for i := range N {
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
		t.Logf("%-26s agreement=%.1f%%  meanKL(R1||X)=%.4f  top5overlap=%.1f%%  perplexity=%.3f (R1 %.3f)",
			name, 100*float64(agree)/float64(N), klSum/float64(N), 100*t5Sum/float64(N),
			math.Exp(nll/float64(N)), math.Exp(r1NLL/float64(N)))
	}

	// ---- resident int4 ----
	os.Setenv("GOINFER_SSM_RESIDENT", "1")
	defer os.Unsetenv("GOINFER_SSM_RESIDENT")
	res, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load resident int4: %v", err)
	}
	defer res.Close()
	if !res.ResidentActive() {
		t.Fatal("9B int4 did not go resident")
	}
	rf := res.ResidentForwardForTest()
	resDist := make([][]float64, N)
	resArg := make([]int, N)
	rf.Reset()
	for i := range N {
		lg, e := rf.Forward(res.EmbedResidentForTest(ids[i]), i)
		if e != nil {
			t.Fatal(e)
		}
		resDist[i] = softmaxOf(lg)
		resArg[i] = argmaxOf(lg)
	}
	score("int4-resident vs R1", resDist, resArg)

	// ---- free-running coherence + speed ----
	prompts := []string{"The capital of France is", "Q: What is 2+2?\nA:", "The three primary colors are"}
	gen := func(p string, n int) string {
		pid, _ := tk.Encode(p, true)
		ch, _ := res.Generate(context.Background(), pid, n, decoder.SamplingParams{Temperature: 0})
		var out []int
		for tok := range ch {
			out = append(out, tok)
		}
		s, _ := tk.Decode(out)
		return s
	}
	for _, p := range prompts {
		t.Logf("  PROMPT %q → %q", p, gen(p, 24))
	}
	// speed: best-of-3 rounds, 32 decode steps each
	pid, _ := tk.Encode("Tell me about the solar system.", true)
	best := time.Hour
	for range 3 {
		t0 := time.Now()
		ch, _ := res.Generate(context.Background(), pid, 32, decoder.SamplingParams{Temperature: 0})
		nt := 0
		for range ch {
			nt++
		}
		if d := time.Since(t0) / time.Duration(max1(nt, 1)); d < best {
			best = d
		}
	}
	t.Logf("int4-resident SPEED: %.1f ms/token = %.1f tok/s", float64(best.Microseconds())/1000, 1e6/float64(best.Microseconds()))
}

func max1(a, b int) int {
	if a > b {
		return a
	}
	return b
}
