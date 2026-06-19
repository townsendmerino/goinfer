//go:build gpu

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

// Benign-vs-harmful characterization of int4-resident Nemotron-H's ~7.5% argmax disagreements vs
// f32. Near-lossless perplexity/KL with sub-95% agreement should mean the disagreements are
// coin-flips on near-tied tokens (int4 picked f32's #2/#3 where f32 was itself indifferent), not
// confident-token mistakes. The robust signal is the CORRELATION: benign ⇒ disagreements
// concentrate at SMALL f32 top1–top2 margins; harmful ⇒ at LARGE margins (f32 confident, int4 wrong).
func TestNemotronBenignHarmful(t *testing.T) {
	if os.Getenv("GOINFER_SSM_QUALITY") == "" {
		t.Skip("nemotron benign/harmful analysis (slow: 9B f32 + int4); set GOINFER_SSM_QUALITY=1")
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
	ids, _ := tk.Encode(ssmHeldOut, true)
	N := len(ids) - 1
	t.Logf("held-out: %d tokens scored", N)

	// ---- f32 CPU reference: keep the full distribution per position ----
	os.Unsetenv("GOINFER_SSM_RESIDENT")
	r1, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load R1: %v", err)
	}
	r1Dist := make([][]float64, N)
	r1Arg := make([]int, N)
	cache := r1.NewCache(len(ids) + 1)
	for i := 0; i < N; i++ {
		lg, e := r1.ForwardForTest(ids[i], cache)
		if e != nil {
			t.Fatal(e)
		}
		r1Dist[i] = softmaxOf(lg)
		r1Arg[i] = argmaxOf(lg)
	}
	r1.Close()

	// ---- int4 resident ----
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
	resArg := make([]int, N)
	resDist := make([][]float64, N)
	rf.Reset()
	for i := 0; i < N; i++ {
		lg, e := rf.Forward(res.EmbedResidentForTest(ids[i]), i)
		if e != nil {
			t.Fatal(e)
		}
		resDist[i] = softmaxOf(lg)
		resArg[i] = argmaxOf(lg)
	}

	// helpers over f32's distribution
	top2 := func(d []float64) (i1, i2 int, p1, p2 float64) {
		i1, i2, p1, p2 = 0, 0, -1, -1
		for i, p := range d {
			if p > p1 {
				i2, p2 = i1, p1
				i1, p1 = i, p
			} else if p > p2 {
				i2, p2 = i, p
			}
		}
		return
	}
	rankUnder := func(d []float64, tok int) int { // 1 = highest-prob
		pt := d[tok]
		r := 1
		for _, p := range d {
			if p > pt {
				r++
			}
		}
		return r
	}

	// ---- analysis ----
	agree, disagree, top2agree := 0, 0, 0
	var marginAgree, marginDisagree []float64 // f32 top1-top2 margin at agree / disagree
	var conf1Agree, conf1Disagree []float64   // f32 top-1 prob
	rankHist := map[string]int{"2": 0, "3": 0, "4-5": 0, "6-10": 0, ">10": 0}
	klDisagree := 0.0
	// "harmful" = int4 picked a token f32 ranked low (>3) AND f32 was confident there
	// (top1-top2 margin > 0.10). benign = everything else among disagreements.
	harmful := 0
	var harmfulPos []int
	for i := 0; i < N; i++ {
		i1, i2, p1, p2 := top2(r1Dist[i])
		margin := p1 - p2
		if resArg[i] == r1Arg[i] {
			agree++
			marginAgree = append(marginAgree, margin)
			conf1Agree = append(conf1Agree, p1)
			top2agree++
			continue
		}
		disagree++
		marginDisagree = append(marginDisagree, margin)
		conf1Disagree = append(conf1Disagree, p1)
		if resArg[i] == i1 || resArg[i] == i2 { // (i1 can't equal here since disagree, so == i2)
			top2agree++
		}
		rk := rankUnder(r1Dist[i], resArg[i])
		switch {
		case rk <= 2:
			rankHist["2"]++
		case rk == 3:
			rankHist["3"]++
		case rk <= 5:
			rankHist["4-5"]++
		case rk <= 10:
			rankHist["6-10"]++
		default:
			rankHist[">10"]++
		}
		// per-position KL(f32 || int4)
		var kl float64
		for v, p := range r1Dist[i] {
			if p > 1e-9 {
				kl += p * (math.Log(p) - math.Log(math.Max(resDist[i][v], 1e-12)))
			}
		}
		klDisagree += kl
		if rk > 3 && margin > 0.10 {
			harmful++
			harmfulPos = append(harmfulPos, i)
		}
	}
	mean := func(xs []float64) float64 {
		if len(xs) == 0 {
			return 0
		}
		s := 0.0
		for _, x := range xs {
			s += x
		}
		return s / float64(len(xs))
	}
	median := func(xs []float64) float64 {
		if len(xs) == 0 {
			return 0
		}
		c := append([]float64(nil), xs...)
		sort.Float64s(c)
		return c[len(c)/2]
	}
	benign := disagree - harmful

	t.Logf("agreement: %d/%d = %.1f%%  (disagreements: %d)", agree, N, 100*float64(agree)/float64(N), disagree)
	t.Logf("TOP-2 agreement: %d/%d = %.1f%%  (int4 pick is f32's #1 or #2)", top2agree, N, 100*float64(top2agree)/float64(N))
	t.Logf("f32 top-1 prob   — at AGREE: mean=%.3f median=%.3f | at DISAGREE: mean=%.3f median=%.3f",
		mean(conf1Agree), median(conf1Agree), mean(conf1Disagree), median(conf1Disagree))
	t.Logf("f32 top1-top2 margin — at AGREE: mean=%.3f median=%.3f | at DISAGREE: mean=%.3f median=%.3f",
		mean(marginAgree), median(marginAgree), mean(marginDisagree), median(marginDisagree))
	t.Logf("int4-pick rank under f32 (disagreements): #2=%d #3=%d #4-5=%d #6-10=%d >10=%d",
		rankHist["2"], rankHist["3"], rankHist["4-5"], rankHist["6-10"], rankHist[">10"])
	if disagree > 0 {
		t.Logf("mean per-position KL at disagreements: %.4f", klDisagree/float64(disagree))
	}
	t.Logf("BENIGN %d/%d = %.1f%% of disagreements (HARMFUL = rank>3 AND f32 margin>0.10: %d, positions %v)",
		benign, disagree, 100*float64(benign)/float64(max1(disagree, 1)), harmful, harmfulPos)

	// ---- free-running coherence (factual / instruction / code) ----
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
	for _, p := range []string{
		"The boiling point of water at sea level is",
		"List three primary colors:",
		"Write a Python function that returns the sum of a list:",
	} {
		t.Logf("  GEN %q → %q", p, gen(p, 40))
	}
}
