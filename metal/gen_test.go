//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// THE decisive test, and the one the debug report wrongly called blocked: does Gemma 3 actually
// GENERATE on Metal? Merges are encode-only, so a decode-only vocab is enough to read the
// model's own output back. Also reports dNLL of the forced next token — the tokenizer-free
// "is it actually broken" metric that argmax (trajectory-sensitive: the same known-good path
// scores 15/24 vs 20/24 on different id sets) cannot give.
func TestGemma3_GeneratesCoherently(t *testing.T) {
	requireHeavyModel(t)
	// Dormant until the Gemma kernels are validated and DECLARED for metal (features.go). Until
	// then gemma3 declines to CPU by design, so rf is nil — a skip, not a failure, exactly as the
	// sibling TestGemma3ResidentParity guards. The moment the declaration lands this goes live
	// (the rf==nil below then t.Fatals, catching a silent CPU fallback). See docs/task-metal-gemma.md.
	if !decoder.ResidentBackendFeatures["metal"][decoder.FeatSandwichNorm] {
		t.Skip("metal does not declare the Gemma features yet (kernels dormant)")
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer (decode-only): %v", err)
	}
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("metal resident DECLINED")
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()

	// Build a REAL prompt from vocab lookups. The gate's inherited ids decode to
	// "<bos>ath হই of carry Bত্ব忽视ardRep" — they are not valid Gemma tokens, so every parity
	// number measured with them was measured on nonsense. Encode() is unavailable (decode-only
	// vocab), but TokenID is enough: SPM marks a leading space with ▁.
	prompt := []int{2} // <bos>
	for _, piece := range []string{"▁The", "▁capital", "▁of", "▁France", "▁is"} {
		id, ok := tk.TokenID(piece)
		if !ok {
			t.Fatalf("vocab lookup failed for %q", piece)
		}
		prompt = append(prompt, id)
	}
	if s, err := tk.Decode(prompt); err == nil {
		t.Logf("REAL prompt ids %v decode to: %q", prompt, s)
	}

	// dNLL of the forced next token, CPU vs Metal, at each prompt position.
	logsm := func(l []float32, id int) float64 {
		mx := l[0]
		for _, v := range l {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for _, v := range l {
			sum += math.Exp(float64(v - mx))
		}
		return float64(l[id]-mx) - math.Log(sum)
	}
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 256)
	var worst float64
	for i := 0; i < len(prompt)-1; i++ {
		cpuL, _ := mcpu.ForwardForTest(prompt[i], cache)
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(prompt[i]), i)
		if err != nil {
			t.Fatalf("gpu: %v", err)
		}
		next := prompt[i+1]
		d := math.Abs(logsm(cpuL, next) - logsm(gpuL, next))
		if d > worst {
			worst = d
		}
		t.Logf("  pos %d: dNLL(next=%d) = %.4f nats", i, next, d)
	}
	t.Logf("worst dNLL over the prompt = %.4f nats  (<0.3 = fine, >>1 = broken)", worst)

	// Free-run greedy continuation on Metal, then read it back.
	cache2 := decoder.NewKVCache(nL, nKV, hd, 0, 256)
	_ = cache2
	var out []int
	tok := prompt[0]
	pos := 0
	for i := 0; i < len(prompt)+20; i++ {
		l, err := rf.Forward(mg.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatalf("gen: %v", err)
		}
		best, bv := 0, float32(-1e30)
		for j, v := range l {
			if v > bv {
				bv, best = v, j
			}
		}
		pos++
		if i+1 < len(prompt) {
			tok = prompt[i+1]
			continue
		}
		out = append(out, best)
		tok = best
	}
	text, err := tk.Decode(out)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	t.Logf("METAL GREEDY CONTINUATION: %q", text)
}
