//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGemma3ResidentParity is the real-checkpoint gate for Gemma 3 on CUDA: the CUDA resident
// forward must agree with the CPU forward on a real gemma-3-4b-it, under the repo's 3%
// near-tie rule (gpu/kv_i8_parity_test.go). Gemma exercises five features no other CUDA-
// admitted family does: (1+w) RMS, sandwich norms, GeGLU, embed scale, per-layer RoPE base.
func TestGemma3ResidentParity(t *testing.T) {
	requireHeavyModel(t)
	residentCosineParity(t, os.ExpandEnv("$HOME/models/gemma-3-4b-it"),
		"The capital of France is Paris. The city is")
}

// TestDenseResidentParity is the CONTROL for the Gemma gate: the same harness on the
// known-good dense Qwen path, so "is 0.999 good?" has an answer measured on this box rather
// than assumed. Without it, a Gemma cosine cannot be judged.
func TestDenseResidentParity(t *testing.T) {
	requireHeavyModel(t)
	residentCosineParity(t, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
		"The capital of France is Paris. The city is")
}

// residentCosineParity drives BOTH paths from the SAME text, tokenized by the model's OWN
// tokenizer.
//
// It used to take hardcoded token ids, and the Gemma gate's were ones I invented and never
// decoded: they were not valid Gemma tokens at all, so every Gemma parity number reported from
// here (and inherited by the Metal port) was measured on gibberish. The parity CLAIM survived
// that — both paths got identical input, and agreement is agreement — but the ANALYSIS did not:
// the control ran on real Qwen ids while Gemma ran on nonsense, and nonsense flattens the
// logits, which inflates near-ties and depresses exact-argmax. I read that signature as
// "Gemma is noisier, probably the 262k vocab" instead of as a confound I had created.
//
// Encoding real text makes a wrong id impossible BY CONSTRUCTION rather than by eyeballing, and
// makes the two models comparable: same sentence, each in its own vocabulary.
func residentCosineParity(t *testing.T, path, text string) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED — admission says it should be admitted")
	}
	t.Logf("admitted; required features = %v", mc.RequiredResidentFeatures())

	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	_, _, _, _, _, _, vocab := mc.Dims()
	minCos := 1.0
	tk, err := loadTok(path)
	if err != nil {
		t.Skipf("no tokenizer for %s: %v", path, err)
	}
	prompt, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(prompt) < 4 {
		t.Fatalf("encode(%q) gave only %d ids — too short to gate anything", text, len(prompt))
	}
	// Prove the ids ROUND-TRIP before trusting a single number measured on them. This is the
	// check whose absence made the whole exercise worthless.
	back, err := tk.Decode(prompt)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !strings.Contains(strings.ReplaceAll(back, " ", ""), "capitalofFrance") {
		t.Fatalf("prompt ids do not round-trip: encode(%q) -> %v -> %q. The gate would be measuring "+
			"gibberish, which is exactly how a Gemma parity number got reported on ids that decoded "+
			"to \"<bos>ath হই of carry\".", text, prompt, back)
	}
	t.Logf("prompt: %d ids, round-trips to %q", len(prompt), back)

	cache := mcpu.NewCache(len(prompt) + 2)
	exact, hard := 0, 0
	worst := 0.0
	for i, tok := range prompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		// Cosine is the real signal here: argmax over a 262k vocab is a coarse, high-variance
		// statistic, so a low exact-match rate can be pure near-tie churn rather than a bug.
		var dot, na, nb float64
		for j := range cpuL {
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na) * math.Sqrt(nb))
		if c < minCos {
			minCos = c
		}
		t.Logf("  pos %2d cosine %.6f", i, c)
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			exact++
			continue
		}
		lo, hi := cpuL[0], cpuL[0]
		for _, v := range cpuL {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
		if gap > worst {
			worst = gap
		}
		if gap > 0.03 {
			hard++
			t.Errorf("pos %d: CPU=%d CUDA=%d gap=%.3f%% > 3%% (vocab %d)", i, ca, ga, gap*100, vocab)
		}
	}
	t.Logf("CUDA vs CPU: %d/%d exact | worst near-tie %.3f%% | hard fails %d | min cosine %.6f",
		exact, len(prompt), worst*100, hard, minCos)
	// The GATE is the repo's own rule (gpu/kv_i8_parity_test.go): argmax must match, or differ
	// only inside a 3% near-tie — asserted per position above. Cosine is logged as a DIAGNOSTIC,
	// with only a gross-breakage floor: an early draft of this test asserted cosine ≥ 0.999 and
	// that bar failed the SHIPPED dense Qwen path (min 0.9936), which is why the control below
	// exists. W4A8 int4 does not reproduce CPU int4 to 0.999; a tighter floor here would encode a
	// number no backend meets.
	if minCos < 0.95 {
		t.Errorf("logit cosine %.6f < 0.95 — far below the dense control (~0.99); that is gross breakage, not int4 noise", minCos)
	}
}

// loadTok mirrors cmd/serve's loader: a .gguf carries its tokenizer in its own metadata, an HF
// dir has tokenizer.json/SentencePiece alongside the weights.
func loadTok(path string) (*tokenizer.Tokenizer, error) {
	if strings.HasSuffix(path, ".gguf") {
		return tokenizer.LoadGGUF(path)
	}
	return tokenizer.Load(path)
}
