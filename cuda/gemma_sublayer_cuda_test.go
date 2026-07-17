//go:build cuda

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// captureSublayersForTest runs one resident token with per-sublayer capture on, returning the
// dp4a-path attention contribution (o-proj out, post sandwich-norm) and MLP contribution (down
// out) per layer — the CUDA analogue of decoder.ForwardSubCapture.
func (r *cudaResident) captureSublayersForTest(emb []float32, pos int) (attn, mlp [][]float32, err error) {
	err = r.do(func() error {
		r.subCap = true
		r.subAttnC = make([][]float32, r.nLayers)
		r.subMLPC = make([][]float32, r.nLayers)
		defer func() { r.subCap = false }()
		e := r.launchToken(emb, pos)
		attn, mlp = r.subAttnC, r.subMLPC
		return e
	})
	return
}

// TestGemmaSublayerCUDA answers the Metal box's Fork-2 question directly: does CUDA's dp4a W4A8
// path show the same 2-6x amplitude inflation Metal shows on the o-proj contribution at
// channels 1723/227 (L31-33), or is CUDA's amplitude clean? If clean, Metal's blow-up is a
// kernel scale bug separate from int4 quant-hostility.
//
// int4 = the byte-identical Q4_K_M gguf (sha 882e8d2d) run through the CUDA RESIDENT dp4a path
// (not CPU int4); f32 truth = the real bf16 safetensors via decoder.ForwardSubCapture. Metal's
// numbers (from the relay): 1723 L32 attn +175 vs truth +27 (~6.5x); 227 L33 attn +63 (flipped)
// vs truth -12.
func TestGemmaSublayerCUDA(t *testing.T) {
	gguf := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	bf16 := os.ExpandEnv("$HOME/models/gemma-3-4b-it")
	if _, e := os.Stat(gguf); e != nil {
		t.Skipf("no gguf at %s", gguf)
	}
	if _, e := os.Stat(bf16); e != nil {
		t.Skipf("no bf16 at %s", bf16)
	}
	prompt := []int{2, 669, 5279, 529, 7001, 563}
	const probe = 5
	channels := []int{1723, 227}

	// CUDA resident dp4a int4 from the shared gguf.
	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (cuda): %v", err)
	}
	rfIface := mc.ResidentForwardForTest()
	if rfIface == nil {
		mc.Close()
		t.Fatal("gguf gemma-3 did not go resident on CUDA — need the dp4a path for this test")
	}
	rf := rfIface.(*cudaResident)
	var cAttn, cMLP [][]float32
	for i, tok := range prompt {
		emb := mc.EmbedResidentForTest(tok)
		if i == probe {
			a, m, err := rf.captureSublayersForTest(emb, i)
			if err != nil {
				mc.Close()
				t.Fatalf("cuda capture: %v", err)
			}
			cAttn, cMLP = a, m
		} else if _, err := rf.Forward(emb, i); err != nil {
			mc.Close()
			t.Fatalf("cuda warm pos %d: %v", i, err)
		}
	}
	mc.Close()

	// f32 truth from bf16.
	mf, err := decoder.Load(bf16, decoder.Options{Quant: ""})
	if err != nil {
		t.Fatalf("load bf16 (f32): %v", err)
	}
	defer mf.Close()
	cache := mf.NewCache(len(prompt) + 1)
	var fAttn, fMLP [][]float32
	for i, tok := range prompt {
		if i == probe {
			a, m, err := mf.ForwardSubCapture(tok, cache)
			if err != nil {
				t.Fatalf("f32 ForwardSubCapture: %v", err)
			}
			fAttn, fMLP = a, m
		} else if _, err := mf.ForwardForTest(tok, cache); err != nil {
			t.Fatalf("f32 warm pos %d: %v", i, err)
		}
	}

	ratio := func(v, ref float32) float64 {
		if math.Abs(float64(ref)) < 1e-3 {
			return math.NaN()
		}
		return float64(v) / float64(ref)
	}
	sgn := func(x float32) string {
		if (x > 0) != (x >= 0) {
			return "0"
		}
		if x > 0 {
			return "+"
		}
		return "-"
	}
	for _, c := range channels {
		t.Logf("=== channel %d: CUDA dp4a vs f32 truth (Metal inflates o-proj 2-6x here) ===", c)
		for l := 31; l <= 33; l++ {
			fa, ca := fAttn[l][c], cAttn[l][c]
			fm, cm := fMLP[l][c], cMLP[l][c]
			flip := ""
			if sgn(ca) != sgn(fa) {
				flip = "  <- attn SIGN-FLIP"
			}
			t.Logf("  L%d attn: f32=%8.2f cuda=%8.2f (%.2fx) | mlp: f32=%8.2f cuda=%8.2f (%.2fx)%s",
				l, fa, ca, ratio(ca, fa), fm, cm, ratio(cm, fm), flip)
		}
	}
}
