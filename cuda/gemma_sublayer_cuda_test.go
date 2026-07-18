//go:build cuda

package cuda

import (
	"math"
	"os"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// captureSublayersForTest runs one resident token with per-sublayer capture on, returning the
// dp4a-path attention contribution (o-proj out, post sandwich-norm) and MLP contribution (down
// out) per layer — the CUDA analogue of decoder.ForwardSubCapture.
func (r *cudaResident) captureSublayersForTest(emb []float32, pos int) (attn, mlp, ctx, mlpPre [][]float32, err error) {
	err = r.do(func() error {
		r.subCap = true
		r.subAttnC = make([][]float32, r.nLayers)
		r.subMLPC = make([][]float32, r.nLayers)
		r.subCtxC = make([][]float32, r.nLayers)
		r.subMLPpreC = make([][]float32, r.nLayers)
		defer func() { r.subCap = false }()
		e := r.launchToken(emb, pos)
		attn, mlp, ctx, mlpPre = r.subAttnC, r.subMLPC, r.subCtxC, r.subMLPpreC
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
			a, m, _, _, err := rf.captureSublayersForTest(emb, i)
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
			a, m, _, _, err := mf.ForwardSubCapture(tok, cache)
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

// TestGemmaContextCUDA is the cross-box discriminator the Metal box asked for: the pre-o-proj
// attention CONTEXT (qDim), which every proven-faithful kernel downstream feeds on. If Metal's
// context matches CUDA's, the divergence is in the weights (candidate 3); if it differs, it is
// the attention path (candidate 2). CUDA's context is the reference the Metal box cannot produce
// on one machine.
//
// This reports CUDA-int4 context vs f32 truth (is CUDA's attention faithful?) and the
// top-magnitude context elements, so the Metal box overlays its own r.ctx at the same qDim
// indices. CPU-int4 (same gguf) is printed too as the parity cross-check on the resident path.
func TestGemmaContextCUDA(t *testing.T) {
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

	// CUDA resident int4 context.
	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (cuda): %v", err)
	}
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		mc.Close()
		t.Fatal("gguf gemma-3 not resident on CUDA")
	}
	var cudaCtx [][]float32
	for i, tok := range prompt {
		emb := mc.EmbedResidentForTest(tok)
		if i == probe {
			_, _, ctx, _, err := rf.captureSublayersForTest(emb, i)
			if err != nil {
				mc.Close()
				t.Fatalf("cuda ctx capture: %v", err)
			}
			cudaCtx = ctx
		} else if _, err := rf.Forward(emb, i); err != nil {
			mc.Close()
			t.Fatalf("cuda warm %d: %v", i, err)
		}
	}
	mc.Close()

	// CPU contexts: f32 truth (bf16) and int4 (same gguf, resident-path parity cross-check).
	cpuCtx := func(path, quant string) [][]float32 {
		m, err := decoder.Load(path, decoder.Options{Quant: quant})
		if err != nil {
			t.Fatalf("load %s (%s): %v", path, quant, err)
		}
		defer m.Close()
		cache := m.NewCache(len(prompt) + 1)
		var ctx [][]float32
		for i, tok := range prompt {
			if i == probe {
				_, _, c, _, err := m.ForwardSubCapture(tok, cache)
				if err != nil {
					t.Fatalf("ForwardSubCapture: %v", err)
				}
				ctx = c
			} else if _, err := m.ForwardForTest(tok, cache); err != nil {
				t.Fatalf("warm %d: %v", i, err)
			}
		}
		return ctx
	}
	f32Ctx := cpuCtx(bf16, "")
	cpuI4Ctx := cpuCtx(gguf, "int4")

	qDim := len(f32Ctx[0])
	nL := len(f32Ctx)
	t.Logf("pre-o-proj attention context, qDim=%d, pos %d", qDim, probe)

	// The TARGET CURVE for the Metal f32-KV fix: cosine(int4 ctx, f32) per layer, full depth.
	// Metal's f32-KV context should land ON this curve (goinfer f32-KV, int4-weight), not merely
	// "up" from its f16-KV crater. CUDA and CPU int4 agree — either is the reference.
	t.Logf("TARGET CURVE — cosine(int4 ctx, f32-truth) per layer [Metal f32-KV should reproduce this]:")
	for l := 0; l < nL; l++ {
		t.Logf("  L%-2d  CUDA-int4=%.6f  CPU-int4=%.6f", l, cosine(cudaCtx[l], f32Ctx[l]), cosine(cpuI4Ctx[l], f32Ctx[l]))
	}

	for _, l := range []int{31, 32, 33} {
		cCos := cosine(cudaCtx[l], f32Ctx[l])
		pCos := cosine(cpuI4Ctx[l], f32Ctx[l])
		t.Logf("L%d: cosine(CUDA-int4 ctx, f32) = %.6f | cosine(CPU-int4 ctx, f32) = %.6f", l, cCos, pCos)
		// Top-magnitude context elements (the massive channel lives here and sets the o-proj int8
		// scale). These qDim indices are what the Metal box overlays its r.ctx onto.
		idx := make([]int, qDim)
		for i := range idx {
			idx[i] = i
		}
		fl := f32Ctx[l]
		sort.Slice(idx, func(a, b int) bool { return math.Abs(float64(fl[idx[a]])) > math.Abs(float64(fl[idx[b]])) })
		t.Logf("   top-6 |ctx| qDim-idx:  %-8s %12s %12s %12s", "idx", "f32", "CUDA-int4", "CPU-int4")
		for r := 0; r < 6 && r < qDim; r++ {
			q := idx[r]
			t.Logf("                         %-8d %12.4f %12.4f %12.4f", q, f32Ctx[l][q], cudaCtx[l][q], cpuI4Ctx[l][q])
		}
	}
}
