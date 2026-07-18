//go:build cuda

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemmaBOSBuild looks at how L0 builds the BOS token's massive activation — the Metal box
// found Metal under-builds it (goinfer |resid into L1| = 12491, Metal = 1461, 8.5x too weak),
// corrupting Gemma's attention sink and craterng every downstream context. This traces the
// build on the CUDA side: does the amplification live in L0's ATTENTION contribution or its
// MLP contribution, and does CUDA-int4 reach f32's magnitude (so the bug is Metal-specific) or
// under-build like Metal (an int4 issue CUDA tolerates)?
//
// Probe is pos 0 (BOS, id 2) — the sink token, where the massive activation is constructed.
func TestGemmaBOSBuild(t *testing.T) {
	gguf := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	bf16 := os.ExpandEnv("$HOME/models/gemma-3-4b-it")
	if _, e := os.Stat(gguf); e != nil {
		t.Skipf("no gguf at %s", gguf)
	}
	if _, e := os.Stat(bf16); e != nil {
		t.Skipf("no bf16 at %s", bf16)
	}
	const bos = 2
	const massiveCh = 443 // the dominant hidden channel at the sink

	l2 := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s)
	}
	add := func(a, b []float32) []float32 {
		o := make([]float32, len(a))
		for i := range a {
			o[i] = a[i] + b[i]
		}
		return o
	}

	// CUDA int4 build at BOS.
	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load gguf (cuda): %v", err)
	}
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		mc.Close()
		t.Fatal("gguf gemma-3 not resident on CUDA")
	}
	cudaEmb := append([]float32(nil), mc.EmbedResidentForTest(bos)...)
	cAttn, cMLP, _, cMLPpre, err := rf.captureSublayersForTest(cudaEmb, 0)
	if err != nil {
		mc.Close()
		t.Fatalf("cuda capture: %v", err)
	}
	mc.Close()

	// CPU f32 reference build at BOS.
	mf, err := decoder.Load(bf16, decoder.Options{Quant: ""})
	if err != nil {
		t.Fatalf("load bf16 (f32): %v", err)
	}
	fEmb := append([]float32(nil), mf.EmbedResidentForTest(bos)...)
	fcache := mf.NewCache(2)
	fAttn, fMLP, _, fMLPpre, err := mf.ForwardSubCapture(bos, fcache)
	mf.Close()
	if err != nil {
		t.Fatalf("f32 ForwardSubCapture: %v", err)
	}

	// Norm progression through L0: embedding -> +attn -> +mlp (= residual into L1).
	report := func(tag string, emb, attn0, mlp0 []float32) {
		afterAttn := add(emb, attn0)
		intoL1 := add(afterAttn, mlp0)
		t.Logf("%s  |emb|=%.1f  ->(+attn)  |=%.1f  ->(+mlp)  |=%.1f  (into L1)", tag, l2(emb), l2(afterAttn), l2(intoL1))
		t.Logf("     attn contribution |=%.1f  mlp contribution |=%.1f", l2(attn0), l2(mlp0))
		t.Logf("     ch %d:  emb=%.2f  attn=%.2f  mlp=%.2f  intoL1=%.2f",
			massiveCh, emb[massiveCh], attn0[massiveCh], mlp0[massiveCh], intoL1[massiveCh])
	}
	t.Logf("=== BOS (id %d) massive-activation build at L0 ===", bos)
	report("f32 ", fEmb, fAttn[0], fMLP[0])
	report("cuda", cudaEmb, cAttn[0], cMLP[0])

	// The compute-vs-norm split: the down output BEFORE the post-MLP sandwich norm. The norm
	// divides by this vector's near-zero RMS, so its DIRECTION at ch443 is what the norm amplifies
	// into the massive activation. Metal's is 0.4 (whole |down|=7.1); if CUDA's is ~5.9 the bug is
	// the MLP compute/int8-crush under-weighting ch443; if CUDA's is also ~0.4 the norm reopens.
	t.Logf("--- pre-post-MLP-norm DOWN output at BOS L0 (the norm's input direction) ---")
	t.Logf("f32   down: |=%.3f  ch%d=%.4f", l2(fMLPpre[0]), massiveCh, fMLPpre[0][massiveCh])
	t.Logf("cuda  down: |=%.3f  ch%d=%.4f", l2(cMLPpre[0]), massiveCh, cMLPpre[0][massiveCh])
}
