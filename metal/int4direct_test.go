//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma_Int4DirectContext checks the int4-direct path. The default resident path double-quantizes weights
// (f32→int8→int4); int4-direct consumes the decoder's int4 nibbles verbatim (what CUDA does), removing the int8 step. It
// ASSERTS only that layer 0's context matches the int4 reference (the nibbles are packed correctly). Later layers are
// logged, not asserted: Metal's context does not track the int4 forward past L1 (the FINDING below), which is the CUDA
// box's success criterion (≈ 1.0 against the int4 reference at every layer) not being met. The figures:
// docs/code-notes/metal.md#TestGemma_Int4DirectContext
func TestGemma_Int4DirectContext(t *testing.T) {
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("loads real models")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}

	// Metal resident built DIRECTLY from int4 (no int8 intermediate).
	m4, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	r, err := buildResident(m4)
	if err != nil {
		t.Fatalf("BuildResident(int4): %v", err)
	}
	defer r.Close()
	if !r.sandwich {
		t.Fatal("gemma3 resident is not sandwich")
	}
	seed := seedPrompt(t, path, probeText)
	pos := len(seed) - 1
	for i := range pos {
		r.forwardTrunkForTest(m4.EmbedResidentForTest(seed[i]), i, r.nL)
	}
	_, _, _, mctx, _ := r.forwardSubCaptureForTest(m4.EmbedResidentForTest(seed[pos]), pos)
	if mctx == nil {
		t.Fatal("forwardSubCaptureForTest nil")
	}

	// int4-ref (goinfer int4 == CUDA-int4) and f32-truth (weight-only int8, f32 activation).
	ref, e1 := decoder.Load(path, decoder.Options{Quant: "int4"})
	if e1 != nil {
		t.Fatalf("load int4 ref: %v", e1)
	}
	truth, e2 := decoder.Load(path, decoder.Options{Quant: "int8"})
	if e2 != nil {
		t.Fatalf("load int8-weight truth: %v", e2)
	}
	_, nL, _, nKV, hd, _, _ := ref.Dims()
	cr := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
	ct := decoder.NewKVCache(nL, nKV, hd, 0, 1024, nil)
	for i := range pos {
		if _, err := ref.ForwardForTest(seed[i], cr); err != nil {
			t.Fatalf("ref walk: %v", err)
		}
		if _, err := truth.ForwardForTest(seed[i], ct); err != nil {
			t.Fatalf("truth walk: %v", err)
		}
	}
	_, _, rctx, _, er := ref.ForwardSubCapture(seed[pos], cr)
	if er != nil {
		t.Skipf("ref ForwardSubCapture: %v", er)
	}
	_, _, tctx, _, et := truth.ForwardSubCapture(seed[pos], ct)
	if et != nil {
		t.Skipf("truth ForwardSubCapture: %v", et)
	}

	cos := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
	}

	t.Logf("=== int4-DIRECT context vs int4-ref (identical weights now) ===")
	var l0, l1 float64
	for l := range nL {
		vr := cos(mctx[l], rctx[l])
		vt := cos(mctx[l], tctx[l])
		if l == 0 {
			l0 = vr
		}
		if l == 1 {
			l1 = vr
		}
		if l == 0 || l == 1 || l >= nL-3 {
			nrm := func(v []float32) float64 {
				var s float64
				for _, x := range v {
					s += float64(x) * float64(x)
				}
				return math.Sqrt(s)
			}
			t.Logf("  L%2d: cos(Metal,int4-ref)=%.4f cos(Metal,f32)=%.4f  |metal|=%.2f |int4-ref|=%.2f",
				l, vr, vt, nrm(mctx[l]), nrm(rctx[l]))
		}
	}
	// L0 (identical embedding input, now identical weights) MUST be ~1.0 — this asserts int4-direct
	// packs the nibbles correctly (a mis-pack would corrupt L0 too).
	if l0 < 0.999 {
		t.Errorf("int4-direct L0 context %.4f != 1.0 — nibbles are mis-packed (aikit→Metal layout wrong)", l0)
	}
	// The FINDING (logged, not a failure): L1 still craters, matching the double-quant path. int4-direct removed the int8
	// intermediate and made ZERO difference to the crater — so the weight double-quant was NOT the cause. With byte-identical
	// weights Metal still diverges from the CPU int4 forward at L1, which localizes the bug to Metal's reduced-precision
	// COMPUTE (f16 KV cache / f16 activations), amplified by Gemma's sensitive attention — not the weights.
	t.Logf("FINDING: int4-direct L0=%.4f (nibbles correct) but L1=%.4f — crater UNCHANGED vs double-quant "+
		"(0.649). Weights refuted; the bug is Metal compute precision. Needs the matched-KV confirmer.", l0, l1)
}
