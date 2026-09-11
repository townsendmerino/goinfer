//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// relL2 is ||got-want|| / ||want||. It sits beside the cosine in these gates because a cosine is
// blind to a uniform scale, and a missing logit scale is exactly a uniform scale.
func relL2(got, want []float32) float64 {
	var d, n float64
	for i := range want {
		e := float64(got[i]) - float64(want[i])
		d += e * e
		n += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(d / (n + 1e-30))
}

// TestCohereResidentParityCUDA is the resident-vs-CPU numeric gate that the Cohere smoke test
// (admission + no NaN) never was, for audit-2026-09-10 C-04. The smoke test could not see either
// half of C-04:
//   - every decode token's FINAL norm ran RMSNorm where Cohere's is a mean-centred LayerNorm;
//   - the default-on batched prefill ran RMSNorm at every norm site and re-normalised with a
//     post-attention norm that parallelBlock never allocates.
//
// The CPU side is the same quant, so the comparison isolates the forward, not the weight quantizer.
// Two measures are held, with bars fixed before the fix was written: cosine >= 0.995, and relative
// L2 <= 0.10. A batched-prefill DECLINE passes: sending Cohere to the sequential path is the fix's
// contract until the batched glue carries the LayerNorm and the parallel-block reuse.
func TestCohereResidentParityCUDA(t *testing.T) {
	for _, ckpt := range []string{"../testdata/cohere-tiny", "../testdata/cohere2-tiny"} {
		t.Run(filepath.Base(ckpt), func(t *testing.T) { cohereResidentParityCUDA(t, ckpt) })
	}
}

func cohereResidentParityCUDA(t *testing.T, ckpt string) {
	requireDeviceAndFixture(t, ckpt)
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("%s did not go CUDA-resident (%T) — decline: %s", ckpt, rf, mRes.ResidentDecline())
	}
	if !cr.layerNorm || !cr.parallelBlock {
		t.Fatalf("fixture resident has layerNorm=%v parallelBlock=%v — this gate only means something "+
			"when both are set", cr.layerNorm, cr.parallelBlock)
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	_, _, _, _, _, _, vocab := mCPU.Dims()
	prompt := make([]int, 12) // >= 8: the batched prefill's own threshold
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
	}

	// Decode: every position through the sequential resident path.
	rf.Reset()
	cache := mCPU.NewCache(len(prompt))
	worstCos, worstRel := 1.0, 0.0
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lr, lc)
		worstCos = math.Min(worstCos, cos)
		worstRel = math.Max(worstRel, relL2(lr, lc))
	}
	t.Logf("decode, %d positions: worst cosine %.6f, worst relL2 %.4f", len(prompt), worstCos, worstRel)
	if worstCos < 0.995 || worstRel > 0.10 {
		t.Errorf("decode diverges from the CPU forward: worst cosine %.6f (want >= 0.995), worst relL2 %.4f "+
			"(want <= 0.10) — the final norm is not the family's LayerNorm (audit C-04)", worstCos, worstRel)
	}

	// Prefill: the batched path, or its decline.
	rf.Reset()
	embs := make([][]float32, len(prompt))
	for i, tok := range prompt {
		embs[i] = mRes.EmbedResidentForTest(tok)
	}
	got, err := cr.PrefillLast(context.Background(), embs, 0)
	switch {
	case errors.Is(err, errPrefillDeclined):
		t.Logf("batched prefill declined, so the sequential path serves the prompt: %v", err)
	case err != nil:
		t.Fatalf("PrefillLast: %v", err)
	default:
		want, err := mCPU.PrefillLogitsForTest(context.Background(), prompt, mCPU.NewCache(len(prompt)))
		if err != nil {
			t.Fatalf("cpu prefill: %v", err)
		}
		cos, _ := cosF32(got, want)
		rel := relL2(got, want)
		t.Logf("batched prefill last-token logits vs CPU: cosine %.6f relL2 %.4f", cos, rel)
		if cos < 0.995 || rel > 0.10 {
			t.Errorf("batched prefill diverges from the CPU prefill: cosine %.6f relL2 %.4f — it runs RMSNorm "+
				"and a missing post-attention norm on a LayerNorm/parallel-block family (audit C-04)", cos, rel)
		}
	}
}

// TestGraniteDenseResidentPrefillLogitScaleCUDA is C-04's third half, which is not Cohere's alone:
// the batched prefill tail applies the final softcap but never the family's logit scale, while
// decode (step) applies both. Dense Granite is CUDA-resident with logits_scaling, and it takes the
// batched path, so its seed-token logits came back unscaled. That is invisible to argmax and wrong
// for every sampled first token. The bars are cosine >= 0.999 and relL2 <= 0.02 between batched and
// sequential resident logits over the same prompt: the same weights and the same quantization, so
// only the tail differs.
func TestGraniteDenseResidentPrefillLogitScaleCUDA(t *testing.T) {
	const ckpt = "../testdata/granite-dense-tiny"
	requireDeviceAndFixture(t, ckpt)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("%s did not go CUDA-resident (%T) — decline: %s", ckpt, rf, m.ResidentDecline())
	}
	if cr.logitScale == 0 || cr.logitScale == 1 {
		t.Fatalf("resident logitScale = %v — this gate only means something with a real logit scale", cr.logitScale)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := make([]int, 12)
	embs := make([][]float32, len(prompt))
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
		embs[i] = m.EmbedResidentForTest(prompt[i])
	}
	rf.Reset()
	var seq []float32
	for i, e := range embs {
		if seq, err = rf.Forward(e, i); err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
	}
	seq = append([]float32(nil), seq...)
	rf.Reset()
	got, err := cr.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v — this gate needs granite-dense on the batched path (the premise of "+
			"C-04's third half); if it now declines, the tail is unreachable for it", err)
	}
	cos, _ := cosF32(got, seq)
	rel := relL2(got, seq)
	t.Logf("granite-dense (logitScale %.4f): batched vs sequential last-token logits: cosine %.6f relL2 %.4f",
		cr.logitScale, cos, rel)
	if cos < 0.999 || rel > 0.02 {
		t.Errorf("batched prefill logits differ from sequential decode: cosine %.6f relL2 %.4f — the "+
			"prefill tail drops the logit scale (audit C-04)", cos, rel)
	}
}
