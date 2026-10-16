//go:build darwin && goinfer_testhooks

package metal

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma1ResidentParityMetal: Gemma 1 / CodeGemma (model_type gemma) is admitted to Metal by feature (embed scale,
// GeGLU, (1+w) RMSNorm, all of which gemma3 exercises there), but its own combination — pre-norm, MQA, head_dim 256 on
// the releases (32 here, != hidden/heads) — is not one gemma3 covers. This runs testdata/gemma1-tiny through the resident
// decode token by token and compares every position's logits with the CPU's. Both sides are int4: Metal has no int8 GEMV
// (int8int8 runs int4 there), so a CPU-int8 side would compare two quantizations (see
// prompthidden_resident_parity_test.go for the measured size of that).
func TestGemma1ResidentParityMetal(t *testing.T) {
	const ckpt = "../testdata/gemma1-tiny"
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("gemma1-tiny did not go resident — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()
	_, _, _, _, _, _, vocab := mCPU.Dims()
	cache := mCPU.NewCache(16)
	rf.Reset()
	worst := 1.0
	for i := range 16 {
		tok := (i*37 + 11) % vocab
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lc, lr)
		worst = min(worst, cos)
	}
	t.Logf("gemma1-tiny resident (%s) vs CPU int4, 16 positions: worst logit cosine %.6f", mRes.DecodePath(), worst)
	if worst < 0.999 {
		t.Errorf("worst logit cosine %.6f < 0.999", worst)
	}
}

// TestGemma2ResidentParityMetal: Gemma 2 needs the attention-score softcap, which Metal compiles into its kernels for a
// model that has one. Both tiny fixtures (cap 5, window 4, head dimension 32) run through the resident decode token by
// token against the CPU at int4, 16 positions.
//
// A diagnostic, not a gate: it logs the worst logit cosine and asserts nothing about it. It was G2-T
// (docs/tasks/task-forward-architecture-2026-10.md §4.6), whose 0.999 bar the fixtures' own no-softcap baseline sat at; by
// owner decision OP1-M1 the tiny fixtures are diagnostics, and the softcap's evidence is the kernel gates G2-K and G2-D. The
// lanes and the batched prefill are TestGemma2Softcap_lanesAndPrefillMetal's.
func TestGemma2ResidentParityMetal(t *testing.T) {
	for _, ckpt := range []string{"../testdata/gemma2-tiny", "../testdata/gemma2-hd-tiny"} {
		t.Run(filepath.Base(ckpt), func(t *testing.T) {
			mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			defer mRes.Close()
			rf := mRes.ResidentForwardForTest()
			if rf == nil {
				t.Fatalf("did not go resident: decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
			}
			if mRes.AttnLogitSoftcapResident() == 0 {
				t.Fatal("the fixture has no attention softcap: this test no longer gates what it claims")
			}
			mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load cpu: %v", err)
			}
			defer mCPU.Close()
			_, _, _, _, _, _, vocab := mCPU.Dims()
			cache := mCPU.NewCache(16)
			rf.Reset()
			worst := 1.0
			for i := range 16 {
				tok := (i*37 + 11) % vocab
				lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
				if err != nil {
					t.Fatalf("resident forward[%d]: %v", i, err)
				}
				lc, err := mCPU.ForwardForTest(tok, cache)
				if err != nil {
					t.Fatalf("cpu forward[%d]: %v", i, err)
				}
				cos, _ := cosF32(lc, lr)
				worst = min(worst, cos)
			}
			t.Logf("resident (%s) vs CPU int4, 16 positions: worst logit cosine %.6f (diagnostic)", mRes.DecodePath(), worst)

		})
	}
}
