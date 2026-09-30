//go:build darwin && goinfer_testhooks

package metal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma1ResidentParityMetal: Gemma 1 / CodeGemma (model_type gemma) is admitted to Metal by feature (embed scale,
// GeGLU, (1+w) RMSNorm, all of which gemma3 exercises there), but its own combination — pre-norm, MQA, head_dim 256 on
// the releases (32 here, != hidden/heads) — had not run resident. This runs testdata/gemma1-tiny through the resident
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

// TestGemma2DeclinesResidentMetal: Gemma 2 needs the attention-score softcap, which no resident backend implements,
// so a Metal load must decline to the CPU and say why, rather than go resident and silently drop the cap.
func TestGemma2DeclinesResidentMetal(t *testing.T) {
	m, err := decoder.Load("../testdata/gemma2-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() != nil {
		t.Fatalf("gemma2-tiny went resident on Metal (%s), which has no attention-softcap kernel", m.DecodePath())
	}
	if d := m.ResidentDecline(); !strings.Contains(d, string(decoder.FeatAttnLogitSoftcap)) {
		t.Fatalf("the decline does not name the attention softcap: %q", d)
	}
	t.Logf("decline: %s", m.ResidentDecline())
}
