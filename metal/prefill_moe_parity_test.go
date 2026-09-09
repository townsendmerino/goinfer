//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// moePrefillPrompt mirrors TestPrefillParity/TestPrefillParityGemma's own shape: an arbitrary
// fixed token sequence, not tied to the fixture's real tokenizer.
var moePrefillPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88, 21, 64, 9, 150}

// TestPrefillParityMoE is G8's second gate (docs/task-gpu-paths-2026-09.md): the batched f16 MMA
// prefill path, extended this row to run a MoE layer's FFN row by row off the batched residual
// (reusing the unchanged per-token decode MoE dispatch chain — encodeMoERoute/encodeMoEExperts/
// encodeMoESharedExpert — the same "batch attention, loop the FFN" shape CUDA's own batched
// prefill already uses for MoE), must match the sequential Forward loop's last-token logits.
// Same structure and bar as TestPrefillParity/TestPrefillParityGemma: argmax match, cosine >=
// 0.95 (prefill's f16 activations vs decode's int8 mean a high-but-not-exact cosine is expected).
//
// testdata/mixtral-tiny: the simplest MoE shape available (no shared expert, softmax routing, no
// group-limited routing, MHA) — deliberately chosen to isolate the row-loop mechanics from the
// gated-shared-expert/sigmoid-routing paths TestPrefillParityMoEGatedShared below covers
// separately.
func TestPrefillParityMoE(t *testing.T) {
	testPrefillParityMoEFixture(t, "../testdata/mixtral-tiny", false)
}

// TestPrefillParityMoEGatedShared exercises the OTHER branch encodeMoESharedExpert's
// parameterization (this row) must get right: a sigmoid-GATED always-on shared expert
// (FeatMoEGatedShared, Qwen2-MoE), not just the ungated GLM/DeepSeek shape the plain mixtral
// fixture above never touches at all (mixtral has no shared expert).
func TestPrefillParityMoEGatedShared(t *testing.T) {
	testPrefillParityMoEFixture(t, "../testdata/tiny-qwen2-moe", true)
}

func testPrefillParityMoEFixture(t *testing.T, ckpt string, wantGatedShared bool) {
	t.Helper()
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no fixture (%s/config.json)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	feats := m.RequiredResidentFeatures()
	hasMoE, hasGatedShared := false, false
	for _, f := range feats {
		hasMoE = hasMoE || f == decoder.FeatMoE
		hasGatedShared = hasGatedShared || f == decoder.FeatMoEGatedShared
	}
	if !hasMoE {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises FeatMoE; "+
			"if the fixture changed, this test no longer gates what it claims", feats)
	}
	if hasGatedShared != wantGatedShared {
		t.Fatalf("fixture FeatMoEGatedShared=%v, want %v", hasGatedShared, wantGatedShared)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if !r.prefillOK {
		t.Fatal("prefillOK is false for this fixture — the MoE admission (prefillFeatures) did not take")
	}

	prompt := moePrefillPrompt
	M := len(prompt)

	// sequential path (correct reference): Forward each token; keep the last logits.
	var seq []float32
	for i, id := range prompt {
		seq = r.Forward(id, i)
	}
	seqArg := argmaxF(seq)

	// prefill path: embeddings for the prompt, one PrefillLast call at the SAME startPos=0 the
	// sequential loop just used — PrefillLast's own KV writes overwrite whatever the sequential
	// run left at those exact positions (same pattern TestPrefillParity's own twin uses).
	embs := make([][]float32, M)
	for i, id := range prompt {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		if r.embedScale > 1 {
			for j := range e {
				e[j] *= r.embedScale
			}
		}
		embs[i] = e
	}
	pre := r.PrefillLast(embs, 0)
	preArg := argmaxF(pre)

	cos := cosF(seq, pre)
	t.Logf("moe prefill vs sequential last-token: argmax pre=%d seq=%d, cosine=%.5f", preArg, seqArg, cos)
	if math.IsNaN(float64(cos)) || math.IsInf(float64(cos), 0) {
		t.Fatalf("moe prefill parity FAIL: cosine is %v — degenerate (NaN/Inf) logits", cos)
	}
	if preArg != seqArg {
		t.Fatalf("moe prefill parity FAIL: last-token argmax %d != sequential %d (cosine %.4f)", preArg, seqArg, cos)
	}
	if cos < 0.95 {
		t.Fatalf("moe prefill parity FAIL: cosine %.4f too low (bug?)", cos)
	}
	t.Logf("moe prefill last-token argmax matches sequential ✓ (cosine %.4f)", cos)
}
