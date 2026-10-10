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

// TestPrefillParityMoE is G8's second gate (docs/tasks/task-gpu-paths-2026-09.md): the batched f16 MMA
// prefill path's MoE FFN must match the sequential Forward loop's last-token logits. G8 first ran
// that FFN row by row off the batched residual, through the per-token decode MoE dispatch chain
// (encodeMoERoute/encodeMoEExperts/encodeMoESharedExpert); the default is now expert-major (all
// rows routed at once, then each active expert over its rows), and GOINFER_MOE_EXPERT_MAJOR=0
// restores the row loop (D-D01, audit-metal-2026-09-30.md).
// Same structure and bar as TestPrefillParity/TestPrefillParityGemma: argmax match, cosine >=
// 0.95 (prefill's f16 activations vs decode's int8 mean a high-but-not-exact cosine is expected).
//
// testdata/mixtral-tiny: the simplest MoE shape available (no shared expert, softmax routing, no
// group-limited routing, MHA) — deliberately chosen to isolate the row-loop mechanics from the
// gated-shared-expert/sigmoid-routing paths TestPrefillParityMoEGatedShared below covers
// separately.
func TestPrefillParityMoE(t *testing.T) {
	testPrefillParityMoEFixture(t, "../testdata/mixtral-tiny", false, 0.95)
}

// TestPrefillParityMoEGatedShared exercises the other branch encodeMoESharedExpert's parameterization must get right: a
// sigmoid-GATED always-on shared expert (FeatMoEGatedShared, Qwen2-MoE), not just the ungated GLM/DeepSeek shape the plain
// mixtral fixture above never touches (mixtral has no shared expert).
//
// The fixtures are the two committed gated-shared tiny checkpoints, both DeltaNet hybrids whose MoE layers take the same
// batched shared-expert dispatch (D-B01). testdata/tiny-qwen2-moe, a per-machine download, cannot serve: its
// moe_intermediate_size of 44 is rejected by the int4 group of 32, so BuildResident declines it. The cosine bar is 0.999, not
// the plain MoE test's 0.95: on these fixtures the gate is a small term, and dropping the sigmoid (gate fixed at 1) still
// clears 0.95 on qwen35-tiny while reading below 0.999 where unmutated runs read above it. At 0.999 that mutation fails both.
func TestPrefillParityMoEGatedShared(t *testing.T) {
	for _, ckpt := range []string{"../testdata/qwen35-tiny", "../testdata/qwen3next-tiny"} {
		t.Run(ckpt[len("../testdata/"):], func(t *testing.T) { testPrefillParityMoEFixture(t, ckpt, true, 0.999) })
	}
}

func testPrefillParityMoEFixture(t *testing.T, ckpt string, wantGatedShared bool, minCos float64) {
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
	if cos < minCos {
		t.Fatalf("moe prefill parity FAIL: cosine %.5f under %.3f (bug?)", cos, minCos)
	}
	t.Logf("moe prefill last-token argmax matches sequential ✓ (cosine %.4f)", cos)
}
