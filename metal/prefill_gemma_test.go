//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// gemmaPrefillPrompt mirrors TestPrefillParity's own shape: an arbitrary fixed token sequence
// long enough to exercise the batched f16 MMA path meaningfully, not tied to the fixture's real
// tokenizer.
var gemmaPrefillPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88, 21, 64, 9, 150}

// TestPrefillParityGemma is the G8 gate (docs/task-gpu-paths-2026-09.md): the batched f16 MMA
// prefill path, extended this row to admit Gemma's sandwich norms / (1+w) RMS offset / GeGLU
// (prefillFeatures, metal/model.go), must match the sequential Forward loop's last-token logits
// on a REAL Gemma checkpoint — same structure as TestPrefillParity, same bar (argmax match,
// cosine >= 0.95: prefill's f16 activations vs decode's int8 mean a high-but-not-exact cosine is
// expected, not a bug).
//
// testdata/gemma3-vl-tiny (not a downloaded heavy model): a real small Gemma3 VL checkpoint,
// already used by gpu/gemma3_resident_parity_test.go (G6) for the identical reason — only its
// text tower is exercised here (a plain int token prompt never touches the vision encoder).
func TestPrefillParityGemma(t *testing.T) {
	const ckpt = "../testdata/gemma3-vl-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture (%s)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if !r.sandwich {
		t.Fatal("fixture is not a sandwich-norm family — this test no longer gates what it claims")
	}
	if !r.prefillOK {
		t.Fatal("prefillOK is false for this fixture — the Gemma admission (prefillFeatures) " +
			"did not take, or the per-layer-geometry guard incorrectly declined a uniform-geometry family")
	}

	prompt := gemmaPrefillPrompt
	M := len(prompt)

	// sequential path (correct reference): Forward each token; keep the last logits.
	var seq []float32
	for i, id := range prompt {
		seq = r.Forward(id, i)
	}
	seqArg := argmaxF(seq)

	// prefill path: embeddings for the prompt, one PrefillLast call at the SAME startPos=0 the
	// sequential loop just used — PrefillLast's own KV writes overwrite whatever the sequential
	// run left at those exact positions (same pattern TestPrefillParity's own twin uses; no
	// explicit reset exists at this level, only metalResident's DeltaNet-specific one).
	embs := make([][]float32, M)
	for i, id := range prompt {
		e := make([]float32, r.H)
		r.embed.Row(id, e)
		// Match loadEmbedRow's own embed-scale step (r.Forward, used for the sequential
		// reference above, applies this internally) — PrefillLast takes precomputed embeddings
		// and, like production's embedResident caller, does not scale them itself.
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
	t.Logf("gemma prefill vs sequential last-token: argmax pre=%d seq=%d, cosine=%.5f", preArg, seqArg, cos)
	if math.IsNaN(float64(cos)) || math.IsInf(float64(cos), 0) {
		t.Fatalf("gemma prefill parity FAIL: cosine is %v — degenerate (NaN/Inf) logits", cos)
	}
	if preArg != seqArg {
		t.Fatalf("gemma prefill parity FAIL: last-token argmax %d != sequential %d (cosine %.4f)", preArg, seqArg, cos)
	}
	if cos < 0.95 {
		t.Fatalf("gemma prefill parity FAIL: cosine %.4f too low (bug?)", cos)
	}
	t.Logf("gemma prefill last-token argmax matches sequential ✓ (cosine %.4f)", cos)
}

// TestPrefillDeclinesGemma4PerLayerGeom pins the OTHER half of G8's admission: dense Gemma 4's
// local/global attention layers genuinely differ in head_dim (256 vs 512), which PrefillLast's
// uniform-g0 fast path (reads r.layers[0].geom once, reuses it for every layer) cannot represent
// — a family that shares Gemma 3's exact ResidentFeature set otherwise (see decoder/features.go's
// residentPerLayerGeomBackends comment: "Gemma 3 and dense Gemma 4 derive the IDENTICAL feature
// set"), so admitting FeatSandwichNorm/FeatGatedGELU/etc without this separate guard would have
// silently admitted Gemma 4 here too, right after the Gemma 3 admission this row's whole point
// was to land. testdata/gemma4-dense-twogeom-tiny is built specifically to have two distinct
// head_dim geometries (unlike every other tiny fixture, which is uniform by construction).
func TestPrefillDeclinesGemma4PerLayerGeom(t *testing.T) {
	const ckpt = "../testdata/gemma4-dense-twogeom-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture (%s)", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if r.prefillOK {
		t.Fatal("dense Gemma 4 (per-layer local/global head_dim) must decline batched prefill — " +
			"the uniform-g0 fast path cannot represent it, even though it shares Gemma 3's feature set")
	}
}
