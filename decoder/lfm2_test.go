package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"slices"
	"strings"
	"testing"
)

// TestLFM2_arch pins the descriptor facts that the LFM2 forward depends on and that a
// config-key mistake would silently zero. The two explicit >0 assertions are not padding:
// both were real bugs, found 2026-08-31 by differencing against HF on the released
// LFM2.5-2.6B, and both produced a running, fluent, WRONG model rather than an error.
//
//   - NormEps came back 0 because LFM2 spells the key "norm_eps" and the adapter read
//     cfg.RMSNormEps. Uniform 1.0185x scale on the first norm; logits cosine 0.897.
//   - AttnScale came back 0 because the Architecture literal simply omitted it. Every q·k
//     score is then 0, so softmax returns a UNIFORM average over the context. Invisible at
//     one token (softmax of one element is 1.0 at any scale) — it needs >= 2 tokens to show.
//
// Both had a MATCHING argmax against HF while broken, so neither a smoke test nor a greedy
// decode would have caught them.
func TestLFM2_arch(t *testing.T) {
	arch, schema, err := resolveArchitecture(representativeConfig("lfm2"))
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.NormEps <= 0 {
		t.Errorf("NormEps = %v, want >0 (read from norm_eps, NOT rms_norm_eps)", arch.NormEps)
	}
	if arch.AttnScale <= 0 {
		t.Errorf("AttnScale = %v, want >0 (head_dim**-0.5)", arch.AttnScale)
	}
	if arch.lfm2 == nil {
		t.Fatal("arch.lfm2 is nil")
	}
	if arch.lfm2.ConvLCache != 3 || arch.lfm2.ConvDim != 64 {
		t.Errorf("conv geometry = {dim %d, K %d}, want {64, 3}", arch.lfm2.ConvDim, arch.lfm2.ConvLCache)
	}
	// layer_types is what selects the mixer per layer; a loader that ignored it would run
	// attention everywhere and still produce logits.
	for i, wantConv := range []bool{true, true, false, true} {
		if got := arch.isConvLayer(i); got != wantConv {
			t.Errorf("isConvLayer(%d) = %v, want %v", i, got, wantConv)
		}
	}
	if !arch.QKNorm || !arch.TiedLMHead {
		t.Errorf("QKNorm=%v TiedLMHead=%v, want both true", arch.QKNorm, arch.TiedLMHead)
	}
	if schema.FinalNorm != "model.embedding_norm.weight" {
		t.Errorf("FinalNorm = %q, want model.embedding_norm.weight", schema.FinalNorm)
	}
}

// TestResolveArchitecture_guardFiresRed proves validateResolved can go RED. A guard that
// never fires is indistinguishable from one that does not work, and this one exists only to
// catch fields an adapter forgot — so the omission it catches must be demonstrated, not
// assumed. Mirrors cmd/gate's mutation discipline.
func TestResolveArchitecture_guardFiresRed(t *testing.T) {
	t.Run("missing_norm_eps_family_validator", func(t *testing.T) {
		// For lfm2 the FAMILY validator catches this first and never reaches
		// validateResolved. Both layers are asserted rather than one, because which one
		// fires is an implementation detail and the requirement is that SOMETHING does.
		cfg := representativeConfig("lfm2")
		cfg.NormEps = 0 // exactly what reading rms_norm_eps for this family produces
		_, _, err := resolveArchitecture(cfg)
		if err == nil {
			t.Fatal("resolveArchitecture accepted NormEps=0 — the guard did not fire")
		}
		if !strings.Contains(err.Error(), "norm_eps") {
			t.Errorf("error %q does not name norm_eps", err)
		}
	})
	t.Run("missing_norm_eps_resolved_guard", func(t *testing.T) {
		// validateResolved's own branch, reached directly — this is the one that covers
		// families with no eps check of their own, which is the whole point of it existing.
		a := &Architecture{Name: "probe", Norm: NormRMS, AttnScale: 0.25} // NormEps omitted
		err := a.validateResolved()
		if err == nil {
			t.Fatal("validateResolved accepted NormEps=0 — the guard did not fire")
		}
		if !strings.Contains(err.Error(), "NormEps") {
			t.Errorf("error %q does not name NormEps", err)
		}
	})
	t.Run("missing_attn_scale", func(t *testing.T) {
		a := &Architecture{Name: "probe", Norm: NormRMS, NormEps: 1e-5} // AttnScale omitted
		err := a.validateResolved()
		if err == nil {
			t.Fatal("validateResolved accepted AttnScale=0 — the guard did not fire")
		}
		if !strings.Contains(err.Error(), "AttnScale") {
			t.Errorf("error %q does not name AttnScale", err)
		}
	})
	t.Run("well_formed_passes", func(t *testing.T) {
		// The other half of a red-capable gate: it must still go green on a good descriptor,
		// or "it fires" would just mean "it always fires".
		if _, _, err := resolveArchitecture(representativeConfig("lfm2")); err != nil {
			t.Fatalf("well-formed lfm2 config rejected: %v", err)
		}
	})
}

// TestLFM2_declinesResidentRunner pins BOTH independent reasons LFM2 stays on the CPU. They
// answer different questions and a future change could remove either one alone:
//
//   - decodeRunnerEligible is the ARCH-SHAPE gate: 22 of 30 layers are a gated short conv with
//     a rolling window that no uniform-layer runner can express, regardless of backend.
//   - FeatShortConv is the BACKEND-CAPABILITY gate: nobody implements it, so all decline.
//
// The arch-shape half also feeds the capability matrix's GPU-resident column directly. Before
// this was pinned, LFM2 fell through every switch arm in decodeRunnerEligible and the generated
// matrix published "GPU-resident: yes" for a family no backend can run.
func TestLFM2_declinesResidentRunner(t *testing.T) {
	arch, _, err := resolveArchitecture(representativeConfig("lfm2"))
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.decodeRunnerEligible() {
		t.Error("decodeRunnerEligible() = true — LFM2 has its own forward (runLayersLFM2); " +
			"the resident runner would treat its conv layers as attention")
	}
	feats := arch.residentFeatures()
	if !slices.Contains(feats, FeatShortConv) {
		t.Errorf("residentFeatures() = %v, missing FeatShortConv — without it LFM2's profile is "+
			"{FeatQKNorm}, which every backend implements, so all three would admit it", feats)
	}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if missing := missingFeatures(feats, residentBackendFeatures[be]); len(missing) == 0 {
			t.Errorf("%s reports no missing features for LFM2 — it must decline", be)
		}
	}
}

// TestLFM2_textParity gates the whole LFM2 forward — both mixer kinds, QK-norm, GQA, SwiGLU,
// the tied head — against an HF golden. Fixture: scripts/pin_lfm2_tiny.py.
//
// The LAST-LOGIT COSINE is the load-bearing assertion here, not the continuation: this
// tiny-random model emits a constant continuation (all 88s), which a broken forward could
// plausibly reproduce. The 256-wide logit vector is what actually discriminates. The
// continuation is kept as a cheap regression tripwire, not as the proof.
func TestLFM2_textParity(t *testing.T) {
	const golden = "../testdata/lfm2_tiny_text_golden.json"
	const ckpt = "../testdata/lfm2-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_lfm2_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_lfm2_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("lfm2 text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d (got %v want %v)", i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
			break
		}
	}

	// tiny-golden is SUB-T3, and the merge derives status from the method, so this lands as
	// "experimental" rather than "validated" — which is the honest tier for a family whose
	// only numeric gate is a seeded 4-layer fixture. The released LFM2.5-2.6B was differenced
	// against HF by hand during bring-up (bit-exact after the two fixes), but that ran off a
	// 5 GB local checkpoint with no committed gate, so it is NOT claimed here.
	emitParityRow(t, "lfm2", "tiny-golden", "HF f32 (lfm2-tiny seeded fixture)",
		100.0, float64(cos), float64(cos))
}
