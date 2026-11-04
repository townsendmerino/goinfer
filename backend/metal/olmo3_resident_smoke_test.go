//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/program"
)

// TestOlmo3ResidentSmokeMetal is the smoke gate for FeatPostOnlyNorm + FeatQKNormWhole (docs/tasks/task-gpu-paths-2026-09.md,
// G5): the model goes resident, no logit of a 24-token decode or of the batched prefill is NaN, and the prefill is batched.
//
// Deliberately NOT a resident-vs-CPU cosine floor: the seeded synthetic fixture (testdata/olmo3-tiny)
// cannot separate a real bug from quantization noise. The correctness gate is one level lower: TestQKNorm_wholeVector
// (qknorm_whole_test.go) compares the whole-vector qk_norm dispatch geometry against the production kernel with an exact
// per-component comparison. The post-only pre-norm skip (quant_vec instead of rmsnorm_quant) reuses a kernel ctx-before-o-proj
// already dispatches, so it rests on that coverage plus the admission and no-NaN checks here.
func TestOlmo3ResidentSmokeMetal(t *testing.T) {
	testOlmoFamilyResidentSmoke(t, "../../testdata/olmo3-tiny", "int8int8")
}

// TestOlmoHybridResidentParityMetal is OM-1 (docs/tasks/task-olmo-hybrid-metal-webgpu-2026-11.md): Olmo Hybrid's DeltaNet
// widens its write gate (NegEigval) and runs its output norm at an epsilon of its own (ONormEps). olmo_hybrid-tiny goes
// resident on Metal at int4, the one quant its build takes (TestOlmoHybridBuildDeclinesOtherQuants), and holds
// TestQwen35ResidentParityMetal's bar against the CPU at int4. With either setting dropped at the build, it falls outside the bar.
func TestOlmoHybridResidentParityMetal(t *testing.T) {
	const fx = "../../testdata/olmo_hybrid-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no olmo_hybrid-tiny weights: %v", err)
	}
	t.Run("prompts take the sequential loop", func(t *testing.T) { // OM-2: so the as-built run covers them
		m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
		if m.ResidentForwardForTest() == nil {
			t.Fatalf("not resident: %s", m.ResidentDecline())
		}
		if batched, why := m.PrefillPath(); batched {
			t.Errorf("olmo_hybrid-tiny's prefill is batched on Metal; OM-1 no longer covers its prompts (%s)", why)
		}
	})
	t.Run("as built", func(t *testing.T) { qwen35ResidentParityMetal(t, fx, "int4") })
	for _, plant := range []struct {
		name string
		on   *bool
	}{{"NegEigval dropped", &dnetNegEigvalOffForTest}, {"ONormEps dropped", &dnetONormEpsOffForTest}} {
		t.Run(plant.name, func(t *testing.T) {
			*plant.on = true
			defer func() { *plant.on = false }()
			p := deltaNetParityMetal(t, fx, "int4")
			t.Logf("planted (%s): worst %.6f, drift %.4f, replay %.9f", plant.name, p.worst, p.first-p.worst, p.replay)
			if !p.failsBar() {
				t.Errorf("with %s, olmo_hybrid-tiny still holds the bar: the gate does not see the setting", plant.name)
			}
		})
	}
}

// TestOlmoHybridBuildDeclinesOtherQuants: admission passes olmo_hybrid-tiny at f32 and int8int8 on Metal, as it does every Metal
// family at f32 and every Gated-DeltaNet family at int8int8 (the census's rows record admission, not the build), and the build
// refuses both, so the load runs on the CPU and no resident path exists at either quant for OM-1 to hold.
func TestOlmoHybridBuildDeclinesOtherQuants(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct{ quant, why string }{{"f32", "weights loaded at f32"}, {"int8int8", "int4 only"}} {
		m, err := decoder.Load("../../testdata/olmo_hybrid-tiny", decoder.Options{Backend: "metal", Quant: c.quant})
		if err != nil {
			t.Fatalf("%s: %v", c.quant, err)
		}
		if m.ResidentForwardForTest() != nil {
			t.Errorf("%s: went resident on Metal (%s)", c.quant, m.DecodePath())
		}
		if !strings.Contains(m.DecodePath(), "running on cpu") || !strings.Contains(m.DecodePath(), c.why) {
			t.Errorf("%s: decode path %q does not run on the CPU naming %q", c.quant, m.DecodePath(), c.why)
		}
		t.Logf("%s: %s", c.quant, m.DecodePath())
		m.Close()
	}
}

// TestDeltaNetOpParams_layersMustAgree: the block holds one set of DeltaNet parameters, so a model whose DeltaNet layers disagree
// on one declines at the build, naming the layer. qwen35-tiny's program is rewritten on one DeltaNet layer, one parameter at a time.
func TestDeltaNetOpParams_layersMustAgree(t *testing.T) {
	m, err := decoder.Load("../../testdata/qwen35-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, err := deltaNetOpParams(m); err != nil {
		t.Fatalf("the unmodified model's layers disagree: %v", err)
	}
	last := func() *program.Op {
		var op *program.Op
		for _, ops := range m.Program().Layers {
			for i := range ops {
				if ops[i].Kind == program.OpDeltaNet {
					op = &ops[i]
				}
			}
		}
		return op
	}
	op := last()
	if op == nil {
		t.Fatal("no DeltaNet layer")
	}
	saved := op.Delta
	defer func() { op.Delta = saved }()
	for _, c := range []struct {
		name   string
		change func(p *program.DeltaNetOpParams)
	}{
		{"NegEigval", func(p *program.DeltaNetOpParams) { p.NegEigval = !p.NegEigval }},
		{"ONormEps", func(p *program.DeltaNetOpParams) { p.ONormEps = 1e-3 }},
	} {
		op.Delta = saved
		c.change(&op.Delta)
		if _, err := deltaNetOpParams(m); err == nil || !strings.Contains(err.Error(), "differ") {
			t.Errorf("%s disagreeing: %v, want the decline", c.name, err)
		}
		if r, err := buildResident(m); err == nil {
			_ = r.Close()
			t.Errorf("%s disagreeing: the build went through", c.name)
		} else if !strings.Contains(err.Error(), "differ") {
			t.Errorf("%s disagreeing: the build declined for another reason: %v", c.name, err)
		}
	}
}

// TestDeltaNetPrefill_readsBothParameters: the batched prefill's mixer (encodePrefillDeltaMixer) binds the write gate's scale and
// the output norm's epsilon. Olmo Hybrid's prefill is not batched, so the check sets them on qwen35-tiny's resident: each changes
// the prefill's last logits from the as-built run, and from the other.
func TestDeltaNetPrefill_readsBothParameters(t *testing.T) {
	m, err := decoder.Load("../../testdata/qwen35-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	if batched, why := m.PrefillPath(); !batched {
		t.Fatalf("qwen35-tiny's prefill is not batched: %s", why)
	}
	r := rf.(*metalResident).r
	pf := rf.(decoder.Prefiller)
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, 40)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*37 + 11) % vocab)
	}
	run := func() []float32 {
		rf.Reset()
		lg, err := pf.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	differ := func(a, b []float32) int {
		n := 0
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				n++
			}
		}
		return n
	}
	base := run()
	if d := differ(base, run()); d != 0 {
		t.Fatalf("two as-built prefills differ in %d logits", d)
	}
	beta, eps := r.uDnBetaScale, r.uDnONormEps
	defer func() { r.uDnBetaScale, r.uDnONormEps = beta, eps }()
	r.uDnBetaScale = NewBufferFloats(r.d, []float32{2})
	betaRun := run()
	r.uDnBetaScale = beta
	r.uDnONormEps = NewBufferFloats(r.d, []float32{1})
	epsRun := run()
	t.Logf("logits differing from the as-built prefill of %d: betaScale 2 %d, a distinct epsilon %d", len(base), differ(base, betaRun), differ(base, epsRun))
	if differ(base, betaRun) == 0 {
		t.Error("with betaScale 2 the prefill's logits are unchanged: its gates do not read the scale")
	}
	if differ(base, epsRun) == 0 {
		t.Error("with a distinct epsilon the prefill's logits are unchanged: its gated norm does not read it")
	}
	if differ(betaRun, epsRun) == 0 {
		t.Error("the two settings give the same logits")
	}
}

func testOlmoFamilyResidentSmoke(t *testing.T, ckpt, quant string) {
	t.Helper()
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: quant})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	feats := mRes.RequiredResidentFeatures()
	hasPostOnly, hasQKWhole := false, false
	for _, f := range feats {
		hasPostOnly = hasPostOnly || f == decoder.FeatPostOnlyNorm
		hasQKWhole = hasQKWhole || f == decoder.FeatQKNormWhole
	}
	if !hasPostOnly || !hasQKWhole {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises both "+
			"FeatPostOnlyNorm and FeatQKNormWhole; if the fixture changed, this test no longer "+
			"gates what it claims", feats)
	}
	t.Logf("required feats: %v, missing from prefill: %v", feats, mRes.MissingResidentFeatures(prefillFeatures))

	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			ckpt, mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	const ntok = 24
	_, _, _, _, _, _, vocab := mRes.Dims()
	rf.Reset()
	for i := range ntok {
		tok := (i*37 + 3) % vocab
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		for j, v := range lr {
			if v != v { // NaN check without importing math
				t.Fatalf("resident forward[%d]: logit[%d] is NaN", i, j)
			}
		}
	}

	// Prefill check: Olmo 3 admits batched prefill; Olmo Hybrid declines due to DeltaNet recurrence.
	mr, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("rf is not *metalResident")
	}
	batched, reason := mRes.PrefillPath()
	if mr.r.dnet != nil {
		if batched {
			t.Fatalf("Olmo Hybrid has recurrent DeltaNet state and must decline batched prefill, got %s", reason)
		}
		t.Logf("Olmo Hybrid properly declined batched prefill: %s", reason)
	} else {
		if !batched {
			t.Fatalf("Olmo 3 reported sequential prefill path (%q); expected batched prefill", reason)
		}
		t.Logf("Olmo 3 prefill path: %s", reason)

		decoder.SetKnobEnvForTest(t, mRes, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
		pf, ok := rf.(decoder.Prefiller)
		if !ok {
			t.Fatalf("rf does not implement decoder.Prefiller")
		}
		rf.Reset()
		embs := make([][]float32, ntok)
		for i := range ntok {
			tok := (i*37 + 3) % vocab
			embs[i] = mRes.EmbedResidentForTest(tok)
		}
		preLogits, err := pf.PrefillLast(t.Context(), embs, 0)
		if err != nil {
			t.Fatalf("Olmo 3 PrefillLast error: %v", err)
		}
		for j, v := range preLogits {
			if v != v {
				t.Fatalf("Olmo 3 PrefillLast: logit[%d] is NaN", j)
			}
		}
		t.Logf("Olmo 3 batched prefill succeeded with %d tokens", ntok)
	}
}
