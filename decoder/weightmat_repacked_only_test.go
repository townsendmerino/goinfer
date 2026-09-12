package decoder

import (
	"math"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// fakeQuantBackend4 implements only QuantBackend4 (the staged per-token int4 consult) — a
// minimal stand-in for a webgpu-class backend, for wantsCanonicalInt4's routing test.
type fakeQuantBackend4 struct{ fakeMatmulBTBackend }

func (fakeQuantBackend4) MatmulW4A8(a []float32, q4 []byte, q4s []float32, group int, dst []float32, M, K, N int) bool {
	return false
}

// TestWantsCanonicalInt4_onlyExplicitCPUUnlocksRepackedOnly is the load-bearing routing
// decision behind quantizeEmbedWM/quantizeBatchedProjWM: repacked-only is a PROMISE ("no GPU
// backend will ever touch this *Model") that must be STATED via Options.Backend=="cpu"
// literally, never inferred from an empty/unspecified value — see wantsCanonicalInt4's own doc
// comment for why (metal's own test suite loads generically, backend unset, then hands the
// result to Metal's OWN resident-build machinery completely outside decoder.Load's dispatch;
// decoder.Load cannot see that coming, so "unspecified" cannot mean "definitely CPU-only").
func TestWantsCanonicalInt4_onlyExplicitCPUUnlocksRepackedOnly(t *testing.T) {
	plain := &fakeMatmulBTBackend{}
	cases := []struct {
		name        string
		backendName string
		be          Backend
		want        bool
	}{
		{"explicit cpu, plain backend", "cpu", plain, false},
		{"empty/unspecified, plain backend", "", plain, true},
		{"a real GPU name, plain backend (name/be mismatch is not this func's job to catch)", "metal", plain, true},
		{"explicit cpu, but be also implements QuantBackend4 (belt-and-braces)", "cpu", &fakeQuantBackend4{}, true},
		{"explicit cpu, but be also implements QuantBatchBackend4 (belt-and-braces)", "cpu", &fakeBatchBackend4{}, true},
		{"explicit cpu, but be also implements ResidencyBackend (belt-and-braces)", "cpu", &fakeResidencyBackend{Backend: plain}, true},
	}
	for _, c := range cases {
		if got := wantsCanonicalInt4(c.backendName, c.be); got != c.want {
			t.Errorf("%s: wantsCanonicalInt4(%q, ...) = %v, want %v", c.name, c.backendName, got, c.want)
		}
	}
}

// syntheticEmbedF32 builds a deterministic, non-trivial [rows,cols] f32 matrix — large enough
// (rows=8, cols=64) to satisfy Int4Row4Usable's rows%4==0/cols%group==0 shape gate on a core that
// supports it at all, small enough to keep the test fast.
func syntheticEmbedF32(rows, cols int) []float32 {
	f32 := make([]float32, rows*cols)
	for i := range f32 {
		// A smooth, non-constant, non-monotonic pattern so no group's dynamic range collapses
		// to zero (which would make every int4 code identical and hide a wiring bug).
		f32[i] = float32(math.Sin(float64(i)*0.37)) * float32((i%7)+1)
	}
	return f32
}

// TestQuantizeEmbedWM_repackedOnlyWhenEligible is M-22's adoption gate (aikit v1.41.0, cross-
// referenced from goinfer's audit-2026-09-10.md's own M-22 — an unrelated finding sharing the
// label by coincidence of two repos' independent numbering): needCanonical=false on a shape this
// core can build row4 for must produce a repacked-only WeightMat (IsInt4() true, Int4()'s own ok
// false — no canonical bytes resident at all); needCanonical=true must produce the existing
// canonical(+row4) "both" policy unchanged.
func TestQuantizeEmbedWM_repackedOnlyWhenEligible(t *testing.T) {
	const rows, cols = 8, 64
	if !linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod) — nothing to test here")
	}
	f32 := syntheticEmbedF32(rows, cols)

	repackedOnly := quantizeEmbedWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, false)
	if !repackedOnly.IsInt4() {
		t.Fatal("needCanonical=false: result is not int4-resident at all")
	}
	if _, _, _, ok := repackedOnly.Int4(); ok {
		t.Error("needCanonical=false: Int4()'s ok is true — canonical bytes are resident, " +
			"repacked-only construction did not happen")
	}
	if _, _, ok := repackedOnly.Int4Row4(); !ok {
		t.Error("needCanonical=false: Int4Row4() has no data — neither layout is present")
	}
	if repackedOnly.Int4Layout() != "row4" {
		t.Errorf("needCanonical=false: Int4Layout() = %q, want %q", repackedOnly.Int4Layout(), "row4")
	}

	both := quantizeEmbedWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, true)
	if !both.IsInt4() {
		t.Fatal("needCanonical=true: result is not int4-resident")
	}
	if _, _, _, ok := both.Int4(); !ok {
		t.Error("needCanonical=true: Int4()'s ok is false — canonical bytes are NOT resident, " +
			"the existing policy regressed")
	}
}

// TestQuantizeEmbedWM_repackedOnlyMatchesCanonical proves the memory-layout swap is a pure
// storage change: Row() (the per-token embed lookup) and matmul() (the LM head projection, via
// the SAME dispatch every other int4 tensor uses) must be bit-identical between the repacked-
// only and canonical(+row4) constructions of the identical source data.
func TestQuantizeEmbedWM_repackedOnlyMatchesCanonical(t *testing.T) {
	const rows, cols = 8, 64
	if !linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	f32 := syntheticEmbedF32(rows, cols)
	repackedOnly := quantizeEmbedWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, false)
	canonical := quantizeEmbedWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, true)

	// Row(): the per-token embedding lookup, layout-independent per aikit's own contract.
	for r := 0; r < rows; r++ {
		gotR, wantR := make([]float32, cols), make([]float32, cols)
		repackedOnly.Row(r, gotR)
		canonical.Row(r, wantR)
		for c := range gotR {
			if gotR[c] != wantR[c] {
				t.Fatalf("Row(%d)[%d]: repacked-only = %v, canonical = %v — not bit-identical",
					r, c, gotR[c], wantR[c])
			}
		}
	}

	// matmul(): the LM head's own dispatch (decoder/weightmat.go's matmul, CPU backend — no
	// staged consult on a plain CPU backend, so this exercises w.MatmulBTW4A8Into directly, the
	// same path IsInt4()'s dispatch fix (this same audit item) routes both constructions through).
	be, err := NewBackend("")
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	a := make([]float32, cols)
	for i := range a {
		a[i] = float32(math.Cos(float64(i) * 0.11))
	}
	gotDst, wantDst := make([]float32, rows), make([]float32, rows)
	matmul(be, &repackedOnly, a, gotDst, 1)
	matmul(be, &canonical, a, wantDst, 1)
	for i := range gotDst {
		if gotDst[i] != wantDst[i] {
			t.Fatalf("matmul dst[%d]: repacked-only = %v, canonical = %v — not bit-identical",
				i, gotDst[i], wantDst[i])
		}
	}

	// matmulInto: the caller-supplied-Workspace sibling (decode's actual hot path via
	// decodeScratch). Note its own IsInt4() guard is NOT independently load-bearing for
	// CORRECTNESS: if it were wrong, matmulInto falls through to matmul() at the end, which has
	// the correct guard and still produces the right answer — only the ws Workspace reuse (the
	// whole point of matmulInto existing) would silently degrade to matmul()'s pooled-Workspace
	// round-trip. Checked here for parity anyway, since the two code paths should never diverge.
	var ws linalg.Workspace
	gotInto, wantInto := make([]float32, rows), make([]float32, rows)
	matmulInto(&ws, be, &repackedOnly, a, gotInto, 1)
	matmulInto(&ws, be, &canonical, a, wantInto, 1)
	for i := range gotInto {
		if gotInto[i] != wantInto[i] {
			t.Fatalf("matmulInto dst[%d]: repacked-only = %v, canonical = %v — not bit-identical",
				i, gotInto[i], wantInto[i])
		}
	}
}

// TestQuantizeEmbedWM_ineligibleShapeFallsBackToCanonical: a shape Int4Row4Usable rejects (rows
// not a multiple of 4) must still produce a WORKING WeightMat via the existing
// repackW4A8IfEligible policy even when needCanonical is false — no silent data loss, matching
// what every other int4 tensor already gets from quantizeWM.
func TestQuantizeEmbedWM_ineligibleShapeFallsBackToCanonical(t *testing.T) {
	const rows, cols = 6, 64 // 6 is not a multiple of 4
	if linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Fatal("test setup: this shape is supposed to be row4-ineligible")
	}
	f32 := syntheticEmbedF32(rows, cols)
	wm := quantizeEmbedWM(linalg.WrapF32(f32, rows, cols), quantInt4, false)
	if !wm.IsInt4() {
		t.Fatal("result is not int4-resident at all")
	}
	if _, _, _, ok := wm.Int4(); !ok {
		t.Error("Int4()'s ok is false for an ineligible shape — the fallback to the canonical " +
			"policy did not happen, so this tensor has no canonical bytes and no row4 either")
	}
}

// TestLoad_embedInt4RepackedOnlyEndToEnd proves the needCanonical WIRING through the full call
// chain — decoder.Load -> loadWeights -> loadGGUFWeights/buildWeightsFromSafetensors ->
// streamQuantizedEmbed/quantizeEmbedWM — not just the unit-level functions above. A real GGUF
// fixture on the plain CPU backend (which implements none of QuantBackend4/QuantBatchBackend4/
// ResidencyBackend, so wantsCanonicalInt4 must say false) with Options.EmbedInt4 set must come
// out of a real Load() call with Embed built repacked-only, proving every hop of the threading
// (Load's own wantsCanonicalInt4(be) call, and the ~9 intermediate signatures between it and the
// two construction sites) actually carries the right value rather than silently dropping it.
func TestLoad_embedInt4RepackedOnlyEndToEnd(t *testing.T) {
	const fixture = "testdata/gptoss_tiny.gguf"
	if !linalg.Int4Row4Usable(4, int4GroupSize, int4GroupSize) {
		// A crude but sufficient proxy for "this core can do row4 at all" — the real shape check
		// runs inside Load on the fixture's actual embedding dims.
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	m, err := Load(fixture, Options{Quant: "int4", EmbedInt4: true, Backend: "cpu"})
	if err != nil {
		t.Skipf("could not load %s at int4/embed-int4: %v", fixture, err)
	}
	defer m.Close()
	layout := m.w.Embed.Int4Layout()
	if layout != "row4" {
		t.Errorf("m.w.Embed.Int4Layout() = %q after a real Load(EmbedInt4:true, Backend:\"cpu\") "+
			"call, want %q — the needCanonical wiring dropped the value somewhere between Load "+
			"and the construction site", layout, "row4")
	}
}

// TestIsW4A8_trueForRepackedOnly is the gate isW4A8 is FOR: causalAttention/gatedMLP's
// `w4a8BatchEnabled && isW4A8(&lw.QProj) && ...` decides whether a repacked-only tensor ever
// reaches the batch dispatch this whole extension depends on. TestMatmulW4A8Batch_
// repackedOnlyMatchesCanonical (below) proves the dispatch itself is correct once reached, but
// bypasses this gate by calling wmW4A8Op/matmulW4A8Batch directly — this closes that gap.
func TestIsW4A8_trueForRepackedOnly(t *testing.T) {
	const rows, cols = 8, 64
	if !linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	wm := quantizeBatchedProjWM(linalg.WrapF32(syntheticEmbedF32(rows, cols), rows, cols), quantInt4, false)
	if wm.Int4Layout() != "row4" {
		t.Fatalf("test setup: Int4Layout() = %q, want %q", wm.Int4Layout(), "row4")
	}
	if !isW4A8(&wm) {
		t.Error("isW4A8(repacked-only tensor) = false, want true — this tensor would never reach " +
			"the batch dispatch in causalAttention/gatedMLP at all")
	}
}

// TestQuantizeBatchedProjWM_repackedOnlyWhenEligible mirrors
// TestQuantizeEmbedWM_repackedOnlyWhenEligible for the attention Q/K/V and MLP gate/up policy
// (quantizeBatchedProjWM): needCanonical=false on an eligible shape must produce a repacked-only
// WeightMat; needCanonical=true must keep the existing canonical(+row4) "both" policy.
func TestQuantizeBatchedProjWM_repackedOnlyWhenEligible(t *testing.T) {
	const rows, cols = 8, 64
	if !linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	f32 := syntheticEmbedF32(rows, cols)
	repackedOnly := quantizeBatchedProjWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, false)
	if repackedOnly.Int4Layout() != "row4" {
		t.Errorf("needCanonical=false: Int4Layout() = %q, want %q", repackedOnly.Int4Layout(), "row4")
	}
	both := quantizeBatchedProjWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4, true)
	if _, _, _, ok := both.Int4(); !ok {
		t.Error("needCanonical=true: Int4()'s ok is false — canonical bytes are not resident")
	}
}

// TestIsBatchedProjTensor_matchesOnlyTheStandardNames pins isBatchedProjTensor's suffix
// matching: the five standard llama.cpp names (with a layer prefix, matching real usage) match;
// an MoE expert-stacked tensor, an MoE router, and an MLA-style split projection do not — those
// intentionally fall back to the existing canonical(+row4) policy (see quantizeBatchedProjWM's
// own doc comment for why).
func TestIsBatchedProjTensor_matchesOnlyTheStandardNames(t *testing.T) {
	for _, name := range []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
	} {
		if !isBatchedProjTensor(name) {
			t.Errorf("isBatchedProjTensor(%q) = false, want true", name)
		}
	}
	for _, name := range []string{
		"blk.0.ffn_gate_exps.weight", // MoE expert-stacked
		"blk.0.ffn_gate_inp.weight",  // MoE router
		"blk.0.attn_q_a.weight",      // MLA split projection
		"blk.0.ffn_down.weight",      // down-proj, deliberately not covered
		"blk.0.attn_output.weight",   // o_proj, deliberately not covered
	} {
		if isBatchedProjTensor(name) {
			t.Errorf("isBatchedProjTensor(%q) = true, want false", name)
		}
	}
}

// TestMatmulW4A8Batch_repackedOnlyMatchesCanonical is the load-bearing correctness proof for
// extending repacked-only to the attention Q/K/V / MLP gate/up policy: the REAL production
// dispatch (wmW4A8Op building the op, matmulW4A8Batch running it — exactly what
// causalAttention/gatedMLP call, decoder/attention.go:87 and decoder/mlp.go:441) must produce
// bit-identical output whether the three fused ops are repacked-only or canonical(+row4).
func TestMatmulW4A8Batch_repackedOnlyMatchesCanonical(t *testing.T) {
	const rows, cols = 8, 64
	if !linalg.Int4Row4Usable(rows, cols, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	be, err := NewBackend("")
	if err != nil {
		t.Fatalf("NewBackend: %v", err)
	}
	a := make([]float32, cols)
	for i := range a {
		a[i] = float32(math.Cos(float64(i) * 0.11))
	}

	build := func(needCanonical bool) []linalg.WeightMat {
		out := make([]linalg.WeightMat, 3)
		for i := range out {
			// A distinct pattern per tensor so a bug that mixes up ops (wrong Dst, wrong op
			// order) would show up as a mismatch rather than accidentally cancelling out.
			f32 := syntheticEmbedF32(rows, cols)
			for j := range f32 {
				f32[j] *= float32(i + 1)
			}
			out[i] = quantizeBatchedProjWM(linalg.WrapF32(f32, rows, cols), quantInt4, needCanonical)
		}
		return out
	}
	run := func(wms []linalg.WeightMat) []float32 {
		dst := make([]float32, 3*rows)
		ops := make([]linalg.W4A8Op, 3)
		var group int
		for i := range wms {
			ops[i], group = wmW4A8Op(&wms[i], dst[i*rows:(i+1)*rows])
		}
		var ws linalg.Workspace
		matmulW4A8Batch(be, &ws, a, 1, cols, group, ops)
		return dst
	}

	repackedOnly := build(false)
	if repackedOnly[0].Int4Layout() != "row4" {
		t.Fatalf("test setup: repackedOnly[0].Int4Layout() = %q, want %q — the eligibility check "+
			"above should have caught this", repackedOnly[0].Int4Layout(), "row4")
	}
	canonical := build(true)

	gotDst := run(repackedOnly)
	wantDst := run(canonical)
	for i := range gotDst {
		if gotDst[i] != wantDst[i] {
			t.Fatalf("dst[%d]: repacked-only batch = %v, canonical batch = %v — not bit-identical",
				i, gotDst[i], wantDst[i])
		}
	}
}

// TestLoad_qkvGateUpRepackedOnlyEndToEnd is TestLoad_embedInt4RepackedOnlyEndToEnd's twin for
// Q/K/V/gate/up: plain int4 quant (no --embed-int4 needed — this is the ordinary, default int4
// path) on the CPU backend must build every projection matched by isBatchedProjTensor
// repacked-only, proving the full mat/streamMatBatched/loadProj/quantizeBatchedProjWM wiring.
func TestLoad_qkvGateUpRepackedOnlyEndToEnd(t *testing.T) {
	const fixture = "testdata/gptoss_tiny.gguf"
	if !linalg.Int4Row4Usable(4, int4GroupSize, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	m, err := Load(fixture, Options{Quant: "int4", Backend: "cpu"})
	if err != nil {
		t.Skipf("could not load %s at int4: %v", fixture, err)
	}
	defer m.Close()
	if len(m.w.Layers) == 0 {
		t.Fatal("test setup: loaded model has no layers")
	}
	l := m.w.Layers[0]
	checks := []struct {
		name string
		wm   linalg.WeightMat
	}{
		{"QProj", l.QProj}, {"KProj", l.KProj}, {"VProj", l.VProj},
		{"GateProj", l.GateProj}, {"UpProj", l.UpProj},
	}
	sawRow4 := false
	for _, c := range checks {
		if !c.wm.IsInt4() {
			continue // this family/shape didn't end up int4 at all — not this test's concern
		}
		if layout := c.wm.Int4Layout(); layout == "row4" {
			sawRow4 = true
		} else if _, _, _, ok := c.wm.Int4(); !ok {
			t.Errorf("layer 0 %s: Int4Layout() = %q and Int4()'s ok is false — neither canonical "+
				"nor row4, a real construction failure", c.name, layout)
		}
	}
	if !sawRow4 {
		t.Error("none of layer 0's Q/K/V/gate/up projections came out row4 — the needCanonical " +
			"wiring dropped the value somewhere between Load and the construction sites")
	}
}

// TestBackendReport_int4LayoutVisible is item 2 of the gate-redesign followup: the repacked-only
// decision must be diagnosable by reading BackendReport(), not by comparing tok/s. Backend:"cpu"
// (the promise) must show a "-only" layout; an unspecified backend (no promise made) must show
// the existing canonical(+row4) policy untouched.
func TestBackendReport_int4LayoutVisible(t *testing.T) {
	const fixture = "testdata/gptoss_tiny.gguf"
	if !linalg.Int4Row4Usable(4, int4GroupSize, int4GroupSize) {
		t.Skip("this core cannot build row4 at all (non-arm64, or no dotprod)")
	}
	promised, err := Load(fixture, Options{Quant: "int4", EmbedInt4: true, Backend: "cpu"})
	if err != nil {
		t.Skipf("could not load %s: %v", fixture, err)
	}
	defer promised.Close()
	if report := promised.BackendReport(); !strings.Contains(report, "row4-only") {
		t.Errorf("Backend:\"cpu\" BackendReport() = %q, want it to mention \"row4-only\"", report)
	}

	unpromised, err := Load(fixture, Options{Quant: "int4", EmbedInt4: true})
	if err != nil {
		t.Skipf("could not load %s: %v", fixture, err)
	}
	defer unpromised.Close()
	if report := unpromised.BackendReport(); strings.Contains(report, "row4-only") {
		t.Errorf("unspecified-backend BackendReport() = %q, mentions \"row4-only\" — the promise "+
			"was never made, canonical(+row4) should have stayed the policy", report)
	}

	// serve's own startup/diagnostic surface (internal/serveapp/main.go) reports via
	// DecodePath(), not BackendReport() — both must carry the same visibility.
	if dp := promised.DecodePath(); !strings.Contains(dp, "row4-only") {
		t.Errorf("Backend:\"cpu\" DecodePath() = %q, want it to mention \"row4-only\"", dp)
	}
	if dp := unpromised.DecodePath(); strings.Contains(dp, "row4-only") {
		t.Errorf("unspecified-backend DecodePath() = %q, mentions \"row4-only\"", dp)
	}
}
