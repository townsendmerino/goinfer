//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4Router_residentIdxParity is Split-B task 2a, the ROUTER-FIRST gate: before any expert
// GEMV output is compared, prove the resident router selects the SAME experts as the CPU router —
// a binary idx[] equality check, the one MoE failure a whole-forward cosine can't localize (a
// flipped expert is a different computation, not a small error).
//
// The resident router uses gemv_f32_f32 (cuda/router_f32.cu) — a PURE-f32 projection, NOT the
// shared int8-activation gemv_f32_a8. That is a deliberate choice, not a reuse: gemv_f32_a8 would
// quantize the router input rn to int8 (~1e-2), which can flip a top-k decision near a tie. An
// earlier version of this test ran that int8 path and found no flip — but the gemma4-moe-tiny
// fixture's 0.12 routing margin was CONSTRUCTED by least-squares to be wide, so that result is
// CIRCULAR for a trained 128-expert/top-8 router whose 8th-vs-9th boundary is far tighter. f32xf32
// quantizes NOTHING, so the only residual is f32 reduction order (~1e-6) — routing cannot flip from
// activation quant at ANY expert count. This test therefore verifies the kernel we actually ship;
// the 128/top-8 re-run is no longer a correctness precondition (there is no quant perturbation to
// re-check), only a nice-to-have when a real router is available.
func TestGemma4Router_residentIdxParity(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()

	// Kernels: gemv_f32_f32 (routerF32PTX, pure-f32 router projection) + moe_route (moePTX).
	rmod, err := ctx.LoadModule(routerF32PTX)
	if err != nil {
		t.Fatalf("router_f32 module: %v", err)
	}
	mmod, err := ctx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("moe module: %v", err)
	}
	fGemv, err := rmod.Function("gemv_f32_f32")
	if err != nil {
		t.Fatalf("gemv_f32_f32: %v", err)
	}
	fRoute, err := mmod.Function("moe_route")
	if err != nil {
		t.Fatalf("moe_route: %v", err)
	}
	stream := mustStream(t, ctx)

	// CPU int4 forward (the resident path is int4: experts int4, router f32), capturing per-decision
	// idx[] and the finalized router input rn.
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()
	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	cache := m.NewCache(len(prompt))
	for i, tok := range prompt {
		if _, err := m.ForwardForTest(tok, cache); err != nil {
			t.Fatalf("forward pos %d: %v", i, err)
		}
	}
	idxAll, rnAll := decoder.RouterCaptureForTest()
	if len(idxAll) == 0 || len(idxAll) != len(rnAll) {
		t.Fatalf("capture empty or mismatched: %d idx, %d rn", len(idxAll), len(rnAll))
	}

	// MoE layer indices (routerProj differs per layer). Decisions are token-outer, layer-inner, so
	// decision d → moeLayers[d % nMoELayers] (asserted below).
	var moeLayers []int
	for l := 0; l < 64; l++ {
		if _, _, _, _, _, ok := m.Gemma4MoERouterForTest(l); ok {
			moeLayers = append(moeLayers, l)
		}
	}
	if len(moeLayers) == 0 {
		t.Fatal("no gemma4 MoE layers found via accessor")
	}
	if len(idxAll)%len(moeLayers) != 0 {
		t.Fatalf("%d decisions not a multiple of %d MoE layers — decision→layer mapping unsafe", len(idxAll), len(moeLayers))
	}

	mismatches := 0
	for d := range idxAll {
		layer := moeLayers[d%len(moeLayers)]
		proj, bias, nE, topK, hidden, ok := m.Gemma4MoERouterForTest(layer)
		if !ok {
			t.Fatalf("decision %d: router accessor failed for layer %d", d, layer)
		}
		rn := rnAll[d]
		if len(rn) != hidden {
			t.Fatalf("decision %d: rn len %d != hidden %d", d, len(rn), hidden)
		}

		// Device: rn (f32, NO quant) → gemv_f32_f32 → moe_route. Exactly the resident chain.
		dRn := mustAlloc[float32](t, ctx, hidden)
		dProj := mustAlloc[float32](t, ctx, nE*hidden)
		dBias := mustAlloc[float32](t, ctx, nE)
		dLogits := mustAlloc[float32](t, ctx, nE)
		dIdx := mustAlloc[uint32](t, ctx, topK)
		dWgt := mustAlloc[float32](t, ctx, topK)
		_ = gc.CopyHtoD(bg, dRn, rn)
		_ = gc.CopyHtoD(bg, dProj, proj)
		_ = gc.CopyHtoD(bg, dBias, bias)

		_ = fGemv.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			gc.Arg(dProj), gc.Arg(dRn), gc.ArgValue(int32(nE)), gc.ArgValue(int32(hidden)), gc.Arg(dLogits))
		// moe_route: sigmoid=0 (softmax), norm=1 (unconditional renorm), scale=1, nGroup=1, topkGroup=1.
		// Per-expert scale is applied OUTSIDE the router and does not affect selection.
		_ = fRoute.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1},
			gc.Arg(dLogits), gc.Arg(dBias), gc.Arg(dIdx), gc.Arg(dWgt),
			gc.ArgValue(int32(nE)), gc.ArgValue(int32(topK)), gc.ArgValue(int32(0)), gc.ArgValue(int32(1)),
			gc.ArgValue(float32(1)), gc.ArgValue(int32(1)), gc.ArgValue(int32(1)))
		_ = stream.Synchronize(bg)
		gpuIdx := make([]uint32, topK)
		_ = gc.CopyDtoH(bg, gpuIdx, dIdx)

		// Binary equality against the CPU selection (order-independent: top-k is a SET).
		cpu := idxAll[d]
		if !sameExpertSet(cpu, gpuIdx) {
			mismatches++
			t.Errorf("decision %d (layer %d): resident idx %v != CPU idx %v — pure-f32 router diverges from CPU "+
				"(f32 reduction order should never flip a real margin — check the kernel)", d, layer, gpuIdx, cpu)
		}
	}
	if mismatches == 0 {
		t.Logf("router idx parity: %d/%d decisions bit-equal (CPU f32 router vs resident gemv_f32_f32, pure f32, "+
			"no activation quant) — routing retired from the resident suspect list at any expert count", len(idxAll), len(idxAll))
	}
}

// TestGemma4_perExpertScaleFold isolates the on-GPU per-expert-scale fold (scale_wgt_by_expert,
// cuda/router_f32.cu) before it is wired into the resident MoE combine. Gemma-4 multiplies each
// routed weight by a learned per-EXPERT scale indexed by the selected expert id:
// wts[k] = (topv[k]/sum) * perExpertScale[idx[k]]. The generic moe_route has only a single scalar
// routed_scaling_factor, so this is a new op. It must run on-device (idx is a moe_route output; a
// host fold would sync per token), so verify the kernel's indexed multiply against a CPU oracle.
func TestGemma4_perExpertScaleFold(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()
	rmod, err := ctx.LoadModule(routerF32PTX)
	if err != nil {
		t.Fatalf("router_f32 module: %v", err)
	}
	fScale, err := rmod.Function("scale_wgt_by_expert")
	if err != nil {
		t.Fatalf("scale_wgt_by_expert: %v", err)
	}
	stream := mustStream(t, ctx)

	// top-2 of 4: weights from moe_route, selected expert ids, learned per-expert scale.
	wgt := []float32{0.6, 0.4}
	idx := []uint32{2, 0}
	perExpertScale := []float32{1.5, 0.5, 2.0, 1.0}
	nE, K := len(perExpertScale), len(wgt)
	want := make([]float32, K)
	for k := 0; k < K; k++ {
		want[k] = wgt[k] * perExpertScale[idx[k]] // 0.6*2.0=1.2 ; 0.4*1.5=0.6
	}

	dWgt := mustAlloc[float32](t, ctx, K)
	dIdx := mustAlloc[uint32](t, ctx, K)
	dScale := mustAlloc[float32](t, ctx, nE)
	_ = gc.CopyHtoD(bg, dWgt, wgt)
	_ = gc.CopyHtoD(bg, dIdx, idx)
	_ = gc.CopyHtoD(bg, dScale, perExpertScale)
	_ = fScale.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: uint32(K), BlockY: 1, BlockZ: 1},
		gc.Arg(dWgt), gc.Arg(dIdx), gc.Arg(dScale), gc.ArgValue(int32(K)))
	_ = stream.Synchronize(bg)
	got := make([]float32, K)
	_ = gc.CopyDtoH(bg, got, dWgt)

	for k := 0; k < K; k++ {
		if d := got[k] - want[k]; d > 1e-6 || d < -1e-6 {
			t.Errorf("wgt[%d]: got %.6f want %.6f (idx=%d scale=%.3f)", k, got[k], want[k], idx[k], perExpertScale[idx[k]])
		}
	}
	t.Logf("per-expert-scale fold: %v * scale[%v] = %v (matches CPU)", wgt, idx, got)
}

// TestGemma4Expert_geluTanhChain isolates the gemma4 MoE EXPERT function on the resident path:
// gemv_w4a8_moe (indexed stacked gate‖up) → glu_quant(act=GELU_TANH) → gemv_w4a8_moe (down), for
// ONE expert, vs the CPU expert on the same input. This combination — the gelu-tanh glu_quant
// epilogue between two INDEXED-expert GEMVs — ships in neither Gemma-3 (dense, so glu_quant but not
// indexed) nor Mixtral/GLM (indexed, but SiLU), so it has never actually run. Rather than argue it
// safe by composition (the same reasoning that said hd=512 "should" work, which we tested anyway),
// run it. A NEGATIVE CONTROL proves the check isn't vacuous: the same chain with act=SILU must
// diverge — if a SiLU misdispatch scored as well as gelu-tanh, the gate would be blind to exactly
// the failure it exists to catch.
func TestGemma4Expert_geluTanhChain(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()
	gmod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("glue module: %v", err)
	}
	mmod, err := ctx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("moe module: %v", err)
	}
	fQuant := mustFn(t, gmod, "quant_vec")
	fGlu := mustFn(t, gmod, "glu_quant")
	fMoE := mustFn(t, mmod, "gemv_w4a8_moe")
	stream := mustStream(t, ctx)

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()

	layer, expert := firstMoELayer(t, m), 0
	_, _, _, _, hidden, ok := m.Gemma4MoERouterForTest(layer)
	if !ok {
		t.Fatalf("router accessor failed for layer %d", layer)
	}
	xe := make([]float32, hidden)
	for i := range xe {
		xe[i] = float32(math.Sin(float64(i)*0.13)) * 0.8 // normed-hidden scale
	}
	edownRef, gateUp, down, moeInter, _, ok := m.Gemma4MoEExpertForTest(layer, expert, xe)
	if !ok {
		t.Fatalf("expert accessor failed (layer %d expert %d)", layer, expert)
	}
	guHost, err := packWeightStack(gateUp)
	if err != nil {
		t.Fatalf("pack gate‖up: %v", err)
	}
	dnHost, err := packWeightStack(down)
	if err != nil {
		t.Fatalf("pack down: %v", err)
	}
	if guHost.kind != "int4" || dnHost.kind != "int4" {
		t.Fatalf("expert weights are %q/%q, not int4 — the resident MoE GEMV is int4-only", guHost.kind, dnHost.kind)
	}

	// upload int4 weights (packed uint32 + f16 group scales) — the moe_gemv_test pattern.
	upW := func(h hostW) (*gc.Buffer[uint32], *gc.Buffer[uint16]) {
		w := mustAlloc[uint32](t, ctx, len(h.wpk))
		g := mustAlloc[uint16](t, ctx, len(h.ws16))
		_ = gc.CopyHtoD(bg, w, h.wpk)
		_ = gc.CopyHtoD(bg, g, h.ws16)
		return w, g
	}
	guW, guGs := upW(guHost)
	dnW, dnGs := upW(dnHost)

	guN := 2 * moeInter
	dXe := mustAlloc[float32](t, ctx, hidden)
	dMq := mustAlloc[int32](t, ctx, hidden/4)
	dMSc := mustAlloc[float32](t, ctx, 1)
	dIdx := mustAlloc[uint32](t, ctx, 1)
	dMoeGU := mustAlloc[float32](t, ctx, guN)
	dMoeQ := mustAlloc[int32](t, ctx, moeInter/4)
	dMoeSc := mustAlloc[float32](t, ctx, 1)
	dMoeScr := mustAlloc[float32](t, ctx, moeInter)
	dEdown := mustAlloc[float32](t, ctx, hidden)
	_ = gc.CopyHtoD(bg, dXe, xe)
	_ = gc.CopyHtoD(bg, dIdx, []uint32{0})

	// The full resident expert chain, activation selectable so the negative control can flip it.
	ck := func(what string, e error) {
		if e != nil {
			t.Fatalf("%s: %v", what, e)
		}
	}
	runChain := func(act int32) []float32 {
		ck("quant_vec", fQuant.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			gc.Arg(dXe), gc.ArgValue(int32(hidden)), gc.Arg(dMq), gc.Arg(dMSc)))
		ck("gemv gate‖up", fMoE.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((guN + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			gc.Arg(guW), gc.Arg(dMq), gc.Arg(guGs), gc.Arg(dMSc), gc.Arg(dIdx),
			gc.ArgValue(int32(0)), gc.ArgValue(int32(guN)), gc.ArgValue(int32(guN)),
			gc.ArgValue(int32(hidden/8)), gc.ArgValue(int32(hidden/32)), gc.Arg(dMoeGU)))
		ck("glu_quant", fGlu.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			gc.Arg(dMoeGU), gc.Arg(dMoeGU), gc.ArgValue(int32(0)), gc.ArgValue(int32(moeInter)),
			gc.ArgValue(int32(moeInter)), gc.ArgValue(act), gc.Arg(dMoeQ), gc.Arg(dMoeSc), gc.Arg(dMoeScr)))
		ck("gemv down", fMoE.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32((hidden + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
			gc.Arg(dnW), gc.Arg(dMoeQ), gc.Arg(dnGs), gc.Arg(dMoeSc), gc.Arg(dIdx),
			gc.ArgValue(int32(0)), gc.ArgValue(int32(hidden)), gc.ArgValue(int32(hidden)),
			gc.ArgValue(int32(moeInter/8)), gc.ArgValue(int32(moeInter/32)), gc.Arg(dEdown)))
		ck("sync", stream.Synchronize(bg))
		out := make([]float32, hidden)
		ck("D2H", gc.CopyDtoH(bg, out, dEdown))
		return out
	}

	gelu := runChain(0) // ACT_GELU_TANH — what gemma4 ships
	silu := runChain(1) // ACT_SILU — negative control
	cosGelu := cosine(edownRef, gelu)
	cosSilu := cosine(edownRef, silu)
	t.Logf("single-expert gelu-tanh chain vs CPU: cosine=%.6f | SiLU control=%.6f", cosGelu, cosSilu)
	if cosGelu < 0.97 {
		t.Errorf("gelu-tanh expert chain cosine %.6f < 0.97 — the indexed-GEMV × gelu-tanh combination diverges from CPU", cosGelu)
	}
	if cosSilu >= cosGelu {
		t.Errorf("negative control failed: SiLU cosine %.6f ≥ gelu-tanh %.6f — the gate can't tell the activations apart", cosSilu, cosGelu)
	}
}

// mustFn resolves a kernel or fails loudly (a missing symbol is a build/embed error, not a bug).
func mustFn(t *testing.T, mod *gc.Module, name string) *gc.Function {
	t.Helper()
	f, err := mod.Function(name)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return f
}

// firstMoELayer returns the lowest gemma4 MoE layer index (fails if none).
func firstMoELayer(t *testing.T, m *decoder.Model) int {
	t.Helper()
	for l := 0; l < 64; l++ {
		if _, _, _, _, _, ok := m.Gemma4MoERouterForTest(l); ok {
			return l
		}
	}
	t.Fatal("no gemma4 MoE layer found")
	return -1
}

// TestGemma4_rmsnormNW_scaleVec isolates the two elementwise kernels the gemma4 MoE orchestration
// adds (cuda/router_f32.cu): rmsnorm_nw (weightless OUT-OF-PLACE RMSNorm — the router norms raw h
// without mutating it) and scale_vec (the per-layer output scalar). rmsnorm_nw is a reduction, the
// same class as the qk_norm reduction that the hd=512 sweep exercised, so verify it vs a CPU oracle.
func TestGemma4_rmsnormNW_scaleVec(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()
	rmod, err := ctx.LoadModule(routerF32PTX)
	if err != nil {
		t.Fatalf("router_f32 module: %v", err)
	}
	fNW := mustFn(t, rmod, "rmsnorm_nw")
	fSc := mustFn(t, rmod, "scale_vec")
	stream := mustStream(t, ctx)

	const H = 256
	const eps = float32(1e-6)
	src := make([]float32, H)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.17)) * 1.7
	}
	// CPU rmsNormNoWeight: x * rsqrt(mean(x^2)+eps).
	var ss float64
	for _, v := range src {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(H)+float64(eps)))
	want := make([]float32, H)
	for i := range src {
		want[i] = src[i] * inv
	}

	dSrc := mustAlloc[float32](t, ctx, H)
	dDst := mustAlloc[float32](t, ctx, H)
	_ = gc.CopyHtoD(bg, dSrc, src)
	if e := fNW.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		gc.Arg(dSrc), gc.Arg(dDst), gc.ArgValue(int32(H)), gc.ArgValue(eps)); e != nil {
		t.Fatalf("rmsnorm_nw: %v", e)
	}
	// scale by 0.75 in place, and confirm src (the input) is UNTOUCHED (out-of-place).
	if e := fSc.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
		gc.Arg(dDst), gc.ArgValue(float32(0.75)), gc.ArgValue(int32(H))); e != nil {
		t.Fatalf("scale_vec: %v", e)
	}
	_ = stream.Synchronize(bg)
	got := make([]float32, H)
	gotSrc := make([]float32, H)
	_ = gc.CopyDtoH(bg, got, dDst)
	_ = gc.CopyDtoH(bg, gotSrc, dSrc)

	var maxRel float64
	for i := range got {
		ref := want[i] * 0.75
		if r := math.Abs(float64(got[i]-ref)) / (math.Abs(float64(ref)) + 1e-6); r > maxRel {
			maxRel = r
		}
		if gotSrc[i] != src[i] {
			t.Fatalf("rmsnorm_nw mutated its INPUT at %d (%.6f != %.6f) — not out-of-place", i, gotSrc[i], src[i])
		}
	}
	t.Logf("rmsnorm_nw + scale_vec vs CPU: maxRelErr=%.2e, input preserved", maxRel)
	if maxRel > 1e-3 {
		t.Errorf("rmsnorm_nw*0.75 maxRelErr %.3e > 1e-3", maxRel)
	}
}

// sameExpertSet compares a CPU selection ([]int) and a resident selection ([]uint32) as SETS.
func sameExpertSet(a []int, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	seen := make(map[int]bool, len(a))
	for _, x := range a {
		seen[x] = true
	}
	for _, x := range b {
		if !seen[int(x)] {
			return false
		}
	}
	return true
}
