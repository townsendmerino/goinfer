//go:build cuda

package cuda

import (
	"context"
	"errors"
	"io/fs"
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
// It also settles a concrete kernel decision. The CPU gemma4 router is pure f32×f32 (routerProj
// f32, rn f32); the resident selection kernel gemv_f32_a8 takes an INT8-quantized activation, so a
// resident router would quantize rn and CPU does not. This test replays the CPU forward's captured
// router inputs (routerRnBuf) through the exact resident chain — quant_vec (int8, maxabs/127) →
// gemv_f32_a8 → moe_route (softmax, no bias, unconditional renorm) — and asserts idx[] matches the
// CPU selection. If int8-on-rn flips any decision, task 2c must add a pure-f32 router GEMV; if it
// holds (the fixture's constructed min routing margin is 0.12, » the sub-1% int8 perturbation),
// gemv_f32_a8 is safe to reuse for gemma4's router.
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

	// Kernels: quant_vec (gluePTX) for the int8 activation, gemv_f32_a8 + moe_route (moePTX).
	gmod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("glue module: %v", err)
	}
	mmod, err := ctx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("moe module: %v", err)
	}
	fQuant, err := gmod.Function("quant_vec")
	if err != nil {
		t.Fatalf("quant_vec: %v", err)
	}
	fGemv, err := mmod.Function("gemv_f32_a8")
	if err != nil {
		t.Fatalf("gemv_f32_a8: %v", err)
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

		// Device: rn → int8 (quant_vec) → gemv_f32_a8 → moe_route. Exactly the resident chain.
		dRn := mustAlloc[float32](t, ctx, hidden)
		dQ := mustAlloc[int32](t, ctx, hidden/4)
		dScale := mustAlloc[float32](t, ctx, 1)
		dProj := mustAlloc[float32](t, ctx, nE*hidden)
		dBias := mustAlloc[float32](t, ctx, nE)
		dLogits := mustAlloc[float32](t, ctx, nE)
		dIdx := mustAlloc[uint32](t, ctx, topK)
		dWgt := mustAlloc[float32](t, ctx, topK)
		_ = gc.CopyHtoD(bg, dRn, rn)
		_ = gc.CopyHtoD(bg, dProj, proj)
		_ = gc.CopyHtoD(bg, dBias, bias)

		_ = fQuant.LaunchOn(bg, stream, gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			gc.Arg(dRn), gc.ArgValue(int32(hidden)), gc.Arg(dQ), gc.Arg(dScale))
		_ = fGemv.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(nE), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			gc.Arg(dProj), gc.Arg(dQ), gc.Arg(dScale), gc.ArgValue(int32(nE)), gc.ArgValue(int32(hidden)), gc.Arg(dLogits))
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
			t.Errorf("decision %d (layer %d): resident idx %v != CPU idx %v — int8-on-rn FLIPPED the router",
				d, layer, gpuIdx, cpu)
		}
	}
	if mismatches == 0 {
		t.Logf("router idx parity: %d/%d decisions bit-equal (CPU f32 router vs resident int8-rn gemv_f32_a8) — "+
			"gemv_f32_a8 is safe for gemma4's router; no pure-f32 router GEMV needed", len(idxAll), len(idxAll))
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
