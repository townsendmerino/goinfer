//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4Router_residentIdxParity is the 9c Step-5 ROUTER-FIRST gate: before any expert GEMV
// output is compared, prove the Metal resident router selects the SAME experts as the CPU router —
// a binary idx[] equality check, the one MoE failure a whole-forward cosine cannot localize (a
// flipped expert is a DIFFERENT computation, not a small error). Routing is gated first and
// independently because it is the only discrete-failure path in the whole dense‖MoE delta.
//
// The resident router uses gemv_f32_f32 (metal/gemma4_moe.go) — a PURE-f32 projection, NOT the
// shared int8-activation gemv_wf32_a8. Deliberate, not reuse: gemv_wf32_a8 quantizes the router
// input rn to int8 (~1e-2), which can flip a top-k decision near a tie; the gemma4-moe-tiny
// fixture's margin (0.268, measured by the Step-5 pre-flight noise-floor check) was CONSTRUCTED wide,
// so a "no flip" result on the int8 path would be circular for a trained 128-expert/top-8 router
// whose 8th-vs-9th boundary is far tighter. f32×f32 quantizes NOTHING, so the only residual is f32
// reduction order (simdgroup tree vs CPU sequential, ~1e-6) — routing cannot flip from activation
// quant at ANY expert count. Mirrors cuda/gemma4_router_parity_test.go exactly.
//
// The captured CPU rn is the FINALIZED router input (weightlessNorm(h)·routerScale·hidden^-0.5), so
// it is paired with the RAW routerProj (Gemma4MoERouterForTest) — NOT the build-time RouterProjScaled
// (which folds routerScale·hidden^-0.5 into its columns and pairs with the UNSCALED rmsnorm_nw
// output). Both compute identical logits; using raw-proj + scaled-rn here avoids a double fold.
func TestGemma4Router_residentIdxParity(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture (%s) — scp from the box / run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile (incl. gemma4MoeKernels): %v", err)
	}
	pGemv, err := d.NewComputePipeline(lib, "gemv_f32_f32")
	if err != nil {
		t.Fatalf("pipeline gemv_f32_f32: %v", err)
	}
	pRoute, err := d.NewComputePipeline(lib, "moe_route")
	if err != nil {
		t.Fatalf("pipeline moe_route: %v", err)
	}

	// CPU int4 forward (the resident path is int4: experts int4, router f32), capturing per-decision
	// idx[] and the finalized router input rn.
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()
	decoder.SetRouterCaptureForTest(true)
	defer decoder.SetRouterCaptureForTest(false)
	cache := m.NewCache(len(twoGeomPrompt))
	for i, tok := range twoGeomPrompt {
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
	for l := range 64 {
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

	q := d.NewCommandQueue()
	mismatches := 0
	for dcs := range idxAll {
		layer := moeLayers[dcs%len(moeLayers)]
		proj, bias, nE, topK, hidden, ok := m.Gemma4MoERouterForTest(layer)
		if !ok {
			t.Fatalf("decision %d: router accessor failed for layer %d", dcs, layer)
		}
		rn := rnAll[dcs]
		if len(rn) != hidden {
			t.Fatalf("decision %d: rn len %d != hidden %d", dcs, len(rn), hidden)
		}

		// Device: rn (f32, NO quant) → gemv_f32_f32 → moe_route. Exactly the resident router chain.
		logits := d.NewBufferLen(nE)
		q.Run1D(pGemv, nE*32, 32, NewBufferFloats(d, proj), NewBufferFloats(d, rn), logits, NewBufferU32(d, uint32(hidden)))
		// moe_route: sigmoid=0 (softmax), norm=1 (unconditional renorm), scale=1, nGroup=1, topkGroup=1.
		// Per-expert scale is applied OUTSIDE the router and does not affect selection.
		idxBuf := NewBufferUint32s(d, make([]uint32, topK))
		wgtBuf := d.NewBufferLen(topK)
		q.Run1D(pRoute, 1, 1, logits, NewBufferFloats(d, bias), idxBuf, wgtBuf,
			NewBufferU32(d, uint32(nE)), NewBufferU32(d, uint32(topK)),
			NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferFloats(d, []float32{1}),
			NewBufferU32(d, 1), NewBufferU32(d, 1))
		gpuIdx := idxBuf.U32s()

		// Binary equality against the CPU selection (order-independent: top-k is a SET).
		cpu := idxAll[dcs]
		got := make([]int, len(gpuIdx))
		for j, v := range gpuIdx {
			got[j] = int(v)
		}
		if !sameSet(cpu, got) {
			mismatches++
			t.Errorf("decision %d (layer %d): resident idx %v != CPU idx %v — pure-f32 router diverges from CPU "+
				"(f32 reduction order should never flip a real margin — check the kernel)", dcs, layer, got, cpu)
		}
	}
	if mismatches == 0 {
		t.Logf("router idx parity: %d/%d decisions bit-equal (CPU f32 router vs resident gemv_f32_f32, pure f32, no "+
			"activation quant) — routing retired from the Metal resident suspect list at any expert count", len(idxAll), len(idxAll))
	}
}
