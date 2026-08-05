//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4Expert_geluTanhChain isolates the gemma4 MoE EXPERT function on the Metal resident path:
// gemv_w4a8_moe (indexed stacked gate‖up) → swiglu_quant(act=GELU_TANH) → gemv_w4a8_moe (down), for
// ONE expert, vs the CPU expert on the same input. This combination — the gelu-tanh GeGLU epilogue
// between two INDEXED-expert GEMVs — ships in neither Gemma-3 (dense: GeGLU but not indexed) nor
// Mixtral/GLM (indexed: but SiLU), so it has never actually run on Metal. Rather than argue it safe
// by composition (the same reasoning that said hd=512 "should" work, which we tested anyway), run it.
// A NEGATIVE CONTROL proves the check isn't vacuous: the same chain with act=SILU must diverge — if a
// SiLU misdispatch scored as well as gelu-tanh, the gate would be blind to exactly the failure it
// exists to catch. Mirrors cuda/gemma4_router_parity_test.go TestGemma4Expert_geluTanhChain.
func TestGemma4Expert_geluTanhChain(t *testing.T) {
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
		t.Fatalf("compile: %v", err)
	}
	pQv, err := d.NewComputePipeline(lib, "quant_vec")
	if err != nil {
		t.Fatalf("pipeline quant_vec: %v", err)
	}
	pMoE, err := d.NewComputePipeline(lib, "gemv_w4a8_moe")
	if err != nil {
		t.Fatalf("pipeline gemv_w4a8_moe: %v", err)
	}
	pSw, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		t.Fatalf("pipeline swiglu_quant: %v", err)
	}

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()

	// first MoE layer, expert 0.
	layer := -1
	for l := 0; l < 64; l++ {
		if _, _, _, _, _, ok := m.Gemma4MoERouterForTest(l); ok {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatal("no gemma4 MoE layer found")
	}
	_, _, _, _, hidden, ok := m.Gemma4MoERouterForTest(layer)
	if !ok {
		t.Fatalf("router accessor failed for layer %d", layer)
	}
	xe := make([]float32, hidden)
	for i := range xe {
		xe[i] = float32(math.Sin(float64(i)*0.13)) * 0.8 // normed-hidden scale
	}
	edownRef, gateUp, down, moeInter, _, ok := m.Gemma4MoEExpertForTest(layer, 0, xe)
	if !ok {
		t.Fatalf("expert accessor failed (layer %d expert 0)", layer)
	}

	// Pack the expert's fused gate‖up and down into W4A8 (idx=[0] → the buffer's own rows).
	guW, guS, err := int4Buf(d, gateUp)
	if err != nil {
		t.Fatalf("pack gate‖up: %v", err)
	}
	dnW, dnS, err := int4Buf(d, down)
	if err != nil {
		t.Fatalf("pack down: %v", err)
	}

	q := d.NewCommandQueue()
	uHidden, uMoeInter := d.NewBufferU32(uint32(hidden)), d.NewBufferU32(uint32(moeInter))
	idx0, slot0 := d.NewBufferUint32s([]uint32{0}), d.NewBufferU32(0)
	uRowsGU, uRowsDown := d.NewBufferU32(uint32(2*moeInter)), d.NewBufferU32(uint32(hidden))

	// The full resident expert chain, activation selectable so the negative control can flip it.
	runChain := func(act uint32) []float32 {
		mq, mSc := byteBuf(d, hidden), d.NewBufferLen(1)
		q.Run1D(pQv, 256, 256, d.NewBufferFloats(xe), mq, mSc, uHidden)
		// gate‖up: idx[0]*rowsPerExpert + row, rowsPerExpert = 2*moeInter, K = hidden.
		moeGU := d.NewBufferLen(2 * moeInter)
		q.Run1DBatchTG(pMoE, (2*moeInter)*32, 256, 1, hidden*2,
			guW, guS, mq, mSc, moeGU, uHidden, idx0, slot0, uRowsGU)
		// swiglu: gate @0, up @moeInter → dq/dSc.
		dq, dSc := byteBuf(d, moeInter), d.NewBufferLen(1)
		q.Run1D(pSw, 256, 256, moeGU, moeGU.At(moeInter*4), dq, dSc, uMoeInter, d.NewBufferU32(act))
		// down: rowsPerExpert = hidden, K = moeInter → edown[hidden].
		edown := d.NewBufferLen(hidden)
		q.Run1DBatchTG(pMoE, hidden*32, 256, 1, moeInter*2,
			dnW, dnS, dq, dSc, edown, uMoeInter, idx0, slot0, uRowsDown)
		return edown.Floats()
	}

	gelu := runChain(0) // ACT_GELU_TANH — what gemma4 ships
	silu := runChain(1) // ACT_SILU — negative control
	cosGelu, _ := cosMaxAbs(edownRef, gelu)
	cosSilu, _ := cosMaxAbs(edownRef, silu)
	t.Logf("single-expert gelu-tanh chain vs CPU: cosine=%.6f | SiLU control=%.6f", cosGelu, cosSilu)
	if cosGelu < 0.97 {
		t.Errorf("gelu-tanh expert chain cosine %.6f < 0.97 — the indexed-GEMV × gelu-tanh combination diverges from CPU", cosGelu)
	}
	if cosSilu >= cosGelu {
		t.Errorf("negative control failed: SiLU cosine %.6f ≥ gelu-tanh %.6f — the gate can't tell the activations apart", cosSilu, cosGelu)
	}
}
