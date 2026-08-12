//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
)

// TestMoEIndexedGemv proves the two halves of expert selection agree: the packing puts expert e
// at e*rowsPerExpert, and the kernel's indexed read (wrow = idx[slot]*rowsPerExpert + row)
// lands on it.
//
// Tested TOGETHER on purpose. Each half is individually plausible and separately tested
// (TestPackWeightStack, the dense GEMV parity), but a stride disagreement between them is
// invisible to both: the GEMV happily reads whatever is at the offset it computed and returns a
// number. The failure has no crash and no NaN — just a token routed through the wrong expert's
// weights, which is the MoE bug class that does not announce itself.
//
// The reference is the SAME kernel on the SAME expert packed ALONE, so it isolates the indexing
// from the arithmetic: identical math, identical bytes, only the offset differs. Anything but
// bit-equality means the index is wrong.
func TestMoEIndexedGemv(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()

	moeMod, err := cx.LoadModule(moePTX)
	if err != nil {
		t.Fatalf("LoadModule(moe): %v", err)
	}
	fnMoE, err := moeMod.Function("gemv_w4a8_moe")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_moe): %v", err)
	}
	fnWacc, err := moeMod.Function("gemv_w4a8_moe_wacc")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_moe_wacc): %v", err)
	}
	stream := mustStream(t, cx)

	const nE, N, K, group, k = 6, 16, 128, 32, 3
	kw, kg, kd4 := K/8, K/group, K/4

	mk := func(seed uint32) linalg.WeightMat {
		f := make([]float32, N*K)
		s := seed
		for i := range f {
			s = s*1664525 + 1013904223
			f[i] = float32(int32(s>>8)%2000-1000) / 1000
		}
		return linalg.QuantizeInt4(f, N, K, group)
	}
	mats := make([]linalg.WeightMat, nE)
	ptrs := make([]*linalg.WeightMat, nE)
	for e := range mats {
		mats[e] = mk(uint32(31 + e*101)) // distinct per expert: reading the wrong one MUST differ
		ptrs[e] = &mats[e]
	}
	stacked, err := packWeightStack(ptrs...)
	if err != nil {
		t.Fatalf("packWeightStack: %v", err)
	}

	// One int8 activation, shared by every run.
	act := make([]int32, kd4)
	var s uint32 = 987
	for i := range act {
		var p int32
		for b := range 4 {
			s = s*1664525 + 1013904223
			p |= (int32(int8(s>>24)) & 0xff) << (8 * uint(b))
		}
		act[i] = p
	}
	aScale := float32(0.017)

	dA := mustAlloc[int32](t, cx, kd4)
	dAs := mustAlloc[float32](t, cx, 1)
	dIdx := mustAlloc[uint32](t, cx, k)
	dWgt := mustAlloc[float32](t, cx, k)
	dOut := mustAlloc[float32](t, cx, N)
	defer dA.Close()
	defer dAs.Close()
	defer dIdx.Close()
	defer dWgt.Close()
	defer dOut.Close()
	for _, e := range []error{gc.CopyHtoD(bg, dA, act), gc.CopyHtoD(bg, dAs, []float32{aScale})} {
		if e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}

	upload := func(h hostW) (*gc.Buffer[uint32], *gc.Buffer[uint16]) {
		w := mustAlloc[uint32](t, cx, len(h.wpk))
		g := mustAlloc[uint16](t, cx, len(h.ws16))
		if e := gc.CopyHtoD(bg, w, h.wpk); e != nil {
			t.Fatalf("H2D W: %v", e)
		}
		if e := gc.CopyHtoD(bg, g, h.ws16); e != nil {
			t.Fatalf("H2D gs: %v", e)
		}
		return w, g
	}
	dW, dGs := upload(stacked)
	defer dW.Close()
	defer dGs.Close()

	run := func(fn *gc.Function, W *gc.Buffer[uint32], gs *gc.Buffer[uint16], rowsPerExpert, slot int, wacc bool) []float32 {
		zero := make([]float32, N)
		if e := gc.CopyHtoD(bg, dOut, zero); e != nil {
			t.Fatalf("zero dst: %v", e)
		}
		cfg := gc.LaunchConfig{GridX: uint32((N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		var e error
		if wacc {
			e = fn.LaunchOn(bg, stream, cfg, gc.Arg(W), gc.Arg(dA), gc.Arg(gs), gc.Arg(dAs),
				gc.Arg(dIdx), gc.Arg(dWgt), gc.ArgValue(int32(slot)), gc.ArgValue(int32(rowsPerExpert)),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dOut))
		} else {
			e = fn.LaunchOn(bg, stream, cfg, gc.Arg(W), gc.Arg(dA), gc.Arg(gs), gc.Arg(dAs),
				gc.Arg(dIdx), gc.ArgValue(int32(slot)), gc.ArgValue(int32(rowsPerExpert)),
				gc.ArgValue(int32(N)), gc.ArgValue(int32(kw)), gc.ArgValue(int32(kg)), gc.Arg(dOut))
		}
		if e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		out := make([]float32, N)
		if e := gc.CopyDtoH(bg, out, dOut); e != nil {
			t.Fatalf("D2H: %v", e)
		}
		return out
	}

	// For each expert: route slot 0 to it, and compare against that expert packed ALONE
	// (rowsPerExpert=N, idx=0 ⇒ offset 0). Same kernel, same bytes, only the index differs.
	for e := range nE {
		if err := gc.CopyHtoD(bg, dIdx, []uint32{uint32(e), 0, 0}); err != nil {
			t.Fatalf("H2D idx: %v", err)
		}
		got := run(fnMoE, dW, dGs, N, 0, false)

		alone, err := packWeight(ptrs[e])
		if err != nil {
			t.Fatalf("packWeight: %v", err)
		}
		aW, aGs := upload(alone)
		if err := gc.CopyHtoD(bg, dIdx, []uint32{0, 0, 0}); err != nil {
			t.Fatalf("H2D idx: %v", err)
		}
		want := run(fnMoE, aW, aGs, N, 0, false)
		aW.Close()
		aGs.Close()

		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("expert %d row %d: indexed read gave %v, the same expert packed alone gives %v — "+
					"the stacked offset and the kernel's stride DISAGREE, so the token is routed through "+
					"another expert's weights. No crash, no NaN: just the wrong expert.",
					e, i, got[i], want[i])
			}
		}
	}
	t.Logf("all %d experts: indexed read == same expert packed alone, bit-exact", nE)

	// The routing must actually SELECT: distinct experts must give distinct results, or the test
	// above would pass just as well against a kernel that ignored idx entirely.
	if err := gc.CopyHtoD(bg, dIdx, []uint32{0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	e0 := run(fnMoE, dW, dGs, N, 0, false)
	if err := gc.CopyHtoD(bg, dIdx, []uint32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	e1 := run(fnMoE, dW, dGs, N, 0, false)
	same := true
	for i := range e0 {
		if e0[i] != e1[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("expert 0 and expert 1 produced IDENTICAL output — the kernel is ignoring idx, so every " +
			"per-expert check above passed vacuously")
	}

	// wacc: dst += wgt[slot] * result. Zeroed dst ⇒ exactly wgt*plain.
	const w0 = 0.25
	if err := gc.CopyHtoD(bg, dIdx, []uint32{2, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := gc.CopyHtoD(bg, dWgt, []float32{w0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	plain := run(fnMoE, dW, dGs, N, 0, false)
	acc := run(fnWacc, dW, dGs, N, 0, true)
	for i := range acc {
		want := float64(plain[i]) * w0
		if d := math.Abs(float64(acc[i]) - want); d > 1e-5*math.Abs(want)+1e-6 {
			t.Errorf("wacc row %d: %v, want wgt*plain = %v (|d|=%.3g) — the expert combine is not "+
				"applying the routed weight", i, acc[i], want, d)
		}
	}
	t.Logf("wacc applies the routed weight (dst += wgt*result) and matches wgt*plain")
}
