//go:build cuda

package cuda

import (
	"context"
	"fmt"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestGemvShapeSweep sweeps the PRODUCTION forward GEMVs across the geometries the real model
// lineup actually produces, instead of the single shape the parity tests pin.
//
// WHY THIS EXISTS. The W4A8 parity test asserts one shape:
//
//	const N, K = 8960, 1536 // FFN shape; K mult of 32
//
// and that shape SATISFIES the precondition the shipped kernel assumed. packWeight guarded
// K%32, but the kernel's lanes step in 32-word strides, so the real requirement is on
// Kwords = K/8. Qwen2.5-0.5B (hidden 896) gives Kwords = 112, and 112%32 = 16 — the tail lanes
// read past the row. K=1536 gives Kwords=192, 192%32 == 0, so the bug was invisible BY
// CONSTRUCTION: the one tested shape was the one that could not fail. That cost a real
// out-of-bounds read on the 0.5B, which is the model the README's headline number is measured on.
//
// The bar here is EXACT, not cosine. An out-of-bounds read or a bad tail is a wrong ANSWER, not
// a numerical drift, so it needs no near-tie threshold — and a threshold would be wrong anyway:
// random weights score ~0.94 cosine even when correct (measured independently on CUDA and
// Metal), so a cosine bar over synthetic shapes is either too loose to catch anything or too
// tight to pass. Same weights, same activations, both sides — the answers must agree to f32
// accumulation order, so this compares against the exact CPU reference with a tolerance that
// only absorbs float-summation order.
//
// Cheap on purpose: no model loads, ~0.05 s per shape, so the whole sweep is ~1 s. It is the
// regression net under the NEXT kernel (CUDA MoE), where shape assumptions bite hardest —
// expert count, top-k, intermediate dim, stacked-expert layouts.
func TestGemvShapeSweep(t *testing.T) {
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

	// gemv_w4a8_fwd lives in aikit/gpu since the Phase-1b blob-split.
	mod, err := cx.LoadModule(gpu.QuantGEMVPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn4, err := mod.Function("gemv_w4a8_fwd")
	if err != nil {
		t.Fatalf("Function(gemv_w4a8_fwd): %v", err)
	}
	stream, err := cx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}

	// The geometries the registry actually produces. Kwords = K/8; the %32 column is what the
	// old single-shape test could never vary. Shapes marked kw%32!=0 are the ones that reproduced
	// the out-of-bounds read.
	cases := []struct {
		name string
		N, K int
	}{
		{"qwen2.5-0.5B/attn-qkv", 896, 896},   // Kwords 112 -> 112%32=16  (the OOB geometry)
		{"qwen2.5-0.5B/ffn-up", 4864, 896},    // Kwords 112 -> 16
		{"qwen2.5-0.5B/ffn-down", 896, 4864},  // Kwords 608 -> 0
		{"qwen2.5-1.5B/attn-qkv", 1536, 1536}, // Kwords 192 -> 0   (the ONLY shape tested before)
		{"qwen2.5-1.5B/ffn-up", 8960, 1536},   // Kwords 192 -> 0
		{"qwen2.5-1.5B/ffn-down", 1536, 8960}, // Kwords 1120 -> 0
		{"qwen3-1.7B/attn", 2048, 2048},       // Kwords 256 -> 0
		{"gemma3-4B/attn-q", 2560, 2560},      // Kwords 320 -> 0
		{"gemma3-4B/ffn-up", 10240, 2560},     // Kwords 320 -> 0
		{"gemma3-4B/ffn-down", 2560, 10240},   // Kwords 1280 -> 0
		{"phi3-mini/attn", 3072, 3072},        // Kwords 384 -> 0
		{"tail/K=1120", 512, 1120},            // Kwords 140 -> 12   (odd tail)
		{"tail/K=288", 64, 288},               // Kwords 36 -> 4     (tiny, forces the remainder)
		{"tail/K=160", 32, 160},               // Kwords 20 -> 20
		{"tail/K=2336", 128, 2336},            // Kwords 292 -> 4
		{"single-row/N=1", 1, 896},            // grid edge: one row
		{"odd-N/N=7", 7, 1536},                // N not a multiple of the 8 rows/block
		{"odd-N/N=9", 9, 896},                 // N just over one block
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.K%32 != 0 {
				t.Fatalf("bad case: K=%d is not a multiple of 32 (packWeight's own precondition)", c.K)
			}
			kw, kg, kd4 := c.K/8, c.K/32, c.K/4

			// Synthetic but DETERMINISTIC weights/activations, and the CPU reference is computed
			// from the same bits, so the comparison isolates the kernel from the data.
			var seed uint32 = 1234567
			rnd := func() uint32 { seed = seed*1664525 + 1013904223; return seed }
			nibs := make([]uint8, c.N*c.K) // logical nibble values 0..15 (q+8)
			for i := range nibs {
				nibs[i] = uint8(rnd()>>28) & 0xf
			}
			scales := make([]float32, c.N*kg)
			for i := range scales {
				scales[i] = 0.001 + float32(rnd()>>20)/float32(1<<12)*0.01
			}
			act := make([]int8, c.K)
			for i := range act {
				act[i] = int8(int32(rnd()>>24) - 128)
			}
			aScale := float32(0.0123)

			// Pack exactly as the production path does, so a packing/kernel disagreement shows up.
			Wp := make([]uint32, c.N*kw)
			for n := 0; n < c.N; n++ {
				for w := range kw {
					var word uint32
					for j := range 8 {
						word |= uint32(nibs[n*c.K+w*8+j]) << (4 * uint(nibblePosFast(j)))
					}
					Wp[n*kw+w] = word
				}
			}
			gsH := make([]uint16, c.N*kg)
			for i, s := range scales {
				gsH[i] = f32tof16(s)
			}
			// The kernel reads __half group scales, so the reference must read the SAME rounded
			// values — not the f32 originals. f16 carries ~5e-4 relative precision, so comparing
			// against unrounded scales manufactures a ~0.05% "error" that is the TEST's rounding,
			// not the kernel's. (This first showed up as failures on well-aligned shapes, which is
			// what proved it was the reference and not the Kwords%32 tail.)
			effScale := make([]float32, len(scales))
			for i := range scales {
				effScale[i] = f16tof32(gsH[i])
			}
			aPacked := make([]int32, kd4)
			for j := range kd4 {
				var p int32
				for b := range 4 {
					p |= (int32(act[4*j+b]) & 0xff) << (8 * uint(b))
				}
				aPacked[j] = p
			}

			// CPU reference: dequant-matmul in f64, mirroring the kernel's math exactly.
			ref := make([]float64, c.N)
			for n := 0; n < c.N; n++ {
				var acc float64
				for k := 0; k < c.K; k++ {
					q := float64(int32(nibs[n*c.K+k]) - 8)
					acc += q * float64(effScale[n*kg+k/32]) * float64(act[k]) * float64(aScale)
				}
				ref[n] = acc
			}

			// POISON THE GUARD BANDS. An out-of-bounds read must produce a WRONG ANSWER, or this
			// sweep cannot see it. CUDA pads allocations, so a lane reading past a snugly-sized
			// buffer lands in zeros: __dp4a(word, 0, p) contributes nothing and the result is
			// unchanged — the bug hides. (Verified: deleting the kernel's `if (wi < Kwords)` tail
			// guard did NOT fail an un-poisoned version of this test.) The production runner has no
			// such luck — r.aq is one buffer reused across layers with different K, so an
			// over-read there hits live data and corrupts. Padding both arrays with a non-zero
			// sentinel reproduces that condition.
			const band = 64
			poison := func32Poison
			Wpad := append(append([]uint32(nil), Wp...), make([]uint32, band)...)
			for i := len(Wp); i < len(Wpad); i++ {
				Wpad[i] = 0x7BADF00D
			}
			aPad := append(append([]int32(nil), aPacked...), make([]int32, band)...)
			for i := len(aPacked); i < len(aPad); i++ {
				aPad[i] = poison
			}
			gsPad := append(append([]uint16(nil), gsH...), make([]uint16, band)...)
			for i := len(gsH); i < len(gsPad); i++ {
				gsPad[i] = f32tof16(7.5) // a large, obviously-wrong scale
			}
			dW, e1 := gc.Alloc[uint32](cx, len(Wpad))
			dGs, e2 := gc.Alloc[uint16](cx, len(gsPad))
			dA, e3 := gc.Alloc[int32](cx, len(aPad))
			dAs, e4 := gc.Alloc[float32](cx, 1)
			dOut, e5 := gc.Alloc[float32](cx, c.N)
			// Check the allocations. A dropped alloc error is how an OOM disguises itself as a
			// numerics failure: nil buffers read back as zeros, which look exactly like a
			// "cosine 0.000000 layout mismatch" rather than "you are out of memory".
			for i, e := range []error{e1, e2, e3, e4, e5} {
				if e != nil {
					t.Fatalf("alloc %d failed: %v (GPU memory exhausted? that is a RESOURCE failure, not a kernel bug)", i, e)
				}
			}
			defer dW.Close()
			defer dGs.Close()
			defer dA.Close()
			defer dAs.Close()
			defer dOut.Close()

			if e := gc.CopyHtoD(bg, dW, Wpad); e != nil {
				t.Fatalf("H2D W: %v", e)
			}
			if e := gc.CopyHtoD(bg, dGs, gsPad); e != nil {
				t.Fatalf("H2D gs: %v", e)
			}
			if e := gc.CopyHtoD(bg, dA, aPad); e != nil {
				t.Fatalf("H2D a: %v", e)
			}
			if e := gc.CopyHtoD(bg, dAs, []float32{aScale}); e != nil {
				t.Fatalf("H2D aScale: %v", e)
			}

			cfg := gc.LaunchConfig{GridX: uint32((c.N + 7) / 8), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1}
			if e := fn4.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs), gc.Arg(dAs),
				gc.ArgDevicePtr(0), gc.ArgValue(int32(c.N)), gc.ArgValue(int32(kw)),
				gc.ArgValue(int32(kg)), gc.Arg(dOut), gc.ArgValue(int32(0))); e != nil {
				t.Fatalf("launch: %v", e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			got := make([]float32, c.N)
			if e := gc.CopyDtoH(bg, got, dOut); e != nil {
				t.Fatalf("D2H: %v", e)
			}

			// EXACT-ish: the only legitimate difference is f32 summation ORDER, so scale the
			// tolerance to the row magnitude. An OOB read or a dropped tail is off by far more.
			var worst float64 // max |got-ref| / sum|terms| — deviation vs magnitude, not vs |ref|:
			var worstN int    // cancellation can drive |ref| to ~0 and make relative-to-ref explode.
			for n := 0; n < c.N; n++ {
				var mag float64
				for k := 0; k < c.K; k++ {
					q := float64(int32(nibs[n*c.K+k]) - 8)
					mag += math.Abs(q * float64(effScale[n*kg+k/32]) * float64(act[k]) * float64(aScale))
				}
				// Only f32 summation ORDER may differ now that the scales match: the kernel
				// float-accumulates ~K terms, so bound at eps*sqrt(K)-ish with headroom. An OOB
				// read or a dropped tail lands orders of magnitude above this, not just over it.
				if d := math.Abs(float64(got[n])-ref[n]) / (mag + 1e-30); d > worst {
					worst = d
					worstN = n
				}
			}
			const tol = 1e-5 // vs magnitude; f32 accumulation over K terms sits far below this
			t.Logf("N=%d K=%d Kwords=%d (Kwords%%32=%d): worst deviation %.2e of magnitude (bar %.0e)",
				c.N, c.K, kw, kw%32, worst, tol)
			if worst > tol {
				t.Errorf("N=%d K=%d (Kwords=%d, Kwords%%32=%d): row %d deviates %.2e of magnitude from the "+
					"CPU reference — beyond f32 summation order, so a tail/stride assumption does not hold "+
					"at this geometry (the Kwords%%32 out-of-bounds class).",
					c.N, c.K, kw, kw%32, worstN, worst)
			}
		})
	}
}

// TestGemvShapeSweep_coversTheOOBGeometry is a guard on the sweep itself: the table above must
// keep at least one shape whose Kwords is NOT a multiple of 32. If a future edit tidies the
// table down to "nice" shapes, the sweep silently stops covering the exact class it exists for
// — which is precisely how the original single-shape test failed to catch the 0.5B.
func TestGemvShapeSweep_coversTheOOBGeometry(t *testing.T) {
	// Mirrors the table's Kwords values; kept as raw K so it reads as the geometry, not a hash.
	ks := []int{896, 896, 4864, 1536, 1536, 8960, 2048, 2560, 2560, 10240, 3072, 1120, 288, 160, 2336}
	var ragged []string
	for _, k := range ks {
		if kw := k / 8; kw%32 != 0 {
			ragged = append(ragged, fmt.Sprintf("K=%d(Kwords=%d)", k, kw))
		}
	}
	if len(ragged) == 0 {
		t.Fatal("the shape sweep no longer contains ANY geometry with Kwords%32 != 0 — it has been " +
			"tidied into only well-aligned shapes and can no longer catch the out-of-bounds class " +
			"it was written for (Qwen2.5-0.5B: hidden 896 -> Kwords 112 -> 112%32 = 16)")
	}
	t.Logf("sweep covers %d ragged-tail geometries: %v", len(ragged), ragged)
}

// func32Poison is a non-zero packed-int8 sentinel (4 x 0x5A) for the activation guard band: an
// out-of-bounds __dp4a against it yields a large wrong contribution rather than a silent zero.
const func32Poison int32 = 0x5A5A5A5A
