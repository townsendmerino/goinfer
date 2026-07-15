//go:build cuda

package cuda

import (
	"context"
	_ "embed"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

//go:embed testdata/gemv_w4a8_fast.ptx
var gemvW4A8FastPTX []byte

//go:embed testdata/gemv_w4a8_coal.ptx
var gemvW4A8CoalPTX []byte

//go:embed testdata/gemv_w4a8_v4.ptx
var gemvW4A8V4PTX []byte

//go:embed testdata/gemv_w4a8_coal2.ptx
var gemvW4A8Coal2PTX []byte

//go:embed testdata/gemv_w4a8_coal3.ptx
var gemvW4A8Coal3PTX []byte

// nibblePosFast + permuteFast now live in kernels.go (shared with the production backend).

// runW4A8Variant packs real-shaped int4 weights in the fast nibble-permuted layout,
// validates cosine 1.0 vs a logical CPU reference (packing-independent), and reports
// % of ~448 GB/s peak — the number that says whether int4 became memory-bound.
func runW4A8Variant(t *testing.T, ptx []byte, fnName, label string) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	ctx, _ := dev.Primary()
	defer ctx.Close()
	bg := context.Background()

	const N, K = 8960, 1536
	const grp = 32
	Kwords, Kgroups, Kdiv4 := K/8, K/grp, K/4

	qv := make([]uint8, N*K)
	gs := make([]uint16, N*Kgroups)
	a := make([]int32, Kdiv4)
	aB := make([]int8, K)
	var seed uint32 = 99
	nib := func() uint8 { seed = seed*1664525 + 1013904223; return uint8((seed >> 28) & 0xf) }
	for i := range qv {
		qv[i] = nib()
	}
	for i := range gs {
		gs[i] = f32tof16(0.02 + 0.001*float32(i%7))
	}
	for k := 0; k < K; k++ {
		seed = seed*1664525 + 1013904223
		aB[k] = int8(seed >> 24)
	}
	for j := 0; j < Kdiv4; j++ {
		var p int32
		for b := 0; b < 4; b++ {
			p |= (int32(aB[4*j+b]) & 0xff) << (8 * b)
		}
		a[j] = p
	}
	const aScale = float32(0.03)

	Wp := make([]uint32, N*Kwords)
	for n := 0; n < N; n++ {
		for m := 0; m < Kwords; m++ {
			var word uint32
			for i := 0; i < 8; i++ {
				word |= uint32(qv[n*K+8*m+i]) << (4 * nibblePosFast(i))
			}
			Wp[n*Kwords+m] = word
		}
	}
	ref := make([]float32, N)
	for n := 0; n < N; n++ {
		var facc float64
		for g := 0; g < Kgroups; g++ {
			var iacc int64
			for k := g * grp; k < (g+1)*grp; k++ {
				iacc += (int64(qv[n*K+k]) - 8) * int64(aB[k])
			}
			facc += float64(iacc) * float64(f16tof32(gs[n*Kgroups+g]))
		}
		ref[n] = float32(facc) * aScale
	}

	dW, _ := gc.Alloc[uint32](ctx, len(Wp))
	dGs, _ := gc.Alloc[uint16](ctx, len(gs))
	dA, _ := gc.Alloc[int32](ctx, len(a))
	dOut, _ := gc.Alloc[float32](ctx, N)
	_ = gc.CopyHtoD(bg, dW, Wp)
	_ = gc.CopyHtoD(bg, dGs, gs)
	_ = gc.CopyHtoD(bg, dA, a)
	mod, _ := ctx.LoadModule(ptx)
	fn, err := mod.Function(fnName)
	if err != nil {
		t.Fatalf("Function %s: %v", fnName, err)
	}
	stream, _ := ctx.NewStream()
	const wpb = 8
	cfg := gc.LaunchConfig{GridX: uint32((N + wpb - 1) / wpb), GridY: 1, GridZ: 1, BlockX: wpb * 32, BlockY: 1, BlockZ: 1}
	launch := func() {
		_ = fn.LaunchOn(bg, stream, cfg, gc.Arg(dW), gc.Arg(dA), gc.Arg(dGs),
			gc.ArgValue(aScale), gc.ArgValue(int32(N)), gc.ArgValue(int32(Kwords)), gc.ArgValue(int32(Kgroups)), gc.Arg(dOut))
	}
	for i := 0; i < 3; i++ {
		launch()
	}
	_ = stream.Synchronize(bg)
	got := make([]float32, N)
	_ = gc.CopyDtoH(bg, got, dOut)
	var d, ng, nr, maxAbs float64
	for n := 0; n < N; n++ {
		d += float64(got[n]) * float64(ref[n])
		ng += float64(got[n]) * float64(got[n])
		nr += float64(ref[n]) * float64(ref[n])
		if x := math.Abs(float64(got[n] - ref[n])); x > maxAbs {
			maxAbs = x
		}
	}
	cos := d / (math.Sqrt(ng)*math.Sqrt(nr) + 1e-30)
	if cos < 0.9999 {
		t.Fatalf("%s cosine %.6f maxAbs %.4g — layout/unpack mismatch", label, cos, maxAbs)
	}
	wbytes := int64(N)*int64(K)/2 + int64(N)*int64(Kgroups)*2
	bestUs := 1e18
	for r := 0; r < 8; r++ {
		s, _ := ctx.NewEvent()
		e, _ := ctx.NewEvent()
		_ = s.Record(stream)
		const it = 50
		for i := 0; i < it; i++ {
			launch()
		}
		_ = e.Record(stream)
		_ = stream.Synchronize(bg)
		el, _ := s.Elapsed(e)
		if us := float64(el.Microseconds()) / it; us < bestUs {
			bestUs = us
		}
	}
	gbps := float64(wbytes) / (bestUs * 1e-6) / 1e9
	t.Logf("%s [%d×%d] cosine %.6f; %.1f us | %.0f GB/s = %.0f%% peak (naive 43%%)",
		label, N, K, cos, bestUs, gbps, gbps/448*100)
}

// TestGemvW4A8Fast — even/odd + __vsub4 unpack, same strided access (isolates the ALU).
func TestGemvW4A8Fast(t *testing.T) {
	runW4A8Variant(t, gemvW4A8FastPTX, "gemv_w4a8_fast", "W4A8-FAST")
}

// TestGemvW4A8Coal — coalesced consecutive-word reads + 4-lane segmented reduction.
func TestGemvW4A8Coal(t *testing.T) {
	runW4A8Variant(t, gemvW4A8CoalPTX, "gemv_w4a8_coal", "W4A8-COAL")
}

// TestGemvW4A8V4 — uint4 group load (coalesced AND group-aligned, no segmented reduction).
func TestGemvW4A8V4(t *testing.T) { runW4A8Variant(t, gemvW4A8V4PTX, "gemv_w4a8_v4", "W4A8-V4") }

// TestGemvW4A8Coal2 — coalesced reads, scale-per-word float accumulate (drops the 2 shfl/word).
func TestGemvW4A8Coal2(t *testing.T) {
	runW4A8Variant(t, gemvW4A8Coal2PTX, "gemv_w4a8_coal2", "W4A8-COAL2")
}

// TestGemvW4A8Coal3 — COAL2 + 2x ILP unroll (two loads in flight per lane, 32-remainder).
func TestGemvW4A8Coal3(t *testing.T) {
	runW4A8Variant(t, gemvW4A8Coal3PTX, "gemv_w4a8_coal3", "W4A8-COAL3")
}

//go:embed testdata/gemv_w4a8_coal4.ptx
var gemvW4A8Coal4PTX []byte

// TestGemvW4A8Coal4 — 4x ILP unroll (four loads in flight per lane, 32-remainder).
func TestGemvW4A8Coal4(t *testing.T) {
	runW4A8Variant(t, gemvW4A8Coal4PTX, "gemv_w4a8_coal4", "W4A8-COAL4")
}
