//go:build cuda

package cuda

import (
	"context"
	_ "embed"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

//go:embed testdata/gemv_w4a8.ptx
var gemvW4A8PTX []byte

// f32tof16 now lives in kernels.go (shared with the production packer).

func f16tof32(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32((h >> 10) & 0x1f)
	m := uint32(h & 0x3ff)
	if e == 0 {
		return math.Float32frombits(s)
	}
	return math.Float32frombits(s | (e-15+127)<<23 | m<<13)
}

// TestGemvW4A8 validates the int4 W4A8 GEMV (spec §3 packing: group=32, nibble=q+8, f16 scale)
// against an exact CPU reference, then measures int4 bandwidth — the hot path for the matching-
// quant (int4) apples-to-apples vs Ollama-on-q4_k_m. Run: CGO_ENABLED=0 go test -tags cuda -run W4A8 -v
func TestGemvW4A8(t *testing.T) {
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

	const N, K = 8960, 1536 // FFN shape; K mult of 32
	const grp = 32
	Kwords := K / 8    // nibbles packed 8/uint32
	Kgroups := K / grp // one f16 scale per 32 nibbles
	Kdiv4 := K / 4     // activation int8 packed 4/int

	// synthetic: nibbles q+8 in [1,15]; f16 group scales; int8 activation.
	Wp := make([]uint32, N*Kwords)
	gs := make([]uint16, N*Kgroups)
	a := make([]int32, Kdiv4)
	var seed uint32 = 99
	nib := func() uint32 { seed = seed*1664525 + 1013904223; return (seed >> 28) & 0xf } // 0..15
	for i := range Wp {
		var w uint32
		for j := 0; j < 8; j++ {
			w |= nib() << (4 * j)
		}
		Wp[i] = w
	}
	for i := range gs {
		gs[i] = f32tof16(0.02 + 0.001*float32(i%7))
	}
	i8 := func() int32 { seed = seed*1664525 + 1013904223; return int32(int8(seed >> 24)) }
	for i := range a {
		var p int32
		for b := 0; b < 4; b++ {
			p |= (i8() & 0xff) << (8 * b)
		}
		a[i] = p
	}
	const aScale = float32(0.03)

	// CPU ref
	ref := make([]float32, N)
	for n := 0; n < N; n++ {
		var facc float64
		for g := 0; g < Kgroups; g++ {
			var iacc int64
			for w := 0; w < 4; w++ {
				word := Wp[n*Kwords+4*g+w]
				for j := 0; j < 8; j++ {
					nb := int64((word>>(4*j))&0xf) - 8
					ai := a[8*g+2*w+j/4]
					av := int64(int8(ai >> (8 * (j % 4))))
					iacc += nb * av
				}
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
	mod, _ := ctx.LoadModule(gemvW4A8PTX)
	fn, err := mod.Function("gemv_w4a8")
	if err != nil {
		t.Fatalf("Function: %v", err)
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
	var d, ng, nr float64
	for n := 0; n < N; n++ {
		d += float64(got[n]) * float64(ref[n])
		ng += float64(got[n]) * float64(got[n])
		nr += float64(ref[n]) * float64(ref[n])
	}
	cos := d / (math.Sqrt(ng)*math.Sqrt(nr) + 1e-30)
	if cos < 0.9999 {
		t.Fatalf("W4A8 GEMV cosine %.6f vs CPU ref — packing mismatch", cos)
	}
	// bandwidth: int4 = N*K/2 weight bytes + N*Kgroups*2 scale bytes
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
	t.Logf("W4A8 GEMV [%d×%d] correct (cosine %.6f); %.1f us | %.2f MB | %.0f GB/s = %.0f%% peak",
		N, K, cos, bestUs, float64(wbytes)/1e6, gbps, gbps/448*100)
}
