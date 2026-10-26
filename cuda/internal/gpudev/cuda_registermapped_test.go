//go:build linux

package gpudev

import "testing"

// TestCUDA_registerMappedHostWeight_zeroCopy is TestCUDA_mappedHostWeight_zeroCopy's twin for
// RegisterMappedHostBuffer: the weight bytes are already resident in an ordinary Go slice (as they
// would be after decoding a checkpoint) and get PINNED IN PLACE rather than allocated-then-copied.
// Same bar: a real W4A8 GEMV must read the pinned-in-place bytes bit-identically to a device buffer.
func TestCUDA_registerMappedHostWeight_zeroCopy(t *testing.T) {
	d, q, _ := setup(t, "vadd")
	g, err := d.NewQuantGEMV()
	if err != nil {
		t.Fatalf("NewQuantGEMV: %v", err)
	}
	const N, Kwords = 64, 32
	Kgroups := (Kwords + 3) / 4
	rng := lcg(13)
	W := make([]uint32, N*Kwords) // the caller's own slice — already fully populated, as production has it
	for i := range W {
		W[i] = uint32(rng.word())
	}
	a := make([]int32, 2*Kwords)
	for i := range a {
		a[i] = rng.word()
	}
	pow2 := []float32{0.03125, 0.0625, 0.125, 0.25}
	gs16 := make([]uint16, N*Kgroups)
	for i := range gs16 {
		gs16[i] = f32tof16(pow2[i%len(pow2)])
	}
	bias := make([]float32, N)
	const aScale = float32(0.02)

	dA := NewBufferOf(d, a)
	dGS := NewBufferOf(d, gs16)
	dAS := NewBufferOf(d, []float32{aScale})
	dBias := NewBufferOf(d, bias)

	run := func(wArg Buffer) []float32 {
		dst := NewBufferLenOf[float32](d, N)
		cfg := GEMVGrid(N, GEMVWarpsPerBlock)
		if err := q.Launch(g.W4A8, cfg,
			Arg(wArg), Arg(dA), Arg(dGS), Arg(dAS), Arg(dBias),
			ArgValue(int32(N)), ArgValue(int32(Kwords)), ArgValue(int32(Kgroups)),
			Arg(dst), ArgValue(int32(0))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		out := make([]float32, N)
		if err := Download(dst, out); err != nil {
			t.Fatalf("Download: %v", err)
		}
		d.ReleaseBuf(dst)
		return out
	}

	dW := NewBufferOf(d, W) // baseline: the same bytes in a device buffer
	gotDev := run(dW)

	mb, err := d.RegisterMappedHostBuffer(asBytes(W)) // zero-copy: pin W's own bytes in place
	if err != nil {
		t.Fatalf("RegisterMappedHostBuffer: %v", err)
	}
	defer func() { _ = mb.Close() }()
	gotMapped := run(mb.Buffer())

	for n := range N {
		if gotDev[n] != gotMapped[n] {
			t.Fatalf("n=%d: device %.6f != registered-mapped-host %.6f — RegisterMappedHostBuffer read WRONG",
				n, gotDev[n], gotMapped[n])
		}
	}
	t.Logf("W4A8 GEMV over %d×%d int4 weight: device-buffer == registered-in-place-host-buffer, bit-identical", N, Kwords)
}
