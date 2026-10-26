//go:build linux

package gpudev

import "testing"

// TestCUDA_mappedHostWeight_zeroCopy proves a kernel reads MAPPED PINNED HOST memory identically to a
// device buffer — the correctness foundation for host↔VRAM MoE expert streaming (an int4 model whose
// experts exceed VRAM keeps the always-resident core in device memory and the experts host-mapped,
// and the GEMV streams them over PCIe). Runs the real W4A8 GEMV twice on IDENTICAL bytes: once with
// the weight in a device Buffer, once in a MappedHostBuffer. Bit-identical is the bar — the kernel,
// dispatch, and reduction order are unchanged; only where the weight lives differs. A failure means
// the UVA host-pointer-as-device-pointer assumption does not hold on this hardware.
func TestCUDA_mappedHostWeight_zeroCopy(t *testing.T) {
	d, q, _ := setup(t, "vadd")
	g, err := d.NewQuantGEMV()
	if err != nil {
		t.Fatalf("NewQuantGEMV: %v", err)
	}
	const N, Kwords = 64, 32
	Kgroups := (Kwords + 3) / 4
	rng := lcg(7)
	W := make([]uint32, N*Kwords)
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

	dW := NewBufferOf(d, W) // baseline: weight in a device buffer
	gotDev := run(dW)

	mb, err := d.NewMappedHostBuffer(len(W) * 4) // zero-copy: the SAME bytes in mapped pinned host memory
	if err != nil {
		t.Fatalf("NewMappedHostBuffer: %v", err)
	}
	defer func() { _ = mb.Close() }()
	copy(mb.Bytes(), asBytes(W))
	gotMapped := run(mb.Buffer())

	for n := range N {
		if gotDev[n] != gotMapped[n] {
			t.Fatalf("n=%d: device %.6f != mapped-host %.6f — the kernel read the mapped host weight WRONG (UVA/zero-copy broken)",
				n, gotDev[n], gotMapped[n])
		}
	}
	t.Logf("W4A8 GEMV over %d×%d int4 weight: device-buffer == mapped-host-buffer, bit-identical — zero-copy host read verified", N, Kwords)
}
