//go:build linux

package gpudev

import (
	_ "embed"
	"testing"
	"time"
)

// roofline.cu's PTX is embedded HERE, in a test file, because these kernels are a
// measuring instrument rather than something a consumer runs. TestKernelFMALint_
// coversEmbeddedPTX treats a test-only embed as not requiring a bit-identity
// declaration; roofline.cu carries one anyway, since "no float arithmetic at all" is
// worth stating rather than leaving a reader to check.
//
//go:embed testdata/roofline.ptx
var rooflinePTX []byte

// bestOf runs f n times and returns the SHORTEST run. A ceiling is a maximum: nothing
// another process does can make this device faster than it is, so contention can only
// depress a sample. Averaging would fold in whatever else was running.
func bestOf(n int, f func() error) (time.Duration, error) {
	best := time.Duration(1) << 62
	for range n {
		t0 := time.Now()
		if err := f(); err != nil {
			return 0, err
		}
		if d := time.Since(t0); d < best {
			best = d
		}
	}
	return best, nil
}

// TestDeviceCeilings measures the three roofs and logs them, and — this is the part
// that makes it a test rather than a benchmark — checks that the probes are measuring
// what they claim.
//
// THE SELF-CHECK IS THE POINT. A probe that reports a number cannot be trusted just
// because the number looks reasonable; that is exactly how the amd64 GEMM benchmark
// reported "~50% of peak" against another machine's constant for months. So:
//
//   - dp4a_peak and imad_peak run the SAME loop shape and the same chain count, and
//     differ only in whether the instruction retires 1 or 4 int8 MACs. Their
//     instruction rates must therefore be close. If dp4a's were far lower, the probe
//     would be measuring loop overhead or a dependency stall rather than issue rate,
//     and the 4x MAC advantage it reports would be fiction.
//   - stream_read must beat both probes' memory traffic by orders of magnitude (it is
//     the only one that touches DRAM in its loop) and must not exceed any plausible
//     bus. A figure above ~4 TB/s would mean the buffer fit in cache.
//
// The bounds are deliberately loose. This runs on whatever GPU is present, and the
// assertion worth making portably is "this probe is coherent", not "this device is
// fast". Exact figures for nvidia-rtx2070s are in docs/internal/roofline-2026-08.md.
func TestDeviceCeilings(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(dev.ReleaseObjects)
	lib, err := dev.CompileLibrary(rooflinePTX)
	if err != nil {
		t.Fatalf("CompileLibrary(roofline.ptx): %v", err)
	}
	q := dev.NewCommandQueue()

	// --- DRAM streaming read -------------------------------------------------
	// 512 MiB is past any current GPU's last-level cache, which is what makes this
	// a DRAM measurement rather than a cache one.
	const streamBytes = 512 << 20
	var streamGBs float64
	{
		p, err := dev.NewComputePipeline(lib, "stream_read")
		if err != nil {
			t.Fatalf("stream_read: %v", err)
		}
		threads := 40 * 32 * 256 // enough waves that no SM runs dry
		src := dev.MustBuf(streamBytes, streamBytes/4, "roofline-src")
		out := NewBufferLenOf[int32](dev, threads)
		n4 := NewBufferOf(dev, []uint32{uint32(streamBytes / 16)})
		tot := NewBufferOf(dev, []uint32{uint32(threads)})
		d, err := bestOf(5, func() error { return q.Run1D(p, threads, 256, src, out, n4, tot) })
		if err != nil {
			t.Fatalf("stream_read run: %v", err)
		}
		streamGBs = float64(streamBytes) / d.Seconds() / 1e9
		t.Logf("streaming read : %7.1f GB/s  (%d MiB in %v)", streamGBs, streamBytes>>20, d)
	}

	// --- int32 multiply-add and __dp4a issue rates ---------------------------
	rate := func(name string) float64 {
		p, err := dev.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		const threads, iters, unroll = 40 * 1024, 20000, 16
		seed := NewBufferLenOf[int32](dev, 256)
		out := NewBufferLenOf[int32](dev, threads)
		it := NewBufferOf(dev, []uint32{iters})
		tot := NewBufferOf(dev, []uint32{threads})
		d, err := bestOf(5, func() error { return q.Run1D(p, threads, 256, seed, out, it, tot) })
		if err != nil {
			t.Fatalf("%s run: %v", name, err)
		}
		return float64(threads) * float64(iters) * float64(unroll) / d.Seconds() / 1e9
	}
	imad, dp4a := rate("imad_peak"), rate("dp4a_peak")
	t.Logf("int32 mul-add  : %7.1f G instr/s -> %8.1f GMAC/s (int8)", imad, imad)
	t.Logf("__dp4a         : %7.1f G instr/s -> %8.1f GMAC/s (int8)", dp4a, dp4a*4)

	// Coherence, not performance. Same loop, same chain count, one instruction each.
	if r := dp4a / imad; r < 0.5 || r > 2 {
		t.Errorf("dp4a/imad instruction rate ratio is %.2f, want ~1 (%.1f vs %.1f G instr/s)\n"+
			"  The two probes run the SAME loop with the SAME chain count and differ only in "+
			"which instruction they issue, so a large gap means one of them is measuring "+
			"something other than issue rate — loop overhead or a dependency stall. Treat the "+
			"4x int8 MAC advantage dp4a reports as unproven until this is ~1.", r, dp4a, imad)
	}
	if streamGBs <= 0 || streamGBs > 4000 {
		t.Errorf("streaming read reported %.1f GB/s, which is not a plausible DRAM figure\n"+
			"  Above ~4 TB/s means the %d MiB buffer was served from cache and this is not a "+
			"DRAM measurement at all.", streamGBs, streamBytes>>20)
	}
}
