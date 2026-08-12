//go:build cuda

package cuda

import (
	"context"
	_ "embed"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// gemvPTX is the W8A8 GEMV hot-path kernel (NVRTC, compute_75, __dp4a int8 dot).
//
//go:embed testdata/gemv_w8a8.ptx
var gemvPTX []byte

// TestGemvW8A8Bandwidth is the spike's decisive-proxy experiment: decode is
// weight-streaming-bound, and WebGPU sits ~37% below the 2070's bandwidth ceiling
// because of its dispatch/glue wall. This measures what a *hand CUDA quant GEMV*
// achieves in isolation — the ceiling a megakernel could approach once the glue is
// gone. Correctness: exact int accumulation vs a CPU reference (the packing must
// match); Bandwidth: weight bytes / CUDA-event kernel time, as % of the ~448 GB/s
// peak. A high % here (≫ WebGPU's 37%) is the "the kernel is competent, the lane is
// real" signal; a low % is an early NO-GO. Run: CGO_ENABLED=0 go test -tags cuda -run Bandwidth -v
func TestGemvW8A8Bandwidth(t *testing.T) {
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

	// qwen2.5-1.5b FFN up/gate projection shape: N=8960 rows, K=1536. int8 weights.
	const N, K = 8960, 1536
	const Kdiv4 = K / 4
	wBytes := int64(N) * int64(K) // 1 byte/int8 weight — the dominant read

	// synthetic packed int8: W[N,Kdiv4] and a[Kdiv4] as int32 (4 int8 each), wScale[N].
	W := make([]int32, N*Kdiv4)
	a := make([]int32, Kdiv4)
	wScale := make([]float32, N)
	var seed uint32 = 12345
	rnd := func() int32 { seed = seed*1664525 + 1013904223; return int32(int8(seed >> 24)) }
	pack := func() int32 { return (rnd()&0xff)<<0 | (rnd()&0xff)<<8 | (rnd()&0xff)<<16 | (rnd()&0xff)<<24 }
	for i := range W {
		W[i] = pack()
	}
	for i := range a {
		a[i] = pack()
	}
	for i := range wScale {
		wScale[i] = 0.01
	}
	const aScale = float32(0.02)

	// CPU reference (exact int dp4 accumulation, then scale).
	dp4 := func(x, y int32) int32 {
		var s int32
		for b := range 4 {
			s += int32(int8(x>>(8*b))) * int32(int8(y>>(8*b)))
		}
		return s
	}
	ref := make([]float32, N)
	for n := range N {
		var acc int32
		for k := range Kdiv4 {
			acc += dp4(W[n*Kdiv4+k], a[k])
		}
		ref[n] = float32(acc) * aScale * wScale[n]
	}

	// device buffers
	dW := mustAlloc[int32](t, ctx, len(W))
	dA := mustAlloc[int32](t, ctx, len(a))
	dS := mustAlloc[float32](t, ctx, N)
	dOut := mustAlloc[float32](t, ctx, N)
	defer dW.Close()
	defer dA.Close()
	defer dS.Close()
	defer dOut.Close()
	if err := gc.CopyHtoD(bg, dW, W); err != nil {
		t.Fatalf("H2D W: %v", err)
	}
	_ = gc.CopyHtoD(bg, dA, a)
	_ = gc.CopyHtoD(bg, dS, wScale)

	mod, err := ctx.LoadModule(gemvPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("gemv_w8a8")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}
	stream := mustStream(t, ctx)

	// 8 warps/block → 8 output rows/block; grid covers N rows.
	const warpsPerBlock = 8
	cfg := gc.LaunchConfig{
		GridX: uint32((N + warpsPerBlock - 1) / warpsPerBlock), GridY: 1, GridZ: 1,
		BlockX: warpsPerBlock * 32, BlockY: 1, BlockZ: 1,
	}
	launch := func() error {
		return fn.LaunchOn(bg, stream, cfg,
			gc.Arg(dW), gc.Arg(dA), gc.Arg(dS), gc.ArgValue(aScale),
			gc.ArgValue(int32(N)), gc.ArgValue(int32(Kdiv4)), gc.Arg(dOut))
	}

	// warm + correctness
	for range 3 {
		if err := launch(); err != nil {
			t.Fatalf("launch: %v", err)
		}
	}
	_ = stream.Synchronize(bg)
	got := make([]float32, N)
	_ = gc.CopyDtoH(bg, got, dOut)
	var dot, ng, nr float64
	maxAbs := 0.0
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		ng += float64(got[n]) * float64(got[n])
		nr += float64(ref[n]) * float64(ref[n])
		if d := math.Abs(float64(got[n] - ref[n])); d > maxAbs {
			maxAbs = d
		}
	}
	cos := dot / (math.Sqrt(ng)*math.Sqrt(nr) + 1e-30)
	if cos < 0.9999 {
		t.Fatalf("GEMV incorrect vs CPU ref: cosine %.6f, maxAbs %.4g (packing mismatch corrupts the result)", cos, maxAbs)
	}

	// bandwidth: best of ≥5 event-timed runs
	bestUs := 1e18
	for range 8 {
		start, _ := ctx.NewEvent()
		done, _ := ctx.NewEvent()
		_ = start.Record(stream)
		const iters = 50
		for range iters {
			_ = launch()
		}
		_ = done.Record(stream)
		_ = stream.Synchronize(bg)
		el, _ := start.Elapsed(done)
		if us := float64(el.Microseconds()) / iters; us < bestUs {
			bestUs = us
		}
	}
	gbps := float64(wBytes) / (bestUs * 1e-6) / 1e9
	const peak = 448.0 // RTX 2070 SUPER GDDR6 ~448 GB/s
	t.Logf("W8A8 GEMV [%d×%d] correct (cosine %.6f vs CPU ref):", N, K, cos)
	t.Logf("  kernel %.1f us | weight read %.2f MB | %.0f GB/s = %.0f%% of ~%.0f GB/s peak",
		bestUs, float64(wBytes)/1e6, gbps, gbps/peak*100, peak)
	t.Logf("  (WebGPU decode sits ~37%% of peak — a hand quant GEMV clearing that is the 'lane is real' signal)")
}
