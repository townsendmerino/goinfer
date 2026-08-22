//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestQuantVec_scaleIsExactMaxabs pins the ONE property that makes quant_vec's reduction safe to
// restructure: its scale is max|x|/127, and max is EXACT and ORDER-INDEPENDENT. Any correct
// reduction tree — the __syncthreads() ladder, a warp shuffle, a single thread in a loop — must
// produce the identical float, bit for bit. A sum reduction would have no such property, which is
// why this test can exist for this kernel and not for its neighbours.
//
// So this is not a tolerance check. It asserts EQUALITY against a CPU maxabs computed in a
// different order, which is exactly the invariant a shuffle rewrite must not break: a wrong lane
// mask, a missed partial, or an out-of-range lane seeded with the wrong identity all show up here
// as a changed scale, and would otherwise show up as a silent quantization drift on every token.
func TestQuantVec_scaleIsExactMaxabs(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("LoadModule(gluePTX): %v", err)
	}
	fn, err := mod.Function("quant_vec")
	if err != nil {
		t.Fatalf("quant_vec: %v", err)
	}
	stream := mustStream(t, ctx)

	// Sizes chosen to exercise the reduction's edges, not just the happy path:
	//   255/257  N not a multiple of the 256-thread block — the strided load loop's tail
	//   2048..5120 the production activation widths
	//   8        far fewer elements than threads, so most lanes hold only the identity
	for _, n := range []int{8, 255, 256, 257, 2048, 4096, 5120} {
		rng := rand.New(rand.NewSource(int64(n)))
		host := make([]float32, n)
		want := float32(0)
		for i := range host {
			// Deliberately asymmetric and wide-ranged: a maxabs bug that only reads positives, or
			// only the first warp, must not survive.
			v := float32(rng.NormFloat64() * float64(1+i%7))
			if i%13 == 0 {
				v = -v * 3
			}
			host[i] = v
			if a := float32(math.Abs(float64(v))); a > want {
				want = a
			}
		}
		dx := mustAlloc[float32](t, ctx, n)
		dq := mustAlloc[int32](t, ctx, n/4+1)
		dsc := mustAlloc[float32](t, ctx, 1)
		if err := gc.CopyHtoD(bg, dx, host); err != nil {
			t.Fatalf("n=%d CopyHtoD: %v", n, err)
		}
		cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4}
		if err := fn.LaunchOn(bg, stream, cfg, gc.Arg(dx), gc.ArgValue(int32(n)), gc.Arg(dq), gc.Arg(dsc)); err != nil {
			t.Fatalf("n=%d launch: %v", n, err)
		}
		if err := stream.Synchronize(bg); err != nil {
			t.Fatalf("n=%d sync: %v", n, err)
		}
		got := make([]float32, 1)
		if err := gc.CopyDtoH(bg, got, dsc); err != nil {
			t.Fatalf("n=%d CopyDtoH: %v", n, err)
		}
		if wantScale := want / 127.0; got[0] != wantScale {
			t.Errorf("n=%d: scale %v, want EXACTLY %v (maxabs %v / 127) — max is order-independent, "+
				"so any correct reduction tree must produce this float bit-for-bit; a difference means "+
				"the reduction dropped or double-counted a lane", n, got[0], wantScale, want)
		}
	}
}
