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

// TestGluQuant_scaleIsExactMaxabs is the same invariant for glu_quant, and it can be a PERMANENT
// gate rather than a one-shot instrument for one reason: glu_quant writes its pre-quantization
// values to dscratch in GLOBAL memory. So the oracle needs no knowledge of the activation, no
// replication of silu/gelu-tanh, and no dependence on a non-IEEE intrinsic — read back what the
// kernel itself produced and take its maximum in a different order.
//
// (rmsnorm_quant gets no equivalent: its normed[] never leaves shared memory, and CUDA's rsqrtf is
// not IEEE-exact, so its bit-identity was proven by before/after capture at change time instead.)
func TestGluQuant_scaleIsExactMaxabs(t *testing.T) {
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
	fn, err := mod.Function("glu_quant")
	if err != nil {
		t.Fatalf("glu_quant: %v", err)
	}
	stream := mustStream(t, ctx)
	// Both activations: ACT_GELU_TANH(0) is gemma's, ACT_SILU(1) everyone else's. The reduction is
	// identical either way, but a maxabs bug that only shows on one sign distribution would not be.
	for _, act := range []int32{0, 1} {
		for _, inter := range []int{64, 255, 4096} {
			rng := rand.New(rand.NewSource(int64(inter) + int64(act)*101))
			gu := make([]float32, 2*inter)
			for i := range gu {
				gu[i] = float32(rng.NormFloat64() * 2)
			}
			dgu := mustAlloc[float32](t, ctx, 2*inter)
			dscr := mustAlloc[float32](t, ctx, inter)
			dq := mustAlloc[int32](t, ctx, inter/4+1)
			dsc := mustAlloc[float32](t, ctx, 1)
			if err := gc.CopyHtoD(bg, dgu, gu); err != nil {
				t.Fatalf("CopyHtoD: %v", err)
			}
			cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4}
			if e := fn.LaunchOn(bg, stream, cfg, gc.Arg(dgu), gc.Arg(dgu), gc.ArgValue(int32(0)),
				gc.ArgValue(int32(inter)), gc.ArgValue(int32(inter)), gc.ArgValue(act),
				gc.Arg(dq), gc.Arg(dsc), gc.Arg(dscr)); e != nil {
				t.Fatalf("act=%d inter=%d launch: %v", act, inter, e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			scale := make([]float32, 1)
			scratch := make([]float32, inter)
			if e := gc.CopyDtoH(bg, scale, dsc); e != nil {
				t.Fatalf("CopyDtoH scale: %v", e)
			}
			if e := gc.CopyDtoH(bg, scratch, dscr); e != nil {
				t.Fatalf("CopyDtoH scratch: %v", e)
			}
			want := float32(0)
			for _, v := range scratch {
				if a := float32(math.Abs(float64(v))); a > want {
					want = a
				}
			}
			if got, wantScale := scale[0], want/127.0; got != wantScale {
				t.Errorf("act=%d inter=%d: scale %v, want EXACTLY %v — the kernel's OWN dscratch has "+
					"maxabs %v, and max is order-independent, so any correct reduction must agree "+
					"bit-for-bit", act, inter, got, wantScale, want)
			}
		}
	}
}
