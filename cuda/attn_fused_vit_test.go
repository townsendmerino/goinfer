//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestAttnVit_logicAndBMIdentity is gates 1 and 2 of docs/measurements/vision-tower-attn-PREREGISTERED.md for the R8 phase-A
// kernel (attn_fused_vit.cu, hd 72 padded to 80, non-causal):
//  1. worst per-row cosine >= 0.99999 against an f64 reference computed on f16-ROUNDED operands and sequenced the way the kernel is
//     (BN=64 key tiles in order, running max/denominator, the tile's P rounded to f16 before the PV product), so operand precision is
//     factored out and what remains is kernel logic — a wrong tail mask, a padded lane leaking, a mismapped fragment;
//  2. the bm64 and bm128 arms are BIT-IDENTICAL (non-causal, so BM changes no row's arithmetic order).
//
// M values cross the 64/128 tile heights and leave a partial last key tile (17, 70, 130, 517).
func TestAttnVit_logicAndBMIdentity(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	defer dev.ReleaseAll()
	mod, err := dev.CompileLibrary(attnFusedVitPTX)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p64, err := dev.NewComputePipeline(mod, "attn_vit_hd72_bm64")
	if err != nil {
		t.Fatal(err)
	}
	p128, err := dev.NewComputePipeline(mod, "attn_vit_hd72_bm128")
	if err != nil {
		t.Fatal(err)
	}
	stream := dev.NewCommandQueue()
	const hd, scale = 72, float32(0.11785113)
	smem := uint32(2 * (64*(80+8) + 80*(64+8)))
	for _, nH := range []int{4, 16} {
		for _, M := range []int{17, 70, 130, 517} {
			t.Run(fmt.Sprintf("nH%d/M%d", nH, M), func(t *testing.T) {
				qDim := nH * hd
				q, kc, vc := make([]float32, M*qDim), make([]float32, M*qDim), make([]float32, M*qDim)
				for i := range q {
					q[i] = float32(math.Sin(float64(i)*0.37)) * 0.9
					kc[i] = float32(math.Cos(float64(i)*0.21)) * 1.1
					vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
				}
				run := func(p Pipeline, bm int) []float32 {
					qb, kb, vb, ob := gpu.NewBufferLenOf[float32](dev, len(q)), gpu.NewBufferLenOf[float32](dev, len(kc)), gpu.NewBufferLenOf[float32](dev, len(vc)), gpu.NewBufferLenOf[float32](dev, len(q))
					for b, src := range map[Buffer][]float32{qb: q, kb: kc, vb: vc} {
						if e := gpu.Upload(b, src); e != nil {
							t.Fatal(e)
						}
					}
					out := make([]float32, len(q))
					for i := range out {
						out[i] = float32(math.NaN()) // a lane the kernel fails to write stays NaN and fails the cosine
					}
					if e := gpu.Upload(ob, out); e != nil {
						t.Fatal(e)
					}
					cfg := LaunchConfig{GridX: uint32(nH), GridY: uint32((M + bm - 1) / bm), GridZ: 1, BlockX: uint32(bm / 16 * 32), BlockY: 1, BlockZ: 1, SharedMemBytes: smem}
					if e := stream.Launch(p, cfg, Arg(qb), Arg(kb), Arg(vb), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nH)), gpu.ArgValue(scale), gpu.ArgValue(int32(M)), Arg(ob)); e != nil {
						t.Fatal(e)
					}
					if e := stream.Sync(); e != nil {
						t.Fatal(e)
					}
					if e := gpu.Download(ob, out); e != nil {
						t.Fatal(e)
					}
					return out
				}
				o64, o128 := run(p64, 64), run(p128, 128)
				for i := range o64 {
					if math.Float32bits(o64[i]) != math.Float32bits(o128[i]) {
						t.Fatalf("bm64 and bm128 differ at %d: %g vs %g", i, o64[i], o128[i])
					}
				}
				worst, at := 1.0, ""
				for m := 0; m < M; m++ {
					for h := 0; h < nH; h++ {
						mRun, lRun := math.Inf(-1), 0.0
						acc := make([]float64, hd)
						for s0 := 0; s0 < M; s0 += 64 {
							nk := min(64, M-s0)
							sc := make([]float64, nk)
							tmax := math.Inf(-1)
							for j := 0; j < nk; j++ {
								var dot float64
								for d := 0; d < hd; d++ {
									dot += float64(f16rne(q[m*qDim+h*hd+d])) * float64(f16rne(kc[(s0+j)*qDim+h*hd+d]))
								}
								sc[j] = dot * float64(scale)
								tmax = math.Max(tmax, sc[j])
							}
							mNew := math.Max(mRun, tmax)
							alpha := math.Exp(mRun - mNew)
							if math.IsInf(mRun, -1) {
								alpha = 0
							}
							var sum float64
							for j := range sc {
								sc[j] = math.Exp(sc[j] - mNew)
								sum += sc[j]
							}
							for d := range acc {
								acc[d] *= alpha
							}
							for j := range sc {
								pj := float64(f16rne(float32(sc[j])))
								for d := 0; d < hd; d++ {
									acc[d] += pj * float64(f16rne(vc[(s0+j)*qDim+h*hd+d]))
								}
							}
							lRun = alpha*lRun + sum
							mRun = mNew
						}
						var dot, na, nb float64
						for d := 0; d < hd; d++ {
							x, y := float64(o64[m*qDim+h*hd+d]), acc[d]/lRun
							dot += x * y
							na += x * x
							nb += y * y
						}
						if cos := dot / (math.Sqrt(na) * math.Sqrt(nb)); !(cos >= worst) {
							worst, at = cos, fmt.Sprintf("m=%d h=%d", m, h)
						}
					}
				}
				t.Logf("nH=%d M=%d: worst per-row cosine vs f16-operand f64 reference = %.8f (%s); bm64 == bm128 bit for bit", nH, M, worst, at)
				if !(worst >= 0.99999) {
					t.Errorf("worst cosine %.8f at %s: kernel logic defect (the reference uses the kernel's own f16 operands)", worst, at)
				}
			})
		}
	}
}
