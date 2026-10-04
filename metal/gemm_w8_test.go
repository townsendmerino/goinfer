//go:build darwin

package metal

import (
	"math"
	"testing"
)

// TestGemmW8Tile_matchesReference (int8 slice 2, docs/tasks/task-metal-int8-2026-10.md): gemm_w4f16_tile<TM, TN, true>,
// the batched pass's GEMM over the native int8 path's weights (row-major int8 codes, one f32 scale a row), against a
// float64 reference over the same inputs: A as f16, each weight dequantized to f16 exactly as the kernel stages it
// (half(q*scale)). Each output must be within one f16 rounding of the reference plus the f32 accumulation error over K
// products (4·sqrt(K)·2^-24·Σ|a·w|); a wrong row, column or code is off by orders of magnitude more. Every tile must equal the 64 × 64 one bit for bit (the int4 tiles' property: a tile changes only which
// threadgroup owns an output). Shapes: the 0.5B's and 1.5B's GEMMs, 8-72 rows, the three epilogues.
func TestGemmW8Tile_matchesReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	tiles := []struct {
		name   string
		tm, tn int
	}{{"m64n64", 64, 64}, {"m32n64", 32, 64}, {"m64n32", 64, 32}, {"m32n32", 32, 32}, {"m16n32", 16, 32}}
	pipes := make([]Pipeline, len(tiles))
	for i, tl := range tiles {
		if pipes[i], err = d.NewComputePipeline(lib, "gemm_w8f16_"+tl.name); err != nil {
			t.Fatalf("pipeline %s: %v", tl.name, err)
		}
	}
	cq := d.NewCommandQueue()
	seed := uint32(80808)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	checked := 0
	for _, sh := range []struct{ N, K int }{{1152, 896}, {896, 896}, {9728, 896}, {896, 4864}, {2048, 1536}, {520, 512}} {
		N, K := sh.N, sh.K
		codes := make([]int8, N*K)
		for i := range codes {
			codes[i] = int8(int(rnd()%255) - 127)
		}
		scale := make([]float32, N)
		for i := range scale {
			scale[i] = float32(int(rnd()%1000)+1) * 2e-5
		}
		wq, ws := NewBufferInt8(d, codes), NewBufferFloats(d, scale)
		bias := make([]float32, N)
		for i := range bias {
			bias[i] = float32(int(rnd()%2001)-1000) * 1e-3
		}
		bB, uN, uK := NewBufferFloats(d, bias), NewBufferU32(d, uint32(N)), NewBufferU32(d, uint32(K))
		wd := make([]float64, N*K) // the dequantized f16 weight, as the kernel stages it
		for n := range N {
			for k := range K {
				wd[n*K+k] = float64(f16ToF32(f32ToF16(float32(codes[n*K+k]) * scale[n])))
			}
		}
		for _, rows := range []int{8, 16, 24, 40, 72} {
			av := make([]uint16, rows*K)
			for i := range av {
				av[i] = f32ToF16(float32(int(rnd()%2001)-1000) * 1e-3)
			}
			resid := make([]uint16, rows*N)
			for i := range resid {
				resid[i] = f32ToF16(float32(int(rnd()%2001)-1000) * 1e-2)
			}
			a, uM := NewBufferU16s(d, av), NewBufferU32(d, uint32(rows))
			for mode := uint32(0); mode < 3; mode++ {
				uMode := NewBufferU32(d, mode)
				var first []uint16
				for ti, tl := range tiles {
					c := NewBufferU16s(d, resid)
					e := cq.Begin()
					e.Dispatch2D(pipes[ti], (N+tl.tn-1)/tl.tn, (rows+tl.tm-1)/tl.tm, 128, 1, a, wq, ws, c, uM, uN, uK, bB, uMode)
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("N %d K %d rows %d mode %d %s: %v", N, K, rows, mode, tl.name, err)
					}
					got := append([]uint16(nil), c.U16s()[:rows*N]...)
					d.ReleaseBuf(c)
					if first == nil {
						first = got
						for m := range rows {
							for n := range N {
								var ref, abs float64
								for k := range K {
									x := float64(f16ToF32(av[m*K+k])) * wd[n*K+k]
									ref += x
									abs += math.Abs(x)
								}
								switch mode {
								case 1:
									ref += float64(bias[n])
								case 2:
									ref += float64(f16ToF32(resid[m*N+n]))
								}
								g := float64(f16ToF32(got[m*N+n]))
								// one f16 rounding of the result, plus f32 accumulation over K products (sqrt(K) u Σ|a·w|, x4)
								bound := math.Abs(ref)*math.Pow(2, -10) + 4*math.Sqrt(float64(K))*math.Pow(2, -24)*abs + 1e-7
								if diff := math.Abs(g - ref); diff > bound {
									t.Fatalf("N %d K %d rows %d mode %d [%d,%d]: kernel %g, reference %g (diff %g > bound %g)",
										N, K, rows, mode, m, n, g, ref, diff, bound)
								}
							}
						}
						checked += rows * N
						continue
					}
					for i := range got {
						if got[i] != first[i] {
							t.Fatalf("N %d K %d rows %d mode %d: %s differs from m64n64 at %d", N, K, rows, mode, tl.name, i)
						}
					}
				}
			}
		}
	}
	t.Logf("%d outputs within the bound of the reference; every tile equal to m64n64", checked)
}
