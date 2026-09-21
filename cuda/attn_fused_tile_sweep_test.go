//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttnFusedTile_smallMSweep is the pre-registered threshold sweep of attn-fused-tile128-default-PREREGISTERED.md:
// attention-only kernel launches, 64x64 vs 128x64 interleaved, per (hd, M, startPos) paired ratio t128/t64. It prints the
// table and the T the registered rule picks (smallest sweep M such that every M >= T at every startPos has ratio <= 1.03).
func TestAttnFusedTile_smallMSweep(t *testing.T) {
	requireHeavyModel(t)
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	if r.bAttnBM128hd64 == (Pipeline{}) || r.bAttnBM128hd128 == (Pipeline{}) {
		t.Skip("tile pipelines not loaded (set GOINFER_CUDA_FAST_PREFILL=1)")
	}
	Ms := []int{16, 32, 48, 64, 96, 128, 192, 256, 384, 512}
	SPs := []int{0, 1024, 3400}
	type hdcfg struct{ hd, nH, nKV int }
	worstByM := map[int]float64{}
	for _, hc := range []hdcfg{{128, 12, 2}, {64, 14, 2}} {
		fmt.Printf("hd%d nH=%d nKV=%d   M x startPos -> t128/t64 (t64 us)\n", hc.hd, hc.nH, hc.nKV)
		for _, M := range Ms {
			line := fmt.Sprintf("  M=%-4d", M)
			for _, sp := range SPs {
				qDim, kvDim := hc.nH*hc.hd, hc.nKV*hc.hd
				nKeys := sp + M
				q, kc, vc := make([]float32, M*qDim), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
				for i := range q {
					q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
				}
				for i := range kc {
					kc[i] = float32(math.Cos(float64(i) * 0.21))
					vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
				}
				var ratio, t64us float64
				err := r.do(func() error {
					qb, kb, vb, ob := r.af(len(q)), r.af(len(kc)), r.af(len(vc)), r.af(M*qDim)
					for b, src := range map[Buffer][]float32{qb: q, kb: kc, vb: vc} {
						if e := gpu.Upload(b, src); e != nil {
							return e
						}
					}
					args := []gpu.KernelArg{Arg(qb), Arg(kb), Arg(vb), gpu.ArgValue(int32(hc.nH)), gpu.ArgValue(int32(hc.nKV)), gpu.ArgValue(int32(hc.hd)),
						gpu.ArgValue(int32(sp)), gpu.ArgValue(float32(0.125)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M)), Arg(ob), ArgNull()}
					p64, sh64 := r.attnFusedFor(hc.hd)
					c64 := LaunchConfig{GridX: uint32(hc.nH), GridY: uint32((M + 63) / 64), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: sh64}
					p128 := tilePipeline(r, 3, hc.hd)
					c128 := LaunchConfig{GridX: uint32(hc.nH), GridY: uint32((M + 127) / 128), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: sh64}
					const reps, per = 7, 20
					timeIt := func(p Pipeline, c LaunchConfig) (float64, error) {
						t0 := time.Now()
						for i := 0; i < per; i++ {
							if e := r.launch(p, c, args...); e != nil {
								return 0, e
							}
						}
						if e := r.stream.Sync(); e != nil {
							return 0, e
						}
						return float64(time.Since(t0).Microseconds()) / per, nil
					}
					for i := 0; i < 3; i++ { // warm-up both
						timeIt(p64, c64)
						timeIt(p128, c128)
					}
					var a, b []float64
					for i := 0; i < reps; i++ {
						x, e := timeIt(p64, c64)
						if e != nil {
							return e
						}
						y, e := timeIt(p128, c128)
						if e != nil {
							return e
						}
						a, b = append(a, x), append(b, y)
					}
					sort.Float64s(a)
					sort.Float64s(b)
					t64us, ratio = a[reps/2], b[reps/2]/a[reps/2]
					return nil
				})
				if err != nil {
					t.Fatalf("M=%d sp=%d: %v", M, sp, err)
				}
				line += fmt.Sprintf("  sp%-4d %5.2fx (%6.0f)", sp, ratio, t64us)
				if ratio > worstByM[M] {
					worstByM[M] = ratio
				}
			}
			fmt.Println(line)
		}
	}
	T := -1
	for i := len(Ms) - 1; i >= 0; i-- {
		if worstByM[Ms[i]] > 1.03 {
			break
		}
		T = Ms[i]
	}
	fmt.Printf("worst ratio by M (both hd, all startPos): %v\nREGISTERED RULE PICKS T = %d (-1 = none)\n", worstByM, T)
}
