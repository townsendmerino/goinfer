//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttnFusedTile_32x64BitIdentical: the 32-row query tile (BN unchanged) must equal the shipped 64x64 kernel bit for bit
// when there is no sliding window (attn-fused-tile-PREREGISTERED.md): every row sees the same key tiles in the same order,
// and the extra fully-masked tiles a taller block visits are exact no-ops. The window case is deliberately excluded: the tile
// grouping there depends on the block's first row, so identity is not expected (it is covered by TestAttnFused_vsF16Reference).
func TestAttnFusedTile_32x64BitIdentical(t *testing.T) {
	for _, arm := range []struct {
		name   string
		id, bm int
	}{{"32x64", 1, 32}, {"128x64", 3, 128}} {
		t.Run(arm.name, func(t *testing.T) { tileBitIdentical(t, arm.id, arm.bm) })
	}
}

func tileBitIdentical(t *testing.T, armID, bm int) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	if r.bAttnFused64 == (Pipeline{}) || r.bAttnBM32x64hd64 == (Pipeline{}) || r.bAttnBM32x64hd128 == (Pipeline{}) || r.bAttnBM128hd128 == (Pipeline{}) {
		t.Skip("attn_fused tile pipelines not loaded (set GOINFER_CUDA_FAST_PREFILL=1)")
	}
	const scale = 0.125
	checked := 0
	for _, hd := range []int{64, 128} {
		for _, M := range []int{1, 7, 16, 17, 31, 32, 33, 48, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257, 517} {
			for _, sp := range []int{0, 5, 384, 1024} {
				for _, g := range []struct {
					nH, nKV int
					sinks   bool
				}{{4, 4, false}, {12, 2, false}, {8, 2, true}} {
					name := fmt.Sprintf("hd%d/M%d/sp%d/nH%d/nKV%d/sinks=%v", hd, M, sp, g.nH, g.nKV, g.sinks)
					qDim, kvDim := g.nH*hd, g.nKV*hd
					nKeys := sp + M
					q, kc, vc := make([]float32, M*qDim), make([]float32, nKeys*kvDim), make([]float32, nKeys*kvDim)
					for i := range q {
						q[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
					}
					for i := range kc {
						kc[i] = float32(math.Cos(float64(i) * 0.21))
						vc[i] = float32(math.Sin(float64(i)*0.13)) * 0.7
					}
					var a0, a1 []float32
					err := r.do(func() error {
						qb, kb, vb := r.af(len(q)), r.af(len(kc)), r.af(len(vc))
						o0, o1 := r.af(M*qDim), r.af(M*qDim)
						sinkArg := ArgNull()
						if g.sinks {
							sv := make([]float32, g.nH)
							for i := range sv {
								sv[i] = float32(math.Cos(float64(i)*1.7)) * 2
							}
							sb := r.af(len(sv))
							if e := gpu.Upload(sb, sv); e != nil {
								return e
							}
							sinkArg = Arg(sb)
						}
						for b, src := range map[Buffer][]float32{qb: q, kb: kc, vb: vc} {
							if e := gpu.Upload(b, src); e != nil {
								return e
							}
						}
						args := func(dst Buffer) []gpu.KernelArg {
							return []gpu.KernelArg{Arg(qb), Arg(kb), Arg(vb), gpu.ArgValue(int32(g.nH)), gpu.ArgValue(int32(g.nKV)),
								gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(sp)), gpu.ArgValue(float32(scale)), gpu.ArgValue(int32(0)),
								gpu.ArgValue(int32(M)), Arg(dst), sinkArg}
						}
						p0, sh0 := r.attnFusedFor(hd)
						if e := r.launch(p0, LaunchConfig{GridX: uint32(g.nH), GridY: uint32((M + 63) / 64), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: sh0}, args(o0)...); e != nil {
							return e
						}
						if e := r.launch(tilePipeline(r, armID, hd), LaunchConfig{GridX: uint32(g.nH), GridY: uint32((M + bm - 1) / bm), GridZ: 1, BlockX: uint32(bm / 16 * 32), BlockY: 1, BlockZ: 1,
							SharedMemBytes: uint32(2 * (64*(hd+attnFusedKPAD) + hd*(64+attnFusedKPAD)))}, args(o1)...); e != nil {
							return e
						}
						if e := r.stream.Sync(); e != nil {
							return e
						}
						a0, a1 = make([]float32, M*qDim), make([]float32, M*qDim)
						if e := gpu.Download(o0, a0); e != nil {
							return e
						}
						return gpu.Download(o1, a1)
					})
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					for i := range a0 {
						if math.Float32bits(a0[i]) != math.Float32bits(a1[i]) {
							t.Fatalf("%s: tile arm differs from 64x64 at %d: %g vs %g", name, i, a0[i], a1[i])
						}
					}
					checked++
				}
			}
		}
	}
	t.Logf("tile arm == 64x64 bit for bit on %d shapes", checked)
}
