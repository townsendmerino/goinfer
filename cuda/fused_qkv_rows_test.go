//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"os"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// fusedQKVRowsFixture builds random int4 Q/K/V weights (+ f16 group scales, optional biases) and an activation for one geometry, and runs either the original
// fused_rms_qkv (rowsPerWarp == 0) or fused_rms_qkv_rows at the given rowsPerWarp, returning the three outputs. n launches are issued back to back (timing runs use n>1).
func fusedQKVRowsRun(t *testing.T, rf *cudaResident, H, qDim, kvDim, rowsPerWarp, n int, bias bool, seed int64) (q, k, v []float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	Kwords, Kgroups := H/8, H/32
	nrows := qDim + 2*kvDim
	mkW := func(rows int) []uint32 {
		w := make([]uint32, rows*Kwords)
		for i := range w {
			w[i] = rng.Uint32()
		}
		return w
	}
	mkS := func(rows int) []uint16 {
		s := make([]uint16, rows*Kgroups)
		for i := range s {
			s[i] = uint16(0x2400 + rng.Intn(0x1400)) // positive f16 in ~[0.0004, 0.6]
		}
		return s
	}
	mkF := func(n int, scale float64) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(rng.NormFloat64() * scale)
		}
		return f
	}
	x := mkF(H, 2)
	nrm := mkF(H, 1)
	q, k, v = make([]float32, qDim), make([]float32, kvDim), make([]float32, kvDim)
	err := rf.do(func() error {
		xb, nb := rf.up32(x), rf.up32(nrm)
		Wq, Wk, Wv := rf.upu32(mkW(qDim)), rf.upu32(mkW(kvDim)), rf.upu32(mkW(kvDim))
		Sq, Sk, Sv := rf.upu16(mkS(qDim)), rf.upu16(mkS(kvDim)), rf.upu16(mkS(kvDim))
		bq, bk, bv := ArgNull(), ArgNull(), ArgNull()
		if bias {
			bq, bk, bv = Arg(rf.up32(mkF(qDim, 0.5))), Arg(rf.up32(mkF(kvDim, 0.5))), Arg(rf.up32(mkF(kvDim, 0.5)))
		}
		qb, kb, vb := rf.af(qDim), rf.af(kvDim), rf.af(kvDim)
		shm := uint32((H + 256 + H/4) * 4)
		for i := 0; i < n; i++ {
			var e error
			if rowsPerWarp == 0 {
				e = rf.launch(rf.fQKV, LaunchConfig{GridX: uint32((nrows + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
					Arg(xb), Arg(nb), gpu.ArgValue(int32(H)), gpu.ArgValue(float32(1e-6)), gpu.ArgValue(int32(0)),
					Arg(Wq), Arg(Sq), bq, Arg(Wk), Arg(Sk), bk, Arg(Wv), Arg(Sv), bv,
					gpu.ArgValue(int32(qDim)), gpu.ArgValue(int32(kvDim)), gpu.ArgValue(int32(Kwords)), gpu.ArgValue(int32(Kgroups)),
					Arg(qb), Arg(kb), Arg(vb))
			} else {
				rowsPerBlock := 8 * rowsPerWarp
				e = rf.launch(rf.fQKVRows, LaunchConfig{GridX: uint32((nrows + rowsPerBlock - 1) / rowsPerBlock), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
					Arg(xb), Arg(nb), gpu.ArgValue(int32(H)), gpu.ArgValue(float32(1e-6)), gpu.ArgValue(int32(0)),
					Arg(Wq), Arg(Sq), bq, Arg(Wk), Arg(Sk), bk, Arg(Wv), Arg(Sv), bv,
					gpu.ArgValue(int32(qDim)), gpu.ArgValue(int32(kvDim)), gpu.ArgValue(int32(Kwords)), gpu.ArgValue(int32(Kgroups)), gpu.ArgValue(int32(rowsPerWarp)),
					Arg(qb), Arg(kb), Arg(vb))
			}
			if e != nil {
				return e
			}
		}
		if e := rf.stream.Sync(); e != nil {
			return e
		}
		for _, d := range []struct {
			b Buffer
			o []float32
		}{{qb, q}, {kb, k}, {vb, v}} {
			if e := gpu.Download(d.b, d.o); e != nil {
				return e
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("fused qkv (rowsPerWarp=%d): %v", rowsPerWarp, err)
	}
	return q, k, v
}

// TestFusedQKVRowsBitIdentical: fused_rms_qkv_rows equals the original fused_rms_qkv BIT FOR BIT for every rows-per-warp, with and without biases, on the real
// projection geometries (D7 3584/3584/512, 1.5B 1536/1536/256, 0.5B 896/896/128, and a non-multiple-of-block row count). rowsPerWarp only changes which warp computes which
// row, so any differing bit is a defect.
func TestFusedQKVRowsBitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a small model for a device context)")
	}
	m, err := decoder.Load(modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || rf.fQKVRows == (Pipeline{}) {
		t.Skip("resident path declined or fused_rms_qkv_rows not loaded")
	}
	geoms := []struct {
		name     string
		H, q, kv int
	}{
		{"D7", 3584, 3584, 512}, {"1.5B", 1536, 1536, 256}, {"0.5B", 896, 896, 128}, {"odd", 1024, 1000, 168},
	}
	checked := 0
	for _, g := range geoms {
		for _, bias := range []bool{false, true} {
			rq, rk, rv := fusedQKVRowsRun(t, rf, g.H, g.q, g.kv, 0, 1, bias, 7)
			for _, rpw := range []int{1, 2, 3, 4, 6, 8, 16} {
				q, k, v := fusedQKVRowsRun(t, rf, g.H, g.q, g.kv, rpw, 1, bias, 7)
				for _, c := range []struct {
					n    string
					a, b []float32
				}{{"q", rq, q}, {"k", rk, k}, {"v", rv, v}} {
					for i := range c.a {
						checked++
						if math.Float32bits(c.a[i]) != math.Float32bits(c.b[i]) {
							t.Fatalf("%s bias=%v rowsPerWarp=%d: %s[%d] differs: original %v, rows kernel %v", g.name, bias, rpw, c.n, i, c.a[i], c.b[i])
						}
					}
				}
			}
		}
	}
	t.Logf("fused_rms_qkv_rows == fused_rms_qkv over %d output elements (4 geometries x bias/no-bias x 7 rows-per-warp values)", checked)
}

// TestFusedQKVRowsBench launches every variant many times per geometry; run it under ncu (gpu__time_duration) to compare kernel times. It asserts nothing.
//
//	GOINFER_HEAVY_TESTS=1 ncu --metrics gpu__time_duration.sum -k regex:fused_rms_qkv --csv --page raw <test binary> -test.run TestFusedQKVRowsBench
func TestFusedQKVRowsBench(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_FQKV_BENCH") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_FQKV_BENCH=1")
	}
	m, err := decoder.Load(modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest().(*cudaResident)
	for _, g := range fusedQKVBenchGeoms {
		t.Logf("GEOM %s", g.name)
		for _, rpw := range fusedQKVBenchRPWs {
			t.Logf("  launching rowsPerWarp=%d", rpw)
			fusedQKVRowsRun(t, rf, g.H, g.q, g.kv, rpw, 3, true, 1)
		}
	}
}

// fusedQKVBenchGeoms are the (hidden, qDim, kvDim) of the dense families goinfer runs on CUDA, for the rows-per-warp selection table.
var fusedQKVBenchGeoms = []struct {
	name     string
	H, q, kv int
}{
	{"D7 qwen2.5-7b", 3584, 3584, 512}, {"qwen2.5-3b", 2048, 2048, 256}, {"1.5B", 1536, 1536, 256}, {"0.5B", 896, 896, 128},
	{"qwen3-4b", 2560, 4096, 1024}, {"qwen3-1.7b", 2048, 2048, 1024}, {"llama3-8b/mistral-7b", 4096, 4096, 1024},
	{"gemma3-1b", 1152, 1024, 256}, {"phi3-mini (MHA)", 3072, 3072, 3072}, {"olmo/llama-1b", 2048, 2048, 512},
}

var fusedQKVBenchRPWs = []int{0, 1, 2, 4, 8, 16}

// fusedGURun runs fused_rms_gu (rowsPerWarp == 0) or fused_rms_gu_rows at the given rows-per-warp on random int4 gate/up weights for one (hidden, intermediate) geometry, n launches
// back to back.
func fusedGURun(t *testing.T, rf *cudaResident, H, I, rowsPerWarp, n int, seed int64) (g, u []float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	Kwords, Kgroups := H/8, H/32
	mkW := func(rows int) []uint32 {
		w := make([]uint32, rows*Kwords)
		for i := range w {
			w[i] = rng.Uint32()
		}
		return w
	}
	mkS := func(rows int) []uint16 {
		s := make([]uint16, rows*Kgroups)
		for i := range s {
			s[i] = uint16(0x2400 + rng.Intn(0x1400))
		}
		return s
	}
	mkF := func(n int, scale float64) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(rng.NormFloat64() * scale)
		}
		return f
	}
	x, nrm := mkF(H, 2), mkF(H, 1)
	g, u = make([]float32, I), make([]float32, I)
	if err := rf.do(func() error {
		xb, nb := rf.up32(x), rf.up32(nrm)
		Wg, Wu := rf.upu32(mkW(I)), rf.upu32(mkW(I))
		Sg, Su := rf.upu16(mkS(I)), rf.upu16(mkS(I))
		gb, ub := rf.af(I), rf.af(I)
		shm := uint32((H + 256 + H/4) * 4)
		for i := 0; i < n; i++ {
			var e error
			if rowsPerWarp == 0 {
				e = rf.launch(rf.fGU, LaunchConfig{GridX: uint32((2*I + 63) / 64), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
					Arg(xb), Arg(nb), gpu.ArgValue(int32(H)), gpu.ArgValue(float32(1e-6)), gpu.ArgValue(int32(0)),
					Arg(Wg), Arg(Sg), Arg(Wu), Arg(Su), gpu.ArgValue(int32(I)), gpu.ArgValue(int32(Kwords)), gpu.ArgValue(int32(Kgroups)), Arg(gb), Arg(ub))
			} else {
				rows := 8 * rowsPerWarp
				e = rf.launch(rf.fGURows, LaunchConfig{GridX: uint32((2*I + rows - 1) / rows), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
					Arg(xb), Arg(nb), gpu.ArgValue(int32(H)), gpu.ArgValue(float32(1e-6)), gpu.ArgValue(int32(0)),
					Arg(Wg), Arg(Sg), Arg(Wu), Arg(Su), gpu.ArgValue(int32(I)), gpu.ArgValue(int32(Kwords)), gpu.ArgValue(int32(Kgroups)), gpu.ArgValue(int32(rowsPerWarp)), Arg(gb), Arg(ub))
			}
			if e != nil {
				return e
			}
		}
		if e := rf.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.Download(gb, g); e != nil {
			return e
		}
		return gpu.Download(ub, u)
	}); err != nil {
		t.Fatalf("fused gu (rowsPerWarp=%d): %v", rowsPerWarp, err)
	}
	return g, u
}

var fusedGUBenchGeoms = []struct {
	name string
	H, I int
}{
	{"D7 qwen2.5-7b", 3584, 18944}, {"qwen2.5-3b", 2048, 11008}, {"1.5B", 1536, 8960}, {"0.5B", 896, 4864}, {"qwen3-4b", 2560, 9728},
	{"qwen3-1.7b", 2048, 6144}, {"llama3-8b", 4096, 14336}, {"gemma3-1b", 1152, 6912}, {"phi3-mini", 3072, 8192},
}

var fusedGUBenchRPWs = []int{0, 8, 16, 32}

// TestFusedGURowsBitIdentical: fused_rms_gu_rows equals the original fused_rms_gu BIT FOR BIT at every rows-per-warp, on the real (hidden, intermediate) geometries.
func TestFusedGURowsBitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	m, err := decoder.Load(modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || rf.fGURows == (Pipeline{}) {
		t.Skip("resident path declined or fused_rms_gu_rows not loaded")
	}
	checked := 0
	for _, g := range fusedGUBenchGeoms {
		rg, ru := fusedGURun(t, rf, g.H, g.I, 0, 1, 3)
		for _, rpw := range []int{1, 2, 3, 4, 8, 16, 32} {
			ng, nu := fusedGURun(t, rf, g.H, g.I, rpw, 1, 3)
			for i := range rg {
				checked += 2
				if math.Float32bits(rg[i]) != math.Float32bits(ng[i]) || math.Float32bits(ru[i]) != math.Float32bits(nu[i]) {
					t.Fatalf("%s (H=%d I=%d) rowsPerWarp=%d: row %d differs: gate %v/%v up %v/%v", g.name, g.H, g.I, rpw, i, rg[i], ng[i], ru[i], nu[i])
				}
			}
		}
	}
	t.Logf("fused_rms_gu_rows == fused_rms_gu over %d output elements (%d geometries x 7 rows-per-warp values)", checked, len(fusedGUBenchGeoms))
}

// TestFusedGUBench launches every variant per geometry for ncu timing (asserts nothing). GOINFER_FGU_BENCH=1.
func TestFusedGUBench(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_FGU_BENCH") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_FGU_BENCH=1")
	}
	m, err := decoder.Load(modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest().(*cudaResident)
	for _, g := range fusedGUBenchGeoms {
		for _, rpw := range fusedGUBenchRPWs {
			fusedGURun(t, rf, g.H, g.I, rpw, 15, 1)
		}
	}
}

// TestWaveRowsPerWarpFor pins the rule on the measured card (40 SMs, 1024 threads/SM, 64 KB shared per SM) to the values the microbenchmark validated, and its "cannot tell" answers.
func TestWaveRowsPerWarpFor(t *testing.T) {
	sms, thr, sm := 40, 1024, 65536
	smem := func(h int) int { return (h + 256 + h/4) * 4 }
	for _, c := range []struct {
		name    string
		rows, h int
		want    int
	}{
		{"D7 gate/up", 37888, 3584, 40}, {"D7 qkv", 4608, 3584, 5}, {"1.5B gate/up", 17920, 1536, 14}, {"1.5B qkv", 2048, 1536, 2},
		{"0.5B qkv (no gain: stays original)", 1152, 896, 1}, {"llama3-8b gate/up", 28672, 4096, 30}, {"gemma3-1b gate/up", 13824, 1152, 11},
	} {
		if got := waveRowsPerWarpFor(c.rows, smem(c.h), sms, thr, sm); got != c.want {
			t.Errorf("%s: waveRowsPerWarpFor = %d, want %d", c.name, got, c.want)
		}
	}
	for _, bad := range [][5]int{{0, 1000, 40, 1024, 65536}, {100, 0, 40, 1024, 65536}, {100, 1000, 0, 1024, 65536}, {100, 1000, 40, 128, 65536}, {100, 70000, 40, 1024, 65536}} {
		if got := waveRowsPerWarpFor(bad[0], bad[1], bad[2], bad[3], bad[4]); got != 0 {
			t.Errorf("waveRowsPerWarpFor(%v) = %d, want 0 (cannot tell)", bad, got)
		}
	}
}
