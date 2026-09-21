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

// TestFlashDecodeRowsBitIdentical is gate G1 of attn-decode-fa-verify-PREREGISTERED.md: for every row of a verify batch the
// multi-row lane's output equals the M=1 lane's output at that row's position, BIT FOR BIT (math.Float32bits), on random K/V at the
// geometries of the 0.5B (hd64, G=7), the 1.5B (hd128, G=6), gemma3-1b (hd256, G=4) and D7 (hd128, G=7); S in {2,4,8,16}; M in
// {1,2,3,5,9,16}; batches placed to straddle a per-change (per = ceil(span/S)) and, with a sliding window, a winStart change.
// Any differing bit is a defect: the multi-row kernel is only worth having if it IS the M=1 kernel, several rows at a time.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeRowsBitIdentical -v
func TestFlashDecodeRowsBitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints for their geometry)")
	}
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	for _, mc := range []struct{ name, file string }{
		{"hd64-G7", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"hd128-G6", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"hd256-G4", "gemma3-1b-q4_k_m.gguf"},
		{"hd128-G7", "qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(mc.name, func(t *testing.T) {
			path := modelPath(mc.file)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			rf, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Skip("resident path declined")
			}
			if rf.faCombineRows == (Pipeline{}) {
				t.Fatal("multi-row kernels did not load")
			}
			hd, nKV, nH := rf.layers[0].hd, rf.layers[0].nKV, rf.nH
			kvDim, qDim := nKV*hd, nH*hd
			const maxKeys, maxM = 2600, faMaxRows
			rng := rand.New(rand.NewSource(11))
			q := make([]float32, maxM*qDim)
			k := make([]float32, maxKeys*kvDim)
			v := make([]float32, maxKeys*kvDim)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			for i := range k {
				k[i] = float32(rng.NormFloat64())
				v[i] = float32(rng.NormFloat64())
			}
			var qb, kb, vb, cb, q1, c1 Buffer
			if e := rf.do(func() error {
				qb, kb, vb, cb = rf.af(len(q)), rf.af(len(k)), rf.af(len(v)), rf.af(maxM*qDim)
				q1, c1 = rf.af(qDim), rf.af(qDim)
				for _, u := range []struct {
					b Buffer
					x []float32
				}{{qb, q}, {kb, k}, {vb, v}} {
					if e := gpu.Upload(u.b, u.x); e != nil {
						return e
					}
				}
				return nil
			}); e != nil {
				t.Fatalf("setup: %v", e)
			}
			checked, straddled := 0, 0
			for _, window := range []int{0, 300} {
				for _, S := range []int{2, 4, 8, 16} {
					rf.faSplit = S
					// n0 values around per changes for this S (span multiples of S) and around the window edge.
					var n0s []int
					for _, base := range []int{S * 40, S * 97, 700, 1024, 2048} {
						n0s = append(n0s, base-3, base-1, base, base+2)
					}
					n0s = append(n0s, 1, 8, 9, 31)
					if window > 0 {
						n0s = append(n0s, window-4, window, window+1)
					}
					for _, n0 := range n0s {
						for _, M := range []int{1, 2, 3, 5, 9, 16} {
							if n0+M > maxKeys {
								continue
							}
							runs := faRowRuns(n0, M, window, S)
							if len(runs) > 1 {
								straddled++
							}
							got := make([]float32, M*qDim)
							if e := rf.do(func() error {
								for _, run := range runs {
									if e := rf.flashDecodeRowsLaunch(qb, kb, vb, cb, hd, nKV, run); e != nil {
										return e
									}
								}
								return rf.stream.Sync()
							}); e != nil {
								t.Fatalf("multi-row launch: %v", e)
							}
							// Download the rows that were written.
							full := make([]float32, maxM*qDim)
							if e := rf.do(func() error { return gpu.Download(cb, full) }); e != nil {
								t.Fatalf("download: %v", e)
							}
							copy(got, full[:M*qDim])
							for i := 0; i < M; i++ {
								nk := n0 + i
								ws := 0
								if window > 0 && nk > window {
									ws = nk - window
								}
								ref := make([]float32, qDim)
								if e := rf.do(func() error {
									if e := gpu.Upload(q1, q[i*qDim:(i+1)*qDim]); e != nil {
										return e
									}
									if e := rf.flashDecodeLaunch(q1, kb, vb, c1, hd, nKV, ws, nk); e != nil {
										return e
									}
									if e := rf.stream.Sync(); e != nil {
										return e
									}
									return gpu.Download(c1, ref)
								}); e != nil {
									t.Fatalf("M=1 reference: %v", e)
								}
								for j := range ref {
									checked++
									if math.Float32bits(ref[j]) != math.Float32bits(got[i*qDim+j]) {
										t.Fatalf("NOT BIT-IDENTICAL: window=%d S=%d n0=%d M=%d row %d (nKeys %d) elem %d: multi-row %v (%#x) vs M=1 %v (%#x); runs %+v",
											window, S, n0, M, i, nk, j, got[i*qDim+j], math.Float32bits(got[i*qDim+j]), ref[j], math.Float32bits(ref[j]), runs)
									}
								}
							}
						}
					}
				}
			}
			t.Logf("%s (nH=%d nKV=%d hd=%d G=%d R=%d): %d elements bit-identical to the M=1 lane; %d batches spanned >1 run", mc.name, nH, nKV, hd, nH/nKV, faRowsPerCTA(hd, nH/nKV), checked, straddled)
		})
	}
}
