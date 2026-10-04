//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

// TestEP02_adjacentRows is E-P02's probe (docs/audit-metal-2026-09-30.md, "E-P02"): the batched step's gate|up at B rows
// on the 7B shape (N 37,888, K 3,584, R 4), as production runs it below the fragment's calibrated size (B dispatches of
// gemv_w4a8_sa_rows4, each streaming the whole weight) and as one dispatch with each tile's B threadgroups adjacent
// (mc3_gemv_w4a8_sa_rows_adj4). Every output float must be equal; the GPU time of 16 reps per command buffer, best of 15
// buffers, arms alternated buffer by buffer. The audit's kill line: B = 2 above 0.65 ms. GOINFER_EP02=1.
func TestEP02_adjacentRows(t *testing.T) {
	if os.Getenv("GOINFER_EP02") != "1" {
		t.Skip("set GOINFER_EP02=1: a timed kernel probe")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pRow, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_rows4")
	if err != nil {
		t.Fatal(err)
	}
	pAdj, err := d.NewComputePipeline(lib, "mc3_gemv_w4a8_sa_rows_adj4")
	if err != nil {
		t.Fatal(err)
	}
	const R = 4
	for _, sh := range []struct {
		what string
		N, K int
	}{{"7B gate|up", 37888, 3584}, {"7B qkv-width", 4608, 3584}, {"1.5B gate|up", 17920, 1536}} {
		N, K := sh.N, sh.K
		rng := rand.New(rand.NewSource(int64(N)))
		wv := make([]uint32, N*K/8)
		for i := range wv {
			wv[i] = rng.Uint32()
		}
		sc := make([]uint16, N*K/32)
		for i := range sc {
			sc[i] = f32ToF16(rng.Float32()*0.01 + 0.001)
		}
		wq, ws := NewBufferUint32s(d, wv), NewBufferU16s(d, sc)
		uK, uN := NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(N))
		q := d.NewCommandQueue()
		for _, B := range []int{2, 3, 4} {
			a := make([]int8, B*K)
			for i := range a {
				a[i] = int8(rng.Intn(255) - 127)
			}
			asc := make([]float32, B)
			for i := range asc {
				asc[i] = 0.01 + 0.001*float32(i)
			}
			aq, aS, uB := NewBufferInt8(d, a), NewBufferFloats(d, asc), NewBufferU32(d, uint32(B))
			outRow, outAdj := d.NewBufferLen(B*N), d.NewBufferLen(B*N)
			enc := func(e *Encoder, adj bool) {
				if adj {
					e.DispatchTG(pAdj, B*N*32/R, 256, K*2, wq, ws, aq, aS, outAdj, uK, uN, uB)
					return
				}
				for m := range B {
					e.DispatchTG(pRow, N*32/R, 256, K*2, wq, ws, aq.At(m*K), aS.At(4*m), outRow.At(4*m*N), uK)
				}
			}
			for _, adj := range []bool{false, true} {
				e := q.Begin()
				enc(e, adj)
				e.End()
			}
			x, y := outRow.Floats()[:B*N], outAdj.Floats()[:B*N]
			for i := range x {
				if x[i] != y[i] {
					t.Fatalf("%s B=%d: output %d: adjacent %v, per-row %v", sh.what, B, i, y[i], x[i])
				}
			}
			const per = 16
			best := map[bool]time.Duration{false: time.Hour, true: time.Hour}
			for b := range 17 {
				for _, adj := range []bool{b%2 == 0, b%2 != 0} {
					e := q.Begin()
					for range per {
						enc(e, adj)
					}
					e.End()
					if dt := time.Duration((e.GPUEnd() - e.GPUStart()) * 1e9); b >= 2 && dt < best[adj] {
						best[adj] = dt
					}
				}
			}
			r, a2 := float64(best[false])/per/1e6, float64(best[true])/per/1e6
			fmt.Fprintf(os.Stderr, "[e-p02] %s B=%d: per-row %.3f ms, adjacent %.3f ms (best of 15, %d per buffer); per-row/adjacent %.3f; outputs equal\n",
				sh.what, B, r, a2, per, r/a2)
		}
	}
}
