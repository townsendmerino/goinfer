//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"
)

// TestGemmTile_bitIdentical is A-P01's kernel gate (docs/audit-metal-2026-09-30.md): every tile of gemm_w4f16_tile
// against gemm_w4f16_store, the R16 kernel production runs at 64 × 64, at the 1.5B's and 7B's prefill GEMM shapes
// (qkv, o, gate/up width, down) plus a ragged N, over pass sizes that cross both of gemmTile's thresholds, in all three
// epilogues (store, +bias, +=residual). Every f16 output must match bit for bit. The 64 × 64 instance of the template
// is checked too: it is the template at production's own tile, so a difference there would be the template's, not
// the tile's.
func TestGemmTile_bitIdentical(t *testing.T) {
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
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	ref := pipe("gemm_w4f16_store")
	tiles := []struct {
		name   string
		tm, tn int
	}{{"gemm_w4f16_m64n64", 64, 64}, {"gemm_w4f16_m32n64", 32, 64}, {"gemm_w4f16_m64n32", 64, 32}, {"gemm_w4f16_m32n32", 32, 32},
		{"gemm_w4f16_m16n64", 16, 64}, {"gemm_w4f16_m16n32", 16, 32}}
	cq := d.NewCommandQueue()
	seed := uint32(64032)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	half := func(f float32) uint16 { return f32ToF16(f) }
	checked := 0
	for _, sh := range []struct {
		what string
		N, K int
	}{{"1.5B qkv", 2048, 1536}, {"1.5B o", 1536, 1536}, {"1.5B down", 1536, 8960}, {"gate/up width", 4096, 1536},
		{"7B qkv", 4608, 3584}, {"7B o", 3584, 3584}, {"ragged N", 520, 512}} {
		N, K := sh.N, sh.K
		w := d.NewBufferLen(N * K / 8)
		wv := w.U32s()[:N*K/8]
		for i := range wv {
			wv[i] = rnd()
		}
		sc := make([]uint16, N*K/32)
		for i := range sc {
			sc[i] = half(float32(int(rnd()%2001)-1000) * 2e-5)
		}
		ws := NewBufferU16s(d, sc)
		bias := make([]float32, N)
		for i := range bias {
			bias[i] = float32(int(rnd()%2001)-1000) * 1e-3
		}
		bB, uN, uK := NewBufferFloats(d, bias), NewBufferU32(d, uint32(N)), NewBufferU32(d, uint32(K))
		for _, rows := range []int{8, 16, 24, 32, 40, 48, 64, 72} {
			av := make([]uint16, rows*K)
			for i := range av {
				av[i] = half(float32(int(rnd()%2001)-1000) * 1e-3)
			}
			resid := make([]uint16, rows*N)
			for i := range resid {
				resid[i] = half(float32(int(rnd()%2001)-1000) * 1e-2)
			}
			a, uM := NewBufferU16s(d, av), NewBufferU32(d, uint32(rows))
			for mode := range uint32(3) {
				uMode := NewBufferU32(d, mode)
				run := func(p Pipeline, tm, tn int) []uint16 {
					c := NewBufferU16s(d, resid) // mode 2 adds into it
					e := cq.Begin()
					e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, a, w, ws, c, uM, uN, uK, bB, uMode)
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("%s rows %d mode %d: %v", sh.what, rows, mode, err)
					}
					return append([]uint16(nil), c.U16s()[:rows*N]...)
				}
				want := run(ref, 64, 64)
				for _, tl := range tiles {
					got := run(pipe(tl.name), tl.tm, tl.tn)
					diff := 0
					for i := range want {
						if got[i] != want[i] {
							diff++
						}
					}
					if diff != 0 {
						t.Errorf("%s (N %d, K %d), %d rows, mode %d: %s differs from gemm_w4f16_store in %d of %d outputs",
							sh.what, N, K, rows, mode, tl.name, diff, rows*N)
					}
					checked += rows * N
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[gemm-tile] %d outputs across 7 shapes, 8-72 rows, 3 epilogues, 6 tiles: compared with gemm_w4f16_store\n", checked)
}

// TestGemmTile_selector pins gemmTile's choices (A-P01, then D-B02's padding rule): which tile each policy gives at
// small and large passes, so a selector change is a visible test change.
func TestGemmTile_selector(t *testing.T) {
	pf := &prefillState{}
	defer func() { gemmTilePolicy = "" }()
	for _, c := range []struct {
		policy       string
		rows, N      int
		wantM, wantN int
	}{
		{"", 16, 2048, 16, 32}, {"", 16, 17920, 16, 32}, {"", 40, 17920, 64, 64}, {"", 40, 8192, 16, 32}, {"", 32, 17920, 32, 64}, {"", 40, 1536, 16, 32}, {"", 64, 2048, 64, 32},
		{"", 72, 1536, 32, 64}, {"", 512, 17920, 64, 64}, {"", 16, 2056, 16, 32}, {"", 96, 4096, 32, 64},
		{"", 24, 2048, 32, 32}, {"", 160, 17920, 64, 64}, {"", 48, 4096, 16, 32}, {"", 104, 2048, 64, 64}, {"", 192, 1536, 64, 64},
		{"a01", 16, 2048, 32, 32}, {"a01", 32, 17920, 32, 64}, {"a01", 40, 1536, 64, 32}, {"a01", 64, 2048, 64, 32},
		{"a01", 72, 1536, 64, 64}, {"a01", 512, 17920, 64, 64}, {"a01", 16, 2056, 32, 64},
		{"shipped", 16, 2048, 64, 64}, {"bm32", 16, 2048, 32, 64}, {"bn32", 16, 2048, 64, 32},
	} {
		gemmTilePolicy = c.policy
		_, tm, tn := pf.gemmTile(c.rows, c.N)
		if tm != c.wantM || tn != c.wantN {
			t.Errorf("policy %q, %d rows × N %d: tile %d×%d, want %d×%d", c.policy, c.rows, c.N, tm, tn, c.wantM, c.wantN)
		}
	}
}
