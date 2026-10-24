//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// TestKSD4_w32MatchesViTGEMM is D4-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D4 gates"): S17's tower_gemm_w32 against
// the kernel aikit's GEMMF32Plan picks (gemm_f32_sg_big or gemm_f32_sg), the GEMM eg2Ops.gemm runs today, C = A · Wᵀ at six
// (M, N, K) cells, random f32, tower_gemm_w32 from the precise library the towers compile (eg2MSL + gvMSL). A differing byte
// means the swap is not a bit-identical change.
func TestKSD4_w32MatchesViTGEMM(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	vit, err := d.NewViT()
	if err != nil {
		t.Fatal(err)
	}
	lib, err := d.CompileLibraryPrecise(eg2MSL+towerGemmMSL+gvMSL, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	w32, err := d.NewComputePipeline(lib, "tower_gemm_w32")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(41))
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	fill := func(n int) Buffer {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return NewBufferFloats(d, v)
	}
	red := 0
	for _, c := range [][3]int{{256, 768, 768}, {1024, 3072, 768}, {1024, 768, 3072}, {100, 768, 768}, {37, 1536, 1536}, {512, 512, 64}} {
		M, N, K := c[0], c[1], c[2]
		A, W := fill(M*K), fill(N*K)
		old, nw := d.NewBufferLen(M*N), d.NewBufferLen(M*N)
		p, gx, gy, tgx, tgy := vit.GEMMF32Plan(M, N, K)
		e := q.Begin()
		e.Dispatch2D(p, gx, gy, tgx, tgy, A, W, old, u(M), u(N), u(K))
		e.Dispatch2D(w32, (N+63)/64, (M+63)/64, 512, 1, A, W, nw, W, u(M), u(N), u(K), u(0), u(0))
		e.End()
		a, b := slices.Clone(old.U32s()[:M*N]), nw.U32s()[:M*N]
		diff := 0
		for i := range a {
			if a[i] != b[i] {
				diff++
			}
		}
		fmt.Fprintf(os.Stderr, "[D4-G1] M=%d N=%d K=%d (aikit sg_big: %v): %d of %d outputs differ\n", M, N, K, M%32 == 0 && N%32 == 0 && K%8 == 0, diff, M*N)
		if diff != 0 {
			red++
		}
	}
	if red != 0 {
		t.Errorf("tower_gemm_w32 differs from aikit's GEMM in %d of 6 cells: not a bit-identical swap", red)
	}
}
