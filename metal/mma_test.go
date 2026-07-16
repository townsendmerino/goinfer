//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// TestMMAPrefill — the make-or-break for the prefill direction. Two questions: (1) does MSL
// simdgroup_matrix (8×8 f16 MMA) compile + run cgo-free at MSL 3.1 via purego? (2) does a tiled
// MMA GEMM amortize the weight read across M prompt rows, so prefill(M) ≪ M × single-GEMV?
// Unlike the int4 decode GEMV (scalar-MAC/issue-bound, no amortization), MMA is a different
// compute path (512 MACs/instr). If this amortizes, a fast prefill is viable.
func TestMMAPrefill(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	// C[M,N] = A[M,K] · B[K,N], f16 in / f32 out. One simdgroup per 8×8 C tile, K-loop of 8×8 MMAs.
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void mma_gemm(device const half* A[[buffer(0)]], device const half* B[[buffer(1)]],
    device float* C[[buffer(2)]], constant uint& M[[buffer(3)]], constant uint& N[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint sgpt[[simdgroups_per_threadgroup]]) {
    uint sg = tgid*sgpt + sgid;
    uint tilesN = N/8u;
    uint tr = sg / tilesN, tc = sg % tilesN;
    if (tr*8u >= M) return;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        simdgroup_half8x8 a, b;
        simdgroup_load(a, A + (tr*8u)*K + k, K);      // A[8×8] at (tr*8, k)
        simdgroup_load(b, B + k*N + tc*8u, N);        // B[8×8] at (k, tc*8)
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C + (tr*8u)*N + tc*8u, N);
}
// Blocked: each simdgroup owns RPS row-tiles (RPS*8 rows) × one col-tile. Per k-step it loads
// ONE B tile and reuses it across all RPS A tiles — cutting weight re-reads RPS×. This is the
// amortization mechanism prefill needs.
#define RPS 4
kernel void mma_gemm_blk(device const half* A[[buffer(0)]], device const half* B[[buffer(1)]],
    device float* C[[buffer(2)]], constant uint& M[[buffer(3)]], constant uint& N[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint tgid[[threadgroup_position_in_grid]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint sgpt[[simdgroups_per_threadgroup]]) {
    uint sg = tgid*sgpt + sgid;
    uint tilesN = N/8u;
    uint rblk = sg / tilesN, tc = sg % tilesN;   // rblk = which RPS-row-tile block
    uint r0 = rblk*RPS;
    if (r0*8u >= M) return;
    simdgroup_float8x8 acc[RPS];
    for (uint r=0;r<RPS;r++) acc[r] = make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        simdgroup_half8x8 b;
        simdgroup_load(b, B + k*N + tc*8u, N);       // ONE weight tile, reused across RPS rows
        for (uint r=0;r<RPS;r++) {
            if ((r0+r)*8u >= M) break;
            simdgroup_half8x8 a;
            simdgroup_load(a, A + ((r0+r)*8u)*K + k, K);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
    }
    for (uint r=0;r<RPS;r++) {
        if ((r0+r)*8u >= M) break;
        simdgroup_store(acc[r], C + ((r0+r)*8u)*N + tc*8u, N);
    }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("MMA compile FAILED (simdgroup_matrix unavailable cgo-free?): %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "mma_gemm")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Log("simdgroup_matrix compiles + pipelines cgo-free at MSL 3.1 ✓")

	// ---- correctness on a small shape ----
	{
		const M, K, N = 16, 32, 24
		rng := rand.New(rand.NewSource(1))
		A := make([]uint16, M*K)
		B := make([]uint16, K*N)
		af := make([]float32, M*K)
		bf := make([]float32, K*N)
		for i := range A {
			af[i] = rng.Float32()*2 - 1
			A[i] = f32ToF16(af[i])
		}
		for i := range B {
			bf[i] = rng.Float32()*2 - 1
			B[i] = f32ToF16(bf[i])
		}
		q := d.NewCommandQueue()
		out := d.NewBufferLen(M * N)
		q.Run1D(pipe, (M/8)*(N/8)*32, 32, d.NewBufferU16s(A), d.NewBufferU16s(B), out,
			d.NewBufferU32(M), d.NewBufferU32(N), d.NewBufferU32(K))
		got := out.Floats()
		var maxErr float64
		for m := range M {
			for n := range N {
				var ref float64
				for k := range K {
					ref += float64(af16(af[m*K+k])) * float64(af16(bf[k*N+n])) // ref in f16-rounded inputs
				}
				if e := math.Abs(ref - float64(got[m*N+n])); e > maxErr {
					maxErr = e
				}
			}
		}
		if maxErr > 0.05 {
			t.Fatalf("MMA correctness FAIL: maxErr=%.4f", maxErr)
		}
		t.Logf("MMA GEMM correctness: maxErr=%.5f vs CPU (f16 inputs) ✓", maxErr)
	}

	// ---- amortization: prefill gate/up shape [M×K]·[K×N], M = prompt length ----
	const K, N = 1536, 17920 // gate/up: K=H, N=2I
	B := d.NewBufferU16s(make([]uint16, K*N))
	q := d.NewCommandQueue()
	prof := func(reps int, run func(int)) time.Duration {
		for range 4 {
			run(reps)
		}
		best := time.Hour
		for range 15 {
			t0 := time.Now()
			run(reps)
			if dt := time.Since(t0); dt < best {
				best = dt
			}
		}
		return best / time.Duration(reps)
	}
	// single-token int4 GEMV baseline (what the current prefill pays per prompt token).
	mainLib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("main lib: %v", err)
	}
	pSA, _ := d.NewComputePipeline(mainLib, "gemv_w4a8_sa")
	wq := d.NewBufferUint32s(make([]uint32, N*(K/8)))
	sc := d.NewBufferU16s(make([]uint16, N*(K/32)))
	aq := byteBuf(d, K)
	asc := d.NewBufferFloats(make([]float32, 1))
	outN := d.NewBufferLen(N)
	uK := d.NewBufferU32(K)
	gemv1 := prof(100, func(r int) { q.Run1DBatch(pSA, N*32, 256, r, wq, sc, aq, asc, outN, uK) })
	t.Logf("single-token int4 GEMV gate/up: %.1f us", float64(gemv1.Microseconds()))

	pBlk, err := d.NewComputePipeline(lib, "mma_gemm_blk")
	if err != nil {
		t.Fatalf("blk pipeline: %v", err)
	}
	uN, uKb := d.NewBufferU32(N), d.NewBufferU32(K)
	for _, M := range []int{8, 64, 256, 512} {
		A := d.NewBufferU16s(make([]uint16, M*K))
		C := d.NewBufferLen(M * N)
		uM := d.NewBufferU32(uint32(M))
		mma := prof(30, func(r int) { q.Run1DBatch(pipe, (M/8)*(N/8)*32, 256, r, A, B, C, uM, uN, uKb) })
		mrows := (M + 31) / 32 // RPS=4 row-tiles per simdgroup → M/32 blocks
		blk := prof(30, func(r int) { q.Run1DBatch(pBlk, mrows*(N/8)*32, 256, r, A, B, C, uM, uN, uKb) })
		naiveRow := float64(mma.Microseconds()) / float64(M)
		blkRow := float64(blk.Microseconds()) / float64(M)
		t.Logf("M=%3d: naive %5.1f us/row | blocked %5.1f us/row | GEMV %.0f us/row => blocked %.1fx faster than the loop",
			M, naiveRow, blkRow, float64(gemv1.Microseconds()), float64(gemv1.Microseconds())/blkRow)
	}
}

// af16 rounds a float32 through f16 (matches what the GPU sees for f16 inputs).
func af16(f float32) float32 { return f16ToF32(f32ToF16(f)) }

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	man := uint32(h & 0x3FF)
	if exp == 0 {
		if man == 0 {
			return math.Float32frombits(sign)
		}
		f := float32(man) / 1024 * float32(math.Exp2(-14))
		if sign != 0 {
			f = -f
		}
		return f
	}
	if exp == 0x1F {
		return math.Float32frombits(sign | 0x7F800000 | man<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | man<<13)
}
