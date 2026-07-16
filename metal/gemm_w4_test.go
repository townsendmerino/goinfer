//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// prefillGemmSrc — the core prefill kernel: C[M×N] = A[M×K](f16) · Wᵀ where W[N×K] is the
// resident int4/W4A8 weight (packed nibbles + f16 group scales), dequanted to f16 IN-KERNEL
// (no extra RAM). Blocked: each simdgroup owns RPS row-tiles × one 8-col output tile, dequants
// each 8×8 weight tile once (into a per-simdgroup threadgroup scratch, transposed to Wᵀ) and
// reuses it across the RPS activation tiles via simdgroup_matrix MMA. K stepped by 8; since
// 8|k and a group is 32, an 8-tile never crosses a group boundary → one word + one scale per
// output row per tile.
const prefillGemmSrc = `
#include <metal_stdlib>
using namespace metal;
#define RPS 4
kernel void gemm_w4f16(device const half* A[[buffer(0)]], device const uint* W[[buffer(1)]],
    device const half* WS[[buffer(2)]], device float* C[[buffer(3)]],
    constant uint& M[[buffer(4)]], constant uint& N[[buffer(5)]], constant uint& K[[buffer(6)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint sgpt[[simdgroups_per_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    threadgroup half wscr[8*64];                 // up to 8 simdgroups × 8×8 f16 scratch
    uint sg = tgid*sgpt + sgid;
    uint tilesN = N/8u;
    uint rblk = sg / tilesN, tc = sg % tilesN;
    uint r0 = rblk*RPS;
    if (r0*8u >= M) return;
    uint n0 = tc*8u;
    threadgroup half* scr = wscr + sgid*64u;
    uint wpr = K/8u, gpr = K/32u;
    simdgroup_float8x8 acc[RPS];
    for (uint r=0;r<RPS;r++) acc[r]=make_filled_simdgroup_matrix<float,8,8>(0.0);
    for (uint k=0; k<K; k+=8u) {
        if (lane < 8u) {                          // dequant W[n0+lane][k..k+8] → scr[kl][lane]
            uint nl = lane;
            uint word = W[(n0+nl)*wpr + k/8u];
            float sc = float(WS[(n0+nl)*gpr + k/32u]);
            for (uint kl=0; kl<8u; kl++)
                scr[kl*8u + nl] = half(float(int((word >> (4u*kl)) & 0xF) - 8) * sc);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 b; simdgroup_load(b, scr, 8);         // b[kl][nl] = Wᵀ
        for (uint r=0;r<RPS;r++) {
            if ((r0+r)*8u >= M) break;
            simdgroup_half8x8 a; simdgroup_load(a, A + ((r0+r)*8u)*K + k, K);
            simdgroup_multiply_accumulate(acc[r], a, b, acc[r]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<RPS;r++) {
        if ((r0+r)*8u >= M) break;
        simdgroup_store(acc[r], C + ((r0+r)*8u)*N + n0, N);
    }
}`

func TestPrefillGemmW4(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillGemmSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemm_w4f16")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	// ---- correctness: C[M×N] = A · dequant(W)ᵀ vs CPU on the SAME packed nibbles ----
	const M, K, N = 32, 128, 40
	rng := rand.New(rand.NewSource(3))
	// random weight rows, packed via the validated W4A8 packer; f16 scales.
	words := make([]uint32, N*(K/8))
	scalesH := make([]uint16, N*(K/32))
	deq := make([][]float32, N) // CPU dequant reference (exactly what the kernel reconstructs)
	for n := range N {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(row)
		copy(words[n*(K/8):(n+1)*(K/8)], w)
		dr := make([]float32, K)
		for g := range K / 32 {
			scalesH[n*(K/32)+g] = f32ToF16(s[g])
			sc := f16ToF32(scalesH[n*(K/32)+g])
			for e := range 32 {
				k := g*32 + e
				nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
				dr[k] = af16(float32(nib) * sc) // kernel stores the dequant as f16 (scr is half)
			}
		}
		deq[n] = dr
	}
	A := make([]uint16, M*K)
	Af := make([]float32, M*K)
	for i := range A {
		Af[i] = af16(rng.Float32()*2 - 1)
		A[i] = f32ToF16(Af[i])
	}
	ref := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var acc float64
			for k := range K {
				acc += float64(Af[m*K+k]) * float64(deq[n][k])
			}
			ref[m*N+n] = float32(acc)
		}
	}
	q := d.NewCommandQueue()
	C := d.NewBufferLen(M * N)
	q.Run1D(pipe, ((M+31)/32)*(N/8)*32, 256, d.NewBufferU16s(A), d.NewBufferUint32s(words),
		d.NewBufferU16s(scalesH), C, d.NewBufferU32(M), d.NewBufferU32(N), d.NewBufferU32(K))
	got := C.Floats()
	var dot, na, nb, maxAbs float64
	for i := range ref {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxAbs {
			maxAbs = dd
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.99999 || maxAbs > 0.05 {
		t.Fatalf("prefill GEMM parity FAIL: cos=%.7f maxAbs=%.4f (got[0]=%v ref[0]=%v)", cos, maxAbs, got[0], ref[0])
	}
	t.Logf("prefill int4-dequant MMA GEMM %dx%dx%d vs CPU: cos=%.7f maxAbs=%.4f — PARITY ✓", M, K, N, cos, maxAbs)

	// ---- throughput at gate/up dims vs the single-token int4 GEMV loop ----
	const K2, N2 = 1536, 17920
	mainLib, _ := d.CompileLibrary(allKernels, MSL3_1)
	pSA, _ := d.NewComputePipeline(mainLib, "gemv_w4a8_sa")
	wq := d.NewBufferUint32s(make([]uint32, N2*(K2/8)))
	scb := d.NewBufferU16s(make([]uint16, N2*(K2/32)))
	aq := byteBuf(d, K2)
	asc := d.NewBufferFloats(make([]float32, 1))
	o1 := d.NewBufferLen(N2)
	uK := d.NewBufferU32(K2)
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
	gemv1 := prof(100, func(r int) { q.Run1DBatch(pSA, N2*32, 256, r, wq, scb, aq, asc, o1, uK) })
	W2 := d.NewBufferUint32s(make([]uint32, N2*(K2/8)))
	WS2 := d.NewBufferU16s(make([]uint16, N2*(K2/32)))
	uN2, uK2 := d.NewBufferU32(N2), d.NewBufferU32(K2)
	t.Logf("baseline single-token int4 GEMV gate/up: %.0f us/token", float64(gemv1.Microseconds()))
	for _, mm := range []int{64, 256, 512} {
		Ab := d.NewBufferU16s(make([]uint16, mm*K2))
		Cb := d.NewBufferLen(mm * N2)
		uM := d.NewBufferU32(uint32(mm))
		g := prof(20, func(r int) {
			q.Run1DBatch(pipe, ((mm+31)/32)*(N2/8)*32, 256, r, Ab, W2, WS2, Cb, uM, uN2, uK2)
		})
		t.Logf("M=%3d: prefill GEMM %.1f us/row vs GEMV %.0f => %.1fx faster",
			mm, float64(g.Microseconds())/float64(mm), float64(gemv1.Microseconds()),
			float64(gemv1.Microseconds())/(float64(g.Microseconds())/float64(mm)))
	}
}
