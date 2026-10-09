//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"
)

// mc3Kernels is MC3's S0 probe (docs/tasks/task-concurrency-2026-09.md), TEST-ONLY: a W4A8 GEMM for a SMALL number of
// activation rows M (2-8 decode sequences), built on R18's shipped rows-per-simdgroup kernel. Each simdgroup computes
// R output rows x M activation rows: per group it loads each of its R weight rows' uint4 and half scale ONCE, and each
// of the M rows' 32 staged activations once, then forms R*M integer group sums. Output (r, m) keeps the M = 1 kernel's
// lane-to-group order, UNP8V integer math and float(gi)*scale accumulation for activation row m, so it is
// bit-identical to gemv_w4a8_sa_rows<R> run on row m alone.
//
// Activations are staged as M*K shorts (M rows back to back) and M per-row scales (per-row activation quantization,
// the decode path's): asc[m]. The question it answers is T(M)/T(1) at real decode shapes: how much of a second, third,
// fourth sequence's GEMV rides free on weights already loaded.
const mc3Kernels = `
template <uint R, uint M>
kernel void gemv_w4a8_sa_mr(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], threadgroup short* As [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<M*K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R][M];
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) { SA_ROWS_UNROLL for (uint m=0;m<M;m++) acc[r][m]=0.0f; }
    for (uint g=lane; g<G; g+=32u) {
        uint4 w[R]; float s[R];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) { w[r] = wq[(row0+r)*G + g]; s[r] = float(sct[(row0+r)*G + g]); }
        SA_ROWS_UNROLL for (uint m=0;m<M;m++) {
            threadgroup const short4* a4s = reinterpret_cast<threadgroup const short4*>(As + m*K + g*32u);
            short4 a[8];
            SA_ROWS_UNROLL for (uint j=0;j<8u;j++) a[j]=a4s[j];
            SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
                int gi = UNP8V(w[r].x,a) + UNP8V(w[r].y,a+2) + UNP8V(w[r].z,a+4) + UNP8V(w[r].w,a+6);
                acc[r][m] += float(gi) * s[r];
            }
        }
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
        SA_ROWS_UNROLL for (uint m=0;m<M;m++) {
            float v = simd_sum(acc[r][m]);
            if (lane==0) out[m*N + row0+r] = v*asc[m];
        }
    }
}
// Variant H (MLX's masked form, shared across the M rows): activations staged as HALF pre-scaled by 16^-(k mod 4) —
// exact, since |a| <= 127 and 127 * 2^-12 is a normal half — plus 8*sum(a) per (row, group). Each weight word is split
// into two ushorts, each nibble MASKED in place (no shift) and converted ONCE, then dotted with every row's four
// pre-scaled activations. Every product n*a and every partial sum is an integer below 2^24, so the group sum equals the
// integer gi exactly whatever the order, and float(gi)*scale accumulates per (r, m) in the shipped lane order ->
// bit-identical to gemv_w4a8_sa_rows.
template <uint R, uint M>
kernel void gemv_w4a8_sa_mrh(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], constant uint& N[[buffer(6)]], threadgroup half* Ah [[threadgroup(0)]],
    uint tgid[[threadgroup_position_in_grid]], uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    uint G = K>>5u;
    threadgroup float* S8 = reinterpret_cast<threadgroup float*>(Ah + M*K); // [M][G] of 8*sum(a)
    constexpr float P4[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};
    for (uint i=tid;i<M*K;i+=tgs) Ah[i] = half(float(aq[i]) * P4[i & 3u]);
    for (uint i=tid;i<M*G;i+=tgs) {
        device const char* ap = aq + (i / G)*K + (i % G)*32u;
        int sa = 0;
        for (uint j=0;j<32u;j++) sa += int(ap[j]);
        S8[i] = float(8*sa);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row0 = (tgid*(tgs>>5u) + sgid)*R;
    float acc[R][M];
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) { SA_ROWS_UNROLL for (uint m=0;m<M;m++) acc[r][m]=0.0f; }
    for (uint g=lane; g<G; g+=32u) {
        uint4 w[R]; float s[R];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) { w[r] = wq[(row0+r)*G + g]; s[r] = float(sct[(row0+r)*G + g]); }
        float gf[R][M];
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) { SA_ROWS_UNROLL for (uint m=0;m<M;m++) gf[r][m] = -S8[m*G + g]; }
        SA_ROWS_UNROLL for (uint t=0;t<8u;t++) {                 // 8 ushorts per group, 4 nibbles each
            float4 x[M];
            SA_ROWS_UNROLL for (uint m=0;m<M;m++) {
                threadgroup const half4* a4 = reinterpret_cast<threadgroup const half4*>(Ah + m*K + g*32u);
                x[m] = float4(a4[t]);
            }
            SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
                uint word = (t < 2u) ? w[r].x : (t < 4u) ? w[r].y : (t < 6u) ? w[r].z : w[r].w;
                uint u = (t & 1u) ? (word >> 16) : (word & 0xFFFFu);
                float4 wf = float4(float(u & 0xFu), float(u & 0xF0u), float(u & 0xF00u), float(u & 0xF000u));
                SA_ROWS_UNROLL for (uint m=0;m<M;m++) gf[r][m] += dot(wf, x[m]);
            }
        }
        SA_ROWS_UNROLL for (uint r=0;r<R;r++) { SA_ROWS_UNROLL for (uint m=0;m<M;m++) acc[r][m] += gf[r][m] * s[r]; }
    }
    SA_ROWS_UNROLL for (uint r=0;r<R;r++) {
        SA_ROWS_UNROLL for (uint m=0;m<M;m++) {
            float v = simd_sum(acc[r][m]);
            if (lane==0) out[m*N + row0+r] = v*asc[m];
        }
    }
}
template [[host_name("gemv_w4a8_sa_mrh_r1_m1")]] kernel decltype(gemv_w4a8_sa_mrh<1,1>) gemv_w4a8_sa_mrh<1,1>;
template [[host_name("gemv_w4a8_sa_mrh_r2_m1")]] kernel decltype(gemv_w4a8_sa_mrh<2,1>) gemv_w4a8_sa_mrh<2,1>;
template [[host_name("gemv_w4a8_sa_mrh_r4_m1")]] kernel decltype(gemv_w4a8_sa_mrh<4,1>) gemv_w4a8_sa_mrh<4,1>;
template [[host_name("gemv_w4a8_sa_mrh_r1_m2")]] kernel decltype(gemv_w4a8_sa_mrh<1,2>) gemv_w4a8_sa_mrh<1,2>;
template [[host_name("gemv_w4a8_sa_mrh_r2_m2")]] kernel decltype(gemv_w4a8_sa_mrh<2,2>) gemv_w4a8_sa_mrh<2,2>;
template [[host_name("gemv_w4a8_sa_mrh_r4_m2")]] kernel decltype(gemv_w4a8_sa_mrh<4,2>) gemv_w4a8_sa_mrh<4,2>;
template [[host_name("gemv_w4a8_sa_mrh_r1_m4")]] kernel decltype(gemv_w4a8_sa_mrh<1,4>) gemv_w4a8_sa_mrh<1,4>;
template [[host_name("gemv_w4a8_sa_mrh_r2_m4")]] kernel decltype(gemv_w4a8_sa_mrh<2,4>) gemv_w4a8_sa_mrh<2,4>;
template [[host_name("gemv_w4a8_sa_mrh_r4_m4")]] kernel decltype(gemv_w4a8_sa_mrh<4,4>) gemv_w4a8_sa_mrh<4,4>;
template [[host_name("gemv_w4a8_sa_mrh_r1_m8")]] kernel decltype(gemv_w4a8_sa_mrh<1,8>) gemv_w4a8_sa_mrh<1,8>;
template [[host_name("gemv_w4a8_sa_mrh_r2_m8")]] kernel decltype(gemv_w4a8_sa_mrh<2,8>) gemv_w4a8_sa_mrh<2,8>;
template [[host_name("gemv_w4a8_sa_mr_r1_m2")]] kernel decltype(gemv_w4a8_sa_mr<1,2>) gemv_w4a8_sa_mr<1,2>;
template [[host_name("gemv_w4a8_sa_mr_r1_m4")]] kernel decltype(gemv_w4a8_sa_mr<1,4>) gemv_w4a8_sa_mr<1,4>;
template [[host_name("gemv_w4a8_sa_mr_r1_m8")]] kernel decltype(gemv_w4a8_sa_mr<1,8>) gemv_w4a8_sa_mr<1,8>;
template [[host_name("gemv_w4a8_sa_mr_r2_m2")]] kernel decltype(gemv_w4a8_sa_mr<2,2>) gemv_w4a8_sa_mr<2,2>;
template [[host_name("gemv_w4a8_sa_mr_r2_m4")]] kernel decltype(gemv_w4a8_sa_mr<2,4>) gemv_w4a8_sa_mr<2,4>;
template [[host_name("gemv_w4a8_sa_mr_r2_m8")]] kernel decltype(gemv_w4a8_sa_mr<2,8>) gemv_w4a8_sa_mr<2,8>;
template [[host_name("gemv_w4a8_sa_mr_r4_m2")]] kernel decltype(gemv_w4a8_sa_mr<4,2>) gemv_w4a8_sa_mr<4,2>;
template [[host_name("gemv_w4a8_sa_mr_r4_m4")]] kernel decltype(gemv_w4a8_sa_mr<4,4>) gemv_w4a8_sa_mr<4,4>;
`

// TestMC3SmallMProbe: EXPLORATORY S0 for MC3. For each SA-family decode shape of the 1.5B and 7B it times, per
// dispatch with the S0 bench's SLC-defeating weight rotation:
//   - the shipped M = 1 kernel (gemv_w4a8_sa_rows<R>, R18) run M times, once per activation row — what M sequences
//     cost today, one GEMV each;
//   - the probe's single M-row dispatch;
//
// after checking every probe output bit for bit against the shipped kernel on the same row.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3SmallMProbe$' -v ./metal/
func TestMC3SmallMProbe(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (allocates ~0.5 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+mc3Kernels, MSL3_1)
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
	cq := d.NewCommandQueue()
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-s0 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(2463534242)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	type shape struct {
		model, cat string
		N, K, R    int // R: the shipped rows-per-simdgroup for this GEMV (R18: 2 for qkv/o, 4 for gate/up)
	}
	shapes := []shape{
		{"1.5B", "qkv", 2048, 1536, 2}, {"1.5B", "o", 1536, 1536, 2}, {"1.5B", "gate/up", 17920, 1536, 4},
		{"7B", "qkv", 4608, 3584, 2}, {"7B", "o", 3584, 3584, 2}, {"7B", "gate/up", 37888, 3584, 4},
	}
	for _, s := range shapes {
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		const maxM = 8
		aqb := d.NewBufferBytes(maxM * s.K)
		av := aqb.Int8s()[:maxM*s.K]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		ascs := make([]float32, maxM)
		for i := range ascs {
			ascs[i] = 0.01 + 0.001*float32(i)
		}
		asc := NewBufferFloats(d, ascs)
		uK, uN := NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N))
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		shipped := pipe(fmt.Sprintf("gemv_w4a8_sa_rows%d", s.R))
		// the shipped kernel reads activation row m through an offset view of aq, and its scale through asc.At(4m)
		shippedRow := func(e *Encoder, c, m int, out Buffer) {
			e.DispatchTG(shipped, s.N*32/s.R, 256, s.K*2, ws[c], ss[c], aqb.At(m*s.K), asc.At(4*m), out.At(4*m*s.N), uK)
		}
		// reference outputs, one shipped dispatch per activation row
		ref := d.NewBufferLen(maxM * s.N)
		e := cq.Begin()
		for m := range maxM {
			shippedRow(e, 0, m, ref)
		}
		e.End()
		refV := append([]float32(nil), ref.Floats()[:maxM*s.N]...)

		per := max(8, 2*copies)
		line := fmt.Sprintf("%-4s %-8s N=%-6d K=%-5d R=%d:", s.model, s.cat, s.N, s.K, s.R)
		t1 := timeIt(per, func(e *Encoder, i int) { shippedRow(e, i%copies, 0, ref) })
		line += fmt.Sprintf("  shipped M=1 %.3f ms", t1)
		for _, M := range []int{2, 4, 8} {
			for _, R := range []int{1, 2, 4} {
				if R == 4 && M == 8 {
					continue
				}
				name := fmt.Sprintf("gemv_w4a8_sa_mr_r%d_m%d", R, M)
				if s.N%(8*R) != 0 || M*s.K*2 > d.MaxThreadgroupMemoryLength() {
					continue
				}
				p := pipe(name)
				out := d.NewBufferLen(M * s.N)
				e := cq.Begin()
				e.DispatchTG(p, s.N*32/R, 256, M*s.K*2, ws[0], ss[0], aqb, asc, out, uK, uN)
				e.End()
				got := out.Floats()[:M*s.N]
				diff := 0
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(refV[i]) {
						diff++
					}
				}
				if diff != 0 {
					t.Errorf("%s %s %s: %d of %d outputs differ from the shipped kernel row by row", s.model, s.cat, name, diff, M*s.N)
				}
				tm := timeIt(per, func(e *Encoder, i int) {
					e.DispatchTG(p, s.N*32/R, 256, M*s.K*2, ws[i%copies], ss[i%copies], aqb, asc, out, uK, uN)
				})
				line += fmt.Sprintf("  M=%d R=%d %.3f ms (T/T1 %.2f, %.2fx vs M sequential)", M, R, tm, tm/t1, float64(M)*t1/tm)
			}
		}
		hb("%s", line)
		line = fmt.Sprintf("%-4s %-8s  H (masked, half-staged):", s.model, s.cat)
		for _, M := range []int{1, 2, 4, 8} {
			for _, R := range []int{1, 2, 4} {
				if R == 4 && M == 8 {
					continue
				}
				shm := M*s.K*2 + M*(s.K/32)*4
				if s.N%(8*R) != 0 || shm > d.MaxThreadgroupMemoryLength() {
					continue
				}
				name := fmt.Sprintf("gemv_w4a8_sa_mrh_r%d_m%d", R, M)
				p := pipe(name)
				out := d.NewBufferLen(M * s.N)
				e := cq.Begin()
				e.DispatchTG(p, s.N*32/R, 256, shm, ws[0], ss[0], aqb, asc, out, uK, uN)
				e.End()
				got := out.Floats()[:M*s.N]
				diff := 0
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(refV[i]) {
						diff++
					}
				}
				if diff != 0 {
					t.Errorf("%s %s %s: %d of %d outputs differ from the shipped kernel row by row", s.model, s.cat, name, diff, M*s.N)
				}
				tm := timeIt(per, func(e *Encoder, i int) {
					e.DispatchTG(p, s.N*32/R, 256, shm, ws[i%copies], ss[i%copies], aqb, asc, out, uK, uN)
				})
				line += fmt.Sprintf("  M=%d R=%d %.3f (T/T1 %.2f, %.2fx)", M, R, tm, tm/t1, float64(M)*t1/tm)
			}
		}
		hb("%s", line)
		d.ReleaseAll()
	}
}

// TestMC3S0PrefillGEMMSmallM: MC3's matrix-unit S0 (docs/tasks/task-concurrency-2026-09.md). The existing prefill GEMM
// (gemm_w4f16_store, the f16 simdgroup_matrix kernel fast prefill uses) timed at M = 1..64 rows on decode shapes,
// against the production decode GEMV (gemv_w4a8_sa_rows, R18b) run M times — what M sequences cost today, one GEMV
// each. Its 64x64 tile runs every token row's MMAs whether or not the row exists (the kernel's own comment), so its
// time should be ~flat up to M = 64: the question is where that flat line sits against M GEMVs, i.e. from which M a
// matrix-unit batched decode pays even before a small-M tile is written. f16 activations: NOT bit-identical to the
// W4A8 decode path (it is fast prefill's numerics) — this measures cost only.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3S0PrefillGEMMSmallM$' -v ./metal/
func TestMC3S0PrefillGEMMSmallM(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (allocates ~0.5 GB of GPU buffers)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	libD, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile decode: %v", err)
	}
	libP, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefill: %v", err)
	}
	gemm, err := d.NewComputePipeline(libP, "gemm_w4f16_store")
	if err != nil {
		t.Fatalf("pipeline gemm_w4f16_store: %v", err)
	}
	cq := d.NewCommandQueue()
	t0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[mc3-s0mma %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(97531)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	med := func(xs []float64) float64 { s := append([]float64(nil), xs...); sort.Float64s(s); return s[len(s)/2] }
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := cq.Begin()
			for i := range per {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 {
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return med(ms)
	}
	type shape struct {
		model, cat string
		N, K, R    int
	}
	for _, s := range []shape{
		{"1.5B", "qkv", 2048, 1536, 4}, {"1.5B", "o", 1536, 1536, 2}, {"1.5B", "gate/up", 17920, 1536, 4},
		{"7B", "qkv", 4608, 3584, 4}, {"7B", "o", 3584, 3584, 2}, {"7B", "gate/up", 37888, 3584, 4},
	} {
		bytes := s.N*s.K/2 + s.N*(s.K/32)*2
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		const maxM = 64
		aq := d.NewBufferBytes(s.K)
		for i := range aq.Int8s()[:s.K] {
			aq.Int8s()[i] = int8(rnd()%255) - 127
		}
		asc := NewBufferFloats(d, []float32{0.01})
		ah := make([]uint16, maxM*s.K)
		for i := range ah {
			ah[i] = f32ToF16(float32(int(rnd()%255)-127) * 0.01)
		}
		A := NewBufferU16s(d, ah)
		C := d.NewBufferLen(maxM * s.N) // half output needs maxM*N*2 bytes; floats give room to spare
		out := d.NewBufferLen(s.N)
		bias := d.NewBufferLen(s.N)
		uK, uN, uMode := NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N)), NewBufferU32(d, 0)
		var ws, ss []Buffer
		for range copies {
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rnd()%1000+1) * 1e-5)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		per := max(8, 2*copies)
		gemv, err := d.NewComputePipeline(libD, fmt.Sprintf("gemv_w4a8_sa_rows%d", s.R))
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		t1 := timeIt(per, func(e *Encoder, i int) {
			e.DispatchTG(gemv, s.N*32/s.R, 256, s.K*2, ws[i%copies], ss[i%copies], aq, asc, out, uK)
		})
		line := fmt.Sprintf("%-4s %-8s N=%-6d K=%-5d  decode GEMV %.3f ms:", s.model, s.cat, s.N, s.K, t1)
		for _, M := range []int{1, 2, 4, 8, 16, 32, 64} {
			uM := NewBufferU32(d, uint32(M))
			gx, gy := (s.N+63)/64, (M+63)/64
			tm := timeIt(per, func(e *Encoder, i int) {
				e.Dispatch2D(gemm, gx, gy, 128, 1, A, ws[i%copies], ss[i%copies], C, uM, uN, uK, bias, uMode)
			})
			line += fmt.Sprintf("  M=%d %.3f (%.2fx M GEMVs)", M, tm, float64(M)*t1/tm)
		}
		hb("%s", line)
		d.ReleaseAll()
	}
}
