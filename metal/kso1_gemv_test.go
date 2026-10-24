//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
)

// kso1ProtoKernels are O1-G0's prototype native Q5_K and Q6_K GEMVs at R rows a simdgroup (docs/tasks/task-kernel-survey-2026-10.md,
// "KS-O1 on Metal: gates"), appended to allKernels. Q5_K is GGUF's 176-byte block verbatim (d, dmin, scales[12], qh[32],
// qs[128]) on q4k_rows_acc's lane map. Q6_K is a row of NB 210-byte blocks split into planes: NB×128 bytes of ql, NB×64 of
// qh, NB×16 of int8 scales, NB halves of d.
const kso1ProtoKernels = `
template <uint R>
inline void kso1p_q5k_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, j = (lane & 7u) >> 1u, h = lane & 1u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        device const float4* alo4 = (device const float4*)(ax + b*256u + j*64u + h*16u);
        float4 al[4], ah[4];
        float slo = 0.0f, shi = 0.0f;
        for (uint i = 0u; i < 4u; i++) {
            al[i] = alo4[i]; ah[i] = alo4[8u + i];
            slo += (al[i].x + al[i].y) + (al[i].z + al[i].w);
            shi += (ah[i].x + ah[i].y) + (ah[i].z + ah[i].w);
        }
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*176ul + b*176u;
            half2 dd = as_type<half2>(*(device const uint*)blk);
            float sc0, m0, sc1, m1; q4k_scale_min(2u*j, blk + 4, sc0, m0); q4k_scale_min(2u*j + 1u, blk + 4, sc1, m1);
            uint4 q = *(device const uint4*)(blk + 48u + j*32u + h*16u);
            uint4 qh = *(device const uint4*)(blk + 16u + h*16u);
            float dlo = 0.0f, dhi = 0.0f;
            for (uint i = 0u; i < 4u; i++) {
                uint lo = (q[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j)) & 0x01010101u) << 4u);
                uint hi = ((q[i] >> 4u) & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j + 1u)) & 0x01010101u) << 4u);
                dlo += dot(float4(as_type<uchar4>(lo)), al[i]);
                dhi += dot(float4(as_type<uchar4>(hi)), ah[i]);
            }
            float d = float(dd.x), dm = float(dd.y);
            acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
// kso1p_q6k_s6: four 6-bit codes (bytes 0..63) minus 32, as signed bytes, with no borrow between bytes.
inline float4 kso1p_q6k_s6(uint t) { return float4(as_type<char4>(((t | 0x80808080u) - 0x20202020u) ^ 0x80808080u)); }
template <uint R>
inline void kso1p_q6k_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, n = (lane & 7u) >> 2u, l0 = (lane & 3u)*8u, is = (lane & 3u) >> 1u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        device const float4* y4 = (device const float4*)(ax + b*256u + n*128u + l0);
        float4 a[4][2];
        for (uint t = 0u; t < 4u; t++) { a[t][0] = y4[8u*t]; a[t][1] = y4[8u*t + 1u]; }
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            device const uchar* row = wq + (ulong)(row0 + r)*(ulong)NB*210ul;
            uint2 qa = *(device const uint2*)(row + b*128u + n*64u + l0);
            uint2 qb = *(device const uint2*)(row + b*128u + n*64u + 32u + l0);
            uint2 qh = *(device const uint2*)(row + NB*128u + b*64u + n*32u + l0);
            device const char* sc = (device const char*)(row + NB*192u + b*16u + n*8u + is);
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint w = 0u; w < 2u; w++) {
                d0 += dot(kso1p_q6k_s6((qa[w] & 0x0F0F0F0Fu) | ((qh[w] & 0x03030303u) << 4u)), a[0][w]);
                d1 += dot(kso1p_q6k_s6((qb[w] & 0x0F0F0F0Fu) | (((qh[w] >> 2u) & 0x03030303u) << 4u)), a[1][w]);
                d2 += dot(kso1p_q6k_s6(((qa[w] >> 4u) & 0x0F0F0F0Fu) | (((qh[w] >> 4u) & 0x03030303u) << 4u)), a[2][w]);
                d3 += dot(kso1p_q6k_s6(((qb[w] >> 4u) & 0x0F0F0F0Fu) | (((qh[w] >> 6u) & 0x03030303u) << 4u)), a[3][w]);
            }
            float d = float(*(device const half*)(row + NB*208u + b*2u));
            acc[r] += d*((float(sc[0])*d0 + float(sc[2])*d1) + (float(sc[4])*d2 + float(sc[6])*d3));
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void kso1_q5k_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; kso1p_q5k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template <uint R>
kernel void kso1_q6k_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; kso1p_q6k_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
// kso1p_q5kb_rows_acc is O1-G0's amended Q5_K: kso1p_q5k_rows_acc with the scale bytes read as three words and each lane's two scales and
// mins decoded by masks and shifts (no branch); SPLIT splits each dot chain into two halves.
template <uint R, bool SPLIT>
inline void kso1p_q5kb_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, j = (lane & 7u) >> 1u, h = lane & 1u;
    const bool hiw = j >= 2u;
    const uint sh = ((2u*j) & 3u)*8u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        device const float4* alo4 = (device const float4*)(ax + b*256u + j*64u + h*16u);
        float4 al[4], ah[4];
        float slo = 0.0f, shi = 0.0f;
        for (uint i = 0u; i < 4u; i++) {
            al[i] = alo4[i]; ah[i] = alo4[8u + i];
            slo += (al[i].x + al[i].y) + (al[i].z + al[i].w);
            shi += (ah[i].x + ah[i].y) + (ah[i].z + ah[i].w);
        }
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*176ul + b*176u;
            half2 dd = as_type<half2>(*(device const uint*)blk);
            device const uint* sw = (device const uint*)(blk + 4);
            uint s0 = sw[0], s1 = sw[1], s2 = sw[2];
            uint scw = hiw ? ((s2 & 0x0F0F0F0Fu) | ((s0 >> 2u) & 0x30303030u)) : (s0 & 0x3F3F3F3Fu);
            uint mw = hiw ? (((s2 >> 4u) & 0x0F0F0F0Fu) | ((s1 >> 2u) & 0x30303030u)) : (s1 & 0x3F3F3F3Fu);
            float sc0 = float((scw >> sh) & 0xFFu), sc1 = float((scw >> (sh + 8u)) & 0xFFu);
            float m0 = float((mw >> sh) & 0xFFu), m1 = float((mw >> (sh + 8u)) & 0xFFu);
            uint4 q = *(device const uint4*)(blk + 48u + j*32u + h*16u);
            uint4 qh = *(device const uint4*)(blk + 16u + h*16u);
            float dl[2] = {0.0f, 0.0f}, dh[2] = {0.0f, 0.0f};
            for (uint i = 0u; i < 4u; i++) {
                uint lo = (q[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j)) & 0x01010101u) << 4u);
                uint hi = ((q[i] >> 4u) & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j + 1u)) & 0x01010101u) << 4u);
                uint c = SPLIT ? (i >> 1u) : 0u;
                dl[c] += dot(float4(as_type<uchar4>(lo)), al[i]);
                dh[c] += dot(float4(as_type<uchar4>(hi)), ah[i]);
            }
            float dlo = SPLIT ? dl[0] + dl[1] : dl[0], dhi = SPLIT ? dh[0] + dh[1] : dh[0];
            float d = float(dd.x), dm = float(dd.y);
            acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R, bool SPLIT>
kernel void kso1_q5kb_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; kso1p_q5kb_rows_acc<R, SPLIT>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template [[host_name("kso1_q5kb_rows1")]] kernel decltype(kso1_q5kb_rows<1, false>) kso1_q5kb_rows<1, false>;
template [[host_name("kso1_q5kb_rows2")]] kernel decltype(kso1_q5kb_rows<2, false>) kso1_q5kb_rows<2, false>;
template [[host_name("kso1_q5kb_rows4")]] kernel decltype(kso1_q5kb_rows<4, false>) kso1_q5kb_rows<4, false>;
template [[host_name("kso1_q5kb_rows8")]] kernel decltype(kso1_q5kb_rows<8, false>) kso1_q5kb_rows<8, false>;
template [[host_name("kso1_q5kc_rows1")]] kernel decltype(kso1_q5kb_rows<1, true>) kso1_q5kb_rows<1, true>;
template [[host_name("kso1_q5kc_rows2")]] kernel decltype(kso1_q5kb_rows<2, true>) kso1_q5kb_rows<2, true>;
template [[host_name("kso1_q5kc_rows4")]] kernel decltype(kso1_q5kb_rows<4, true>) kso1_q5kb_rows<4, true>;
template [[host_name("kso1_q5kc_rows8")]] kernel decltype(kso1_q5kb_rows<8, true>) kso1_q5kb_rows<8, true>;
// kso1p_q4kb_rows_acc is the shipped q4k_rows_acc with kso1p_q5kb_rows_acc's branchless scale decode, for an ungraded side reading.
template <uint R>
inline void kso1p_q4kb_rows_acc(device const uchar* wq, device const float* ax, uint K, uint N, uint row0, uint lane,
    thread float* acc) {
#pragma clang fp reassociate(off)
    const uint NB = K >> 8u, ix = lane >> 3u, j = (lane & 7u) >> 1u, h = lane & 1u;
    const bool hiw = j >= 2u;
    const uint sh = ((2u*j) & 3u)*8u;
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = 0.0f;
    for (uint b = ix; b < NB; b += 4u) {
        device const float4* alo4 = (device const float4*)(ax + b*256u + j*64u + h*16u);
        float4 al[4], ah[4];
        float slo = 0.0f, shi = 0.0f;
        for (uint i = 0u; i < 4u; i++) {
            al[i] = alo4[i]; ah[i] = alo4[8u + i];
            slo += (al[i].x + al[i].y) + (al[i].z + al[i].w);
            shi += (ah[i].x + ah[i].y) + (ah[i].z + ah[i].w);
        }
        SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) {
            if (row0 + r >= N) break;
            device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*144ul + b*144u;
            half2 dd = as_type<half2>(*(device const uint*)blk);
            device const uint* sw = (device const uint*)(blk + 4);
            uint s0 = sw[0], s1 = sw[1], s2 = sw[2];
            uint scw = hiw ? ((s2 & 0x0F0F0F0Fu) | ((s0 >> 2u) & 0x30303030u)) : (s0 & 0x3F3F3F3Fu);
            uint mw = hiw ? (((s2 >> 4u) & 0x0F0F0F0Fu) | ((s1 >> 2u) & 0x30303030u)) : (s1 & 0x3F3F3F3Fu);
            float sc0 = float((scw >> sh) & 0xFFu), sc1 = float((scw >> (sh + 8u)) & 0xFFu);
            float m0 = float((mw >> sh) & 0xFFu), m1 = float((mw >> (sh + 8u)) & 0xFFu);
            uint4 q = *(device const uint4*)(blk + 16u + j*32u + h*16u);
            float dlo = 0.0f, dhi = 0.0f;
            for (uint i = 0u; i < 4u; i++) {
                uint w = q[i];
                dlo += dot(float4(float(w & 15u), float((w >> 8u) & 15u), float((w >> 16u) & 15u), float((w >> 24u) & 15u)), al[i]);
                dhi += dot(float4(float((w >> 4u) & 15u), float((w >> 12u) & 15u), float((w >> 20u) & 15u), float(w >> 28u)), ah[i]);
            }
            float d = float(dd.x), dm = float(dd.y);
            acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);
        }
    }
    SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) acc[r] = simd_sum(acc[r]);
}
template <uint R>
kernel void kso1_q4kb_rows(device const uchar* wq[[buffer(0)]], device const float* ax[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]], constant uint& N[[buffer(4)]],
    uint tgid[[threadgroup_position_in_grid]], uint sgid[[simdgroup_index_in_threadgroup]],
    uint lane[[thread_index_in_simdgroup]]) {
    uint row0 = (tgid*8u + sgid)*R;
    if (row0 >= N) return;
    float acc[R]; kso1p_q4kb_rows_acc<R>(wq, ax, K, N, row0, lane, acc);
    if (lane==0) { SA_ROWS_UNROLL for (uint r = 0u; r < R; r++) if (row0 + r < N) out[row0 + r] = acc[r]; }
}
template [[host_name("kso1_q4kb_rows4")]] kernel decltype(kso1_q4kb_rows<4>) kso1_q4kb_rows<4>;
template [[host_name("kso1_q4kb_rows8")]] kernel decltype(kso1_q4kb_rows<8>) kso1_q4kb_rows<8>;
template [[host_name("kso1_q5k_rows1")]] kernel decltype(kso1_q5k_rows<1>) kso1_q5k_rows<1>;
template [[host_name("kso1_q5k_rows2")]] kernel decltype(kso1_q5k_rows<2>) kso1_q5k_rows<2>;
template [[host_name("kso1_q5k_rows4")]] kernel decltype(kso1_q5k_rows<4>) kso1_q5k_rows<4>;
template [[host_name("kso1_q5k_rows8")]] kernel decltype(kso1_q5k_rows<8>) kso1_q5k_rows<8>;
template [[host_name("kso1_q6k_rows1")]] kernel decltype(kso1_q6k_rows<1>) kso1_q6k_rows<1>;
template [[host_name("kso1_q6k_rows2")]] kernel decltype(kso1_q6k_rows<2>) kso1_q6k_rows<2>;
template [[host_name("kso1_q6k_rows4")]] kernel decltype(kso1_q6k_rows<4>) kso1_q6k_rows<4>;
template [[host_name("kso1_q6k_rows8")]] kernel decltype(kso1_q6k_rows<8>) kso1_q6k_rows<8>;
`

// kso1Q5KRaw is n rows of random Q5_K blocks over K: d and dmin small positive f16, random scales, qh and qs.
func kso1Q5KRaw(rng *rand.Rand, n, K int) []byte {
	raw := make([]byte, n*(K/256)*176)
	for b := 0; b < len(raw); b += 176 {
		d, dm := f32ToF16(float32(0.002+0.01*rng.Float64())), f32ToF16(float32(0.001+0.005*rng.Float64()))
		raw[b], raw[b+1], raw[b+2], raw[b+3] = byte(d), byte(d>>8), byte(dm), byte(dm>>8)
		for i := b + 4; i < b+176; i++ {
			raw[i] = byte(rng.Intn(256))
		}
	}
	return raw
}

// kso1Q6KPlanar is n rows of random Q6_K blocks over K in the planar row layout (ql, qh, scales, d planes): random codes,
// int8 scales, small positive f16 d.
func kso1Q6KPlanar(rng *rand.Rand, n, K int) []byte {
	NB := K / 256
	raw := make([]byte, n*NB*210)
	for r := range n {
		row := raw[r*NB*210 : (r+1)*NB*210]
		for i := range NB * 208 {
			row[i] = byte(rng.Intn(256))
		}
		for b := range NB {
			d := f32ToF16(float32(0.0005 + 0.002*rng.Float64()))
			row[NB*208+2*b], row[NB*208+2*b+1] = byte(d), byte(d>>8)
		}
	}
	return raw
}

// TestKSO1_gemvTiming is O1-G0: the prototype Q5_K and Q6_K rows kernels at R = 1, 2, 4, 8 against gemv_w8_f32_rows2 (the int8
// copy they would replace) at Phi-3 mini's shapes, the LM head also against gemv_w8f32_head, beside a streaming-read ceiling;
// C1-G0's method (weights rotated over >= 256 MB of copies, back-to-back dispatches timed by GPU timestamps, median of 7). The
// kill rule sums each kind over Phi-3's tensors (32 qkv; 16 down + the head). A timed run: hold the timing lock
// (GOINFER_KSO1_TIMING=1).
func TestKSO1_gemvTiming(t *testing.T) {
	if os.Getenv("GOINFER_KSO1_TIMING") == "" {
		t.Skip("set GOINFER_KSO1_TIMING=1 (a timed run under the timing lock)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels+kso1ProtoKernels+gemvS0Kernels[len("\n#include <metal_stdlib>\nusing namespace metal;\n"):], MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	q := d.NewCommandQueue()
	const reps = 7
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := range reps + 1 {
			e := q.Begin()
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
		sort.Float64s(ms)
		return ms[len(ms)/2]
	}
	{
		const n4 = 512 << 20 >> 4
		buf := d.NewBufferLen(n4 * 4)
		out, un4 := d.NewBufferLen(4), NewBufferU32(d, n4)
		ms := timeIt(4, func(e *Encoder, _ int) { e.Dispatch(pipe("s0_stream"), 1048576, 256, buf, out, un4) })
		fmt.Fprintf(os.Stderr, "[O1-G0] streaming read, 512 MB: %.3f ms -> %.1f GB/s\n", ms, float64(n4*16)/ms/1e6)
		d.ReleaseAll()
	}
	rng := rand.New(rand.NewSource(61))
	type sum struct {
		w8   float64
		best [3][4]float64
	}
	sums := map[string]*sum{"Q5_K": {}, "Q6_K": {}}
	for _, s := range []struct {
		name, kind  string
		N, K, count int
	}{{"qkv", "Q5_K", 9216, 3072, 32}, {"down (Q6_K layers)", "Q6_K", 3072, 8192, 16}, {"head", "Q6_K", 32064, 3072, 1}} {
		nbytes, w8bytes := s.N*(s.K/256)*176, s.N*s.K
		if s.kind == "Q6_K" {
			nbytes = s.N * (s.K / 256) * 210
		}
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(w8bytes)))))
		var nat, w8, w8s []Buffer
		for range copies {
			raw := kso1Q5KRaw(rng, s.N, s.K)
			if s.kind == "Q6_K" {
				raw = kso1Q6KPlanar(rng, s.N, s.K)
			}
			nb := d.NewBufferBytes(len(raw))
			iv := nb.Int8s()[:len(raw)]
			for i, c := range raw {
				iv[i] = int8(c)
			}
			wb := d.NewBufferBytes(w8bytes)
			wv := wb.U32s()[:w8bytes/4]
			for i := range wv {
				wv[i] = rng.Uint32()
			}
			sc := make([]float32, s.N)
			for i := range sc {
				sc[i] = 0.001
			}
			nat, w8, w8s = append(nat, nb), append(w8, wb), append(w8s, NewBufferFloats(d, sc))
		}
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		ba, out, uK, uN := NewBufferFloats(d, a), d.NewBufferLen(s.N), NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N))
		per := max(8, 2*copies)
		line := fmt.Sprintf("%-18s N=%-5d K=%-5d native %6.1f MB, int8 %6.1f MB:", s.name, s.N, s.K, float64(nbytes)/1e6, float64(w8bytes)/1e6)
		p2 := pipe("gemv_w8_f32_rows2")
		old := timeIt(per, func(e *Encoder, i int) {
			e.Dispatch(p2, (s.N+1)/2*32, 256, w8[i%copies], w8s[i%copies], ba, out, uK, uN)
		})
		line += fmt.Sprintf("  w8_rows2 %.3f ms (%.0f GB/s)", old, float64(w8bytes)/1e6/old)
		if s.name == "head" {
			ph := pipe("gemv_w8f32_head")
			hm := timeIt(per, func(e *Encoder, i int) {
				e.Dispatch(ph, s.N*32, 32, ba, ba, w8[i%copies], w8s[i%copies], out, uK)
			})
			line += fmt.Sprintf("  w8f32_head %.3f ms (%.0f GB/s)", hm, float64(w8bytes)/1e6/hm)
		}
		sm := sums[s.kind]
		vars := []string{"q6k"}
		if s.kind == "Q5_K" {
			vars = []string{"q5k", "q5kb", "q5kc"}
		}
		for vi, v := range vars {
			for ri, R := range []int{1, 2, 4, 8} {
				p := pipe(fmt.Sprintf("kso1_%s_rows%d", v, R))
				ms := timeIt(per, func(e *Encoder, i int) {
					e.Dispatch(p, (s.N+R-1)/R*32, 256, nat[i%copies], ba, out, uK, uN)
				})
				sm.best[vi][ri] += float64(s.count) * ms
				line += fmt.Sprintf("  %s R=%d %.3f ms (%.0f GB/s, %.2fx)", v, R, ms, float64(nbytes)/1e6/ms, old/ms)
			}
		}
		sm.w8 += float64(s.count) * old
		fmt.Fprintf(os.Stderr, "[O1-G0] %s\n", line)
		d.ReleaseAll()
	}
	for _, k := range []string{"Q5_K", "Q6_K"} {
		sm := sums[k]
		line := fmt.Sprintf("%s summed over Phi-3: w8_rows2 %.3f ms;", k, sm.w8)
		for vi, v := range map[string][]string{"Q5_K": {"q5k", "q5kb", "q5kc"}, "Q6_K": {"q6k"}}[k] {
			for ri, R := range []int{1, 2, 4, 8} {
				line += fmt.Sprintf("  %s R=%d %.3f ms (%.3f of int8)", v, R, sm.best[vi][ri], sm.best[vi][ri]/sm.w8)
			}
		}
		fmt.Fprintf(os.Stderr, "[O1-G0] %s\n", line)
	}
}

// kso1ScaleMin is ggml's get_scale_min_k4 on a block's 12 scale bytes.
func kso1ScaleMin(j int, q []byte) (sc, m float64) {
	if j < 4 {
		return float64(q[j] & 63), float64(q[j+4] & 63)
	}
	return float64(q[j+4]&15 | (q[j-4]>>6)<<4), float64(q[j+4]>>4 | (q[j]>>6)<<4)
}

// kso1DequantQ5K is one row of GGUF Q5_K blocks over K in float64, ggml's dequantize_row_q5_K.
func kso1DequantQ5K(row []byte, K int) []float64 {
	y := make([]float64, K)
	for b := range K / 256 {
		blk := row[b*176 : (b+1)*176]
		d, dm := float64(f16ToF32(uint16(blk[0])|uint16(blk[1])<<8)), float64(f16ToF32(uint16(blk[2])|uint16(blk[3])<<8))
		qh, qs := blk[16:48], blk[48:176]
		for j := range 4 {
			sc0, m0 := kso1ScaleMin(2*j, blk[4:16])
			sc1, m1 := kso1ScaleMin(2*j+1, blk[4:16])
			for l := range 32 {
				lo, hi := float64(qs[32*j+l]&15), float64(qs[32*j+l]>>4)
				if qh[l]&(1<<(2*j)) != 0 {
					lo += 16
				}
				if qh[l]&(2<<(2*j)) != 0 {
					hi += 16
				}
				y[256*b+64*j+l] = d*sc0*lo - dm*m0
				y[256*b+64*j+32+l] = d*sc1*hi - dm*m1
			}
		}
	}
	return y
}

// kso1Q6KToPlanar converts one row of GGUF Q6_K blocks (ql[128], qh[64], scales[16], d) into the planar row layout.
func kso1Q6KToPlanar(row []byte, K int) []byte {
	NB := K / 256
	out := make([]byte, NB*210)
	for b := range NB {
		blk := row[b*210 : (b+1)*210]
		copy(out[b*128:], blk[:128])
		copy(out[NB*128+b*64:], blk[128:192])
		copy(out[NB*192+b*16:], blk[192:208])
		copy(out[NB*208+b*2:], blk[208:210])
	}
	return out
}

// kso1DequantQ6K is one row of GGUF Q6_K blocks over K in float64, ggml's dequantize_row_q6_K.
func kso1DequantQ6K(row []byte, K int) []float64 {
	y := make([]float64, K)
	for b := range K / 256 {
		blk := row[b*210 : (b+1)*210]
		d := float64(f16ToF32(uint16(blk[208]) | uint16(blk[209])<<8))
		for n := range 2 {
			ql, qh, sc := blk[64*n:], blk[128+32*n:], blk[192+8*n:]
			for l := range 32 {
				is := l / 16
				q := [4]int{int(ql[l]&15 | (qh[l]&3)<<4), int(ql[l+32]&15 | (qh[l]>>2&3)<<4), int(ql[l]>>4 | (qh[l]>>4&3)<<4), int(ql[l+32]>>4 | (qh[l]>>6&3)<<4)}
				for t := range 4 {
					y[256*b+128*n+32*t+l] = d * float64(int8(sc[is+2*t])) * float64(q[t]-32)
				}
			}
		}
	}
	return y
}

// TestKSO1_protoMatchesRef checks O1-G0's prototypes compute the GEMV before they are timed: every row within G-Q1's bar
// (1e-5 of sum|w*a|) of the float64 reference, at Phi-3's qkv K and a partial threadgroup.
func TestKSO1_protoMatchesRef(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels+kso1ProtoKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(67))
	const N, K = 37, 3072
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for _, kind := range []string{"q5k", "q5kb", "q5kc", "q6k"} {
		var raw, up []byte
		var ref func(row []byte, K int) []float64
		rb := K / 256 * 176
		if kind != "q6k" {
			raw, ref = kso1Q5KRaw(rng, N, K), kso1DequantQ5K
			up = raw
		} else {
			rb = K / 256 * 210
			raw = make([]byte, N*rb)
			for i := range raw {
				raw[i] = byte(rng.Intn(256))
			}
			for r := range N {
				for b := range K / 256 {
					h := f32ToF16(float32(0.0005 + 0.002*rng.Float64()))
					raw[r*rb+b*210+208], raw[r*rb+b*210+209] = byte(h), byte(h>>8)
				}
				up = append(up, kso1Q6KToPlanar(raw[r*rb:(r+1)*rb], K)...)
			}
			ref = kso1DequantQ6K
		}
		wb := d.NewBufferBytes(len(up))
		iv := wb.Int8s()[:len(up)]
		for i, c := range up {
			iv[i] = int8(c)
		}
		ba, uK, uN := NewBufferFloats(d, a), NewBufferU32(d, K), NewBufferU32(d, N)
		for _, R := range []int{1, 2, 4, 8} {
			p, err := d.NewComputePipeline(lib, fmt.Sprintf("kso1_%s_rows%d", kind, R))
			if err != nil {
				t.Fatal(err)
			}
			out := d.NewBufferLen(N)
			e := q.Begin()
			e.Dispatch(p, (N+R-1)/R*32, 256, wb, ba, out, uK, uN)
			e.End()
			worst := 0.0
			for r := range N {
				w := ref(raw[r*rb:(r+1)*rb], K)
				var want, mag float64
				for i := range K {
					want += w[i] * float64(a[i])
					mag += math.Abs(w[i] * float64(a[i]))
				}
				worst = max(worst, math.Abs(float64(out.Floats()[r])-want)/mag)
			}
			fmt.Fprintf(os.Stderr, "[O1-G0 check] %s R=%d worst |err|/sum|w*a| %.2e\n", kind, R, worst)
			if worst > 1e-5 {
				t.Errorf("%s R=%d: worst %.2e over 1e-5", kind, R, worst)
			}
		}
	}
}

// TestKSO1_q4kSideReading is an ungraded side reading beside O1-G0: the shipped gemv_q4k_f32_rows8 against the same loop with
// q5kb's branchless scale decode (kso1_q4kb_rows) at Phi-3's Q4_K shapes, C1-G0's method, and how many outputs differ in bits.
// A timed run: hold the timing lock (GOINFER_KSO1_TIMING=1).
func TestKSO1_q4kSideReading(t *testing.T) {
	if os.Getenv("GOINFER_KSO1_TIMING") == "" {
		t.Skip("set GOINFER_KSO1_TIMING=1 (a timed run under the timing lock)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels+kso1ProtoKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(71))
	var sumOld, sumNew [2]float64
	for _, s := range []struct {
		name        string
		N, K, count int
	}{{"o", 3072, 3072, 32}, {"gate|up", 16384, 3072, 32}, {"down (Q4_K layers)", 3072, 8192, 16}} {
		bytes := s.N * (s.K / 256) * 144
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		var ws []Buffer
		for range copies {
			raw := ksc1Q4KRaw(rng, s.N, s.K)
			wb := d.NewBufferBytes(len(raw))
			iv := wb.Int8s()[:len(raw)]
			for i, c := range raw {
				iv[i] = int8(c)
			}
			ws = append(ws, wb)
		}
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		ba, uK, uN := NewBufferFloats(d, a), NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N))
		run := func(name string, R int) []uint32 {
			out := d.NewBufferLen(s.N)
			e := q.Begin()
			e.Dispatch(pipe(name), (s.N+R-1)/R*32, 256, ws[0], ba, out, uK, uN)
			e.End()
			return append([]uint32(nil), out.U32s()[:s.N]...)
		}
		ref := run("gemv_q4k_f32_rows8", 8)
		line := fmt.Sprintf("%-18s N=%-5d K=%-5d %6.1f MB:", s.name, s.N, s.K, float64(bytes)/1e6)
		out := d.NewBufferLen(s.N)
		per := max(8, 2*copies)
		for ki, k := range []struct {
			name string
			R    int
		}{{"gemv_q4k_f32_rows8", 8}, {"kso1_q4kb_rows8", 8}, {"kso1_q4kb_rows4", 4}} {
			p := pipe(k.name)
			var ms []float64
			for r := range 8 {
				e := q.Begin()
				for i := range per {
					e.Dispatch(p, (s.N+k.R-1)/k.R*32, 256, ws[i%copies], ba, out, uK, uN)
				}
				e.End()
				if r > 0 {
					ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
				}
			}
			sort.Float64s(ms)
			m := ms[len(ms)/2]
			diff := 0
			for i, v := range run(k.name, k.R) {
				if v != ref[i] {
					diff++
				}
			}
			if ki == 0 {
				sumOld[0] += float64(s.count) * m
			} else {
				sumNew[ki-1] += float64(s.count) * m
			}
			line += fmt.Sprintf("  %s %.3f ms (%.0f GB/s, %d of %d bits differ)", k.name, m, float64(bytes)/1e6/m, diff, s.N)
		}
		fmt.Fprintf(os.Stderr, "[O1 side] %s\n", line)
		d.ReleaseAll()
	}
	fmt.Fprintf(os.Stderr, "[O1 side] Q4_K summed over Phi-3: shipped R=8 %.3f ms; q4kb R=8 %.3f (%.3f); q4kb R=4 %.3f (%.3f)\n",
		sumOld[0], sumNew[0], sumNew[0]/sumOld[0], sumNew[1], sumNew[1]/sumOld[0])
}
