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

// gemvS0Kernels are LOADS-ONLY twins of the production decode GEMVs, for the fixed-cost scoping of Metal decode
// (docs/measurements/metal-decode-gemv-s0-2026-09-26.md). Each reads exactly what its production kernel reads from
// device memory, in the same order and with the same thread mapping — weight words, group scales, and (coal)
// the activation bytes — but replaces the nibble unpack and multiply-accumulate with a cheap fold, so the gap
// between a production kernel and its twin is the cost of the arithmetic, not of the memory traffic. s0_stream is
// a plain coalesced streaming read: this machine's practical read ceiling for a single dispatch.
const gemvS0Kernels = `
#include <metal_stdlib>
using namespace metal;

// twin of gemv_w4a8_sa: activation staged in threadgroup memory, one simdgroup per row, lane-strided uint4 groups
kernel void s0_sa_loads(device const uint4* wq[[buffer(0)]], device const half* sct[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], threadgroup short* As [[threadgroup(0)]], uint tgid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]],
    uint sgid[[simdgroup_index_in_threadgroup]], uint lane[[thread_index_in_simdgroup]]) {
    for (uint i=tid;i<K;i+=tgs) As[i]=short(aq[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint G = K>>5u;
    uint row = tgid*(tgs>>5u) + sgid;
    device const uint4* wr = wq + (uint)row*G;
    device const half*  sr = sct + (uint)row*G;
    uint x = 0u; float s = 0.0f;
    for (uint g=lane; g<G; g+=32u) { uint4 w = wr[g]; x ^= w.x ^ w.y ^ w.z ^ w.w; s += float(sr[g]); }
    float acc = simd_sum(float(x & 0xFFFFu) + s + float(As[lane]));
    if (lane==0) out[row] = acc*asc[0];
}

// twin of gemv_w4a8_coal/_resid (W4A8_BODY): one simdgroup per row, lane-strided words, 8 activation bytes and the
// group scale re-read from device per word
kernel void s0_coal_loads(device const uint* bq[[buffer(0)]], device const half* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    uint wpr = K/8u;
    device const uint* brow = bq + (uint)gid*wpr;
    device const half* srow = bsc + (uint)gid*(K/32u);
    uint x = 0u; int a8 = 0; float s = 0.0f;
    for (uint wi = lid; wi < wpr; wi += 32u) {
        x ^= brow[wi];
        device const char* a = aq + wi*8u;
        a8 += int(a[0]) + int(a[1]) + int(a[2]) + int(a[3]) + int(a[4]) + int(a[5]) + int(a[6]) + int(a[7]);
        s += float(srow[wi>>2]);
    }
    float acc = simd_sum(float(x & 0xFFFFu) + float(a8) + s);
    if (lid == 0) out[gid] += acc*asc[0];
}

// a plain coalesced streaming read of n4 uint4s: each thread folds the elements i, i+nthreads, ...
kernel void s0_stream(device const uint4* p[[buffer(0)]], device uint* out[[buffer(1)]], constant uint& n4[[buffer(2)]],
    uint gid[[thread_position_in_grid]], uint nth[[threads_per_grid]]) {
    uint x = 0u;
    for (uint i = gid; i < n4; i += nth) { uint4 v = p[i]; x ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (x == 0x9E3779B9u) out[0] = x; // almost never true; keeps the loads live
}
`

// TestMetalGEMVS0 times each decode GEMV category at its REAL shape, standalone: the production kernel, its
// loads-only twin, and the int8 LM head kernel, plus a streaming-read ceiling. Weights are random and rotated over
// enough copies (>= 256 MB) that no dispatch finds its matrix in the system-level cache, as in real decode, where every
// layer's matrix is distinct. Time per dispatch = a command buffer of back-to-back dispatches, GPU timestamps,
// median of 7.
//
//	GOINFER_METAL_GEMV_S0=1 go test -run '^TestMetalGEMVS0$' -v ./metal/
func TestMetalGEMVS0(t *testing.T) {
	if os.Getenv("GOINFER_METAL_GEMV_S0") != "1" {
		t.Skip("set GOINFER_METAL_GEMV_S0=1 (allocates ~0.5-1 GB of GPU buffers; a minute or two of GPU time)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels+"\n"+gemvS0Kernels[len("\n#include <metal_stdlib>\nusing namespace metal;\n"):], MSL3_1)
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
		fmt.Fprintf(os.Stderr, "[gemv-s0 %6.1fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(format, a...))
	}
	seed := uint32(2463534242)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const reps = 7
	median := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	// timeIt runs `per` dispatches per command buffer via enc, reps times, and returns the median ms per dispatch.
	timeIt := func(per int, enc func(e *Encoder, i int)) float64 {
		var ms []float64
		for r := 0; r < reps+1; r++ {
			e := cq.Begin()
			for i := 0; i < per; i++ {
				enc(e, i)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if r > 0 { // rep 0 warms
				ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(per))
			}
		}
		return median(ms)
	}

	// streaming ceiling
	{
		const n4 = 512 << 20 >> 4 // 512 MB
		buf := d.NewBufferLen(n4 * 4)
		w := buf.U32s()[:n4*4]
		for i := range w {
			w[i] = rnd()
		}
		out, un4 := d.NewBufferLen(1), NewBufferU32(d, n4)
		p := pipe("s0_stream")
		for _, threads := range []int{1 << 16, 1 << 18, 1 << 20} {
			ms := timeIt(4, func(e *Encoder, _ int) { e.Dispatch(p, threads, 256, buf, out, un4) })
			hb("streaming read, 512 MB, %7d threads: %.3f ms -> %.1f GB/s", threads, ms, float64(n4*16)/ms/1e6)
		}
		d.ReleaseAll()
	}

	type shape struct {
		model, cat string
		fam        string // "sa" (gemv_w4a8_sa family), "coal" (gemv_w4a8_resid), "w8" (gemv_w8a8_coal)
		N, K       int
	}
	shapes := []shape{
		{"1.5B", "qkv", "sa", 2048, 1536}, {"1.5B", "o", "sa", 1536, 1536}, {"1.5B", "gate/up", "sa", 17920, 1536},
		{"1.5B", "down", "coal", 1536, 8960}, {"1.5B", "LM head", "w8", 151936, 1536},
		{"7B", "qkv", "sa", 4608, 3584}, {"7B", "o", "sa", 3584, 3584}, {"7B", "gate/up", "sa", 37888, 3584},
		{"7B", "down", "coal", 3584, 18944}, {"7B", "LM head", "w8", 152064, 3584},
	}
	type res struct {
		s                shape
		mb, prodMs, ldMs float64
	}
	var results []res
	for _, s := range shapes {
		var bytes int // weight bytes read per dispatch
		if s.fam == "w8" {
			bytes = s.N*s.K + s.N*4
		} else {
			bytes = s.N*s.K/2 + s.N*(s.K/32)*2
		}
		copies := max(1, int(math.Ceil(float64(256<<20)/float64(bytes))))
		copies = min(copies, 64)
		aqb := d.NewBufferBytes(s.K * 10)
		av := aqb.Int8s()[:s.K*10]
		for i := range av {
			av[i] = int8(rnd()%255) - 127
		}
		asc := NewBufferFloats(d, []float32{0.01})
		out := d.NewBufferLen(s.N)
		uK := NewBufferU32(d, uint32(s.K))
		var ws, ss []Buffer
		for c := 0; c < copies; c++ {
			if s.fam == "w8" {
				wb := d.NewBufferBytes(s.N * s.K)
				v := wb.U32s()[:s.N*s.K/4]
				for i := range v {
					v[i] = rnd()
				}
				sc := make([]float32, s.N)
				for i := range sc {
					sc[i] = 0.001
				}
				ws, ss = append(ws, wb), append(ss, NewBufferFloats(d, sc))
				continue
			}
			nw := s.N * s.K / 8
			wb := d.NewBufferLen(nw)
			v := wb.U32s()[:nw]
			for i := range v {
				v[i] = rnd()
			}
			sc := make([]uint16, s.N*(s.K/32))
			for i := range sc {
				sc[i] = f32ToF16(0.001)
			}
			ws, ss = append(ws, wb), append(ss, NewBufferU16s(d, sc))
		}
		per := max(8, 2*copies)
		var prod, ld float64
		switch s.fam {
		case "sa":
			pp, lp := pipe("gemv_w4a8_sa"), pipe("s0_sa_loads")
			prod = timeIt(per, func(e *Encoder, i int) {
				e.DispatchTG(pp, s.N*32, 256, s.K*2, ws[i%copies], ss[i%copies], aqb, asc, out, uK)
			})
			ld = timeIt(per, func(e *Encoder, i int) {
				e.DispatchTG(lp, s.N*32, 256, s.K*2, ws[i%copies], ss[i%copies], aqb, asc, out, uK)
			})
		case "coal":
			pp, lp := pipe("gemv_w4a8_resid"), pipe("s0_coal_loads")
			prod = timeIt(per, func(e *Encoder, i int) {
				e.Dispatch(pp, s.N*32, 32, ws[i%copies], ss[i%copies], aqb, asc, out, uK)
			})
			ld = timeIt(per, func(e *Encoder, i int) {
				e.Dispatch(lp, s.N*32, 32, ws[i%copies], ss[i%copies], aqb, asc, out, uK)
			})
		case "w8":
			pp := pipe("gemv_w8a8_coal")
			prod = timeIt(per, func(e *Encoder, i int) {
				e.Dispatch(pp, s.N*32, 32, aqb, asc, ws[i%copies], ss[i%copies], out, uK)
			})
		}
		mb := float64(bytes) / 1e6
		results = append(results, res{s, mb, prod, ld})
		ldStr := "—"
		if ld > 0 {
			ldStr = fmt.Sprintf("%.3f ms (%.0f GB/s)", ld, mb/ld)
		}
		hb("%-4s %-8s %-4s N=%-6d K=%-5d %7.1f MB x%-2d copies: production %.3f ms (%.0f GB/s) | loads-only %s",
			s.model, s.cat, s.fam, s.N, s.K, mb, copies, prod, mb/prod, ldStr)
		d.ReleaseAll() // this shape's copies, activations and uniforms; the next shape allocates its own
	}
	fmt.Fprintf(os.Stderr, "\n=== decode GEMV S0 (standalone, real shapes, SLC-defeating rotation, median of %d) ===\n", reps)
	fmt.Fprintf(os.Stderr, "  %-4s %-8s %-5s %9s %12s %9s %14s %9s %s\n", "", "category", "kern", "MB", "production", "GB/s", "loads-only", "GB/s", "production / loads-only")
	for _, r := range results {
		if r.ldMs > 0 {
			fmt.Fprintf(os.Stderr, "  %-4s %-8s %-5s %9.1f %9.3f ms %9.0f %11.3f ms %9.0f %.2fx\n", r.s.model, r.s.cat, r.s.fam, r.mb, r.prodMs, r.mb/r.prodMs, r.ldMs, r.mb/r.ldMs, r.prodMs/r.ldMs)
		} else {
			fmt.Fprintf(os.Stderr, "  %-4s %-8s %-5s %9.1f %9.3f ms %9.0f %14s %9s\n", r.s.model, r.s.cat, r.s.fam, r.mb, r.prodMs, r.mb/r.prodMs, "—", "")
		}
	}
}
