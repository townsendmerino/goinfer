//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"sort"
	"testing"
)

// ksc1Q4KRaw is n rows of random Q4_K super-blocks over K: d and dmin small positive f16, random scales and quants.
func ksc1Q4KRaw(rng *rand.Rand, n, K int) []byte {
	raw := make([]byte, n*(K/256)*144)
	for b := 0; b < len(raw); b += 144 {
		d, dm := f32ToF16(float32(0.002+0.01*rng.Float64())), f32ToF16(float32(0.001+0.005*rng.Float64()))
		raw[b], raw[b+1], raw[b+2], raw[b+3] = byte(d), byte(d>>8), byte(dm), byte(dm>>8)
		for i := b + 4; i < b+144; i++ {
			raw[i] = byte(rng.Intn(256))
		}
	}
	return raw
}

// TestKSC1_w8RowsMatch is C1-G1 (b) (docs/tasks/task-kernel-survey-2026-10.md, "KS-C1 (a)+(b) gates"): gemv_w8_f32_rows<R> and its
// residual and bias variants, R = 1, 2, 4, against gemv_w8_f32 and its variants at G-Q1's int8 shapes (a partial threadgroup,
// Phi-3's qkv, its Q6_K down layers, the LM head), random int8 rows and per-row scales, f32 activations with outliers, in the
// fast-math and the precise library. Byte-equal.
func TestKSC1_w8RowsMatch(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(47))
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		lib, err := compile(allKernels, MSL3_1)
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
		for _, sh := range [][2]int{{5, 64}, {9216, 3072}, {3072, 8192}, {32064, 3072}} {
			N, K := sh[0], sh[1]
			w := d.NewBufferBytes(N * K)
			wv := w.Int8s()[:N*K]
			for i := range wv {
				wv[i] = int8(rng.Intn(255) - 127)
			}
			sc, a, bias, init := make([]float32, N), make([]float32, K), make([]float32, N), make([]float32, N)
			for i := range sc {
				sc[i], bias[i], init[i] = float32(0.001+0.01*rng.Float64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			for i := range a {
				a[i] = float32(rng.NormFloat64())
				if i%997 == 0 {
					a[i] = float32(300 * rng.NormFloat64())
				}
			}
			bs, ba, bb, uK, uN := NewBufferFloats(d, sc), NewBufferFloats(d, a), NewBufferFloats(d, bias), NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(N))
			for _, v := range []string{"", "_resid", "_bias"} {
				run := func(name string, R int) []uint32 {
					out := NewBufferFloats(d, slices.Clone(init))
					args := []Buffer{w, bs, ba, out, uK, uN}
					if v == "_bias" {
						args = append(args, bb)
					}
					e := q.Begin()
					e.Dispatch(pipe(name), (N+R-1)/R*32, 256, args...)
					e.End()
					return slices.Clone(out.U32s()[:N])
				}
				want := run("gemv_w8_f32"+v, 1)
				for _, R := range []int{1, 2, 4} {
					got := run(fmt.Sprintf("gemv_w8_f32%s_rows%d", v, R), R)
					diff := 0
					for i := range want {
						if want[i] != got[i] {
							diff++
						}
					}
					fmt.Fprintf(os.Stderr, "[C1-G1b] precise=%v N=%d K=%d %s R=%d: %d of %d differ\n", precise, N, K, "gemv_w8_f32"+v, R, diff, N)
					if diff != 0 {
						t.Errorf("precise=%v N=%d K=%d gemv_w8_f32%s_rows%d: %d outputs differ", precise, N, K, v, R, diff)
					}
				}
			}
		}
	}
}

// TestKSC1_gemvTiming is C1-G0: gemv_q4k_f32 and gemv_w8_f32 at Phi-3 mini's shapes (Q4_K o, gate|up, down; int8 qkv and the
// Q6_K down layers), each beside the new rows kernels at R = 1, 2, 4 and a streaming-read ceiling; random weights rotated over
// enough copies (>= 256 MB) to defeat the system cache; per dispatch, a command buffer of back-to-back dispatches timed by
// GPU timestamps, the median of 7. A timed run: hold the timing lock (GOINFER_KSC1_TIMING=1).
func TestKSC1_gemvTiming(t *testing.T) {
	if os.Getenv("GOINFER_KSC1_TIMING") == "" {
		t.Skip("set GOINFER_KSC1_TIMING=1 (a timed run under the timing lock)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels+"\n"+gemvS0Kernels[len("\n#include <metal_stdlib>\nusing namespace metal;\n"):], MSL3_1)
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
		fmt.Fprintf(os.Stderr, "[C1-G0] streaming read, 512 MB: %.3f ms -> %.1f GB/s\n", ms, float64(n4*16)/ms/1e6)
		d.ReleaseAll()
	}
	rng := rand.New(rand.NewSource(53))
	for _, s := range []struct {
		name string
		q4k  bool
		N, K int
	}{{"o", true, 3072, 3072}, {"gate|up", true, 16384, 3072}, {"down", true, 3072, 8192}, {"qkv", false, 9216, 3072}, {"down (Q6_K layers)", false, 3072, 8192}} {
		bytes := s.N * s.K
		if s.q4k {
			bytes = s.N * (s.K / 256) * 144
		}
		copies := min(64, max(1, int(math.Ceil(float64(256<<20)/float64(bytes)))))
		var ws, ss []Buffer
		for range copies {
			if s.q4k {
				raw := ksc1Q4KRaw(rng, s.N, s.K)
				wb := d.NewBufferBytes(len(raw))
				iv := wb.Int8s()[:len(raw)]
				for i, c := range raw {
					iv[i] = int8(c)
				}
				ws = append(ws, wb)
				continue
			}
			wb := d.NewBufferBytes(s.N * s.K)
			wv := wb.U32s()[:s.N*s.K/4]
			for i := range wv {
				wv[i] = rng.Uint32()
			}
			sc := make([]float32, s.N)
			for i := range sc {
				sc[i] = 0.001
			}
			ws, ss = append(ws, wb), append(ss, NewBufferFloats(d, sc))
		}
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		ba, out, uK, uN := NewBufferFloats(d, a), d.NewBufferLen(s.N), NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N))
		per := max(8, 2*copies)
		line := fmt.Sprintf("%-18s N=%-5d K=%-5d %6.1f MB:", s.name, s.N, s.K, float64(bytes)/1e6)
		base := "gemv_w8_f32"
		if s.q4k {
			base = "gemv_q4k_f32"
		}
		var old float64
		Rs := []int{0, 1, 2, 4}
		if s.q4k {
			Rs = append(Rs, 8)
		}
		for _, R := range Rs {
			name, n := base, s.N*32
			if R > 0 {
				name, n = fmt.Sprintf("%s_rows%d", base, R), (s.N+R-1)/R*32
			}
			p := pipe(name)
			ms := timeIt(per, func(e *Encoder, i int) {
				if s.q4k {
					e.Dispatch(p, n, 256, ws[i%copies], ba, out, uK, uN)
				} else {
					e.Dispatch(p, n, 256, ws[i%copies], ss[i%copies], ba, out, uK, uN)
				}
			})
			if R == 0 {
				old = ms
			}
			label := "old"
			if R > 0 {
				label = fmt.Sprintf("R=%d", R)
			}
			line += fmt.Sprintf("  %s %.3f ms (%.0f GB/s, %.2fx)", label, ms, float64(bytes)/1e6/ms, old/ms)
		}
		fmt.Fprintf(os.Stderr, "[C1-G0] %s\n", line)
		d.ReleaseAll()
	}
}
