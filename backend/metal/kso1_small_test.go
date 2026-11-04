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

// TestKSO1_smallShapeReading is an ungraded reading after O1-G7: gemv_q6k_f32_rows<R> at R = 1, 2, 4 against gemv_w8_f32_rows2 at
// Qwen2.5-Coder-1.5B's Q6_K shapes (ffn_down, attn_v) and Phi-3's ffn_down, back-to-back dispatches over copies rotating
// >= 256 MiB of native bytes, median of 7. A timed run: hold the timing lock (GOINFER_KSO1_TIMING=1).
func TestKSO1_smallShapeReading(t *testing.T) {
	if os.Getenv("GOINFER_KSO1_TIMING") == "" {
		t.Skip("set GOINFER_KSO1_TIMING=1 (a timed run under the timing lock)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	extra := ""
	for _, R := range []int{1, 2} {
		extra += fmt.Sprintf("template [[host_name(\"gemv_q6k_f32_rows%d\")]] kernel decltype(gemv_q6k_f32_rows<%d>) gemv_q6k_f32_rows<%d>;\n", R, R, R)
	}
	lib, err := d.CompileLibrary(allKernels+extra, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("%s: %v", n, err)
		}
		return p
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(5))
	for _, s := range []struct {
		name string
		N, K int
	}{{"1.5B down", 1536, 8960}, {"1.5B v", 256, 1536}, {"phi3 down", 3072, 8192}} {
		copies := min(1024, int(math.Ceil(float64(256<<20)/float64(s.N*(s.K/256)*210)))) // >= 256 MiB of native bytes, more of int8
		var nat, w8, w8s []Buffer
		for range copies {
			raw := kso1Q6KPlanes(kso1Q6KGGUF(rng, s.N, s.K), s.N, s.K)
			nb := d.NewBufferBytes(len(raw))
			iv := nb.Int8s()[:len(raw)]
			for i, c := range raw {
				iv[i] = int8(c)
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
			nat, w8, w8s = append(nat, nb), append(w8, wb), append(w8s, NewBufferFloats(d, sc))
		}
		a := make([]float32, s.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		ba, out, uK, uN := NewBufferFloats(d, a), d.NewBufferLen(s.N), NewBufferU32(d, uint32(s.K)), NewBufferU32(d, uint32(s.N))
		tm := func(f func(e *Encoder, i int)) float64 {
			var ms []float64
			for r := range 8 {
				e := q.Begin()
				for i := range max(64, 2*copies) {
					f(e, i)
				}
				e.End()
				if err := e.Err(); err != nil {
					t.Fatalf("dispatch: %v", err)
				}
				if r > 0 {
					ms = append(ms, (e.GPUEnd()-e.GPUStart())*1e3/float64(max(64, 2*copies)))
				}
			}
			sort.Float64s(ms)
			return ms[3]
		}
		line := fmt.Sprintf("%-10s N=%d K=%d:", s.name, s.N, s.K)
		line += fmt.Sprintf(" w8_rows2 %.4f ms", tm(func(e *Encoder, i int) {
			e.Dispatch(pipe("gemv_w8_f32_rows2"), (s.N+1)/2*32, 256, w8[i%copies], w8s[i%copies], ba, out, uK, uN)
		}))
		for _, R := range []int{1, 2, 4} {
			line += fmt.Sprintf("  q6k R=%d %.4f ms", R, tm(func(e *Encoder, i int) {
				e.Dispatch(pipe(fmt.Sprintf("gemv_q6k_f32_rows%d", R)), (s.N+R-1)/R*32, 256, nat[i%copies], ba, out, uK, uN)
			}))
		}
		fmt.Fprintln(os.Stderr, "[O1 small]", line)
		d.ReleaseAll()
	}
}
