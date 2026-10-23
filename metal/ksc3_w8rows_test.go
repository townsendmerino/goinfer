//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSC3_w8RowsMatchShipped is C3-G1's kernel half (docs/tasks/task-kernel-survey-2026-10.md, "KS-C3 gates"): each int8
// rows twin against its shipped one-row kernel at the 1.5B's and the 0.5B's qkv (bias), o (residual), gate|up and down
// (residual) shapes, R = 2 and 4, random int8 weights and activations and f32 scales, in the fast-math and the precise
// library. Every output byte-equal.
func TestKSC3_w8RowsMatchShipped(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	type shape struct {
		name, shipped, twin string
		N, K                int
		bias, resid, coal   bool
	}
	var shapes []shape
	for _, m := range []struct{ H, qkv, I int }{{1536, 2048, 8960}, {896, 1152, 4864}} {
		shapes = append(shapes,
			shape{fmt.Sprintf("qkv H=%d", m.H), "gemv_w8a8_sa_bias", "gemv_w8a8_sa_bias_rows", m.qkv, m.H, true, false, false},
			shape{fmt.Sprintf("o H=%d", m.H), "gemv_w8a8_sa_resid", "gemv_w8a8_sa_resid_rows", m.H, m.H, false, true, false},
			shape{fmt.Sprintf("gate|up H=%d", m.H), "gemv_w8a8_sa", "gemv_w8a8_sa_rows", 2 * m.I, m.H, false, false, false},
			shape{fmt.Sprintf("down H=%d", m.H), "gemv_w8a8_body_resid", "gemv_w8a8_body_resid_staged", m.H, m.I, false, true, true})
	}
	rng := rand.New(rand.NewSource(31))
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
		for _, s := range shapes {
			w := d.NewBufferBytes(s.N * s.K)
			wv := w.U32s()[:s.N*s.K/4]
			for i := range wv {
				wv[i] = rng.Uint32()
			}
			sc := make([]float32, s.N)
			for i := range sc {
				sc[i] = float32(rng.NormFloat64()*0.001 + 0.002)
			}
			a := d.NewBufferBytes(s.K)
			av := a.Int8s()[:s.K]
			for i := range av {
				av[i] = int8(rng.Intn(255) - 127)
			}
			bias, init := make([]float32, s.N), make([]float32, s.N)
			for i := range bias {
				bias[i], init[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			ws, asc, bb, uK := NewBufferFloats(d, sc), NewBufferFloats(d, []float32{0.0123}), NewBufferFloats(d, bias), NewBufferU32(d, uint32(s.K))
			run := func(p Pipeline, R int) []uint32 {
				out := NewBufferFloats(d, slices.Clone(init))
				args := []Buffer{w, ws, a, asc, out}
				if s.bias {
					args = append(args, bb)
				}
				args = append(args, uK)
				e := q.Begin()
				switch {
				case R > 0:
					e.DispatchTG(p, s.N*32/R, 256, s.K*2, args...)
				case s.coal:
					e.Dispatch(p, s.N*32, 32, args...)
				default:
					e.DispatchTG(p, s.N*32, 256, s.K*2, args...)
				}
				e.End()
				return slices.Clone(out.U32s()[:s.N])
			}
			want := run(pipe(s.shipped), 0)
			for _, R := range []int{2, 4} {
				got := run(pipe(fmt.Sprintf("%s%d", s.twin, R)), R)
				diff := 0
				for i := range want {
					if want[i] != got[i] {
						diff++
					}
				}
				fmt.Fprintf(os.Stderr, "[C3-G1] precise=%v %s N=%d K=%d R=%d: %d of %d outputs differ\n", precise, s.name, s.N, s.K, R, diff, s.N)
				if diff != 0 {
					t.Errorf("precise=%v %s R=%d: %d outputs differ from %s", precise, s.name, R, diff, s.shipped)
				}
			}
		}
	}
}

// ksc3Decode runs n greedy decode steps from token 13 on a's resident through Forward and returns every step's logits.
func ksc3Decode(t *testing.T, m *decoder.Model, a *metalResident, n int) [][]float32 {
	a.Reset()
	var out [][]float32
	tok := 13
	for pos := range n {
		lg, err := a.Forward(m.EmbedResidentForTest(tok), pos)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
		tok = 0
		for i, v := range lg {
			if v > lg[tok] {
				tok = i
			}
		}
	}
	return out
}

// TestKSC3_decodeIdentity is C3-G1's model half: llama-tiny (and, with GOINFER_HEAVY_TESTS=1, the 0.5B's .gguf) at
// int8int8, 32 greedy decode steps, every step's logits with the int8 rows kernels against the one-row kernels on one
// resident (gemvRows cleared for the old arm). Byte-equal; the resident must be native int8 with rows set.
func TestKSC3_decodeIdentity(t *testing.T) {
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		paths = append(paths, os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"))
	}
	for _, path := range paths {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8", DirectLoad: true, ResidentContext: 256})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("%s: not Metal-resident: %s", path, m.ResidentDecline())
		}
		rows := a.r.gemvRows
		if !a.r.w8 || rows.gu == 0 {
			t.Fatalf("%s: native int8 %v, rows %+v: the int8 rows kernels are not in use", path, a.r.w8, rows)
		}
		on := ksc3Decode(t, m, a, 32)
		a.r.gemvRows = struct{ qkv, o, gu, down int }{}
		off := ksc3Decode(t, m, a, 32)
		a.r.gemvRows = rows
		d := 0
		for s := range on {
			for i := range on[s] {
				if math.Float32bits(on[s][i]) != math.Float32bits(off[s][i]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[C3-G1] %s at int8int8, rows %+v: %d of %d logits differ over 32 steps\n", path, rows, d, 32*len(on[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the int8 rows and one-row kernels", path, d)
		}
		m.Close()
	}
}

// TestKSC3_decodeSpeed is C3-G3 (and C3-G3b with GOINFER_KSC3_MODEL naming the 1.5B's .gguf): the 0.5B's .gguf at int8int8,
// 128 greedy decode tokens after a 32-token prompt (Forward steps), the int8 rows kernels against the one-row kernels on one
// resident, A B B A over 10 pairs after 2 warm runs each, each run's start load printed. A timed run: hold the timing lock
// (GOINFER_KSC3_TIMING=1). A fit refusal prints FIT-REFUSED, which a night script turns into exit 75.
func TestKSC3_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSC3_TIMING") == "" {
		t.Skip("set GOINFER_KSC3_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if p := os.Getenv("GOINFER_KSC3_MODEL"); p != "" {
		path = p
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8", DirectLoad: true, ResidentContext: 256})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[C3-G3] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	defer m.Close()
	fmt.Fprintf(os.Stderr, "[C3-G3] %s at int8int8\n", path)
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	rows := a.r.gemvRows
	if !a.r.w8 || rows.gu == 0 {
		t.Fatalf("native int8 %v, rows %+v: the int8 rows kernels are not in use", a.r.w8, rows)
	}
	const prompt, gen = 32, 128
	run := func(oneRow bool) float64 {
		if oneRow {
			a.r.gemvRows = struct{ qkv, o, gu, down int }{}
		}
		defer func() { a.r.gemvRows = rows }()
		a.Reset()
		tok := 13
		for pos := range prompt {
			if err := a.ForwardNoLogits(m.EmbedResidentForTest((pos*7919+13)%1000), pos); err != nil {
				t.Fatal(err)
			}
		}
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg, err := a.Forward(m.EmbedResidentForTest(tok), pos)
			if err != nil {
				t.Fatal(err)
			}
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
		}
		return time.Since(t0).Seconds() / gen
	}
	for range 2 {
		run(true)
		run(false)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var one, rws float64
		if i%2 == 0 {
			one, rws = run(true), run(false)
		} else {
			rws, one = run(false), run(true)
		}
		ratios = append(ratios, rws/one)
		if rws < one {
			below++
		}
		fmt.Fprintf(os.Stderr, "[C3-G3] pair %d (load %s): one-row %.3f ms/token, rows %.3f ms/token, rows/one-row %.4f\n", i, la[:len(la)-1], one*1e3, rws*1e3, rws/one)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[C3-G3] median rows/one-row %.4f, %d of 10 pairs below 1 (rows %+v)\n", (ratios[4]+ratios[5])/2, below, rows)
}
