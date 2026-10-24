//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSC2_stagedMatchShipped is C2b-G1's kernel half (docs/tasks/task-kernel-survey-2026-10.md, "KS-C2 wiring gates"):
// gemv_w4a8_resid_staged<2,4> against gemv_w4a8_resid and gemv_w4a8_coal_staged<2,4> against gemv_w4a8_coal at the 1.5B's
// (1,536 x 8,960) and the 7B's (3,584 x 18,944) down shapes, random int4 weights, f16 scales and int8 activations, in the
// fast-math and the precise library. Every output byte-equal.
func TestKSC2_stagedMatchShipped(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(37))
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
		for _, sh := range []struct{ N, K int }{{1536, 8960}, {3584, 18944}} {
			w := d.NewBufferLen(sh.N * sh.K / 8)
			wv := w.U32s()[:sh.N*sh.K/8]
			for i := range wv {
				wv[i] = rng.Uint32()
			}
			sc := make([]uint16, sh.N*(sh.K/32))
			for i := range sc {
				sc[i] = f32ToF16(float32(rng.Intn(1000)+1) * 1e-5)
			}
			a := d.NewBufferBytes(sh.K)
			av := a.Int8s()[:sh.K]
			for i := range av {
				av[i] = int8(rng.Intn(255) - 127)
			}
			init := make([]float32, sh.N)
			for i := range init {
				init[i] = float32(rng.NormFloat64())
			}
			ws, asc, uK := NewBufferU16s(d, sc), NewBufferFloats(d, []float32{0.0123}), NewBufferU32(d, uint32(sh.K))
			run := func(name string, R int) []uint32 {
				out := NewBufferFloats(d, slices.Clone(init))
				e := q.Begin()
				if R == 0 {
					e.Dispatch(pipe(name), sh.N*32, 32, w, ws, a, asc, out, uK)
				} else {
					e.DispatchTG(pipe(name), sh.N*32/R, 256, sh.K, w, ws, a, asc, out, uK)
				}
				e.End()
				return slices.Clone(out.U32s()[:sh.N])
			}
			for _, pair := range [][2]string{{"gemv_w4a8_resid", "gemv_w4a8_resid_staged"}, {"gemv_w4a8_coal", "gemv_w4a8_coal_staged"}} {
				want := run(pair[0], 0)
				for _, R := range []int{2, 4} {
					got := run(fmt.Sprintf("%s%d", pair[1], R), R)
					diff := 0
					for i := range want {
						if want[i] != got[i] {
							diff++
						}
					}
					fmt.Fprintf(os.Stderr, "[C2b-G1] precise=%v %s%d N=%d K=%d: %d of %d outputs differ from %s\n", precise, pair[1], R, sh.N, sh.K, diff, sh.N, pair[0])
					if diff != 0 {
						t.Errorf("precise=%v %s%d N=%d K=%d: %d outputs differ from %s", precise, pair[1], R, sh.N, sh.K, diff, pair[0])
					}
				}
			}
		}
	}
}

// ksc2Load loads path resident on Metal with the staged coal kernels' old per-word form (old) or the masked one.
func ksc2Load(t *testing.T, path string, opts decoder.Options, old bool) (*decoder.Model, *metalResident) {
	t.Helper()
	prev := ksc2OldDownForTest
	ksc2OldDownForTest = old
	defer func() { ksc2OldDownForTest = prev }()
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { m.Close() })
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("%s: not Metal-resident: %s", path, m.ResidentDecline())
	}
	return m, a
}

// ksc2Decode runs n greedy decode steps from token 13 through Forward and returns every step's logits.
func ksc2Decode(t *testing.T, m *decoder.Model, a *metalResident, n int) [][]float32 {
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

// TestKSC2_decodeIdentity is C2b-G1's model half: qwen35-tiny (a DeltaNet layer runs gemv_w4a8_coal_staged) at int4 and, with
// GOINFER_HEAVY_TESTS=1, the 1.5B's Metal .giw, 32 greedy decode steps, every step's logits from a resident built with the
// masked kernels against one built with KSC2_OLD_DOWN. Byte-equal.
func TestKSC2_decodeIdentity(t *testing.T) {
	cases := []struct {
		path string
		opts decoder.Options
	}{{"../testdata/qwen35-tiny", decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256}}}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		cases = append(cases, struct {
			path string
			opts decoder.Options
		}{os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw"), decoder.Options{Backend: "metal", ResidentContext: 256}})
	}
	for _, c := range cases {
		mo, ao := ksc2Load(t, c.path, c.opts, true)
		old := ksc2Decode(t, mo, ao, 32)
		mn, an := ksc2Load(t, c.path, c.opts, false)
		nw := ksc2Decode(t, mn, an, 32)
		d := 0
		for s := range old {
			for i := range old[s] {
				if math.Float32bits(old[s][i]) != math.Float32bits(nw[s][i]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[C2b-G1] %s: down rows %d; %d of %d logits differ over 32 steps\n", c.path, an.r.gemvRows.down, d, 32*len(old[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the masked and the per-word staged kernels", c.path, d)
		}
	}
}

// TestKSC2_decodeSpeed is C2b-G2: the 1.5B's Metal .giw, two residents in one process (KSC2_OLD_DOWN and the masked form),
// 128 greedy decode tokens after a 32-token prompt, A B B A over 10 pairs after 2 warm runs each, each pair's start load
// printed. A timed run: hold the timing lock (GOINFER_KSC2_TIMING=1).
func TestKSC2_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSC2_TIMING") == "" {
		t.Skip("set GOINFER_KSC2_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw")
	opts := decoder.Options{Backend: "metal", ResidentContext: 256}
	mo, ao := ksc2Load(t, path, opts, true)
	mn, an := ksc2Load(t, path, opts, false)
	if an.r.gemvRows.down == 0 {
		t.Fatal("the 1.5B's down projection does not take the staged kernel")
	}
	const prompt, gen = 32, 128
	run := func(m *decoder.Model, a *metalResident) float64 {
		a.Reset()
		for pos := range prompt {
			if err := a.ForwardNoLogits(m.EmbedResidentForTest((pos*7919+13)%1000), pos); err != nil {
				t.Fatal(err)
			}
		}
		tok := 13
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
		run(mo, ao)
		run(mn, an)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, n float64
		if i%2 == 0 {
			o, n = run(mo, ao), run(mn, an)
		} else {
			n, o = run(mn, an), run(mo, ao)
		}
		ratios = append(ratios, n/o)
		if n < o {
			below++
		}
		fmt.Fprintf(os.Stderr, "[C2b-G2] pair %d (load %s): old %.3f ms/token, masked %.3f ms/token, new/old %.4f\n", i, la[:len(la)-1], o*1e3, n*1e3, n/o)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[C2b-G2] median new/old %.4f, %d of 10 pairs below 1\n", (ratios[4]+ratios[5])/2, below)
}
