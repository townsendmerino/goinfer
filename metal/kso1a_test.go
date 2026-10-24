//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// kso1aOld rebuilds the pre-KS-O1a library source from src (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1a"): the
// q4k_scale_min2 definition removed and each call to it put back as the two per-byte q4k_scale_min calls, so every other
// line of the library stands where it did.
func kso1aOld(t testing.TB, src string) string {
	const def = "// q4k_scale_min2:"
	i := strings.Index(src, def)
	if i < 0 {
		t.Fatal("kso1aOld: q4k_scale_min2's comment moved")
	}
	end := strings.Index(src[i:], "\n}\n")
	if end < 0 {
		t.Fatal("kso1aOld: q4k_scale_min2 has no end")
	}
	src = src[:i] + src[i+end+3:]
	const call = "q4k_scale_min2(j, blk + 4, sc0, m0, sc1, m1);"
	if n := strings.Count(src, call); n < 2 {
		t.Fatalf("kso1aOld: %d calls of q4k_scale_min2, want decode's and the step's", n)
	}
	return strings.ReplaceAll(src, call, "q4k_scale_min(2u*j, blk + 4, sc0, m0); q4k_scale_min(2u*j + 1u, blk + 4, sc1, m1);")
}

// TestKSO1a_q4kBitsUnchanged is O1a-G1's kernel half: the resident library's gemv_q4k_f32_rows<R> (plain, residual, bias;
// R = 1, 2, 4, 8) and the step's mc3_gemv_q4k_f32_rows<B> (B = 1, 2, 4, 8; modes 0-2), built from the source before and after
// KS-O1a, on random Q4_K blocks (every 6-bit scale and min value occurs) at a partial threadgroup and Phi-3's o, gate|up and down
// shapes, in the fast-math and the precise library. Byte-equal.
func TestKSO1a_q4kBitsUnchanged(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	src := allKernels + mc3RowsKernels
	rng := rand.New(rand.NewSource(73))
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		oldLib, err := compile(kso1aOld(t, src), MSL3_1)
		if err != nil {
			t.Fatalf("compile old: %v", err)
		}
		newLib, err := compile(src, MSL3_1)
		if err != nil {
			t.Fatalf("compile new: %v", err)
		}
		pipe := func(arm, n string) Pipeline {
			lib := newLib
			if arm == "old" {
				lib = oldLib
			}
			p, e := d.NewComputePipeline(lib, n)
			if e != nil {
				t.Fatalf("%s %s: %v", arm, n, e)
			}
			return p
		}
		q := d.NewCommandQueue()
		for _, sh := range [][2]int{{37, 256}, {3072, 3072}, {16384, 3072}, {3072, 8192}} {
			N, K := sh[0], sh[1]
			raw := ksc1Q4KRaw(rng, N, K)
			w := d.NewBufferBytes(len(raw))
			iv := w.Int8s()[:len(raw)]
			for i, c := range raw {
				iv[i] = int8(c)
			}
			const B = 8
			a, bias, init := make([]float32, B*K), make([]float32, N), make([]float32, B*N)
			for i := range a {
				a[i] = float32(rng.NormFloat64())
				if i%997 == 0 {
					a[i] = float32(300 * rng.NormFloat64())
				}
			}
			for i := range bias {
				bias[i] = float32(rng.NormFloat64())
			}
			for i := range init {
				init[i] = float32(rng.NormFloat64())
			}
			ba, bb, uK, uN := NewBufferFloats(d, a), NewBufferFloats(d, bias), NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(N))
			cmp := func(label string, run func(arm string) []uint32) {
				want, got := run("old"), run("new")
				diff := 0
				for i := range want {
					if want[i] != got[i] {
						diff++
					}
				}
				fmt.Fprintf(os.Stderr, "[O1a-G1] precise=%v N=%d K=%d %s: %d of %d differ\n", precise, N, K, label, diff, len(want))
				if diff != 0 {
					t.Errorf("precise=%v N=%d K=%d %s: %d outputs differ", precise, N, K, label, diff)
				}
			}
			for _, v := range []string{"", "_resid", "_bias"} {
				for _, R := range []int{1, 2, 4, 8} {
					name := fmt.Sprintf("gemv_q4k_f32%s_rows%d", v, R)
					cmp(name, func(arm string) []uint32 {
						out := NewBufferFloats(d, slices.Clone(init[:N]))
						args := []Buffer{w, ba, out, uK, uN}
						if v == "_bias" {
							args = append(args, bb)
						}
						e := q.Begin()
						e.Dispatch(pipe(arm, name), (N+R-1)/R*32, 256, args...)
						e.End()
						return slices.Clone(out.U32s()[:N])
					})
				}
			}
			uLda, uLdo := NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(N))
			for mode := range 3 {
				for _, Bs := range []int{1, 2, 4, 8} {
					name := fmt.Sprintf("mc3_gemv_q4k_f32_rows%d", Bs)
					cmp(fmt.Sprintf("%s mode %d", name, mode), func(arm string) []uint32 {
						out := NewBufferFloats(d, slices.Clone(init[:Bs*N]))
						e := q.Begin()
						e.Dispatch(pipe(arm, name), N*32, 256, w, ba, out, uK, uN, uLda, uLdo, bb, NewBufferU32(d, uint32(mode)))
						e.End()
						return slices.Clone(out.U32s()[:Bs*N])
					})
				}
			}
		}
		d.ReleaseAll()
	}
}

// TestKSO1a_phi3Speed is O1a-G3, with O1a-G2 inside it: Phi-3 mini 4k at --quant q4k (the registered GOINFER_PHI3_GGUF asset),
// one resident, its three Q4_K pipelines swapped between the library built from the source before KS-O1a (kso1aOld) and the
// resident's own; 128 greedy decode tokens after a 32-token prompt through ForwardEmb, A B B A over 10 pairs after 2 warm runs
// each, each pair's start load printed. Every run's tokens and final logits must equal the first run's (O1a-G2). A fit
// refusal prints FIT-REFUSED. A timed run: hold the timing lock (GOINFER_KSO1_TIMING=1).
func TestKSO1a_phi3Speed(t *testing.T) {
	if os.Getenv("GOINFER_KSO1_TIMING") == "" {
		t.Skip("set GOINFER_KSO1_TIMING=1 (a timed run under the timing lock)")
	}
	phi3 := decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF")
	mg, err := decoder.Load(phi3, decoder.Options{Quant: "q4k", ResidentContext: 256})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[O1a-G3] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	defer mg.Close()
	if why := residentMemoryDecline(mg); why != "" {
		fmt.Fprintf(os.Stderr, "[O1a-G3] FIT-REFUSED: %s\n", why)
		t.Fatalf("Metal's fit guard declines Phi-3: %s", why)
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.q4kLane {
		t.Fatal("the resident did not take the q4k lane")
	}
	src, err := withAttnSoftcap(allKernels+mc3RowsKernels+kvF32Kernels+kvRowsKernels, r.attnSoftcap)
	if err != nil {
		t.Fatal(err)
	}
	lib, err := r.d.CompileLibrary(kso1aOld(t, src), MSL3_1)
	if err != nil {
		t.Fatalf("compile the old library: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := r.d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	R := r.q4kR
	nw := [3]Pipeline{r.pQ4K, r.pQ4KResid, r.pQ4KBias}
	old := [3]Pipeline{pipe(fmt.Sprintf("gemv_q4k_f32_rows%d", R)), pipe(fmt.Sprintf("gemv_q4k_f32_resid_rows%d", R)), pipe(fmt.Sprintf("gemv_q4k_f32_bias_rows%d", R))}
	use := func(a [3]Pipeline) { r.pQ4K, r.pQ4KResid, r.pQ4KBias = a[0], a[1], a[2] }
	defer use(nw)
	const prompt, gen = 32, 128
	_, _, _, _, _, _, vocab := mg.Dims()
	var refToks []int
	var refLogits []float32
	runs := 0
	run := func(a [3]Pipeline) float64 {
		use(a)
		for pos := range prompt {
			r.ForwardEmb(mg.EmbedResidentForTest((pos*7919+13)%vocab), pos)
		}
		tok := 13
		toks := make([]int, 0, gen)
		var lg []float32
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg = r.ForwardEmb(mg.EmbedResidentForTest(tok), pos)
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
			toks = append(toks, tok)
		}
		s := time.Since(t0).Seconds() / gen
		runs++
		if refToks == nil {
			refToks, refLogits = toks, slices.Clone(lg)
		} else if !slices.Equal(toks, refToks) || !slices.Equal(lg, refLogits) {
			t.Errorf("O1a-G2: run %d's tokens or final logits differ from the first run's", runs)
		}
		return s
	}
	for range 2 {
		run(old)
		run(nw)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, n float64
		if i%2 == 0 {
			o, n = run(old), run(nw)
		} else {
			n, o = run(nw), run(old)
		}
		ratios = append(ratios, n/o)
		if n < o {
			below++
		}
		fmt.Fprintf(os.Stderr, "[O1a-G3] pair %d (load %s): old %.2f ms/token (%.1f tok/s), new %.2f ms/token (%.1f tok/s), new/old %.4f\n",
			i, la[:len(la)-1], o*1e3, 1/o, n*1e3, 1/n, n/o)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[O1a-G3] median new/old %.4f, %d of 10 pairs below 1; O1a-G2: %d runs, tokens and final logits identical: %v\n",
		(ratios[4]+ratios[5])/2, below, runs, !t.Failed())
}
