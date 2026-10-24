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

// ksc5OldSrc is allKernels with attention's shipped single-tile form compiled back (KSC5_OLD_ATTN).
func ksc5OldSrc(src string) string { return "#define KSC5_OLD_ATTN 1\n" + src }

// TestKSC5_attentionMatchesOld is C5-G1 (a) (docs/tasks/task-kernel-survey-2026-10.md, "KS-C5 gates"): attention against its
// KSC5_OLD_ATTN form at head dims 64 and 128, nKeys 1 to 4096 (the single-tile path), window 0 and 64, a sink on and off, 4
// query heads over 2 KV heads, in the fast-math and the precise library. Every output byte-equal. GOINFER_KSC5_PLANT=1 compiles
// the registered plant into the new form (the sum's stride-1 shuffle adds the lane's own value) and expects red.
func TestKSC5_attentionMatchesOld(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	newSrc := allKernels
	if os.Getenv("GOINFER_KSC5_PLANT") != "" {
		const at = "v = v + simd_shuffle_down(v, st);"
		if !strings.Contains(newSrc, at) {
			t.Fatal("plant site not found")
		}
		newSrc = strings.Replace(newSrc, at, "v = v + (st == 1u ? v : simd_shuffle_down(v, st));", 1)
	}
	const nH, nKV = 4, 2
	rng := rand.New(rand.NewSource(53))
	total := 0
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		libNew, err := compile(newSrc, MSL3_1)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		libOld, err := compile(ksc5OldSrc(allKernels), MSL3_1)
		if err != nil {
			t.Fatalf("compile old: %v", err)
		}
		pNew, err := d.NewComputePipeline(libNew, "attention")
		if err != nil {
			t.Fatal(err)
		}
		pOld, err := d.NewComputePipeline(libOld, "attention")
		if err != nil {
			t.Fatal(err)
		}
		q := d.NewCommandQueue()
		for _, hd := range []int{64, 128} {
			for _, nKeys := range []int{1, 7, 8, 15, 16, 17, 31, 100, 128, 129, 512, 1000, 4096} {
				kvDim := nKV * hd
				halves := func(n int) Buffer {
					v := make([]uint16, n)
					for i := range v {
						v[i] = f32ToF16(float32(rng.NormFloat64()))
					}
					return NewBufferU16s(d, v)
				}
				qv := make([]float32, nH*hd)
				for i := range qv {
					qv[i] = float32(rng.NormFloat64())
				}
				qb, kc, vc := NewBufferFloats(d, qv), halves(nKeys*kvDim), halves(nKeys*kvDim)
				sinks := NewBufferFloats(d, []float32{0.5, -1, 2, 0.1})
				for _, window := range []int{0, 64} {
					for _, sink := range []uint32{0, 1} {
						run := func(p Pipeline) []uint32 {
							out := d.NewBufferLen(nH * hd)
							e := q.Begin()
							e.Dispatch(p, nH*tgReduceAttn, tgReduceAttn, qb, kc, vc, out, NewBufferU32(d, nH), NewBufferU32(d, nKV),
								NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(nKeys)), NewBufferFloats(d, []float32{float32(1 / math.Sqrt(float64(hd)))}),
								NewBufferU32(d, uint32(window)), sinks, NewBufferU32(d, sink))
							e.End()
							return slices.Clone(out.U32s()[:nH*hd])
						}
						want, got := run(pOld), run(pNew)
						diff := 0
						for i := range want {
							if want[i] != got[i] {
								diff++
							}
						}
						total += diff
						if diff != 0 {
							fmt.Fprintf(os.Stderr, "[C5-G1] precise=%v hd=%d nKeys=%d window=%d sink=%d: %d of %d differ\n", precise, hd, nKeys, window, sink, diff, nH*hd)
							t.Errorf("precise=%v hd=%d nKeys=%d window=%d sink=%d: %d outputs differ", precise, hd, nKeys, window, sink, diff)
						}
					}
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[C5-G1] %d cells (2 libraries × 2 head dims × 13 key counts × 2 windows × 2 sinks): %d outputs differ in all\n", 2*2*13*2*2, total)
}

// ksc5Load loads path with the decode library in its old (KSC5_OLD_ATTN) or new form.
func ksc5Load(t *testing.T, path string, old bool) (*decoder.Model, *metalResident) {
	t.Helper()
	prev := ksc5OldAttnForTest
	ksc5OldAttnForTest = old
	defer func() { ksc5OldAttnForTest = prev }()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[C5] FIT-REFUSED: %v\n", err)
		}
		t.Fatalf("load %s: %v", path, err)
	}
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("%s: not Metal-resident: %s", path, m.ResidentDecline())
	}
	return m, a
}

// ksc5Run prefills a prompt of `prompt` tokens (ForwardNoLogits) and decodes n greedy steps, returning every step's logits.
func ksc5Run(t *testing.T, m *decoder.Model, a *metalResident, prompt, n int) [][]float32 {
	prompt = min(prompt, a.r.ctxCap-n-1) // llama-tiny's context is 128
	a.Reset()
	for pos := range prompt {
		if err := a.ForwardNoLogits(m.EmbedResidentForTest((pos*7919+13)%min(1000, a.r.V)), pos); err != nil {
			t.Fatal(err)
		}
	}
	var out [][]float32
	tok := 13
	for pos := prompt; pos < prompt+n; pos++ {
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

// TestKSC5_decodeIdentity is C5-G1 (b): llama-tiny (and, with GOINFER_HEAVY_TESTS=1, the 0.5B's and 1.5B's Metal .giw) at int4,
// a 300-token prompt (llama-tiny: as long as its 128-position context allows) then 32 greedy decode steps (the legacy attention), every step's logits, a resident built with
// each library. Byte-equal. GOINFER_KSC5_OLDPLANT=1 plants a defect inside the KSC5_OLD_ATTN branch (its max tree takes min)
// and expects red, so the define reaches the old arm.
func TestKSC5_decodeIdentity(t *testing.T) {
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		for _, n := range []string{"0.5b", "1.5b"} {
			paths = append(paths, os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-%s-instruct-q4_k_m.int4.metal.v%d.giw", n, decoder.GIWFormatVersion)))
		}
	}
	plant := os.Getenv("GOINFER_KSC5_OLDPLANT") != ""
	for _, path := range paths {
		mNew, aNew := ksc5Load(t, path, false)
		on := ksc5Run(t, mNew, aNew, 300, 32)
		mNew.Close()
		if plant {
			const at = "#ifdef KSC5_OLD_ATTN\n        for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]);"
			mambaSrcPlantForTest = [2]string{at, strings.Replace(at, "=max(", "=min(", 1)}
		}
		mOld, aOld := ksc5Load(t, path, true)
		mambaSrcPlantForTest = [2]string{}
		off := ksc5Run(t, mOld, aOld, 300, 32)
		mOld.Close()
		d := 0
		for s := range on {
			for i := range on[s] {
				if math.Float32bits(on[s][i]) != math.Float32bits(off[s][i]) {
					d++
				}
			}
		}
		fmt.Fprintf(os.Stderr, "[C5-G1] %s (plant in the old arm %v): %d of %d logits differ over 32 steps after a 300-token prompt\n", path, plant, d, 32*len(on[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the new and old attention", path, d)
		}
	}
}

// TestKSC5_decodeSpeed is C5-G2: the 1.5B's Metal .giw at int4, two residents in one process (old and new library), 128 greedy
// decode tokens after a 384-token prompt (keys 385–512), A B B A over 10 pairs after 2 warm runs each, each pair's start load
// printed. A timed run: hold the timing lock (GOINFER_KSC5_TIMING=1).
func TestKSC5_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSC5_TIMING") == "" {
		t.Skip("set GOINFER_KSC5_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	mOld, aOld := ksc5Load(t, path, true)
	defer mOld.Close()
	mNew, aNew := ksc5Load(t, path, false)
	defer mNew.Close()
	const prompt, gen = 384, 128
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
		run(mOld, aOld)
		run(mNew, aNew)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, n float64
		if i%2 == 0 {
			o, n = run(mOld, aOld), run(mNew, aNew)
		} else {
			n, o = run(mNew, aNew), run(mOld, aOld)
		}
		ratios = append(ratios, n/o)
		if n < o {
			below++
		}
		fmt.Fprintf(os.Stderr, "[C5-G2] pair %d (load %s): old %.3f ms/token, new %.3f ms/token, new/old %.4f\n", i, la[:len(la)-1], o*1e3, n*1e3, n/o)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[C5-G2] median new/old %.4f, %d of 10 pairs below 1\n", (ratios[4]+ratios[5])/2, below)
}
