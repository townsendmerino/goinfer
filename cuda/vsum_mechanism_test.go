//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestVsumMechanism is R6 step 1 (docs/tasks/red-october.md): why did the V-sum spike's KL against the
// f32/f64 reference read 1.0585x exact's on D7@8000 while its argmax metrics were better?
// Pre-registered in docs/measurements/vsum-mechanism-PREREGISTERED.md; the hypotheses, the numeric tests
// and the decision rule are there and are not restated here.
//
// It re-uses Phase A's cached reference rows and re-scores arms only: exact (skVsumSplit=0) and the spike
// at each S in GOINFER_VSUM_MECH_S (default 1,2,4,8,16). S=1 must be BIT-IDENTICAL to exact — it is the
// built-in defect detector, because a one-chunk "split" is the same fold in the same order.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_SPLITKV_VSUM_SPLIT=16 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVsumMechanism -v -timeout 2h
//
// GOINFER_SPLITKV_VSUM_SPLIT must be >= max(S): it sizes the partials buffer at backend setup.
func TestVsumMechanism(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoint; needs Phase A's reference files)")
	}
	maxS := 0
	if v, err := strconv.Atoi(strings.TrimSpace(os.Getenv("GOINFER_SPLITKV_VSUM_SPLIT"))); err == nil {
		maxS = v
	}
	var ss []int
	sSpec := os.Getenv("GOINFER_VSUM_MECH_S")
	if sSpec == "" {
		sSpec = "1,2,4,8,16"
	}
	for _, f := range strings.Split(sSpec, ",") {
		s, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || s < 1 {
			t.Fatalf("GOINFER_VSUM_MECH_S: bad entry %q", f)
		}
		if s > maxS {
			t.Skipf("S=%d needs GOINFER_SPLITKV_VSUM_SPLIT >= %d (it sizes the partials buffer at setup); have %d", s, s, maxS)
		}
		ss = append(ss, s)
	}
	K := 8000
	if v, err := strconv.Atoi(os.Getenv("GOINFER_VSUM_MECH_K")); err == nil && v > 0 {
		K = v
	}
	nPrompts := 10
	if v, err := strconv.Atoi(os.Getenv("GOINFER_VSUM_MECH_PROMPTS")); err == nil && v > 0 {
		nPrompts = v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home: %v", err)
	}
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref")
	path := os.Getenv("GOINFER_CUDA_GATE_MODEL_D7")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("cuda resident did not engage")
	}
	if rf.skVsumPartial == (Pipeline{}) || rf.skVsumCombine == (Pipeline{}) || !rf.splitkvAttn {
		t.Fatal("split-KV or the spike pipelines are not loaded — every 'spike' arm would be the exact path")
	}
	rf.skMinKeys = 0

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	setLabel, promptFiles := decoder.PrefillGatePromptSet()
	if nPrompts > len(promptFiles) {
		nPrompts = len(promptFiles)
	}
	t.Logf("D7 K=%d prompt set %q, %d prompts, S=%v", K, setLabel, nPrompts, ss)

	// per-arm accumulators; index 0 is exact, 1.. follow ss.
	type acc struct {
		kl, klHead, klTail, klFromExact float64
		hf                              int
		agree                           float64
		bucket                          [3]float64 // KL by position bucket
		perPromptKL                     []float64
		differing                       int
	}
	arms := append([]int{0}, ss...)
	accs := make([]acc, len(arms))
	bucketOf := func(i int) int { // i is the continuation row index, 1-based decode rows
		switch {
		case i <= 16:
			return 0
		case i <= 48:
			return 1
		}
		return 2
	}
	var bucketN [3]int
	t0 := time.Now()
	s1Identical := true
	for pi := 0; pi < nPrompts; pi++ {
		ids := decoder.PrefillGateProseIDsForTest(t, tk, promptFiles[pi], K)
		_, refTokens, refLogits, err := decoder.ReadPrefillReferenceForTest(filepath.Join(refDir, fmt.Sprintf("D7-K%d-p%d.bin", K, pi)))
		if err != nil {
			t.Fatalf("reference p%d: %v", pi, err)
		}
		rows := make([][][]float32, len(arms))
		for ai, s := range arms {
			rows[ai] = vsumArm(t, rf, m, ids[:K], K, refTokens, s)
		}
		exact := rows[0]
		for ai, s := range arms {
			a := &accs[ai]
			var kl, head, tail, fromExact float64
			bk := [3]float64{}
			hf, nd := 0, 0
			for i := range rows[ai] {
				h, tl := klSplit(refLogits[i], rows[ai][i], 100)
				kl += h + tl
				if pi == 0 && i%16 == 0 { // the split must sum to the gate's own KL, or it is measuring something else
					if want := decoder.KLDivergenceForTest(refLogits[i], rows[ai][i]); math.Abs(h+tl-want) > 1e-7*math.Max(1, math.Abs(want)) {
						t.Fatalf("klSplit %.12g + %.12g != KLDivergenceForTest %.12g (arm S=%d row %d)", h, tl, want, s, i)
					}
				}
				head += h
				tail += tl
				if i > 0 {
					bk[bucketOf(i)] += h + tl
					if pi == 0 && ai == 0 {
						bucketN[bucketOf(i)]++
					}
				}
				if _, _, flip := decoder.NearTieArgmaxForTest(refLogits[i], rows[ai][i]); flip {
					hf++
				}
				if ai > 0 {
					fromExact += decoder.KLDivergenceForTest(exact[i], rows[ai][i])
					if !sameLogits(exact[i], rows[ai][i]) {
						nd++
						if s == 1 {
							s1Identical = false
						}
					}
				}
			}
			n := float64(len(rows[ai]))
			ag, _ := decoder.TeacherForcedTop1AgreementForTest(rows[ai], refTokens)
			a.kl += kl / n
			a.klHead += head / n
			a.klTail += tail / n
			a.klFromExact += fromExact / n
			a.hf += hf
			a.agree += ag
			a.differing += nd
			a.perPromptKL = append(a.perPromptKL, kl/n)
			for b := range bk {
				a.bucket[b] += bk[b]
			}
		}
		fmt.Printf("[vsum-mech] prompt %2d/%d done, S=1 identical so far=%v, elapsed %s\n", pi+1, nPrompts, s1Identical, time.Since(t0).Round(time.Second))
	}

	np := float64(nPrompts)
	fmt.Printf("=== VSUM MECHANISM D7 K=%d, %d prompts x 64 positions ===\n", K, nPrompts)
	fmt.Printf("%-6s %10s %10s %10s %10s %8s %9s %11s %10s %10s %10s\n", "arm", "meanKL", "head100", "tail", "KL/exact", "HF", "agree%", "KL(ex||arm)", "b1-16", "b17-48", "b49-63")
	base := accs[0]
	for ai, s := range arms {
		a := accs[ai]
		name := "exact"
		if s > 0 {
			name = fmt.Sprintf("S=%d", s)
		}
		fmt.Printf("%-6s %10.6f %10.6f %10.6f %10.4f %8d %9.2f %11.6f %10.6f %10.6f %10.6f\n", name,
			a.kl/np, a.klHead/np, a.klTail/np, a.kl/base.kl, a.hf, a.agree/np*100, a.klFromExact/np,
			a.bucket[0]/(np*float64(bucketN[0])), a.bucket[1]/(np*float64(bucketN[1])), a.bucket[2]/(np*float64(bucketN[2])))
	}
	fmt.Printf("paired per-prompt KL delta (arm - exact): mean ± s.e., #prompts higher\n")
	for ai, s := range arms {
		if s == 0 {
			continue
		}
		d := make([]float64, nPrompts)
		up := 0
		for p := range d {
			d[p] = accs[ai].perPromptKL[p] - base.perPromptKL[p]
			if d[p] > 0 {
				up++
			}
		}
		mu, se := meanSE(d)
		fmt.Printf("  S=%-2d delta %+0.6f ± %0.6f  higher on %d/%d prompts; head delta %+0.6f, tail delta %+0.6f; rowsDiffering=%d\n",
			s, mu, se, up, nPrompts, (accs[ai].klHead-base.klHead)/np, (accs[ai].klTail-base.klTail)/np, accs[ai].differing)
	}
	if !s1Identical {
		t.Errorf("DEFECT: S=1 is not bit-identical to the exact path on at least one row — a one-chunk split must be the same fold")
	}
}

// klSplit returns KL(p‖q) split into the part over the top-`k` tokens of p and the rest, with the same
// 1e-12 floor as decoder.KLDivergenceForTest so head+tail equals it.
func klSplit(pLogits, qLogits []float32, k int) (head, tail float64) {
	p, q := softmax64(pLogits), softmax64(qLogits)
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return p[idx[a]] > p[idx[b]] })
	inHead := make([]bool, len(p))
	for _, i := range idx[:min(k, len(idx))] {
		inHead[i] = true
	}
	for i, pi := range p {
		if pi < 1e-12 {
			continue
		}
		v := pi * math.Log(pi/(q[i]+1e-300))
		if inHead[i] {
			head += v
		} else {
			tail += v
		}
	}
	return head, tail
}

func softmax64(x []float32) []float64 {
	mx := float64(x[0])
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	out := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		out[i] = math.Exp(float64(v) - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func meanSE(x []float64) (mean, se float64) {
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	if len(x) < 2 {
		return mean, 0
	}
	var ss float64
	for _, v := range x {
		ss += (v - mean) * (v - mean)
	}
	return mean, math.Sqrt(ss/float64(len(x)-1)) / math.Sqrt(float64(len(x)))
}
