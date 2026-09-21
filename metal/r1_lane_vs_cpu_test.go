//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR1_laneVsCPU is experiment X2 of the R1 (W4F16 decode lane) re-investigation. X1
// (r1_gu_reference_test.go) showed that, within the resident int4 model, the f16 lane's gate/up
// GEMV sits on the f64 reference and the shipped W4A8 lane is the coarse arm (its single
// per-tensor int8 activation scale zeroes 97% of the FFN-26 input at the attention-sink
// position). That leaves the one question that decides R1's fate: against an EXTERNAL
// reference, is the f16 lane's full-model output better, equal, or worse than W4A8's?
//
// Reference: the CPU backend, Options{Backend:"cpu", Quant:"int8"} — weight-only per-row int8,
// f32 activations, exact f64-accumulating attention (GOINFER_CPU_FAST_ATTENTION=0, as
// decoder/prefill_ref_gen_test.go forces it). The repo's own S reference is Quant:"" (f32 weights,
// ~6 GB) and does not fit beside anything on this 16 GB machine today (~5 GB free); "int8" is the
// same choice prefill_ref_gen_test.go's d7RefQuant documents for the same reason. It is an
// external reference: its weight requantisation differs from Metal's int4-g32, so BOTH Metal arms
// carry the same weight-quant noise floor against it and the PAIRED comparison between arms is
// what carries information. The CPU int4 path is deliberately NOT used (it quantizes activations).
//
// Teacher-forced, no generation: the same N prompt tokens are fed one per step to the CPU
// reference and to each Metal arm; per position we score cosine(logits), KL(softmax(cpu) ||
// softmax(arm)) in float64, top-1 agreement and hard flips (decoder.NearTieArgmaxForTest).
// Both Metal arms come from ONE model load: the lane is toggled at runtime via
// r.decodeLaneW4F16 (canUseF16Lane reads the field on every call); Forward(id,pos) calls
// setPos(pos), which rewrites curNKeys=pos+1 and the KV slot for pos, so restarting the arm at
// pos 0 is clean (no other per-sequence state accumulates in forwardLogits).
//
// Memory: ONE checkpoint resident at a time — the CPU model is Closed before the Metal load.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR1_laneVsCPU$' -v
func TestR1_laneVsCPU(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint twice, sequentially)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	t.Setenv("GOINFER_METAL_DECODE_LANE", "")   // lane OFF at load; toggled at runtime below
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0") // exact f64-accumulating CPU attention (reference)

	logf := func(format string, args ...any) {
		t.Helper()
		t.Logf(format, args...)
		fmt.Fprintf(os.Stderr, "[r1-x2] "+format+"\n", args...)
	}

	const refQuant = "int8"
	const prompt = "package main\n\nimport (\n\t\"fmt\"\n\t\"sort\"\n)\n\n// topK returns the k largest values of xs in descending order.\nfunc topK(xs []float64, k int) []float64 {\n\tout := append([]float64(nil), xs...)\n\tsort.Sort(sort.Reverse(sort.Float64Slice(out)))\n\tif k < len(out) {\n\t\tout = out[:k]\n\t}\n\treturn out\n}\n\nfunc main() {\n\tfmt.Println(topK([]float64{3, 1, 4, 1, 5, 9, 2, 6}, 3))\n}\n"

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ids, err := tk.Encode(prompt, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	N := min(48, len(ids))
	ids = ids[:N]
	logf("prompt: %d tokens encoded, using N=%d positions (teacher-forced, prompt tokens only); reference=cpu/%s", len(ids), N, refQuant)

	// ---- Step A: CPU reference (int8 weights, f32 activations). Closed before Step B. ----
	cpuLogits := make([][]float32, N)
	{
		mc, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: refQuant})
		if err != nil {
			t.Fatalf("cpu load (%s): %v", refQuant, err)
		}
		cache := mc.NewCache(N + 4)
		for pos := 0; pos < N; pos++ {
			lg, err := mc.ForwardForTest(ids[pos], cache)
			if err != nil {
				mc.Close()
				t.Fatalf("cpu forward pos %d: %v", pos, err)
			}
			cpuLogits[pos] = append([]float32(nil), lg...)
			if pos%8 == 7 || pos == N-1 {
				fmt.Fprintf(os.Stderr, "[r1-x2] cpu ref: %d/%d positions done\n", pos+1, N)
			}
		}
		if err := mc.Close(); err != nil {
			t.Fatalf("cpu close: %v", err)
		}
		logf("cpu reference done: V=%d, model closed before the Metal load", len(cpuLogits[0]))
	}

	// ---- Step B: Metal, one load, two arms by runtime lane toggle. ----
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("metal load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built for this model")
	}
	r := rf.r
	if r.decodeLaneW4F16 {
		t.Fatalf("lane ON at load despite GOINFER_METAL_DECODE_LANE=\"\"")
	}
	r.decodeLaneW4F16 = true
	elig := 0
	for l := 0; l < r.nL; l++ {
		if r.canUseF16Lane(l) {
			elig++
		}
	}
	r.decodeLaneW4F16 = false
	logf("metal: H=%d I=%d nL=%d V=%d; f16-lane-eligible layers with the lane on: %d/%d", r.H, r.I, r.nL, r.V, elig, r.nL)
	if elig != r.nL {
		t.Fatalf("f16 lane eligible on %d/%d layers — premise changed", elig, r.nL)
	}
	if len(cpuLogits[0]) != r.V {
		t.Fatalf("vocab mismatch: cpu %d vs metal %d", len(cpuLogits[0]), r.V)
	}

	runArm := func(name string, f16 bool) [][]float32 {
		r.decodeLaneW4F16 = f16
		defer func() { r.decodeLaneW4F16 = false }()
		out := make([][]float32, N)
		for pos := 0; pos < N; pos++ {
			lg := r.Forward(ids[pos], pos) // returns the shared r.logitsHost — copy it
			out[pos] = append([]float32(nil), lg...)
		}
		if err := r.takeExecErr(); err != nil {
			t.Fatalf("%s arm: resident exec error: %v", name, err)
		}
		fmt.Fprintf(os.Stderr, "[r1-x2] metal %s arm: %d/%d positions done\n", name, N, N)
		return out
	}
	w4a8 := runArm("w4a8", false)
	f16 := runArm("f16", true)
	if r.decodeLaneW4F16 {
		t.Fatalf("lane field not restored")
	}

	// Sanity: the toggle must have taken — the two arms may not be bit-identical everywhere.
	identical := 0
	for pos := 0; pos < N; pos++ {
		same := true
		for i := range w4a8[pos] {
			if w4a8[pos][i] != f16[pos][i] {
				same = false
				break
			}
		}
		if same {
			identical++
		}
	}
	logf("arms bit-identical at %d/%d positions (must be < N or the lane toggle did not take)", identical, N)
	if identical == N {
		t.Fatalf("W4A8 and f16 arms bit-identical at every position: the lane toggle did not take")
	}

	// ---- Metrics ----
	argmax := func(v []float32) int {
		best := 0
		for i, x := range v {
			if x > v[best] {
				best = i
			}
		}
		return best
	}
	cosine := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			x, y := float64(a[i]), float64(b[i])
			dot += x * y
			na += x * x
			nb += y * y
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-300)
	}
	logSoftmax := func(v []float32) []float64 {
		mx := float64(v[0])
		for _, x := range v {
			if float64(x) > mx {
				mx = float64(x)
			}
		}
		var s float64
		for _, x := range v {
			s += math.Exp(float64(x) - mx)
		}
		lse := mx + math.Log(s)
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = float64(x) - lse
		}
		return out
	}
	// KL(p || q) with p = softmax(ref), q = softmax(arm), in float64.
	kl := func(ref, arm []float32) float64 {
		lp, lq := logSoftmax(ref), logSoftmax(arm)
		var s float64
		for i := range lp {
			p := math.Exp(lp[i])
			if p > 0 {
				s += p * (lp[i] - lq[i])
			}
		}
		return s
	}

	type armStats struct {
		name                      string
		cos, klv, gap             []float64
		agree, hard               []bool
		sumCos, minCos, sumKL     float64
		nAgree, nHard, nSoftFlips int
	}
	score := func(name string, arm [][]float32) *armStats {
		s := &armStats{name: name, minCos: 2}
		s.cos = make([]float64, N)
		s.klv = make([]float64, N)
		s.gap = make([]float64, N)
		s.agree = make([]bool, N)
		s.hard = make([]bool, N)
		for pos := 0; pos < N; pos++ {
			s.cos[pos] = cosine(cpuLogits[pos], arm[pos])
			s.klv[pos] = kl(cpuLogits[pos], arm[pos])
			agree, gapPct, hardFail := decoder.NearTieArgmaxForTest(cpuLogits[pos], arm[pos])
			s.agree[pos], s.gap[pos], s.hard[pos] = agree, gapPct, hardFail
			s.sumCos += s.cos[pos]
			s.minCos = math.Min(s.minCos, s.cos[pos])
			s.sumKL += s.klv[pos]
			if agree {
				s.nAgree++
			} else if hardFail {
				s.nHard++
			} else {
				s.nSoftFlips++
			}
		}
		return s
	}
	sw := score("w4a8", w4a8)
	sf := score("f16", f16)

	// Paired counts.
	laneAgree, onlyOne, onlyW, onlyF, f16Lower := 0, 0, 0, 0, 0
	for pos := 0; pos < N; pos++ {
		if argmax(w4a8[pos]) == argmax(f16[pos]) {
			laneAgree++
		}
		if sw.agree[pos] != sf.agree[pos] {
			onlyOne++
			if sw.agree[pos] {
				onlyW++
			} else {
				onlyF++
			}
		}
		if sf.klv[pos] < sw.klv[pos] {
			f16Lower++
		}
	}

	// ---- Per-position table ----
	logf("=== PER-POSITION TABLE (reference = cpu/%s, f32 activations) ===", refQuant)
	logf("%4s %7s %6s | %10s %10s %6s %7s | %10s %10s %6s %7s | %6s %10s",
		"pos", "tok", "cpuArg", "w4a8.cos", "w4a8.KL", "w.top1", "w.gap%", "f16.cos", "f16.KL", "f.top1", "f.gap%", "lanes=", "ratioKL f/w")
	for pos := 0; pos < N; pos++ {
		mark := func(agree, hard bool) string {
			switch {
			case agree:
				return "ok"
			case hard:
				return "HARD"
			default:
				return "soft"
			}
		}
		lanes := "same"
		if argmax(w4a8[pos]) != argmax(f16[pos]) {
			lanes = "DIFF"
		}
		logf("%4d %7d %6d | %10.7f %10.3e %6s %7.3f | %10.7f %10.3e %6s %7.3f | %6s %10.3f",
			pos, ids[pos], argmax(cpuLogits[pos]),
			sw.cos[pos], sw.klv[pos], mark(sw.agree[pos], sw.hard[pos]), 100*sw.gap[pos],
			sf.cos[pos], sf.klv[pos], mark(sf.agree[pos], sf.hard[pos]), 100*sf.gap[pos],
			lanes, sf.klv[pos]/(sw.klv[pos]+1e-300))
	}

	// ---- Pooled summary ----
	logf("=== POOLED SUMMARY (N=%d positions, reference=cpu/%s) ===", N, refQuant)
	for _, s := range []*armStats{sw, sf} {
		logf("%-5s mean_cos=%.8f min_cos=%.8f mean_KL=%.6e top1_agree=%d/%d (%.4f) hard_flips=%d soft_flips=%d",
			s.name, s.sumCos/float64(N), s.minCos, s.sumKL/float64(N), s.nAgree, N, float64(s.nAgree)/float64(N), s.nHard, s.nSoftFlips)
	}
	logf("lane-vs-lane top1 agreement: %d/%d (%.4f); positions where exactly one arm matches cpu argmax (McNemar d): %d (only w4a8: %d, only f16: %d); f16 KL lower on %d/%d positions",
		laneAgree, N, float64(laneAgree)/float64(N), onlyOne, onlyW, onlyF, f16Lower, N)

	// Rules from metal/prefill_gate_ref_test.go's header, applied with f16 as "fast" and W4A8 as
	// "exact" (the shipped arm). Reported, not asserted: the verdict belongs to the write-up.
	ruleA := sf.nHard <= sw.nHard+int(math.Ceil(2*math.Sqrt(float64(sw.nHard))))
	ruleB := float64(sf.nAgree)/float64(N) >= float64(sw.nAgree)/float64(N)-2*math.Sqrt(float64(onlyOne))/float64(N)
	ruleC := sf.sumKL <= sw.sumKL
	logf("rule (a) hard flips f16 %d <= w4a8 %d + 2*sqrt(%d) = %.2f: %v", sf.nHard, sw.nHard, sw.nHard, float64(sw.nHard)+2*math.Sqrt(float64(sw.nHard)), ruleA)
	logf("rule (b) top1 f16 %.4f >= w4a8 %.4f - 2*sqrt(d=%d)/N = %.4f: %v", float64(sf.nAgree)/float64(N), float64(sw.nAgree)/float64(N), onlyOne, float64(sw.nAgree)/float64(N)-2*math.Sqrt(float64(onlyOne))/float64(N), ruleB)
	logf("rule (c) mean KL f16 %.6e <= w4a8 %.6e: %v (f16 lower on %d/%d positions; ratio f16/w4a8 = %.4f)", sf.sumKL/float64(N), sw.sumKL/float64(N), ruleC, f16Lower, N, sf.sumKL/(sw.sumKL+1e-300))
	logf("all three pooled rules hold: %v", ruleA && ruleB && ruleC)
}
