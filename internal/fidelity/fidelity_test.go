package fidelity

import (
	"bytes"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// flipP is the per-position probability that equal arms disagree about agreement (discordance).
const flipP = 0.03

// synth builds a cell whose candidate is `klScale`× the exact arm's KL (with prompt-to-prompt spread and per-position
// noise) and whose agreement differs by agreeShift (probability points).
func synth(rng *rand.Rand, name string, prompts, positions int, klScale, agreeShift float64) Cell {
	c := Cell{Name: name}
	for p := 0; p < prompts; p++ {
		base := 0.02 * math.Exp(rng.NormFloat64()*0.6) // prompts differ a lot in difficulty, as the logs show
		pAgree := 0.85 + 0.05*rng.NormFloat64()
		var pr Prompt
		pr.ID = name + "-" + string(rune('a'+p%26))
		for i := 0; i < positions; i++ {
			// The arms are CORRELATED, as real candidate/exact arms are: both see the same position, so they mostly
			// agree or disagree with the reference together. flipP is the per-position discordance for equal arms
			// (~3%, the order the gates' recorded d/N shows); agreeShift moves the candidate's agreement on top of it.
			shared := math.Exp(rng.NormFloat64() * 0.3)
			ea := rng.Float64() < pAgree
			ca := ea
			if rng.Float64() < flipP { // a discordant position: which arm gets it right is a fair coin for equal arms
				ea = rng.Float64() < 0.5
				ca = !ea
			}
			if agreeShift < 0 && ca && rng.Float64() < -agreeShift/pAgree {
				ca = false
			}
			e := Position{Agree: ea, KL: base * shared * math.Exp(rng.NormFloat64()*0.1)}
			cd := Position{Agree: ca, KL: klScale * base * shared * math.Exp(rng.NormFloat64()*0.1)}
			e.HardFlip = !e.Agree && rng.Float64() < 0.05
			cd.HardFlip = !cd.Agree && rng.Float64() < 0.05
			pr.Exact = append(pr.Exact, e)
			pr.Candidate = append(pr.Candidate, cd)
		}
		c.Prompts = append(c.Prompts, pr)
	}
	return c
}

func passRate(t *testing.T, seed uint64, trials, cells, prompts int, klScale, agreeShift float64, c Criteria) float64 {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7))
	pass := 0
	for i := 0; i < trials; i++ {
		var cs []Cell
		for k := 0; k < cells; k++ {
			cs = append(cs, synth(rng, string(rune('A'+k)), prompts, 64, klScale, agreeShift))
		}
		v, err := NonInferior(cs, c)
		if err != nil {
			t.Fatal(err)
		}
		if v.Pass {
			pass++
		}
	}
	return float64(pass) / float64(trials)
}

func TestNonInferior_refusesZeroMargin(t *testing.T) {
	_, err := NonInferior([]Cell{{Name: "x"}}, Criteria{AgreeMarginPts: 1, KLMargin: 0})
	if err == nil || !strings.Contains(err.Error(), "superiority") {
		t.Fatalf("a zero KL margin must be refused and point at Superiority, got %v", err)
	}
	_, err = NonInferior([]Cell{{Name: "x"}}, Criteria{AgreeMarginPts: 0, KLMargin: 0.1})
	if err == nil {
		t.Fatal("a zero agreement margin must be refused")
	}
}

// The calibration property TE12 is about: at a sensible pooled prompt count, truly equal arms pass nearly always, and a
// candidate truly outside the margin nearly never does. (The old no-margin criteria pass equal arms ~half the time.)
func TestNonInferior_calibrated(t *testing.T) {
	c := Criteria{AgreeMarginPts: 1, KLMargin: 0.10}
	equal := passRate(t, 1, 200, 5, 10, 1.0, 0, c) // 5 cells × 10 prompts, pooled: 50 prompts
	if equal < 0.85 {
		t.Fatalf("equal arms passed only %.0f%% of 200 simulated gates", 100*equal)
	}
	worseKL := passRate(t, 2, 200, 5, 10, 1.25, 0, c) // 25% more KL, outside the 10% margin
	if worseKL > 0.05 {
		t.Fatalf("a candidate at 1.25x KL passed %.0f%% of gates; the one-sided bound should hold it near 0", 100*worseKL)
	}
	worseAgree := passRate(t, 3, 200, 5, 10, 1.0, -0.04, c) // 4 points worse, outside the 1-point margin
	if worseAgree > 0.05 {
		t.Fatalf("a candidate 4 pts worse on agreement passed %.0f%% of gates", 100*worseAgree)
	}
	t.Logf("pass rate: equal %.0f%%, 1.25x KL %.0f%%, -4 pts agreement %.0f%%", 100*equal, 100*worseKL, 100*worseAgree)
}

// The old Metal-pooled KL rule (candidate mean <= exact mean, no margin) is a coin flip for equal arms — the finding
// that motivated the margin. Kept as a test so the reason is executable, not just prose.
func TestNoMarginRule_isACoinFlipForEqualArms(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 7))
	pass := 0
	const trials = 400
	for i := 0; i < trials; i++ {
		var all []Prompt
		for k := 0; k < 5; k++ {
			all = append(all, synth(rng, "c", 10, 64, 1.0, 0).Prompts...)
		}
		s, err := Summarize(all, 0.05)
		if err != nil {
			t.Fatal(err)
		}
		if s.MeanKLCand <= s.MeanKLExact {
			pass++
		}
	}
	if r := float64(pass) / trials; r < 0.35 || r > 0.65 {
		t.Fatalf("the no-margin rule passed equal arms %.0f%% of the time; expected about half", 100*r)
	}
}

func TestPerCellNeverVetoes(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 7))
	var cs []Cell
	for k := 0; k < 6; k++ {
		cs = append(cs, synth(rng, string(rune('A'+k)), 12, 64, 1.0, 0))
	}
	// One cell's candidate is much worse; pooled it is diluted but the cell is reported.
	cs = append(cs, synth(rng, "bad", 3, 64, 1.6, 0))
	v, err := NonInferior(cs, Criteria{AgreeMarginPts: 1, KLMargin: 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if bad := v.PerCell["bad"]; bad.KLRatio < 1.3 {
		t.Fatalf("the bad cell's own ratio should be reported (got %.2f)", bad.KLRatio)
	}
	if !v.Pass {
		t.Logf("pooled verdict failed (%v); acceptable only if the pooled bound itself fails, never because of one cell", v.Reasons)
		if v.Pooled.KLRatioUCB <= 1.2 && v.AgreePass && v.FlipsPass {
			t.Fatal("verdict failed although every pooled criterion held: a per-cell veto leaked in")
		}
	}
}

func TestSuperiority_equalArmsRarelyWin(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 7))
	wins := 0
	for i := 0; i < 200; i++ {
		var cs []Cell
		for k := 0; k < 5; k++ {
			cs = append(cs, synth(rng, "c", 10, 64, 1.0, 0))
		}
		if _, ok, err := Superiority(cs, 0.05); err != nil {
			t.Fatal(err)
		} else if ok {
			wins++
		}
	}
	if wins > 10 {
		t.Fatalf("equal arms 'won' a superiority test %d of 200 times", wins)
	}
}

func TestWorstCase(t *testing.T) {
	if n := WorstCasePrompts(0.10, 0.05); n != 29 {
		t.Fatalf("excluding a 10%% bad-prompt rate at 95%% needs 29 prompts, got %d", n)
	}
	if r := WorstCaseExcluded(10, 0.05); math.Abs(r-0.259) > 0.001 {
		t.Fatalf("10 passing prompts exclude %.3f, want ~0.259", r)
	}
}

func TestPromptSetHash(t *testing.T) {
	a := PromptSetHash([][]int{{1, 2, 3}, {4}})
	if a != PromptSetHash([][]int{{1, 2, 3}, {4}}) {
		t.Fatal("hash must be deterministic")
	}
	for _, other := range [][][]int{{{1, 2}, {3, 4}}, {{4}, {1, 2, 3}}, {{1, 2, 3}, {5}}} {
		if PromptSetHash(other) == a {
			t.Fatalf("different prompt sets hashed equal: %v", other)
		}
	}
}

func TestWritePositionsAndErrors(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 7))
	c := synth(rng, "c", 2, 3, 1, 0)
	var buf bytes.Buffer
	if err := WritePositions(&buf, []Cell{c}); err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(buf.String(), "\n"); n != 2*3*2 {
		t.Fatalf("want 12 lines (2 prompts × 3 positions × 2 arms), got %d", n)
	}
	c.Prompts[0].Exact = c.Prompts[0].Exact[:1]
	if _, err := Summarize(c.Prompts, 0.05); err == nil {
		t.Fatal("misaligned arms must be an error")
	}
}
