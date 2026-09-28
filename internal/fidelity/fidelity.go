// Package fidelity grades a candidate path against an exact path, both scored position by position against a reference
// (the prefill and decode fidelity gates' protocol), under the criteria TE12 of
// docs/tasks/task-test-efficiency-2026-09.md pre-registers for the next gate that uses the protocol.
//
// What changed from the per-file criteria the gates carry today, and why:
//
//   - Every "no worse than" criterion has a margin. Without one it is a superiority test: equal arms pass it about half
//     the time at any prompt count. TE12 measured the Metal pooled KL criterion (candidate <= exact, lower on at least
//     half the prompts) failing 17 of 47 verdicts that way. NonInferior refuses a margin <= 0; Superiority is its own
//     function, so a pre-registration has to name it.
//   - Agreement and hard flips are pooled across the gate's cells. Per-cell values are reported, never a per-cell veto
//     (task-prefill-gap.md §3.2).
//   - The PROMPT is the sampling unit for agreement and KL. Positions within a prompt are correlated, so a bound computed
//     as if 64 positions × 10 prompts were 640 independent draws is too narrow.
//   - A worst-case claim ("no prompt regresses") is a separate criterion. It states the bad-prompt rate it excludes, and
//     WorstCasePrompts gives the prompt count that excludes it.
//   - Prompt sets are hashed (PromptSetHash), so a reference is never scored against logits for different text.
//
// Sizing is scripts/power.py's job (its `fidelity` subcommand uses the same formulas as this package).
package fidelity

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

// Position is one scored continuation position of one arm against the reference.
type Position struct {
	Agree    bool    // argmax(arm) == argmax(reference)
	HardFlip bool    // a flip outside the near-tie band (decoder.NearTieArgmaxForTest's hf)
	KL       float64 // KL(reference ‖ arm)
}

// Prompt is one prompt of one cell: the candidate's and the exact arm's positions, index-aligned.
type Prompt struct {
	ID        string
	Candidate []Position
	Exact     []Position
}

// Cell is one (model, K) cell of the gate.
type Cell struct {
	Name    string
	Prompts []Prompt
}

// Criteria is what a gate pre-registers. Margins are in natural units: AgreeMarginPts in percentage points of top-1
// agreement, KLMargin as a fraction of the exact arm's mean KL (0.10 means the candidate's mean may be up to 1.10×).
type Criteria struct {
	AgreeMarginPts float64
	KLMargin       float64
	Alpha          float64 // one-sided; 0.05 if zero
}

// Summary is one arm-pair's totals over a set of prompts.
type Summary struct {
	Prompts, Positions          int
	FlipsCand, FlipsExact       int
	AgreeCandPct, AgreeExactPct float64 // pooled over positions, percent
	MeanKLCand, MeanKLExact     float64 // mean over prompts of each prompt's mean KL
	AgreeDiffPts, AgreeDiffLCB  float64 // candidate − exact, prompt-clustered, and its one-sided lower bound
	KLRatio, KLRatioUCB         float64 // mean KL candidate ÷ exact (ratio of means), and its one-sided upper bound
	DiscordantPositions         int     // positions where exactly one arm agrees with the reference
}

// Verdict is the pooled decision, with the per-cell summaries reported alongside.
type Verdict struct {
	Criteria          Criteria
	Pooled            Summary
	PerCell           map[string]Summary
	FlipsPass         bool
	AgreePass, KLPass bool
	Pass              bool
	Reasons           []string
}

func z(alpha float64) float64 {
	if alpha <= 0 {
		alpha = 0.05
	}
	return math.Sqrt2 * math.Erfinv(1-2*alpha)
}

func meanPos(ps []Position, f func(Position) float64) float64 {
	if len(ps) == 0 {
		return 0
	}
	var s float64
	for _, p := range ps {
		s += f(p)
	}
	return s / float64(len(ps))
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Summarize pools prompts. Agreement and KL bounds cluster by prompt; flips and agreement rates pool positions.
func Summarize(prompts []Prompt, alpha float64) (Summary, error) {
	var s Summary
	zz := z(alpha)
	var dAgree, klC, klE []float64
	var agreeC, agreeE float64
	for _, p := range prompts {
		if len(p.Candidate) != len(p.Exact) || len(p.Candidate) == 0 {
			return s, fmt.Errorf("fidelity: prompt %q: %d candidate vs %d exact positions", p.ID, len(p.Candidate), len(p.Exact))
		}
		s.Prompts++
		s.Positions += len(p.Candidate)
		for i := range p.Candidate {
			c, e := p.Candidate[i], p.Exact[i]
			if c.HardFlip {
				s.FlipsCand++
			}
			if e.HardFlip {
				s.FlipsExact++
			}
			agreeC += b2f(c.Agree)
			agreeE += b2f(e.Agree)
			if c.Agree != e.Agree {
				s.DiscordantPositions++
			}
		}
		dAgree = append(dAgree, 100*(meanPos(p.Candidate, func(x Position) float64 { return b2f(x.Agree) })-
			meanPos(p.Exact, func(x Position) float64 { return b2f(x.Agree) })))
		klC = append(klC, meanPos(p.Candidate, func(x Position) float64 { return x.KL }))
		klE = append(klE, meanPos(p.Exact, func(x Position) float64 { return x.KL }))
	}
	if s.Prompts == 0 {
		return s, errors.New("fidelity: no prompts")
	}
	n := float64(s.Prompts)
	s.AgreeCandPct = 100 * agreeC / float64(s.Positions)
	s.AgreeExactPct = 100 * agreeE / float64(s.Positions)
	md, sd := meanSD(dAgree)
	s.AgreeDiffPts = md
	s.AgreeDiffLCB = md - zz*sd/math.Sqrt(n)
	mc, _ := meanSD(klC)
	me, _ := meanSD(klE)
	s.MeanKLCand, s.MeanKLExact = mc, me
	if me > 0 {
		// Ratio of means with a delta-method bound, clustered by prompt: Var(R) ≈ Var(c_i − R·e_i) / (n·ē²).
		r := mc / me
		resid := make([]float64, len(klC))
		for i := range klC {
			resid[i] = klC[i] - r*klE[i]
		}
		_, sr := meanSD(resid)
		s.KLRatio = r
		s.KLRatioUCB = r + zz*sr/(math.Sqrt(n)*me)
	} else {
		s.KLRatio, s.KLRatioUCB = math.Inf(1), math.Inf(1)
	}
	return s, nil
}

func meanSD(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	var m float64
	for _, x := range xs {
		m += x
	}
	m /= float64(len(xs))
	if len(xs) < 2 {
		return m, 0
	}
	var v float64
	for _, x := range xs {
		v += (x - m) * (x - m)
	}
	return m, math.Sqrt(v / float64(len(xs)-1))
}

// NonInferior grades the gate's decision cells, pooled. It refuses a criterion without a positive margin: a
// "no worse than" test with margin 0 is a superiority test, and must be registered as one (Superiority).
func NonInferior(cells []Cell, c Criteria) (Verdict, error) {
	if c.AgreeMarginPts <= 0 || c.KLMargin <= 0 {
		return Verdict{}, fmt.Errorf("fidelity: a non-inferiority criterion needs a positive margin (agreement %g pts, KL %g); "+
			"with none it is a superiority test — register Superiority explicitly", c.AgreeMarginPts, c.KLMargin)
	}
	if c.Alpha == 0 {
		c.Alpha = 0.05
	}
	v := Verdict{Criteria: c, PerCell: map[string]Summary{}}
	var all []Prompt
	for _, cell := range cells {
		s, err := Summarize(cell.Prompts, c.Alpha)
		if err != nil {
			return v, fmt.Errorf("cell %s: %w", cell.Name, err)
		}
		v.PerCell[cell.Name] = s
		all = append(all, cell.Prompts...)
	}
	p, err := Summarize(all, c.Alpha)
	if err != nil {
		return v, err
	}
	v.Pooled = p
	// Hard flips: Poisson-noise-aware, pooled, exactly the form the gates use today.
	v.FlipsPass = float64(p.FlipsCand) <= float64(p.FlipsExact)+2*math.Sqrt(float64(p.FlipsExact))
	v.AgreePass = p.AgreeDiffLCB >= -c.AgreeMarginPts
	v.KLPass = p.KLRatioUCB <= 1+c.KLMargin
	v.Pass = v.FlipsPass && v.AgreePass && v.KLPass
	if !v.FlipsPass {
		v.Reasons = append(v.Reasons, fmt.Sprintf("hard flips %d > exact %d + 2√exact", p.FlipsCand, p.FlipsExact))
	}
	if !v.AgreePass {
		v.Reasons = append(v.Reasons, fmt.Sprintf("agreement lower bound %+.2f pts < −%.2f", p.AgreeDiffLCB, c.AgreeMarginPts))
	}
	if !v.KLPass {
		v.Reasons = append(v.Reasons, fmt.Sprintf("KL ratio upper bound %.3f > %.3f", p.KLRatioUCB, 1+c.KLMargin))
	}
	return v, nil
}

// Superiority is the margin-free test, named: the candidate's KL upper bound must be below the exact arm's (ratio < 1)
// and its agreement lower bound above zero. Registering it means claiming the candidate is better, not merely no worse.
func Superiority(cells []Cell, alpha float64) (Summary, bool, error) {
	var all []Prompt
	for _, cell := range cells {
		all = append(all, cell.Prompts...)
	}
	s, err := Summarize(all, alpha)
	if err != nil {
		return s, false, err
	}
	return s, s.KLRatioUCB < 1 && s.AgreeDiffLCB > 0, nil
}

// WorstCasePrompts is the number of prompts, all of them passing, that excludes a bad-prompt rate of rate or more at
// confidence 1−alpha (the zero-failure binomial bound: (1−rate)^n <= alpha). 10 prompts exclude only ~26% at 95%.
func WorstCasePrompts(rate, alpha float64) int {
	if rate <= 0 || rate >= 1 || alpha <= 0 || alpha >= 1 {
		return 0
	}
	return int(math.Ceil(math.Log(alpha) / math.Log(1-rate)))
}

// WorstCaseExcluded is the bad-prompt rate that n passing prompts exclude at confidence 1−alpha.
func WorstCaseExcluded(n int, alpha float64) float64 {
	if n <= 0 {
		return 1
	}
	return 1 - math.Pow(alpha, 1/float64(n))
}

// PromptSetHash is the sha256 of a prompt set's token ids (length-prefixed, in order). A reference cached for one prompt
// set must never be scored against another; TE6(a)'s reference key includes this hash.
func PromptSetHash(prompts [][]int) string {
	h := sha256.New()
	var b [8]byte
	for _, p := range prompts {
		binary.LittleEndian.PutUint64(b[:], uint64(len(p)))
		h.Write(b[:])
		for _, t := range p {
			binary.LittleEndian.PutUint64(b[:], uint64(int64(t)))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// WritePositions writes one JSON line per (cell, prompt, position, arm), the per-position log TE12 asks the next run of
// the protocol to keep, so the within-prompt vs between-prompt variance split can size positions per prompt.
func WritePositions(w io.Writer, cells []Cell) error {
	enc := json.NewEncoder(w)
	names := make([]string, 0, len(cells))
	for _, c := range cells {
		names = append(names, c.Name)
	}
	sort.Strings(names)
	byName := map[string]Cell{}
	for _, c := range cells {
		byName[c.Name] = c
	}
	for _, name := range names {
		for _, p := range byName[name].Prompts {
			for i := range p.Candidate {
				for _, arm := range []struct {
					n string
					x Position
				}{{"candidate", p.Candidate[i]}, {"exact", p.Exact[i]}} {
					if err := enc.Encode(map[string]any{"cell": name, "prompt": p.ID, "pos": i, "arm": arm.n,
						"agree": arm.x.Agree, "hard_flip": arm.x.HardFlip, "kl": arm.x.KL}); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}
