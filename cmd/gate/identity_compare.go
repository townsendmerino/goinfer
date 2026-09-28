package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// `gate identity`'s COMPARISON. Bytes decide; everything else here only describes a difference.

// dumpMeta mirrors the dumper's per-cell .json (identity_dumper.go.tmpl).
type dumpMeta struct {
	ID         string `json:"id"`
	OK         bool   `json:"ok"`
	Err        string `json:"err,omitempty"`
	DecodePath string `json:"decode_path"`
	Vocab      int    `json:"vocab"`
	Prompts    []struct {
		Len    int   `json:"len"`
		Steps  int   `json:"steps"`
		Tokens []int `json:"tokens"`
	} `json:"prompts"`
	LoadMS int64 `json:"load_ms"`
	RunMS  int64 `json:"run_ms"`
}

// dump is one run's output for one cell.
type dump struct {
	Meta    *dumpMeta
	Bin     []byte
	Missing string // the run produced no result for this cell (crash, timeout), and why
}

func readDump(dir, id, crash string) *dump {
	b, err := os.ReadFile(filepath.Join(dir, id+".json"))
	if err != nil {
		why := "no result written"
		if crash != "" {
			why += " — " + crash
		}
		return &dump{Missing: why}
	}
	var m dumpMeta
	if err := json.Unmarshal(b, &m); err != nil {
		return &dump{Missing: "unreadable result: " + err.Error()}
	}
	d := &dump{Meta: &m}
	d.Bin, _ = os.ReadFile(filepath.Join(dir, id+".bin"))
	return d
}

// dumpDiff describes how two runs of one cell differ.
type dumpDiff struct {
	Identical bool
	Why       string // a structural difference (steps, vocab, sizes, tokens), "" if none
	// First differing logit, located.
	FirstPrompt, FirstStep, FirstLogit int
	Located                            bool
	MaxAbs                             float64 // over the steps both runs captured; +Inf if one side is NaN/Inf where the other is not
	ArgmaxAgree, ArgmaxTotal           int
	FirstArgmaxPrompt, FirstArgmaxStep int // -1 when every compared step agrees
	TokensDiffer                       bool
}

// compareDumps compares run b against run a. Both must have OK metas.
func compareDumps(a, b *dump) dumpDiff {
	d := dumpDiff{FirstArgmaxPrompt: -1, FirstArgmaxStep: -1}
	am, bm := a.Meta, b.Meta
	stepsEqual := len(am.Prompts) == len(bm.Prompts)
	for i := 0; stepsEqual && i < len(am.Prompts); i++ {
		if am.Prompts[i].Steps != bm.Prompts[i].Steps || am.Prompts[i].Len != bm.Prompts[i].Len {
			stepsEqual = false
		}
	}
	tokensEqual := stepsEqual
	for i := 0; tokensEqual && i < len(am.Prompts); i++ {
		tokensEqual = intsEqual(am.Prompts[i].Tokens, bm.Prompts[i].Tokens)
	}
	d.TokensDiffer = !tokensEqual
	if stepsEqual && am.Vocab == bm.Vocab && bytes.Equal(a.Bin, b.Bin) && tokensEqual {
		d.Identical = true
		return d
	}
	switch {
	case am.Vocab != bm.Vocab:
		d.Why = fmt.Sprintf("vocab %d vs %d", am.Vocab, bm.Vocab)
		return d
	case !stepsEqual:
		d.Why = fmt.Sprintf("captured steps differ: %s vs %s", stepsOf(am), stepsOf(bm))
	case len(a.Bin) != len(b.Bin):
		d.Why = fmt.Sprintf("dump sizes %d vs %d bytes", len(a.Bin), len(b.Bin))
	}
	v := am.Vocab
	if v == 0 {
		return d
	}
	// Walk prompt by prompt over the steps both captured.
	offA, offB := 0, 0
	for p := 0; p < len(am.Prompts) && p < len(bm.Prompts); p++ {
		sa, sb := am.Prompts[p].Steps, bm.Prompts[p].Steps
		n := min(sa, sb)
		for s := 0; s < n; s++ {
			ra := floatsAt(a.Bin, offA+s*v, v)
			rb := floatsAt(b.Bin, offB+s*v, v)
			if ra == nil || rb == nil {
				break
			}
			for i := range ra {
				if math.Float32bits(ra[i]) != math.Float32bits(rb[i]) {
					if !d.Located {
						d.Located, d.FirstPrompt, d.FirstStep, d.FirstLogit = true, p, s, i
					}
					x, y := float64(ra[i]), float64(rb[i])
					diff := math.Abs(x - y)
					if math.IsNaN(diff) || math.IsInf(diff, 0) {
						diff = math.Inf(1)
					}
					if diff > d.MaxAbs {
						d.MaxAbs = diff
					}
				}
			}
			d.ArgmaxTotal++
			if argmax32(ra) == argmax32(rb) {
				d.ArgmaxAgree++
			} else if d.FirstArgmaxPrompt < 0 {
				d.FirstArgmaxPrompt, d.FirstArgmaxStep = p, s
			}
		}
		offA += sa * v
		offB += sb * v
	}
	if d.Why == "" && !d.Located && d.TokensDiffer {
		d.Why = "logits identical but the generated tokens differ"
	}
	return d
}

func floatsAt(b []byte, idx, n int) []float32 {
	if (idx+n)*4 > len(b) {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[(idx+i)*4:]))
	}
	return out
}

// argmax32 is the first maximum; NaN never wins (so an all-NaN row is argmax 0 on both sides).
func argmax32(xs []float32) int {
	best := 0
	for i, x := range xs {
		if x > xs[best] || (math.IsNaN(float64(xs[best])) && !math.IsNaN(float64(x))) {
			best = i
		}
	}
	return best
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func stepsOf(m *dumpMeta) string {
	s := "["
	for i, p := range m.Prompts {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprint(p.Steps)
	}
	return s + "]"
}

// describe renders a non-identical diff in one line.
func (d dumpDiff) describe() string {
	s := ""
	if d.Why != "" {
		s = d.Why + "; "
	}
	if d.Located {
		s += fmt.Sprintf("first differing logit: prompt %d step %d (0 = prefill) index %d; max |diff| %.3g", d.FirstPrompt, d.FirstStep, d.FirstLogit, d.MaxAbs)
	}
	if d.ArgmaxTotal > 0 {
		s += fmt.Sprintf("; argmax agrees %d/%d steps", d.ArgmaxAgree, d.ArgmaxTotal)
		if d.FirstArgmaxPrompt >= 0 {
			s += fmt.Sprintf(" (first disagreement prompt %d step %d)", d.FirstArgmaxPrompt, d.FirstArgmaxStep)
		}
	}
	if d.TokensDiffer {
		s += "; generated tokens differ"
	}
	return s
}

// Cell verdicts.
const (
	vIdentical = "IDENTICAL"
	vTolerant  = "WITHIN TOLERANCE"
	vDifferent = "DIFFERENT"
	vNotRun    = "NOT RUN"
)

// cellOutcome is one cell's full result: determinism (new vs new) and identity (old vs new).
type cellOutcome struct {
	Cell          *identityCell
	Verdict       string
	Detail        string
	Deterministic bool // new build run twice was byte-identical
	DetChecked    bool // both new runs produced a result to compare
	DetDiff       dumpDiff
	Tolerance     float64 // set in tolerance mode
	OldPath       string
	NewPath       string
}

// identityTolFactor is the tolerance mode's bar: a backend whose own run-to-run max |diff| on a cell
// is r is compared old-vs-new at max |diff| <= identityTolFactor*r, with every step's argmax agreeing
// and the same tokens generated. Twice the observed noise, because two independent draws of that
// noise can each sit at its maximum, in opposite directions.
const identityTolFactor = 2.0

// judgeCell decides one cell from its three runs (new#1, new#2, old). backend is what was asked for:
// a GPU cell whose NEW build fell back to the CPU is NOT RUN on that backend — a CPU identity
// reported under "metal" would vouch for a device path nothing executed.
func judgeCell(c *identityCell, backend string, n1, n2, old *dump) cellOutcome {
	out := cellOutcome{Cell: c}
	if c.NotRun != "" {
		out.Verdict, out.Detail = vNotRun, c.NotRun
		return out
	}
	errOf := func(d *dump) string {
		if d.Missing != "" {
			return d.Missing
		}
		if !d.Meta.OK {
			return d.Meta.Err
		}
		return ""
	}
	e1, e2, eo := errOf(n1), errOf(n2), errOf(old)
	if n1.Meta != nil {
		out.NewPath = n1.Meta.DecodePath
	}
	if old.Meta != nil {
		out.OldPath = old.Meta.DecodePath
	}
	switch {
	case e1 != "" && eo != "":
		out.Verdict = vNotRun
		out.Detail = "neither build ran it — new: " + e1
		if eo != e1 {
			out.Detail += " | old: " + eo
		}
		return out
	case e1 != "" || eo != "":
		// One build runs the cell and the other cannot: a behaviour change, not a skip.
		out.Verdict = vDifferent
		if e1 != "" {
			out.Detail = "the new build fails where the old runs: " + e1
		} else {
			out.Detail = "the old build fails where the new runs: " + eo
		}
		return out
	}
	if backend != "cpu" && !pathOnBackend(out.NewPath, backend) {
		out.Verdict = vNotRun
		out.Detail = fmt.Sprintf("backend %s declined — the new build ran %q", backend, out.NewPath)
		return out
	}
	if e2 == "" {
		out.DetChecked = true
		out.DetDiff = compareDumps(n1, n2)
		out.Deterministic = out.DetDiff.Identical
	}
	id := compareDumps(old, n1)
	pathNote := ""
	if out.OldPath != out.NewPath {
		pathNote = fmt.Sprintf(" [decode path old %q, new %q]", out.OldPath, out.NewPath)
	}
	switch {
	case !out.DetChecked:
		// Without a second new run there is no determinism evidence; identity alone is still bytes.
		if id.Identical {
			out.Verdict, out.Detail = vIdentical, "(determinism unchecked: second new run failed: "+e2+")"+pathNote
		} else {
			out.Verdict, out.Detail = vDifferent, id.describe()+pathNote
		}
	case out.Deterministic && id.Identical:
		out.Verdict, out.Detail = vIdentical, pathNote
	case out.Deterministic:
		out.Verdict, out.Detail = vDifferent, id.describe()+pathNote
	default:
		// Tolerance mode: this backend is not bitwise reproducible on this cell, so byte identity
		// against the old build means nothing either way.
		det := out.DetDiff
		out.Tolerance = identityTolFactor * det.MaxAbs
		ok := det.Why == "" && id.Why == "" && !id.TokensDiffer && id.FirstArgmaxPrompt < 0 && id.MaxAbs <= out.Tolerance
		if ok {
			out.Verdict = vTolerant
			out.Detail = fmt.Sprintf("old vs new max |diff| %.3g <= %.3g (%.0fx the new build's own run-to-run %.3g); argmax agrees %d/%d%s",
				id.MaxAbs, out.Tolerance, identityTolFactor, det.MaxAbs, id.ArgmaxAgree, id.ArgmaxTotal, pathNote)
		} else {
			out.Verdict = vDifferent
			out.Detail = fmt.Sprintf("outside tolerance %.3g (%.0fx run-to-run %.3g): %s%s", out.Tolerance, identityTolFactor, det.MaxAbs, id.describe(), pathNote)
			if det.Why != "" {
				out.Detail += "; run-to-run itself differs structurally: " + det.Why
			}
		}
	}
	return out
}

// pathOnBackend: DecodePath names the executing backend first ("metal-resident (int4)",
// "webgpu:metal-resident (int4)", "webgpu-staged (…)"); a CPU fallback starts "cpu (".
func pathOnBackend(path, backend string) bool {
	return len(path) >= len(backend) && path[:len(backend)] == backend
}
