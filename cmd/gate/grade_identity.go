package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"time"
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
		for s := range n {
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
	var s strings.Builder
	s.WriteString("[")
	for i, p := range m.Prompts {
		if i > 0 {
			s.WriteString(" ")
		}
		s.WriteString(fmt.Sprint(p.Steps))
	}
	return s.String() + "]"
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

// rowArch maps a manifest `machine` string to a GOARCH, or "" when it cannot tell.
func rowArch(machine string) string {
	m := strings.ToLower(machine)
	switch {
	case strings.Contains(m, "arm64"), strings.Contains(m, "mac"), strings.Contains(m, "darwin"):
		return "arm64"
	case strings.Contains(m, "amd64"), strings.Contains(m, "x86_64"), strings.Contains(m, "linux"):
		return "amd64"
	}
	return ""
}

// inheritedMethod parses "identity-inherited (<method> @ <rev>)".
func inheritedMethod(m string) (inner, rev string, ok bool) {
	const pre = "identity-inherited ("
	if !strings.HasPrefix(m, pre) || !strings.HasSuffix(m, ")") {
		return "", "", false
	}
	body := strings.TrimSuffix(strings.TrimPrefix(m, pre), ")")
	i := strings.LastIndex(body, " @ ")
	if i <= 0 || i+3 >= len(body) {
		return "", "", false
	}
	return body[:i], body[i+3:], true
}

func cpuCarriedMethod(m string) bool {
	return m == "full-forward-oracle" || m == "weightDiff" || strings.HasPrefix(m, "shared-path (via ")
}

// identityInheritance decides one family. arch is this machine's GOARCH (runtime.GOARCH in the CLI; a parameter so the
// tests can hold it fixed), head the repo's HEAD sha, date the day the row is stamped with (the run's, recorded).
func identityInheritance(o identityOpts, old, nw *identitySide, head, arch, host, date string, mf manifestFamily, f identityFamily) inheritance {
	in := inheritance{Family: f.Name}
	no := func(format string, a ...any) { in.Reasons = append(in.Reasons, fmt.Sprintf(format, a...)) }
	inner, originRev := mf.Method, mf.ValidatedAt
	if i, r, ok := inheritedMethod(mf.Method); ok {
		inner, originRev = i, r
	}
	if f.Verdict != vIdentical {
		no("verdict %s, not IDENTICAL", f.Verdict)
	}
	if mf.Status != "validated" {
		no("manifest status %q, not validated", mf.Status)
	}
	if !cpuCarriedMethod(inner) {
		no("method %q is not one the CPU reference carries (real-model-oracle inherits only on its own backend)", inner)
	}
	if o.Assets != "real" {
		no("-assets %s: inheriting a T3 needs real checkpoints, not tiny fixtures", o.Assets)
	}
	if o.Backend != "cpu" {
		no("-backend %s: T3 oracles here validate the CPU path", o.Backend)
	}
	if ra := rowArch(mf.Machine); ra == "" {
		no("cannot tell the arch of machine %q", mf.Machine)
	} else if ra != arch {
		no("validated on %s (%s), this run is %s: identity holds within an arch, not across", mf.Machine, ra, arch)
	}
	if mf.ValidatedAt == "" || !strings.HasPrefix(old.SHA, mf.ValidatedAt) {
		no("<old-rev> %s is not the row's validated_at %q", old.Short, mf.ValidatedAt)
	}
	if head == "" || !strings.HasPrefix(nw.SHA, head) && !strings.HasPrefix(head, nw.SHA) {
		no("<new-rev> %s is not HEAD (the merge stamps HEAD)", nw.Short)
	}
	quants := map[string]bool{}
	var assets []string
	seen := map[string]bool{}
	for _, c := range f.Cells {
		if c.Verdict != vIdentical {
			if c.Verdict != vNotRun {
				no("cell %s @ %s is %s", c.Cell.Asset, c.Cell.Quant, c.Verdict)
			}
			continue
		}
		quants[c.Cell.Quant] = true
		if !seen[c.Cell.Asset] {
			seen[c.Cell.Asset] = true
			assets = append(assets, c.Cell.Asset)
		}
	}
	var missing []string
	for _, q := range identityRecordQuants {
		if !quants[q] {
			missing = append(missing, q)
		}
	}
	if len(missing) > 0 {
		no("quants %v not run IDENTICAL (pass -quant %s)", missing, strings.Join(identityRecordQuants, ","))
	}
	if len(in.Reasons) > 0 {
		return in
	}
	sort.Strings(assets)
	ref := fmt.Sprintf("identity-inherited %s: full logits at %s byte-identical to %s (its validated_at), %s/cpu, real "+
		"checkpoint(s) %s, quants %s, %d cell(s), gate identity. Original validation: %s",
		date, nw.Short, old.Short, host, strings.Join(assets, ", "),
		strings.Join(identityRecordQuants, "/"), f.Ran, mf.Reference)
	row := map[string]any{
		"family":    f.Name,
		"method":    fmt.Sprintf("identity-inherited (%s @ %s)", inner, originRev),
		"reference": ref,
		"metrics":   mf.Metrics,
	}
	b, _ := json.Marshal(row)
	in.Row = "PARITY_ROW " + string(b)
	return in
}

// identityGradeInputs is what identity's report reads beyond the judged cells: the run's times and log directory, and
// the host, HEAD, GOARCH and date the inheritance rows are stamped with. The runner resolves them and records them.
type identityGradeInputs struct {
	T0        time.Time     `json:"t0"`
	End       time.Time     `json:"end"`
	BuildWall time.Duration `json:"build_wall"`
	RunWall   time.Duration `json:"run_wall"`
	Logs      string        `json:"logs"`
	Host      string        `json:"host"`
	Head      string        `json:"head"`
	Arch      string        `json:"arch"`
	Date      string        `json:"date"`
}

// gradeIdentity is identity's report and verdict on judged cells, live or replayed: determinism, the per-family
// verdicts, what would be recorded, the -record inheritance rows (handed to writeRows, which a replay answers from the
// record), and the verdict.
func gradeIdentity(w io.Writer, o identityOpts, old, nw *identitySide, manifest map[string]manifestFamily, selected []string,
	outs []cellOutcome, in identityGradeInputs, writeRows func(rows []string) error) int {
	t0, buildWall, runWall, logs := in.T0, in.BuildWall, in.RunWall, in.Logs
	fams := summarizeFamilies(selected, outs)

	fmt.Fprintf(w, "\n%s== determinism: the new build (%s) run twice, separate processes, on %s ==%s\n", bold, nw.Short, o.Backend, off)
	det, nondet, unchecked := 0, 0, 0
	for _, c := range outs {
		switch {
		case c.Verdict == vNotRun:
		case !c.DetChecked:
			unchecked++
		case c.Deterministic:
			det++
		default:
			nondet++
			fmt.Fprintf(w, "  %sNONDETERMINISTIC%s %-44s %s\n", red, off, c.Cell.ID, c.DetDiff.describe())
		}
	}
	fmt.Fprintf(w, "  %d cell(s) byte-identical to themselves, %d not, %d unchecked (second run failed)\n", det, nondet, unchecked)
	if nondet > 0 {
		if o.Backend == "cpu" {
			fmt.Fprintf(w, "  %sKILL CRITERION (TE6(b)): run-to-run nondeterminism on the CPU — impossible under the bit-identical discipline, and a finding in its own right.%s\n", red, off)
		} else {
			fmt.Fprintf(w, "  %s%s is not bitwise deterministic on %d cell(s): those are compared in TOLERANCE MODE (max |diff| <= %.0fx the cell's own run-to-run max |diff|, argmax and tokens exact) and reported WITHIN TOLERANCE, never IDENTICAL.%s\n",
				amber, o.Backend, nondet, identityTolFactor, off)
		}
	}

	fmt.Fprintf(w, "\n%s== identity: old %s vs new %s, per family ==%s\n", bold, old.Short, nw.Short, off)
	nIdent, nTol, nDiff, nNR := 0, 0, 0, 0
	for _, f := range fams {
		col := green
		switch f.Verdict {
		case vDifferent:
			col = red
			nDiff++
		case vTolerant:
			col = amber
			nTol++
		case vNotRun:
			col = amber
			nNR++
		default:
			nIdent++
		}
		cov := fmt.Sprintf("%d/%d cell(s) ran", f.Ran, f.Ran+f.NotRun)
		switch f.Verdict {
		case vIdentical:
			note := "validation inheritable from " + old.Short
			if o.Giw != "" {
				note = "the two builds' sidecars give the same logits"
			}
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s %s\n", col, f.Verdict, off, f.Name, cov, note)
		case vTolerant:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s not byte-identical (nondeterministic backend); tolerance-inheritable only if the owner accepts it\n", col, f.Verdict, off, f.Name, cov)
		case vDifferent:
			note := "goes to its reference gate"
			if o.Giw != "" {
				note = "the two builds' sidecars do not give the same logits"
			}
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s %s\n", col, f.Verdict, off, f.Name, cov, note)
		case vNotRun:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %s\n", col, f.Verdict, off, f.Name, "nothing ran — see NOT RUN")
			continue
		}
		for _, c := range f.Cells {
			if c.Verdict == vNotRun {
				continue
			}
			line := fmt.Sprintf("      %-10s %-38s %s", c.Verdict, c.Cell.Asset+" @ "+c.Cell.Quant, orDash(c.NewPath))
			if c.Verdict != vIdentical || strings.TrimSpace(c.Detail) != "" {
				line += " — " + strings.TrimSpace(c.Detail)
			}
			fmt.Fprintln(w, line)
		}
	}

	printNotRun(w, cellsOf(outs))

	fmt.Fprintf(w, "\n%s== would record (the parity manifest is NOT written by this item) ==%s\n", bold, off)
	host := in.Host
	anyRec := false
	var notValidated []string
	if o.Giw != "" {
		anyRec = true
		fmt.Fprintf(w, "  nothing — -giw compares two builds' sidecars of one source; no family's parity row is inherited from it\n")
	}
	for _, f := range fams {
		if f.Verdict != vIdentical || o.Giw != "" {
			continue
		}
		anyRec = true
		mf := manifest[f.Name]
		va := mf.ValidatedAt
		if va == "" {
			va = "never"
		}
		if mf.ValidatedAt == "" || !strings.HasPrefix(old.SHA, mf.ValidatedAt) {
			notValidated = append(notValidated, f.Name+"@"+va)
		}
		fmt.Fprintf(w, "  %-18s method \"identity-inherited from %s\" at %s — %s, %s, %s assets, %d cell(s); today: %s @ %s (%s)\n",
			f.Name, old.Short, nw.Short, o.Backend, host, o.Assets, f.Ran, mf.Method, va, mf.Machine)
	}
	if !anyRec {
		fmt.Fprintf(w, "  nothing — no family came back IDENTICAL\n")
	}
	if len(notValidated) > 0 {
		fmt.Fprintf(w, "  %sNOTE%s old %s is not the validated_at of %d of these: identity inherits only what %s itself carries —\n"+
			"       chain through the revs that were checked, or pass the family's validated_at as <old-rev>:\n       %s\n",
			amber, off, old.Short, len(notValidated), old.Short, wrapWords(notValidated, 96, "       "))
	}
	fmt.Fprintf(w, "  scope: this machine (%s) and backend (%s) only — identity is per machine and per backend (the CPU reference is bit-identical within an arch, not across).\n", host, o.Backend)

	if o.Record != "" {
		var rows []string
		fmt.Fprintf(w, "\n%s== inheritance (-record %s) ==%s\n", bold, o.Record, off)
		for _, f := range fams {
			if f.Verdict == vNotRun {
				continue
			}
			inh := identityInheritance(o, old, nw, in.Head, in.Arch, host, in.Date, manifest[f.Name], *f)
			if inh.Row != "" {
				rows = append(rows, inh.Row)
				fmt.Fprintf(w, "  %sINHERITS%s %-18s\n", green, off, f.Name)
				continue
			}
			fmt.Fprintf(w, "  not      %-18s %s\n", f.Name, strings.Join(inh.Reasons, "; "))
		}
		if len(rows) == 0 {
			fmt.Fprintf(w, "  nothing eligible; %s not written\n", o.Record)
		} else if err := writeRows(rows); err != nil {
			fmt.Fprintf(w, "  %swrite %s: %v%s\n", red, o.Record, err, off)
		} else {
			fmt.Fprintf(w, "  wrote %d row(s) to %s. Before merging, check each row's checkpoint against its original reference, then:\n"+
				"    GOINFER_MANIFEST_MACHINE=<the row's machine name> go test ./decoder/ -run TestParityManifest_merge -merge-rows %s\n",
				len(rows), o.Record, o.Record)
		}
	}

	end := in.End
	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  families %d: %d IDENTICAL, %d WITHIN TOLERANCE, %d DIFFERENT, %d NOT RUN (a skip is not a pass)\n", len(fams), nIdent, nTol, nDiff, nNR)
	fmt.Fprintf(w, "  wall     %s (%s → %s): builds %s, runs %s\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"), end.Format("15:04:05"),
		buildWall.Round(time.Second), runWall.Round(time.Second))
	keep := "removed (pass -keep to retain dumps)"
	if o.Keep {
		keep = "kept"
	}
	fmt.Fprintf(w, "  logs     %s (dumps/binaries %s; worktrees removed)\n", logs, keep)
	switch {
	case nondet > 0 && o.Backend == "cpu":
		fmt.Fprintf(w, "  %sRED — the CPU is not deterministic run to run on %d cell(s)%s\n", red, nondet, off)
		return 1
	case nDiff > 0:
		fmt.Fprintf(w, "  %sRED — %d family(ies) DIFFERENT: they, and only they, go to their reference gates%s\n", red, nDiff, off)
		return 1
	case nIdent+nTol == 0:
		fmt.Fprintf(w, "  %sREFUSED — no family ran%s\n", amber, off)
		return 2
	}
	fmt.Fprintf(w, "  %sGREEN — every family that ran is %s%s\n", green, map[bool]string{true: "IDENTICAL or WITHIN TOLERANCE", false: "IDENTICAL"}[nTol > 0], off)
	return 0
}
func summarizeFamilies(selected []string, outs []cellOutcome) []*identityFamily {
	by := map[string]*identityFamily{}
	for _, f := range selected {
		by[f] = &identityFamily{Name: f}
	}
	for _, c := range outs {
		f := by[c.Cell.Family]
		f.Cells = append(f.Cells, c)
		if c.Verdict == vNotRun {
			f.NotRun++
		} else {
			f.Ran++
		}
		if c.DetChecked {
			f.DetCheck++
			if !c.Deterministic {
				f.NonDet++
			}
		}
	}
	var out []*identityFamily
	for _, name := range selected {
		f := by[name]
		f.Verdict = vNotRun
		if f.Ran > 0 {
			f.Verdict = vIdentical
		}
		for _, c := range f.Cells {
			switch c.Verdict {
			case vDifferent:
				f.Verdict = vDifferent
			case vTolerant:
				if f.Verdict == vIdentical {
					f.Verdict = vTolerant
				}
			}
		}
		out = append(out, f)
	}
	return out
}

func cellsOf(outs []cellOutcome) []*identityCell {
	var cs []*identityCell
	for _, c := range outs {
		if c.Verdict == vNotRun {
			cc := *c.Cell
			cc.NotRun = c.Detail
			cs = append(cs, &cc)
		}
	}
	return cs
}

func printNotRun(w io.Writer, cells []*identityCell) {
	var nr []*identityCell
	for _, c := range cells {
		if c.NotRun != "" {
			nr = append(nr, c)
		}
	}
	if len(nr) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%s== NOT RUN (%d cell(s)) — reported, never counted as a pass ==%s\n", bold, len(nr), off)
	// Collapse the per-quant copies of one reason.
	type key struct{ fam, asset, why string }
	seen := map[key][]string{}
	var order []key
	for _, c := range nr {
		k := key{c.Family, c.Asset, c.NotRun}
		if _, ok := seen[k]; !ok {
			order = append(order, k)
		}
		seen[k] = append(seen[k], c.Quant)
	}
	for _, k := range order {
		fmt.Fprintf(w, "  %s%-8s%s %-18s %-22s [%s] %s\n", amber, vNotRun, off, k.fam, k.asset, strings.Join(seen[k], ","), trunc(k.why, 150))
	}
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func wrapWords(words []string, width int, indent string) string {
	var b strings.Builder
	col := 0
	for i, wd := range words {
		if i > 0 {
			if col+1+len(wd) > width {
				b.WriteString("\n" + indent)
				col = 0
			} else {
				b.WriteByte(' ')
				col++
			}
		}
		b.WriteString(wd)
		col += len(wd)
	}
	return b.String()
}

var identityRecordQuants = []string{"f32", "int8int8", "int4"}

// dumpBinSize is the bytes of logits a result's prompts call for: one float32 per vocab entry per captured step.
func dumpBinSize(m *dumpMeta) int {
	n := 0
	for _, p := range m.Prompts {
		n += p.Steps
	}
	return n * m.Vocab * 4
}
