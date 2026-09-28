package main

// `gate identity -record FILE` (TE6(b), docs/tasks/task-test-efficiency-2026-09.md; owner decision 2026-09-28: a
// family's validation may be inherited by identity). It writes PARITY_ROW lines for the decoder package's
// TestParityManifest_merge, the manifest's one writer, and never writes the manifest itself.
//
// A family is eligible only when identity is proof of the SAME numerics its T3 oracle validated:
//   - its row is `validated` under a T3 method (or an earlier identity-inherited one);
//   - the run used real checkpoints, not the tiny fixtures (a tiny fixture cannot reach the shape-dependent kernel
//     paths a released checkpoint does);
//   - the backend is the CPU, and the inner method is one the CPU reference carries (full-forward-oracle, weightDiff,
//     shared-path). A real-model-oracle row validated a GPU-resident path, which only that backend can inherit;
//   - this machine's arch is the arch the row was validated on (the CPU reference is bit-identical within an arch, not
//     across);
//   - <old-rev> is the row's validated_at, and <new-rev> is HEAD (the merge stamps HEAD);
//   - every cell of the family came back IDENTICAL, covering all three CPU quants (f32, int8int8, int4), so whichever
//     quant the oracle ran at is covered.
//
// The row keeps the original metrics, and its method carries the original method and the rev its oracle ran at:
// "identity-inherited (<method> @ <rev>)". The checkpoint that ran is named in the reference, and the merge is a
// deliberate step after checking it against the row's own reference.

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"
)

var identityRecordQuants = []string{"f32", "int8int8", "int4"}

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

type inheritance struct {
	Family  string
	Row     string   // the PARITY_ROW line, when eligible
	Reasons []string // why not, when not
}

// identityInheritance decides one family. arch is this machine's GOARCH (runtime.GOARCH in the CLI; a parameter so the
// tests can hold it fixed), head the repo's HEAD sha.
func identityInheritance(o identityOpts, old, nw *identitySide, head, arch, host string, mf manifestFamily, f identityFamily) inheritance {
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
		time.Now().Format("2006-01-02"), nw.Short, old.Short, host, strings.Join(assets, ", "),
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

func identityRecordArch() string { return runtime.GOARCH }
