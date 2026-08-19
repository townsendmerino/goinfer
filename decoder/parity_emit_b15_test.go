package decoder

// B15 regression gates: the manifest WRITER must not produce claims the manifest READER has to
// reject. One sweep with EMIT_MANIFEST=1 promoted four families from experimental to *validated*
// while their method still said tiny-golden, and wrote mellum's method as "real-oracle" — a
// string no tier rule recognises. TestParityManifest_methodTier caught both, which is that gate
// working; these two catch them at the source, and — the part that matters — they run in plain
// CI, where the emitter itself never does.

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestParityEmit_methodVocabulary is the census that would have caught mellum's typo without a
// heavy run. emitParityRow's own guard fires only under GOINFER_MANIFEST_EMIT with real assets
// present — which is precisely why "real-oracle" survived for months at a call site nobody
// re-read. Scanning the SOURCE needs neither, so the vocabulary is enforced on every push.
func TestParityEmit_methodVocabulary(t *testing.T) {
	// emitParityRow(t, "family", "method", ... — the method is the second string literal.
	call := regexp.MustCompile(`emitParityRow\(t,\s*("(?:[^"\\]|\\.)*"|\w+),\s*"((?:[^"\\]|\\.)*)"`)
	seen := 0
	for path, src := range goTestFiles(t) {
		for _, mm := range call.FindAllStringSubmatch(stripLineComments(src), -1) {
			seen++
			if !knownParityMethod(mm[2]) {
				t.Errorf("%s: emitParityRow method %q is not in the manifest vocabulary %v",
					path, mm[2], sortedKeys(parityMethods))
			}
		}
	}
	// A regex that silently matches nothing would pass forever. Pin a floor well under the real
	// count so a call-shape change fails loudly instead of disabling the census.
	if seen < 10 {
		t.Errorf("found only %d emitParityRow call sites — the census regex no longer matches the "+
			"call shape, so it is not checking anything", seen)
	}
	t.Logf("checked %d emitParityRow call sites against the vocabulary", seen)
}

// TestParityMerge_noPromotionWithoutMethod drives the REAL merge (applyParityRows, the same
// function the -merge-rows tool calls) and asserts the exact defect: a sub-T3 row must not
// leave the family at "validated".
func TestParityMerge_noPromotionWithoutMethod(t *testing.T) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	load := func() *parityManifest {
		var m parityManifest
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("unmarshal manifest: %v", err)
		}
		return &m
	}
	metrics := `{"argmax_pct":100.0,"cosine_min":1.00000,"cosine_mean":1.00000}`
	row := func(fam, meth string) string {
		return `PARITY_ROW {"family":"` + fam + `","method":"` + meth + `","reference":"unit test","metrics":` + metrics + `}`
	}
	statusOf := func(m *parityManifest, fam string) (string, string) {
		f := m.Families[fam]
		return f.Status, methodString(f.Method)
	}

	// THE DEFECT: tiny-golden evidence, merged, previously came out "validated".
	m := load()
	if _, err := applyParityRows(m, row("mixtral", "tiny-golden"), "deadbee", "2026-01-01", "unit"); err != nil {
		t.Fatalf("merge: %v", err)
	}
	if st, meth := statusOf(m, "mixtral"); st != "experimental" || meth != "tiny-golden" {
		t.Errorf("tiny-golden row merged to status=%q method=%q, want experimental/tiny-golden — "+
			"this is B15: the writer promoting a family the tier rule would then reject", st, meth)
	}

	// The other direction still works: a T3 method does validate.
	m = load()
	if _, err := applyParityRows(m, row("mixtral", "real-model-oracle"), "deadbee", "2026-01-01", "unit"); err != nil {
		t.Fatalf("merge: %v", err)
	}
	if st, _ := statusOf(m, "mixtral"); st != "validated" {
		t.Errorf("real-model-oracle row merged to status=%q, want validated", st)
	}

	// DEMOTION, which the old code could not express: a family whose evidence is downgraded
	// must lose "validated", not keep it beside a weaker method.
	m = load()
	if st, _ := statusOf(m, "mellum"); st != "validated" {
		t.Fatalf("precondition: mellum should be validated in the committed manifest, got %q", st)
	}
	if _, err := applyParityRows(m, row("mellum", "tiny-golden"), "deadbee", "2026-01-01", "unit"); err != nil {
		t.Fatalf("merge: %v", err)
	}
	if st, _ := statusOf(m, "mellum"); st != "experimental" {
		t.Errorf("downgraded evidence left mellum at status=%q, want experimental", st)
	}

	// mellum's actual typo, as a row: rejected rather than written.
	m = load()
	if _, err := applyParityRows(m, row("mellum", "real-oracle"), "deadbee", "2026-01-01", "unit"); err == nil {
		t.Error(`method "real-oracle" was accepted — it is one word short of the T3 name ` +
			`"real-model-oracle" and reached the committed manifest once already`)
	}

	// And the merge must still refuse a family it does not know.
	m = load()
	if _, err := applyParityRows(m, row("not_a_family", "real-model-oracle"), "deadbee", "2026-01-01", "unit"); err == nil {
		t.Error("PARITY_ROW for an unknown family was accepted")
	}
}

// stripLineComments blanks // comments so the census counts CALLS, not prose about calls — this
// file and parity_emit_test.go both spell the call shape in their own doc comments, and a census
// that flagged its own documentation would get "fixed" by weakening the regex. Naive on purpose:
// no emitParityRow argument contains "//".
func stripLineComments(src string) string {
	var b strings.Builder
	for line := range strings.SplitSeq(src, "\n") {
		if i := strings.Index(line, "//"); i >= 0 {
			line = line[:i]
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}
