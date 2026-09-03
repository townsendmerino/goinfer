//go:build gpu && goinfer_testhooks

package gpu

import (
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"strings"
	"testing"
)

// srcOf reads one of this package's files for the source-level guards below. Several of these
// defects have no reachable behavioural test on this machine — they need a Nemotron or DeltaNet
// model resident on a GPU — and a test that skipped would be the "a skip is not a pass" trap
// rather than coverage.
func srcOf(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(b)
}

// N-12: BuildResident admits Nemotron regardless of DecodeRunnerEligible, and the block-kind
// switch had NO default — so an unhandled kind (nemoMoE) appended a layer with nil weights and
// decoderunner.go's gemv nil-dereferenced on the first token. decoder/residency.go calls
// DecodeRunnerEligible "the one predicate every backend's admission funnels through" and
// documents declining there; this bypass skips exactly that, which makes the switch the last
// place the shape is checked. Reachable through the exported ResidencyBackend.BuildResident.
func TestNemotronBlockKind_hasADefault(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "residency.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var found, hasDefault bool
	ast.Inspect(af, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok || sw.Tag == nil {
			return true
		}
		var tag strings.Builder
		_ = printer.Fprint(&tag, fset, sw.Tag)
		if !strings.Contains(tag.String(), "NemotronBlockKind") {
			return true
		}
		found = true
		for _, st := range sw.Body.List {
			if cc, ok := st.(*ast.CaseClause); ok && cc.List == nil {
				hasDefault = true
			}
		}
		return true
	})
	if !found {
		t.Fatal("no switch on NemotronBlockKind — this guard is watching nothing")
	}
	if !hasDefault {
		t.Error("the NemotronBlockKind switch has no default: an unhandled kind appends a layer " +
			"with nil weights and nil-derefs at the first token (N-12)")
	}
}

// N-13: GOINFER_SSM_W8A16 binds the INT8-weight kernel (one byte per weight, f32 row scales). A
// *ResidentW4A8 projection packs two 4-bit nibbles per byte with f16 GROUP scales, so
// dispatching it there reads nibble pairs as int8 weights — silent garbage, not an error.
func TestSSMW8A16_declinesInt4Projections(t *testing.T) {
	src := srcOf(t, "decoderunner.go")
	if !strings.Contains(src, "ResidentW4A8") || !strings.Contains(src, "GOINFER_SSM_W8A16") {
		t.Fatal("the W8A16 flag or the W4A8 type moved; this guard is watching nothing")
	}
	// The flag must be cleared when an int4 projection is present, not merely warned about.
	if !strings.Contains(src, "w8a16 = false") {
		t.Error("GOINFER_SSM_W8A16 is not cleared when a projection is *ResidentW4A8: the int8 " +
			"kernel would read nibble pairs as int8 weights (N-13)")
	}
}

// N-14: ensureAttnWide compiles THREE variants (f32, f16 KV, int8 KV) and returned early when
// the FIRST existed — so a failure on the second or third left those nil while the next call
// reported success, and a kvF16/kvI8 plan then bound a nil pipeline. The R-30 class, in the file
// R-30 fixed.
func TestEnsureAttnWide_guardsOnTheLastPipeline(t *testing.T) {
	src := srcOf(t, "attention.go")
	i := strings.Index(src, "func (c *Context) ensureAttnWide()")
	if i < 0 {
		t.Fatal("ensureAttnWide not found")
	}
	head := src[i:min(i+700, len(src))]
	if strings.Contains(head, "if c.attnWidePipeline != nil {") {
		t.Error("ensureAttnWide early-returns on attnWidePipeline, the FIRST of three variants: " +
			"a failure on the f16 or int8 variant leaves it nil while the next call reports " +
			"success (N-14)")
	}
	if !strings.Contains(head, "if c.attnI8WidePipeline != nil {") {
		t.Error("ensureAttnWide does not guard on the LAST variant it assigns")
	}
}

// N-15: DecodeRunner.ReadMambaCap is EXPORTED and panicked on a failed buffer map — "test-only"
// is a comment, not a compiler constraint. And residentDecoder.Reset dropped every WriteBuffer
// error, so a failed C-01 re-zero left the previous sequence's recurrent state in place and the
// next generation continued from it silently.
func TestGPU_deviceBoundaryErrorsAreNotDropped(t *testing.T) {
	dr := srcOf(t, "decoderunner.go")
	i := strings.Index(dr, "func (r *DecodeRunner) ReadMambaCap")
	if i < 0 {
		t.Fatal("ReadMambaCap not found")
	}
	body := dr[i:min(i+2000, len(dr))]
	if strings.Contains(body, `panic("ReadMambaCap`) {
		t.Error("ReadMambaCap still panics on a failed map; it is exported, so that takes the " +
			"caller's process down (N-15)")
	}
	if !strings.Contains(dr[i:min(i+200, len(dr))], "err error)") {
		t.Error("ReadMambaCap does not return an error")
	}

	rs := srcOf(t, "residency.go")
	j := strings.Index(rs, "func (rd *residentDecoder) Reset()")
	if j < 0 {
		t.Fatal("residentDecoder.Reset not found")
	}
	reset := rs[j:min(j+1600, len(rs))]
	for _, ln := range strings.Split(reset, "\n") {
		s := strings.TrimSpace(ln)
		if strings.HasPrefix(s, "//") {
			continue
		}
		if strings.HasPrefix(s, "rd.c.queue.WriteBuffer(") {
			t.Errorf("Reset drops a WriteBuffer error: %s\n"+
				"a failed re-zero leaves the previous sequence's recurrent state resident and "+
				"the next generation continues from it (N-15)", s)
		}
	}
	if !strings.Contains(reset, "rd.resetErr") {
		t.Error("Reset records no error; it cannot return one (the cross-backend interface " +
			"returns nothing), so it must record for Forward to surface")
	}
	// And Forward/ForwardN must actually surface it — recording without surfacing is the same
	// silent failure with an extra field.
	if strings.Count(rs, "return nil, rd.resetErr") < 2 {
		t.Error("resetErr is recorded but not surfaced by both Forward and ForwardN")
	}
}
