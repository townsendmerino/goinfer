package decoder

import (
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"strings"
	"testing"
)

// task-never-swap-2026-09.md S2: a family's per-layer loader closure is safe to stream
// (build -> sink.layer -> release, the loadQ35/loadGptOss shape) only if EVERY tensor it reads is
// genuinely per-layer — named "blk.{i}." + something, never a bare/model-level name. gpt-oss's own
// closure was verified this way by hand before its fix landed (byte-identical to the resident
// path, on a real fixture — decoder/testdata/gptoss_tiny.gguf, internal/prequant's
// TestGptOss_streamedMatchesResident). laguna, granite (the Mamba-2+MoE hybrid, arch.granite —
// not the plain dense Granite family gguf_granite_permute_test.go covers), nemotron and llama4
// have NO comparable fixture (no small, tokenizer-bearing, architecture-correct GGUF for any of
// them exists in this repo or under ~/models at the time this landed) — building one per family
// from scratch is real, separate work this pass did not do (a genuinely different Mamba-2/MoE
// tensor set per family). This test is the structural half of the proof that DID ship with the
// code: same technique stream_test.go's own TestTranscode_writesViaTempThenRenames already uses
// for a property real execution can't force either — parse the source and check the invariant
// directly, since the property this checks (nothing is read that would silently corrupt a
// streamed bundle) IS provable from source, independent of a real checkpoint's numbers.
//
// Verified, not assumed, when this landed: read every one of these four closures by hand,
// confirmed zero non-per-layer tensor reads, THEN wrote this test to keep that true — not the
// other way around (a test written first and never checked against the real code proves nothing).
func TestStreamableFamilyClosures_onlyReadPerLayerTensors(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "gguf.go", nil, 0)
	if err != nil {
		t.Fatalf("parse gguf.go: %v", err)
	}
	render := func(e ast.Expr) string {
		var b strings.Builder
		_ = printer.Fprint(&b, fset, e)
		return b.String()
	}

	// One entry per family this test covers. loaderVar is the closure's own variable name
	// (loadLaguna := func(i int) error { ... }), found by scanning for that assignment anywhere
	// in the file — independent of which enclosing "if arch.X != nil" block it sits in, so a
	// future reordering of the file doesn't silently stop covering a family.
	families := []string{"loadLaguna", "loadGranite", "loadNemo", "loadL4", "loadGptOss", "loadQ35"}

	found := map[string]bool{}
	for _, want := range families {
		var lit *ast.FuncLit
		ast.Inspect(af, func(n ast.Node) bool {
			asn, ok := n.(*ast.AssignStmt)
			if !ok || len(asn.Lhs) != 1 || len(asn.Rhs) != 1 {
				return true
			}
			id, ok := asn.Lhs[0].(*ast.Ident)
			if !ok || id.Name != want {
				return true
			}
			fl, ok := asn.Rhs[0].(*ast.FuncLit)
			if !ok {
				return true
			}
			lit = fl
			return false
		})
		if lit == nil {
			t.Errorf("%s: closure not found in gguf.go — this guard is watching nothing for this family", want)
			continue
		}
		found[want] = true

		var badReads []string
		ast.Inspect(lit, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 {
				return true
			}
			fname := render(call.Fun)
			switch fname {
			case "mat", "vec", "vnorm", "streamMat", "stackedExperts", "stackedExpertBias", "flat":
				// The tensor-name argument is always first for every one of these helpers.
			default:
				return true
			}
			arg := call.Args[0]
			switch a := arg.(type) {
			case *ast.BinaryExpr:
				// The per-layer shape: p + "suffix". Anything else under a BinaryExpr (e.g. two
				// bare literals concatenated) is unexpected enough to flag rather than silently pass.
				if a.Op == token.ADD {
					if id, ok := a.X.(*ast.Ident); ok && id.Name == "p" {
						return true
					}
				}
				badReads = append(badReads, render(arg)+" (non-p+ BinaryExpr)")
			case *ast.BasicLit:
				badReads = append(badReads, render(arg)+" (bare literal, no per-layer prefix at all)")
			default:
				badReads = append(badReads, render(arg)+" (dynamic/unrecognized name expression)")
			}
			return true
		})
		if len(badReads) > 0 {
			t.Errorf("%s reads a tensor name that is not per-layer (blk.{i}.*): %v\n"+
				"a non-per-layer read inside a streaming closure means that data is either missing "+
				"from every layer but the one that happened to read it, or silently duplicated into "+
				"every layer — either way a streamed bundle would NOT match the resident build, and "+
				"this family must not be removed from needsResidentSerialize without fixing this first",
				want, badReads)
		}
	}
	for _, want := range families {
		if !found[want] {
			t.Errorf("missing coverage for %s — add it back to the families list once its closure exists again", want)
		}
	}
}
