package decoder

import (
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"strings"
	"testing"
)

// S2 (docs/tasks/task-never-swap-2026-09.md): a family's per-layer loader closure is safe to
// stream (build -> sink.layer -> release, the loadQ35/loadGptOss shape) only if EVERY tensor it reads is genuinely
// per-layer: named "blk.{i}." + something, never a bare/model-level name.
//
// gpt-oss's closure is also proven byte-identical to the resident path on a real fixture (decoder/testdata/gptoss_tiny.gguf,
// internal/prequant's TestGptOss_streamedMatchesResident). laguna, granite (the Mamba-2+MoE hybrid, arch.granite, not the
// plain dense Granite family gguf_granite_permute_test.go covers), nemotron and llama4 have no comparable GGUF fixture in
// this repo, so this test is the structural half of the proof: it parses the source and checks the invariant directly, the
// technique stream_test.go's TestTranscode_writesViaTempThenRenames uses for a property real execution cannot force. The
// property (nothing is read that would silently corrupt a streamed bundle) is provable from source, independent of a real
// checkpoint's numbers.
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
	families := []string{"loadLaguna", "loadGranite", "loadNemo", "loadL4", "loadGptOss", "loadQ35", "loadG4"}

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
				"this family's streaming branch must not ship until this is fixed",
				want, badReads)
		}
	}
	for _, want := range families {
		if !found[want] {
			t.Errorf("missing coverage for %s — add it back to the families list once its closure exists again", want)
		}
	}
}
