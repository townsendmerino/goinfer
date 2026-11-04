//go:build cuda && goinfer_testhooks

package cuda

import (
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"testing"
)

// TestMropePosWindow_absoluteNotChunkRelative pins the plan's top risk directly: mropePosWindow
// must slice the WHOLE-PROMPT, ABSOLUTE-indexed mropePos array by absolute startPos, never by a
// chunk-relative index. A prompt whose (t,h,w) triples differ at every absolute position makes any
// off-by-a-chunk slicing bug visible immediately (a wrong window returns the wrong triples, not a
// coincidentally-correct one).
func TestMropePosWindow_absoluteNotChunkRelative(t *testing.T) {
	whole := make([][3]int, 20)
	for i := range whole {
		whole[i] = [3]int{i, i + 100, i + 200} // every absolute position distinguishable
	}
	// A "second chunk" starting at absolute position 12, width 5 — mirrors prefillChunked's own
	// i>0 pass, where embeddings is chunk-relative (embeddings[i:i+n]) but mropePos must not be.
	got := mropePosWindow(whole, 12, 5)
	want := whole[12:17]
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mropePosWindow(whole, startPos=12, M=5) = %v, want %v (absolute window)", got, want)
	}
	// The bug this test exists to catch: slicing as if startPos were 0 (chunk-relative).
	wrongRelative := whole[0:5]
	if reflect.DeepEqual(got, wrongRelative) {
		t.Fatalf("mropePosWindow returned the CHUNK-RELATIVE window %v — this is the exact regression the function was extracted to prevent", wrongRelative)
	}
}

// TestMropePosWindow_firstChunk pins the startPos=0 case (the unchunked / first-pass path) as its
// own explicit assertion, not just implied by the general case above.
func TestMropePosWindow_firstChunk(t *testing.T) {
	whole := make([][3]int, 8)
	for i := range whole {
		whole[i] = [3]int{i, i, i}
	}
	got := mropePosWindow(whole, 0, 8)
	if !reflect.DeepEqual(got, whole) {
		t.Fatalf("mropePosWindow(whole, 0, len(whole)) = %v, want the whole slice %v", got, whole)
	}
}

// TestPrefillCoreCallsMropePosWindow mirrors TestPrefillCoreCallsCheckPrefillShmemImg's exact
// pattern: proves prefillCore's SOURCE actually calls mropePosWindow, rather than inlining the
// slice expression by hand at the call site — the inlined form is exactly what would silently
// regress to chunk-relative slicing on a future edit, with no test catching it (the real fixture
// is 14 tokens, far below the default chunk width, so it can never exercise the multi-pass loop
// that would make the bug visible).
func TestPrefillCoreCallsMropePosWindow(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "prefill.go", nil, 0)
	if err != nil {
		t.Fatalf("parse prefill.go: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "prefillCore" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("prefillCore not found in prefill.go — this guard is watching nothing")
	}
	found := false
	ast.Inspect(fn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch f := call.Fun.(type) {
		case *ast.Ident:
			if f.Name == "mropePosWindow" {
				found = true
			}
		case *ast.SelectorExpr:
			if f.Sel.Name == "mropePosWindow" {
				found = true
			}
		}
		return true
	})
	if !found {
		t.Error("prefillCore does not call mropePosWindow — the absolute-vs-chunk-relative slicing guard would never fire, and a future edit could silently inline the wrong (chunk-relative) slice")
	}
}
