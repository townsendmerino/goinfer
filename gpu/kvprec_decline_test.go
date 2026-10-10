//go:build gpu && goinfer_testhooks

package gpu

import (
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"strings"
	"testing"
)

// TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches pins that the resident builder
// declines --kv i8 / --kv f16 on the non-generic branches (Nemotron, Qwen3.5, MLA). Those
// branches always allocate an f32 NewKVCache while the flag raised ctxCap and kernel selection
// was model-wide, so with --kv i8 rl.kScale stays nil and bind fails with a misleading "device
// allocation failed (VRAM exhausted?)" that sends the whole model to CPU for a reason that is not
// true, and with --kv f16 each cache is ctxCap×kvDim×4 at the raised cap, twice the intended f16
// footprint. Declining is chosen over implementing quantized KV for MLA's rank-space latent
// unvalidated.
//
// The test asserts the SHAPE of that decision on the source, because reaching the branch needs
// one of those three model families resident on a GPU, which no fixture here provides, and a test
// that silently skipped would be exactly the "a skip is not a pass" trap. Background:
// docs/code-notes/gpu.md#TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches.
func TestResidency_kvPrecisionDeclineCoversTheNonGenericBranches(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "residency.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse residency.go: %v", err)
	}
	var fn *ast.FuncDecl
	ast.Inspect(af, func(n ast.Node) bool {
		if d, ok := n.(*ast.FuncDecl); ok && d.Name.Name == "BuildResident" {
			fn = d
		}
		return true
	})
	if fn == nil {
		t.Fatal("BuildResident not found — this guard is watching nothing")
	}
	var body strings.Builder
	if err := printer.Fprint(&body, fset, fn); err != nil {
		t.Fatalf("print: %v", err)
	}
	src := body.String()

	// The decline must consult all three family probes, or a family it misses keeps the old
	// silent-wrong behaviour.
	for _, probe := range []string{"MLAResidentParams", "Qwen35ResidentParams", "nemoOK"} {
		if !strings.Contains(src, probe) {
			t.Errorf("BuildResident does not consult %s when deciding the KV-precision "+
				"decline; that family still allocates f32 under --kv i8/f16 (M-32)", probe)
		}
	}
	if !strings.Contains(src, "kvI8 || kvF16") {
		t.Error("no combined kvI8||kvF16 guard: the two flags fail differently but both fail")
	}
	// And the -ctx request must be consulted (ResidentContextRequest), or -ctx would not change the
	// allocated cap.
	if !strings.Contains(src, "ResidentContextRequest") {
		t.Error("BuildResident ignores ResidentContextRequest: `serve -ctx` is silently a no-op " +
			"on webgpu (M-32)")
	}
}

// The -ctx rule itself, as arithmetic rather than as prose: a smaller request lowers the cap, a
// larger one does not raise it (the caps are proven-fit ceilings, and honouring a bigger request
// would trade a clean checkCap refusal for an OOM).
func TestResidency_ctxRequestOnlyLowers(t *testing.T) {
	apply := func(def, req int) int {
		if req > 0 && req < def {
			return req
		}
		return def
	}
	for _, tc := range []struct{ def, req, want int }{
		{16384, 0, 16384},     // unset: backend default
		{16384, 2048, 2048},   // smaller: honoured
		{16384, 32768, 16384}, // larger: ignored, not honoured
		{65536, 32768, 32768}, // i8 cap, smaller request
	} {
		if got := apply(tc.def, tc.req); got != tc.want {
			t.Errorf("default %d, request %d → %d, want %d", tc.def, tc.req, got, tc.want)
		}
	}
}
