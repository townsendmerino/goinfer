//go:build gpu

package kernels

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoBufferLeak pins that LiveBufferBytes returns to its pre-call baseline after each path
// that once leaked accounting while the GPU memory itself was released: Readback (readbackRaw now
// owns the throwaway *DeviceBuffer wrapper and closes it) and FusedMLP (the xn/mid buffer is
// wrapped once, not twice). It drives each path several times and asserts the exact baseline
// every time, which a threshold check could pass for a while on a slow leak. The resident weight
// fixture (rmsWDev/gateRM/upRM/downRM) is created once and stays live across iterations on
// purpose, so the baseline is taken after it. bufaccount.go points callers here. Background:
// docs/code-notes/gpu.md#TestNoBufferLeak.
func TestNoBufferLeak(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	t.Run("FusedMLP", func(t *testing.T) {
		f := newMLPFixture(t, ctx, 256, 512)
		defer f.close()
		if _, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false); err != nil {
			t.Fatalf("warm-up FusedMLP: %v", err)
		}
		base := LiveBufferBytes()
		for i := range 5 {
			if _, err := ctx.FusedMLP(f.x, f.rmsWDev, f.gateRM, f.upRM, f.downRM, f.eps, false); err != nil {
				t.Fatalf("FusedMLP iter %d: %v", i, err)
			}
			if got := LiveBufferBytes(); got != base {
				t.Errorf("FusedMLP iter %d: LiveBufferBytes = %d, want it back at the baseline %d "+
					"(V-22: a wrapper is being alloc'd and never Close'd)", i, got, base)
			}
		}
	})

	t.Run("vision host wrappers", func(t *testing.T) {
		// Reuse TestVisionLayerNorm_parity's shape: a smaller ad hoc size once SIGTRAPed inside the
		// wgpu-native driver on CreateBuffer, unrelated to the accounting under test.
		const rows, h = 257, 1152
		src := make([]float32, rows*h)
		w := make([]float32, h)
		b := make([]float32, h)
		for i := range src {
			src[i] = float32(i%7) - 3
		}
		for i := range w {
			w[i] = 1
		}
		if _, err := ctx.LayerNormRowsHost(src, w, b, rows, h, 1e-6); err != nil {
			t.Fatalf("warm-up LayerNormRowsHost: %v", err)
		}
		if _, err := ctx.softmaxRowsHost(src, rows, h, 0.1); err != nil {
			t.Fatalf("warm-up softmaxRowsHost: %v", err)
		}
		if _, err := ctx.geluHost(src); err != nil {
			t.Fatalf("warm-up geluHost: %v", err)
		}
		base := LiveBufferBytes()
		for i := range 5 {
			if _, err := ctx.LayerNormRowsHost(src, w, b, rows, h, 1e-6); err != nil {
				t.Fatalf("LayerNormRowsHost iter %d: %v", i, err)
			}
			if _, err := ctx.softmaxRowsHost(src, rows, h, 0.1); err != nil {
				t.Fatalf("softmaxRowsHost iter %d: %v", i, err)
			}
			if _, err := ctx.geluHost(src); err != nil {
				t.Fatalf("geluHost iter %d: %v", i, err)
			}
			if got := LiveBufferBytes(); got != base {
				t.Errorf("iter %d: LiveBufferBytes = %d, want it back at the baseline %d "+
					"(V-22: sd/wd/bd/xd released raw, bypassing Close/accountFree)", i, got, base)
			}
		}
	})
}

// wgpuCalls type-checks files (package path, against exp's compiled export data) and returns each call that creates a buffer on a
// *wgpu.Device, or releases or destroys a *wgpu.Buffer (called or as a method value), outside the ledger's own wrappers (ledger.go).
func wgpuCalls(t *testing.T, fset *token.FileSet, path string, files []*ast.File, exp map[string]string) []string {
	t.Helper()
	imp := importer.ForCompiler(fset, "gc", func(p string) (io.ReadCloser, error) {
		f, ok := exp[p]
		if !ok {
			return nil, fmt.Errorf("no export data for %s", p)
		}
		return os.Open(f)
	})
	info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}}
	if _, err := (&types.Config{Importer: imp, FakeImportC: true}).Check(path, fset, files, info); err != nil {
		t.Fatalf("type-check %s: %v", path, err)
	}
	var out []string
	for _, f := range files {
		if filepath.Base(fset.Position(f.Pos()).Filename) == "ledger.go" {
			continue
		}
		// A buffer's Release or Destroy counts whether it is called or kept as a method value (b.Release appended to a
		// release list frees the buffer later, the same as a call).
		ast.Inspect(f, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			recv := info.Types[sel.X].Type
			if recv == nil {
				return true
			}
			switch r := recv.String(); {
			case strings.HasSuffix(r, "wgpu.Device") && strings.Contains(sel.Sel.Name, "CreateBuffer"),
				strings.HasSuffix(r, "wgpu.Buffer") && (sel.Sel.Name == "Release" || sel.Sel.Name == "Destroy"):
				out = append(out, fmt.Sprintf("%s: %s.%s", fset.Position(sel.Pos()), r, sel.Sel.Name))
			}
			return true
		})
	}
	return out
}

// TestLedger_everyBufferIsLeased (G-MM3b-1, I1 and I2 static): in the gpu module's non-test code (this package and the gpu
// package, with the test-hook files), no buffer is created on a *wgpu.Device, nor a *wgpu.Buffer released or destroyed (called,
// or kept as a method value), except
// through the ledger's wrappers, so every buffer is a lease and every free closes one. Typed, so a pipeline's or a bind group's
// Release is not mistaken for a buffer's; it goes red on a planted call of each kind.
func TestLedger_everyBufferIsLeased(t *testing.T) {
	pkgs := []string{"github.com/townsendmerino/goinfer/gpu/internal/kernels", "github.com/townsendmerino/goinfer/gpu"}
	out, err := exec.Command("go", append([]string{"list", "-export", "-deps", "-json", "-tags", "gpu goinfer_testhooks"}, pkgs...)...).Output()
	if err != nil {
		t.Fatalf("go list -export: %v", err)
	}
	type listed struct {
		ImportPath, Export, Dir string
		GoFiles                 []string
	}
	exp, dirs := map[string]string{}, map[string]listed{}
	for dec := json.NewDecoder(bytes.NewReader(out)); dec.More(); {
		var p listed
		if err := dec.Decode(&p); err != nil {
			t.Fatal(err)
		}
		exp[p.ImportPath], dirs[p.ImportPath] = p.Export, p
	}
	fset := token.NewFileSet()
	plant, err := parser.ParseFile(fset, "plant.go", "package x\nimport \"github.com/oliverbestmann/webgpu/wgpu\"\n"+
		"func f(d *wgpu.Device, b *wgpu.Buffer, p *wgpu.ComputePipeline) { d.CreateBuffer(nil); b.Release(); b.Destroy(); p.Release(); k := b.Release; _ = k }\n", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := wgpuCalls(t, fset, "x", []*ast.File{plant}, exp); len(got) != 4 {
		t.Fatalf("the planted file gives %d, want 4 (a buffer's Release kept as a method value counts; a pipeline's does not): %v", len(got), got)
	}
	read := 0
	for _, path := range pkgs {
		p := dirs[path]
		var files []*ast.File
		for _, name := range p.GoFiles {
			f, err := parser.ParseFile(fset, filepath.Join(p.Dir, name), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			files = append(files, f)
		}
		read += len(files)
		for _, c := range wgpuCalls(t, fset, path, files, exp) {
			t.Errorf("%s: outside the ledger's wrappers (create with c.createBuffer and its siblings, free with releaseBuf or destroyBuf)", c)
		}
	}
	if read < 50 { // 61 files across the two packages when written: far fewer means the listing is mis-scoped
		t.Fatalf("read %d files", read)
	}
	t.Logf("type-checked %d non-test files", read)
}
