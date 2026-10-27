//go:build gpu

package kernels

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/mem"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
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
	isBuf := func(e ast.Expr) bool {
		t := info.Types[e].Type
		return t != nil && strings.HasSuffix(t.String(), "wgpu.Buffer")
	}
	releasing := func(t types.Type) bool { // an interface with a Release method: a buffer stored in one is freed by dispatch
		it, ok := t.Underlying().(*types.Interface)
		if !ok {
			return false
		}
		for i := range it.NumMethods() {
			if it.Method(i).Name() == "Release" {
				return true
			}
		}
		return false
	}
	for _, f := range files {
		if filepath.Base(fset.Position(f.Pos()).Filename) == "ledger.go" {
			continue
		}
		// A buffer converted to an interface with a Release method (appended to a release list, in a slice literal, or passed
		// as an argument) is freed by dispatch, bypassing releaseBuf.
		ast.Inspect(f, func(n ast.Node) bool {
			flag := func(e ast.Expr) {
				out = append(out, fmt.Sprintf("%s: a *wgpu.Buffer kept as an interface with Release", fset.Position(e.Pos())))
			}
			switch x := n.(type) {
			case *ast.CallExpr:
				if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "append" && len(x.Args) > 1 {
					if sl, ok := info.Types[x.Args[0]].Type.Underlying().(*types.Slice); ok && releasing(sl.Elem()) {
						for _, a := range x.Args[1:] {
							if isBuf(a) {
								flag(a)
							}
						}
					}
					return true
				}
				if sig, ok := info.Types[x.Fun].Type.(*types.Signature); ok {
					for i, a := range x.Args {
						if i < sig.Params().Len() && releasing(sig.Params().At(i).Type()) && isBuf(a) {
							flag(a)
						}
					}
				}
			case *ast.CompositeLit:
				if sl, ok := info.Types[x].Type.Underlying().(*types.Slice); ok && releasing(sl.Elem()) {
					for _, e := range x.Elts {
						if isBuf(e) {
							flag(e)
						}
					}
				}
			}
			return true
		})
	}
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
		GoFiles, CgoFiles       []string
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
		"func f(d *wgpu.Device, b *wgpu.Buffer, p *wgpu.ComputePipeline) { d.CreateBuffer(nil); b.Release(); b.Destroy(); p.Release(); k := b.Release; _ = k\n"+
		"\tvar keep []interface{ Release() }; keep = append(keep, b, p); _ = keep }\n", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := wgpuCalls(t, fset, "x", []*ast.File{plant}, exp); len(got) != 5 {
		t.Fatalf("the planted file gives %d, want 5 (a buffer's Release kept as a method value, and a buffer appended to a release list, count; a pipeline's do not): %v", len(got), got)
	}
	read := 0
	for _, path := range pkgs {
		p := dirs[path]
		var files []*ast.File
		for _, name := range append(p.GoFiles, p.CgoFiles...) {
			f, err := parser.ParseFile(fset, filepath.Join(p.Dir, name), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			files = append(files, f)
		}
		read += len(files)
		for _, c := range wgpuCalls(t, fset, path, files, exp) {
			t.Errorf("%s: outside the ledger's wrappers (create with c.createBuffer and its siblings, free with releaseBuf)", c)
		}
	}
	if read < 50 { // 61 files across the two packages when written: far fewer means the listing is mis-scoped
		t.Fatalf("read %d files", read)
	}
	t.Logf("type-checked %d non-test files", read)
}

// TestLedger_anUnreleasedBufferIsCounted (G-MM3bf-2's plant): a leased buffer nobody releases, once the GC frees it, closes its
// lease through the collection cleanup and is counted on its Context (Collected), which the I2 check reports; a released one is
// not counted.
func TestLedger_anUnreleasedBufferIsCounted(t *testing.T) {
	leakExempt.Store(true) // the dropped buffer is the point of this test; set before Close's defer so it covers Close
	defer leakExempt.Store(false)
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	kept, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: 256, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatal(err)
	}
	releaseBuf(kept)
	func() {
		if _, err := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: 512, Usage: wgpu.BufferUsageStorage}); err != nil {
			t.Fatal(err)
		}
	}()
	for i := 0; i < 50; i++ {
		if n, _ := c.Collected(); n > 0 {
			break
		}
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
	}
	if n, b := c.Collected(); n != 1 || b != 512 {
		t.Fatalf("collected %d buffers, %d bytes; want 1, 512 (the dropped one only)", n, b)
	}
	if u := c.Ledger().Usage(); u.Leases != 0 {
		t.Fatalf("%d leases open after the dropped buffer was collected", u.Leases)
	}
}
