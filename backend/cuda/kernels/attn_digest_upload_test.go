package kernels

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"testing"
)

// adRigFiles are the files that drive the attention digests through adRig.
var adRigFiles = []string{"attn_digest_rig_test.go", "attn_digest_test.go"}

// TestAttnDigestRig_uploadsSynchronize holds the digest rig to one upload path: every host-to-device copy in its files is made
// inside adUpload, and adUpload synchronizes the context after the copy. The rig launches on a non-blocking stream, which a
// pageable copy does not order against, so a raw copy lets a KV kernel read its block table before it lands and fault out of
// bounds when another process loads the card (docs/measurements/cuda-kernels-red-2026-10-30/README.md). It reads source, so it
// needs no device and runs untagged.
func TestAttnDigestRig_uploadsSynchronize(t *testing.T) {
	fset := token.NewFileSet()
	sawUpload := false
	for _, name := range adRigFiles {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			copyAt, syncAfter := token.NoPos, false
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				switch sel.Sel.Name {
				case "CopyHtoD", "CopyFrom", "CopyHtoDAsync":
					if fd.Name.Name != "adUpload" {
						t.Errorf("%s: %s copies to the device outside adUpload; route it through adUpload so the copy lands before a launch reads it",
							fset.Position(call.Pos()), fd.Name.Name)
					}
					copyAt = call.Pos()
				case "Synchronize":
					if x, ok := sel.X.(*ast.SelectorExpr); ok && x.Sel.Name == "cx" && copyAt.IsValid() && call.Pos() > copyAt {
						syncAfter = true
					}
				}
				return true
			})
			if fd.Name.Name == "adUpload" {
				sawUpload = true
				if !copyAt.IsValid() || !syncAfter {
					t.Errorf("adUpload must copy and then synchronize the context (cx.Synchronize) after the copy")
				}
			}
		}
	}
	if !sawUpload {
		t.Errorf("no adUpload in %v: the rig's uploads have no single synchronized path", adRigFiles)
	}
}

// TestKernels_uploadsSynchronize holds the rest of the package to copyHtoD, as TestAttnDigestRig_uploadsSynchronize holds the rig to adUpload: a test that
// copies to the device any other way can launch on a buffer that has not landed. It reads source, so it needs no device and runs untagged.
func TestKernels_uploadsSynchronize(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil || len(files) == 0 {
		t.Fatalf("no test files: %v", err)
	}
	fset := token.NewFileSet()
	sawHelper, sawCaller := false, false
	for _, name := range files {
		if slices.Contains(adRigFiles, name) {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			inHelper := fd.Name.Name == "copyHtoD"
			sawHelper = sawHelper || inHelper
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch fn := call.Fun.(type) {
				case *ast.Ident:
					sawCaller = sawCaller || fn.Name == "copyHtoD"
				case *ast.IndexExpr:
					if sel, ok := fn.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "Alloc" && fd.Name.Name != "allocBuffer" {
						t.Errorf("%s: %s allocates with gc.Alloc; use allocBuffer or mustAlloc, or copyHtoD cannot find the buffer's context", fset.Position(call.Pos()), fd.Name.Name)
					}
				case *ast.SelectorExpr:
					switch fn.Sel.Name {
					case "CopyHtoD", "CopyFrom":
						if !inHelper {
							t.Errorf("%s: %s copies to the device outside copyHtoD; route it through copyHtoD so the copy lands before a launch", fset.Position(call.Pos()), fd.Name.Name)
						}
					}
				}
				return true
			})
		}
	}
	if !sawHelper || !sawCaller {
		t.Errorf("copyHtoD defined %v, called %v: the scan read nothing", sawHelper, sawCaller)
	}
}
