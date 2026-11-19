package testgate

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"go/parser"
	"go/token"
)

// recorder takes a gate's skip without stopping the calling test, so the test can read the message.
type recorder struct {
	testing.TB
	skipped bool
	msg     string
}

func (r *recorder) Skip(args ...any) { r.skipped, r.msg = true, fmt.Sprint(args...) }
func (r *recorder) Skipf(format string, args ...any) {
	r.skipped, r.msg = true, fmt.Sprintf(format, args...)
}

func TestRequireHeavy_skipsUnsetWithTheToken(t *testing.T) {
	t.Setenv("GOINFER_HEAVY_TESTS", "")
	r := &recorder{TB: t}
	RequireHeavy(r)
	if !r.skipped || !strings.HasPrefix(r.msg, SkipHeavy) || !strings.Contains(r.msg, "GOINFER_HEAVY_TESTS=1") {
		t.Fatalf("unset: skipped %v, message %q; want a skip beginning %q", r.skipped, r.msg, SkipHeavy)
	}
	t.Setenv("GOINFER_HEAVY_TESTS", "1")
	r = &recorder{TB: t}
	RequireHeavy(r)
	if r.skipped {
		t.Fatalf("GOINFER_HEAVY_TESTS=1: skipped with %q", r.msg)
	}
}

func TestRequireDevice_skipsOnAProbeErrorWithTheToken(t *testing.T) {
	r := &recorder{TB: t}
	RequireDevice(r, "cuda", errors.New("cuInit: CUDA_ERROR_NO_DEVICE"))
	if !r.skipped || r.msg != SkipDevice+" no cuda device: cuInit: CUDA_ERROR_NO_DEVICE" {
		t.Fatalf("probe error: skipped %v, message %q", r.skipped, r.msg)
	}
	r = &recorder{TB: t}
	RequireDevice(r, "cuda", nil)
	if r.skipped {
		t.Fatalf("nil probe error: skipped with %q", r.msg)
	}
}

func TestModelPath_honoursTheModelsDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("GOINFER_MODELS_DIR", dir)
	if got, want := ModelPath("m.gguf"), filepath.Join(dir, "m.gguf"); got != want {
		t.Fatalf("ModelPath = %q, want %q", got, want)
	}
	t.Setenv("GOINFER_MODELS_DIR", "")
	h, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home directory: %v", err)
	}
	if got, want := ModelPath("m.gguf"), filepath.Join(h, "models", "m.gguf"); got != want {
		t.Fatalf("ModelPath with GOINFER_MODELS_DIR unset = %q, want %q", got, want)
	}
}

func TestRequireModel_skipsAnAbsentAssetWithTheToken(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("GOINFER_MODELS_DIR", dir)
	r := &recorder{TB: t}
	RequireModel(r, "absent.gguf")
	if !r.skipped || !strings.HasPrefix(r.msg, SkipAsset+" absent.gguf") {
		t.Fatalf("absent: skipped %v, message %q; want a skip beginning %q", r.skipped, r.msg, SkipAsset)
	}
	if err := os.WriteFile(filepath.Join(dir, "present.gguf"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	r = &recorder{TB: t}
	if p := RequireModel(r, "present.gguf"); r.skipped || p != filepath.Join(dir, "present.gguf") {
		t.Fatalf("present: skipped %v (%q), path %q", r.skipped, r.msg, p)
	}
}

// TestNoProductionImporter: only _test.go files import this package, in every module of the tree. That is what makes its
// environment reads test-only, which TestEnvVars_docAndCodeAgree's exemption of this directory relies on.
func TestNoProductionImporter(t *testing.T) {
	const self = "github.com/townsendmerino/goinfer/internal/testgate"
	root := filepath.Join("..", "..")
	scanned, test := 0, 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if n := d.Name(); n == ".git" || n == "vendor" || n == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
		if err != nil {
			return nil // not Go this toolchain parses (a fixture); it imports nothing that builds
		}
		for _, im := range f.Imports {
			if path, _ := strconv.Unquote(im.Path.Value); path == self {
				if strings.HasSuffix(p, "_test.go") {
					test++
				} else {
					t.Errorf("%s imports %s: a non-test file; its environment reads would become production reads", p, self)
				}
			}
		}
		scanned++
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if scanned < 1000 || test == 0 {
		t.Fatalf("scanned %d .go files, %d _test.go importers: the walk did not see the tree", scanned, test)
	}
}
