package decoder

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testassets"
)

// assetPath is what decoder's heavy gates call to resolve an asset through testdata/assets.json
// (internal/testassets). It SKIPS when the asset is absent, with the reason the predicate produced, and
// fails when the registry itself cannot be read: the file is tracked, so that is a harness bug.
func assetPath(tb testing.TB, env string) string {
	tb.Helper()
	p, err := testassets.Lookup(env)
	if errors.Is(err, testassets.ErrRegistryUnreadable) {
		tb.Fatalf("asset %s: %v", env, err)
	}
	if err != nil {
		tb.Skipf("asset %s: %v", env, err)
	}
	return p
}

// goTestFiles is every _test.go file in the checkout outside testdata, keyed by its path from decoder/.
func goTestFiles(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.Walk("..", func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi.IsDir() || !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		if strings.Contains(p, "/testdata/") {
			return nil
		}
		b, rerr := os.ReadFile(p)
		if rerr == nil {
			out[p] = string(b)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	return out
}
