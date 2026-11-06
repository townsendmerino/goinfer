//go:build goinfer_testhooks

package decoder

import (
	"errors"
	"testing"
)

// AssetPathForTest resolves a heavy test asset through testdata/assets.json — the same predicate
// decoder's own gates and the sweep preflight use — and skips with the reason when it is absent. A registry that cannot be read
// or parsed fails the test: the file is tracked, so that is a harness bug.
//
// It exists so backend packages (cuda, metal, gpu) resolve assets the one way rather than each
// hand-rolling env + join + stat. The predicate itself lives in assets.go; only this four-line
// skip wrapper is per-consumer, because assets.go must not import testing.
func AssetPathForTest(tb testing.TB, env string) string {
	tb.Helper()
	p, err := lookupAsset(env)
	if errors.Is(err, errRegistryUnreadable) {
		tb.Fatalf("asset %s: %v", env, err)
	}
	if err != nil {
		tb.Skipf("asset %s: %v", env, err)
	}
	return p
}
