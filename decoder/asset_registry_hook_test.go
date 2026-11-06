//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// recTB records the first Fatalf or Skipf and stops the caller, as testing.TB does.
type recTB struct {
	testing.TB
	fatal, skip string
}

type recStop struct{}

func (r *recTB) Helper() {}
func (r *recTB) Fatalf(f string, a ...any) {
	r.fatal = fmt.Sprintf(f, a...)
	panic(recStop{})
}
func (r *recTB) Skipf(f string, a ...any) {
	r.skip = fmt.Sprintf(f, a...)
	panic(recStop{})
}

func callAssetPathForTest(env string) (r *recTB) {
	r = &recTB{}
	defer func() {
		if v := recover(); v != nil {
			if _, ok := v.(recStop); !ok {
				panic(v)
			}
		}
	}()
	AssetPathForTest(r, env)
	return r
}

// TestAssetPathForTest_unreadableRegistryFailsAndMissingWeightsSkip — testdata/assets.json is tracked, so a registry that cannot be
// read or parsed is a broken harness and fails the test; a registered asset whose weights are absent still skips.
func TestAssetPathForTest_unreadableRegistryFailsAndMissingWeightsSkip(t *testing.T) {
	t.Run("no registry above the directory", func(t *testing.T) {
		t.Chdir(t.TempDir())
		if r := callAssetPathForTest("GOINFER_DFLASH_F32"); r.fatal == "" {
			t.Errorf("an unreadable registry did not fail the test (skip=%q)", r.skip)
		}
	})
	t.Run("registry does not parse", func(t *testing.T) {
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "testdata"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "testdata/assets.json"), []byte("{"), 0o644); err != nil {
			t.Fatal(err)
		}
		sub := filepath.Join(root, "a", "b")
		if err := os.MkdirAll(sub, 0o755); err != nil {
			t.Fatal(err)
		}
		t.Chdir(sub)
		if r := callAssetPathForTest("GOINFER_DFLASH_F32"); r.fatal == "" {
			t.Errorf("an unparsable registry did not fail the test (skip=%q)", r.skip)
		}
	})
	t.Run("registered asset, weights absent", func(t *testing.T) {
		t.Setenv("GOINFER_MODELS", t.TempDir())
		t.Setenv("GOINFER_DFLASH_F32", "")
		r := callAssetPathForTest("GOINFER_DFLASH_F32")
		if r.skip == "" || r.fatal != "" {
			t.Errorf("absent weights: skip=%q fatal=%q, want a skip", r.skip, r.fatal)
		}
	})
}
