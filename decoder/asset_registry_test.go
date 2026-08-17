package decoder

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// THE GATE SIDE OF THE SHARED ASSET REGISTRY (testdata/assets.json).
//
// Every heavy gate used to resolve its own asset: read an env var, fall back to a path it spelled
// out itself, and decide presence with os.Stat. The sweep's preflight did the same thing again in
// bash with `[ -e ]`. Two implementations of "is this asset present", free to disagree, and they did:
//
//   * a DIRECTORY satisfies `-e`, so preflight reported .gguf assets RESOLVED while naming the
//     directory above them. Four gates were costed by that.
//   * GOINFER_QWEN35_GOLDEN's real requirement is a readable manifest.json INSIDE the directory,
//     which `-e` on the directory cannot express -- preflight said present, the gate skipped.
//   * GOINFER_PREQUANT_GGUF had three different fallbacks across four call sites and, at
//     loadInt4Model, none at all -- so one box ran different gates against different files.
//
// Now both sides read testdata/assets.json and apply the predicate it states. The two
// implementations (this file and scripts/asset_registry.py) are checked against each other by
// TestAssetRegistry_agreesWithPreflight rather than assumed to match.

// assetPath is what gates call. It SKIPS when the asset is absent, exactly as the hand-written
// resolutions it replaces did, but with the reason the predicate produced rather than a bare
// "no model at <path>".
func assetPath(tb testing.TB, env string) string {
	tb.Helper()
	p, err := lookupAsset(env)
	if err != nil {
		tb.Skipf("asset %s: %v", env, err)
	}
	return p
}

// ---------------------------------------------------------------------------------------------
// The registry's own gates.
// ---------------------------------------------------------------------------------------------

func goTestFiles(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.Walk(repoRoot, func(p string, fi os.FileInfo, err error) error {
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

// TestAssetRegistry_usedByExists — used_by is not decoration. A gate that is renamed or deleted must
// not leave an entry behind still claiming to cover it, because that entry is what a reader consults
// to decide whether an asset still matters.
func TestAssetRegistry_usedByExists(t *testing.T) {
	assets, err := loadAssetRegistry()
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	files := goTestFiles(t)
	for _, a := range assets {
		for _, g := range a.UsedBy {
			pat := regexp.MustCompile(`func ` + regexp.QuoteMeta(g) + `\(`)
			found := false
			for _, src := range files {
				if pat.MatchString(src) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s used_by names %s, which is not defined in any _test.go — "+
					"a renamed or deleted gate leaving a stale claim of coverage", a.Env, g)
			}
		}
	}
}

// TestAssetRegistry_noDirectReads — the drift gate. A registered variable read with os.Getenv
// somewhere other than this file means a second resolution has grown back, which is precisely the
// condition the registry removed. Enforcement is scoped exactly to what is REGISTERED; everything
// else is reported by `asset_registry.py census` and not policed here.
func TestAssetRegistry_noDirectReads(t *testing.T) {
	assets, err := loadAssetRegistry()
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	files := goTestFiles(t)
	for _, a := range assets {
		pat := regexp.MustCompile(`os\.Getenv\("` + regexp.QuoteMeta(a.Env) + `"\)`)
		for p, src := range files {
			if strings.HasSuffix(p, "asset_registry_test.go") {
				continue
			}
			if pat.MatchString(src) {
				t.Errorf("%s reads %s directly — use assetPath(t, %q) so the gate and the sweep "+
					"preflight apply the same predicate", p, a.Env, a.Env)
			}
		}
	}
}

// TestAssetRegistry_agreesWithPreflight — the two implementations of the predicate, compared instead
// of trusted. Bash cannot read JSON and Go cannot be called from the preflight, so the predicate
// necessarily exists twice; what makes that acceptable is this test, not good intentions.
func TestAssetRegistry_agreesWithPreflight(t *testing.T) {
	// No cmd.Dir: the script path is already relative to the test's working directory, and setting
	// Dir would apply repoRoot twice. The script locates the registry from its own __file__.
	cmd := exec.Command("python3", filepath.Join(repoRoot, "scripts/asset_registry.py"), "verdicts")
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("cannot run scripts/asset_registry.py (%v) — the two predicates are UNCOMPARED on "+
			"this box, not proven equal", err)
	}
	var py map[string]struct {
		Path   string `json:"path"`
		OK     bool   `json:"ok"`
		Source string `json:"source"`
	}
	if err := json.Unmarshal(out, &py); err != nil {
		t.Fatalf("parse verdicts: %v", err)
	}
	assets, err := loadAssetRegistry()
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	if len(py) != len(assets) {
		t.Errorf("python reported %d assets, Go sees %d", len(py), len(assets))
	}
	for _, a := range assets {
		v, ok := py[a.Env]
		if !ok {
			t.Errorf("%s: python produced no verdict", a.Env)
			continue
		}
		p, gerr := lookupAsset(a.Env)
		if (gerr == nil) != v.OK {
			t.Errorf("%s: PREDICATES DISAGREE — python ok=%v (%s), Go ok=%v (%v)",
				a.Env, v.OK, v.Source, gerr == nil, gerr)
			continue
		}
		// Same verdict is not enough: the two sides must pick the SAME FILE, or a gate and the
		// preflight can both report success about different assets.
		if v.OK {
			gabs, _ := filepath.Abs(p)
			pabs, _ := filepath.Abs(v.Path)
			if gabs != pabs {
				t.Errorf("%s: both resolve, but to DIFFERENT paths — python %s, Go %s",
					a.Env, pabs, gabs)
			}
		}
	}
}
