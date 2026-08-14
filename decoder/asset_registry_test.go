package decoder

import (
	"encoding/json"
	"fmt"
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

type assetSpec struct {
	Env        string   `json:"env"`
	Kind       string   `json:"kind"`
	MinBytes   int64    `json:"min_bytes"`
	Members    []string `json:"members"`
	MembersAny []string `json:"members_any"`
	Candidates []string `json:"candidates"`
	UsedBy     []string `json:"used_by"`
	Note       string   `json:"note"`
}

// repoRoot is "..": every package that uses this helper is one level below the checkout root, which
// is also what the existing "../testdata/..." golden paths throughout these tests assume.
const repoRoot = ".."

func loadAssetRegistry() ([]assetSpec, error) {
	raw, err := os.ReadFile(filepath.Join(repoRoot, "testdata/assets.json"))
	if err != nil {
		return nil, err
	}
	var doc struct {
		Assets []assetSpec `json:"assets"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	return doc.Assets, nil
}

func modelsRoot() string {
	if m := os.Getenv("GOINFER_MODELS"); m != "" {
		return m
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(home, "models")
}

func expandCandidate(c string) string {
	switch {
	case strings.HasPrefix(c, "$REPO/"):
		return filepath.Join(repoRoot, strings.TrimPrefix(c, "$REPO/"))
	case strings.HasPrefix(c, "$MODELS/"):
		return filepath.Join(modelsRoot(), strings.TrimPrefix(c, "$MODELS/"))
	}
	return c
}

// satisfiesAsset is THE PREDICATE, and it must stay behaviourally identical to satisfies() in
// scripts/asset_registry.py. The reason string matters as much as the boolean: "not found" printed
// for a path that exists is the single most misleading thing this can say.
func satisfiesAsset(a assetSpec, p string) (bool, string) {
	fi, err := os.Stat(p) // follows symlinks, as the python side's exists()/is_file() do
	if err != nil {
		return false, "does not exist"
	}
	switch a.Kind {
	case "file":
		if fi.IsDir() {
			return false, "is a DIRECTORY, but this asset is a file"
		}
		if !fi.Mode().IsRegular() {
			return false, "exists but is not a regular file"
		}
		min := a.MinBytes
		if min < 1 {
			min = 1
		}
		if fi.Size() < min {
			return false, fmt.Sprintf("only %d bytes (min_bytes %d) — a stub or a truncated copy",
				fi.Size(), min)
		}
		return true, ""
	case "dir":
		if !fi.IsDir() {
			return false, "is not a directory"
		}
		var missing []string
		for _, m := range a.Members {
			if _, err := os.Stat(filepath.Join(p, m)); err != nil {
				missing = append(missing, m)
			}
		}
		if len(missing) > 0 {
			return false, "directory exists but is missing " + strings.Join(missing, ", ")
		}
		if len(a.MembersAny) > 0 {
			ok := false
			for _, m := range a.MembersAny {
				if _, err := os.Stat(filepath.Join(p, m)); err == nil {
					ok = true
					break
				}
			}
			if !ok {
				return false, "directory exists but has none of " + strings.Join(a.MembersAny, ", ")
			}
		}
		return true, ""
	}
	return false, "unknown kind " + a.Kind + " in the registry"
}

// lookupAsset resolves one registered asset. An explicit env value ALWAYS wins and is checked by the
// same predicate: silently falling back to a candidate when the operator named a path would run the
// gate against a different file than the one they asked for, which is worse than skipping.
func lookupAsset(env string) (string, error) {
	assets, err := loadAssetRegistry()
	if err != nil {
		return "", fmt.Errorf("asset registry unreadable: %w", err)
	}
	for _, a := range assets {
		if a.Env != env {
			continue
		}
		if cur := os.Getenv(env); cur != "" {
			ok, why := satisfiesAsset(a, cur)
			if ok {
				return cur, nil
			}
			return "", fmt.Errorf("%s=%s %s", env, cur, why)
		}
		var tried []string
		for _, c := range a.Candidates {
			p := expandCandidate(c)
			ok, why := satisfiesAsset(a, p)
			if ok {
				return p, nil
			}
			tried = append(tried, fmt.Sprintf("%s (%s)", p, why))
		}
		return "", fmt.Errorf("%s unset and no candidate usable: %s", env, strings.Join(tried, "; "))
	}
	return "", fmt.Errorf("%s is not in testdata/assets.json", env)
}

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
