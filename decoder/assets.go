package decoder

// The heavy-asset resolver: testdata/assets.json and the ONE predicate that decides whether an
// asset is present. Regular (untagged) file, not a _test.go, so backend packages can reach it
// through AssetPathForTest — before that each hand-rolled os.Getenv + filepath.Join + os.Stat,
// which is the drift TestAssetRegistry_noDirectReads exists to stop.
//
// The `testing`-dependent skip wrapper is deliberately NOT here: this file must not import
// testing. Callers wrap it (assetPath in the gates, AssetPathForTest for other packages).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

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
		min := max(a.MinBytes, 1)
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
