package decoder

// Parity validation manifest + staleness detector (docs/task-parity-coverage.md
// Item 1). The manifest testdata/parity_manifest.json records, per family, the
// exact set of source files its numerics depend on (named shared sets via `uses`
// plus per-family `own`), a content hash of that set, and the validation metrics.
//
// TestParityManifest_fresh is model-free (no assets) and runs every push:
//
//   - STRUCTURE: every file path in every shared set and every family's `own`
//     list must exist on disk (catches renames).
//   - COVERAGE: the manifest's family keys must equal the capability matrix's
//     family set (adding a family without a manifest row fails CI).
//   - HASH/ENFORCEMENT: for each family, re-hash the SORTED, DEDUPED union of its
//     dependency files (+ aikit_version); for VALIDATED families a mismatch vs.
//     the recorded deps_hash fails ("parity stale") — pending families do not.
//
// Run `go test ./decoder -run ParityManifest -update` to fill/refresh deps_hash;
// the plain run is the staleness gate. The -update flag is shared with the
// capability matrix test (var updateMatrix in capability_matrix_test.go).

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// parityManifest mirrors testdata/parity_manifest.json. familyParity uses
// json.RawMessage for fields the test must preserve verbatim on -update
// (metrics/status/dates/etc.) while still letting us read+rewrite deps_hash and
// inspect uses/own/validated_at. Field order matches the on-disk schema so that
// re-marshaling produces a stable, zero-diff layout.
type parityManifest struct {
	AikitVersion string                  `json:"aikit_version"`
	SharedSets   map[string][]string     `json:"shared_sets"`
	Families     map[string]familyParity `json:"families"`
}

type familyParity struct {
	Uses        []string        `json:"uses"`
	Own         []string        `json:"own"`
	DepsHash    string          `json:"deps_hash"`
	Status      json.RawMessage `json:"status"`
	ValidatedAt json.RawMessage `json:"validated_at"`
	Date        json.RawMessage `json:"date"`
	Reference   json.RawMessage `json:"reference"`
	Machine     json.RawMessage `json:"machine"`
	Method      json.RawMessage `json:"method"`
	Metrics     json.RawMessage `json:"metrics"`
}

const parityManifestPath = "../testdata/parity_manifest.json"

// repoPath resolves a repo-root-relative path (as stored in the manifest, e.g.
// "decoder/model.go") to a path usable from the decoder package directory.
func repoPath(p string) string { return filepath.Join("..", p) }

// familyDepFiles returns the sorted, deduped union of all files reachable from a
// family's `uses` shared sets plus its `own` list.
func familyDepFiles(m *parityManifest, fam familyParity) []string {
	seen := map[string]bool{}
	for _, set := range fam.Uses {
		for _, f := range m.SharedSets[set] {
			seen[f] = true
		}
	}
	for _, f := range fam.Own {
		seen[f] = true
	}
	out := make([]string, 0, len(seen))
	for f := range seen {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// freshDepsHash computes the deterministic content hash over a family's
// dependency set: for each path (sorted) write path + NUL + file bytes + NUL,
// then mix in the aikit_version. Returns "sha256:" + hex.
func freshDepsHash(m *parityManifest, fam familyParity) (string, error) {
	h := sha256.New()
	for _, p := range familyDepFiles(m, fam) {
		b, err := os.ReadFile(repoPath(p))
		if err != nil {
			return "", err
		}
		h.Write([]byte(p))
		h.Write([]byte{0})
		h.Write(b)
		h.Write([]byte{0})
	}
	h.Write([]byte("aikit_version=" + m.AikitVersion))
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func TestParityManifest_fresh(t *testing.T) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatalf("read %s: %v", parityManifestPath, err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal %s: %v", parityManifestPath, err)
	}

	// STRUCTURE: every referenced source path must exist on disk.
	var missing []string
	checkExists := func(p string) {
		if _, err := os.Stat(repoPath(p)); err != nil {
			missing = append(missing, p)
		}
	}
	for _, files := range m.SharedSets {
		for _, f := range files {
			checkExists(f)
		}
	}
	for _, fam := range m.Families {
		for _, f := range fam.Own {
			checkExists(f)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Fatalf("parity manifest references missing files (rename?): %v", missing)
	}

	// COVERAGE: manifest family keys must equal the capability matrix family set.
	rows, err := buildMatrix(t)
	if err != nil {
		t.Fatalf("build matrix: %v", err)
	}
	matrixFams := map[string]bool{}
	for _, r := range rows {
		matrixFams[r.Name] = true
	}
	var inManifestNotMatrix, inMatrixNotManifest []string
	for fam := range m.Families {
		if !matrixFams[fam] {
			inManifestNotMatrix = append(inManifestNotMatrix, fam)
		}
	}
	for fam := range matrixFams {
		if _, ok := m.Families[fam]; !ok {
			inMatrixNotManifest = append(inMatrixNotManifest, fam)
		}
	}
	if len(inManifestNotMatrix) > 0 || len(inMatrixNotManifest) > 0 {
		sort.Strings(inManifestNotMatrix)
		sort.Strings(inMatrixNotManifest)
		t.Fatalf("parity manifest / capability matrix family mismatch:\n  in manifest but not matrix: %v\n  in matrix but not manifest: %v (add a parity_manifest.json row)",
			inManifestNotMatrix, inMatrixNotManifest)
	}

	// HASH + ENFORCEMENT (and -update rewrite).
	famKeys := make([]string, 0, len(m.Families))
	for fam := range m.Families {
		famKeys = append(famKeys, fam)
	}
	sort.Strings(famKeys)

	for _, fam := range famKeys {
		f := m.Families[fam]
		fresh, err := freshDepsHash(&m, f)
		if err != nil {
			t.Fatalf("hash deps for %s: %v", fam, err)
		}
		if *updateMatrix {
			f.DepsHash = fresh
			m.Families[fam] = f
			continue
		}
		// Only enforce for validated families (non-null validated_at).
		if string(f.ValidatedAt) == "null" || len(f.ValidatedAt) == 0 {
			continue
		}
		if fresh != f.DepsHash {
			var validatedAt string
			_ = json.Unmarshal(f.ValidatedAt, &validatedAt)
			t.Fatalf("parity stale for %s: numerics changed since %s — re-run T3 (scripts/parity_sweep.sh) and update parity_manifest.json", fam, validatedAt)
		}
	}

	if *updateMatrix {
		out, err := json.MarshalIndent(&m, "", "  ")
		if err != nil {
			t.Fatalf("marshal manifest: %v", err)
		}
		out = append(out, '\n')
		if err := os.WriteFile(parityManifestPath, out, 0o644); err != nil {
			t.Fatalf("write %s: %v", parityManifestPath, err)
		}
		t.Logf("wrote %s", parityManifestPath)
	}
}
