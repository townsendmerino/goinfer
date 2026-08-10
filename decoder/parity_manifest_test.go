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
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// mergeRowsPath, when set, makes TestParityManifest_merge fold the collected
// PARITY_ROW lines (emitted by the real-checkpoint gates under GOINFER_MANIFEST_EMIT)
// from that file into the manifest — the machine-written alternative to hand-editing
// validation fields. Driven by scripts/parity_sweep.sh EMIT_MANIFEST=1.
var mergeRowsPath = flag.String("merge-rows", "", "merge collected PARITY_ROW lines from this file into the parity manifest")

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
	Status      string          `json:"status"`
	ValidatedAt json.RawMessage `json:"validated_at"`
	Date        json.RawMessage `json:"date"`
	Reference   json.RawMessage `json:"reference"`
	Machine     json.RawMessage `json:"machine"`
	Method      string          `json:"method"`
	Metrics     json.RawMessage `json:"metrics"`
}

const parityManifestPath = "../testdata/parity_manifest.json"

// repoPath resolves a repo-root-relative path (as stored in the manifest, e.g.
// "decoder/model.go") to a path usable from the decoder package directory.
func repoPath(p string) string { return filepath.Join("..", p) }

// rawString quotes s as a JSON string literal for the RawMessage manifest fields.
func rawString(s string) json.RawMessage { b, _ := json.Marshal(s); return b }

// shortHEAD returns the short HEAD commit SHA stamped into newly-validated rows.
func shortHEAD(t *testing.T) string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		t.Fatalf("git rev-parse HEAD: %v", err)
	}
	return strings.TrimSpace(string(out))
}

// TestParityManifest_merge folds collected PARITY_ROW lines into the manifest (Item
// 1d): for each row it sets that family's status=validated, fills method/reference/
// metrics from the gate's measurement, and stamps validated_at (short HEAD SHA), date
// (today), and machine (GOINFER_MANIFEST_MACHINE, default linux-62gb) — the same stamp
// across one run. Other families and all uses/own are preserved; every deps_hash is
// then recomputed (the freshness hashing) and the manifest rewritten deterministically
// (same path as -update). An unknown family in a row is a hard error (no silent drop).
// Skips unless -merge-rows is set, so plain `go test` never touches the manifest.
func TestParityManifest_merge(t *testing.T) {
	if *mergeRowsPath == "" {
		t.Skip("set -merge-rows <file> to merge collected PARITY_ROW lines")
	}
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatalf("read %s: %v", parityManifestPath, err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal %s: %v", parityManifestPath, err)
	}
	rowsRaw, err := os.ReadFile(*mergeRowsPath)
	if err != nil {
		t.Fatalf("read rows %s: %v", *mergeRowsPath, err)
	}

	sha := shortHEAD(t)
	date := time.Now().Format("2006-01-02")
	machine := os.Getenv("GOINFER_MANIFEST_MACHINE")
	if machine == "" {
		machine = "linux-62gb"
	}

	applied := []string{}
	for line := range strings.SplitSeq(string(rowsRaw), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "PARITY_ROW ") {
			continue
		}
		var row struct {
			Family    string          `json:"family"`
			Method    string          `json:"method"`
			Reference string          `json:"reference"`
			Metrics   json.RawMessage `json:"metrics"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "PARITY_ROW ")), &row); err != nil {
			t.Fatalf("bad PARITY_ROW line %q: %v", line, err)
		}
		f, ok := m.Families[row.Family]
		if !ok {
			t.Fatalf("PARITY_ROW for unknown family %q (not in %s)", row.Family, parityManifestPath)
		}
		f.Status = "validated"
		f.Method = row.Method
		f.Reference = rawString(row.Reference)
		f.Metrics = row.Metrics
		f.ValidatedAt = rawString(sha)
		f.Date = rawString(date)
		f.Machine = rawString(machine)
		m.Families[row.Family] = f
		applied = append(applied, row.Family)
	}
	if len(applied) == 0 {
		t.Fatalf("no PARITY_ROW lines in %s", *mergeRowsPath)
	}

	// Recompute every deps_hash (same as -update) so the freshness gate stays green.
	for fam := range m.Families {
		f := m.Families[fam]
		fresh, err := freshDepsHash(&m, f)
		if err != nil {
			t.Fatalf("hash deps for %s: %v", fam, err)
		}
		f.DepsHash = fresh
		m.Families[fam] = f
	}
	out, err := json.MarshalIndent(&m, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	out = append(out, '\n')
	if err := os.WriteFile(parityManifestPath, out, 0o644); err != nil {
		t.Fatalf("write %s: %v", parityManifestPath, err)
	}
	sort.Strings(applied)
	t.Logf("merged %d row(s) into %s at %s (%s): %v", len(applied), parityManifestPath, sha, date, applied)
}

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

	var stale []string
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
			scope := strings.Join(f.Uses, "+")
			if len(f.Own) > 0 {
				scope += "+own(" + strings.Join(f.Own, ",") + ")"
			}
			stale = append(stale, fmt.Sprintf("  %-16s stale since %s  [covers: %s]", fam, validatedAt, scope))
		}
	}
	// Collect EVERY mismatch and fail once with the full blast radius. A staleness is
	// systemic when a shared file changes (every family that `uses` that set restales at
	// once); a per-family t.Fatalf in sorted order would surface a 9-family staleness as
	// "cohere is RED" and hide its own scope. The scope tag (which shared sets / own files
	// each stale hash covers) makes the cause — the one changed shared file — a glance away.
	if len(stale) > 0 {
		t.Fatalf("parity stale for %d validated family(ies) — numerics changed since the noted commit:\n%s\n"+
			"Fix: re-run T3 (scripts/parity_sweep.sh) then -update; or, for a provably non-numeric core edit "+
			"(a guarded diagnostic seam, comment, rename), scripts/refresh_parity_hashes.sh.",
			len(stale), strings.Join(stale, "\n"))
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

// t3Methods is parity-coverage-policy.md's authoritative list of T3 `method` values. A row is only
// allowed to claim `status: "validated"` — the status the capability matrix and the README's
// "supported" count read as a real gate — if its method is one of these.
//
//	full-forward-oracle   cosine/argmax vs a full bf16 forward of a released checkpoint
//	real-model-oracle     int8-resident vs a bf16 reference (when both won't co-reside)
//	weightDiff            GGUF-vs-safetensors to Q8_0 tolerance, when no oracle is feasible
//	shared-path (via X)   an alias family riding X's already-validated forward AND deps_hash
var t3Methods = map[string]bool{
	"full-forward-oracle": true,
	"real-model-oracle":   true,
	"weightDiff":          true,
}

func isT3Method(m string) bool {
	return t3Methods[m] || strings.HasPrefix(m, "shared-path (via ")
}

// TestParityManifest_methodTier is the claim-discipline gate: it makes "validated" MEAN T3.
//
// WHY IT EXISTS. `parity-coverage-policy.md` has always defined which methods clear T3, but nothing
// enforced it — `Method` was parsed as a `json.RawMessage` and never compared to the list. That let
// five rows sit at `status: validated` with `method: tiny-golden` / `tiny-golden+coherent`: a T1
// artifact (cosine vs the family's own seeded tiny golden — no released checkpoint involved)
// recorded in a T3 slot, and therefore counted as "supported". The staleness gate could not catch it
// because it keys on `deps_hash` freshness, which says nothing about how the row was validated.
//
// THE HONEST ALTERNATIVE, so a weak row does not have to lie. `status: "experimental"` records a
// family whose gate is real but sub-T3. Such a row keeps its method and metrics, renders distinctly
// in the capability matrix, and is EXCLUDED from the supported count. Downgrading is not a
// regression — it is the matrix finally saying what the row always was.
func TestParityManifest_methodTier(t *testing.T) {
	m, err := loadParityManifest()
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	var bad []string
	for _, name := range sortedKeys(m.Families) {
		f := m.Families[name]
		switch f.Status {
		case "validated":
			if !isT3Method(f.Method) {
				bad = append(bad, fmt.Sprintf("  %-18s method=%q — not a T3 method", name, f.Method))
			}
		case "experimental":
			if f.Method == "" {
				bad = append(bad, fmt.Sprintf("  %-18s status=experimental but no method recorded", name))
			}
		case "pending", "":
			// nothing claimed, nothing to check.
		default:
			bad = append(bad, fmt.Sprintf("  %-18s unknown status %q", name, f.Status))
		}
	}
	if len(bad) > 0 {
		t.Errorf("manifest rows claim a tier their method does not support:\n%s\n\n"+
			"A row may claim status:\"validated\" ONLY with a T3 method (%s, or \"shared-path (via X)\").\n"+
			"If the gate is real but sub-T3 (e.g. tiny-golden), record status:\"experimental\" instead —\n"+
			"it keeps the method and metrics, renders as experimental in the capability matrix, and is\n"+
			"excluded from the supported count. See docs/parity-coverage-policy.md.",
			strings.Join(bad, "\n"), strings.Join(sortedKeys(t3Methods), ", "))
	}
}

// sortedKeys is a tiny helper so the gate's output is deterministic.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
