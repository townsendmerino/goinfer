package decoder

// Parity validation manifest and staleness detector (docs/completed/task-parity-coverage.md). The manifest
// testdata/parity_manifest.json records, per family, the source files its numerics depend on (shared sets via `uses`
// plus per-family `own`), a content hash of that set, and the validation metrics.
//
// TestParityManifest_fresh is model-free (no assets) and runs every push:
//
//   - STRUCTURE: every path in every shared set and every family's `own` list must exist on disk (catches renames).
//   - COVERAGE: the manifest's family keys must equal the capability matrix's family set (a family added without a
//     manifest row fails CI).
//   - HASH/ENFORCEMENT: for each family, re-hash the SORTED, DEDUPED union of its dependency files plus the root
//     go.mod's aikit pin; for VALIDATED families a mismatch against the recorded deps_hash fails ("parity stale"),
//     pending families do not.
//
// Run `go test ./decoder -run ParityManifest -update` to fill/refresh deps_hash; the plain run is the staleness gate.
// The -update flag is shared with the capability matrix test (var updateMatrix in capability_matrix_test.go).

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// mergeRowsPath, when set, makes TestParityManifest_merge fold the collected
// PARITY_ROW lines (emitted by the real-checkpoint gates under GOINFER_MANIFEST_EMIT)
// from that file into the manifest — the machine-written alternative to hand-editing
// validation fields. Driven by EMIT_MANIFEST=1 go run ./cmd/gate parity.
var mergeRowsPath = flag.String("merge-rows", "", "merge collected PARITY_ROW lines from this file into the parity manifest")

// parityManifest mirrors testdata/parity_manifest.json. familyParity uses json.RawMessage for fields the test must
// preserve verbatim on -update (metrics, status, dates) while still reading and rewriting deps_hash and reading
// uses/own/validated_at. Field order matches the on-disk schema so re-marshaling gives a zero-diff layout.
//
// There is deliberately no AikitVersion field: freshDepsHash reads the root go.mod's aikit require at hash time
// (rootAikitVersion), so no stored value can drift from the pin. History: docs/code-notes/decoder.md#parityManifest.
type parityManifest struct {
	SharedSets map[string][]string     `json:"shared_sets"`
	Families   map[string]familyParity `json:"families"`
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
	Method      json.RawMessage `json:"method"`
	Metrics     json.RawMessage `json:"metrics"`
}

const parityManifestPath = "../testdata/parity_manifest.json"

// writeManifest serialises the manifest back to disk faithfully: no HTML escaping (a ">" inside a `reference` string
// stays ">") and a JSON `null` Method stays null (Method is a RawMessage). scripts/refresh_parity_hashes.sh aborts when
// the update changes more than deps_hash, so a lossy round trip reads as a real edit; fix the writer, do not loosen that
// guard. History: docs/code-notes/decoder.md#writeManifest.
func writeManifest(m *parityManifest) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		return err
	}
	return os.WriteFile(parityManifestPath, buf.Bytes(), 0o644)
}

// methodString reads Method's JSON literal as a string; a null or absent value reads as "".
func methodString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

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
// (today), and machine (GOINFER_MANIFEST_MACHINE, default GOOS-GOARCH) — the same stamp
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
		// Not a hardcoded box name: a fixed label mislabels every row re-validated from another machine. GOOS-GOARCH cannot be
		// wrong that way; pass GOINFER_MANIFEST_MACHINE for a friendlier name. History: docs/code-notes/decoder.md#TestParityManifest_merge.machine.
		machine = runtime.GOOS + "-" + runtime.GOARCH
	}

	applied, err := applyParityRows(&m, string(rowsRaw), sha, date, machine)
	if err != nil {
		t.Fatal(err)
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
	if err := writeManifest(&m); err != nil {
		t.Fatalf("write manifest: %v", err)
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

// aikitVersionRE matches the aikit require line in a go.mod file — the same pattern
// TestAikitPinsAgree (aikit_pin_test.go) uses per-module; rootAikitVersion below is its
// root-go.mod-only twin.
var aikitVersionRE = regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit (v[0-9][^\s]*)`)

// rootAikitVersion reads the ROOT go.mod's aikit require directly. freshDepsHash mixes in whatever go.mod pins at the
// moment it hashes, so the manifest holds no hand-typed aikit version that could drift.
func rootAikitVersion() (string, error) {
	b, err := os.ReadFile(repoPath("go.mod"))
	if err != nil {
		return "", err
	}
	m := aikitVersionRE.FindStringSubmatch(string(b))
	if m == nil {
		return "", fmt.Errorf("root go.mod does not require github.com/townsendmerino/aikit")
	}
	return m[1], nil
}

// TestRootAikitVersion_readsFromGoMod pins that rootAikitVersion actually reads go.mod rather than returning a stale or
// hardcoded value. It deliberately asserts no specific version string (a second place to bump on every aikit release);
// it re-parses go.mod with a SEPARATE regexp match, not rootAikitVersion's own machinery, and requires exact agreement,
// so a bug in the function (wrong path, wrong pattern) cannot pass by coincidence.
func TestRootAikitVersion_readsFromGoMod(t *testing.T) {
	got, err := rootAikitVersion()
	if err != nil {
		t.Fatalf("rootAikitVersion: %v", err)
	}
	if !strings.HasPrefix(got, "v") {
		t.Errorf("rootAikitVersion() = %q, want a v-prefixed version", got)
	}

	b, err := os.ReadFile(repoPath("go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}
	want := regexp.MustCompile(`(?m)^\tgithub\.com/townsendmerino/aikit (v\S+)$`).FindStringSubmatch(string(b))
	if want == nil {
		t.Fatal("go.mod has no tab-indented aikit require line — test setup assumption broke")
	}
	if got != want[1] {
		t.Errorf("rootAikitVersion() = %q, want %q (go.mod's own require line, independently parsed)", got, want[1])
	}
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
	aikitVer, err := rootAikitVersion()
	if err != nil {
		return "", err
	}
	h.Write([]byte("aikit_version=" + aikitVer))
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
	var emptyDeps []string
	// Count what was actually ENFORCED, not just what came back stale. An empty `stale` means
	// either "every validated family's hash matched" or "no family was enforced at all" — the
	// loop `continue`s past every validated_at:null row, so a manifest whose rows were all
	// pending would report zero staleness having checked nothing. Zero has to say which.
	enforced := 0
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
		enforced++
		// A validated family whose uses/own sets name no files hashes only the aikit pin, so no edit to its forward can ever
		// restale it and its green covers nothing. Keyed on status, not validated_at: an experimental family can carry a
		// validated_at without claiming validation, and gets its sets when it is promoted.
		if f.Status == "validated" && len(familyDepFiles(&m, f)) == 0 {
			emptyDeps = append(emptyDeps, fam)
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
	if len(emptyDeps) > 0 {
		t.Fatalf("%d validated family(ies) name no dependency files, so their deps_hash covers no source "+
			"and can never go stale: %s. Give each its uses/own sets (audit-2026-09-10 G-02).",
			len(emptyDeps), strings.Join(emptyDeps, ", "))
	}
	// Collect EVERY mismatch and fail once with the full blast radius. A staleness is
	// systemic when a shared file changes (every family that `uses` that set restales at
	// once); a per-family t.Fatalf in sorted order would surface a 9-family staleness as
	// "cohere is RED" and hide its own scope. The scope tag (which shared sets / own files
	// each stale hash covers) makes the cause — the one changed shared file — a glance away.
	if len(stale) > 0 {
		t.Fatalf("parity stale for %d validated family(ies) — numerics changed since the noted commit:\n%s\n"+
			"Fix: re-run T3 (go run ./cmd/gate parity) then -update; or, for a provably non-numeric core edit "+
			"(a guarded diagnostic seam, comment, rename), scripts/refresh_parity_hashes.sh.",
			len(stale), strings.Join(stale, "\n"))
	}
	if !*updateMatrix {
		if enforced == 0 {
			t.Fatalf("staleness gate enforced ZERO families of %d — every row is validated_at:null, so this "+
				"gate passed having checked nothing. That is not a green manifest.", len(famKeys))
		}
		t.Logf("staleness: %d/%d families enforced (the rest are validated_at:null and carry no hash to check)",
			enforced, len(famKeys))
	}

	if *updateMatrix {
		if err := writeManifest(&m); err != nil {
			t.Fatalf("write manifest: %v", err)
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
	if inner, _, ok := identityInherited(m); ok {
		return t3Methods[inner] || strings.HasPrefix(inner, "shared-path (via ")
	}
	return t3Methods[m] || strings.HasPrefix(m, "shared-path (via ")
}

// identityInherited parses "identity-inherited (<method> @ <rev>)" (TE6(b), docs/tasks/task-test-efficiency-2026-09.md;
// parity-coverage-policy.md): a family whose full logits at the current rev are byte-identical, on the same arch and
// backend, to the build its T3 oracle ran at. <method> is that original T3 method and <rev> the rev it ran at; both
// carry through chains of inheritance unchanged, and the row keeps the original metrics. It clears T3 exactly when the
// inner method does.
func identityInherited(m string) (inner, rev string, ok bool) {
	const pre = "identity-inherited ("
	if !strings.HasPrefix(m, pre) || !strings.HasSuffix(m, ")") {
		return "", "", false
	}
	body := strings.TrimSuffix(strings.TrimPrefix(m, pre), ")")
	i := strings.LastIndex(body, " @ ")
	if i <= 0 || i+3 >= len(body) {
		return "", "", false
	}
	return body[:i], body[i+3:], true
}

// TestParityManifest_methodTier is the claim-discipline gate: it makes "validated" MEAN T3. parity-coverage-policy.md
// defines which methods clear T3; this test enforces it. A T1 method (`tiny-golden`, cosine against the family's own
// seeded tiny golden, no released checkpoint) must not sit at `status: "validated"`, where the capability matrix and the
// README's supported count read it. The staleness gate cannot catch that: it keys on deps_hash freshness, which says
// nothing about how the row was validated.
//
// A weak row does not have to lie: `status: "experimental"` keeps its method and metrics, renders distinctly in the
// capability matrix and is excluded from the supported count. Downgrading is not a regression.
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
			if !isT3Method(methodString(f.Method)) {
				bad = append(bad, fmt.Sprintf("  %-18s method=%q — not a T3 method", name, methodString(f.Method)))
			}
		case "experimental":
			if methodString(f.Method) == "" {
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

// applyParityRows folds PARITY_ROW lines into m and returns the families it touched. It is a function, not an inline
// loop body, so the regression gates in parity_emit_b15_test.go drive the real merge rather than a test-only copy of it.
func applyParityRows(m *parityManifest, rows, sha, date, machine string) ([]string, error) {
	var applied []string
	for line := range strings.SplitSeq(rows, "\n") {
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
			return nil, fmt.Errorf("bad PARITY_ROW line %q: %w", line, err)
		}
		if !knownParityMethod(row.Method) {
			return nil, fmt.Errorf("PARITY_ROW for %q carries method %q, which is not in the "+
				"manifest vocabulary %v", row.Family, row.Method, sortedKeys(parityMethods))
		}
		f, ok := m.Families[row.Family]
		if !ok {
			return nil, fmt.Errorf("PARITY_ROW for unknown family %q (not in %s)", row.Family, parityManifestPath)
		}
		// Status is DERIVED FROM THE METHOD, not asserted: a T3 method means validated, anything else experimental. A row can
		// therefore demote a family whose evidence was downgraded, and a sweep with EMIT_MANIFEST=1 cannot promote a
		// tiny-golden row to supported. TestParityManifest_methodTier enforces the same rule on the file.
		if isT3Method(row.Method) {
			f.Status = "validated"
		} else {
			f.Status = "experimental"
		}
		f.Method = rawString(row.Method)
		f.Reference = rawString(row.Reference)
		f.Metrics = row.Metrics
		f.ValidatedAt = rawString(sha)
		f.Date = rawString(date)
		f.Machine = rawString(machine)
		m.Families[row.Family] = f
		applied = append(applied, row.Family)
	}
	if len(applied) == 0 {
		return nil, fmt.Errorf("no PARITY_ROW lines found")
	}
	return applied, nil
}
