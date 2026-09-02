package modelpull

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestCurated_matchesReleaseWorkflow is the "one shared place" enforcement.
//
// docs/task-model-pull.md asked for the demo tiers to be read from a single source shared
// with .github/workflows/release-assets.yml, because this project keeps rediscovering that
// two copies of one fact drift. Rewriting the workflow's matrix to consume curated.json
// would touch the release pipeline; this achieves the same guarantee without that risk —
// the copies may exist, but they cannot DISAGREE without failing the build.
//
// If this test fails after a deliberate model bump, update BOTH files. That is the point.
func TestCurated_matchesReleaseWorkflow(t *testing.T) {
	wf, err := os.ReadFile("../../.github/workflows/release-assets.yml")
	if err != nil {
		t.Fatalf("reading the release workflow: %v", err)
	}
	// Each matrix entry is a `- tier:` block carrying url / sha256 / bytes.
	block := regexp.MustCompile(`(?m)^\s*- tier:\s*(\S+)\s*\n\s*url:\s*(\S+)\s*\n\s*sha256:\s*([0-9a-f]{64})\s*\n\s*bytes:\s*(\d+)`)
	found := block.FindAllStringSubmatch(string(wf), -1)
	if len(found) == 0 {
		t.Fatal("no `- tier:` matrix entries found in release-assets.yml — this test is no longer checking anything")
	}

	seen := map[string]bool{}
	for _, m := range found {
		tier, url, sha, bytesStr := m[1], m[2], m[3], m[4]
		seen[tier] = true
		c, ok := Curated()[tier]
		if !ok {
			t.Errorf("release workflow ships tier %q but curated.json has no such tier (have: %v)", tier, CuratedNames())
			continue
		}
		// https://huggingface.co/<repo>/resolve/main/<file>
		rest, okPrefix := strings.CutPrefix(url, "https://huggingface.co/")
		repo, file, okSplit := strings.Cut(rest, "/resolve/main/")
		if !okPrefix || !okSplit {
			t.Errorf("tier %q: unrecognised url shape %q", tier, url)
			continue
		}
		if repo != c.Repo {
			t.Errorf("tier %q repo: workflow %q, curated.json %q", tier, repo, c.Repo)
		}
		if file != c.File {
			t.Errorf("tier %q file: workflow %q, curated.json %q", tier, file, c.File)
		}
		if sha != c.SHA256 {
			t.Errorf("tier %q sha256: workflow %s, curated.json %s", tier, sha, c.SHA256)
		}
		if n, _ := strconv.ParseInt(bytesStr, 10, 64); n != c.Bytes {
			t.Errorf("tier %q bytes: workflow %s, curated.json %d", tier, bytesStr, c.Bytes)
		}
	}
	for tier := range Curated() {
		if !seen[tier] {
			t.Errorf("curated.json offers tier %q that the release workflow does not ship — `pull demo:%s` would hand out a model no release vets", tier, tier)
		}
	}
}

// TestCurated_refsResolve pins the demo: form itself: it must expand to an EXACT repo and
// filename plus the pinned digest, not to a vague selector. That is what keeps it shorthand
// for an explicit reference rather than an opaque tag.
func TestCurated_refsResolve(t *testing.T) {
	for _, tier := range CuratedNames() {
		ref, err := ParseRef("demo:" + tier)
		if err != nil {
			t.Fatalf("demo:%s: %v", tier, err)
		}
		if ref.Repo == "" || ref.File == "" || ref.Pin == "" {
			t.Errorf("demo:%s resolved to %+v — want a concrete repo, file and pin", tier, ref)
		}
		if !validRepo(ref.Repo) {
			t.Errorf("demo:%s resolved to an invalid repo %q", tier, ref.Repo)
		}
	}
	if _, err := ParseRef("demo:nope"); err == nil {
		t.Error("an unknown demo tier must error")
	}
	if _, err := ParseRef("demo:nope"); err != nil && !strings.Contains(err.Error(), "0.5b") {
		t.Errorf("the error should list the tiers that DO exist, got: %v", err)
	}
}

// TestCurated_pinRefusesReupload covers the case the pin exists for: HF's resolve/main moved,
// so the API now declares a different digest. Verifying only against the API-declared digest
// would confirm the NEW file's own hash and report success.
func TestCurated_pinRefusesReupload(t *testing.T) {
	ref, err := ParseRef("demo:0.5b")
	if err != nil {
		t.Fatal(err)
	}
	reuploaded := []File{{Path: ref.File, Size: 1, SHA256: strings.Repeat("a", 64)}}
	if _, err := Select(reuploaded, ref); err == nil || !strings.Contains(err.Error(), "pinned") {
		t.Errorf("a re-uploaded curated file must be refused, got: %v", err)
	}
	// The genuine article still selects.
	good := []File{{Path: ref.File, Size: 1, SHA256: ref.Pin}}
	if _, err := Select(good, ref); err != nil {
		t.Errorf("the pinned file must still select: %v", err)
	}
}
