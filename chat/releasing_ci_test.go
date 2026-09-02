package chat

import (
	"os"
	"strings"
	"testing"
)

// M-34: RELEASING.md said the standalone (no-workspace) build was proved by "CI's
// standalone-build step (below)", and there was no such step — `.github/workflows/ci.yml` had
// no GOWORK=off anywhere, every submodule job running `go work init`. The B-01…B-04 class was
// caught only by a human running Step 2 by hand at tag time, and demo/agent — the fifth module,
// carrying the MCP-SDK demo — compiled for the FIRST time there.
//
// A prose claim about CI is exactly the kind that rots silently, so this asserts the artifact
// exists and does what the sentence says. It lives in chat/ only because the repo has no test
// package for repo-root concerns; the paths are explicit.
func TestReleasing_standaloneBuildGateExists(t *testing.T) {
	wf, err := os.ReadFile("../.github/workflows/standalone-build.yml")
	if err != nil {
		t.Fatalf("RELEASING.md relies on a standalone-build workflow that is not here: %v", err)
	}
	src := string(wf)
	for _, want := range []string{
		`GOWORK: "off"`, // the point of the gate
		"tags:",         // triggered by tag pushes, not every push
	} {
		if !strings.Contains(src, want) {
			t.Errorf("standalone-build.yml lacks %q", want)
		}
	}
	// Every shipped submodule must be covered, or the gate proves less than the doc claims.
	for _, mod := range []string{"gpu", "cuda", "metal", "demo/agent"} {
		if !strings.Contains(src, "module: "+mod) {
			t.Errorf("standalone-build.yml does not cover the %s module", mod)
		}
	}
	// It must NOT create a workspace — that would silently defeat the whole check. Comment
	// lines are skipped: the workflow EXPLAINS why it does not run `go work init`, and matching
	// that explanation is the same "a check that matches its own comment" trap this audit has
	// now produced three times.
	for _, ln := range strings.Split(src, "\n") {
		if strings.HasPrefix(strings.TrimSpace(ln), "#") {
			continue
		}
		if strings.Contains(ln, "go work init") {
			t.Errorf("standalone-build.yml runs `go work init` (%s): it would then prove the "+
				"opposite of what RELEASING.md claims, since a workspace is exactly what the "+
				"shipped tag must not need", strings.TrimSpace(ln))
		}
	}

	// And demo/agent must be built by ordinary CI too, or its only coverage is at tag time.
	ci, err := os.ReadFile("../.github/workflows/ci.yml")
	if err != nil {
		t.Fatalf("read ci.yml: %v", err)
	}
	if !strings.Contains(string(ci), "./demo/agent/...") {
		t.Error("ci.yml never builds ./demo/agent/...: the fifth module has no per-push coverage")
	}
}
