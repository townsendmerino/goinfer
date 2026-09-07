package pull

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// R14 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the README named opencode
// alongside Claude Code as "a real agent" target for three releases before a recipe for it
// existed anywhere in the tree — a cold user had to reconstruct opencode's provider config from
// outside knowledge, and that reconstruction cost the whole run's only safety incident. The
// README's own claim ("Pointing a real agent (Claude Code, opencode) at it: docs/integrations/")
// is the thing that silently went stale; this reads it back and checks the promise against the
// directory it names, so a THIRD harness added to that sentence without a matching page fails
// here instead of waiting for the next cold-user run to find it.
//
// Mutation: name a harness in README.md's integrations sentence without adding its page under
// docs/integrations/ (e.g. add ", cursor" to the parenthetical) — this test goes red naming the
// missing docs/integrations/cursor.md.
func TestIntegrationsDoc_everyHarnessTheReadmeNamesHasAPage(t *testing.T) {
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	m := regexp.MustCompile(`Pointing a real agent \(([^)]+)\) at`).FindSubmatch(readme)
	if m == nil {
		t.Fatal("README.md no longer has the \"Pointing a real agent (...)\" sentence this test reads — " +
			"update the regex here to match its replacement, don't just delete this test")
	}
	names := strings.Split(string(m[1]), ",")
	if len(names) == 0 {
		t.Fatal("parsed zero harness names out of the README sentence")
	}
	for _, name := range names {
		name = strings.TrimSpace(name)
		slug := strings.ReplaceAll(strings.ToLower(name), " ", "-")
		path := "../docs/integrations/" + slug + ".md"
		if _, err := os.Stat(path); err != nil {
			t.Errorf("README names %q as a real-agent target but %s does not exist", name, path)
		}
	}
}
