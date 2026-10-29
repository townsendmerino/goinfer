package config_test

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/config"
)

var updateDocs = flag.Bool("update", false, "rewrite the generated parts of docs/env-vars.md from the schema")

// knobParagraph is the generated paragraph of docs/env-vars.md: the per-model knobs, in schema order.
func knobParagraph() string {
	var names []string
	for _, s := range config.Settings {
		if s.EnvRole == config.EnvKnob {
			names = append(names, "`"+s.Env+"`")
		}
	}
	var b strings.Builder
	line := fmt.Sprintf("These %d knobs are snapshotted when a model is loaded, not read on every forward:", len(names))
	for i, n := range names {
		w := n + ","
		if i == len(names)-1 {
			w = n + "."
		}
		if len(line)+1+len(w) > 120 {
			b.WriteString(line + "\n")
			line = w
			continue
		}
		line += " " + w
	}
	b.WriteString(line + "\n")
	return b.String()
}

var generatedKnobs = regexp.MustCompile(`(?s)(<!-- generated: knobs -->\n).*?(<!-- /generated -->)`)

// TestEnvVarsDoc is G-SB10-12 (docs/tasks/task-subsystems-2026-10.md § "Step 4's gates"): docs/env-vars.md's knob
// paragraph is the schema's knobs, and every variable in the schema is named on that page.
func TestEnvVarsDoc(t *testing.T) {
	const path = "../../docs/env-vars.md"
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	doc := string(raw)
	if !generatedKnobs.MatchString(doc) {
		t.Fatalf("%s has no <!-- generated: knobs --> block", path)
	}
	want := generatedKnobs.ReplaceAllLiteralString(doc, "<!-- generated: knobs -->\n"+knobParagraph()+"<!-- /generated -->")
	if *updateDocs {
		if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
			t.Fatal(err)
		}
		doc = want
	}
	if doc != want {
		t.Errorf("%s's knob paragraph is not the schema's: run go test ./internal/config -run TestEnvVarsDoc -update", path)
	}
	for _, s := range config.Settings {
		if s.Env != "" && !strings.Contains(doc, "`"+s.Env+"`") && !strings.Contains(doc, "`"+s.Env+"=") {
			t.Errorf("%s (setting %s) is not named on %s", s.Env, s.Name, path)
		}
	}
}
