package config_test

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/config"
)

var updateDocs = flag.Bool("update", false, "rewrite the generated parts of docs/env-vars.md from the schema and docs/env-vars.meta.json")

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

// envMeta is docs/env-vars.meta.json: the page's tables.
type envMeta struct {
	Tables []envTable `json:"tables"`
}

type envTable struct {
	ID      string   `json:"id"`
	Columns []string `json:"columns"`
	Tests   bool     `json:"tests"` // its variables are read by tests, not production, so the schema does not hold them
	Rows    []struct {
		Vars    []string `json:"vars"`
		Label   string   `json:"label"`   // the first cell, when it is not the bare name of Vars[0]
		Outside string   `json:"outside"` // why the row's variables are not schema settings; empty when they are
		Cells   []string `json:"cells"`
	} `json:"rows"`
}

// text is t as the page shows it.
func (t envTable) text() string {
	var b strings.Builder
	b.WriteString("| " + strings.Join(t.Columns, " | ") + " |\n|" + strings.Repeat("---|", len(t.Columns)) + "\n")
	for _, r := range t.Rows {
		first := r.Label
		if first == "" {
			first = "`" + r.Vars[0] + "`"
		}
		b.WriteString("| " + strings.Join(append([]string{first}, r.Cells...), " | ") + " |\n")
	}
	return b.String()
}

// checkRows is G-SB10-17: every variable a row documents is a schema setting, unless its table is a tests table or the
// row says why not (and then it is not one), named in its first cell, and in one row only.
func checkRows(m envMeta, settings []config.Setting) []string {
	env := map[string]bool{}
	for _, s := range settings {
		if s.Env != "" {
			env[s.Env] = true
		}
	}
	var bad []string
	seen := map[string]string{}
	for _, t := range m.Tables {
		for _, r := range t.Rows {
			if len(r.Vars) == 0 {
				bad = append(bad, fmt.Sprintf("table %s: a row documents no variable", t.ID))
			}
			for _, v := range r.Vars {
				switch {
				case r.Outside != "" && env[v]:
					bad = append(bad, fmt.Sprintf("table %s: %s is a setting in internal/config, but its row says it is not", t.ID, v))
				case !t.Tests && r.Outside == "" && !env[v]:
					bad = append(bad, fmt.Sprintf("table %s: %s is not a setting in internal/config", t.ID, v))
				}
				if r.Label != "" && !strings.Contains(r.Label, v) {
					bad = append(bad, fmt.Sprintf("table %s: %s is not named in its row's label %q", t.ID, v, r.Label))
				}
				if prev, ok := seen[v]; ok {
					bad = append(bad, fmt.Sprintf("%s is documented in tables %s and %s", v, prev, t.ID))
				}
				seen[v] = t.ID
			}
		}
	}
	return bad
}

func readEnvMeta(t *testing.T) envMeta {
	t.Helper()
	raw, err := os.ReadFile("../../docs/env-vars.meta.json")
	if err != nil {
		t.Fatal(err)
	}
	var m envMeta
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("docs/env-vars.meta.json: %v", err)
	}
	return m
}

// TestEnvVarsDoc is G-SB10-12, 16 and 17 (docs/tasks/task-subsystems-2026-10.md § SB10): docs/env-vars.md's knob
// paragraph is the schema's knobs, its tables are docs/env-vars.meta.json's rows over schema settings, and every
// variable in the schema is named on the page.
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
	meta := readEnvMeta(t)
	for _, line := range checkRows(meta, config.Settings) {
		t.Error(line)
	}
	for _, tb := range meta.Tables {
		block := regexp.MustCompile(`(?s)<!-- generated: table ` + regexp.QuoteMeta(tb.ID) + ` -->\n.*?<!-- /generated -->`)
		if !block.MatchString(want) {
			t.Errorf("%s has no <!-- generated: table %s --> block", path, tb.ID)
			continue
		}
		want = block.ReplaceAllLiteralString(want, "<!-- generated: table "+tb.ID+" -->\n"+tb.text()+"<!-- /generated -->")
	}
	if *updateDocs {
		if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
			t.Fatal(err)
		}
		doc = want
	}
	if doc != want {
		t.Errorf("%s's generated parts are not the schema's and docs/env-vars.meta.json's: run go test ./internal/config -run TestEnvVarsDoc -update", path)
	}
	for _, s := range config.Settings {
		if s.Env != "" && !strings.Contains(doc, "`"+s.Env+"`") && !strings.Contains(doc, "`"+s.Env+"=") {
			t.Errorf("%s (setting %s) is not named on %s", s.Env, s.Name, path)
		}
	}
}

// TestEnvVarsDoc_plants shows checkRows red: a row naming a variable the schema lacks, one named twice, and a label
// that does not name its variable.
func TestEnvVarsDoc_plants(t *testing.T) {
	for _, c := range []struct {
		name, want string
		edit       func(*envMeta)
	}{
		{"not in schema", "table serving: GOINFER_PLANTED is not a setting", func(m *envMeta) { m.Tables[0].Rows[0].Vars = []string{"GOINFER_PLANTED"} }},
		{"twice", "GOINFER_API_KEY is documented in tables serving and residency", func(m *envMeta) {
			m.Tables[1].Rows[0].Vars = append(m.Tables[1].Rows[0].Vars, "GOINFER_API_KEY")
		}},
		{"label", "table serving: GOINFER_API_KEY is not named in its row's label", func(m *envMeta) { m.Tables[0].Rows[0].Label = "`X`" }},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := readEnvMeta(t)
			c.edit(&m)
			got := checkRows(m, config.Settings)
			if !slices.ContainsFunc(got, func(l string) bool { return strings.HasPrefix(l, c.want) }) {
				t.Errorf("planted %s not reported; checkRows said %q", c.name, got)
			}
		})
	}
}
