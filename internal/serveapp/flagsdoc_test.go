package serveapp

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

var updateFlagsDoc = flag.Bool("update", false, "rewrite docs/flags.md from serve's flags instead of asserting it is fresh")

// flagsDoc is docs/flags.md's generator and its drift test. The page is built from the flags serve really registers
// (registerFlags, on a fresh FlagSet) plus curated text in docs/flags.meta.json, so a new flag cannot ship without a
// line on the site and a removed flag cannot linger there — the same shape as TestEnvVars_docAndCodeAgree and the
// capability matrix. The long help text stays in --help on purpose (docs/tasks/task-site-2026-09.md; main.go's own
// note on the cold-user run that could not skim it): the page carries one curated sentence per flag instead.

type flagMeta struct {
	Name       string `json:"name"`
	Group      string `json:"group"`
	Summary    string `json:"summary"`
	Type       string `json:"type"`
	Default    string `json:"default"`
	Deprecated string `json:"deprecated"`
	See        string `json:"see"`
}

type flagsMeta struct {
	Groups      map[string]string `json:"groups"`
	Flags       []flagMeta        `json:"flags"`
	Subcommands []struct {
		Cmd     string `json:"cmd"`
		Summary string `json:"summary"`
	} `json:"subcommands"`
}

// goTypeName names a flag's value type when the Go type says it; custom values and enums need "type" in the metadata.
func goTypeName(f *flag.Flag) (string, bool) {
	switch fmt.Sprintf("%T", f.Value) {
	case "*flag.boolValue":
		return "bool", true
	case "*flag.stringValue":
		return "string", true
	case "*flag.intValue", "*flag.int64Value":
		return "int", true
	case "*flag.durationValue":
		return "duration", true
	case "*flag.float64Value":
		return "number", true
	}
	return "", false
}

// cell makes s safe inside a Markdown table cell.
func cell(s string) string { return strings.ReplaceAll(strings.ReplaceAll(s, "|", `\|`), "\n", " ") }

func TestFlagsDoc(t *testing.T) {
	// direct-load's default reads this variable at registration; the page must not depend on the machine it was
	// generated on. (moe-pager's default depends on the OS and is pinned in the metadata.)
	t.Setenv("GOINFER_GGUF_DIRECT", "")
	root := filepath.Join("..", "..")
	raw, err := os.ReadFile(filepath.Join(root, "docs", "flags.meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta flagsMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatalf("docs/flags.meta.json: %v", err)
	}

	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	registerFlags(fs)
	registered := map[string]*flag.Flag{}
	fs.VisitAll(func(f *flag.Flag) { registered[f.Name] = f })
	shared := map[string]bool{}
	cfs := flag.NewFlagSet("chat", flag.ContinueOnError)
	loadflags.Register(cfs, loadflags.Chat)
	cfs.VisitAll(func(f *flag.Flag) { shared[f.Name] = true })

	var problems []string
	bad := func(format string, a ...any) { problems = append(problems, fmt.Sprintf(format, a...)) }
	seen := map[string]bool{}
	usedGroups := map[string]bool{}
	var groupOrder []string
	for _, m := range meta.Flags {
		if seen[m.Name] {
			bad("%s: listed twice in docs/flags.meta.json", m.Name)
		}
		seen[m.Name] = true
		f, ok := registered[m.Name]
		if !ok {
			bad("%s: in docs/flags.meta.json but serve registers no such flag — remove the entry", m.Name)
			continue
		}
		if _, ok := meta.Groups[m.Group]; !ok {
			bad("%s: group %q has no entry in \"groups\"", m.Name, m.Group)
		}
		if !usedGroups[m.Group] {
			usedGroups[m.Group] = true
			groupOrder = append(groupOrder, m.Group)
		}
		if strings.TrimSpace(m.Summary) == "" {
			bad("%s: empty summary", m.Name)
		}
		if n := len(m.Summary); n > 420 {
			bad("%s: summary is %d characters; the page is a skim — the long text belongs in --help", m.Name, n)
		}
		if _, ok := goTypeName(f); !ok && m.Type == "" {
			bad("%s: value type %T says nothing about what it takes; set \"type\" in docs/flags.meta.json", m.Name, f.Value)
		}
		if isDep := strings.HasPrefix(f.Usage, "DEPRECATED"); isDep != (m.Deprecated != "") {
			bad("%s: --help says deprecated=%v but docs/flags.meta.json says %q", m.Name, isDep, m.Deprecated)
		}
		if m.See != "" {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(m.See))); err != nil {
				bad("%s: see %q does not exist", m.Name, m.See)
			}
		}
	}
	var missing []string
	for name := range registered {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	for _, name := range missing {
		bad("%s: serve registers this flag but docs/flags.meta.json has no entry — add a group and a one-line summary", name)
	}
	for g := range meta.Groups {
		if !usedGroups[g] {
			bad("group %q has no flags", g)
		}
	}
	if len(problems) > 0 {
		t.Fatalf("docs/flags.meta.json and serve's flags disagree:\n  %s", strings.Join(problems, "\n  "))
	}

	var b bytes.Buffer
	b.WriteString("# serve flag reference\n\n")
	b.WriteString("<!-- Generated by `go test ./internal/serveapp -run TestFlagsDoc -update` from serve's real flags and docs/flags.meta.json. Do not edit by hand: a flag without an entry in the metadata fails that test. -->\n\n")
	fmt.Fprintf(&b, "All %d flags of `serve`, grouped, with the type and default of each. This page is a skim: one sentence per flag. "+
		"`serve --help` carries the full text of every flag, including the trade-off each one makes and what was measured. "+
		"Where a flag also has an environment variable, see [the environment variable registry](env-vars.md).\n\n", len(registered))
	b.WriteString("Every flag takes one or two dashes (`-model` and `--model` are the same). A flag marked *also in `goinfer-chat`* is shared with that binary.\n\n")
	b.WriteString("## Subcommands\n\nBesides serving, the binary has subcommands, each with its own `-h`:\n\n| Command | What it does |\n|---|---|\n")
	for _, s := range meta.Subcommands {
		fmt.Fprintf(&b, "| `serve %s` | %s |\n", cell(s.Cmd), cell(s.Summary))
	}
	b.WriteString("\n")
	for _, g := range groupOrder {
		fmt.Fprintf(&b, "## %s\n\n%s\n\n| Flag | Type | Default | What it does |\n|---|---|---|---|\n", g, meta.Groups[g])
		for _, m := range meta.Flags {
			if m.Group != g {
				continue
			}
			f := registered[m.Name]
			typ := m.Type
			if typ == "" {
				typ, _ = goTypeName(f)
			}
			def := m.Default
			switch {
			case def != "":
			case f.DefValue == "":
				def = "—"
			default:
				def = "`" + f.DefValue + "`"
			}
			what := m.Summary
			if m.Deprecated != "" {
				what = "**Deprecated: " + m.Deprecated + ".** " + what
			}
			if m.See != "" {
				what += " [More](" + strings.TrimPrefix(m.See, "docs/") + ")."
			}
			if shared[m.Name] {
				what += " *Also in `goinfer-chat`.*"
			}
			fmt.Fprintf(&b, "| `--%s` | `%s` | %s | %s |\n", m.Name, cell(typ), cell(def), cell(what))
		}
		b.WriteString("\n")
	}
	got := b.Bytes()

	path := filepath.Join(root, "docs", "flags.md")
	if *updateFlagsDoc {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s (%d flags)", path, len(registered))
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read docs/flags.md: %v (generate it with: go test ./internal/serveapp -run TestFlagsDoc -update)", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("docs/flags.md is stale against serve's flags or docs/flags.meta.json; regenerate with: go test ./internal/serveapp -run TestFlagsDoc -update")
	}
}
