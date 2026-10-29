package config_test

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/config"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// surfaces is what the tree registers today: the schema is checked against it, never the other way round.
type surfaces struct {
	options  []string                  // exported decoder.Options fields
	env      map[string]config.EnvRole // testdata/env_reads.txt and the knobs in decoder/knobs.go
	flags    map[string]config.Apps    // loadflags.Register for each app, plus serve's flags in docs/flags.meta.json
	defaults map[string]string         // DefValue of each loadflags flag on a fresh FlagSet, GOINFER_GGUF_DIRECT unset
}

var goinferName = regexp.MustCompile(`"(GOINFER_[A-Z0-9_]+)"`)

func readSurfaces(t *testing.T) surfaces {
	t.Helper()
	s := surfaces{env: map[string]config.EnvRole{}, flags: map[string]config.Apps{}, defaults: map[string]string{}}
	ot := reflect.TypeOf(decoder.Options{})
	for i := range ot.NumField() {
		if f := ot.Field(i); f.IsExported() {
			s.options = append(s.options, f.Name)
		}
	}

	reads, err := os.ReadFile("../../testdata/env_reads.txt")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(reads), "\n") {
		name, rest, ok := strings.Cut(line, "\t")
		if !ok || strings.HasPrefix(line, "#") {
			continue
		}
		switch {
		case strings.HasPrefix(rest, "startup:"):
			s.env[name] = config.EnvStartup
		case strings.HasPrefix(rest, "diagnostic:"):
			s.env[name] = config.EnvDiagnostic
		default:
			t.Fatalf("env_reads.txt: %q has no startup: or diagnostic: role", line)
		}
	}
	knobs, err := os.ReadFile("../../decoder/knobs.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range goinferName.FindAllStringSubmatch(string(knobs), -1) {
		s.env[m[1]] = config.EnvKnob
	}

	t.Setenv("GOINFER_GGUF_DIRECT", "")
	for _, app := range []struct {
		la  loadflags.App
		bit config.Apps
	}{{loadflags.Chat, config.Chat}, {loadflags.Serve, config.Serve}} {
		fs := flag.NewFlagSet("config", flag.ContinueOnError)
		loadflags.Register(fs, app.la)
		fs.VisitAll(func(f *flag.Flag) {
			s.flags[f.Name] |= app.bit
			s.defaults[f.Name] = f.DefValue
		})
	}
	meta, err := os.ReadFile("../../docs/flags.meta.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Flags []struct{ Name string } `json:"flags"`
	}
	if err := json.Unmarshal(meta, &doc); err != nil {
		t.Fatal(err)
	}
	for _, f := range doc.Flags {
		s.flags[f.Name] |= config.Serve
	}
	return s
}

// check returns one line per way settings and the tree disagree, or the table is malformed (G-SB10-1, G-SB10-2).
func check(settings []config.Setting, s surfaces, goos string) []string {
	var bad []string
	badf := func(format string, a ...any) { bad = append(bad, fmt.Sprintf(format, a...)) }
	names, flags, envs, options := map[string]bool{}, map[string]string{}, map[string]string{}, map[string]string{}
	claim := func(seen map[string]string, surface, value, by string) {
		if value == "" {
			return
		}
		if prev, ok := seen[value]; ok {
			badf("%s %s claimed by both %s and %s", surface, value, prev, by)
		}
		seen[value] = by
	}
	for _, st := range settings {
		if names[st.Name] {
			badf("setting %s listed twice", st.Name)
		}
		names[st.Name] = true
		claim(flags, "flag", st.Flag, st.Name)
		claim(envs, "env", st.Env, st.Name)
		claim(options, "option", st.Option, st.Name)
		if st.Flag == "" && st.Env == "" && st.Option == "" {
			badf("setting %s has no surface", st.Name)
		}
		if (st.Class == config.Internal) != (st.Reason != "") {
			badf("setting %s: an config.Internal setting, and only one, carries a Reason", st.Name)
		}
		if st.Class == config.Internal && (st.Flag != "" || st.Env != "") {
			badf("setting %s is config.Internal but has a flag or variable", st.Name)
		}
		if (st.PerModel != "") != strings.Contains(st.Help, "{per-model}") || (st.ServeHelp != "") != strings.Contains(st.Help, "{serve-only}") {
			badf("setting %s: a {per-model} or {serve-only} mark without its field, or the reverse", st.Name)
		}
		if st.Flag != "" {
			if apps, ok := s.flags[st.Flag]; !ok {
				badf("stale: flag %s (setting %s) is registered nowhere", st.Flag, st.Name)
			} else if apps != st.Apps {
				badf("flag %s: schema says apps %b, the tree registers it in %b", st.Flag, st.Apps, apps)
			}
		}
		if st.Env != "" {
			if role, ok := s.env[st.Env]; !ok {
				badf("stale: variable %s (setting %s) is read nowhere", st.Env, st.Name)
			} else if role != st.EnvRole {
				badf("variable %s: schema says role %d, the tree reads it as %d", st.Env, st.EnvRole, role)
			}
		}
		if st.Option != "" && !slices.Contains(s.options, st.Option) {
			badf("stale: decoder.Options.%s (setting %s) does not exist", st.Option, st.Name)
		}
		if def, ok := s.defaults[st.Flag]; ok && st.Flag != "" {
			want := st.DefaultFor(goos)
			if want != def {
				badf("flag %s: schema default %q, registered default %q on %s", st.Flag, want, def, goos)
			}
		}
	}
	for _, f := range sortedKeys(s.flags) {
		if _, ok := flags[f]; !ok {
			badf("flag %s is in no setting", f)
		}
	}
	for _, e := range sortedKeys(s.env) {
		if _, ok := envs[e]; !ok {
			badf("variable %s is in no setting", e)
		}
	}
	for _, o := range s.options {
		if _, ok := options[o]; !ok {
			badf("decoder.Options.%s is in no setting", o)
		}
	}
	return bad
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// TestInventory is G-SB10-1 and G-SB10-2: config.Settings claims every option, variable, knob and flag the tree has,
// each once, names nothing stale, and records each shared loading flag's registered default.
func TestInventory(t *testing.T) {
	for _, line := range check(config.Settings, readSurfaces(t), runtime.GOOS) {
		t.Error(line)
	}
}

// TestInventory_plants is G-SB10-3: check reports a planted surface missing from the schema, a planted stale
// setting, a duplicate claim and a wrong default, each by name.
func TestInventory_plants(t *testing.T) {
	real := readSurfaces(t)
	cases := []struct {
		name     string
		edit     func(*surfaces, *[]config.Setting)
		wantLine string
	}{
		{"extra option", func(s *surfaces, _ *[]config.Setting) { s.options = append(s.options, "PlantedField") }, "decoder.Options.PlantedField is in no setting"},
		{"extra variable", func(s *surfaces, _ *[]config.Setting) { s.env["GOINFER_PLANTED"] = config.EnvDiagnostic }, "variable GOINFER_PLANTED is in no setting"},
		{"extra knob", func(s *surfaces, _ *[]config.Setting) { s.env["GOINFER_PLANTED_KNOB"] = config.EnvKnob }, "variable GOINFER_PLANTED_KNOB is in no setting"},
		{"extra flag", func(s *surfaces, _ *[]config.Setting) { s.flags["planted-flag"] = config.Serve }, "flag planted-flag is in no setting"},
		{"stale setting", func(_ *surfaces, st *[]config.Setting) {
			*st = append(*st, config.Setting{Name: "gone", Flag: "gone", Apps: config.Serve})
		}, "stale: flag gone (setting gone) is registered nowhere"},
		{"double claim", func(_ *surfaces, st *[]config.Setting) {
			*st = append(*st, config.Setting{Name: "again", Env: "GOINFER_API_KEY", EnvRole: config.EnvStartup})
		}, "env GOINFER_API_KEY claimed by both api-key and again"},
		{"wrong default", func(s *surfaces, _ *[]config.Setting) { s.defaults["ctx"] = "4096" }, `flag ctx: schema default "0", registered default "4096"`},
		{"mark without key", func(_ *surfaces, st *[]config.Setting) { (*st)[0].Help += "{per-model}" }, "setting backend: a {per-model} or {serve-only} mark"},
		{"role changed", func(s *surfaces, _ *[]config.Setting) { s.env["GOINFER_API_KEY"] = config.EnvDiagnostic }, "variable GOINFER_API_KEY: schema says role 1, the tree reads it as 2"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := surfaces{options: slices.Clone(real.options), env: clone(real.env), flags: clone(real.flags), defaults: clone(real.defaults)}
			st := slices.Clone(config.Settings)
			c.edit(&s, &st)
			got := check(st, s, runtime.GOOS)
			if !slices.ContainsFunc(got, func(l string) bool { return strings.HasPrefix(l, c.wantLine) }) {
				t.Errorf("planted %s not reported; check said %q", c.name, got)
			}
		})
	}
}

func clone[V any](m map[string]V) map[string]V {
	out := make(map[string]V, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
