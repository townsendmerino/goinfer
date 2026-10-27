package paritykey

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// side is one family "f" with dependency files decoder/arch.go (a builder) and decoder/attention.go (an executor),
// whose contents are arch and attn, its program list progs, and its programs digest digest.
func side(today, shadow, arch, attn, digest string, progs ...string) Side {
	return Side{Tail: "kernel_fingerprints=t",
		Today: map[string]string{"f": today}, Shadow: map[string]string{"f": shadow},
		Files:    map[string][]string{"f": {"decoder/arch.go", "decoder/attention.go"}},
		Programs: map[string][]string{"f": progs}, Digest: map[string]string{"f": digest},
		Content: map[string]string{"decoder/arch.go": arch, "decoder/attention.go": attn}}
}

// TestCompare_classes is W4: every class and reason, from sides built by hand.
func TestCompare_classes(t *testing.T) {
	base := side("t1", "s1", "a1", "x1", "p1", "fx")
	cases := []struct {
		name          string
		before, after Side
		class         Class
		explained     bool
	}{
		{"nothing moved", base, side("t1", "s1", "a1", "x1", "p1", "fx"), AgreeFresh, true},
		{"an executor moved", base, side("t2", "s2", "a1", "x2", "p1", "fx"), AgreeStale, true},
		{"a builder and its program moved", base, side("t2", "s2", "a2", "x1", "p2", "fx"), AgreeStale, true},
		{"a builder moved, no program did", base, side("t2", "s1", "a2", "x1", "p1", "fx"), OldOnly, true},
		{"a builder and an executor moved, the shadow key did not", base, side("t2", "s1", "a2", "x2", "p1", "fx"), OldOnly, false},
		{"a program moved, no dependency did", base, side("t1", "s2", "a1", "x1", "p2", "fx"), NewOnly, false},
		{"the program list changed", side("t1", "t1", "a1", "x1", ""), side("t1", "s2", "a1", "x1", "p1", "fx"), NewOnly, true},
		{"the program list and a program changed", base, side("t1", "s2", "a1", "x1", "p2", "fx", "fy"), NewOnly, true},
		{"today's tail moved, the shadow key did not", base, func() Side { s := side("t2", "s1", "a1", "x1", "p1", "fx"); s.Tail = "kernel_fingerprints=u"; return s }(), OldOnly, false},
	}
	for _, c := range cases {
		v := Compare(c.before, c.after)
		if len(v) != 1 || v[0].Class != c.class || v[0].Explained != c.explained {
			t.Errorf("%s: got %+v, want class %s explained %v", c.name, v, c.class, c.explained)
		}
	}
	after := base
	after.Today = map[string]string{"g": "t"}
	got := Compare(base, after)
	if len(got) != 2 || got[0].Family != "f" || got[0].Class != Removed || got[1].Family != "g" || got[1].Class != Added {
		t.Errorf("added and removed: got %+v", got)
	}
}

// TestBuilderFile names the builders and keeps every other file in the key.
func TestBuilderFile(t *testing.T) {
	for _, p := range []string{"decoder/program.go", "decoder/blockspec.go", "decoder/arch.go", "decoder/config.go", "decoder/registry.go", "decoder/arch_llama.go"} {
		if !BuilderFile(p) {
			t.Errorf("%s is a builder", p)
		}
	}
	for _, p := range []string{"decoder/program_cpu.go", "decoder/attention.go", "decoder/arch_llama_test.go", "decoder/weights.go", "decoder/gemma12.go", "scripts/arch_x.go"} {
		if BuilderFile(p) {
			t.Errorf("%s is not a builder", p)
		}
	}
}

// TestKey_onTheTree hashes every family of the working tree with both keys, so a manifest or table the package cannot
// read fails here rather than in the shadow job.
func TestKey_onTheTree(t *testing.T) {
	src := Dir("../..")
	s, err := LoadSide(src)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(s.Tail, KernelTailPrefix) {
		t.Errorf("today's tail %q: the tree's key is not on its kernel table", s.Tail)
	}
	with := 0
	for name := range s.Today {
		has := len(s.Programs[name]) > 0
		if has {
			with++
		}
		if same := s.Today[name] == s.Shadow[name]; same == has {
			t.Errorf("%s: %d programs, and today's key equal to the shadow key is %v", name, len(s.Programs[name]), same)
		}
	}
	if with == 0 {
		t.Fatal("no family has programs")
	}
}

// TestManifestJSON_legacyTree reads a tree from before the records (one testdata/parity_manifest.json) unchanged, so
// gate shadow and the VK0 replay still compare a push that crosses the change.
func TestManifestJSON_legacyTree(t *testing.T) {
	dir := t.TempDir()
	legacy := []byte(`{"shared_sets": {"core": ["a.go"]}, "families": {"f": {"uses": ["core"]}}}`)
	if err := os.MkdirAll(filepath.Join(dir, "testdata"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, LegacyManifestPath), legacy, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ManifestJSON(Dir(dir))
	if err != nil || string(got) != string(legacy) {
		t.Fatalf("legacy tree: %q, %v", got, err)
	}
	if _, err := ManifestJSON(Dir(t.TempDir())); err == nil {
		t.Fatal("a tree with neither layout read as a manifest")
	}
}
