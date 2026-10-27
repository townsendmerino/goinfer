package decoder

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// TestParityKey_matchesFreshDepsHash is VK5's P1 (docs/tasks/task-subsystems-2026-10.md §5.8): internal/paritykey's
// today key equals freshDepsHash for every family, so the shadow job's "today" column is the gate's own hash.
func TestParityKey_matchesFreshDepsHash(t *testing.T) {
	raw, err := parityManifestJSON()
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	src := paritykey.Dir("..")
	pm, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := paritykey.TodayTail(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Families) == 0 || len(pm.Families) != len(m.Families) {
		t.Fatalf("%d families in the manifest, %d read by paritykey", len(m.Families), len(pm.Families))
	}
	for name, fam := range m.Families {
		want, err := freshDepsHash(&m, fam)
		if err != nil {
			t.Fatal(err)
		}
		got, err := paritykey.Key(src, pm, pm.Families[name], tail)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: paritykey %s, freshDepsHash %s", name, got, want)
		}
	}
}

// isBuilder is the decoder's own statement of which files build programs (paritykey.BuilderFile's list, written again so
// W1 compares two computations): the program builder, the block spec, the architecture and config types, the registry,
// and every architecture adapter.
func isBuilder(p string) bool {
	switch p {
	case "decoder/program.go", "decoder/blockspec.go", "decoder/arch.go", "decoder/config.go", "decoder/registry.go":
		return true
	}
	base := filepath.Base(p)
	return filepath.Dir(p) == "decoder" && strings.HasPrefix(base, "arch_") && !strings.HasSuffix(base, "_test.go")
}

// TestParityKey_shadowMatchesDecoder is VK5 week 2's W1 (docs/tasks/task-subsystems-2026-10.md §5.8): the shadow key
// computed here (the family's dependency files less the builders, its fixtures' program fingerprints for every GOARCH,
// then freshDepsHash's kernel tail) equals paritykey.ShadowKey for every family, and a family with no programs keeps
// today's key.
func TestParityKey_shadowMatchesDecoder(t *testing.T) {
	raw, err := parityManifestJSON()
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	var progs map[string]map[string]string
	b, err := os.ReadFile(programFingerprintsPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &progs); err != nil {
		t.Fatal(err)
	}
	kt, err := paritykey.ReadKernelTable(paritykey.Dir(repoPath(".")))
	if err != nil {
		t.Fatal(err)
	}
	tail := "kernel_fingerprints=" + paritykey.KernelTableDigest(kt)
	src := paritykey.Dir("..")
	pm, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	with := 0
	for name, fam := range m.Families {
		want, err := freshDepsHash(&m, fam)
		if err != nil {
			t.Fatal(err)
		}
		if len(fam.Programs) > 0 {
			with++
			h := sha256.New()
			for _, p := range familyDepFiles(&m, fam) {
				if isBuilder(p) {
					continue
				}
				c, err := os.ReadFile(repoPath(p))
				if err != nil {
					t.Fatal(err)
				}
				if strings.HasSuffix(p, ".go") {
					if c, err = goCodeTokens(p, c); err != nil {
						t.Fatal(err)
					}
				}
				h.Write([]byte(p + "\x00"))
				h.Write(c)
				h.Write([]byte{0})
			}
			fx := append([]string(nil), fam.Programs...)
			sort.Strings(fx)
			archs := make([]string, 0, len(progs))
			for a := range progs {
				archs = append(archs, a)
			}
			sort.Strings(archs)
			d := sha256.New()
			for _, a := range archs {
				d.Write([]byte(a + "\n"))
				for _, f := range fx {
					d.Write([]byte(f + "=" + progs[a][f] + "\n"))
				}
			}
			h.Write([]byte("programs=" + hex.EncodeToString(d.Sum(nil)) + "\n" + tail))
			want = "sha256:" + hex.EncodeToString(h.Sum(nil))
		}
		got, err := paritykey.ShadowKey(src, pm, pm.Families[name], tail, progs)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: paritykey's shadow key %s, decoder's %s", name, got, want)
		}
	}
	t.Logf("%d families, %d with programs", len(m.Families), with)
}

// TestParityPrograms_mapping is VK5 week 2's W2: every fixture in the program table is in exactly one family's
// programs, has an entry for every GOARCH, and its config's adapter is defined in a file of that family's own list, so
// the program stands for that family's adapter.
func TestParityPrograms_mapping(t *testing.T) {
	raw, err := parityManifestJSON()
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	var progs map[string]map[string]string
	b, err := os.ReadFile(programFingerprintsPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &progs); err != nil {
		t.Fatal(err)
	}
	owner := map[string]string{}
	for name, fam := range m.Families {
		for _, fx := range fam.Programs {
			if prev, ok := owner[fx]; ok {
				t.Errorf("%s is in both %s's and %s's programs", fx, prev, name)
			}
			owner[fx] = name
			for arch, tab := range progs {
				if _, ok := tab[fx]; !ok {
					t.Errorf("%s: %s is not in the %s program table", name, fx, arch)
				}
			}
			cfg, err := loadConfig(os.DirFS(repoPath(fx)), "config.json")
			if err != nil {
				t.Errorf("%s: %s: %v", name, fx, err)
				continue
			}
			adapter, ok := registry[cfg.ModelType]
			if !ok {
				t.Errorf("%s: %s's model_type %q has no adapter", name, fx, cfg.ModelType)
				continue
			}
			file, _ := runtime.FuncForPC(reflect.ValueOf(adapter).Pointer()).FileLine(reflect.ValueOf(adapter).Pointer())
			def := "decoder/" + filepath.Base(file)
			found := false
			for _, o := range fam.Own {
				found = found || o == def
			}
			if !found {
				t.Errorf("%s lists %s, whose adapter is defined in %s, not in the family's own files %v", name, fx, def, fam.Own)
			}
		}
	}
	for arch, tab := range progs {
		for fx := range tab {
			if _, ok := owner[fx]; !ok {
				t.Errorf("%s (%s program table) is in no family's programs", fx, arch)
			}
		}
	}
	t.Logf("%d fixtures mapped to families", len(owner))
}
