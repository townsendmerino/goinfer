package decoder

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/paritykey"
)

// TestParityKey_matchesFreshDepsHash is VK5's P1 and the builder switch's B1 (docs/tasks/task-subsystems-2026-10.md §5.8):
// paritykey's today key, read under the tree's key marker, equals freshDepsHash for every family, so the shadow job's
// "today" column is the gate's own hash.
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
	def, err := paritykey.TodayDefinition(src)
	if err != nil {
		t.Fatal(err)
	}
	progs, err := paritykey.ReadProgramTable(src)
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
		got, err := paritykey.TodayKey(src, pm, pm.Families[name], def, tail, progs)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: paritykey %s, freshDepsHash %s", name, got, want)
		}
	}
}

// isBuilder is the decoder's own statement of which files build programs (paritykey.BuilderFile's list, written again so
// P1 and W1 compare two computations): the program builder, the block spec, the architecture and config types, the registry,
// and every architecture adapter (gemma12.go holds gemma's and gemma2's).
func isBuilder(p string) bool {
	switch p {
	case "decoder/program.go", "decoder/blockspec.go", "decoder/arch.go", "decoder/config.go", "decoder/registry.go", "decoder/gemma12.go":
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
	// The kernel table's digest, computed here: each configuration in sorted order and a newline, then its kernels as
	// sorted "name=fingerprint" lines.
	var kt map[string]map[string]string
	kb, err := os.ReadFile(repoPath("testdata/kernel_fingerprints.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(kb, &kt); err != nil {
		t.Fatal(err)
	}
	kd := sha256.New()
	cfgs := make([]string, 0, len(kt))
	for c := range kt {
		cfgs = append(cfgs, c)
	}
	sort.Strings(cfgs)
	for _, c := range cfgs {
		kd.Write([]byte(c + "\n"))
		names := make([]string, 0, len(kt[c]))
		for k := range kt[c] {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, k := range names {
			kd.Write([]byte(k + "=" + kt[c][k] + "\n"))
		}
	}
	tail := "kernel_fingerprints=" + hex.EncodeToString(kd.Sum(nil))
	src := paritykey.Dir("..")
	pm, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	loaderTab, err := paritykey.ReadLoaderTable(src)
	if err != nil {
		t.Fatal(err)
	}
	with, qualifying := 0, 0
	for name, fam := range m.Families {
		want, err := freshDepsHash(&m, fam)
		if err != nil {
			t.Fatal(err)
		}
		if len(fam.Programs) > 0 {
			with++
			// The loader rule, computed here (docs/measurements/sb4-2026-10.md §4): every program fixture in the loader table for every
			// GOARCH, each one's GGUF beside it where the table has one.
			loaderFx := weekThreeLoaderFixtures(fam.Programs, loaderTab)
			dropWeights := len(loaderFx) > 0
			dropGGUF := false
			for _, f := range loaderFx {
				dropGGUF = dropGGUF || strings.HasSuffix(f, ".gguf")
			}
			if dropWeights {
				qualifying++
			}
			h := sha256.New()
			for _, p := range familyDepFiles(&m, fam) {
				if isBuilder(p) || (dropWeights && p == "decoder/weights.go") || (dropGGUF && p == "decoder/gguf.go") {
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
			loaderLine := ""
			if dropWeights {
				sort.Strings(loaderFx)
				ld := sha256.New()
				for _, a := range sortedStringKeys(loaderTab) {
					ld.Write([]byte(a + "\n"))
					for _, f := range loaderFx {
						for _, mode := range sortedStringKeys(loaderTab[a][f]) {
							ld.Write([]byte(f + " " + mode + "=" + loaderTab[a][f][mode] + "\n"))
						}
					}
				}
				loaderLine = "loaders=" + hex.EncodeToString(ld.Sum(nil)) + "\n"
			}
			h.Write([]byte("programs=" + hex.EncodeToString(d.Sum(nil)) + "\n" + loaderLine + tail))
			want = "sha256:" + hex.EncodeToString(h.Sum(nil))
		}
		got, err := paritykey.ShadowKey(src, pm, pm.Families[name], tail, progs, loaderTab)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: paritykey's shadow key %s, decoder's %s", name, got, want)
		}
	}
	t.Logf("%d families, %d with programs, %d under the loader rule", len(m.Families), with, qualifying)
}

// sortedStringKeys is m's keys in order.
func sortedStringKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// weekThreeLoaderFixtures is the loader rule's fixture list, nil when a program fixture lacks an entry for some GOARCH.
func weekThreeLoaderFixtures(programs []string, tab map[string]map[string]map[string]string) []string {
	var out []string
	for _, f := range programs {
		for _, arch := range tab {
			if len(arch[f]) == 0 {
				return nil
			}
		}
		out = append(out, f)
		both := true
		for _, arch := range tab {
			both = both && len(arch[f+".gguf"]) > 0
		}
		if both {
			out = append(out, f+".gguf")
		}
	}
	return out
}

// TestParityLoaders_mapping is VK5 week 3's L2: exactly the families the registration names fall under the loader rule, each GGUF in the
// loader table sits in exactly one family's list, and each listed fixture has an entry for every GOARCH.
func TestParityLoaders_mapping(t *testing.T) {
	src := paritykey.Dir("..")
	pm, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	tab, err := paritykey.ReadLoaderTable(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(tab) < 2 {
		t.Fatalf("the loader table covers %d GOARCHes", len(tab))
	}
	var withPrograms, under, keep []string
	ggufs := map[string][]string{}
	for name, fam := range pm.Families {
		if len(fam.Programs) == 0 {
			continue
		}
		withPrograms = append(withPrograms, name)
		fx, dropped := paritykey.LoaderRule(fam, tab)
		if len(fx) == 0 {
			keep = append(keep, name)
			continue
		}
		under = append(under, name)
		for _, f := range fx {
			if strings.HasSuffix(f, ".gguf") {
				ggufs[f] = append(ggufs[f], name)
			}
		}
		if !slices.Contains(dropped, "decoder/weights.go") {
			t.Errorf("%s qualifies and keeps weights.go: dropped %v", name, dropped)
		}
		hasGGUF := slices.ContainsFunc(fx, func(f string) bool { return strings.HasSuffix(f, ".gguf") })
		if slices.Contains(dropped, "decoder/gguf.go") != hasGGUF {
			t.Errorf("%s: GGUF fixture %v, dropped %v", name, hasGGUF, dropped)
		}
	}
	if len(under) != 30 || !slices.Equal(keep, []string{"mellum"}) {
		sort.Strings(keep)
		t.Errorf("%d under the rule of %d with programs, keeping %v; want 30, keeping [mellum]", len(under), len(withPrograms), keep)
	}
	for _, f := range []string{"testdata/gemma1-tiny.gguf", "testdata/gemma2-hd-tiny.gguf", "testdata/glm-tiny.gguf"} {
		if len(ggufs[f]) != 1 {
			t.Errorf("%s sits in %v, want exactly one family", f, ggufs[f])
		}
	}
	// Read from the table, not from the families: LoaderRule skips a GGUF it cannot place, so a new one, one on a single GOARCH, or one not
	// named after a program fixture would otherwise pass unseen.
	for arch, fixtures := range tab {
		for f := range fixtures {
			if strings.HasSuffix(f, ".gguf") && len(ggufs[f]) != 1 {
				t.Errorf("%s has %s in the loader table, in families %v; want exactly one", arch, f, ggufs[f])
			}
		}
	}
	for f := range ggufs {
		for arch, fixtures := range tab {
			if len(fixtures[f]) == 0 {
				t.Errorf("%s is in a family's list with no %s entry", f, arch)
			}
		}
	}
	if len(ggufs) != 3 {
		t.Errorf("GGUF fixtures %v, want the three the table holds", ggufs)
	}
}

// TestParityPrograms_mapping is VK5 week 2's W2: every fixture in the program table is in exactly one family's
// programs, has an entry for every GOARCH, and its config's adapter is defined in a file of that family's own list, so
// the program stands for that family's adapter. Where several families own that file, the fixture's model_type names
// the family.
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
			// Where several families own the adapter's file, the family the fixture's model_type names takes it.
			var owners []string
			for other, f := range m.Families {
				for _, o := range f.Own {
					if o == def {
						owners = append(owners, other)
					}
				}
			}
			if len(owners) > 1 && name != cfg.ModelType {
				sort.Strings(owners)
				t.Errorf("%s lists %s, whose adapter file %s is owned by %v: it belongs to the family its model_type %q names", name, fx, def, owners, cfg.ModelType)
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
