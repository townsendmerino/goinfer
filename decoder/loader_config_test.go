package decoder

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/paritykey"
)

// fixtureConfigEntries is the entries the loader table holds for fx's parsed config, beside its per-mode tensor digests.
func fixtureConfigEntries(fx string, cfg *Config) map[string]string {
	return nil
}

// TestLoaderIdentity_ggufConfigPlant: for every family under the loader rule with a GGUF loader fixture, a numeric change
// to the config the GGUF parser hands the model (planted here as a changed rms_norm_eps) moves the family's shadow key.
// The shadow key drops decoder/gguf.go for such a family, so the loader table must stand for the GGUF config parsers too.
func TestLoaderIdentity_ggufConfigPlant(t *testing.T) {
	src := paritykey.Dir("..")
	m, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := paritykey.TodayTail(src)
	if err != nil {
		t.Fatal(err)
	}
	progs, err := paritykey.ReadProgramTable(src)
	if err != nil {
		t.Fatal(err)
	}
	tab, err := paritykey.ReadLoaderTable(src)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for name, fam := range m.Families {
		fixtures, dropped := paritykey.LoaderRule(fam, tab)
		for _, fx := range fixtures {
			if !strings.HasSuffix(fx, ".gguf") {
				continue
			}
			w, err := loadForIdentity(filepath.Join("..", fx), "f32")
			if err != nil {
				t.Fatalf("%s: %v", fx, err)
			}
			planted := w.Cfg
			planted.RMSNormEps *= 2
			key := func(cfg *Config) string {
				k, err := paritykey.ShadowKey(src, m, fam, tail, progs, withLoaderEntries(tab, fx, fixtureConfigEntries(fx, cfg)))
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				return k
			}
			if key(&w.Cfg) == key(&planted) {
				t.Errorf("%s: a changed rms_norm_eps in %s's parsed config leaves the shadow key unchanged, though it drops %v", name, fx, dropped)
			}
			checked++
		}
	}
	if checked != 3 {
		t.Errorf("checked %d GGUF loader fixtures, want 3 (gemma1-tiny, gemma2-hd-tiny, glm-tiny)", checked)
	}
}

// withLoaderEntries is a copy of tab with entries set on fx for every GOARCH.
func withLoaderEntries(tab map[string]map[string]map[string]string, fx string, entries map[string]string) map[string]map[string]map[string]string {
	out := map[string]map[string]map[string]string{}
	for arch, fxs := range tab {
		out[arch] = map[string]map[string]string{}
		for f, modes := range fxs {
			out[arch][f] = map[string]string{}
			for mode, d := range modes {
				out[arch][f][mode] = d
			}
		}
		for mode, d := range entries {
			out[arch][fx][mode] = d
		}
	}
	return out
}
