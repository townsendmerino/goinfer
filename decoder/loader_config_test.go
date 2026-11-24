package decoder

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/paritykey"
)

// loaderConfigMode is a GGUF fixture's loader table entry for its parsed config: configIdentity of the Cfg its loader
// hands the model. A tensor identity leaves Cfg out and the program fingerprint reads the fixture directory's config.json,
// not the .gguf, so this entry is what stands for the GGUF config parsers once the shadow key drops decoder/gguf.go. Parsing
// is the same on every GOARCH, so the entry is too.
const loaderConfigMode = "config"

// fixtureConfigEntries is the entries the loader table holds for fx's parsed config, beside its per-mode tensor digests:
// loaderConfigMode for a GGUF fixture, none for a directory fixture (its config.json reaches the key through the program
// fingerprint).
func fixtureConfigEntries(fx string, cfg *Config) map[string]string {
	if !strings.HasSuffix(fx, ".gguf") {
		return nil
	}
	return map[string]string{loaderConfigMode: configIdentity(cfg)}
}

// configIdentity is a SHA-256 over every field of cfg, exported or not and json-tagged or not, by reflection: pointers
// followed (nil marked), slices with their length, maps in sorted key order, floats by their bits. A kind it cannot hash
// (func, chan, interface) panics, so a new field of one fails here rather than dropping out of the digest.
func configIdentity(cfg *Config) string {
	h := sha256.New()
	var walk func(path string, v reflect.Value)
	walk = func(path string, v reflect.Value) {
		switch v.Kind() {
		case reflect.Pointer:
			if v.IsNil() {
				fmt.Fprintf(h, "%s=nil\n", path)
				return
			}
			walk(path+"*", v.Elem())
		case reflect.Struct:
			for i := range v.NumField() {
				walk(path+"."+v.Type().Field(i).Name, v.Field(i))
			}
		case reflect.Slice, reflect.Array:
			fmt.Fprintf(h, "%s#%d\n", path, v.Len())
			for i := range v.Len() {
				walk(fmt.Sprintf("%s[%d]", path, i), v.Index(i))
			}
		case reflect.Map:
			keys := v.MapKeys()
			slices.SortFunc(keys, func(a, b reflect.Value) int { return cmp.Compare(fmt.Sprint(a), fmt.Sprint(b)) })
			fmt.Fprintf(h, "%s#%d\n", path, len(keys))
			for _, k := range keys {
				walk(fmt.Sprintf("%s[%v]", path, k), v.MapIndex(k))
			}
		case reflect.Float32, reflect.Float64:
			fmt.Fprintf(h, "%s=%x\n", path, math.Float64bits(v.Float()))
		case reflect.Bool, reflect.String, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			fmt.Fprintf(h, "%s=%v\n", path, v)
		default:
			panic(fmt.Sprintf("configIdentity: %s: unsupported kind %s", path, v.Kind()))
		}
	}
	walk("Cfg", reflect.ValueOf(cfg).Elem())
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
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
			maps.Copy(out[arch][f], modes)
		}
		maps.Copy(out[arch][fx], entries)
	}
	return out
}
