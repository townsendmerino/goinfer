package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// modelOptionFields is every Model field modelFromOptions derives from Options, plus a knob, so a
// constructor that drops one shows up as a field that differs.
type modelOptionFields struct {
	kvF16, kvPrecI8, kvI8, disableFit, moeCache, exactPrefill bool
	resCtxReq, moeSlots                                       int
	extraBytes, extraKVPerPos                                 int64
	cpuFastAttn                                               string
}

func optionFieldsOf(m *Model) modelOptionFields {
	v, _ := m.Knob(knobCPUFastAttention)
	return modelOptionFields{m.kvF16, m.kvPrecI8, m.kvI8, m.disableFit, m.moeCache, m.exactPrefill,
		m.resCtxReq, m.moeSlots, m.extraBytes, m.extraKVPerPos, v}
}

// TestConstructors_carryTheSameLoadOptions: every way to build a Model honours the per-model load
// options. They were separate struct literals, and they had drifted — LoadGGUFBytes (the baked-in
// raw-GGUF chat build) never read ExtraResidentBytes/ExtraResidentKVPerPosition, and NewModel (the
// baked-in prequant chat build) read nothing but the backend, so --kv, --fit and --exact-prefill were
// silently ignored there. One tiny model through all four paths, the same Options, the same fields.
func TestConstructors_carryTheSameLoadOptions(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(ggufPath, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	blob, err := SerializeWeights(loadTinyGGUFWeights(t, raw, "llama"), "tiny")
	if err != nil {
		t.Fatal(err)
	}
	giwPath := filepath.Join(dir, "tiny.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, kv := range []string{"f16", "i8"} {
		opts := Options{Backend: "cpu", KVPrecision: kv, KVQuant: kv, ResidentContext: 1234, DisableFit: true,
			MoECacheExperts: true, MoECacheSlots: 7, ExtraResidentBytes: 99, ExtraResidentKVPerPosition: 5,
			ExactPrefill: true, Knobs: &Knobs{knobCPUFastAttention: "0"}}
		want := modelOptionFields{kvF16: kv == "f16", kvPrecI8: kv == "i8", kvI8: kv == "i8", disableFit: true,
			moeCache: true, exactPrefill: true, resCtxReq: 1234, moeSlots: 7, extraBytes: 99, extraKVPerPos: 5,
			cpuFastAttn: "0"}

		build := map[string]func() (*Model, error){
			"Load(.giw)":          func() (*Model, error) { return Load(giwPath, opts) },
			"Load(.gguf), direct": func() (*Model, error) { return Load(ggufPath, opts) },
			"LoadGGUFBytes":       func() (*Model, error) { return LoadGGUFBytes(raw, opts) },
			"NewModelWithOptions": func() (*Model, error) {
				w, err := LoadSerializedWeights(blob)
				if err != nil {
					return nil, err
				}
				return NewModelWithOptions(w, opts)
			},
		}
		for name, f := range build {
			m, err := f()
			if err != nil {
				t.Fatalf("kv=%s %s: %v", kv, name, err)
			}
			if got := optionFieldsOf(m); got != want {
				t.Errorf("kv=%s %s dropped a load option:\n got  %+v\n want %+v", kv, name, got, want)
			}
			m.Close()
		}
	}
}

// TestNewModelWithOptions_refusesStreamWeights: paging reads a .giw's file mapping, which weights
// built in memory do not have, so the option is refused rather than silently doing nothing.
func TestNewModelWithOptions_refusesStreamWeights(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	blob, err := SerializeWeights(loadTinyGGUFWeights(t, raw, "llama"), "tiny")
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatal(err)
	}
	if m, err := NewModelWithOptions(w, Options{Backend: "cpu", StreamWeights: true}); err == nil {
		m.Close()
		t.Fatal("NewModelWithOptions accepted StreamWeights for in-memory weights")
	}
}
