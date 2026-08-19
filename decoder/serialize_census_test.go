package decoder

// THE SILENT-DROP CENSUS: does a .giw round-trip preserve every per-layer field the loader
// populated? Generic over the struct, not over a remembered list.
//
// WHY THIS EXISTS. `canSerialize` is a hand-maintained blocklist of families the writer cannot
// express, and on 2026-08-19 it was found to have DRIFTED for two of them:
//
//   gpt_oss  accepted -> AttnSinks 8 -> 0, and the bundle LOADED CLEAN (silent wrong answers)
//   laguna   accepted -> reader rejected it at load ("layer 1 QProj: 128 rows, arch expects 64")
//
// Neither was exotic: gpt-oss needed ONE []float32 field written. The defect was not difficulty, it
// was that adding a field to LayerWeights and forgetting to add it to serialize.go produces no
// error anywhere — and the gate meant to catch that (TestCanSerialize_refusesUnrepresentable) asks
// only "is this family on the list?", which is the same memory that failed in the first place.
//
// So this test asks the struct instead: populate a model, serialize, load, and compare EVERY field
// of LayerWeights by reflection. A field that was non-empty and comes back empty is a silent drop,
// whatever family introduced it and whoever forgot it. Refused families are skipped — refusing is a
// correct answer; silently dropping is not.
//
// It runs on committed tiny fixtures only: no assets, no GPU, no network.

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// populated reports whether a LayerWeights field holds anything, for the kinds this struct uses.
// Exported WeightMat fields are measured by Rows(); unexported sub-struct pointers (delta, qattn,
// mla, mamba) only by nil-ness, since reflect forbids Interface() on values reached through an
// unexported field — nil-ness is exactly the check that matters for them anyway.
func populated(v reflect.Value) (bool, int) {
	switch v.Kind() {
	case reflect.Slice:
		return v.Len() > 0, v.Len()
	case reflect.Ptr:
		return !v.IsNil(), 1
	case reflect.Struct:
		if v.CanInterface() && v.CanAddr() {
			if m, ok := v.Addr().Interface().(*linalg.WeightMat); ok {
				return m.Rows() > 0, m.Rows()
			}
		}
		return false, 0
	}
	return false, 0
}

func TestSerializeCensus_noSilentFieldDrop(t *testing.T) {
	fixtures := []string{"laguna-xs2-tiny", "qwen3_5-tiny", "internlm2-tiny", "internlm3-tiny",
		"laguna-m1-tiny", "laguna-xs21-tiny", "gptoss_tiny.gguf"}
	checked := 0
	for _, fx := range fixtures {
		path := filepath.Join("testdata", fx)
		if _, err := os.Stat(path); err != nil {
			continue
		}
		t.Run(fx, func(t *testing.T) {
			m, err := Load(path, Options{})
			if err != nil {
				t.Skipf("load: %v", err)
			}
			defer m.Close()
			if e := canSerialize(m.w.arch); e != nil {
				t.Skipf("refused (a correct answer): %v", e)
			}
			blob, err := SerializeWeights(m.w, "census")
			if err != nil {
				t.Fatalf("serialize: %v", err)
			}
			w2, err := LoadSerializedWeights(blob)
			if err != nil {
				// A family the writer accepts but the reader cannot load is also a defect — it
				// hands the user an artifact that fails only later.
				t.Fatalf("canSerialize ACCEPTED this family but the bundle does not load: %v", err)
			}
			checked++
			lt := reflect.TypeOf(m.w.Layers[0])
			for li := range m.w.Layers {
				if li >= len(w2.Layers) {
					t.Fatalf("round-trip lost layer %d", li)
				}
				before := reflect.ValueOf(&m.w.Layers[li]).Elem()
				after := reflect.ValueOf(&w2.Layers[li]).Elem()
				for f := range lt.NumField() {
					name := lt.Field(f).Name
					hadIt, n := populated(before.Field(f))
					hasIt, _ := populated(after.Field(f))
					if hadIt && !hasIt {
						t.Errorf("SILENT DROP: layer %d field %s was populated (%d) and is empty "+
							"after a .giw round-trip — serialize.go does not write it, and nothing "+
							"else would have told you", li, name, n)
					}
				}
			}
		})
	}
	if checked == 0 {
		t.Skip("no representable fixture available")
	}
	t.Logf("censused %d representable fixture(s) field-by-field", checked)
}
