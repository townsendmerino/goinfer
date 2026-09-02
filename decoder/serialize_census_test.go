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
	"strings"
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
	case reflect.Pointer:
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

// censusList is the fixture set this census round-trips. See the completeness gate below.
var censusList = []string{
	"testdata/laguna-xs2-tiny", "testdata/qwen3_5-tiny", "testdata/internlm2-tiny",
	"testdata/internlm3-tiny", "testdata/laguna-m1-tiny", "testdata/laguna-xs21-tiny",
	"testdata/gptoss_tiny.gguf",
	"../testdata/deepseek-tiny", // MLA (q-LoRA + latent KV)
	"../testdata/kimi-tiny",     // MLA, the 384-expert shape
	"../testdata/granite-tiny",  // Mamba-2 + attention hybrid
	"../testdata/nemotron-tiny", // Mamba-2, single-op-per-block
	"../testdata/nemotron3nano-tiny",
	"../testdata/llama4-tiny", // iRoPE + dense/MoE interleave
	"../testdata/glm-tiny", "../testdata/mixtral-tiny", "../testdata/phi3-tiny",
	"../testdata/cohere-tiny", "../testdata/qwen35-tiny", "../testdata/qwen3next-tiny",
	"../testdata/gemma4-moe-tiny", "../testdata/tiny-qwen2-moe",
	// Added 2026-09-02 by TestSerializeCensus_everyFixtureIsListedOrExcluded, which found
	// eleven committed fixtures this list had never mentioned. lfm2-tiny is the one that
	// mattered: serialize.go dropped its entire conv mixer (audit-2026-09-02 C-03).
	"../testdata/lfm2-tiny", "../testdata/cohere2-tiny", "../testdata/gemma3-vl-tiny",
	"../testdata/gemma4-dense-twogeom-tiny", "../testdata/gemma4-moe-kv-tiny",
	"../testdata/gemma4-moe-unified-tiny", "../testdata/glm-tiny-bias",
	"../testdata/glm-tiny.gguf", "../testdata/qwen25vl-tiny", "testdata/qwen3_5_moe-tiny",
}

// censusExcluded names a committed model fixture the census deliberately does NOT round-trip, with
// the reason. Every fixture must be in censusList or here — TestSerializeCensus_everyFixtureIsListedOrExcluded.
//
// A LIST THAT NOBODY CHECKS IS THE DEFECT THIS TEST WAS BUILT TO REPLACE, AND IT CAME BACK. The
// census's own preamble says the guard before it failed because it "asks only 'is this family on
// the list?'" — and the replacement asked a list too, 21 hand-maintained paths. ../testdata/lfm2-tiny
// has been committed since ed112b0 and was on none of them, so `grep shortConv decoder/serialize.go`
// returned zero matches for an entire shipped family while this census reported green over 21
// others (audit-2026-09-02 C-03). Discovery plus an explicit exclusion is what makes the list a
// decision instead of a memory.
//
// The reasons here are COST, measured, not judgement: the census loads, serializes and greedy-decodes
// every fixture, and 29 of them together take 0.27s.
var censusExcluded = map[string]string{
	"gpt2":                 "529 MB — the real GPT-2, not a tiny fixture. Its per-layer fields are the generic dense set every *-tiny fixture here covers, and its distinctive state (learned PosEmbed) is MODEL-level, which this per-layer census does not reflect over at all.",
	"gemma4-dense-scaled":  "449 MB. gemma4's per-layer state (PLE, the dense‖MoE sub-block, two-geometry head dims) is covered by four other gemma4 fixtures on the list.",
	"mellum-mellum2-slice": "4.0 GB — a real-weight 4-layer slice. Measured: still running after 90s while all 29 listed fixtures together take 0.27s. mellum's per-layer fields are the generic set.",
	"siglip-tiny":          "a vision encoder, not a decoder — Load refuses it, so there are no LayerWeights to census.",
	"mistral-tiny-window":  "config-only fixture (no model.safetensors); it exists to pin sliding-window CONFIG parsing, and Load cannot open it.",
}

func TestSerializeCensus_noSilentFieldDrop(t *testing.T) {
	// Both fixture roots: decoder/testdata holds the families brought up in this package, and the
	// repo-root testdata holds the older ones. The MLA and Mamba-2 families live in the latter, and
	// they are the whole point of the v6 tail — a census that could not reach them would have
	// "passed" while the new code went unexercised.
	fixtures := censusList
	checked := 0
	for _, fx := range fixtures {
		path := fx
		if _, err := os.Stat(path); err != nil {
			continue
		}
		t.Run(filepath.Base(fx), func(t *testing.T) {
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
			// DECODE IDENTITY, not just field presence. A field can survive a round-trip and still
			// be wrong — restored at the wrong offset, or restored while some sibling it depends on
			// was not. Greedy-decoding both models and requiring the SAME tokens is what turns "the
			// tail is written" into "the model that comes back is the model that went in". This is
			// the check the qwen3_5_moe and gemma4 round-trip tests already made by hand; here it
			// runs for every family the census reaches.
			m2, err := NewModel(w2, "cpu")
			if err != nil {
				t.Fatalf("new model from round-tripped weights: %v", err)
			}
			prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
			a, b := greedyN(t, m, prompt, 6), greedyN(t, m2, prompt, 6)
			if !slicesEqualInt(a, b) {
				t.Errorf(".giw round-trip CHANGED THE DECODE:\n direct:     %v\n round-trip: %v", a, b)
			}
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

// EVERY COMMITTED FIXTURE IS LISTED OR EXPLICITLY EXCLUDED.
//
// The census above is only as complete as its fixture list, and that list is hand-maintained — the
// exact shape whose failure it was written to replace. This walks both testdata roots, treats an
// entry with a config.json (or a .gguf/.giw file) as a model fixture, and requires each to be
// censused or excluded with a written reason. It is a STAT, not a load, so it costs nothing and
// cannot be the reason someone trims the list.
func TestSerializeCensus_everyFixtureIsListedOrExcluded(t *testing.T) {
	listed := map[string]bool{}
	for _, p := range censusList {
		listed[filepath.Base(p)] = true
	}
	found := map[string]bool{}
	for _, root := range []string{"testdata", "../testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			name := e.Name()
			switch {
			case e.IsDir():
				if _, err := os.Stat(filepath.Join(root, name, "config.json")); err != nil {
					continue // a golden dir, a tokenizer fixture: not a model
				}
			case strings.HasSuffix(name, ".gguf"), strings.HasSuffix(name, ".giw"):
			default:
				continue
			}
			found[name] = true
			reason, excluded := censusExcluded[name]
			switch {
			case listed[name] && excluded:
				t.Errorf("%s is both censused and in censusExcluded — the two disagree about "+
					"whether this fixture is covered", name)
			case listed[name]:
			case excluded && strings.TrimSpace(reason) == "":
				t.Errorf("%s is in censusExcluded with an EMPTY reason", name)
			case excluded:
			default:
				t.Errorf("committed model fixture %s is in neither censusList nor censusExcluded. "+
					"An unlisted fixture is a family this census does not speak for, and that is "+
					"exactly how serialize.go came to drop LFM2's entire conv mixer while this "+
					"test reported green over 21 other families (audit-2026-09-02 C-03).", name)
			}
		}
	}
	if len(found) == 0 {
		t.Fatal("no model fixture found under either testdata root — the scan is broken, and a " +
			"broken scan makes this gate pass over nothing")
	}
	for name := range censusExcluded {
		if !found[name] {
			t.Errorf("censusExcluded names %q, which is not a committed fixture — a stale exemption "+
				"reads as a fixture somebody considered", name)
		}
	}
	t.Logf("%d committed model fixture(s): %d censused, %d explicitly excluded",
		len(found), len(found)-len(censusExcluded), len(censusExcluded))
}
