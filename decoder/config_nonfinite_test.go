package decoder

import (
	"encoding/json"
	"math"
	"os"
	"testing"
	"testing/fstest"
)

// Released Nemotron-H checkpoints write config.json with Python's json.dump, which emits bare Infinity: the 4B's time_step_limit is
// [0.0, Infinity], and Go's encoding/json refuses the whole file. loadConfig accepts those tokens, outside strings only, and only there:
// the serve request decoders stay strict (internal/serveapp's TestRequestJSON_nonFiniteStaysRejected).

func TestLenientConfigJSON_tokens(t *testing.T) {
	type doc struct {
		A []float64 `json:"a"`
		S string    `json:"s"`
		N float64   `json:"NaN"`
	}
	cases := []struct {
		name, in string
		wantA    []float64
		wantS    string
		wantErr  bool
	}{
		{"infinity", `{"a":[0.0, Infinity]}`, []float64{0, math.MaxFloat64}, "", false},
		{"negative infinity", `{"a":[-Infinity, 1]}`, []float64{-math.MaxFloat64, 1}, "", false},
		{"nan becomes null", `{"a":[1, NaN, 2]}`, []float64{1, 0, 2}, "", false},
		{"in a string it is left alone", `{"s":"Infinity and NaN and -Infinity"}`, nil, "Infinity and NaN and -Infinity", false},
		{"an escaped quote does not end the string", `{"s":"a\" Infinity \" b","a":[Infinity]}`, []float64{math.MaxFloat64}, `a" Infinity " b`, false},
		{"a key named NaN is a key", `{"NaN": 3, "a":[NaN]}`, []float64{0}, "", false},
		{"only whole tokens: InfinityX stays invalid", `{"a":[InfinityX]}`, nil, "", true},
		{"only whole tokens: aNaN stays invalid", `{"a":[aNaN]}`, nil, "", true},
		{"a number is not a token", `{"a":[1e5, 2.5]}`, []float64{1e5, 2.5}, "", false},
	}
	for _, c := range cases {
		var d doc
		err := json.Unmarshal(lenientConfigJSON([]byte(c.in)), &d)
		if (err != nil) != c.wantErr {
			t.Errorf("%s: error %v, want error %v", c.name, err, c.wantErr)
			continue
		}
		if c.wantErr {
			continue
		}
		if len(d.A) != len(c.wantA) {
			t.Errorf("%s: a = %v, want %v", c.name, d.A, c.wantA)
			continue
		}
		for i := range d.A {
			if d.A[i] != c.wantA[i] {
				t.Errorf("%s: a[%d] = %v, want %v", c.name, i, d.A[i], c.wantA[i])
			}
		}
		if d.S != c.wantS {
			t.Errorf("%s: s = %q, want %q", c.name, d.S, c.wantS)
		}
	}
}

// A config without the tokens comes back as it came, byte for byte.
func TestLenientConfigJSON_cleanInputUntouched(t *testing.T) {
	in := []byte(`{"model_type":"llama","hidden_size":64,"note":"no tokens here"}`)
	if got := lenientConfigJSON(in); string(got) != string(in) {
		t.Errorf("changed a clean config: %s", got)
	}
}

// The real 4B config: plain encoding/json refuses it (so the fixture really carries the token), loadConfig reads it.
func TestLoadConfig_realNemotronH4B(t *testing.T) {
	raw, err := os.ReadFile("../testdata/nemotron-h-configs/h-4b-instruct-128k.config.json")
	if err != nil {
		t.Fatal(err)
	}
	var plain Config
	if json.Unmarshal(raw, &plain) == nil {
		t.Fatal("VOID: the fixture parses with encoding/json, so it no longer carries the token this test exists for")
	}
	cfg, err := loadConfig(fstest.MapFS{"config.json": {Data: raw}}, "config.json")
	if err != nil {
		t.Fatalf("loadConfig on the released 4B config: %v", err)
	}
	if cfg.ModelType != "nemotron_h" || cfg.HiddenDim != 3072 || cfg.NumHeads != 32 || cfg.MambaNumHeads != 112 {
		t.Errorf("parsed fields wrong: model_type %q, hidden %d, heads %d, mamba heads %d", cfg.ModelType, cfg.HiddenDim, cfg.NumHeads, cfg.MambaNumHeads)
	}
}
