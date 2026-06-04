package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestEOSIDs covers the scalar / list / absent shapes HF emits for
// eos_token_id directly on a Config.
func TestEOSIDs(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []int
	}{
		{"absent", "", nil},
		{"scalar", "1", []int{1}},
		{"list", "[1, 106]", []int{1, 106}},
		{"qwen3", "[151645, 151643]", []int{151645, 151643}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := Config{}
			if c.raw != "" {
				cfg.EOSTokenID = []byte(c.raw)
			}
			got := cfg.EOSIDs()
			if !eqInts(got, c.want) {
				t.Errorf("EOSIDs() = %v, want %v", got, c.want)
			}
		})
	}
}

// TestResolveEOSIDs verifies the load-time merge of config.json and
// generation_config.json — the path Qwen3 chat relies on: config.json carries
// only <|im_end|> (151645) while generation_config adds <|endoftext|>
// (151643), and both must stop a turn. Deduped, config.json first.
func TestResolveEOSIDs(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// config.json eos is a scalar; generation_config adds a second id plus a
	// duplicate of the first (must be deduped).
	write("generation_config.json", `{"eos_token_id": [151645, 151643]}`)
	cfg := &Config{EOSTokenID: []byte("151645")}
	got := resolveEOSIDs(dir, cfg)
	want := []int{151645, 151643}
	if !eqInts(got, want) {
		t.Errorf("resolveEOSIDs = %v, want %v (config first, deduped)", got, want)
	}

	// Missing generation_config.json → just config.json's ids (best-effort).
	empty := t.TempDir()
	if got := resolveEOSIDs(empty, &Config{EOSTokenID: []byte("[2]")}); !eqInts(got, []int{2}) {
		t.Errorf("resolveEOSIDs (no gen config) = %v, want [2]", got)
	}
}

func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
