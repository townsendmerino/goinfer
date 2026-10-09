package decoder

import (
	"encoding/json"
	"slices"
	"testing"
)

// A config.json whose eos_token_id is JSON null (transformers writes the key with None when a model leaves it to
// generation_config.json: Qwen3-ASR's text_config does) must have NO config-side stop id. json.Unmarshal of null into an int
// succeeds and leaves 0, so EOSIDs() once returned [0], and token id 0 ended generation: in Qwen's vocabulary id 0 is "!",
// so every Qwen3-ASR transcription stopped at its first "!" (G-S14c4, 2026-10-08: 2 of the 2 LibriSpeech clips with one).
func TestConfigEOSIDs_nullIsAbsent(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string // "" = the key is absent
		want []int
	}{
		{"absent", "", nil},
		{"null", "null", nil},
		{"one", "151645", []int{151645}},
		{"list", "[151645,151643]", []int{151645, 151643}},
		{"zero is a real id when written", "0", []int{0}},
	} {
		var c Config
		if tc.raw != "" {
			c.EOSTokenID = json.RawMessage(tc.raw)
		}
		if got := c.EOSIDs(); !slices.Equal(got, tc.want) {
			t.Errorf("%s: EOSIDs() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// Through the loader, for the real nesting: the tiny Qwen3-ASR fixture's thinker_config.text_config carries eos_token_id null.
// The loaded model must not stop on id 0 (the unit above is the same defect without the loader; this is the path serve takes).
func TestQwen3ASR_loadedModelDoesNotStopOnIDZero(t *testing.T) {
	m := loadTinyASR(t)
	if slices.Contains(m.eosIDs, 0) {
		t.Fatalf("m.eosIDs = %v: id 0 (%q in Qwen's vocabulary) would end generation", m.eosIDs, "!")
	}
}
