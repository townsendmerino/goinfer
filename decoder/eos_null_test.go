package decoder

import (
	"slices"
	"testing"
)

// Through the loader, for the real nesting: the tiny Qwen3-ASR fixture's thinker_config.text_config carries eos_token_id null.
// The loaded model must not stop on id 0 (decoder/config's TestConfigEOSIDs_nullIsAbsent is the same defect without the loader; this is the path serve takes).
func TestQwen3ASR_loadedModelDoesNotStopOnIDZero(t *testing.T) {
	m := loadTinyASR(t)
	if slices.Contains(m.eosIDs, 0) {
		t.Fatalf("m.eosIDs = %v: id 0 (%q in Qwen's vocabulary) would end generation", m.eosIDs, "!")
	}
}
