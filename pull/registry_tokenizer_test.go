package pull

import (
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// R8 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): granite-4.0-h-tiny — a
// registry-recommended checkpoint — loaded with "tokenizer.ggml.pre=\"dbrx\" is not a known
// pre-tokenizer; falling back to cl100k" on every pull. A registry entry backed by parity gates
// cannot ship with a tokenizer this build declines to walk: recommending it to a first-time user
// means recommending a checkpoint whose token ids may differ from HF and llama.cpp, silently,
// unless they happen to read a startup warning.
//
// registryTokenizerFixtures maps a registry short name to a COMMITTED GGUF (or GGUF-header-only)
// fixture carrying that checkpoint's real tokenizer metadata, so this test reads the checkpoint's
// ACTUAL tokenizer.ggml.pre rather than assuming one from the family name — the same discipline
// pull/registry_test.go's TestRegistry_digestsMatchLocalFiles applies to the download digest.
// Deliberately short, like recommendedCheckpoints itself: an entry here exists because someone
// fetched or extracted the real metadata, not because a fixture would be convenient to have.
var registryTokenizerFixtures = map[string]string{
	"granite-4.0-h-tiny": "../tokenizer/testdata/granite-dbrx-meta.gguf",
	"qwen2.5-coder-0.5b": "../testdata/qwen2-gguf/qwen2.5-0.5b-instruct-q8_0.gguf",
}

func TestRegistry_noEntryHasATokenizerDecline(t *testing.T) {
	checked := 0
	for _, c := range RecommendedAll() {
		path, ok := registryTokenizerFixtures[c.Name]
		if !ok {
			t.Logf("SKIP %s: no committed tokenizer fixture registered in registryTokenizerFixtures "+
				"(this is a skip, not a pass)", c.Name)
			continue
		}
		tk, err := tokenizer.LoadGGUF(path)
		if err != nil {
			t.Errorf("%s: load fixture %s: %v", c.Name, path, err)
			continue
		}
		if d := tk.PreTokenizerDecline(); d != "" {
			t.Errorf("%s: PreTokenizerDecline = %q — a registry entry cannot recommend a "+
				"checkpoint this build cannot walk the pre-tokenizer for; either measure the "+
				"shape (tokenizer/gguf.go's byteLevelKnobs) or remove the entry until it does",
				c.Name, d)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no registry entry had a fixture to check — this test would pass having checked " +
			"nothing, which is the failure mode it exists to prevent")
	}
	t.Logf("checked %d of %d registry entries for a tokenizer decline", checked, len(RecommendedAll()))
}
