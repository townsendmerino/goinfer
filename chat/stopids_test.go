package chat

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// stopIDsTokenizer is a byte-level tokenizer whose vocabulary holds some turn markers and not others.
func stopIDsTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	const raw = `{
		"model": {"type": "BPE", "vocab": {"a":0,"b":1,"<|im_start|>":2,"<|im_end|>":3,"<|eot_id|>":4,"<|end_of_text|>":5}, "merges": []},
		"decoder": {"type": "ByteLevel"}
	}`
	tk, err := tokenizer.LoadJSONBytes([]byte(raw))
	if err != nil {
		t.Fatalf("LoadJSONBytes: %v", err)
	}
	return tk
}

// TestTemplate_StopIDs: StopIDs is the end-of-sequence and end-of-turn ids the tokenizer has, then each stop string that is
// one token, each id once and in that order; a marker the vocabulary lacks contributes nothing.
func TestTemplate_StopIDs(t *testing.T) {
	tk := stopIDsTokenizer(t)
	sp := tk.Special()
	if sp.EOS != -1 || sp.EndOfTurn != 3 {
		t.Fatalf("VOID: the fixture's EOS is %d and end of turn %d, want -1 and 3", sp.EOS, sp.EndOfTurn)
	}
	// ChatML's stop is <|im_end|>, which is also the end-of-turn id: it appears once.
	if got := ChatML().StopIDs(tk); !slices.Equal(got, []int{3}) {
		t.Errorf("ChatML: %v, want [3]", got)
	}
	// Llama 3's stops are markers this vocabulary has (in its order) plus any it lacks, which drop out.
	want := []int{3}
	for _, s := range Llama3().Stops().Strings {
		if id, ok := tk.TokenID(s); ok && !slices.Contains(want, id) {
			want = append(want, id)
		}
	}
	if len(want) < 2 {
		t.Fatalf("VOID: none of Llama 3's stops %v is in the fixture's vocabulary", Llama3().Stops().Strings)
	}
	if got := Llama3().StopIDs(tk); !slices.Equal(got, want) {
		t.Errorf("Llama 3: %v, want %v", got, want)
	}
}
