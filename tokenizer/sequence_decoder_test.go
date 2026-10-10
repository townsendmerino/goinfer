package tokenizer

import (
	"encoding/json"
	"slices"
	"testing"
)

// A tokenizer.json whose ByteLevel decoder is wrapped in a Sequence (LFM2-VL's, S10 of
// docs/tasks/task-multimodal-support-2026-10.md) loads as byte-level and encodes exactly as the bare form; a Sequence with
// anything else in it keeps the SentencePiece routing. Before the fix the wrapped form fell to the SentencePiece path and
// refused to load ("required token <unk>"), so LFM2.5-VL could not tokenize at all.
func TestLoad_sequenceByteLevelDecoder(t *testing.T) {
	enc, _ := buildByteLevelTables()
	vocab := map[string]int{}
	for b := range 256 {
		vocab[string(enc[b])] = b
	}
	sp := string(enc[' '])
	vocab["he"], vocab["ll"], vocab["hell"], vocab[sp+"w"] = 256, 257, 258, 259
	merges := [][2]string{{"h", "e"}, {"l", "l"}, {"he", "ll"}, {sp, "w"}}
	build := func(decoder any) []byte {
		raw, err := json.Marshal(map[string]any{
			"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": merges},
			"pre_tokenizer": map[string]any{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
			"decoder":       decoder,
		})
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	bl := map[string]any{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}
	bare, err := LoadJSONBytes(build(bl))
	if err != nil {
		t.Fatalf("bare ByteLevel: %v", err)
	}
	wrapped, err := LoadJSONBytes(build(map[string]any{"type": "Sequence", "decoders": []any{bl}}))
	if err != nil {
		t.Fatalf("Sequence[ByteLevel]: %v", err)
	}
	if bare.mode != modeByteLevel || wrapped.mode != modeByteLevel {
		t.Fatalf("modes %d and %d, want both modeByteLevel", bare.mode, wrapped.mode)
	}
	for _, s := range []string{"hello world", "hell", " w\n\t x"} {
		a, err1 := bare.Encode(s, false)
		b, err2 := wrapped.Encode(s, false)
		if err1 != nil || err2 != nil || !slices.Equal(a, b) {
			t.Errorf("%q: bare %v (%v), wrapped %v (%v)", s, a, err1, b, err2)
		}
	}
	// A Sequence with another member is not taken for byte-level.
	tj := &tokenizerJSON{}
	if err := json.Unmarshal(build(map[string]any{"type": "Sequence", "decoders": []any{bl, map[string]any{"type": "Strip"}}}), tj); err != nil {
		t.Fatal(err)
	}
	if byteLevelDecoder(tj) {
		t.Error("Sequence[ByteLevel, Strip] was taken for byte-level")
	}
}
