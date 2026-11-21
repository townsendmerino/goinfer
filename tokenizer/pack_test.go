package tokenizer

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

// packFixture is a small SentencePiece-style byte-fallback BPE tokenizer.json: the 256 byte pieces, a few words, a
// merge whose right piece is outside the vocabulary (mergeTable.byStr), a merge listed twice (the later rank wins), and
// added tokens, one of them rstrip.
func packFixture(t *testing.T) []byte {
	t.Helper()
	vocab := map[string]int{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<|turn>": 5, "<turn|>": 6}
	for b := range 256 {
		vocab[fmt.Sprintf("<0x%02X>", b)] = 7 + b
	}
	next := 7 + 256
	words := []string{"▁", "a", "b", "c", "é", "ab", "abc", "▁a", "▁ab", "▁abc", "ca", "bé", "▁\n", "\n", "\n\n", "x", "y", "xy"}
	for _, w := range words {
		vocab[w] = next
		next++
	}
	merges := [][2]string{{"a", "b"}, {"ab", "c"}, {"▁", "a"}, {"▁a", "b"}, {"▁ab", "c"}, {"c", "a"}, {"b", "é"},
		{"x", "y"}, {"y", "zz"}, {"a", "b"}, {"\n", "\n"}}
	added := []map[string]any{
		{"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}, {"id": 2, "content": "<bos>"},
		{"id": 3, "content": "<unk>"}, {"id": 4, "content": "<mask>"},
		{"id": 5, "content": "<|turn>", "rstrip": true}, {"id": 6, "content": "<turn|>"},
		{"id": next, "content": "<extra_only_added>"},
	}
	doc := map[string]any{
		"added_tokens": added,
		"model":        map[string]any{"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": vocab, "merges": merges},
		"normalizer":   map[string]any{"type": "Replace", "pattern": map[string]any{"String": " "}, "content": "▁"},
		"decoder":      map[string]any{"type": "Sequence", "decoders": []any{map[string]any{"type": "ByteFallback"}}},
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// packCorpus is random text over the fixture's alphabet, with added-token surfaces, spaces, newlines and runes outside
// the vocabulary, so byte fallback, the trie and rstrip all run.
func packCorpus(n int, seed int64) []string {
	rng := rand.New(rand.NewSource(seed))
	parts := []string{"a", "b", "c", "é", "x", "y", "z", " ", "\n", "\t", "ab", "abc", "<|turn>", "<turn|>", "<eos>",
		"<bos", "日本", "🙂", "\u00a0", "<extra_only_added>", "  "}
	out := make([]string, n)
	for i := range out {
		var b strings.Builder
		for range rng.Intn(24) {
			b.WriteString(parts[rng.Intn(len(parts))])
		}
		out[i] = b.String()
	}
	return out
}

// TestPack_equalsJSON: a tokenizer loaded from its pack has the same tables as one parsed from the tokenizer.json, and
// encodes and decodes every corpus text the same, with and without special-token parsing.
func TestPack_equalsJSON(t *testing.T) {
	raw := packFixture(t)
	want, err := LoadJSONBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	pack, err := BuildPack(raw)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadPack(pack)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []struct {
		name     string
		got, exp any
	}{
		{"vocab", got.vocab, want.vocab}, {"idToPiece", got.idToPiece, want.idToPiece},
		{"merges", got.merges, want.merges}, {"special", got.special, want.special},
		{"byteFallback", got.byteFallback, want.byteFallback}, {"bytePiece", got.bytePiece, want.bytePiece},
		{"byteToVal", got.byteToVal, want.byteToVal}, {"unkPiece", got.unkPiece, want.unkPiece},
		{"unkID", got.unkID, want.unkID}, {"prependSpace", got.prependSpace, want.prependSpace},
		{"stripLeadingSpace", got.stripLeadingSpace, want.stripLeadingSpace}, {"ignoreMerges", got.ignoreMerges, want.ignoreMerges},
		{"added", got.added, want.added}, {"rstrip", got.rstrip, want.rstrip}, {"isAdded", got.isAdded, want.isAdded},
		{"mode", got.mode, want.mode}, {"scoreRank", got.scoreRank, want.scoreRank},
	} {
		if !reflect.DeepEqual(f.got, f.exp) {
			t.Errorf("%s differs: pack %v, json %v", f.name, f.got, f.exp)
		}
	}
	if len(want.merges.byStr) == 0 {
		t.Fatal("fixture has no merge outside the vocabulary: the byStr path is not exercised")
	}
	for _, s := range packCorpus(2000, 1) {
		for _, special := range []bool{true, false} {
			a, errA := want.encode(s, true, special)
			b, errB := got.encode(s, true, special)
			if errA != nil || errB != nil || !reflect.DeepEqual(a, b) {
				t.Fatalf("encode(%q, special=%v): json %v %v, pack %v %v", s, special, a, errA, b, errB)
			}
			da, _ := want.Decode(a)
			db, _ := got.Decode(b)
			if da != db {
				t.Fatalf("Decode(%v): json %q, pack %q", a, da, db)
			}
		}
	}
}

// TestBuildPack_deterministic: the same tokenizer.json always gives the same pack bytes, so a pack's digest can be pinned.
func TestBuildPack_deterministic(t *testing.T) {
	raw := packFixture(t)
	a, err := BuildPack(raw)
	if err != nil {
		t.Fatal(err)
	}
	for range 5 {
		b, err := BuildPack(raw)
		if err != nil {
			t.Fatal(err)
		}
		if sha256.Sum256(a) != sha256.Sum256(b) {
			t.Fatal("BuildPack output differs between runs")
		}
	}
}

// TestBuildPack_refusesOtherFamilies: only the SentencePiece-style BPE family is packed.
func TestBuildPack_refusesOtherFamilies(t *testing.T) {
	byteLevel := `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"decoder":{"type":"ByteLevel"}}`
	if _, err := BuildPack([]byte(byteLevel)); err == nil {
		t.Fatal("BuildPack accepted a byte-level tokenizer")
	}
}

// reseal recomputes a pack's trailing checksum after a test edits its body, so the structural checks behind the
// checksum are reached.
func reseal(p []byte) []byte {
	body := p[:len(p)-sha256.Size]
	sum := sha256.Sum256(body)
	return append(append([]byte(nil), body...), sum[:]...)
}

// TestLoadPack_refusesDamage: a truncated pack, a flipped bit, and structural damage behind a valid checksum (counts
// past the end, offsets out of order, an id out of range, trailing bytes) are each an ErrPack error, never a panic.
func TestLoadPack_refusesDamage(t *testing.T) {
	pack, err := BuildPack(packFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	try := func(name string, p []byte) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("%s: LoadPack panicked: %v", name, r)
			}
		}()
		if _, err := LoadPack(p); !errors.Is(err, ErrPack) {
			t.Errorf("%s: err = %v, want ErrPack", name, err)
		}
	}
	for n := 0; n < len(pack); n += 97 {
		try(fmt.Sprintf("truncated to %d", n), pack[:n])
	}
	for i := 0; i < len(pack); i += 131 {
		p := append([]byte(nil), pack...)
		p[i] ^= 0x10
		try(fmt.Sprintf("bit flipped at %d", i), p)
	}
	header := len(packMagic) + 8 + 24 // magic, version, flags, special tokens
	set := func(off int, v uint32) []byte {
		p := append([]byte(nil), pack...)
		binary.LittleEndian.PutUint32(p[off:], v)
		return reseal(p)
	}
	try("string count huge", set(header, 1<<30))
	try("blob length past end", set(header+4, 1<<30))
	try("offset out of order", set(header+8+4, 1<<20))
	try("bad version", set(len(packMagic), 99))
	try("unk id out of range", set(len(packMagic)+8+20, 1<<20))
	body := pack[:len(pack)-sha256.Size]
	try("trailing bytes", reseal(append(append(append([]byte(nil), body...), 0, 0, 0, 0), make([]byte, sha256.Size)...)))
}

// FuzzLoadPack: no input makes LoadPack panic; whatever it accepts encodes without panicking.
func FuzzLoadPack(f *testing.F) {
	var tt testing.T
	pack, err := BuildPack(packFixture(&tt))
	if err != nil {
		f.Fatal(err)
	}
	f.Add(pack)
	f.Add(reseal(pack))
	f.Fuzz(func(t *testing.T, b []byte) {
		tk, err := LoadPack(b)
		if err != nil {
			return
		}
		_, _ = tk.Encode("ab c <|turn> é 日本", true)
		if len(b) > sha256.Size {
			_, _ = LoadPack(reseal(b))
		}
	})
}
