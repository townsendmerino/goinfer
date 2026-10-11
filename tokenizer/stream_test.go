package tokenizer

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// streamAll feeds ids through a Stream and returns every Write's text, in order, and the Flush.
func streamAll(t *testing.T, tk *Tokenizer, ids []int) (writes []string, flush string) {
	t.Helper()
	s := tk.NewStream()
	for _, id := range ids {
		w, err := s.Write(id)
		if err != nil {
			t.Fatalf("Write(%d): %v", id, err)
		}
		writes = append(writes, w)
	}
	return writes, s.Flush()
}

// spmStreamTokenizer is a SentencePiece-shaped tokenizer with the dummy-prefix strip on: whole words with the ▁ marker, and
// byte-fallback pieces for the four bytes of 🦄 and the three of €.
func spmStreamTokenizer() (*Tokenizer, map[string]int) {
	pieces := []string{"▁The", "▁capital", "▁of", "▁France", "▁is", "▁Paris", ".", "▁", "ok"}
	ids := map[string]int{}
	for i, p := range pieces {
		ids[p] = i
	}
	byteToVal := map[int32]byte{}
	for _, b := range append([]byte("🦄"), []byte("€")...) {
		byteToVal[int32(len(pieces))] = b
		pieces = append(pieces, "")
	}
	return &Tokenizer{idToPiece: pieces, byteToVal: byteToVal, stripLeadingSpace: true}, ids
}

// The streamed text equals the whole-sequence Decode, the first space is stripped once, and per-id Decode (the mistake the
// docs' smallest program used to make) is shown to differ.
func TestStream_equalsDecodeAndKeepsSpaces(t *testing.T) {
	tk, id := spmStreamTokenizer()
	ids := []int{id["▁The"], id["▁capital"], id["▁of"], id["▁France"], id["▁is"], id["▁Paris"], id["."]}
	want, err := tk.Decode(ids)
	if err != nil || want != "The capital of France is Paris." {
		t.Fatalf("Decode = %q, %v", want, err)
	}
	writes, flush := streamAll(t, tk, ids)
	if got := strings.Join(writes, "") + flush; got != want {
		t.Fatalf("stream = %q, want %q", got, want)
	}
	var perID strings.Builder
	for _, i := range ids {
		s, _ := tk.Decode([]int{i})
		perID.WriteString(s)
	}
	if perID.String() == want {
		t.Fatalf("per-id Decode matched the whole decode (%q); the premise of Stream is gone", perID.String())
	}
}

// A character spread over byte-fallback tokens is held until its last byte arrives: no Write returns invalid UTF-8, the
// 🦄 appears whole in one Write, and the pieces on both sides are kept.
func TestStream_holdsPartialCharacters(t *testing.T) {
	tk, id := spmStreamTokenizer()
	n := len(tk.idToPiece)
	uni := []int{n - 7, n - 6, n - 5, n - 4} // 🦄
	euro := []int{n - 3, n - 2, n - 1}       // €
	ids := append(append(append([]int{id["▁The"]}, uni...), id["▁is"]), euro...)
	want, _ := tk.Decode(ids)
	writes, flush := streamAll(t, tk, ids)
	for i, w := range writes {
		if !utf8.ValidString(w) {
			t.Fatalf("Write %d returned invalid UTF-8 %q", i, w)
		}
	}
	if flush != "" {
		t.Fatalf("Flush = %q, want empty: every character was completed", flush)
	}
	if got := strings.Join(writes, ""); got != want || !strings.Contains(got, "🦄") || !strings.Contains(got, "€") {
		t.Fatalf("stream = %q, want %q", got, want)
	}
	for i := range uni[:3] {
		if writes[1+i] != "" {
			t.Fatalf("byte %d of 🦄 was written early: %q", i, writes[1+i])
		}
	}
	if writes[4] != "🦄" {
		t.Fatalf("the last byte of 🦄 wrote %q, want the whole character", writes[4])
	}
}

// A generation that ends inside a character: Flush returns the held bytes, as the whole-sequence Decode writes them.
func TestStream_flushReturnsATruncatedCharacter(t *testing.T) {
	tk, id := spmStreamTokenizer()
	n := len(tk.idToPiece)
	ids := []int{id["▁The"], n - 7, n - 6}
	want, _ := tk.Decode(ids)
	writes, flush := streamAll(t, tk, ids)
	if got := strings.Join(writes, "") + flush; got != want {
		t.Fatalf("stream+flush = %q, want %q", got, want)
	}
	if flush == "" {
		t.Fatal("Flush returned nothing for a generation that ended mid-character")
	}
}

// The strip is spent by the first text, not by the first id: a leading id that renders nothing does not make the next
// "▁word" keep a space the whole-sequence decode removes, and a second sequence's Stream strips again.
func TestStream_stripsTheSequenceSpaceOnce(t *testing.T) {
	tk, id := spmStreamTokenizer()
	ids := []int{id["▁"], id["▁of"], id["▁is"]}
	want, _ := tk.Decode(ids)
	writes, flush := streamAll(t, tk, ids)
	if got := strings.Join(writes, "") + flush; got != want {
		t.Fatalf("stream = %q, want %q", got, want)
	}
	if again, _ := streamAll(t, tk, ids); strings.Join(again, "") != strings.Join(writes, "") {
		t.Fatal("a second Stream did not strip its own leading space")
	}
}

func TestStream_outOfRangeIDIsAnError(t *testing.T) {
	tk, _ := spmStreamTokenizer()
	s := tk.NewStream()
	if _, err := s.Write(10_000); err == nil {
		t.Fatal("an id outside the vocabulary must be an error")
	}
	if _, err := s.Write(-1); err == nil {
		t.Fatal("a negative id must be an error")
	}
}

// byteLevelStreamTokenizer is a byte-level (GPT-2 style) tokenizer whose pieces spell 🦄's bytes through the byte table, one per token.
func byteLevelStreamTokenizer() (*Tokenizer, []int, []int) {
	dec := map[rune]byte{}
	for b := 0; b < 256; b++ {
		dec[rune(0x100+b)] = byte(b)
	}
	for _, r := range "abcdefghijklmnopqrstuvwxyz" {
		dec[r] = byte(r)
	}
	pieces := []string{"hello", "Ġworld"} // Ġ is 0x100+0x20: a space
	dec['Ġ'] = ' '
	for _, b := range []byte("🦄") {
		pieces = append(pieces, string(rune(0x100+int(b))))
	}
	return &Tokenizer{mode: modeByteLevel, idToPiece: pieces, byteDecoder: dec}, []int{0, 1}, []int{2, 3, 4, 5}
}

func TestStream_byteLevelHoldsPartialCharacters(t *testing.T) {
	tk, words, uni := byteLevelStreamTokenizer()
	ids := append(append([]int{}, words...), uni...)
	want, err := tk.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	writes, flush := streamAll(t, tk, ids)
	for i, w := range writes {
		if !utf8.ValidString(w) {
			t.Fatalf("Write %d returned invalid UTF-8 %q", i, w)
		}
	}
	if got := strings.Join(writes, "") + flush; got != want {
		t.Fatalf("stream = %q, want %q", got, want)
	}
	if !strings.HasSuffix(want, "🦄") {
		t.Fatalf("test setup: %q does not end with the emoji", want)
	}
}
