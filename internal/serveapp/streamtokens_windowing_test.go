package serveapp

import (
	"context"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// streamTokensTestTokenizer builds a tiny, fully-controlled byte-level tokenizer via
// tokenizer.LoadJSONBytes (the exported, in-memory loader) — internal/serveapp cannot reach
// tokenizer.Tokenizer's unexported fields directly, and no real on-disk tokenizer with a known,
// hand-picked vocabulary is available to this package. Byte-level mode's decode has no
// cross-token state at all (unlike SentencePiece's byte-fallback fusion, proven separately by
// tokenizer.TestDecodeContinuation_isIncrementallyAssociative), so it is sufficient here: this
// test's job is the STOP-STRING WINDOWING half of the R-08 fix, not the decode half.
func streamTokensTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	const raw = `{
		"model": {"type": "BPE", "vocab": {"a":0,"b":1,"c":2,"EN":3,"D":4,"ST":5,"OP":6,"E":7,"N":8,"O":9}, "merges": []},
		"decoder": {"type": "ByteLevel"}
	}`
	tk, err := tokenizer.LoadJSONBytes([]byte(raw))
	if err != nil {
		t.Fatalf("LoadJSONBytes: %v", err)
	}
	return tk
}

// referenceStreamTokens is streamTokens's PRE-R-08 algorithm: full DecodeContinuation and a
// full-text firstStop/completeUTF8/stopTailHold scan every token, no windowing. Kept here,
// deliberately, as the reference streamTokens is checked against — not read from git history,
// so this differential test still means something after the optimized version is the only one
// left in openai.go.
func referenceStreamTokens(tk *tokenizer.Tokenizer, ids []int, stops []string) (emitted, stopHit string) {
	var text string
	printed := 0
	var out []byte
	for i := range ids {
		full, _ := tk.DecodeContinuation(ids[:i+1])
		text = full
		if cut, which, hit := firstStop(text, stops); hit {
			if cut > printed {
				out = append(out, text[printed:cut]...)
			}
			return string(out), which
		}
		end := completeUTF8(text)
		if safe := len(text) - stopTailHold(text, stops); safe < end {
			end = safe
		}
		if end > printed {
			out = append(out, text[printed:end]...)
			printed = end
		}
	}
	if len(text) > printed {
		out = append(out, text[printed:]...)
	}
	return string(out), ""
}

// TestStreamTokens_windowedScanMatchesFullRescan is audit R-08's differential gate: the fix
// bounds firstStop/completeUTF8/stopTailHold to text[printed:] instead of the whole accumulated
// text (see streamTokens's own comment on why text[:printed] can never contain, or be the start
// of, a stop match). That argument is checked here empirically, the way this repo's own rules
// ask for a subtle invariant to be checked, across many random token streams and stop-string
// configurations designed to produce split matches (a stop spelled across two tokens: "EN"+"D"),
// near-misses that must NOT fire ("EN" alone, or "EN"+"O" which is not "END"), and matches that
// start right at a token boundary a naive window could clip.
func TestStreamTokens_windowedScanMatchesFullRescan(t *testing.T) {
	tk := streamTokensTestTokenizer(t)
	lm := &loadedModel{tk: tk}
	palette := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} // a b c EN D ST OP E N O
	stopSets := [][]string{
		{"END"},
		{"STOP"},
		{"END", "STOP"},
		{"ENDO"}, // spans three tokens (E/EN + N + D + O combinations) — a longer split match
	}

	rng := rand.New(rand.NewSource(1))
	const trials = 300
	for trial := range trials {
		stops := stopSets[trial%len(stopSets)]
		n := 5 + rng.Intn(30)
		ids := make([]int, n)
		for i := range ids {
			ids[i] = palette[rng.Intn(len(palette))]
		}

		wantEmit, wantStopHit := referenceStreamTokens(tk, ids, stops)

		stream := make(chan int, len(ids))
		for _, id := range ids {
			stream <- id
		}
		close(stream)
		var got []byte
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		gr := genRequest{maxTokens: len(ids) + 1000, stopStrings: stops}
		_, _, gotStopHit := lm.streamTokens(ctx, cancel, stream, gr, nil, func(s string) {
			got = append(got, s...)
		})

		if string(got) != wantEmit {
			t.Fatalf("trial %d (ids=%v stops=%v): streamTokens emitted %q, reference emitted %q",
				trial, ids, stops, string(got), wantEmit)
		}
		if gotStopHit != wantStopHit {
			t.Fatalf("trial %d (ids=%v stops=%v): streamTokens stopHit=%q, reference stopHit=%q",
				trial, ids, stops, gotStopHit, wantStopHit)
		}
	}
}
