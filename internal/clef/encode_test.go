package clef

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// D12's encoder gate (docs/tasks/task-constrained-confidence.md): the prompt text, the token ids and every span identical to the
// official encoder's on all 150 recorded items (testdata/decisions/clef/encoder.jsonl, pinned by scripts/pin_clef_d10.py from the
// checkpoint's own joint_schema_model.py, with its tokenizer calls instrumented).
//
// THE REPLAY TOKENIZER. The dump records, per item, the full input_ids and every fragment the reference tokenized (text and token
// count), in call order. The fragments partition input_ids as prefix + state + schema + suffix, so each fragment's ids can be sliced
// back out of input_ids. TestEncode_matchesReference replays them: the Go encoder must call the tokenizer with exactly the recorded
// texts in exactly the recorded order, and what it assembles must equal the recorded input_ids and spans. That proves the PROMPT
// STRING and the assembly with no tokenizer asset, so it runs on every checkout; the tokenizer itself is TestEncode_realTokenizer's.

type dumpItem struct {
	ID        string `json:"id"`
	InputIDs  []int  `json:"input_ids"`
	NTokens   int    `json:"n_tokens"`
	Fragments []struct {
		Text    string `json:"text"`
		NTokens int    `json:"n_tokens"`
	} `json:"fragments"`
	Questions []struct {
		ID          string   `json:"id"`
		OptionIDs   []string `json:"option_ids"`
		OptionSpans [][2]int `json:"option_spans"`
		Span        [2]int   `json:"question_span"`
		Type        int      `json:"type"`
	} `json:"questions"`
}

type recordItem struct {
	ID      string          `json:"id"`
	Request json.RawMessage `json:"request"`
}

func readJSONL[T any](t *testing.T, path string) []T {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("no fixture (a skip would hide this; scripts/pin_clef_d10.py): %v", err)
	}
	defer f.Close()
	var out []T
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		var v T
		if err := json.Unmarshal(sc.Bytes(), &v); err != nil {
			t.Fatal(err)
		}
		out = append(out, v)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func loadDump(t *testing.T) (map[string]dumpItem, []recordItem) {
	t.Helper()
	dumps := map[string]dumpItem{}
	for _, d := range readJSONL[dumpItem](t, "../../testdata/decisions/clef/encoder.jsonl") {
		dumps[d.ID] = d
	}
	recs := readJSONL[recordItem](t, "../../testdata/decisions/clef/records.jsonl")
	if len(recs) != 150 || len(dumps) != 150 {
		t.Fatalf("fixture has %d records and %d dumps, want 150 each", len(recs), len(dumps))
	}
	return dumps, recs
}

// replayTokenizer returns each fragment's recorded ids, asserting the text it is asked for is the recorded one. The reference calls the
// tokenizer for the schema fragments, then the prefix, the suffix, and the state; the ids come from input_ids at the matching offsets.
func replayTokenizer(t *testing.T, d dumpItem) (Tokenize, func() int) {
	t.Helper()
	nFrag := len(d.Fragments)
	var prefixN, suffixN, stateN, schemaN int
	// The last three calls are prefix, suffix, state; everything before them is schema.
	prefixN, suffixN, stateN = d.Fragments[nFrag-3].NTokens, d.Fragments[nFrag-2].NTokens, d.Fragments[nFrag-1].NTokens
	for _, f := range d.Fragments[:nFrag-3] {
		schemaN += f.NTokens
	}
	if prefixN+stateN+schemaN+suffixN != len(d.InputIDs) {
		t.Fatalf("%s: fragments sum to %d tokens, input_ids has %d", d.ID, prefixN+stateN+schemaN+suffixN, len(d.InputIDs))
	}
	prefixIDs := d.InputIDs[:prefixN]
	stateIDs := d.InputIDs[prefixN : prefixN+stateN]
	schemaIDs := d.InputIDs[prefixN+stateN : prefixN+stateN+schemaN]
	suffixIDs := d.InputIDs[prefixN+stateN+schemaN:]
	call, schemaOff := 0, 0
	tok := func(text string, parseSpecial bool) ([]int, error) {
		if call >= nFrag {
			return nil, fmt.Errorf("%s: the encoder tokenized more fragments than the reference (%d)", d.ID, nFrag)
		}
		want := d.Fragments[call]
		if text != want.Text {
			return nil, fmt.Errorf("%s: fragment %d is %q, the reference tokenized %q", d.ID, call, text, want.Text)
		}
		// Only the prefix and the suffix (the last-but-two and last-but-one calls) carry chat markers and may parse special tokens.
		if wantSpecial := call == nFrag-3 || call == nFrag-2; parseSpecial != wantSpecial {
			return nil, fmt.Errorf("%s: fragment %d parseSpecial=%v, want %v", d.ID, call, parseSpecial, wantSpecial)
		}
		var ids []int
		switch {
		case call == nFrag-3:
			ids = prefixIDs
		case call == nFrag-2:
			ids = suffixIDs
		case call == nFrag-1:
			ids = stateIDs
		default:
			ids = schemaIDs[schemaOff : schemaOff+want.NTokens]
			schemaOff += want.NTokens
		}
		call++
		return append([]int(nil), ids...), nil
	}
	return tok, func() int { return call }
}

func TestEncode_matchesReference(t *testing.T) {
	dumps, recs := loadDump(t)
	var nTok, nQ int
	kinds := map[int]int{}
	for _, r := range recs {
		d, ok := dumps[r.ID]
		if !ok {
			t.Fatalf("record %s has no dump", r.ID)
		}
		tok, calls := replayTokenizer(t, d)
		got, err := Encode(r.Request, tok, 0)
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		if calls() != len(d.Fragments) {
			t.Errorf("%s: tokenized %d fragments, the reference %d", r.ID, calls(), len(d.Fragments))
		}
		if len(got.InputIDs) != d.NTokens {
			t.Errorf("%s: %d tokens, want %d", r.ID, len(got.InputIDs), d.NTokens)
		}
		for i := range d.InputIDs {
			if i >= len(got.InputIDs) || got.InputIDs[i] != d.InputIDs[i] {
				t.Errorf("%s: input_ids differ at %d", r.ID, i)
				break
			}
		}
		if len(got.Questions) != len(d.Questions) {
			t.Fatalf("%s: %d questions, want %d", r.ID, len(got.Questions), len(d.Questions))
		}
		for i, w := range d.Questions {
			g := got.Questions[i]
			if g.ID != w.ID || g.Type != w.Type || g.QuestionSpan != w.Span || fmt.Sprint(g.OptionIDs) != fmt.Sprint(w.OptionIDs) || fmt.Sprint(g.OptionSpans) != fmt.Sprint(w.OptionSpans) {
				t.Errorf("%s question %d: got %+v, want %+v", r.ID, i, g, w)
			}
			kinds[g.Type]++
		}
		nTok += len(got.InputIDs)
		nQ += len(got.Questions)
	}
	t.Logf("150 items, %d questions, %d tokens: prompt text, ids and spans identical to the reference (noul %d, choice %d, score %d)", nQ, nTok, kinds[TypeNoul], kinds[TypeChoice], kinds[TypeScore])
	if nTok != 44823 {
		t.Errorf("the fixture's token total is %d, the D10 record says 44,823", nTok)
	}
	if kinds[TypeNoul] == 0 || kinds[TypeChoice] == 0 || kinds[TypeScore] == 0 {
		t.Errorf("a question type is missing from the fixture: %v", kinds)
	}
}

// Able to fail: a wrong rendering of ANY fragment must turn the replay red, or the test above only checks counts. Four mutations of the
// request, each of which changes a fragment's text the reference would not have produced.
func TestEncode_replayCatchesAWrongPrompt(t *testing.T) {
	dumps, recs := loadDump(t)
	r := recs[0]
	d := dumps[r.ID]
	for name, mutate := range map[string]func(string) string{
		"the state loses a character": func(s string) string { return strings.Replace(s, `"account"`, `"acount"`, 1) },
		"instructions altered":        func(s string) string { return strings.Replace(s, "Apply this", "Apply That", 1) },
		"the type is renamed":         func(s string) string { return strings.Replace(s, `"type":"noul"`, `"type":"score"`, 1) },
		"the question id changes":     func(s string) string { return strings.Replace(s, `"q":`, `"r":`, 1) },
	} {
		mutated := mutate(string(r.Request))
		if mutated == string(r.Request) {
			// The record is written with ", " separators by the pin script, so retry against the compact form.
			var v any
			if err := json.Unmarshal(r.Request, &v); err != nil {
				t.Fatal(err)
			}
			b, _ := json.Marshal(v)
			mutated = mutate(string(b))
		}
		if mutated == string(r.Request) {
			t.Fatalf("%s: the mutation did not change the request, so this proves nothing", name)
		}
		tok, _ := replayTokenizer(t, d)
		if _, err := Encode([]byte(mutated), tok, 0); err == nil {
			t.Errorf("%s: the replay accepted a prompt the reference did not build", name)
		}
	}
}
