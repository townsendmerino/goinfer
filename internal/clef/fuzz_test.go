package clef

import (
	"bytes"
	"strings"
	"testing"
)

// parse takes a request body straight from the client, and its hand parser recurses, so it relies on encoding/json's depth limit
// having refused anything deeper first: an overflowed stack is fatal, not a recoverable panic. Each document is balanced, so only
// that limit can refuse it, and each fits under the default request cap.
func TestParse_refusesNestingDeepEnoughToOverflowTheStack(t *testing.T) {
	const depth = 2 << 20
	for name, doc := range map[string]string{
		"array":  strings.Repeat("[", depth) + strings.Repeat("]", depth),
		"object": strings.Repeat(`{"a":`, depth) + "1" + strings.Repeat("}", depth),
	} {
		if _, err := parse([]byte(doc)); err == nil {
			t.Errorf("a %s nested %d deep was accepted", name, depth)
		}
	}
}

// FuzzParse holds the client's request through parse and Encode. Encode's tokenizer is a stand-in: the walk over the request is what is under test.
func FuzzParse(f *testing.F) {
	for _, s := range []string{
		`{"state":"s","questions":{"q":{"type":"choice","instructions":"i","criteria":{"a":"x","b":"y"}}}}`,
		`{"a":1,"a":2,"b":[1,2.5e3,true,false,null,"é🦊"]}`,
		`[]`, `{}`, `""`, `0`, `-0.0`, `1e400`, `"\ud800"`, ` `, `{`, "\xff",
	} {
		f.Add([]byte(s))
	}
	tok := func(text string, _ bool) ([]int, error) { return make([]int, min(len(text), 8)), nil }
	f.Fuzz(func(t *testing.T, data []byte) {
		if _, err := parse(data); err == nil && len(bytes.TrimSpace(data)) == 0 {
			t.Fatal("an empty document parsed")
		}
		_, _ = Encode(data, tok, 64)
	})
}
