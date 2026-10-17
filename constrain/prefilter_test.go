package constrain

import (
	"math"
	"testing"
)

// TestMasker_firstBytePrefilterExact is the correctness gate for nextBytes' prefilter (the first byte, and the second
// at structural positions): at every state each grammar passes through on its way through a set of documents (every
// byte prefix, the complete document included), the mask with the prefilter must equal the mask from the full grammar
// walk, for every id of a vocabulary made of the adversarial set plus every 1-4 byte substring of the documents. The
// grammars are the four kinds: free JSON, a schema, one tool, and the union over the 19 agent tools.
func TestMasker_firstBytePrefilterExact(t *testing.T) {
	schema, err := JSONSchema([]byte(`{"type":"object","properties":{
		"kind":{"enum":["a","bc"]},"n":{"type":"integer"},"x":{"type":"number"},"ok":{"type":"boolean"},
		"tags":{"type":"array","items":{"type":"string"}},"inner":{"type":"object","properties":{"note":{"type":"string"}}}},
		"required":["kind","n"]}`))
	if err != nil {
		t.Fatal(err)
	}
	one, err := ToolCallGrammar("<tool_call>\n", "\n</tool_call>", "arguments", "search", false,
		[]byte(`{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}`))
	if err != nil {
		t.Fatal(err)
	}
	union, err := ToolCallsGrammar("<tool_call>\n", "\n</tool_call>", "arguments", false, opencodeKenTools(t))
	if err != nil {
		t.Fatal(err)
	}
	call := func(body string) string { return "<tool_call>\n" + body + "\n</tool_call>" }
	cases := []struct {
		name string
		g    Grammar
		docs []string
	}{
		{"json", JSON(), []string{`{"a": [1, -2.5e3, true, null, "x\"yé\\"], "b": {}}`, `[]`, `"s"`, `-0.1`}},
		{"schema", schema, []string{
			`{"kind": "bc", "n": -12, "x": 1.5e-2, "ok": false, "tags": ["p", "q\n"], "inner": {"note": "é"}}`,
			`{"n":3,"kind":"a"}`,
		}},
		{"tool", one, []string{call(`{"name": "search", "arguments": {"query": "find \"x\"", "limit": 20}}`)}},
		{"union", union, []string{
			call(`{"name": "read", "arguments": {"filePath": "/repo/cmd/serve/main.go", "limit": 2000}}`),
			call(`{"name": "bash", "arguments": {"command": "go test ./...", "timeout": 120000, "workdir": "/repo"}}`),
			call(`{"name": "ken_search", "arguments": {"query": "prepare", "languages": ["go"]}}`),
		}},
	}

	vocab := buildAdversarialVocab()
	seen := map[string]bool{}
	for _, c := range cases {
		for _, d := range c.docs {
			for i := range d {
				for n := 1; n <= 4 && i+n <= len(d); n++ {
					if s := d[i : i+n]; !seen[s] {
						seen[s] = true
						vocab = append(vocab, []byte(s))
					}
				}
			}
		}
	}
	eos := []int{0}

	for _, c := range cases {
		fast := NewMasker(c.g.Clone(), vocab, eos).StopWhenComplete()
		slow := NewMasker(c.g.Clone(), vocab, eos).StopWhenComplete()
		slow.walkAll = true
		states, prefiltered := 0, 0
		for _, d := range c.docs {
			g := c.g.Clone()
			g.Reset()
			for i := 0; i <= len(d); i++ {
				if i > 0 {
					b := []byte{d[i-1]}
					if !g.TryBytes(b) {
						t.Fatalf("%s: the grammar refuses its own document %q at byte %d", c.name, d, i-1)
					}
					g.Commit(b)
				}
				lf := make([]float32, len(vocab))
				ls := make([]float32, len(vocab))
				fast.MaskAt(g, lf)
				slow.MaskAt(g, ls)
				var next nextTable
				fast.nextBytes(g, &next)
				for id, tok := range vocab {
					mf, ms := math.IsInf(float64(lf[id]), -1), math.IsInf(float64(ls[id]), -1)
					if mf != ms {
						t.Fatalf("%s, after %q: id %d (%q): prefiltered mask says masked=%v, full walk says masked=%v",
							c.name, d[:i], id, tok, mf, ms)
					}
					if len(tok) > 0 && !next.admits(tok) && id != 0 {
						prefiltered++
					}
				}
				states++
			}
		}
		if prefiltered == 0 {
			t.Fatalf("%s: the prefilter rejected nothing over %d states, so this case checks nothing", c.name, states)
		}
		t.Logf("%s: %d states, %d id-states decided by the table alone, of %d", c.name, states, prefiltered, states*len(vocab))
	}
}
