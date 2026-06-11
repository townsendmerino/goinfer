package constrain

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
)

// Track 2.1 (testing campaign): constrain is the only ATTACKER-SUPPLIED grammar
// surface — cmd/serve compiles a caller's response_format JSON Schema on every
// request via JSONSchema. The contract is the repo promise: a typed error or a
// clean compile, never a panic/hang, and — the structural property — if a schema
// compiles, the masker it produces must always be able to drive SOME complete,
// conforming document (it can never paint itself into a corner where no token,
// not even EOS, is legal). These fuzz targets enforce both.

// seedSchemas are valid + edge-case schemas planted in the corpus so the mutator
// starts from structurally interesting points rather than random bytes.
var seedSchemas = []string{
	`{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}`,
	`{"type":"object","additionalProperties":false,"properties":{"color":{"enum":["red","green","blue"]},"kind":{"const":"widget"}}}`,
	`{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}`,
	`{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}`,
	`{"type":"object","additionalProperties":false,"properties":{"u":{"type":"object","additionalProperties":false,"properties":{"id":{"type":"integer"}},"required":["id"]}},"required":["u"]}`,
	`{"type":"string"}`,
	`{"type":"number"}`,
	`{"type":"boolean"}`,
	`{"type":"null"}`,
	`{"const":"x"}`,
	`{"enum":[1,2,3]}`,
}

// FuzzJSONSchema fuzzes the full response_format path: compile an arbitrary
// schema, and if it compiles, drive the resulting Masker to a complete document
// and check the output actually conforms. `choices` steers which allowed token
// is taken at each step, so the mutator can explore distinct generation paths
// deterministically.
func FuzzJSONSchema(f *testing.F) {
	for _, s := range seedSchemas {
		f.Add([]byte(s), []byte{0, 1, 2, 3, 7, 11})
	}
	f.Fuzz(func(t *testing.T, schema, choices []byte) {
		g, err := JSONSchema(schema)
		if err != nil {
			return // a typed compile error is the correct outcome, not a bug
		}
		// Re-parse for the independent conformance oracle (schema compiled, so it
		// is within the supported subset `conforms` understands).
		var doc map[string]any
		if err := json.Unmarshal(schema, &doc); err != nil {
			t.Fatalf("schema compiled but no longer parses as object: %q (%v)", schema, err)
		}
		driveAndValidate(t, g, doc, schema, choices)
	})
}

// driveAndValidate walks the masked decode the way a sampler would: at each step
// it Processes the logits, takes one of the surviving tokens (chosen by the fuzz
// `choices` stream), and stops at EOS. The two invariants:
//   - never a dead-end: a non-complete document must always leave at least one
//     legal token (the masker masks EOS until CanEnd, so empty-allowed +
//     !CanEnd means the grammar accepted an unsatisfiable schema).
//   - the completed output parses and conforms to the schema.
func driveAndValidate(t *testing.T, g Grammar, doc map[string]any, schema, choices []byte) {
	t.Helper()
	// A COMPLETE byte vocabulary (every single byte + EOS) so the only possible
	// dead-end is a grammar that contradicts itself — never "the test vocab can't
	// spell a byte this schema legitimately requires" (e.g. a non-ASCII enum
	// literal). With a real tokenizer the vocab covers the bytes too.
	tokens, eos := fullByteVocab()
	m := NewMasker(g, tokens, []int{eos}).StopWhenComplete()
	neg := float32(math.Inf(-1))
	logits := make([]float32, len(tokens))
	var generated []int
	var progress []int // allowed tokens that actually advance the document (non-WS, non-EOS)
	var out bytes.Buffer
	ci := 0
	const budget = 2000 // large schemas may not finish; that is not a bug, just stop
	for range budget {
		for i := range logits {
			logits[i] = 0
		}
		m.Process(generated, logits)
		// Between JSON tokens, whitespace is optional — it never unblocks a state,
		// so it can't count as progress; a schema is satisfiable from such a state
		// only if some NON-whitespace token is legal (or the document can already
		// end). INSIDE a string/key/enum literal, though, a whitespace byte is
		// required content (e.g. a property named " ", or const "a b"), so it does
		// count. We read the grammar's top frame to tell the two apart precisely.
		inLiteral := topFrameInLiteral(g)
		progress = progress[:0]
		for id, v := range logits {
			if v <= neg || id == eos {
				continue
			}
			if !inLiteral && isWSByte(tokens[id]) {
				continue
			}
			progress = append(progress, id)
		}
		if m.CanEnd() {
			break // a complete, valid document is reachable here — stop and validate
		}
		if len(progress) == 0 {
			t.Fatalf("unsatisfiable: schema %q compiled but can never complete — no productive token after %q", schema, out.Bytes())
		}
		var c byte
		if len(choices) > 0 {
			c = choices[ci%len(choices)]
			ci++
		}
		pick := progress[int(c)%len(progress)]
		generated = append(generated, pick)
		out.Write(tokens[pick])
	}
	if !m.CanEnd() {
		return // ran out of budget mid-document — not a bug
	}
	// The masker reached a completion point: the bytes must be valid, conforming JSON.
	dec := json.NewDecoder(bytes.NewReader(out.Bytes()))
	dec.UseNumber() // any-magnitude integers/numbers are legal; don't overflow the oracle
	var val any
	if err := dec.Decode(&val); err != nil {
		t.Fatalf("schema %q: constrained output is not valid JSON: %q (%v)", schema, out.Bytes(), err)
	}
	if err := conforms(doc, val); err != nil {
		t.Fatalf("schema %q: constrained output %q does not conform: %v", schema, out.Bytes(), err)
	}
}

// topFrameInLiteral reports whether the grammar is currently inside a string,
// object key, or enum literal — states where a whitespace byte is meaningful
// content rather than optional structural spacing. Used by the fuzz satisfiability
// check to avoid mistaking required whitespace content for a livelock.
func topFrameInLiteral(g Grammar) bool {
	sg, ok := g.(*schemaGrammar)
	if !ok || len(sg.stack) == 0 {
		return false
	}
	switch sg.stack[len(sg.stack)-1].state {
	case fsStr, fsStrEsc, fsStrU, fsObjKeyStr, fsEnum:
		return true
	}
	return false
}

// isWSByte reports whether tok is a non-empty token made entirely of JSON
// whitespace (space, tab, LF, CR) — such a token can never advance the grammar.
func isWSByte(tok []byte) bool {
	if len(tok) == 0 {
		return false
	}
	for _, b := range tok {
		if b != ' ' && b != '\t' && b != '\n' && b != '\r' {
			return false
		}
	}
	return true
}

// fullByteVocab returns a vocabulary with one token per byte value (0x00–0xff)
// plus a trailing EOS id, so the masker can spell any byte a schema may require.
func fullByteVocab() (tokens [][]byte, eos int) {
	tokens = make([][]byte, 256, 257)
	for b := range 256 {
		tokens[b] = []byte{byte(b)}
	}
	eos = len(tokens)
	tokens = append(tokens, nil) // EOS: no surface bytes
	return tokens, eos
}

// FuzzSchemaGrammarBytes feeds arbitrary byte streams straight into a compiled
// grammar's TryBytes/Commit machine for each seed schema, asserting the byte-level
// acceptor never panics and honours its own contract: a byte that fails TryBytes
// must not be the one we Commit. This exercises step() on inputs a real decode
// would never reach (the masker only commits validated tokens).
func FuzzSchemaGrammarBytes(f *testing.F) {
	f.Add([]byte(`{"name":"x","age":0}`))
	f.Add([]byte("[1,2]"))
	f.Add([]byte(`{"u":{"id":-5}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		for _, s := range seedSchemas {
			g, err := JSONSchema([]byte(s))
			if err != nil {
				continue
			}
			g.Reset()
			for _, b := range data {
				one := []byte{b}
				if g.TryBytes(one) {
					g.Commit(one)
				}
			}
			_ = g.CanEnd() // must not panic regardless of where the stream left the stack
		}
	})
}
