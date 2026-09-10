package constrain

import (
	"encoding/json"
	"math"
	"testing"
)

// buildAdversarialVocab returns a vocabulary that exercises every byte class the plain-string
// fast path reasons about, exhaustively at 1 and 2 bytes plus targeted longer shapes. It is
// deliberately NOT a real tokenizer's vocab: a real one may simply not contain the awkward
// tokens (a lone backslash, a quote followed by a control byte), and those are exactly the
// cases where a wrong fast path would be wrong.
func buildAdversarialVocab() [][]byte {
	interesting := []byte{
		0x00, 0x01, 0x09, 0x0a, 0x0d, 0x1f, // control bytes, incl. the boundary 0x1f
		0x20, '"', '\\', '/', '{', '}', '[', ']', ':', ',',
		'a', 'z', '0', '9', 0x7f, 0x80, 0xc3, 0xa9, 0xff, // ASCII, DEL, UTF-8 continuation/lead
	}
	var v [][]byte
	v = append(v, nil, []byte{}) // empty-surface ids (control/pad tokens)
	for _, a := range interesting {
		v = append(v, []byte{a})
	}
	for _, a := range interesting {
		for _, b := range interesting {
			v = append(v, []byte{a, b})
		}
	}
	// Longer shapes where the FIRST byte decides a different state than a later one — the
	// reason a control byte cannot be treated as unconditionally illegal.
	for _, s := range []string{
		`"` + "\n", `"` + "\x00", `\n`, `\\`, `\"`, `ab"cd`, `ab\cd`, "ab\x01cd",
		`hello`, `héllo`, `"`, `""`, `"":`, `xyz"`, "\x1f\x20", " ", "  ",
	} {
		v = append(v, []byte(s))
	}
	return v
}

// TestPlainString_exact is the correctness gate for the fast path, and it is a PROOF over the
// whole vocabulary rather than a sample: for every id, the masked logit produced with the
// bitmap must equal the one produced by the full grammar walk, at every plain-string state a
// document passes through.
//
// If this ever fails, the optimization is silently letting the model emit JSON that violates
// the schema — the exact promise constrained decoding exists to make — so it is checked
// against the slow path directly rather than against an expected list someone maintains.
func TestPlainString_exact(t *testing.T) {
	vocab := buildAdversarialVocab()
	eos := []int{0}

	type Inner struct {
		Note string `json:"note"`
	}
	type Doc struct {
		Name  string   `json:"name"`
		Tags  []string `json:"tags"`
		Inner Inner    `json:"inner"`
	}
	schema, err := GrammarFromStruct(Doc{})
	if err != nil {
		t.Fatalf("GrammarFromStruct: %v", err)
	}

	// Every prefix that lands inside a string, across object values, array elements and a
	// nested object — plus non-string states, which must be unaffected by the fast path.
	prefixes := []string{
		`{"name":"`, `{"name":"A`, `{"name":"Ada Lovelace`,
		`{"name":"a","tags":["`, `{"name":"a","tags":["x`, `{"name":"a","tags":["x","y`,
		`{"name":"a","tags":[],"inner":{"note":"`, `{"name":"a","tags":[],"inner":{"note":"deep`,
		`{`, `{"`, `{"name`, `{"name":`, `{"name":"a"`, `{"name":"a",`, // key/structure states
		`{"name":"a","tags":[],"inner":{"note":"n"}}`, // complete
	}

	for _, gname := range []string{"schema", "json"} {
		for _, prefix := range prefixes {
			var base Grammar
			if gname == "schema" {
				base = schema
			} else {
				base = JSON()
			}

			fast := NewMasker(base.Clone(), vocab, eos)
			slow := NewMasker(base.Clone(), vocab, eos)
			slow.plainOK = newBitset(len(vocab)) // all zero ⇒ never fast-pathed: the reference

			gf, gs := fast.GrammarClone(), slow.GrammarClone()
			gf.Reset()
			gs.Reset()
			// A prefix the grammar rejects is not a state this test can reach; skip it
			// rather than assert on an undefined state.
			if !gf.TryBytes([]byte(prefix)) {
				continue
			}
			gf.Commit([]byte(prefix))
			gs.Commit([]byte(prefix))

			lf := make([]float32, len(vocab))
			ls := make([]float32, len(vocab))
			fast.MaskAt(gf, lf)
			slow.MaskAt(gs, ls)

			plain := inPlainString(gf)
			for id := range vocab {
				bf, bs := math.IsInf(float64(lf[id]), -1), math.IsInf(float64(ls[id]), -1)
				if bf != bs {
					t.Fatalf("%s grammar, prefix %q (plainString=%v): id %d (%q) — fast path says masked=%v, full walk says masked=%v",
						gname, prefix, plain, id, vocab[id], bf, bs)
				}
			}
		}
	}
}

// TestToolGrammar_plainStringExact is P-17 (audit-2026-09-10): toolGrammar had no InPlainString
// at all, so inPlainString's type assertion never matched it and every forced tool call paid
// the full walk on its own JSON body — the exact regression TestPlainString_exact above already
// gates for schemaGrammar/jsonGrammar directly. Same harness, applied to *toolGrammar's own
// wrapped shape (a literal prefix, then {"name":const,"arguments":<paramSchema>}, then a literal
// suffix) so a wrong phase boundary (e.g. treating the literal prefix/suffix as plain-string-able)
// would be caught here, not just proven absent by construction.
func TestToolGrammar_plainStringExact(t *testing.T) {
	vocab := buildAdversarialVocab()
	eos := []int{0}

	paramSchema, err := SchemaFromStruct(struct {
		Query string   `json:"query"`
		Tags  []string `json:"tags"`
	}{})
	if err != nil {
		t.Fatalf("SchemaFromStruct: %v", err)
	}
	base, err := ToolCallGrammar("<tool_call>\n", "\n</tool_call>", "arguments", "search", false, paramSchema)
	if err != nil {
		t.Fatalf("ToolCallGrammar: %v", err)
	}

	prefixes := []string{
		// Phase 0: the literal prefix, partial and complete — never plain-string-able.
		"", "<tool_call", "<tool_call>\n",
		// Phase 1: inside the JSON value, at various points including string state.
		`<tool_call>
{"name":"`, `<tool_call>
{"name":"search","arguments":{"query":"`, `<tool_call>
{"name":"search","arguments":{"query":"find x`,
		`<tool_call>
{"name":"search","arguments":{"query":"q","tags":["`,
		`<tool_call>
{"name":"search","arguments":{"query":"q","tags":["a`,
		// Complete JSON value, into phase 2 (the literal suffix) — never plain-string-able.
		`<tool_call>
{"name":"search","arguments":{"query":"q","tags":[]}}`,
	}

	var everEngaged bool
	for _, prefix := range prefixes {
		fast := NewMasker(base.Clone(), vocab, eos)
		slow := NewMasker(base.Clone(), vocab, eos)
		slow.plainOK = newBitset(len(vocab)) // all zero ⇒ never fast-pathed: the reference

		gf, gs := fast.GrammarClone(), slow.GrammarClone()
		gf.Reset()
		gs.Reset()
		if !gf.TryBytes([]byte(prefix)) {
			continue // a prefix the grammar rejects is not a reachable state
		}
		gf.Commit([]byte(prefix))
		gs.Commit([]byte(prefix))

		lf := make([]float32, len(vocab))
		ls := make([]float32, len(vocab))
		fast.MaskAt(gf, lf)
		slow.MaskAt(gs, ls)

		plain := inPlainString(gf)
		everEngaged = everEngaged || plain
		for id := range vocab {
			bf, bs := math.IsInf(float64(lf[id]), -1), math.IsInf(float64(ls[id]), -1)
			if bf != bs {
				t.Fatalf("prefix %q (plainString=%v): id %d (%q) — fast path says masked=%v, full walk says masked=%v",
					prefix, plain, id, vocab[id], bf, bs)
			}
		}
	}
	// TAUTOLOGY GUARD: without InPlainString, inPlainString(gf) is always false (the type
	// assertion never matches), so fast and slow compute the identical thing and every prefix
	// above would pass trivially without ever exercising the fast path this test is about.
	if !everEngaged {
		t.Fatal("no prefix ever reported plainString=true — InPlainString never engaged, so this test proved nothing")
	}
}

// TestPlainString_classification pins the bitmap's meaning: exactly the tokens with no '"',
// no '\' and no byte < 0x20 (and a non-empty surface) are marked, and nothing else. A drift
// here would not fail the exactness test if it were conservative, so it is asserted directly.
func TestPlainString_classification(t *testing.T) {
	vocab := buildAdversarialVocab()
	bs := buildPlainString(vocab)
	for id, tok := range vocab {
		want := len(tok) > 0
		for _, c := range tok {
			if c < 0x20 || c == '"' || c == '\\' {
				want = false
				break
			}
		}
		if bs.has(id) != want {
			t.Errorf("id %d (%q): marked=%v want=%v", id, tok, bs.has(id), want)
		}
	}
}

// TestPlainString_endToEnd is the promise itself: a constrained generation driven only by the
// masker still yields JSON that unmarshals into the target struct. Guards against a fast path
// that is self-consistent but lets an invalid document through.
func TestPlainString_endToEnd(t *testing.T) {
	vocab := buildAdversarialVocab()
	g, err := GrammarFromStruct(struct {
		Name string `json:"name"`
	}{})
	if err != nil {
		t.Fatal(err)
	}
	m := NewMasker(g, vocab, []int{0}).StopWhenComplete()
	gg := m.GrammarClone()
	gg.Reset()
	doc := `{"name":"héllo"}`
	if !gg.TryBytes([]byte(doc)) {
		t.Fatalf("the grammar rejects a valid document: %s", doc)
	}
	gg.Commit([]byte(doc))
	if !gg.CanEnd() {
		t.Error("a complete document should be able to end")
	}
	var out struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal([]byte(doc), &out); err != nil || out.Name != "héllo" {
		t.Errorf("unmarshal: %v, got %q", err, out.Name)
	}
}
