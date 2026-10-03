package constrain

import (
	"bytes"
	"encoding/json"
	"math"
	"regexp"
	"strings"
	"testing"
)

// The whitespace bound (maxStructuralWS): whitespace is legal at every JSON structural boundary, so when a model's
// preferred token is illegal there (a bare number where the schema says "string") the mask used to leave whitespace as
// the top legal token and generation padded whitespace until max_tokens, silently. These tests pin the bound at the
// byte level on both grammars and then reproduce the actual runaway through a Masker.

type wsOne struct {
	Q string `json:"q"`
}

func spaces(n int) []byte { return bytes.Repeat([]byte{' '}, n) }

func schemaG(t *testing.T) Grammar {
	t.Helper()
	g, err := GrammarFromStruct(wsOne{})
	if err != nil {
		t.Fatal(err)
	}
	g.Reset()
	return g
}

func commitStr(g Grammar, s string) { g.Commit([]byte(s)) }

// Between a ':' and its value the bound is maxValueWS: that is where the runaway happens, and where a long pad also
// makes the forced token's context unnatural (see maxValueWS).
func TestWhitespaceBound_valuePosition(t *testing.T) {
	g := schemaG(t)
	commitStr(g, `{"q":`) // the value position: a string must open with '"', whitespace is the only other legal byte

	if !g.TryBytes(spaces(maxValueWS)) {
		t.Errorf("%d whitespace byte(s) after ':' must be legal (the canonical `\": \"`)", maxValueWS)
	}
	if g.TryBytes(spaces(maxValueWS + 1)) {
		t.Errorf("%d whitespace bytes after ':' must be refused (the runaway)", maxValueWS+1)
	}
	if g.TryBytes([]byte(" \n")) || g.TryBytes([]byte("\n\n")) {
		t.Error("a second whitespace byte of any kind after ':' must be refused")
	}
	// the generation is steered, not dead-ended: the string's opening quote stays legal after the maximal pad
	if !g.TryBytes(append(spaces(maxValueWS), '"')) {
		t.Error("the string's opening quote must stay legal after a maximal pad")
	}
	// ...and a bare number is still illegal where the schema says string (the bound limits formatting, not values)
	if g.TryBytes([]byte("6")) || g.TryBytes([]byte(" 6")) {
		t.Error("a bare number must stay illegal where the schema says string")
	}
}

// Everywhere else between tokens the bound is maxStructuralWS.
func TestWhitespaceBound_structuralPositions(t *testing.T) {
	for name, prefix := range map[string]string{
		"after '{', before the first key": `{`,
		"after a key, before the colon":   `{"q"`,
		"after a value, before '}'":       `{"q":"a"`,
	} {
		t.Run(name, func(t *testing.T) {
			g := schemaG(t)
			commitStr(g, prefix)
			if !g.TryBytes(spaces(maxStructuralWS)) {
				t.Errorf("%d whitespace bytes in a row must be legal here", maxStructuralWS)
			}
			if g.TryBytes(spaces(maxStructuralWS + 1)) {
				t.Errorf("%d whitespace bytes in a row must be refused here", maxStructuralWS+1)
			}
			// every whitespace byte kind counts, mixed
			if g.TryBytes([]byte(strings.Repeat(" \n\t\r", maxStructuralWS/4+1))) {
				t.Errorf("a mixed whitespace run longer than %d must be refused", maxStructuralWS)
			}
		})
	}
}

func TestWhitespaceBound_resetsAfterAnyOtherByte(t *testing.T) {
	g := schemaG(t)
	commitStr(g, `{`)
	commitStr(g, string(spaces(maxStructuralWS))) // a maximal run ...
	commitStr(g, `"q"`)                           // ... then a real token resets the count
	commitStr(g, string(spaces(maxStructuralWS))) // so another maximal run is fine
	if !g.TryBytes([]byte(`:`)) {
		t.Fatal("':' must be legal after a maximal run that follows a real token")
	}
	if g.TryBytes(spaces(1)) {
		t.Error("one more whitespace byte past the maximum must be refused")
	}
}

func TestWhitespaceBound_stringContentIsNotStructural(t *testing.T) {
	g := schemaG(t)
	commitStr(g, `{"q": "`)
	// a string's own spaces are content: 10x the bound is legal and the string still closes
	if !g.TryBytes(append(spaces(10*maxStructuralWS), []byte(`"}`)...)) {
		t.Error("spaces inside a string must never count toward the structural bound")
	}
	// and they do not leak a count into the structure after the string closes
	commitStr(g, string(spaces(10*maxStructuralWS))+`"`)
	if !g.TryBytes(append(spaces(maxStructuralWS), '}')) {
		t.Error("a full structural run must be legal right after a string full of spaces")
	}
}

func TestWhitespaceBound_trailingAfterCompleteDocument(t *testing.T) {
	g := schemaG(t)
	commitStr(g, `{"q":"a"}`)
	if !g.CanEnd() {
		t.Fatal("document should be complete")
	}
	if !g.TryBytes(spaces(maxStructuralWS)) || g.TryBytes(spaces(maxStructuralWS+1)) {
		t.Errorf("trailing whitespace after a complete document must stop at %d", maxStructuralWS)
	}
	commitStr(g, string(spaces(maxStructuralWS)))
	if !g.CanEnd() {
		t.Error("a complete document must still be able to end after maximal trailing whitespace")
	}
	if g.TryBytes(spaces(1)) {
		t.Error("only EOS may follow a document plus a maximal trailing run")
	}
}

func TestWhitespaceBound_jsonGrammar(t *testing.T) {
	g := JSON()
	g.Reset()
	commitStr(g, `[1,`)
	if !g.TryBytes(spaces(maxStructuralWS)) || g.TryBytes(spaces(maxStructuralWS+1)) {
		t.Errorf("the generic JSON grammar must bound structural whitespace at %d", maxStructuralWS)
	}
	if !g.TryBytes(append(spaces(maxStructuralWS), '2')) {
		t.Error("a value must stay legal after a maximal run")
	}
	// trailing
	g.Reset()
	commitStr(g, `{}`)
	if !g.TryBytes(spaces(maxStructuralWS)) || g.TryBytes(spaces(maxStructuralWS+1)) {
		t.Errorf("trailing whitespace after a complete JSON value must stop at %d", maxStructuralWS)
	}
	// strings are exempt
	g.Reset()
	commitStr(g, `["`)
	if !g.TryBytes(append(spaces(10*maxStructuralWS), []byte(`"]`)...)) {
		t.Error("spaces inside a JSON string must not count")
	}
}

// TryBytes must leave the count alone (it snapshots and restores), and Clone must carry it.
func TestWhitespaceBound_snapshotAndClone(t *testing.T) {
	for name, mk := range map[string]func() Grammar{
		"schema": func() Grammar { return schemaG(t) },
		"json":   func() Grammar { g := JSON(); g.Reset(); return g },
	} {
		t.Run(name, func(t *testing.T) {
			g := mk()
			if name == "schema" {
				commitStr(g, `{`) // a structural position (the value position's bound is 1, too small for this test)
			} else {
				commitStr(g, `[`)
			}
			commitStr(g, string(spaces(maxStructuralWS-4))) // 4 left
			for range 3 {
				if g.TryBytes(spaces(5)) {
					t.Fatal("5 more must be refused with 4 left")
				}
				if !g.TryBytes(spaces(4)) {
					t.Fatal("4 more must be legal with 4 left: a failed TryBytes leaked its count")
				}
			}
			c := g.Clone()
			if !c.TryBytes(spaces(4)) || c.TryBytes(spaces(5)) {
				t.Error("a clone must carry the whitespace count")
			}
			commitStr(c, string(spaces(4)))
			if !g.TryBytes(spaces(4)) {
				t.Error("committing on a clone must not move the original's count")
			}
		})
	}
}

// TestWhitespaceBound_stringTypedNumberRunaway reproduces the real failure through a Masker: a schema whose field is a
// STRING, and a "model" that wants a bare number at that position, then whitespace, then anything. Before the bound the
// mask left whitespace as the best legal token forever (the generation ran to max_tokens with no error); now the run
// ends, the opening quote becomes the only legal choice, and the document completes as valid JSON.
func TestWhitespaceBound_stringTypedNumberRunaway(t *testing.T) {
	g, err := GrammarFromStruct(wsOne{})
	if err != nil {
		t.Fatal(err)
	}
	tokens, eos := fullByteVocab()
	m := NewMasker(g, tokens, []int{eos})

	// scripted preferences: highest score wins after masking
	score := func(out []byte) []float32 {
		l := make([]float32, len(tokens))
		set := func(b byte, v float32) { l[int(b)] = v }
		set('6', 10)       // the model wants a bare number
		set(' ', 5)        // whitespace next
		set('"', 3)        // a quote only if nothing better is legal
		set('}', 2)        // closing
		l[eos] = 1         // EOS is masked until the document is complete
		for i := range l { // keep every other legal byte barely above -inf so the argmax is always defined
			if l[i] == 0 {
				l[i] = -1
			}
		}
		switch {
		case bytes.HasSuffix(out, []byte(`"6`)) && bytes.Count(out, []byte(`"`)) == 3:
			set('"', 20) // inside the string value after one char: close it
		case bytes.HasSuffix(out, []byte(`"6"`)):
			set('}', 20) // then close the object (the model would otherwise prefer whitespace again)
		case bytes.HasSuffix(out, []byte(`}`)):
			l[eos] = 20
		}
		return l
	}

	var gen []int
	var out []byte
	const maxSteps = 400
	finished := false
	for step := 0; step < maxSteps; step++ {
		l := score(out)
		m.Process(gen, l)
		best, bestV := -1, float32(math.Inf(-1))
		for id, v := range l {
			if v > bestV {
				best, bestV = id, v
			}
		}
		if best < 0 || math.IsInf(float64(bestV), -1) {
			t.Fatalf("step %d: no legal token (the grammar dead-ended); output so far %q", step, out)
		}
		if best == eos {
			finished = true
			break
		}
		gen = append(gen, best)
		out = append(out, tokens[best]...)
	}
	if !finished {
		t.Fatalf("generation did not finish in %d steps: the whitespace runaway (output %d bytes, ends %q)",
			maxSteps, len(out), tail(out, 30))
	}
	var v struct{ Q string }
	if err := json.Unmarshal(out, &v); err != nil {
		t.Fatalf("output is not valid JSON for the schema: %v\n%q", err, out)
	}
	if v.Q != "6" {
		t.Errorf("q = %q, want %q (the quote is forced, then the model's digit lands inside the string)", v.Q, "6")
	}
	if m := regexp.MustCompile(`"q" *:( *)"`).FindSubmatch(out); m == nil || len(m[1]) > maxValueWS {
		t.Errorf("the pad between \"q\": and its value must be at most %d byte(s); output %q", maxValueWS, out)
	}
	if run := longestWSRun(out); run > maxStructuralWS {
		t.Errorf("the output contains a %d-byte whitespace run, over the %d bound", run, maxStructuralWS)
	}
}

func tail(b []byte, n int) []byte {
	if len(b) <= n {
		return b
	}
	return b[len(b)-n:]
}

func longestWSRun(b []byte) int {
	best, cur, inStr, esc := 0, 0, false, false
	for _, c := range b {
		switch {
		case inStr:
			if esc {
				esc = false
			} else if c == '\\' {
				esc = true
			} else if c == '"' {
				inStr = false
			}
			cur = 0
		case c == '"':
			inStr, cur = true, 0
		case isWS(c):
			cur++
			best = max(best, cur)
		default:
			cur = 0
		}
	}
	return best
}
