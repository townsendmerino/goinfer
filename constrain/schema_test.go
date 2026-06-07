package constrain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// A small byte-piece "vocabulary" for the property test: every printable ASCII
// byte (enough to spell any JSON) plus a few multi-byte pieces (to exercise
// tokens that span grammar-state boundaries) and an EOS id at the end.
func testVocab() (tokens [][]byte, eos int) {
	for b := byte(0x20); b <= 0x7e; b++ {
		tokens = append(tokens, []byte{b})
	}
	for _, s := range []string{"true", "false", "null", "{}", "[]", "\":\"", "\",\""} {
		tokens = append(tokens, []byte(s))
	}
	eos = len(tokens)
	tokens = append(tokens, nil) // EOS: no surface bytes
	return tokens, eos
}

func isCloser(bs []byte) bool {
	return len(bs) == 1 && (bs[0] == '"' || bs[0] == '}' || bs[0] == ']')
}

// genConstrained drives the Masker with random sampling among the allowed tokens
// (biased toward closers so strings/containers terminate quickly), returning the
// produced bytes. It asserts the grammar never dead-ends and always reaches a
// complete document.
// maxDigitRun, when >0, stops the random walk from extending a digit run past
// that length (by preferring non-digit tokens) — keeps generated numbers within
// Go's fixed-width range for the struct round-trip. The grammar itself does NOT
// bound magnitude (JSON Schema integer/number don't); 0 = no cap.
func genConstrained(t *testing.T, g Grammar, seed int64, maxDigitRun int) []byte {
	t.Helper()
	tokens, eos := testVocab()
	m := NewMasker(g, tokens, []int{eos}).StopWhenComplete()
	rng := rand.New(rand.NewSource(seed))
	var generated []int
	var out bytes.Buffer
	neg := float32(math.Inf(-1))
	for step := 0; step < 5000; step++ {
		logits := make([]float32, len(tokens))
		m.Process(generated, logits)
		var allowed []int
		for id, v := range logits {
			if v > neg {
				allowed = append(allowed, id)
			}
		}
		if len(allowed) == 0 {
			t.Fatalf("seed %d: grammar dead-ended after %q (no allowed token)", seed, out.String())
		}
		pool := allowed
		if maxDigitRun > 0 && trailingDigits(out.Bytes()) >= maxDigitRun {
			var nd []int
			for _, id := range allowed {
				if bs := tokens[id]; len(bs) == 0 || bs[0] < '0' || bs[0] > '9' {
					nd = append(nd, id)
				}
			}
			if len(nd) > 0 {
				pool = nd // force the number to end rather than grow unboundedly
			}
		}
		var closers []int
		for _, id := range pool {
			if id != eos && isCloser(tokens[id]) {
				closers = append(closers, id)
			}
		}
		var pick int
		if len(closers) > 0 && rng.Float64() < 0.5 {
			pick = closers[rng.Intn(len(closers))]
		} else {
			pick = pool[rng.Intn(len(pool))]
		}
		if pick == eos {
			return out.Bytes()
		}
		generated = append(generated, pick)
		out.Write(tokens[pick])
	}
	t.Fatalf("seed %d: did not complete in 5000 steps: %q", seed, out.String())
	return nil
}

func trailingDigits(b []byte) int {
	n := 0
	for i := len(b) - 1; i >= 0 && b[i] >= '0' && b[i] <= '9'; i-- {
		n++
	}
	return n
}

// schemas exercised by the property test, with their compiled grammar.
func propertySchemas(t *testing.T) map[string][]byte {
	t.Helper()
	return map[string][]byte{
		"flat object": []byte(`{
			"type":"object","additionalProperties":false,
			"properties":{"name":{"type":"string"},"age":{"type":"integer"},"vip":{"type":"boolean"}},
			"required":["name","age"]}`),
		"enum + const": []byte(`{
			"type":"object","additionalProperties":false,
			"properties":{"color":{"enum":["red","green","blue"]},"kind":{"const":"widget"},"n":{"type":"number"}},
			"required":["color","kind"]}`),
		"array bounds": []byte(`{
			"type":"object","additionalProperties":false,
			"properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}},
			"required":["tags"]}`),
		"nested": []byte(`{
			"type":"object","additionalProperties":false,
			"properties":{"user":{"type":"object","additionalProperties":false,
				"properties":{"id":{"type":"integer"},"roles":{"type":"array","items":{"enum":["a","b"]}}},
				"required":["id"]}},
			"required":["user"]}`),
		"top-level array": []byte(`{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}`),
	}
}

// TestSchema_propertyValidates is the acceptance gate: for many random
// constrained generations under each supported schema, the output ALWAYS parses
// and conforms to the schema (checked by an independent validator).
func TestSchema_propertyValidates(t *testing.T) {
	for name, schema := range propertySchemas(t) {
		var doc map[string]any
		if err := json.Unmarshal(schema, &doc); err != nil {
			t.Fatalf("%s: bad schema: %v", name, err)
		}
		for i := 0; i < 200; i++ {
			g, err := JSONSchema(schema)
			if err != nil {
				t.Fatalf("%s: compile: %v", name, err)
			}
			out := genConstrained(t, g, int64(i)+1, 0)
			// UseNumber so huge-but-valid JSON numbers don't overflow the oracle
			// (the grammar correctly allows any-magnitude integers/numbers).
			dec := json.NewDecoder(bytes.NewReader(out))
			dec.UseNumber()
			var val any
			if err := dec.Decode(&val); err != nil {
				t.Fatalf("%s seed %d: output is not valid JSON: %q (%v)", name, i, out, err)
			}
			if err := conforms(doc, val); err != nil {
				t.Fatalf("%s seed %d: output %q does not conform: %v", name, i, out, err)
			}
		}
	}
}

// TestGrammarFromStruct_roundTrip is the killer demo as a test: constrained
// generation against a struct-derived grammar always json.Unmarshals into the
// struct (with unknown fields disallowed — additionalProperties:false).
func TestGrammarFromStruct_roundTrip(t *testing.T) {
	type Address struct {
		City string `json:"city"`
		Zip  string `json:"zip,omitempty"`
	}
	type Person struct {
		Name    string   `json:"name"`
		Age     int      `json:"age"`
		Active  bool     `json:"active"`
		Tags    []string `json:"tags"`
		Address Address  `json:"address"`
		Email   *string  `json:"email,omitempty"`
	}
	for i := 0; i < 200; i++ {
		g, err := GrammarFromStruct(Person{})
		if err != nil {
			t.Fatalf("GrammarFromStruct: %v", err)
		}
		out := genConstrained(t, g, int64(i)+1, 15) // cap digits so ints fit int64
		dec := json.NewDecoder(bytes.NewReader(out))
		dec.DisallowUnknownFields()
		var p Person
		if err := dec.Decode(&p); err != nil {
			t.Fatalf("seed %d: unmarshal into struct failed: %q (%v)", i, out, err)
		}
	}
}

// TestSchema_unsupported confirms unenforceable keywords are loud errors.
func TestSchema_unsupported(t *testing.T) {
	for _, s := range []string{
		`{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":true}`,
		`{"type":"wat"}`,
		`{}`,
	} {
		if _, err := JSONSchema([]byte(s)); err == nil {
			t.Errorf("expected error compiling %s", s)
		}
	}
}

// TestSchema_rejectsInvalid spot-checks that the grammar refuses non-conforming
// prefixes (the masker would −∞ the token that produced them).
func TestSchema_rejectsInvalid(t *testing.T) {
	schema := []byte(`{"type":"object","additionalProperties":false,
		"properties":{"n":{"type":"integer"}},"required":["n"]}`)
	g, err := JSONSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		`{"n":1.5`,  // integer field, fraction not allowed
		`{"x":`,     // unknown property (additionalProperties:false)
		`{"n":"hi"`, // wrong type
		`{}`,        // missing required "n" → '}' rejected
		`[`,         // wrong top-level type
	} {
		g.Reset()
		if g.TryBytes([]byte(bad)) {
			t.Errorf("grammar wrongly accepted invalid prefix %q", bad)
		}
	}
	// And a valid one is accepted + completes.
	g.Reset()
	if !g.TryBytes([]byte(`{"n":-42}`)) {
		t.Error("grammar rejected a valid document")
	}
	g.Commit([]byte(`{"n":-42}`))
	if !g.CanEnd() {
		t.Error("CanEnd false after a complete document")
	}
}

// --- independent oracle: validate a parsed value against a schema doc ---

func conforms(schema map[string]any, val any) error {
	if c, ok := schema["const"]; ok {
		return eqJSON(c, val)
	}
	if e, ok := schema["enum"].([]any); ok {
		for _, cand := range e {
			if eqJSON(cand, val) == nil {
				return nil
			}
		}
		return fmt.Errorf("value %v not in enum", val)
	}
	switch schema["type"] {
	case "object":
		m, ok := val.(map[string]any)
		if !ok {
			return fmt.Errorf("want object, got %T", val)
		}
		props, _ := schema["properties"].(map[string]any)
		for k := range m {
			if _, known := props[k]; !known {
				return fmt.Errorf("unknown property %q", k)
			}
		}
		if rl, ok := schema["required"].([]any); ok {
			for _, r := range rl {
				if _, present := m[r.(string)]; !present {
					return fmt.Errorf("missing required %q", r)
				}
			}
		}
		for k, sub := range props {
			if v, present := m[k]; present {
				if err := conforms(sub.(map[string]any), v); err != nil {
					return fmt.Errorf("property %q: %w", k, err)
				}
			}
		}
	case "array":
		a, ok := val.([]any)
		if !ok {
			return fmt.Errorf("want array, got %T", val)
		}
		if mi, ok := schema["minItems"].(float64); ok && len(a) < int(mi) {
			return fmt.Errorf("array len %d < minItems %d", len(a), int(mi))
		}
		if ma, ok := schema["maxItems"].(float64); ok && len(a) > int(ma) {
			return fmt.Errorf("array len %d > maxItems %d", len(a), int(ma))
		}
		items, _ := schema["items"].(map[string]any)
		for i, e := range a {
			if err := conforms(items, e); err != nil {
				return fmt.Errorf("item %d: %w", i, err)
			}
		}
	case "string":
		if _, ok := val.(string); !ok {
			return fmt.Errorf("want string, got %T", val)
		}
	case "integer":
		num, ok := val.(json.Number)
		if !ok || strings.ContainsAny(num.String(), ".eE") {
			return fmt.Errorf("want integer, got %v", val)
		}
	case "number":
		if _, ok := val.(json.Number); !ok {
			return fmt.Errorf("want number, got %T", val)
		}
	case "boolean":
		if _, ok := val.(bool); !ok {
			return fmt.Errorf("want boolean, got %T", val)
		}
	case "null":
		if val != nil {
			return fmt.Errorf("want null, got %v", val)
		}
	}
	return nil
}

func eqJSON(a, b any) error {
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	if !bytes.Equal(ab, bb) {
		return fmt.Errorf("%s != %s", ab, bb)
	}
	return nil
}
