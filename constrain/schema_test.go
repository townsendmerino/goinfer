package constrain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
	"time"
)

// A small byte-piece "vocabulary" for the property test: every printable ASCII
// byte (enough to spell any JSON) plus a few multi-byte pieces (to exercise
// tokens that span grammar-state boundaries) and an EOS id at the end.
func testVocab() (tokens [][]byte, eos int) {
	for b := byte(0x20); b <= 0x7e; b++ {
		tokens = append(tokens, []byte{b})
	}
	tokens = append(tokens, []byte("\n"), []byte("\t")) // whitespace control bytes (tool wrappers use \n)
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
	for range 5000 {
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
		for i := range 200 {
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
	for i := range 200 {
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

// TestSchema_rejectsUnsatisfiable covers the unenforceable/unsatisfiable schema
// classes found by FuzzJSONSchema: compile must reject them loudly rather than
// silently drop the constraint and let the masker emit non-conforming output.
func TestSchema_rejectsUnsatisfiable(t *testing.T) {
	reject := []string{
		// required names a property absent from properties → constraint silently
		// dropped, grammar emits {} which the schema forbids.
		`{"type":"object","required":[""]}`,
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["b"]}`,
		// required must be an array of strings.
		`{"type":"object","properties":{"a":{"type":"string"}},"required":"a"}`,
		`{"type":"object","properties":{"a":{"type":"string"}},"required":[1]}`,
		// array bounds: maxItems < minItems can never close.
		`{"type":"array","items":{"type":"integer"},"minItems":5,"maxItems":2}`,
		// negative bounds are not valid non-negative integers.
		`{"type":"array","items":{"type":"integer"},"maxItems":-1}`,
		`{"type":"array","items":{"type":"integer"},"minItems":-3}`,
	}
	for _, s := range reject {
		if _, err := JSONSchema([]byte(s)); err == nil {
			t.Errorf("expected compile error for unsatisfiable schema %s", s)
		}
	}
	// Adjacent valid forms must still compile (no over-rejection).
	accept := []string{
		`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`,
		`{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}`,
		`{"type":"array","items":{"type":"integer"},"minItems":0}`,
		`{"type":"array","items":{"type":"integer"}}`,
	}
	for _, s := range accept {
		if _, err := JSONSchema([]byte(s)); err != nil {
			t.Errorf("valid schema rejected: %s (%v)", s, err)
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

// M-29: enum/const literals went through json.Marshal, which HTML-ESCAPES by default.
//
// For {"enum":["<",">","="]} the grammar demanded "<", so the natural continuation `<`
// was masked at −∞ and a greedy model slid to whichever member remained reachable. The output
// still VALIDATES — the oracle marshals both sides the same way — which is why no round-trip
// test could see it. The assertion has to be on the literal bytes the grammar forces.
func TestEncodeLiteral_doesNotHTMLEscape(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"<", `"<"`},
		{">", `">"`},
		{"a&b", `"a&b"`},
		{"x", `"x"`},
	} {
		got, err := encodeLiteral(tc.in)
		if err != nil {
			t.Fatalf("encodeLiteral(%q): %v", tc.in, err)
		}
		if string(got) != tc.want {
			t.Errorf("encodeLiteral(%q) = %s, want %s — an escaped literal is unreachable for "+
				"the model even though the result still validates (M-29)", tc.in, got, tc.want)
		}
	}
	// And no trailing newline: json.Encoder appends one, and a literal must be exact.
	if got, _ := encodeLiteral("x"); string(got) != `"x"` {
		t.Errorf("encodeLiteral left trailing bytes: %q", got)
	}
}

// The masker half of M-29: `<` must actually be a legal first token for that enum.
func TestSchemaGrammar_htmlUnsafeEnumIsReachable(t *testing.T) {
	g, err := JSONSchema([]byte(`{"enum":["<",">","="]}`))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if !g.TryBytes([]byte(`"<`)) {
		t.Error(`the grammar rejects "< as a prefix: the < member of the enum is unreachable, ` +
			`so a model asked to choose it must pick a different one (M-29)`)
	}
}

// The precision half of M-29: an integer enum above 2^53 must survive the schema decode.
// float64 turns 9007199254740993 into ...992, and the grammar then forces a literal the
// caller never wrote — which validates against nothing the caller can check.
func TestSchemaGrammar_largeIntegerEnumKeepsItsDigits(t *testing.T) {
	const big = "9007199254740993" // 2^53 + 1, not representable in float64
	g, err := JSONSchema([]byte(`{"enum":[` + big + `]}`))
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if !g.TryBytes([]byte(big)) {
		t.Errorf("the grammar rejects %s, the only member of its own enum — the schema was "+
			"decoded through float64 and re-encoded with different digits (M-29)", big)
	}
	// The premise, so this cannot pass for the wrong reason: float64 really does change it.
	if fmt.Sprintf("%.0f", 9007199254740993.0) == big {
		t.Skip("float64 now represents 2^53+1 exactly; this test no longer describes the defect")
	}
}

// M-30: `{"type":"object","properties":{}}` is THE canonical no-argument tool schema —
// OpenAI's examples, most MCP servers, every Pydantic/zod no-arg tool — and it was a compile
// error, so a named tool_choice for such a tool came back 400. An OMITTED `parameters`
// already worked, so the two spellings of "no arguments" disagreed.
func TestToolCallGrammar_noArgumentSchemas(t *testing.T) {
	for name, schema := range map[string]string{
		"explicit empty properties": `{"type":"object","properties":{}}`,
		"absent properties":         `{"type":"object"}`,
		"already closed":            `{"type":"object","properties":{},"additionalProperties":false}`,
	} {
		t.Run(name, func(t *testing.T) {
			g, err := ToolCallGrammar("", "", "arguments", "ping", false, []byte(schema))
			if err != nil {
				t.Fatalf("ToolCallGrammar(%s): %v — a no-argument tool is refused, and "+
					"serveapp turns that into a 400 under a named tool_choice (M-30)", schema, err)
			}
			if !g.TryBytes([]byte(`{"name":"ping","arguments":{}}`)) {
				t.Errorf("%s: the grammar rejects the only call such a tool can make", schema)
			}
		})
	}

	// NOT narrowed: an explicit additionalProperties:true is still a freeform object, still
	// unconstrainable, and must still be refused rather than silently turned into "{} only".
	// Without this, the fix would look identical to one that accepts everything.
	if _, err := ToolCallGrammar("", "", "arguments", "ping", false,
		[]byte(`{"type":"object","properties":{},"additionalProperties":true}`)); err == nil {
		t.Error("additionalProperties:true compiled — a freeform object was silently narrowed " +
			"to the empty object, which is tighter than the schema means")
	}
}

// M-28: SchemaFromStruct did not flatten embedded structs, and the "json.Unmarshal always
// succeeds" contract was false three separate ways.
//
// The embedded case is the worst of them because it is SILENT: `type Person struct { Base;
// Name string }` produced {"Base":{…},"name":…}, the model satisfied that schema, and
// json.Unmarshal accepted the result WITHOUT ERROR while leaving every promoted field zero.
// A round-trip test that only checks `err == nil` passes on it.
func TestSchemaFromStruct_embeddedFieldsArePromoted(t *testing.T) {
	type Base struct {
		ID int `json:"id"`
	}
	type Person struct {
		Base
		Name string `json:"name"`
	}
	raw, err := SchemaFromStruct(Person{})
	if err != nil {
		t.Fatalf("SchemaFromStruct: %v", err)
	}
	var doc struct {
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse schema: %v", err)
	}
	if _, promoted := doc.Properties["id"]; !promoted {
		t.Errorf("embedded Base was not promoted; properties = %v. json.Unmarshal expects `id` "+
			"at the top level, and the un-promoted schema round-trips WITHOUT ERROR while "+
			"leaving ID zero (M-28)", slices.Sorted(maps.Keys(doc.Properties)))
	}
	if _, asObject := doc.Properties["Base"]; asObject {
		t.Error(`the embedded struct appears as a property named "Base"`)
	}
	if !slices.Contains(doc.Required, "id") {
		t.Errorf("promoted `id` is not required; required = %v", doc.Required)
	}

	// THE END-TO-END CHECK the shape argument rests on: what the grammar produces must
	// actually populate the promoted field, not merely parse.
	var p Person
	if err := json.Unmarshal([]byte(`{"id":7,"name":"x"}`), &p); err != nil || p.ID != 7 {
		t.Fatalf("premise: json.Unmarshal promotes id (got ID=%d, err=%v)", p.ID, err)
	}
	if err := json.Unmarshal([]byte(`{"Base":{"id":7},"name":"x"}`), &p); err != nil {
		t.Fatalf("premise: the OLD schema's output must parse without error — that is what "+
			"made this silent (err=%v)", err)
	}
}

// An embedded field WITH an explicit json name is a normal property, per encoding/json's
// own rule. A fix that promoted unconditionally would be wrong in the other direction.
func TestSchemaFromStruct_namedEmbeddedIsNotPromoted(t *testing.T) {
	type Base struct {
		ID int `json:"id"`
	}
	type Person struct {
		Base `json:"base"`
		Name string `json:"name"`
	}
	raw, err := SchemaFromStruct(Person{})
	if err != nil {
		t.Fatalf("SchemaFromStruct: %v", err)
	}
	if !strings.Contains(string(raw), `"base"`) {
		t.Errorf("a json-named embedded field was promoted anyway: %s", raw)
	}
}

// The other two halves of the false contract, both of which used to compile to a schema the
// model could satisfy and json.Unmarshal could not accept.
func TestSchemaFromStruct_selfUnmarshalingTypes(t *testing.T) {
	t.Run("time.Time maps to string", func(t *testing.T) {
		type Ev struct {
			At time.Time `json:"at"`
		}
		raw, err := SchemaFromStruct(Ev{})
		if err != nil {
			t.Fatalf("SchemaFromStruct: %v", err)
		}
		if !strings.Contains(string(raw), `"at":{"type":"string"}`) {
			t.Errorf("time.Time did not map to string: %s\n"+
				"as an object it has no exported fields, so the grammar forced `{}` — the one "+
				"value time.Time's UnmarshalJSON rejects (M-28)", raw)
		}
	})
	t.Run("a struct with no exported fields is refused", func(t *testing.T) {
		// The unexported field is the POINT: a struct whose only field is unexported has no
		// JSON shape to derive. staticcheck cannot know that, so the suppression is explicit.
		//lint:ignore U1000 deliberately unexported — the fixture is a struct with no exported fields
		type Opaque struct{ hidden int }
		type Wrap struct {
			O Opaque `json:"o"`
		}
		if _, err := SchemaFromStruct(Wrap{}); err == nil {
			t.Error("compiled to `{}`-only instead of erroring; the model can then produce only " +
				"the empty object, which is not what the field means")
		}
	})
}

// The unsigned half. This is also where the audit found the "test supplies its own calling
// convention" trap: the property test at schema_test.go worked around unbounded integers with
// its OWN 15-digit cap, which made the schema look adequate.
func TestSchemaFromStruct_unsignedRejectsNegative(t *testing.T) {
	type Q struct {
		N uint32 `json:"n"`
	}
	raw, err := SchemaFromStruct(Q{})
	if err != nil {
		t.Fatalf("SchemaFromStruct: %v", err)
	}
	g, err := JSONSchema(raw)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if g.TryBytes([]byte(`{"n":-`)) {
		t.Error(`a uint32 field accepts a leading "-": json.Unmarshal then FAILS on the ` +
			`grammar's own output (M-28)`)
	}
	// Signed fields must be unaffected — a fix that banned '-' everywhere would pass above.
	type S struct {
		N int32 `json:"n"`
	}
	sraw, _ := SchemaFromStruct(S{})
	sg, err := JSONSchema(sraw)
	if err != nil {
		t.Fatalf("compile signed: %v", err)
	}
	if !sg.TryBytes([]byte(`{"n":-`)) {
		t.Error("an int32 field now rejects a negative number")
	}

	// STATED, NOT FIXED: magnitude is still unbounded, which is exactly why the docstring no
	// longer promises json.Unmarshal always succeeds. Pinned so the gap cannot be forgotten
	// or quietly "fixed" without updating the contract that describes it.
	type B struct {
		N uint8 `json:"n"`
	}
	braw, _ := SchemaFromStruct(B{})
	bg, _ := JSONSchema(braw)
	if !bg.TryBytes([]byte(`{"n":99999`)) {
		t.Log("uint8 now bounds magnitude — update SchemaFromStruct's docstring, which says it does not")
	}
}
