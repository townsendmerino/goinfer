package constrain

import (
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"
)

// pydanticSchema reads one of the schemas Pydantic emitted (testdata/pydantic/gen.py).
func pydanticSchema(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/pydantic/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func mustSchema(t *testing.T, schema []byte) Grammar {
	t.Helper()
	g, err := JSONSchema(schema)
	if err != nil {
		t.Fatalf("compile %s: %v", schema, err)
	}
	return g
}

// docCase is a document and whether a grammar must accept it whole.
type docCase struct {
	doc  string
	want bool
	why  string
}

func checkDocs(t *testing.T, g Grammar, cases []docCase) {
	t.Helper()
	for _, c := range cases {
		if got := accepts(g, c.doc); got != c.want {
			t.Errorf("accepts(%s) = %v, want %v: %s", c.doc, got, c.want, c.why)
		}
	}
}

// A nested model: Pydantic puts the inner model under `$defs` and points at it with `$ref`, once bare and once beside
// a description.
func TestSchema_refPydanticNestedModel(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "nested_model"))
	addr := `{"street":"1 Main","city":"Ely"}`
	checkDocs(t, g, []docCase{
		{`{"name":"a","address":` + addr + `,"shipping":` + addr + `}`, true, "a valid document"},
		{`{"shipping":` + addr + `,"address":` + addr + `,"name":"a"}`, true, "keys in any order"},
		{`{"name":"a","address":` + addr + `}`, false, "shipping is required"},
		{`{"name":"a","address":"1 Main","shipping":` + addr + `}`, false, "the referenced schema is an object"},
		{`{"name":"a","address":{"street":"1 Main"},"shipping":` + addr + `}`, false, "the referenced schema requires city"},
		{`{"name":"a","address":{"street":"1 Main","city":"Ely","zip":"x"},"shipping":` + addr + `}`, false, "the referenced schema declares no zip"},
		{`{"name":"a","address":{"street":1,"city":"Ely"},"shipping":` + addr + `}`, false, "street is a string"},
		{`{"name":"a","address":` + addr + `,"shipping":{"city":"Ely"}}`, false, "a $ref beside a description is held to its target too"},
	})
}

// commentChain is n comments, each the only reply to the one before.
func commentChain(n int, innermost string) string {
	return strings.Repeat(`{"text":"","replies":[`, n-1) + innermost + strings.Repeat(`]}`, n-1)
}

// A recursive model: the root is a `$ref` to a definition whose array items point back at it.
func TestSchema_refPydanticRecursiveModel(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "recursive_model"))
	checkDocs(t, g, []docCase{
		{`{"text":"a"}`, true, "replies is optional"},
		{`{"text":"a","replies":[]}`, true, "no replies"},
		{`{"text":"a","replies":[{"text":"b"},{"text":"c","replies":[{"text":"d","replies":[]}]}]}`, true, "replies nest"},
		{`{"replies":[]}`, false, "text is required"},
		{`{"text":"a","replies":[{"replies":[]}]}`, false, "text is required in a reply too"},
		{`{"text":"a","replies":["b"]}`, false, "a reply is a comment, not a string"},
		{`{"text":"a","replies":[{"text":"b","likes":1}]}`, false, "a reply has a comment's keys only"},
		{`{"text":"a","replies":{"text":"b"}}`, false, "replies is an array"},
	})

	// The recursion is cut at the nesting cap. A comment sits two levels below its parent (the array, then the object) and
	// its required text one level below it, so comments fit at levels 0, 2, ... up to the last even level under the cap.
	fit := maxSchemaDepth / 2
	checkDocs(t, g, []docCase{
		{commentChain(fit, `{"text":""}`), true, "the deepest comment that fits"},
		{commentChain(fit, `{"text":"","replies":[]}`), true, "the deepest comment may say it has no replies"},
		{commentChain(fit+1, `{"text":""}`), false, "one comment deeper than fits"},
	})
	sg := g.(*schemaGrammar)
	sg.Reset()
	deepest := 0
	for _, b := range []byte(commentChain(fit, `{"text":""}`)) {
		sg.Commit([]byte{b})
		deepest = max(deepest, len(sg.stack))
	}
	if deepest > maxSchemaDepth+1 {
		t.Errorf("the frame stack reached %d frames; a schema without $ref is held to %d", deepest, maxSchemaDepth+1)
	}

	// At the cut the grammar steers, it does not dead-end: from inside the deepest comment's replies array the only way
	// on is to close it, and every level closes after it.
	sg.Reset()
	prefix := strings.Repeat(`{"text":"","replies":[`, fit)
	if n, _ := pinAccept(sg, []byte(prefix)); n != len(prefix) {
		t.Fatalf("the deepest replies array was refused at byte %d", n)
	}
	if sg.TryBytes([]byte(`{`)) || !sg.TryBytes([]byte(`]`)) {
		t.Error("inside the deepest replies array a reply must be refused and the closing bracket allowed")
	}
	closeAll := strings.Repeat(`]}`, fit)
	if !sg.TryBytes([]byte(closeAll)) {
		t.Error("the document cannot be closed from the deepest level")
	}
}

// refPairs are schemas that use `$ref`, each beside the schema it must behave as: its references written out in place.
var refPairs = []struct{ name, ref, inline string }{
	{"$defs",
		`{"type":"object","properties":{"a":{"$ref":"#/$defs/P"},"b":{"type":"array","items":{"$ref":"#/$defs/P"}}},"required":["a"],
		  "$defs":{"P":{"type":"object","properties":{"x":{"type":"integer"},"tag":{"enum":["u","v"]}},"required":["x"]}}}`,
		`{"type":"object","properties":{"a":{"type":"object","properties":{"x":{"type":"integer"},"tag":{"enum":["u","v"]}},"required":["x"]},
		  "b":{"type":"array","items":{"type":"object","properties":{"x":{"type":"integer"},"tag":{"enum":["u","v"]}},"required":["x"]}}},"required":["a"]}`},
	{"definitions",
		`{"definitions":{"Name":{"type":"string"}},"type":"array","items":{"$ref":"#/definitions/Name"},"maxItems":2}`,
		`{"type":"array","items":{"type":"string"},"maxItems":2}`},
	{"a root $ref",
		`{"$defs":{"Root":{"type":"object","properties":{"n":{"type":"number","minimum":0}},"additionalProperties":false}},"$ref":"#/$defs/Root"}`,
		`{"type":"object","properties":{"n":{"type":"number","minimum":0}},"additionalProperties":false}`},
	{"a chain of references",
		`{"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"$ref":"#/$defs/C","title":"b"},"C":{"type":"boolean"}},"type":"object","properties":{"f":{"$ref":"#/$defs/A"}},"required":["f"]}`,
		`{"type":"object","properties":{"f":{"type":"boolean"}},"required":["f"]}`},
	{"a pointer into properties",
		`{"type":"object","properties":{"first":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]},"second":{"$ref":"#/properties/first"}}}`,
		`{"type":"object","properties":{"first":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]},"second":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}}}`},
	{"escaped pointer tokens",
		`{"$defs":{"a/b":{"type":"null"},"c~d":{"const":"k"},"e f":{"type":"integer"}},"type":"object",
		  "properties":{"x":{"$ref":"#/$defs/a~1b"},"y":{"$ref":"#/$defs/c~0d"},"z":{"$ref":"#/$defs/e%20f"}},"required":["x","y","z"]}`,
		`{"type":"object","properties":{"x":{"type":"null"},"y":{"const":"k"},"z":{"type":"integer"}},"required":["x","y","z"]}`},
	{"nested definitions",
		`{"$defs":{"Outer":{"$defs":{"Inner":{"type":"string","format":"date"}},"type":"object","properties":{"when":{"$ref":"#/$defs/Outer/$defs/Inner"}},"required":["when"]}},"$ref":"#/$defs/Outer"}`,
		`{"type":"object","properties":{"when":{"type":"string","format":"date"}},"required":["when"]}`},
}

// TestSchema_refBehavesAsInlined: a schema with references allows the same next bytes, at every prefix of the documents
// its inlined form accepts and of one-edit variants of them, as the inlined form does.
func TestSchema_refBehavesAsInlined(t *testing.T) {
	pairs := refPairs
	nested := pydanticSchema(t, "nested_model")
	pairs = append(pairs, struct{ name, ref, inline string }{"pydantic nested model", string(nested),
		`{"type":"object","properties":{"name":{"type":"string"},
		  "address":{"type":"object","properties":{"street":{"type":"string"},"city":{"type":"string"}},"required":["street","city"]},
		  "shipping":{"type":"object","properties":{"street":{"type":"string"},"city":{"type":"string"}},"required":["street","city"]}},
		  "required":["name","address","shipping"]}`})
	for _, p := range pairs {
		ref, err := JSONSchema([]byte(p.ref))
		if err != nil {
			t.Errorf("%s: %v", p.name, err)
			continue
		}
		inline := mustSchema(t, []byte(p.inline))
		rng := rand.New(rand.NewPCG(7, uint64(len(p.name))))
		walks := 0
		for range 40 {
			doc, ok := pinWalkDoc(inline, rng)
			if !ok {
				continue
			}
			walks++
			if a, b := pinMask(ref, doc), pinMask(inline, doc); a != b {
				t.Errorf("%s: along %q the schema with $ref allows different next bytes than its inlined form", p.name, doc)
			}
			for range 4 {
				m := pinMutate(doc, rng)
				rn, rend := pinAccept(ref, m)
				in, iend := pinAccept(inline, m)
				if rn != in || rend != iend {
					t.Errorf("%s: %q is accepted for %d bytes (may end: %v) with $ref and %d (may end: %v) inlined", p.name, m, rn, rend, in, iend)
				}
			}
		}
		if walks < 20 {
			t.Errorf("%s: only %d walks completed", p.name, walks)
		}
	}
}

// TestSchema_refErrors: what a reference cannot do stays a compile error that names it.
func TestSchema_refErrors(t *testing.T) {
	for _, c := range []struct{ schema, want string }{
		{`{"$ref":"https://example.com/address.json"}`, "only a reference into the same document"},
		{`{"$ref":"address.json#/$defs/A"}`, "only a reference into the same document"},
		{`{"$ref":"#anchor"}`, "JSON Pointer"},
		{`{"$ref":"#/$defs/Missing","$defs":{}}`, `"#/$defs/Missing"`},
		{`{"$ref":"#/$defs/x"}`, `"#/$defs/x"`},
		{`{"$ref":5}`, "$ref must be a string"},
		{`{"$defs":{"A":"nope"},"$ref":"#/$defs/A"}`, "schema object"},
		{`{"$defs":{"A":{"type":"string"}},"$ref":"#/$defs/A","type":"string"}`, `"type" beside $ref`},
		{`{"$defs":{"A":{"type":"string"}},"type":"object","properties":{"x":{"$ref":"#/$defs/A","enum":["a"]}}}`, `"enum" beside $ref`},
		{`{"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"$ref":"#/$defs/A"}},"$ref":"#/$defs/A"}`, "refers to itself"},
		{`{"$ref":"#"}`, "refers to itself"},
		{`{"$defs":{"T":{"type":"object","properties":{"l":{"$ref":"#/$defs/T"}},"required":["l"]}},"$ref":"#/$defs/T"}`, "levels of nesting"},
		{`{"$defs":{"L":{"type":"array","items":{"$ref":"#/$defs/L"},"minItems":1}},"$ref":"#/$defs/L"}`, "levels of nesting"},
		{`{"$defs":{"A":{"type":"string","pattern":"x"}},"$ref":"#/$defs/A"}`, `"pattern"`},
		{`{"$dynamicRef":"#x"}`, `"$dynamicRef"`},
		{`{"$defs":{"A":{"$id":"https://example.com/a","type":"string"}},"$ref":"#/$defs/A"}`, "$id"},
	} {
		_, err := JSONSchema([]byte(c.schema))
		if err == nil {
			t.Errorf("%s compiled; want an error naming %s", c.schema, c.want)
		} else if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not name %s", c.schema, err, c.want)
		}
	}
	// A definition nothing refers to constrains nothing, so what it holds is not compiled.
	if _, err := JSONSchema([]byte(`{"$defs":{"Unused":{"type":"string","pattern":"x"}},"definitions":{"Old":5},"type":"string"}`)); err != nil {
		t.Errorf("a definition nothing refers to failed the compile: %v", err)
	}
}

// refChain is n definitions, each an array of the next, ending in a string: references without a cycle.
func refChain(n int) string {
	var b strings.Builder
	b.WriteString(`{"$ref":"#/$defs/D0","$defs":{`)
	for i := range n {
		fmt.Fprintf(&b, `"D%d":{"type":"array","items":{"$ref":"#/$defs/D%d"}},`, i, i+1)
	}
	fmt.Fprintf(&b, `"D%d":{"type":"string"}}}`, n)
	return b.String()
}

// TestSchema_refNestingErrorIsKept: only a reference cycle is cut at the nesting cap. References without a cycle are
// their inlined form, whose nesting past the cap is a compile error, and a cycle elsewhere in the document does not
// excuse nesting written out in place.
func TestSchema_refNestingErrorIsKept(t *testing.T) {
	if _, err := JSONSchema([]byte(refChain(maxSchemaDepth))); err != nil {
		t.Errorf("a chain of references within the nesting cap was refused: %v", err)
	}
	if _, err := JSONSchema([]byte(refChain(maxSchemaDepth + 6))); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Errorf("a chain of references past the nesting cap: got %v, want the nesting error its inlined form gets", err)
	}
	cyclic := `{"type":"object","properties":{"deep":` + string(nestedArraySchema(maxSchemaDepth+6)) +
		`,"node":{"$ref":"#/$defs/N"}},"$defs":{"N":{"type":"object","properties":{"next":{"$ref":"#/$defs/N"}}}}}`
	if _, err := JSONSchema([]byte(cyclic)); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Errorf("nesting written out past the cap, beside a reference cycle: got %v, want the nesting error", err)
	}
}

// TestSchema_refExpansionIsBounded: a family of definitions that all refer to each other is refused once its expansion
// passes the compiler's budget, instead of being compiled at every nesting level.
func TestSchema_refExpansionIsBounded(t *testing.T) {
	const n = 48
	var b strings.Builder
	b.WriteString(`{"$ref":"#/$defs/D0","$defs":{`)
	for i := range n {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, `"D%d":{"type":"object","properties":{`, i)
		for j := range n {
			if j > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `"p%d":{"$ref":"#/$defs/D%d"}`, j, j)
		}
		b.WriteString(`}}`)
	}
	b.WriteString(`}}`)
	_, err := JSONSchema([]byte(b.String()))
	if err == nil || !strings.Contains(err.Error(), "$ref expansions") {
		t.Errorf("%d definitions that each refer to all %d: got %v, want the expansion budget's error", n, n, err)
	}
}

// refGenSchemas are schemas with references whose constrained generations must conform (TestSchema_refGenerates).
func refGenSchemas(t *testing.T) map[string][]byte {
	t.Helper()
	m := map[string][]byte{
		"pydantic nested model":    pydanticSchema(t, "nested_model"),
		"pydantic recursive model": pydanticSchema(t, "recursive_model"),
		"optional self reference":  []byte(`{"$defs":{"N":{"type":"object","properties":{"v":{"type":"integer"},"next":{"$ref":"#/$defs/N"}},"required":["v"]}},"$ref":"#/$defs/N"}`),
		"root recursion":           []byte(`{"type":"object","properties":{"name":{"enum":["a","b"]},"kids":{"type":"array","items":{"$ref":"#"},"maxItems":2}},"required":["name"],"additionalProperties":false}`),
		"mutual recursion": []byte(`{"$defs":{"A":{"type":"object","properties":{"b":{"$ref":"#/$defs/B"},"k":{"type":"boolean"}},"required":["k"]},
			"B":{"type":"array","items":{"$ref":"#/$defs/A"}}},"$ref":"#/$defs/A"}`),
	}
	for _, p := range refPairs {
		m["pair/"+p.name] = []byte(p.ref)
	}
	return m
}

// TestSchema_refGenerates: constrained generation under a schema with references never dead-ends, including under a
// reference cycle, and what it produces conforms to the schema by the independent validator.
func TestSchema_refGenerates(t *testing.T) {
	t.Parallel()
	for name, schema := range refGenSchemas(t) {
		checkGenerates(t, name, schema, 150)
	}
}

// TestToolCallGrammar_refIsHeldToItsTarget: a tool parameter behind a `$ref` is held to the referenced schema, where it
// used to be any JSON value. A reference the compiler cannot follow still is.
func TestToolCallGrammar_refIsHeldToItsTarget(t *testing.T) {
	params := `{"type":"object","properties":{"to":{"$ref":"#/$defs/Addr"},"cc":{"$ref":"https://example.com/addr.json"}},"required":["to"],
		"$defs":{"Addr":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`
	g, err := ToolCallGrammar("", "", "arguments", "send", false, []byte(params))
	if err != nil {
		t.Fatal(err)
	}
	checkDocs(t, g, []docCase{
		{`{"name":"send","arguments":{"to":{"city":"Ely"}}}`, true, "a valid call"},
		{`{"name":"send","arguments":{"to":"Ely"}}`, false, "the referenced schema is an object"},
		{`{"name":"send","arguments":{"to":{"town":"Ely"}}}`, false, "the referenced schema declares city only"},
		{`{"name":"send","arguments":{"to":{"city":"Ely"},"cc":[1,{"a":null}]}}`, true, "a reference out of the document leaves the value free"},
	})
	// The recursion bound holds on the tool path too: a self-referential parameter schema compiles and nests to the cap.
	tree := `{"type":"object","properties":{"label":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#"}}},"required":["label"]}`
	tg, err := ToolCallGrammar("", "", "arguments", "plant", false, []byte(tree))
	if err != nil {
		t.Fatal(err)
	}
	nest := func(n int) string {
		return `{"name":"plant","arguments":` + strings.Repeat(`{"label":"","kids":[`, n-1) + `{"label":""}` + strings.Repeat(`]}`, n-1) + `}`
	}
	// The arguments sit one level down, so nodes fit at levels 1, 3, ... under the cap: as many as a bare schema's.
	fit := maxSchemaDepth / 2
	if !accepts(tg, nest(fit)) {
		t.Errorf("a tree %d deep was refused", fit)
	}
	if accepts(tg, nest(fit+1)) {
		t.Errorf("a tree %d deep, past the nesting cap, was accepted", fit+1)
	}
}

// oracleResolve is the validator's own reading of a local reference, written apart from the compiler's.
func oracleResolve(root map[string]any, ref string) (map[string]any, error) {
	var cur any = root
	if frag := strings.TrimPrefix(ref, "#"); frag != "" {
		for _, tok := range strings.Split(frag, "/")[1:] {
			tok = strings.NewReplacer("%20", " ", "~1", "/", "~0", "~").Replace(tok)
			switch v := cur.(type) {
			case map[string]any:
				cur = v[tok]
			case []any:
				i, err := strconv.Atoi(tok)
				if err != nil || i < 0 || i >= len(v) {
					return nil, fmt.Errorf("oracle: $ref %q: no index %q", ref, tok)
				}
				cur = v[i]
			default:
				return nil, fmt.Errorf("oracle: $ref %q: nothing at %q", ref, tok)
			}
		}
	}
	target, ok := cur.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("oracle: $ref %q does not name a schema", ref)
	}
	return target, nil
}
