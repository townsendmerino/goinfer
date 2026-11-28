package constrain

import (
	"fmt"
	"math/rand/v2"
	"os"
	"reflect"
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

// arraysOf is n arrays nested in each other around leaf. They have no minimum length, so a nest that is cut short is
// still a schema some document fits.
func arraysOf(n int, leaf string) string {
	for range n {
		leaf = `{"type":"array","items":` + leaf + `}`
	}
	return leaf
}

// TestSchema_refNestingBesideACycle: a reference cycle excuses the nesting it accounts for and no other. Nesting
// written out past the cap is the nesting error wherever it stands in a document that also has a cycle.
func TestSchema_refNestingBesideACycle(t *testing.T) {
	const cycle = `"N":{"type":"object","properties":{"next":{"$ref":"#/$defs/N"}}}`
	deep := arraysOf(maxSchemaDepth+6, `{"type":"string"}`)
	for _, c := range []struct{ name, schema string }{
		{"behind a reference of its own, compiled before the cycle",
			`{"type":"object","properties":{"deep":{"$ref":"#/$defs/Deep"},"node":{"$ref":"#/$defs/N"}},"$defs":{` + cycle + `,"Deep":` + deep + `}}`},
		{"behind a reference of its own, compiled after the cycle",
			`{"type":"object","properties":{"z":{"$ref":"#/$defs/Deep"},"a":{"$ref":"#/$defs/N"}},"$defs":{` + cycle + `,"Deep":` + deep + `}}`},
		{"in the definition the root refers to, beside a property that recurses elsewhere",
			`{"$ref":"#/$defs/Big","$defs":{"Big":{"type":"object","properties":{"deep":` + deep + `,"node":{"$ref":"#/$defs/N"}}},` + cycle + `}}`},
		{"as a property of the recursive definition",
			`{"$ref":"#/$defs/R","$defs":{"R":{"type":"object","properties":{"next":{"$ref":"#/$defs/R"},"deep":` + deep + `}}}}`},
		{"as a property one level inside the cycle",
			`{"$ref":"#/$defs/R","$defs":{"R":{"type":"object","properties":{"child":{"type":"object","properties":{"sub":{"$ref":"#/$defs/R"},"deep":` + deep + `}}}}}}`},
		{"in the second of two definitions that refer to each other",
			`{"$ref":"#/$defs/A","$defs":{"A":{"type":"object","properties":{"b":{"$ref":"#/$defs/B"}}},"B":{"type":"object","properties":{"a":{"$ref":"#/$defs/A"},"deep":` + deep + `}}}}`},
		{"as the cycle's own body, behind a reference",
			`{"$ref":"#/$defs/R","$defs":{"R":` + arraysOf(maxSchemaDepth+6, `{"$ref":"#/$defs/R"}`) + `}}`},
		{"as the cycle's own body, at the root", arraysOf(maxSchemaDepth+6, `{"$ref":"#"}`)},
		{"as the body of a definition the document does not enter the cycle by",
			`{"$ref":"#/$defs/A","$defs":{"A":{"type":"object","properties":{"b":{"$ref":"#/$defs/B"}}},"B":` + arraysOf(maxSchemaDepth+6, `{"$ref":"#/$defs/A"}`) + `}}`},
		{"as a recursive body the document enters part of the way down",
			`{"$ref":"#/$defs/R` + strings.Repeat(`/items`, 10) + `","$defs":{"R":` + arraysOf(maxSchemaDepth+6, `{"$ref":"#/$defs/R"}`) + `}}`},
	} {
		if _, err := JSONSchema([]byte(c.schema)); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
			t.Errorf("nesting written out past the cap, %s: got %v, want the nesting error", c.name, err)
		}
	}
	// The count is the inlined form's: under a property, a definition compiles while its leaf sits within the cap.
	for _, levels := range []int{maxSchemaDepth - 1, maxSchemaDepth} {
		nest := arraysOf(levels, `{"type":"string"}`)
		_, inlined := JSONSchema([]byte(`{"type":"object","properties":{"deep":` + nest + `}}`))
		_, err := JSONSchema([]byte(`{"type":"object","properties":{"deep":{"$ref":"#/$defs/Deep"},"node":{"$ref":"#/$defs/N"}},"$defs":{` + cycle + `,"Deep":` + nest + `}}`))
		if (err == nil) != (inlined == nil) || (inlined == nil) != (levels < maxSchemaDepth) {
			t.Errorf("%d arrays under a property: beside a cycle %v, written in place %v; want both to compile below %d arrays only", levels, err, inlined, maxSchemaDepth)
		}
	}
	// And it starts at the level the document itself is compiled at.
	const base = 3
	for _, levels := range []int{maxSchemaDepth - base - 1, maxSchemaDepth - base} {
		doc := mustMap(`{"type":"object","properties":{"deep":{"$ref":"#/$defs/Deep"},"node":{"$ref":"#/$defs/N"}},"$defs":{` + cycle + `,"Deep":` + arraysOf(levels, `{"type":"string"}`) + `}}`)
		if _, err := compileDocument(doc, false, base); (err == nil) != (levels < maxSchemaDepth-base) {
			t.Errorf("%d arrays under a property of a document compiled at level %d: %v; want it to compile below %d arrays only", levels, base, err, maxSchemaDepth-base)
		}
	}
}

// TestSchema_refNestingOwedToACycle: nesting a cycle accounts for is cut, not refused, also where the recursive
// definition holds nesting of its own that fits. The document then nests to the cap and no deeper, whichever of the
// two the levels come from.
func TestSchema_refNestingOwedToACycle(t *testing.T) {
	const member = 40
	g := mustSchema(t, []byte(`{"$ref":"#/$defs/N","$defs":{"N":{"type":"object","properties":{"next":{"$ref":"#/$defs/N"},"deep":`+
		arraysOf(member, `{"type":"string"}`)+`}}}}`))
	// doc is nodes objects nested through "next", the innermost holding arrays arrays nested in "deep".
	doc := func(nodes, arrays int) string {
		return strings.Repeat(`{"next":`, nodes-1) + `{"deep":` + strings.Repeat(`[`, arrays) + strings.Repeat(`]`, arrays) + `}` + strings.Repeat(`}`, nodes-1)
	}
	const nodes = 30
	fit := maxSchemaDepth + 1 - nodes // the innermost node is at level nodes-1, and its arrays go down to the cap
	checkDocs(t, g, []docCase{
		{doc(1, member), true, "the member written out in full in the outermost node"},
		{doc(1, member+1), false, "the member has no more arrays than the schema writes"},
		{doc(nodes, fit), true, "the member of a nested node, down to the nesting cap"},
		{doc(nodes, fit+1), false, "one level past the nesting cap"},
	})
	// What the count leaves out: two definitions that each write out 40 levels and end in a reference to the other
	// nest 80 levels on their first trip round, but that trip goes through the cycle's own references, so it is cut
	// as any recursion is.
	half := func(other string) string { return arraysOf(member, `{"$ref":"#/$defs/`+other+`"}`) }
	if _, err := JSONSchema([]byte(`{"$ref":"#/$defs/A","$defs":{"A":` + half("B") + `,"B":` + half("A") + `}}`)); err != nil {
		t.Errorf("two definitions of %d levels each that refer to each other: %v, want the cycle cut", member, err)
	}
}

// TestSchema_refNestingCountsWhatIsCompiled: the count reads a schema as the compiler does, so nesting under a keyword
// the compiler does not read for that schema is not counted against a recursive document.
func TestSchema_refNestingCountsWhatIsCompiled(t *testing.T) {
	deep := arraysOf(maxSchemaDepth+6, `{"type":"string"}`)
	recursive := func(extra string) string {
		return `{"$ref":"#/$defs/N","$defs":{"N":{"type":"object","properties":{"next":{"$ref":"#/$defs/N"},"x":` + extra + `}}}}`
	}
	for name, unread := range map[string]string{
		"properties of a string":           `{"type":"string","properties":{"deep":` + deep + `}}`,
		"items of an object":               `{"type":"object","properties":{"k":{"type":"boolean"}},"items":` + deep + `}`,
		"properties beside an enum":        `{"enum":["a","b"],"properties":{"deep":` + deep + `}}`,
		"items beside a const":             `{"const":1,"items":` + deep + `}`,
		"items of a list with no array":    `{"type":["string","null"],"items":` + deep + `}`,
		"properties of a list, no object":  `{"type":["integer","null"],"properties":{"deep":` + deep + `}}`,
		"items with an empty type beside":  `{"type":"","properties":{"k":{"type":"boolean"}},"items":` + deep + `}`,
		"a definition nothing refers to":   `{"type":"boolean","$defs":{"Unused":` + deep + `}}`,
		"a type list with a schema in it":  `{"enum":[1],"type":[{"deep":` + deep + `}]}`,
		"properties that are not a schema": `{"type":"object","properties":{"k":{"type":"boolean"},"n":{"enum":[[[[1]]]]}}}`,
	} {
		if _, err := JSONSchema([]byte(recursive(unread))); err != nil {
			t.Errorf("a recursive document with nesting the compiler does not read (%s): %v", name, err)
		}
	}
}

// TestSchema_refShapeOnRandomDocuments: on small random documents, refShape is what its definition gives when read off
// directly, with a cycle found by following every path, and in a document without a cycle plain is the nesting of the
// deepest path.
func TestSchema_refShapeOnRandomDocuments(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 14))
	const rounds = 500
	cyclicDocs := 0
	for range rounds {
		defs := 1 + rng.IntN(4)
		var gen func(level int) string
		gen = func(level int) string {
			switch pick := rng.IntN(7); {
			case pick == 0:
				return `{"type":"integer"}`
			case pick <= 2 && level < 4:
				return `{"type":"array","items":` + gen(level+1) + `}`
			case pick == 3 && level < 4:
				return `{"type":"object","properties":{"p":` + gen(level+1) + `,"q":` + gen(level+1) + `}}`
			case pick == 4 && level < 4:
				return `{"anyOf":[` + gen(level+1) + `,{"type":"null"}]}`
			}
			return fmt.Sprintf(`{"$ref":"#/$defs/D%d"}`, rng.IntN(defs))
		}
		var b strings.Builder
		b.WriteString(`{"type":"object","properties":{"r":` + gen(1) + `},"$defs":{`)
		for i := range defs {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `"D%d":%s`, i, gen(1))
		}
		b.WriteString(`}}`)
		r := newRefState(mustMap(b.String()), false, 0)
		sh, err := r.readShape()
		if err != nil {
			t.Fatal(err)
		}
		id := func(s map[string]any) uintptr { return reflect.ValueOf(s).Pointer() }
		levels := func(kid refKid) int {
			if kid.nested {
				return 1
			}
			return 0
		}
		// leads[{a, b}]: a path of one edge or more goes from schema a to schema b.
		schemas := map[uintptr]map[string]any{}
		var collect func(s map[string]any)
		collect = func(s map[string]any) {
			if schemas[id(s)] == nil {
				schemas[id(s)] = s
				for _, kid := range r.below(s) {
					collect(kid.s)
				}
			}
		}
		collect(r.root)
		leads := map[[2]uintptr]bool{}
		for a, s := range schemas {
			var follow func(s map[string]any)
			follow = func(s map[string]any) {
				for _, kid := range r.below(s) {
					if to := [2]uintptr{a, id(kid.s)}; !leads[to] {
						leads[to] = true
						follow(kid.s)
					}
				}
			}
			follow(s)
		}
		sameCycle := func(a, b uintptr) bool { return a == b || leads[[2]uintptr{a, b}] && leads[[2]uintptr{b, a}] }
		// written is the nesting below s with none of its own cycle's references followed, and counted the deepest
		// written among the schemas of s's cycle, which is what refShape.plain is defined as from s down.
		writtenMemo, countedMemo := map[uintptr]int{}, map[uintptr]int{}
		var written, counted func(s map[string]any) int
		written = func(s map[string]any) int {
			if deep, ok := writtenMemo[id(s)]; ok {
				return deep
			}
			deep := 0
			for _, kid := range r.below(s) {
				switch {
				case !sameCycle(id(s), id(kid.s)):
					deep = max(deep, levels(kid)+counted(kid.s))
				case !kid.ref:
					deep = max(deep, levels(kid)+written(kid.s))
				}
			}
			writtenMemo[id(s)] = deep
			return deep
		}
		counted = func(s map[string]any) int {
			if deep, ok := countedMemo[id(s)]; ok {
				return deep
			}
			deep := 0
			for other, m := range schemas {
				if sameCycle(id(s), other) {
					deep = max(deep, written(m))
				}
			}
			countedMemo[id(s)] = deep
			return deep
		}
		anyCycle := false
		for a := range schemas {
			anyCycle = anyCycle || leads[[2]uintptr{a, a}]
		}
		if sh.cyclic != anyCycle {
			t.Fatalf("cyclic is %v, and following every path finds a cycle: %v\n%s", sh.cyclic, anyCycle, b.String())
		}
		if want := counted(r.root); sh.plain != want {
			t.Fatalf("plain is %d, its definition read directly gives %d\n%s", sh.plain, want, b.String())
		}
		if anyCycle {
			cyclicDocs++
			continue
		}
		// longest is the deepest nesting a path from s reaches.
		var longest func(s map[string]any) int
		longest = func(s map[string]any) int {
			deep := 0
			for _, kid := range r.below(s) {
				deep = max(deep, levels(kid)+longest(kid.s))
			}
			return deep
		}
		if want := longest(r.root); sh.plain != want {
			t.Fatalf("no cycle: plain is %d, the deepest path nests %d\n%s", sh.plain, want, b.String())
		}
	}
	if cyclicDocs < rounds/5 || cyclicDocs > rounds*4/5 {
		t.Errorf("%d of %d documents had a cycle: the test wants both kinds", cyclicDocs, rounds)
	}
}

// TestSchema_refShapeOfASchemaThatHoldsItself: nesting with no reference in it cannot go round in a document read from
// JSON, but a schema built in Go can hold itself. The count then ends, in the nesting error.
func TestSchema_refShapeOfASchemaThatHoldsItself(t *testing.T) {
	self := map[string]any{"type": "object"}
	self["properties"] = map[string]any{"self": self, "next": map[string]any{"$ref": "#/$defs/R"}}
	doc := map[string]any{"$ref": "#/$defs/R", "$defs": map[string]any{"R": self}}
	if _, err := compileDocument(doc, false, 0); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Errorf("a schema that holds itself: got %v, want the nesting error", err)
	}
}

// TestSchema_refNestingIgnoresAnUnrelatedCycle: whether a chain of definitions without a cycle compiles does not change
// when a recursive definition is put beside it or at its end, whichever the compiler reaches first.
func TestSchema_refNestingIgnoresAnUnrelatedCycle(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 27))
	const rounds = 300
	refused := 0
	for range rounds {
		// The chain: definitions D0, D1, ..., each some arrays around a reference to the next, the last around leaf.
		total := maxSchemaDepth - 5 + rng.IntN(10)
		var plain, cyclic strings.Builder
		for i, left := 0, total; left > 0; i++ {
			n := 1 + rng.IntN(left)
			if left -= n; left > 0 {
				fmt.Fprintf(&plain, `"D%d":%s,`, i, arraysOf(n, fmt.Sprintf(`{"$ref":"#/$defs/D%d"}`, i+1)))
			} else {
				fmt.Fprintf(&plain, `"D%d":%s,`, i, arraysOf(n, `{"$ref":"#/$defs/Leaf"}`))
			}
		}
		cyclic.WriteString(plain.String())
		const leaf = `"Leaf":{"type":"object","properties":{"v":{"type":"integer"}}}`
		const node = `"N":{"type":"object","properties":{"next":{"$ref":"#/$defs/N"}}}`
		props := `"deep":{"$ref":"#/$defs/D0"}`
		plainDoc := `{"type":"object","properties":{` + props + `},"$defs":{` + plain.String() + leaf + `}}`
		switch rng.IntN(3) {
		case 0: // the cycle beside the chain, compiled before it
			props = `"a":{"$ref":"#/$defs/N"},` + props
			cyclic.WriteString(leaf + `,` + node)
		case 1: // beside it, compiled after
			props += `,"z":{"$ref":"#/$defs/N"}`
			cyclic.WriteString(leaf + `,` + node)
		default: // the chain's own leaf recurses
			cyclic.WriteString(`"Leaf":{"type":"object","properties":{"v":{"type":"integer"},"again":{"$ref":"#/$defs/Leaf"}}}`)
		}
		cyclicDoc := `{"type":"object","properties":{` + props + `},"$defs":{` + cyclic.String() + `}}`
		_, errPlain := JSONSchema([]byte(plainDoc))
		_, errCyclic := JSONSchema([]byte(cyclicDoc))
		if (errPlain == nil) != (errCyclic == nil) {
			t.Fatalf("a chain of %d arrays: without a cycle %v, with one %v\n%s", total, errPlain, errCyclic, cyclicDoc)
		}
		if errPlain != nil {
			refused++
		}
	}
	if refused < rounds/10 || refused > rounds*9/10 {
		t.Errorf("%d of %d chains were refused: the chains do not straddle the nesting cap", refused, rounds)
	}
}

// TestSchema_refCyclesAreNeverTheNestingError: definitions that refer to each other at random, none writing out more
// than a few levels of its own, never fail with the nesting error, whatever else stops one compiling.
func TestSchema_refCyclesAreNeverTheNestingError(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 8))
	const rounds = 400
	compiled := 0
	for range rounds {
		defs := 1 + rng.IntN(5)
		var gen func(level int) string
		gen = func(level int) string {
			switch pick := rng.IntN(10); {
			case pick == 0:
				return `{"type":"integer"}`
			case pick == 1:
				return fmt.Sprintf(`{"anyOf":[{"$ref":"#/$defs/D%d"},{"type":"null"}]}`, rng.IntN(defs))
			case pick == 2:
				return fmt.Sprintf(`{"oneOf":[{"type":"null"},{"$ref":"#/$defs/D%d"}]}`, rng.IntN(defs))
			case pick == 3 && level < 3:
				return `{"type":"array","items":` + gen(level+1) + `}`
			case pick == 4 && level < 3:
				return `{"type":["null","array"],"items":` + gen(level+1) + `}`
			case pick == 5 && level < 3:
				return `{"type":"object","properties":{"p":` + gen(level+1) + `,"q":` + gen(level+1) + `}}`
			case pick == 6 && level < 3:
				return `{"type":["object","null"],"properties":{"p":` + gen(level+1) + `}}`
			case pick == 7 && level < 3:
				return `{"properties":{"p":` + gen(level+1) + `}}`
			}
			return fmt.Sprintf(`{"$ref":"#/$defs/D%d"}`, rng.IntN(defs))
		}
		var b strings.Builder
		b.WriteString(`{"$ref":"#/$defs/D0","$defs":{`)
		for i := range defs {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `"D%d":{"type":"object","properties":{"k":{"type":"boolean"},"x":%s,"y":%s}}`, i, gen(1), gen(1))
		}
		b.WriteString(`}}`)
		switch _, err := JSONSchema([]byte(b.String())); {
		case err == nil:
			compiled++
		case strings.Contains(err.Error(), "nesting"):
			t.Fatalf("%v\n%s", err, b.String())
		}
	}
	if compiled < rounds/2 {
		t.Errorf("%d of %d random recursive documents compiled; the test wants most of them to", compiled, rounds)
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

// TestToolCallGrammar_refNestingBesideACycle: a tool's parameters that write out nesting past the cap beside a reference
// cycle are refused by the strict compile and keep their grammar in the lenient one, cut at the cap: the arguments stay
// held to the schema, and the nesting goes down to the cap and no deeper.
func TestToolCallGrammar_refNestingBesideACycle(t *testing.T) {
	params := `{"type":"object","properties":{"label":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#"}},"deep":{"$ref":"#/$defs/Deep"}},
		"required":["label"],"$defs":{"Deep":` + arraysOf(maxSchemaDepth+6, `{"type":"string"}`) + `}}`
	if _, err := JSONSchema([]byte(params)); err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Fatalf("the strict compile of the parameters: got %v, want the nesting error", err)
	}
	g, err := ToolCallGrammar("", "", "arguments", "plant", false, []byte(params))
	if err != nil {
		t.Fatal(err)
	}
	// The arguments are at level 1 and "deep" at level 2, so its arrays fit at levels 2 to the cap.
	fit := maxSchemaDepth - 1
	call := func(arrays int) string {
		return `{"name":"plant","arguments":{"label":"a","deep":` + strings.Repeat(`[`, arrays) + strings.Repeat(`]`, arrays) + `}}`
	}
	checkDocs(t, g, []docCase{
		{`{"name":"plant","arguments":{"label":"a","kids":[{"label":"b"}]}}`, true, "a valid call"},
		{`{"name":"plant","arguments":{"label":1}}`, false, "the arguments are still held to their schema"},
		{call(fit), true, "the written-out nesting, down to the nesting cap"},
		{call(fit + 1), false, "one level past the nesting cap"},
	})
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
