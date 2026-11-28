package constrain

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

// A discriminated union as Pydantic writes it: oneOf over two models, each with a const tag, and a `discriminator`.
func TestSchema_variantPydanticDiscriminatedUnion(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "discriminated_union"))
	checkDocs(t, g, []docCase{
		{`{"pet":{"pet_type":"cat","meows":3},"n":1}`, true, "a cat"},
		{`{"pet":{"pet_type":"dog","barks":1.5},"n":1}`, true, "a dog"},
		{`{"n":1,"pet":{"barks":2,"pet_type":"dog"}}`, true, "the tag may come after the keys it explains"},
		{`{"pet":{"meows":3,"pet_type":"cat"},"n":1}`, true, "a cat, tag last"},
		{`{"pet":{"pet_type":"cat","barks":1.5},"n":1}`, false, "a cat does not bark"},
		{`{"pet":{"pet_type":"dog","meows":3},"n":1}`, false, "a dog does not meow"},
		{`{"pet":{"meows":3,"pet_type":"dog"},"n":1}`, false, "after meows the tag is cat"},
		{`{"pet":{"barks":1,"pet_type":"cat"},"n":1}`, false, "after barks the tag is dog"},
		{`{"pet":{"pet_type":"cat"},"n":1}`, false, "a cat requires meows"},
		{`{"pet":{"meows":3},"n":1}`, false, "a cat requires its tag"},
		{`{"pet":{"pet_type":"bird","meows":1},"n":1}`, false, "no such tag"},
		{`{"pet":{"meows":1.5,"pet_type":"cat"},"n":1}`, false, "meows is an integer"},
		{`{"pet":{"pet_type":"cat","meows":3,"barks":1},"n":1}`, false, "a cat's keys only"},
		{`{"pet":{"pet_type":"cat","pet_type":"cat","meows":3},"n":1}`, false, "a key once"},
		{`{"pet":{},"n":1}`, false, "neither model is empty"},
		{`{"pet":null,"n":1}`, false, "pet is not optional"},
	})
}

// The same union as an array's items: every item picks its own alternative.
func TestSchema_variantPydanticUnionItems(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "union_items"))
	checkDocs(t, g, []docCase{
		{`{"pets":[]}`, true, "no pets"},
		{`{"pets":[{"pet_type":"cat","meows":1},{"pet_type":"dog","barks":2},{"meows":0,"pet_type":"cat"}]}`, true, "a cat, a dog, a cat"},
		{`{"pets":[{"pet_type":"cat","meows":1},{"pet_type":"cat","barks":2}]}`, false, "the second item is neither"},
		{`{"pets":[{"pet_type":"dog","barks":2},{"pet_type":"dog"}]}`, false, "the second dog has no barks"},
	})
}

// A union with no tag keyword: anyOf over the same two models. The keys and the const still tell them apart.
func TestSchema_variantPydanticUntaggedUnion(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "untagged_union"))
	checkDocs(t, g, []docCase{
		{`{"pet":{"pet_type":"cat","meows":3}}`, true, "a cat"},
		{`{"pet":{"barks":0.5,"pet_type":"dog"}}`, true, "a dog"},
		{`{"pet":{"pet_type":"dog","meows":3}}`, false, "a dog does not meow"},
		{`{"pet":{"pet_type":"cat","meows":3,"barks":1}}`, false, "one model's keys"},
	})
}

const (
	objA = `{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`
	objB = `{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}`
)

// variantCases are unions of objects, with documents each must accept and refuse.
var variantCases = []struct {
	name, schema string
	yes, no      []string
}{
	{"told apart by their keys", `{"anyOf":[` + objA + `,` + objB + `]}`,
		[]string{`{"a":1}`, `{"b":"x"}`}, []string{`{}`, `{"a":1,"b":"x"}`, `{"a":"x"}`, `{"b":1}`, `null`}},
	{"a shared key with one schema", `{"anyOf":[
		{"type":"object","properties":{"id":{"type":"integer"},"a":{"type":"string"}},"required":["id"]},
		{"type":"object","properties":{"id":{"type":"integer"},"b":{"type":"boolean"}},"required":["id","b"]}]}`,
		[]string{`{"id":1}`, `{"id":1,"a":"x"}`, `{"b":true,"id":2}`}, []string{`{"id":1,"a":"x","b":true}`, `{"b":true}`, `{"id":"1"}`}},
	{"one alternative's keys are a subset of the other's", `{"anyOf":[
		{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},
		{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]}]}`,
		[]string{`{"a":1}`, `{"a":1,"b":"x"}`, `{"b":"x","a":1}`}, []string{`{"b":"x"}`, `{}`}},
	{"a tag with several literals", `{"anyOf":[
		{"type":"object","properties":{"kind":{"enum":["a1","a2"]},"x":{"type":"integer"}},"required":["kind","x"]},
		{"type":"object","properties":{"kind":{"const":"b"},"y":{"type":"boolean"}},"required":["kind","y"]}]}`,
		[]string{`{"kind":"a1","x":1}`, `{"kind":"a2","x":1}`, `{"kind":"b","y":true}`, `{"y":false,"kind":"b"}`},
		[]string{`{"kind":"a1","y":true}`, `{"kind":"b","x":1}`, `{"x":1,"kind":"b"}`, `{"kind":"a3","x":1}`}},
	{"a key typed differently", `{"anyOf":[
		{"type":"object","properties":{"v":{"type":"integer"},"t":{"const":"i"}},"required":["v","t"]},
		{"type":"object","properties":{"v":{"type":"string"},"t":{"const":"s"}},"required":["v","t"]}]}`,
		[]string{`{"v":1,"t":"i"}`, `{"t":"s","v":"x"}`, `{"v":"x","t":"s"}`}, []string{`{"v":1,"t":"s"}`, `{"t":"i","v":"x"}`, `{"v":"x","t":"i"}`, `{"v":null,"t":"i"}`}},
	{"a key typed differently, and no tag", `{"anyOf":[
		{"type":"object","properties":{"v":{"type":"integer"}},"required":["v"]},
		{"type":"object","properties":{"v":{"type":"string"},"w":{"type":"null"}},"required":["v"]}]}`,
		[]string{`{"v":1}`, `{"v":"x","w":null}`, `{"v":"x"}`, `{"w":null,"v":"x"}`}, []string{`{"v":1,"w":null}`, `{"w":null,"v":1}`, `{"w":null}`}},
	{"objects beside a string and null", `{"anyOf":[` + objA + `,{"type":"null"},` + objB + `,{"type":"string"}]}`,
		[]string{`{"a":1}`, `{"b":"x"}`, `null`, `"s"`}, []string{`{}`, `1`, `{"a":1,"b":"x"}`}},
	{"a key that is a different object in each", `{"anyOf":[
		{"type":"object","properties":{"kind":{"const":"a"},"data":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}},"required":["kind","data"]},
		{"type":"object","properties":{"kind":{"const":"b"},"data":{"type":"object","properties":{"y":{"type":"string"}},"required":["y"]}},"required":["kind","data"]}]}`,
		[]string{`{"kind":"a","data":{"x":1}}`, `{"data":{"x":1},"kind":"a"}`, `{"kind":"b","data":{"y":"s"}}`, `{"data":{"y":"s"},"kind":"b"}`},
		[]string{`{"data":{"x":1},"kind":"b"}`, `{"kind":"a","data":{"y":"s"}}`, `{"kind":"b","data":{"x":1}}`, `{"kind":"a","data":{}}`}},
	{"a key that is a union in one alternative", `{"anyOf":[
		{"type":"object","properties":{"v":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["v"]},
		{"type":"object","properties":{"v":{"type":"string"},"b":{"type":"boolean"}},"required":["v","b"]}]}`,
		[]string{`{"v":1}`, `{"v":null}`, `{"v":"x","b":true}`, `{"b":false,"v":"x"}`}, []string{`{"v":"x"}`, `{"v":1,"b":true}`, `{"b":true,"v":null}`}},
	{"a literal two alternatives share", `{"anyOf":[
		{"type":"object","properties":{"k":{"enum":["x","y"]}},"required":["k"]},
		{"type":"object","properties":{"k":{"enum":["y","z"]},"b":{"type":"integer"}},"required":["k","b"]},
		{"type":"object","properties":{"k":{"const":"z"},"c":{"type":"integer"}},"required":["k","c"]}]}`,
		[]string{`{"k":"x"}`, `{"k":"y"}`, `{"k":"y","b":1}`, `{"k":"z","b":1}`, `{"k":"z","c":1}`, `{"c":1,"k":"z"}`},
		[]string{`{"k":"z"}`, `{"k":"x","b":1}`, `{"k":"y","c":1}`, `{"c":1,"k":"y"}`, `{"k":"z","b":1,"c":1}`}},
	{"tags of four types", `{"anyOf":[
		{"type":"object","properties":{"k":{"const":1},"a":{"type":"null"}},"required":["k","a"]},
		{"type":"object","properties":{"k":{"const":"1"},"b":{"type":"null"}},"required":["k","b"]},
		{"type":"object","properties":{"k":{"const":true},"c":{"type":"null"}},"required":["k","c"]},
		{"type":"object","properties":{"k":{"const":null},"d":{"type":"null"}},"required":["k","d"]}]}`,
		[]string{`{"k":1,"a":null}`, `{"k":"1","b":null}`, `{"k":true,"c":null}`, `{"k":null,"d":null}`, `{"d":null,"k":null}`},
		[]string{`{"k":1,"b":null}`, `{"k":"1","a":null}`, `{"k":true,"d":null}`, `{"a":null,"k":"1"}`, `{"k":12,"a":null}`}},
	{"an empty object beside a full one", `{"anyOf":[{"type":"object","additionalProperties":false},` + objA + `]}`,
		[]string{`{}`, `{"a":1}`}, []string{`{"b":1}`, `[]`}},
	{"a negative literal beside a non-negative number", `{"anyOf":[
		{"type":"object","properties":{"v":{"const":-1}},"required":["v"]},
		{"type":"object","properties":{"v":{"type":"integer","minimum":0},"b":{"type":"boolean"}},"required":["v","b"]}]}`,
		[]string{`{"v":-1}`, `{"v":0,"b":true}`, `{"b":true,"v":7}`}, []string{`{"v":-1,"b":true}`, `{"v":0}`, `{"b":true,"v":-1}`, `{"v":-2}`}},
	{"oneOf with a discriminator, through $ref", `{"$defs":{
		"Circle":{"type":"object","properties":{"shape":{"const":"circle"},"r":{"type":"number"}},"required":["shape","r"]},
		"Rect":{"type":"object","properties":{"shape":{"const":"rect"},"w":{"type":"number"},"h":{"type":"number"}},"required":["shape","w","h"]}},
		"oneOf":[{"$ref":"#/$defs/Circle"},{"$ref":"#/$defs/Rect"}],
		"discriminator":{"propertyName":"shape","mapping":{"circle":"#/$defs/Circle","rect":"#/$defs/Rect"}}}`,
		[]string{`{"shape":"circle","r":1.5}`, `{"w":1,"h":2,"shape":"rect"}`}, []string{`{"shape":"circle","w":1,"h":2}`, `{"shape":"rect","w":1}`, `{"r":1,"shape":"rect"}`}},
	{"oneOf told apart by a required key", `{"oneOf":[` + objA + `,` + objB + `]}`,
		[]string{`{"a":1}`, `{"b":"x"}`}, []string{`{}`, `{"a":1,"b":"x"}`}},
	{"a union of objects inside a union of objects", `{"anyOf":[
		{"type":"object","properties":{"t":{"const":"leaf"},"v":{"type":"integer"}},"required":["t","v"]},
		{"type":"object","properties":{"t":{"const":"pair"},"l":{"anyOf":[` + objA + `,` + objB + `]}},"required":["t","l"]}]}`,
		[]string{`{"t":"leaf","v":1}`, `{"t":"pair","l":{"a":1}}`, `{"l":{"b":"x"},"t":"pair"}`}, []string{`{"t":"leaf","l":{"a":1}}`, `{"t":"pair","l":{}}`, `{"l":{"a":1},"t":"leaf"}`}},
}

func TestSchema_variantCases(t *testing.T) {
	for _, c := range variantCases {
		g, err := JSONSchema([]byte(c.schema))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		for _, doc := range c.yes {
			if !accepts(g, doc) {
				t.Errorf("%s: %s refused", c.name, doc)
			}
		}
		for _, doc := range c.no {
			if accepts(g, doc) {
				t.Errorf("%s: %s accepted", c.name, doc)
			}
		}
	}
}

// TestSchema_variantIsItsAlternatives: a union of objects allows, byte for byte, what its alternatives compiled one
// by one and run side by side allow.
func TestSchema_variantIsItsAlternatives(t *testing.T) {
	for _, c := range variantCases {
		checkUnionOfAlternatives(t, c.name, []byte(c.schema))
	}
	for _, name := range []string{"discriminated_union", "untagged_union"} {
		// The union is the `pet` property: compare it as a document of its own.
		checkUnionOfAlternatives(t, "pydantic "+name, hoistProperty(t, pydanticSchema(t, name), "pet"))
	}
}

// hoistProperty returns the schema of one property of a root object as a document, with the root's definitions.
func hoistProperty(t *testing.T, schema []byte, name string) []byte {
	t.Helper()
	root := mustMap(string(schema))
	prop := maps.Clone(root["properties"].(map[string]any)[name].(map[string]any))
	prop["$defs"] = root["$defs"]
	raw, err := json.Marshal(prop)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// TestSchema_variantGenerates: constrained generation under a union of objects never dead-ends, and what it produces
// conforms to the schema by the independent validator.
func TestSchema_variantGenerates(t *testing.T) {
	t.Parallel()
	schemas := map[string][]byte{
		"pydantic discriminated union": pydanticSchema(t, "discriminated_union"),
		"pydantic union items":         pydanticSchema(t, "union_items"),
		"pydantic untagged union":      pydanticSchema(t, "untagged_union"),
	}
	for _, c := range variantCases {
		schemas["case/"+c.name] = []byte(c.schema)
	}
	for _, name := range slices.Sorted(maps.Keys(schemas)) {
		checkGenerates(t, name, schemas[name], 120)
	}
}

// TestSchema_variantErrors: a union of objects the grammar cannot follow names the property in the way.
func TestSchema_variantErrors(t *testing.T) {
	two := func(a, b string) string {
		return `{"anyOf":[{"type":"object","properties":{"v":` + a + `},"required":["v"]},{"type":"object","properties":{"v":` + b + `,"w":{"type":"null"}},"required":["v"]}]}`
	}
	var wide strings.Builder
	wide.WriteString(`{"anyOf":[`)
	for half := range 2 {
		wide.WriteString(`{"type":"object","properties":{`)
		for i := range 33 {
			fmt.Fprintf(&wide, `"p%d_%d":{"type":"null"},`, half, i)
		}
		wide.WriteString(`"z":{"type":"null"}}}`)
		if half == 0 {
			wide.WriteByte(',')
		}
	}
	wide.WriteString(`]}`)
	for _, c := range []struct{ schema, want string }{
		{two(`{"type":"array","items":{"type":"integer"}}`, `{"type":"array","items":{"type":"string"}}`), `property "v" is an array in alternatives 0 and 1`},
		{two(`{"type":"integer"}`, `{"type":"number"}`), `property "v" is a number in alternatives 0 and 1`},
		{two(`{"type":"string","format":"date"}`, `{"type":"string"}`), `property "v" is a string in alternatives 0 and 1`},
		{two(`{"const":"x"}`, `{"type":"string"}`), `property "v"`},
		{two(`{"const":1}`, `{"type":"integer"}`), `property "v"`},
		{wide.String(), "more than 64 properties"},
		{manyObjects(33, 2), "more than 64 object alternatives"},
		{`{"oneOf":[{"type":"object","properties":{"a":{"type":"integer"}}},{"type":"object","properties":{"b":{"type":"integer"}}}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[` + objA + `,{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[
			{"type":"object","properties":{"k":{"enum":["x","y"]}},"required":["k"]},
			{"type":"object","properties":{"k":{"enum":["y","z"]}},"required":["k"]}]}`, "oneOf alternatives 0 and 1 can both match"},
	} {
		_, err := JSONSchema([]byte(c.schema))
		show := c.schema
		if len(show) > 160 {
			show = show[:160] + "…"
		}
		if err == nil {
			t.Errorf("%s compiled; want an error naming %s", show, c.want)
		} else if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not name %s", show, err, c.want)
		}
	}
}

// manyObjects is an anyOf of groups alternatives, each an anyOf of per objects that differ in their one key.
func manyObjects(per, groups int) string {
	var b strings.Builder
	b.WriteString(`{"anyOf":[`)
	for g := range groups {
		if g > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"anyOf":[`)
		for i := range per {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `{"type":"object","properties":{"k%d_%d":{"type":"null"}},"required":["k%d_%d"]}`, g, i, g, i)
		}
		b.WriteString(`]}`)
	}
	b.WriteString(`]}`)
	return b.String()
}

// TestSchema_unionWorkIsBounded: a oneOf whose alternatives take a great many comparisons to tell apart is refused
// by a count, not compiled for as long as it takes. Here 64 objects share 63 required nested objects, each of 17
// required keys, and differ in a last key.
func TestSchema_unionWorkIsBounded(t *testing.T) {
	object := func(n int, name string, value func(i int) string) string {
		var props, required []string
		for i := range n {
			key := fmt.Sprintf("%s%02d", name, i)
			props = append(props, strconv.Quote(key)+":"+value(i))
			required = append(required, strconv.Quote(key))
		}
		return `{"type":"object","properties":{` + strings.Join(props, ",") + `},"required":[` + strings.Join(required, ",") + `]}`
	}
	union := func(inner string) string {
		alts := make([]string, 64)
		for alt := range alts {
			alts[alt] = object(64, "p", func(i int) string {
				if i == 63 {
					return fmt.Sprintf(`{"const":%d}`, alt)
				}
				return inner
			})
		}
		return `{"oneOf":[` + strings.Join(alts, ",") + `]}`
	}
	null := func(int) string { return `{"type":"null"}` }
	_, err := JSONSchema([]byte(union(object(17, "q", null))))
	if err == nil || !strings.Contains(err.Error(), "comparisons") {
		t.Fatalf("got %v, want the refusal by comparison count", err)
	}
	// The same union with small alternatives is well inside the count.
	if _, err := JSONSchema([]byte(union(`{"type":"null"}`))); err != nil {
		t.Fatalf("64 flat alternatives: %v", err)
	}
}

// TestSchema_variantOneOfExclusive: a oneOf over objects compiles when each pair is shown apart, by a tag whose
// literals differ, a key one requires and the other does not declare, or a shared required key of different types.
func TestSchema_variantOneOfExclusive(t *testing.T) {
	for name, schema := range map[string]string{
		"tags that differ":        `{"oneOf":[{"type":"object","properties":{"k":{"const":"a"}},"required":["k"]},{"type":"object","properties":{"k":{"enum":["b","c"]}},"required":["k"]}]}`,
		"a required key":          `{"oneOf":[` + objA + `,` + objB + `]}`,
		"a key of two types":      `{"oneOf":[{"type":"object","properties":{"v":{"type":"integer"}},"required":["v"]},{"type":"object","properties":{"v":{"type":"string"}},"required":["v"]}]}`,
		"a tag only one requires": `{"oneOf":[{"type":"object","properties":{"k":{"const":"a"},"x":{"type":"null"}},"required":["k"]},{"type":"object","properties":{"k":{"const":"b"}}}]}`,
		"arrays that cannot be the same length": `{"oneOf":[
			{"type":"object","properties":{"l":{"type":"array","items":{"type":"null"},"maxItems":1}},"required":["l"]},
			{"type":"object","properties":{"l":{"type":"array","items":{"type":"null"},"minItems":2},"m":{"type":"null"}},"required":["l","m"]}]}`,
	} {
		if name == "arrays that cannot be the same length" {
			// The pair is exclusive, but two arrays under one key are not told apart by a first byte.
			if _, err := JSONSchema([]byte(schema)); err == nil || !strings.Contains(err.Error(), `property "l" is an array`) {
				t.Errorf("%s: got %v, want the error about property l", name, err)
			}
			continue
		}
		if _, err := JSONSchema([]byte(schema)); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// TestToolCallGrammar_variant: a tool whose parameters are a union of objects is held to it, and one alternative the
// compiler cannot express leaves the arguments any object, as before.
func TestToolCallGrammar_variant(t *testing.T) {
	params := `{"anyOf":[` + objA + `,` + objB + `]}`
	g, err := ToolCallGrammar("", "", "arguments", "t", false, []byte(params))
	if err != nil {
		t.Fatal(err)
	}
	call := func(args string) string { return `{"name":"t","arguments":` + args + `}` }
	checkDocs(t, g, []docCase{
		{call(`{"a":1}`), true, "the first alternative"},
		{call(`{"b":"x"}`), true, "the second alternative"},
		{call(`{"a":1,"b":"x"}`), false, "one alternative's keys"},
		{call(`{}`), false, "each alternative requires its key"},
	})
	free := `{"anyOf":[` + objA + `,{"type":"object"}]}`
	g, err = ToolCallGrammar("", "", "arguments", "t", false, []byte(free))
	if err != nil {
		t.Fatal(err)
	}
	checkDocs(t, g, []docCase{
		{call(`{"anything":[1,2]}`), true, "a freeform object alternative takes any object"},
		{call(`{}`), true, "the empty one too"},
	})
	clash := `{"type":"object","properties":{"arg":{"anyOf":[
		{"type":"object","properties":{"v":{"type":"integer"}},"required":["v"]},
		{"type":"object","properties":{"v":{"type":"number"},"w":{"type":"null"}},"required":["v"]}]}}}`
	g, err = ToolCallGrammar("", "", "arguments", "t", false, []byte(clash))
	if err != nil {
		t.Fatal(err)
	}
	checkDocs(t, g, []docCase{
		{call(`{"arg":{"v":1.5,"w":null}}`), true, "a union the compiler cannot follow"},
		{call(`{"arg":"anything"}`), true, "... leaves the value free"},
		{call(`{"other":1}`), false, "the object around it is still held to its keys"},
	})
}

// TestSchema_variantStackAndFrame: a union of objects is followed with one frame, as a plain object is, so a nested
// union's stack is as deep as the nesting and no deeper; and a frame is the size it was.
func TestSchema_variantStackAndFrame(t *testing.T) {
	if strconv.IntSize == 64 {
		if size := unsafe.Sizeof(frame{}); size != 64 {
			t.Errorf("a frame is %d bytes, want 64: a mask step copies the frame stack once per candidate token", size)
		}
	}
	g := mustSchema(t, pydanticSchema(t, "union_items")).(*schemaGrammar)
	doc := `{"pets":[{"pet_type":"cat","meows":1},{"barks":2.5,"pet_type":"dog"}]}`
	deepest := 0
	g.Reset()
	for i := range len(doc) {
		if !g.TryBytes([]byte{doc[i]}) {
			t.Fatalf("refused at %q", doc[:i+1])
		}
		g.Commit([]byte{doc[i]})
		deepest = max(deepest, len(g.stack))
	}
	if !g.CanEnd() {
		t.Fatal("the document is not complete")
	}
	// The root object, the array, the item, and the item's value: four frames, as the document nests.
	if deepest != 4 {
		t.Errorf("the stack reached %d frames, want 4", deepest)
	}
}

// TestFieldConfidence_variant: a union's tag is reported as the enum it is, at its path, with the probability the
// model gave the alternative it wrote; a field of the chosen alternative is reported at its path too.
func TestFieldConfidence_variant(t *testing.T) {
	got := designedConfidence(t, pydanticSchema(t, "discriminated_union"), `{"pet":{"pet_type":"dog","barks":2},"n":17}`,
		map[string]map[byte]float64{
			`{"pet":{"pet_type":"`:                      {'d': 0.75, 'c': 0.25},
			`{"pet":{"pet_type":"dog","barks":2},"n":1`: {'7': 0.6, '8': 0.4},
		})
	tag, ok := got["pet.pet_type"]
	if !ok || tag.Kind != "enum" || string(tag.Value) != `"dog"` {
		t.Fatalf("the tag is not reported as an enum at pet.pet_type: %+v (all: %+v)", tag, got)
	}
	if math.Abs(tag.Confidence-0.75) > 1e-6 || math.Abs(tag.Distribution["cat"]-0.25) > 1e-6 {
		t.Errorf("tag confidence %v with distribution %v, want 0.75 with cat at 0.25", tag.Confidence, tag.Distribution)
	}
	if n, ok := got["n"]; !ok || n.Kind != "integer" || math.Abs(n.Confidence-0.6) > 1e-6 {
		t.Errorf("n is reported as %+v, want an integer at confidence 0.6", n)
	}
	if _, ok := got["pet.barks"]; ok {
		t.Errorf("pet.barks is a number, which is not a reported kind: %+v", got["pet.barks"])
	}
}

// designedConfidence generates target under schema, one byte a token, with confidence capture on, and returns the
// reported fields by path. design gives, for a prefix of target, the probability of each next byte it names; anywhere
// else the next byte of target has all the mass.
func designedConfidence(t *testing.T, schema []byte, target string, design map[string]map[byte]float64) map[string]FieldConfidence {
	t.Helper()
	tokens, eos := fullByteVocab()
	m := NewMasker(mustSchema(t, schema), tokens, []int{eos}).StopWhenComplete().CaptureConfidence(ConfidenceOptions{})
	var gen []int
	out := ""
	for range 200 {
		logits := make([]float32, len(tokens))
		for i := range logits {
			logits[i] = -30
		}
		switch d, ok := design[out]; {
		case ok:
			for b, p := range d {
				logits[b] = float32(math.Log(p))
			}
		case out == target:
			logits[eos] = 0
		default:
			logits[target[len(out)]] = 0
		}
		m.Process(gen, logits)
		pick := 0
		for i, v := range logits {
			if v > logits[pick] {
				pick = i
			}
		}
		gen = append(gen, pick)
		if pick == eos {
			break
		}
		out += string(tokens[pick])
	}
	if out != target {
		t.Fatalf("generated %q, want %q", out, target)
	}
	fields, err := m.FieldConfidence(gen)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]FieldConfidence{}
	for _, f := range fields {
		got[f.Path] = f
	}
	return got
}

// TestSchema_variantRandomIsItsAlternatives builds unions of random small objects (and now and then a scalar) and, for
// each one that compiles, checks it against its alternatives run side by side. The shapes are drawn to collide: a few
// key names, and values that agree, differ by type, differ by literal, nest and repeat.
func TestSchema_variantRandomIsItsAlternatives(t *testing.T) {
	t.Parallel()
	values := []string{
		`{"type":"integer"}`, `{"type":"string"}`, `{"type":"null"}`, `{"type":"boolean"}`, `{"type":"number"}`,
		`{"const":"x"}`, `{"const":"y"}`, `{"enum":["x","z"]}`, `{"const":1}`, `{"enum":[1,2,"x"]}`,
		`{"type":"object","properties":{"p":{"type":"integer"}},"required":["p"]}`,
		`{"type":"object","properties":{"q":{"const":"x"},"p":{"type":"string"}},"required":["q"]}`,
		`{"type":"array","items":{"type":"integer"},"maxItems":2}`,
		`{"anyOf":[{"type":"integer"},{"type":"null"}]}`,
		`{"anyOf":[{"const":"y"},{"type":"object","properties":{"p":{"type":"integer"}}}]}`,
	}
	names := []string{"a", "b", "c", "k"}
	rng := rand.New(rand.NewPCG(7, 2026))
	object := func() string {
		var props, required []string
		for _, name := range names {
			if rng.IntN(3) == 0 {
				continue
			}
			props = append(props, fmt.Sprintf("%q:%s", name, values[rng.IntN(len(values))]))
			if rng.IntN(2) == 0 {
				required = append(required, strconv.Quote(name))
			}
		}
		return `{"type":"object","additionalProperties":false,"properties":{` + strings.Join(props, ",") + `},"required":[` + strings.Join(required, ",") + `]}`
	}
	compiled, variants := 0, 0
	for i := range 160 {
		alts := []string{object(), object()}
		for rng.IntN(3) == 0 {
			alts = append(alts, object())
		}
		if rng.IntN(4) == 0 {
			alts = append(alts, values[rng.IntN(len(values))])
		}
		schema := `{"anyOf":[` + strings.Join(alts, ",") + `]}`
		g, err := JSONSchema([]byte(schema))
		if err != nil {
			continue
		}
		compiled++
		if g.(*schemaGrammar).root.kind == kVariant {
			variants++
		}
		checkUnionOfAlternatives(t, fmt.Sprintf("random union %d %s", i, schema), []byte(schema))
		if t.Failed() {
			return
		}
	}
	t.Logf("%d of 160 random unions compiled, %d of them as a union of objects", compiled, variants)
	if compiled < 40 || variants < 25 {
		t.Errorf("%d of 160 random unions compiled, %d of them as a union of objects: too few to say anything", compiled, variants)
	}
}

// TestSchema_variantRandomOneOf: when a oneOf over random objects compiles, the compiler has claimed that no document
// the grammar writes matches two alternatives. The independent validator, which reads the alternatives as JSON Schema
// does (undeclared keys allowed), counts the matches of every generated document.
func TestSchema_variantRandomOneOf(t *testing.T) {
	t.Parallel()
	values := []string{
		`{"type":"integer"}`, `{"type":"string"}`, `{"type":"null"}`, `{"type":"number"}`, `{"type":"integer","minimum":0}`,
		`{"const":"x"}`, `{"const":"y"}`, `{"enum":["x","z"]}`, `{"const":1}`, `{"const":-1}`, `{"const":1.5}`, `{"enum":[1,2,"x"]}`,
		`{"type":"object","properties":{"p":{"type":"integer"}},"required":["p"]}`,
		`{"type":"object","properties":{"q":{"const":"x"},"p":{"type":"string"}},"required":["q"]}`,
		`{"type":"array","items":{"type":"integer"},"maxItems":1}`,
		`{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":2}`,
		`{"anyOf":[{"type":"integer"},{"type":"null"}]}`,
		`{"properties":{"p":{"type":"integer"}},"required":["p"]}`, // no type: it matches whatever is not an object too
	}
	names := []string{"a", "b", "k"}
	rng := rand.New(rand.NewPCG(11, 2026))
	object := func() string {
		var props, required []string
		for len(props) == 0 {
			for _, name := range names {
				if rng.IntN(3) == 0 {
					continue
				}
				props = append(props, fmt.Sprintf("%q:%s", name, values[rng.IntN(len(values))]))
				if rng.IntN(3) != 0 {
					required = append(required, strconv.Quote(name))
				}
			}
		}
		return `{"type":"object","properties":{` + strings.Join(props, ",") + `},"required":[` + strings.Join(required, ",") + `]}`
	}
	compiled := 0
	for i := range 600 {
		alts := []string{object(), object()}
		if rng.IntN(3) == 0 {
			alts = append(alts, object())
		}
		schema := `{"oneOf":[` + strings.Join(alts, ",") + `]}`
		if _, err := JSONSchema([]byte(schema)); err != nil {
			continue
		}
		compiled++
		checkGenerates(t, fmt.Sprintf("random oneOf %d %s", i, schema), []byte(schema), 8)
	}
	if compiled < 150 {
		t.Errorf("only %d of 600 random oneOf schemas compiled: too few to say anything", compiled)
	}
}

// A recursive union of objects: the cycle runs through one alternative, and at the nesting cap only the other is left.
func TestSchema_variantRecursive(t *testing.T) {
	const tree = `{"$defs":{
		"Num":{"type":"object","properties":{"t":{"const":"num"},"v":{"type":"number"}},"required":["t","v"]},
		"Add":{"type":"object","properties":{"t":{"const":"add"},"l":{"$ref":"#/$defs/Expr"},"r":{"$ref":"#/$defs/Expr"}},"required":["t","l","r"]},
		"Expr":{"oneOf":[{"$ref":"#/$defs/Num"},{"$ref":"#/$defs/Add"}],"discriminator":{"propertyName":"t"}}},
		"$ref":"#/$defs/Expr"}`
	g := mustSchema(t, []byte(tree)).(*schemaGrammar)
	num := `{"t":"num","v":1}`
	add := func(l, r string) string { return `{"t":"add","l":` + l + `,"r":` + r + `}` }
	left := func(n int) string {
		doc := num
		for range n {
			doc = add(doc, num)
		}
		return doc
	}
	checkDocs(t, g, []docCase{
		{num, true, "a leaf"},
		{add(num, add(num, num)), true, "a small tree"},
		{`{"t":"add","l":` + num + `}`, false, "an add requires both sides"},
		{`{"t":"num","v":1,"l":` + num + `}`, false, "a leaf has no sides"},
		{`{"l":` + num + `,"t":"num","r":` + num + `}`, false, "after l the tag is add"},
		// A leaf's value sits one level below it, so leaves fit at levels 0 to maxSchemaDepth-1.
		{left(maxSchemaDepth - 1), true, "the deepest tree that fits"},
		{left(maxSchemaDepth), false, "one level deeper than fits"},
	})
	deepest, doc := 0, left(maxSchemaDepth-1)
	g.Reset()
	for i := range len(doc) {
		g.Commit([]byte{doc[i]})
		deepest = max(deepest, len(g.stack))
	}
	if deepest > maxSchemaDepth+1 {
		t.Errorf("the frame stack reached %d frames, over the %d every schema is held to", deepest, maxSchemaDepth+1)
	}
	// At the cap the tag offers the leaf only: an add there could never be finished.
	prefix := strings.Repeat(`{"t":"add","l":`, maxSchemaDepth-1) + `{"t":"`
	g.Reset()
	g.Commit([]byte(prefix))
	if g.TryBytes([]byte("a")) || !g.TryBytes([]byte("n")) {
		t.Error("at the nesting cap the tag must offer num and not add")
	}

	const defs = `"$defs":{
		"Leaf":{"type":"object","properties":{"t":{"const":"leaf"},"v":{"type":"integer"}},"required":["t","v"]},
		"Node":{"type":"object","properties":{"t":{"const":"node"},"kids":{"type":"array","items":{"$ref":"#/$defs/Tree"}}},"required":["t","kids"]},
		"Tree":{"oneOf":[{"$ref":"#/$defs/Leaf"},{"$ref":"#/$defs/Node"}]}}`
	checkGenerates(t, "a recursive union of objects", []byte(`{`+defs+`,"$ref":"#/$defs/Tree"}`), 60)
	checkUnionOfAlternatives(t, "a recursive union of objects", []byte(`{`+defs+`,"oneOf":[{"$ref":"#/$defs/Leaf"},{"$ref":"#/$defs/Node"}]}`))
}
