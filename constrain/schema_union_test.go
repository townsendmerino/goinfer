package constrain

import (
	"bytes"
	"encoding/json"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// An Optional field: Pydantic writes it as anyOf of the type and null, for a scalar and for a nested model alike.
func TestSchema_unionPydanticOptionalField(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "optional_field"))
	checkDocs(t, g, []docCase{
		{`{"name":"a"}`, true, "every optional field left out"},
		{`{"name":"a","nickname":null,"age":null,"home":null}`, true, "every optional field null"},
		{`{"name":"a","nickname":"n","age":3,"home":{"street":"s","city":"c"}}`, true, "every optional field set"},
		{`{"name":null}`, false, "name is not optional"},
		{`{"name":"a","nickname":3}`, false, "nickname is a string or null"},
		{`{"name":"a","age":"3"}`, false, "age is an integer or null"},
		{`{"name":"a","age":1.5}`, false, "age is an integer"},
		{`{"name":"a","age":true}`, false, "age is not a boolean"},
		{`{"name":"a","home":"s"}`, false, "home is an address or null"},
		{`{"name":"a","home":{"street":"s"}}`, false, "a set home is a whole address"},
		{`{"name":"a","home":{"street":"s","city":"c","zip":1}}`, false, "a set home has an address's keys only"},
		{`{"name":"a","home":nul}`, false, "null is spelled out"},
	})
}

// unionCases are schemas whose union the first byte of a value decides, with documents each must accept and refuse.
var unionCases = []struct {
	name, schema string
	yes, no      []string
}{
	{"type list: string or null", `{"type":["string","null"]}`, []string{`"a"`, `""`, `null`}, []string{`1`, `true`, `[]`, `{}`}},
	{"type list: integer or string", `{"type":["integer","string"]}`, []string{`5`, `-5`, `"5"`}, []string{`5.5`, `null`}},
	{"type list: number covers integer", `{"type":["number","integer"]}`, []string{`1.5`, `2`, `-0.5e3`}, []string{`"2"`}},
	{"type list: one entry", `{"type":["boolean"]}`, []string{`true`, `false`}, []string{`null`, `0`}},
	{"type list: an entry twice", `{"type":["string","string","null"]}`, []string{`"a"`, `null`}, []string{`0`}},
	{"type list: scalars", `{"type":["boolean","null","integer"]}`, []string{`true`, `false`, `null`, `7`}, []string{`"x"`, `7.5`}},
	{"type list: object or null", `{"type":["object","null"],"properties":{"a":{"type":"string"}},"required":["a"]}`,
		[]string{`{"a":"x"}`, `null`}, []string{`{}`, `"x"`, `{"a":null}`}},
	{"type list: array or null", `{"type":["array","null"],"items":{"type":"integer"},"maxItems":2}`,
		[]string{`[]`, `[1,2]`, `null`}, []string{`[1,2,3]`, `["a"]`, `1`}},
	{"type list: every type", `{"type":["object","array","string","number","boolean","null"],"properties":{"k":{"type":"null"}},"items":{"type":"boolean"}}`,
		[]string{`{"k":null}`, `{}`, `[true]`, `"s"`, `1.5`, `false`, `null`}, []string{`{"j":1}`, `[1]`}},
	{"type list: minimum 0 on the numeric entry", `{"type":["integer","string"],"minimum":0}`, []string{`1`, `0`, `"-1"`}, []string{`-1`, `1.5`}},
	{"type list: format on the string entry", `{"type":["string","null"],"format":"date"}`, []string{`"2024-01-02"`, `null`}, []string{`"x"`, `1`}},
	{"anyOf: string or null", `{"anyOf":[{"type":"string"},{"type":"null"}]}`, []string{`"a"`, `null`}, []string{`1`, `false`}},
	{"anyOf: number or string", `{"anyOf":[{"type":"number"},{"type":"string"}]}`, []string{`1.5`, `"a"`}, []string{`null`, `[1]`}},
	{"anyOf: object or null", `{"anyOf":[{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]},{"type":"null"}]}`,
		[]string{`{"id":1}`, `null`}, []string{`{}`, `1`, `{"id":null}`}},
	{"anyOf: string or array of strings", `{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]}`,
		[]string{`"a"`, `[]`, `["a","b"]`}, []string{`[1]`, `1`}},
	{"anyOf: number covers integer", `{"anyOf":[{"type":"integer"},{"type":"number"}]}`, []string{`1`, `1.5`}, []string{`"1"`}},
	{"anyOf: integer covers a non-negative integer", `{"anyOf":[{"type":"integer","minimum":0},{"type":"integer"}]}`, []string{`-1`, `1`}, []string{`1.5`}},
	{"anyOf: literals merge", `{"anyOf":[{"enum":["a","b"]},{"type":"null"},{"const":"c"},{"type":"boolean"}]}`,
		[]string{`"a"`, `"b"`, `"c"`, `null`, `true`, `false`}, []string{`"d"`, `0`}},
	{"anyOf: a literal beside a type it does not belong to", `{"anyOf":[{"const":"auto"},{"type":"integer"}]}`, []string{`"auto"`, `5`}, []string{`"x"`, `5.5`}},
	{"anyOf: a negative literal beside a non-negative number", `{"anyOf":[{"const":-1},{"type":"integer","minimum":0}]}`, []string{`-1`, `0`, `12`}, []string{`-2`, `-12`}},
	{"anyOf: a string covers a string literal", `{"anyOf":[{"type":"string"},{"const":"a"}]}`, []string{`"a"`, `"b"`}, []string{`1`}},
	{"anyOf: a string covers a formatted string", `{"anyOf":[{"type":"string","format":"date"},{"type":"string"}]}`, []string{`"2024-01-02"`, `"x"`}, []string{`1`}},
	{"anyOf: an integer covers an integer literal", `{"anyOf":[{"type":"integer"},{"enum":[1,2]},{"type":"null"}]}`, []string{`1`, `3`, `null`}, []string{`1.5`}},
	{"anyOf: nested", `{"anyOf":[{"anyOf":[{"type":"string"},{"type":"null"}]},{"type":"integer"}]}`, []string{`"a"`, `null`, `3`}, []string{`3.5`, `true`}},
	{"anyOf: the same alternative twice", `{"anyOf":[{"type":"array","items":{"type":"string"}},{"type":"array","items":{"type":"string"},"title":"again"}]}`,
		[]string{`["a"]`}, []string{`[1]`}},
	{"anyOf: through $ref", `{"$defs":{"S":{"type":"string"},"N":{"type":"null"}},"anyOf":[{"$ref":"#/$defs/S"},{"$ref":"#/$defs/N"}],"description":"d","default":null}`,
		[]string{`"a"`, `null`}, []string{`1`}},
	{"oneOf: string or null", `{"oneOf":[{"type":"string"},{"type":"null"}]}`, []string{`"a"`, `null`}, []string{`1`}},
	{"oneOf: one alternative", `{"oneOf":[{"type":"string"}]}`, []string{`"a"`}, []string{`null`}},
	{"oneOf: literals that differ", `{"oneOf":[{"enum":["a",1]},{"enum":["b",2]},{"type":"null"}]}`, []string{`"a"`, `1`, `"b"`, `2`, `null`}, []string{`"c"`, `3`}},
	{"oneOf: a negative literal or a non-negative integer", `{"oneOf":[{"const":-1},{"type":"integer","minimum":0}]}`, []string{`-1`, `0`, `5`}, []string{`-2`, `0.5`}},
	{"oneOf: a fraction or an integer type's neighbours", `{"oneOf":[{"enum":[1.5,2.5]},{"enum":["1.5"]},{"type":"null"}]}`, []string{`1.5`, `2.5`, `"1.5"`, `null`}, []string{`1`, `"2.5"`}},
	{"oneOf: an array or a string", `{"oneOf":[{"type":"array","items":{"type":"integer"},"maxItems":0},{"type":"string"}]}`, []string{`[]`, `"a"`}, []string{`[1]`}},
	{"a union as a property and as items", `{"type":"object","properties":{"v":{"type":["integer","null"]},"l":{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]}}},"required":["v"]}`,
		[]string{`{"v":null}`, `{"v":3,"l":["a",null]}`}, []string{`{"v":"3"}`, `{"v":1,"l":[1]}`, `{}`}},
}

func TestSchema_unionByFirstByte(t *testing.T) {
	for _, c := range unionCases {
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

// A top-level scalar union may end as soon as what is written is a value of one alternative, as a bare scalar may.
func TestSchema_unionTopLevelMayEnd(t *testing.T) {
	g := mustSchema(t, []byte(`{"anyOf":[{"type":"integer"},{"type":"null"}]}`))
	g.Reset()
	g.Commit([]byte(`12`))
	if !g.CanEnd() || !g.TryBytes([]byte(`3`)) {
		t.Error("after 12 the document may end and may also go on")
	}
	g.Reset()
	g.Commit([]byte(`nul`))
	if g.CanEnd() {
		t.Error("nul is not a value")
	}
	g.Commit([]byte(`l`))
	if !g.CanEnd() {
		t.Error("null is a value")
	}
}

// unionAlternatives returns the alternatives of a root-level union (anyOf, oneOf or a type list), each as a document
// of its own that keeps the root's definitions.
func unionAlternatives(t *testing.T, schema []byte) [][]byte {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()
	var root map[string]any
	if err := dec.Decode(&root); err != nil {
		t.Fatal(err)
	}
	var alts []map[string]any
	for _, key := range []string{"anyOf", "oneOf"} {
		list, _ := root[key].([]any)
		for _, a := range list {
			alts = append(alts, maps.Clone(a.(map[string]any)))
		}
	}
	if types, ok := root["type"].([]any); ok {
		for _, typ := range types {
			alt := maps.Clone(root)
			alt["type"] = typ
			alts = append(alts, alt)
		}
	}
	var out [][]byte
	for _, alt := range alts {
		for _, defs := range []string{"$defs", "definitions"} {
			if d, ok := root[defs]; ok {
				alt[defs] = d
			}
		}
		raw, err := json.Marshal(alt)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, raw)
	}
	return out
}

// checkUnionOfAlternatives compares a union's grammar with its alternatives compiled one by one and run side by side
// (unionGrammar): along documents the alternatives accept, and one-edit variants of them, the legal next bytes and
// the points where the document may end must be the same.
func checkUnionOfAlternatives(t *testing.T, name string, schema []byte) {
	t.Helper()
	g, err := JSONSchema(schema)
	if err != nil {
		t.Errorf("%s: %v", name, err)
		return
	}
	alts := unionAlternatives(t, schema)
	if len(alts) == 0 {
		t.Fatalf("%s: no root-level union to compare", name)
	}
	side := &unionGrammar{}
	for _, alt := range alts {
		side.all = append(side.all, mustSchema(t, alt))
	}
	rng := rand.New(rand.NewPCG(11, uint64(len(name))))
	walks := 0
	for i := range 20 * len(alts) {
		doc, ok := pinWalkDoc(side.all[i%len(alts)], rng)
		if !ok {
			continue
		}
		walks++
		if a, b := pinMask(g, doc), pinMask(side, doc); a != b {
			t.Errorf("%s: along %q the union allows different next bytes than its alternatives side by side", name, doc)
		}
		for range 4 {
			m := pinMutate(doc, rng)
			gn, gend := pinAccept(g, m)
			sn, send := pinAccept(side, m)
			if gn != sn || gend != send {
				t.Errorf("%s: %q is accepted for %d bytes (may end: %v) by the union and %d (may end: %v) by its alternatives", name, m, gn, gend, sn, send)
			}
		}
	}
	if walks < 10*len(alts) {
		t.Errorf("%s: only %d walks completed", name, walks)
	}
}

// TestSchema_unionIsItsAlternatives: a union accepts exactly what its alternatives accept between them.
func TestSchema_unionIsItsAlternatives(t *testing.T) {
	n := 0
	for _, c := range unionCases {
		if strings.HasPrefix(c.name, "a union as a property") {
			continue // not a root-level union
		}
		n++
		checkUnionOfAlternatives(t, c.name, []byte(c.schema))
	}
	if n < 25 {
		t.Fatalf("only %d root-level unions compared", n)
	}
}

// TestSchema_unionErrors: a union the grammar cannot follow by a value's first byte, or cannot show to be what the
// schema says, is a compile error that names what is in the way.
func TestSchema_unionErrors(t *testing.T) {
	many := `{"anyOf":[` + strings.TrimSuffix(strings.Repeat(`{"type":"null"},`, 65), ",") + `]}`
	for _, c := range []struct{ schema, want string }{
		{`{"anyOf":[]}`, "anyOf must be a non-empty array"},
		{`{"oneOf":"x"}`, "oneOf must be a non-empty array"},
		{`{"anyOf":[5]}`, "anyOf alternative 0 is not a schema"},
		{many, "more than 64 alternatives"},
		{`{"type":[]}`, `"type" lists no types`},
		{`{"type":["string","wat"]}`, `unsupported type "wat"`},
		{`{"type":["string",5]}`, `"type" list`},
		{`{"type":["array","null"]}`, "items"},
		{`{"anyOf":[{"type":"string"},{"type":"null"}],"type":"string"}`, `"type" beside anyOf`},
		{`{"oneOf":[{"type":"string"},{"type":"null"}],"enum":["a",null]}`, `"enum" beside oneOf`},
		{`{"anyOf":[{"type":"string"}],"oneOf":[{"type":"string"}]}`, "beside"},
		{`{"type":"object","properties":{"p":{"type":"string"}},"discriminator":{"propertyName":"p"}}`, `"discriminator"`},
		{`{"anyOf":[{"type":"string","pattern":"x"},{"type":"null"}]}`, `anyOf alternative 0: constrain: unsupported schema keyword "pattern"`},
		{`{"anyOf":[{"type":"array","items":{"type":"string"}},{"type":"array","items":{"type":"integer"}}]}`, "anyOf alternatives 0 and 1 are both arrays"},
		{`{"anyOf":[{"type":"string","format":"date"},{"type":"string","format":"uuid"}]}`, "anyOf alternatives 0 and 1 are both strings"},
		{`{"anyOf":[{"type":"integer"},{"type":"number","minimum":0}]}`, "anyOf alternatives 0 and 1 are both numbers"},
		{`{"anyOf":[{"type":"integer"},{"const":1.5}]}`, "anyOf alternative 1"},
		{`{"anyOf":[{"type":"string","format":"date"},{"const":"x"}]}`, "anyOf alternative 1"},
		{`{"oneOf":[{"type":"integer"},{"type":"number"}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[{"type":"string"},{"const":"a"}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[{"enum":[1,2]},{"enum":[2,3]}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[{"enum":[1]},{"enum":[1.0]}]}`, "oneOf alternatives 0 and 1 can both match"},
		{`{"oneOf":[{"type":"string"},{"type":"null"},{"type":"string"}]}`, "oneOf alternatives 0 and 2 can both match"},
		{`{"oneOf":[{"type":"array","items":{"type":"integer"},"maxItems":0},{"type":"array","items":{"type":"integer"},"minItems":1}]}`, "oneOf alternatives 0 and 1 are both arrays"},
		{`{"$defs":{"A":{"anyOf":[{"$ref":"#/$defs/A"},{"type":"null"}]}},"$ref":"#/$defs/A"}`, "refers to itself"},
		{`{"allOf":[{"type":"string"}]}`, `"allOf"`},
		{`{"not":{"type":"string"}}`, `"not"`},
	} {
		_, err := JSONSchema([]byte(c.schema))
		show := c.schema
		if len(show) > 120 {
			show = show[:120] + "…"
		}
		if err == nil {
			t.Errorf("%s compiled; want an error naming %s", show, c.want)
		} else if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not name %s", show, err, c.want)
		}
	}
}

// TestSchema_typeListBesideProperties: a `type` list beside `properties` is read. Such a schema used to compile as a
// plain object, the list unread, so it could not say null and a list without "object" still forced an object.
func TestSchema_typeListBesideProperties(t *testing.T) {
	g := mustSchema(t, []byte(`{"type":["object","null"],"properties":{"a":{"type":"string"}}}`))
	checkDocs(t, g, []docCase{
		{`{"a":"x"}`, true, "the object entry"},
		{`null`, true, "the null entry"},
		{`"x"`, false, "neither entry"},
	})
	g = mustSchema(t, []byte(`{"type":["string","null"],"properties":{"a":{"type":"string"}}}`))
	checkDocs(t, g, []docCase{
		{`"x"`, true, "the string entry"},
		{`null`, true, "the null entry"},
		{`{"a":"x"}`, false, "the list has no object entry"},
	})
}

// A recursive Optional: the cycle runs through a union, and is cut at the nesting cap with null as the way out.
func TestSchema_unionPydanticRecursiveOptional(t *testing.T) {
	g := mustSchema(t, pydanticSchema(t, "recursive_optional"))
	chain := func(n int, last string) string {
		return strings.Repeat(`{"value":1,"next":`, n-1) + last + strings.Repeat(`}`, n-1)
	}
	checkDocs(t, g, []docCase{
		{`{"value":1}`, true, "next left out"},
		{`{"value":1,"next":null}`, true, "next null"},
		{chain(3, `{"value":3,"next":null}`), true, "three nodes"},
		{`{"value":1,"next":{"next":null}}`, false, "a node requires its value"},
		{`{"value":1,"next":3}`, false, "next is a node or null"},
		// A node's value sits one level below it, so nodes fit at levels 0 to maxSchemaDepth-1.
		{chain(maxSchemaDepth, `{"value":0}`), true, "the deepest node that fits"},
		{chain(maxSchemaDepth, `{"value":0,"next":null}`), true, "the deepest node may say it is the last"},
		{chain(maxSchemaDepth+1, `{"value":0}`), false, "one node deeper than fits"},
	})
}

// unionGenSchemas are schemas with unions whose constrained generations must conform (TestSchema_unionGenerates).
func unionGenSchemas(t *testing.T) map[string][]byte {
	t.Helper()
	m := map[string][]byte{
		"pydantic optional field":     pydanticSchema(t, "optional_field"),
		"pydantic recursive optional": pydanticSchema(t, "recursive_optional"),
		"pydantic mixed unions":       pydanticSchema(t, "mixed_unions"),
	}
	for _, c := range unionCases {
		m["case/"+c.name] = []byte(c.schema)
	}
	return m
}

// TestSchema_unionGenerates: constrained generation under a schema with unions never dead-ends, and what it produces
// conforms to the schema by the independent validator, which counts a oneOf's matching alternatives itself.
func TestSchema_unionGenerates(t *testing.T) {
	t.Parallel()
	schemas := unionGenSchemas(t)
	for _, name := range slices.Sorted(maps.Keys(schemas)) {
		checkGenerates(t, name, schemas[name], 120)
	}
}

// checkGenerates runs n seeded constrained generations under schema and validates each with conforms.
func checkGenerates(t *testing.T, name string, schema []byte, n int) {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		t.Fatalf("%s: bad schema: %v", name, err)
	}
	for i := range n {
		g, err := JSONSchema(schema)
		if err != nil {
			t.Fatalf("%s: compile: %v", name, err)
		}
		out := genConstrained(t, g, int64(i)+1, 0)
		vd := json.NewDecoder(bytes.NewReader(out))
		vd.UseNumber()
		var val any
		if err := vd.Decode(&val); err != nil {
			t.Fatalf("%s seed %d: output is not valid JSON: %q (%v)", name, i, out, err)
		}
		if err := conforms(doc, val); err != nil {
			t.Fatalf("%s seed %d: output %q does not conform: %v", name, i, out, err)
		}
	}
}

// TestToolCallGrammar_typeListIsHeldToItsTypes: in a tool's parameters a type list, anyOf and oneOf are enforced where
// they used to leave the value free. ken_search's `languages` and `queries` are ["null","array"] of strings.
func TestToolCallGrammar_typeListIsHeldToItsTypes(t *testing.T) {
	var ken ToolSpec
	for _, tl := range opencodeKenTools(t) {
		if tl.Name == "ken_search" {
			ken = tl
		}
	}
	g, err := ToolCallGrammar("", "", "arguments", ken.Name, false, ken.Parameters)
	if err != nil {
		t.Fatal(err)
	}
	call := func(args string) string { return `{"name":"ken_search","arguments":` + args + `}` }
	checkDocs(t, g, []docCase{
		{call(`{"query":"prepare","languages":["go"]}`), true, "an array of strings"},
		{call(`{"query":"prepare","languages":null,"queries":["a","b"]}`), true, "null, and an array of strings"},
		{call(`{"query":"prepare","languages":"go"}`), false, "a bare string is neither null nor an array"},
		{call(`{"query":"prepare","languages":[1]}`), false, "the array's items are strings"},
		{call(`{"query":"prepare","queries":{"a":1}}`), false, "an object is neither null nor an array"},
	})

	params := `{"type":"object","properties":{
		"mode":{"anyOf":[{"enum":["fast","slow"]},{"type":"null"}]},
		"limit":{"oneOf":[{"type":"integer"},{"type":"number"}]},
		"filter":{"anyOf":[{"type":"string"},{"not":{"type":"string"}}]},
		"where":{"anyOf":[{"type":"string"},{"type":"null"}],"minLength":1}}}`
	tg, err := ToolCallGrammar("", "", "arguments", "t", false, []byte(params))
	if err != nil {
		t.Fatal(err)
	}
	tcall := func(args string) string { return `{"name":"t","arguments":` + args + `}` }
	checkDocs(t, tg, []docCase{
		{tcall(`{"mode":"fast"}`), true, "an enum alternative"},
		{tcall(`{"mode":null}`), true, "the null alternative"},
		{tcall(`{"mode":"medium"}`), false, "anyOf is enforced"},
		{tcall(`{"limit":1.5}`), true, "a oneOf whose alternatives overlap is held as anyOf"},
		{tcall(`{"limit":"1"}`), false, "... and to its alternatives"},
		{tcall(`{"filter":[1,{"a":null}]}`), true, "an alternative the compiler cannot express leaves the value free"},
		{tcall(`{"where":[1]}`), true, "an assertion beside anyOf leaves the value free, as before"},
	})
}
