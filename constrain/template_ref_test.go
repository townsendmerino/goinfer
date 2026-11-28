package constrain

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// The convention's rows for references and unions (template.go's table).
func TestTemplateFromSchema_refsAndUnions(t *testing.T) {
	address := "{\n        \"street\": \"\",\n        \"city\": \"\"\n    }"
	cases := []struct{ name, schema, want string }{
		{"pydantic nested model", string(pydanticSchema(t, "nested_model")),
			"{\n    \"name\": \"\",\n    \"address\": " + address + ",\n    \"shipping\": " + address + "\n}"},
		{"pydantic optional field", string(pydanticSchema(t, "optional_field")),
			"{\n    \"name\": \"\",\n    \"nickname\": \"\",\n    \"age\": 0,\n    \"home\": " + address + "\n}"},
		{"pydantic recursive model", string(pydanticSchema(t, "recursive_model")),
			"{\n    \"text\": \"\",\n    \"replies\": [\n        {}\n    ]\n}"},
		{"pydantic discriminated union", string(pydanticSchema(t, "discriminated_union")),
			"{\n    \"pet\": {\n        \"pet_type\": \"cat\",\n        \"meows\": 0\n    },\n    \"n\": 0\n}"},
		{"a reference at the root", `{"definitions":{"S":{"type":"integer"}},"$ref":"#/definitions/S"}`, `0`},
		{"anyOf takes its first alternative that is not null", `{"anyOf":[{"type":"null"},{"type":"boolean"},{"type":"string"}]}`, `false`},
		{"oneOf likewise", `{"oneOf":[{"const":null},{"type":"array","items":{"type":"integer"}}]}`, "[\n    0\n]"},
		{"a union of nulls is null", `{"anyOf":[{"type":"null"}]}`, `null`},
		{"a recursive optional", `{"$defs":{"N":{"type":"object","properties":{"v":{"type":"integer"},"next":{"anyOf":[{"$ref":"#/$defs/N"},{"type":"null"}]}}}},"$ref":"#/$defs/N"}`,
			"{\n    \"v\": 0,\n    \"next\": {}\n}"},
		// A schema is being written out however the template came to it: as the root, by a property, or by a reference
		// spelled another way.
		{"a reference back to the root", `{"type":"object","properties":{"name":{"type":"string"},"kids":{"type":"array","items":{"$ref":"#"}}}}`,
			"{\n    \"name\": \"\",\n    \"kids\": [\n        {}\n    ]\n}"},
		{"a reference back to an enclosing property", `{"type":"object","properties":{"a":{"type":"object","properties":{"v":{"type":"integer"},"again":{"$ref":"#/properties/a"}}}}}`,
			"{\n    \"a\": {\n        \"v\": 0,\n        \"again\": {}\n    }\n}"},
		{"one target under two spellings", `{"$defs":{"N":{"type":"object","properties":{"v":{"type":"integer"},"next":{"$ref":"#/%24defs/N"}}}},"$ref":"#/$defs/N"}`,
			"{\n    \"v\": 0,\n    \"next\": {}\n}"},
		{"a recursive array", `{"$defs":{"L":{"type":"array","items":{"$ref":"#/$defs/L"}}},"$ref":"#/$defs/L"}`, "[\n    []\n]"},
		// A schema is an array when an array is what stands in its place, by a reference or as a union's alternative.
		{"a recursive array, named by the reference to it", `{"$defs":{"L":{"type":"array","items":{"$ref":"#"}}},"$ref":"#/$defs/L"}`, "[\n    []\n]"},
		{"a recursive array, named by a property that refers to it", `{"type":"object","properties":{"list":{"$ref":"#/$defs/L"}},"$defs":{"L":{"type":"array","items":{"$ref":"#/properties/list"}}}}`,
			"{\n    \"list\": [\n        []\n    ]\n}"},
		{"a recursive array, named by the union it is an alternative of", `{"anyOf":[{"type":"null"},{"type":"array","items":{"$ref":"#"}}]}`, "[\n    []\n]"},
		{"a recursive array behind two references and a union", `{"$defs":{"A":{"$ref":"#/$defs/B"},"B":{"oneOf":[{"type":"array","items":{"$ref":"#"}},{"type":"null"}]}},"$ref":"#/$defs/A"}`, "[\n    []\n]"},
		{"a recursive object, named by the reference to it", `{"$defs":{"N":{"type":"object","properties":{"next":{"$ref":"#"}}}},"$ref":"#/$defs/N"}`, "{\n    \"next\": {}\n}"},
		{"a reference to itself and nothing else", `{"$ref":"#"}`, `{}`},
		// A schema a reference entered, met again as its enclosing schema's own child, is still being written out when
		// the second writing of it is done.
		{"a recursive array, entered by a reference into its items", `{"$defs":{"X":{"type":"array","items":{"type":"object","properties":{"a":{"$ref":"#/$defs/X"},"b":{"$ref":"#"}}}}},"$ref":"#/$defs/X/items/properties/a"}`,
			"[\n    {\n        \"a\": [],\n        \"b\": []\n    }\n]"},
		{"a recursive object, entered by a reference to one of its properties", `{"$defs":{"X":{"type":"object","properties":{"a":{"type":"object","properties":{"v":{"type":"integer"},"up":{"$ref":"#/$defs/X"}}},"b":{"$ref":"#/$defs/X/properties/a"}}}},"$ref":"#/$defs/X/properties/a"}`,
			"{\n    \"v\": 0,\n    \"up\": {\n        \"a\": {\n            \"v\": 0,\n            \"up\": {}\n        },\n        \"b\": {}\n    }\n}"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := TemplateFromSchema([]byte(c.schema))
			if err != nil {
				t.Fatal(err)
			}
			if got != c.want {
				t.Fatalf("got\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// A reference that cannot be followed is refused: one that names nothing, one out of the document, one that is not a
// string. A document whose references fan out is refused by a count: a template is the schema written out in full, so
// it has no memo to keep it small.
func TestTemplateFromSchema_refErrors(t *testing.T) {
	var fan strings.Builder
	fan.WriteString(`{"$defs":{"L0":{"type":"string"}`)
	for i := 1; i <= 24; i++ {
		fmt.Fprintf(&fan, `,"L%d":{"type":"object","properties":{"a":{"$ref":"#/$defs/L%d"},"b":{"$ref":"#/$defs/L%d"}}}`, i, i-1, i-1)
	}
	fan.WriteString(`},"$ref":"#/$defs/L24"}`)
	for _, c := range []struct{ schema, want string }{
		{`{"$ref":"#/$defs/Missing"}`, "nothing in the document"},
		{`{"$ref":"https://example.com/x.json"}`, "only a reference into the same document"},
		{`{"$ref":5}`, "$ref must be a string"},
		{fan.String(), "values"},
	} {
		_, err := TemplateFromSchema([]byte(c.schema))
		show := c.schema
		if len(show) > 100 {
			show = show[:100] + "…"
		}
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error naming %q", show, err, c.want)
		}
	}
}

// tschema is a schema of a document that TestTemplateFromSchema_onRandomDocuments builds. The document is kept as a
// tree, a reference holding its target, so that its template can be written without reading the document.
type tschema struct {
	kind   string     // "string", "integer", "null", "array", "object", "anyOf", "oneOf" or "ref"
	kids   []*tschema // an array's items, an object's properties p and q, a union's two alternatives
	target *tschema   // a reference's
	ptr    string     // where the schema is in the document
	tree   int        // which of the document's trees it is in: the root's, or a definition's
}

func (n *tschema) standsFor() bool { return n.kind == "ref" || n.kind == "anyOf" || n.kind == "oneOf" }

// write is the schema as JSON; defs are the definitions the root schema carries.
func (n *tschema) write(b *strings.Builder, defs []*tschema) {
	switch n.kind {
	case "ref":
		fmt.Fprintf(b, `{"$ref":%q`, n.target.ptr)
	case "array":
		b.WriteString(`{"type":"array","items":`)
		n.kids[0].write(b, nil)
	case "object":
		b.WriteString(`{"type":"object","properties":{"p":`)
		n.kids[0].write(b, nil)
		b.WriteString(`,"q":`)
		n.kids[1].write(b, nil)
		b.WriteByte('}')
	case "anyOf", "oneOf":
		fmt.Fprintf(b, `{%q:[`, n.kind)
		n.kids[0].write(b, nil)
		b.WriteByte(',')
		n.kids[1].write(b, nil)
		b.WriteByte(']')
	default:
		fmt.Fprintf(b, `{"type":%q`, n.kind)
	}
	for i, d := range defs {
		fmt.Fprintf(b, `%s"D%d":`, []string{`,"$defs":{`, `,`}[min(i, 1)], i)
		d.write(b, nil)
	}
	if len(defs) > 0 {
		b.WriteByte('}')
	}
	b.WriteByte('}')
}

// tcounts is what template met on the way: the values it may still write, how deep it went, and the kinds of place it
// came to.
type tcounts struct {
	budget  int
	deepest int // the most properties and items any value is nested under
	cut     int // a reference back into a schema being written out
	arrays  int // of those, written []
	stoodIn int // of those, a [] decided by what a reference or a union was being written as
	again   int // a schema written out while it was already being written out
	after   int // a cut into a schema that had been written out again inside itself, once that writing had ended
	// ended is true for a schema from the end of a writing of it inside itself to the end of the writing around that.
	ended map[*tschema]bool
}

// template is the convention's value for n, as encoding/json decodes one. open is the schemas being written out,
// outermost first: the one after a reference or a union is the schema written in its place. depth is the properties
// and items n is nested under.
func (n *tschema) template(open []*tschema, depth int, c *tcounts) any {
	if c.budget--; c.budget < 0 {
		return nil
	}
	c.deepest = max(c.deepest, depth)
	if slices.Contains(open, n) {
		c.again++
		c.ended[n] = false
		defer func() { c.ended[n] = true }()
	} else {
		defer delete(c.ended, n)
	}
	open = append(open, n)
	switch n.kind {
	case "ref":
		for i := len(open) - 1; i >= 0; i-- {
			if open[i] != n.target {
				continue
			}
			c.cut++
			if c.ended[n.target] {
				c.after++
			}
			at := i
			for at < len(open) && open[at].standsFor() {
				at++
			}
			if at == len(open) || open[at].kind != "array" {
				return map[string]any{}
			}
			c.arrays++
			if at > i {
				c.stoodIn++
			}
			return []any{}
		}
		return n.target.template(open, depth, c)
	case "anyOf", "oneOf":
		for _, alt := range n.kids {
			if v := alt.template(open, depth, c); v != nil {
				return v
			}
		}
		return nil
	case "object":
		return map[string]any{"p": n.kids[0].template(open, depth+1, c), "q": n.kids[1].template(open, depth+1, c)}
	case "array":
		return []any{n.kids[0].template(open, depth+1, c)}
	case "string":
		return ""
	case "integer":
		return float64(0)
	}
	return nil
}

// TestTemplateFromSchema_onRandomDocuments: on small random documents whose references name any schema of the
// document (a definition, a schema in the middle of one, the root, one being written out), the template is the
// convention's value, written directly from the document's tree with the schemas being written out kept in order.
// A document is left out only when that value passes one of TemplateFromSchema's caps.
func TestTemplateFromSchema_onRandomDocuments(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 21))
	const rounds = 10000
	var seen tcounts
	skipped := 0
	for range rounds {
		var all []*tschema
		var trees [][]*tschema // the root's schemas, then each definition's
		var gen func(ptr string, level int) *tschema
		gen = func(ptr string, level int) *tschema {
			if level == 0 {
				trees = append(trees, nil)
			}
			n := &tschema{ptr: ptr, tree: len(trees) - 1}
			all = append(all, n)
			trees[n.tree] = append(trees[n.tree], n)
			pick := rng.IntN(16)
			if level == 3 {
				pick = rng.IntN(8) // no deeper: a scalar or a reference
			}
			switch {
			case pick == 0:
				n.kind = "string"
			case pick == 1:
				n.kind = "integer"
			case pick == 2:
				n.kind = "null"
			case pick < 8:
				n.kind = "ref"
			case pick < 11:
				n.kind = "array"
				n.kids = []*tschema{gen(ptr+"/items", level+1)}
			case pick < 14:
				n.kind = "object"
				n.kids = []*tschema{gen(ptr+"/properties/p", level+1), gen(ptr+"/properties/q", level+1)}
			default:
				n.kind = []string{"anyOf", "oneOf"}[rng.IntN(2)]
				n.kids = []*tschema{gen(ptr+"/"+n.kind+"/0", level+1), gen(ptr+"/"+n.kind+"/1", level+1)}
			}
			return n
		}
		root := gen("#", 0)
		defs := make([]*tschema, 1+rng.IntN(3))
		for i := range defs {
			defs[i] = gen(fmt.Sprintf("#/$defs/D%d", i), 0)
		}
		for _, n := range all {
			if n.kind != "ref" {
				continue
			}
			from := all
			if own := trees[n.tree]; len(own) > 1 && rng.IntN(2) == 0 {
				from = own // half of the references stay in their own tree, where recursion is written
			}
			n.target = from[rng.IntN(len(from))]
		}
		var b strings.Builder
		root.write(&b, defs)
		c := tcounts{budget: 20000, ended: map[*tschema]bool{}}
		want := root.template(nil, 0, &c)
		got, err := TemplateFromSchema([]byte(b.String()))
		if c.budget < 0 || c.deepest > maxSchemaDepth {
			skipped++ // a template past a cap of TemplateFromSchema's; the budget is under maxTemplateValues
			continue
		}
		if err != nil {
			t.Fatalf("%v\n%s", err, b.String())
		}
		var have any
		if err := json.Unmarshal([]byte(got), &have); err != nil {
			t.Fatalf("the template is not JSON: %v\n%s\n%s", err, got, b.String())
		}
		if !reflect.DeepEqual(have, want) {
			direct, _ := json.Marshal(want)
			t.Fatalf("the template is\n%s\nthe convention read directly gives\n%s\nfor\n%s", got, direct, b.String())
		}
		seen.cut += min(c.cut, 1)
		seen.arrays += min(c.arrays, 1)
		seen.stoodIn += min(c.stoodIn, 1)
		seen.again += min(c.again, 1)
		seen.after += min(c.after, 1)
	}
	t.Logf("of %d documents: %d cut a reference, %d wrote [] for one, %d of those by what a reference or a union was written as, %d wrote a schema out again inside itself, %d cut a reference to one after that; %d left out as past a cap",
		rounds, seen.cut, seen.arrays, seen.stoodIn, seen.again, seen.after, skipped)
	for _, k := range []struct {
		what     string
		n, least int
	}{
		{"cut a reference", seen.cut, rounds / 100},
		{"wrote [] for a cut reference", seen.arrays, rounds / 100},
		{"had a [] decided by what a reference or a union was written as", seen.stoodIn, rounds / 100},
		{"wrote a schema out again inside itself", seen.again, rounds / 100},
		{"cut a reference to a schema written out again inside itself, after that writing", seen.after, rounds / 1000},
	} {
		if k.n < k.least {
			t.Errorf("%d of %d documents %s: the test wants %d", k.n, rounds, k.what, k.least)
		}
	}
	if skipped > rounds/10 {
		t.Errorf("%d of %d documents were left out as past a cap", skipped, rounds)
	}
}
