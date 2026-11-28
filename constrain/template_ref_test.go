package constrain

import (
	"fmt"
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

// A reference the grammar would refuse is refused here too, and a document whose references fan out is refused by a
// count: a template is the schema written out in full, so it has no memo to keep it small.
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
