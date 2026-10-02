package constrain

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

// cardTemplate is the information-extraction example from zai-org/GLM-OCR's model card (revision 2e85a628), verbatim:
// the layout the model was trained to read.
const cardTemplate = `{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}`

const cardSchema = `{"type":"object","additionalProperties":false,
 "properties":{
  "id_number":{"type":"string"},"last_name":{"type":"string"},"first_name":{"type":"string"},"date_of_birth":{"type":"string"},
  "address":{"type":"object","additionalProperties":false,"properties":{
     "street":{"type":"string"},"city":{"type":"string"},"state":{"type":"string"},"zip_code":{"type":"string"}}},
  "dates":{"type":"object","additionalProperties":false,"properties":{
     "issue_date":{"type":"string"},"expiration_date":{"type":"string"}}},
  "sex":{"type":"string"}}}`

// The card's own example is the oracle for the layout: a schema listing the same keys in the same order must come out
// byte-identical to it (four-space indent, `"key": ""`, no trailing newline). Reordering the schema's properties must
// reorder the template, because the order is what the model reads.
func TestTemplateFromSchema_cardExampleByteExact(t *testing.T) {
	got, err := TemplateFromSchema([]byte(cardSchema))
	if err != nil {
		t.Fatal(err)
	}
	if got != cardTemplate {
		t.Fatalf("template differs from the model card's example\n got:\n%s\nwant:\n%s", got, cardTemplate)
	}
	rev := `{"type":"object","additionalProperties":false,"properties":{"sex":{"type":"string"},"id_number":{"type":"string"}}}`
	got, err = TemplateFromSchema([]byte(rev))
	if err != nil {
		t.Fatal(err)
	}
	if want := "{\n    \"sex\": \"\",\n    \"id_number\": \"\"\n}"; got != want {
		t.Fatalf("property order not kept:\n%s", got)
	}
}

// One row per line of the convention table in template.go's doc comment.
func TestTemplateFromSchema_convention(t *testing.T) {
	cases := []struct{ name, schema, want string }{
		{"string", `{"type":"string"}`, `""`},
		{"integer", `{"type":"integer"}`, `0`},
		{"number", `{"type":"number"}`, `0`},
		{"boolean", `{"type":"boolean"}`, `false`},
		{"null", `{"type":"null"}`, `null`},
		{"string list", `{"type":"array","items":{"type":"string"}}`, "[\n    \"\"\n]"},
		{"nullable string takes the non-null type", `{"type":["null","integer"]}`, `0`},
		{"string enum", `{"enum":["paid","open"]}`, `""`},
		{"numeric enum", `{"enum":[3,4]}`, `0`},
		{"const", `{"const":"USD"}`, `"USD"`},
		{"numeric const", `{"const":42}`, `42`},
		{"type inferred from properties", `{"properties":{"a":{"type":"integer"}},"additionalProperties":false}`, "{\n    \"a\": 0\n}"},
		{"empty closed object", `{"type":"object","properties":{},"additionalProperties":false}`, `{}`},
		{"array of objects shows one element", `{"type":"array","items":{"type":"object","additionalProperties":false,"properties":{"sku":{"type":"string"},"qty":{"type":"integer"}}}}`,
			"[\n    {\n        \"sku\": \"\",\n        \"qty\": 0\n    }\n]"},
		{"object with every kind", `{"type":"object","additionalProperties":false,"properties":{
			"name":{"type":"string"},"n":{"type":"integer"},"x":{"type":"number"},"ok":{"type":"boolean"},
			"tags":{"type":"array","items":{"type":"string"}},
			"inner":{"type":"object","additionalProperties":false,"properties":{"k":{"type":"string"}}}}}`,
			"{\n    \"name\": \"\",\n    \"n\": 0,\n    \"x\": 0,\n    \"ok\": false,\n    \"tags\": [\n        \"\"\n    ],\n    \"inner\": {\n        \"k\": \"\"\n    }\n}"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := TemplateFromSchema([]byte(c.schema))
			if err != nil {
				t.Fatal(err)
			}
			if got != c.want {
				t.Fatalf("got %q\nwant %q", got, c.want)
			}
		})
	}
}

// A property NAMED like a schema keyword is an ordinary key. A reader that took the first "type" key it saw anywhere
// would turn this object into a string template.
func TestTemplateFromSchema_keywordNamedFields(t *testing.T) {
	schema := `{"type":"object","additionalProperties":false,"properties":{
		"type":{"type":"string"},"properties":{"type":"integer"},"items":{"type":"array","items":{"type":"string"}},
		"enum":{"type":"boolean"},"required":{"type":"string"},"const":{"type":"string"}}}`
	got, err := TemplateFromSchema([]byte(schema))
	if err != nil {
		t.Fatal(err)
	}
	want := "{\n    \"type\": \"\",\n    \"properties\": 0,\n    \"items\": [\n        \"\"\n    ],\n    \"enum\": false,\n    \"required\": \"\",\n    \"const\": \"\"\n}"
	if got != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
}

func TestTemplateFromSchema_errors(t *testing.T) {
	for name, schema := range map[string]string{
		"not json":       `{`,
		"trailing data":  `{"type":"string"}0`,
		"array no items": `{"type":"array"}`,
		"node not obj":   `{"type":"object","properties":{"a":5}}`,
	} {
		if _, err := TemplateFromSchema([]byte(schema)); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

type tmplLine struct {
	SKU   string  `json:"sku"`
	Qty   int     `json:"qty"`
	Price float64 `json:"price"`
}

type tmplBase struct {
	Currency string `json:"currency"`
}

type tmplDoc struct {
	tmplBase                       // promoted, as encoding/json promotes it
	Vendor   string                `json:"vendor"`
	Paid     bool                  `json:"paid"`
	Lines    []tmplLine            `json:"lines"`
	Tags     []string              `json:"tags,omitempty"`
	Note     *string               `json:"note"`
	When     time.Time             `json:"when"`
	Blob     []byte                `json:"blob"`
	Skipped  string                `json:"-"`
	Fixed    [2]int                `json:"fixed"`
	Inner    struct{ A, B string } `json:"inner"`
	Type     string                `json:"type"` // a keyword-named field
	hidden   string
}

// TemplateFromStruct follows the struct's FIELD order (not the schema map's alphabetical one), the same convention,
// pointers and omitempty fields included, []byte and time.Time as strings, slices as ONE example element rather than the
// null a marshalled zero value gives, json:"-" and unexported fields absent, an embedded struct promoted.
func TestTemplateFromStruct(t *testing.T) {
	_ = tmplDoc{}.hidden
	got, err := TemplateFromStruct(tmplDoc{})
	if err != nil {
		t.Fatal(err)
	}
	want := `{
    "currency": "",
    "vendor": "",
    "paid": false,
    "lines": [
        {
            "sku": "",
            "qty": 0,
            "price": 0
        }
    ],
    "tags": [
        ""
    ],
    "note": "",
    "when": "",
    "blob": "",
    "fixed": [
        0
    ],
    "inner": {
        "A": "",
        "B": ""
    },
    "type": ""
}`
	if got != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
	// The zero value marshalled is NOT the template: a nil slice is null. That is why this function exists.
	if zero, _ := json.Marshal(tmplDoc{}); !strings.Contains(string(zero), `"lines":null`) {
		t.Errorf("premise changed: the marshalled zero value is %s", zero)
	}
}

// The single-source-of-truth invariant: the template built from a struct and the template built from the schema derived
// from the same struct describe the same shape (compared as parsed JSON, since the schema map sorts its keys), and the
// struct decodes the template without error, so a model that echoes the template back is already a valid value.
func TestTemplateFromStruct_agreesWithSchemaAndDecodes(t *testing.T) {
	v := tmplDoc{}
	st, err := TemplateFromStruct(v)
	if err != nil {
		t.Fatal(err)
	}
	schema, err := SchemaFromStruct(v)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := TemplateFromSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal([]byte(st), &a); err != nil {
		t.Fatalf("struct template is not JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(sc), &b); err != nil {
		t.Fatalf("schema template is not JSON: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("the struct's template and its schema's template disagree:\n%s\n%s", st, sc)
	}
	var back tmplDoc
	if err := json.Unmarshal([]byte(st), &back); err == nil {
		t.Fatal("expected the blank time.Time to be rejected by Unmarshal (\"\" is not RFC 3339): the test's premise changed")
	}
	type plain struct {
		Name  string     `json:"name"`
		N     int        `json:"n"`
		Lines []tmplLine `json:"lines"`
	}
	pt, err := TemplateFromStruct(plain{})
	if err != nil {
		t.Fatal(err)
	}
	var p plain
	if err := json.Unmarshal([]byte(pt), &p); err != nil {
		t.Fatalf("a struct decodes its own template: %v", err)
	}
	g, err := GrammarFromStruct(plain{})
	if err != nil || g == nil {
		t.Fatalf("grammar: %v", err)
	}
}

func TestTemplateFromStruct_refusesWhatTheGrammarRefuses(t *testing.T) {
	type rec struct {
		Next []rec `json:"next"`
	}
	for name, v := range map[string]any{"not a struct": 3, "recursive": rec{}, "nested no fields": struct{ In struct{} }{}} {
		if _, err := TemplateFromStruct(v); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
