package constrain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
)

// A JSON TEMPLATE is the empty-valued example of a schema's shape: the object with every key present and every value
// blank. It is what GLM-OCR's information-extraction prompt carries
// (`请按下列JSON格式输出图中信息:` followed by `{"id_number": "", "address": {"street": "", ...}}`, the model card's
// example): the model is told the SHAPE by example, not by a JSON Schema, and the grammar (JSONSchema /
// GrammarFromStruct) is what guarantees it. A schema-only request gives the model no template to read, and a
// template-only one gives no guarantee, so both are built from the one source (docs/tasks/task-glm-ocr-2026-10.md O5).
//
// THE CONVENTION, one table for both entry points (a test pins every row):
//
//	object          every declared property, in the SCHEMA's order (TemplateFromSchema) or the struct's FIELD order
//	                (TemplateFromStruct), optional ones included; nested objects nest
//	string          ""
//	integer, number 0
//	boolean         false
//	null            null
//	enum            "" for string options, 0 for numeric ones, false/true/null for those literals (the first
//	                option's kind; the grammar, not the template, says which values are legal)
//	const           the constant itself
//	array           ONE example element in brackets (the element's own template), so a list of line items shows what
//	                one line item looks like. An array of strings is [""].
//
// Output is indented four spaces, as the card's example is. A key that spells a keyword ("type", "properties",
// "items", "required", "enum") is an ordinary key: only a schema node's own keywords are read as keywords.

// tmplNode is one node of the template tree: an object (keys/kids in order), an array (elem), or a literal.
type tmplNode struct {
	keys []string
	kids []*tmplNode
	elem *tmplNode
	lit  string // a scalar's rendering; "" with keys==nil && elem==nil means the string ""
	kind byte   // 'o' object, 'a' array, 'l' literal
}

// TemplateFromSchema returns the JSON template of a JSON Schema document (the subset JSONSchema compiles). Property
// order is the order the schema document lists them in, which is why this reads the raw bytes with an ordered decoder
// rather than going through a map.
func TemplateFromSchema(schema []byte) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()
	doc, err := readOrdered(dec)
	if err != nil {
		return "", fmt.Errorf("constrain: parse schema: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return "", fmt.Errorf("constrain: parse schema: unexpected data after the top-level value")
	}
	n, err := templateNode(doc, 0)
	if err != nil {
		return "", err
	}
	return n.render(), nil
}

// TemplateFromStruct returns the JSON template of a Go struct type, in the struct's field order: the zero value's shape
// with the template convention's blanks (so slices are one-element examples, not null, which a marshalled zero value
// would give). It refuses what SchemaFromStruct refuses, so the template and the grammar derived from the same value
// cannot disagree about which types are legal.
func TemplateFromStruct(v any) (string, error) {
	if _, err := SchemaFromStruct(v); err != nil {
		return "", err
	}
	t := reflect.TypeOf(v)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	n, err := typeTemplate(t)
	if err != nil {
		return "", err
	}
	return n.render(), nil
}

// orderedObj is a JSON object that remembers its key order.
type orderedObj struct {
	keys []string
	vals map[string]any
}

// readOrdered decodes one JSON value, objects as *orderedObj, arrays as []any, scalars as decoded (json.Number for
// numbers).
func readOrdered(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	d, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch d {
	case '{':
		o := &orderedObj{vals: map[string]any{}}
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return nil, err
			}
			k, _ := kt.(string)
			v, err := readOrdered(dec)
			if err != nil {
				return nil, err
			}
			if _, dup := o.vals[k]; !dup {
				o.keys = append(o.keys, k)
			}
			o.vals[k] = v
		}
		if _, err := dec.Token(); err != nil { // the closing }
			return nil, err
		}
		return o, nil
	case '[':
		arr := []any{}
		for dec.More() {
			v, err := readOrdered(dec)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		if _, err := dec.Token(); err != nil { // the closing ]
			return nil, err
		}
		return arr, nil
	}
	return nil, fmt.Errorf("unexpected delimiter %v", d)
}

func templateNode(v any, depth int) (*tmplNode, error) {
	if depth > maxSchemaDepth {
		return nil, fmt.Errorf("constrain: schema nests deeper than %d levels", maxSchemaDepth)
	}
	s, ok := v.(*orderedObj)
	if !ok {
		return nil, fmt.Errorf("constrain: a schema node must be an object, got %T", v)
	}
	if c, has := s.vals["const"]; has {
		return &tmplNode{kind: 'l', lit: literalOf(c)}, nil
	}
	if e, has := s.vals["enum"]; has {
		if arr, ok := e.([]any); ok && len(arr) > 0 {
			return &tmplNode{kind: 'l', lit: blankOf(arr[0])}, nil
		}
	}
	typ := schemaType(s)
	switch typ {
	case "object":
		n := &tmplNode{kind: 'o'}
		props, _ := s.vals["properties"].(*orderedObj)
		if props == nil {
			return n, nil
		}
		for _, k := range props.keys {
			kid, err := templateNode(props.vals[k], depth+1)
			if err != nil {
				return nil, fmt.Errorf("property %q: %w", k, err)
			}
			n.keys = append(n.keys, k)
			n.kids = append(n.kids, kid)
		}
		return n, nil
	case "array":
		items, ok := s.vals["items"].(*orderedObj)
		if !ok {
			return nil, fmt.Errorf("constrain: array needs an \"items\" schema")
		}
		el, err := templateNode(items, depth+1)
		if err != nil {
			return nil, fmt.Errorf("array items: %w", err)
		}
		return &tmplNode{kind: 'a', elem: el}, nil
	case "integer", "number":
		return &tmplNode{kind: 'l', lit: "0"}, nil
	case "boolean":
		return &tmplNode{kind: 'l', lit: "false"}, nil
	case "null":
		return &tmplNode{kind: 'l', lit: "null"}, nil
	}
	return &tmplNode{kind: 'l', lit: `""`}, nil
}

// schemaType is the node's type: its "type" (the first non-null entry of a list), else inferred from the keywords
// that only one shape has, else "string".
func schemaType(s *orderedObj) string {
	switch t := s.vals["type"].(type) {
	case string:
		return t
	case []any:
		for _, e := range t {
			if str, ok := e.(string); ok && str != "null" {
				return str
			}
		}
	}
	if _, ok := s.vals["properties"]; ok {
		return "object"
	}
	if _, ok := s.vals["items"]; ok {
		return "array"
	}
	return "string"
}

// blankOf is the template literal for an enum whose first option is v: the blank of that option's kind.
func blankOf(v any) string {
	switch v.(type) {
	case json.Number:
		return "0"
	case bool:
		return "false"
	case nil:
		return "null"
	}
	return `""`
}

// literalOf renders a const's value.
func literalOf(v any) string {
	if b, err := encodeLiteral(v); err == nil {
		return string(b)
	}
	return `""`
}

// typeTemplate is templateNode's reflect twin: the same convention, struct field order. Embedded structs are promoted
// the way encoding/json (and structSchema) promote them.
func typeTemplate(t reflect.Type) (*tmplNode, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return &tmplNode{kind: 'l', lit: `""`}, nil
	case reflect.Bool:
		return &tmplNode{kind: 'l', lit: "false"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return &tmplNode{kind: 'l', lit: "0"}, nil
	case reflect.Struct:
		if decodesFromText(t) {
			return &tmplNode{kind: 'l', lit: `""`}, nil
		}
		n := &tmplNode{kind: 'o'}
		if err := addStructFields(n, t); err != nil {
			return nil, err
		}
		return n, nil
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return &tmplNode{kind: 'l', lit: `""`}, nil // []byte is a base64 string in JSON
		}
		el, err := typeTemplate(t.Elem())
		if err != nil {
			return nil, err
		}
		return &tmplNode{kind: 'a', elem: el}, nil
	}
	return nil, fmt.Errorf("unsupported type %v", t)
}

func addStructFields(n *tmplNode, t reflect.Type) error {
	for f := range t.Fields() {
		if f.Anonymous {
			et := f.Type
			if et.Kind() == reflect.Pointer {
				et = et.Elem()
			}
			if !f.IsExported() && et.Kind() != reflect.Struct {
				continue
			}
		} else if !f.IsExported() {
			continue
		}
		name, _, skip := jsonField(f)
		if skip {
			continue
		}
		ft := f.Type
		if ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		if f.Anonymous && !hasJSONName(f) && ft.Kind() == reflect.Struct && !unmarshalsItself(f.Type) {
			sub := &tmplNode{kind: 'o'}
			if err := addStructFields(sub, ft); err != nil {
				return fmt.Errorf("embedded %s: %w", f.Name, err)
			}
			for i, k := range sub.keys {
				if !slices.Contains(n.keys, k) { // an outer field of the same name shadows the promoted one
					n.keys = append(n.keys, k)
					n.kids = append(n.kids, sub.kids[i])
				}
			}
			continue
		}
		kid, err := typeTemplate(ft)
		if err != nil {
			return fmt.Errorf("field %s: %w", f.Name, err)
		}
		if i := slices.Index(n.keys, name); i >= 0 {
			n.kids[i] = kid // a same-named outer field replaces a promoted one in place
			continue
		}
		n.keys = append(n.keys, name)
		n.kids = append(n.kids, kid)
	}
	return nil
}

const templateIndent = "    "

// render writes the tree with four-space indentation, the card's layout.
func (n *tmplNode) render() string {
	var b strings.Builder
	n.write(&b, 0)
	return b.String()
}

func (n *tmplNode) write(b *strings.Builder, depth int) {
	pad := strings.Repeat(templateIndent, depth+1)
	switch n.kind {
	case 'o':
		if len(n.keys) == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteString("{\n")
		for i, k := range n.keys {
			b.WriteString(pad)
			b.WriteString(jsonString(k))
			b.WriteString(": ")
			n.kids[i].write(b, depth+1)
			if i < len(n.keys)-1 {
				b.WriteByte(',')
			}
			b.WriteByte('\n')
		}
		b.WriteString(strings.Repeat(templateIndent, depth))
		b.WriteByte('}')
	case 'a':
		b.WriteString("[\n")
		b.WriteString(pad)
		n.elem.write(b, depth+1)
		b.WriteByte('\n')
		b.WriteString(strings.Repeat(templateIndent, depth))
		b.WriteByte(']')
	default:
		b.WriteString(n.lit)
	}
}

// jsonString quotes s without HTML escaping (a key spelled "a<b" stays that way in the prompt).
func jsonString(s string) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s)
	return strings.TrimSuffix(buf.String(), "\n")
}
