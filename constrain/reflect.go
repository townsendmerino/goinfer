package constrain

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// GrammarFromStruct derives a JSON Schema from a Go struct (via its json tags)
// and compiles it into a Grammar — so a constrained generation can only produce
// JSON that json.Unmarshal accepts into that struct. "A struct the model cannot
// violate."
//
//	type Person struct {
//	    Name  string `json:"name"`
//	    Age   int    `json:"age"`
//	    Email string `json:"email,omitempty"` // optional
//	}
//	g, _ := constrain.GrammarFromStruct(Person{})
//	// drive the Masker with g; then:
//	var p Person
//	_ = json.Unmarshal(out, &p) // always succeeds
func GrammarFromStruct(v any) (Grammar, error) {
	schema, err := SchemaFromStruct(v)
	if err != nil {
		return nil, err
	}
	return JSONSchema(schema)
}

// SchemaFromStruct reflects a Go value's type into a JSON Schema document (the
// supported subset). The value is only inspected for its type; field values are
// ignored. Pointer and omitempty fields are optional; all others are required.
func SchemaFromStruct(v any) ([]byte, error) {
	t := reflect.TypeOf(v)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("constrain: GrammarFromStruct needs a struct, got %v", t)
	}
	m, err := structSchema(t)
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// structSchema builds the object schema for a struct type.
func structSchema(t reflect.Type) (map[string]any, error) {
	props := map[string]any{}
	var required []string
	for f := range t.Fields() {
		if !f.IsExported() {
			continue
		}
		name, optional, skip := jsonField(f)
		if skip {
			continue
		}
		ft := f.Type
		if ft.Kind() == reflect.Pointer { // *T is the optional/nullable form
			optional = true
			ft = ft.Elem()
		}
		ps, err := typeSchema(ft)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", f.Name, err)
		}
		props[name] = ps
		if !optional {
			required = append(required, name)
		}
	}
	m := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		m["required"] = required
	}
	return m, nil
}

// typeSchema maps a Go type to its JSON Schema node.
func typeSchema(t reflect.Type) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Struct:
		return structSchema(t)
	case reflect.Slice, reflect.Array:
		items, err := typeSchema(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	default:
		return nil, fmt.Errorf("unsupported type %v", t)
	}
}

// jsonField resolves a struct field's JSON name and whether it's optional, from
// its `json:"..."` tag. skip is true for `json:"-"`.
func jsonField(f reflect.StructField) (name string, optional, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	name = f.Name
	if tag != "" {
		parts := strings.Split(tag, ",")
		if parts[0] != "" {
			name = parts[0]
		}
		for _, opt := range parts[1:] {
			if opt == "omitempty" {
				optional = true
			}
		}
	}
	return name, optional, false
}
