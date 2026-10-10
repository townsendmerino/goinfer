package constrain

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
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
//	_ = json.Unmarshal(out, &p) // the SHAPE is guaranteed; see below
//
// WHAT IS GUARANTEED is the shape: the object's keys, which are required, and each
// value's JSON type. Embedded structs are promoted as encoding/json promotes them, and
// an unsigned field cannot be handed a negative number.
//
// WHAT IS NOT is MAGNITUDE. JSON Schema integer has no width, so a uint8 field can be
// given 99999 and json.Unmarshal will return an error for it. Two other cases that would make
// the schema silently wrong are compile errors instead: a type with its own UnmarshalJSON is
// refused (its fields do not describe the JSON it accepts) unless it decodes from a string via
// UnmarshalText, in which case it maps to "string"; and a struct with no exported fields is
// refused instead of compiling to "{} only".
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
	m, err := structSchema(t, map[reflect.Type]bool{})
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// structSchema builds the object schema for a struct type. visited holds the struct types on
// the current recursion path so a self-referential type (e.g. `type Node struct{ Children
// []Node }`) is a clean error, not a stack overflow: it can never be a finite closed-object
// grammar. It is removed on the way back up, so the same type reused across sibling fields is
// still fine.
func structSchema(t reflect.Type, visited map[reflect.Type]bool) (map[string]any, error) {
	if visited[t] {
		return nil, fmt.Errorf("constrain: recursive type %v is unsupported", t)
	}
	visited[t] = true
	defer delete(visited, t)
	props := map[string]any{}
	var required []string
	for f := range t.Fields() {
		// An anonymous field's reflect name IS its type name, so an embedded struct of UNEXPORTED
		// type reads IsExported()==false. encoding/json does not skip it (its typeFields has the same
		// special case), so its exported promoted fields must reach the schema, or json.Unmarshal
		// would populate fields the schema never mentions.
		if f.Anonymous {
			et := f.Type
			if et.Kind() == reflect.Pointer {
				et = et.Elem()
			}
			if !f.IsExported() && et.Kind() != reflect.Struct {
				continue // embedded field of unexported NON-struct type — json ignores this one too
			}
		} else if !f.IsExported() {
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
		// EMBEDDED STRUCTS ARE PROMOTED, as encoding/json promotes them. An anonymous field with no
		// json tag name contributes its fields to the PARENT object; emitting it as a property called
		// "Base" would give a schema json.Unmarshal accepts while leaving every promoted field zero.
		// An anonymous field WITH a json tag is a normal named field, per json's rules.
		if f.Anonymous && !hasJSONName(f) && ft.Kind() == reflect.Struct && !unmarshalsItself(f.Type) {
			sub, err := structSchema(ft, visited)
			if err != nil {
				return nil, fmt.Errorf("embedded %s: %w", f.Name, err)
			}
			for k, v := range sub["properties"].(map[string]any) {
				// An outer field of the same name shadows the promoted one, which is
				// also json's rule (shallower depth wins).
				if _, clash := props[k]; !clash {
					props[k] = v
				}
			}
			if req, ok := sub["required"].([]string); ok && !optional {
				for _, k := range req {
					if !slices.Contains(required, k) {
						required = append(required, k)
					}
				}
			}
			continue
		}
		ps, err := typeSchema(ft, visited)
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

// hasJSONName reports whether f's json tag supplies an explicit name (as opposed to
// being absent, empty, or options-only like `json:",omitempty"`). An embedded field
// with an explicit name is a normal property, not a promotion.
func hasJSONName(f reflect.StructField) bool {
	tag, ok := f.Tag.Lookup("json")
	if !ok {
		return false
	}
	name, _, _ := strings.Cut(tag, ",")
	return name != ""
}

// unmarshalsItself reports whether t (or *t) decodes from JSON through its own
// UnmarshalJSON / UnmarshalText rather than field-by-field, so its exported fields —
// of which there are often none — say nothing about what JSON it accepts.
func unmarshalsItself(t reflect.Type) (self bool) {
	pt := reflect.PointerTo(t)
	return t.Implements(jsonUnmarshaler) || pt.Implements(jsonUnmarshaler) ||
		t.Implements(textUnmarshaler) || pt.Implements(textUnmarshaler)
}

// decodesFromText reports whether t decodes from a JSON STRING via UnmarshalText —
// time.Time, net/netip types, and most custom scalar wrappers.
func decodesFromText(t reflect.Type) bool {
	return t.Implements(textUnmarshaler) || reflect.PointerTo(t).Implements(textUnmarshaler)
}

var (
	jsonUnmarshaler = reflect.TypeFor[json.Unmarshaler]()
	textUnmarshaler = reflect.TypeFor[encoding.TextUnmarshaler]()
)

// typeSchema maps a Go type to its JSON Schema node. visited threads through to structSchema
// to break recursive-type cycles.
func typeSchema(t reflect.Type, visited map[reflect.Type]bool) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		// minimum:0 so an unsigned field cannot be handed a negative number. This does NOT bound
		// magnitude; see GrammarFromStruct.
		return map[string]any{"type": "integer", "minimum": 0}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Struct:
		// A type that decodes from a JSON string via UnmarshalText is a STRING, not an object
		// (time.Time is the common one): an object schema from its zero exported fields would force
		// `{}`, the one output UnmarshalJSON then rejects.
		if decodesFromText(t) {
			return map[string]any{"type": "string"}, nil
		}
		if unmarshalsItself(t) {
			return nil, fmt.Errorf("constrain: %v has its own UnmarshalJSON, so its fields do "+
				"not describe the JSON it accepts; pass an explicit schema for it", t)
		}
		if !hasExportedFields(t) {
			return nil, fmt.Errorf("constrain: %v has no exported fields, so the only JSON it "+
				"could constrain is {}; pass an explicit schema if that is intended", t)
		}
		return structSchema(t, visited)
	case reflect.Slice, reflect.Array:
		// []byte is encoding/json's own special case: it marshals as a base64 STRING, never as an
		// array of integers, so an element-type schema would make json.Unmarshal fail on every
		// grammar-legal output.
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string"}, nil
		}
		items, err := typeSchema(t.Elem(), visited)
		if err != nil {
			return nil, err
		}
		schema := map[string]any{"type": "array", "items": items}
		// A fixed-size Go array must constrain the model to exactly that many elements: json.Unmarshal
		// into [N]T silently truncates or zero-pads on a wrong count rather than failing, so an
		// unconstrained length would be a silent shape violation.
		if t.Kind() == reflect.Array {
			schema["minItems"] = t.Len()
			schema["maxItems"] = t.Len()
		}
		return schema, nil
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

// hasExportedFields reports whether t has at least one exported, non-json:"-" field.
//
// An anonymous embed of an unexported-TYPE struct counts when that struct has exported fields
// of its own: encoding/json and structSchema both promote them. typeSchema calls this gate for
// every non-top-level struct (SchemaFromStruct does not, for the top-level one), so ignoring
// such an embed would accept a type at the top level and refuse the same type when nested.
func hasExportedFields(t reflect.Type) bool {
	for f := range t.Fields() {
		if f.Anonymous {
			et := f.Type
			if et.Kind() == reflect.Pointer {
				et = et.Elem()
			}
			if !f.IsExported() && et.Kind() == reflect.Struct {
				if hasExportedFields(et) { // recurse: the embed's own exported fields still promote
					return true
				}
				continue
			}
		}
		if !f.IsExported() {
			continue
		}
		if _, _, skip := jsonField(f); !skip {
			return true
		}
	}
	return false
}
