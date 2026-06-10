package constrain

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// JSON Schema → Grammar. JSONSchema compiles a (subset of) JSON Schema into the
// same incremental byte-level Grammar the free JSON validator implements, so the
// streaming Masker drives it unchanged: at each step only the tokens that keep
// the output a valid prefix of a schema-conforming document survive the mask.
//
// Supported subset (the part that matters for structured output):
//   - object: properties (required + optional), additionalProperties:false (closed)
//   - string, number, integer, boolean, null
//   - enum, const
//   - array: items (one schema), minItems, maxItems
//   - arbitrary nesting of the above
//
// Unsupported keywords are a compile error rather than a silent no-op, so a
// caller never thinks a constraint is in force when it isn't.

// schemaKind is the compiled shape of a schema node. boolean/null/const collapse
// onto kEnum (a fixed set of literal encodings), so the grammar has five shapes.
type schemaKind int

const (
	kObject schemaKind = iota
	kArray
	kString
	kNumber // integer is kNumber with intOnly
	kEnum   // enum / const / boolean / null: match one of a fixed literal set
)

// node is a compiled schema node (immutable after compile; shared across frames).
type node struct {
	kind schemaKind

	// object
	props        []propNode // index i ↔ bit i in the seen/cand masks (≤64 props)
	requiredMask uint64     // bit i set if props[i] is required

	// array
	items    *node
	minItems int
	maxItems int // <0 = unbounded

	// number
	intOnly bool

	// enum: the allowed values' compact JSON encodings (≤64 entries)
	enum [][]byte
}

type propNode struct {
	name   string
	schema *node
}

// JSONSchema compiles a JSON Schema document into a Grammar. It returns an error
// for an unsupported keyword/shape (so an unenforceable constraint is loud).
func JSONSchema(schema []byte) (Grammar, error) {
	var doc map[string]any
	if err := json.Unmarshal(schema, &doc); err != nil {
		return nil, fmt.Errorf("constrain: parse schema: %w", err)
	}
	n, err := compile(doc)
	if err != nil {
		return nil, err
	}
	g := &schemaGrammar{root: n}
	g.Reset()
	return g, nil
}

// compile turns one JSON Schema object into a node.
func compile(s map[string]any) (*node, error) {
	// enum / const first — they pin a fixed set of literals regardless of type.
	if c, ok := s["const"]; ok {
		enc, err := encodeLiteral(c)
		if err != nil {
			return nil, err
		}
		return &node{kind: kEnum, enum: [][]byte{enc}}, nil
	}
	if e, ok := s["enum"]; ok {
		vals, ok := e.([]any)
		if !ok || len(vals) == 0 {
			return nil, fmt.Errorf("constrain: enum must be a non-empty array")
		}
		if len(vals) > 64 {
			return nil, fmt.Errorf("constrain: enum with >64 entries unsupported")
		}
		enum := make([][]byte, len(vals))
		for i, v := range vals {
			enc, err := encodeLiteral(v)
			if err != nil {
				return nil, err
			}
			enum[i] = enc
		}
		return &node{kind: kEnum, enum: enum}, nil
	}

	typ, _ := s["type"].(string)
	switch typ {
	case "object":
		return compileObject(s)
	case "array":
		return compileArray(s)
	case "string":
		return &node{kind: kString}, nil
	case "number":
		return &node{kind: kNumber}, nil
	case "integer":
		return &node{kind: kNumber, intOnly: true}, nil
	case "boolean":
		return &node{kind: kEnum, enum: [][]byte{[]byte("true"), []byte("false")}}, nil
	case "null":
		return &node{kind: kEnum, enum: [][]byte{[]byte("null")}}, nil
	case "":
		// "properties" with no explicit type is conventionally an object.
		if _, ok := s["properties"]; ok {
			return compileObject(s)
		}
		return nil, fmt.Errorf("constrain: schema needs a \"type\", \"enum\", or \"const\"")
	default:
		return nil, fmt.Errorf("constrain: unsupported type %q", typ)
	}
}

func compileObject(s map[string]any) (*node, error) {
	// Closed objects only: the grammar can't enforce an open additionalProperties
	// (that would need a free-JSON sub-grammar). Reject an explicit `true`.
	if ap, ok := s["additionalProperties"]; ok {
		if b, isBool := ap.(bool); !isBool || b {
			return nil, fmt.Errorf("constrain: only additionalProperties:false is supported")
		}
	}
	propsRaw, _ := s["properties"].(map[string]any)
	if len(propsRaw) > 64 {
		return nil, fmt.Errorf("constrain: object with >64 properties unsupported")
	}
	names := make([]string, 0, len(propsRaw))
	for name := range propsRaw {
		names = append(names, name)
	}
	sort.Strings(names) // deterministic index→bit assignment
	declared := make(map[string]bool, len(names))
	for _, name := range names {
		declared[name] = true
	}
	// A `required` entry naming an undeclared property is unenforceable: the
	// closed object can never emit that key, so the constraint would be silently
	// dropped and the masker would happily produce output the schema rejects.
	// Reject it loudly (the package's invariant: an unenforceable schema fails to
	// compile rather than mis-constrain at decode time).
	req := map[string]bool{}
	if rraw, present := s["required"]; present {
		rl, ok := rraw.([]any)
		if !ok {
			return nil, fmt.Errorf("constrain: required must be an array of property names")
		}
		for _, r := range rl {
			rs, ok := r.(string)
			if !ok {
				return nil, fmt.Errorf("constrain: required entries must be strings, got %T", r)
			}
			if !declared[rs] {
				return nil, fmt.Errorf("constrain: required property %q is not declared in properties", rs)
			}
			req[rs] = true
		}
	}
	n := &node{kind: kObject}
	for i, name := range names {
		ps, ok := propsRaw[name].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("constrain: property %q is not a schema object", name)
		}
		child, err := compile(ps)
		if err != nil {
			return nil, fmt.Errorf("constrain: property %q: %w", name, err)
		}
		n.props = append(n.props, propNode{name: name, schema: child})
		if req[name] {
			n.requiredMask |= 1 << uint(i)
		}
	}
	return n, nil
}

func compileArray(s map[string]any) (*node, error) {
	itemsRaw, ok := s["items"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("constrain: array needs an \"items\" schema")
	}
	items, err := compile(itemsRaw)
	if err != nil {
		return nil, fmt.Errorf("constrain: array items: %w", err)
	}
	n := &node{kind: kArray, items: items, maxItems: -1}
	if v, present, err := intKeyword(s, "minItems"); err != nil {
		return nil, err
	} else if present {
		n.minItems = v
	}
	if v, present, err := intKeyword(s, "maxItems"); err != nil {
		return nil, err
	} else if present {
		// maxItems < minItems is unsatisfiable: the masker could never close the
		// array (too few items to satisfy minItems, too many to add another), so
		// it would livelock on whitespace. Reject at compile.
		if v < n.minItems {
			return nil, fmt.Errorf("constrain: maxItems %d < minItems %d (unsatisfiable)", v, n.minItems)
		}
		n.maxItems = v
	}
	return n, nil
}

// intKeyword reads a JSON Schema integer keyword. present is false when the key
// is absent; it errors for a non-numeric, non-integral, or negative value (the
// array-bound keywords are defined as non-negative integers).
func intKeyword(s map[string]any, key string) (val int, present bool, err error) {
	raw, ok := s[key]
	if !ok {
		return 0, false, nil
	}
	f, ok := raw.(float64)
	if !ok {
		return 0, true, fmt.Errorf("constrain: %s must be a number", key)
	}
	if f < 0 || f != math.Trunc(f) {
		return 0, true, fmt.Errorf("constrain: %s must be a non-negative integer, got %v", key, raw)
	}
	return int(f), true, nil
}

// encodeLiteral renders an enum/const value to the compact JSON bytes the model
// must reproduce exactly (json.Marshal gives canonical, space-free output).
func encodeLiteral(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("constrain: encode enum/const literal: %w", err)
	}
	return b, nil
}
