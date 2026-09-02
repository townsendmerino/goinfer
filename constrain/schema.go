package constrain

import (
	"bytes"
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
	nonNeg  bool // reject a leading '-' (unsigned Go kinds; JSON Schema minimum:0)

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
	// UseNumber, not plain Unmarshal: an integer enum/const above 2^53 decoded into
	// float64 comes back out of encodeLiteral with different digits, and the grammar
	// then forces a literal the caller never wrote (M-29).
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
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
// schemaKeywords are the keywords compile enforces plus the annotation-only
// keywords it may safely ignore. Any other keyword is an assertion we do NOT enforce
// (pattern, minimum, maxLength, oneOf, $ref, uniqueItems, …) — the package contract
// is that these are a compile error, not a silent no-op, so a caller can't believe a
// constraint is in force that isn't (M27). `format` is annotation-only by JSON Schema
// 2020-12 default, so it's allowed and ignored rather than rejected.
var schemaKeywords = map[string]bool{
	// enforced
	"type": true, "enum": true, "const": true,
	"properties": true, "required": true, "additionalProperties": true, "items": true,
	"minItems": true, "maxItems": true, "minimum": true,
	// annotation-only — allowed, not enforced
	"title": true, "description": true, "default": true, "examples": true, "format": true,
	"$schema": true, "$id": true, "$comment": true,
	"deprecated": true, "readOnly": true, "writeOnly": true,
}

func checkSchemaKeys(s map[string]any) error {
	for k := range s {
		if !schemaKeywords[k] {
			return fmt.Errorf("constrain: unsupported schema keyword %q — it would be silently ignored, so the constraint the caller expects is not enforced; remove it or use a supported keyword", k)
		}
	}
	return nil
}

func compile(s map[string]any) (*node, error) {
	if err := checkSchemaKeys(s); err != nil {
		return nil, err
	}
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
	case "number", "integer":
		nn, err := nonNegativeKeyword(s)
		if err != nil {
			return nil, err
		}
		return &node{kind: kNumber, intOnly: typ == "integer", nonNeg: nn}, nil
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

// validPropertyName rejects property names the key grammar can't match byte-for-byte:
// keyStep treats an unescaped '"' as the key terminator and does not accept a JSON
// escape ('\'), and control bytes never appear literally in a JSON string. Such a
// name compiles but is unsatisfiable — mid-generation every logit goes to −∞, the
// exact wedged state the fuzzer flags as a bug. Reject at compile instead (M27).
func validPropertyName(name string) error {
	for i := 0; i < len(name); i++ {
		if c := name[i]; c == '"' || c == '\\' || c < 0x20 || c == 0x7f {
			return fmt.Errorf("constrain: property name %q contains a byte (0x%02x) the key grammar cannot match (\", \\, or a control character)", name, c)
		}
	}
	return nil
}

func compileObject(s map[string]any) (*node, error) {
	// Closed objects only: the grammar can't enforce an open additionalProperties
	// (that would need a free-JSON sub-grammar). Reject an explicit `true`.
	apClosed := false
	if ap, ok := s["additionalProperties"]; ok {
		if b, isBool := ap.(bool); !isBool || b {
			return nil, fmt.Errorf("constrain: only additionalProperties:false is supported")
		}
		apClosed = true // ok && the value is literal false
	}
	propsRaw, _ := s["properties"].(map[string]any)
	if len(propsRaw) > 64 {
		return nil, fmt.Errorf("constrain: object with >64 properties unsupported")
	}
	// An object with no declared properties and no `additionalProperties:false` is the
	// freeform "any object" shape (the standard freeform tool-arguments schema). The
	// grammar can only build a CLOSED object, so it would compile to "{} only" — far
	// tighter than the schema means. Reject it loudly; an explicit closed empty object
	// (additionalProperties:false, no properties) legitimately matches just "{}" (M27).
	if len(propsRaw) == 0 && !apClosed {
		return nil, fmt.Errorf("constrain: object with no properties is unconstrainable (a freeform object needs a free-JSON sub-grammar); declare properties or set additionalProperties:false for an empty object")
	}
	names := make([]string, 0, len(propsRaw))
	for name := range propsRaw {
		if err := validPropertyName(name); err != nil {
			return nil, err
		}
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
	// Both forms: json.Number from the UseNumber decode above, float64 from a
	// map built in Go (mustMap, and callers that construct a schema directly).
	var f float64
	switch v := raw.(type) {
	case json.Number:
		p, err := v.Float64()
		if err != nil {
			return 0, true, fmt.Errorf("constrain: %s must be a number", key)
		}
		f = p
	case float64:
		f = v
	default:
		return 0, true, fmt.Errorf("constrain: %s must be a number", key)
	}
	if f < 0 || f != math.Trunc(f) {
		return 0, true, fmt.Errorf("constrain: %s must be a non-negative integer, got %v", key, raw)
	}
	return int(f), true, nil
}

// encodeLiteral renders an enum/const value to the compact JSON bytes the model must
// reproduce exactly.
//
// NOT json.Marshal, which HTML-ESCAPES by default (M-29). For {"enum":["<",">","="]}
// it emits "\u003c", so the grammar masks the natural continuation `<` at −∞ and a
// greedy model slides to whichever member IS reachable. The output still validates —
// the oracle marshals both sides the same way — so the failure is silent and shows up
// only as the model "preferring" a different literal than it did unconstrained.
//
// json.Number is emitted VERBATIM: the schema is decoded with UseNumber precisely so
// an integer enum above 2^53 is not routed through float64 and re-encoded with lost
// precision, which would make the literal unreachable for a different reason.
func encodeLiteral(v any) ([]byte, error) {
	if n, ok := v.(json.Number); ok {
		return []byte(n.String()), nil
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, fmt.Errorf("constrain: encode enum/const literal: %w", err)
	}
	// Encode appends a newline; the literal must be the exact bytes and nothing more.
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// nonNegativeKeyword reads `minimum`, which is supported for the value 0 ONLY — the
// grammar can enforce "no leading minus" as a prefix rule, which is all minimum:0 is,
// but a general numeric bound needs magnitude comparison the byte FSM does not do.
// Anything else is refused rather than silently ignored, per this compiler's rule that
// an unenforceable constraint is loud. `minimum` was previously not a known keyword at
// all, so every schema carrying it already errored; this only ever widens what compiles.
//
// SchemaFromStruct emits minimum:0 for the unsigned Go kinds (M-28).
func nonNegativeKeyword(s map[string]any) (bool, error) {
	raw, ok := s["minimum"]
	if !ok {
		return false, nil
	}
	var f float64
	switch v := raw.(type) {
	case json.Number:
		p, err := v.Float64()
		if err != nil {
			return false, fmt.Errorf("constrain: minimum must be a number")
		}
		f = p
	case float64:
		f = v
	default:
		return false, fmt.Errorf("constrain: minimum must be a number")
	}
	if f != 0 {
		return false, fmt.Errorf("constrain: only minimum:0 is supported (a general numeric "+
			"bound is not enforceable by a byte grammar), got %v", raw)
	}
	return true, nil
}
