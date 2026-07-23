package constrain

import "testing"

// TestSchema_M27_hardening gates the schema-compile contract: an unsupported
// assertion keyword is a compile error (not a silently dropped constraint), a
// property name the key grammar can't match byte-for-byte is rejected, and a
// freeform object (no properties, not closed) is rejected while an explicit closed
// empty object is allowed.
func TestSchema_M27_hardening(t *testing.T) {
	// (a) unsupported assertion keywords → error.
	for _, s := range []string{
		`{"type":"string","pattern":"^x$"}`,
		`{"type":"integer","minimum":3}`,
		`{"type":"string","minLength":2}`,
		`{"type":"string","maxLength":8}`,
		`{"oneOf":[{"type":"string"}]}`,
		`{"$ref":"#/$defs/x"}`,
		`{"type":"array","items":{"type":"integer"},"uniqueItems":true}`,
	} {
		if _, err := JSONSchema([]byte(s)); err == nil {
			t.Errorf("unsupported-keyword schema compiled without error: %s", s)
		}
	}
	// annotation-only keywords are allowed (and ignored). format is annotation-only
	// by JSON Schema 2020-12 default.
	for _, s := range []string{
		`{"type":"string","description":"a name","title":"Name"}`,
		`{"type":"string","format":"email"}`,
		`{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false,"$comment":"hi"}`,
	} {
		if _, err := JSONSchema([]byte(s)); err != nil {
			t.Errorf("annotation-only schema rejected: %s: %v", s, err)
		}
	}
	// (b) property names with a byte the key grammar can't emit → error.
	for _, s := range []string{
		`{"type":"object","properties":{"a\"b":{"type":"string"}},"additionalProperties":false}`, // embedded quote
		`{"type":"object","properties":{"a\\b":{"type":"string"}},"additionalProperties":false}`, // embedded backslash
	} {
		if _, err := JSONSchema([]byte(s)); err == nil {
			t.Errorf("unsatisfiable property name compiled: %s", s)
		}
	}
	// (c) freeform object rejected; explicit closed empty object allowed.
	if _, err := JSONSchema([]byte(`{"type":"object"}`)); err == nil {
		t.Error("freeform object compiled — want error")
	}
	if _, err := JSONSchema([]byte(`{"type":"object","additionalProperties":false}`)); err != nil {
		t.Errorf("closed empty object rejected: %v", err)
	}
}

// TestSchemaFromStruct_recursive gates M27(d): a self-referential type is a clean
// error, not a stack overflow, while a type reused across sibling fields still works
// (visited is scoped to the recursion path, not global).
func TestSchemaFromStruct_recursive(t *testing.T) {
	type recNode struct {
		Name     string    `json:"name"`
		Children []recNode `json:"children"`
	}
	if _, err := SchemaFromStruct(recNode{}); err == nil {
		t.Error("recursive struct compiled — want error, not stack overflow")
	}

	type leaf struct {
		V string `json:"v"`
	}
	type twin struct {
		A leaf `json:"a"`
		B leaf `json:"b"`
	}
	if _, err := SchemaFromStruct(twin{}); err != nil {
		t.Errorf("sibling reuse of a non-recursive type rejected: %v", err)
	}
}
