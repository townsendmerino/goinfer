package gemma4tools

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSafe_stripsNestedControlTokens(t *testing.T) {
	nested := `<|<|"|>"|>`
	got := Safe(nested)
	if strings.Contains(got, Quote) {
		t.Fatalf("Safe(%q) = %q still contains %q", nested, got, Quote)
	}
	deep := `<|<|<|"|>"|>"|>`
	if got2 := Safe(deep); strings.Contains(got2, Quote) {
		t.Fatalf("Safe(%q) = %q still contains %q", deep, got2, Quote)
	}
	const levels = 50000
	var b strings.Builder
	b.Grow(levels * 5)
	for range levels {
		b.WriteString(`<|`)
	}
	for range levels {
		b.WriteString(`"|>`)
	}
	out := Safe(b.String())
	if out != "" {
		t.Fatalf("Safe(50000 nested levels) left %d bytes, want empty", len(out))
	}
}

func TestSchema_deepNestingBoundedAndLinear(t *testing.T) {
	const depth = 500
	var sb strings.Builder
	for range depth {
		sb.WriteString(`{"type":"array","items":`)
	}
	sb.WriteString(`{"type":"string","description":"leaf"}`)
	for range depth {
		sb.WriteByte('}')
	}
	out := Schema(json.RawMessage(sb.String()))
	if out == "" {
		t.Fatal("Schema returned empty output")
	}
	if strings.Contains(out, "leaf") {
		t.Fatalf("Schema did not cap recursion at depth %d", MaxSchemaDepth)
	}
}
