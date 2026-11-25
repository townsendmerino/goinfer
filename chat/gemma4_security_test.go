package chat

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGemmaSafe_stripsNestedControlTokens(t *testing.T) {
	nested := `<|<|"|>"|>`
	got := gemmaSafe(nested)
	if strings.Contains(got, gq) {
		t.Fatalf("gemmaSafe(%q) = %q still contains %q", nested, got, gq)
	}
	deep := `<|<|<|"|>"|>"|>`
	if got2 := gemmaSafe(deep); strings.Contains(got2, gq) {
		t.Fatalf("gemmaSafe(%q) = %q still contains %q", deep, got2, gq)
	}
	// Linear-time guard: 50,000 nested levels (~250 KB) must strip completely in milliseconds, not O(N^2).
	const levels = 50000
	var b strings.Builder
	b.Grow(levels * 5)
	for range levels {
		b.WriteString(`<|`)
	}
	for range levels {
		b.WriteString(`"|>`)
	}
	out := gemmaSafe(b.String())
	if out != "" {
		t.Fatalf("gemmaSafe(50000 nested levels) left %d bytes, want empty", len(out))
	}
}

func TestGemmaSchema_deepNestingBoundedAndLinear(t *testing.T) {
	// Build a deeply nested {"items":{"items":...}} schema past maxGemmaSchemaDepth.
	const depth = 500
	var sb strings.Builder
	for range depth {
		sb.WriteString(`{"type":"array","items":`)
	}
	sb.WriteString(`{"type":"string","description":"leaf"}`)
	for range depth {
		sb.WriteByte('}')
	}
	out := gemmaSchema(json.RawMessage(sb.String()))
	if out == "" {
		t.Fatal("gemmaSchema returned empty output")
	}
	if strings.Contains(out, "leaf") {
		t.Fatalf("gemmaSchema did not cap recursion at depth %d", maxGemmaSchemaDepth)
	}
}
