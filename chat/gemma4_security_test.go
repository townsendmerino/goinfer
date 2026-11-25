package chat

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGemmaSafe_stripsNestedControlTokens(t *testing.T) {
	// Single-pass ReplaceAll turned "<|<|\"|>\"|>" into "<|\"|>", breaking out of Gemma 4's string quote.
	nested := `<|<|"|>"|>`
	got := gemmaSafe(nested)
	if strings.Contains(got, gq) {
		t.Fatalf("gemmaSafe(%q) = %q still contains %q", nested, got, gq)
	}
	deep := `<|<|<|"|>"|>"|>`
	if got2 := gemmaSafe(deep); strings.Contains(got2, gq) {
		t.Fatalf("gemmaSafe(%q) = %q still contains %q", deep, got2, gq)
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
