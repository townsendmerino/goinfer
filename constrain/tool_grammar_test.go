package constrain

import (
	"encoding/json"
	"strings"
	"testing"
)

var weatherParams = []byte(`{"type":"object","additionalProperties":false,
	"properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},
	"required":["location"]}`)

// TestToolGrammar_property: every constrained generation is exactly the family
// wrapper around a JSON object that names the tool and whose arguments conform.
func TestToolGrammar_property(t *testing.T) {
	cases := []struct {
		name, prefix, suffix, argsKey string
		array                         bool
	}{
		{"chatml", "<tool_call>\n", "\n</tool_call>", "arguments", false},
		{"llama3", "", "", "parameters", false},
		{"mistral", "[TOOL_CALLS] ", "", "arguments", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for i := 0; i < 100; i++ {
				g, err := ToolCallGrammar(c.prefix, c.suffix, c.argsKey, "get_weather", c.array, weatherParams)
				if err != nil {
					t.Fatalf("build: %v", err)
				}
				out := string(genConstrained(t, g, int64(i)+1, 0))
				if !strings.HasPrefix(out, c.prefix) || !strings.HasSuffix(out, c.suffix) {
					t.Fatalf("wrapper wrong: %q", out)
				}
				inner := strings.TrimSuffix(strings.TrimPrefix(out, c.prefix), c.suffix)
				// Pull the call object (unwrap the array for Mistral).
				obj := inner
				if c.array {
					var arr []json.RawMessage
					if err := json.Unmarshal([]byte(inner), &arr); err != nil || len(arr) != 1 {
						t.Fatalf("array form wrong: %q (%v)", inner, err)
					}
					obj = string(arr[0])
				}
				var call struct {
					Name       string         `json:"name"`
					Arguments  map[string]any `json:"arguments"`
					Parameters map[string]any `json:"parameters"`
				}
				if err := json.Unmarshal([]byte(obj), &call); err != nil {
					t.Fatalf("call not JSON: %q (%v)", obj, err)
				}
				if call.Name != "get_weather" {
					t.Errorf("name = %q, want get_weather", call.Name)
				}
				args := call.Arguments
				if args == nil {
					args = call.Parameters
				}
				if _, ok := args["location"]; !ok {
					t.Errorf("required arg location missing: %q", obj)
				}
				for k := range args {
					if k != "location" && k != "unit" {
						t.Errorf("unexpected arg %q (additionalProperties:false): %q", k, obj)
					}
				}
			}
		})
	}
}
