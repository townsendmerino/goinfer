package serveapp

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// H-a (docs/measurements/harness-dialects-2026-10-09.md): /v1/responses decoded `tools` only in the Chat Completions
// shape, so a Responses-shaped function tool — {"type":"function","name",…} at top level, what Codex and every harness's
// Responses mode send — arrived with an empty name and no schema, under HTTP 200.

func TestToolSpec_decodesBothWireShapes(t *testing.T) {
	params := `{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`
	var nested, flat toolSpec
	if err := json.Unmarshal([]byte(`{"type":"function","function":{"name":"get_weather","description":"Weather.","parameters":`+params+`}}`), &nested); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"type":"function","name":"get_weather","description":"Weather.","parameters":`+params+`,"strict":false}`), &flat); err != nil {
		t.Fatal(err)
	}
	if nested.Function.Name != "get_weather" || string(nested.Function.Parameters) != params {
		t.Fatalf("nested shape changed: %+v", nested)
	}
	if !reflect.DeepEqual(nested, flat) {
		t.Errorf("flat (Responses) shape decodes differently from nested:\n nested %+v\n flat   %+v", nested, flat)
	}
	// Nested wins when a body somehow carries both.
	var both toolSpec
	if err := json.Unmarshal([]byte(`{"type":"function","name":"outer","function":{"name":"inner"}}`), &both); err != nil {
		t.Fatal(err)
	}
	if both.Function.Name != "inner" {
		t.Errorf("both shapes: name %q, want the nested one", both.Function.Name)
	}
}

func TestFunctionTools_dropsWhatCannotRunAndNamesIt(t *testing.T) {
	var tools []toolSpec
	body := `[
	  {"type":"function","name":"shell","parameters":{"type":"object"}},
	  {"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
	  {"type":"web_search"},
	  {"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}
	]`
	if err := json.Unmarshal([]byte(body), &tools); err != nil {
		t.Fatal(err)
	}
	kept, dropped := functionTools(tools)
	var names []string
	for _, k := range kept {
		names = append(names, k.Function.Name)
	}
	if !reflect.DeepEqual(names, []string{"shell", "read_file"}) {
		t.Errorf("kept %v, want [shell read_file]", names)
	}
	if !reflect.DeepEqual(dropped, []string{"custom:apply_patch", "web_search"}) {
		t.Errorf("dropped %v, want [custom:apply_patch web_search]", dropped)
	}

	w := httptest.NewRecorder()
	dropUnrunnableTools(w, &tools)
	if got := w.Header().Get("X-Goinfer-Tools-Dropped"); got != "custom:apply_patch,web_search" {
		t.Errorf("header = %q", got)
	}
	if len(tools) != 2 {
		t.Errorf("request tools not filtered in place: %d left", len(tools))
	}
}

func TestForcedTool_acceptsTheResponsesToolChoiceShape(t *testing.T) {
	tools := []chat.Tool{{Name: "a"}, {Name: "b"}}
	for _, tc := range []string{
		`{"type":"function","function":{"name":"b"}}`, // Chat Completions
		`{"type":"function","name":"b"}`,              // Responses
	} {
		got := forcedTool(json.RawMessage(tc), tools, false)
		if got == nil || got.Name != "b" {
			t.Errorf("tool_choice %s: forced %+v, want b", tc, got)
		}
	}
}
