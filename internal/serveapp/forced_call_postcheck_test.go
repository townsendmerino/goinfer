package serveapp

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// postCheckServer serves one model, "m", on tmpl, whose every generation is reply (one script token per character).
func postCheckServer(t *testing.T, tmpl *chat.Template, reply string) *httptest.Server {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode(reply, false)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{}
	lm := &loadedModel{tk: tk, name: "m", scripted: append([]int{}, ids...)}
	lm.tmpl = s.cfg.tuneTemplate(tmpl)
	s.models = map[string]*loadedModel{"m": lm}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", s.handleResponses)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

// mustCallRequest is requestFor's body with tool_choice set to the route's must-call form (OpenAI "required", Anthropic
// "any"), or to auto.
func mustCallRequest(route string, stream, mustCall bool) map[string]any {
	b := requestFor(route, true, stream)
	switch {
	case route == "/v1/messages" && mustCall:
		b["tool_choice"] = map[string]any{"type": "any"}
	case route == "/v1/messages":
		b["tool_choice"] = map[string]any{"type": "auto"}
	case mustCall:
		b["tool_choice"] = "required"
	default:
		b["tool_choice"] = "auto"
	}
	return b
}

func postRaw(t *testing.T, url string, body map[string]any) (int, string) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(raw)
}

const postCheckProse = "It returns 5."

// leaksProse reports whether a reply carries any of postCheckProse: whole, or streamed a character per delta (each its own
// JSON string, so the first, "I", appears quoted on its own).
func leaksProse(body string) bool {
	return strings.Contains(body, postCheckProse) || strings.Contains(body, `"I"`)
}

// TestForcedCall_postCheckRefusesProse: on a family with no JSON call form (Gemma 4), a must-call tool_choice cannot be
// constrained, so the reply is checked once parsed. A prose reply is an error on every route, streamed or not, and none of
// its prose reaches the client. Controls: the same reply under auto is the answer, and a reply that calls the tool is the
// call.
func TestForcedCall_postCheckRefusesProse(t *testing.T) {
	call := `<|tool_call>call:exec_command{cmd:<|"|>cat calc.go<|"|>}<tool_call|>`
	prose, called := postCheckServer(t, chat.Gemma4(), postCheckProse), postCheckServer(t, chat.Gemma4(), call)
	for _, route := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages"} {
		for _, stream := range []bool{false, true} {
			code, body := postRaw(t, prose.URL+route, mustCallRequest(route, stream, true))
			refused := strings.Contains(body, "tool_choice required a tool call and the model answered without one")
			if stream {
				refused = refused && code == http.StatusOK // headers are out before generation: the error is the stream's
			} else {
				refused = refused && code == http.StatusInternalServerError
			}
			if !refused || leaksProse(body) {
				t.Errorf("%s stream=%v, prose under a must-call choice: %d %s; want the post-check's error and none of the prose", route, stream, code, body)
			}
			if code, body := postRaw(t, prose.URL+route, mustCallRequest(route, stream, false)); code != http.StatusOK || !leaksProse(body) {
				t.Errorf("%s stream=%v, the same prose under auto: %d %s; want the answer", route, stream, code, body)
			}
			if code, body := postRaw(t, called.URL+route, mustCallRequest(route, stream, true)); code != http.StatusOK || !strings.Contains(body, "exec_command") || strings.Contains(body, "answered without one") {
				t.Errorf("%s stream=%v, a call under a must-call choice: %d %s; want the call", route, stream, code, body)
			}
		}
	}
}

// TestForcedCall_enforcementPerModel: /v1/models (and /health, through pathFields) reports how a must-call choice is
// enforced on each model's template.
func TestForcedCall_enforcementPerModel(t *testing.T) {
	native := templateFromGolden(t, "qwen3.5-9b").WithToolFormat(chat.ToolFormatTemplate)
	for name, c := range map[string]struct {
		tmpl *chat.Template
		want string
	}{
		"chatml (Mellum2.1)":    {mellum21Template(t), "grammar"},
		"llama3":                {chat.Llama3(), "grammar"},
		"gemma4":                {chat.Gemma4(), "post-check"},
		"qwen3.5, native calls": {native, "post-check"},
		"no tool template":      {nil, ""},
	} {
		if got := forcedCallEnforcement(c.tmpl); got != c.want {
			t.Errorf("%s: %q, want %q", name, got, c.want)
		}
	}
	defer func(on bool) { toolUnionEnabled = on }(toolUnionEnabled)
	toolUnionEnabled = false
	if got := forcedCallEnforcement(chat.Gemma4()); got != "off" {
		t.Errorf("GOINFER_TOOL_UNION=0: %q, want off", got)
	}
}
