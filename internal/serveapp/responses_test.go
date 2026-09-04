package serveapp

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

type respObj struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Status string `json:"status"`
	Model  string `json:"model"`
	Output []struct {
		Type      string `json:"type"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
		Content   []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func (o respObj) text() string {
	for _, it := range o.Output {
		if it.Type == "message" && len(it.Content) > 0 {
			return it.Content[0].Text
		}
	}
	return ""
}

// TestServe_responses is the Inc4 gate for /v1/responses: create (non-stream),
// streaming event shapes, text.format → constrained JSON, a tool round-trip, and
// store/previous_response_id continuation. Gated on GOINFER_SERVE_MODEL.
func TestServe_responses(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the responses test")
	}
	srv, err := newServer(config{models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 2})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	post := func(body string) *http.Response {
		r, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		return r
	}
	create := func(body string) respObj {
		r := post(body)
		defer r.Body.Close()
		if r.StatusCode != 200 {
			t.Fatalf("status %d", r.StatusCode)
		}
		var o respObj
		if err := json.NewDecoder(r.Body).Decode(&o); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return o
	}

	// 1. Non-stream create.
	o := create(`{"model":"m","input":"Say hi in one word.","max_output_tokens":16,"temperature":0}`)
	if o.Object != "response" || o.Status != "completed" || o.text() == "" {
		t.Fatalf("create: object=%q status=%q text=%q", o.Object, o.Status, o.text())
	}
	if o.Usage.InputTokens == 0 || o.Usage.OutputTokens == 0 {
		t.Errorf("usage = %+v", o.Usage)
	}
	t.Logf("create text: %q", o.text())

	// 2. text.format json_schema → constrained, valid JSON conforming to the schema.
	jo := create(`{"model":"m","input":"Return a JSON object for a person with a name and an integer age.","max_output_tokens":80,"temperature":0,
		"text":{"format":{"type":"json_schema","name":"person","schema":
			{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}}}`)
	var person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	dec := json.NewDecoder(strings.NewReader(jo.text()))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&person); err != nil {
		t.Errorf("text.format output not schema JSON: %q (%v)", jo.text(), err)
	}

	// 3. Streaming: response.created + output_text.delta(s) + response.completed.
	r := post(`{"model":"m","input":"Count to three.","max_output_tokens":16,"temperature":0,"stream":true}`)
	defer r.Body.Close()
	var created, deltas, completed int
	var streamed strings.Builder
	sc := bufio.NewScanner(r.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || strings.Contains(line, "[DONE]") {
			continue
		}
		var ev struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
		}
		json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev)
		switch ev.Type {
		case "response.created":
			created++
		case "response.output_text.delta":
			deltas++
			streamed.WriteString(ev.Delta)
		case "response.completed":
			completed++
		}
	}
	if created != 1 || completed != 1 || deltas == 0 {
		t.Errorf("stream events: created=%d deltas=%d completed=%d", created, deltas, completed)
	}
	t.Logf("streamed %d deltas: %q", deltas, streamed.String())

	// 4. Tool round-trip: forced function call → a function_call output item.
	to := create(`{"model":"m","input":"What is the weather in Paris?","max_output_tokens":64,"temperature":0,
		"tool_choice":{"type":"function","function":{"name":"get_weather"}},
		"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city.",
			"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]}`)
	var fc bool
	for _, it := range to.Output {
		if it.Type == "function_call" && it.Name == "get_weather" {
			fc = true
			var args map[string]any
			if json.Unmarshal([]byte(it.Arguments), &args) != nil {
				t.Errorf("function_call arguments not JSON: %q", it.Arguments)
			}
		}
	}
	if !fc {
		t.Errorf("tool round-trip: no get_weather function_call in %+v", to.Output)
	}

	// 5. store/previous_response_id continuation.
	first := create(`{"model":"m","input":"My favorite color is blue.","max_output_tokens":16,"temperature":0,"store":true}`)
	if first.ID == "" {
		t.Fatal("no response id to continue from")
	}
	cont := create(`{"model":"m","input":"What did I say my favorite color was?","max_output_tokens":16,"temperature":0,"previous_response_id":"` + first.ID + `"}`)
	if cont.Status != "completed" || cont.text() == "" {
		t.Errorf("continuation: status=%q text=%q", cont.Status, cont.text())
	}
	t.Logf("continuation text: %q", cont.text())
	// Unknown previous_response_id → 404.
	r404 := post(`{"model":"m","input":"hi","previous_response_id":"resp_doesnotexist"}`)
	r404.Body.Close()
	if r404.StatusCode != http.StatusNotFound {
		t.Errorf("unknown previous_response_id: status %d, want 404", r404.StatusCode)
	}
}

// TestMaybeStore_preservesToolCallsForPreviousResponseID pins V-18 (docs/review-2026-09-04.md):
// maybeStore used to store only the lead text (often empty, when the model went straight into a
// tool call), dropping the tool calls entirely. serveResponsesWith appends a stored entry's
// messages VERBATIM onto the next request's input (see the previous_response_id branch), so a
// later request's own function_call_output — correctly decoded into a {Role:"tool"} turn by
// M-18's fix — would answer a call that, as far as the reconstructed conversation shows, was
// never made: user → assistant("") → tool(result). No model or HTTP server needed — maybeStore
// and the store it writes to are plain Go values.
func TestMaybeStore_preservesToolCallsForPreviousResponseID(t *testing.T) {
	s := &server{responses: newResponseStore(8)}
	tc := apiToolCall{ID: "call_1", Type: "function"}
	tc.Function.Name = "get_weather"
	tc.Function.Arguments = `{"city":"nyc"}`

	s.maybeStore(true, "resp_1", "m", []chatMessage{{Role: "user", Content: rawStr("weather in nyc?")}},
		"", []apiToolCall{tc})

	entry := s.responses.get("resp_1")
	if entry == nil {
		t.Fatal("maybeStore did not store an entry")
	}
	last := entry.messages[len(entry.messages)-1]
	if last.Role != "assistant" {
		t.Fatalf("stored last message role = %q, want assistant", last.Role)
	}
	if len(last.ToolCalls) != 1 || last.ToolCalls[0].ID != "call_1" ||
		last.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("stored assistant turn lost its tool calls: %+v (V-18)", last.ToolCalls)
	}

	// The precise failure mode: continuing with a function_call_output for call_1 must land
	// right after an assistant turn that actually CARRIES call_1, not after an empty one.
	contMsgs, err := responseInputToMessages(json.RawMessage(
		`[{"type":"function_call_output","call_id":"call_1","output":"68F and sunny"}]`))
	if err != nil {
		t.Fatal(err)
	}
	full := append(append([]chatMessage(nil), entry.messages...), contMsgs...)
	if len(full) < 2 {
		t.Fatal("reconstructed conversation is too short to check")
	}
	toolTurn := full[len(full)-1]
	assistantTurn := full[len(full)-2]
	if toolTurn.Role != "tool" || toolTurn.ToolCallID != "call_1" {
		t.Fatalf("tool turn = %+v, want Role=tool ToolCallID=call_1", toolTurn)
	}
	if assistantTurn.Role != "assistant" || len(assistantTurn.ToolCalls) == 0 ||
		assistantTurn.ToolCalls[0].ID != "call_1" {
		t.Errorf("the tool result's matching call is missing from the reconstructed history "+
			"(user → assistant(\"\") → tool(result), V-18): assistant turn = %+v", assistantTurn)
	}
}

// TestRespondTools_callsMaybeStoreWithTheToolCalls is the wiring guard: the unit test above
// proves maybeStore preserves tool calls when GIVEN them, not that respondTools actually passes
// them — the exact shape of gap this session's audit keeps finding (a helper with a test, and a
// call site nobody checked).
func TestRespondTools_callsMaybeStoreWithTheToolCalls(t *testing.T) {
	src, err := os.ReadFile("responses.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	i := strings.Index(body, "func (s *server) respondTools(")
	if i < 0 {
		t.Fatal("respondTools not found — this guard is watching nothing")
	}
	j := strings.Index(body[i:], "\nfunc ")
	if j < 0 {
		j = len(body) - i
	}
	fn := body[i : i+j]
	if !strings.Contains(fn, "s.maybeStore(store, id, lm.name, messages, lead, toolCalls)") {
		t.Error("respondTools no longer calls maybeStore with toolCalls — a tool-calling turn " +
			"would again store no ToolCalls, breaking previous_response_id continuity (V-18)")
	}
}
