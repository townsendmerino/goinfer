package main

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
