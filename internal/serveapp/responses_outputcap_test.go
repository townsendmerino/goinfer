package serveapp

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/connect"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// A /v1/responses request that names no output cap gets the limit `connect` writes into an agent's configuration, not
// the 512 tokens the other routes default to: Codex sends no cap. Through the real handler on the committed tiny chat
// model, whose window is 128 tokens, so the limit is 64 and a 512-token default would run to the room left instead.
func TestResponses_noCapTakesTheWindowsOutputLimit(t *testing.T) {
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: tinyChatCheckpoint(t)}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	type usage struct {
		Status string `json:"status"`
		Usage  struct {
			InputTokens      int `json:"input_tokens"`
			OutputTokens     int `json:"output_tokens"`
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	post := func(path string, body map[string]any) usage {
		t.Helper()
		b, _ := json.Marshal(body)
		resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(string(b)))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", path, resp.StatusCode, raw)
		}
		var u usage
		if err := json.Unmarshal(raw, &u); err != nil {
			t.Fatalf("%s: %v in %s", path, err, raw)
		}
		return u
	}
	const window = 128
	limit := connect.OutputLimit(window)

	// The control: with a cap far over the window the tiny model writes until the window is full, so it does not stop by
	// itself before the limit and the assertion below is about the default, not about where the model happened to end.
	// A fixed seed, because an unseeded request samples.
	ctl := post("/v1/responses", map[string]any{"model": "tiny", "input": "say hi", "temperature": 0, "seed": 1, "max_output_tokens": 500})
	if ctl.Usage.OutputTokens <= limit {
		t.Fatalf("the control wrote %d tokens, not more than the limit %d: this model stops too early for the test to tell the defaults apart", ctl.Usage.OutputTokens, limit)
	}
	got := post("/v1/responses", map[string]any{"model": "tiny", "input": "say hi", "temperature": 0, "seed": 1})
	if got.Usage.OutputTokens != limit || got.Status != "incomplete" {
		t.Errorf("no cap named: %d output tokens, status %q; want connect.OutputLimit(%d) = %d and incomplete", got.Usage.OutputTokens, got.Status, window, limit)
	}
	// The other OpenAI route keeps its own default: with no cap it runs to the room left in the window here.
	chat := post("/v1/chat/completions", map[string]any{"model": "tiny", "temperature": 0, "seed": 1, "messages": []any{map[string]any{"role": "user", "content": "say hi"}}})
	if chat.Usage.CompletionTokens != ctl.Usage.OutputTokens {
		t.Errorf("/v1/chat/completions with no cap wrote %d tokens; want the %d the room allows (its default is unchanged)", chat.Usage.CompletionTokens, ctl.Usage.OutputTokens)
	}
}
