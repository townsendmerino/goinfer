package serveapp

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// piOverflowPatterns and opencodeOverflowPatterns are the two agents' own overflow matchers, as their sources list them
// (Pi: packages/ai/src/utils/overflow.ts; opencode: packages/ai/src/provider-error.ts). An error body one of them
// matches makes that agent compact and retry.
var piOverflowPatterns = compileAll(
	`prompt (?:is )?too long`, `prompt exceeds max length`, `request_too_large`, `input is too long for requested model`,
	`exceeds the context window`, `exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))`,
	`input token count.*exceeds the maximum`, `maximum context length is \d+ tokens`,
	`exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?`,
	`input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)`, `exceeds the limit of \d+`,
	`exceeds the available context size`, `greater than the context length`, `context window exceeds limit`,
	`exceeded model token limit`, `too large for model with \d+ maximum context length`,
	`prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?`, `model_context_window_exceeded`,
	`prompt too long; exceeded (?:max )?context length`, `range of input length should be`, `context[_ ]length[_ ]exceeded`,
	`too many tokens`, `token limit exceeded`)

var opencodeOverflowPatterns = compileAll(
	`prompt is too long`, `input is too long for requested model`, `exceeded this model context window limit`,
	`exceeds the context window`, `exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))`,
	`input token count.*exceeds the maximum`, `number of input tokens exceeds maximum length`,
	`tokens in request more than max tokens allowed`, `requested input length \d+ exceeds maximum input length`,
	`maximum context length is \d+ tokens`, `exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?`,
	`input(?: \(\d+ tokens\))? is longer than the model'?s context length`, `exceeds the limit of \d+`,
	`exceeds the available context size`, `greater than the context length`,
	"`inputs` tokens \\+ `max_new_tokens` must be <= \\d+", `context window exceeds limit`, `exceeded model token limit`,
	`context[_ ]length[_ ]exceeded`, `context length is only \d+ tokens`, `input length.*exceeds.*context length`,
	`prompt (?:too long|超长)`, `too large for model with \d+ maximum context length`,
	`prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?`, `model_context_window_exceeded`,
	`too many tokens`, `token limit exceeded`, `request entity too large`, `payload too large`, `request too large`)

// piNotOverflow are Pi's exclusions: a message matching one is never read as an overflow.
var piNotOverflow = compileAll(`rate limit`, `too many requests`)

func compileAll(pats ...string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(pats))
	for i, p := range pats {
		out[i] = regexp.MustCompile("(?i)" + p)
	}
	return out
}

func anyMatch(res []*regexp.Regexp, s string) string {
	for _, re := range res {
		if re.MatchString(s) {
			return re.String()
		}
	}
	return ""
}

// The two branches' wording against both agents' matchers: an overflow compaction can cure is recognised by each, in
// either dialect, and one it cannot cure is recognised by neither, with or without the -ctx remedy.
func TestContextOverflow_wordingAgainstTheAgentsMatchers(t *testing.T) {
	cure := &contextOverflowError{prompt: 20000, ctx: 16384, modelWindow: 32768, fixed: 7000, reserved: 4096}
	for _, dialect := range []string{"openai", "anthropic"} {
		msg := cure.message(dialect)
		if cure.helpless() || anyMatch(piOverflowPatterns, msg) == "" || anyMatch(opencodeOverflowPatterns, msg) == "" || anyMatch(piNotOverflow, msg) != "" {
			t.Errorf("%s: a curable overflow is not recognised by both agents: %q", dialect, msg)
		}
		if !strings.Contains(msg, "-ctx 20001 or more (up to 32768)") {
			t.Errorf("%s: the message does not name the -ctx that would accept the prompt: %q", dialect, msg)
		}
	}
	if msg := cure.message("anthropic"); !strings.HasPrefix(msg, "prompt is too long: 20000 tokens > 16384 maximum") {
		t.Errorf("the Messages wording does not open as Claude Code's matcher expects: %q", msg)
	}
	if msg := cure.message("openai"); !strings.Contains(msg, "maximum context length is 16384 tokens") {
		t.Errorf("the OpenAI wording lacks the phrase its agents match: %q", msg)
	}
	for _, e := range []*contextOverflowError{
		{prompt: 20000, ctx: 16384, modelWindow: 32768, fixed: 13000, reserved: 4096},
		{prompt: 40000, ctx: 32768, modelWindow: 32768, fixed: 30000, reserved: 8192},
	} {
		for _, dialect := range []string{"openai", "anthropic"} {
			msg := e.message(dialect)
			if !e.helpless() {
				t.Fatalf("fixed %d + reserved %d against %d is not read as incurable", e.fixed, e.reserved, e.ctx)
			}
			if p := anyMatch(piOverflowPatterns, msg); p != "" {
				t.Errorf("%s: Pi would compact on an incurable overflow (pattern %s): %q", dialect, p, msg)
			}
			if p := anyMatch(opencodeOverflowPatterns, msg); p != "" {
				t.Errorf("%s: opencode would compact on an incurable overflow (pattern %s): %q", dialect, p, msg)
			}
			for _, bad := range []string{"context length", "context window", "context_length"} {
				if strings.Contains(strings.ToLower(msg), bad) {
					t.Errorf("%s: the incurable message says %q, which agents match anywhere in a 400: %q", dialect, bad, msg)
				}
			}
		}
	}
	if e := (&contextOverflowError{prompt: 20000, ctx: 16384, modelWindow: 32768, fixed: -1, reserved: 4096}); e.helpless() {
		t.Error("a route that did not measure its fixed part was read as incurable")
	}
}

// The room counted for a reply: the request's cap, at least 1,024 and at most a quarter of the window, so a client that
// asks for 32,000 on every request does not turn every overflow into an incurable one.
func TestOverflowReserve(t *testing.T) {
	for _, c := range []struct{ maxTokens, ctx, want int }{
		{32000, 32768, 8192}, {32000, 16384, 4096}, {256, 16384, 1024}, {2000, 16384, 2000}, {8, 128, 32}, {64, 128, 32},
	} {
		if got := overflowReserve(c.maxTokens, c.ctx); got != c.want {
			t.Errorf("overflowReserve(%d, %d) = %d, want %d", c.maxTokens, c.ctx, got, c.want)
		}
	}
}

// Through the real handlers, on the committed tiny chat model (a 128-token window): on each route a conversation that
// is too long gets the dialect's curable overflow, and a system prompt that leaves no room gets the operator message
// with no overflow code.
func TestContextOverflow_throughTheHandlers(t *testing.T) {
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: tinyChatCheckpoint(t)}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	post := func(path string, body map[string]any) (int, string, string) {
		b, _ := json.Marshal(body)
		resp, err := http.Post(ts.URL+path, "application/json", strings.NewReader(string(b)))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get("Content-Type"), string(raw)
	}
	type errBody struct {
		Error struct {
			Message string  `json:"message"`
			Type    string  `json:"type"`
			Code    *string `json:"code"`
		} `json:"error"`
	}
	long, short := strings.Repeat("word ", 60), "say hi"
	bigSystem := strings.Repeat("rule ", 30)
	user := func(s string) []any { return []any{map[string]any{"role": "user", "content": s}} }

	t.Run("chat completions", func(t *testing.T) {
		code, _, raw := post("/v1/chat/completions", map[string]any{"model": "tiny", "max_tokens": 8,
			"messages": []any{map[string]any{"role": "system", "content": "be brief"}, map[string]any{"role": "user", "content": long}}})
		var e errBody
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || e.Error.Code == nil || *e.Error.Code != "context_length_exceeded" || !strings.Contains(e.Error.Message, "maximum context length is 128 tokens") {
			t.Errorf("a long conversation: %d %s", code, raw)
		}
		code, _, raw = post("/v1/chat/completions", map[string]any{"model": "tiny", "max_tokens": 8,
			"messages": []any{map[string]any{"role": "system", "content": bigSystem}, map[string]any{"role": "user", "content": short}}})
		e = errBody{}
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || e.Error.Code != nil || !strings.Contains(e.Error.Message, "shortening the conversation will not help") || anyMatch(opencodeOverflowPatterns, raw) != "" || anyMatch(piOverflowPatterns, raw) != "" {
			t.Errorf("a system prompt that leaves no room: %d %s", code, raw)
		}
	})
	// With tools, the definitions are part of the fixed part: one long description leaves no room, whatever the history.
	t.Run("chat completions with tools", func(t *testing.T) {
		tool := func(desc string) []any {
			return []any{map[string]any{"type": "function", "function": map[string]any{"name": "f", "description": desc,
				"parameters": map[string]any{"type": "object", "properties": map[string]any{}}}}}
		}
		code, _, raw := post("/v1/chat/completions", map[string]any{"model": "tiny", "max_tokens": 8, "tools": tool(strings.Repeat("does a thing ", 12)),
			"messages": user(short)})
		var e errBody
		_ = json.Unmarshal([]byte(raw), &e)
		if strings.Contains(raw, "no tool-calling template") {
			t.Fatalf("the tiny chat model has no tool template, so the tools route is not covered: %s", raw)
		}
		if code != 400 || e.Error.Code != nil || !strings.Contains(e.Error.Message, "shortening the conversation will not help") {
			t.Errorf("tool definitions that leave no room: %d %s", code, raw)
		}
	})
	t.Run("messages", func(t *testing.T) {
		code, _, raw := post("/v1/messages", map[string]any{"model": "tiny", "max_tokens": 8, "system": "be brief", "messages": user(long)})
		var e errBody
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || e.Error.Type != "invalid_request_error" || !regexp.MustCompile(`^prompt is too long: \d+ tokens > 128 maximum`).MatchString(e.Error.Message) {
			t.Errorf("a long conversation: %d %s", code, raw)
		}
		code, _, raw = post("/v1/messages", map[string]any{"model": "tiny", "max_tokens": 8, "system": bigSystem, "messages": user(short)})
		e = errBody{}
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || !strings.Contains(e.Error.Message, "shortening the conversation will not help") || anyMatch(opencodeOverflowPatterns, raw) != "" || anyMatch(piOverflowPatterns, raw) != "" {
			t.Errorf("a system prompt that leaves no room: %d %s", code, raw)
		}
	})
	t.Run("responses", func(t *testing.T) {
		code, _, raw := post("/v1/responses", map[string]any{"model": "tiny", "max_output_tokens": 8, "instructions": "be brief", "input": long})
		var e errBody
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || e.Error.Code == nil || *e.Error.Code != "context_length_exceeded" {
			t.Errorf("a long conversation, not streaming: %d %s", code, raw)
		}
		code, ctype, raw := post("/v1/responses", map[string]any{"model": "tiny", "max_output_tokens": 8, "instructions": "be brief", "input": long, "stream": true})
		var failed struct {
			Type     string `json:"type"`
			Response struct {
				Status string `json:"status"`
				Error  struct {
					Code string `json:"code"`
				} `json:"error"`
			} `json:"response"`
		}
		sc := bufio.NewScanner(strings.NewReader(raw))
		for sc.Scan() {
			if d, ok := strings.CutPrefix(sc.Text(), "data: "); ok && strings.Contains(d, "response.failed") {
				_ = json.Unmarshal([]byte(d), &failed)
			}
		}
		if code != 200 || !strings.HasPrefix(ctype, "text/event-stream") || failed.Type != "response.failed" || failed.Response.Status != "failed" || failed.Response.Error.Code != "context_length_exceeded" {
			t.Errorf("a long conversation, streaming: %d %q %s", code, ctype, raw)
		}
		code, ctype, raw = post("/v1/responses", map[string]any{"model": "tiny", "max_output_tokens": 8, "instructions": bigSystem, "input": short, "stream": true})
		e = errBody{}
		_ = json.Unmarshal([]byte(raw), &e)
		if code != 400 || strings.HasPrefix(ctype, "text/event-stream") || e.Error.Code != nil || !strings.Contains(e.Error.Message, "shortening the conversation will not help") {
			t.Errorf("instructions that leave no room, streaming: %d %q %s", code, ctype, raw)
		}
	})
}
