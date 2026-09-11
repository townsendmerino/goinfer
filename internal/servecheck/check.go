// Package servecheck drives a running goinfer server through the conversation a HARNESS
// would, over the real routes, and prints a per-feature verdict with a number.
//
// It exists because of a gap docs/task-embed-and-harness-ux.md §3.4 names: ten of the
// serveapp test files skip without a model, so the routes Claude Code, dsh, Open-WebUI and
// Continue actually drive are exercised by nothing. A route a harness uses that no test
// touches is the "gate that cannot fail" one level up — the same defect class this project
// keeps finding in its own instruments.
//
// It is a CLIENT, not an embedded server: it checks whatever is actually running, with the
// flags its operator chose, which is the thing a harness will meet. That also makes it usable
// against a staging box, and keeps it out of serve's startup path.
//
// Every row prints a NUMBER, not just ok — TTFT and tok/s at a realistic turn size are what a
// harness user needs before choosing a model, and they are exactly what a pass/fail alone
// hides.
package servecheck

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"strings"
	"time"
)

// Result is one row of the report.
type Result struct {
	Name   string
	OK     bool
	Detail string // the number or the reason — never empty
	Skip   bool   // not applicable to this server (e.g. no chat model loaded)
}

// Client drives one server.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string // "" ⇒ discovered from /v1/models
	HTTP    *http.Client
}

func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(c.BaseURL, "/")+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	h := c.HTTP
	if h == nil {
		h = &http.Client{Timeout: 10 * time.Minute}
	}
	return h.Do(req)
}

// errBody reads a failed response into a short, printable reason.
func errBody(resp *http.Response) string {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
	s := strings.TrimSpace(string(b))
	if s == "" {
		return resp.Status
	}
	return resp.Status + ": " + s
}

// Models checks /v1/models and records the served ids. Every later row needs one, so a
// failure here short-circuits the rest rather than producing a column of identical errors.
func (c *Client) Models(ctx context.Context) (Result, []string) {
	resp, err := c.do(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return Result{Name: "models list", Detail: err.Error()}, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{Name: "models list", Detail: errBody(resp)}, nil
	}
	var out struct {
		Data []struct {
			ID         string `json:"id"`
			DecodePath string `json:"decode_path"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return Result{Name: "models list", Detail: "unparseable: " + err.Error()}, nil
	}
	var ids []string
	for _, m := range out.Data {
		ids = append(ids, m.ID)
	}
	if len(ids) == 0 {
		return Result{Name: "models list", OK: true, Detail: "0 models (no generative model loaded)"}, nil
	}
	d := fmt.Sprintf("%d model", len(ids))
	if len(ids) > 1 {
		d += "s"
	}
	if out.Data[0].DecodePath != "" {
		d += " · " + out.Data[0].DecodePath
	}
	return Result{Name: "models list", OK: true, Detail: d}, ids
}

// sse yields the data payload of each SSE frame.
func sse(resp *http.Response, fn func(data string) bool) error {
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		if !fn(strings.TrimSpace(strings.TrimPrefix(line, "data:"))) {
			return nil
		}
	}
	return sc.Err()
}

// Chat streams a short completion and reports TTFT and the inter-token rate — the two numbers
// a harness user needs before picking a model — plus whether a usage block arrived, which
// several harnesses require for accounting and which a stream can silently omit.
func (c *Client) Chat(ctx context.Context, model, prompt string, maxTokens int, label string) Result {
	res := Result{Name: label}
	body := map[string]any{
		"model": model, "stream": true, "temperature": 0, "max_tokens": maxTokens,
		"messages":       []map[string]string{{"role": "user", "content": prompt}},
		"stream_options": map[string]any{"include_usage": true},
	}
	start := time.Now()
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = errBody(resp)
		return res
	}
	var ttft time.Duration
	var n int
	var last time.Time
	usage := false
	_ = sse(resp, func(data string) bool {
		if data == "[DONE]" {
			return false
		}
		var ev struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			return true
		}
		if ev.Usage != nil {
			usage = true
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			if n == 0 {
				ttft = time.Since(start)
			}
			n++
			last = time.Now()
		}
		return true
	})
	if n == 0 {
		res.Detail = "stream produced no tokens"
		return res
	}
	d := fmt.Sprintf("TTFT %.2fs", ttft.Seconds())
	if n > 1 {
		d += fmt.Sprintf(" · %.1f tok/s", float64(n-1)/last.Sub(start.Add(ttft)).Seconds())
	}
	if usage {
		d += " · usage present"
	} else {
		d += " · NO usage block"
	}
	res.OK = usage
	res.Detail = d
	return res
}

// Structured checks the promise the README makes: a schema the model cannot violate. Uses a
// top-level integer on purpose — that is M-27's shape, where StopWhenComplete used to truncate
// at the first complete document and return a single digit.
func (c *Client) Structured(ctx context.Context, model string) Result {
	res := Result{Name: "structured output"}
	body := map[string]any{
		"model": model, "temperature": 0, "max_tokens": 16,
		"messages": []map[string]string{{"role": "user", "content": "Reply with just the number 366."}},
		"response_format": map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "n", "schema": map[string]any{"type": "integer"}},
		},
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = errBody(resp)
		return res
	}
	var out struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		res.Detail = "unparseable response"
		return res
	}
	txt := strings.TrimSpace(out.Choices[0].Message.Content)
	var n json.Number
	if err := json.Unmarshal([]byte(txt), &n); err != nil {
		res.Detail = fmt.Sprintf("%q does not parse as the schema's integer", txt)
		return res
	}
	// V-17 (docs/review-2026-09-04.md): the prompt asks for "366" specifically BECAUSE that is
	// M-27's shape — StopWhenComplete used to stop at the first complete document and return a
	// single truncated digit. A check that only confirms the output PARSES as SOME json.Number
	// cannot detect that: a truncated "3" parses exactly as cleanly as the real "366" and this
	// row would report OK against the very regression it exists to catch. Compare the VALUE.
	if n.String() != "366" {
		res.Detail = fmt.Sprintf(`{"type":"integer"} → %s, want 366 (truncated at the first `+
			`complete digit? M-27)`, n)
		return res
	}
	res.OK, res.Detail = true, fmt.Sprintf(`{"type":"integer"} → %s`, n)
	return res
}

// Stop checks that a stop sequence ends the turn and is NOT echoed — the failure mode is a
// stop string split across token boundaries leaking into the output, which a harness sees as
// corrupted text rather than as a bug here. It also checks that the count actually REACHED the
// stop ("4" before "5"). A reply that never gets there, such as "Sure, I can count.", contains no
// "5" either, and used to pass (audit-2026-09-10 G-12).
func (c *Client) Stop(ctx context.Context, model string) Result {
	res := Result{Name: "stop sequences"}
	body := map[string]any{
		"model": model, "temperature": 0, "max_tokens": 64,
		"messages": []map[string]string{{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."}},
		"stop":     []string{"5"},
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = errBody(resp)
		return res
	}
	var out struct {
		Choices []struct {
			Message      struct{ Content string } `json:"message"`
			FinishReason string                   `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		res.Detail = "unparseable response"
		return res
	}
	txt := out.Choices[0].Message.Content
	if strings.Contains(txt, "5") {
		res.Detail = fmt.Sprintf("stop string leaked into the output: %q", trunc(txt, 60))
		return res
	}
	if !strings.Contains(txt, "4") {
		res.Detail = fmt.Sprintf("the reply never reached the stop sequence, so this proves nothing about it: %q", trunc(txt, 60))
		return res
	}
	res.OK = true
	res.Detail = fmt.Sprintf("not leaked · finish_reason=%q", out.Choices[0].FinishReason)
	return res
}

// CountTokens checks that /v1/messages/count_tokens agrees with what the server then bills as
// prompt usage. A harness budgets context with the first number and is charged the second; a
// silent disagreement is a wrong context-window calculation on every request.
func (c *Client) CountTokens(ctx context.Context, model string) Result {
	res := Result{Name: "count_tokens"}
	const prompt = "The quick brown fox jumps over the lazy dog, repeatedly and with enthusiasm."
	msgs := []map[string]string{{"role": "user", "content": prompt}}

	resp, err := c.do(ctx, http.MethodPost, "/v1/messages/count_tokens",
		map[string]any{"model": model, "messages": msgs})
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = errBody(resp)
		return res
	}
	var counted struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&counted); err != nil {
		res.Detail = "unparseable count_tokens response"
		return res
	}

	r2, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": model, "temperature": 0, "max_tokens": 1,
		"messages": msgs,
	})
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer r2.Body.Close()
	if r2.StatusCode != http.StatusOK {
		res.Detail = errBody(r2)
		return res
	}
	var billed struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(r2.Body).Decode(&billed); err != nil {
		res.Detail = "unparseable completion response"
		return res
	}
	if counted.InputTokens != billed.Usage.PromptTokens {
		res.Detail = fmt.Sprintf("count_tokens says %d, usage bills %d — a harness budgets with the first and pays the second",
			counted.InputTokens, billed.Usage.PromptTokens)
		return res
	}
	res.OK, res.Detail = true, fmt.Sprintf("%d tokens, matches usage", counted.InputTokens)
	return res
}

func trunc(s string, n int) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// visionTestPNG returns a tiny solid-red PNG for the Vision row's fixed test image — generated
// here rather than embedded as an opaque base64 blob, so the color the question asks about is
// defined right next to the question itself.
func visionTestPNG() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	red := color.RGBA{R: 220, G: 20, B: 20, A: 255}
	for y := range 8 {
		for x := range 8 {
			img.Set(x, y, red)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img) // encoding an in-memory RGBA image never fails
	return buf.Bytes()
}

// Vision drives the multimodal round-trip an image-capable harness needs: a fixed, solid-color
// test image plus a question with exactly one right answer. A server with no vision tower loaded
// 400s every image request the same way regardless of which model is targeted
// (vision_serve.go's serveVisionChatWith), so that case is reported as a SKIP, not a failure —
// same "property of how the operator started serve, not a broken route" rule Tools already
// applies to a checkpoint that cannot use tools.
func (c *Client) Vision(ctx context.Context, model string) Result {
	res := Result{Name: "vision"}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(visionTestPNG())
	body := map[string]any{
		"model": model, "temperature": 0, "max_tokens": 16,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "What color is this image? Reply with exactly one word."},
				{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
			},
		}},
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		detail := errBody(resp)
		if strings.Contains(detail, "no vision tower") {
			res.Skip = true
			res.Detail = "no vision tower loaded (start the server with --vision <dir> to check this row)"
			return res
		}
		res.Detail = detail
		return res
	}
	var out struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		res.Detail = "unparseable response"
		return res
	}
	txt := strings.ToLower(strings.TrimSpace(out.Choices[0].Message.Content))
	if !strings.Contains(txt, "red") {
		res.Detail = fmt.Sprintf("answered %q, want a color naming red", trunc(out.Choices[0].Message.Content, 60))
		return res
	}
	res.OK = true
	res.Detail = fmt.Sprintf("correctly named the test image's color: %q", trunc(out.Choices[0].Message.Content, 40))
	return res
}

// Tools drives the round-trip a harness actually needs: the model asks for a tool, the caller
// answers, and the model uses the answer. Both halves are checked, because they fail separately —
// a server can emit a well-formed tool_call and then choke on the `role:"tool"` message coming
// back, which a harness experiences as a conversation that dies on turn two.
//
// Cold-user run 2026-09-06 scenario B could not test this at all: no OpenAI-speaking agent CLI was
// installed on the machine, so the sub-test did not run
// (docs/measurements/cold-user-2026-09-06.md). This row is the part of that coverage which does
// not depend on a third-party CLI being present, and it is the "checkable" half
// docs/task-embed-and-harness-ux.md §3.4 lists under `serve check`.
//
// The tool is deliberately one a small model can get right: one required string argument, an
// obvious call site. A model too small to emit a tool call at all is reported as a SKIP, not a
// failure — that is a fact about the model, and reporting it red would train an operator to
// ignore the row.
func (c *Client) Tools(ctx context.Context, model string) Result {
	res := Result{Name: "tools, OpenAI"}
	tools := []map[string]any{{
		"type": "function",
		"function": map[string]any{
			"name":        "get_weather",
			"description": "Get the current weather in a given city.",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		},
	}}
	msgs := []map[string]any{
		{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
	}
	body := map[string]any{
		"model": model, "temperature": 0, "max_tokens": 96,
		"messages": msgs, "tools": tools,
	}
	call, res := c.toolCall(ctx, body, res)
	if call.id == "" {
		return res
	}

	// Turn two: hand the result back and require the model to consume it. The assistant message
	// must be echoed verbatim-enough that the server can pair the tool result with its call.
	msgs = append(msgs,
		map[string]any{"role": "assistant", "content": nil, "tool_calls": []map[string]any{{
			"id": call.id, "type": "function",
			"function": map[string]any{"name": call.name, "arguments": call.args},
		}}},
		map[string]any{"role": "tool", "tool_call_id": call.id, "content": `{"temp_c":14,"sky":"rain"}`},
	)
	body["messages"] = msgs
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = "tool result rejected: " + err.Error()
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = "tool result rejected: " + errBody(resp)
		return res
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content   string            `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		res.Detail = "unparseable response to the tool result"
		return res
	}
	// A 200 is not a use of the result (audit-2026-09-10 G-12): asking for the tool again is M-18's
	// agent-livelock shape, and an empty answer is a conversation that dies on turn two.
	if msg := out.Choices[0].Message; len(msg.ToolCalls) > 0 {
		res.Detail = "turn two asked for the tool again instead of answering — the agent-livelock shape (M-18)"
		return res
	} else if strings.TrimSpace(msg.Content) == "" {
		res.Detail = "empty answer to the tool result — the conversation dies on turn two"
		return res
	}
	res.OK = true
	res.Detail = fmt.Sprintf("call %s(%s) → result → answer in 2 turns", call.name, trunc(call.args, 32))
	return res
}

// ToolsHarness is Tools' harness-scale twin. R11 (docs/measurements/cold-user-2026-09-06-nobara-pc.md):
// `Tools` above passes with a ONE-function schema — "tools, OpenAI ... ok" — and a real agent
// harness (opencode) driving the SAME server with the same model still printed a tool call as
// plain-text JSON instead of a real OpenAI tool_calls response, twice, burning ~350 s finding
// out. Confirmed by direct isolation that goinfer's own tool-calling was not the fault (a raw
// curl with a one-tool schema got a proper tool_calls response) — the gap is specifically
// "does the model still call the right tool once the schema looks like a real agent's", which
// `Tools`'s minimal schema cannot expose. This row reproduces that shape: a dozen tools with
// nested object parameters, the same target (get_weather) and prompt as `Tools`, so the schema
// SIZE is the only variable that differs between the two rows.
//
// A model that cannot manage this is reported as a SKIP, not a failure — same rule as `Tools`:
// this is a property of the checkpoint (small models under a large tool schema), and a red row
// here would train an operator to ignore it exactly the way `Tools` already guards against for
// the minimal case.
func (c *Client) ToolsHarness(ctx context.Context, model string) Result {
	res := Result{Name: "tools, harness-scale"}
	tools := harnessScaleTools()
	msgs := []map[string]any{
		{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
	}
	body := map[string]any{
		"model": model, "temperature": 0, "max_tokens": 96,
		"messages": msgs, "tools": tools,
	}
	call, res := c.toolCall(ctx, body, res)
	if call.id == "" {
		if res.Skip {
			res.Detail = "model did not call the tool under a harness-scale (" +
				fmt.Sprint(len(tools)) + "-tool) schema — " + res.Detail
		}
		return res
	}
	res.OK = true
	res.Detail = fmt.Sprintf("call %s(%s) among %d tools", call.name, trunc(call.args, 32), len(tools))
	return res
}

// harnessScaleTools is a dozen tools shaped like a real coding-agent's — file read/write/edit,
// shell, search, plus get_weather among them — several with nested object parameters, matching
// what opencode's own "build" agent actually sends (the shape that broke under R11).
func harnessScaleTools() []map[string]any {
	str := map[string]any{"type": "string"}
	fn := func(name, desc string, params map[string]any, required []string) map[string]any {
		return map[string]any{"type": "function", "function": map[string]any{
			"name": name, "description": desc,
			"parameters": map[string]any{"type": "object", "properties": params, "required": required},
		}}
	}
	return []map[string]any{
		fn("read_file", "Read a file's contents.", map[string]any{"path": str, "offset": map[string]any{"type": "integer"}, "limit": map[string]any{"type": "integer"}}, []string{"path"}),
		fn("write_file", "Write content to a file, creating it if absent.", map[string]any{"path": str, "content": str}, []string{"path", "content"}),
		fn("edit_file", "Replace one exact string match in a file.", map[string]any{"path": str, "old_string": str, "new_string": str}, []string{"path", "old_string", "new_string"}),
		fn("list_dir", "List files in a directory.", map[string]any{"path": str}, []string{"path"}),
		fn("grep", "Search file contents by regex.", map[string]any{"pattern": str, "path": str, "case_insensitive": map[string]any{"type": "boolean"}}, []string{"pattern"}),
		fn("glob", "Find files by glob pattern.", map[string]any{"pattern": str}, []string{"pattern"}),
		fn("run_shell", "Run a shell command and return its output.", map[string]any{"command": str, "timeout_seconds": map[string]any{"type": "integer"}}, []string{"command"}),
		fn("git_status", "Show working tree status.", map[string]any{"path": str}, nil),
		fn("web_search", "Search the web for a query.", map[string]any{"query": str, "max_results": map[string]any{"type": "integer"}}, []string{"query"}),
		fn("fetch_url", "Fetch a URL's content.", map[string]any{"url": str}, []string{"url"}),
		fn("get_weather", "Get the current weather in a given city.", map[string]any{"city": str}, []string{"city"}),
		fn("create_task", "Track a task with nested subtasks.", map[string]any{
			"title": str,
			"subtasks": map[string]any{"type": "array", "items": map[string]any{
				"type": "object", "properties": map[string]any{"title": str, "done": map[string]any{"type": "boolean"}},
			}},
		}, []string{"title"}),
	}
}

// toolCallInfo is the one tool call Tools cares about.
type toolCallInfo struct{ id, name, args string }

// toolCall runs turn one and extracts the call. It returns the updated Result so the caller can
// return it directly on any non-call outcome; a zero id means "stop here", with Skip already set
// when the reason is the model rather than the server.
func (c *Client) toolCall(ctx context.Context, body map[string]any, res Result) (toolCallInfo, Result) {
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		res.Detail = err.Error()
		return toolCallInfo{}, res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		res.Detail = errBody(resp)
		return toolCallInfo{}, res
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		res.Detail = "unparseable response"
		return toolCallInfo{}, res
	}
	calls := out.Choices[0].Message.ToolCalls
	if len(calls) == 0 {
		// The server answered cleanly; this model just did not choose the tool. That is a
		// property of the checkpoint, not a broken route.
		res.Skip = true
		res.Detail = fmt.Sprintf("model answered without calling the tool (finish_reason=%q) — "+
			"too small for tool use, or its template has no tool section",
			out.Choices[0].FinishReason)
		return toolCallInfo{}, res
	}
	call := toolCallInfo{id: calls[0].ID, name: calls[0].Function.Name, args: calls[0].Function.Arguments}
	if call.id == "" {
		res.Detail = "tool_call has no id — a harness cannot pair the result with the call"
		return toolCallInfo{}, res
	}
	if call.name != "get_weather" {
		res.Detail = fmt.Sprintf("called %q, not the only tool offered", call.name)
		return toolCallInfo{}, res
	}
	// Arguments are a JSON STRING in the OpenAI shape, not an object; a server that emits an
	// object here breaks every client that json.Unmarshal's the field.
	var args map[string]any
	if err := json.Unmarshal([]byte(call.args), &args); err != nil {
		res.Detail = fmt.Sprintf("arguments are not a JSON string: %q", trunc(call.args, 60))
		return toolCallInfo{}, res
	}
	return call, res
}
