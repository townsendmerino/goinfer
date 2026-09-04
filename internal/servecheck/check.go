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
	"encoding/json"
	"fmt"
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
// corrupted text rather than as a bug here.
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
