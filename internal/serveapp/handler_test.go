package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/inproc"
)

func inprocHandler(t *testing.T, extra ...string) (*http.Client, io.Closer) {
	t.Helper()
	h, closer, err := NewHandler(append([]string{"-model", "tiny=" + tinyChatCheckpoint(t), "-backend", "cpu", "-quant", "int8int8"}, extra...))
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	return &http.Client{Transport: inproc.Transport(h)}, closer
}

func inprocChat(t *testing.T, c *http.Client, ctx context.Context, body string, header ...string) *http.Response {
	t.Helper()
	req, _ := http.NewRequestWithContext(ctx, "POST", "http://goinfer/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("in-process request: %v", err)
	}
	return resp
}

// The handler NewHandler returns serves the API with no listener: a chat completion answers, a stream arrives as
// separate events, the model list names the model, and after Close the model is gone.
func TestNewHandler_servesTheAPIInProcess(t *testing.T) {
	c, closer := inprocHandler(t)
	resp := inprocChat(t, c, context.Background(), `{"model":"tiny","max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`)
	var out struct {
		Choices []struct {
			Message struct {
				Role string `json:"role"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err := json.Unmarshal(raw, &out); err != nil || resp.StatusCode != 200 || len(out.Choices) != 1 || out.Choices[0].Message.Role != "assistant" || out.Usage.PromptTokens == 0 || out.Usage.CompletionTokens == 0 {
		t.Fatalf("chat completion in process: %d %s (%v)", resp.StatusCode, raw, err)
	}

	resp = inprocChat(t, c, context.Background(), `{"model":"tiny","max_tokens":6,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	events, done := 0, false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if d, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			if d == "[DONE]" {
				done = true
			} else {
				events++
			}
		}
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || events < 2 || !done {
		t.Errorf("stream in process: status %d, %d events, done %v", resp.StatusCode, events, done)
	}

	if err := closer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	resp = inprocChat(t, c, context.Background(), `{"model":"tiny","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}`)
	raw, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("a request after Close: %d %s, want 404 (the model is unloaded)", resp.StatusCode, raw)
	}
}

// -api-key is honoured in process: a request without the key is refused and one with it is served.
func TestNewHandler_apiKey(t *testing.T) {
	c, closer := inprocHandler(t, "-api-key", "s3cret")
	defer closer.Close()
	body := `{"model":"tiny","max_tokens":2,"messages":[{"role":"user","content":"hi"}]}`
	resp := inprocChat(t, c, context.Background(), body)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Errorf("no key: status %d, want 401", resp.StatusCode)
	}
	resp = inprocChat(t, c, context.Background(), body, "Authorization", "Bearer s3cret")
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("with the key: status %d, want 200", resp.StatusCode)
	}
}

// Cancelling a stream's context partway leaves no goroutine behind, and Close returns promptly afterwards because
// nothing is still in flight.
func TestNewHandler_cancelMidStream(t *testing.T) {
	c, closer := inprocHandler(t)
	before := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resp := inprocChat(t, c, ctx, `{"model":"tiny","max_tokens":100,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	sc := bufio.NewScanner(resp.Body)
	got := 0
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "data: ") {
			if got++; got == 2 {
				cancel()
				break
			}
		}
	}
	resp.Body.Close()
	if got < 2 {
		t.Fatalf("the stream ended after %d events, before it could be cancelled", got)
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > before && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > before {
		t.Errorf("%d goroutines after the cancelled stream, %d before it", n, before)
	}
	t0 := time.Now()
	if err := closer.Close(); err != nil || time.Since(t0) > 3*time.Second {
		t.Errorf("Close after a cancelled stream: %v in %s", err, time.Since(t0))
	}
}
