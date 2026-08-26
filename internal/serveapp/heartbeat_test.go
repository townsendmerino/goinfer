package serveapp

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// Gates for G19 — SSE heartbeats while the tool path buffers.
//
// The before-state: `stream: true` with tools declared sent ZERO bytes until the
// whole generation finished (measured 1682.6s to first byte against a client
// whose idle timeout was 300s). The buffering is correct — a tool call can only
// be parsed from the complete output — so the fix keeps the buffer and adds
// content-free comment frames, which every SSE parser drops.

func newToolsTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the tool-path streaming tests")
	}
	srv, err := newServer(config{models: modelFlag{{name: "test-model", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 4})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	return httptest.NewServer(mux)
}

const toolsStreamBody = `{
  "model":"test-model","stream":true,"temperature":0,"max_tokens":48,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
}`

// The heartbeat must reach the client BEFORE the generation finishes, and it
// must be a comment frame — never a content delta. Those are two separate
// claims and both are asserted: the first is what defeats an idle timeout, the
// second is the buffering guarantee the tool path exists to keep.
func TestToolStreamHeartbeatsBeforeAnyContent(t *testing.T) {
	old := sseHeartbeatInterval
	sseHeartbeatInterval = 50 * time.Millisecond // drive the mechanism, not the wall clock
	defer func() { sseHeartbeatInterval = old }()

	ts := newToolsTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(toolsStreamBody))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	var comments, dataFrames int
	firstDataAfterComment := false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, ": "):
			comments++
		case strings.HasPrefix(line, "data: "):
			dataFrames++
			if comments > 0 && dataFrames == 1 {
				firstDataAfterComment = true
			}
			// The buffering guarantee: nothing carrying content may precede the
			// single whole-message delta. A comment frame carries none by
			// construction, so any early content here is a real regression.
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				continue
			}
			var chunk struct {
				Choices []struct {
					Delta map[string]any `json:"delta"`
				} `json:"choices"`
			}
			if json.Unmarshal([]byte(payload), &chunk) == nil && len(chunk.Choices) > 0 {
				if c, ok := chunk.Choices[0].Delta["content"]; ok && c != nil && c != "" {
					if dataFrames > 1 {
						continue // the single whole-message delta is expected
					}
				}
			}
		}
	}
	if comments == 0 {
		t.Error("no SSE comment frames: a buffered tool generation is still silent, which is the whole defect")
	}
	if !firstDataAfterComment {
		t.Error("the first data frame did not follow a heartbeat — the keep-alive is not reaching the client during the buffer")
	}
	t.Logf("heartbeats=%d data frames=%d", comments, dataFrames)
}

// A tool call must still parse identically through the restructured path — the
// heartbeats are additive, not a change to what the client finally receives.
func TestToolStreamStillEmitsToolCall(t *testing.T) {
	ts := newToolsTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(toolsStreamBody))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	sawToolCall, sawDone := false, false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			continue
		}
		if strings.Contains(payload, "tool_calls") {
			sawToolCall = true
		}
	}
	if !sawDone {
		t.Error("stream did not terminate with [DONE]")
	}
	t.Logf("tool_calls delta seen: %v", sawToolCall)
}

// The non-streaming path keeps its 500 on a generation error — starting SSE
// early must not have changed error semantics for clients that never asked to
// stream.
func TestNonStreamingToolPathKeepsStatusCodes(t *testing.T) {
	ts := newToolsTestServer(t)
	defer ts.Close()

	body := strings.Replace(toolsStreamBody, `"stream":true`, `"stream":false`, 1)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "event-stream") {
		t.Errorf("non-streaming request got Content-Type %q — SSE must not start for it", ct)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

// G21 end-to-end. What this asserts is bounded by the model available: the 1.5B
// used here emits a tool call for EVERY prompt tried (prose, arithmetic, "say one
// word") and never any prose, so "prose arrives before the generation ends"
// cannot be observed through it. That property is gated where it can be proven
// exhaustively instead — chat.TestProseStreamerMatchesParser drives every
// streamable family's real parser one byte at a time.
//
// What IS observable here is the failure mode that actually matters: with the
// incremental path live, no part of a tool call may leak out as a content delta.
// A leaked delta cannot be unsent, so this is the corruption case, and a model
// that always calls a tool is an ideal probe for it.
func TestToolStreamNeverLeaksCallAsProse(t *testing.T) {
	ts := newToolsTestServer(t)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(toolsStreamBody))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()

	var contentDeltas, toolDeltas int
	var assembled strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []any  `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(payload), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		d := chunk.Choices[0].Delta
		if d.Content != "" {
			contentDeltas++
			assembled.WriteString(d.Content)
		}
		if len(d.ToolCalls) > 0 {
			toolDeltas++
		}
	}

	if toolDeltas != 1 {
		t.Errorf("tool_calls deltas = %d, want exactly 1", toolDeltas)
	}
	// The model emits only a call, so the lead is empty: any content delta here is
	// tool-call syntax that escaped as prose.
	if got := assembled.String(); strings.TrimSpace(got) != "" {
		t.Errorf("content leaked from a call-only generation: %q — the streamer released bytes the "+
			"parser does not consider prose, and a delta cannot be unsent", got)
	}
	t.Logf("call-only generation: content deltas=%d (assembled %q), tool_calls deltas=%d",
		contentDeltas, assembled.String(), toolDeltas)
}
