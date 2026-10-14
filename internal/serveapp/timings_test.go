package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// timingsTurn sends body to route on c and returns the turn's goinfer_timings and its usage, from the response
// (buffered) or from the event that closes the stream: chat's final usage chunk, the Messages message_delta, the
// Responses response.completed.
func timingsTurn(t *testing.T, c *http.Client, route, body string) (tm, usage map[string]any) {
	t.Helper()
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "http://goinfer"+route, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("%s: %d %s", route, resp.StatusCode, raw)
	}
	var objs []map[string]any
	if !strings.Contains(body, `"stream":true`) {
		var m map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
			t.Fatal(err)
		}
		objs = append(objs, m)
	} else {
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			if d, ok := strings.CutPrefix(sc.Text(), "data: "); ok && d != "[DONE]" {
				var m map[string]any
				if err := json.Unmarshal([]byte(d), &m); err != nil {
					t.Fatalf("bad SSE data %q: %v", d, err)
				}
				objs = append(objs, m)
			}
		}
	}
	for _, o := range objs {
		u, _ := o["usage"].(map[string]any)
		if r, ok := o["response"].(map[string]any); ok {
			u, _ = r["usage"].(map[string]any)
		}
		if tm, ok := u["goinfer_timings"].(map[string]any); ok {
			return tm, u
		}
	}
	t.Fatalf("%s %s: no goinfer_timings in the reply", route, body)
	return nil, nil
}

func num(t *testing.T, m map[string]any, k string) float64 {
	t.Helper()
	v, ok := m[k].(float64)
	if !ok {
		t.Fatalf("goinfer_timings has no %s: %v", k, m)
	}
	return v
}

// TestTimings_onEveryRoute is X1's "done" (docs/tasks/task-turn-telemetry-2026-09.md, "X1, registered 2026-10-14"):
// on the CPU tiny model, every route, buffered and streamed, carries goinfer_timings with every field, and the
// values hold together: prefill within time to first token, the host gap within decode, the prefill split adding up
// to the prompt, and decode counting at least the completion's tokens.
func TestTimings_onEveryRoute(t *testing.T) {
	c, closer := inprocHandler(t)
	defer closer.Close()
	for _, tc := range []struct{ route, body, promptKey, completionKey string }{
		{"/v1/chat/completions", `{"model":"tiny","max_tokens":6,"temperature":0,"messages":[{"role":"user","content":"hi there"}]}`, "prompt_tokens", "completion_tokens"},
		{"/v1/chat/completions", `{"model":"tiny","max_tokens":6,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi there"}]}`, "prompt_tokens", "completion_tokens"},
		{"/v1/messages", `{"model":"tiny","max_tokens":6,"temperature":0,"messages":[{"role":"user","content":"hi there"}]}`, "input_tokens", "output_tokens"},
		{"/v1/messages", `{"model":"tiny","max_tokens":6,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi there"}]}`, "", "output_tokens"},
		{"/v1/responses", `{"model":"tiny","max_output_tokens":6,"temperature":0,"input":"hi there"}`, "input_tokens", "output_tokens"},
		{"/v1/responses", `{"model":"tiny","max_output_tokens":6,"temperature":0,"stream":true,"input":"hi there"}`, "input_tokens", "output_tokens"},
	} {
		name := tc.route
		if strings.Contains(tc.body, `"stream":true`) {
			name += " streamed"
		}
		t.Run(name, func(t *testing.T) {
			tm, u := timingsTurn(t, c, tc.route, tc.body)
			queue, prefill, ttft := num(t, tm, "queue_ms"), num(t, tm, "prefill_ms"), num(t, tm, "ttft_ms")
			decode, gap := num(t, tm, "decode_ms"), num(t, tm, "host_gap_ms")
			computed, reused, dTok := num(t, tm, "prefill_tokens"), num(t, tm, "prefill_reused_tokens"), num(t, tm, "decode_tokens")
			if _, ok := tm["decode_overlapped"].(bool); !ok {
				t.Errorf("no decode_overlapped: %v", tm)
			}
			if queue < 0 || prefill < 0 || prefill > ttft || gap < 0 || gap > decode {
				t.Errorf("inconsistent: queue %v, prefill %v, ttft %v, decode %v, host gap %v", queue, prefill, ttft, decode, gap)
			}
			if tc.promptKey != "" {
				if p := num(t, u, tc.promptKey); computed+reused != p {
					t.Errorf("prefill %v computed + %v reused, the prompt %v", computed, reused, p)
				}
			}
			if comp := num(t, u, tc.completionKey); dTok < comp || dTok == 0 {
				t.Errorf("decode_tokens %v against %v completion tokens", dTok, comp)
			}
		})
	}
}

// TestTimings_absentNotZero: a turn the decoder does not time (here the scripted model, which bypasses it) reports
// serve's own queue and first-token times and leaves every decoder field out, never zero.
func TestTimings_absentNotZero(t *testing.T) {
	ts := scriptedServer(t, "4")
	for _, route := range []string{"/v1/chat/completions", "/v1/messages", "/v1/responses"} {
		objs := postScripted(t, ts.URL+route, requestFor(route, false, false))
		u, _ := objs[0]["usage"].(map[string]any)
		tm, ok := u["goinfer_timings"].(map[string]any)
		if !ok {
			t.Fatalf("%s: no goinfer_timings: %v", route, objs[0]["usage"])
		}
		for _, k := range []string{"queue_ms", "ttft_ms"} {
			if _, ok := tm[k].(float64); !ok {
				t.Errorf("%s: %s absent: %v", route, k, tm)
			}
		}
		for _, k := range []string{"prefill_ms", "prefill_tokens", "prefill_reused_tokens", "decode_ms", "decode_tokens", "host_gap_ms", "decode_overlapped"} {
			if v, ok := tm[k]; ok {
				t.Errorf("%s: %s = %v on a turn the decoder did not time; it must be absent", route, k, v)
			}
		}
	}
}
