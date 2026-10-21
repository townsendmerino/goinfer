package serveapp

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// usageReading is one request's prompt accounting: input_tokens and the reused count as the route reports them, the request
// log's reused=, and count_tokens' prompt (Messages only).
type usageReading struct{ input, cacheRead, logged, count int }

// TestMessagesUsage_cacheRead: two requests sharing a prefix, each pair on a fresh server over the committed tiny fixture with
// one KV session. The second reply reports the reused prefix as cache_read_input_tokens and the prefilled rest as input_tokens:
// the request log's reused=, /v1/responses' cached_tokens for the same pair and count_tokens' prompt agree with it. A streamed
// reply's message_delta reports what the unstreamed reply does; its message_start, sent before the prefill, the whole prompt.
func TestMessagesUsage_cacheRead(t *testing.T) {
	user := func(s string) map[string]any { return map[string]any{"role": "user", "content": s} }
	first := []any{user(strings.Repeat("abc ", 10))}
	second := []any{first[0], map[string]any{"role": "assistant", "content": "Sure."}, user("And then?")}
	reusedRE := regexp.MustCompile(`reused=(\d+)`)
	num := func(m map[string]any, k string) int { f, _ := m[k].(float64); return int(f) }

	serve := func(t *testing.T) (string, *bytes.Buffer) {
		srv, err := newServer(config{models: modelFlag{{name: "tiny", path: tinyFillerCheckpoint(t)}}, kvSessions: 1,
			load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
		if err != nil {
			t.Fatalf("newServer on the tiny fixture: %v", err)
		}
		var logbuf bytes.Buffer
		mux := http.NewServeMux()
		mux.HandleFunc("POST /v1/messages", srv.traceRequests(&logbuf, srv.handleMessages))
		mux.HandleFunc("POST /v1/messages/count_tokens", srv.handleCountTokens)
		mux.HandleFunc("POST /v1/responses", srv.traceRequests(&logbuf, srv.handleResponses))
		ts := httptest.NewServer(mux)
		t.Cleanup(ts.Close)
		return ts.URL, &logbuf
	}
	post := func(t *testing.T, url string, body map[string]any) []byte {
		t.Helper()
		b, _ := json.Marshal(body)
		resp, err := http.Post(url, "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d %s", url, resp.StatusCode, raw)
		}
		return raw
	}
	logged := func(t *testing.T, logbuf *bytes.Buffer) int {
		t.Helper()
		m := reusedRE.FindStringSubmatch(logbuf.String())
		if m == nil {
			t.Fatalf("no reused= on the request line: %q", logbuf.String())
		}
		n, _ := strconv.Atoi(m[1])
		return n
	}

	messages := func(t *testing.T, url string, logbuf *bytes.Buffer, msgs []any, stream bool) usageReading {
		t.Helper()
		logbuf.Reset()
		raw := post(t, url+"/v1/messages", map[string]any{"model": "tiny", "max_tokens": 4, "temperature": 0, "messages": msgs, "stream": stream})
		var ct map[string]any
		if err := json.Unmarshal(post(t, url+"/v1/messages/count_tokens", map[string]any{"model": "tiny", "messages": msgs}), &ct); err != nil {
			t.Fatal(err)
		}
		r := usageReading{count: num(ct, "input_tokens"), logged: logged(t, logbuf)}
		var u map[string]any
		if !stream {
			var msg map[string]any
			if err := json.Unmarshal(raw, &msg); err != nil {
				t.Fatal(err)
			}
			u, _ = msg["usage"].(map[string]any)
		} else {
			var start map[string]any
			for _, frame := range strings.Split(string(raw), "\n\n") {
				_, data, ok := strings.Cut(frame, "\ndata: ")
				if !ok {
					continue
				}
				var ev map[string]any
				if err := json.Unmarshal([]byte(data), &ev); err != nil {
					t.Fatalf("event %q: %v", frame, err)
				}
				switch ev["type"] {
				case "message_start":
					start, _ = ev["message"].(map[string]any)["usage"].(map[string]any)
				case "message_delta":
					u, _ = ev["usage"].(map[string]any)
				}
			}
			if _, has := start["cache_read_input_tokens"]; has || num(start, "input_tokens") != r.count {
				t.Errorf("message_start usage %v: want the whole prompt (%d) as input_tokens and no cache_read_input_tokens", start, r.count)
			}
		}
		if _, has := u["cache_read_input_tokens"]; !has {
			t.Fatalf("usage %v has no cache_read_input_tokens", u)
		}
		r.input, r.cacheRead = num(u, "input_tokens"), num(u, "cache_read_input_tokens")
		return r
	}
	responses := func(t *testing.T, url string, logbuf *bytes.Buffer, msgs []any) usageReading {
		t.Helper()
		logbuf.Reset()
		var resp map[string]any
		if err := json.Unmarshal(post(t, url+"/v1/responses", map[string]any{"model": "tiny", "max_output_tokens": 4, "temperature": 0, "input": msgs}), &resp); err != nil {
			t.Fatal(err)
		}
		u, _ := resp["usage"].(map[string]any)
		details, _ := u["input_tokens_details"].(map[string]any)
		return usageReading{input: num(u, "input_tokens"), cacheRead: num(details, "cached_tokens"), logged: logged(t, logbuf)}
	}

	got := map[string][2]usageReading{}
	for _, arm := range []string{"messages", "messages stream", "responses"} {
		url, logbuf := serve(t)
		var pair [2]usageReading
		for i, msgs := range [][]any{first, second} {
			if arm == "responses" {
				pair[i] = responses(t, url, logbuf, msgs)
			} else {
				pair[i] = messages(t, url, logbuf, msgs, arm == "messages stream")
			}
		}
		got[arm] = pair
		t.Logf("%s: first %+v, second %+v", arm, pair[0], pair[1])
	}

	m, s, r := got["messages"], got["messages stream"], got["responses"]
	if m[1].cacheRead <= 0 {
		t.Fatalf("the second request reports cache_read_input_tokens %d: the shared prefix was not reported as reused", m[1].cacheRead)
	}
	for _, arm := range []string{"messages", "messages stream"} {
		for i, x := range got[arm] {
			if x.input+x.cacheRead != x.count {
				t.Errorf("%s request %d: input_tokens %d + cache_read_input_tokens %d is not the prompt (count_tokens %d)", arm, i+1, x.input, x.cacheRead, x.count)
			}
			if x.cacheRead != x.logged {
				t.Errorf("%s request %d: cache_read_input_tokens %d, the request log's reused=%d", arm, i+1, x.cacheRead, x.logged)
			}
		}
	}
	if s != m {
		t.Errorf("streamed %+v and unstreamed %+v usage differ", s, m)
	}
	for i := range 2 {
		if m[i].cacheRead != r[i].cacheRead || m[i].count != r[i].input {
			t.Errorf("request %d: Messages reused %d of %d, /v1/responses cached_tokens %d of %d", i+1, m[i].cacheRead, m[i].count, r[i].cacheRead, r[i].input)
		}
	}
}
