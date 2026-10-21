package serveapp

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// TestLogitBias_reachesTheSamplerOnEveryRoute: logit_bias on /v1/chat/completions, /v1/completions and /v1/responses (OpenAI's
// field) and on /v1/messages (the extension) changes what the model writes. Through the real handlers on the committed tiny chat
// model, greedy, one token: +100 on a token the model would not pick makes it the pick, and -100 on the model's own pick moves
// the pick off it.
func TestLogitBias_reachesTheSamplerOnEveryRoute(t *testing.T) {
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: tinyChatCheckpoint(t)}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatal(err)
	}
	lm := srv.models["tiny"]
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	user := []any{map[string]any{"role": "user", "content": "say hi"}}
	routes := []struct {
		path string
		body func(bias map[string]float64) map[string]any
		text func(raw []byte) string
	}{
		{"/v1/chat/completions",
			func(b map[string]float64) map[string]any {
				return map[string]any{"model": "tiny", "temperature": 0, "max_tokens": 1, "messages": user, "logit_bias": b}
			},
			func(raw []byte) string {
				var r struct {
					Choices []struct {
						Message struct{ Content string } `json:"message"`
					} `json:"choices"`
				}
				_ = json.Unmarshal(raw, &r)
				if len(r.Choices) == 0 {
					return ""
				}
				return r.Choices[0].Message.Content
			}},
		{"/v1/completions",
			func(b map[string]float64) map[string]any {
				return map[string]any{"model": "tiny", "temperature": 0, "max_tokens": 1, "prompt": "say hi", "logit_bias": b}
			},
			func(raw []byte) string {
				var r struct {
					Choices []struct{ Text string } `json:"choices"`
				}
				_ = json.Unmarshal(raw, &r)
				if len(r.Choices) == 0 {
					return ""
				}
				return r.Choices[0].Text
			}},
		{"/v1/responses",
			func(b map[string]float64) map[string]any {
				return map[string]any{"model": "tiny", "temperature": 0, "max_output_tokens": 1, "input": "say hi", "logit_bias": b}
			},
			func(raw []byte) string {
				var r struct {
					Output []struct {
						Content []struct{ Text string } `json:"content"`
					} `json:"output"`
				}
				_ = json.Unmarshal(raw, &r)
				var s strings.Builder
				for _, o := range r.Output {
					for _, c := range o.Content {
						s.WriteString(c.Text)
					}
				}
				return s.String()
			}},
		{"/v1/messages",
			func(b map[string]float64) map[string]any {
				return map[string]any{"model": "tiny", "temperature": 0, "max_tokens": 1, "messages": user, "logit_bias": b}
			},
			func(raw []byte) string {
				var r struct {
					Content []struct{ Text string } `json:"content"`
				}
				_ = json.Unmarshal(raw, &r)
				if len(r.Content) == 0 {
					return ""
				}
				return r.Content[0].Text
			}},
	}
	post := func(path string, body map[string]any) string {
		t.Helper()
		if body["logit_bias"] == nil || len(body["logit_bias"].(map[string]float64)) == 0 {
			delete(body, "logit_bias")
		}
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
		return string(raw)
	}
	idOf := func(s string) int {
		t.Helper()
		ids, err := lm.tk.Encode(s, false)
		if err != nil || len(ids) != 1 {
			t.Fatalf("encode %q: %v %v, want one id", s, ids, err)
		}
		return ids[0]
	}
	bans := 0
	for _, r := range routes {
		plain := r.text([]byte(post(r.path, r.body(nil))))
		target := "Q"
		if plain == target {
			target = "Z"
		}
		x := idOf(target)
		forced := r.text([]byte(post(r.path, r.body(map[string]float64{strconv.Itoa(x): 100}))))
		if forced != target {
			t.Errorf("%s: logit_bias +100 on %q (id %d) wrote %q; the unbiased pick was %q", r.path, target, x, forced, plain)
		}
		// The ban: -100 on the greedy pick, where the pick shows as text that is one id (the chat-shaped routes trim the tiny
		// model's first pick to ""). The value takes the same path through prepare on every route, which the +100 case covers.
		pid, err := lm.tk.Encode(plain, false)
		if plain == "" || err != nil || len(pid) != 1 {
			continue
		}
		bans++
		if banned := r.text([]byte(post(r.path, r.body(map[string]float64{strconv.Itoa(pid[0]): -100})))); banned == plain {
			t.Errorf("%s: logit_bias -100 on the greedy pick %q (id %d) still wrote it", r.path, plain, pid[0])
		}
	}
	if bans == 0 {
		t.Fatal("no route's greedy pick was a visible single id, so the -100 case never ran")
	}
}

// TestLogitBias_rejectsWhatItCannotApply: a key that is not a token id in the vocabulary, a value outside [-100,100], or too
// many entries is a 400, so a mistyped bias is never silently dropped.
func TestLogitBias_rejectsWhatItCannotApply(t *testing.T) {
	big := map[string]float64{}
	for i := 0; i <= maxLogitBiasEntries; i++ {
		big[strconv.Itoa(i)] = 1
	}
	for _, c := range []struct {
		name string
		in   map[string]float64
		ok   bool
	}{
		{"a token id and a ban", map[string]float64{"5": -100}, true},
		{"the range's ends", map[string]float64{"0": -100, "255": 100}, true},
		{"not a number", map[string]float64{"abc": 1}, false},
		{"negative id", map[string]float64{"-1": 1}, false},
		{"past the vocabulary", map[string]float64{"256": 1}, false},
		{"over 100", map[string]float64{"5": 100.5}, false},
		{"under -100", map[string]float64{"5": -101}, false},
		{"too many entries", big, false},
	} {
		got, err := parseLogitBias(c.in, 256)
		if c.ok && (err != nil || len(got) != len(c.in)) {
			t.Errorf("%s: %v, %v; want accepted", c.name, got, err)
		}
		if !c.ok && err == nil {
			t.Errorf("%s: accepted; want a 400", c.name)
		}
	}
}
