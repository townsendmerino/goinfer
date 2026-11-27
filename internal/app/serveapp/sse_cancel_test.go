package serveapp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/goinfer/engine"
)

// sseProbeLimit is how many tokens sseProbeEngine streams at most; every probe request asks for more, so only a cancelled
// context ends a generation sooner.
const sseProbeLimit = 200

// sseProbeEngine streams tok until its context ends or sseProbeLimit tokens have gone, and counts the generations it
// started and the ones whose context ended them.
type sseProbeEngine struct {
	tok          int
	started, cut atomic.Int32
}

func (e *sseProbeEngine) Start(ctx context.Context, _ engine.Request) *engine.Stream {
	e.started.Add(1)
	ch := make(chan int)
	go func() {
		defer close(ch)
		for range sseProbeLimit {
			if ctx.Err() != nil {
				e.cut.Add(1)
				return
			}
			select {
			case ch <- e.tok:
			case <-ctx.Done():
				e.cut.Add(1)
				return
			}
		}
	}()
	return engine.NewStream(ch, nil, nil)
}

// flakyWriter accepts ok writes, then fails every write as a dead or stalled client does.
type flakyWriter struct {
	hdr http.Header
	ok  int
}

func (f *flakyWriter) Header() http.Header { return f.hdr }
func (f *flakyWriter) WriteHeader(int)     {}
func (f *flakyWriter) Write(b []byte) (int, error) {
	if f.ok <= 0 {
		return 0, errors.New("broken pipe")
	}
	f.ok--
	return len(b), nil
}
func (f *flakyWriter) Flush() {}

// sseProbeServer serves one text model "m" and one vision model "vlm" on Mellum2.1's template, both generating through eng.
func sseProbeServer(t *testing.T) (*server, *sseProbeEngine) {
	t.Helper()
	tk := scriptTokenizer(t)
	ids, err := tk.Encode("a", false)
	if err != nil || len(ids) != 1 {
		t.Fatalf("encode a: %v %v", ids, err)
	}
	imgIDs, err := tk.Encode("<IMG>", false)
	if err != nil || len(imgIDs) == 0 {
		t.Fatalf("encode <IMG>: %v", err)
	}
	eng := &sseProbeEngine{tok: ids[0]}
	s := &server{responses: newResponseStore(8)}
	lm := &loadedModel{tk: tk, name: "m", fakeEngine: eng}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	vlm := &loadedModel{tk: tk, name: "vlm", fakeEngine: eng, vision: fakeVisionTower{tok: imgIDs[0]}}
	vlm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	s.models = map[string]*loadedModel{"m": lm, "vlm": vlm}
	return s, eng
}

// TestSSEStream_writeFailureCancelsGeneration: on every streaming route, a frame that fails to write (a client that went
// away or stopped reading past the write deadline) cancels the context the generation runs under, so the turn ends
// instead of decoding to max_tokens into a dead socket while holding the model's slot.
func TestSSEStream_writeFailureCancelsGeneration(t *testing.T) {
	post := func(h func(http.ResponseWriter, *http.Request), body map[string]any) func(*server, http.ResponseWriter) {
		return func(_ *server, w http.ResponseWriter) {
			b, _ := json.Marshal(body)
			h(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(string(b))))
		}
	}
	with := func(b map[string]any, k string, v any) map[string]any { b[k] = v; return b }
	const many = 4 * sseProbeLimit
	thinking := map[string]any{"type": "enabled", "budget_tokens": many}
	visionMsgs := []any{map[string]any{"role": "user", "content": "describe"}}
	img := []imageRef{{Data: []byte("img")}}
	for _, c := range []struct {
		name string
		ok   int // writes the client accepts before failing
		run  func(*server, http.ResponseWriter)
	}{
		{"chat", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleChat, with(requestFor("/v1/chat/completions", false, true), "max_tokens", many))(s, w)
		}},
		{"chat mid-stream", 3, func(s *server, w http.ResponseWriter) {
			post(s.handleChat, with(requestFor("/v1/chat/completions", false, true), "max_tokens", many))(s, w)
		}},
		{"chat tools", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleChat, with(requestFor("/v1/chat/completions", true, true), "max_tokens", many))(s, w)
		}},
		{"completions", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleCompletions, map[string]any{"model": "m", "prompt": "hi", "stream": true, "max_tokens": many})(s, w)
		}},
		{"messages", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleMessages, with(requestFor("/v1/messages", false, true), "max_tokens", many))(s, w)
		}},
		{"messages thinking", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleMessages, with(with(requestFor("/v1/messages", false, true), "max_tokens", 2*many), "thinking", thinking))(s, w)
		}},
		{"messages tools", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleMessages, with(requestFor("/v1/messages", true, true), "max_tokens", many))(s, w)
		}},
		{"responses", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleResponses, with(requestFor("/v1/responses", false, true), "max_output_tokens", many))(s, w)
		}},
		{"responses tools", 0, func(s *server, w http.ResponseWriter) {
			post(s.handleResponses, with(requestFor("/v1/responses", true, true), "max_output_tokens", many))(s, w)
		}},
		{"vision chat", 0, func(s *server, w http.ResponseWriter) {
			var req chatReq
			b, _ := json.Marshal(map[string]any{"model": "vlm", "stream": true, "max_tokens": many, "messages": visionMsgs})
			if err := json.Unmarshal(b, &req); err != nil {
				t.Fatal(err)
			}
			s.serveVisionChatWith(w, httptest.NewRequest(http.MethodPost, "/", nil), req, img, s.models["vlm"])
		}},
		{"vision messages", 0, func(s *server, w http.ResponseWriter) {
			var req anthropicReq
			b, _ := json.Marshal(map[string]any{"model": "vlm", "stream": true, "max_tokens": many, "messages": visionMsgs})
			if err := json.Unmarshal(b, &req); err != nil {
				t.Fatal(err)
			}
			s.serveVisionMessages(w, httptest.NewRequest(http.MethodPost, "/", nil), &req, s.models["vlm"], img)
		}},
		{"vision messages thinking", 0, func(s *server, w http.ResponseWriter) {
			var req anthropicReq
			b, _ := json.Marshal(map[string]any{"model": "vlm", "stream": true, "max_tokens": 2 * many, "messages": visionMsgs, "thinking": thinking})
			if err := json.Unmarshal(b, &req); err != nil {
				t.Fatal(err)
			}
			s.serveVisionMessages(w, httptest.NewRequest(http.MethodPost, "/", nil), &req, s.models["vlm"], img)
		}},
		{"vision responses", 0, func(s *server, w http.ResponseWriter) {
			n := many
			msgs := []chatMessage{{Role: "user", Content: json.RawMessage(`"describe"`)}}
			s.serveVisionResponses(w, httptest.NewRequest(http.MethodPost, "/", nil), s.models["vlm"],
				responseReq{Model: "vlm", Stream: true}, msgs, img, sampling{MaxTokens: &n}, "resp_probe", 1, false)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			s, eng := sseProbeServer(t)
			c.run(s, &flakyWriter{hdr: http.Header{}, ok: c.ok})
			if eng.started.Load() == 0 {
				t.Fatal("no generation started: the probe did not reach the stream path")
			}
			if cut, started := eng.cut.Load(), eng.started.Load(); cut != started {
				t.Errorf("%d of %d generations ran on after the stream's write failed: the failure did not cancel the generation's context", started-cut, started)
			}
		})
	}
}
