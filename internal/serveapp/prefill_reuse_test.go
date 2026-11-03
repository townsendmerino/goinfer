package serveapp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/townsendmerino/goinfer/engine"
)

// reuseEngine streams ids with a report whose reuse is rep's (KV4's §5.16 vendor extension reads it).
type reuseEngine struct {
	ids []int
	rep engine.Report
}

func (e reuseEngine) Start(context.Context, engine.Request) *engine.Stream {
	ch := make(chan int, len(e.ids))
	for _, id := range e.ids {
		ch <- id
	}
	close(ch)
	rep := e.rep
	return engine.NewStream(ch, func() *engine.Report { return &rep }, nil)
}

// TestUsage_prefillReuseSources is KV4's §5.16 reuse-source extension (docs/tasks/task-kv-architecture-2026-10.md): beside
// usage.prefill_reused_tokens, usage.prefill_reuse splits them into own continuation, shared (from another conversation), host and
// disk, with the blocks promoted, on a streamed and a whole chat completion. With no reuse it is absent.
func TestUsage_prefillReuseSources(t *testing.T) {
	tk := keepTokenizer(t)
	ids, err := tk.Encode("Sure.", false)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		rep  engine.Report
		want map[string]float64 // nil: the extension is absent
	}{
		{"own and shared", engine.Report{PrefillReused: 40, PrefillShared: 24}, map[string]float64{
			"own_tokens": 16, "shared_tokens": 24, "host_tokens": 0, "disk_tokens": 0, "promoted_blocks": 0}},
		{"a host restore", engine.Report{PrefillReused: 30, KVSnapRestored: 30}, map[string]float64{
			"own_tokens": 0, "shared_tokens": 0, "host_tokens": 30, "disk_tokens": 0, "promoted_blocks": 0}},
		{"cold", engine.Report{}, nil},
	} {
		s := &server{}
		lm := &loadedModel{tk: tk, name: "m", fakeEngine: reuseEngine{ids: ids, rep: c.rep}}
		lm.tmpl = s.cfg.tuneTemplate(qwen35Template(t))
		s.models = map[string]*loadedModel{"m": lm}
		mux := http.NewServeMux()
		mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
		ts := httptest.NewServer(mux)
		for _, stream := range []bool{false, true} {
			body := map[string]any{"model": "m", "max_tokens": 8, "stream": stream,
				"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
			if stream {
				body["stream_options"] = map[string]any{"include_usage": true}
			}
			var u map[string]any
			for _, o := range postScripted(t, ts.URL+"/v1/chat/completions", body) {
				if m, ok := o["usage"].(map[string]any); ok {
					u = m
				}
			}
			if u == nil {
				t.Fatalf("%s, stream %v: no usage", c.name, stream)
			}
			if got := u["prefill_reused_tokens"]; got != float64(c.rep.PrefillReused) {
				t.Errorf("%s, stream %v: prefill_reused_tokens %v, want %d", c.name, stream, got, c.rep.PrefillReused)
			}
			src, has := u["prefill_reuse"].(map[string]any)
			if c.want == nil {
				if _, present := u["prefill_reuse"]; present {
					t.Errorf("%s, stream %v: prefill_reuse %v on a cold prefill, want it absent", c.name, stream, u["prefill_reuse"])
				}
				continue
			}
			if !has {
				t.Fatalf("%s, stream %v: no prefill_reuse beside prefill_reused_tokens: %v", c.name, stream, u)
			}
			for k, v := range c.want {
				if src[k] != v {
					t.Errorf("%s, stream %v: prefill_reuse.%s = %v, want %v", c.name, stream, k, src[k], v)
				}
			}
		}
		ts.Close()
	}
}
