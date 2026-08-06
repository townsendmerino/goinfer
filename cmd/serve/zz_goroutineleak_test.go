package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestServe_goroutineLeakCheck drives the serve request lifecycle (generation goroutines,
// the per-model queue, warm-KV sessions) and the demote loop on a tiny CPU model, then asks
// the runtime's goroutineleak profiler (GOEXPERIMENT=goroutineleakprofile) whether any goinfer
// goroutine is left permanently blocked. Skips when the experiment is off (profile nil) or the
// tiny fixture is absent. Run with:
//
//	GOEXPERIMENT=goroutineleakprofile go test ./cmd/serve -run TestServe_goroutineLeakCheck -v
func TestServe_goroutineLeakCheck(t *testing.T) {
	p := pprof.Lookup("goroutineleak")
	if p == nil {
		t.Skip("goroutineleak profile unavailable — build with GOEXPERIMENT=goroutineleakprofile")
	}
	// The serve path needs a tokenizer; the committed tiny parity fixtures ship weights
	// only, so point at a real checkpoint (a .gguf carries its own tokenizer).
	ckpt := os.Getenv("GOINFER_SERVE_MODEL")
	if ckpt == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the serve goroutine-leak check")
	}
	srv, err := newServer(config{
		models:     modelFlag{{name: "m", path: ckpt}},
		backend:    "cpu",
		quant:      "int8int8",
		kvSessions: 2, // exercise warm-KV session goroutines/state
		maxQueue:   4, // exercise the backpressure queue
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)

	body := `{"model":"m","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"hello there, write something"}]}`
	stream := `{"model":"m","max_tokens":16,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi again"}]}`

	// Burst of overlapping requests: fills the queue (some 429), streams and non-streams,
	// reuses the warm-KV prefix across the shared system/user prefix.
	var wg sync.WaitGroup
	for i := range 16 {
		payload := body
		if i%2 == 0 {
			payload = stream
		}
		wg.Add(1)
		go func(pl string) {
			defer wg.Done()
			r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(pl))
			if err != nil {
				return
			}
			// Fully drain + close so nothing is a client-side artifact.
			buf := make([]byte, 4096)
			for {
				if _, e := r.Body.Read(buf); e != nil {
					break
				}
			}
			r.Body.Close()
		}(payload)
	}
	wg.Wait()

	// Demote loop: start and stop — must exit on the stop signal, not leak.
	stop := make(chan struct{})
	go demoteLoop(srv, time.Millisecond, stop)
	time.Sleep(10 * time.Millisecond)
	close(stop)

	ts.Close() // stop accepting; in-flight already drained above

	// Let everything settle, drop references, GC so the leak analysis is precise.
	time.Sleep(300 * time.Millisecond)
	runtime.GC()
	runtime.GC()

	var sb strings.Builder
	if err := p.WriteTo(&sb, 1); err != nil {
		t.Fatalf("write goroutineleak profile: %v", err)
	}
	report := sb.String()
	for ln := range strings.SplitSeq(report, "\n") {
		if after, ok := strings.CutPrefix(ln, "goroutineleak profile: total "); ok {
			t.Logf("goroutineleak total: %s", after)
			break
		}
	}
	leaks := 0
	for block := range strings.SplitSeq(report, "\n\n") {
		if strings.Contains(block, "@") && strings.Contains(block, "townsendmerino/goinfer") {
			leaks++
		}
	}
	if leaks > 0 {
		t.Errorf("goroutineleak: %d leaked goroutine(s) touching goinfer code:\n%s", leaks, report)
	}
}
