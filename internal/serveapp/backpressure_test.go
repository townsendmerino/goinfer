package serveapp

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
)

// TestServe_backpressure is the Inc5 gate: with a small --max-queue, a burst of
// concurrent requests to one model gets 429s (queue full) once the bounded queue
// fills, every request returns (no deadlock), and the served ones are 200. Gated
// on GOINFER_SERVE_MODEL.
func TestServe_backpressure(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the backpressure test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8",
		kvSessions: 0, maxQueue: 2, // cap = 1 running + 2 waiting
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	const burst = 12
	var wg sync.WaitGroup
	var mu sync.Mutex
	codes := map[int]int{}
	for range burst {
		wg.Go(func() {
			// A non-trivial generation so the burst overlaps and the queue fills.
			r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
				strings.NewReader(`{"model":"m","max_tokens":48,"temperature":0,"messages":[{"role":"user","content":"Write a haiku about Go."}]}`))
			if err != nil {
				t.Errorf("POST: %v", err)
				return
			}
			retry := r.Header.Get("Retry-After")
			r.Body.Close()
			mu.Lock()
			codes[r.StatusCode]++
			mu.Unlock()
			if r.StatusCode == http.StatusTooManyRequests && retry == "" {
				t.Errorf("429 without Retry-After header")
			}
		})
	}
	wg.Wait() // no deadlock: every request returned

	total := codes[200] + codes[http.StatusTooManyRequests]
	t.Logf("burst %d: 200=%d 429=%d (other=%d)", burst, codes[200], codes[429], burst-total)
	if total != burst {
		t.Errorf("unexpected statuses: %v (want only 200/429)", codes)
	}
	if codes[429] == 0 {
		t.Errorf("expected some 429s under a burst of %d with max-queue 2; got %v", burst, codes)
	}
	if codes[200] == 0 {
		t.Errorf("expected some 200s; got %v", codes)
	}
}
