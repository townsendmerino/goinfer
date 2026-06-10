package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Track 5 (testing campaign): the untested interaction is multi-model + admin +
// sessions under concurrency. These tests drive real small GGUFs (env-gated, so
// CI skips) and run under -race. Bars: zero races, zero goroutine leaks, every
// response a valid OpenAI shape or the correct status, and a --session-dir
// restart restores warm KV without corrupting the continuation.

// chaosModels builds a 2-model registry from GOINFER_SERVE_MODEL (+ optional
// GOINFER_SERVE_MODEL2; absent ⇒ the same file served under a second name).
func chaosModels(t *testing.T) modelFlag {
	t.Helper()
	p1 := os.Getenv("GOINFER_SERVE_MODEL")
	if p1 == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> (and optionally GOINFER_SERVE_MODEL2) for the chaos test")
	}
	models := modelFlag{{name: "m1", path: p1}}
	if p2 := os.Getenv("GOINFER_SERVE_MODEL2"); p2 != "" {
		models = append(models, modelSpec{name: "m2", path: p2})
	} else {
		models = append(models, modelSpec{name: "m2", path: p1}) // same weights, second served id
	}
	return models
}

func chaosMux(srv *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", srv.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
	mux.HandleFunc("POST /v1/responses", srv.handleResponses)
	mux.HandleFunc("POST /admin/models/load", srv.handleAdminLoad)
	mux.HandleFunc("POST /admin/models/unload", srv.handleAdminUnload)
	return mux
}

// okStatus is the set of acceptable HTTP statuses under chaos — anything else
// (notably 500 / a transport error on a non-cancelled request) is a failure.
func okStatus(code int) bool {
	switch code {
	case http.StatusOK, http.StatusTooManyRequests, http.StatusConflict,
		http.StatusNotFound, http.StatusBadRequest, http.StatusForbidden:
		return true
	}
	return false
}

// TestServe_soakChaos hammers two models with mixed stream/non-stream requests
// (some constrained), while an admin goroutine load/unloads a scratch model and
// some clients disconnect mid-SSE — all under -race — then asserts no goroutine
// leak and only well-formed responses.
func TestServe_soakChaos(t *testing.T) {
	models := chaosModels(t) // skips without the env model
	srv, err := newServer(config{
		models: models, backend: "cpu", quant: "int8int8",
		kvSessions: 2, maxQueue: 4, allowAdmin: true,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(chaosMux(srv))
	defer ts.Close()
	scratchPath := models[0].path

	// Warm up one request per model so lazy per-request goroutines (and HTTP/1
	// keep-alive readers) are already spun up before we sample the baseline.
	for _, m := range []string{"m1", "m2"} {
		r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"`+m+`","max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatalf("warmup %s: %v", m, err)
		}
		io.Copy(io.Discard, r.Body)
		r.Body.Close()
	}
	time.Sleep(200 * time.Millisecond)
	runtime.GC()
	baseline := runtime.NumGoroutine()

	deadline := time.Now().Add(20 * time.Second)
	var bad int64 // count of unacceptable responses
	var streamCancels, served int64
	var wg sync.WaitGroup

	body := func(model string, stream, constrained bool) string {
		rf := ""
		if constrained {
			rf = `,"response_format":{"type":"json_object"}`
		}
		return fmt.Sprintf(`{"model":%q,"stream":%t,"max_tokens":8,"temperature":0%s,"messages":[{"role":"user","content":"Reply briefly."}]}`,
			model, stream, rf)
	}

	// Request workers.
	const workers = 8
	for i := range workers {
		wg.Go(func() {
			client := &http.Client{}
			defer client.CloseIdleConnections()
			n := 0
			for time.Now().Before(deadline) {
				n++
				model := []string{"m1", "m2"}[n&1]
				stream := n%3 == 0
				constrained := n%5 == 0
				if stream && n%2 == 0 {
					// Disconnect mid-stream: cancel after the first chunk arrives.
					ctx, cancel := context.WithCancel(context.Background())
					req, _ := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/chat/completions",
						strings.NewReader(body(model, true, false)))
					req.Header.Set("Content-Type", "application/json")
					resp, err := client.Do(req)
					if err != nil {
						cancel()
						continue
					}
					br := bufio.NewReader(resp.Body)
					br.ReadString('\n') // first SSE line, then bail
					cancel()
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
					atomic.AddInt64(&streamCancels, 1)
					continue
				}
				resp, err := client.Post(ts.URL+"/v1/chat/completions", "application/json",
					strings.NewReader(body(model, stream, constrained)))
				if err != nil {
					atomic.AddInt64(&bad, 1)
					t.Errorf("worker %d POST: %v", i, err)
					continue
				}
				code := resp.StatusCode
				retry := resp.Header.Get("Retry-After")
				buf, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				if !okStatus(code) {
					atomic.AddInt64(&bad, 1)
					t.Errorf("worker %d: unacceptable status %d: %s", i, code, truncate(buf))
					continue
				}
				if code == http.StatusTooManyRequests && retry == "" {
					atomic.AddInt64(&bad, 1)
					t.Errorf("429 without Retry-After")
				}
				if code == http.StatusOK && !stream && !validChatJSON(buf) {
					atomic.AddInt64(&bad, 1)
					t.Errorf("200 body not a valid chat completion: %s", truncate(buf))
				}
				atomic.AddInt64(&served, 1)
			}
		})
	}

	// Admin churn: load a scratch model, then unload it, in a loop. Expect a clean
	// 200, or 409 (busy / already loaded), never a 500.
	wg.Go(func() {
		client := &http.Client{}
		defer client.CloseIdleConnections()
		for time.Now().Before(deadline) {
			lr, err := client.Post(ts.URL+"/admin/models/load", "application/json",
				strings.NewReader(fmt.Sprintf(`{"name":"scratch","path":%q}`, scratchPath)))
			if err == nil {
				if !okStatus(lr.StatusCode) {
					atomic.AddInt64(&bad, 1)
					t.Errorf("admin load: status %d", lr.StatusCode)
				}
				io.Copy(io.Discard, lr.Body)
				lr.Body.Close()
			}
			ur, err := client.Post(ts.URL+"/admin/models/unload", "application/json",
				strings.NewReader(`{"name":"scratch"}`))
			if err == nil {
				if !okStatus(ur.StatusCode) {
					atomic.AddInt64(&bad, 1)
					t.Errorf("admin unload: status %d", ur.StatusCode)
				}
				io.Copy(io.Discard, ur.Body)
				ur.Body.Close()
			}
			time.Sleep(50 * time.Millisecond)
		}
	})

	wg.Wait()
	t.Logf("served=%d stream-cancels=%d bad=%d", served, streamCancels, bad)
	if bad != 0 {
		t.Fatalf("%d unacceptable responses under chaos", bad)
	}

	// No goroutine leak: per-request goroutines (incl. cancelled streams) must
	// drain back toward the baseline once traffic stops.
	if leaked := waitGoroutines(baseline + 8); leaked > baseline+8 {
		t.Errorf("goroutine leak: baseline %d, settled at %d", baseline, leaked)
	}
}

// waitGoroutines polls until NumGoroutine drops to <= target (up to ~5s),
// returning the final count. GC between polls so finished goroutines are reaped.
func waitGoroutines(target int) int {
	last := runtime.NumGoroutine()
	for range 50 {
		runtime.GC()
		last = runtime.NumGoroutine()
		if last <= target {
			return last
		}
		time.Sleep(100 * time.Millisecond)
	}
	return last
}

func validChatJSON(b []byte) bool {
	var m struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Role string `json:"role"`
			} `json:"message"`
		} `json:"choices"`
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(b, &m) != nil {
		return false
	}
	return m.Object == "chat.completion" && len(m.Choices) > 0
}

func truncate(b []byte) string {
	if len(b) > 160 {
		return string(b[:160]) + "…"
	}
	return string(b)
}

// TestServe_warmKVRestore is the --session-dir gate: a continuation served after
// a process restart (warm KV restored from disk) must be byte-identical to the
// same continuation served by a never-restarted server (deterministic greedy
// decode), i.e. KV restore is lossless. Uses one model (GOINFER_SERVE_MODEL).
func TestServe_warmKVRestore(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the warm-KV restore test")
	}
	dir := t.TempDir()

	// chat sends a (possibly multi-turn) conversation greedily and returns the
	// assistant text. server must already be running behind ts.
	chat := func(ts *httptest.Server, msgs string) (string, error) {
		r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"m","max_tokens":16,"temperature":0,"messages":`+msgs+`}`))
		if err != nil {
			return "", err
		}
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		if r.StatusCode != http.StatusOK {
			return "", fmt.Errorf("status %d: %s", r.StatusCode, truncate(b))
		}
		var resp struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(b, &resp); err != nil || len(resp.Choices) == 0 {
			return "", fmt.Errorf("bad body: %s", truncate(b))
		}
		return resp.Choices[0].Message.Content, nil
	}

	const turn1 = `[{"role":"user","content":"Name three primary colors."}]`

	// Round 1 on a session-backed server: generates the assistant reply and warms
	// a KV session for the conversation, then snapshot to disk.
	srvA, err := newServer(config{models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 2, sessionDir: dir})
	if err != nil {
		t.Fatalf("newServer A: %v", err)
	}
	tsA := httptest.NewServer(chaosMux(srvA))
	a1, err := chat(tsA, turn1)
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}
	lmA := srvA.models["m"]
	lmA.mu.Lock()
	if err := lmA.sessions.save(sessionSubdir(dir, lmA.fp)); err != nil {
		t.Fatalf("session save: %v", err)
	}
	lmA.mu.Unlock()
	tsA.Close()

	// The follow-up conversation = turn 1 + the assistant reply + a new user turn.
	// Its token prefix is the warmed session, so the restarted server reuses it.
	follow := fmt.Sprintf(`[{"role":"user","content":"Name three primary colors."},{"role":"assistant","content":%q},{"role":"user","content":"Now name three more."}]`, a1)

	// Restarted server: same session-dir, restore from disk.
	srvB, err := newServer(config{models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 2, sessionDir: dir})
	if err != nil {
		t.Fatalf("newServer B: %v", err)
	}
	lmB := srvB.models["m"]
	if restored := lmB.sessions.load(sessionSubdir(dir, lmB.fp)); restored == 0 {
		t.Fatalf("warm restore loaded 0 sessions — nothing to test")
	}
	tsB := httptest.NewServer(chaosMux(srvB))
	defer tsB.Close()
	warm, err := chat(tsB, follow)
	if err != nil {
		t.Fatalf("warm follow-up: %v", err)
	}

	// Cold reference: a fresh server with NO session reuse, same follow-up.
	srvC, err := newServer(config{models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 0})
	if err != nil {
		t.Fatalf("newServer C: %v", err)
	}
	tsC := httptest.NewServer(chaosMux(srvC))
	defer tsC.Close()
	cold, err := chat(tsC, follow)
	if err != nil {
		t.Fatalf("cold follow-up: %v", err)
	}

	if warm != cold {
		t.Fatalf("warm-KV restore changed the continuation:\n warm: %q\n cold: %q", warm, cold)
	}
	t.Logf("warm-restored continuation byte-identical to cold prefill: %q", cold)
}
