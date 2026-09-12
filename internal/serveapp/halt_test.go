package serveapp

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// K2 gate (docs/task-halt-2026-09.md): 32 concurrent generations at -max-inflight 32
// (saturated), halt, assert every one of them stopped because of the halt (not a natural
// completion), and measure time-to-quiescence. Then resume and assert a request succeeds.
//
// "Every stream ended cancelled" (the doc's own words) needs one honest caveat, found while
// building this test, not assumed going in: goinfer serializes ALL generations for one model
// behind a single mutex (loadedModel.mu, "the single decode worker" — openai.go's own doc
// comment). With -max-inflight 32 admitting all 32 requests, only ONE of them is ever actually
// registered in K1's cancel registry and streaming at a time; the other 31 are blocked
// acquiring that mutex, which is not context-aware and would otherwise run to natural
// completion one at a time regardless of a halt (a real bug this task's own K2 work found and
// fixed in tryEnter/enter — see halt.go and openai.go's doc comments on both). The fixed
// behavior: a request already streaming when halt lands gets finish_reason "cancelled"; a
// request still queued behind the mutex gets refused with a 503 {"error":"halted"} the moment
// it would otherwise have started, WITHOUT ever running a real generation. Both are "the halt
// switch worked" outcomes; neither is "ran to completion after being told to stop". This test
// asserts exactly that split, not a literal 32/32 "cancelled".
//
// Gated on GOINFER_SERVE_MODEL like this package's other real-model tests.
func TestServe_haltUnderLoad(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the halt-under-load test")
	}
	const n = 32
	srv, err := newServer(config{
		models: modelFlag{{name: "t", path: path}}, backend: "cpu", quant: "int8int8",
		allowAdmin: true, maxInflight: n, maxQueue: n, // queue capacity 1+maxQueue must cover all n
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.haltGate(limitInflight(make(chan struct{}, n), srv.handleChat)))
	mux.HandleFunc("POST /admin/halt", srv.handleAdminHalt)
	mux.HandleFunc("POST /admin/resume", srv.handleAdminResume)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	longBody := `{"model":"t","max_tokens":200,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Count from 1 to 1000, one number per line."}]}`

	type result struct {
		finishReason string // "" if the request never got a chat chunk at all (e.g. 503 halted)
		status       int
		haltedBody   string // set when the response was a 503 {"error":"halted",...}
	}
	results := make([]result, n)
	var wg sync.WaitGroup
	var launched sync.WaitGroup
	launched.Add(n)
	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			req, err := http.NewRequest("POST", ts.URL+"/v1/chat/completions", strings.NewReader(longBody))
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				launched.Done()
				return
			}
			req.Header.Set("Content-Type", "application/json")
			launched.Done() // the goroutine is about to call Do — NOT the same as the request being
			// admitted: Do() blocks until the server sends response headers, which for a request
			// still queued behind the model's single decode mutex does not happen until its turn
			// comes up. Waiting on Do() to return here (as an earlier version of this test did)
			// self-deadlocks — nothing frees that mutex until the halt this goroutine is itself
			// blocking the issuing of.
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			defer resp.Body.Close()
			results[i].status = resp.StatusCode
			if resp.StatusCode == http.StatusServiceUnavailable {
				b := make([]byte, 512)
				nRead, _ := resp.Body.Read(b)
				results[i].haltedBody = string(b[:nRead])
				return
			}
			sc := bufio.NewScanner(resp.Body)
			for sc.Scan() {
				line := sc.Text()
				if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
					continue
				}
				var chunk map[string]any
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
					continue
				}
				if choices, ok := chunk["choices"].([]any); ok && len(choices) > 0 {
					if c0, ok := choices[0].(map[string]any); ok {
						if fr, ok := c0["finish_reason"].(string); ok && fr != "" {
							results[i].finishReason = fr
						}
					}
				}
			}
		}(i)
	}

	// Wait for every goroutine to have been launched (not for its request to complete — most
	// won't, until the halt below), then give the server a moment to actually accept all 32 TCP
	// connections and have each one claim its queue slot, so this genuinely exercises
	// "saturated" rather than racing the halt against requests still arriving.
	launched.Wait()
	time.Sleep(300 * time.Millisecond)

	haltStart := time.Now()
	haltResp, err := http.Post(ts.URL+"/admin/halt", "application/json", strings.NewReader(`{"reason":"load test"}`))
	if err != nil {
		t.Fatalf("POST halt: %v", err)
	}
	var haltResult struct {
		QuiescedInMs int64 `json:"quiesced_in_ms"`
	}
	json.NewDecoder(haltResp.Body).Decode(&haltResult)
	haltResp.Body.Close()
	adminRoundTrip := time.Since(haltStart)
	t.Logf("K2 time-to-quiescence: %dms (admin round trip %s), model=%s box=%s/%s",
		haltResult.QuiescedInMs, adminRoundTrip, path, runtime.GOOS, runtime.GOARCH)

	// Every one of the 32 goroutines must finish promptly now — the halt call itself already
	// blocked until quiescence, so this should return almost immediately; a generous bound
	// catches a genuine hang rather than a slow-but-fine finish.
	doneCh := make(chan struct{})
	go func() { wg.Wait(); close(doneCh) }()
	select {
	case <-doneCh:
	case <-time.After(15 * time.Second):
		t.Fatal("not all 32 requests finished within 15s of the halt call returning")
	}

	var cancelled, haltedRefused, other int
	for i, r := range results {
		switch {
		case r.finishReason == "cancelled":
			cancelled++
		case r.status == http.StatusServiceUnavailable && strings.Contains(r.haltedBody, `"halted"`):
			haltedRefused++
		default:
			other++
			t.Errorf("request %d: status=%d finish_reason=%q body=%q — neither cancelled nor halted-refused",
				i, r.status, r.finishReason, r.haltedBody)
		}
	}
	t.Logf("32 requests: %d cancelled (were actively streaming), %d refused with 503 halted (were still queued), %d unexplained",
		cancelled, haltedRefused, other)
	if other > 0 {
		t.Errorf("%d of %d requests ended some way other than cancelled/halted-refused", other, n)
	}
	if cancelled == 0 {
		t.Error("0 requests reached finish_reason \"cancelled\" — expected at least the one actively streaming when halt landed")
	}

	// Resume, then a request must succeed normally.
	resumeResp, err := http.Post(ts.URL+"/admin/resume", "application/json", nil)
	if err != nil {
		t.Fatalf("POST resume: %v", err)
	}
	resumeResp.Body.Close()
	shortResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"t","max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("post-resume request: %v", err)
	}
	defer shortResp.Body.Close()
	if shortResp.StatusCode != http.StatusOK {
		t.Errorf("post-resume request status = %d, want 200", shortResp.StatusCode)
	}
}
