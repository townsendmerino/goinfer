package serveapp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// K1 gate (docs/task-halt-2026-09.md): start a long generation, cancel it by id partway
// through, and assert (1) the stream stops promptly after the cancel request completes —
// the practical form of "no token timestamped after the cancel" this test can check without
// per-token wall-clock instrumentation that does not exist anywhere in the tree today, (2) the
// final SSE event names the cancel reason, the finish_reason chunk says "cancelled", and the
// registry no longer lists the id, and (3) a FOLLOWING request on the same model produces
// byte-identical output to one taken BEFORE the cancel — proof the resident/session state the
// cancelled generation touched was left exactly as clean as the existing error path leaves it
// (drive/driveVL add no new cleanup; see generations.go's doc comment for why).
//
// Gated on GOINFER_SERVE_MODEL (a real .gguf) like this package's other real-model tests
// (chaos_test.go, admin_test.go) — CI skips it for want of the asset; run it locally with the
// env var set.
func TestServe_cancelByID(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the cancel-by-id test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "t", path: path}}, backend: "cpu", quant: "int8int8", allowAdmin: true,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("GET /admin/generations", srv.handleAdminGenerationsList)
	mux.HandleFunc("POST /admin/generations/{id}/cancel", srv.handleAdminGenerationCancel)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	const shortBody = `{"model":"t","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}`

	// Baseline BEFORE the cancelled generation: short, deterministic (greedy, temp 0), so its
	// output can be reproduced exactly afterward if — and only if — the cancel left resident/
	// session state sane.
	baselineBefore := runChat(t, ts, shortBody)

	// The long generation: 4k max_tokens, streamed, cancelled partway through.
	longBody := fmt.Sprintf(`{"model":"t","max_tokens":%d,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Count from 1 to 1000, one number per line."}]}`, 4096)
	req, err := http.NewRequest("POST", ts.URL+"/v1/chat/completions", strings.NewReader(longBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("start long generation: %v", err)
	}
	defer resp.Body.Close()
	t.Logf("long generation response status: %d", resp.StatusCode)
	sc := bufio.NewScanner(resp.Body)

	// First SSE data line carries the id (chatChunk's own id field). Read a few chunks to be
	// sure the generation is genuinely under way before cancelling — cancelling before the
	// server has even registered it would race the registry, not test the cancel path.
	var id string
	chunksSeen := 0
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			continue
		}
		if idv, ok := chunk["id"].(string); ok && idv != "" {
			id = idv
		}
		chunksSeen++
		if chunksSeen >= 3 { // a few tokens in — the generation is genuinely running
			break
		}
	}
	if id == "" {
		t.Fatalf("never observed a generation id on the stream (saw %d chunks)", chunksSeen)
	}

	// Confirm the registry actually has it before cancelling (proves K1's registration path,
	// not just the cancel path).
	listResp, err := http.Get(ts.URL + "/admin/generations")
	if err != nil {
		t.Fatalf("GET /admin/generations: %v", err)
	}
	var list struct {
		Generations []generationSnapshot `json:"generations"`
	}
	json.NewDecoder(listResp.Body).Decode(&list)
	listResp.Body.Close()
	found := false
	for _, g := range list.Generations {
		if g.ID == id {
			found = true
		}
	}
	if !found {
		t.Errorf("generation %q not in GET /admin/generations before cancel: %+v", id, list.Generations)
	}

	// Cancel it.
	cancelResp, err := http.Post(ts.URL+"/admin/generations/"+id+"/cancel", "application/json",
		strings.NewReader(`{"reason":"test cancel"}`))
	if err != nil {
		t.Fatalf("POST cancel: %v", err)
	}
	var cancelResult struct {
		Found bool `json:"found"`
	}
	json.NewDecoder(cancelResp.Body).Decode(&cancelResult)
	cancelResp.Body.Close()
	if !cancelResult.Found {
		t.Fatal("cancel reported found:false — the generation had already ended, this run raced itself")
	}
	cancelledAt := time.Now()

	// Drain the rest of the stream, watching for the finish chunk and the goinfer_cancelled
	// event, and timing how long it takes to arrive.
	sawCancelledFinish := false
	sawCancelledEvent := ""
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			continue
		}
		if gc, ok := chunk["goinfer_cancelled"].(map[string]any); ok {
			if r, ok := gc["reason"].(string); ok {
				sawCancelledEvent = r
			}
		}
		if choices, ok := chunk["choices"].([]any); ok && len(choices) > 0 {
			if c0, ok := choices[0].(map[string]any); ok {
				if fr, ok := c0["finish_reason"].(string); ok && fr == "cancelled" {
					sawCancelledFinish = true
				}
			}
		}
	}
	quiescedAfter := time.Since(cancelledAt)
	// The practical stand-in for "no token timestamped after the cancel": the stream must not
	// still be arriving long after the cancel request returned. A generous bound (this is a
	// CPU test model, not a latency gate) — the point is "promptly", not a specific number.
	if quiescedAfter > 5*time.Second {
		t.Errorf("stream took %v to quiesce after cancel — expected well under 5s on a CPU test model", quiescedAfter)
	}
	if !sawCancelledFinish {
		t.Error("no chunk carried finish_reason \"cancelled\"")
	}
	if sawCancelledEvent != "test cancel" {
		t.Errorf("goinfer_cancelled event reason = %q, want %q", sawCancelledEvent, "test cancel")
	}

	// The registry entry must be gone now (drive's own defer gens.remove).
	listResp2, err := http.Get(ts.URL + "/admin/generations")
	if err != nil {
		t.Fatalf("GET /admin/generations (after): %v", err)
	}
	var list2 struct {
		Generations []generationSnapshot `json:"generations"`
	}
	json.NewDecoder(listResp2.Body).Decode(&list2)
	listResp2.Body.Close()
	for _, g := range list2.Generations {
		if g.ID == id {
			t.Errorf("generation %q still in the registry after it ended", id)
		}
	}

	// The parity check: repeat the EXACT baseline request. Byte-identical output to the
	// pre-cancel baseline is the proof that the cancelled generation left resident/session
	// state exactly as clean as a normal completion would have — the same property a fresh
	// process would have, without needing to actually spawn one.
	baselineAfter := runChat(t, ts, shortBody)
	if baselineAfter != baselineBefore {
		t.Errorf("post-cancel generation diverged from the pre-cancel baseline:\n before: %q\n after:  %q",
			baselineBefore, baselineAfter)
	}
}

// runChat POSTs a non-streaming chat completion and returns the assistant's content string.
func runChat(t *testing.T, ts *httptest.Server, body string) string {
	t.Helper()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("chat POST: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode chat response: %v", err)
	}
	if len(out.Choices) == 0 {
		t.Fatalf("chat response had no choices")
	}
	return out.Choices[0].Message.Content
}
