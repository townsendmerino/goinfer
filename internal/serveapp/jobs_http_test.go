package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func jobsTestMux(srv *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/jobs", srv.handleCreateJob)
	mux.HandleFunc("GET /v1/jobs/{id}", srv.handleGetJob)
	mux.HandleFunc("GET /v1/jobs/{id}/events", srv.handleJobEvents)
	mux.HandleFunc("DELETE /v1/jobs/{id}", srv.handleCancelJob)
	return mux
}

// readSSEContent reads an SSE stream of chatChunk frames from body until [DONE] or ctx ends,
// concatenating every delta.content it sees.
func readSSEContent(ctx context.Context, body io.Reader) (string, error) {
	sc := bufio.NewScanner(body)
	var sb strings.Builder
	for sc.Scan() {
		select {
		case <-ctx.Done():
			return sb.String(), ctx.Err()
		default:
		}
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return sb.String(), nil
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // a comment/heartbeat frame, or something this decoder doesn't need
		}
		for _, c := range chunk.Choices {
			sb.WriteString(c.Delta.Content)
		}
	}
	return sb.String(), sc.Err()
}

// TestJobs_submitPollReattachMatchesSyncRun is the task doc's own J3 gate: start a job, kill the
// client mid-stream, reconnect, and assemble a transcript byte-identical to an uninterrupted
// /v1/chat/completions run of the same seed.
func TestJobs_submitPollReattachMatchesSyncRun(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the jobs re-attach test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8", kvSessions: 0,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(jobsTestMux(srv))
	defer ts.Close()

	const body = `{"model":"m","max_tokens":48,"temperature":0,"seed":42,"messages":[{"role":"user","content":"Write a haiku about Go."}]}`

	// Reference: an uninterrupted synchronous run of the same seed.
	refResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("reference POST: %v", err)
	}
	defer refResp.Body.Close()
	var ref struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(refResp.Body).Decode(&ref); err != nil || len(ref.Choices) == 0 {
		t.Fatalf("decode reference response: %v", err)
	}
	refContent := ref.Choices[0].Message.Content

	// Submit the same request as a job.
	jobResp, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/jobs: %v", err)
	}
	defer jobResp.Body.Close()
	if jobResp.StatusCode != http.StatusAccepted {
		b, _ := io.ReadAll(jobResp.Body)
		t.Fatalf("POST /v1/jobs status = %d, body = %s", jobResp.StatusCode, b)
	}
	var created struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := json.NewDecoder(jobResp.Body).Decode(&created); err != nil {
		t.Fatalf("decode job creation response: %v", err)
	}
	if created.Status != string(jobPending) {
		t.Fatalf("initial status = %q, want %q", created.Status, jobPending)
	}

	// Connect to /events, read a FEW frames, then disconnect early (simulating a client that
	// went away mid-stream) — the job keeps running in the background regardless.
	firstCtx, firstCancel := context.WithTimeout(context.Background(), 5*time.Second)
	req, _ := http.NewRequestWithContext(firstCtx, "GET", ts.URL+"/v1/jobs/"+created.ID+"/events", nil)
	firstResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /events (first attempt): %v", err)
	}
	sc := bufio.NewScanner(firstResp.Body)
	gotAFrame := false
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "data: ") {
			gotAFrame = true
			break
		}
	}
	if !gotAFrame {
		t.Fatal("did not observe even one event before disconnecting")
	}
	firstResp.Body.Close()
	firstCancel() // the disconnect

	// Wait for the job to actually finish in the background (it must not have been cancelled by
	// the client going away — only DELETE /v1/jobs/{id} cancels a job).
	deadline := time.Now().Add(10 * time.Second)
	for {
		st, err := http.Get(ts.URL + "/v1/jobs/" + created.ID)
		if err != nil {
			t.Fatalf("GET /v1/jobs/%s: %v", created.ID, err)
		}
		var polled struct {
			Status string `json:"status"`
			Result *struct {
				Content      string `json:"content"`
				FinishReason string `json:"finish_reason"`
			} `json:"result"`
		}
		derr := json.NewDecoder(st.Body).Decode(&polled)
		st.Body.Close()
		if derr != nil {
			t.Fatalf("decode poll response: %v", derr)
		}
		if polled.Status == string(jobDone) {
			if polled.Result == nil {
				t.Fatal("job done but result is nil")
			}
			if polled.Result.Content != refContent {
				t.Fatalf("poll result mismatch:\n  reference: %q\n  job:       %q", refContent, polled.Result.Content)
			}
			break
		}
		if polled.Status == string(jobFailed) || polled.Status == string(jobCancelled) {
			t.Fatalf("job ended as %q, want done", polled.Status)
		}
		if time.Now().After(deadline) {
			t.Fatalf("job did not finish within the deadline (last status %q)", polled.Status)
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Re-attach: a SECOND, independent connection to /events after the job is already done must
	// replay the WHOLE transcript from the start — byte-identical to the reference run.
	reCtx, reCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer reCancel()
	req2, _ := http.NewRequestWithContext(reCtx, "GET", ts.URL+"/v1/jobs/"+created.ID+"/events", nil)
	reResp, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("GET /events (re-attach): %v", err)
	}
	defer reResp.Body.Close()
	replayed, err := readSSEContent(reCtx, reResp.Body)
	if err != nil {
		t.Fatalf("reading replayed events: %v", err)
	}
	if replayed != refContent {
		t.Fatalf("re-attach replay mismatch:\n  reference: %q\n  replayed:  %q", refContent, replayed)
	}
}

// TestJobs_deleteCancelsAQueuedJob: DELETE must stop a job that is still waiting for admission,
// not only one that is already running — the specific gap a plain sync.Mutex-based wait couldn't
// close, mirrored here at the HTTP-surface level (J1 is what actually makes this possible).
func TestJobs_deleteCancelsAQueuedJob(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the jobs cancel test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, backend: "cpu", quant: "int8int8",
		kvSessions: 0, maxQueue: 4,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(jobsTestMux(srv))
	defer ts.Close()

	longBody := `{"model":"m","max_tokens":128,"temperature":0,"messages":[{"role":"user","content":"Count from 1 to 100."}]}`
	shortBody := `{"model":"m","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}`

	// Occupy the one decode worker with a long job, then queue a second, short job behind it.
	holder, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(longBody))
	if err != nil {
		t.Fatalf("POST holder job: %v", err)
	}
	holder.Body.Close()

	queued, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(shortBody))
	if err != nil {
		t.Fatalf("POST queued job: %v", err)
	}
	var q struct{ ID string }
	json.NewDecoder(queued.Body).Decode(&q)
	queued.Body.Close()

	// Give the holder time to actually claim the turn (not a guarantee, but generous) before
	// cancelling the one still waiting.
	time.Sleep(50 * time.Millisecond)

	del, err := http.NewRequest("DELETE", ts.URL+"/v1/jobs/"+q.ID, nil)
	if err != nil {
		t.Fatal(err)
	}
	delResp, err := http.DefaultClient.Do(del)
	if err != nil {
		t.Fatalf("DELETE: %v", err)
	}
	var delOut struct {
		Cancelled bool `json:"cancelled"`
	}
	json.NewDecoder(delResp.Body).Decode(&delOut)
	delResp.Body.Close()
	if !delOut.Cancelled {
		t.Fatal("DELETE reported cancelled=false for a job that should still exist (queued or running)")
	}

	deadline := time.Now().Add(10 * time.Second)
	for {
		st, err := http.Get(ts.URL + "/v1/jobs/" + q.ID)
		if err != nil {
			t.Fatalf("GET: %v", err)
		}
		var polled struct{ Status string }
		json.NewDecoder(st.Body).Decode(&polled)
		st.Body.Close()
		if polled.Status == string(jobCancelled) {
			return
		}
		if polled.Status == string(jobDone) {
			t.Fatal("queued job ran to completion instead of being cancelled — DELETE did not reach a still-queued job")
		}
		if time.Now().After(deadline) {
			t.Fatalf("job never reached a terminal state (last status %q)", polled.Status)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
