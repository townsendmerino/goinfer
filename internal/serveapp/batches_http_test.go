package serveapp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func batchesTestMux(srv *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
	mux.HandleFunc("POST /v1/messages", srv.handleMessages)
	mux.HandleFunc("POST /v1/jobs", srv.handleCreateJob)
	mux.HandleFunc("POST /v1/files", srv.handleCreateFile)
	mux.HandleFunc("GET /v1/files/{id}", srv.handleGetFile)
	mux.HandleFunc("GET /v1/files/{id}/content", srv.handleGetFileContent)
	mux.HandleFunc("POST /v1/batches", srv.handleCreateBatch)
	mux.HandleFunc("GET /v1/batches/{id}", srv.handleGetBatch)
	mux.HandleFunc("POST /v1/batches/{id}/cancel", srv.handleCancelBatch)
	mux.HandleFunc("POST /v1/messages/batches", srv.handleCreateMessageBatch)
	mux.HandleFunc("GET /v1/messages/batches/{id}", srv.handleGetMessageBatch)
	mux.HandleFunc("GET /v1/messages/batches/{id}/results", srv.handleGetMessageBatchResults)
	mux.HandleFunc("POST /v1/messages/batches/{id}/cancel", srv.handleCancelMessageBatch)
	return mux
}

func uploadJSONLFile(t *testing.T, ts *httptest.Server, filename string, lines []string) string {
	t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := part.Write([]byte(strings.Join(lines, "\n") + "\n")); err != nil {
		t.Fatalf("write multipart body: %v", err)
	}
	if err := w.WriteField("purpose", "batch"); err != nil {
		t.Fatalf("WriteField: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	resp, err := http.Post(ts.URL+"/v1/files", w.FormDataContentType(), &body)
	if err != nil {
		t.Fatalf("POST /v1/files: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("POST /v1/files status = %d, body = %s", resp.StatusCode, b)
	}
	var out struct{ ID string }
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode file response: %v", err)
	}
	return out.ID
}

func postJSON(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

// TestBatches_openAIRoundTrip is the task doc's own J4 gate for the OpenAI dialect: an input JSONL
// with a good line and a deliberately-bad line (an image, out of scope this pass) round-trips
// through files → batches → poll → output/error content, with custom_id ordering preserved and
// the good line's content matching an uninterrupted /v1/chat/completions call of the same seed —
// and the bad line lands in the error file without ever failing the batch itself.
func TestBatches_openAIRoundTrip(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the batches round-trip test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}, kvSessions: 0,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	const goodBody = `{"model":"m","max_tokens":32,"temperature":0,"seed":42,"messages":[{"role":"user","content":"Write a haiku about Go."}]}`

	// Reference: an uninterrupted synchronous run of the same seed.
	refResp := postJSON(t, ts.URL+"/v1/chat/completions", json.RawMessage(goodBody))
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

	badBody := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`

	inputLines := []string{
		mustJSONLine(t, map[string]any{"custom_id": "req-good", "method": "POST", "url": "/v1/chat/completions", "body": json.RawMessage(goodBody)}),
		mustJSONLine(t, map[string]any{"custom_id": "req-bad", "method": "POST", "url": "/v1/chat/completions", "body": json.RawMessage(badBody)}),
	}
	fileID := uploadJSONLFile(t, ts, "in.jsonl", inputLines)

	batchResp := postJSON(t, ts.URL+"/v1/batches", map[string]any{
		"input_file_id": fileID, "endpoint": "/v1/chat/completions", "completion_window": "24h",
	})
	defer batchResp.Body.Close()
	var created struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := json.NewDecoder(batchResp.Body).Decode(&created); err != nil {
		t.Fatalf("decode batch creation response: %v", err)
	}
	if created.Status != "in_progress" {
		t.Fatalf("initial status = %q, want in_progress", created.Status)
	}

	var polled struct {
		Status       string `json:"status"`
		OutputFileID string `json:"output_file_id"`
		ErrorFileID  string `json:"error_file_id"`
	}
	deadline := time.Now().Add(15 * time.Second)
	for {
		st, err := http.Get(ts.URL + "/v1/batches/" + created.ID)
		if err != nil {
			t.Fatalf("GET /v1/batches/%s: %v", created.ID, err)
		}
		derr := json.NewDecoder(st.Body).Decode(&polled)
		st.Body.Close()
		if derr != nil {
			t.Fatalf("decode poll response: %v", derr)
		}
		if polled.Status == "completed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("batch did not complete within the deadline (last status %q)", polled.Status)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if polled.OutputFileID == "" {
		t.Fatal("batch completed but output_file_id is empty")
	}
	if polled.ErrorFileID == "" {
		t.Fatal("batch completed but error_file_id is empty — the bad line should have produced one")
	}

	outLines := fetchJSONLLines(t, ts, polled.OutputFileID)
	if len(outLines) != 1 {
		t.Fatalf("output file has %d lines, want 1", len(outLines))
	}
	if outLines[0]["custom_id"] != "req-good" {
		t.Fatalf("output line custom_id = %v, want req-good", outLines[0]["custom_id"])
	}
	respObj, _ := outLines[0]["response"].(map[string]any)
	bodyObj, _ := respObj["body"].(map[string]any)
	choices, _ := bodyObj["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("output line has no choices: %+v", outLines[0])
	}
	msg, _ := choices[0].(map[string]any)["message"].(map[string]any)
	if got := msg["content"]; got != refContent {
		t.Fatalf("batch output content mismatch:\n  reference: %q\n  batch:     %q", refContent, got)
	}

	errLines := fetchJSONLLines(t, ts, polled.ErrorFileID)
	if len(errLines) != 1 {
		t.Fatalf("error file has %d lines, want 1", len(errLines))
	}
	if errLines[0]["custom_id"] != "req-bad" {
		t.Fatalf("error line custom_id = %v, want req-bad", errLines[0]["custom_id"])
	}
	if errLines[0]["response"] != nil {
		t.Fatalf("error line carries a non-nil response: %+v", errLines[0])
	}
}

func mustJSONLine(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

func fetchJSONLLines(t *testing.T, ts *httptest.Server, fileID string) []map[string]any {
	t.Helper()
	resp, err := http.Get(ts.URL + "/v1/files/" + fileID + "/content")
	if err != nil {
		t.Fatalf("GET file content: %v", err)
	}
	defer resp.Body.Close()
	var out []map[string]any
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Fatalf("parse JSONL line %q: %v", raw, err)
		}
		out = append(out, m)
	}
	return out
}

// TestBatches_anthropicRoundTrip mirrors the OpenAI test for the inline-request dialect: no file
// upload (Anthropic's Message Batches API has none), a good line and a bad line (missing
// max_tokens), polled to "ended", results fetched from the dedicated results endpoint.
func TestBatches_anthropicRoundTrip(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the batches round-trip test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}, kvSessions: 0,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	temp := 0.0
	goodParams := map[string]any{
		"model": "m", "max_tokens": 32, "temperature": temp,
		"messages": []map[string]any{{"role": "user", "content": "Write a haiku about Go."}},
	}

	// Reference: an uninterrupted synchronous /v1/messages call, greedy (temperature 0), so it's
	// reproducible without a seed field (anthropicReq has none).
	refResp := postJSON(t, ts.URL+"/v1/messages", goodParams)
	defer refResp.Body.Close()
	var ref struct {
		Content []struct{ Text string } `json:"content"`
	}
	if err := json.NewDecoder(refResp.Body).Decode(&ref); err != nil || len(ref.Content) == 0 {
		t.Fatalf("decode reference response: %v", err)
	}
	refText := ref.Content[0].Text

	badParams := map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
		// max_tokens deliberately omitted
	}

	createResp := postJSON(t, ts.URL+"/v1/messages/batches", map[string]any{
		"requests": []map[string]any{
			{"custom_id": "req-good", "params": goodParams},
			{"custom_id": "req-bad", "params": badParams},
		},
	})
	defer createResp.Body.Close()
	var created struct {
		ID               string `json:"id"`
		ProcessingStatus string `json:"processing_status"`
	}
	if err := json.NewDecoder(createResp.Body).Decode(&created); err != nil {
		t.Fatalf("decode batch creation response: %v", err)
	}
	if created.ProcessingStatus != "in_progress" {
		t.Fatalf("initial processing_status = %q, want in_progress", created.ProcessingStatus)
	}

	deadline := time.Now().Add(15 * time.Second)
	for {
		st, err := http.Get(ts.URL + "/v1/messages/batches/" + created.ID)
		if err != nil {
			t.Fatalf("GET batch: %v", err)
		}
		var polled struct {
			ProcessingStatus string `json:"processing_status"`
		}
		derr := json.NewDecoder(st.Body).Decode(&polled)
		st.Body.Close()
		if derr != nil {
			t.Fatalf("decode poll response: %v", derr)
		}
		if polled.ProcessingStatus == "ended" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("batch did not end within the deadline (last status %q)", polled.ProcessingStatus)
		}
		time.Sleep(50 * time.Millisecond)
	}

	resultsResp, err := http.Get(ts.URL + "/v1/messages/batches/" + created.ID + "/results")
	if err != nil {
		t.Fatalf("GET results: %v", err)
	}
	defer resultsResp.Body.Close()
	sc := bufio.NewScanner(resultsResp.Body)
	byID := map[string]map[string]any{}
	for sc.Scan() {
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Fatalf("parse results line %q: %v", raw, err)
		}
		byID[m["custom_id"].(string)] = m
	}

	goodResult, ok := byID["req-good"]["result"].(map[string]any)
	if !ok {
		t.Fatalf("req-good has no result object: %+v", byID["req-good"])
	}
	if goodResult["type"] != "succeeded" {
		t.Fatalf("req-good result type = %v, want succeeded: %+v", goodResult["type"], goodResult)
	}
	msg, _ := goodResult["message"].(map[string]any)
	content, _ := msg["content"].([]any)
	if len(content) == 0 {
		t.Fatalf("req-good message has no content: %+v", msg)
	}
	gotText, _ := content[0].(map[string]any)["text"].(string)
	if gotText != refText {
		t.Fatalf("batch result text mismatch:\n  reference: %q\n  batch:     %q", refText, gotText)
	}

	badResult, ok := byID["req-bad"]["result"].(map[string]any)
	if !ok {
		t.Fatalf("req-bad has no result object: %+v", byID["req-bad"])
	}
	if badResult["type"] != "errored" {
		t.Fatalf("req-bad result type = %v, want errored", badResult["type"])
	}
}

// TestBatches_cancelStopsQueuedLines mirrors TestJobs_deleteCancelsAQueuedJob (jobs_http_test.go):
// cancelling a batch must stop lines that are still waiting for admission, not only ones already
// running — the batch-level form of the same J1 gap DELETE /v1/jobs/{id} closes for a single job.
func TestBatches_cancelStopsQueuedLines(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the batches cancel test")
	}
	srv, err := newServer(config{
		models: modelFlag{{name: "m", path: path}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}, kvSessions: 0, maxQueue: 8,
	})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	longBody := `{"model":"m","max_tokens":128,"temperature":0,"messages":[{"role":"user","content":"Count from 1 to 100."}]}`
	holder, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(longBody))
	if err != nil {
		t.Fatalf("POST holder job: %v", err)
	}
	holder.Body.Close()
	time.Sleep(50 * time.Millisecond) // give the holder time to actually claim the turn

	shortBody := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"Say hi."}]}`
	inputLines := []string{
		mustJSONLine(t, map[string]any{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": json.RawMessage(shortBody)}),
		mustJSONLine(t, map[string]any{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": json.RawMessage(shortBody)}),
	}
	fileID := uploadJSONLFile(t, ts, "in.jsonl", inputLines)

	batchResp := postJSON(t, ts.URL+"/v1/batches", map[string]any{
		"input_file_id": fileID, "endpoint": "/v1/chat/completions",
	})
	var created struct{ ID string }
	json.NewDecoder(batchResp.Body).Decode(&created)
	batchResp.Body.Close()

	cancelResp, err := http.Post(ts.URL+"/v1/batches/"+created.ID+"/cancel", "application/json", nil)
	if err != nil {
		t.Fatalf("POST cancel: %v", err)
	}
	cancelResp.Body.Close()

	deadline := time.Now().Add(10 * time.Second)
	var polled struct {
		Status        string               `json:"status"`
		RequestCounts struct{ Failed int } `json:"request_counts"`
	}
	for {
		st, err := http.Get(ts.URL + "/v1/batches/" + created.ID)
		if err != nil {
			t.Fatalf("GET: %v", err)
		}
		json.NewDecoder(st.Body).Decode(&polled)
		st.Body.Close()
		if polled.Status == "cancelled" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("batch never reached cancelled (last status %q)", polled.Status)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if polled.RequestCounts.Failed != 2 {
		t.Fatalf("request_counts.failed = %d, want 2 (both lines cancelled rather than completing)",
			polled.RequestCounts.Failed)
	}
}
