package serveapp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// J4 (task-work-queue-2026-09.md): the two batch APIs, both thin translations onto the job store
// J2/J3 already built — see batches_run.go for the per-line execution and batches_finalize.go for
// completion + output assembly. This file is HTTP-shape only: decode, project, translate status
// vocabulary. Text-only chat scope this pass, matching J3's own line.

const maxFileUploadMemory = 32 << 20 // in-memory threshold before ParseMultipartForm spills to disk

// --- OpenAI: POST /v1/files, GET /v1/files/{id}, GET /v1/files/{id}/content ---

func (s *server) handleCreateFile(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(maxFileUploadMemory); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid multipart/form-data body: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeErr(w, http.StatusBadRequest, "missing multipart field \"file\": "+err.Error())
		return
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading uploaded file: "+err.Error())
		return
	}
	purpose := r.FormValue("purpose")
	if purpose == "" {
		purpose = "batch"
	}
	f := s.files.put(header.Filename, purpose, data)
	writeJSON(w, http.StatusOK, fileObject(f))
}

func fileObject(f *storedFile) map[string]any {
	return map[string]any{
		"id": f.ID, "object": "file", "bytes": len(f.Bytes),
		"created_at": f.CreatedAt.Unix(), "filename": f.Filename, "purpose": f.Purpose,
	}
}

func (s *server) handleGetFile(w http.ResponseWriter, r *http.Request) {
	f := s.files.get(r.PathValue("id"))
	if f == nil {
		writeErr(w, http.StatusNotFound, "file not found")
		return
	}
	writeJSON(w, http.StatusOK, fileObject(f))
}

func (s *server) handleGetFileContent(w http.ResponseWriter, r *http.Request) {
	f := s.files.get(r.PathValue("id"))
	if f == nil {
		writeErr(w, http.StatusNotFound, "file not found")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(f.Bytes)
}

// --- OpenAI: POST /v1/batches, GET /v1/batches/{id}, POST /v1/batches/{id}/cancel ---

// batchInputLine is one line of an OpenAI batch input JSONL file.
type batchInputLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

func (s *server) handleCreateBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		InputFileID      string `json:"input_file_id"`
		Endpoint         string `json:"endpoint"`
		CompletionWindow string `json:"completion_window"`
	}
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Endpoint != "/v1/chat/completions" {
		writeErr(w, http.StatusBadRequest,
			fmt.Sprintf("endpoint %q is not supported this pass; only \"/v1/chat/completions\" is", req.Endpoint))
		return
	}
	f := s.files.get(req.InputFileID)
	if f == nil {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("input_file_id %q not found", req.InputFileID))
		return
	}

	var lines []batchInputLine
	var customIDs []string
	var reqs []chatReq
	sc := bufio.NewScanner(bytes.NewReader(f.Bytes))
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var l batchInputLine
		if err := json.Unmarshal([]byte(raw), &l); err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("input file line %d: invalid JSON: %s", lineNo, err.Error()))
			return
		}
		if l.CustomID == "" {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("input file line %d: custom_id is required", lineNo))
			return
		}
		var creq chatReq
		if err := json.Unmarshal(l.Body, &creq); err != nil {
			// A malformed per-line BODY is that line's own problem, not the whole batch's — record
			// it as a request whose validation will fail immediately in runBatchChatLine instead of
			// rejecting the batch outright (matches real batch-API "one bad line never blocks the
			// rest" semantics, task doc's own framing).
			creq = chatReq{}
		}
		lines = append(lines, l)
		customIDs = append(customIDs, l.CustomID)
		reqs = append(reqs, creq)
	}
	if err := sc.Err(); err != nil {
		writeErr(w, http.StatusBadRequest, "reading input file: "+err.Error())
		return
	}
	if len(lines) == 0 {
		writeErr(w, http.StatusBadRequest, "input file has no request lines")
		return
	}

	b := s.batches.create("openai_chat", req.Endpoint, req.CompletionWindow, req.InputFileID, customIDs)
	for i, creq := range reqs {
		go runBatchChatLine(s, b, i, creq)
	}
	go s.finalizeBatch(b.ID)

	writeJSON(w, http.StatusOK, openAIBatchObject(b))
}

func openAIBatchObject(b *batchRecord) map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()
	status := b.Status // internal vocabulary already matches OpenAI's almost verbatim
	total, completed, failed := 0, 0, 0
	for _, r := range b.Results {
		if r.CustomID == "" {
			continue // this line hasn't finished yet — every real result has a non-empty CustomID
		}
		total++
		if r.ErrType == "" && r.Body != nil {
			completed++
		} else {
			failed++
		}
	}
	out := map[string]any{
		"id": b.ID, "object": "batch", "endpoint": b.Endpoint, "input_file_id": b.InputFileID,
		"completion_window": b.CompletionWindow, "status": status,
		"created_at": b.Created.Unix(),
		"request_counts": map[string]any{
			"total": len(b.CustomIDs), "completed": completed, "failed": failed,
		},
	}
	if b.OutputFileID != "" {
		out["output_file_id"] = b.OutputFileID
	}
	if b.ErrorFileID != "" {
		out["error_file_id"] = b.ErrorFileID
	}
	if b.CompletedAt != nil {
		out["completed_at"] = b.CompletedAt.Unix()
	}
	return out
}

func (s *server) handleGetBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.get(r.PathValue("id"))
	if b == nil {
		writeErr(w, http.StatusNotFound, "batch not found")
		return
	}
	writeJSON(w, http.StatusOK, openAIBatchObject(b))
}

func (s *server) handleCancelBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.get(r.PathValue("id"))
	if b == nil {
		writeErr(w, http.StatusNotFound, "batch not found")
		return
	}
	b.requestCancel(s.jobs)
	writeJSON(w, http.StatusOK, openAIBatchObject(b))
}

// --- Anthropic: POST /v1/messages/batches, GET .../{id}, GET .../{id}/results, POST .../{id}/cancel ---

type anthropicBatchLineReq struct {
	CustomID string          `json:"custom_id"`
	Params   json.RawMessage `json:"params"`
}

func (s *server) handleCreateMessageBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Requests []anthropicBatchLineReq `json:"requests"`
	}
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	if len(req.Requests) == 0 {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "requests must not be empty")
		return
	}
	customIDs := make([]string, len(req.Requests))
	reqs := make([]anthropicReq, len(req.Requests))
	for i, line := range req.Requests {
		if line.CustomID == "" {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("requests[%d]: custom_id is required", i))
			return
		}
		customIDs[i] = line.CustomID
		var areq anthropicReq
		_ = json.Unmarshal(line.Params, &areq) // a malformed params object is THIS line's own error, see handleCreateBatch's identical comment
		reqs[i] = areq
	}

	b := s.batches.create("anthropic_messages", "/v1/messages", "", "", customIDs)
	for i, areq := range reqs {
		go runBatchMessageLine(s, b, i, areq)
	}
	go s.finalizeBatch(b.ID)

	writeJSON(w, http.StatusOK, anthropicBatchObject(b))
}

func anthropicBatchObject(b *batchRecord) map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()
	// Anthropic vocabulary: in_progress | canceling | ended. No "finalizing" — folds into
	// in_progress; "completed" and "cancelled" both surface as "ended", matching the real API.
	status := "in_progress"
	switch b.Status {
	case "cancelling":
		status = "canceling"
	case "completed", "cancelled":
		status = "ended"
	}
	processing, succeeded, errored, canceled := 0, 0, 0, 0
	for _, r := range b.Results {
		if r.CustomID == "" {
			processing++
			continue
		}
		switch {
		case r.ErrType == "cancelled":
			canceled++
		case r.ErrType != "" || r.Body == nil:
			errored++
		default:
			succeeded++
		}
	}
	out := map[string]any{
		"id": b.ID, "type": "message_batch", "processing_status": status,
		"created_at": b.Created.Format(time.RFC3339),
		"request_counts": map[string]any{
			"processing": processing, "succeeded": succeeded, "errored": errored,
			"canceled": canceled, "expired": 0,
		},
		"results_url": nil,
	}
	if status == "ended" {
		out["results_url"] = "/v1/messages/batches/" + b.ID + "/results"
	}
	if b.CompletedAt != nil {
		out["ended_at"] = b.CompletedAt.Format(time.RFC3339)
	}
	return out
}

func (s *server) handleGetMessageBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	writeJSON(w, http.StatusOK, anthropicBatchObject(b))
}

func (s *server) handleGetMessageBatchResults(w http.ResponseWriter, r *http.Request) {
	b := s.batches.get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Status != "completed" && b.Status != "cancelled" {
		writeAnthropicErr(w, http.StatusConflict, "invalid_request_error", "batch results are not ready yet")
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	for _, res := range b.Results {
		line := map[string]any{"custom_id": res.CustomID}
		switch {
		case res.ErrType == "cancelled":
			line["result"] = map[string]any{"type": "canceled"}
		case res.ErrType != "" || res.Body == nil:
			line["result"] = map[string]any{"type": "errored",
				"error": map[string]any{"type": res.ErrType, "message": res.ErrMsg}}
		default:
			line["result"] = map[string]any{"type": "succeeded", "message": res.Body}
		}
		lineBytes, _ := json.Marshal(line)
		_, _ = w.Write(lineBytes)
		_, _ = w.Write([]byte("\n"))
	}
}

func (s *server) handleCancelMessageBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	b.requestCancel(s.jobs)
	writeJSON(w, http.StatusOK, anthropicBatchObject(b))
}
