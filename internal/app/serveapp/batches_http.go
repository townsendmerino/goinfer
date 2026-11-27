package serveapp

import (
	"fmt"
	"io"
	"net/http"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/batchapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

func (s *server) handleCreateFile(w http.ResponseWriter, r *http.Request) {
	form, ok := parseUpload(w, r, "invalid multipart/form-data body")
	if !ok {
		return
	}
	defer form.RemoveAll()
	file, header, err := r.FormFile("file")
	if err != nil {
		writeErr(w, http.StatusBadRequest, "missing multipart field \"file\": "+err.Error())
		return
	}
	defer file.Close()
	data := make([]byte, header.Size)
	if _, err := io.ReadFull(file, data); err != nil {
		writeErr(w, http.StatusBadRequest, "reading uploaded file: "+err.Error())
		return
	}
	purpose := r.FormValue("purpose")
	if purpose == "" {
		purpose = "batch"
	}
	f := s.files.Put(header.Filename, purpose, data)
	writeJSON(w, http.StatusOK, batchapi.FileObject(f))
}

func (s *server) handleGetFile(w http.ResponseWriter, r *http.Request) {
	f := s.files.Get(r.PathValue("id"))
	if f == nil {
		writeErr(w, http.StatusNotFound, batchapi.FileNotFound)
		return
	}
	writeJSON(w, http.StatusOK, batchapi.FileObject(f))
}

func (s *server) handleGetFileContent(w http.ResponseWriter, r *http.Request) {
	f := s.files.Get(r.PathValue("id"))
	if f == nil {
		writeErr(w, http.StatusNotFound, batchapi.FileNotFound)
		return
	}
	batchapi.WriteFileContent(w, f.Bytes)
}

func (s *server) finalizeBatch(id string) {
	b := s.batches.Get(id)
	if b == nil {
		return
	}
	b.Finalize(func(files ...stores.NewFile) []*stores.StoredFile {
		return s.files.PutAll(files...)
	})
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
	f := s.files.Get(req.InputFileID)
	if f == nil {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("input_file_id %q not found", req.InputFileID))
		return
	}
	customIDs, reqs, err := batchapi.ParseChatBatch(f.Bytes)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if batchTooLarge(w, len(customIDs)) {
		return
	}
	b := s.batches.Create("openai_chat", req.Endpoint, req.CompletionWindow, req.InputFileID, customIDs)
	if b == nil {
		writeErr(w, http.StatusTooManyRequests, batchapi.BudgetMsg(s.batches.MaxActiveLines(), len(customIDs)))
		return
	}
	runBatchPool(len(reqs), func(i int) { runBatchChatLine(s, b, i, reqs[i]) })
	go s.finalizeBatch(b.ID)

	writeJSON(w, http.StatusOK, b.OpenAIObject())
}

func (s *server) handleGetBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.Get(r.PathValue("id"))
	if b == nil {
		writeErr(w, http.StatusNotFound, "batch not found")
		return
	}
	writeJSON(w, http.StatusOK, b.OpenAIObject())
}

func (s *server) handleCancelBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.Get(r.PathValue("id"))
	if b == nil {
		writeErr(w, http.StatusNotFound, "batch not found")
		return
	}
	b.RequestCancel(s.jobs)
	writeJSON(w, http.StatusOK, b.OpenAIObject())
}

func (s *server) handleCreateMessageBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Requests []batchapi.AnthropicLineReq `json:"requests"`
	}
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	customIDs, reqs, status, kind, msg := batchapi.ParseMessageBatch(req.Requests)
	if status != 0 {
		writeAnthropicErr(w, status, kind, msg)
		return
	}
	b := s.batches.Create("anthropic_messages", "/v1/messages", "", "", customIDs)
	if b == nil {
		writeAnthropicErr(w, http.StatusTooManyRequests, "rate_limit_error", batchapi.BudgetMsg(s.batches.MaxActiveLines(), len(customIDs)))
		return
	}
	runBatchPool(len(reqs), func(i int) { runBatchMessageLine(s, b, i, reqs[i]) })
	go s.finalizeBatch(b.ID)

	writeJSON(w, http.StatusOK, b.AnthropicObject())
}

func (s *server) handleGetMessageBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.Get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	writeJSON(w, http.StatusOK, b.AnthropicObject())
}

func (s *server) handleGetMessageBatchResults(w http.ResponseWriter, r *http.Request) {
	b := s.batches.Get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	results, ready := b.TerminalResults()
	if !ready {
		writeAnthropicErr(w, http.StatusConflict, "invalid_request_error", "batch results are not ready yet")
		return
	}
	batchapi.WriteMessageBatchResults(w, results)
}

func (s *server) handleCancelMessageBatch(w http.ResponseWriter, r *http.Request) {
	b := s.batches.Get(r.PathValue("id"))
	if b == nil {
		writeAnthropicErr(w, http.StatusNotFound, "not_found_error", "batch not found")
		return
	}
	b.RequestCancel(s.jobs)
	writeJSON(w, http.StatusOK, b.AnthropicObject())
}
