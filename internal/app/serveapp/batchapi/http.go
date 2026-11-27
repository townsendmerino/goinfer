package batchapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/anthropicapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
	"github.com/townsendmerino/goinfer/internal/batchio"
)

// FileNotFound answers a /v1/files lookup for an ID absent from the bounded file store.
const FileNotFound = "file not found: an unknown id, or a file dropped to stay within the store's limits (256 files, -max-file-store-bytes)"

// FileObject builds an OpenAI /v1/files JSON object from f.
func FileObject(f *stores.StoredFile) map[string]any {
	return map[string]any{
		"id": f.ID, "object": "file", "bytes": len(f.Bytes),
		"created_at": f.CreatedAt.Unix(), "filename": f.Filename, "purpose": f.Purpose,
	}
}

// WriteFileContent streams data to w in 256 KiB chunks, renewing the write deadline per chunk.
func WriteFileContent(w http.ResponseWriter, data []byte) {
	rc := http.NewResponseController(w)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	const chunkBytes = 256 << 10
	for off := 0; off < len(data); off += chunkBytes {
		end := min(off+chunkBytes, len(data))
		_ = rc.SetWriteDeadline(time.Now().Add(streamframe.WriteTimeout))
		if _, err := w.Write(data[off:end]); err != nil {
			return
		}
	}
}

// BudgetMsg formats the 429 refusal message when the server-wide active-line budget is full.
func BudgetMsg(limit, n int) string {
	return fmt.Sprintf("the server is running its limit of batch requests (%d across all batches); this batch of %d does not fit until earlier lines finish: retry later",
		limit, n)
}

// ParseChatBatch parses an OpenAI JSONL batch input file into custom IDs and per-line ChatReq structs.
func ParseChatBatch(data []byte) (customIDs []string, reqs []wirecore.ChatReq, err error) {
	lines, err := batchio.ParseInput(data)
	if err != nil {
		return nil, nil, err
	}
	if err := batchio.CheckUnique(lines); err != nil {
		return nil, nil, fmt.Errorf("input file: %w", err)
	}
	customIDs = make([]string, 0, len(lines))
	reqs = make([]wirecore.ChatReq, 0, len(lines))
	for _, l := range lines {
		var creq wirecore.ChatReq
		if err := json.Unmarshal(l.Body, &creq); err != nil {
			creq = wirecore.ChatReq{}
		}
		customIDs = append(customIDs, l.CustomID)
		reqs = append(reqs, creq)
	}
	return customIDs, reqs, nil
}

// AnthropicLineReq is one item in an Anthropic Message Batch requests array.
type AnthropicLineReq struct {
	CustomID string          `json:"custom_id"`
	Params   json.RawMessage `json:"params"`
}

// ParseMessageBatch validates an Anthropic Message Batch request slice into custom IDs and per-line Req structs.
func ParseMessageBatch(lines []AnthropicLineReq) (customIDs []string, reqs []anthropicapi.Req, status int, kind, msg string) {
	if len(lines) == 0 {
		return nil, nil, http.StatusBadRequest, "invalid_request_error", "requests must not be empty"
	}
	if len(lines) > stores.MaxBatchLines {
		return nil, nil, http.StatusRequestEntityTooLarge, "request_too_large",
			fmt.Sprintf("a batch holds at most %d requests, this one has %d: split it", stores.MaxBatchLines, len(lines))
	}
	customIDs = make([]string, len(lines))
	reqs = make([]anthropicapi.Req, len(lines))
	for i, line := range lines {
		if line.CustomID == "" {
			return nil, nil, http.StatusBadRequest, "invalid_request_error",
				fmt.Sprintf("requests[%d]: custom_id is required", i)
		}
		customIDs[i] = line.CustomID
		var areq anthropicapi.Req
		_ = json.Unmarshal(line.Params, &areq)
		reqs[i] = areq
	}
	if err := batchio.CheckUniqueIDs(customIDs); err != nil {
		return nil, nil, http.StatusBadRequest, "invalid_request_error", "requests: " + err.Error()
	}
	return customIDs, reqs, 0, "", ""
}

// WriteMessageBatchResults streams completed Anthropic Message Batch JSONL lines to w.
func WriteMessageBatchResults(w http.ResponseWriter, results []stores.BatchLineResult) {
	rc := http.NewResponseController(w)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	for _, res := range results {
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
		lineBytes, err := json.Marshal(line)
		if err != nil {
			continue
		}
		_ = rc.SetWriteDeadline(time.Now().Add(streamframe.WriteTimeout))
		if _, err := w.Write(lineBytes); err != nil {
			return
		}
		if _, err := w.Write([]byte("\n")); err != nil {
			return
		}
	}
}
