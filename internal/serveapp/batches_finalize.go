package serveapp

import (
	"bytes"
	"encoding/json"
	"time"
)

// finalizeBatch waits for every line to finish (b.wg.Wait() — no polling, see batchRecord's own
// doc comment) then assembles the batch's output. Launched with `go s.finalizeBatch(id)` right
// after a batch's lines are all submitted (batches_http.go), so it runs concurrently with them.
func (s *server) finalizeBatch(id string) {
	b := s.batches.get(id)
	if b == nil {
		return
	}
	b.wg.Wait()

	b.mu.Lock()
	defer b.mu.Unlock()
	cancelled := b.Status == "cancelling"
	now := time.Now()
	b.CompletedAt = &now
	if b.Kind == "openai_chat" {
		out, errs := assembleOpenAIOutput(b)
		b.OutputFileID = s.files.put(id+"_output.jsonl", "batch_output", out).ID
		if len(errs) > 0 {
			b.ErrorFileID = s.files.put(id+"_error.jsonl", "batch_error", errs).ID
		}
	}
	// anthropic: b.Results IS the output — handleGetMessageBatchResults reads it directly, no
	// file store involved (Anthropic's batch API has no separate Files concept to translate into).
	if cancelled {
		b.Status = "cancelled"
	} else {
		b.Status = "completed"
	}
}

// assembleOpenAIOutput builds the output and error JSONL files from b.Results, in ORIGINAL input
// order, split exactly the way the real API splits them: a line with a body goes to output, a line
// with an error goes to error — never both, never neither.
func assembleOpenAIOutput(b *batchRecord) (output, errors []byte) {
	var out, errs bytes.Buffer
	for _, r := range b.Results {
		line := map[string]any{"id": "batch_req_" + reqID(), "custom_id": r.CustomID}
		var buf *bytes.Buffer
		if r.ErrType != "" || r.Body == nil {
			line["response"] = nil
			line["error"] = map[string]any{"code": r.ErrType, "message": r.ErrMsg}
			buf = &errs
		} else {
			line["response"] = map[string]any{"status_code": r.StatusCode, "body": r.Body}
			line["error"] = nil
			buf = &out
		}
		lineBytes, _ := json.Marshal(line)
		buf.Write(lineBytes)
		buf.WriteByte('\n')
	}
	return out.Bytes(), errs.Bytes()
}
