package batchapi

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
)

func TestParseAndPool(t *testing.T) {
	ids, reqs, err := ParseChatBatch([]byte(`{"custom_id":"c1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}` + "\n"))
	if err != nil || len(ids) != 1 || ids[0] != "c1" || reqs[0].Model != "m" {
		t.Fatalf("ParseChatBatch = %v, %+v, %v", ids, reqs, err)
	}
	mIDs, mReqs, status, _, _ := ParseMessageBatch([]AnthropicLineReq{{CustomID: "m1", Params: json.RawMessage(`{"model":"m","max_tokens":8}`)}})
	if status != 0 || len(mIDs) != 1 || mReqs[0].Model != "m" {
		t.Fatalf("ParseMessageBatch = %v, %+v, status=%d", mIDs, mReqs, status)
	}

	var count atomic.Int32
	var wg sync.WaitGroup
	wg.Add(20)
	RunPool(20, func(int) {
		count.Add(1)
		wg.Done()
	})
	wg.Wait()
	if count.Load() != 20 {
		t.Fatalf("RunPool ran %d tasks, want 20", count.Load())
	}

	shutErr := errors.New("shutdown")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(shutErr)
	if r, failed := ClassifyDriveError(ctx, shutErr, 499, "id1", context.Canceled, ""); !failed || r.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("ClassifyDriveError = %+v, %v", r, failed)
	}
}
