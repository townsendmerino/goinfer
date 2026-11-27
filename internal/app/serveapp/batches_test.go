package serveapp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

// A batch line with an image or audio part is refused by its part type, before any payload is decoded: a payload that is not
// valid base64 gets the unsupported-media refusal, naming what it refuses, never a base64 decode error.
func TestBatchLines_imagePartsRefusedBeforeDecode(t *testing.T) {
	bs := stores.NewBatchStore(0)
	s := &server{batches: bs}
	b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"oa-image", "oa-audio"})
	image := chatReq{Messages: []chatMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!not-base64!!!"}}]`)}}}
	audio := chatReq{Messages: []chatMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"input_audio","input_audio":{"data":"!!!not-base64!!!","format":"wav"}}]`)}}}
	runBatchChatLine(s, b, 0, image)
	runBatchChatLine(s, b, 1, audio)
	maxTok := 8
	ab := bs.Create("anthropic_messages", "/v1/messages", "", "", []string{"an-image"})
	areq := anthropicReq{MaxTokens: &maxTok, Messages: []anthropicMessage{{Role: "user", Content: json.RawMessage(
		`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!not-base64!!!"}}]`)}}}
	runBatchMessageLine(s, ab, 0, areq)
	for _, c := range []struct {
		r    stores.BatchLineResult
		want string
	}{
		{b.ResultAt(0), "do not support image or audio inputs"},
		{b.ResultAt(1), "do not support image or audio inputs"},
		{ab.ResultAt(0), "do not support image inputs"},
	} {
		if c.r.StatusCode != http.StatusBadRequest || !strings.Contains(c.r.ErrMsg, c.want) {
			t.Errorf("line %s: status %d, message %q; want 400 with %q", c.r.CustomID, c.r.StatusCode, c.r.ErrMsg, c.want)
		}
	}
}

// The server-wide line budget: Create reserves every line of a batch at once and refuses a batch that would exceed the budget, and each
// line returns its reservation when it finishes, whether it completed, failed validation, panicked, or was cancelled.
func TestBatchStore_activeLineBudgetIsReservedAndReleased(t *testing.T) {
	bs := stores.NewBatchStoreWithMaxActive(0, 3)
	s := &server{batches: bs}
	b1 := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"a", "b"})
	if b1 == nil {
		t.Fatal("a batch inside the budget was refused")
	}
	if b := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"c", "d"}); b != nil {
		t.Fatal("a batch past the server-wide line budget was created")
	}
	if got := bs.ActiveLineCount(); got != 2 {
		t.Fatalf("active lines after one admitted and one refused batch = %d, want 2", got)
	}
	js, err := jobs.NewStore("", 0)
	if err != nil {
		t.Fatal(err)
	}
	b1.RequestCancel(js)
	runBatchPool(2, func(i int) { runBatchChatLine(s, b1, i, chatReq{}) })
	b1.Wait()
	if got := bs.ActiveLineCount(); got != 0 {
		t.Fatalf("active lines after every line of a cancelled batch finished = %d, want 0", got)
	}
	b2 := bs.Create("anthropic_messages", "/v1/messages", "", "", []string{"x", "y", "z"})
	if b2 == nil {
		t.Fatal("the released budget was not available to the next batch")
	}
	runBatchPool(3, func(i int) { runBatchMessageLine(s, b2, i, anthropicReq{}) })
	b2.Wait()
	if got := bs.ActiveLineCount(); got != 0 {
		t.Fatalf("active lines after every line failed validation = %d, want 0", got)
	}
	// A registered model with nothing loaded behind it: the line panics past validation, and recoverBatchLine fails it.
	ps := &server{batches: bs, models: map[string]*loadedModel{"m": {name: "m"}}}
	b3 := bs.Create("openai_chat", "/v1/chat/completions", "", "f", []string{"p"})
	runBatchChatLine(ps, b3, 0, chatReq{Model: "m", Messages: []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}}})
	if r := b3.ResultAt(0); r.StatusCode != http.StatusInternalServerError {
		t.Fatalf("the panicking line's result = %+v, want a recovered 500", r)
	}
	if got := bs.ActiveLineCount(); got != 0 {
		t.Fatalf("active lines after a line panicked = %d, want 0", got)
	}
	b3.LineFinished() // a second release of the same line is a no-op, never a negative count
	if got := bs.ActiveLineCount(); got != 0 {
		t.Fatalf("active lines after a repeated release = %d, want 0", got)
	}
}

// A saturated line budget is a 429 in each dialect's error shape, and no batch is created.
func TestBatches_serverWideLineBudgetIs429(t *testing.T) {
	srv := &server{files: stores.NewFileStore(8, 0), batches: stores.NewBatchStoreWithMaxActive(8, 1)}
	ts := httptest.NewServer(batchesTestMux(srv))
	defer ts.Close()

	row := func(id string) string {
		return mustJSONLine(t, map[string]any{"custom_id": id, "method": "POST", "url": "/v1/chat/completions", "body": map[string]any{}})
	}
	fileID := uploadJSONLFile(t, ts, "in.jsonl", []string{row("a"), row("b")})
	resp := postJSON(t, ts.URL+"/v1/batches", map[string]any{"input_file_id": fileID, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
	defer resp.Body.Close()
	var oa struct {
		Error struct{ Type, Message string } `json:"error"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&oa)
	if resp.StatusCode != http.StatusTooManyRequests || oa.Error.Type != "rate_limit_error" {
		t.Errorf("OpenAI dialect: status %d, error %+v; want 429 rate_limit_error", resp.StatusCode, oa.Error)
	}

	req := map[string]any{"params": map[string]any{}}
	resp2 := postJSON(t, ts.URL+"/v1/messages/batches", map[string]any{"requests": []map[string]any{
		{"custom_id": "x", "params": req["params"]}, {"custom_id": "y", "params": req["params"]}}})
	defer resp2.Body.Close()
	var an struct {
		Type  string                         `json:"type"`
		Error struct{ Type, Message string } `json:"error"`
	}
	_ = json.NewDecoder(resp2.Body).Decode(&an)
	if resp2.StatusCode != http.StatusTooManyRequests || an.Type != "error" || an.Error.Type != "rate_limit_error" {
		t.Errorf("Anthropic dialect: status %d, body %+v; want 429 rate_limit_error", resp2.StatusCode, an)
	}
	if n := srv.batches.Len(); n != 0 {
		t.Errorf("a refused batch must not be created (%d in the store)", n)
	}
}

// A batch line whose drive was cut short by server shutdown is a 503 server_shutdown even though the drive returned its
// context error; a failed drive outside shutdown stays a 500.
func TestBatchLines_shutdownOutranksTheDriveError(t *testing.T) {
	shut, shutCancel := context.WithCancelCause(context.Background())
	shutCancel(errServerShutdown)
	if r, failed := driveLineError(shut, "a", context.Canceled, ""); !failed || r.StatusCode != http.StatusServiceUnavailable || !strings.Contains(r.ErrMsg, "shutting down") {
		t.Errorf("a drive cut short by shutdown: %+v (failed %v); want 503 server_shutdown", r, failed)
	}
	if r, failed := driveLineError(context.Background(), "b", errors.New("boom"), ""); !failed || r.StatusCode != 500 {
		t.Errorf("a failed drive: %+v (failed %v); want 500", r, failed)
	}
	if _, failed := driveLineError(context.Background(), "c", nil, ""); failed {
		t.Error("a completed drive reads as failed")
	}
}
