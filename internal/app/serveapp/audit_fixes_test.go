package serveapp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/runtime/whisper"
)

type panicCloser struct{}

func (panicCloser) Close() error { panic("close exploded") }

func TestStartDrain_panicStillReleasesLivenessLock(t *testing.T) {
	key := panicCloser{}
	ml := &modelLiveness{refs: 1}
	lm := &loadedModel{name: "boom", closer: key}
	realSrv := newTestDrainServer(lm, ml)
	done := realSrv.startDrain(lm, ml, true)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("startDrain did not finish")
	}
	// ml.rw must be unlocked (acquirable without deadlock) while the draining mark stays set so /health does not lie.
	if !ml.rw.TryRLock() {
		t.Fatal("ml.rw remained write-locked after drain panic")
	}
	ml.rw.RUnlock()
	realSrv.regMu.RLock()
	_, stillDraining := realSrv.draining[lm.name]
	realSrv.regMu.RUnlock()
	if !stillDraining {
		t.Fatal("draining mark should stay set after a drain panic")
	}
}

func newTestDrainServer(lm *loadedModel, ml *modelLiveness) *server {
	s := &server{models: map[string]*loadedModel{}}
	s.liveness = map[io.Closer]*modelLiveness{lm.closer: ml}
	return s
}

func TestSSEHeartbeat_stopIsIdempotent(t *testing.T) {
	rec := httptest.NewRecorder()
	ss := newSSEWriter(rec, rec)
	stop := sseHeartbeat(ss)
	stop()
	stop() // second call must not panic on closed channel
}

type failingWriter struct {
	hdr http.Header
}

func (f *failingWriter) Header() http.Header       { return f.hdr }
func (f *failingWriter) WriteHeader(int)           {}
func (f *failingWriter) Write([]byte) (int, error) { return 0, errors.New("broken pipe") }
func (f *failingWriter) Flush()                    {}

func TestSSEWriter_onErrFiresOnWriteFailure(t *testing.T) {
	fw := &failingWriter{hdr: http.Header{}}
	ss := newSSEWriter(fw, fw)
	called := 0
	ss.onErr = func() { called++ }
	if err := ss.frame("data: hi\n\n"); err == nil {
		t.Fatal("expected write error")
	}
	if called != 1 {
		t.Fatalf("onErr called %d times, want 1", called)
	}
}

func TestBatch_registerAndEvictFailed(t *testing.T) {
	bs := newBatchStore(1)
	js, err := jobs.NewStore("", 8)
	if err != nil {
		t.Fatal(err)
	}
	b1 := bs.create("openai_chat", "/v1/chat/completions", "24h", "file_1", []string{"c1"})
	b1.requestCancel(js)
	ctx, cancel := context.WithCancel(context.Background())
	js.CreatePending("j1", "m", nil)
	if registerBatchJob(b1, 0, "j1", js, cancel) {
		t.Fatal("registerBatchJob returned true on an already-cancelling batch")
	}
	if ctx.Err() == nil {
		t.Fatal("expected cancel to have been invoked when batch was already cancelling")
	}

	// Failed batches must be evictable once capacity is exceeded.
	b1.mu.Lock()
	b1.Status = "failed"
	b1.mu.Unlock()
	b2 := bs.create("openai_chat", "/v1/chat/completions", "24h", "file_2", []string{"c2"})
	if bs.get(b1.ID) != nil {
		t.Fatalf("failed batch %s was not evicted when store exceeded cap", b1.ID)
	}
	if bs.get(b2.ID) == nil {
		t.Fatalf("new batch %s missing", b2.ID)
	}
}

func TestParseEmbedItems_rejectsOverInputLimitImmediately(t *testing.T) {
	arr := make([]string, maxEmbedInputs+1)
	raw, _ := json.Marshal(arr)
	if _, err := parseEmbedItems(raw); err == nil || !strings.Contains(err.Error(), "too many inputs") {
		t.Fatalf("parseEmbedItems over maxEmbedInputs: err = %v, want 'too many inputs'", err)
	}
}

type fakeSpeech struct{}

func (fakeSpeech) TranscribeWith([]float32, whisper.TranscribeOptions) (*whisper.LongResult, error) {
	return &whisper.LongResult{}, nil
}
func (fakeSpeech) Text([]int) string       { return "" }
func (fakeSpeech) HasLanguage(string) bool { return true }

func TestSpeech_rejectsNaNTemperature(t *testing.T) {
	s := &server{
		models: map[string]*loadedModel{},
		speech: map[string]*speechModel{"whisper": {name: "whisper", eng: fakeSpeech{}}},
	}
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	_ = mw.WriteField("model", "whisper")
	_ = mw.WriteField("temperature", "NaN")
	_ = mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.handleTranscriptions(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
}
