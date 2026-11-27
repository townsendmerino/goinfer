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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/runtime/whisper"
)

type panicCloser struct{}

func (panicCloser) Close() error { panic("close exploded") }

func TestStartDrain_panicStillReleasesLivenessLock(t *testing.T) {
	key := panicCloser{}
	ml := &modelLiveness{refs: 1}
	lm := &loadedModel{modelIdentity: modelIdentity{name: "boom", closer: key}}
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
	s := &server{modelRegistry: modelRegistry{models: map[string]*loadedModel{}}}
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
	called := 0
	send := sseJSONSender(fw, fw, func() { called++ })
	send("progress", map[string]string{"step": "1"})
	send("progress", map[string]string{"step": "2"})
	if called != 1 {
		t.Fatalf("sseJSONSender cancel called %d times, want 1", called)
	}
}

func TestBatch_registerAndEvictFailed(t *testing.T) {
	bs := stores.NewBatchStore(1)
	js, err := jobs.NewStore("", 8)
	if err != nil {
		t.Fatal(err)
	}
	b1 := bs.Create("openai_chat", "/v1/chat/completions", "24h", "file_1", []string{"c1"})
	b1.RequestCancel(js)
	ctx, cancel := context.WithCancel(context.Background())
	js.CreatePending("j1", "m", nil)
	if b1.RegisterJob(0, "j1", js, cancel) {
		t.Fatal("RegisterJob returned true on an already-cancelling batch")
	}
	if ctx.Err() == nil {
		t.Fatal("expected cancel to have been invoked when batch was already cancelling")
	}

	// Failed batches must stay terminal under RequestCancel and be evictable once capacity is exceeded.
	b1.MarkFailed()
	b1.RequestCancel(js)
	if st := b1.Status(); st != "failed" {
		t.Fatalf("RequestCancel on failed batch flipped status to %q, want failed", st)
	}
	b2 := bs.Create("openai_chat", "/v1/chat/completions", "24h", "file_2", []string{"c2"})
	if bs.Get(b1.ID) != nil {
		t.Fatalf("failed batch %s was not evicted when store exceeded cap", b1.ID)
	}
	if bs.Get(b2.ID) == nil {
		t.Fatalf("new batch %s missing", b2.ID)
	}
}

func TestRunBatchPool_boundsPeakConcurrency(t *testing.T) {
	const total = 32
	var active, peak, done atomic.Int64
	var wg sync.WaitGroup
	wg.Add(total)
	runBatchPool(total, func(i int) {
		defer wg.Done()
		cur := active.Add(1)
		for {
			prev := peak.Load()
			if cur <= prev || peak.CompareAndSwap(prev, cur) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		active.Add(-1)
		done.Add(1)
	})
	wg.Wait()
	if done.Load() != total {
		t.Fatalf("completed %d items, want %d", done.Load(), total)
	}
	if p := peak.Load(); p > batchLineWorkers {
		t.Fatalf("peak concurrency = %d, want <= %d", p, batchLineWorkers)
	}
}

func TestLifeContext_shutdownRecordsInterrupted(t *testing.T) {
	// Shutdown cause maps to jobs.Interrupted (distinct from user jobs.Cancelled).
	shutCtx, shutCancel := context.WithCancelCause(context.Background())
	shutCancel(errServerShutdown)
	got := notAdmitted(shutCtx, "", &loadedModel{modelIdentity: modelIdentity{name: "m"}})
	if got.state != jobs.Interrupted || got.status != http.StatusServiceUnavailable {
		t.Fatalf("notAdmitted(shutdown) = %+v, want Interrupted/503", got)
	}

	userCtx, userCancel := context.WithCancel(context.Background())
	userCancel()
	gotUser := notAdmitted(userCtx, "", &loadedModel{modelIdentity: modelIdentity{name: "m"}})
	if gotUser.state != jobs.Cancelled {
		t.Fatalf("notAdmitted(user cancel) = %+v, want Cancelled", gotUser)
	}

	// End-to-end: a background job queued behind a full turn slot ends as jobs.Interrupted when lifeCtx shuts down.
	lifeCtx, lifeCancel := context.WithCancelCause(context.Background())
	js, err := jobs.NewStore("", 8)
	if err != nil {
		t.Fatal(err)
	}
	lm := &loadedModel{modelIdentity: modelIdentity{name: "m"}}
	lm.setConcurrency(config{maxConcurrent: 1, maxQueue: 4})
	// Occupy the lone turn slot so runJob blocks in tryEnter.
	if ok, _ := lm.tryEnter(context.Background(), admissionRecord{id: "blocker"}, nil); !ok {
		t.Fatal("failed to claim blocker turn")
	}
	defer lm.exit()

	s := &server{modelRegistry: modelRegistry{models: map[string]*loadedModel{"m": lm}}, serverStores: serverStores{jobs: js}, opState: opState{lifeCtx: lifeCtx}}
	js.CreatePending("job_shut", "m", []int{1})
	done := make(chan struct{})
	go func() {
		defer close(done)
		runJob(s, lm, func() {}, genRequest{id: "job_shut", promptIDs: []int{1}}, admissionRecord{promptIDs: []int{1}}, 1)
	}()
	time.Sleep(10 * time.Millisecond)
	lifeCancel(errServerShutdown)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("runJob did not exit when server lifeCtx cancelled")
	}
	j, ok := js.Snapshot("job_shut")
	if !ok {
		t.Fatal("job_shut missing from store")
	}
	if j.State != jobs.Interrupted {
		t.Fatalf("job_shut state = %q, want %q", j.State, jobs.Interrupted)
	}
}

func TestRoutes_fileUploadSemaphoreReturns503(t *testing.T) {
	s := &server{
		modelRegistry: modelRegistry{models: map[string]*loadedModel{}},
		serverStores:  serverStores{files: stores.NewFileStore(0, 0)},
	}
	mux, _ := s.routes(config{}, "", io.Discard)

	entered := make(chan struct{}, 8)
	unblock := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			pr, pw := io.Pipe()
			go func() {
				entered <- struct{}{}
				<-unblock
				_ = pw.Close()
			}()
			req := httptest.NewRequest(http.MethodPost, "/v1/files", pr)
			req.Header.Set("Content-Type", "multipart/form-data; boundary=abc")
			mux.ServeHTTP(httptest.NewRecorder(), req)
		})
	}
	for range 8 {
		<-entered
	}
	// Give handlers a moment to acquire the upload semaphore and block on ParseMultipartForm.
	time.Sleep(20 * time.Millisecond)

	req := httptest.NewRequest(http.MethodPost, "/v1/files", strings.NewReader(""))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=abc")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	close(unblock)
	wg.Wait()
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("9th concurrent POST /v1/files status = %d, want 503; body=%s", rec.Code, rec.Body.String())
	}
}

func TestBodyCaps_autoTextCapClampedTo64MiB(t *testing.T) {
	if got := autoTextBodyCap(0); got != maxBodyBytes {
		t.Fatalf("autoTextBodyCap(0) = %d, want floor %d", got, maxBodyBytes)
	}
	if got := autoTextBodyCap(10 << 20); got != 10<<20 {
		t.Fatalf("autoTextBodyCap(10 MiB) = %d, want 10 MiB", got)
	}
	if got := autoTextBodyCap(512 << 20); got != maxAutoTextBodyBytes {
		t.Fatalf("autoTextBodyCap(512 MiB) = %d, want clamp %d", got, maxAutoTextBodyBytes)
	}
}

type closableFakeEmbedder struct {
	closed atomic.Bool
}

func (e *closableFakeEmbedder) Encode(string, bool) ([]float32, error) { return []float32{1}, nil }
func (e *closableFakeEmbedder) EncodeBatch([]string, []bool, int) ([][]float32, error) {
	return nil, nil
}
func (e *closableFakeEmbedder) HiddenDim() int { return 1 }
func (e *closableFakeEmbedder) Close() error   { e.closed.Store(true); return nil }

type closableFakeSpeech struct {
	fakeSpeech
	closed atomic.Bool
}

func (f *closableFakeSpeech) Close() error { f.closed.Store(true); return nil }

func TestHandlerCloser_closesEmbedderAndSpeech(t *testing.T) {
	emb := &closableFakeEmbedder{}
	sp := &closableFakeSpeech{}
	sm := &speechModel{name: "whisper", eng: sp}
	s := &server{
		modelRegistry: modelRegistry{models: map[string]*loadedModel{}, speech: map[string]*speechModel{"whisper": sm}},
		embedService:  embedService{embed: emb},
	}
	_, stop := context.WithCancelCause(context.Background())
	hc := &handlerCloser{srv: s, stop: stop}
	if err := hc.Close(); err != nil {
		t.Fatalf("handlerCloser.Close: %v", err)
	}
	if !emb.closed.Load() {
		t.Fatal("handlerCloser.Close did not close s.embed")
	}
	if !sp.closed.Load() {
		t.Fatal("handlerCloser.Close did not close speech engine")
	}
	if !sm.closed {
		t.Fatal("handlerCloser.Close did not mark speechModel closed")
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
		modelRegistry: modelRegistry{models: map[string]*loadedModel{}, speech: map[string]*speechModel{"whisper": {name: "whisper", eng: fakeSpeech{}}}},
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

type fakeVisionTower struct{ tok int }

func (f fakeVisionTower) family() string { return "gemma3" }
func (f fakeVisionTower) kind() string   { return "gemma3" }
func (f fakeVisionTower) imageTok() int  { return f.tok }
func (f fakeVisionTower) close()         {}
func (f fakeVisionTower) prep(_ *loadedModel, _ imageRef) (imagePrep, error) {
	return imagePrep{
		n:        1,
		block:    "<IMG>",
		features: func() ([]float32, error) { return []float32{1}, nil },
	}, nil
}

func TestServeVisionResponses_checksAllMediaAndReasoning(t *testing.T) {
	tk := scriptTokenizer(t)
	ids, err := tk.Encode("<think>\nLook at pixels.\n</think>\n\nA cat.", false)
	if err != nil {
		t.Fatal(err)
	}
	imgIDs, err := tk.Encode("<IMG>", false)
	if err != nil || len(imgIDs) == 0 {
		t.Fatalf("encode <IMG>: %v", err)
	}
	s := &server{serverStores: serverStores{responses: newResponseStore(8)}}
	lm := &loadedModel{
		modelIdentity:     modelIdentity{tk: tk, name: "vlm"},
		modelModalities:   modelModalities{vision: fakeVisionTower{tok: imgIDs[0]}},
		modelScriptedSeam: modelScriptedSeam{scripted: ids},
	}
	lm.fakeEngine = scriptedEngine{lm}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	lm.setConcurrency(config{maxConcurrent: 1, maxQueue: 2})
	s.models = map[string]*loadedModel{"vlm": lm}

	// 1. Multi-media [image, audio] must reject audio even when imgs[0] is an image.
	recAudio := httptest.NewRecorder()
	s.serveVisionResponses(recAudio, httptest.NewRequest(http.MethodPost, "/v1/responses", nil), lm,
		responseReq{Model: "vlm"}, nil, []imageRef{{Data: []byte("img")}, {Data: []byte("wav"), Audio: true}},
		sampling{}, "resp_audio", 1, false)
	if recAudio.Code != http.StatusBadRequest || !strings.Contains(recAudio.Body.String(), "audio tower") {
		t.Fatalf("mixed [image, audio] status=%d body=%s, want 400 audio tower error", recAudio.Code, recAudio.Body.String())
	}

	// 2. Non-streaming vision Responses emits reasoning item + message and stores reasoning text.
	msgs := []chatMessage{{Role: "user", Content: json.RawMessage(`"describe"`)}}
	rec := httptest.NewRecorder()
	s.serveVisionResponses(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", nil), lm,
		responseReq{Model: "vlm"}, msgs, []imageRef{{Data: []byte("img")}},
		sampling{}, "resp_vis_1", 1, true)
	if rec.Code != http.StatusOK {
		t.Fatalf("non-streaming vision Responses status=%d body=%s", rec.Code, rec.Body.String())
	}
	var respObj map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &respObj); err != nil {
		t.Fatal(err)
	}
	out, _ := respObj["output"].([]any)
	if len(out) != 2 || out[0].(map[string]any)["type"] != "reasoning" || out[1].(map[string]any)["type"] != "message" {
		t.Fatalf("vision Responses output = %v, want [reasoning, message]", out)
	}
	if stored := s.responses.Get("resp_vis_1"); stored == nil || len(stored.Messages) < 2 || stored.Messages[1].ReasoningContent != "Look at pixels." {
		t.Fatalf("stored vision response missing reasoning: stored=%+v", stored)
	}

	// 3. Streaming vision Responses emits response.reasoning_text.delta events.
	recStream := httptest.NewRecorder()
	s.serveVisionResponses(recStream, httptest.NewRequest(http.MethodPost, "/v1/responses", nil), lm,
		responseReq{Model: "vlm", Stream: true}, msgs, []imageRef{{Data: []byte("img")}},
		sampling{}, "resp_vis_2", 1, false)
	if !strings.Contains(recStream.Body.String(), "response.reasoning_text.delta") || !strings.Contains(recStream.Body.String(), "Look at pixels.") {
		t.Fatalf("streaming vision Responses missing reasoning delta: %s", recStream.Body.String())
	}
}
