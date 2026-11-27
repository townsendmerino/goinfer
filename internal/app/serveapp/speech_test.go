package serveapp

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/whisper"
)

// G-S14i1: the shapes of POST /v1/audio/transcriptions and its refusals, on a stub engine.

type stubSpeech struct {
	got  whisper.TranscribeOptions
	n    int // samples seen
	segs []whisper.Segment
}

var stubWords = map[int]string{1: " Hello", 2: " world", 3: " -->", 4: " again"}

func (s *stubSpeech) TranscribeWith(x []float32, o whisper.TranscribeOptions) (*whisper.LongResult, error) {
	s.got, s.n = o, len(x)
	r := &whisper.LongResult{Language: "<|en|>"}
	for _, sg := range s.segs {
		r.Segments = append(r.Segments, sg)
		r.Tokens = append(r.Tokens, sg.Tokens...)
	}
	return r, nil
}
func (s *stubSpeech) Text(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(stubWords[id])
	}
	return b.String()
}
func (s *stubSpeech) HasLanguage(c string) bool { return c == "en" || c == "fr" }

func speechServer(t *testing.T) (*server, *stubSpeech) {
	t.Helper()
	st := &stubSpeech{segs: []whisper.Segment{
		{Start: 0.5, End: 3.21, Tokens: []int{1, 2}, Seek: 0, Temperature: 0, AvgLogprob: -0.25, CompressionRatio: 0.9, NoSpeechProb: 0.01},
		{Start: 3661.5, End: 3662.0044, Tokens: []int{3, 4}, Seek: 366150, Temperature: 0.2, AvgLogprob: -0.5, CompressionRatio: 1.1, NoSpeechProb: 0.02}}}
	return &server{modelRegistry: modelRegistry{models: map[string]*loadedModel{}, speech: map[string]*speechModel{"whisper-small": {name: "whisper-small", eng: st}}}}, st
}

func post(t *testing.T, s *server, fields map[string]string, repeated map[string][]string, wav []byte) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		mw.WriteField(k, v)
	}
	for k, vs := range repeated {
		for _, v := range vs {
			mw.WriteField(k, v)
		}
	}
	if wav != nil {
		fw, _ := mw.CreateFormFile("file", "a.wav")
		fw.Write(wav)
	}
	mw.Close()
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.handleTranscriptions(rec, req)
	return rec
}

func TestTranscriptions_formats(t *testing.T) {
	s, st := speechServer(t)
	wav := wavBytes(16000, 16000) // 1 s
	cases := map[string]string{
		"json":    `{"text":"Hello world --> again"}`,
		"text":    "Hello world --> again\n",
		"srt":     "1\n00:00:00,500 --> 00:00:03,210\nHello world\n\n2\n01:01:01,500 --> 01:01:02,004\n--> again\n\n",
		"vtt":     "WEBVTT\n\n00:00:00.500 --> 00:00:03.210\nHello world\n\n01:01:01.500 --> 01:01:02.004\n-> again\n\n",
		"verbose": "",
	}
	_ = cases
	for _, f := range []string{"json", "text"} {
		rec := post(t, s, map[string]string{"model": "whisper-small", "response_format": f}, nil, wav)
		if rec.Code != 200 {
			t.Fatalf("%s: %d %s", f, rec.Code, rec.Body.String())
		}
		got := rec.Body.String()
		if f == "json" {
			var j map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &j); err != nil || len(j) != 1 {
				t.Fatalf("json: %q (%v)", got, err)
			}
			got = `{"text":"` + j["text"] + `"}`
		}
		if got != cases[f] {
			t.Errorf("%s: got %q, want %q", f, got, cases[f])
		}
	}
	// srt: the third segment's text starts with "-->" on the stub (token 3), which the writer must not emit as an arrow; vtt likewise
	for _, f := range []string{"srt", "vtt"} {
		rec := post(t, s, map[string]string{"response_format": f}, nil, wav) // no model: the only speech model is the default
		if rec.Code != 200 {
			t.Fatalf("%s: %d %s", f, rec.Code, rec.Body.String())
		}
		want := cases[f]
		if f == "srt" {
			want = strings.Replace(want, "--> again", "-> again", 1)
		}
		if rec.Body.String() != want {
			t.Errorf("%s:\n got %q\nwant %q", f, rec.Body.String(), want)
		}
	}
	// verbose_json: the fields, the language as a name, the duration, and the texts stripped
	rec := post(t, s, map[string]string{"response_format": "verbose_json"}, nil, wav)
	var v struct {
		Task, Language, Text string
		Duration             float64
		Segments             []map[string]any
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatal(err)
	}
	if v.Task != "transcribe" || v.Language != "english" || v.Duration != 1 || v.Text != "Hello world --> again" || len(v.Segments) != 2 {
		t.Errorf("verbose_json: %+v", v)
	}
	for _, k := range []string{"id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"} {
		if _, ok := v.Segments[1][k]; !ok {
			t.Errorf("segment lacks %q", k)
		}
	}
	if v.Segments[0]["text"] != "Hello world" || v.Segments[1]["temperature"] != 0.2 || v.Segments[1]["seek"] != 366150.0 {
		t.Errorf("segments: %v", v.Segments)
	}
	// the options the route gave the engine: OpenAI's policy, the ladder, language and task
	if len(st.got.Temperatures) != 6 || *st.got.LogprobThreshold != -1.0 || *st.got.CompressionRatioThreshold != 2.4 || *st.got.NoSpeechThreshold != 0.6 || !st.got.ConditionOnPrev || st.got.Task != "transcribe" || st.got.Language != "" {
		t.Errorf("default options: %+v", st.got)
	}
	post(t, s, map[string]string{"temperature": "0.4", "language": "French"}, nil, wav)
	if len(st.got.Temperatures) != 1 || st.got.Temperatures[0] != 0.4 || st.got.Language != "fr" || st.n != 16000 {
		t.Errorf("temperature 0.4 and language French: %+v (%d samples)", st.got, st.n)
	}
}

func TestTranscriptions_refusals(t *testing.T) {
	s, _ := speechServer(t)
	wav := wavBytes(16000, 16000)
	for name, c := range map[string]struct {
		fields   map[string]string
		repeated map[string][]string
		wav      []byte
		code     int
		msg      string
	}{
		"no file":               {map[string]string{}, nil, nil, 400, "file is required"},
		"not a wav":             {map[string]string{}, nil, []byte("ID3 not a wav"), 400, "16-bit PCM WAV"},
		"too long":              {map[string]string{}, nil, wavBytes((speechMaxSeconds+5)*16000, 16000), 400, "over the 600 s"},
		"unknown language":      {map[string]string{"language": "klingon"}, nil, wav, 400, "language"},
		"language not in model": {map[string]string{"language": "de"}, nil, wav, 400, "language"},
		"unknown model":         {map[string]string{"model": "nope"}, nil, wav, 404, "not found"},
		"prompt":                {map[string]string{"prompt": "hello"}, nil, wav, 400, "prompt"},
		"stream":                {map[string]string{"stream": "true"}, nil, wav, 400, "stream"},
		"word granularity":      {map[string]string{}, map[string][]string{"timestamp_granularities[]": {"word"}}, wav, 400, "timestamp_granularities"},
		"bad response format":   {map[string]string{"response_format": "docx"}, nil, wav, 400, "response_format"},
		"bad temperature":       {map[string]string{"temperature": "2"}, nil, wav, 400, "temperature"},
	} {
		rec := post(t, s, c.fields, c.repeated, c.wav)
		if rec.Code != c.code || !strings.Contains(rec.Body.String(), c.msg) {
			t.Errorf("%s: %d %q, want %d containing %q", name, rec.Code, rec.Body.String(), c.code, c.msg)
		}
	}
	// two speech models and no model field: the request must say which
	s.speech["other"] = &speechModel{name: "other", eng: &stubSpeech{}}
	if rec := post(t, s, map[string]string{}, nil, wav); rec.Code != 400 || !strings.Contains(rec.Body.String(), "model is required") {
		t.Errorf("two speech models, no model: %d %q", rec.Code, rec.Body.String())
	}
	// a non-multipart body
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", strings.NewReader(`{"model":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	s.handleTranscriptions(rec, req)
	if rec.Code != 400 {
		t.Errorf("a JSON body: %d", rec.Code)
	}
}

func TestSpeechModels_listedInModels(t *testing.T) {
	s, _ := speechServer(t)
	rec := httptest.NewRecorder()
	s.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var out struct{ Data []map[string]any }
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Data) != 1 || out.Data[0]["id"] != "whisper-small" {
		t.Fatalf("models: %v", out.Data)
	}
	if caps, _ := out.Data[0]["capabilities"].([]any); len(caps) != 1 || caps[0] != "audio.transcriptions" {
		t.Errorf("capabilities: %v", out.Data[0]["capabilities"])
	}
	if _, has := out.Data[0]["decode_path"]; has {
		t.Errorf("a speech model must not publish decoder paths")
	}
}

func TestClock(t *testing.T) {
	for in, want := range map[float64]string{0: "00:00:00,000", 0.5: "00:00:00,500", 59.9996: "00:01:00,000", 3661.5: "01:01:01,500", 36000: "10:00:00,000"} {
		if got := clock(in, ","); got != want {
			t.Errorf("clock(%v) = %q, want %q", in, got, want)
		}
	}
}

// blockSpeech is an engine whose transcription runs until release is closed or its context ends. calls counts transcriptions
// started; started receives one value per start.
type blockSpeech struct {
	release chan struct{}
	started chan struct{}
	calls   atomic.Int32
	closed  atomic.Bool
}

func newBlockSpeech() *blockSpeech {
	return &blockSpeech{release: make(chan struct{}), started: make(chan struct{}, 16)}
}

func (b *blockSpeech) TranscribeWithContext(ctx context.Context, x []float32, o whisper.TranscribeOptions) (*whisper.LongResult, error) {
	b.calls.Add(1)
	b.started <- struct{}{}
	select {
	case <-b.release:
		return &whisper.LongResult{}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
func (b *blockSpeech) TranscribeWith(x []float32, o whisper.TranscribeOptions) (*whisper.LongResult, error) {
	return b.TranscribeWithContext(context.Background(), x, o)
}
func (b *blockSpeech) Text([]int) string       { return "" }
func (b *blockSpeech) HasLanguage(string) bool { return true }
func (b *blockSpeech) Close() error            { b.closed.Store(true); return nil }

// speechReq builds a one-second transcription request carrying ctx.
func speechReq(ctx context.Context) *http.Request {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, _ := mw.CreateFormFile("file", "a.wav")
	fw.Write(wavBytes(16000, 16000))
	mw.Close()
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req
}

// serveAsync runs one transcription request on its own goroutine; the recorder arrives on the returned channel when it returns.
func serveAsync(s *server, req *http.Request) <-chan *httptest.ResponseRecorder {
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		s.handleTranscriptions(rec, req)
		done <- rec
	}()
	return done
}

// waitAdmitted waits until n requests hold an admission slot on sm, failing the test if 10s pass first.
func waitAdmitted(t *testing.T, sm *speechModel, n int) {
	t.Helper()
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
		if sm.slots.admitted() >= n {
			return
		}
	}
	t.Fatalf("%d requests admitted after 10s, expected %d", sm.slots.admitted(), n)
}

func blockSpeechServer() (*server, *speechModel, *blockSpeech) {
	eng := newBlockSpeech()
	sm := &speechModel{name: "w", eng: eng}
	return &server{modelRegistry: modelRegistry{models: map[string]*loadedModel{}, speech: map[string]*speechModel{"w": sm}}}, sm, eng
}

// One transcription runs and speechMaxWaiting wait; one more is refused at once with 429 and Retry-After, and the queued ones
// all complete once the running one does.
func TestTranscriptions_queueIsBounded(t *testing.T) {
	s, sm, eng := blockSpeechServer()
	var queued []<-chan *httptest.ResponseRecorder
	queued = append(queued, serveAsync(s, speechReq(context.Background())))
	<-eng.started
	for range speechMaxWaiting {
		queued = append(queued, serveAsync(s, speechReq(context.Background())))
	}
	waitAdmitted(t, sm, 1+speechMaxWaiting)
	select {
	case rec := <-serveAsync(s, speechReq(context.Background())):
		if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") != "1" {
			t.Errorf("a request past the queue: %d, Retry-After %q; want 429 and Retry-After 1", rec.Code, rec.Header().Get("Retry-After"))
		}
	case <-time.After(2 * time.Second):
		t.Fatal("a request past the queue waited instead of being refused with 429")
	}
	close(eng.release)
	for i, d := range queued {
		if rec := <-d; rec.Code != http.StatusOK {
			t.Errorf("queued request %d: %d %s, want 200", i, rec.Code, rec.Body.String())
		}
	}
	if got := eng.calls.Load(); got != int32(1+speechMaxWaiting) {
		t.Errorf("transcriptions run = %d, want %d", got, 1+speechMaxWaiting)
	}
}

// A waiting request whose client goes away leaves the queue at once, without running; a running one whose client goes away has
// its context cancelled through TranscribeWithContext.
func TestTranscriptions_cancelledRequestLeaves(t *testing.T) {
	s, sm, eng := blockSpeechServer()
	runCtx, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()
	running := serveAsync(s, speechReq(runCtx))
	<-eng.started
	waitCtx, cancelWait := context.WithCancel(context.Background())
	waiting := serveAsync(s, speechReq(waitCtx))
	waitAdmitted(t, sm, 2)
	cancelWait()
	select {
	case rec := <-waiting:
		if rec.Code != statusCancelled {
			t.Errorf("a cancelled waiter: %d %s, want %d", rec.Code, rec.Body.String(), statusCancelled)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("a cancelled waiter stayed queued behind the running transcription")
	}
	if got := sm.slots.admitted(); got != 1 {
		t.Errorf("admitted after the waiter left = %d, want 1", got)
	}
	cancelRun()
	select {
	case rec := <-running:
		if rec.Code != statusCancelled {
			t.Errorf("a cancelled running request: %d %s, want %d", rec.Code, rec.Body.String(), statusCancelled)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelling a running request's context did not end its transcription")
	}
	if got := eng.calls.Load(); got != 1 {
		t.Errorf("transcriptions run = %d, want 1 (the waiter must not run)", got)
	}
}

// handlerCloser.Close waits for the running transcription before closing the engine, and a waiter that gets the slot after Close
// is refused with 503 instead of running on the closed engine.
func TestTranscriptions_closeWaitsAndLaterWaitersGet503(t *testing.T) {
	s, sm, eng := blockSpeechServer()
	running := serveAsync(s, speechReq(context.Background()))
	<-eng.started
	waiting := serveAsync(s, speechReq(context.Background()))
	waitAdmitted(t, sm, 2)

	_, stop := context.WithCancelCause(context.Background())
	closed := make(chan error, 1)
	go func() { closed <- (&handlerCloser{srv: s, stop: stop}).Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned while a transcription was running")
	case <-time.After(50 * time.Millisecond):
	}
	if eng.closed.Load() {
		t.Fatal("the engine was closed under a running transcription")
	}
	close(eng.release)
	if rec := <-running; rec.Code != http.StatusOK {
		t.Errorf("the running request: %d %s, want 200", rec.Code, rec.Body.String())
	}
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !eng.closed.Load() {
		t.Error("Close did not close the engine")
	}
	if rec := <-waiting; rec.Code != http.StatusOK && rec.Code != http.StatusServiceUnavailable {
		t.Errorf("the waiter: %d %s, want 200 (it ran before Close) or 503", rec.Code, rec.Body.String())
	}

	// Deterministic: the slot is held here, Close runs to completion, and only then is the waiter let through.
	s2, sm2, eng2 := blockSpeechServer()
	if err := sm2.slots.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	late := serveAsync(s2, speechReq(context.Background()))
	waitAdmitted(t, sm2, 1)
	closed2 := make(chan error, 1)
	go func() { closed2 <- (&handlerCloser{srv: s2, stop: stop}).Close() }()
	select {
	case err := <-closed2:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close waited on a request that had not started transcribing")
	}
	sm2.slots.release()
	select {
	case rec := <-late:
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("a waiter woken after Close: %d %s, want 503", rec.Code, rec.Body.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a waiter woken after Close did not return")
	}
	if got := eng2.calls.Load(); got != 0 {
		t.Errorf("transcriptions run on the closed engine = %d, want 0", got)
	}
}
