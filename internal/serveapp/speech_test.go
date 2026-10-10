package serveapp

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/whisper"
)

// G-S14i1 of docs/tasks/task-multimodal-support-2026-10.md: the shapes of POST /v1/audio/transcriptions and its refusals, on a stub engine.

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
	return &server{models: map[string]*loadedModel{}, speech: map[string]*speechModel{"whisper-small": {name: "whisper-small", eng: st}}}, st
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
