package serveapp

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/internal/whisper"
	"github.com/townsendmerino/goinfer/multimodal"
)

// A Whisper model is served as a speech model (docs/tasks/task-multimodal-support-2026-10.md, G-S14i): it is not a decoder.Model, so it lives in server.speech and not in server.models, which everything
// (sessions, KV slots, templates, adapters, the resident path) assumes holds decoders. It answers POST /v1/audio/transcriptions and is listed by GET /v1/models.

// speechEngine is what the route needs of a Whisper model; *whisper.Transcriber is the real one, a stub stands in for the handler tests.
type speechEngine interface {
	TranscribeWith(samples []float32, o whisper.TranscribeOptions) (*whisper.LongResult, error)
	Text(ids []int) string
	HasLanguage(code string) bool
}

type speechModel struct {
	name, dir string
	eng       speechEngine
	mu        sync.Mutex // one transcription at a time: a window is a whole encoder pass, and two at once only slow each other
}

// speechMaxSeconds is the longest clip a request may carry: a cost bound (each 30 s window is an encoder pass and a decode), not the model's limit; a longer clip is refused, never cut.
const speechMaxSeconds = 600

// isWhisperDir reports whether path is a directory holding an OpenAI Whisper checkpoint in Hugging Face's layout (config.json model_type "whisper").
func isWhisperDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir() && visionModelType(path) == "whisper"
}

// loadSpeech loads a Whisper directory as a speech model. A directory missing the tokenizer or generation_config.json is refused here, by name, and not at the first request.
func loadSpeech(spec modelSpec, cfg config) (*speechModel, error) {
	for _, f := range []string{"tokenizer.json", "generation_config.json", "model.safetensors"} {
		if _, err := os.Stat(filepath.Join(spec.path, f)); err != nil {
			if f == "model.safetensors" {
				if _, e2 := os.Stat(filepath.Join(spec.path, "model.safetensors.index.json")); e2 == nil {
					continue
				}
			}
			return nil, fmt.Errorf("--model %q: a Whisper directory needs %s (%v)", spec.path, f, err)
		}
	}
	t0 := time.Now()
	tr, err := whisper.LoadTranscriber(spec.path)
	if err != nil {
		return nil, fmt.Errorf("--model %q: %w", spec.path, err)
	}
	name := spec.name
	if name == "" {
		if len(cfg.models) == 1 && cfg.name != "" {
			name = cfg.name
		} else {
			name = filepath.Base(strings.TrimRight(spec.path, "/"))
		}
	}
	fmt.Fprintf(os.Stderr, "loaded Whisper %q (%d mels, d_model %d, %d+%d layers, %d languages; float32, CPU; POST /v1/audio/transcriptions) in %s\n",
		name, tr.Mels, tr.Dec.Cfg.D, tr.Enc.Cfg.Layers, tr.Dec.Cfg.Layers, len(tr.Gen.LangToID), time.Since(t0).Round(time.Millisecond))
	return &speechModel{name: name, dir: spec.path, eng: tr}, nil
}

// speechByName is an exact lookup under the read lock.
func (s *server) speechByName(name string) *speechModel {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	return s.speech[name]
}

// speechFor picks the speech model a request names: the named one, or the only one when the field is absent.
func (s *server) speechFor(name string) (*speechModel, bool) {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	if name != "" {
		sm, ok := s.speech[name]
		return sm, ok
	}
	if len(s.speech) == 1 {
		for _, sm := range s.speech {
			return sm, true
		}
	}
	return nil, false
}

// transcriptionPolicy is OpenAI's default decode policy (the Whisper reference's): the temperature ladder, the three thresholds, and conditioning on the earlier text.
func transcriptionPolicy(temperature float64) whisper.TranscribeOptions {
	o := whisper.TranscribeOptions{ConditionOnPrev: true}
	one := func(v float64) *float64 { return &v }
	o.CompressionRatioThreshold, o.LogprobThreshold, o.NoSpeechThreshold = one(2.4), one(-1.0), one(0.6)
	if temperature > 0 {
		o.Temperatures = []float64{temperature}
	} else {
		o.Temperatures = []float64{0, 0.2, 0.4, 0.6, 0.8, 1.0}
	}
	return o
}

type transcriptionSegment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}

// srtTime and vttTime format seconds as HH:MM:SS,mmm and HH:MM:SS.mmm (hours always present).
func clock(sec float64, sep string) string {
	ms := int64(sec*1000 + 0.5)
	h, ms := ms/3600000, ms%3600000
	m, ms := ms/60000, ms%60000
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", h, m, ms/1000, sep, ms%1000)
}

// handleTranscriptions is POST /v1/audio/transcriptions (OpenAI's route, multipart/form-data); the accepted fields and what is refused are in the task doc's G-S14i.
func (s *server) handleTranscriptions(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeErr(w, http.StatusBadRequest, "the request must be multipart/form-data with a file part: "+err.Error())
		return
	}
	form := r.MultipartForm
	val := func(k string) string {
		if v := form.Value[k]; len(v) > 0 {
			return strings.TrimSpace(v[0])
		}
		return ""
	}
	for _, f := range []string{"prompt", "stream"} {
		if val(f) != "" && val(f) != "false" {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("%s is not supported by this server", f))
			return
		}
	}
	for _, g := range append(form.Value["timestamp_granularities[]"], form.Value["timestamp_granularities"]...) {
		if g != "segment" {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("timestamp_granularities %q is not supported: only \"segment\"", g))
			return
		}
	}
	format := val("response_format")
	if format == "" {
		format = "json"
	}
	switch format {
	case "json", "text", "srt", "vtt", "verbose_json":
	default:
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("response_format %q is not supported: json, text, srt, vtt or verbose_json", format))
		return
	}
	sm, ok := s.speechFor(val("model"))
	if !ok {
		if val("model") == "" {
			writeErr(w, http.StatusBadRequest, "model is required: more than one speech model is served, or none (served: "+strings.Join(s.servedNames(), ", ")+")")
		} else {
			s.modelNotFound(w, val("model"))
		}
		return
	}
	language := ""
	if l := val("language"); l != "" {
		if language = whisper.LanguageCode(l); language == "" || !sm.eng.HasLanguage(language) {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("language %q is not one this model knows", l))
			return
		}
	}
	temperature := 0.0
	if t := val("temperature"); t != "" {
		v, err := strconv.ParseFloat(t, 64)
		if err != nil || v < 0 || v > 1 {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("temperature %q must be a number from 0 to 1", t))
			return
		}
		temperature = v
	}
	files := form.File["file"]
	if len(files) == 0 {
		writeErr(w, http.StatusBadRequest, "file is required")
		return
	}
	fh, err := files[0].Open()
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading file: "+err.Error())
		return
	}
	raw, err := io.ReadAll(fh)
	fh.Close()
	if err != nil {
		writeErr(w, http.StatusBadRequest, "reading file: "+err.Error())
		return
	}
	samples, err := multimodal.DecodeWAVAnyRate(raw)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "file: only 16-bit PCM WAV is supported ("+err.Error()+")")
		return
	}
	if len(samples) > speechMaxSeconds*audio.WhisperSampleRate {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("the audio clip is %.1f s, over the %d s one request may carry", float64(len(samples))/audio.WhisperSampleRate, speechMaxSeconds))
		return
	}
	opts := transcriptionPolicy(temperature)
	opts.Language, opts.Task, opts.Seed = language, "transcribe", uint64(time.Now().UnixNano())
	sm.mu.Lock()
	res, err := sm.eng.TranscribeWith(samples, opts)
	sm.mu.Unlock()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "transcription: "+err.Error())
		return
	}
	segs := make([]transcriptionSegment, len(res.Segments))
	for i, sg := range res.Segments {
		segs[i] = transcriptionSegment{ID: i, Seek: sg.Seek, Start: sg.Start, End: sg.End, Text: strings.TrimSpace(sm.eng.Text(sg.Tokens)), Tokens: sg.Tokens,
			Temperature: sg.Temperature, AvgLogprob: sg.AvgLogprob, CompressionRatio: sg.CompressionRatio, NoSpeechProb: sg.NoSpeechProb}
	}
	text := strings.TrimSpace(sm.eng.Text(res.Tokens))
	switch format {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, text+"\n")
	case "srt", "vtt":
		var b strings.Builder
		sep := ","
		if format == "vtt" {
			b.WriteString("WEBVTT\n\n")
			sep = "."
		}
		for i, sg := range segs {
			if format == "srt" {
				fmt.Fprintf(&b, "%d\n", i+1)
			}
			fmt.Fprintf(&b, "%s --> %s\n%s\n\n", clock(sg.Start, sep), clock(sg.End, sep), strings.ReplaceAll(sg.Text, "-->", "->"))
		}
		if format == "vtt" {
			w.Header().Set("Content-Type", "text/vtt; charset=utf-8")
		} else {
			w.Header().Set("Content-Type", "application/x-subrip; charset=utf-8")
		}
		io.WriteString(w, b.String())
	case "verbose_json":
		writeJSON(w, http.StatusOK, map[string]any{"task": "transcribe", "language": whisper.LanguageName(res.Language), "duration": float64(len(samples)) / audio.WhisperSampleRate, "text": text, "segments": segs})
	default:
		writeJSON(w, http.StatusOK, map[string]string{"text": text})
	}
}
