package embeddinggemma2

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/townsendmerino/aikit/audio"
)

// Audio (Phase A, docs/tasks/task-embeddinggemma2.md). EmbeddingGemma 2's audio tower is gemma4_audio, which aikit's
// audio package runs (Gemma4Features, then Gemma4AudioEncoder), so an audio clip's soft tokens come from aikit and
// replace their placeholder rows in the encoder's input, unscaled, as an image's do.

type audioTokens struct {
	Audio int `json:"audio_token_id"`
	BOA   int `json:"boa_token_id"`
	EOA   int `json:"eoa_token_index"` // the config's name for it (not eoa_token_id)
}

type audioTower struct {
	enc    *audio.Gemma4AudioEncoder
	tok    audioTokens
	accel  AudioAccelerator // nil: the blocks run on the CPU (audio_accel.go)
	device string           // AudioDevice's account when not plain CPU
}

// EnableAudio loads the checkpoint's audio tower (from the encoder's directory) so EmbedAudio works. Its blocks run on
// the encoder's accelerator when one of that name is registered for audio (RegisterAudioAccelerator), on the CPU
// otherwise (AudioDevice says which).
func (e *Encoder) EnableAudio() error {
	if e.dir == "" {
		return fmt.Errorf("embeddinggemma2: the encoder was not loaded from a directory")
	}
	raw, err := os.ReadFile(filepath.Join(e.dir, "config.json"))
	if err != nil {
		return fmt.Errorf("embeddinggemma2: %w", err)
	}
	var cfg struct {
		audioTokens
		Audio *json.RawMessage `json:"audio_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("embeddinggemma2: config.json: %w", err)
	}
	if cfg.Audio == nil || cfg.audioTokens.Audio <= 0 || cfg.BOA <= 0 || cfg.EOA <= 0 {
		return fmt.Errorf("embeddinggemma2: this checkpoint has no audio_config or audio token ids")
	}
	enc, err := audio.LoadGemma4AudioEncoder(e.dir)
	if err != nil {
		return fmt.Errorf("embeddinggemma2: audio tower: %w", err)
	}
	if enc.TextHiddenSize != e.m.cfg.Hidden {
		return fmt.Errorf("embeddinggemma2: the audio embedder emits %d wide, the encoder takes %d", enc.TextHiddenSize, e.m.cfg.Hidden)
	}
	e.aud = &audioTower{enc: enc, tok: cfg.audioTokens}
	e.bindAudioAccel()
	return nil
}

// AudioEnabled reports whether EnableAudio has loaded the tower.
func (e *Encoder) AudioEnabled() bool { return e.aud != nil }

// EmbedAudioTask embeds one audio input for a server: it loads the audio tower on first use (a server that never sees
// audio never pays for it), then EmbedAudio. prompt names a task prompt or is "" for none.
func (e *Encoder) EmbedAudioTask(samples []float32, text, prompt string) ([]float32, int, error) {
	visMu.Lock()
	if e.aud == nil {
		t0 := time.Now()
		if err := e.EnableAudio(); err != nil {
			visMu.Unlock()
			return nil, 0, err
		}
		if e.onTower != nil {
			e.onTower("audio", e.AudioDevice(), time.Since(t0))
		}
	}
	visMu.Unlock()
	return e.EmbedAudio(AudioInput{Samples: samples, Text: text, Prompt: prompt})
}

// AudioDevice says where the audio tower runs ("" before EnableAudio).
func (e *Encoder) AudioDevice() string {
	if e.aud == nil {
		return ""
	}
	if e.aud.device == "" {
		return "CPU"
	}
	return e.aud.device
}

// MaxAudioSeconds is the longest clip the tower reads (the reference extractor truncates at 30 s; here a longer clip is
// refused instead, so nothing is dropped silently).
const MaxAudioSeconds = 30

// AudioInput is one audio-bearing input: 16 kHz mono samples in [-1, 1], optional text after the audio, and the
// named prompt before it ("" for none).
type AudioInput struct {
	Samples []float32
	Text    string
	Prompt  string
}

// TokenizeAudio returns the ids for an audio input with n soft tokens: <bos>, the prompt's text, <|audio>, n audio
// tokens, <audio|>, the text, <eos> (probed through sentence-transformers, Phase A Gate 0), and the index of the
// first soft token.
func (e *Encoder) TokenizeAudio(in AudioInput, n int) ([]int, int, error) {
	if e.aud == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: audio is not enabled (EnableAudio)")
	}
	t := e.aud.tok
	return e.tokenizeMedia(in.Prompt, in.Text, t.BOA, t.Audio, t.EOA, n)
}

// tokenizeMedia is the layout an image or an audio clip takes: <bos>, the prompt's text, open, n placeholders, close,
// the text, <eos>, and the index of the first placeholder.
func (e *Encoder) tokenizeMedia(prompt, text string, open, ph, close, n int) ([]int, int, error) {
	p, err := e.PromptText(prompt)
	if err != nil {
		return nil, 0, err
	}
	ids := []int{e.bos}
	if p != "" {
		pi, err := e.tok.Encode(p, false)
		if err != nil {
			return nil, 0, err
		}
		ids = append(ids, pi...)
	}
	ids = append(ids, open)
	pos := len(ids)
	for range n {
		ids = append(ids, ph)
	}
	ids = append(ids, close)
	if text != "" {
		ti, err := e.tok.Encode(text, false)
		if err != nil {
			return nil, 0, err
		}
		ids = append(ids, ti...)
	}
	ids = append(ids, e.eos)
	if len(ids) > MaxTokens {
		return nil, 0, fmt.Errorf("input is %d tokens, over the model's %d-token context", len(ids), MaxTokens)
	}
	return ids, pos, nil
}

// AudioLogMel is the tower's input for 16 kHz samples: aikit's log-mel [T, 128] over the T valid frames.
func AudioLogMel(samples []float32) ([]float32, int, error) { return audio.Gemma4Features(samples) }

// AudioFeatures runs the log-mel and the tower on 16 kHz samples: the soft tokens [n, hidden] and n.
func (e *Encoder) AudioFeatures(samples []float32) ([]float32, int, error) {
	if len(samples) > MaxAudioSeconds*audio.Gemma4SampleRate {
		return nil, 0, fmt.Errorf("embeddinggemma2: audio is %.1f s, over the %d s the model reads", float64(len(samples))/audio.Gemma4SampleRate, MaxAudioSeconds)
	}
	mel, T, err := audio.Gemma4Features(samples)
	if err != nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: %w", err)
	}
	return e.AudioFeaturesFrom(mel, T)
}

// AudioFeaturesFrom is AudioFeatures from a log-mel already computed (a test feeds the reference extractor's own).
func (e *Encoder) AudioFeaturesFrom(mel []float32, T int) ([]float32, int, error) {
	if e.aud == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: audio is not enabled (EnableAudio)")
	}
	return e.audioFeaturesOn(e.aud.accel, mel, T)
}

// AudioTower is the loaded aikit tower (nil before EnableAudio), for per-stage tests.
func (e *Encoder) AudioTower() *audio.Gemma4AudioEncoder {
	if e.aud == nil {
		return nil
	}
	return e.aud.enc
}

// EmbedAudio is the sentence embedding of an audio input, unit length, and its token count.
func (e *Encoder) EmbedAudio(in AudioInput) ([]float32, int, error) {
	f, n, err := e.AudioFeatures(in.Samples)
	if err != nil {
		return nil, 0, err
	}
	return e.EmbedAudioFeatures(in, f, n)
}

// EmbedAudioFeatures is EmbedAudio from soft tokens already computed.
func (e *Encoder) EmbedAudioFeatures(in AudioInput, feats []float32, n int) ([]float32, int, error) {
	ids, pos, err := e.TokenizeAudio(in, n)
	if err != nil {
		return nil, 0, err
	}
	return e.embedSpliced(ids, pos, feats, n)
}

// embedSpliced embeds ids with n soft-token rows written over the placeholders at pos (unscaled), then pools.
func (e *Encoder) embedSpliced(ids []int, pos int, feats []float32, n int) ([]float32, int, error) {
	x, err := e.m.EmbedTokens(ids)
	if err != nil {
		return nil, 0, err
	}
	H := e.m.cfg.Hidden
	if len(feats) != n*H {
		return nil, 0, fmt.Errorf("embeddinggemma2: %d soft-token values for %d rows of %d", len(feats), n, H)
	}
	copy(x[pos*H:(pos+n)*H], feats)
	var h []float32
	if e.accel != nil {
		h, _, err = e.accel.ForwardEmbeds(x, len(ids), false)
	} else {
		h, _, err = e.m.forwardEmbeds(x, len(ids), false)
	}
	if err != nil {
		return nil, 0, err
	}
	return poolNormalize(h, len(ids), e.m.cfg.EmbeddingDim), len(ids), nil
}

// DecodeWAV reads a RIFF WAV of 16-bit PCM, mono, at 16 kHz into samples in [-1, 1] (s/32768). Any other format,
// channel count or rate is refused: the tower takes 16 kHz and nothing here resamples.
func DecodeWAV(data []byte) ([]float32, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("audio: not a RIFF/WAVE file")
	}
	var fmtOK bool
	for off := 12; off+8 <= len(data); {
		id, size := string(data[off:off+4]), int(binary.LittleEndian.Uint32(data[off+4:off+8]))
		body := off + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("audio: WAV fmt chunk of %d bytes", size)
			}
			f := data[body : body+size]
			format, ch := binary.LittleEndian.Uint16(f[0:2]), binary.LittleEndian.Uint16(f[2:4])
			rate, bits := binary.LittleEndian.Uint32(f[4:8]), binary.LittleEndian.Uint16(f[14:16])
			if format != 1 || bits != 16 || ch != 1 || rate != audio.Gemma4SampleRate {
				return nil, fmt.Errorf("audio: WAV is format %d, %d-bit, %d channel(s) at %d Hz; want 16-bit PCM mono at 16000 Hz (no resampling is done)", format, bits, ch, rate)
			}
			fmtOK = true
		case "data":
			if !fmtOK {
				return nil, fmt.Errorf("audio: WAV data before its fmt chunk")
			}
			n := size / 2
			out := make([]float32, n)
			for i := range n {
				out[i] = float32(int16(binary.LittleEndian.Uint16(data[body+2*i:]))) / 32768
			}
			return out, nil
		}
		off = body + size + size&1
	}
	return nil, fmt.Errorf("audio: WAV has no data chunk")
}
