package towers

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Audio clip length and soft-token ceilings per family.
const (
	Gemma4AudioMaxSoftTokens = 750
	Gemma4AudioMaxSeconds    = 30
	QwenASRMaxSeconds        = 600
	VoxtralMaxSeconds        = 600
)

// CloseAudioEncoder closes a lazily loaded audio encoder on model unload; overridable in tests.
var CloseAudioEncoder = func(field string, enc interface{ Close() error }) error { return enc.Close() }

// Gemma4Audio lazily loads and runs a Gemma 4 checkpoint's conformer audio tower.
type Gemma4Audio struct {
	Dir    string
	Tok    int
	Device string
	once   sync.Once
	enc    *audio.Gemma4AudioEncoder
	err    error
	Acc    embeddinggemma2.AudioAccelerator
}

func (g *Gemma4Audio) Family() string { return "Gemma 4 audio" }

func (g *Gemma4Audio) Prompt(m AudioModel, tm *chat.Template, system string, turns []chat.Turn, clip media.ImageRef) (VisionInput, error) {
	if !m.HasTemplate {
		return VisionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := LastUserTurn(turns)
	if idx < 0 {
		return VisionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	return g.ClipPrompt(m, tm, system, turns, idx, clip)
}

func (g *Gemma4Audio) Close() {
	if g.Acc != nil {
		_ = g.Acc.Close()
		g.Acc = nil
	}
	g.enc = nil
}

// Encoder lazily loads the Gemma 4 audio encoder and optional Metal accelerator.
func (g *Gemma4Audio) Encoder(name string) (*audio.Gemma4AudioEncoder, error) {
	g.once.Do(func() {
		t0 := time.Now()
		g.enc, g.err = audio.LoadGemma4AudioEncoder(g.Dir)
		if g.err == nil {
			fmt.Fprintf(os.Stderr, "audio: loaded the Gemma 4 audio tower for %q in %s\n", name, time.Since(t0).Round(time.Millisecond))
			if g.Device != "" {
				acc, err := embeddinggemma2.NewAudioAccelerator(g.Device, g.enc)
				if err != nil {
					fmt.Fprintf(os.Stderr, "audio: the %s audio tower declined (%v); the tower runs on the CPU\n", g.Device, err)
				} else {
					g.Acc = acc
				}
			}
		}
	})
	return g.enc, g.err
}

func (g *Gemma4Audio) forward(enc *audio.Gemma4AudioEncoder, mel []float32, T int) ([]float32, error) {
	if g.Acc == nil {
		return enc.Forward(mel, T)
	}
	h, n, err := enc.Subsample(mel, T)
	if err != nil {
		return nil, err
	}
	hb, err := g.Acc.Blocks(h, n)
	if err != nil {
		return nil, err
	}
	return enc.FinishBlocks(hb, n)
}

// ClipPrompt decodes WAV audio, extracts Gemma 4 mel features, and splices the <|audio|> block into turns[idx].
func (g *Gemma4Audio) ClipPrompt(m AudioModel, tm *chat.Template, system string, turns []chat.Turn, idx int, clip media.ImageRef) (VisionInput, error) {
	wav, err := multimodal.ParseWAV(clip.Data)
	if err != nil {
		return VisionInput{}, err
	}
	if n := wav.Samples(); n > Gemma4AudioMaxSeconds*audio.Gemma4SampleRate {
		return VisionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s the model reads", float64(n)/audio.Gemma4SampleRate, Gemma4AudioMaxSeconds)
	}
	samples := wav.Decode()
	mel, T, err := audio.Gemma4Features(samples)
	if err != nil {
		return VisionInput{}, fmt.Errorf("audio features: %w", err)
	}
	n := audio.Gemma4SoftTokens(T)
	if n > Gemma4AudioMaxSoftTokens {
		return VisionInput{}, fmt.Errorf("the audio clip is %.1f s, over the 30 s (%d soft tokens) the model reads", float64(len(samples))/16000, Gemma4AudioMaxSoftTokens)
	}
	if n <= 0 {
		return VisionInput{}, fmt.Errorf("the audio clip is too short (%d samples)", len(samples))
	}
	hiddenDim := m.HiddenDim
	features := func() ([]float32, error) {
		enc, err := g.Encoder(m.Name)
		if err != nil {
			return nil, fmt.Errorf("gemma4 audio tower: %w", err)
		}
		feats, err := g.forward(enc, mel, T)
		if err != nil {
			return nil, fmt.Errorf("gemma4 audio tower: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("gemma4 audio tower emitted %d values, want %d x %d", len(feats), n, hiddenDim)
		}
		return feats, nil
	}
	block := multimodal.Gemma4AudioBlock(n)
	turns[idx].Content = block + turns[idx].Content
	ids, err := EncodeVisionSegments(m.TK, tm, system, turns, block)
	if err != nil {
		return VisionInput{}, fmt.Errorf("encode: %w", err)
	}
	pos, ln := multimodal.FindImageRun(ids, g.Tok)
	if ln != n {
		return VisionInput{}, fmt.Errorf("audio placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", ln, n)
	}
	return VisionInput{IDs: ids, Features: features, ImgHash: multimodal.HashImageBytes(clip.Data), ImgPos: pos, ImgLen: ln, Gemma4: true}, nil
}

// QwenASRAudio lazily loads and runs a Qwen3-ASR audio encoder and projector.
type QwenASRAudio struct {
	Dir  string
	once sync.Once
	Enc  *audio.QwenASREncoder
	err  error
}

func (q *QwenASRAudio) Family() string { return "Qwen3-ASR audio" }

func (q *QwenASRAudio) Prompt(m AudioModel, _ *chat.Template, system string, _ []chat.Turn, clip media.ImageRef) (VisionInput, error) {
	return q.ClipPrompt(m, system, clip)
}

func (q *QwenASRAudio) Close() {
	if q.Enc != nil {
		_ = CloseAudioEncoder("qwenASR", q.Enc)
		q.Enc = nil
	}
}

// Encoder lazily loads the Qwen3-ASR audio encoder.
func (q *QwenASRAudio) Encoder() (*audio.QwenASREncoder, error) {
	q.once.Do(func() { q.Enc, q.err = audio.LoadQwenASREncoder(q.Dir) })
	return q.Enc, q.err
}

// ClipPrompt decodes WAV audio and builds a Qwen3-ASR transcription prompt.
func (q *QwenASRAudio) ClipPrompt(m AudioModel, system string, clip media.ImageRef) (VisionInput, error) {
	wav, err := multimodal.ParseWAV(clip.Data)
	if err != nil {
		return VisionInput{}, err
	}
	if n := wav.Samples(); n > QwenASRMaxSeconds*audio.WhisperSampleRate {
		return VisionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s one request may carry", float64(n)/audio.WhisperSampleRate, QwenASRMaxSeconds)
	}
	samples := wav.Decode()
	f, err := audio.QwenASRFeatures(samples)
	if err != nil {
		return VisionInput{}, fmt.Errorf("audio features: %w", err)
	}
	n := audio.QwenASRTokenCount(f.Valid)
	hiddenDim := m.HiddenDim
	features := func() ([]float32, error) {
		enc, err := q.Encoder()
		if err != nil {
			return nil, fmt.Errorf("qwen3-asr audio encoder: %w", err)
		}
		emb, got, err := enc.Forward(f.Data, f.T, f.Valid)
		if err != nil {
			return nil, fmt.Errorf("qwen3-asr audio encoder: %w", err)
		}
		if got != n || len(emb) != n*hiddenDim {
			return nil, fmt.Errorf("qwen3-asr audio encoder emitted %d tokens (%d values), want %d x %d", got, len(emb), n, hiddenDim)
		}
		return emb, nil
	}
	ids, pos, err := multimodal.QwenASRPromptOpenLanguage(m.TK, system, n)
	if err != nil {
		return VisionInput{}, fmt.Errorf("encode: %w", err)
	}
	return VisionInput{IDs: ids, Features: features, ImgHash: multimodal.HashImageBytes(clip.Data), ImgPos: pos, ImgLen: n, ASR: true, ReplyPrefix: multimodal.QwenASRLanguageWord}, nil
}

// VoxtralAudio lazily loads and runs Voxtral Mini's audio encoder and projector.
type VoxtralAudio struct {
	Dir     string
	Backend string
	once    sync.Once
	Device  audio.WhisperDevice
	Enc     *audio.VoxtralAudio
	err     error
}

func (v *VoxtralAudio) Family() string { return "Voxtral audio" }

func (v *VoxtralAudio) Prompt(m AudioModel, _ *chat.Template, _ string, _ []chat.Turn, clip media.ImageRef) (VisionInput, error) {
	return v.ClipPrompt(m, clip)
}

func (v *VoxtralAudio) Close() {
	if v.Device != nil {
		_ = v.Device.Close()
		v.Device = nil
	}
	if v.Enc != nil {
		_ = CloseAudioEncoder("voxtral", v.Enc)
		v.Enc = nil
	}
}

// Encoder lazily loads the Voxtral audio encoder and attaches any available Whisper GPU device.
func (v *VoxtralAudio) Encoder(name string) (*audio.VoxtralAudio, error) {
	v.once.Do(func() {
		v.Enc, v.err = audio.LoadVoxtralAudio(v.Dir)
		if v.err == nil {
			var where string
			v.Device, where = towerfit.AttachWhisperDevice(v.Enc.Enc, v.Backend, os.Stderr)
			fmt.Fprintf(os.Stderr, "audio: loaded the Voxtral audio tower for %q (encoder on %s)\n", name, where)
		}
	})
	return v.Enc, v.err
}

// ClipPrompt decodes WAV audio and builds a Voxtral transcription prompt.
func (v *VoxtralAudio) ClipPrompt(m AudioModel, clip media.ImageRef) (VisionInput, error) {
	wav, err := multimodal.ParseWAV(clip.Data)
	if err != nil {
		return VisionInput{}, err
	}
	if n := wav.Samples(); n > VoxtralMaxSeconds*audio.WhisperSampleRate {
		return VisionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s one request may carry", float64(n)/audio.WhisperSampleRate, VoxtralMaxSeconds)
	}
	samples := wav.Decode()
	n := multimodal.VoxtralAudioTokens(len(samples))
	hiddenDim := m.HiddenDim
	features := func() ([]float32, error) {
		wins, err := audio.WhisperFeaturesWindows(samples, 128)
		if err != nil {
			return nil, fmt.Errorf("audio features: %w", err)
		}
		enc, err := v.Encoder(m.Name)
		if err != nil {
			return nil, fmt.Errorf("voxtral audio encoder: %w", err)
		}
		emb, err := enc.Embed(wins)
		if err != nil {
			return nil, fmt.Errorf("voxtral audio encoder: %w", err)
		}
		if len(emb) != n*hiddenDim {
			return nil, fmt.Errorf("voxtral audio encoder emitted %d values, want %d x %d", len(emb), n, hiddenDim)
		}
		return emb, nil
	}
	ids, pos, err := multimodal.VoxtralPrompt(m.TK, n, "")
	if err != nil {
		return VisionInput{}, fmt.Errorf("encode: %w", err)
	}
	return VisionInput{IDs: ids, Features: features, ImgHash: multimodal.HashImageBytes(clip.Data), ImgPos: pos, ImgLen: n, ASR: true}, nil
}
