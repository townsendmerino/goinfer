package serveapp

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// audioTower is one family's audio input: the clip-to-prompt builder and the encoder it loads on the first clip, so a
// server that never hears audio never pays for it. Each registry entry holds its own (the encoders are not shared
// between entries), and an entry holds at most one: a request carrying a clip carries nothing else (visionPromptN).
type audioTower interface {
	// family names the tower in errors and keys the per-image feature cache (withFeatureCache); it must stay exact.
	family() string
	// prompt builds the multimodal prompt for clip. tm is the request's template; the families that lay the prompt
	// out themselves ignore it and the turns.
	prompt(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, clip imageRef) (visionInput, error)
	// close releases the encoder and any device it was placed on, if one was loaded. Idempotent.
	close()
}

// closeAudioEncoder closes one of an entry's lazily loaded audio encoders at unload. A variable so a test can count the
// calls.
var closeAudioEncoder = func(field string, enc interface{ Close() error }) error { return enc.Close() }

// gemma4Audio is a Gemma 4 checkpoint's audio tower (one with an audio_config), beside its vision tower.
type gemma4Audio struct {
	dir string
	tok int // <|audio|>
	// device is "metal" when the conformer blocks run on Metal (EmbeddingGemma 2's accelerator), "" for the CPU; acc is
	// that accelerator once built, nil if it declined (the CPU then, said once).
	device string
	once   sync.Once
	enc    *audio.Gemma4AudioEncoder
	err    error
	acc    embeddinggemma2.AudioAccelerator
}

func (g *gemma4Audio) family() string { return "Gemma 4 audio" }

func (g *gemma4Audio) prompt(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, clip imageRef) (visionInput, error) {
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	return g.clipPrompt(lm, tm, system, turns, idx, clip)
}

func (g *gemma4Audio) close() {
	if g.acc != nil {
		_ = g.acc.Close()
		g.acc = nil
	}
	g.enc = nil
}

// encoder is the audio tower, loaded on the first clip (aikit's gemma4_audio, CPU, float32). name is the entry's, for
// the load line.
func (g *gemma4Audio) encoder(name string) (*audio.Gemma4AudioEncoder, error) {
	g.once.Do(func() {
		t0 := time.Now()
		g.enc, g.err = audio.LoadGemma4AudioEncoder(g.dir)
		if g.err == nil {
			fmt.Fprintf(os.Stderr, "audio: loaded the Gemma 4 audio tower for %q in %s\n", name, time.Since(t0).Round(time.Millisecond))
			if g.device != "" {
				acc, err := embeddinggemma2.NewAudioAccelerator(g.device, g.enc)
				if err != nil {
					fmt.Fprintf(os.Stderr, "audio: the %s audio tower declined (%v); the tower runs on the CPU\n", g.device, err)
				} else {
					g.acc = acc
				}
			}
		}
	})
	return g.enc, g.err
}

// forward is the audio tower: the conformer blocks on the accelerator when there is one between aikit's Subsample and
// FinishBlocks on the host; otherwise aikit's CPU Forward.
func (g *gemma4Audio) forward(enc *audio.Gemma4AudioEncoder, mel []float32, T int) ([]float32, error) {
	if g.acc == nil {
		return enc.Forward(mel, T)
	}
	h, n, err := enc.Subsample(mel, T)
	if err != nil {
		return nil, err
	}
	hb, err := g.acc.Blocks(h, n)
	if err != nil {
		return nil, err
	}
	return enc.FinishBlocks(hb, n)
}

// gemma4AudioMaxSoftTokens is the processor's cap on one clip's soft tokens (E2B's processor_config.json,
// audio_seq_length: 750, 30 s at 40 ms per token). A longer clip is refused rather than cut.
const gemma4AudioMaxSoftTokens = 750

// gemma4AudioMaxSeconds is the same cap in seconds of 16 kHz audio.
const gemma4AudioMaxSeconds = 30

// clipPrompt is the Gemma 4 audio path: the WAV (16-bit PCM at any rate, 1-8 channels, downmixed and resampled to 16
// kHz mono by resample_poly's filter) through aikit's log-mel and the audio tower (embed_audio baked in) -> the prompt
// with the audio block, <|audio> + n x <|audio|> + <audio|>, before turns[idx]'s text, as HF's processor writes it (no
// newline on either side). The run is generated through GenerateGemma4VL, which splices any media run the same way: rows
// unscaled, PAD for PLE's token-identity term.
func (g *gemma4Audio) clipPrompt(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, idx int, clip imageRef) (visionInput, error) {
	wav, err := multimodal.ParseWAV(clip.data)
	if err != nil {
		return visionInput{}, err
	}
	// Checked on the samples, not the soft-token count: the extractor itself stops at 30 s, so a longer clip would
	// otherwise be cut without a word.
	if n := wav.Samples(); n > gemma4AudioMaxSeconds*audio.Gemma4SampleRate {
		return visionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s the model reads", float64(n)/audio.Gemma4SampleRate, gemma4AudioMaxSeconds)
	}
	samples := wav.Decode()
	mel, T, err := audio.Gemma4Features(samples)
	if err != nil {
		return visionInput{}, fmt.Errorf("audio features: %w", err)
	}
	n := audio.Gemma4SoftTokens(T)
	if n > gemma4AudioMaxSoftTokens {
		return visionInput{}, fmt.Errorf("the audio clip is %.1f s, over the 30 s (%d soft tokens) the model reads", float64(len(samples))/16000, gemma4AudioMaxSoftTokens)
	}
	if n <= 0 {
		return visionInput{}, fmt.Errorf("the audio clip is too short (%d samples)", len(samples))
	}
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		enc, err := g.encoder(lm.name)
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
	ids, err := encodeVisionSegments(lm, tm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	pos, ln := multimodal.FindImageRun(ids, g.tok)
	if ln != n {
		return visionInput{}, fmt.Errorf("audio placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", ln, n)
	}
	return visionInput{ids: ids, features: features, imgHash: multimodal.HashImageBytes(clip.data), imgPos: pos, imgLen: ln, gemma4: true}, nil
}

// qwenASRAudio is a Qwen3-ASR checkpoint's audio encoder and projector, which live beside the decoder in its own
// safetensors.
type qwenASRAudio struct {
	dir  string
	once sync.Once
	enc  *audio.QwenASREncoder
	err  error
}

func (q *qwenASRAudio) family() string { return "Qwen3-ASR audio" }

func (q *qwenASRAudio) prompt(lm *loadedModel, _ *chat.Template, system string, _ []chat.Turn, clip imageRef) (visionInput, error) {
	return q.clipPrompt(lm, system, clip)
}

// close releases the encoder, which keeps its checkpoint mapped until Close (aikit embed.SafetensorsFile.Close).
func (q *qwenASRAudio) close() {
	if q.enc != nil {
		_ = closeAudioEncoder("qwenASR", q.enc)
		q.enc = nil
	}
}

// qwenASRMaxSeconds is the longest clip a request may carry. The model has no 30 s window (its encoder attends within 8 s blocks), so this is a cost bound, not the model's limit: a
// clip over it is refused rather than cut.
const qwenASRMaxSeconds = 600

// encoder is the Qwen3-ASR audio encoder and projector, loaded on the first clip (aikit's, CPU, float32).
func (q *qwenASRAudio) encoder() (*audio.QwenASREncoder, error) {
	q.once.Do(func() { q.enc, q.err = audio.LoadQwenASREncoder(q.dir) })
	return q.enc, q.err
}

// clipPrompt is the Qwen3-ASR path: the WAV (16-bit PCM at any rate, downmixed and resampled to 16 kHz mono as for Gemma 4's audio) through the Qwen3-ASR front end, whose
// frame count fixes the number of audio tokens, into the fixed prompt layout with the assistant turn already opened by "language" (multimodal.QwenASRPromptOpenLanguage; the
// model still writes the name and <asr_text>). The encoder runs lazily inside features(). The reply is the model's raw "language <Name><asr_text><transcription>", the word
// supplied from the prompt via replyPrefix, so completion_tokens does not count it. Text in the user turn is not part of the template and is ignored, the system message is passed through.
func (q *qwenASRAudio) clipPrompt(lm *loadedModel, system string, clip imageRef) (visionInput, error) {
	wav, err := multimodal.ParseWAV(clip.data)
	if err != nil {
		return visionInput{}, err
	}
	if n := wav.Samples(); n > qwenASRMaxSeconds*audio.WhisperSampleRate {
		return visionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s one request may carry", float64(n)/audio.WhisperSampleRate, qwenASRMaxSeconds)
	}
	samples := wav.Decode()
	f, err := audio.QwenASRFeatures(samples)
	if err != nil {
		return visionInput{}, fmt.Errorf("audio features: %w", err)
	}
	n := audio.QwenASRTokenCount(f.Valid)
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		enc, err := q.encoder()
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
	ids, pos, err := multimodal.QwenASRPromptOpenLanguage(lm.tk, system, n)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	return visionInput{ids: ids, features: features, imgHash: multimodal.HashImageBytes(clip.data), imgPos: pos, imgLen: n, asr: true, replyPrefix: multimodal.QwenASRLanguageWord}, nil
}

// voxtralAudio is Voxtral Mini's audio tower and projector, which live beside the decoder in its own safetensors.
type voxtralAudio struct {
	dir     string
	backend string // the backend the tower is placed on at its load (-vision-device cpu makes it "cpu")
	once    sync.Once
	device  audio.WhisperDevice // the caller's to close, after the encoder
	enc     *audio.VoxtralAudio
	err     error
}

func (v *voxtralAudio) family() string { return "Voxtral audio" }

func (v *voxtralAudio) prompt(lm *loadedModel, _ *chat.Template, _ string, _ []chat.Turn, clip imageRef) (visionInput, error) {
	return v.clipPrompt(lm, clip)
}

// close releases the device the tower was placed on, then the encoder, which keeps its checkpoint mapped until Close.
func (v *voxtralAudio) close() {
	if v.device != nil {
		_ = v.device.Close()
		v.device = nil
	}
	if v.enc != nil {
		_ = closeAudioEncoder("voxtral", v.enc)
		v.enc = nil
	}
}

// voxtralMaxSeconds is the longest clip a request may carry. Each 30 s window is a full encoder pass (about a minute on a CPU), so this is a cost bound, not the model's limit; a clip over it is
// refused rather than cut.
const voxtralMaxSeconds = 600

// encoder is Voxtral's audio tower and projector (aikit's, CPU, float32), loaded on the first clip. name is the entry's,
// for the load line.
func (v *voxtralAudio) encoder(name string) (*audio.VoxtralAudio, error) {
	v.once.Do(func() {
		v.enc, v.err = audio.LoadVoxtralAudio(v.dir)
		if v.err == nil {
			var where string
			v.device, where = attachWhisperDevice(v.enc.Enc, v.backend, os.Stderr)
			fmt.Fprintf(os.Stderr, "audio: loaded the Voxtral audio tower for %q (encoder on %s)\n", name, where)
		}
	})
	return v.enc, v.err
}

// clipPrompt is the Voxtral path: the WAV (16-bit PCM at any rate, downmixed and resampled to 16 kHz mono as for the other audio models) through the Whisper-style 128-mel front end, one
// 30 s window at a time (the clip is padded up to whole windows), into the transcription request mistral_common lays out (multimodal.VoxtralPrompt, with no language: the model detects
// it). The tower and projector run lazily inside features(). The reply is the model's raw text; text in the user turn is ignored, the system message is not used.
func (v *voxtralAudio) clipPrompt(lm *loadedModel, clip imageRef) (visionInput, error) {
	wav, err := multimodal.ParseWAV(clip.data)
	if err != nil {
		return visionInput{}, err
	}
	if n := wav.Samples(); n > voxtralMaxSeconds*audio.WhisperSampleRate {
		return visionInput{}, fmt.Errorf("the audio clip is %.1f s, over the %d s one request may carry", float64(n)/audio.WhisperSampleRate, voxtralMaxSeconds)
	}
	samples := wav.Decode()
	n := multimodal.VoxtralAudioTokens(len(samples))
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		wins, err := audio.WhisperFeaturesWindows(samples, 128)
		if err != nil {
			return nil, fmt.Errorf("audio features: %w", err)
		}
		enc, err := v.encoder(lm.name)
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
	ids, pos, err := multimodal.VoxtralPrompt(lm.tk, n, "")
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	return visionInput{ids: ids, features: features, imgHash: multimodal.HashImageBytes(clip.data), imgPos: pos, imgLen: n, asr: true}, nil
}
