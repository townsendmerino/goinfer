package whisper

import (
	"fmt"
	"path/filepath"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Transcriber is a whole short-form Whisper: aikit's log-mel front end and encoder, this package's decoder, and the checkpoint's tokenizer. One 30 s window per call.
type Transcriber struct {
	Mels int
	Enc  *audio.WhisperEncoder
	Dec  *Decoder
	Gen  GenConfig
	Tok  *tokenizer.Tokenizer
}

// LoadTranscriber reads a Whisper checkpoint directory (config.json, generation_config.json, tokenizer.json and the safetensors).
func LoadTranscriber(dir string) (*Transcriber, error) {
	enc, err := audio.LoadWhisperEncoder(dir)
	if err != nil {
		return nil, err
	}
	dec, err := Load(dir)
	if err != nil {
		return nil, err
	}
	gen, err := LoadGenConfig(dir)
	if err != nil {
		return nil, err
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	return &Transcriber{Mels: enc.Cfg.MelBins, Enc: enc, Dec: dec, Gen: gen, Tok: tk}, nil
}

// MaxSamples is one window: 30 s of 16 kHz audio. A longer clip is long-form (S14.4d) and is refused here.
const MaxSamples = 30 * audio.WhisperSampleRate

// Encode runs the front end and the encoder over one window and returns the encoder's last hidden state.
func (t *Transcriber) Encode(samples []float32) ([]float32, error) {
	if len(samples) > MaxSamples {
		return nil, fmt.Errorf("whisper: the clip is %.1f s, over the 30 s one window holds (long-form decoding is not built)", float64(len(samples))/audio.WhisperSampleRate)
	}
	feats, _, err := audio.WhisperFeatures(samples, t.Mels)
	if err != nil {
		return nil, err
	}
	return t.Enc.Forward(feats)
}

// Transcribe is transformers' short-form greedy generate and batch_decode(skip_special_tokens=True) for one clip: the text (as the tokenizer decodes it, so with the leading space Whisper writes),
// the result and the language. language is a code ("en") or empty to detect.
func (t *Transcriber) Transcribe(samples []float32, language, task string) (string, *Result, error) {
	enc, err := t.Encode(samples)
	if err != nil {
		return "", nil, err
	}
	res, err := t.Dec.Generate(enc, t.Gen, language, task)
	if err != nil {
		return "", nil, err
	}
	return t.Text(res.IDs), res, nil
}

// Text decodes generated ids to text, skipping the special tokens (every id from the stop token up: the stop token, the language and task tokens, the timestamps).
func (t *Transcriber) Text(ids []int) string {
	var keep []int
	for _, id := range ids {
		if id < t.Gen.EOS {
			keep = append(keep, id)
		}
	}
	s, _ := t.Tok.Decode(keep)
	return s
}
