package multimodal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Voxtral Mini's audio constants (docs/tasks/task-multimodal-support-2026-10.md, S14.4b).
const (
	// VoxtralChunkSamples is one encoder window: 30 s at 16 kHz. The processor pads every clip UP to a multiple of it.
	VoxtralChunkSamples = 480000
	// VoxtralTokensPerChunk is the audio tokens one chunk yields: the encoder's 1500 frames stacked four to a row.
	VoxtralTokensPerChunk = 375
)

// VoxtralChunks is how many thirty-second encoder windows a clip of n samples occupies: the clip is padded up to a whole number of them, and an empty clip still takes one.
func VoxtralChunks(nSamples int) int {
	if nSamples <= 0 {
		return 1
	}
	return (nSamples + VoxtralChunkSamples - 1) / VoxtralChunkSamples
}

// VoxtralAudioTokens is the number of [AUDIO] placeholders a clip of n samples needs.
func VoxtralAudioTokens(nSamples int) int { return VoxtralChunks(nSamples) * VoxtralTokensPerChunk }

// voxtralPromptDefects are the planted-defect switches of voxtral_test.go; the zero value is the correct prompt.
type voxtralPromptDefects struct {
	noBOS     bool // omit <s>
	swapInst  bool // [/INST] where [INST] belongs and the reverse
	noColon   bool // "lang en" for "lang:en"
	countDiff int  // the placeholder count off by this much
	noTranscr bool // omit [TRANSCRIBE]
}

// VoxtralPrompt builds the token ids of a Voxtral transcription request, the way mistral_common's encode_transcription lays it out (and so transformers'
// apply_transcription_request): <s> [INST] [BEGIN_AUDIO] n x [AUDIO] [/INST] then, for a non-empty language, the plain-text ids of "lang:<code>", then [TRANSCRIBE]. The language
// text is ordinary text (it is encoded without parsing special tokens). It returns the ids and the position of the first placeholder.
func VoxtralPrompt(tk *tokenizer.Tokenizer, n int, language string) (ids []int, audioPos int, err error) {
	return voxtralPrompt(tk, n, language, voxtralPromptDefects{})
}

func voxtralPrompt(tk *tokenizer.Tokenizer, n int, language string, d voxtralPromptDefects) (ids []int, audioPos int, err error) {
	if n <= 0 {
		return nil, 0, fmt.Errorf("multimodal: %d audio tokens", n)
	}
	need := func(s string) (int, error) {
		id, ok := tk.TokenID(s)
		if !ok {
			return 0, fmt.Errorf("multimodal: the tokenizer has no %s token (not a Voxtral tokenizer)", s)
		}
		return id, nil
	}
	bos, inst, begin, audio, endInst, transcribe := tk.Special().BOS, 0, 0, 0, 0, 0
	for _, p := range []struct {
		dst *int
		s   string
	}{{&inst, "[INST]"}, {&begin, "[BEGIN_AUDIO]"}, {&audio, "[AUDIO]"}, {&endInst, "[/INST]"}, {&transcribe, "[TRANSCRIBE]"}} {
		if *p.dst, err = need(p.s); err != nil {
			return nil, 0, err
		}
	}
	if bos < 0 {
		return nil, 0, fmt.Errorf("multimodal: the tokenizer has no <s> token")
	}
	if d.swapInst {
		inst, endInst = endInst, inst
	}
	if !d.noBOS {
		ids = append(ids, bos)
	}
	ids = append(ids, inst, begin)
	audioPos = len(ids)
	for range n + d.countDiff {
		ids = append(ids, audio)
	}
	ids = append(ids, endInst)
	if language != "" {
		text := "lang:" + language
		if d.noColon {
			text = "lang " + language
		}
		lang, err := tk.EncodeSegments([]tokenizer.Segment{{Text: text}}, false)
		if err != nil {
			return nil, 0, err
		}
		ids = append(ids, lang...)
	}
	if !d.noTranscr {
		ids = append(ids, transcribe)
	}
	return ids, audioPos, nil
}
