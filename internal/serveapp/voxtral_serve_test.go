package serveapp

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S14e4a (iii) of docs/tasks/task-multimodal-support-2026-10.md: the serve wiring of Voxtral Mini on the tiny fixture. A real tekken.json is 14.9 MB and not committed, so the tokenizer here is a
// Tekken file built in the test: the 256 byte tokens and 40 special tokens, the seven the prompt uses at the ids the real file gives them. Only special ids appear in a no-language prompt, so
// nothing else of the vocabulary is exercised (the real file is G-S14e4a (i) and (ii), in tokenizer/ and multimodal/).
func tinyTekkenJSON(t *testing.T) []byte {
	t.Helper()
	type v struct {
		Rank       int    `json:"rank"`
		TokenBytes string `json:"token_bytes"`
	}
	type sp struct {
		Rank      int    `json:"rank"`
		TokenStr  string `json:"token_str"`
		IsControl bool   `json:"is_control"`
	}
	names := map[int]string{0: "<unk>", 1: "<s>", 2: "</s>", 3: "[INST]", 4: "[/INST]", 24: "[AUDIO]", 25: "[BEGIN_AUDIO]", 34: "[TRANSCRIBE]"}
	var specials []sp
	for i := range 40 {
		n, ok := names[i]
		if !ok {
			n = fmt.Sprintf("<SPECIAL_%d>", i)
		}
		specials = append(specials, sp{i, n, true})
	}
	var vocab []v
	for i := range 256 {
		vocab = append(vocab, v{i, base64.StdEncoding.EncodeToString([]byte{byte(i)})})
	}
	cfg := map[string]any{
		"pattern":          `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		"num_vocab_tokens": 256, "default_vocab_size": 296, "default_num_special_tokens": 40, "version": "v7",
	}
	raw, err := json.Marshal(map[string]any{"config": cfg, "vocab": vocab, "special_tokens": specials})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func sineWAV(n, rate int) []byte {
	b := wavBytes(n, rate)
	for i := range n {
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(int16(8000*math.Sin(float64(i)*0.05)+3000*math.Sin(float64(i)*0.31))))
	}
	return b
}

func tinyVoxtralLM(t *testing.T) (*loadedModel, string) {
	t.Helper()
	dir, err := filepath.Abs(filepath.Join("..", "..", "testdata", "voxtral-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Skipf("no tiny Voxtral fixture: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	tk, err := tokenizer.LoadTekken(tinyTekkenJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	return &loadedModel{model: m, tk: tk, name: "voxtral-tiny", voxtralDir: dir}, dir
}

// TestVoxtralPrompt_wiring: voxtralPrompt builds the processor's no-language request (ids equal multimodal.VoxtralPrompt's, which G-S14e4a (ii) pins to the processor's own on the real file), counts
// the placeholders from the clip's length (375 per 30 s window), and its features are the Whisper-style windows through the tower and projector, bit for bit; a generation through it runs.
func TestVoxtralPrompt_wiring(t *testing.T) {
	lm, dir := tinyVoxtralLM(t)
	for _, c := range []struct {
		name    string
		samples int
		windows int
	}{{"6 s", 96000, 1}, {"35 s", 560000, 2}} {
		wav := sineWAV(c.samples, 16000)
		vi, err := lm.voxtralPrompt(imageRef{mediaType: "audio/wav", data: wav, audio: true})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		n := c.windows * multimodal.VoxtralTokensPerChunk
		want, pos, err := multimodal.VoxtralPrompt(lm.tk, n, "")
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(vi.ids, want) || vi.imgPos != pos || vi.imgLen != n || !vi.asr {
			t.Fatalf("%s: prompt %d ids, audio at %d (%d rows, asr %v); want %d ids, at %d (%d rows)", c.name, len(vi.ids), vi.imgPos, vi.imgLen, vi.asr, len(want), pos, n)
		}
		if got := vi.ids[len(vi.ids)-2:]; got[0] != 4 || got[1] != 34 { // [/INST] [TRANSCRIBE]: no language text between
			t.Fatalf("%s: the prompt ends %v, want [/INST] [TRANSCRIBE] = [4 34]", c.name, got)
		}
		emb, err := vi.features()
		if err != nil {
			t.Fatalf("%s: features: %v", c.name, err)
		}
		samples, _ := multimodal.DecodeWAVAnyRate(wav)
		wins, err := audio.WhisperFeaturesWindows(samples, 128)
		if err != nil {
			t.Fatal(err)
		}
		va, err := audio.LoadVoxtralAudio(dir)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := va.Embed(wins)
		if err != nil {
			t.Fatal(err)
		}
		if len(emb) != n*lm.model.Config().HiddenDim || !slices.Equal(emb, ref) {
			t.Fatalf("%s: features differ from the encoder's own output (%d values, %d expected)", c.name, len(emb), len(ref))
		}
		ch, g := lm.model.GenerateAudio(context.Background(), vi.ids, vi.imgPos, vi.imgLen, vi.features, 6, decoder.SamplingParams{})
		var got []int
		for id := range ch {
			got = append(got, id)
		}
		if err := g.Err(); err != nil || len(got) == 0 {
			t.Fatalf("%s: generation through the served prompt: %d tokens, err %v", c.name, len(got), err)
		}
	}
}

// TestVoxtralPrompt_refusals: a clip over the cost bound, a non-WAV, and a model the tokenizer of which has no [AUDIO] (loadVoxtral) are refused by name.
func TestVoxtralPrompt_refusals(t *testing.T) {
	lm, _ := tinyVoxtralLM(t)
	if _, err := lm.voxtralPrompt(imageRef{data: wavBytes((voxtralMaxSeconds+10)*16000, 16000), audio: true}); err == nil || !strings.Contains(err.Error(), "over the") {
		t.Errorf("a %d s clip: want the cost-bound refusal, got %v", voxtralMaxSeconds+10, err)
	}
	if _, err := lm.voxtralPrompt(imageRef{data: []byte("not a wav"), audio: true}); err == nil {
		t.Error("a non-WAV was accepted")
	}
	s := &server{models: map[string]*loadedModel{"m": {name: "m", tk: lm.tk}}}
	if err := s.loadVoxtral("dir"); err != nil {
		t.Errorf("loadVoxtral with an [AUDIO] token: %v", err)
	}
	// a tokenizer without [AUDIO]: the tiny Tekken minus the name
	raw := strings.Replace(string(tinyTekkenJSON(t)), "[AUDIO]", "[AUDIOX]", 1)
	tk, err := tokenizer.LoadTekken([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	s = &server{models: map[string]*loadedModel{"m": {name: "m", tk: tk}}}
	if err := s.loadVoxtral("dir"); err == nil || !strings.Contains(err.Error(), "[AUDIO]") {
		t.Errorf("a tokenizer with no [AUDIO]: want a named refusal, got %v", err)
	}
}
