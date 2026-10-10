//go:build realckpt

package whisper

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// G-S14f2 of docs/tasks/task-multimodal-support-2026-10.md: openai/whisper-small, goinfer float32 on the CPU against transformers 5.15.0 float32 on the LibriSpeech clip. The reference is
// scripts/pin_whisper_decoder_real.py's. Bars: (a) the detected language is the reference's; (b) the logits at the last prompt position, cosine >= 0.9999 and the same argmax; (c) the greedy
// transcription byte-equal (ids and decoded text); (d) teacher-forced on the reference's own tokens, worst cosine >= 0.9999 and no position where the argmax (after the same suppression) differs.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_WHISPER_REF=<pin output dir> [GOINFER_WHISPER_DIR=<checkpoint>] go test -tags realckpt ./internal/whisper/ -run TestWhisperSmall_real -v -timeout 30m
func TestWhisperSmall_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	ref := os.Getenv("GOINFER_WHISPER_REF")
	if ref == "" {
		t.Skip("set GOINFER_WHISPER_REF to scripts/pin_whisper_decoder_real.py's output directory")
	}
	dir := os.Getenv("GOINFER_WHISPER_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "whisper-small")
	}
	rd := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	ints := func(name string, v any) {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			t.Fatal(err)
		}
	}
	var refPrompt, refGen []int
	var refLang int
	var meta struct{ Text string }
	ints("libri.prompt.json", &refPrompt)
	ints("libri.gen.json", &refGen)
	ints("libri.lang.json", &refLang)
	ints("meta.json", &meta)
	samples := rd("libri.in.f32")

	tr, err := LoadTranscriber(dir)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := tr.Encode(samples)
	if err != nil {
		t.Fatal(err)
	}
	V := tr.Dec.Cfg.Vocab
	argmax := func(l []float32) int {
		b := 0
		for i, v := range l {
			if v > l[b] {
				b = i
			}
		}
		return b
	}
	suppress := func(l []float32, first bool) []float32 {
		l = slices.Clone(l)
		for _, id := range tr.Gen.Suppress {
			l[id] = float32(math.Inf(-1))
		}
		if first {
			for _, id := range tr.Gen.BeginSuppress {
				l[id] = float32(math.Inf(-1))
			}
		}
		return l
	}

	// (a) the language
	tok, id, err := tr.Dec.Detect(enc, tr.Gen)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("(a) detected language %s (id %d); the reference's id %d", tok, id, refLang)
	if id != refLang {
		t.Errorf("(a) language id %d, reference %d", id, refLang)
	}

	// (b) the prompt's last position
	st, err := tr.Dec.NewState(enc)
	if err != nil {
		t.Fatal(err)
	}
	pl, err := st.Forward(refPrompt, false)
	if err != nil {
		t.Fatal(err)
	}
	pref := rd("libri.plogits.f32")
	c := cos(pl[0], pref)
	t.Logf("(b) last prompt position: logit cosine %.9f, argmax %d (reference %d)", c, argmax(pl[0]), argmax(pref))
	if c < 0.9999 || argmax(pl[0]) != argmax(pref) {
		t.Errorf("(b) prompt logits cosine %.9f, argmax %d against %d", c, argmax(pl[0]), argmax(pref))
	}

	// (c) the greedy transcription
	text, res, err := tr.Transcribe(samples, "", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("(c) %d generated ids (reference %d), prompt %v (reference %v), language %s, text %q", len(res.IDs), len(refGen), res.Prompt, refPrompt, res.Language, text)
	if !slices.Equal(res.IDs, refGen) || !slices.Equal(res.Prompt, refPrompt) || text != meta.Text {
		t.Errorf("(c) the transcription differs from transformers': ids %v against %v, prompt %v against %v, text %q against %q", res.IDs, refGen, res.Prompt, refPrompt, text, meta.Text)
	}

	// (d) teacher-forced
	st2, err := tr.Dec.NewState(enc)
	if err != nil {
		t.Fatal(err)
	}
	all, err := st2.Forward(append(slices.Clone(refPrompt), refGen[:len(refGen)-1]...), true)
	if err != nil {
		t.Fatal(err)
	}
	tf := rd("libri.tf.f32")
	if len(tf) != len(refGen)*V {
		t.Fatalf("instrument: the reference holds %d teacher-forced values for %d positions of %d", len(tf), len(refGen), V)
	}
	worst, bad := 1.0, 0
	for i := range refGen {
		row := all[len(refPrompt)-1+i]
		worst = math.Min(worst, cos(row, tf[i*V:(i+1)*V]))
		if argmax(suppress(row, i == 0)) != refGen[i] {
			bad++
		}
	}
	t.Logf("(d) teacher-forced over %d positions: worst logit cosine %.9f, %d positions where the argmax is not the reference's token", len(refGen), worst, bad)
	if worst < 0.9999 || bad != 0 {
		t.Errorf("(d) worst cosine %.9f, %d argmax differences", worst, bad)
	}
}
