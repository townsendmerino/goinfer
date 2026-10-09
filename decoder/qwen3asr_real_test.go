//go:build realckpt

package decoder

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S14c1 (real, the text path) and G-S14c3 (real, the composition) of docs/tasks/task-multimodal-support-2026-10.md (registered before this code): Qwen3-ASR-0.6B, goinfer float32 on the
// CPU, against transformers float32 (scripts/pin_qwen3asr_real.py: the checkpoint converted by a pure rename and proved by transcribing the clip). c1: logits for a fixed token sequence,
// cosine >= 0.9999, argmax and a 6-token continuation equal. c3: the LibriSpeech clip through the Go front end, encoder, projector, soft-token splice and decoder: the prompt ids equal the
// processor's, the greedy transcription equals transformers' token for token, and with transformers' path forced every generated position's logits have cosine >= 0.9999.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QWEN3ASR_REF=<pin output dir> [GOINFER_QWEN3ASR_DIR=<checkpoint, with tokenizer.json>] go test -tags realckpt ./decoder/ -run TestQwen3ASRReal -v -timeout 30m
func TestQwen3ASRReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ref := os.Getenv("GOINFER_QWEN3ASR_REF")
	if ref == "" {
		t.Skip("set GOINFER_QWEN3ASR_REF to scripts/pin_qwen3asr_real.py's output directory")
	}
	dir := os.Getenv("GOINFER_QWEN3ASR_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3-asr-0.6b")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Text struct {
			IDs          []int
			Continuation []int
		}
		Transcribe struct {
			PromptIDs    []int `json:"prompt_ids"`
			GeneratedIDs []int `json:"generated_ids"`
			Text         string
			Vocab        int
		}
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	rd := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return q3aF32(b)
	}
	t0 := time.Now()
	m, err := Load(dir, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	t.Logf("loaded in %.1fs: arch %s, %d layers, hidden %d, vocab %d, quant %s", time.Since(t0).Seconds(), m.w.arch.Name, m.w.arch.NumLayers, m.w.arch.HiddenDim, m.w.arch.VocabSize, m.Quant())
	ctx := context.Background()

	// c1: the text path.
	cache := m.NewCache(len(meta.Text.IDs) + 8)
	tl, err := m.prefillLogits(ctx, meta.Text.IDs, cache)
	if err != nil {
		t.Fatal(err)
	}
	want := rd("text.logits.f32")
	cos := q3aCos(tl, want)
	cont := q3aGreedy(t, m, cache, tl, len(meta.Text.Continuation))
	t.Logf("c1 text path: last-row cosine %.9f, argmax %d (HF %d), continuation %v (HF %v)", cos, argmax(tl), argmax(want), cont, meta.Text.Continuation)
	if cos < 0.9999 || argmax(tl) != argmax(want) || !slices.Equal(cont, meta.Text.Continuation) {
		t.Errorf("c1: text path differs from transformers (cosine %.9f, continuation %v against %v)", cos, cont, meta.Text.Continuation)
	}

	// c3: the composition.
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadQwenASREncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	f, err := audio.QwenASRFeatures(rd("libri.in.f32"))
	if err != nil {
		t.Fatal(err)
	}
	t1 := time.Now()
	emb, n, err := enc.Forward(f.Data, f.T, f.Valid)
	if err != nil {
		t.Fatal(err)
	}
	ids, pos, err := multimodal.QwenASRPrompt(tk, "", n, "")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, meta.Transcribe.PromptIDs) {
		t.Fatalf("c3 instrument: the prompt ids (%d) differ from the processor's (%d): %v against %v", len(ids), len(meta.Transcribe.PromptIDs), ids, meta.Transcribe.PromptIDs)
	}
	t.Logf("c3 instrument: the Go prompt (%d ids, %d audio tokens at %d) equals the processor's; the encoder took %.1fs", len(ids), n, pos, time.Since(t1).Seconds())
	hf := meta.Transcribe.GeneratedIDs
	eos := m.eosIDs
	if len(hf) == 0 || !slices.Contains(eos, hf[len(hf)-1]) {
		t.Fatalf("c3 instrument: transformers' generation %v does not end in one of goinfer's stop ids %v", hf, eos)
	}
	// teacher-forced: HF's path through goinfer, the logits at every position that predicted a generated token
	cache2 := m.NewCache(len(ids) + len(hf) + 1)
	lg, err := m.prefillLogitsAudio(ctx, ids, emb, pos, n, cache2)
	if err != nil {
		t.Fatal(err)
	}
	tf := rd("libri.tf_logits.f32")
	V := meta.Transcribe.Vocab
	worst, argmaxMiss := 1.0, 0
	for k, id := range hf {
		c := q3aCos(lg, tf[k*V:(k+1)*V])
		worst = min(worst, c)
		if argmax(lg) != id {
			argmaxMiss++
		}
		if k < len(hf)-1 {
			if lg, err = m.forward(id, cache2); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("c3 teacher-forced: %d positions, worst logit cosine %.9f, positions where goinfer's argmax is not transformers' token: %d", len(hf), worst, argmaxMiss)
	if worst < 0.9999 {
		t.Errorf("c3: worst teacher-forced cosine %.9f, bar 0.9999", worst)
	}
	// free-run greedy through the public path
	t2 := time.Now()
	ch, g := m.GenerateAudio(ctx, ids, pos, n, func() ([]float32, error) { return emb, nil }, 64, SamplingParams{})
	var got []int
	for id := range ch {
		got = append(got, id)
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	text, _ := tk.Decode(got)
	t.Logf("c3 free run (%.1fs): %q", time.Since(t2).Seconds(), text)
	t.Logf("c3 transformers:      %q", meta.Transcribe.Text)
	if !slices.Equal(got, hf[:len(hf)-1]) {
		t.Errorf("c3: the greedy transcription differs from transformers' (%d tokens against %d)", len(got), len(hf)-1)
	}
}
