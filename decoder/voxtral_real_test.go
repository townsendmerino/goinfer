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

// G-S14e3 of docs/tasks/task-multimodal-support-2026-10.md (registered 2026-10-09 before any code): the REAL Voxtral Mini 3B, goinfer float32 on the CPU, against transformers float32
// (scripts/pin_voxtral_real.py: the reference loads with no missing keys and is proved real by transcribing the LibriSpeech clip). Registered bars, on the LibriSpeech clip: the prompt ids equal
// the processor's (the Tekken tokenizer and VoxtralPrompt on the real tekken.json); the tower output compared per row (1500 rows) and the projector output per row (375 rows) at cosine >=
// 0.9999, the first and last rows named; the prompt-row logits at cosine >= 0.9999 with the argmax equal; the greedy transcription BYTE-EQUAL (token for token) to transformers'; and with the
// reference's tokens forced, every generated position's logits at cosine >= 0.9999. The second case (the clip six times, two windows, 750 audio tokens) is RECORD ONLY: it exercises the chunking
// on real weights and its readings are logged, never graded.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_VOXTRAL_REF=<pin output dir> [GOINFER_VOXTRAL_DIR=<checkpoint, with tekken.json>] go test -tags realckpt ./decoder/ -run TestVoxtralReal -v -timeout 60m
func TestVoxtralReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ref := os.Getenv("GOINFER_VOXTRAL_REF")
	if ref == "" {
		t.Skip("set GOINFER_VOXTRAL_REF to scripts/pin_voxtral_real.py's output directory")
	}
	dir := os.Getenv("GOINFER_VOXTRAL_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "voxtral-mini-3b-2507")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Cases map[string]struct {
			PromptIDs int    `json:"prompt_ids"`
			Windows   int    `json:"windows"`
			AudioRows int    `json:"audio_rows"`
			Generated int    `json:"generated"`
			Text      string `json:"text"`
			Vocab     int    `json:"vocab"`
			EncRows   int    `json:"enc_rows"`
		} `json:"cases"`
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
	ints := func(name string) []int {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return q3aInts(t, b)
	}
	t0 := time.Now()
	m, err := Load(dir, Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	t.Logf("loaded in %.1fs: arch %s, %d layers, hidden %d, %d heads / %d kv, head_dim %d, vocab %d, quant %s, rope base %v, tied head %v",
		time.Since(t0).Seconds(), a.Name, a.NumLayers, a.HiddenDim, a.NumHeads, a.NumKVHeads, a.HeadDim, a.VocabSize, m.Quant(), a.RoPEGlobalBase, a.TiedLMHead)
	if a.Name != "voxtral" || a.TiedLMHead || a.HeadDim != 128 {
		t.Fatalf("instrument: unexpected architecture (name %q, tied head %v, head_dim %d)", a.Name, a.TiedLMHead, a.HeadDim)
	}
	tekken, err := os.ReadFile(filepath.Join(dir, "tekken.json"))
	if err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.LoadTekken(tekken)
	if err != nil {
		t.Fatal(err)
	}
	va, err := audio.LoadVoxtralAudio(dir)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	hidden := a.HiddenDim
	eos := 2 // </s>, generation_config.json's eos_token_id

	for _, name := range []string{"libri", "libri6x"} {
		c, ok := meta.Cases[name]
		if !ok {
			t.Logf("case %s not in the reference (smoke run?): skipped", name)
			continue
		}
		graded := name == "libri"
		bad := func(format string, args ...any) {
			t.Helper()
			if graded {
				t.Errorf(format, args...)
			} else {
				t.Logf("RECORD ONLY (not graded): "+format, args...)
			}
		}
		samples := rd(name + ".in.f32")
		refIDs := ints(name + ".ids.json")
		refGen := ints(name + ".gen.json")

		// The processor's own request, rebuilt by goinfer's tokenizer and prompt builder.
		wins, err := audio.WhisperFeaturesWindows(samples, 128)
		if err != nil {
			t.Fatal(err)
		}
		ids, pos, err := multimodal.VoxtralPrompt(tk, multimodal.VoxtralAudioTokens(len(samples)), "en")
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, refIDs) {
			t.Fatalf("%s instrument: the Go prompt (%d ids) differs from the processor's (%d)", name, len(ids), len(refIDs))
		}
		refFeats := rd(name + ".feats.f32")
		var fw, fmax float64 = 1, 0
		for w, f := range wins {
			fw = min(fw, q3aCos(f, refFeats[w*len(f):(w+1)*len(f)]))
			for i := range f {
				fmax = max(fmax, float64(abs32(f[i]-refFeats[w*len(f)+i])))
			}
		}
		t.Logf("%s: features (Go front end vs the processor's): worst window cosine %.9f, max |diff| %.3e", name, fw, fmax)

		t1 := time.Now()
		emb, tower, err := va.EmbedWithTower(wins)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: tower and projector over %d window(s) took %.1fs", name, len(wins), time.Since(t1).Seconds())
		rowCos := func(got, want []float32, h int) (worst float64, worstAt int, first, last float64) {
			worst = 1
			rows := len(got) / h
			for r := range rows {
				c := q3aCos(got[r*h:(r+1)*h], want[r*h:(r+1)*h])
				if r == 0 {
					first = c
				}
				if r == rows-1 {
					last = c
				}
				if c < worst {
					worst, worstAt = c, r
				}
			}
			return
		}
		refTower, refEmb := rd(name+".enc.f32"), rd(name+".embeds.f32")
		if len(tower) != len(refTower) || len(emb) != len(refEmb) {
			t.Fatalf("%s instrument: tower %d / embeds %d values against the reference's %d / %d", name, len(tower), len(emb), len(refTower), len(refEmb))
		}
		tw, twAt, tf0, tfN := rowCos(tower, refTower, 1280)
		ew, ewAt, ef0, efN := rowCos(emb, refEmb, hidden)
		t.Logf("%s: tower rows: worst cosine %.9f at row %d (first row %.9f, last row %.9f); projector rows: worst %.9f at row %d (first %.9f, last %.9f)", name, tw, twAt, tf0, tfN, ew, ewAt, ef0, efN)
		if tw < 0.9999 || ew < 0.9999 {
			bad("%s: tower worst row cosine %.9f (row %d), projector worst %.9f (row %d); bar 0.9999", name, tw, twAt, ew, ewAt)
		}

		// The prompt, then the greedy transcription, free-running.
		cache := m.NewCache(len(ids) + len(refGen) + 8)
		t2 := time.Now()
		lg, err := m.prefillLogitsAudio(ctx, ids, emb, pos, c.AudioRows, cache)
		if err != nil {
			t.Fatal(err)
		}
		refLogits := rd(name + ".logits.f32")
		cos := q3aCos(lg, refLogits)
		t.Logf("%s: prompt prefill %.1fs; last-position logit cosine %.9f, argmax %d (reference %d)", name, time.Since(t2).Seconds(), cos, argmax(lg), argmax(refLogits))
		if cos < 0.9999 || argmax(lg) != argmax(refLogits) {
			bad("%s: prompt logits differ from transformers (cosine %.9f, argmax %d against %d)", name, cos, argmax(lg), argmax(refLogits))
		}
		var gen []int
		l := lg
		for len(gen) < len(refGen)+8 { // allow a few more tokens than the reference made, so a missed stop is seen
			id := argmax(l)
			gen = append(gen, id)
			if id == eos {
				break
			}
			if l, err = m.forward(id, cache); err != nil {
				t.Fatal(err)
			}
		}
		text, _ := tk.Decode(gen[:len(gen)-boolInt(len(gen) > 0 && gen[len(gen)-1] == eos)])
		first := -1
		for i := 0; i < min(len(gen), len(refGen)); i++ {
			if gen[i] != refGen[i] {
				first = i
				break
			}
		}
		t.Logf("%s: free-run greedy: %d tokens (reference %d), first difference at %d (-1 = none); text %q", name, len(gen), len(refGen), first, text)
		t.Logf("%s: reference text %q", name, c.Text)
		if !slices.Equal(gen, refGen) {
			bad("%s: the greedy transcription differs from transformers' (first difference at token %d; %d against %d tokens)", name, first, len(gen), len(refGen))
		}

		// Teacher-forced: the reference's own tokens through goinfer, every generated position's logits.
		cache2 := m.NewCache(len(ids) + len(refGen) + 1)
		l2, err := m.prefillLogitsAudio(ctx, ids, emb, pos, c.AudioRows, cache2)
		if err != nil {
			t.Fatal(err)
		}
		tfRef := rd(name + ".tf_logits.f32")
		worst, miss := 1.0, 0
		for k, id := range refGen {
			cc := q3aCos(l2, tfRef[k*c.Vocab:(k+1)*c.Vocab])
			worst = min(worst, cc)
			if argmax(l2) != id {
				miss++
			}
			if k < len(refGen)-1 {
				if l2, err = m.forward(id, cache2); err != nil {
					t.Fatal(err)
				}
			}
		}
		t.Logf("%s: teacher-forced: %d positions, worst logit cosine %.9f, positions where goinfer's argmax is not the reference's token: %d", name, len(refGen), worst, miss)
		if worst < 0.9999 {
			bad("%s: worst teacher-forced logit cosine %.9f, bar 0.9999", name, worst)
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
