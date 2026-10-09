//go:build realckpt

package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S14c4's goinfer arms (docs/tasks/task-multimodal-support-2026-10.md, registered before this code): every WAV in $GOINFER_S14C4_DIR (scripts/s14c4_data.py) transcribed by Qwen3-ASR-0.6B through
// the library path (Go front end, encoder, projector, soft-token splice, GenerateAudio, greedy, 256 new tokens at most) at the checkpoint's own weights (arm "f32") and at serve's defaults (arm
// "int4": Quant int4, int4 head). The encoder is float32 in both, so it runs once per clip. Writes {arm: {id: raw text}} to $GOINFER_S14C4_OUT; scripts/s14c4_grade.py scores it.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S14C4_DIR=<wav dir> GOINFER_S14C4_OUT=<json> [GOINFER_S14C4_LIMIT=<clips>] go test -tags realckpt ./decoder/ -run TestQwen3ASRWER_arms -v -timeout 120m
func TestQwen3ASRWER_arms(t *testing.T) {
	requireHeavyModel(t)
	data, out := os.Getenv("GOINFER_S14C4_DIR"), os.Getenv("GOINFER_S14C4_OUT")
	if data == "" || out == "" {
		t.Skip("set GOINFER_S14C4_DIR (scripts/s14c4_data.py's output) and GOINFER_S14C4_OUT")
	}
	dir := os.Getenv("GOINFER_QWEN3ASR_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "qwen3-asr-0.6b")
	}
	raw, err := os.ReadFile(filepath.Join(data, "refs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var refs map[string]string
	if err := json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(refs))
	for id := range refs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if lim := os.Getenv("GOINFER_S14C4_LIMIT"); lim != "" {
		var n int
		fmt.Sscanf(lim, "%d", &n)
		if n > 0 && n < len(ids) {
			ids = ids[:n]
		}
	}
	T0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[s14c4 %5.0fs] %s\n", time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadQwenASREncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	type clip struct {
		ids []int
		pos int
		emb []float32
		n   int
	}
	clips := map[string]clip{}
	for i, id := range ids {
		wav, err := os.ReadFile(filepath.Join(data, id+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := multimodal.DecodeWAVAnyRate(wav)
		if err != nil {
			t.Fatal(err)
		}
		f, err := audio.QwenASRFeatures(samples)
		if err != nil {
			t.Fatal(err)
		}
		emb, n, err := enc.Forward(f.Data, f.T, f.Valid)
		if err != nil {
			t.Fatal(err)
		}
		pids, pos, err := multimodal.QwenASRPrompt(tk, "", n, "")
		if err != nil {
			t.Fatal(err)
		}
		clips[id] = clip{pids, pos, emb, n}
		if (i+1)%10 == 0 || i+1 == len(ids) {
			hb("encoder: %d/%d clips", i+1, len(ids))
		}
	}
	result := map[string]map[string]string{}
	for _, arm := range []struct {
		name string
		opts Options
	}{{"f32", Options{Backend: "cpu"}}, {"int4", Options{Backend: "cpu", Quant: "int4", EmbedInt4: true}},
		// Amendment A1 (2026-10-09), record only: serve's int4 with the head table at the int8 pin, to see whether the int4 head is what damages the first token.
		{"int4h8", Options{Backend: "cpu", Quant: "int4", EmbedInt4: false}}} {
		m, err := Load(dir, arm.opts)
		if err != nil {
			t.Fatalf("%s: Load: %v", arm.name, err)
		}
		hb("arm %s loaded: %s, head table %s", arm.name, m.DecodePath(), m.HeadTable())
		res := map[string]string{}
		for i, id := range ids {
			c := clips[id]
			ch, g := m.GenerateAudio(context.Background(), c.ids, c.pos, c.n, func() ([]float32, error) { return c.emb, nil }, 256, SamplingParams{})
			var toks []int
			for tok := range ch {
				toks = append(toks, tok)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("%s %s: %v", arm.name, id, err)
			}
			res[id], _ = tk.Decode(toks)
			if (i+1)%10 == 0 || i+1 == len(ids) {
				hb("arm %s: %d/%d clips, last %q", arm.name, i+1, len(ids), res[id])
			}
		}
		m.Close()
		result[arm.name] = res
	}
	b, _ := json.MarshalIndent(result, "", " ")
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
	hb("done: %d clips, 3 arms, wrote %s", len(ids), out)
}
