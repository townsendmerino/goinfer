package multimodal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14b3 of docs/tasks/task-multimodal-support-2026-10.md (registered before this code): Qwen3-ASR-0.6B's audio encoder and projector, goinfer float32 on the CPU, against transformers
// float32 on three clips (the LibriSpeech clip, 12 s of synthetic signal, 46.8 s of the LibriSpeech clip tiled so that several attention windows span the clip). The reference is
// scripts/pin_qwen3asr_real.py's, which converts the checkpoint by a pure rename (transformers 5.15.0 cannot read it as shipped) and proves the conversion by transcribing the clip.
// Bar: every output token's cosine >= 0.9999 and the token count exact. (a) feeds the encoder the reference's own features, isolating the encoder; (b) feeds it the Go front end's, recorded.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QWEN3ASR_REF=<pin output dir> [GOINFER_QWEN3ASR_DIR=<checkpoint>] go test ./multimodal/ -run TestQwenASREncoder_real -v -timeout 30m
func TestQwenASREncoder_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
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
		Cases []struct {
			Name              string
			Samples, T, Valid int
			Tokens            int
		}
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	enc, err := audio.LoadQwenASREncoder(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("loaded %s in %.1fs: %+v", filepath.Base(dir), time.Since(t0).Seconds(), enc.Cfg)
	rd := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return f32s(b)
	}
	oneMinusCos := map[string]float64{}
	for _, c := range meta.Cases {
		feats, tower, out := rd(c.Name+".feats.f32"), rd(c.Name+".tower.f32"), rd(c.Name+".out.f32")
		if audio.QwenASRTokenCount(c.Valid) != c.Tokens {
			t.Errorf("%s: QwenASRTokenCount(%d) = %d, the reference has %d", c.Name, c.Valid, audio.QwenASRTokenCount(c.Valid), c.Tokens)
		}
		t1 := time.Now()
		gotTower, n, err := enc.Tower(feats, c.T, c.Valid)
		if err != nil || n != c.Tokens {
			t.Fatalf("%s: tower tokens %d (want %d), err %v", c.Name, n, c.Tokens, err)
		}
		gotOut, _, _ := enc.Forward(feats, c.T, c.Valid)
		el := time.Since(t1).Seconds()
		cosT, absT := cosineRows(gotTower, tower, c.Tokens)
		cosO, absO := cosineRows(gotOut, out, c.Tokens)
		t.Logf("(a) %-9s %4d tokens (T %d): tower worst cosine %.8f (max |diff| %.2e), projected %.8f (%.2e); %.1fs for both passes", c.Name, c.Tokens, c.T, cosT, absT, cosO, absO, el)
		if cosT < 0.9999 || cosO < 0.9999 {
			t.Errorf("%s: worst per-token cosine %.8f (tower) / %.8f (projected), bar 0.9999", c.Name, cosT, cosO)
		}
		oneMinusCos[c.Name] = 1 - cosO
		// (b) the Go front end's features instead of the reference's.
		x := rd(c.Name + ".in.f32")
		gf, err := audio.QwenASRFeatures(x)
		if err != nil {
			t.Fatal(err)
		}
		o2, n2, err := enc.Forward(gf.Data, gf.T, gf.Valid)
		if err != nil || n2 != c.Tokens {
			t.Fatalf("%s: Go front end -> encoder gave %d tokens (err %v), want %d", c.Name, n2, err, c.Tokens)
		}
		cos2, abs2 := cosineRows(o2, out, c.Tokens)
		t.Logf("(b) %-9s Go front end -> encoder: projected worst cosine %.8f (max |diff| %.2e)", c.Name, cos2, abs2)
	}
	if a, b := oneMinusCos["long46s8"], oneMinusCos["synth12s"]; b > 0 && a > 10*b {
		t.Logf("NOTE: the 46.8 s clip's (1 - cosine) %.2e is more than 10x the 12 s clip's %.2e: the windowing's own error, recorded", a, b)
	}
}
