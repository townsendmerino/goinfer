package decoder

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestQwen35_textIdentityHashes is P8a's G3 instrument (docs/measurements/p8a-qwen35-vl-2026-09/
// preregistration.md): the TEXT path of a qwen3_5 / qwen3_5_moe model must be byte-identical
// before and after the image seam lands. It hashes the raw f32 bits of the logits at every
// prompt position and every greedy decode step, for fixed prompts on the two tiny fixtures and
// (when present) the real 0.8B, and either writes them (GOINFER_G3_WRITE=path — run on the
// PRE-change tree) or compares against a file written earlier (GOINFER_G3_CHECK=path). With
// neither set it skips, and a skip is not a pass.
func TestQwen35_textIdentityHashes(t *testing.T) {
	write, check := os.Getenv("GOINFER_G3_WRITE"), os.Getenv("GOINFER_G3_CHECK")
	if write == "" && check == "" {
		t.Skip("set GOINFER_G3_WRITE=path (pre-change tree) or GOINFER_G3_CHECK=path (post-change tree)")
	}
	models := []struct{ name, dir string }{
		{"tiny-dense", "testdata/qwen3_5-tiny"},
		{"tiny-moe", "testdata/qwen3_5_moe-tiny"},
		{"real-0.8b", filepath.Join(os.Getenv("HOME"), "models", "qwen3.5-0.8b")},
	}
	got := map[string][]string{}
	for _, mc := range models {
		if _, err := os.Stat(filepath.Join(mc.dir, "config.json")); err != nil {
			t.Logf("%s: no checkpoint at %s — not hashed", mc.name, mc.dir)
			continue
		}
		m, err := Load(mc.dir, Options{})
		if err != nil {
			t.Fatalf("Load(%s): %v", mc.dir, err)
		}
		vocab := m.w.arch.VocabSize
		for pi, prompt := range [][]int{{3, 9, 27, 81, 5, 11}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}} {
			cache := m.NewCache(len(prompt) + 12)
			var hs []string
			var logits []float32
			for _, id := range prompt {
				id %= vocab
				l, err := m.forward(id, cache)
				if err != nil {
					t.Fatalf("%s forward: %v", mc.name, err)
				}
				hs = append(hs, hashF32(l))
				logits = l
			}
			for range 12 { // greedy decode
				best := 0
				for i, v := range logits {
					if v > logits[best] {
						best = i
					}
				}
				l, err := m.forward(best, cache)
				if err != nil {
					t.Fatalf("%s decode forward: %v", mc.name, err)
				}
				hs = append(hs, hashF32(l))
				logits = l
			}
			got[mc.name+"/p"+string(rune('0'+pi))] = hs
		}
		m.Close()
	}
	if len(got) == 0 {
		t.Fatal("no model was hashed — nothing to compare")
	}
	if write != "" {
		b, _ := json.MarshalIndent(got, "", " ")
		if err := os.WriteFile(write, b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d hash lists to %s", len(got), write)
		return
	}
	raw, err := os.ReadFile(check)
	if err != nil {
		t.Fatal(err)
	}
	var want map[string][]string
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	for k, w := range want {
		g, ok := got[k]
		if !ok {
			t.Errorf("%s: in baseline but not hashed now (checkpoint missing?)", k)
			continue
		}
		if len(g) != len(w) {
			t.Errorf("%s: %d hashes, baseline %d", k, len(g), len(w))
			continue
		}
		for i := range w {
			if g[i] != w[i] {
				t.Errorf("%s step %d: logits bits differ from the pre-change baseline", k, i)
				break
			}
		}
		t.Logf("%s: %d steps byte-identical to baseline", k, len(w))
	}
}

func hashF32(v []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, f := range v {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
