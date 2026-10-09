package multimodal

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S14e1, the prompt half (docs/tasks/task-multimodal-support-2026-10.md): the transcription-request ids mistral_common 1.12.0 builds for 2 s, 35 s and 65 s clips (1, 2 and 3 thirty-second
// chunks) with language "en", "fr" and none, pinned by scripts/pin_tekken.py. The tekken.json is not committed; the test reads it from the Voxtral Mini directory under
// GOINFER_MODELS_DIR (default ~/models) and skips without it, and refuses a file whose sha256 is not the golden's.

type voxtralGolden struct {
	SHA      string `json:"tekken_sha256"`
	Requests []struct {
		Seconds      float64 `json:"seconds"`
		Language     *string `json:"language"`
		IDs          []int   `json:"ids"`
		AudioSamples []int   `json:"audio_samples"`
	} `json:"requests"`
}

func loadVoxtral(t *testing.T) (*tokenizer.Tokenizer, *voxtralGolden) {
	t.Helper()
	f, err := os.Open(filepath.Join("..", "testdata", "tekken_golden.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g voxtralGolden
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		home, _ := os.UserHomeDir()
		root = filepath.Join(home, "models")
	}
	raw, err := os.ReadFile(filepath.Join(root, "voxtral-mini-3b-2507", "tekken.json"))
	if err != nil {
		t.Skipf("no tekken.json under %s: %v", root, err)
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != g.SHA {
		t.Fatalf("tekken.json sha256 %s is not the golden's %s", got, g.SHA)
	}
	tk, err := tokenizer.LoadTekken(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tk, &g
}

func lang(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func TestVoxtralPrompt_matchesMistralCommon(t *testing.T) {
	tk, g := loadVoxtral(t)
	if len(g.Requests) != 9 {
		t.Fatalf("golden has %d requests, want 9", len(g.Requests))
	}
	for _, r := range g.Requests {
		samples := int(r.Seconds * 16000)
		// mistral_common returns ONE audio array per request, padded up to a whole number of 30 s windows.
		total := 0
		for _, n := range r.AudioSamples {
			if n%VoxtralChunkSamples != 0 {
				t.Errorf("%.0f s: mistral_common's audio is %d samples, not a multiple of %d", r.Seconds, n, VoxtralChunkSamples)
			}
			total += n
		}
		if got, want := VoxtralChunks(samples), total/VoxtralChunkSamples; got != want {
			t.Errorf("%.0f s: VoxtralChunks = %d, mistral_common padded the audio to %d window(s)", r.Seconds, got, want)
		}
		ids, pos, err := VoxtralPrompt(tk, VoxtralAudioTokens(samples), lang(r.Language))
		if err != nil {
			t.Fatal(err)
		}
		if !equal(ids, r.IDs) {
			t.Errorf("%.0f s lang %q: ids differ\n got %v\nwant %v", r.Seconds, lang(r.Language), edges(ids), edges(r.IDs))
		}
		if audio, _ := tk.TokenID("[AUDIO]"); ids[pos] != audio || ids[pos-1] == audio {
			t.Errorf("%.0f s: audioPos %d does not point at the first [AUDIO]", r.Seconds, pos)
		}
	}
}

func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func edges(a []int) []int {
	if len(a) <= 14 {
		return a
	}
	return append(append(append([]int(nil), a[:5]...), -1), a[len(a)-8:]...)
}

// Each plausible mistake in the builder must put at least one request over the bar.
func TestVoxtralPrompt_plantedDefects(t *testing.T) {
	tk, g := loadVoxtral(t)
	cases := []struct {
		name string
		d    voxtralPromptDefects
	}{
		{"no BOS", voxtralPromptDefects{noBOS: true}},
		{"[INST] and [/INST] swapped", voxtralPromptDefects{swapInst: true}},
		{"language without the colon", voxtralPromptDefects{noColon: true}},
		{"one placeholder too few", voxtralPromptDefects{countDiff: -1}},
		{"one placeholder too many", voxtralPromptDefects{countDiff: +1}},
		{"no [TRANSCRIBE]", voxtralPromptDefects{noTranscr: true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bad := 0
			for _, r := range g.Requests {
				ids, _, err := voxtralPrompt(tk, VoxtralAudioTokens(int(r.Seconds*16000)), lang(r.Language), c.d)
				if err != nil {
					t.Fatal(err)
				}
				if !equal(ids, r.IDs) {
					bad++
				}
			}
			if bad == 0 {
				t.Errorf("the planted defect %q matches every golden request: the gate is blind to it", c.name)
			}
			t.Logf("%d of %d requests differ", bad, len(g.Requests))
		})
	}
	t.Run("chunk count by floor instead of ceil", func(t *testing.T) {
		bad := 0
		for _, r := range g.Requests {
			if samples := int(r.Seconds * 16000); samples/VoxtralChunkSamples != VoxtralChunks(samples) {
				bad++
			}
		}
		if bad == 0 {
			t.Error("floor and ceil agree on every golden clip length: the golden cannot tell them apart")
		}
	})
}
