package site

import (
	"regexp"
	"strings"
	"testing"
)

// The real data holds: every row transcribes the snapshot, every derived status equals the snapshot's, the shares
// recompute to its Result table, and the headline is a cited Fact. The three vision families the rule depends on come
// out supported, which is why the rule reads modality and not only tasks (gemma3's tasks do not say vision).
// No real row is T since the 2026-10-09 snapshot (mistral-small3.2 gained its images); the T derivation is pinned by
// TestCheckOllama_refuses' "the matrix moved (qwen3_5 loses its images)" case.
func TestCheckOllama_theRealDataHolds(t *testing.T) {
	in := realInputs(t)
	if err := CheckOllama(repoRoot, in); err != nil {
		t.Fatal(err)
	}
	m, err := Derive(in)
	if err != nil {
		t.Fatal(err)
	}
	status := map[string]string{}
	for _, e := range m.Ollama.Entries {
		status[e.Tag] = e.Status
	}
	for tag, want := range map[string]string{"gemma3": "S", "gemma4": "S", "qwen2.5vl": "S", "qwen3-vl": "S",
		"mistral-small3.2": "S", "nemotron-3-super": "U", "gemma2": "S", "codegemma": "S", "llava": "N", "nomic-embed-text": "S"} {
		if status[tag] != want {
			t.Errorf("%s: derived %q, want %q", tag, status[tag], want)
		}
	}
	if len(m.Ollama.Entries) != 60 || len(m.Ollama.Top()) != 20 || len(m.Ollama.Rest()) != 40 {
		t.Errorf("entries %d, top %d, rest %d", len(m.Ollama.Entries), len(m.Ollama.Top()), len(m.Ollama.Rest()))
	}
}

// Each gate can fail. A case corrupts one input and must be refused with a message naming the problem.
func TestCheckOllama_refuses(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		edit       func(in *Inputs)
	}{
		{"a mistyped pull count", "row 1 (llama3.1): pulls", func(in *Inputs) { in.Ollama.Rows[0].PullsM -= 0.4 }}, // relative: a refresh moves the printed count
		{"a family changed", "families", func(in *Inputs) { in.Ollama.Rows[4].Families = []string{"qwen3"} }},
		{"the matrix moved (qwen3_5 loses its images)", "new snapshot needed", func(in *Inputs) {
			for i := range in.Rows {
				if in.Rows[i].Name == "qwen3_5" {
					in.Rows[i].Tasks, in.Rows[i].Modality = []string{"chat"}, "text"
				}
			}
		}},
		{"the headline no longer a Fact", "not a Fact", func(in *Inputs) {
			in.Claims.Facts = nil
		}},
		{"a headline with another share", "does not carry the S share", func(in *Inputs) {
			// the first percentage in the headline is the supported share, whatever the current snapshot says: a refresh
			// must not break this case by changing the number (it hard-coded 92.4% until the 2026-10-02 snapshot)
			in.Ollama.Headline = regexp.MustCompile(`\d+\.\d%`).ReplaceAllString(in.Ollama.Headline, "99.9%")
		}},
		{"a row dropped", "rows, ollama.json", func(in *Inputs) { in.Ollama.Rows = in.Ollama.Rows[:59] }},
		// The case that slipped through before needs existed: support landing for a row that names no family.
		{"the matrix gains an N row's needs", "the matrix now has llava_next", func(in *Inputs) {
			in.Rows = append(in.Rows, Row{Name: "llava_next", Modality: "text (+ vision tower)", Tasks: []string{"chat", "vision"}})
		}},
		{"a mistyped needs", `needs "mllama" in the snapshot`, func(in *Inputs) {
			for i := range in.Ollama.Rows {
				if in.Ollama.Rows[i].Tag == "llama3.2-vision" {
					in.Ollama.Rows[i].Needs = "mllama2"
				}
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := realInputs(t)
			tc.edit(in)
			err := CheckOllama(repoRoot, in)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
	in := realInputs(t)
	in.Ollama.Rows[2].Families = []string{"encoder:word2vec"}
	if _, err := deriveOllama(in.Ollama, map[string]*Family{}); err == nil || !strings.Contains(err.Error(), "word2vec") {
		t.Fatalf("an unknown encoder kind: %v", err)
	}
}
