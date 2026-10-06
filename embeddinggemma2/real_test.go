//go:build realckpt

package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestReal_parity is Gate 2 (docs/tasks/task-embeddinggemma2.md): the real google/embeddinggemma-2 checkpoint
// (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2, never the archive) against the sentence-transformers reference
// that scripts/pin_embeddinggemma2_real.py records (GOINFER_EG2_GOLDEN, default testdata/embeddinggemma2-real). For each
// of its 48 texts, under the prompt the reference applied (query, document, none, two other named prompts; four
// documents longer than the sliding window): goinfer's input ids equal the reference's exactly, and its embedding has
// cosine >= 0.9999 with the reference's (the pre-registered bar). The regime: CPU, float32, not GPU-resident; the
// reference is sentence-transformers in float32 on the CPU.
func TestReal_parity(t *testing.T) {
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	golden := os.Getenv("GOINFER_EG2_GOLDEN")
	if golden == "" {
		golden = "../testdata/embeddinggemma2-real/golden.json"
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("no reference golden (scripts/pin_embeddinggemma2_real.py writes it): %v", err)
	}
	var g struct {
		ModelRevision string `json:"model_revision"`
		Items         []struct {
			Prompt    string    `json:"prompt"`
			Text      string    `json:"text"`
			IDs       []int     `json:"ids"`
			Embedding []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if rev, err := os.ReadFile(filepath.Join(dir, "REVISION")); err == nil && strings.TrimSpace(string(rev)) != g.ModelRevision {
		t.Fatalf("the checkpoint is revision %s, the golden was pinned on %s", strings.TrimSpace(string(rev)), g.ModelRevision)
	}
	t0 := time.Now()
	e, err := LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "[eg2 %6.1fs] loaded %s (revision %s), %d texts\n", time.Since(t0).Seconds(), dir, g.ModelRevision, len(g.Items))
	worst, longest := 1.0, 0
	for i, it := range g.Items {
		ids, err := e.Tokenize(it.Text, it.Prompt)
		if err != nil {
			t.Fatalf("item %d: %v", i, err)
		}
		if len(ids) != len(it.IDs) {
			t.Errorf("item %d (prompt %q): %d ids, reference %d", i, it.Prompt, len(ids), len(it.IDs))
			continue
		}
		for j := range ids {
			if ids[j] != it.IDs[j] {
				t.Errorf("item %d (prompt %q): id %d is %d, reference %d", i, it.Prompt, j, ids[j], it.IDs[j])
				break
			}
		}
		v, err := e.Model().Embed(it.IDs)
		if err != nil {
			t.Fatal(err)
		}
		var dot, nr, md float64
		for j, w := range it.Embedding {
			dot += float64(v[j]) * w
			nr += w * w
			md = math.Max(md, math.Abs(float64(v[j])-w))
		}
		cos := dot / math.Sqrt(nr)
		worst, longest = math.Min(worst, cos), max(longest, len(ids))
		fmt.Fprintf(os.Stderr, "[eg2 %6.1fs] item %2d prompt %-15q %5d tokens: cosine %.9f max|d| %.2e\n", time.Since(t0).Seconds(), i, it.Prompt, len(ids), cos, md)
		if cos < 0.9999 {
			t.Errorf("item %d (prompt %q, %d tokens): cosine %.9f, under 0.9999", i, it.Prompt, len(ids), cos)
		}
	}
	fmt.Fprintf(os.Stderr, "[eg2 %6.1fs] RESULT %d texts, worst cosine %.9f, longest %d tokens\n", time.Since(t0).Seconds(), len(g.Items), worst, longest)
}
