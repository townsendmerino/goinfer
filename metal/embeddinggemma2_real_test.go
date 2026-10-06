//go:build darwin

package metal

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// TestEG2Metal_realParity is Phase M's Gate 2 (docs/tasks/task-embeddinggemma2.md): the real google/embeddinggemma-2
// (GOINFER_EG2_DIR, default ~/models/embeddinggemma-2, never the archive) on the Metal accelerator against the
// committed sentence-transformers golden (testdata/embeddinggemma2-real/golden.json, 48 texts, 10 to 1,771 tokens):
// identical ids, and every embedding at cosine >= 0.9999 with the reference. It also times the Metal and the CPU
// forward on each text, alternating, in this process; that timing is exploratory (a by-day read, not a speed result).
//
//	GOINFER_EG2_REAL=1 go test -count=1 -run '^TestEG2Metal_realParity$' -v ./metal/
func TestEG2Metal_realParity(t *testing.T) {
	if os.Getenv("GOINFER_EG2_REAL") != "1" {
		t.Skip("set GOINFER_EG2_REAL=1 (loads the real checkpoint)")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_EG2_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "embeddinggemma-2")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", dir)
	}
	raw, err := os.ReadFile("../testdata/embeddinggemma2-real/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Items []struct {
			Prompt    string    `json:"prompt"`
			Text      string    `json:"text"`
			IDs       []int     `json:"ids"`
			Embedding []float64 `json:"embedding"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	cpu, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	gpu, err := embeddinggemma2.LoadEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gpu.UseAccelerator("metal"); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "[eg2m %6.1fs] loaded %s twice (CPU and Metal), %d texts\n", time.Since(t0).Seconds(), dir, len(g.Items))
	worst := 1.0
	var tg, tc float64
	type row struct {
		n    int
		g, c float64
	}
	var rows []row
	for i, it := range g.Items {
		ids, err := gpu.Tokenize(it.Text, it.Prompt)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(ids) != fmt.Sprint(it.IDs) {
			t.Fatalf("item %d: ids differ from the reference", i)
		}
		s := time.Now()
		v, _, err := gpu.EmbedText(it.Text, it.Prompt)
		if err != nil {
			t.Fatalf("item %d: metal: %v", i, err)
		}
		dg := time.Since(s).Seconds()
		s = time.Now()
		if _, _, err := cpu.EmbedText(it.Text, it.Prompt); err != nil {
			t.Fatal(err)
		}
		dc := time.Since(s).Seconds()
		tg, tc = tg+dg, tc+dc
		rows = append(rows, row{len(ids), dg, dc})
		var dot, nr float64
		for j, w := range it.Embedding {
			dot += float64(v[j]) * w
			nr += w * w
		}
		cos := dot / math.Sqrt(nr)
		worst = math.Min(worst, cos)
		fmt.Fprintf(os.Stderr, "[eg2m %6.1fs] item %2d %-15q %5d tokens: cosine %.9f; metal %.3f s, cpu %.3f s\n", time.Since(t0).Seconds(), i, it.Prompt, len(ids), cos, dg, dc)
		if cos < 0.9999 {
			t.Errorf("item %d (%d tokens): cosine %.9f, under 0.9999", i, len(ids), cos)
		}
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].n > rows[b].n })
	fmt.Fprintf(os.Stderr, "[eg2m %6.1fs] RESULT %d texts, worst cosine %.9f; exploratory timing: metal %.1f s, cpu %.1f s (%.1fx); longest %d tokens metal %.2f s, cpu %.2f s\n",
		time.Since(t0).Seconds(), len(g.Items), worst, tg, tc, tc/tg, rows[0].n, rows[0].g, rows[0].c)
}
