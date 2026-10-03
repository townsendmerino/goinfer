package decide

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// D8 on REAL weights (docs/tasks/task-constrained-confidence.md): D7's five questions about one state, as the two decision templates build them, through
// decoder.PromptHiddenMany and through one PromptHidden each. It reports (1) how much the shared answer differs from the separate one on a 32-layer 9B model's
// final-norm hidden state, and (2) how long each took. The timing is an EXPLORATORY single run by day and is never quoted as a result: it shows whether sharing
// pays on the real thing and what it shares, not by how much.
//
//	D8_MODEL   a Qwen3.5-family directory with a tokenizer (default ~/models/clef-flash), loaded at int8int8 on the CPU
//
// Skips unless GOINFER_HEAVY_TESTS=1. Estimated 4-6 minutes at K=256; a heartbeat goes to stderr as each prompt finishes.
func TestPromptHiddenMany_realWeights(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("GOINFER_HEAVY_TESTS=1 not set")
	}
	dir := os.Getenv("D8_MODEL")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "clef-flash")
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.CanSharePrefix() {
		t.Fatal("the real model does not take the checkpoint path")
	}
	raw, err := os.ReadFile("../../docs/measurements/decisions-d7-2026-09-28/prompts.json")
	if err != nil {
		t.Fatal(err)
	}
	var P struct {
		States map[string][]struct {
			Text string `json:"text"`
		} `json:"states"`
	}
	if err := json.Unmarshal(raw, &P); err != nil {
		t.Fatal(err)
	}
	state := P.States["256"][0].Text
	type q struct {
		kind, text string
		opts, desc []string
	}
	qs := []q{
		{KindNoul, "Is the customer asking for a refund?", noulOptions, nil},
		{KindScore, "How frustrated is the customer?", scoreOptions, nil},
		{KindChoice, "Which team should handle this?", []string{"billing", "technical", "sales", "account"}, nil},
		{KindNoul, "Does the customer mention a deadline?", noulOptions, nil},
		{KindChoice, "How should the customer be contacted?", []string{"email", "phone", "chat", "none"}, nil},
	}
	pt := NewPlainTokenizer(tk)
	ctx := context.Background()
	t0 := time.Now()
	beat := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s +%s] %s\n", time.Now().Format("15:04:05"), time.Since(t0).Round(time.Second), fmt.Sprintf(f, a...))
	}
	for _, tmpl := range []string{TemplateBare, TemplateChat} {
		var prompts [][]int
		for _, x := range qs {
			var ids []int
			var err error
			if tmpl == TemplateChat {
				ids, err = pt.EncodeChat(RenderChat(x.kind, state, x.text, x.opts, x.desc...))
			} else {
				ids, err = pt.EncodePlain(Render(x.kind, state, x.text, x.opts, x.desc...))
			}
			if err != nil {
				t.Fatal(err)
			}
			prompts = append(prompts, ids)
		}
		groups, plen := shareGroupsForReport(prompts)
		total := 0
		for _, p := range prompts {
			total += len(p)
		}
		beat("%s: %d prompts, %d tokens in all; groups %v with shared prefixes %v", tmpl, len(prompts), total, groups, plen)

		tS := time.Now()
		shared, err := m.PromptHiddenMany(ctx, prompts)
		if err != nil {
			t.Fatal(err)
		}
		dShared := time.Since(tS)
		beat("%s: shared %.1f s", tmpl, dShared.Seconds())
		tA := time.Now()
		worst := 0.0
		for i, p := range prompts {
			h, err := m.PromptHidden(ctx, p)
			if err != nil {
				t.Fatal(err)
			}
			var ne, nb float64
			for j := range h {
				d := float64(shared[i][j]) - float64(h[j])
				ne, nb = ne+d*d, nb+float64(h[j])*float64(h[j])
			}
			worst = math.Max(worst, math.Sqrt(ne/nb))
			beat("%s: prompt %d alone done (%d tokens), relative L2 shared vs alone %.3g", tmpl, i, len(p), math.Sqrt(ne/nb))
		}
		dAlone := time.Since(tA)
		t.Logf("%s (K=256, 5 questions, EXPLORATORY): shared %.1f s, separate %.1f s (%.2fx); worst relative L2 on the final-norm hidden state %.3g", tmpl, dShared.Seconds(), dAlone.Seconds(), dAlone.Seconds()/dShared.Seconds(), worst)
		if worst > 1e-4 {
			t.Errorf("%s: worst relative L2 %.3g on the real model exceeds 1e-4", tmpl, worst)
		}
	}
}

// shareGroupsForReport mirrors decoder.shareGroups's grouping for the log line only (its function is unexported): groups by a shared prefix of at least 16 tokens.
func shareGroupsForReport(prompts [][]int) (members [][]int, prefix []int) {
	assigned := make([]bool, len(prompts))
	lcp := func(a, b []int) int {
		n := min(len(a), len(b))
		i := 0
		for i < n && a[i] == b[i] {
			i++
		}
		return i
	}
	for i := range prompts {
		if assigned[i] {
			continue
		}
		g, p := []int{i}, len(prompts[i])-1
		assigned[i] = true
		for j := i + 1; j < len(prompts); j++ {
			if l := min(lcp(prompts[i], prompts[j]), len(prompts[j])-1); !assigned[j] && l >= 16 && min(p, l) >= 16 {
				p = min(p, l)
				g = append(g, j)
				assigned[j] = true
			}
		}
		if len(g) == 1 {
			p = 0
		}
		members, prefix = append(members, g), append(prefix, p)
	}
	return members, prefix
}
