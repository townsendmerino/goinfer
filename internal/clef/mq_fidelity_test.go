package clef

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// The multi-question run harness (D13 addendum, decisions-d13-clef-multiquestion-2026-10-03.md): goinfer's Clef pipeline over the five-question records scripts/pin_clef_mq.py built
// and referenced, one row per record with every question's probabilities. Not a gate: with CLEF_MQ_ARM unset it skips, and the verdict is the grader's (grade_mq.py).
//
//	CLEF_MQ_ARM      f32 | int8int8   (the quant the backbone loads at; the head is always f32)
//	CLEF_MQ_DIR      the directory holding records.jsonl and probs_f32.jsonl from pin_clef_mq.py
//	CLEF_MQ_OUT      the jsonl to append to (rows already there are skipped)
//	CLEF_MODEL_DIR   the checkpoint (default ~/models/clef-flash; never the archive)
//	CLEF_MQ_WALL     wall-clock deadline in minutes (default 25)
func TestMultiQuestionArm_run(t *testing.T) {
	arm := os.Getenv("CLEF_MQ_ARM")
	if arm == "" {
		t.Skip("CLEF_MQ_ARM unset: this is a run harness, not a gate")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("CLEF_MODEL_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "clef-flash")
	}
	if strings.HasPrefix(dir, "/srv/models") || strings.HasPrefix(dir, "/Volumes/") {
		t.Fatalf("CLEF_MODEL_DIR %s is the archive, not the bench set", dir)
	}
	mqDir, outPath := os.Getenv("CLEF_MQ_DIR"), os.Getenv("CLEF_MQ_OUT")
	if mqDir == "" || outPath == "" {
		t.Fatal("CLEF_MQ_DIR and CLEF_MQ_OUT are required")
	}
	wall := 25
	if v := os.Getenv("CLEF_MQ_WALL"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &wall); err != nil {
			t.Fatalf("CLEF_MQ_WALL=%q", v)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(wall)*time.Minute)
	defer cancel()

	recs := readJSONL[recordItem](t, filepath.Join(mqDir, "records.jsonl"))
	type refQ struct {
		ID        string   `json:"id"`
		OptionIDs []string `json:"option_ids"`
	}
	type refRow struct {
		ID        string `json:"id"`
		NTokens   int    `json:"n_tokens"`
		Questions []refQ `json:"questions"`
	}
	ref := map[string]refRow{}
	for _, r := range readJSONL[refRow](t, filepath.Join(mqDir, "probs_f32.jsonl")) {
		ref[r.ID] = r
	}
	done := map[string]bool{}
	if b, err := os.ReadFile(outPath); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			var row struct{ ID string }
			if json.Unmarshal([]byte(l), &row) == nil && row.ID != "" {
				done[row.ID] = true
			}
		}
	}
	t0 := time.Now()
	logf := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s +%s] %s\n", time.Now().Format("15:04:05"), time.Since(t0).Round(time.Second), fmt.Sprintf(f, a...))
	}
	logf("arm %s: loading %s through modelload (the path serve uses)", arm, dir)
	res, err := modelload.Load(ctx, modelload.Request{
		Spec: dir, Opts: decoder.Options{Backend: "cpu", Quant: arm}, ExplicitQuant: arm, GuardAdvice: "use a smaller quant",
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer res.Model.Close()
	head, err := LoadHead(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewModel(res.Model, head, TokenizerFunc(res.Tokenizer), 0)
	if err != nil {
		t.Fatal(err)
	}
	logf("loaded: quant %q, %d records", res.Model.Quant(), len(recs))
	f, err := os.OpenFile(outPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for i, r := range recs {
		if done[r.ID] {
			continue
		}
		want, ok := ref[r.ID]
		if !ok {
			t.Fatalf("%s: no reference row", r.ID)
		}
		start := time.Now()
		out, err := m.Decide(ctx, r.Request)
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		if len(out.Answers) != len(want.Questions) {
			t.Fatalf("%s: %d answers, the reference has %d questions", r.ID, len(out.Answers), len(want.Questions))
		}
		if out.InputTokens != want.NTokens {
			t.Errorf("%s: %d input tokens, the reference encoded %d", r.ID, out.InputTokens, want.NTokens)
		}
		qs := make([]map[string]any, len(out.Answers))
		for j, a := range out.Answers {
			if a.ID != want.Questions[j].ID || strings.Join(a.Options, "\x00") != strings.Join(want.Questions[j].OptionIDs, "\x00") {
				t.Errorf("%s question %d: id/options %q %v, the reference has %q %v", r.ID, j, a.ID, a.Options, want.Questions[j].ID, want.Questions[j].OptionIDs)
			}
			qs[j] = map[string]any{"id": a.ID, "option_ids": a.Options, "probs": a.Probs}
		}
		row, _ := json.Marshal(map[string]any{"id": r.ID, "arm": arm, "n_tokens": out.InputTokens, "questions": qs, "seconds": time.Since(start).Seconds(), "state_truncated": out.StateTruncated})
		if _, err := f.Write(append(row, '\n')); err != nil {
			t.Fatal(err)
		}
		logf("[%d/%d] %s: %d tokens, %d questions, %.0f s", i+1, len(recs), r.ID, out.InputTokens, len(out.Answers), time.Since(start).Seconds())
	}
	logf("arm %s done", arm)
}
