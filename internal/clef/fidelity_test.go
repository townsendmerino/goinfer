package clef

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// D13's fidelity harness (docs/measurements/decisions-d13-clef-fidelity-2026-10-03.md): one ARM of goinfer's Clef pipeline over the 150 D10 records, loaded through
// the same modelload path serve uses, one row per record written as it finishes, so a run is resumable and watchable. It is a RUN HARNESS, not a gate: without
// CLEF_FIDELITY_ARM it skips, and it asserts only that the rows it wrote are well formed. The verdict is the grader's (decisions-d13-clef-fidelity-2026-10/grade.py),
// against the reference rows scripts/pin_clef_d10.py wrote.
//
//	CLEF_FIDELITY_ARM    f32 | int8int8 | int4   (the quant the backbone loads at; the head is always f32)
//	CLEF_FIDELITY_OUT    the jsonl to append to (rows already there are skipped)
//	CLEF_MODEL_DIR       the Clef-flash directory (default ~/models/clef-flash), or a GGUF file of its backbone (the q4k arm). NEVER /srv/models or /Volumes: a run off the archive reads a 5400 rpm disk
//	CLEF_HEAD_DIR        the directory holding joint_head.safetensors and joint_head_config.json (default: CLEF_MODEL_DIR, which must then be that directory; a GGUF has no head, so the q4k arm sets it)
//	CLEF_FIDELITY_EVERY  take every Nth record (default 1), CLEF_FIDELITY_LIMIT stop after N new rows (default all)
//	CLEF_FIDELITY_WALL   wall-clock deadline in minutes (default 170), so a hang fails loudly
//
// Run with `go test -v -timeout 4h -run TestFidelityArm_run ./internal/clef/`. A heartbeat goes to stderr every 45 s: records and TOKENS done of total and elapsed.
// No time-remaining estimate on purpose: records differ in length, so the tokens-done fraction is what to read.
func TestFidelityArm_run(t *testing.T) {
	arm := os.Getenv("CLEF_FIDELITY_ARM")
	if arm == "" {
		t.Skip("CLEF_FIDELITY_ARM unset: this is a run harness, not a gate")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("CLEF_MODEL_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "clef-flash")
	}
	headDir := os.Getenv("CLEF_HEAD_DIR")
	if headDir == "" {
		headDir = dir
	}
	if strings.HasPrefix(headDir, "/srv/models") || strings.HasPrefix(headDir, "/Volumes/") {
		t.Fatalf("CLEF_HEAD_DIR %s is the archive, not the bench set", headDir)
	}
	if strings.HasPrefix(dir, "/srv/models") || strings.HasPrefix(dir, "/Volumes/") {
		t.Fatalf("CLEF_MODEL_DIR %s is the archive, not the bench set (CLAUDE.md, \"Models: stored in the archive, benchmarked from local disk\")", dir)
	}
	outPath := os.Getenv("CLEF_FIDELITY_OUT")
	if outPath == "" {
		t.Fatal("CLEF_FIDELITY_OUT is required")
	}
	envInt := func(k string, def int) int {
		if v := os.Getenv(k); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				t.Fatalf("%s=%q", k, v)
			}
			return n
		}
		return def
	}
	every, limit, wall := max(1, envInt("CLEF_FIDELITY_EVERY", 1)), envInt("CLEF_FIDELITY_LIMIT", 0), envInt("CLEF_FIDELITY_WALL", 170)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(wall)*time.Minute)
	defer cancel()

	recs := readJSONL[recordItem](t, "../../testdata/decisions/clef/records.jsonl")
	if len(recs) != 150 {
		t.Fatalf("records.jsonl has %d records, want 150", len(recs))
	}
	var work []recordItem
	for i, r := range recs {
		if i%every == 0 {
			work = append(work, r)
		}
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
	head, err := LoadHead(headDir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := NewModel(res.Model, head, TokenizerFunc(res.Tokenizer), 0)
	if err != nil {
		t.Fatal(err)
	}
	logf("loaded: quant %q, %d records to run (%d already in %s)", res.Model.Quant(), len(work)-countIn(work, done), countIn(work, done), outPath)

	f, err := os.OpenFile(outPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// Total tokens to run, from the encoder dump (the same ids), so the heartbeat can say how much of the work is done.
	enc := map[string]int{}
	for _, d := range readJSONL[dumpItem](t, "../../testdata/decisions/clef/encoder.jsonl") {
		enc[d.ID] = d.NTokens
	}
	var totalTok, doneTok, n int
	for _, r := range work {
		if !done[r.ID] {
			totalTok += enc[r.ID]
		}
	}
	lastBeat := time.Now()
	for _, r := range work {
		if done[r.ID] {
			continue
		}
		if limit > 0 && n >= limit {
			break
		}
		start := time.Now()
		out, err := m.Decide(ctx, r.Request)
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		if len(out.Answers) != 1 {
			t.Fatalf("%s: %d answers for one question", r.ID, len(out.Answers))
		}
		if out.InputTokens != enc[r.ID] {
			t.Errorf("%s: %d input tokens, the encoder dump has %d: a tokenizer or template drift would grade a different prompt", r.ID, out.InputTokens, enc[r.ID])
		}
		a := out.Answers[0]
		row, _ := json.Marshal(map[string]any{"id": r.ID, "arm": arm, "n_tokens": out.InputTokens, "option_ids": a.Options, "probs": a.Probs,
			"seconds": time.Since(start).Seconds(), "state_truncated": out.StateTruncated})
		if _, err := f.Write(append(row, '\n')); err != nil {
			t.Fatal(err)
		}
		n++
		doneTok += out.InputTokens
		if time.Since(lastBeat) >= 45*time.Second {
			lastBeat = time.Now()
			logf("heartbeat: %d records, %d of %d tokens (%.0f%%), %.0f ms/token so far", n, doneTok, totalTok, 100*float64(doneTok)/float64(max(1, totalTok)), 1000*time.Since(t0).Seconds()/float64(max(1, doneTok)))
		}
	}
	logf("arm %s done: %d new rows, %d tokens, %.0f s of run (%.0f ms/token including the load)", arm, n, doneTok, time.Since(t0).Seconds(), 1000*time.Since(t0).Seconds()/float64(max(1, doneTok)))
}

func countIn(work []recordItem, done map[string]bool) int {
	n := 0
	for _, r := range work {
		if done[r.ID] {
			n++
		}
	}
	return n
}
