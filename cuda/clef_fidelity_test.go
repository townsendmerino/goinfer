//go:build cuda

package cuda

import (
	"bufio"
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
	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// D13's CUDA-resident arm (docs/measurements/decisions-d13-clef-fidelity-2026-10-03.md, amendment 5b): the same run harness as internal/clef's TestFidelityArm_run, but with the
// backbone loaded on the CUDA backend so PromptHiddenAll answers from the device (decoder.ResidentResidualAll; decisions-d11-resident-hidden-2026-10-03.md). It lives here because
// the root module may not depend on this one, while this module may import the root's internal packages (the rule is by import path). Rows are full precision, in the same format
// the CPU arms write, so decisions-d13-clef-fidelity-2026-10/grade.py grades them with the same reference.
//
// A RUN HARNESS: without CLEF_FIDELITY_ARM it skips. The arm must be int4 (the 9B at int8int8 is 9.5 GB and does not fit an 8 GB card). The run is VOID, and the harness fails, if the model did
// not load onto the device ("cuda-resident" must appear in Model.DecodePath): a CPU fallback must not be graded as a GPU result (D6b's amendment 2). Environment as the CPU harness's
// (CLEF_FIDELITY_OUT, CLEF_MODEL_DIR, CLEF_FIDELITY_EVERY, CLEF_FIDELITY_LIMIT, CLEF_FIDELITY_WALL).
//
//	cd cuda && go test -c -tags cuda -o clef-fidelity-cuda.test . && CLEF_FIDELITY_ARM=int4 CLEF_FIDELITY_OUT=... ./clef-fidelity-cuda.test -test.run 'TestClefFidelityCUDA_run$' -test.v
func TestClefFidelityCUDA_run(t *testing.T) {
	arm := os.Getenv("CLEF_FIDELITY_ARM")
	if arm == "" {
		t.Skip("CLEF_FIDELITY_ARM unset: this is a run harness, not a gate")
	}
	if arm != "int4" {
		t.Fatalf("CLEF_FIDELITY_ARM=%q: the CUDA arm is int4 (the 9B at int8int8 does not fit an 8 GB card)", arm)
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("CLEF_MODEL_DIR")
	if dir == "" {
		dir = filepath.Join(home, "models", "clef-flash")
	}
	if strings.HasPrefix(dir, "/srv/models") || strings.HasPrefix(dir, "/Volumes/") {
		t.Fatalf("CLEF_MODEL_DIR %s is the archive, not the bench set", dir)
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
	every, limit, wall := max(1, envInt("CLEF_FIDELITY_EVERY", 1)), envInt("CLEF_FIDELITY_LIMIT", 0), envInt("CLEF_FIDELITY_WALL", 60)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(wall)*time.Minute)
	defer cancel()

	type rec struct {
		ID      string          `json:"id"`
		Request json.RawMessage `json:"request"`
	}
	type dump struct {
		ID      string `json:"id"`
		NTokens int    `json:"n_tokens"`
	}
	readJSONL := func(path string, each func(line []byte)) {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("no fixture %s (a skip would hide this): %v", path, err)
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		for sc.Scan() {
			each(append([]byte(nil), sc.Bytes()...))
		}
	}
	var recs []rec
	readJSONL("../testdata/decisions/clef/records.jsonl", func(l []byte) {
		var r rec
		if err := json.Unmarshal(l, &r); err != nil {
			t.Fatal(err)
		}
		recs = append(recs, r)
	})
	if len(recs) != 150 {
		t.Fatalf("records.jsonl has %d records, want 150", len(recs))
	}
	enc := map[string]int{}
	readJSONL("../testdata/decisions/clef/encoder.jsonl", func(l []byte) {
		var d dump
		if err := json.Unmarshal(l, &d); err != nil {
			t.Fatal(err)
		}
		enc[d.ID] = d.NTokens
	})
	done := map[string]bool{}
	if b, err := os.ReadFile(outPath); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			var row struct{ ID string }
			if json.Unmarshal([]byte(l), &row) == nil && row.ID != "" {
				done[row.ID] = true
			}
		}
	}

	// --embed-int4 is ON by default in serve and the CLIs (internal/loadflags) and applies only with -quant int4: it stores the token-embedding/LM-head table at int4 instead of the int8 pin.
	// The Clef head reads LM-head rows, so the flag changes the int4 arms' answers; the arms are graded as SERVED (on) unless CLEF_FIDELITY_EMBED_INT4=0 asks for the int8 pin, which the
	// record then names. It is ignored for every other arm.
	embedInt4 := arm == "int4" && os.Getenv("CLEF_FIDELITY_EMBED_INT4") != "0"
	t0 := time.Now()
	logf := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s +%s] %s\n", time.Now().Format("15:04:05"), time.Since(t0).Round(time.Second), fmt.Sprintf(f, a...))
	}
	logf("arm %s on cuda: loading %s through modelload (the path serve uses)", arm, dir)
	res, err := modelload.Load(ctx, modelload.Request{Spec: dir, Opts: decoder.Options{Backend: "cuda", Quant: arm, EmbedInt4: embedInt4}, ExplicitQuant: arm, GuardAdvice: "use a smaller quant"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer res.Model.Close()
	path := res.Model.DecodePath()
	logf("loaded: quant %q, decode path %q", res.Model.Quant(), path)
	if !strings.Contains(path, "cuda-resident") {
		t.Fatalf("VOID: the model did not load onto the device (decode path %q; decline: %s): a CPU fallback must not be graded as a GPU result", path, res.Model.ResidentDecline())
	}
	head, err := clef.LoadHead(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := clef.NewModel(res.Model, head, clef.TokenizerFunc(res.Tokenizer), 0)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(outPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var n, doneTok, totalTok int
	var work []rec
	for i, r := range recs {
		if i%every == 0 {
			work = append(work, r)
			if !done[r.ID] {
				totalTok += enc[r.ID]
			}
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
			t.Errorf("%s: %d input tokens, the encoder dump has %d", r.ID, out.InputTokens, enc[r.ID])
		}
		a := out.Answers[0]
		row, _ := json.Marshal(map[string]any{"id": r.ID, "arm": "cuda-" + arm, "n_tokens": out.InputTokens, "option_ids": a.Options, "probs": a.Probs,
			"seconds": time.Since(start).Seconds(), "state_truncated": out.StateTruncated, "embed_int4": embedInt4, "decode_path": path})
		if _, err := f.Write(append(row, '\n')); err != nil {
			t.Fatal(err)
		}
		n++
		doneTok += out.InputTokens
		if time.Since(lastBeat) >= 30*time.Second {
			lastBeat = time.Now()
			logf("heartbeat: %d records, %d of %d tokens (%.0f%%)", n, doneTok, totalTok, 100*float64(doneTok)/float64(max(1, totalTok)))
		}
	}
	logf("arm cuda-%s done: %d new rows, %d tokens, %.0f s including the load", arm, n, doneTok, time.Since(t0).Seconds())
}
