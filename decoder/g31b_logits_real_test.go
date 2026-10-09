//go:build realckpt

// Step (b') of docs/tasks/task-multimodal-support-2026-10.md ("Gemma 4 31B, step (b')", registered before this code): goinfer's half. For each arm (int8int8, int4; CPU) it teacher-forces the 12 prompts of
// quant_consistency_real_test.go (qcPrompts, through the checkpoint's own chat template) along the continuation recorded by G-31a (GOINFER_G31B_PATHS, the Paths of qc-31b.json or qc-e4b.json: goinfer's own
// int8int8 greedy path, stop token included) and writes the float32 logits at every continuation position, [positions, vocab], beside the id sequences the Hugging Face side must use
// (scripts/g31b_hf_stream.py reads <out>.seqs.json, so both sides run the same ids). Nothing is graded here (scripts/g31b_grade.py).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_G31B_DIR=<checkpoint dir> GOINFER_G31B_PATHS=<qc json> GOINFER_G31B_OUT=<prefix> [GOINFER_G31B_ARMS=int8int8,int4] [GOINFER_G31B_INT8=<bundle>] [GOINFER_G31B_INT4=<bundle>]
//	  go test -tags realckpt ./decoder/ -run TestG31b_goinferLogits -v -timeout 150m
package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestG31b_goinferLogits(t *testing.T) {
	requireHeavyModel(t)
	dir, pathsFile, out := os.Getenv("GOINFER_G31B_DIR"), os.Getenv("GOINFER_G31B_PATHS"), os.Getenv("GOINFER_G31B_OUT")
	if dir == "" || out == "" {
		t.Skip("set GOINFER_G31B_DIR, GOINFER_G31B_PATHS (not needed for a tiny fixture) and GOINFER_G31B_OUT")
	}
	for _, p := range []string{dir, os.Getenv("GOINFER_G31B_INT8"), os.Getenv("GOINFER_G31B_INT4"), out} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is the archive (CLAUDE.md)", p)
		}
	}
	arms := strings.Split(os.Getenv("GOINFER_G31B_ARMS"), ",")
	if arms[0] == "" {
		arms = []string{"int8int8", "int4"}
	}
	type seq struct {
		IDs  []int `json:"ids"`
		Path []int `json:"path"`
	}
	var seqs []seq
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil { // a tiny fixture: the plumbing control, with synthetic ids and a synthetic path, labelled as such and never a result
		fmt.Fprintf(os.Stderr, "[g31b] SYNTHETIC sequences: no tokenizer.json in %s; this is a plumbing control, not a result\n", dir)
		for i := range qcPrompts {
			ids := make([]int, 12+i)
			for j := range ids {
				ids[j] = 3 + (i*131+j*17)%50
			}
			path := make([]int, 5+i%3)
			for j := range path {
				path[j] = 3 + (i*7+j*13)%50
			}
			seqs = append(seqs, seq{ids, path})
		}
	} else {
		raw, err := os.ReadFile(pathsFile)
		if err != nil {
			t.Fatal(err)
		}
		var qc struct{ Paths [][]int }
		if err := json.Unmarshal(raw, &qc); err != nil || len(qc.Paths) != len(qcPrompts) {
			t.Fatalf("paths file: %v (%d paths, want %d)", err, len(qc.Paths), len(qcPrompts))
		}
		tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		tm, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
		if err != nil {
			t.Fatal(err)
		}
		for i, p := range qcPrompts {
			ids, err := tk.EncodeSegments(tm.RenderSegments("", []chat.Turn{{Role: "user", Content: p}}), false)
			if err != nil {
				t.Fatal(err)
			}
			seqs = append(seqs, seq{ids, qc.Paths[i]})
		}
	}
	b, _ := json.Marshal(seqs)
	if err := os.WriteFile(out+".seqs.json", b, 0o644); err != nil {
		t.Fatal(err)
	}
	T0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[g31b %5.0fs] %s\n", time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	}
	ctx := context.Background()
	for _, arm := range arms {
		path, opts := dir, Options{Backend: "cpu", Quant: arm, EmbedInt4: arm == "int4"} // as G-31a: serve's int4 default on the CPU; a bundle carries its own quant
		switch {
		case arm == "int8int8" && os.Getenv("GOINFER_G31B_INT8") != "":
			path, opts = os.Getenv("GOINFER_G31B_INT8"), Options{Backend: "cpu"}
		case arm == "int4" && os.Getenv("GOINFER_G31B_INT4") != "":
			path, opts = os.Getenv("GOINFER_G31B_INT4"), Options{Backend: "cpu"}
		}
		t0 := time.Now()
		m, err := Load(path, opts)
		if err != nil {
			t.Fatalf("%s load: %v", arm, err)
		}
		hb("%s loaded in %.0fs (%s; head table %s)", arm, time.Since(t0).Seconds(), m.DecodePath(), m.HeadTable())
		f, err := os.Create(out + "." + arm + ".f32")
		if err != nil {
			t.Fatal(err)
		}
		positions, nonFinite := 0, 0
		t0 = time.Now()
		for i, s := range seqs {
			cache := m.NewCache(len(s.IDs) + len(s.Path) + 1)
			l, err := m.prefillLogits(ctx, s.IDs, cache)
			if err != nil {
				t.Fatal(err)
			}
			for j, tok := range s.Path {
				for _, v := range l {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						nonFinite++
						break
					}
				}
				if err := binary.Write(f, binary.LittleEndian, l); err != nil {
					t.Fatal(err)
				}
				positions++
				if j < len(s.Path)-1 { // the stop token is the last path entry: its logits are written, nothing follows it
					if l, err = m.forward(tok, cache); err != nil {
						t.Fatal(err)
					}
				}
			}
			hb("%s prompt %d/%d: %d positions done (%.1fs per position so far)", arm, i+1, len(seqs), len(s.Path), time.Since(t0).Seconds()/float64(positions))
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		m.Close()
		hb("%s: %d positions written to %s.%s.f32; non-finite rows %d", arm, positions, out, arm, nonFinite)
		if nonFinite > 0 {
			t.Errorf("%s: %d non-finite logit rows", arm, nonFinite)
		}
	}
}
