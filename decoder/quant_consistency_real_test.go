//go:build realckpt

// G-31a2 of docs/tasks/task-multimodal-support-2026-10.md ("Gemma 4 31B on nobara ... G-31a"): how well a checkpoint's int4 CPU decode agrees with its OWN
// int8int8 CPU decode. 12 fixed chat prompts go through the checkpoint's real chat template (quantisation is judged only through the template); the int8int8 arm decodes up to 32 greedy tokens
// per prompt (stopping at a stop token), recording its path and, at every step, the margin between its top-1 and top-2 probabilities; the int4 arm is then teacher-forced along that path. The statistic is
// the fraction of positions where the int4 argmax equals the int8int8 argmax; every logit must be finite. The arms are loaded one after the other, never together (the 31B's int8int8 arm is 31 GB).
// Writes $GOINFER_QC_OUT (JSON) for the script that grades it against the sibling checkpoint's, and prints a heartbeat per prompt. A measurement; the verdict is in the night script.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QC_DIR=<checkpoint dir> [GOINFER_QC_INT8=<int8int8 bundle>] [GOINFER_QC_INT4=<bundle or dir for the int4 arm>] GOINFER_QC_OUT=<json> GOINFER_QC_LABEL=<name> \
//	  go test -tags realckpt ./decoder/ -run TestQuantConsistency_real -v -timeout 150m
//
// With no tokenizer.json in GOINFER_QC_DIR (a tiny fixture) the prompts are synthetic id sequences: a plumbing control, labelled as such in the output, never a result.
package decoder

import (
	"context"
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

var qcPrompts = []string{
	"Explain in two or three sentences why the sky is blue.",
	"Write a Python function that returns the n-th Fibonacci number, with a short explanation.",
	"What is the capital of Australia, and what is one interesting fact about it?",
	"Summarize the plot of Romeo and Juliet in a short paragraph.",
	"Give me three tips for learning a new language quickly.",
	"Translate 'Where is the nearest train station?' into French, Spanish and German.",
	"If a train travels 120 km in 1.5 hours, what is its average speed? Show the calculation.",
	"List four differences between a process and a thread in an operating system.",
	"Write a haiku about autumn rain.",
	"Describe how a binary search works and state its time complexity.",
	"What are the main causes of inflation? Answer in a short paragraph.",
	"Write a polite email asking a colleague to reschedule a meeting to next Tuesday.",
}

type qcSplit struct {
	Prompt, Step int
	Ref, Got     int
	RefMargin    float64 // int8int8's top-1 minus top-2 probability at this step
	RefTop       float64 // int8int8's probability of its own top-1
}

type qcResult struct {
	Label, Dir, Int4Path string
	Synthetic            bool
	Prompts              int
	Positions, Agree     int
	Agreement            float64
	NonFinite            int
	Splits               []qcSplit
	Paths                [][]int
	Texts                []string
	LoadInt8Sec          float64
	LoadInt4Sec          float64
	DecodeInt8SecPerTok  float64
	DecodeInt4SecPerTok  float64
}

func TestQuantConsistency_real(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("GOINFER_QC_DIR")
	if dir == "" {
		t.Skip("set GOINFER_QC_DIR to a checkpoint directory (or a tiny fixture for the plumbing control)")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive (CLAUDE.md)", dir)
	}
	int4Path := os.Getenv("GOINFER_QC_INT4")
	if int4Path == "" {
		int4Path = dir
	}
	out := os.Getenv("GOINFER_QC_OUT")
	label := os.Getenv("GOINFER_QC_LABEL")
	if label == "" {
		label = filepath.Base(dir)
	}
	const steps = 32
	T0 := time.Now()
	hb := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[qc %s %5.0fs] %s\n", label, time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	}

	// Prompts as ids.
	var prompts [][]int
	res := qcResult{Label: label, Dir: dir, Int4Path: int4Path}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err == nil {
		tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
		if err != nil {
			t.Fatal(err)
		}
		tm, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range qcPrompts {
			ids, err := tk.EncodeSegments(tm.RenderSegments("", []chat.Turn{{Role: "user", Content: p}}), false)
			if err != nil {
				t.Fatal(err)
			}
			prompts = append(prompts, ids)
		}
		defer func() { // texts for the reader, decoded after the fact
			for _, path := range res.Paths {
				s, _ := tk.Decode(path)
				res.Texts = append(res.Texts, s)
			}
		}()
	} else {
		res.Synthetic = true
		hb("SYNTHETIC prompts: no tokenizer.json in %s; this is a plumbing control, not a result", dir)
		for i := range qcPrompts {
			ids := make([]int, 12+i)
			for j := range ids {
				ids[j] = 3 + (i*131+j*17)%50
			}
			prompts = append(prompts, ids)
		}
	}
	res.Prompts = len(prompts)
	ctx := context.Background()

	finite := func(l []float32) bool {
		for _, v := range l {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return false
			}
		}
		return true
	}
	// run decodes one prompt: forced == nil is greedy until a stop token or `steps`; otherwise it takes forced[k] at step k. It returns the path and a copy of the argmax-relevant stats per step.
	type stepInfo struct {
		argmax     int
		top1, top2 float64
	}
	run := func(m *Model, ids, forced []int) (path []int, infos []stepInfo, nonFinite int) {
		cache := m.NewCache(len(ids) + steps + 1)
		l, err := m.prefillLogits(ctx, ids, cache)
		if err != nil {
			t.Fatal(err)
		}
		for k := range steps {
			if !finite(l) {
				nonFinite++
			}
			lp := logSoftmax64(l)
			a := argmax(l)
			p1, p2 := 0.0, 0.0
			for _, v := range lp {
				p := math.Exp(v)
				if p > p1 {
					p1, p2 = p, p1
				} else if p > p2 {
					p2 = p
				}
			}
			info := stepInfo{argmax: a, top1: p1, top2: p2}
			next := a
			if forced != nil {
				if k >= len(forced) {
					break
				}
				next = forced[k]
			}
			infos = append(infos, info)
			path = append(path, next)
			if forced == nil && m.isStop(next, SamplingParams{}) {
				break
			}
			if l, err = m.forward(next, cache); err != nil {
				t.Fatal(err)
			}
		}
		return path, infos, nonFinite
	}

	// Arm 1: int8int8, greedy.
	t0 := time.Now()
	int8Path, opts8 := dir, Options{Backend: "cpu", Quant: "int8int8"}
	if b := os.Getenv("GOINFER_QC_INT8"); b != "" { // an int8int8 bundle (prequant), for a checkpoint too big to quantise straight from its safetensors
		int8Path, opts8 = b, Options{Backend: "cpu"}
	}
	m8, err := Load(int8Path, opts8)
	if err != nil {
		t.Fatalf("int8int8 load: %v", err)
	}
	res.LoadInt8Sec = time.Since(t0).Seconds()
	hb("int8int8 loaded in %.0fs (%s)", res.LoadInt8Sec, m8.DecodePath())
	type refStep struct{ margin, top1 float64 }
	refStats := make([][]refStep, len(prompts))
	var tokens int
	t0 = time.Now()
	for i, ids := range prompts {
		path, infos, nf := run(m8, ids, nil)
		res.Paths = append(res.Paths, path)
		res.NonFinite += nf
		for _, in := range infos {
			refStats[i] = append(refStats[i], refStep{in.top1 - in.top2, in.top1})
		}
		tokens += len(path)
		hb("int8int8 prompt %d/%d: %d tokens (%.1fs per token so far)", i+1, len(prompts), len(path), time.Since(t0).Seconds()/float64(tokens))
	}
	res.DecodeInt8SecPerTok = time.Since(t0).Seconds() / float64(tokens)
	m8.Close()
	m8 = nil

	// Arm 2: int4, teacher-forced along the int8int8 path.
	t0 = time.Now()
	arm2 := os.Getenv("GOINFER_QC_ARM2_QUANT") // the positive control: "int8int8" makes the second arm the first again, which must read 100%
	if arm2 == "" {
		arm2 = "int4"
	}
	opts4 := Options{Backend: "cpu", Quant: arm2, EmbedInt4: arm2 == "int4"} // int4: serve's default on the CPU; a bundle carries its own quant and --embed-int4 choice
	if int4Path != dir {
		opts4 = Options{Backend: "cpu"}
	}
	m4, err := Load(int4Path, opts4)
	if err != nil {
		t.Fatalf("int4 load: %v", err)
	}
	res.LoadInt4Sec = time.Since(t0).Seconds()
	hb("int4 loaded in %.0fs (%s)", res.LoadInt4Sec, m4.DecodePath())
	t0 = time.Now()
	tokens = 0
	for i, ids := range prompts {
		path := res.Paths[i]
		_, infos, nf := run(m4, ids, path)
		res.NonFinite += nf
		agree := 0
		for k, in := range infos {
			res.Positions++
			if in.argmax == path[k] { // the path was decoded greedily, so path[k] is the int8int8 arm's own argmax at step k
				agree++
				res.Agree++
			} else {
				res.Splits = append(res.Splits, qcSplit{Prompt: i, Step: k, Ref: path[k], Got: in.argmax, RefMargin: refStats[i][k].margin, RefTop: refStats[i][k].top1})
			}
		}
		tokens += len(infos)
		hb("int4 prompt %d/%d: agree %d of %d (%.1fs per token so far)", i+1, len(prompts), agree, len(infos), time.Since(t0).Seconds()/float64(tokens))
	}
	res.DecodeInt4SecPerTok = time.Since(t0).Seconds() / float64(tokens)
	m4.Close()
	res.Agreement = float64(res.Agree) / float64(res.Positions)
	hb("DONE %s: int4 agrees with int8int8 at %d of %d positions = %.2f%%; non-finite logit rows %d; synthetic=%v", label, res.Agree, res.Positions, 100*res.Agreement, res.NonFinite, res.Synthetic)
	if out != "" {
		b, _ := json.MarshalIndent(res, "", " ")
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if res.NonFinite > 0 {
		t.Errorf("%d non-finite logit rows", res.NonFinite)
	}
}
