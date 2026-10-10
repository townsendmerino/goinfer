//go:build realckpt

// G-S3b's Gemma 3 divergence, phase 2 (docs/tasks/task-multimodal-support-2026-10.md, S3). Served, with only the tower
// differing, Gemma 3 4B's reply split at token 10 (the reference at p 0.795 against 0.195) although the Metal tower's
// features matched the CPU tower's to a worst token cosine of 0.9999999. Is that the tower, or the decoder's sensitivity?
//
// Phase 1 (metal/gemma3_tower_dump_test.go, on a Mac) writes the towers' projected features for one image: the CPU
// float32 tower, the Metal f16 tower and the Metal int8 tower. This test runs the int4 decoder on the CPU with serve's
// prompt and teacher-forces every arm along the CPU-tower arm's own greedy path: the CPU tower's features (the
// reference), each Metal tower's, and for each of those the CPU tower's plus random noise matched per soft token to
// that tower's relative L2 deviation (three seeds; shipped-path-is-not-ground-truth's perturbation control). Each arm
// reports its first step whose argmax differs, the reference's probabilities there, how many steps keep the reference's
// argmax, and the mean and worst KL(reference || arm) over the steps, which end at the reference's end of turn. A tower
// arm inside its own noise arms is the decoder's sensitivity to a perturbation of that size; one above them costs more
// than its size alone explains. Two controls come first: the reference's features again (zero) and the same features
// negated (far from zero). A measurement; scripts/gs18g2_grade.py grades its GOINFER_G3_OUT lines.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS=<phase 1 file> [GOINFER_GEMMA3_4B=<dir>] [GOINFER_G3_BACKEND=metal] \
//	  [GOINFER_G3_PROMPT=<text>] [GOINFER_G3_IMAGE=<label>] [GOINFER_G3_OUT=<jsonl>] \
//	  go test -tags realckpt ./decoder/ -run TestGemma3TowerSensitivity -v -timeout 30m
package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGemma3TowerSensitivity(t *testing.T) {
	requireHeavyModel(t)
	featsFile := os.Getenv("GOINFER_G3_FEATS")
	if featsFile == "" {
		t.Skip("set GOINFER_G3_FEATS to phase 1's output (metal/gemma3_tower_dump_test.go)")
	}
	dir := os.Getenv("GOINFER_GEMMA3_4B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-3-4b-it")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	backend := os.Getenv("GOINFER_G3_BACKEND")
	if backend == "" {
		backend = "metal" // G-S3b's arms: --backend metal, the resident declined, so the CPU with Metal's (canonical) layout
	}
	raw, err := os.ReadFile(featsFile)
	if err != nil {
		t.Fatal(err)
	}
	var feats map[string][]float32
	if err := json.Unmarshal(raw, &feats); err != nil {
		t.Fatal(err)
	}
	cpuF, metalF := feats["cpu"], feats["metal"]
	if len(cpuF) == 0 || len(cpuF) != len(metalF) {
		t.Fatalf("features: cpu %d, metal %d values", len(cpuF), len(metalF))
	}

	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	// EmbedInt4 off: what serve's --backend metal loads (loadflags.embedInt4), so the arms use serve's own decoder.
	// Through the directory's sidecar when it exists (`<dir>.int4.metal.giw`, the file serve's Metal load reads, int4 with
	// the plain head), mapped rather than quantized into the heap, which the load guard refuses on the 16 GB Mac. The
	// sidecar generates exactly what the direct load does (prequant.TestDirSidecar_matchesDirectLoad).
	src := dir
	if g := dir + ".int4.metal.giw"; fileExistsG3(g) {
		src = g
		fmt.Fprintf(os.Stderr, "[g3] loading the decoder through its sidecar %s\n", filepath.Base(g))
	}
	m, err := Load(src, Options{Backend: backend, Quant: "int4", EmbedInt4: false, ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	n := len(cpuF) / hidden
	fmt.Fprintf(os.Stderr, "[g3] %s on %s: decode path %s; %d soft tokens\n", filepath.Base(dir), backend, m.DecodePath(), n)

	// serve's prompt, built the way serve builds it (internal/serveapp encodeVisionSegments): the model's chat template
	// rendered as segments, the image block (the processor's "\n\n" on both sides, M-38) spliced in as its own Special
	// segment, then EncodeSegments. Do not hand-write the template and encode it as one string: that merges the template's
	// "\n" with the block's "\n\n" into one token, so the greedy reference is not the served path.
	tm, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	prompt := os.Getenv("GOINFER_G3_PROMPT")
	if prompt == "" {
		prompt = "What does this image show? Answer briefly."
	}
	block := multimodal.Gemma3PromptBlock(n)
	segs, err := multimodal.SpliceImageBlock(tm.RenderSegments("", []chat.Turn{{Role: "user", Content: block + prompt}}), block)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.EncodeSegments(segs, false)
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID(multimodal.ImageSoftToken)
	if !ok {
		t.Fatal("no image soft token")
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, imgTok)
	if imgLen != n {
		t.Fatalf("%d soft tokens in the prompt, want %d (the tokenizer split the special tokens)", imgLen, n)
	}

	const maxSteps = 32
	// run teacher-forces forced (nil: greedy for maxSteps, recording its own argmax path) and returns the logits at every
	// step.
	run := func(f []float32, forced []int) (path []int, logits [][]float32) {
		steps := maxSteps
		if forced != nil {
			steps = len(forced)
		}
		cache := m.NewCache(len(ids) + maxSteps + 1)
		l, err := m.prefillLogitsVL(context.Background(), ids, f, imgPos, imgLen, cache)
		if err != nil {
			t.Fatal(err)
		}
		for k := range steps {
			logits = append(logits, slices.Clone(l)) // forward returns the cache's reused logits buffer (night 2026-10-07: steps 1-31 all read the last)
			next := argmax(l)
			if forced != nil {
				next = forced[k]
			}
			path = append(path, next)
			if l, err = m.forward(next, cache); err != nil {
				t.Fatal(err)
			}
		}
		return path, logits
	}
	refPath, refLogits := run(cpuF, nil)
	refText, _ := tk.Decode(refPath)
	fmt.Fprintf(os.Stderr, "[g3] reference (CPU tower) greedy: %q\n", refText)
	// The graded positions end with the reference's end of turn: a step forced past it predicts nothing a reply contains.
	if eot, ok := tk.TokenID("<end_of_turn>"); ok {
		if i := slices.Index(refPath, eot); i >= 0 {
			refPath, refLogits = refPath[:i+1], refLogits[:i+1]
		}
	}
	steps := len(refPath)
	fmt.Fprintf(os.Stderr, "[g3] %d graded positions\n", steps)
	// GOINFER_G3_SERVED_REPLY names the served CPU-tower arm's reply on this decoder (G-S3b's day run): the reference path
	// must reproduce it, or the steps below do not measure the served split and the run is void.
	if f := os.Getenv("GOINFER_G3_SERVED_REPLY"); f != "" {
		want, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		got, _, _ := strings.Cut(refText, "<end_of_turn>")
		if w := strings.TrimSpace(string(want)); !strings.HasPrefix(got, w) && !strings.HasPrefix(w, got) {
			t.Fatalf("VOID: the reference path %q does not reproduce the served reply %q", refText, want)
		}
		fmt.Fprintf(os.Stderr, "[g3] the reference path reproduces the served reply (%s)\n", filepath.Base(f))
	}

	type armResult struct {
		Name     string  `json:"name"`
		KLMean   float64 `json:"kl_mean"`
		KLMax    float64 `json:"kl_max"`
		Agree    int     `json:"agree"` // steps whose argmax is the reference's token
		FirstOff int     `json:"first_change"`
		RelL2    float64 `json:"rel_l2,omitempty"` // a tower arm's mean relative L2 from the CPU tower, per soft token
	}
	var results []armResult
	report := func(name string, f []float32) armResult {
		_, ls := run(f, refPath)
		r := armResult{Name: name, FirstOff: -1}
		var sumKL float64
		for k, l := range ls {
			kl := klTo(refLogits[k], l)
			sumKL += kl
			r.KLMax = math.Max(r.KLMax, kl)
			if argmax(l) == refPath[k] {
				r.Agree++
			} else if r.FirstOff < 0 {
				r.FirstOff = k
			}
		}
		r.KLMean = sumKL / float64(steps)
		line := fmt.Sprintf("[g3] %-28s KL mean %.5f, max %.5f nats; argmax agrees at %d of %d; ", name, r.KLMean, r.KLMax, r.Agree, steps)
		if r.FirstOff < 0 {
			line += "no change"
		} else {
			lp := logSoftmax64(refLogits[r.FirstOff])
			other := argmax(ls[r.FirstOff])
			line += fmt.Sprintf("first change at step %d: reference %q p %.3f, this %q (reference p %.3f)",
				r.FirstOff, layoutPiece(tk, refPath[r.FirstOff]), math.Exp(lp[refPath[r.FirstOff]]), layoutPiece(tk, other), math.Exp(lp[other]))
		}
		fmt.Fprintln(os.Stderr, line)
		results = append(results, r)
		return r
	}

	// Two controls on the instrument itself. The CPU tower's features again must read exactly zero (the decoder is
	// deterministic), and the same features negated must read far from zero (a wrong image is visible, whatever the
	// image: a mirrored gradient is still a gradient, so reordering the soft tokens would not do).
	report("control: CPU tower again", cpuF)
	neg := make([]float32, len(cpuF))
	for i, v := range cpuF {
		neg[i] = -v
	}
	report("control: features negated", neg)

	// Every tower arm in the file, each followed by its own noise control: per soft token, Gaussian noise scaled to that
	// tower's relative L2 deviation from the CPU tower there (three seeds).
	arms := []struct {
		name string
		f    []float32
	}{{"Metal f16 tower", metalF}}
	if f8 := feats["metal_int8"]; len(f8) == len(cpuF) {
		arms = append(arms, struct {
			name string
			f    []float32
		}{"Metal int8 tower", f8})
	}
	for _, arm := range arms {
		rel := make([]float64, n)
		var sumRel float64
		for i := range n {
			var dd, nn float64
			for j := range hidden {
				a, b := float64(cpuF[i*hidden+j]), float64(arm.f[i*hidden+j])
				dd, nn = dd+(a-b)*(a-b), nn+a*a
			}
			rel[i] = math.Sqrt(dd / nn)
			sumRel += rel[i]
		}
		report(arm.name, arm.f)
		results[len(results)-1].RelL2 = sumRel / float64(n)
		fmt.Fprintf(os.Stderr, "[g3]   (its mean relative L2 from the CPU tower: %.3g)\n", sumRel/float64(n))
		for seed := int64(1); seed <= 3; seed++ {
			rng := rand.New(rand.NewSource(seed))
			f := append([]float32(nil), cpuF...)
			for i := range n {
				row := f[i*hidden : (i+1)*hidden]
				noise := make([]float64, hidden)
				var nn, cn float64
				for j := range noise {
					noise[j] = rng.NormFloat64()
					nn += noise[j] * noise[j]
					cn += float64(row[j]) * float64(row[j])
				}
				s := rel[i] * math.Sqrt(cn/nn)
				for j := range row {
					row[j] += float32(s * noise[j])
				}
			}
			report(fmt.Sprintf("noise at %s's size, seed %d", arm.name, seed), f)
		}
	}

	// GOINFER_G3_OUT: one JSON line per run, appended, for a grader (GOINFER_G3_IMAGE names the image the features are of).
	if out := os.Getenv("GOINFER_G3_OUT"); out != "" {
		line, err := json.Marshal(map[string]any{"image": os.Getenv("GOINFER_G3_IMAGE"), "prompt": prompt, "steps": steps,
			"reference": refText, "decode_path": m.DecodePath(), "arms": results})
		if err != nil {
			t.Fatal(err)
		}
		f, err := os.OpenFile(out, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if _, err := f.Write(append(line, '\n')); err != nil {
			t.Fatal(err)
		}
	}
}

func fileExistsG3(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
