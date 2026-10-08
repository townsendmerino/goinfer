//go:build realckpt

// G-S3b's Gemma 3 divergence, phase 2 (docs/tasks/task-multimodal-support-2026-10.md, S3). Served, with only the tower
// differing, Gemma 3 4B's reply split at token 10 (the reference at p 0.795 against 0.195) although the Metal tower's
// features matched the CPU tower's to a worst token cosine of 0.9999999. Is that the tower, or the decoder's sensitivity?
//
// Phase 1 (metal/gemma3_tower_dump_test.go, on a Mac) writes both towers' projected features for table.png. This test
// runs the int4 decoder on the CPU with serve's prompt and teacher-forces every arm along the CPU-tower arm's own greedy
// path: the CPU tower's features (the reference), the Metal tower's, and the CPU tower's plus random noise matched per
// soft token to the Metal tower's relative L2 deviation (three seeds; shipped-path-is-not-ground-truth's perturbation
// control). Each arm reports its first step whose argmax differs, the reference's probabilities there, and the mean
// and worst KL(reference || arm) over the steps. If the Metal arm sits inside the noise arms, the flip is the decoder's
// sensitivity to a perturbation of that size, not a tower defect. A measurement, not a gate.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS=<phase 1 file> [GOINFER_GEMMA3_4B=<dir>] [GOINFER_G3_BACKEND=metal] \
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
	// EmbedInt4 off: what serve's --backend metal loads (loadflags.embedInt4), so the G-S3b arms' own decoder.
	m, err := Load(dir, Options{Backend: backend, Quant: "int4", EmbedInt4: false, ResidentContext: 4096})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	n := len(cpuF) / hidden
	fmt.Fprintf(os.Stderr, "[g3] %s on %s: decode path %s; %d soft tokens\n", filepath.Base(dir), backend, m.DecodePath(), n)

	// serve's prompt: Gemma 3's template, the block (with the processor's "\n\n" on both sides, M-38) before the text.
	text := "<start_of_turn>user\n" + multimodal.Gemma3PromptBlock(n) + "What does this image show? Answer briefly.<end_of_turn>\n<start_of_turn>model\n"
	ids, err := tk.Encode(text, true)
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

	const steps = 32
	// run teacher-forces forced (nil: greedy, recording its own argmax path) and returns the logits at every step.
	run := func(f []float32, forced []int) (path []int, logits [][]float32) {
		cache := m.NewCache(len(ids) + steps + 1)
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

	report := func(name string, f []float32) {
		_, ls := run(f, refPath)
		first := -1
		var sumKL, maxKL float64
		for k, l := range ls {
			kl := klTo(refLogits[k], l)
			sumKL += kl
			maxKL = math.Max(maxKL, kl)
			if first < 0 && argmax(l) != refPath[k] {
				first = k
			}
		}
		line := fmt.Sprintf("[g3] %-26s KL mean %.5f, max %.5f nats; ", name, sumKL/steps, maxKL)
		if first < 0 {
			line += "argmax agrees at every step"
		} else {
			lp := logSoftmax64(refLogits[first])
			other := argmax(ls[first])
			line += fmt.Sprintf("first argmax change at step %d: reference %q p %.3f, this %q (reference p %.3f)",
				first, layoutPiece(tk, refPath[first]), math.Exp(lp[refPath[first]]), layoutPiece(tk, other), math.Exp(lp[other]))
		}
		fmt.Fprintln(os.Stderr, line)
	}
	report("Metal tower", metalF)

	// The control: per soft token, Gaussian noise scaled to the Metal tower's own relative L2 deviation there.
	rel := make([]float64, n)
	for i := range n {
		var dd, nn float64
		for j := range hidden {
			a, b := float64(cpuF[i*hidden+j]), float64(metalF[i*hidden+j])
			dd, nn = dd+(a-b)*(a-b), nn+a*a
		}
		rel[i] = math.Sqrt(dd / nn)
	}
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
		report(fmt.Sprintf("CPU tower + noise, seed %d", seed), f)
	}
}
