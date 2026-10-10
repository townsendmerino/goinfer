//go:build realckpt

// G-S10q-c of docs/tasks/task-multimodal-support-2026-10.md ("S10, Qwen3-VL MoE", registered before this code): goinfer's
// half, step (b')'s design (g31b_logits_real_test.go) on image prompts. Run once for the 30B and once for the dense
// Qwen3-VL-2B sibling the bar is read against.
//
//   - GOINFER_Q3M_STEP=paths writes <out>.seqs.json: the 8 prompts (the four F2a images x two questions, an explicit system
//     message, the checkpoint's own chat template through chat.Detect, the image block as serve builds it from the
//     checkpoint's own, uncapped preprocessing), each with goinfer's own int8int8 greedy continuation of up to 32 tokens
//     through the production entry (GenerateQwenVLDeepstackSpans): {"ids", "path", "image"}. Recorded once and pinned.
//
//   - GOINFER_Q3M_STEP=logits, for each arm (GOINFER_Q3M_ARMS, default int8int8,int4; CPU): teacher-forces every sequence
//     (ids + path[:-1], the image features from goinfer's own tower) and writes <out>.<arm>.f32, the float32 logits
//     predicting each path token, [positions, vocab]. Nothing is graded here (scripts/q3vlmoe_grade.py).
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_Q3M_DIR=<checkpoint dir> GOINFER_Q3M_OUT=<prefix> GOINFER_Q3M_STEP=paths|logits \
//     go test -tags realckpt ./decoder/ -run TestQwen3VLMoe_goinferLogits -v -timeout 180m
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

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

var q3mImages = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}
var q3mQuestions = []string{"Describe this image in one sentence.", "What text, if any, appears in this image? Answer briefly."}

const q3mSystem = "You are a helpful assistant."

type q3mSeq struct {
	IDs   []int  `json:"ids"`
	Path  []int  `json:"path"`
	Image string `json:"image"` // the absolute path, for the HF side
	Name  string `json:"name"`  // the testdata name
}

// q3mImage is one image's goinfer side: its grid, its merged rows and DeepStack sets (goinfer's own preprocessing and
// aikit's tower), and its prompt block.
type q3mImage struct {
	grid   [3]int
	merged []float32
	deep   [][]float32
	block  string
}

func TestQwen3VLMoe_goinferLogits(t *testing.T) {
	requireHeavyModel(t)
	dir, out, step := os.Getenv("GOINFER_Q3M_DIR"), os.Getenv("GOINFER_Q3M_OUT"), os.Getenv("GOINFER_Q3M_STEP")
	if dir == "" || out == "" || (step != "paths" && step != "logits") {
		t.Skip("set GOINFER_Q3M_DIR, GOINFER_Q3M_OUT and GOINFER_Q3M_STEP (paths | logits)")
	}
	for _, p := range []string{dir, out} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is the archive (CLAUDE.md)", p)
		}
	}
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[q3m %6.0fs] "+format+"\n", append([]any{time.Since(t0).Seconds()}, a...)...)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID(multimodal.QwenImagePad)
	if !ok {
		t.Fatal("no image pad token")
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	merge := enc.Cfg.SpatialMergeSize
	images := map[string]q3mImage{}
	for _, name := range q3mImages {
		data, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		pv, grid, err := multimodal.QwenPreprocess(data, pp)
		if err != nil {
			t.Fatal(err)
		}
		merged, deep, err := enc.ForwardDeepstack(pv, [][3]int{grid})
		if err != nil {
			t.Fatal(err)
		}
		images[name] = q3mImage{grid: grid, merged: merged, deep: deep, block: multimodal.QwenImageBlock(multimodal.QwenMergedTokens(grid, merge))}
		logf("tower: %s grid %v, %d DeepStack sets", name, grid, len(deep))
	}
	enc = nil
	spanOf := func(ids []int) []ImageSpan {
		pos, n := -1, 0
		for i, id := range ids {
			if id == imgTok {
				if pos < 0 {
					pos = i
				}
				n++
			}
		}
		return []ImageSpan{{Pos: pos, Len: n, Hash: 1}}
	}

	if step == "paths" {
		tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
		if err != nil {
			t.Fatal(err)
		}
		m, err := Load(dir, Options{Quant: "int8int8"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		var seqs []q3mSeq
		for _, name := range q3mImages {
			im := images[name]
			for _, q := range q3mQuestions {
				turns := []chat.Turn{{Role: "user", Content: im.block + q}}
				segs, err := multimodal.SpliceImageBlocks(tmpl.RenderSegments(q3mSystem, turns), []string{im.block})
				if err != nil {
					t.Fatal(err)
				}
				ids, err := tk.EncodeSegments(segs, false)
				if err != nil {
					t.Fatal(err)
				}
				feats := func() ([]float32, [][]float32, error) { return im.merged, im.deep, nil }
				stream, g := m.GenerateQwenVLDeepstackSpans(context.Background(), ids, spanOf(ids), feats, [][3]int{im.grid}, merge, imgTok, 32, SamplingParams{})
				var path []int
				for tok := range stream {
					path = append(path, tok)
				}
				if err := g.Err(); err != nil {
					t.Fatal(err)
				}
				if len(path) == 0 {
					t.Fatalf("%s / %q: an empty continuation", name, q)
				}
				seqs = append(seqs, q3mSeq{IDs: ids, Path: path, Image: filepath.Join(must(filepath.Abs("../testdata")), name), Name: name})
				text, _ := tk.Decode(path)
				logf("path: %s / %q: %d prompt ids, %d tokens: %q", name, q, len(ids), len(path), text)
			}
		}
		b, _ := json.Marshal(seqs)
		if err := os.WriteFile(out+".seqs.json", b, 0o644); err != nil {
			t.Fatal(err)
		}
		logf("wrote %s.seqs.json (%d sequences)", out, len(seqs))
		return
	}

	raw, err := os.ReadFile(out + ".seqs.json")
	if err != nil {
		t.Fatal(err)
	}
	var seqs []q3mSeq
	if err := json.Unmarshal(raw, &seqs); err != nil {
		t.Fatal(err)
	}
	arms := strings.Split(os.Getenv("GOINFER_Q3M_ARMS"), ",")
	if arms[0] == "" {
		arms = []string{"int8int8", "int4"}
	}
	for _, arm := range arms {
		m, err := Load(dir, Options{Quant: arm})
		if err != nil {
			t.Fatal(err)
		}
		hidden := m.w.arch.HiddenDim
		f, err := os.Create(out + "." + arm + ".f32")
		if err != nil {
			t.Fatal(err)
		}
		positions := 0
		for k, s := range seqs {
			im, ok := images[s.Name]
			if !ok {
				t.Fatalf("sequence %d names image %q, not one of %v", k, s.Name, q3mImages)
			}
			ids := append(append([]int(nil), s.IDs...), s.Path[:len(s.Path)-1]...)
			spans := spanOf(ids)
			mp, err := mropePositions(ids, imgTok, [][3]int{im.grid}, merge)
			if err != nil {
				t.Fatal(err)
			}
			cache := m.NewCache(len(ids))
			cache.mropePos, cache.mropeDelta = mp, mropeDelta(mp, len(ids))
			cache.deepstack = &deepstackRows{spans: spans, rows: im.deep}
			h := m.embedN(ids)
			spliceImageSpans(h, spans, im.merged, hidden)
			hN, err := m.runLayersFromEmbedN(context.Background(), h, cache, m.cpuFastAttention())
			if err != nil {
				t.Fatal(err)
			}
			first := len(s.IDs) - 1
			lg := m.lmHeadN(hN[first*hidden:], len(s.Path))
			buf := make([]byte, 4*len(lg))
			finite := true
			for i, v := range lg {
				binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					finite = false
				}
			}
			if _, err := f.Write(buf); err != nil {
				t.Fatal(err)
			}
			positions += len(s.Path)
			logf("%s: sequence %d/%d (%d ids), %d positions, finite %v", arm, k+1, len(seqs), len(ids), len(s.Path), finite)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		m.Close()
		logf("%s: wrote %s.%s.f32 (%d positions)", arm, out, arm, positions)
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}
