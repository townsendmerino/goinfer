//go:build cuda && goinfer_testhooks

// G-S10q-e of docs/tasks/task-multimodal-support-2026-10.md ("S10, Qwen3-VL MoE"): the CUDA resident's half of G-S10q-c's instrument.
// decoder/qwen3vlmoe_logits_real_test.go teacher-forces the CPU arms along the pinned continuations and writes the logits that
// predict each path token; this does the same on a CUDA resident (the paged-experts build for the 30B), along the route a served
// image turn takes for a MoE decoder: the CPU prefill, the UploadKV bridge, then the resident's m-RoPE decode. The pinned
// <out>.seqs.json (ids and path per sequence) is read, never written; the output is <out>.cuda.f32, [positions, vocab] float32,
// graded against the Hugging Face logits by scripts/q3vlmoe_cuda_grade.py.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_Q3M_DIR=<checkpoint dir> GOINFER_Q3M_OUT=<prefix> [GOINFER_Q3M_PAGED=0 for a model that fits] \
//	go test -tags 'cuda goinfer_testhooks' ./cuda/ -run '^TestQwen3VLMoe_cudaLogits$' -v -timeout 120m
package cuda

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

type q3mCudaSeq struct {
	IDs  []int  `json:"ids"`
	Path []int  `json:"path"`
	Name string `json:"name"`
}

type q3mCudaImage struct {
	grid   [3]int
	merged []float32
	deep   [][]float32
}

func TestQwen3VLMoe_cudaLogits(t *testing.T) {
	requireHeavyModel(t)
	requireCUDADevice(t)
	dir, out := os.Getenv("GOINFER_Q3M_DIR"), os.Getenv("GOINFER_Q3M_OUT")
	if dir == "" || out == "" {
		t.Skip("set GOINFER_Q3M_DIR and GOINFER_Q3M_OUT (the prefix of the pinned .seqs.json)")
	}
	for _, p := range []string{dir, out} {
		if strings.HasPrefix(p, "/Volumes/") || strings.HasPrefix(p, "/srv/models") {
			t.Fatalf("%s is the archive (CLAUDE.md)", p)
		}
	}
	t0 := time.Now()
	var phase atomic.Value
	phase.Store("start")
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[q3m-cuda %6.0fs] "+format+"\n", append([]any{time.Since(t0).Seconds()}, a...)...)
	}
	done := make(chan struct{})
	defer close(done)
	go func() { // a heartbeat: a CPU prefill of a 1,786-token prompt is minutes with nothing else to log
		for {
			select {
			case <-done:
				return
			case <-time.After(30 * time.Second):
				logf("... %s", phase.Load())
			}
		}
	}()

	raw, err := os.ReadFile(out + ".seqs.json")
	if err != nil {
		t.Fatal(err)
	}
	var seqs []q3mCudaSeq
	if err := json.Unmarshal(raw, &seqs); err != nil {
		t.Fatal(err)
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
	phase.Store("vision tower (CPU, goinfer's own, as G-S10q-c)")
	enc, err := vision.LoadQwen3VisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	merge := enc.Cfg.SpatialMergeSize
	images := map[string]q3mCudaImage{}
	for _, s := range seqs {
		if _, seen := images[s.Name]; seen {
			continue
		}
		data, err := os.ReadFile(filepath.Join("..", "testdata", s.Name))
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
		images[s.Name] = q3mCudaImage{grid: grid, merged: merged, deep: deep}
		logf("tower: %s grid %v, %d DeepStack sets", s.Name, grid, len(deep))
	}
	enc = nil

	phase.Store("model load")
	opts := decoder.Options{Backend: "cuda", Quant: "int4"}
	opts.MoECacheExperts = os.Getenv("GOINFER_Q3M_PAGED") != "0"
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("Load(cuda int4, paged=%v): %v", opts.MoECacheExperts, err)
	}
	defer m.Close()
	if m.ResidentForwardForTest() == nil {
		t.Fatalf("the CUDA resident declined this model: %s", m.ResidentDecline())
	}
	logf("loaded: paged=%v, resident present", opts.MoECacheExperts)

	f, err := os.Create(out + ".cuda.f32")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	positions, nonFinite := 0, 0
	for k, s := range seqs {
		im, ok := images[s.Name]
		if !ok {
			t.Fatalf("sequence %d: no image %q", k, s.Name)
		}
		pos, n := -1, 0
		for i, id := range s.IDs {
			if id == imgTok {
				if pos < 0 {
					pos = i
				}
				n++
			}
		}
		spans := []decoder.ImageSpan{{Pos: pos, Len: n, Hash: 1}}
		phase.Store(fmt.Sprintf("sequence %d/%d: CPU prefill of %d tokens, then %d resident steps", k+1, len(seqs), len(s.IDs), len(s.Path)-1))
		ts := time.Now()
		lg, err := m.QwenVLTeacherForcedLogitsForTest(context.Background(), s.IDs, s.Path, spans, im.merged, im.deep, [][3]int{im.grid}, merge, imgTok)
		if err != nil {
			t.Fatalf("sequence %d: %v", k+1, err)
		}
		if len(lg)%len(s.Path) != 0 {
			t.Fatalf("sequence %d: %d logits for %d positions", k+1, len(lg), len(s.Path))
		}
		buf := make([]byte, 4*len(lg))
		for i, v := range lg {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nonFinite++
			}
		}
		if _, err := f.Write(buf); err != nil {
			t.Fatal(err)
		}
		positions += len(s.Path)
		logf("sequence %d/%d (%d ids): %d positions in %.0fs, non-finite so far %d", k+1, len(seqs), len(s.IDs), len(s.Path), time.Since(ts).Seconds(), nonFinite)
	}
	logf("wrote %s.cuda.f32 (%d positions)", out, positions)
	if nonFinite > 0 {
		t.Errorf("%d non-finite logits", nonFinite)
	}
}
