//go:build realckpt

// G-S10m-c of docs/tasks/task-multimodal-support-2026-10.md (S10, Ministral 3 (Pixtral), registered 2026-10-09 before
// any code): Ministral 3 3B on an image prompt, goinfer's float32 CPU prefill against transformers'
// Mistral3ForConditionalGeneration in float32. Three steps, so the same ids reach both sides (nobara):
//
//  1. GOINFER_S10M_STEP=ids writes <out>/ids.json: the prompt serve builds for glm_ocr/table.png with an explicit system
//     message (chat.Ministral, the image first, multimodal.PixtralImageBlock spliced as serve does).
//
//  2. scripts/pin_pixtral_real.py reads it and writes HF's logits (<out>/hf_last_logits.f32, golden.json's "c").
//
//  3. GOINFER_S10M_STEP=compare runs goinfer end to end (its own preprocessing, aikit's tower, the projector, the causal
//     span prefill, one span per merged row) and grades it. PASS: last-position logit cosine >= 0.999, argmax equal, and
//     argmax agreement >= 95% over the text positions after the image. The registered planted defects must each fail
//     that: (1) the features also written over [IMG_BREAK]/[IMG_END]; (2) a bidirectional image block; (3) the rows
//     shifted by one position.
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_S10M_STEP=ids|compare GOINFER_S10M_OUT=<dir> [GOINFER_MINISTRAL3=<dir>] \
//     go test -tags realckpt ./decoder/ -run TestPixtralFull -v -timeout 60m
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

type s10mIDs struct {
	IDs   []int    `json:"ids"`
	Image string   `json:"image"`
	Runs  [][2]int `json:"runs"` // the [IMG] runs, one per merged row: [start, len]
}

func TestPixtralFull(t *testing.T) {
	requireHeavyModel(t)
	step, out := os.Getenv("GOINFER_S10M_STEP"), os.Getenv("GOINFER_S10M_OUT")
	if step == "" || out == "" {
		t.Skip("set GOINFER_S10M_STEP (ids | compare) and GOINFER_S10M_OUT")
	}
	dir := os.Getenv("GOINFER_MINISTRAL3")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "ministral3-3b-bf16")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	const image = "glm_ocr/table.png"
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S10m-c %5.1fs] "+format+"\n", append([]any{time.Since(t0).Seconds()}, a...)...)
	}
	data, err := os.ReadFile(filepath.Join("../testdata", image))
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadPixtralPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	chw, h, w, err := multimodal.PixtralPreprocess(data, pp)
	if err != nil {
		t.Fatal(err)
	}
	f := pp.PatchSize * pp.MergeSize
	rows, cols := h/f, w/f
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID(multimodal.PixtralImageToken)
	if !ok {
		t.Fatal("no [IMG] token")
	}
	idsFile := filepath.Join(out, "ids.json")

	if step == "ids" {
		block := multimodal.PixtralImageBlock(rows, cols)
		turns := []chat.Turn{{Role: "user", Content: block + "What does this table show? Answer in one sentence."}}
		segs, err := multimodal.SpliceImageBlocks(chat.Ministral().RenderSegments("You are a helpful assistant.", turns), []string{block})
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		runs := multimodal.FindImageRuns(ids, imgTok)
		if len(runs) != rows {
			t.Fatalf("%d [IMG] runs, want one per merged row (%d)", len(runs), rows)
		}
		b, _ := json.Marshal(s10mIDs{IDs: ids, Image: image, Runs: runs})
		if err := os.MkdirAll(out, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(idsFile, b, 0o644); err != nil {
			t.Fatal(err)
		}
		logf("wrote %d ids, %dx%d merged (%d [IMG]): %s", len(ids), rows, cols, rows*cols, idsFile)
		return
	}

	var in s10mIDs
	raw, err := os.ReadFile(idsFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	var g struct {
		C struct {
			Image  string `json:"image"`
			IDs    int    `json:"ids"`
			Img    int    `json:"img"`
			Argmax []int  `json:"argmax"`
		} `json:"c"`
	}
	raw, err = os.ReadFile(filepath.Join(out, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.C.IDs != len(in.IDs) || len(g.C.Argmax) != len(in.IDs) || g.C.Img != rows*cols {
		t.Fatalf("HF's reference is for %d ids (%d argmax) and %d image rows; goinfer has %d ids and %d", g.C.IDs, len(g.C.Argmax), g.C.Img, len(in.IDs), rows*cols)
	}
	lb, err := os.ReadFile(filepath.Join(out, "hf_last_logits.f32"))
	if err != nil {
		t.Fatal(err)
	}
	hfLast := make([]float32, len(lb)/4)
	for i := range hfLast {
		hfLast[i] = math.Float32frombits(binary.LittleEndian.Uint32(lb[4*i:]))
	}

	// goinfer's features, end to end.
	enc, err := vision.LoadPixtralVisionEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := multimodal.LoadPixtralProjector(dir)
	if err != nil {
		t.Fatal(err)
	}
	patches, grid, err := vision.PixtralPatchify(chw, 3, h, w, pp.PatchSize)
	if err != nil {
		t.Fatal(err)
	}
	hid, err := enc.Forward(patches, [][2]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	feats, err := proj.Forward(hid, [][2]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	enc, hid = nil, nil
	logf("tower and projector: %d rows", rows*cols)

	m, err := Load(dir, Options{}) // float32
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	if len(feats) != rows*cols*hidden {
		t.Fatalf("%d feature values, want %d x %d", len(feats), rows*cols, hidden)
	}
	var spans []ImageSpan
	for _, r := range in.Runs {
		spans = append(spans, ImageSpan{Pos: r[0], Len: r[1], Hash: 1})
	}
	after := spans[len(spans)-1].Pos + spans[len(spans)-1].Len // the [IMG_END], then the text

	ctx := context.Background()
	grade := func(label string, ids []int, spans []ImageSpan, feats []float32, causal bool) (cos float64, lastOK bool, pct float64) {
		hN, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)), causal)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		n := len(ids) - after
		logits := m.lmHeadN(hN[after*hidden:], n)
		vocab := len(logits) / n
		agree := 0
		for i := range n {
			if argmax(logits[i*vocab:(i+1)*vocab]) == g.C.Argmax[after+i] {
				agree++
			}
		}
		last := logits[(n-1)*vocab:]
		cos = logitCosine(last, hfLast)
		lastOK = argmax(last) == argmax(hfLast)
		pct = 100 * float64(agree) / float64(n)
		logf("%-46s last cosine %.6f, argmax %d (HF %d), agreement %d/%d = %.1f%% over the text after the image", label, cos, argmax(last), argmax(hfLast), agree, n, pct)
		return cos, lastOK, pct
	}
	pass := func(cos float64, lastOK bool, pct float64) bool { return cos >= 0.999 && lastOK && pct >= 95 }

	if c, ok, p := grade("goinfer", in.IDs, spans, feats, true); !pass(c, ok, p) {
		t.Errorf("G-S10m-c FAIL: cosine %.6f, argmax equal %v, agreement %.1f%%", c, ok, p)
	} else {
		logf("G-S10m-c PASS")
	}
	// The production entry's last logits are the graded path's.
	prod, err := m.prefillLogitsVLSpans(ctx, in.IDs, spans, feats, m.NewCache(len(in.IDs)), true)
	if err != nil {
		t.Fatal(err)
	}
	logf("prefillLogitsVLSpans (production) against HF: last cosine %.6f", logitCosine(prod, hfLast))

	// Planted (1): the features also over each [IMG_BREAK]/[IMG_END] (a row's last feature repeated there), one span
	// per row covering the break.
	var feats1 []float32
	var spans1 []ImageSpan
	for k, s := range spans {
		row := feats[k*cols*hidden : (k+1)*cols*hidden]
		feats1 = append(append(feats1, row...), row[(cols-1)*hidden:]...)
		spans1 = append(spans1, ImageSpan{Pos: s.Pos, Len: s.Len + 1, Hash: 1})
	}
	// Planted (3): every row one position later.
	spans3 := make([]ImageSpan, len(spans))
	for k, s := range spans {
		spans3[k] = ImageSpan{Pos: s.Pos + 1, Len: s.Len, Hash: 1}
	}
	for _, d := range []struct {
		label  string
		spans  []ImageSpan
		feats  []float32
		causal bool
	}{
		{"planted 1: features over the breaks and the end", spans1, feats1, true},
		{"planted 2: a bidirectional image block", spans, feats, false},
		{"planted 3: the rows shifted by one", spans3, feats, true},
	} {
		if c, ok, p := grade(d.label, in.IDs, d.spans, d.feats, d.causal); pass(c, ok, p) {
			t.Errorf("%s stayed green: the gate cannot see it", d.label)
		}
	}
}
