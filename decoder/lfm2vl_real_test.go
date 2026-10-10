//go:build realckpt

// G-S10l-c of docs/tasks/task-multimodal-support-2026-10.md (S10, LFM2.5-VL, registered 2026-10-09 before any code):
// LFM2.5-VL-1.6B on a multi-tile image prompt, goinfer's float32 CPU prefill against transformers'
// Lfm2VlForConditionalGeneration in float32. Three steps, so the same ids reach both sides (nobara):
//
//  1. GOINFER_S10L_STEP=ids writes <out>/ids.json: the prompt serve builds for glm_ocr/table.png, with the checkpoint's
//     own template through chat.Detect (as serve does), an explicit system message, the image block first
//     (multimodal.Lfm2VLImageBlock spliced as serve does).
//
//  2. scripts/pin_lfm2vl_real.py reads it and writes HF's logits (<out>/hf_last_logits.f32, golden.json's "c"), and
//     records whether HF's own chat template and processor give the same ids.
//
//  3. GOINFER_S10L_STEP=compare runs goinfer end to end (its own preprocessing, aikit's tower per tile, the projector, the
//     lfm2 per-token image prefill, one span per tile and thumbnail) and grades it. PASS: last-position logit cosine
//     >= 0.999, argmax equal, argmax agreement >= 95% over the text positions after the image. The registered planted
//     defects must each fail that: (1) the tiles' features in column-major order; (2) the thumbnail's features placed
//     first; (3) features written over the <|img_row_R_col_C|> markers.
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_S10L_STEP=ids|compare GOINFER_S10L_OUT=<dir> [GOINFER_LFM2VL=<dir>] \
//     go test -tags realckpt ./decoder/ -run TestLfm2VLFull -v -timeout 60m
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

type s10lIDs struct {
	IDs    []int    `json:"ids"`
	Image  string   `json:"image"`
	System string   `json:"system"`
	Text   string   `json:"text"`
	Runs   [][2]int `json:"runs"` // the <image> runs, one per tile and the thumbnail: [start, len]
}

func TestLfm2VLFull(t *testing.T) {
	requireHeavyModel(t)
	step, out := os.Getenv("GOINFER_S10L_STEP"), os.Getenv("GOINFER_S10L_OUT")
	if step == "" || out == "" {
		t.Skip("set GOINFER_S10L_STEP (ids | compare) and GOINFER_S10L_OUT")
	}
	dir := os.Getenv("GOINFER_LFM2VL")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "lfm25-vl-1.6b")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	const image, system, text = "glm_ocr/table.png", "You are a helpful assistant.", "What does this table show? Answer in one sentence."
	t0 := time.Now()
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[G-S10l-c %5.1fs] "+format+"\n", append([]any{time.Since(t0).Seconds()}, a...)...)
	}
	data, err := os.ReadFile(filepath.Join("../testdata", image))
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadLfm2VLPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	layout, err := multimodal.Lfm2VLPreprocess(data, pp)
	if err != nil {
		t.Fatal(err)
	}
	if layout.Rows*layout.Cols < 2 || !layout.Thumbnail {
		t.Fatalf("%s lays out as %dx%d (thumbnail %v): G-S10l-c wants a multi-tile image", image, layout.Rows, layout.Cols, layout.Thumbnail)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID(multimodal.Lfm2VLImageToken)
	if !ok {
		t.Fatal("no <image> token")
	}
	idsFile := filepath.Join(out, "ids.json")

	if step == "ids" {
		tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
		if err != nil {
			t.Fatalf("chat.Detect on the checkpoint's template: %v", err)
		}
		block := multimodal.Lfm2VLImageBlock(layout, pp)
		turns := []chat.Turn{{Role: "user", Content: block + text}}
		segs, err := multimodal.SpliceImageBlocks(tmpl.RenderSegments(system, turns), []string{block})
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		runs := multimodal.FindImageRuns(ids, imgTok)
		if len(runs) != len(layout.Tiles) {
			t.Fatalf("%d <image> runs, want one per tile and the thumbnail (%d)", len(runs), len(layout.Tiles))
		}
		b, _ := json.Marshal(s10lIDs{IDs: ids, Image: image, System: system, Text: text, Runs: runs})
		if err := os.MkdirAll(out, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(idsFile, b, 0o644); err != nil {
			t.Fatal(err)
		}
		logf("wrote %d ids (template %s, first id %d), %dx%d tiles + thumbnail, runs %v: %s", len(ids), tmpl.Name(), ids[0], layout.Rows, layout.Cols, layout.Tokens(pp), idsFile)
		return
	}

	var in s10lIDs
	raw, err := os.ReadFile(idsFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	var g struct {
		C struct {
			IDs                int   `json:"ids"`
			Argmax             []int `json:"argmax"`
			HFTemplateIDsEqual bool  `json:"hf_template_ids_equal"`
			HFTemplateIDs      int   `json:"hf_template_ids"`
		} `json:"c"`
	}
	raw, err = os.ReadFile(filepath.Join(out, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.C.IDs != len(in.IDs) || len(g.C.Argmax) != len(in.IDs) {
		t.Fatalf("HF's reference is for %d ids (%d argmax); goinfer has %d", g.C.IDs, len(g.C.Argmax), len(in.IDs))
	}
	logf("goinfer's ids against HF's own chat template and processor: equal %v (HF %d ids, goinfer %d)", g.C.HFTemplateIDsEqual, g.C.HFTemplateIDs, len(in.IDs))
	lb, err := os.ReadFile(filepath.Join(out, "hf_last_logits.f32"))
	if err != nil {
		t.Fatal(err)
	}
	hfLast := make([]float32, len(lb)/4)
	for i := range hfLast {
		hfLast[i] = math.Float32frombits(binary.LittleEndian.Uint32(lb[4*i:]))
	}

	// goinfer's features, end to end, tile by tile.
	enc, err := vision.LoadSiglip2NaFlexEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := multimodal.LoadLfm2VLProjector(dir)
	if err != nil {
		t.Fatal(err)
	}
	var tileFeats [][]float32
	for _, tile := range layout.Tiles {
		patches, grid, err := vision.PatchifyNaFlex(tile.CHW, 3, tile.H, tile.W, pp.PatchSize)
		if err != nil {
			t.Fatal(err)
		}
		hid, err := enc.Forward(patches, grid)
		if err != nil {
			t.Fatal(err)
		}
		f, err := proj.Forward(hid, grid)
		if err != nil {
			t.Fatal(err)
		}
		tileFeats = append(tileFeats, f)
	}
	enc = nil
	logf("tower and projector: %d tiles", len(tileFeats))

	m, err := Load(dir, Options{}) // float32
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	var feats []float32
	for _, f := range tileFeats {
		feats = append(feats, f...)
	}
	var spans []ImageSpan
	for _, r := range in.Runs {
		spans = append(spans, ImageSpan{Pos: r[0], Len: r[1], Hash: 1})
	}
	last := spans[len(spans)-1]
	after := last.Pos + last.Len // the <|image_end|>, then the text

	ctx := context.Background()
	grade := func(label string, spans []ImageSpan, feats []float32) (cos float64, lastOK bool, pct float64) {
		hN, err := m.prefillHiddenVLSpans(ctx, in.IDs, spans, feats, m.NewCache(len(in.IDs)), true)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		n := len(in.IDs) - after
		logits := m.lmHeadN(hN[after*hidden:], n)
		vocab := len(logits) / n
		agree := 0
		for i := range n {
			if argmax(logits[i*vocab:(i+1)*vocab]) == g.C.Argmax[after+i] {
				agree++
			}
		}
		lastL := logits[(n-1)*vocab:]
		cos = logitCosine(lastL, hfLast)
		lastOK = argmax(lastL) == argmax(hfLast)
		pct = 100 * float64(agree) / float64(n)
		logf("%-50s last cosine %.6f, argmax %d (HF %d), agreement %d/%d = %.1f%% over the text after the image", label, cos, argmax(lastL), argmax(hfLast), agree, n, pct)
		return cos, lastOK, pct
	}
	pass := func(cos float64, lastOK bool, pct float64) bool { return cos >= 0.999 && lastOK && pct >= 95 }

	if c, ok, p := grade("goinfer", spans, feats); !pass(c, ok, p) {
		t.Errorf("G-S10l-c FAIL: cosine %.6f, argmax equal %v, agreement %.1f%%", c, ok, p)
	} else {
		logf("G-S10l-c PASS")
	}
	prod, err := m.prefillLogitsVLSpans(ctx, in.IDs, spans, feats, m.NewCache(len(in.IDs)), true)
	if err != nil {
		t.Fatal(err)
	}
	logf("prefillLogitsVLSpans (production) against HF: last cosine %.6f", logitCosine(prod, hfLast))

	nTiles := layout.Rows * layout.Cols
	// Planted (1): the tiles' features in column-major order (every tile is the same size, so the spans are unchanged).
	var colMajor []float32
	for col := range layout.Cols {
		for r := range layout.Rows {
			colMajor = append(colMajor, tileFeats[r*layout.Cols+col]...)
		}
	}
	colMajor = append(colMajor, tileFeats[nTiles]...)
	// Planted (2): the thumbnail's features first, then the tiles', filling the same spans.
	thumbFirst := append(append([]float32(nil), tileFeats[nTiles]...), feats[:len(feats)-len(tileFeats[nTiles])]...)
	// Planted (3): features also over each tile's marker (the position before its run), a run's first row repeated there.
	var feats3 []float32
	var spans3 []ImageSpan
	for k, s := range spans {
		feats3 = append(append(feats3, tileFeats[k][:hidden]...), tileFeats[k]...)
		spans3 = append(spans3, ImageSpan{Pos: s.Pos - 1, Len: s.Len + 1, Hash: 1})
	}
	for _, d := range []struct {
		label string
		spans []ImageSpan
		feats []float32
	}{
		{"planted 1: the tiles in column-major order", spans, colMajor},
		{"planted 2: the thumbnail's features first", spans, thumbFirst},
		{"planted 3: features over the tile markers", spans3, feats3},
	} {
		if c, ok, p := grade(d.label, d.spans, d.feats); pass(c, ok, p) {
			t.Errorf("%s stayed green: the gate cannot see it", d.label)
		}
	}
}
