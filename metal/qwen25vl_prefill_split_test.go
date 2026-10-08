//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-S3c's root cause, the Metal-binary half (docs/tasks/task-multimodal-support-2026-10.md, S3). Served, the Qwen2.5-VL
// image turn's first token is "Table" under --backend metal and "Quarter" under --backend cpu, and the first token comes
// from GenerateQwenVL's CPU prefill (prefillLogitsQwenVL) in both. decoder/qwen25vl_layout_split_real_test.go ran that
// prefill in a binary WITHOUT Metal and found every CPU layout identical ("Quarter"). This runs the same prefill, on the
// same features and prompt, in a binary WITH Metal, the load chosen by GOINFER_SPLIT_BACKEND (metal: the resident is
// built, as serve builds it), and compares its logits bit for bit with that test's canonical int4-head arm (written next to
// the features file as <features>.arm4). A measurement, not a gate.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAYOUT_FEATS=<features>.json GOINFER_SPLIT_BACKEND=metal \
//	  go test -tags goinfer_testhooks ./metal/ -run TestQwen25VLPrefillSplit -v -timeout 20m
func TestQwen25VLPrefillSplit(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy")
	}
	featsFile, be := os.Getenv("GOINFER_LAYOUT_FEATS"), os.Getenv("GOINFER_SPLIT_BACKEND")
	if featsFile == "" || be == "" {
		t.Skip("set GOINFER_LAYOUT_FEATS and GOINFER_SPLIT_BACKEND")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "qwen25vl-3b-instruct")
	raw, err := os.ReadFile(featsFile)
	if err != nil {
		t.Fatal(err)
	}
	var feats struct{ Table []float32 }
	if err := json.Unmarshal(raw, &feats); err != nil {
		t.Fatal(err)
	}
	var want []float32
	if b, err := os.ReadFile(featsFile + ".arm4"); err == nil {
		_ = json.Unmarshal(b, &want)
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	img, err := os.ReadFile("../testdata/glm_ocr/table.png")
	if err != nil {
		t.Fatal(err)
	}
	_, grid, err := multimodal.QwenPreprocess(img, pp)
	if err != nil {
		t.Fatal(err)
	}
	const merge = 2
	n := multimodal.QwenMergedTokens(grid, merge)
	ids, err := tk.Encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+multimodal.QwenImageBlock(n)+
		"What does this image show? Answer briefly.<|im_end|>\n<|im_start|>assistant\n", false)
	if err != nil {
		t.Fatal(err)
	}
	imgTok, _ := tk.TokenID(multimodal.QwenImagePad)
	pos, ln := multimodal.FindImageRun(ids, imgTok)
	if ln != n {
		t.Fatalf("%d pads, want %d", ln, n)
	}
	mp, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{grid}, merge)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: be, Quant: "int4", EmbedInt4: true, ResidentContext: 1600}) // small: the resident must fit beside the owner's session (this prompt is 1406 tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	fmt.Fprintf(os.Stderr, "[split] --backend %s: decode path %s\n", be, m.DecodePath())
	got, err := m.PrefillLogitsQwenVLForTest(context.Background(), ids, feats.Table, pos, ln, mp, m.NewCache(len(ids)))
	if err != nil {
		t.Fatal(err)
	}
	idx := make([]int, len(got))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return got[idx[a]] > got[idx[b]] })
	var mx, s float64
	mx = math.Inf(-1)
	for _, v := range got {
		mx = math.Max(mx, float64(v))
	}
	for _, v := range got {
		s += math.Exp(float64(v) - mx)
	}
	for _, id := range idx[:3] {
		p, _ := tk.Decode([]int{id})
		fmt.Fprintf(os.Stderr, "[split]   %q p %.3f\n", p, math.Exp(float64(got[id])-mx)/s)
	}
	if len(want) == len(got) {
		var maxd float64
		same := 0
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			maxd = math.Max(maxd, d)
			if math.Float32bits(got[i]) == math.Float32bits(want[i]) {
				same++
			}
		}
		fmt.Fprintf(os.Stderr, "[split] against the Metal-less binary's canonical int4-head arm: %d of %d logits bit-identical, max |diff| %.4g\n", same, len(got), maxd)
	}
}
