//go:build realckpt

// G-S3c's root-cause probe (docs/tasks/task-multimodal-support-2026-10.md, S3): the Qwen2.5-VL image turn's first token
// comes from the CPU prefill in both arms (Metal has no resident m-RoPE prefill), and it differed between `--backend
// metal` and `--backend cpu`. The suspect is the CPU's int4 layout, which Load keys on Options.Backend: canonical int4
// under "metal" (wantsRow4Fallback false), the arm64 row4 repack under "cpu".
//
// This test runs the same image-turn prefill on CPU-only loads that differ only in that choice (a binary without Metal
// falls back to the CPU but keeps the layout the requested name implies), plus int8 as a control, and reports:
//   - on the HF golden (qwen25vl_real_golden.json.gz: HF f32 logits for the real processor's output), each load's KL to
//     HF and its top tokens: which layout is the faithful one;
//   - on testdata/glm_ocr/table.png with serve's prompt, each load's top-3 at token 0, to set beside the served logprobs.
//
// A measurement, not a gate: it logs and fails only on a broken harness.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QWEN25VL_3B=~/models/qwen25vl-3b-instruct GOINFER_LAYOUT_FEATS=<scratch>.json \
//	  go test -tags realckpt ./decoder/ -run TestQwen25VLLayoutSplit -v -timeout 30m   (twice: the first run caches the features)
package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

type layoutArm struct {
	name, backend, quant string
	embedInt4            bool
}

// The arms. The int8 control comes first (the others are scored against it on table.png). EmbedInt4 is what serve's CLIs
// turn on by default (internal/loadflags): Qwen2.5-VL-3B ties its embedding to the LM head, so under it the head is int4.
var layoutArms = []layoutArm{
	{"int8 weight-only (control)", "cpu", "int8", false},
	{"row4, int8 head", "cpu", "int4", false},
	{"canonical, int8 head", "metal", "int4", false},
	{"row4, int4 head (serve --backend cpu)", "cpu", "int4", true},
	{"canonical, int4 head (serve --backend metal's CPU side)", "metal", "int4", true},
}

// logSoftmax64 is log p over the vocabulary, in float64.
func logSoftmax64(l []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range l {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for _, v := range l {
		s += math.Exp(float64(v) - mx)
	}
	out := make([]float64, len(l))
	for i, v := range l {
		out[i] = float64(v) - mx - math.Log(s)
	}
	return out
}

// klTo is KL(ref || got) in nats.
func klTo(ref, got []float32) float64 {
	lr, lg := logSoftmax64(ref), logSoftmax64(got)
	var kl float64
	for i := range lr {
		if p := math.Exp(lr[i]); p > 0 {
			kl += p * (lr[i] - lg[i])
		}
	}
	return kl
}

func layoutTopK(l []float32, k int) []int {
	idx := make([]int, len(l))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return l[idx[a]] > l[idx[b]] })
	return idx[:k]
}

func describeTop(tk *tokenizer.Tokenizer, l []float32, k int) string {
	lp := logSoftmax64(l)
	var b strings.Builder
	for i, id := range layoutTopK(l, k) {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%q p %.3f (lp %.3f)", layoutPiece(tk, id), math.Exp(lp[id]), lp[id])
	}
	return b.String()
}

func layoutPiece(tk *tokenizer.Tokenizer, id int) string {
	s, err := tk.Decode([]int{id})
	if err != nil {
		return fmt.Sprintf("<%d>", id)
	}
	return s
}

func freeNow() { runtime.GC(); debug.FreeOSMemory() }

func TestQwen25VLLayoutSplit(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN25VL_3B")
	if strings.HasPrefix(ckpt, "/Volumes/") || strings.HasPrefix(ckpt, "/srv/models") {
		t.Fatalf("%s is the archive", ckpt)
	}
	raw, err := readGolden("../testdata/qwen25vl_real_golden.json.gz")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g struct {
		InputIDs     []int     `json:"input_ids"`
		ImageToken   int       `json:"image_token_id"`
		ImageStart   int       `json:"image_token_start"`
		NImageTokens int       `json:"n_image_tokens"`
		GridTHW      [][3]int  `json:"grid_thw"`
		PixelValues  []float32 `json:"pixel_values"`
		LastLogits   []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	const merge = 2
	tk, err := tokenizer.Load(filepath.Join(ckpt, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}

	pp, err := multimodal.LoadQwenPreprocessConfig(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	img, err := os.ReadFile("../testdata/glm_ocr/table.png")
	if err != nil {
		t.Fatal(err)
	}
	pv, grid, err := multimodal.QwenPreprocess(img, pp)
	if err != nil {
		t.Fatal(err)
	}
	// The tower, float32 on the CPU, in its own process: the features are cached to GOINFER_LAYOUT_FEATS on the first run
	// and the test stops there, so the decoder loads run with the tower's memory returned (the fit guard prices the
	// machine's free memory, and the two together did not fit the Mac by day).
	cacheFile := os.Getenv("GOINFER_LAYOUT_FEATS")
	if cacheFile == "" {
		t.Skip("set GOINFER_LAYOUT_FEATS to a scratch file: the first run caches the tower's features, the second compares")
	}
	var feats struct{ Gold, Table []float32 }
	if b, err := os.ReadFile(cacheFile); err == nil {
		if err := json.Unmarshal(b, &feats); err != nil {
			t.Fatal(err)
		}
	} else {
		enc, err := vision.LoadQwenVisionEncoder(ckpt, false)
		if err != nil {
			t.Fatal(err)
		}
		if feats.Gold, err = enc.Forward(g.PixelValues, g.GridTHW); err != nil {
			t.Fatal(err)
		}
		if feats.Table, err = enc.Forward(pv, [][3]int{grid}); err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(feats)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(cacheFile, b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Skipf("features cached to %s; run again for the comparison", cacheFile)
	}
	goldFeats, tableFeats := feats.Gold, feats.Table

	// serve's prompt for table.png: Qwen2.5-VL's template with its default system turn, the image block inline before the
	// text (multimodal.QwenImageBlock, no newline either side, M-38).
	n := multimodal.QwenMergedTokens(grid, merge)
	text := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + multimodal.QwenImageBlock(n) +
		"What does this image show? Answer briefly.<|im_end|>\n<|im_start|>assistant\n"
	tableIDs, err := tk.Encode(text, false)
	if err != nil {
		t.Fatal(err)
	}
	imgTok, ok := tk.TokenID(multimodal.QwenImagePad)
	if !ok {
		t.Fatal("no image pad token")
	}
	tPos, tLen := multimodal.FindImageRun(tableIDs, imgTok)
	if tLen != n {
		t.Fatalf("table prompt: %d pads, want %d (the tokenizer split the special tokens)", tLen, n)
	}
	fmt.Fprintf(os.Stderr, "[layout] table.png grid %v, %d image tokens at %d, prompt %d tokens\n", grid, n, tPos, len(tableIDs))

	gPos, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatal(err)
	}
	tMPos, err := mropePositions(tableIDs, imgTok, [][3]int{grid}, merge)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "[layout] HF golden top-3: %s\n", describeTop(tk, g.LastLogits, 3))

	// One arm per process (GOINFER_LAYOUT_ARM=<index>): a closed model's memory is not back with the OS in time for the
	// next load's fit guard on a 16 GB Mac by day. Each arm's table.png logits go next to the features file, so the last
	// arm can score every earlier one against the int8 control.
	armSel := os.Getenv("GOINFER_LAYOUT_ARM")
	tableLogits := map[string][]float32{}
	for ai, a := range layoutArms {
		if armSel != "" && armSel != fmt.Sprint(ai) {
			if b, err := os.ReadFile(fmt.Sprintf("%s.arm%d", cacheFile, ai)); err == nil {
				var l []float32
				if json.Unmarshal(b, &l) == nil {
					tableLogits[a.name] = l
				}
			}
			continue
		}
		m, err := Load(ckpt, Options{Backend: a.backend, Quant: a.quant, EmbedInt4: a.embedInt4, ResidentContext: 4096}) // the guard prices KV at this, not 128K
		if err != nil {
			t.Fatalf("%s: %v", a.name, err)
		}
		fmt.Fprintf(os.Stderr, "[layout] %s: decode path %s\n", a.name, m.DecodePath())
		gl, err := m.prefillLogitsQwenVL(context.Background(), g.InputIDs, goldFeats, g.ImageStart, g.NImageTokens, gPos, m.NewCache(len(g.InputIDs)))
		if err != nil {
			t.Fatalf("%s golden prefill: %v", a.name, err)
		}
		fmt.Fprintf(os.Stderr, "[layout] %s, HF golden image: KL(HF||this) %.5f nats, cosine %.6f, top-3 %s\n",
			a.name, klTo(g.LastLogits, gl), logitCosine(gl, g.LastLogits), describeTop(tk, gl, 3))
		tl, err := m.prefillLogitsQwenVL(context.Background(), tableIDs, tableFeats, tPos, tLen, tMPos, m.NewCache(len(tableIDs)))
		if err != nil {
			t.Fatalf("%s table prefill: %v", a.name, err)
		}
		tableLogits[a.name] = tl
		if b, err := json.Marshal(tl); err == nil {
			_ = os.WriteFile(fmt.Sprintf("%s.arm%d", cacheFile, ai), b, 0o644)
		}
		fmt.Fprintf(os.Stderr, "[layout] %s, table.png token 0: top-3 %s\n", a.name, describeTop(tk, tl, 3))
		m.Close()
		m = nil
		freeNow()
	}
	ref := tableLogits[layoutArms[0].name]
	for _, a := range layoutArms[1:] {
		if ref == nil || tableLogits[a.name] == nil {
			continue
		}
		fmt.Fprintf(os.Stderr, "[layout] table.png token 0: KL(int8||%s) %.5f nats\n", a.name, klTo(ref, tableLogits[a.name]))
	}
}
