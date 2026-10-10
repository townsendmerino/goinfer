//go:build realckpt

// G-S10c of docs/tasks/task-multimodal-support-2026-10.md (S10): Qwen3-VL-2B on an image
// prompt, goinfer's float32 CPU prefill (aikit's tower with DeepStack, then the decoder adding each DeepStack set after
// its layer) against transformers' Qwen3VLForConditionalGeneration in float32, on identical inputs: HF's own ids, pixel
// values and grid (scripts/pin_qwen3vl_image_real.py, GOINFER_S10C_OUT). PASS: last-position logit cosine >= 0.999,
// argmax equal, and argmax agreement >= 95% over the text positions after the image. Planted defects, each must fail
// the last-position bar: (1) the DeepStack sets not added; (2) added one layer late; (3) added to the text positions
// too (each set's mean image row at every text position).
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QWEN3VL_2B=<dir> GOINFER_S10C_OUT=<dir> go test -tags realckpt ./decoder/ -run TestQwen3VLImageReal -v -timeout 60m
package decoder

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

func TestQwen3VLImageReal(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN3VL_2B")
	out := os.Getenv("GOINFER_S10C_OUT")
	if out == "" {
		t.Skip("set GOINFER_S10C_OUT to pin_qwen3vl_image_real.py's output")
	}
	var in struct {
		IDs   []int  `json:"input_ids"`
		Grid  [3]int `json:"image_grid_thw"`
		Tok   int    `json:"image_token_id"`
		Merge int    `json:"spatial_merge_size"`
	}
	raw, err := os.ReadFile(filepath.Join(out, "inputs.json"))
	if err != nil {
		t.Skipf("no reference: %v", err)
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	f32s := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(out, name))
		if err != nil {
			t.Fatal(err)
		}
		v := make([]float32, len(b)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return v
	}
	pixels, hfLast := f32s("pixels.f32"), f32s("hf_last_logits.f32")
	var hfArg []int
	ab, err := os.ReadFile(filepath.Join(out, "hf_argmax.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(ab, &hfArg); err != nil {
		t.Fatal(err)
	}

	enc, err := vision.LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	merged, deep, err := enc.ForwardDeepstack(pixels, [][3]int{in.Grid})
	if err != nil {
		t.Fatal(err)
	}
	enc = nil
	m, err := Load(ckpt, Options{}) // float32
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	imgPos, imgLen := -1, 0
	for i, id := range in.IDs {
		if id == in.Tok {
			if imgPos < 0 {
				imgPos = i
			}
			imgLen++
		}
	}
	if imgLen*hidden != len(merged) || len(deep) == 0 {
		t.Fatalf("%d image tokens x %d against %d merged values, %d DeepStack sets", imgLen, hidden, len(merged), len(deep))
	}
	mp, err := mropePositions(in.IDs, in.Tok, [][3]int{in.Grid}, in.Merge)
	if err != nil {
		t.Fatal(err)
	}
	fast := m.cpuFastAttention() // the production prefill's setting

	// prefill is prefillLogitsQwenVL kept open: every position's logits, and the DeepStack rows the caller chooses.
	prefill := func(ds *deepstackRows) (last []float32, am []int) {
		cache := m.NewCache(len(in.IDs) + 1)
		h := m.embedN(in.IDs)
		copy(h[imgPos*hidden:(imgPos+imgLen)*hidden], merged)
		cache.mropePos, cache.mropeDelta = mp, mropeDelta(mp, len(in.IDs))
		cache.deepstack = ds
		hN, err := m.runLayersFromEmbedN(context.Background(), h, cache, fast)
		if err != nil {
			t.Fatal(err)
		}
		all := m.lmHeadN(hN, len(in.IDs))
		V := len(all) / len(in.IDs)
		for i := range in.IDs {
			am = append(am, argmax(all[i*V:(i+1)*V]))
		}
		return all[(len(in.IDs)-1)*V:], am
	}
	real := &deepstackRows{spans: []ImageSpan{{Pos: imgPos, Len: imgLen}}, rows: deep}
	last, am := prefill(real)

	// The production path, through GenerateQwenVLDeepstack's own cache setup, must give the open loop's last logits.
	pc := m.NewCache(len(in.IDs) + 1)
	pc.deepstack = real
	prod, err := m.prefillLogitsQwenVL(context.Background(), in.IDs, merged, imgPos, imgLen, mp, pc)
	if err != nil {
		t.Fatal(err)
	}
	if c := logitCosine(prod, last); c < 0.9999999 {
		t.Fatalf("the open loop is not prefillLogitsQwenVL (cosine %.9f)", c)
	}

	agree, total := 0, 0
	for i := imgPos + imgLen; i < len(in.IDs); i++ {
		total++
		if am[i] == hfArg[i] {
			agree++
		}
	}
	cos := logitCosine(last, hfLast)
	pct := 100 * float64(agree) / float64(max(total, 1))
	fmt.Fprintf(os.Stderr, "[G-S10c] %d ids, image [%d, %d), %d DeepStack sets, fast attention %v: last-position cosine %.6f, argmax %d (HF %d); text-position agreement after the image %d/%d (%.1f%%)\n",
		len(in.IDs), imgPos, imgPos+imgLen, len(deep), fast, cos, argmax(last), argmax(hfLast), agree, total, pct)
	if cos < 0.999 || argmax(last) != argmax(hfLast) || pct < 95 {
		t.Errorf("G-S10c FAIL: cosine %.6f (bar 0.999), argmax %d vs HF %d, agreement %.1f%% (bar 95%%)", cos, argmax(last), argmax(hfLast), pct)
	}

	zero := make([]float32, imgLen*hidden)
	textToo := &deepstackRows{spans: []ImageSpan{{Pos: 0, Len: len(in.IDs)}}}
	for _, set := range deep {
		mean := make([]float32, hidden)
		for r := range imgLen {
			for j := range hidden {
				mean[j] += set[r*hidden+j] / float32(imgLen)
			}
		}
		full := make([]float32, len(in.IDs)*hidden)
		for p := range in.IDs {
			src := mean
			if p >= imgPos && p < imgPos+imgLen {
				src = set[(p-imgPos)*hidden : (p-imgPos+1)*hidden]
			}
			copy(full[p*hidden:], src)
		}
		textToo.rows = append(textToo.rows, full)
	}
	for _, d := range []struct {
		name string
		ds   *deepstackRows
	}{
		{"(1) DeepStack not added", nil},
		{"(2) added one layer late", &deepstackRows{spans: []ImageSpan{{Pos: imgPos, Len: imgLen}}, rows: append([][]float32{zero}, deep...)}},
		{"(3) added to the text positions too", textToo},
	} {
		l, _ := prefill(d.ds)
		c := logitCosine(l, hfLast)
		red := c < 0.999 || argmax(l) != argmax(hfLast)
		fmt.Fprintf(os.Stderr, "[G-S10c] planted %s: last-position cosine %.6f, argmax %d (HF %d), red %v\n", d.name, c, argmax(l), argmax(hfLast), red)
		if !red {
			t.Errorf("planted defect %s left the bar green: the check cannot see it", d.name)
		}
	}
}
