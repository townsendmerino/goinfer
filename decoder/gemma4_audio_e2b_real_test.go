//go:build realckpt

// G-S5b of docs/tasks/task-multimodal-support-2026-10.md (S5, registered before this ran): Gemma 4 E2B on an audio
// prompt, goinfer's float32 CPU prefill against transformers' Gemma4ForConditionalGeneration in float32. Three steps, so
// the same ids reach both sides (run on a box that holds E2B in float32 twice over, about 20 GB each: nobara):
//
//  1. GOINFER_S5B_STEP=ids writes <out>/ids.json: serve's prompt for the mid clip
//     (testdata/embeddinggemma2-audio/mid.wav) through chat.Gemma4 and the image-block splice serve uses, the block
//     multimodal.Gemma4AudioBlock(n) with n = audio.Gemma4SoftTokens(frames).
//
//  2. scripts/pin_gemma4_e2b_audio_full.py reads it and writes HF's logits (<out>/hf_last_logits.f32, hf_argmax.json).
//
//  3. GOINFER_S5B_STEP=compare runs goinfer's prefill (the soft tokens from aikit's log-mel and tower, spliced by
//     prefillLogitsGemma4VL's own loop) and grades it. PASS: last-position logit cosine >= 0.999, argmax equal, and
//     argmax agreement >= 95% over the text positions after the audio block. The three registered planted defects
//     must each fail the last-position bar: (1) the audio rows times the embed scale, (2) PLE's token-identity term
//     from the audio token's id instead of PAD, (3) the <|audio> and <audio|> delimiters dropped from the ids.
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_S5B_STEP=ids|compare GOINFER_S5B_OUT=<dir> [GOINFER_GEMMA4_E2B=<dir>] \
//     go test -tags realckpt ./decoder/ -run TestGemma4E2BAudioFull -v -timeout 60m
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

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

type s5bIDs struct {
	IDs      []int  `json:"ids"`
	AudioPos int    `json:"audio_pos"`
	AudioLen int    `json:"audio_len"`
	Clip     string `json:"clip"`
	Frames   int    `json:"frames"`
}

func TestGemma4E2BAudioFull(t *testing.T) {
	requireHeavyModel(t)
	step, out := os.Getenv("GOINFER_S5B_STEP"), os.Getenv("GOINFER_S5B_OUT")
	if step == "" || out == "" {
		t.Skip("set GOINFER_S5B_STEP (ids | compare) and GOINFER_S5B_OUT")
	}
	dir := os.Getenv("GOINFER_GEMMA4_E2B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	wav, err := os.ReadFile("../testdata/embeddinggemma2-audio/mid.wav")
	if err != nil {
		t.Fatal(err)
	}
	samples, err := multimodal.DecodeWAV(wav)
	if err != nil {
		t.Fatal(err)
	}
	mel, T, err := audio.Gemma4Features(samples)
	if err != nil {
		t.Fatal(err)
	}
	n := audio.Gemma4SoftTokens(T)
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	audioTok, ok := tk.TokenID(multimodal.Gemma4AudioSoftToken)
	if !ok {
		t.Fatal("no audio soft token")
	}
	idsFile := filepath.Join(out, "ids.json")

	if step == "ids" {
		block := multimodal.Gemma4AudioBlock(n)
		segs, err := multimodal.SpliceImageBlock(chat.Gemma4().RenderSegments("", []chat.Turn{{Role: "user", Content: block + "Transcribe this audio."}}), block)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := tk.EncodeSegments(segs, false)
		if err != nil {
			t.Fatal(err)
		}
		pos, ln := multimodal.FindImageRun(ids, audioTok)
		if ln != n {
			t.Fatalf("%d audio tokens in the ids, want %d", ln, n)
		}
		b, _ := json.Marshal(s5bIDs{IDs: ids, AudioPos: pos, AudioLen: ln, Clip: "mid", Frames: T})
		if err := os.MkdirAll(out, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(idsFile, b, 0o644); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[G-S5b] wrote %d ids, audio run [%d, %d), %d frames: %s\n", len(ids), pos, pos+ln, T, idsFile)
		return
	}

	var in s5bIDs
	raw, err := os.ReadFile(idsFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	lb, err := os.ReadFile(filepath.Join(out, "hf_last_logits.f32"))
	if err != nil {
		t.Fatal(err)
	}
	hfLast := make([]float32, len(lb)/4)
	for i := range hfLast {
		hfLast[i] = math.Float32frombits(binary.LittleEndian.Uint32(lb[4*i:]))
	}
	var hfArgmax []int
	ab, err := os.ReadFile(filepath.Join(out, "hf_argmax.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(ab, &hfArgmax); err != nil {
		t.Fatal(err)
	}
	if len(hfArgmax) != len(in.IDs) {
		t.Fatalf("HF argmax for %d positions, %d ids", len(hfArgmax), len(in.IDs))
	}

	enc, err := audio.LoadGemma4AudioEncoder(dir)
	if err != nil {
		t.Fatal(err)
	}
	soft, err := enc.Forward(mel, T)
	if err != nil {
		t.Fatal(err)
	}
	enc = nil
	m, err := Load(dir, Options{}) // float32
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	arch := m.w.arch
	hidden := arch.HiddenDim
	if len(soft) != in.AudioLen*hidden {
		t.Fatalf("%d soft-token values, want %d x %d", len(soft), in.AudioLen, hidden)
	}

	// prefill is prefillLogitsGemma4VL's loop, kept open so the planted defects can change one thing each, and so every
	// position's argmax is visible. The production path's own last logits are checked against it below.
	prefill := func(ids []int, apos int, rows []float32, pleTok int) (last []float32, argmaxes []int) {
		cache := m.NewCache(len(ids) + 1)
		for i, id := range ids {
			var h []float32
			var err error
			if i >= apos && i < apos+in.AudioLen {
				off := (i - apos) * hidden
				h, err = m.runLayersGemma4FromEmbed(append([]float32(nil), rows[off:off+hidden]...), pleTok, cache)
			} else {
				h, err = m.runLayersGemma4(id, cache)
			}
			if err != nil {
				t.Fatal(err)
			}
			l := m.logitsFromHidden(h, cache)
			argmaxes = append(argmaxes, argmax(l))
			last = l
		}
		return last, argmaxes
	}
	pad := arch.gemma4.PadTokenID
	last, am := prefill(in.IDs, in.AudioPos, soft, pad)

	prod, err := m.prefillLogitsGemma4VL(context.Background(), in.IDs, soft, in.AudioPos, in.AudioLen, m.NewCache(len(in.IDs)+1))
	if err != nil {
		t.Fatal(err)
	}
	if c := logitCosine(prod, last); c < 0.9999999 {
		t.Fatalf("the test's open loop is not prefillLogitsGemma4VL (cosine %.9f): the grading would not be of the production path", c)
	}

	after := in.AudioPos + in.AudioLen
	agree, total := 0, 0
	for i := after; i < len(in.IDs); i++ {
		total++
		if am[i] == hfArgmax[i] {
			agree++
		}
	}
	cos := logitCosine(last, hfLast)
	pct := 100 * float64(agree) / float64(max(total, 1))
	fmt.Fprintf(os.Stderr, "[G-S5b] %d ids, audio [%d, %d): last-position cosine %.6f, argmax %d (HF %d); text-position argmax agreement after the block %d/%d (%.1f%%)\n",
		len(in.IDs), in.AudioPos, after, cos, argmax(last), argmax(hfLast), agree, total, pct)
	if cos < 0.999 || argmax(last) != argmax(hfLast) || pct < 95 {
		t.Errorf("G-S5b FAIL: cosine %.6f (bar 0.999), argmax %d vs HF %d, agreement %.1f%% (bar 95%%)", cos, argmax(last), argmax(hfLast), pct)
	}

	// The planted defects, each alone, must fail the last-position bar.
	scaled := append([]float32(nil), soft...)
	for i := range scaled {
		scaled[i] *= float32(arch.EmbedScale)
	}
	l1, _ := prefill(in.IDs, in.AudioPos, scaled, pad)
	l2, _ := prefill(in.IDs, in.AudioPos, soft, audioTok)
	boa, _ := tk.TokenID(multimodal.Gemma4AudioBlockStart)
	eoa, _ := tk.TokenID(multimodal.Gemma4AudioBlockEnd)
	var noDelim []int
	for _, id := range in.IDs {
		if id != boa && id != eoa {
			noDelim = append(noDelim, id)
		}
	}
	p3, _ := multimodal.FindImageRun(noDelim, audioTok)
	l3, _ := prefill(noDelim, p3, soft, pad)
	for _, d := range []struct {
		name string
		l    []float32
	}{{"(1) audio rows times the embed scale", l1}, {"(2) PLE from the audio token's id", l2}, {"(3) delimiters dropped", l3}} {
		c := logitCosine(d.l, hfLast)
		red := c < 0.999 || argmax(d.l) != argmax(hfLast)
		fmt.Fprintf(os.Stderr, "[G-S5b] planted %s: last-position cosine %.6f, argmax %d (HF %d), red %v\n", d.name, c, argmax(d.l), argmax(hfLast), red)
		if !red {
			t.Errorf("planted defect %s left the bar green: the check cannot see it", d.name)
		}
	}
}
