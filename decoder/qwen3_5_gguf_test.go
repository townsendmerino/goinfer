//go:build realckpt

// Real-GGUF gate for Qwen3.8 (llama.cpp arch "qwen35") — the dense sibling of the qwen35moe GGUF
// path, and the loader the safetensors bring-up explicitly deferred.
//
// WHY A SEPARATE GATE FROM THE SAFETENSORS ONE. Different container, different metadata dialect, and
// llama.cpp's converter bakes transforms the HF path never sees — V heads TILED for ggml's
// broadcast, A_log stored as −exp(A_log), the standard norms (1+w)'d with ssm_norm exempt. Those are
// reversed at load so one forward serves both; a mistake in any of them yields correct shapes,
// finite values and confident nonsense, which no shape check catches.
//
// It also pins the two metadata facts that differ from the MoE sibling and would silently mis-size
// the model: block_count COUNTS the NextN/MTP block (65 for a 64-layer model), and layer_types is
// not stated at all — the 3:1 interleave is COMPUTED from full_attention_interval.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_QWEN38_GGUF=~/models/qwen38-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
//	  go test -tags realckpt ./decoder/ -run TestQwen38GGUF -v -timeout 60m
package decoder

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestQwen38GGUF_gate(t *testing.T) {
	requireHeavyModel(t)
	path := assetPath(t, "GOINFER_QWEN38_GGUF")

	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s): %v", path, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "qwen3_5" || a.qwen35 == nil {
		t.Fatalf("arch = %q (qwen35=%v), want qwen3_5", a.Name, a.qwen35 != nil)
	}
	if a.MoE != nil {
		t.Fatal("arch.MoE != nil — the dense GGUF must not resolve to the MoE adapter")
	}
	// 64, not 65: block_count includes the MTP block this decoder does not run. Getting this wrong
	// is not a crash, it is a model with one junk layer.
	if a.NumLayers != 64 {
		t.Errorf("NumLayers = %d, want 64 (block_count 65 minus nextn_predict_layers 1)", a.NumLayers)
	}
	if a.HiddenDim != 5120 || a.HeadDim != 256 || a.NumHeads != 24 || a.NumKVHeads != 4 || a.IntermediateDim != 17408 {
		t.Errorf("geometry = h%d/hd%d/%dq/%dkv/ffn%d, want 5120/256/24/4/17408",
			a.HiddenDim, a.HeadDim, a.NumHeads, a.NumKVHeads, a.IntermediateDim)
	}
	// The interleave is COMPUTED from full_attention_interval=4; llama.cpp states no layer_types.
	lin, full := 0, 0
	for i := range a.NumLayers {
		if a.isLinearLayer(i) {
			lin++
			continue
		}
		full++
		if (i+1)%4 != 0 {
			t.Fatalf("layer %d is full_attention but not a 4th layer — the computed 3:1 pattern is wrong", i)
		}
	}
	if lin != 48 || full != 16 {
		t.Errorf("layer mix = %d linear / %d full, want 48/16", lin, full)
	}
	g := a.qwen35
	if g.NumKeyHeads != 16 || g.NumValueHeads != 48 || g.KeyHeadDim != 128 || g.ValueHeadDim != 128 || g.ConvKernel != 4 {
		t.Errorf("DeltaNet = %dk×%d / %dv×%d conv%d, want 16×128 / 48×128 conv4",
			g.NumKeyHeads, g.KeyHeadDim, g.NumValueHeads, g.ValueHeadDim, g.ConvKernel)
	}
	if a.RotaryDim != 64 {
		t.Errorf("RotaryDim = %d, want 64 (rope.dimension_count)", a.RotaryDim)
	}

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	const prompt = "<|im_start|>user\nName three landmarks in Paris.<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	ids, err := tk.Encode(prompt, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 64, SamplingParams{})
	gen := make([]int, 0, 64)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, _ := tk.Decode(gen)
	ratio := distinctTrigramRatio(text)
	t.Logf("qwen3.8 GGUF Q4_K_M→int4: %d tokens, distinct-trigram %.3f: %q", len(gen), ratio, strings.TrimSpace(text))
	if ratio < 0.5 {
		t.Errorf("distinct-trigram %.3f < 0.5 — degenerate output, the shape a mis-reversed transform produces", ratio)
	}
	low := strings.ToLower(text)
	hits := 0
	for _, want := range []string{"eiffel", "louvre", "notre", "arc de triomphe", "sacr", "seine", "champs", "versailles"} {
		if strings.Contains(low, want) {
			hits++
		}
	}
	if hits < 2 {
		t.Errorf("only %d Paris landmarks named in %q — fluent but wrong is what a bad un-tile looks like",
			hits, strings.TrimSpace(text))
	}
}
