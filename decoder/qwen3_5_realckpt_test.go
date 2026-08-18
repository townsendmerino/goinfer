//go:build realckpt

// Real-checkpoint gate for Qwen3.8 (model_type qwen3_5) — the loader + forward on the
// actual released Qwen/Qwen3.8-27B (27.8B dense, bf16, 18 shards, 55.6 GB).
//
// WHY THIS EXISTS SEPARATELY FROM THE TINY ORACLE. The tiny golden proves the math
// against HF at 4 layers with random weights. It cannot prove the loader reads a REAL
// Qwen3.8 checkpoint, and every surprise in this checkpoint is a NAMING or SHAPE fact
// that a config-built fixture cannot express:
//
//   - The text decoder lives under model.language_model.*, alongside a 333-tensor
//     vision tower (model.visual.*) and a 15-tensor MTP head (mtp.*) that must simply
//     never be requested. Loading the MTP block as layer 64 would be a shape error;
//     loading it as a REPLACEMENT for a real layer would not be.
//
//   - The DeltaNet projections ship as in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
//     — qkv fused, z separate — which is NEITHER qwen3_next's fused pair
//     (in_proj_qkvz + in_proj_ba) nor four fully-separate tensors. The split reader is
//     the right one here, and only a real checkpoint says so.
//
//   - head_dim is 256 while hidden is 5120 at 24 heads, so nH·hd = 6144 ≠ hidden. A
//     loader that derives head_dim from hidden/num_heads gets 213 and fails loudly —
//     but one that derives the Q PROJECTION width from hidden silently mis-shapes it.
//
//   - attn_output_gate is true, so q_proj is DOUBLE width (2·24·256 = 12288 rows).
//
//     GOINFER_HEAVY_TESTS=1 GOINFER_QWEN38=~/models/qwen3.8-27b \
//     go test -tags realckpt ./decoder/ -run TestQwen38Real -v -timeout 180m
package decoder

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestQwen38Real_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN38")

	// int4 weights: 27.8B bf16 is 55.6 GB on disk and would not fit alongside f32
	// activations in 62 GB of RAM. Activations stay f32.
	m, err := Load(ckpt, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "qwen3_5" || a.qwen35 == nil {
		t.Fatalf("arch = %q (qwen35=%v), want qwen3_5", a.Name, a.qwen35 != nil)
	}
	if a.MoE != nil {
		t.Fatal("arch.MoE != nil — Qwen3.8 is the DENSE member of this family")
	}
	if a.NumLayers != 64 || a.HiddenDim != 5120 || a.HeadDim != 256 ||
		a.NumHeads != 24 || a.NumKVHeads != 4 || a.IntermediateDim != 17408 {
		t.Errorf("geometry = %dL/h%d/hd%d/%dq/%dkv/ffn%d, want 64/5120/256/24/4/17408",
			a.NumLayers, a.HiddenDim, a.HeadDim, a.NumHeads, a.NumKVHeads, a.IntermediateDim)
	}
	// The nested text_config was flattened, but the TOP-level model_type ("qwen3_5")
	// won over text_config's ("qwen3_5_text"). Both are registered; assert the config
	// path produced the dense descriptor either way.
	if a.RotaryDim != 64 {
		t.Errorf("RotaryDim = %d, want 64 (partial_rotary_factor 0.25 × head_dim 256 — "+
			"NOT × hidden/num_heads)", a.RotaryDim)
	}
	if a.RoPEGlobalBase != 1e7 {
		t.Errorf("rope_theta = %v, want 1e7", a.RoPEGlobalBase)
	}
	if !a.QKNorm {
		t.Error("QKNorm = false — the softmax layers carry q_norm/k_norm")
	}

	// 3:1 interleave: 48 DeltaNet + 16 softmax, softmax on every 4th layer.
	lin, full := 0, 0
	for i := range a.NumLayers {
		if a.isLinearLayer(i) {
			lin++
			continue
		}
		full++
		if (i+1)%4 != 0 {
			t.Fatalf("layer %d is full_attention but is not a 4th layer — the 3:1 pattern moved", i)
		}
	}
	if lin != 48 || full != 16 {
		t.Errorf("layer mix = %d linear / %d full, want 48/16", lin, full)
	}

	// DeltaNet geometry, on real weights. GVA: 48 value heads over 16 key heads (rep 3).
	g := a.qwen35
	if g.NumKeyHeads != 16 || g.NumValueHeads != 48 || g.KeyHeadDim != 128 || g.ValueHeadDim != 128 {
		t.Errorf("DeltaNet = %dk×%d / %dv×%d, want 16×128 / 48×128",
			g.NumKeyHeads, g.KeyHeadDim, g.NumValueHeads, g.ValueHeadDim)
	}
	if g.ConvKernel != 4 {
		t.Errorf("ConvKernel = %d, want 4", g.ConvKernel)
	}
	if g.FusedDeltaNetProj {
		t.Error("FusedDeltaNetProj = true — Qwen3.8 ships in_proj_qkv/z/a/b (qkv fused, z separate), " +
			"not qwen3_next's in_proj_qkvz + in_proj_ba pair")
	}

	// The DOUBLE-WIDTH gated q_proj, on real weights: attn_output_gate is true, so the
	// projection carries query ‖ gate at 2·nH·hd. Reading it at nH·hd would take the query
	// half only and silently drop the gate.
	//
	// This family holds attention in l.qattn / l.delta (f32, parity-first) rather than in the
	// generic QProj/KProj WeightMats, which stay ZERO-ROW here — so the load itself is the
	// shape assertion: st.TensorF32 is given the expected dims and errors on a mismatch.
	// Restating the lengths keeps that implicit contract visible at the gate.
	l3 := m.w.Layers[3] // first full_attention layer
	if l3.qattn == nil || l3.delta != nil {
		t.Fatalf("layer 3 mixer = (qattn=%v, delta=%v), want the softmax set", l3.qattn != nil, l3.delta != nil)
	}
	if got, want := len(l3.qattn.qProj), 2*a.NumHeads*a.HeadDim*a.HiddenDim; got != want {
		t.Errorf("layer 3 q_proj = %d floats, want %d (query ‖ gate: 2·24·256 × 5120)", got, want)
	}
	if got, want := len(l3.qattn.kProj), a.NumKVHeads*a.HeadDim*a.HiddenDim; got != want {
		t.Errorf("layer 3 k_proj = %d floats, want %d", got, want)
	}
	// Layer 0 is the other mixer, and its projections carry the DeltaNet's own geometry:
	// in_proj_qkv is [2·keyDim + valueDim, hidden] — one FUSED tensor, unlike in_proj_z.
	l0 := m.w.Layers[0]
	if l0.delta == nil || l0.qattn != nil {
		t.Fatalf("layer 0 mixer = (qattn=%v, delta=%v), want the DeltaNet set", l0.qattn != nil, l0.delta != nil)
	}
	keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*g.NumValueHeads
	if got, want := len(l0.delta.inProjQKV), (2*keyDim+valueDim)*a.HiddenDim; got != want {
		t.Errorf("layer 0 in_proj_qkv = %d floats, want %d (q‖k‖v fused)", got, want)
	}
	if got, want := len(l0.delta.inProjZ), valueDim*a.HiddenDim; got != want {
		t.Errorf("layer 0 in_proj_z = %d floats, want %d (SEPARATE from qkv on this family)", got, want)
	}
	// Dense FFN everywhere, including on the DeltaNet layers.
	for _, i := range []int{0, 3, 63} {
		if got := m.w.Layers[i].GateProj.Rows(); got != a.IntermediateDim {
			t.Errorf("layer %d mlp.gate_proj rows = %d, want %d", i, got, a.IntermediateDim)
		}
		if m.w.Layers[i].Router.Rows() != 0 {
			t.Errorf("layer %d has a router — Qwen3.8 is dense", i)
		}
	}

	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	// The special surface forms the chat template depends on must be single ids, or
	// the prompt below reaches the model as literal angle-bracket text.
	for _, tok := range []string{"<|im_start|>", "<|im_end|>", "<think>", "</think>"} {
		if !tk.Has(tok) {
			t.Errorf("tokenizer has no single id for %q — the chat prompt would be mis-encoded", tok)
		}
	}
	// ChatML, as this checkpoint's own chat_template.jinja renders it. The trailing
	// "<think>\n\n</think>\n\n" is the template's non-thinking prelude: without it the
	// model opens a reasoning block and a short budget is spent entirely inside it,
	// failing the gate for a reason that has nothing to do with the forward.
	const prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nName three landmarks in Paris.<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	ids, err := tk.Encode(prompt, false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	// ID PARITY, not a length check. These are the exact ids transformers 5.12's
	// AutoTokenizer produces for the string above on this checkpoint (vocab 248320,
	// byte-level BPE). A length-only check passes when a special token splits into
	// pieces that happen to sum to the same count, and a coherence gate downstream
	// cannot tell "the loader is wrong" from "the model was fed a different prompt".
	wantIDs := []int{
		248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198,
		248045, 846, 198, 657, 2250, 57902, 303, 11751, 13, 248046, 198,
		248045, 74455, 198, 248068, 271, 248069, 271,
	}
	if !slices.Equal(ids, wantIDs) {
		t.Errorf("prompt ids = %v,\nwant %v (transformers 5.12)", ids, wantIDs)
	}

	// The chat template is plain ChatML despite the multimodal wrapper, so Detect
	// resolves it with no new template — asserted rather than assumed, because the
	// vision markers (<|vision_start|>) and the reasoning-effort preamble in the
	// jinja would be equally consistent with a family-specific renderer.
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Errorf("chat.Detect: %v", err)
	} else if tmpl.Name() != "chatml" {
		t.Errorf("chat template detected as %q, want chatml", tmpl.Name())
	}

	out, _ := m.Generate(context.Background(), ids, 96, SamplingParams{})
	gen := make([]int, 0, 96)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, _ := tk.Decode(gen)
	ratio := distinctTrigramRatio(text)
	t.Logf("qwen3.8-27b int4: %d tokens, distinct-trigram %.3f: %q", len(gen), ratio, strings.TrimSpace(text))
	if ratio < 0.5 {
		t.Errorf("distinct-trigram %.3f < 0.5 — degenerate output", ratio)
	}
	// Factual content, not just non-repetition: a mis-read DeltaNet projection is
	// fluent and wrong, and trigram diversity alone does not separate those.
	low := strings.ToLower(text)
	hits := 0
	for _, want := range []string{"eiffel", "louvre", "notre", "arc de triomphe", "sacr", "seine", "champs", "versailles"} {
		if strings.Contains(low, want) {
			hits++
		}
	}
	if hits < 2 {
		t.Errorf("only %d Paris landmarks named in %q — fluent but wrong is the failure mode "+
			"a trigram ratio cannot see", hits, strings.TrimSpace(text))
	}
}
