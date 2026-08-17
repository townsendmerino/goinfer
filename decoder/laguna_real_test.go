//go:build realckpt

// Real-checkpoint gate for Laguna (poolside) — the loader + forward on the actual
// released Laguna-XS.2 (33B-A3B, bf16 safetensors, 14 shards).
//
// WHY THIS EXISTS SEPARATELY FROM T1. The tiny goldens prove the math against HF at
// 4 layers and 8 experts with random weights. They cannot prove that the loader
// reads a REAL Laguna checkpoint, and that is where this family has hidden every
// one of its surprises so far — all three of these were found by reading the real
// checkpoint, and none of them are visible in a tiny fixture built from a config:
//
//   - q_norm/k_norm exist on every layer, are UNCONDITIONAL in the module, and are
//     mentioned nowhere in config.json.
//   - g_proj is [64, 2048] — per-HEAD — even though config says `gating: true`,
//     which the sibling generations' module resolves to per-element.
//   - experts ship PER-EXPERT (mlp.experts.N.*), not as the module's fused 3D
//     parameters, and the shared expert is `shared_expert` (singular) while the
//     module calls it `shared_experts`.
//
// A wrong stride or a misread schema here produces correct shapes, finite values,
// and confident nonsense — so the coherence bar is a distinct-TRIGRAM ratio over a
// CHAT-TEMPLATED prompt, not a distinct-token floor on a raw completion. (A raw
// completion prompt on an instruction-tuned checkpoint measures "did the forward
// avoid total collapse"; on gemma-4-26b that manufactured a false "int4 is broken"
// signal that survived a week.)
//
// M.1 has no gate of this kind and will not get one on this box: it is ~220B
// (89 shards, ~400GB bf16) against 62GB of RAM. Its code path is identical to
// XS.2's apart from config, and this gates that path — the same call made for
// Kimi K2, recorded in docs/task-laguna.md.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_XS2=~/models/laguna-xs2 \
//	  go test -tags realckpt ./decoder/ -run TestLagunaReal -v -timeout 180m
package decoder

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestLagunaReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_LAGUNA_XS2")

	// int4 weights: 33B bf16 is ~63GB on disk and would not fit alongside f32
	// activations in 62GB of RAM. Activations stay f32.
	m, err := Load(ckpt, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "laguna" || a.laguna == nil {
		t.Fatalf("arch = %q (laguna=%v), want laguna", a.Name, a.laguna != nil)
	}
	if a.NumLayers != 40 || a.HiddenDim != 2048 || a.HeadDim != 128 || a.NumKVHeads != 8 {
		t.Errorf("geometry = %dL/%dh/hd%d/kv%d, want 40/2048/128/8",
			a.NumLayers, a.HiddenDim, a.HeadDim, a.NumKVHeads)
	}

	// PER-LAYER QUERY HEADS, on real weights: layer 0 is full_attention with 48
	// heads (q_proj [6144,2048]) and layer 1 is sliding with 64 (q_proj [8192,2048]).
	// A uniform-head reader loads both at 48 and silently mis-shapes 30 of 40 layers.
	full, sliding := 0, 0
	for i := range a.NumLayers {
		if a.isGlobalLayer(i) {
			full++
		} else {
			sliding++
		}
		wantHeads := 64
		if a.isGlobalLayer(i) {
			wantHeads = 48
		}
		if got := a.headsAt(i); got != wantHeads {
			t.Fatalf("headsAt(%d) = %d, want %d (global=%v)", i, got, wantHeads, a.isGlobalLayer(i))
		}
		if got := m.w.Layers[i].QProj.Rows(); got != wantHeads*a.HeadDim {
			t.Fatalf("layer %d q_proj rows = %d, want %d", i, got, wantHeads*a.HeadDim)
		}
	}
	if full != 10 || sliding != 30 {
		t.Errorf("layer types = %d full / %d sliding, want 10/30", full, sliding)
	}

	// The gate, on real weights: [64, 2048] on a sliding layer ⇒ PER-HEAD, which
	// contradicts the checkpoint's own `gating: true`. This is the assertion that
	// would have caught the config-driven reading.
	if got := m.w.Layers[1].GProj.Rows(); got != 64 {
		t.Errorf("layer 1 g_proj rows = %d, want 64 (per-head) — config declares gating:true, "+
			"which the XS-2.1/M.1 module resolves to per-ELEMENT; XS.2's own module hardcodes "+
			"per-head and the SHIPPED TENSOR is what decides", got)
	}
	if got := m.w.Layers[0].GProj.Rows(); got != 48 {
		t.Errorf("layer 0 g_proj rows = %d, want 48 — the gate is per-head at that layer's OWN "+
			"query-head count, so it varies with the layer type too", got)
	}

	// MoE: 256 experts top-8, one dense prefix layer from mlp_layer_types.
	if a.MoE == nil {
		t.Fatal("arch.MoE is nil")
	}
	if a.MoE.NumExperts != 256 || a.MoE.TopK != 8 {
		t.Errorf("MoE = %d experts / top-%d, want 256/8", a.MoE.NumExperts, a.MoE.TopK)
	}
	if a.MoE.IntermediateDim != 512 || a.MoE.SharedIntermediateDim != 512 {
		t.Errorf("expert/shared width = %d/%d, want 512/512", a.MoE.IntermediateDim, a.MoE.SharedIntermediateDim)
	}
	if !a.MoE.SharedUngated {
		t.Error("SharedUngated = false — Laguna adds the shared expert with no outer sigmoid")
	}
	if a.MoE.RoutedScale != 2.5 {
		t.Errorf("RoutedScale = %v, want 2.5", a.MoE.RoutedScale)
	}
	// FirstKDense must come from mlp_layer_types: XS.2 ships NO mlp_only_layers, and
	// reading only that field yields 0 here — which would demand expert tensors on
	// layer 0, where the checkpoint has a plain mlp.gate_proj instead.
	if a.FirstKDense != 1 {
		t.Errorf("FirstKDense = %d, want 1 (from mlp_layer_types; XS.2 has no mlp_only_layers)", a.FirstKDense)
	}
	if m.w.Layers[0].Router.Rows() != 0 {
		t.Error("layer 0 has a router — it is the dense prefix layer")
	}
	if m.w.Layers[1].Router.Rows() != 256 {
		t.Errorf("layer 1 router rows = %d, want 256", m.w.Layers[1].Router.Rows())
	}
	// e_score_correction_bias ships under mlp.experts.* (the vLLM spelling HF
	// rewrites at load), so reading the checkpoint directly must take that name.
	if len(m.w.Layers[1].RouterBias) != 256 {
		t.Errorf("layer 1 router bias len = %d, want 256 — e_score_correction_bias ships under "+
			"mlp.experts.*, not mlp.gate.*", len(m.w.Layers[1].RouterBias))
	}

	tk, err := tokenizer.Load(ckpt)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	// The prompt is the EXACT string poolside's own chat_template.jinja renders for
	// this turn, captured via transformers' apply_chat_template. It is a literal here
	// rather than built through chat.Detect because Laguna's template reaches goinfer
	// through neither channel Detect reads: it ships as a chat_template.jinja SIDECAR
	// (tokenizer_config.json has no chat_template key, so tk.ChatTemplate() is empty),
	// and its markers are plain <system>/<user>/<assistant> tags that match no native
	// template. Hand-writing an approximation would put the model off-distribution and
	// measure the prompt instead of the loader.
	//
	// THE TRAILING "</think>" IS LOAD-BEARING: the template's enable_thinking defaults
	// to false and emits a closing </think> to suppress reasoning. Dropping it puts the
	// model in THINKING mode, where a 48-token budget is spent entirely on reasoning
	// and never reaches an answer — a gate that would then fail for a reason having
	// nothing to do with the forward pass.
	const prompt = "\u3008|EOS|\u3009<system>\n\nYou are a helpful, conversationally-fluent assistant made by " +
		"Poolside. You are here to be helpful to users through natural language conversations.\n</system>\n" +
		"<user>\nName three landmarks in Paris.\n</user>\n<assistant>\n</think>"
	ids, err := tk.Encode(prompt, false) // BOS is already the leading token of the template
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	// transformers tokenizes this exact string to 56 ids. A mismatch means goinfer's
	// added-token trie did not fire on the special surface forms (〈|EOS|〉, </think>),
	// which would silently feed the model a different prompt than the reference.
	if len(ids) != 56 {
		t.Errorf("prompt encoded to %d ids, want 56 (transformers) — special tokens may not have "+
			"been recognized as single ids", len(ids))
	}

	out, _ := m.Generate(context.Background(), ids, 48, SamplingParams{})
	gen := make([]int, 0, 48)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, _ := tk.Decode(gen)
	ratio := distinctTrigramRatio(text)
	t.Logf("laguna XS.2 int4: %d tokens, distinct-trigram %.3f: %q", len(gen), ratio, strings.TrimSpace(text))
	if ratio < 0.5 {
		t.Errorf("distinct-trigram %.3f < 0.5 — degenerate output", ratio)
	}
	// Factual content, not just non-repetition: a mis-strided expert read is fluent
	// and wrong, and trigram diversity alone does not separate those.
	low := strings.ToLower(text)
	hits := 0
	for _, want := range []string{"eiffel", "louvre", "notre", "arc de triomphe", "sacr", "seine", "champs"} {
		if strings.Contains(low, want) {
			hits++
		}
	}
	if hits < 2 {
		t.Errorf("only %d Paris landmarks named in %q — fluent but wrong is the failure mode "+
			"a trigram ratio cannot see", hits, strings.TrimSpace(text))
	}
}
