//go:build realckpt

// Real-GGUF gate for Nemotron 3 Nano (nemotron_h MoE, 30B-A3B) — the loader + MoE forward on
// actual released quantized weights.
//
// WHY THIS EXISTS SEPARATELY FROM T3. T3 (nemotron3nano_real_test.go) proves the forward against
// an HF oracle, but it loads SAFETENSORS. The GGUF path is a different loader with a different
// expert layout — safetensors ships one tensor per expert
// (experts.I.{up,down}_proj), GGUF fuses ALL experts into one 3-D tensor per projection
// (blk.N.ffn_{up,down}_exps.weight) — and it had never completed a forward pass. It was verified
// by reading a real file's header and dequantizing one layer's tensors by hand; correct dims and
// sane values, which is not the same as a model that runs.
//
// That gap matters here more than usual: a fused expert stack read with the wrong stride
// produces correct shapes, finite values, and confident nonsense. This session has now seen that
// failure mode five times in a different subsystem.
//
// THE PROMPT GOES THROUGH THE CHAT TEMPLATE, and the coherence bar is a distinct-TRIGRAM ratio,
// not a distinct-token floor. The dense gate (TestNemotronReal_gate) carries an audit note
// saying exactly this: its raw completion prompt on an instruction-tuned checkpoint measures
// "did the forward avoid TOTAL collapse", not coherence, and on gemma-4-26b that manufactured a
// false "int4 is broken" signal that survived a week — which a distinct<3 floor would not have
// caught, because repetition has more than 3 distinct tokens. New gate, so it starts with the
// pattern that note recommends rather than inheriting the one it warns about.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_NEMOTRON3NANO_GGUF=~/models/nemotron3nano-gguf/... \
//	  go test -tags realckpt ./decoder/ -run TestNemotron3NanoMoEReal -v -timeout 60m
package decoder

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestNemotron3NanoMoEReal_gate(t *testing.T) {
	requireHeavyModel(t)
	gguf := assetPath(t, "GOINFER_NEMOTRON3NANO_GGUF")

	m, err := Load(gguf, Options{Quant: "int8"}) // int8 WEIGHTS, f32 activations — see T3
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "nemotron_h" || a.nemotron == nil {
		t.Fatalf("arch = %q (nemotron=%v), want nemotron_h", a.Name, a.nemotron != nil)
	}
	// The GGUF arch string is nemotron_h_moe and the loader normalizes it; if that ever
	// regresses the model_type would not resolve at all, so reaching here already covers it.
	mamba, attn, moe, mlp := 0, 0, 0, 0
	for _, k := range a.nemotron.blockKind {
		switch k {
		case nemoMamba:
			mamba++
		case nemoAttn:
			attn++
		case nemoMoE:
			moe++
		case nemoMLP:
			mlp++
		}
	}
	t.Logf("blocks: mamba=%d attn=%d moe=%d mlp=%d (of %d)", mamba, attn, moe, mlp, a.NumLayers)
	if mamba != 23 || moe != 23 || attn != 6 || mlp != 0 {
		t.Errorf("block kinds = %d/%d/%d/%d, want 23 mamba / 23 moe / 6 attn / 0 mlp — the "+
			"52-char MEMEM* pattern did not survive the GGUF metadata round-trip", mamba, moe, attn, mlp)
	}
	if a.MoE == nil {
		t.Fatal("arch.MoE is nil — the GGUF expert metadata did not produce a MoE config")
	}
	if a.MoE.NumExperts != 128 || a.MoE.TopK != 6 {
		t.Errorf("MoE = %d experts / top-%d, want 128/6", a.MoE.NumExperts, a.MoE.TopK)
	}
	if a.MoE.SharedIntermediateDim != 3712 {
		t.Errorf("shared expert width = %d, want 3712 — the key that is NOT derivable from "+
			"n_shared_experts*moe_intermediate_size", a.MoE.SharedIntermediateDim)
	}

	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat.Detect: %v — refusing a raw completion prompt on an instruction-tuned "+
			"checkpoint, which is what the dense gate's audit note warns against", err)
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("",
		[]chat.Turn{{Role: "user", Content: "Name three landmarks in Paris."}}), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
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
	t.Logf("generated %d tokens, distinct-trigram %.3f: %q", len(gen), ratio, strings.TrimSpace(text))
	if ratio < 0.5 {
		t.Errorf("distinct-trigram %.3f < 0.5 — degenerate output. A fused expert stack read "+
			"with the wrong stride yields exactly this: finite values, correct shapes, "+
			"confident repetition", ratio)
	}
}
