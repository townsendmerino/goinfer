package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// TestNemotron3NanoMoE_ggufConfig verifies ggufNemotronConfig against REAL metadata —
// fetched directly from bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF (the first ~30MB
// of the Q4_K_M file, containing the full metadata + tensor-info section but none of
// the tensor data) and parsed with this package's own GGUF reader, not hand-crafted or
// assumed from the safetensors config's field names. Two things this specifically
// catches that a synthetic fixture wouldn't: (1) llama.cpp's GGUF architecture string
// is "nemotron_h_moe", NOT "nemotron_h" — a different string from HF's own
// model_type, which must still normalize to "nemotron_h" for goinfer's registry to
// dispatch correctly; (2) the "moe" layers are marked in feed_forward_length (the
// SAME array plain nemotron_h uses for "mlp"), not a separate array — confirmed real,
// not assumed, because this checkpoint's array has no zero-only "dense mlp" pattern to
// tell them apart otherwise.
func TestNemotron3NanoMoE_ggufConfig(t *testing.T) {
	kv := func(vals ...uint32) []any {
		out := make([]any, len(vals))
		for i, v := range vals {
			out[i] = v
		}
		return out
	}
	// nemotron_h_moe.attention.head_count_kv, verbatim from the real file (52 entries;
	// nonzero=2 at attention-layer positions).
	headCountKV := kv(0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	// nemotron_h_moe.feed_forward_length, verbatim (nonzero=1856 at moe-layer positions
	// — this checkpoint has zero plain dense-mlp layers, so every nonzero entry here is
	// "moe", confirmed by cross-referencing against the real hybrid_override_pattern's
	// letter count from the safetensors config, not assumed).
	feedForward := kv(0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 1856, 0, 0, 1856, 0, 1856, 0, 1856, 0, 1856, 0, 1856)
	if len(headCountKV) != 52 || len(feedForward) != 52 {
		t.Fatalf("test data itself is wrong length: kv=%d ff=%d, want 52 each", len(headCountKV), len(feedForward))
	}
	g := &embed.GGUFFile{Metadata: map[string]any{
		"general.architecture":                             "nemotron_h_moe",
		"nemotron_h_moe.block_count":                       uint32(52),
		"nemotron_h_moe.embedding_length":                  uint32(2688),
		"nemotron_h_moe.attention.head_count":              uint32(32),
		"nemotron_h_moe.attention.head_count_kv":           headCountKV,
		"nemotron_h_moe.attention.key_length":              uint32(128),
		"nemotron_h_moe.attention.value_length":            uint32(128),
		"nemotron_h_moe.attention.layer_norm_rms_epsilon":  float32(1e-05),
		"nemotron_h_moe.feed_forward_length":               feedForward,
		"nemotron_h_moe.ssm.time_step_rank":                uint32(64),
		"nemotron_h_moe.ssm.inner_size":                    uint32(4096),
		"nemotron_h_moe.ssm.state_size":                    uint32(128),
		"nemotron_h_moe.ssm.group_count":                   uint32(8),
		"nemotron_h_moe.ssm.conv_kernel":                   uint32(4),
		"nemotron_h_moe.expert_count":                      uint32(128),
		"nemotron_h_moe.expert_used_count":                 uint32(6),
		"nemotron_h_moe.expert_feed_forward_length":        uint32(1856),
		"nemotron_h_moe.expert_shared_feed_forward_length": uint32(3712),
		"nemotron_h_moe.expert_shared_count":               uint32(1),
		"nemotron_h_moe.expert_weights_norm":               true,
		"nemotron_h_moe.expert_weights_scale":              float32(2.5),
		"nemotron_h_moe.expert_group_count":                uint32(1),
		"nemotron_h_moe.expert_group_used_count":           uint32(1),
		"tokenizer.ggml.tokens":                            make([]any, 131072),
	}}

	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatalf("ggufConfig: %v", err)
	}
	if cfg.ModelType != "nemotron_h" {
		t.Errorf("ModelType=%q, want %q (normalized from the GGUF arch string nemotron_h_moe)", cfg.ModelType, "nemotron_h")
	}
	switch {
	case cfg.NumLayers != 52:
		t.Errorf("NumLayers=%d, want 52", cfg.NumLayers)
	case cfg.HiddenDim != 2688:
		t.Errorf("HiddenDim=%d, want 2688", cfg.HiddenDim)
	case cfg.NRoutedExperts != 128:
		t.Errorf("NRoutedExperts=%d, want 128", cfg.NRoutedExperts)
	case cfg.NumExpertsPerTok != 6:
		t.Errorf("NumExpertsPerTok=%d, want 6", cfg.NumExpertsPerTok)
	case cfg.MoeIntermediateSize != 1856:
		t.Errorf("MoeIntermediateSize=%d, want 1856", cfg.MoeIntermediateSize)
	case cfg.MoeSharedExpertIntermediateSize != 3712:
		t.Errorf("MoeSharedExpertIntermediateSize=%d, want 3712", cfg.MoeSharedExpertIntermediateSize)
	case cfg.NSharedExperts != 1:
		t.Errorf("NSharedExperts=%d, want 1", cfg.NSharedExperts)
	case cfg.NormTopKProb == nil || !*cfg.NormTopKProb:
		t.Errorf("NormTopKProb=%v, want true", cfg.NormTopKProb)
	case cfg.RoutedScalingFactor != 2.5:
		t.Errorf("RoutedScalingFactor=%v, want 2.5", cfg.RoutedScalingFactor)
	case cfg.NGroup != 1 || cfg.TopkGroup != 1:
		t.Errorf("NGroup=%d TopkGroup=%d, want 1/1", cfg.NGroup, cfg.TopkGroup)
	}

	// Per-layer classification: attention where head_count_kv[i]>0, moe where
	// feed_forward_length[i]>0 (this checkpoint has no plain "mlp" layers), mamba
	// otherwise — fed all the way through resolveArchitecture into blockKind.
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.nemotron == nil {
		t.Fatal("arch.nemotron is nil")
	}
	if arch.MoE == nil {
		t.Fatal("arch.MoE is nil")
	}
	wantAttn := map[int]bool{5: true, 12: true, 19: true, 26: true, 33: true, 42: true}
	mismatches := 0
	for i := 0; i < 52; i++ {
		got := arch.nemotron.blockKind[i]
		switch {
		case wantAttn[i] && got != nemoAttn:
			t.Errorf("layer %d: got kind %d, want nemoAttn (head_count_kv=2)", i, got)
			mismatches++
		case !wantAttn[i] && feedForward[i].(uint32) > 0 && got != nemoMoE:
			t.Errorf("layer %d: got kind %d, want nemoMoE (feed_forward_length=1856)", i, got)
			mismatches++
		case !wantAttn[i] && feedForward[i].(uint32) == 0 && got != nemoMamba:
			t.Errorf("layer %d: got kind %d, want nemoMamba", i, got)
			mismatches++
		}
		if mismatches > 5 {
			t.Fatal("too many mismatches, stopping")
		}
	}
}
