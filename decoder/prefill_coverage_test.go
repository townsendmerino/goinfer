package decoder

import (
	"fmt"
	"sort"
	"testing"
)

// TestPrefillCoverageAudit enumerates every validated family against the cuda PrefillLast guards it models: not-resident
// (ResidentEligible for cuda) and MoE; the body asserts nothing about the other PrefillLast guards (gemma4-moe, sandwich
// norms, qk-norm, K=V, non-uniform geometry). It reports, per family, whether it GETS batched prefill or FALLS BACK, and
// which guard fires, so "extend the guard" work can be scoped by which guard blocks the most families. Guards are read from
// the resolved Architecture; int4-weight and over-cap declines are checkpoint/prompt-specific, not family-inherent, so they
// are not tabulated.
func TestPrefillCoverageAudit(t *testing.T) {
	// parity-manifest family → representativeConfig key (aliases where they differ).
	families := map[string]string{
		"cohere": "cohere", "cohere2": "cohere2", "deepseek_v2": "deepseek_v2",
		"deepseek_v3": "deepseek_v3", "gemma3": "gemma3", "gemma4": "gemma4",
		"glm4_moe": "glm4_moe", "gpt-oss": "gpt_oss", "gpt2": "gpt2",
		"granitemoehybrid": "granitemoehybrid", "kimi_k2": "kimi_k2", "llama": "llama",
		"llama4_text": "llama4_text", "mellum": "mellum", "mistral": "mistral",
		"mixtral": "mixtral", "nemotron_h": "nemotron_h", "phi3": "phi3",
		"qwen2": "qwen2", "qwen2_5_vl": "qwen2_5_vl", "qwen2_moe": "qwen2_moe",
		"qwen3": "qwen3", "qwen3_5_moe": "qwen3_5_moe",
	}
	names := make([]string, 0, len(families))
	for n := range families {
		names = append(names, n)
	}
	sort.Strings(names)

	batched, fallback := 0, 0
	byGuard := map[string]int{}
	t.Logf("%-18s %-10s  %s", "family", "verdict", "reason (first guard)")
	t.Logf("%s", "------------------------------------------------------------------")
	for _, fam := range names {
		cfg := representativeConfig(families[fam])
		if cfg == nil {
			t.Logf("%-18s %-10s  no representativeConfig", fam, "SKIP")
			continue
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Logf("%-18s %-10s  resolveArchitecture: %v", fam, "SKIP", err)
			continue
		}
		// The guards, in PrefillLast order. Not-resident is the prior gate (never reaches PrefillLast).
		reason := ""
		switch {
		case !ResidentEligible(arch, "cuda"):
			if !arch.decodeRunnerEligible() {
				reason = "not resident (family class unsupported)"
			} else {
				miss := missingFeatures(arch.residentFeatures(), residentBackendFeatures["cuda"])
				reason = fmt.Sprintf("not resident (cuda missing: %v)", miss)
			}
		case arch.MoE != nil:
			reason = "MoE"
		}
		// qk-norm and sandwich norms are not guards: batched prefill applies them via qk_norm_batched / rmsnorm_f32_batched
		// (bit-identical per token; cuda.TestPrefillLast_qwen3 / _gemma3). gemma3 batches only when the gemma resident path is
		// enabled (GOINFER_GEMMA4_RESIDENT). K=V is a Gemma-4-only property (nested in gemma4Params) that Gemma-4's other guards
		// trip first, so it never surfaces as the binding guard and is not modelled.
		if reason == "" {
			batched++
			t.Logf("%-18s %-10s  —", fam, "BATCHED")
		} else {
			fallback++
			byGuard[reason]++
			t.Logf("%-18s %-10s  %s", fam, "fallback", reason)
		}
	}
	t.Logf("%s", "------------------------------------------------------------------")
	t.Logf("BATCHED: %d   FALLBACK: %d", batched, fallback)
	gk := make([]string, 0, len(byGuard))
	for g := range byGuard {
		gk = append(gk, g)
	}
	sort.Slice(gk, func(i, j int) bool { return byGuard[gk[i]] > byGuard[gk[j]] })
	for _, g := range gk {
		t.Logf("  %2d × %s", byGuard[g], g)
	}
}
