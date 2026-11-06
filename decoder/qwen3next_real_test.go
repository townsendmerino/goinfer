package decoder

import (
	"testing"
)

// TestQwen3Next_textParity is the real T1: loads a tiny-random Qwen3-Next
// checkpoint (scripts/pin_qwen3next_tiny.py) and gates the full forward
// (last-logit cosine + greedy continuation, both m.forward and Generate)
// against the real HF oracle — transformers 5.15.0's actual
// Qwen3NextForCausalLM, no trust_remote_code needed (mainline support).
//
// The fixture's config.json is deliberately rewritten (by the pin script,
// automatically) to the REAL release's shape — full_attention_interval and a
// top-level rope_theta, no layer_types, no rope_parameters — so this test
// exercises normalizeQwen3NextLayerTypes and the flat-RoPE path directly, the
// two real deltas from qwen3_5_moe this family exists to test, not just the
// already-populated shape qwen3_5_moe's own fixture already covers.
func TestQwen3Next_textParity(t *testing.T) {
	textParity{
		golden: "../testdata/qwen3next_tiny_text_golden.json", ckpt: "../testdata/qwen3next-tiny", pin: "scripts/pin_qwen3next_tiny.py",
		label:    "qwen3next text parity",
		generate: true, // guards the canBatchN exclusion
	}.run(t)
}
