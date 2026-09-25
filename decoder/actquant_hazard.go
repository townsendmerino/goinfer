package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// ActivationQuantHazard names why a family's output is unusable under int8 ACTIVATION quantization,
// or "" when none is known. Every W4A8/W8A8 path (int4, int8int8, int4mix on the CPU, and every
// resident GPU projection, since no GPU backend has an f32-activation GEMV) scales each activation
// row by one max/127. A family whose projection inputs carry massive outliers loses nearly all of
// each row to rounding.
//
// Measured for Phi-3-mini (queue-engineering.md H2, 2026-09-25): at a filler position the down_proj
// input's max/rms is ~80-90 in most layers, so a per-row scale rounds 99.9% of the row to zero;
// int4 and int8int8 fall to logit cosine ~0 vs f32 by position 16 while weight-only int8 holds
// >= 0.95 and f32 matches HF exactly. Keyed on model type, so Phi-4 (also "phi3") is covered
// without having been measured. This is a guard until per-group activation scales land in the
// kernels; it is not the fix.
func ActivationQuantHazard(modelType string) string {
	if modelType == "phi3" {
		return "Phi-3's activation outliers are rounded to zero by the per-row int8 activation scale every int4/int8int8/int4mix path uses (queue-engineering.md H2)"
	}
	return ""
}

// QuantizesActivations reports whether a --quant runs its projections with int8 activations.
// "int8" (weight-only) and f32 keep activations in f32 on the CPU.
func QuantizesActivations(quant string) bool {
	switch quant {
	case "int4", "int8int8", "int4mix":
		return true
	}
	return false
}

// PeekModelType reads a source's model type without loading any weights: a .gguf's
// general.architecture, or a safetensors directory's config.json model_type. "" when it cannot tell
// (a .giw, an unreadable file); callers treat that as "no known hazard".
func PeekModelType(path string) string {
	if strings.HasSuffix(path, ".gguf") {
		g, err := embed.OpenGGUFMmap(path)
		if err != nil {
			return ""
		}
		defer g.Close()
		arch, _ := g.Str("general.architecture")
		return arch
	}
	raw, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return ""
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	if json.Unmarshal(raw, &c) != nil {
		return ""
	}
	return c.ModelType
}
