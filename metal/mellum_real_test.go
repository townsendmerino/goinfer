//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMellumResidentParity is G11's Metal half (docs/queue-correctness.md): the real-weight validation of Mellum on the Metal
// resident path, which G10's FeatRopeMscale declaration (for gpt-oss's YaRN) admits as a side effect.
// decoder/mellum_slice_test.go (tag realckpt) already proved goinfer's CPU forward matches the HF reference on a real
// 4-layer weight slice; this is the same discipline as TestGPT2ResidentParity/TestGptOssResidentParity pointed at the sliced
// checkpoint directory. Regenerate the slice per docs/queue-correctness.md G11 if absent (4 GB, gitignored; only the golden
// ships).
func TestMellumResidentParity(t *testing.T) {
	requireHeavyModel(t)
	path := "../decoder/testdata/mellum-mellum2-slice"
	if _, err := os.Stat(path + "/model.safetensors"); err != nil {
		t.Skipf("no Mellum2 slice at %s (4 GB, gitignored) — regenerate per docs/queue-correctness.md G11", path)
	}
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatRopeMscale] {
		t.Skip("metal does not declare FeatRopeMscale — see docs/queue-correctness.md G10/G11")
	}

	// The slice's own prompt/continuation (decoder/testdata/mellum_mellum2_slice_golden.json):
	// prompt_ids [2, 1547, 913, 24, 88, 7, 100, 2001], continuation [417, 83711, 12311, 83711].
	seed := []int{2, 1547, 913, 24, 88, 7, 100, 2001, 417, 83711, 12311, 83711}
	st := residentParity(t, path, seed, len(seed))
	// A real trained-weight slice, not a calibrated full checkpoint — same int4-noise floor as
	// the other resident gates (GPT-2 0.95, gpt-oss 0.95), not decoder_slice_test.go's 0.9999
	// (that one compares f32-vs-f32; this compares int4/int8-quantized Metal against int8 CPU).
	assertParity(t, "mellum2 slice (real weights, G11)", st, 0.95)
}
