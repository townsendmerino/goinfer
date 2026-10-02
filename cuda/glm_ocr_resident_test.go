//go:build cuda && goinfer_testhooks

package cuda

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGlmOcrResidentDeclinedCUDA: GLM-OCR must NOT go CUDA-resident. Its rotation is PAIRWISE
// (GPT-J, dims 2d/2d+1, over m-RoPE sections) and every rope kernel in this package is the NeoX
// half-split, so an admitted GLM-OCR runs the wrong rotation with no error. Measured 2026-10-01 with
// the admission check temporarily absent (RTX 2070 SUPER, int8int8, 48-token prompt, per-position
// resident vs CPU at the same quantization on testdata/glm-ocr-tiny): worst cosine -0.34, against
// 0.988 for CPU int8int8 vs CPU f32 (the quantization noise alone) and 1.000000 for a NeoX llama
// fixture built with the same peaked-attention weights. The decline is the fix; the model decodes on
// the CPU until a pairwise rope kernel exists (docs/tasks/task-glm-ocr-2026-10.md, O6), at which
// point this test is the line to replace with a resident-vs-CPU parity gate.
func TestGlmOcrResidentDeclinedCUDA(t *testing.T) {
	const ckpt = "../testdata/glm-ocr-tiny"
	requireDeviceAndFixture(t, ckpt)
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if rf := m.ResidentForwardForTest(); rf != nil {
		if _, ok := rf.(*cudaResident); ok {
			t.Fatalf("glm_ocr went CUDA-resident: the resident rope is NeoX and the model's is pairwise")
		}
	}
	if why := m.ResidentDecline(); !strings.Contains(why, "pairwise-mrope") {
		t.Errorf("decline reason = %q, want it to name pairwise-mrope", why)
	} else {
		t.Logf("decline: %s", why)
	}
}
