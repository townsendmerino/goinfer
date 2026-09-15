//go:build darwin

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestBuildResident_glmMoE_routerNotDeclined gates M-27's residency-side symptom: a GGUF MoE
// router loaded through the generic mat() helper got quantized instead of staying f32, and
// Metal's f32Mat(router) panics on a non-f32 weight kind — caught by BuildResident's own
// recover() and turned into a silent decline (ok=false, err=nil, just a stderr line). At a
// quantized mode this fired for every affected GGUF family; this pins that it no longer does,
// on the GLM branch (the shared loadLayer path covers GLM/Mellum/Qwen3-MoE/DeepSeek2).
func TestBuildResident_glmMoE_routerNotDeclined(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const gguf = "../testdata/glm-tiny.gguf"
	m, err := decoder.Load(gguf, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Skipf("no tiny GLM GGUF at %s (or load failed: %v) — run scripts/pin_glm_tiny_gguf.py", gguf, err)
	}
	defer m.Close()
	b := &metalBackend{}
	_, ok, err := b.BuildResident(m)
	if !ok {
		t.Fatalf("BuildResident declined (err=%v) — router precision regression? (M-27, docs/audit-2026-09-10.md)", err)
	}
	defer b.Close()
}
