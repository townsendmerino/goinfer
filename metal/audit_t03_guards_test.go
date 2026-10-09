//go:build darwin

package metal

import (
	"context"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// T0.3 of docs/tasks/task-metal-audit-2026-10.md: three guards from docs/audit-metal-2026-09-30.md that fail closed.

// TestAttnFAMaxG_matchesKernel: attnFAGroupOK's bound is only as good as its constant, which must be the kernel's own
// array size (F-C03).
func TestAttnFAMaxG_matchesKernel(t *testing.T) {
	m := regexp.MustCompile(`#define ATTN_FA_MAXG (\d+)`).FindStringSubmatch(allKernels)
	if m == nil {
		t.Fatal("kernels.go no longer defines ATTN_FA_MAXG; re-derive attnFAMaxG from attention_fa's arrays")
	}
	if n, _ := strconv.Atoi(m[1]); n != attnFAMaxG {
		t.Fatalf("ATTN_FA_MAXG is %d in kernels.go, attnFAMaxG is %d", n, attnFAMaxG)
	}
}

// TestAttnFA_declinesGroupsAboveMaxG (F-C03): a layer with more query heads per KV head than attention_fa's arrays hold
// stays on the shipped kernel, in the single-token step and the batched step alike, and a group that fits still
// engages.
func TestAttnFA_declinesGroupsAboveMaxG(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, c := range []struct {
		nH, nKV int
		want    bool
	}{
		{16, 2, true},  // G = 8, the arrays' size
		{24, 2, false}, // G = 12
		{32, 2, false}, // G = 16, as Llama 3.1 405B has
		{12, 2, true},  // G = 6, the 1.5B's
	} {
		r := &resident{
			decodeAttnFA: true, attnFAPartial: d.NewBufferLen(1), nH: c.nH,
			layers:   []residLayer{{geom: &attnGeom{hd: 128, nKV: c.nKV}}},
			curNKeys: 2 * attnFADepthFloor,
		}
		if got := r.canUseAttnFA(0); got != c.want {
			t.Errorf("nH %d over nKV %d (G = %d): canUseAttnFA = %v, want %v", c.nH, c.nKV, c.nH/c.nKV, got, c.want)
		}
		if got := r.canUseAttnFAAt(0, 2*attnFADepthFloor); got != c.want {
			t.Errorf("nH %d over nKV %d (G = %d): canUseAttnFAAt = %v, want %v", c.nH, c.nKV, c.nH/c.nKV, got, c.want)
		}
	}
	if attnFAGroupOK(14, 3) || attnFAGroupOK(8, 0) {
		t.Error("attnFAGroupOK admits a group that does not divide, or no KV heads")
	}
}

// TestPrefill_gptossFailsClosed (D-C01): the batched prefill kernels implement neither gpt-oss's attention sink nor its
// clamped SwiGLU with biases, and the expert-major branch hard-wires plain SwiGLU. Today gpt-oss is kept out because
// several of its features are missing from prefillFeatures. This admits every one of them, so the explicit guard is
// the only thing left between gpt-oss and that branch, and checks it holds.
func TestPrefill_gptossFailsClosed(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const path = "../decoder/testdata/gptoss_tiny.gguf"
	// Loaded at int4 with no Backend, which keeps the canonical int4 bytes Metal builds from (wantsCanonicalInt4), and
	// handed to buildResident directly.
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	missing := m.MissingResidentFeatures(prefillFeatures)
	if len(missing) == 0 {
		t.Log("gpt-oss needs no feature prefillFeatures lacks; the explicit guard is already the only one")
	}
	for _, f := range missing {
		prefillFeatures[f] = true
	}
	t.Cleanup(func() {
		for _, f := range missing {
			delete(prefillFeatures, f)
		}
	})
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build the gpt-oss resident: %v", err)
	}
	a := &metalResident{r: r, hidden: r.H}
	if !a.r.attnSink || a.r.moe == nil {
		t.Fatal("the fixture is not gpt-oss-shaped (no attention sink, or no MoE); the test proves nothing")
	}
	if a.r.prefillOK {
		t.Errorf("prefillOK is true for gpt-oss once the feature map admits %v", missing)
	}
	if ok, why := a.PrefillPath(); ok || !strings.Contains(why, "gpt-oss") {
		t.Errorf("with %v admitted to the prefill map, PrefillPath = %v, %q; want sequential, naming gpt-oss", missing, ok, why)
	}
}

// TestPrefill_declinesNonFiniteLogits (A-C02): a NaN or ±Inf in the batched pass's logits declines, so the decoder
// re-runs the prompt sequentially; finite logits pass. So does a non-finite last residual row (S17, G-NF2).
func TestPrefill_declinesNonFiniteLogits(t *testing.T) {
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	a := tinyPrefillResident(t, decoder.Options{Quant: "int4"}, 256)
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), 0); err != nil {
		t.Fatalf("the control declined: %v", err)
	}
	a.poisonPrefillLogitsForTest = true
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), 0); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Errorf("PrefillLast with a NaN logit: err %v, want the non-finite decline", err)
	}
	a.poisonPrefillLogitsForTest = false
	// S17 (G-NF2): an overflowed f16 residual declines too. The int8 head reads its NaN row as finite zeros, which the
	// logit check above cannot see (Gemma 3 4B, docs/tasks/task-multimodal-support-2026-10.md).
	a.r.poisonPrefillResidForTest = true
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), 0); err == nil || !strings.Contains(err.Error(), "residual overflowed") {
		t.Errorf("PrefillLast with an inf in the last residual row: err %v, want the overflow decline", err)
	}
	a.r.poisonPrefillResidForTest = false
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), 0); err != nil {
		t.Errorf("the control after the poisoned pass declined: %v", err)
	}
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for _, c := range []struct {
		v    []float32
		want int
	}{
		{[]float32{1, 2, 3}, -1}, {[]float32{1, nan}, 1}, {[]float32{inf, 0}, 0}, {[]float32{0, -inf}, 1},
		{[]float32{math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32}, -1},
	} {
		if got := firstNonFinite(c.v); got != c.want {
			t.Errorf("firstNonFinite(%v) = %d, want %d", c.v, got, c.want)
		}
	}
}
