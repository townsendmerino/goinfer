//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_declinesPagedGenericMoE is audit-2026-09-10's C-08: PrefillLast's MoE row loop
// (metal/prefill.go, "if L.moe != nil") always calls the NON-paged encodeMoERoute/encodeMoEExperts
// pair, unconditionally — but a paged layer's expGuW/expGuS/expDW/expDS buffers are zero-value
// (moe.go: "stay zero-value when paged"), since a paged layer's real weights live in the slot pool
// instead. Calling the non-paged encoders on a paged layer binds those zero-value buffers, the
// same predicate the dense Gemma-4 MoE guard already uses (!m.HasGemma4MoEResident()) but never
// applied to the GENERIC MoE twin (r.moe != nil && r.moe.paged).
//
// Proven directly on prefillOK rather than by running PrefillLast and comparing logits: prefillOK
// is the ONLY thing every caller (metal/backend.go) checks before invoking PrefillLast at all, so
// if it declines correctly no caller ever reaches the buggy row loop with a paged layer — the same
// reasoning docs/task-gpu-paths-2026-09.md's own G8 Gemma-4 MoE guard rests on.
func TestPrefillLast_declinesPagedGenericMoE(t *testing.T) {
	const ckpt = "../testdata/mixtral-tiny"
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no fixture (%s/config.json)", ckpt)
	}

	// mixtral-tiny: num_local_experts=8, num_experts_per_tok=2 (checked directly below rather than
	// assumed, since a fixture change would silently make GOINFER_METAL_MOE_SLOTS=4 stop paging).
	t.Setenv("GOINFER_METAL_MOE_SLOTS", "4")
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if r.moe == nil {
		t.Fatal("fixture has no generic MoE — this gate needs FeatMoE to mean anything")
	}
	if !r.moe.paged {
		t.Fatalf("r.moe.paged = false with GOINFER_METAL_MOE_SLOTS=4 and 8 experts — test setup "+
			"assumption broke (slots %d >= experts?)", r.moe.slots)
	}
	if r.prefillOK {
		t.Error("prefillOK = true for a PAGED generic MoE — PrefillLast's row loop would bind the " +
			"zero-value expert buffers a paged layer never populates (C-08)")
	}
}

// TestPrefillLast_admitsNonPagedGenericMoE is the same gate's other side: the C-08 fix must be
// scoped to PAGED generic MoE only — declining every generic MoE regardless of paging would
// silently defeat P-15's whole batched-prefill benefit for the common (all-experts-resident) case.
func TestPrefillLast_admitsNonPagedGenericMoE(t *testing.T) {
	const ckpt = "../testdata/mixtral-tiny"
	if _, err := os.Stat(ckpt + "/config.json"); err != nil {
		t.Skipf("no fixture (%s/config.json)", ckpt)
	}
	os.Unsetenv("GOINFER_METAL_MOE_SLOTS")

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if r.moe == nil {
		t.Fatal("fixture has no generic MoE — this gate needs FeatMoE to mean anything")
	}
	if r.moe.paged {
		t.Fatal("r.moe.paged = true with GOINFER_METAL_MOE_SLOTS unset — test setup assumption broke")
	}
	if !r.prefillOK {
		t.Error("prefillOK = false for a NON-paged generic MoE — the C-08 fix must not decline the " +
			"all-experts-resident case, which is what PrefillLast is for in the first place")
	}
}
