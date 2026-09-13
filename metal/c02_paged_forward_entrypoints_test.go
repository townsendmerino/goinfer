//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPagedMoE_forwardEntryPointsDecline gates C-02 (audit-metal-2026-09-12.md): encodeMoEFFN (the
// NON-paged encoder encodeLayer dispatches a MoE layer to) reads the stacked all-E buffers
// (ml.expGuW/expGuS/expDW/expDS), which stay zero-value once a layer is paged (metal/moe.go's own
// comment on moeLayer.pool). Every entry point that reaches encodeLayer WITHOUT routing through
// forwardLogitsMoEPaged's per-layer paged branch used to silently compute a finite, WRONG answer
// off those zero buffers instead of failing.
//
//   - HiddenLast (the one production-reachable entry point — /v1/embeddings) must now DECLINE
//     (return an error), so decoder.Model.HiddenLast falls through to the CPU path exactly like an
//     OOM or cap decline already does (decoder/embed.go).
//   - Forward(id,pos)/ForwardArgmax(id,pos) — the raw resident methods used only by tests/gates;
//     decoder.ResidentForward's production Forward(embedding,pos) always goes through
//     ForwardEmbPipe, which IS paged-aware — must now PANIC via encodeMoEFFN's chokepoint guard
//     rather than returning the zero-value-buffer result, so a gate that lands on the non-paged
//     encoder fails loud instead of certifying wrong numbers.
func TestPagedMoE_forwardEntryPointsDecline(t *testing.T) {
	const ckpt = "../testdata/mixtral-tiny"

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4", MoECacheSlots: 3})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("mixtral-tiny did not go resident: decline=%s", m.ResidentDecline())
	}
	r, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("resident runner is %T, not *metalResident", rf)
	}
	if r.r.moe == nil || !r.r.moe.paged {
		t.Fatalf("expected a paged generic-MoE resident (MoECacheSlots=3 on mixtral-tiny's nE=8 topK=2)")
	}

	emb := m.EmbedResidentForTest(1)

	// HiddenLast must decline, not return a finite garbage vector.
	if _, err := r.HiddenLast(context.Background(), [][]float32{emb}, 0); err == nil {
		t.Error("HiddenLast on a paged MoE resident returned no error — should decline (C-02)")
	} else if !strings.Contains(err.Error(), "paged") {
		t.Errorf("decline error %q does not name paging as the reason", err)
	}

	// The decoder-level seam (decoder.Model.HiddenLast) must transparently fall through to the CPU
	// path on that decline rather than surfacing the error or a garbage vector to the caller — the
	// same fallback resident_embed_seam_test.go already gates for an OOM/cap decline.
	ids := []int{1, 2, 3}
	got, err := m.HiddenLast(ids)
	if err != nil {
		t.Fatalf("decoder.Model.HiddenLast did not fall through to CPU on the resident decline: %v", err)
	}
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu reference: %v", err)
	}
	defer mCPU.Close()
	want, err := mCPU.HiddenLast(ids)
	if err != nil {
		t.Fatalf("cpu reference HiddenLast: %v", err)
	}
	if cos, _ := cosF32(want, got); cos < 0.999 {
		t.Errorf("fallen-through HiddenLast cosine %.6f vs CPU reference — want the CPU path exactly, not a paged-Metal computation", cos)
	}

	// Forward(id,pos)/ForwardArgmax(id,pos): reaching a paged layer through the non-paged encoder
	// must panic, not silently return a wrong result.
	func() {
		defer func() {
			if recover() == nil {
				t.Error("resident.Forward(id,pos) on a paged MoE layer did not panic — C-02 regressed")
			}
		}()
		r.r.Forward(1, 0)
	}()
	func() {
		defer func() {
			if recover() == nil {
				t.Error("resident.ForwardArgmax(id,pos) on a paged MoE layer did not panic — C-02 regressed")
			}
		}()
		r.r.ForwardArgmax(1, 0)
	}()
}
