package decoder

import (
	"testing"
)

// loadDenseTiny loads testdata/llama-tiny — TRACKED in git (712 KB), unlike the MoE/hybrid
// fixtures below, so this one row of the table runs in CI unconditionally; the other two skip
// when their (gitignored, real) fixtures are absent, same convention as every other MoE/hybrid
// test in this package.
func loadDenseTiny(t *testing.T) *Model {
	t.Helper()
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func loadHybridTiny(t *testing.T) *Model {
	t.Helper()
	return loadSkippableTiny(t, "../testdata/bailing_hybrid-tiny", "MLA+KDA hybrid mixer")
}

func loadSkippableTiny(t *testing.T, dir, what string) *Model {
	t.Helper()
	m, err := Load(dir, Options{Quant: "f32"})
	if err != nil {
		t.Skipf("no %s fixture (%s): %v", what, dir, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestPlan_tableDriven is G4 (docs/task-fit-to-hardware.md §6): "a table-driven unit test on
// synthetic headers — dense, MoE, hybrid, every backend, every budget... pins the placement and
// the ctx cap. A change to the priority order is a change to this table, reviewed." Not literally
// synthetic headers (Phase 1 was scoped Load()-based, docs/task-gpu-paths-2026-09.md's G11 entry)
// — real tiny checkpoints instead, budgets scaled to each fixture's OWN measured byte counts
// rather than the doc's literal "6 to 64 GB" (these are toy-sized parity fixtures, not real
// deployment checkpoints, so a literal GB range would never exercise the decline path at all).
func TestPlan_tableDriven(t *testing.T) {
	const ctxWant = 8192

	t.Run("dense/cpu/generous", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		p := m.Plan("cpu", 100*(dense+kv), PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementResident {
			t.Fatalf("Placement = %v, want resident (generous budget): %s", p.Placement, p.Reason)
		}
		if p.Ctx != ctxWant {
			t.Errorf("Ctx = %d, want %d (unpinned but nothing forced a shrink)", p.Ctx, ctxWant)
		}
	})

	t.Run("dense/cuda/tight_shrinks_ctx", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		perPos := m.kvBytesPerPositionAllLayers(false, false)
		// A budget that fits dense + a few hundred positions of KV but not the full 8192 requested.
		budget := dense + perPos*(ctxPlanFloor+500)
		p := m.Plan("cuda", budget, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementResident {
			t.Fatalf("Placement = %v, want resident (still fits at a smaller ctx): %s", p.Placement, p.Reason)
		}
		if p.Ctx >= ctxWant || p.Ctx < ctxPlanFloor {
			t.Errorf("Ctx = %d, want strictly between %d and %d (auto-shrunk, not floored, not unchanged)", p.Ctx, ctxPlanFloor, ctxWant)
		}
	})

	t.Run("dense/metal/pinned_ctx_declines_rather_than_shrinks", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		perPos := m.kvBytesPerPositionAllLayers(false, false)
		budget := dense + perPos*(ctxPlanFloor+500) // same tight budget as above
		p := m.Plan("metal", budget, PlanRequest{Ctx: ctxWant, CtxPinned: true})
		if p.Placement != PlacementDecline {
			t.Fatalf("Placement = %v, want decline (an explicit -ctx that doesn't fit must be refused, not silently shrunk): %s", p.Placement, p.Reason)
		}
		if p.Ctx != ctxWant {
			t.Errorf("Ctx = %d, want unchanged %d — a pinned request is not Plan's to alter", p.Ctx, ctxWant)
		}
	})

	t.Run("dense/cpu/never_declines_falls_to_weight_paged", func(t *testing.T) {
		m := loadDenseTiny(t)
		// A budget below even dense-alone at the floor context.
		p := m.Plan("cpu", 1, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementWeightPaged {
			t.Fatalf("Placement = %v, want weight-paged — CPU must never decline outright: %s", p.Placement, p.Reason)
		}
	})

	t.Run("dense/cuda/impossible_declines", func(t *testing.T) {
		m := loadDenseTiny(t)
		p := m.Plan("cuda", 1, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementDecline {
			t.Fatalf("Placement = %v, want decline (a GPU backend with essentially no budget): %s", p.Placement, p.Reason)
		}
	})

	t.Run("moe/metal/generous_admits_resident", func(t *testing.T) {
		m := loadGemma4MoETiny(t)
		dense := m.ResidentDenseWeightBytes()
		full := m.ResidentWeightBytesPaged(0) - dense
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		p := m.Plan("metal", 100*(dense+full+kv), PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementResident {
			t.Fatalf("Placement = %v, want resident (every expert fits): %s", p.Placement, p.Reason)
		}
		if p.ExpertBytesUsed != p.ExpertBytesFull {
			t.Errorf("ExpertBytesUsed = %d, want == ExpertBytesFull %d when fully resident", p.ExpertBytesUsed, p.ExpertBytesFull)
		}
	})

	t.Run("moe/cuda/tight_caches_a_subset_of_experts", func(t *testing.T) {
		m := loadGemma4MoETiny(t)
		nExperts, topK, isMoE := m.moeGeometry()
		if !isMoE || nExperts < 2 {
			t.Fatalf("fixture has %d experts (isMoE=%v) — need >=2 for a meaningful cache case", nExperts, isMoE)
		}
		dense := m.ResidentDenseWeightBytes()
		full := m.ResidentWeightBytesPaged(0) - dense
		perExpert := full / int64(nExperts)
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		// Room for dense + KV + a bit more than topK experts, but not every expert.
		budget := dense + kv + perExpert*int64(topK+1)
		p := m.Plan("cuda", budget, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementExpertCached {
			t.Fatalf("Placement = %v, want expert-cached: %s", p.Placement, p.Reason)
		}
		if p.Slots < topK || p.Slots >= nExperts {
			t.Errorf("Slots = %d, want in [%d, %d) — capped below every expert, at least top-k", p.Slots, topK, nExperts)
		}
		if p.NeedBytes() > budget {
			t.Errorf("NeedBytes() = %d exceeds the budget %d it was supposedly chosen to fit", p.NeedBytes(), budget)
		}
	})

	t.Run("moe/cuda/below_topk_declines", func(t *testing.T) {
		m := loadGemma4MoETiny(t)
		dense := m.ResidentDenseWeightBytes()
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		// Room for dense + KV and essentially nothing else — not even one expert's worth.
		p := m.Plan("cuda", dense+kv+1, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementDecline {
			t.Fatalf("Placement = %v, want decline (not even top-k slots fit): %s", p.Placement, p.Reason)
		}
	})

	t.Run("moe/nonsense_backend_declines_on_features_not_bytes", func(t *testing.T) {
		m := loadGemma4MoETiny(t)
		// An enormous budget — if this declines, it can only be the feature-eligibility check,
		// not the arithmetic, proving Plan checks eligibility BEFORE spending any byte math on it.
		p := m.Plan("a-backend-that-does-not-exist", 1<<60, PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementDecline {
			t.Fatalf("Placement = %v, want decline (unknown backend implements nothing)", p.Placement)
		}
		if p.DenseBytes != 0 {
			t.Errorf("DenseBytes = %d, want 0 — eligibility must be checked before any byte accounting runs", p.DenseBytes)
		}
	})

	t.Run("hybrid/metal/generous", func(t *testing.T) {
		m := loadHybridTiny(t)
		dense := m.ResidentDenseWeightBytes()
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		p := m.Plan("metal", 100*(dense+kv), PlanRequest{Ctx: ctxWant})
		// A hybrid (MLA+KDA) fixture may not be eligible on every backend — either a clean resident
		// admit or a feature-named decline is acceptable here; a silent panic or a byte-shaped
		// decline (implying the arithmetic ran on a backend that shouldn't have reached it) is not.
		if p.Placement == PlacementDecline && p.DenseBytes != 0 {
			t.Errorf("hybrid fixture declined with byte accounting already computed (DenseBytes=%d) — "+
				"a hybrid family metal doesn't implement should decline on FEATURES, not run the arithmetic first: %s",
				p.DenseBytes, p.Reason)
		}
	})

	// Phase 3 (task-fit-to-hardware.md §7): webgpu admitted to Plan now that M-32 is fixed.
	t.Run("dense/webgpu/generous_admits_like_other_backends", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		kv := m.kvBytesPerPositionAllLayers(false, false) * ctxWant
		p := m.Plan("webgpu", 100*(dense+kv), PlanRequest{Ctx: ctxWant})
		if p.Placement != PlacementResident {
			t.Fatalf("Placement = %v, want resident (generous budget, no precision flag, a generic GQA arch): %s", p.Placement, p.Reason)
		}
		if p.Ctx != ctxWant {
			t.Errorf("Ctx = %d, want %d", p.Ctx, ctxWant)
		}
	})

	t.Run("dense/webgpu/kv_i8_ctx_capped_at_fixed_ceiling_even_with_room", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		perPos := m.kvBytesPerPositionAllLayers(false, true) // i8
		const wantCtx = 100000                               // above the 65536 i8 ceiling
		// Deliberately generous: bytes alone would fit wantCtx, so only the fixed ceiling — not
		// the budget — should be what caps it.
		budget := dense + perPos*int64(wantCtx)*2
		p := m.Plan("webgpu", budget, PlanRequest{Ctx: wantCtx, KVI8: true})
		if p.Placement != PlacementResident {
			t.Fatalf("Placement = %v, want resident (unpinned ctx shrinks to the ceiling, does not decline): %s", p.Placement, p.Reason)
		}
		if p.Ctx != WebGPUCtxCeiling(false, true) {
			t.Errorf("Ctx = %d, want the fixed i8 ceiling %d — byte budget had room past it, so only M-32's cap explains a smaller value", p.Ctx, WebGPUCtxCeiling(false, true))
		}
	})

	t.Run("dense/webgpu/pinned_ctx_above_ceiling_declines", func(t *testing.T) {
		m := loadDenseTiny(t)
		dense := m.ResidentDenseWeightBytes()
		perPos := m.kvBytesPerPositionAllLayers(true, false) // f16
		const wantCtx = 40000                                // above the 32768 f16 ceiling
		budget := dense + perPos*int64(wantCtx)*2            // plenty of bytes; only the ceiling should bite
		p := m.Plan("webgpu", budget, PlanRequest{Ctx: wantCtx, CtxPinned: true, KVF16: true})
		if p.Placement != PlacementDecline {
			t.Fatalf("Placement = %v, want decline (a pinned ctx above webgpu's fixed f16 ceiling must be refused, not silently shrunk): %s", p.Placement, p.Reason)
		}
	})

	t.Run("hybrid/webgpu/kv_precision_declines_on_family_not_bytes", func(t *testing.T) {
		m := loadHybridTiny(t)
		_, _, _, _, _, _, _, _, mlaOK := m.MLAResidentParams()
		_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
		_, _, _, _, _, _, _, nemoOK := m.NemotronResidentParams()
		dense := m.ResidentDenseWeightBytes()
		kv := m.kvBytesPerPositionAllLayers(true, false) * ctxWant
		p := m.Plan("webgpu", 100*(dense+kv), PlanRequest{Ctx: ctxWant, KVF16: true})
		if mlaOK || dnetOK || nemoOK {
			if p.Placement != PlacementDecline {
				t.Fatalf("Placement = %v, want decline (--kv-f16 on a family webgpu's generic KV path doesn't cover): %s", p.Placement, p.Reason)
			}
			if p.DenseBytes != 0 {
				t.Errorf("DenseBytes = %d, want 0 — the M-32 family decline must fire before any byte accounting, mirroring gpu/residency.go's own check", p.DenseBytes)
			}
		} else if p.Placement == PlacementDecline && p.DenseBytes != 0 {
			t.Errorf("declined with byte accounting already computed (DenseBytes=%d) — a feature decline should not run the arithmetic first: %s", p.DenseBytes, p.Reason)
		}
	})
}

// TestPlan_extraBytesReservedAheadOfExperts is the regression this session's own G11 CUDA guard
// work (docs/task-gpu-paths-2026-09.md) traces back to: task-fit-to-hardware.md's motivating
// example (a --drafter attach after BuildResident grabbed VRAM an MoE expert cache had already
// claimed). PlanRequest.ExtraBytes exists so the CALLER can price a companion allocation (a
// drafter, a vision tower) as a FIXED term ahead of the elastic expert-slot count, per §2's "every
// allocation is a term of the plan, including the ones that attach after load."
func TestPlan_extraBytesReservedAheadOfExperts(t *testing.T) {
	m := loadGemma4MoETiny(t)
	nExperts, _, isMoE := m.moeGeometry()
	if !isMoE || nExperts < 2 {
		t.Fatalf("fixture has %d experts (isMoE=%v) — need >=2", nExperts, isMoE)
	}
	dense := m.ResidentDenseWeightBytes()
	full := m.ResidentWeightBytesPaged(0) - dense
	perExpert := full / int64(nExperts)
	kv := m.kvBytesPerPositionAllLayers(false, false) * 8192
	budget := dense + kv + perExpert*int64(nExperts) // enough for EVERY expert with nothing else

	without := m.Plan("cuda", budget, PlanRequest{Ctx: 8192})
	if without.Placement != PlacementResident {
		t.Fatalf("without ExtraBytes: Placement = %v, want resident (every expert fits with room to spare)", without.Placement)
	}

	// Now reserve a "drafter" that eats exactly the room the last expert needed.
	withExtra := m.Plan("cuda", budget, PlanRequest{Ctx: 8192, ExtraBytes: perExpert})
	if withExtra.Placement == PlacementResident {
		t.Fatalf("with ExtraBytes=%d reserved: Placement = %v, want NOT resident — the extra term must "+
			"be priced ahead of the elastic expert slots, not ignored", perExpert, withExtra.Placement)
	}
	if withExtra.NeedBytes()+0 > budget {
		// NeedBytes() does not itself include headroom beyond the budget by construction; this is
		// really asserting Plan chose something that respects the budget once ExtraBytes is added.
		t.Errorf("NeedBytes() = %d exceeds budget %d even after Plan adjusted for ExtraBytes", withExtra.NeedBytes(), budget)
	}
}

// TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch is the M-08 gate
// (docs/audit-2026-09-10.md): the existing "nonsense backend declines" test
// (moe/nonsense_backend_declines_on_features_not_bytes, above) only proves the decline for an
// arch with NON-EMPTY RequiredResidentFeatures — a plain Llama has none, so
// MissingResidentFeatures(nil) returned empty (nothing required, nothing implemented, so
// "nothing missing") and Plan fell through to RESIDENT for ANY backend name, including one that
// does not exist. This is the same shape as "Llama-4/cuda", "dense Gemma-4/webgpu" and
// "Kimi-K2/metal" in the finding: a family whose feature list alone doesn't catch the decline.
func TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch(t *testing.T) {
	m := loadDenseTiny(t)
	if got := m.RequiredResidentFeatures(); len(got) != 0 {
		t.Fatalf("fixture requires %v — need a feature-free arch for this gate to mean anything", got)
	}
	p := m.Plan("a-backend-that-does-not-exist", 1<<60, PlanRequest{Ctx: 8192})
	if p.Placement != PlacementDecline {
		t.Fatalf("Placement = %v, want decline — a feature-free arch on an unregistered backend name "+
			"must still decline (M-08), not fall through to resident: %s", p.Placement, p.Reason)
	}
}
