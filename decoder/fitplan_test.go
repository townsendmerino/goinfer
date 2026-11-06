package decoder

import (
	"os"
	"reflect"
	"testing"
)

// loadDenseTiny loads testdata/llama-tiny, TRACKED in git (unlike the MoE/hybrid fixtures below), so this row of the
// table runs in CI unconditionally; the other two skip when their (gitignored, real) fixtures are absent.
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

// TestPlan_tableDriven is G4 (docs/tasks/task-fit-to-hardware.md §6): a table over dense, MoE, hybrid, every backend
// and every budget that pins the placement and the ctx cap. A change to the priority order is a change to this table,
// reviewed. It runs on real tiny checkpoints through Load() rather than synthetic headers
// (docs/tasks/task-gpu-paths-2026-09.md, G11), with budgets scaled to each fixture's OWN byte counts: a literal "6 to
// 64 GB" range would never reach the decline path on toy-sized fixtures.
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
		kv := m.ResidentKVBytes("cuda", ctxWant, m.kvF16, false) // what Plan prices for CUDA's paged KV (whole blocks, null blocks, tables)
		// Room for dense + KV + a bit more than topK experts, but not every expert.
		budget := dense + kv + perExpert*int64(topK+1)
		// Pinned: with the context free to shrink, resident-first placement (internal/runtime/mem/placement.go:SolvePlacement) fits this fixture's experts, under a KV block, in a block or four of context.
		p := m.Plan("cuda", budget, PlanRequest{Ctx: ctxWant, CtxPinned: true})
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
		p := m.Plan("cuda", dense+kv+1, PlanRequest{Ctx: ctxWant, CtxPinned: true})
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

	// webgpu is admitted to Plan (docs/tasks/task-fit-to-hardware.md §7, Phase 3; M-32).
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

// TestPlan_extraBytesReservedAheadOfExperts: PlanRequest.ExtraBytes lets the CALLER price a companion allocation (a
// drafter, a vision tower) as a FIXED term ahead of the elastic expert-slot count (docs/tasks/task-fit-to-hardware.md
// §2: every allocation is a term of the plan, including the ones that attach after load). The motivating case is a
// --drafter attach after BuildResident taking VRAM an MoE expert cache had already claimed
// (docs/tasks/task-gpu-paths-2026-09.md).
func TestPlan_extraBytesReservedAheadOfExperts(t *testing.T) {
	m := loadGemma4MoETiny(t)
	nExperts, _, isMoE := m.moeGeometry()
	if !isMoE || nExperts < 2 {
		t.Fatalf("fixture has %d experts (isMoE=%v) — need >=2", nExperts, isMoE)
	}
	dense := m.ResidentDenseWeightBytes()
	full := m.ResidentWeightBytesPaged(0) - dense
	perExpert := full / int64(nExperts)
	kv := m.ResidentKVBytes("cuda", 8192, m.kvF16, false) // what Plan prices for CUDA's paged KV (whole blocks, null blocks, tables)
	budget := dense + kv + perExpert*int64(nExperts)      // enough for EVERY expert with nothing else

	without := m.Plan("cuda", budget, PlanRequest{Ctx: 8192, CtxPinned: true})
	if without.Placement != PlacementResident {
		t.Fatalf("without ExtraBytes: Placement = %v, want resident (every expert fits with room to spare)", without.Placement)
	}

	// Now reserve a "drafter" that eats exactly the room the last expert needed.
	withExtra := m.Plan("cuda", budget, PlanRequest{Ctx: 8192, CtxPinned: true, ExtraBytes: perExpert})
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

// TestPlan_unrecognisedBackendDeclinesEvenForAFeatureFreeArch (M-08, docs/audit-2026-09-10.md): a nonsense backend must
// decline even for an arch with EMPTY RequiredResidentFeatures. The "nonsense backend declines" subtest above
// (moe/nonsense_backend_declines_on_features_not_bytes) covers only an arch with non-empty required features; a plain
// Llama requires none, so MissingResidentFeatures(nil) was empty and Plan fell through to RESIDENT for ANY backend
// name. The same shape applies to any family whose feature list alone does not catch the decline.
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

// A fit search calls Plan once per probe. What Plan derives from the model alone (the config's facts, and the programs coverage
// and admission check) comes from the model's memo after the first probe. A plan is the same from a cold memo, from one every
// backend has warmed, and from none (a hand-built Model's). Coverage reaches a plan only where the backend is compiled, so the
// facts the probes stored in the warm memo are also held to fresh ones, and every backend must have stored some: a probe that
// declines before pricing (an eligibility decline) derives none. Decoder's default build demand reports stand in for cuda's and
// metal's in an untagged binary. webgpu warms first, so facts kept without their backend would reach cuda and metal. A model's
// first probes build at most its two programs, none for
// webgpu, which has no operation table, and later probes build none. Not parallel: buildProgramCalls counts the whole process,
// and a test that is not parallel runs alone.
func TestPlan_derivesTheModelsOwnFactsOnce(t *testing.T) {
	backends := []string{"cuda", "metal", "webgpu"}
	reqs := []PlanRequest{{Ctx: 2048}, {Ctx: 32768, KVF16: true}}
	frees := []int64{1 << 26, 1 << 30, 1 << 36}
	compared := map[string]int{}
	// Dense; recurrent MoE; refused by cuda's and metal's operation tables; covered by cuda and metal and not webgpu.
	for _, path := range []string{"../testdata/llama-tiny", "../testdata/qwen3next-tiny", "../testdata/bailing_hybrid-tiny", "../testdata/cohere-tiny"} {
		m, err := Load(path, Options{Backend: "cpu"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if m.fitOf == nil {
			t.Fatal("Load built a model without the fit memo")
		}
		warm := new(fitMemo)
		for _, be := range []string{"webgpu", "cuda", "metal"} {
			m.fitOf = warm
			m.Plan(be, 1<<30, reqs[0])
		}
		for be, got := range warm.facts {
			compared[be]++
			if want := new(fitMemo).factsFor(m.Config(), be); !reflect.DeepEqual(got, want) {
				t.Errorf("%s on %s: the warm memo's facts are %+v, fresh ones %+v", path, be, got, want)
			}
		}
		for _, be := range backends {
			for _, req := range reqs {
				for _, free := range frees {
					m.fitOf = new(fitMemo)
					cold := m.Plan(be, free, req)
					m.fitOf = warm
					fromWarm := m.Plan(be, free, req)
					m.fitOf = nil
					none := m.Plan(be, free, req)
					if !reflect.DeepEqual(cold, fromWarm) || !reflect.DeepEqual(cold, none) {
						t.Errorf("%s on %s, %+v with %d free: a cold memo plans\n%+v\na warm one\n%+v\nand none\n%+v", path, be, req, free, cold,
							fromWarm, none)
					}
				}
			}
		}

		m.fitOf = new(fitMemo)
		before := buildProgramCalls.Load()
		m.Plan("webgpu", 1<<30, reqs[0])
		if n := buildProgramCalls.Load() - before; n != 0 {
			t.Errorf("%s: a webgpu probe built %d programs", path, n)
		}
		for _, be := range backends {
			m.Plan(be, 1<<30, reqs[0])
		}
		if n := buildProgramCalls.Load() - before; n > 2 {
			t.Errorf("%s: the first probe on each backend built %d programs, want at most coverage's and admission's", path, n)
		}
		before, probes := buildProgramCalls.Load(), 0
		for free := int64(1) << 26; free <= 1<<36; free <<= 1 {
			for _, be := range backends {
				m.Plan(be, free, reqs[0])
				probes++
			}
		}
		if n := buildProgramCalls.Load() - before; n != 0 {
			t.Errorf("%s: %d programs built over %d probes after each backend's first", path, n, probes)
		}
	}
	for _, be := range backends {
		if compared[be] == 0 {
			t.Errorf("no fixture's probe on %s stored facts in the memo, so none were compared", be)
		}
	}
}

// A memo asked first with no config (BuildQueryFor before the config is read) must still resolve the config it is given next.
func TestFitMemo_aNilConfigDoesNotLatch(t *testing.T) {
	cfg, err := loadConfig(os.DirFS("../testdata/llama-tiny"), "config.json")
	if err != nil {
		t.Fatal(err)
	}
	memo := new(fitMemo)
	if a := memo.factsFor(nil, "webgpu"); a.resolved {
		t.Fatalf("no config resolved: %+v", a)
	}
	if a := memo.factsFor(cfg, "cuda"); !a.resolved {
		t.Error("after a call with no config, the memo never resolved the config it was given")
	}
}
