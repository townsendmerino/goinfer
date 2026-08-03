//go:build darwin

package metal

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4Paging_bitExact is the PRIMARY gate for the synchronous expert-paging path: paging must
// be computationally TRANSPARENT — the paged forward (experts staged into a bounded LRU slot pool)
// must produce BYTE-IDENTICAL logits to the non-paged forward (all experts resident) on a model that
// fits both ways. Same kernels, same weights (a slot's bytes == the stacked buffer's rows for that
// expert), so the only difference is where the expert weights live — the result must not move a bit.
// N is forced small (< nE) so eviction and cold-start actually FIRE during the forward; a pass where
// the pool never evicts would prove only the easy half. Reports which of the three (cold start,
// eviction, same-expert reuse) triggered, with counts. Metal-vs-CPU stays the inherent argmax/near-tie
// relationship (f16 KV) — that is the SECONDARY gate, not this one.
func TestGemma4Paging_bitExact(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture (%s)", ckpt)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")

	// nE / topK to size the pool.
	mProbe, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load probe: %v", err)
	}
	nE, topK := 0, 0
	for l := 0; l < 64; l++ {
		if _, _, e, k, _, ok := mProbe.Gemma4MoERouterForTest(l); ok {
			nE, topK = e, k
			break
		}
	}
	mProbe.Close()
	if nE <= topK {
		t.Skipf("fixture nE=%d <= topK=%d — can't force paging; width fixture (nE=128) needed", nE, topK)
	}
	// N small enough to force eviction over the prompt, but > topK so some experts survive → reuse.
	N := topK + topK/2
	if N >= nE {
		N = nE - 1
	}
	if N < topK {
		N = topK
	}

	// Non-paged reference (env MOE_SLOTS unset).
	os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
	nonPaged := runG4(t, ckpt)

	// Paged (env MOE_SLOTS=N).
	t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(N))
	mg, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load paged: %v", err)
	}
	defer mg.Close()
	r, err := BuildResident(mg)
	if err != nil {
		t.Fatalf("BuildResident (paged): %v", err)
	}
	defer r.Close()
	if r.g4moe == nil || !r.g4moe.paged {
		t.Fatalf("expected paged build with N=%d (nE=%d topK=%d)", N, nE, topK)
	}
	paged := make([][]float32, len(twoGeomPrompt))
	for i, tok := range twoGeomPrompt {
		paged[i] = append([]float32(nil), r.ForwardEmb(mg.EmbedResidentForTest(tok), i)...)
	}

	// BYTE-IDENTICAL: paged ≡ non-paged, every logit, every position (exact float equality).
	mism := 0
	for i := range paged {
		for j := range paged[i] {
			if paged[i][j] != nonPaged[i][j] {
				mism++
				if mism <= 3 {
					t.Errorf("pos %d logit %d: paged %v != non-paged %v", i, j, paged[i][j], nonPaged[i][j])
				}
			}
		}
	}

	// Pool activity: sum counters across MoE layers, report which of the three fired.
	var cold, evict, hits int
	for l := range r.layers {
		if p := r.layers[l].g4moe; p != nil && p.pool != nil {
			cold += p.pool.coldStarts
			evict += p.pool.evictions
			hits += p.pool.hits
		}
	}
	t.Logf("paged≡non-paged on gemma4-moe-tiny: N=%d/%d slots, topK=%d, %d positions — %d logit mismatches",
		N, nE, topK, len(twoGeomPrompt), mism)
	t.Logf("pool activity: coldStarts=%d evictions=%d sameExpertReuse(hits)=%d", cold, evict, hits)
	if mism != 0 {
		t.Fatalf("%d logit mismatches — paging is NOT byte-identical to the non-paged forward (a paging/staging bug)", mism)
	}
	if evict == 0 {
		t.Errorf("no evictions fired — N too large to exercise pressure; the gate proved only the easy half")
	}
	if cold == 0 {
		t.Errorf("no cold starts fired — pool never staged (paging not exercised)")
	}
	if hits == 0 {
		t.Logf("note: same-expert reuse (hits) did not fire during this forward at N=%d; it is covered "+
			"deterministically by TestExpertPool_lruAndStaging (isolation)", N)
	}
}

// runG4 loads + builds a resident with the current env and runs twoGeomPrompt, returning per-position
// logits (copied). Used for the non-paged reference arm.
func runG4(t *testing.T, ckpt string) [][]float32 {
	t.Helper()
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	out := make([][]float32, len(twoGeomPrompt))
	for i, tok := range twoGeomPrompt {
		out[i] = append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(tok), i)...)
	}
	return out
}
