package decoder

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// loadGemma4MoETiny loads the gitignored tiny gemma4-MoE fixture, skipping (not failing) when it
// is absent — same convention as gemma4_moe_forward_test.go/gemma4_moe_quant_test.go, and required
// here: testdata/gemma4-moe-tiny is in .gitignore on purpose (a real, if small, checkpoint), so it
// is never present in CI and a hard Load failure would redden every push, not just a local run.
func loadGemma4MoETiny(t *testing.T) *Model {
	t.Helper()
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// preM01ResidentWeightBytes is ResidentWeightBytes as it read before M-01: the dense per-layer
// matrices and the generic l.Experts/SharedExpert fields, but none of gemma4's OWN gemma4moe
// sub-block, qwen3_5_moe's delta/qattn mixer, MLA, Mamba-2, LFM2's short conv, or the model-level
// PLE tables. Kept here (not deleted) so the M-01 gate measures the actual delta the fix
// introduces, rather than an absolute floor a fixture's embed/vocab size could satisfy by
// coincidence regardless of whether the omitted fields are counted at all.
func preM01ResidentWeightBytes(w *Weights) int64 {
	n := wmBytes(&w.Embed) + wmBytes(&w.LMHead) + wmBytes(&w.PosEmbed)
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, mat := range []*linalg.WeightMat{
			&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj,
			&l.GateProj, &l.UpProj, &l.DownProj,
			&l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj,
		} {
			n += wmBytes(mat)
		}
		for j := range l.Experts {
			e := &l.Experts[j]
			n += wmBytes(&e.Gate) + wmBytes(&e.Up) + wmBytes(&e.Down)
		}
		n += wmBytes(&l.SharedExpert.Gate) + wmBytes(&l.SharedExpert.Up) + wmBytes(&l.SharedExpert.Down)
	}
	return n
}

// TestResidentWeightBytes_countsGemma4MoEExperts is M-01's break-it-first gate. Asserting an
// absolute floor (ResidentWeightBytes >= expert bytes alone) does not discriminate: this
// fixture's dense/embed bytes already exceed the expert sum by coincidence, so that assertion
// would pass even against the pre-fix code that counts the experts as zero. Asserting against
// preM01ResidentWeightBytes measures the actual delta instead — it must be positive (something
// new got counted) and must equal the gemma4moe router+experts exactly (nothing else in this
// fixture is new: hidden_size_per_layer_input is 0, so no PLE, and gemma4 has no
// delta/qattn/mla/mamba/shortConv fields to contribute).
func TestResidentWeightBytes_countsGemma4MoEExperts(t *testing.T) {
	m := loadGemma4MoETiny(t)

	var expertOnly int64
	sawExperts := false
	for i := range m.w.Layers {
		mo := m.w.Layers[i].gemma4moe
		if mo == nil {
			continue
		}
		expertOnly += wmBytes(&mo.routerProj)
		for e := range mo.expertsGateUp {
			sawExperts = true
			expertOnly += wmBytes(&mo.expertsGateUp[e]) + wmBytes(&mo.expertsDown[e])
		}
	}
	if !sawExperts {
		t.Fatalf("fixture has no gemma4moe experts to count — did enable_moe_block/num_experts change?")
	}

	got, old := m.ResidentWeightBytes(), preM01ResidentWeightBytes(m.w)
	if delta := got - old; delta != expertOnly {
		t.Errorf("ResidentWeightBytes() - preM01 delta = %d, want exactly %d (the gemma4moe "+
			"router+experts) — got %d old %d", delta, expertOnly, got, old)
	}
}

// TestResidentWeightBytesPaged_capsAtSlots is M-02's break-it-first gate for the accounting half
// of the fix: before it, the guard always used the unpaged sum, so a model that would fit under
// GOINFER_METAL_MOE_SLOTS paging (a few GB) was declined on the number it would need fully
// resident (tens of GB, per the audit's Qwen3.5-35B-A3B example). Verifies three properties, then
// cross-checks the exact paged byte count against an independently-written per-layer formula
// rather than trusting the production code's own arithmetic.
func TestResidentWeightBytesPaged_capsAtSlots(t *testing.T) {
	m := loadGemma4MoETiny(t)

	unpaged := m.ResidentWeightBytes()

	// slots<=0 and slots>=every layer's expert count must both be exactly the unpaged number.
	if got := m.ResidentWeightBytesPaged(0); got != unpaged {
		t.Errorf("ResidentWeightBytesPaged(0) = %d, want unpaged %d", got, unpaged)
	}
	maxNE := 0
	for i := range m.w.Layers {
		if mo := m.w.Layers[i].gemma4moe; mo != nil && len(mo.expertsGateUp) > maxNE {
			maxNE = len(mo.expertsGateUp)
		}
	}
	if maxNE < 2 {
		t.Fatalf("fixture has %d experts/layer — need >=2 for a meaningful slots<nE case", maxNE)
	}
	if got := m.ResidentWeightBytesPaged(maxNE); got != unpaged {
		t.Errorf("ResidentWeightBytesPaged(%d) [== max experts/layer] = %d, want unpaged %d", maxNE, got, unpaged)
	}

	// A real cap must strictly shrink the total (paging has to do something).
	slots := 1
	paged := m.ResidentWeightBytesPaged(slots)
	if paged >= unpaged {
		t.Fatalf("ResidentWeightBytesPaged(%d) = %d, want strictly < unpaged %d", slots, paged, unpaged)
	}

	// Cross-check against an INDEPENDENT per-layer formula (not the production code's own
	// pagedExperts closure) — same discipline as preM01ResidentWeightBytes above: the two must
	// agree by construction, not by re-reading the same arithmetic.
	want := unpaged
	for i := range m.w.Layers {
		mo := m.w.Layers[i].gemma4moe
		if mo == nil {
			continue
		}
		nE := len(mo.expertsGateUp)
		if nE == 0 || slots >= nE {
			continue
		}
		var layerFull int64
		for e := range mo.expertsGateUp {
			layerFull += wmBytes(&mo.expertsGateUp[e]) + wmBytes(&mo.expertsDown[e])
		}
		want -= layerFull - layerFull/int64(nE)*int64(slots)
	}
	if paged != want {
		t.Errorf("ResidentWeightBytesPaged(%d) = %d, want %d (independent per-layer recomputation)", slots, paged, want)
	}
}
