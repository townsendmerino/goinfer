package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

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
	m, err := Load("../testdata/gemma4-moe-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

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
