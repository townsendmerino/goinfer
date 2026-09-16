package decoder

import (
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/giw"
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

// TestResidentHostCopyBytes_exemptsPagedExperts is M-02's gate for the host-copy addend (2026-09-
// 09): a unified-memory backend (Metal) holds a quantized HOST WeightMat AND a separately-packed
// device buffer for the SAME dense weights, but a genuinely PAGED expert streams from disk and
// never gets a committed host copy — so the addend must shrink under paging exactly where
// ResidentWeightBytesPaged's OWN estimate does, not stay flat.
func TestResidentHostCopyBytes_exemptsPagedExperts(t *testing.T) {
	m := loadGemma4MoETiny(t)

	unpaged := m.ResidentWeightBytes()
	hostUnpaged := m.ResidentHostCopyBytes(0)
	if hostUnpaged != unpaged {
		t.Errorf("ResidentHostCopyBytes(0) = %d, want exactly ResidentWeightBytes() %d — unpaged, "+
			"every expert is a materialized host copy same as dense", hostUnpaged, unpaged)
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

	slots := 1
	hostPaged := m.ResidentHostCopyBytes(slots)
	if hostPaged >= hostUnpaged {
		t.Fatalf("ResidentHostCopyBytes(%d) = %d, want strictly < unpaged %d — paged experts must "+
			"be exempt from the host-copy addend (they stream, no host materialization)", slots, hostPaged, hostUnpaged)
	}

	// The paged host-copy figure must equal the DENSE-only sum: the unpaged total with EVERY
	// expert class's FULL bytes removed — both the generic l.Experts field (empty on this gemma4
	// fixture, which uses the fused representation instead, but summed anyway so this assertion
	// does not silently assume that) and gemma4moe's fused expertsGateUp/expertsDown.
	var allExpertBytes int64
	for i := range m.w.Layers {
		for j := range m.w.Layers[i].Experts {
			e := &m.w.Layers[i].Experts[j]
			allExpertBytes += wmBytes(&e.Gate) + wmBytes(&e.Up) + wmBytes(&e.Down)
		}
		if mo := m.w.Layers[i].gemma4moe; mo != nil {
			for e := range mo.expertsGateUp {
				allExpertBytes += wmBytes(&mo.expertsGateUp[e]) + wmBytes(&mo.expertsDown[e])
			}
		}
	}
	wantDense := unpaged - allExpertBytes
	if hostPaged != wantDense {
		t.Errorf("ResidentHostCopyBytes(%d) = %d, want %d (dense-only: unpaged minus every expert class)", slots, hostPaged, wantDense)
	}

	// ResidentDenseWeightBytes must equal the SAME dense-only figure computed above, independent
	// of paging — it exists precisely so a caller checking "does the fixed part fit" gets the same
	// answer ResidentHostCopyBytes derives internally when paging is active.
	if got := m.ResidentDenseWeightBytes(); got != wantDense {
		t.Errorf("ResidentDenseWeightBytes() = %d, want %d", got, wantDense)
	}
}

// buildTinyGIWInt4 loads dir at Quant:"int4" and re-serializes it as a weights-only .giw in a
// temp dir (SerializeWeightsToForTarget + giw.WriteStream, the same pattern
// TestGIWRoundTrip_preservesMergedEOSIDs and decoder/moepaging_test.go's buildTinyGIW use) —
// int4, not int8: MmapByteOffset itself doesn't care about kind, but int4 is llama-tiny's own
// resident-eligible quant (matches the audit's own "14B int4 .giw" shape) and lets this test
// share fixture setup with nothing else that would make the two diverge.
func buildTinyGIWInt4(t *testing.T, dir string) string {
	t.Helper()
	m, err := Load(dir, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s, int4): %v", dir, err)
	}
	out := filepath.Join(t.TempDir(), "tiny-int4.giw")
	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create %s: %v", out, err)
	}
	werr := giw.WriteStream(f, nil, func(w io.Writer) (int64, error) {
		return SerializeWeightsToForTarget(w, m.Weights(), "weightbytes-fixture", GIWTargetNone)
	})
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	m.Close()
	if werr != nil {
		t.Fatalf("write %s: %v", out, werr)
	}
	return out
}

// TestResidentHostCopyBytes_exemptsMmapAliasedGIWWeights is M-24's gate (docs/audit-2026-09-10.md):
// a .giw-loaded model's int8/int4 payloads are mmap-ALIASED (LoadSerializedWeights' own doc
// comment — "Big int8/int4 arrays are aliased into data (zero-copy)"), not a heap allocation this
// model's own quantizer produced, so they must NOT count as a second "host copy" alongside the
// separately-estimated device buffer — unlike a heap-loaded (GGUF/safetensors) model, where the
// doubling is real. Both loaded from the SAME testdata/llama-tiny fixture, at the same quant, so
// any difference between them is exactly the mmap-aliasing fix, not a shape difference.
func TestResidentHostCopyBytes_exemptsMmapAliasedGIWWeights(t *testing.T) {
	const dir = "../testdata/llama-tiny"

	heap, err := Load(dir, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(heap): %v", err)
	}
	defer heap.Close()
	heapDevice := heap.ResidentWeightBytesPaged(0)
	heapHost := heap.ResidentHostCopyBytes(0)
	if heapHost != heapDevice {
		t.Fatalf("heap-loaded: ResidentHostCopyBytes(0) = %d, want exactly ResidentWeightBytesPaged(0) "+
			"%d — a genuinely heap-allocated quantized copy must still double (control case)", heapHost, heapDevice)
	}

	giwPath := buildTinyGIWInt4(t, dir)
	mg, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("Load(.giw): %v", err)
	}
	defer mg.Close()
	giwDevice := mg.ResidentWeightBytesPaged(0)
	giwHost := mg.ResidentHostCopyBytes(0)

	// NOT asserted: giwDevice == heapDevice. The .giw round trip can choose a different int4
	// packing kind (canonical vs row4/split-half) than the original heap load, which changes the
	// reported byte size independent of anything this test checks — that's a serialization-format
	// question, not what M-24 is about. What must hold regardless of which kind was chosen: an
	// mmap-backed load's host-copy addend must be strictly smaller than its own device estimate,
	// where a heap-backed load's is exactly equal (asserted above).
	if giwHost >= giwDevice {
		t.Errorf(".giw-loaded: ResidentHostCopyBytes(0) = %d, want strictly < ResidentWeightBytesPaged(0) "+
			"%d — mmap-aliased int4 payloads are still being counted as a second host copy (M-24)", giwHost, giwDevice)
	}
	t.Logf("heap: device=%d host=%d (doubles, correctly) | .giw: device=%d host=%d (mmap-aliased "+
		"portion excluded, %.1f%% of the heap case)", heapDevice, heapHost, giwDevice, giwHost,
		100*float64(giwHost)/float64(heapHost))
}
