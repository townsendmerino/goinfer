package decoder

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type providerFakeResident struct {
	stepPlainResident
	argmaxID  int
	slots     int
	caps      ResidentCaps
	tailExact bool
}

func (p *providerFakeResident) ResidentSubsystems() ResidentSubsystems {
	return ResidentSubsystems{
		Decode:  providerFakeDecode{p: p},
		Prefill: providerFakePrefill{p: p},
		KV:      providerFakeKV{p: p},
		Runtime: providerFakeRuntime{p: p},
	}
}

type providerFakeDecode struct {
	NoopResidentDecode
	p *providerFakeResident
}

func (s providerFakeDecode) GreedyAvailable() bool { return true }
func (s providerFakeDecode) ForwardArgmax([]float32, int) (int, error) {
	return s.p.argmaxID, nil
}

type providerFakePrefill struct {
	NoopResidentPrefill
	p *providerFakeResident
}

func (s providerFakePrefill) PrefillTailExact() bool { return s.p.tailExact }

type providerFakeKV struct {
	NoopResidentKV
	p *providerFakeResident
}

func (s providerFakeKV) KVSlotterAvailable() bool { return true }
func (s providerFakeKV) KVSlots() int             { return s.p.slots }

type providerFakeRuntime struct {
	NoopResidentRuntime
	p *providerFakeResident
}

func (s providerFakeRuntime) ResidentCaps() ResidentCaps { return s.p.caps }

func TestResidentSubsystems_NilAndPlainDefaults(t *testing.T) {
	for _, r := range []any{nil, &stepPlainResident{}} {
		sub := SubsystemsOf(r)
		if sub.Decode.GreedyAvailable() || sub.Decode.TopKAvailable() || sub.Decode.SampleAvailable() || sub.Decode.BatchStepperAvailable() {
			t.Fatalf("unexpected decode capability on %T", r)
		}
		if sub.Prefill.PrefillAvailable() || sub.Prefill.PrefillTailExact() || sub.Prefill.HybridMRoPEPrefill() {
			t.Fatalf("unexpected prefill capability on %T", r)
		}
		if sub.KV.KVSlotterAvailable() || sub.KV.KVSlots() != 1 || sub.KV.ReusableKV(64) != 64 || sub.KV.ReusableKVSlot(0, 64) != 64 {
			t.Fatalf("unexpected KV defaults on %T", r)
		}
		if sub.Runtime.ResidentCaps().ContextCap != 0 || sub.Runtime.DrafterHostAvailable() || sub.Runtime.DecodeVerifyDivergence() != nil {
			t.Fatalf("unexpected runtime defaults on %T", r)
		}
		sub.Runtime.EnterExactAttention()()
	}
}

func TestResidentSubsystems_DirectProviderAndZeroAlloc(t *testing.T) {
	prov := &providerFakeResident{
		argmaxID:  42,
		slots:     4,
		caps:      ResidentCaps{ContextCap: 8192, Quant: "int4"},
		tailExact: true,
	}
	// Note: providerFakeResident intentionally does NOT implement ResidentGreedy, ResidentKVSlotter, ResidentCapsReporter,
	// or PrefillTailExact on the outer struct — proving callers go through ResidentSubsystemProvider rather than legacy assertions.
	if _, ok := any(prov).(ResidentGreedy); ok {
		t.Fatal("providerFakeResident must not implement ResidentGreedy directly")
	}

	m := &Model{resident: prov}
	if got := m.ResidentContextCap(); got != 8192 {
		t.Fatalf("ResidentContextCap() = %d, want 8192", got)
	}
	if got := m.residentSlotCount(); got != 4 {
		t.Fatalf("residentSlotCount() = %d, want 4", got)
	}
	if !m.resPrefill().PrefillTailExact() {
		t.Fatal("PrefillTailExact() = false, want true")
	}

	emb := make([]float32, 8)
	var row [1][]float32
	row[0] = emb
	res, err := m.step(stepBatch{Pass: passToken, Emb: row[:], Pos: 1, Out: outArgmax})
	if err != nil || res.ID != 42 {
		t.Fatalf("m.step(outArgmax) = (%d, %v), want (42, nil)", res.ID, err)
	}

	if allocs := testing.AllocsPerRun(100, func() {
		row[0] = emb
		if _, err := m.step(stepBatch{Pass: passToken, Emb: row[:], Pos: 1, Out: outArgmax}); err != nil {
			t.Fatal(err)
		}
		_ = m.ResidentContextCap()
		_ = m.resKV().KVSlots()
	}); allocs != 0 {
		t.Fatalf("ResidentSubsystemProvider hot path allocated %.1f times/run, want 0", allocs)
	}

	// Swapping m.resident between legacy and provider doubles must immediately reflect the new resident.
	m.resident = &stepPlainResident{}
	if m.canDecode(outArgmax) {
		t.Fatal("swapped plain resident should not offer outArgmax")
	}
	m.resident = &stepNoAllocResident{}
	if !m.canDecode(outArgmax) {
		t.Fatal("swapped stepNoAllocResident should offer outArgmax")
	}
}

type uncomparableValueResident struct {
	ResidentForward
	curve []float64
}

func (u uncomparableValueResident) VerifyCost() []float64 { return u.curve }

func TestResidentSubsystems_UncomparableValueResidentDoesNotPanic(t *testing.T) {
	u := uncomparableValueResident{
		ResidentForward: &stepPlainResident{},
		curve:           []float64{0, 1, 1.8},
	}
	m := &Model{resident: u}
	for range 3 {
		got := m.resRuntime().VerifyCost()
		if len(got) != 3 || got[2] != 1.8 {
			t.Fatalf("VerifyCost() = %v, want [0 1 1.8]", got)
		}
	}
}

// TestResidentSubsystems_NoScatteredMicroAssertions enforces the subsystem boundary across package decoder: no production
// file outside resident_subsystems.go may type-assert any of the 42 optional resident capability micro-interfaces.
func TestResidentSubsystems_NoScatteredMicroAssertions(t *testing.T) {
	banned := map[string]bool{
		"ResidentKVSlotter":             true,
		"ResidentBatchStepper":          true,
		"ResidentRaggedStepper":         true,
		"ResidentMRoPE":                 true,
		"ResidentGreedy":                true,
		"ResidentGreedyChain":           true,
		"ResidentSampleChain":           true,
		"ResidentTopK":                  true,
		"ResidentSample":                true,
		"ResidentPrefillKV":             true,
		"Prefiller":                     true,
		"ResidentHiddenLast":            true,
		"ResidentResidualAll":           true,
		"ResidentAdapter":               true,
		"ResidentImagePrefill":          true,
		"ResidentImageBlocksPrefill":    true,
		"ResidentExpertCache":           true,
		"ResidentSpanPrefill":           true,
		"ResidentHybridMRoPEPrefill":    true,
		"ResidentMRoPEPrefill":          true,
		"ResidentMRoPEDeepstackPrefill": true,
		"PrefillPathReporter":           true,
		"PrefillTailExact":              true,
		"VerifyPathReporter":            true,
		"ResidentCapsReporter":          true,
		"ResidentPrefillBlocks":         true,
		"ResidentKVCopier":              true,
		"ResidentKVSlotActivity":        true,
		"ResidentKVPoolInfo":            true,
		"ResidentReusableKVSlot":        true,
		"ResidentKVUploadPass":          true,
		"ResidentKVDemoter":             true,
		"ResidentReusableKV":            true,
		"ResidentKVSnapshotter":         true,
		"ResidentPrefillMarked":         true,
		"ResidentPrefillChunks":         true,
		"ResidentPrefillSpanScratch":    true,
		"ResidentPrefillRows":           true,
		"ResidentRecurrentCheckpoint":   true,
		"ResidentDrafterHost":           true,
		"ResidentSeedArgmax":            true,
		"ResidentRecurrentSpec":         true,
		"StreamingDrafterHost":          true,
		"ResidentCost":                  true,
		"VerifyCostReporter":            true,
		"DecodeVerifyDiverger":          true,
		"ExactAttentionScoper":          true,
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	for _, ent := range entries {
		name := ent.Name()
		if ent.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || name == "resident_subsystems.go" {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Clean(name), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			ta, ok := n.(*ast.TypeAssertExpr)
			if !ok || ta.Type == nil {
				return true
			}
			if id, ok := ta.Type.(*ast.Ident); ok && banned[id.Name] {
				pos := fset.Position(ta.Pos())
				t.Errorf("%s:%d: ad-hoc micro-interface type assertion .(%s) outside resident_subsystems.go; use ResidentSubsystems instead",
					pos.Filename, pos.Line, id.Name)
			}
			return true
		})
	}
}
