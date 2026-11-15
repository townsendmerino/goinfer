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
	// providerFakeResident embeds only the plain stepPlainResident, so the outer struct is not itself a subsystem value:
	// callers must go through ResidentSubsystemProvider.
	if _, ok := any(prov).(ResidentDecodeSubsystem); ok {
		t.Fatal("providerFakeResident must not implement ResidentDecodeSubsystem directly")
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
	fakeSubsystems
	ResidentForward
	curve []float64
}

func (f uncomparableValueResident) ResidentSubsystems() ResidentSubsystems { return selfSubsystems(f) }

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

// bannedMicroAssertions names the legacy resident capability micro-interfaces that callers no longer type-assert; they
// reach a capability through ResidentSubsystems instead. The legacy interfaces themselves stay declared.
var bannedMicroAssertions = map[string]bool{
	"legacyKVSlotter":             true,
	"legacyBatchStepper":          true,
	"legacyRaggedStepper":         true,
	"legacyMRoPE":                 true,
	"legacyGreedy":                true,
	"legacyGreedyChain":           true,
	"legacySampleChain":           true,
	"legacyTopK":                  true,
	"legacySample":                true,
	"legacyPrefillKV":             true,
	"Prefiller":                   true,
	"legacyHiddenLast":            true,
	"legacyResidualAll":           true,
	"legacyAdapter":               true,
	"legacyImagePrefill":          true,
	"legacyImageBlocksPrefill":    true,
	"legacyExpertCache":           true,
	"legacySpanPrefill":           true,
	"ResidentHybridMRoPEPrefill":  true,
	"legacyMRoPEPrefill":          true,
	"legacyMRoPEDeepstackPrefill": true,
	"PrefillPathReporter":         true,
	"PrefillTailExact":            true,
	"VerifyPathReporter":          true,
	"legacyCapsReporter":          true,
	"legacyPrefillBlocks":         true,
	"legacyKVCopier":              true,
	"legacyKVSlotActivity":        true,
	"legacyKVPoolInfo":            true,
	"legacyReusableKVSlot":        true,
	"legacyKVUploadPass":          true,
	"legacyKVDemoter":             true,
	"legacyReusableKV":            true,
	"legacyKVSnapshotter":         true,
	"legacyPrefillMarked":         true,
	"legacyPrefillChunks":         true,
	"legacyPrefillSpanScratch":    true,
	"legacyPrefillRows":           true,
	"legacyRecurrentCheckpoint":   true,
	"ResidentDrafterHost":         true,
	"legacySeedArgmax":            true,
	"legacyRecurrentSpec":         true,
	"StreamingDrafterHost":        true,
	"legacyCost":                  true,
	"VerifyCostReporter":          true,
	"DecodeVerifyDiverger":        true,
	"ExactAttentionScoper":        true,
}

// TestResidentSubsystems_NoScatteredMicroAssertions enforces the subsystem boundary across package decoder: no production
// file outside resident_subsystems.go may type-assert any of the 42 optional resident capability micro-interfaces.
func TestResidentSubsystems_NoScatteredMicroAssertions(t *testing.T) {
	banned := bannedMicroAssertions

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

// backendTypeAssertAllowed lists the backend type assertions on a legacy micro-interface that are kept on purpose, and
// why. The list may only shrink. Each entry is a TYPE-level check on a nil pointer to the backend's resident
// ("does this resident type implement the hybrid image prefill, as the support table declares"). The subsystem
// contracts cannot say that: ResidentPrefillSubsystem always has the method, and answering it through the contract
// needs a built resident (cudaResident.HybridMRoPEPrefill reads mropePrefillReady), which a table test has not got.
var backendTypeAssertAllowed = map[string]string{
	"backend/cuda/support_table_test.go:ResidentHybridMRoPEPrefill":   "type-level support-table check on (*cudaResident)(nil)",
	"backend/metal/support_table_test.go:ResidentHybridMRoPEPrefill":  "type-level support-table check on (*metalResident)(nil)",
	"backend/webgpu/support_table_test.go:ResidentHybridMRoPEPrefill": "type-level support-table check on (*residentDecoder)(nil)",
}

// TestResidentSubsystems_NoScatteredMicroAssertionsInBackends extends the boundary to the backend modules: no file under
// backend/cuda, backend/metal or backend/webgpu, production or test, may type-assert a legacy micro-interface as
// .(decoder.X). The scan is syntactic, so build tags do not hide a file from it, and it reads the modules from the
// workspace on disk, which the backend modules' own go.mod files need not be able to build.
func TestResidentSubsystems_NoScatteredMicroAssertionsInBackends(t *testing.T) {
	fset := token.NewFileSet()
	used := map[string]bool{}
	for _, mod := range []string{"cuda", "metal", "webgpu"} {
		root := filepath.Join("..", "backend", mod)
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !strings.HasSuffix(path, ".go") {
				return nil
			}
			f, err := parser.ParseFile(fset, path, nil, 0)
			if err != nil {
				return err
			}
			rel := strings.TrimPrefix(filepath.ToSlash(path), "../") // slashes first: on Windows the walked path is "..\\backend\\..."
			ast.Inspect(f, func(n ast.Node) bool {
				ta, ok := n.(*ast.TypeAssertExpr)
				if !ok || ta.Type == nil {
					return true
				}
				sel, ok := ta.Type.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				if pkg, ok := sel.X.(*ast.Ident); !ok || pkg.Name != "decoder" || !bannedMicroAssertions[sel.Sel.Name] {
					return true
				}
				key := rel + ":" + sel.Sel.Name
				if _, ok := backendTypeAssertAllowed[key]; ok {
					used[key] = true
					return true
				}
				t.Errorf("%s:%d: type assertion .(decoder.%s) on a legacy micro-interface; use decoder.SubsystemsOf(r) and its subsystem contract",
					rel, fset.Position(ta.Pos()).Line, sel.Sel.Name)
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for key := range backendTypeAssertAllowed {
		if !used[key] {
			t.Errorf("backendTypeAssertAllowed has %s, but no such assertion exists: delete the entry", key)
		}
	}
}
