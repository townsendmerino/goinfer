package decoder

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// NO FAMILY LOOP OUTSIDE THE PROGRAM.
//
// Every family's layers are a program the walkers run (OP3, docs/tasks/task-forward-architecture-2026-10.md §4.8). A family
// layer loop written beside it, a runLayersXxx dispatched by an `if`, is a second list of families: "runLayers dispatches here"
// and "the batched path must not touch this" were once two hand-written facts and LFM2 was in one of them. The source scan
// notices such a loop being born; the second half checks that every registered model type builds a program whose every
// operation the per-token walker implements.
func TestProgram_noFamilyLoopAndEveryFamilyWalks(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(files)
	re := regexp.MustCompile(`(?m)^func \(m \*Model\) (runLayers[A-Z][A-Za-z0-9]*)\(id int, cache \*KVCache\) \(\[\]float32, error\) \{`)
	scanned := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		scanned++
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			t.Errorf("%s (%s) is a family layer loop outside the program: whatever dispatches to it is a second list of families", m[1], f)
		}
	}
	if scanned == 0 {
		t.Fatal("the scan read no source file — the check is broken, not the tree")
	}

	walked := map[opKind]bool{opNorm: true, opProjQKV: true, opQKNorm: true, opRope: true, opAttnTemp: true, opAttention: true,
		opOutGate: true, opProjO: true, opFFN: true, opResidual: true, opMLA: true, opKDA: true, opShortConv: true, opMamba2: true,
		opDeltaNet: true, opGatedAttn: true, opGemma4Attn: true, opGemma4MoE: true, opPLE: true, opLayerScale: true}
	for _, mt := range modelTypes(t) {
		arch, _, err := resolveArchitecture(representativeConfig(mt))
		if err != nil {
			t.Fatalf("%s: %v", mt, err)
		}
		p := buildProgram(arch, nil)
		if len(p.layers) != arch.NumLayers {
			t.Errorf("%s: a program of %d layers for %d", mt, len(p.layers), arch.NumLayers)
		}
		for l, ops := range p.layers {
			if len(ops) == 0 {
				t.Errorf("%s: layer %d has no operations", mt, l)
			}
			for _, o := range ops {
				if !walked[o.kind] {
					t.Errorf("%s: layer %d has %v, which the per-token walker does not implement", mt, l, o.kind)
				}
			}
		}
	}
}

// C-01, END TO END, ON THE COMMITTED FIXTURE. A prompt of >=2 tokens reached the generic batched stack, whose
// first act on an LFM2 conv layer is rmsNorm against a QNorm that was never loaded: an index-out-of-range in the
// Generate goroutine, where net/http's recover does not reach. TestLFM2_textParity drives m.forward one token at a
// time and never calls prefillLogits, so it cannot see it.
//
// The assertion is "the two paths agree", not "it does not panic": prefillLogits is documented as bit-identical
// to the sequential prefill, and a fix that stopped the crash on a different path would satisfy the weaker claim.
func TestLFM2_multiTokenPrefillMatchesSequential(t *testing.T) {
	const ckpt = "../testdata/lfm2-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Fatalf("the committed LFM2 fixture is missing at %s: %v", ckpt, err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	if m.w.arch.lfm2 == nil {
		t.Fatalf("fixture resolved to arch %q, not lfm2", m.w.arch.Name)
	}
	if m.canBatchN(4) {
		t.Error("canBatchN(4) = true for LFM2 — the batched stack has no conv mixer")
	}
	if batched, reason := m.PrefillPath(); batched {
		t.Errorf("PrefillPath() reports batched (%q); serve prints this at startup, so a wrong "+
			"answer here is published as a fact", reason)
	}

	prompt := []int{3, 11, 47, 5}

	seqCache := m.NewCache(len(prompt) + 1)
	var want []float32
	for _, id := range prompt {
		if want, err = m.forward(id, seqCache); err != nil {
			t.Fatalf("sequential forward: %v", err)
		}
	}

	batchCache := m.NewCache(len(prompt) + 1)
	got, err := m.prefillLogits(context.Background(), prompt, batchCache)
	if err != nil {
		t.Fatalf("prefillLogits: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("prefillLogits returned %d logits, sequential %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("prefill logits diverge from sequential at %d: %v vs %v (prefillLogits "+
				"documents itself as bit-identical)", i, got[i], want[i])
		}
	}
	if batchCache.pos != seqCache.pos {
		t.Errorf("cache position %d after prefill, %d after sequential — the whole prompt must be "+
			"in the cache either way", batchCache.pos, seqCache.pos)
	}
}

// Generate is the reported trigger: `serve --model <lfm2>`, first chat request. It runs the
// prefill in its own goroutine, so the panic took the process rather than the request.
func TestLFM2_generateMultiTokenPrompt(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	ch, gen := m.Generate(context.Background(), []int{3, 11, 47, 5}, 3, SamplingParams{})
	n := 0
	for range ch {
		n++
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if n == 0 {
		t.Fatal("Generate produced no tokens")
	}
}

// THE DENSE LAYER PAGER STAYS OFF A FAMILY WITH A LAYER OUTSIDE THE GENERIC OPERATIONS (audit N-13): its spans cover the generic
// layer's weights, not a mixer's, so its RAM-bound banner would promise what it cannot deliver. Over every registered model type,
// so a family is covered the day it is registered.
func TestProgram_pagerStaysOffNonGenericFamilies(t *testing.T) {
	declined := 0
	for _, mt := range modelTypes(t) {
		arch, _, err := resolveArchitecture(representativeConfig(mt))
		if err != nil {
			t.Fatalf("%s: %v", mt, err)
		}
		if arch.genericKindsOnly() || arch.MoE != nil {
			continue
		}
		if p := newLayerPager(&Weights{arch: arch}, []byte{0}, 1<<20); p != nil {
			t.Errorf("%s: newLayerPager built a pager for a family with a layer outside the generic operations", mt)
		}
		declined++
	}
	if declined == 0 {
		t.Fatal("no dense family with a non-generic layer was checked: the test would test nothing")
	}
}

// The two views of "recurrent" must not disagree: the table's Recurrent bit decides before a cache exists
// (speculative rollback), KVCache.hasRecurrentState decides once one does (truncate, snapshot, session
// reconcile). A family recurrent in one sense and not the other leaks state across sessions (audit C-02/C-03).
//
// TestOwnForward_recurrentBitMatchesTheCacheKinds is DERIVED, not listed: both predicates could omit the same
// state (KDA) and agree, so the cache is asked directly. Every registered family's REAL cache is built by NewCache
// from its representativeConfig and its recurrent kinds are found by reflection (recurrentKinds); the architecture's
// rule (Architecture.recurrent: the own-loop table's bit, or the program's operations), both hasRecurrentState views
// and speculative rollback must all agree with that.
func TestOwnForward_recurrentBitMatchesTheCacheKinds(t *testing.T) {
	for _, mt := range modelTypes(t) {
		t.Run(mt, func(t *testing.T) {
			m, c := realCacheForType(t, mt)
			names, _ := recurrentKinds(c)
			allocated := len(names) > 0
			if m.w.arch.recurrent() != allocated {
				t.Errorf("Architecture.recurrent()=%v, but NewCache allocates recurrent kinds %v", m.w.arch.recurrent(), names)
			}
			if got := c.hasRecurrentState(); got != allocated {
				t.Errorf("KVCache.hasRecurrentState()=%v, but the cache holds recurrent kinds %v — every "+
					"consumer keyed on it (TruncateTo, Session.reconcile, Snapshot) is blind to them", got, names)
			}
			if got := m.hasRecurrentState(); got != allocated {
				t.Errorf("Model.hasRecurrentState()=%v, but the cache holds recurrent kinds %v", got, names)
			}
			if allocated && m.specRollbackSafe() {
				t.Errorf("specRollbackSafe()=true for a family with recurrent kinds %v: a rejected speculative "+
					"suffix is 'rolled back' by a TruncateTo that cannot rewind them", names)
			}
		})
	}
}
