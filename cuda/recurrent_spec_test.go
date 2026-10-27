//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// SP2's tier-0 gates (docs/tasks/task-forward-architecture-2026-10.md §6a, "SP2 on nobara"): G1a, a verify leaves the committed
// recurrent state as it found it and a fold of n rows leaves what a prefill of those rows leaves, byte for byte; G1b, the
// BlockSpec loop with a scripted drafter emits plain greedy's tokens and ends on plain greedy's state; G5, a speculative turn
// commits what kvmgr then scores. Each runs clean and under the two registered plants, which must fail it.
//
// The tiny fixtures take the generic scan twin (head dim 16); qwen3.5-0.8b (GOINFER_HEAVY_TESTS=1, or GOINFER_QWEN35_08B) takes
// the 128-wide one the released models run.

type specModel struct {
	name, path string
	heavy      bool
}

func specModels() []specModel {
	ms := []specModel{
		{"qwen35-tiny", filepath.Join("..", "testdata", "qwen35-tiny"), false},
		{"qwen3next-tiny", filepath.Join("..", "testdata", "qwen3next-tiny"), false},
	}
	p := os.Getenv("GOINFER_QWEN35_08B")
	if p == "" {
		p = os.ExpandEnv("$HOME/models/qwen3.5-0.8b")
	}
	return append(ms, specModel{"qwen3.5-0.8b", p, true})
}

// loadSpecModel loads m on CUDA with the fold path's arena allocated, or skips when its weights are absent.
func loadSpecModel(t *testing.T, m specModel) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	if m.heavy && os.Getenv("GOINFER_HEAVY_TESTS") == "" && os.Getenv("GOINFER_QWEN35_08B") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 for the 0.8B")
	}
	if !hasWeights(m.path) {
		t.Skipf("no fixture at %s", m.path)
	}
	mc, err := decoder.Load(m.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { mc.Close() })
	r := mustResident(t, mc)
	if err := r.PrepareRecurrentSpec(decoder.RecurrentSpecMaxRows); err != nil {
		t.Fatalf("PrepareRecurrentSpec: %v", err)
	}
	if r.RecurrentSpecRows() != decoder.RecurrentSpecMaxRows {
		t.Fatalf("no arena: RecurrentSpecRows() = %d", r.RecurrentSpecRows())
	}
	return mc, r
}

// withSpecPlant runs fn with a fold-path plant armed, and disarms it however fn ends (t.Fatalf included), so no later test runs
// planted.
func withSpecPlant(plant int, fn func()) {
	dnSpecPlantForTest = plant
	defer func() { dnSpecPlantForTest = 0 }()
	fn()
}

// withGuardTripped runs fn with every BlockSpec acceptance guard stopping at its window, restored however fn ends.
func withGuardTripped(fn func()) {
	defer decoder.TripBlockSpecGuardForTest()()
	fn()
}

func hasWeights(dir string) bool {
	for _, f := range []string{"model.safetensors", "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err == nil {
			return true
		}
	}
	return false
}

// dnSnapshot downloads every DeltaNet layer's conv window and state, in layer order.
func dnSnapshot(t *testing.T, r *cudaResident) [][]float32 {
	t.Helper()
	var out [][]float32
	err := r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			if !Ly.isDeltaNet() {
				continue
			}
			for _, b := range []Buffer{Ly.dnWin, Ly.dnState} {
				h := make([]float32, b.Len())
				if e := gpu.Download(b, h); e != nil {
					return e
				}
				out = append(out, h)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	return out
}

// dnRestore uploads a snapshot back into the live window and state.
func dnRestore(t *testing.T, r *cudaResident, snap [][]float32) {
	t.Helper()
	err := r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		i := 0
		for l := range r.layers {
			Ly := &r.layers[l]
			if !Ly.isDeltaNet() {
				continue
			}
			for _, b := range []Buffer{Ly.dnWin, Ly.dnState} {
				if e := gpu.Upload(b, snap[i]); e != nil {
					return e
				}
				i++
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
}

// snapDiff names the first buffer whose bits differ between a and b ("" when every bit agrees). Buffers alternate window, state.
func snapDiff(a, b [][]float32) string {
	for i := range a {
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				kind := "conv window"
				if i%2 == 1 {
					kind = "state"
				}
				return fmt.Sprintf("DeltaNet layer #%d %s, element %d: %g vs %g", i/2, kind, j, a[i][j], b[i][j])
			}
		}
	}
	return ""
}

func idsFor(n, seed, vocab int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = (i*7919 + seed*104729 + 101) % (vocab - 1)
	}
	return out
}

func embsOf(mc *decoder.Model, ids []int) [][]float32 {
	e := make([][]float32, len(ids))
	for i, id := range ids {
		e[i] = mc.EmbedResidentForTest(id)
	}
	return e
}

// verifyFoldFailures runs G1a's cases on one model and returns every failure: verify leaves the state; the n rows FoldVerified marks
// fold to what a prefill of them leaves, both at FlushFold and inside the next verify's own scan; each verified row's argmax equals
// the prefill's, including the next verify's rows, which scan from the folded state.
func verifyFoldFailures(t *testing.T, mc *decoder.Model, r *cudaResident) []string {
	t.Helper()
	_, _, _, _, _, _, vocab := mc.Dims()
	prompt := idsFor(20, 1, vocab)
	if _, err := r.PrefillLastNArgmax(embsOf(mc, prompt), 0); err != nil {
		t.Fatalf("prompt prefill: %v", err)
	}
	base := dnSnapshot(t, r)
	pos := len(prompt)
	next := idsFor(4, 99, vocab) // the next verify's rows
	var fails []string
	for _, w := range []int{2, 4, 8, 16} {
		rows := idsFor(w, w, vocab)
		for n := 1; n <= w; n++ {
			// The reference: a prefill of the n committed rows, then of those rows and the next verify's.
			dnRestore(t, r, base)
			want, err := r.PrefillLastNArgmax(embsOf(mc, rows[:n]), pos)
			if err != nil {
				t.Fatalf("w=%d n=%d: reference prefill: %v", w, n, err)
			}
			wantState := dnSnapshot(t, r)
			dnRestore(t, r, base)
			wantNext, err := r.PrefillLastNArgmax(embsOf(mc, append(append([]int(nil), rows[:n]...), next...)), pos)
			if err != nil {
				t.Fatalf("w=%d n=%d: reference prefill with the next rows: %v", w, n, err)
			}
			for _, fused := range []bool{false, true} {
				how := "at FlushFold"
				if fused {
					how = "in the next verify"
				}
				dnRestore(t, r, base)
				got, err := r.VerifyLastNArgmax(embsOf(mc, rows), pos)
				if err != nil {
					t.Fatalf("w=%d: verify: %v", w, err)
				}
				if d := snapDiff(base, dnSnapshot(t, r)); d != "" {
					fails = append(fails, fmt.Sprintf("w=%d n=%d: the verify wrote the committed state: %s", w, n, d))
				}
				for i := range n {
					if got[i] != want[i] {
						fails = append(fails, fmt.Sprintf("w=%d n=%d: row %d argmax %d, prefill %d", w, n, i, got[i], want[i]))
						break
					}
				}
				if err := r.FoldVerified(n); err != nil {
					t.Fatalf("w=%d n=%d: fold: %v", w, n, err)
				}
				if fused {
					gotNext, err := r.VerifyLastNArgmax(embsOf(mc, next), pos+n)
					if err != nil {
						t.Fatalf("w=%d n=%d: next verify: %v", w, n, err)
					}
					for i := range next {
						if gotNext[i] != wantNext[n+i] {
							fails = append(fails, fmt.Sprintf("w=%d n=%d: next verify row %d argmax %d, prefill %d", w, n, i, gotNext[i], wantNext[n+i]))
							break
						}
					}
				} else if err := r.FlushFold(); err != nil {
					t.Fatalf("w=%d n=%d: flush: %v", w, n, err)
				}
				if d := snapDiff(wantState, dnSnapshot(t, r)); d != "" {
					fails = append(fails, fmt.Sprintf("w=%d n=%d: the fold %s differs from a prefill of the same rows: %s", w, n, how, d))
				}
			}
		}
	}
	return fails
}

// TestRecurrentSpec_verifyFoldMatchesPrefill is G1a, with P1 and P2 registered to fail it.
func TestRecurrentSpec_verifyFoldMatchesPrefill(t *testing.T) {
	for _, m := range specModels() {
		t.Run(m.name, func(t *testing.T) {
			mc, r := loadSpecModel(t, m)
			if fails := verifyFoldFailures(t, mc, r); len(fails) > 0 {
				t.Fatalf("%d failures; first: %s", len(fails), fails[0])
			}
			for _, p := range []struct {
				name  string
				plant int
			}{{"P1-fold-skips-conv", dnSpecPlantFoldSkipsConv}, {"P2-verify-writes-state", dnSpecPlantVerifyWritesState}} {
				var fails []string
				withSpecPlant(p.plant, func() { fails = verifyFoldFailures(t, mc, r) })
				if len(fails) == 0 {
					t.Errorf("plant %s passed G1a: the gate cannot see it", p.name)
				} else {
					t.Logf("plant %s caught [expected]: %d failures; first: %s", p.name, len(fails), fails[0])
				}
			}
		})
	}
}

// scriptedDrafter proposes the target's own greedy continuation (oracle, the tokens after prompt) with one draft corrupted per
// round on a fixed schedule: round k corrupts draft k mod 2w, so over 2w rounds every accepted count 0..w-1 occurs, and half the
// rounds accept every draft.
type scriptedDrafter struct {
	oracle            []int
	promptLen, vocab  int
	width, ctx, round int
}

func (d *scriptedDrafter) FuseContext(rows [][]float32) ([][]float32, error) { return rows, nil }
func (d *scriptedDrafter) ExtendContext(f [][]float32) error                 { d.ctx += len(f); return nil }
func (d *scriptedDrafter) ContextLen() int                                   { return d.ctx }
func (d *scriptedDrafter) TruncateContext(n int) {
	if n < d.ctx {
		d.ctx = n
	}
}
func (d *scriptedDrafter) DraftBlock(in [][]float32) ([][]float32, error) { return in, nil }
func (d *scriptedDrafter) DraftTokens(trunk [][]float32) ([]int, error) {
	out := make([]int, len(trunk))
	bad := d.round % (2 * d.width)
	d.round++
	for i := range out {
		if k := d.ctx - d.promptLen + 1 + i; k < len(d.oracle) {
			out[i] = d.oracle[k]
		}
		if i == bad {
			out[i] = (out[i] + 1) % d.vocab
		}
	}
	return out, nil
}

// scriptedWeights stands for the drafter's weights: the loop reads only BlockSize and MaskTokenID.
type scriptedWeights struct{ decoder.BlockDrafterWeights }

func (scriptedWeights) BlockSize() int   { return decoder.RecurrentSpecMaxRows }
func (scriptedWeights) MaskTokenID() int { return 0 }

// plainGreedy is Model.Generate at temperature 0 under the exact-attention scope BlockSpec holds.
func plainGreedy(t *testing.T, mc *decoder.Model, r *cudaResident, prompt []int, maxNew int) ([]int, *decoder.Generation) {
	t.Helper()
	leave := r.EnterExactAttention()
	defer leave()
	ch, gen := mc.Generate(context.Background(), prompt, maxNew, decoder.SamplingParams{})
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	if e := gen.Err(); e != nil {
		t.Fatalf("plain greedy: %v", e)
	}
	return out, gen
}

// scriptedLoopFailures runs G1b's cases on one model: per width, the speculative tokens equal plain greedy's and the state after
// the turn is plain's, bit for bit.
func scriptedLoopFailures(t *testing.T, mc *decoder.Model, r *cudaResident, maxNew int) []string {
	t.Helper()
	_, _, _, _, _, _, vocab := mc.Dims()
	var fails []string
	for pi, prompt := range [][]int{idsFor(12, 7, vocab), idsFor(33, 8, vocab)} {
		want, _ := plainGreedy(t, mc, r, prompt, maxNew)
		plainState := dnSnapshot(t, r)
		for _, w := range []int{2, 4, 8, 16} {
			d := &scriptedDrafter{oracle: want, promptLen: len(prompt), vocab: vocab, width: w}
			s, err := mc.NewBlockSpecForTest(d, scriptedWeights{}, []int{0})
			if err != nil {
				t.Fatalf("NewBlockSpecForTest: %v", err)
			}
			got, rounds, err := s.Generate(prompt, decoder.BlockSpecOptions{VerifyWidth: w, MaxTokens: maxNew})
			if err != nil {
				t.Fatalf("prompt %d w=%d: Generate: %v", pi, w, err)
			}
			if len(got) != len(want) {
				fails = append(fails, fmt.Sprintf("prompt %d w=%d: %d tokens, plain %d", pi, w, len(got), len(want)))
				continue
			}
			bad := -1
			for i := range want {
				if got[i] != want[i] {
					bad = i
					break
				}
			}
			if bad >= 0 {
				fails = append(fails, fmt.Sprintf("prompt %d w=%d: token %d is %d, plain %d (%d rounds)", pi, w, bad, got[bad], want[bad], rounds))
				continue
			}
			if dd := snapDiff(plainState, dnSnapshot(t, r)); dd != "" {
				fails = append(fails, fmt.Sprintf("prompt %d w=%d: tokens equal, state after the turn differs: %s", pi, w, dd))
			}
		}
	}
	return fails
}

// TestRecurrentSpec_scriptedLoop is G1b: the real BlockSpec loop with a scripted drafter, with P1 and P2 registered to fail it,
// and once with the acceptance guard tripped so the turn ends on plain steps after folded rounds.
func TestRecurrentSpec_scriptedLoop(t *testing.T) {
	for _, m := range specModels() {
		t.Run(m.name, func(t *testing.T) {
			mc, r := loadSpecModel(t, m)
			maxNew := 40
			if m.heavy {
				maxNew = 128
			}
			if fails := scriptedLoopFailures(t, mc, r, maxNew); len(fails) > 0 {
				t.Fatalf("%d failures; first: %s", len(fails), fails[0])
			}
			var fails []string
			withGuardTripped(func() { fails = scriptedLoopFailures(t, mc, r, maxNew) })
			if len(fails) > 0 {
				t.Fatalf("guard tripped: %d failures; first: %s", len(fails), fails[0])
			}
			for _, p := range []struct {
				name  string
				plant int
			}{{"P1-fold-skips-conv", dnSpecPlantFoldSkipsConv}, {"P2-verify-writes-state", dnSpecPlantVerifyWritesState}} {
				var fails []string
				withSpecPlant(p.plant, func() { fails = scriptedLoopFailures(t, mc, r, maxNew) })
				if len(fails) == 0 {
					t.Errorf("plant %s passed G1b: the gate cannot see it", p.name)
				} else {
					t.Logf("plant %s caught [expected]: %d failures; first: %s", p.name, len(fails), fails[0])
				}
			}
		})
	}
}

// TestRecurrentSpec_commitAgreesWithKVManager is G5 on the small models: after a speculative turn A, (1) the live state is a cold
// prefill's of A's prompt and output; (2) a turn B extending A reuses all of A and decodes what a cold B decodes, plainly and
// speculatively; (3) the recurrent checkpoint's buffers are untouched by A's rounds.
func TestRecurrentSpec_commitAgreesWithKVManager(t *testing.T) {
	for _, m := range specModels() {
		t.Run(m.name, func(t *testing.T) {
			mc, r := loadSpecModel(t, m)
			_, _, _, _, _, _, vocab := mc.Dims()
			maxNew := 24
			if m.heavy {
				maxNew = 96
			}
			promptA := idsFor(14, 21, vocab)
			oracleA, _ := plainGreedy(t, mc, r, promptA, maxNew)
			ckptBefore := ckptSnapshot(t, r)

			d := &scriptedDrafter{oracle: oracleA, promptLen: len(promptA), vocab: vocab, width: 4}
			s, err := mc.NewBlockSpecForTest(d, scriptedWeights{}, []int{0})
			if err != nil {
				t.Fatalf("NewBlockSpecForTest: %v", err)
			}
			outA, _, err := s.Generate(promptA, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxNew})
			if err != nil {
				t.Fatalf("speculative A: %v", err)
			}
			afterA := dnSnapshot(t, r)
			if dd := snapDiff(ckptBefore, ckptSnapshot(t, r)); dd != "" {
				t.Errorf("KV8a: the speculative turn wrote the recurrent checkpoint: %s", dd)
			}

			// (2) plain B on the warm slot, then cold B, then speculative B on the warm slot.
			promptB := append(append(append([]int(nil), promptA...), outA...), idsFor(5, 22, vocab)...)
			warmB, gen := plainGreedy(t, mc, r, promptB, maxNew)
			if gen.PrefillReused != len(promptA)+len(outA) {
				t.Errorf("kvmgr reused %d of B's prompt, want A's prompt plus output, %d", gen.PrefillReused, len(promptA)+len(outA))
			}

			// (1) the state A left against a cold prefill of A's prompt and output.
			if _, err := r.PrefillLastNArgmax(embsOf(mc, append(append([]int(nil), promptA...), outA...)), 0); err != nil {
				t.Fatalf("cold prefill of A: %v", err)
			}
			if dd := snapDiff(dnSnapshot(t, r), afterA); dd != "" {
				t.Errorf("the state after speculative A is not a cold prefill's of A's prompt and output: %s", dd)
			}
			mc.ForgetResidentForTest()
			coldB, _ := plainGreedy(t, mc, r, promptB, maxNew)
			if fmt.Sprint(warmB) != fmt.Sprint(coldB) {
				t.Errorf("plain B on A's slot differs from cold B:\n  warm %v\n  cold %v", warmB, coldB)
			}

			// Speculative A again, then speculative B on its slot (BlockSpec's own reuse path).
			mc.ForgetResidentForTest()
			d = &scriptedDrafter{oracle: oracleA, promptLen: len(promptA), vocab: vocab, width: 4}
			if s, err = mc.NewBlockSpecForTest(d, scriptedWeights{}, []int{0}); err != nil {
				t.Fatal(err)
			}
			if _, _, err := s.Generate(promptA, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxNew}); err != nil {
				t.Fatalf("speculative A again: %v", err)
			}
			d.oracle, d.promptLen = coldB, len(promptB)
			specB, _, err := s.Generate(promptB, decoder.BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxNew})
			if err != nil {
				t.Fatalf("speculative B: %v", err)
			}
			if fmt.Sprint(specB) != fmt.Sprint(coldB) {
				t.Errorf("speculative B on A's slot differs from cold B:\n  spec %v\n  cold %v", specB, coldB)
			}
		})
	}
}

// ckptSnapshot downloads the recurrent checkpoint's buffers (nil when the load kept none).
func ckptSnapshot(t *testing.T, r *cudaResident) [][]float32 {
	t.Helper()
	var out [][]float32
	err := r.do(func() error {
		if e := r.stream.Sync(); e != nil {
			return e
		}
		for _, c := range r.ckptSave {
			h := make([]float32, c.Bytes/4)
			if e := gpu.Download(c.Dst, h); e != nil {
				return e
			}
			out = append(out, h)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("checkpoint snapshot: %v", err)
	}
	return out
}

// TestRecurrentSpec_arenaPriced is G3's tier-0 half: the fit guard's companion reserve grows, for a drafter on a DeltaNet model, by
// exactly the arena the resident then allocates for RecurrentSpecMaxRows rows, rounded up to 2 MiB as cuMemAlloc rounds it.
// The price is recomputed here from the resident's own geometry, not read from the formula it checks.
func TestRecurrentSpec_arenaPriced(t *testing.T) {
	requireCUDADevice(t)
	for _, m := range specModels() {
		t.Run(m.name, func(t *testing.T) {
			if m.heavy && os.Getenv("GOINFER_HEAVY_TESTS") == "" && os.Getenv("GOINFER_QWEN35_08B") == "" {
				t.Skip("set GOINFER_HEAVY_TESTS=1 for the 0.8B")
			}
			if !hasWeights(m.path) {
				t.Skipf("no fixture at %s", m.path)
			}
			load := func(drafter bool) (*decoder.Model, *cudaResident) {
				mc, err := decoder.Load(m.path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096, KVPrecision: "f32", Drafter: drafter})
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				return mc, mustResident(t, mc)
			}
			plain, _ := load(false)
			base := plain.ExtraResidentBytes()
			plain.Close()
			mc, r := load(true)
			defer mc.Close()
			priced := mc.ExtraResidentBytes() - base
			dp := r.dnet
			nDN := 0
			for l := range r.layers {
				if r.layers[l].isDeltaNet() {
					nDN++
				}
			}
			rows := decoder.RecurrentSpecMaxRows
			raw := int64(2*nDN*rows*(2*dp.convDim+2*dp.keyDim+2*dp.nv)+rows*dp.valueDim) * 4 // two banks
			want := (raw + (2<<20 - 1)) / (2 << 20) * (2 << 20)
			if priced != want {
				t.Errorf("the reserve grew by %d with a drafter, want %d (%d raw bytes, %d DeltaNet layers)", priced, want, raw, nDN)
			}
			if r.extraBytes < priced {
				t.Errorf("the resident's reserve %d does not hold the arena's %d", r.extraBytes, priced)
			}
			// A narrower arena first: asking for more rows replaces it.
			if err := r.PrepareRecurrentSpec(4); err != nil || r.RecurrentSpecRows() != 4 {
				t.Fatalf("PrepareRecurrentSpec(4): %v, %d rows", err, r.RecurrentSpecRows())
			}
			if err := r.PrepareRecurrentSpec(rows); err != nil || r.RecurrentSpecRows() != rows {
				t.Fatalf("PrepareRecurrentSpec(%d) after 4: %v, %d rows", rows, err, r.RecurrentSpecRows())
			}
			if got := r.RecurrentSpecBytes(); got != raw {
				t.Errorf("allocated %d bytes, priced %d raw", got, raw)
			}
		})
	}
}

// TestRecurrentSpec_seedChunksKeepTheCapture: a block-spec seed longer than one pass is prefilled in PrefillLast's passes with the
// capture armed, and the capture holds every row of the prompt, bit-equal to one pass's (the exact kernels a recurrent model's
// prefill takes are chunk-invariant), with the same seed token.
func TestRecurrentSpec_seedChunksKeepTheCapture(t *testing.T) {
	mc, r := loadSpecModel(t, specModels()[1]) // qwen3next-tiny
	_, _, _, _, _, _, vocab := mc.Dims()
	prompt := embsOf(mc, idsFor(40, 3, vocab))
	taps := []int{0, 2}
	if err := r.SetBatchedCapture(taps); err != nil {
		t.Fatal(err)
	}
	defer r.SetBatchedCapture(nil)
	oneTok, err := r.PrefillSeedArgmax(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	var one [][]float32 // a copy: the next call's passes write the resident's slice in place
	for _, rows := range r.BatchedCapture() {
		one = append(one, append([]float32(nil), rows...))
	}
	r.prefillChunkCap.Store(8) // passes of 8 rows from here on
	defer r.prefillChunkCap.Store(0)
	tok, err := r.PrefillSeedArgmax(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := r.BatchedCapture()
	if tok != oneTok {
		t.Errorf("seed token %d in passes of 8, %d in one pass", tok, oneTok)
	}
	for i := range taps {
		if len(got[i]) != len(one[i]) {
			t.Fatalf("tap %d: %d floats captured in passes of 8, %d in one pass", taps[i], len(got[i]), len(one[i]))
		}
		for j := range one[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(one[i][j]) {
				t.Fatalf("tap %d element %d: %g in passes of 8, %g in one pass", taps[i], j, got[i][j], one[i][j])
			}
		}
	}
}

// TestRecurrentSpec_earlyExitLeavesNothingToFold: a speculative turn stopped by its consumer right after a round's FoldVerified, then
// a second speculative turn on the same resident, must emit plain greedy's tokens and end on plain greedy's state: the stopped turn's
// marked rows must never fold into the new sequence.
func TestRecurrentSpec_earlyExitLeavesNothingToFold(t *testing.T) {
	for _, m := range specModels() {
		t.Run(m.name, func(t *testing.T) {
			mc, r := loadSpecModel(t, m)
			_, _, _, _, _, _, vocab := mc.Dims()
			const maxNew = 24
			prompt := idsFor(14, 41, vocab)
			want, _ := plainGreedy(t, mc, r, prompt, maxNew)
			wantState := dnSnapshot(t, r)
			for _, w := range []int{4, 8} {
				d := &scriptedDrafter{oracle: want, promptLen: len(prompt), vocab: vocab, width: w}
				s, err := mc.NewBlockSpecForTest(d, scriptedWeights{}, []int{0})
				if err != nil {
					t.Fatal(err)
				}
				// The seed's token is emit 1, the first round's burst emit 2: the consumer stops there, after FoldVerified.
				if _, err := s.GenerateStopAfterForTest(prompt, decoder.BlockSpecOptions{VerifyWidth: w, MaxTokens: maxNew}, 2); err != nil {
					t.Fatalf("w=%d: stopped turn: %v", w, err)
				}
				d.ctx, d.round = 0, 0
				got, _, err := s.Generate(prompt, decoder.BlockSpecOptions{VerifyWidth: w, MaxTokens: maxNew})
				if err != nil {
					t.Fatalf("w=%d: second turn: %v", w, err)
				}
				if at := sp2Diverge(got, want); at >= 0 {
					t.Errorf("w=%d: after a stopped turn, the next turn diverges from plain greedy at token %d", w, at)
				}
				if dd := snapDiff(wantState, dnSnapshot(t, r)); dd != "" {
					t.Errorf("w=%d: after a stopped turn, the next turn's state differs from plain greedy's: %s", w, dd)
				}
			}
		})
	}
}
