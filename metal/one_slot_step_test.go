//go:build darwin

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// P29 (docs/queue-performance.md): a one-slot resident prefills its prompt on the batched step, verifies nothing on it, and
// prices the step's scratch without ever refusing the load. Tier 0: the MC3 fixture on every darwin box, and the 0.5B on the
// q4k lane where the file is present (the lane whose prefill is otherwise one token at a time).

// oneSlotStepDiff prefills embs on the resident's one slot twice from position 0: first as production decode, one token at
// a time, then on the step, through PrefillLast when viaRoute (which must pick the step route) and through prefillByStep
// otherwise. After each it decodes cont at the next position. It returns how many last-row logits, K/V elements of the
// prompt's positions, and continuation logits differ bit for bit.
func oneSlotStepDiff(t *testing.T, a *metalResident, embs [][]float32, cont []float32, viaRoute bool) (lg, kv, cg int) {
	t.Helper()
	r, n := a.r, len(embs)
	var want []float32
	for i, e := range embs {
		out, err := a.Forward(e, i)
		if err != nil {
			t.Fatal(err)
		}
		if i == n-1 {
			want = append([]float32(nil), out...)
		}
	}
	wantKV := q4kStepKV(t, r, 0, 0, n)
	wantCont, err := a.Forward(cont, n)
	if err != nil {
		t.Fatal(err)
	}
	wantCont = append([]float32(nil), wantCont...)

	var got []float32
	if viaRoute {
		floor := metalFastPrefillFloorFor(r.knobValue("GOINFER_METAL_FAST_PREFILL_FLOOR"))
		if !a.promptStepOK(n, 0, floor) {
			t.Fatalf("%d tokens: PrefillLast would not take the step route", n)
		}
		got, err = a.PrefillLast(context.Background(), embs, 0)
	} else {
		got, err = a.prefillByStep(context.Background(), embs, 0)
	}
	if err != nil {
		t.Fatalf("%d tokens on the step: %v", n, err)
	}
	got = append([]float32(nil), got...) // a one-row last piece returns Forward's own buffer, which the continuation overwrites
	gotKV := q4kStepKV(t, r, 0, 0, n)
	gotCont, err := a.Forward(cont, n)
	if err != nil {
		t.Fatal(err)
	}
	lg, kv = q4kStepDiff(got, want, gotKV, wantKV)
	cg, _ = q4kStepDiff(gotCont, wantCont, nil, nil)
	return lg, kv, cg
}

// oneSlotEmbs is n prompt embeddings and one continuation embedding from a deterministic id sequence.
func oneSlotEmbs(r *resident, seed uint32, n int) ([][]float32, []float32) {
	ids := q4kStepIDs(seed, n+1)
	embs := make([][]float32, n)
	for i := range n {
		embs[i] = mc3Emb(r, ids[i])
	}
	return embs, mc3Emb(r, ids[n])
}

// oneSlotFixture is the MC3 fixture built with one resident KV slot, which must now carry the step.
func oneSlotFixture(t *testing.T) *metalResident {
	t.Helper()
	_, r := mc3FixtureResident(t, 1, 1024)
	if len(r.kvSlotBufs) != 1 || !r.kvContig || r.batch == nil {
		t.Fatalf("one slot: %d slot buffers, contiguous %v, batched step %v (batchIneligible %q, stepDecline %q)",
			len(r.kvSlotBufs), r.kvContig, r.batch != nil, r.batchIneligible(), r.stepDecline)
	}
	return &metalResident{r: r, hidden: r.H}
}

// TestOneSlotStep_prefillBitIdentical: on one slot, a prompt prefilled on the step leaves the last row's logits and every K/V
// byte of its positions bit-identical to production decode run token by token, and the next decoded token's logits agree
// too. Prompts of 2 to 31 tokens go through PrefillLast's step route; on this lane 32 and up go to the f16 pass, so 32 to 64
// call the step itself.
func TestOneSlotStep_prefillBitIdentical(t *testing.T) {
	a := oneSlotFixture(t)
	for _, c := range []struct {
		n     int
		route bool
	}{{2, true}, {3, true}, {5, true}, {8, true}, {9, true}, {16, true}, {17, true}, {31, true}, {32, false}, {48, false}, {64, false}} {
		embs, cont := oneSlotEmbs(a.r, uint32(1000+c.n), c.n)
		lg, kv, cg := oneSlotStepDiff(t, a, embs, cont, c.route)
		if lg != 0 || kv != 0 || cg != 0 {
			t.Errorf("%d tokens (route %v): %d last-row logits, %d K/V elements, %d continuation logits differ from sequential",
				c.n, c.route, lg, kv, cg)
		}
	}
}

// TestOneSlotStep_q4kLaneBitIdentical is the same check on the q4k lane, where PrefillLast takes the step for every prompt
// of two or more tokens. It needs the Qwen2.5-Coder-0.5B q4_k_m file and skips without it.
func TestOneSlotStep_q4kLaneBitIdentical(t *testing.T) {
	a := q4kStepResident(t, 1, 1024)
	if len(a.r.kvSlotBufs) != 1 || a.VerifyCost() != nil {
		t.Fatalf("q4k one slot: %d slot buffers, verify cost %v", len(a.r.kvSlotBufs), a.VerifyCost())
	}
	for _, n := range []int{2, 3, 8, 9, 17, 31, 32, 48, 64} {
		embs, cont := oneSlotEmbs(a.r, uint32(2000+n), n)
		lg, kv, cg := oneSlotStepDiff(t, a, embs, cont, true)
		if lg != 0 || kv != 0 || cg != 0 {
			t.Errorf("q4k, %d tokens: %d last-row logits, %d K/V elements, %d continuation logits differ from sequential", n, lg, kv, cg)
		}
	}
}

// TestOneSlotStep_plantedRowOffsetGoesRed: with every step row's slot offset one row off, the identity check above fails.
func TestOneSlotStep_plantedRowOffsetGoesRed(t *testing.T) {
	a := oneSlotFixture(t)
	t.Cleanup(func() { stepSlotOffsetPlantForTest = 0 })
	embs, cont := oneSlotEmbs(a.r, 77, 9)
	stepSlotOffsetPlantForTest = 1
	lg, kv, cg := oneSlotStepDiff(t, a, embs, cont, true)
	stepSlotOffsetPlantForTest = 0
	t.Logf("planted: %d last-row logits, %d K/V elements, %d continuation logits differ", lg, kv, cg)
	if lg == 0 && kv == 0 && cg == 0 {
		t.Fatal("a row offset off by one went unseen: the identity check does not cover the step's KV addressing")
	}
}

// TestOneSlotStep_noVerifyOrBatchedDecode: one slot prefills on the step but gives speculation no verify cost and the decoder
// no batched decode; two slots keep both.
func TestOneSlotStep_noVerifyOrBatchedDecode(t *testing.T) {
	one := oneSlotFixture(t)
	var cr decoder.VerifyCostReporter = one
	if !one.stepPrefill() || cr.VerifyCost() != nil {
		t.Errorf("one slot: step prefill %v, verify cost %v; want true and nil", one.stepPrefill(), cr.VerifyCost())
	}
	if _, err := one.PrefillLastNArgmax([][]float32{mc3Emb(one.r, 1), mc3Emb(one.r, 2)}, 0); err == nil {
		t.Error("one slot: PrefillLastNArgmax ran a verify")
	}
	if lo, hi := one.BatchStepRange(); lo != 0 || hi != 0 || one.BatchGreedyDraw() {
		t.Errorf("one slot: BatchStepRange (%d, %d), BatchGreedyDraw %v; want (0, 0) and false", lo, hi, one.BatchGreedyDraw())
	}
	_, r2 := mc3FixtureResident(t, 2, 1024)
	two := &metalResident{r: r2, hidden: r2.H}
	if lo, hi := two.BatchStepRange(); two.VerifyCost() == nil || lo != batchMinSeqs || hi != 2 || !two.BatchGreedyDraw() {
		t.Errorf("two slots: verify cost %v, BatchStepRange (%d, %d), BatchGreedyDraw %v; want unchanged", two.VerifyCost() != nil, lo, hi, two.BatchGreedyDraw())
	}
}

// TestOneSlotStep_scratchPricedNotTheLoad drives the backend's own build at the memory guard's exact edge. A one-slot load
// whose need equals the ceiling is admitted with the step declined; a ceiling of need plus the step's priced scratch admits
// both. The price must cover what buildBatch really allocates (the driver's own count).
func TestOneSlotStep_scratchPricedNotTheLoad(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	load := func() *decoder.Model {
		m, err := decoder.Load(mc3FixtureWriter(t, 4096), decoder.Options{Quant: mc3TestQuant, ResidentContext: 1024, ResidentKVSlots: 1})
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	build := func(m *decoder.Model, ceiling int64) *metalResident {
		metalMemoryCeilingForTest = ceiling
		rf, ok, err := (&metalBackend{}).BuildResident(m)
		if err != nil || !ok {
			t.Fatalf("ceiling %d: the load was refused: %v", ceiling, err)
		}
		a := rf.(*metalResident)
		t.Cleanup(func() { a.r.Close() })
		return a
	}

	m := load()
	t.Cleanup(func() { m.Close() })
	need := residentNeedBytes(m)
	tight := build(m, need)
	if tight.r.batch != nil || tight.r.stepDecline == "" || tight.stepPrefill() {
		t.Fatalf("need %d at a ceiling of %d: step built %v, decline %q; want the load admitted and the step declined",
			need, need, tight.r.batch != nil, tight.r.stepDecline)
	}
	scratch := tight.r.stepScratchBytes()
	before := tight.r.d.CurrentAllocatedSize()
	tight.r.stepHeadroom = scratch
	tight.r.buildBatch()
	got := int64(tight.r.d.CurrentAllocatedSize() - before)
	t.Logf("the step's scratch: priced %d bytes, the driver counts %d", scratch, got)
	if tight.r.batch == nil || got > scratch {
		t.Errorf("with its priced headroom the step built %v, allocating %d bytes against a price of %d", tight.r.batch != nil, got, scratch)
	}

	m2 := load()
	t.Cleanup(func() { m2.Close() })
	roomy := build(m2, residentNeedBytes(m2)+scratch)
	if roomy.r.batch == nil || roomy.r.stepDecline != "" || !roomy.stepPrefill() {
		t.Errorf("a ceiling of need + scratch: step built %v, decline %q; want the step", roomy.r.batch != nil, roomy.r.stepDecline)
	}
}

// TestOneSlotStep_phi3PrefillAB is P29's tier 1 measurement (docs/queue-performance.md, P29): Phi-3 mini 4k at --quant q4k
// on one KV slot, one 512-token prompt prefilled from position 0 by production's sequential decode and by the step route,
// 5 pairs in one process with the arm order alternating. It reports each pair's sequential ÷ step wall time and their
// median; the registration's reading is that median. It loads the real file, so it runs only with GOINFER_P29_AB=1.
//
//	GOINFER_P29_AB=1 go test -count=1 -timeout 20m -run '^TestOneSlotStep_phi3PrefillAB$' -v ./metal/
func TestOneSlotStep_phi3PrefillAB(t *testing.T) {
	if os.Getenv("GOINFER_P29_AB") != "1" {
		t.Skip("set GOINFER_P29_AB=1 (a timed run on the real Phi-3 file)")
	}
	path := os.Getenv("GOINFER_PHI3_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "phi3-mini-4k-gguf", "Phi-3-mini-4k-instruct-q4.gguf")
	}
	if strings.HasPrefix(path, "/Volumes/") || strings.HasPrefix(path, "/srv/models") {
		t.Fatalf("%s is on the archive, not the bench set (CLAUDE.md)", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "q4k", ResidentContext: 2048, ResidentKVSlots: 1})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if why := residentMemoryDecline(m); why != "" { // the guard the server applies; never bypassed here
		t.Skipf("Metal's memory guard declines this load now: %s", why)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	t.Cleanup(func() { r.Close() })
	a := &metalResident{r: r, hidden: r.H}
	if !r.q4kLane || len(r.kvSlotBufs) != 1 || !a.stepPrefill() {
		t.Fatalf("not the one-slot step: %d slot buffers, step prefill %v", len(a.r.kvSlotBufs), a.stepPrefill())
	}
	const n, pairs = 512, 5
	embs, _ := oneSlotEmbs(a.r, 2929, n)
	seq := func() time.Duration {
		t0 := time.Now()
		for i, e := range embs {
			if _, err := a.Forward(e, i); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(t0)
	}
	step := func() time.Duration {
		t0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0)
	}
	seq() // warm both arms once, untimed
	step()
	var ratios []float64
	for p := range pairs {
		var s, b time.Duration
		if p%2 == 0 {
			s, b = seq(), step()
		} else {
			b, s = step(), seq()
		}
		ratios = append(ratios, s.Seconds()/b.Seconds())
		fmt.Fprintf(os.Stderr, "[p29-ab] pair %d (%s first): sequential %.2f s (%.0f tok/s), step %.2f s (%.0f tok/s), ratio %.3f\n",
			p+1, map[bool]string{true: "sequential", false: "step"}[p%2 == 0], s.Seconds(), n/s.Seconds(), b.Seconds(), n/b.Seconds(), ratios[p])
	}
	sorted := append([]float64(nil), ratios...)
	sort.Float64s(sorted)
	t.Logf("P29 tier 1: sequential ÷ step over %d pairs at %d tokens: median %.3f, pairs %v", pairs, n, sorted[pairs/2], ratios)
}
