//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/mem"
)

// OP2b on CUDA, step 2c (docs/tasks/task-forward-architecture-2026-10.md §4.7): the sites that tested r.dnet read "recurrent", so a Mamba-2 resident is
// declined where a Gated-DeltaNet one is; and the KV plan prices attention layers only, plus the Mamba state.

type guardCase struct {
	fixture string
	quant   string
}

func loadGuardResident(t *testing.T, c guardCase) (*decoder.Model, *cudaResident) {
	t.Helper()
	m, err := decoder.Load(filepath.Join("..", "testdata", c.fixture), decoder.Options{Backend: "cuda", Quant: c.quant, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load %s: %v", c.fixture, err)
	}
	t.Cleanup(func() { m.Close() })
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s not resident (%s): %s", c.fixture, m.DecodePath(), m.ResidentDecline())
	}
	return m, r
}

// withTwoSlots gives r a second slot entry with empty tables, so the checks that run only on a multi-slot resident (the batched step, the copy) get
// past the slot count to the guard under test. Nothing is copied or run through it.
func withTwoSlots(r *cudaResident) {
	if r.KVSlots() < 2 {
		var t kvSlotTabs
		for g := range kvGroups {
			t.ids[g] = make([]int32, r.kvMaxBlocks())
		}
		r.kvTabs = append(r.kvTabs, t)
	}
}

// TestNemotronResidentCUDA_guards is OP2b's gate G on CUDA: a Nemotron-H resident is declined by each site that refuses recurrent state (the batched
// step, the prefix snapshot, the KV copy, the verify pass, and the batched prefill by name), and a Gated-DeltaNet one is declined by the same sites as before
// except the prefix snapshot, which carries its state (B35, ResidentRecurrentSnapshots) and admits it.
// llama-tiny is the control, admitted where the recurrent check decides, so a guard that declines everything cannot pass. The batched-prefill decline is
// switched off for the batched-step check (batchStepStaticDeclineOffForTest), because it also refuses a Mamba-2 model and would hide a guard left DeltaNet-only.
// Not covered: the fast-lever switch in prefillCore (prefill.go), which a recurrent model never reaches because the prefill declines it first.
func TestNemotronResidentCUDA_guards(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	batchStepStaticDeclineOffForTest = true
	defer func() { batchStepStaticDeclineOffForTest = false }()
	type result struct {
		batchHi            int
		copyDecl, snapDecl string
		copyErr            error
		verifyBatched      bool
	}
	probe := func(c guardCase) (*cudaResident, result) {
		_, r := loadGuardResident(t, c)
		withTwoSlots(r)
		var res result
		_, res.batchHi = r.BatchStepRange()
		res.copyErr = r.CopyKV(0, 1, 0, decoder.KVBlockRows)
		var d *decoder.KVCopyDecline
		if errors.As(res.copyErr, &d) {
			res.copyDecl = d.Reason
		}
		if d := r.snapshotDecline(decoder.KVBlockRows); d != nil {
			res.snapDecl = d.Reason
		}
		res.verifyBatched, _ = r.VerifyPath()
		return r, res
	}
	_, ctrl := probe(guardCase{fixture: "llama-tiny", quant: "int4"})
	// The control has written nothing, so past the recurrent case the copy's only decline is the rows check.
	if ctrl.batchHi < 2 || !strings.Contains(ctrl.copyDecl, "no longer holds") || ctrl.snapDecl != "" || !ctrl.verifyBatched {
		t.Fatalf("VOID: the control llama-tiny is not admitted where the recurrent check decides (batch hi %d, copy decline %q (err %v), snapshot decline %q, batched verify %v)",
			ctrl.batchHi, ctrl.copyDecl, ctrl.copyErr, ctrl.snapDecl, ctrl.verifyBatched)
	}
	for _, c := range []guardCase{{fixture: "nemotron-tiny", quant: "int4"}, {fixture: "qwen35-tiny", quant: "int4"}} {
		t.Run(c.fixture, func(t *testing.T) {
			r, res := probe(c)
			if res.batchHi != 0 {
				t.Errorf("the batched step admits %s (range up to %d)", c.fixture, res.batchHi)
			}
			if !strings.Contains(res.copyDecl, "recurrent") {
				t.Errorf("CopyKV on %s: decline %q (err %v), want the recurrent-state decline", c.fixture, res.copyDecl, res.copyErr)
			}
			switch c.fixture {
			case "qwen35-tiny": // B35: a Gated-DeltaNet snapshot carries the recurrent state, so it is admitted
				if res.snapDecl != "" {
					t.Errorf("the prefix snapshot of %s: decline %q, want it admitted (B35)", c.fixture, res.snapDecl)
				}
			default:
				if !strings.Contains(res.snapDecl, "Mamba-2") {
					t.Errorf("the prefix snapshot of %s: decline %q, want the Mamba-2 state decline", c.fixture, res.snapDecl)
				}
			}
			if res.verifyBatched {
				t.Errorf("ForwardN's verify pass on %s reports a batched path", c.fixture)
			}
			static := r.prefillStaticDecline()
			if c.fixture == "nemotron-tiny" && (static == nil || !strings.Contains(static.Error(), "Mamba-2 mixer")) {
				t.Errorf("the batched prefill on nemotron-tiny: %v, want a decline naming the Mamba-2 mixer", static)
			}
			if c.fixture == "nemotron-tiny" {
				if batched, why := r.PrefillPath(); batched || !strings.Contains(why, "Mamba-2 mixer has no batched kernel") {
					t.Errorf("PrefillPath on nemotron-tiny = (%v, %q), want a sequential path that names the Mamba-2 mixer and not a projection precision", batched, why)
				}
			}
			if c.fixture == "qwen35-tiny" && static != nil {
				t.Errorf("the batched prefill declines qwen35-tiny (%v), which runs the DeltaNet recurrence batched", static)
			}
		})
	}
}

// TestNemotronResidentCUDA_pricing is OP2b's gate P on CUDA: the KV a Nemotron-H resident allocates and the plan prices is the attention layers' alone (a
// Mamba-2 block and a feed-forward-only block hold none), the build's price and the decoder's agree, and the plan prices the Mamba state the build allocates.
func TestNemotronResidentCUDA_pricing(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	m, r := loadGuardResident(t, guardCase{fixture: "nemotron-tiny", quant: "int4"})
	const ctx = 512
	elem := int64(4)
	if r.kvF16 {
		elem = 2
	}
	var want, attn int64
	var nMamba int
	var stateBytes int64
	for l := range r.layers {
		switch m.NemotronBlockKind(l) { // the family's own kinds, not the resident's
		case 1:
			attn++
			want += 2 * int64(m.KVHeadsAtResident(l)*m.HeadDimAtResident(l)) * elem * ctx
			if r.kc[l].Len() == 0 || r.vc[l].Len() == 0 {
				t.Errorf("attention layer %d has no KV buffer", l)
			}
		default:
			if r.kc[l].Len() != 0 || r.vc[l].Len() != 0 {
				t.Errorf("layer %d (block kind %d) holds a KV buffer of %d elements", l, m.NemotronBlockKind(l), r.kc[l].Len())
			}
		}
		if m.NemotronBlockKind(l) == 0 {
			nMamba++
			stateBytes += int64(r.layers[l].mbWin.Len()+r.layers[l].mbSSM.Len()) * 4
		}
	}
	if attn == 0 || attn == int64(len(r.layers)) || nMamba == 0 {
		t.Fatalf("VOID: nemotron-tiny has %d attention and %d Mamba layers of %d: the reading cannot tell the kinds apart", attn, nMamba, len(r.layers))
	}
	// A slot is its attention layers' blocks (ctx is whole blocks here) and its two tables; a one-slot load adds the null blocks (kvpool.go).
	tabs := int64(kvGroups * 4 * kvBlocksFor(ctx))
	if got := kvBytesForCap(ctx, r.layers); got != want+tabs {
		t.Errorf("kvBytesForCap(%d) = %d, the %d attention layers alone are %d and the tables %d", ctx, got, attn, want, tabs)
	}
	if got, w := m.ResidentKVBytes("cuda", ctx, r.kvF16, false), want+tabs+kvPoolNullBytes(r.layers); got != w {
		t.Errorf("the decoder's ResidentKVBytes(cuda, %d) = %d, want %d", ctx, got, w)
	}
	if got := m.Plan("cuda", 1<<40, decoder.PlanRequest{Ctx: ctx}).StateBytes; got != stateBytes {
		t.Errorf("the plan's state line is %d, the %d Mamba layers' state buffers are %d bytes", got, nMamba, stateBytes)
	}
	t.Logf("nemotron-tiny: %d attention layers of %d price %d KV bytes at %d positions; %d Mamba layers' state is %d bytes", attn, len(r.layers), want, ctx, nMamba, stateBytes)
}

// TestRecurrentStatePlanned_matchesTheBuild is MM8's CUDA gate (docs/tasks/task-subsystems-2026-10.md, MM8): the recurrent state the
// pre-build context plan prices (each layer's conv window and state, at the driver's allocation quantum) equals what the build charges
// for it, on Gated-DeltaNet, Mamba-2 and Olmo Hybrid fixtures; and it is all on the device, on the ledger's state line, before checkKVFits reads
// free VRAM.
func TestRecurrentStatePlanned_matchesTheBuild(t *testing.T) {
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	var planned, atKVFit int64
	recurrentStatePlannedForTest = func(p int64) { planned = p }
	recurrentStateAtKVFitForTest = func(c int64) { atKVFit = c }
	t.Cleanup(func() { recurrentStatePlannedForTest, recurrentStateAtKVFitForTest = nil, nil })
	for _, c := range []guardCase{{"qwen35-tiny", "int4"}, {"nemotron-tiny", "int4"}, {"olmo_hybrid-tiny", "int4"}} {
		planned, atKVFit = -1, -1
		_, r := loadGuardResident(t, c)
		charged := r.recurrentStateCharged()
		t.Logf("%s: the plan priced %d bytes of recurrent state, the build charges %d over %d recurrent layers, the ledger's state line holds %d at the KV fit",
			c.fixture, planned, charged, r.recurrentLayers(), atKVFit)
		if r.recurrentLayers() == 0 || charged <= 0 {
			t.Errorf("%s: %d recurrent layers charging %d bytes; the case reads nothing", c.fixture, r.recurrentLayers(), charged)
		}
		if planned != charged {
			t.Errorf("%s: the context plan priced %d bytes of recurrent state, the build charges %d", c.fixture, planned, charged)
		}
		// The state is on the device before the KV fit reads free VRAM, so that probe needs no price for it (postPoolBytes is for what
		// the build allocates after the pools).
		if atKVFit != charged {
			t.Errorf("%s: the ledger's state line holds %d bytes at the KV fit, the build charges %d for the recurrent state", c.fixture, atKVFit, charged)
		}
	}
}

// recurrentSlotFixtures are the recurrent models the state-slot gates run on: Gated-DeltaNet, Gated-DeltaNet with its own
// settings, and Mamba-2 (docs/tasks/task-kv-architecture-2026-10.md, KV8's plan of 2026-11-03).
var recurrentSlotFixtures = []string{"qwen35-tiny", "olmo_hybrid-tiny", "nemotron-tiny"}

func recurrentSlotsLoad(t *testing.T, fixture string, o decoder.Options) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	defer nemotronFeaturesForTest()()
	o.Backend, o.Quant = "cuda", "int4"
	path := fixture
	if !strings.HasPrefix(path, "..") {
		path = "../../testdata/" + fixture
	}
	m, err := decoder.Load(path, o)
	if err != nil {
		t.Fatalf("load %s: %v", fixture, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, mustResident(t, m)
}

// recurrentSlotSeqs is two different token sequences of n tokens, as embeddings.
func recurrentSlotSeqs(m *decoder.Model, n int) (a, b [][]float32) {
	_, _, _, _, _, _, vocab := m.Dims()
	for i := range n {
		a = append(a, slices.Clone(m.EmbedResidentForTest((i*31+7)%vocab)))
		b = append(b, slices.Clone(m.EmbedResidentForTest((i*17+101)%vocab)))
	}
	return a, b
}

// recurrentSlotSolo decodes embs as one fresh sequence on the bound slot and returns each position's logits.
func recurrentSlotSolo(t *testing.T, r *cudaResident, embs [][]float32) [][]float32 {
	t.Helper()
	r.Reset()
	out := make([][]float32, len(embs))
	for i, e := range embs {
		l, err := r.Forward(e, i)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = slices.Clone(l)
	}
	return out
}

func recurrentSlotBind(t *testing.T, r *cudaResident, s int) {
	t.Helper()
	if err := r.UseKVSlot(s); err != nil {
		t.Fatal(err)
	}
}

func logitsDiffer(got, want []float32) int {
	n := 0
	for j := range want {
		if j >= len(got) || math.Float32bits(got[j]) != math.Float32bits(want[j]) {
			n++
		}
	}
	return n
}

// recurrentSlotInterleaved decodes A on slot 0 and B on slot 1 one token each in turn, and counts the logits that differ from refA
// and refB bit for bit.
func recurrentSlotInterleaved(t *testing.T, r *cudaResident, A, B, refA, refB [][]float32) int {
	t.Helper()
	recurrentSlotBind(t, r, 0)
	r.Reset()
	recurrentSlotBind(t, r, 1)
	r.Reset()
	differ := 0
	for i := range A {
		for s, c := range []struct{ emb, want [][]float32 }{{A, refA}, {B, refB}} {
			recurrentSlotBind(t, r, s)
			l, err := r.Forward(c.emb[i], i)
			if err != nil {
				t.Fatal(err)
			}
			differ += logitsDiffer(l, c.want[i])
		}
	}
	return differ
}

// TestRecurrentSlots_sequencesKeepTheirOwnState is KV8's S-1: on a recurrent model that asked for two KV slots, the resident has two,
// and two sequences decoded one token each in turn each equal the same sequence decoded alone on a one-slot resident, every logit
// of every position.
func TestRecurrentSlots_sequencesKeepTheirOwnState(t *testing.T) {
	for _, fixture := range recurrentSlotFixtures {
		t.Run(fixture, func(t *testing.T) {
			_, two := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			if n := two.KVSlots(); n != 2 {
				t.Fatalf("%d KV slots on a recurrent model that asked for 2", n)
			}
			mo, one := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 1})
			A, B := recurrentSlotSeqs(mo, 12)
			refA, refB := recurrentSlotSolo(t, one, A), recurrentSlotSolo(t, one, B)
			if d := recurrentSlotInterleaved(t, two, A, B, refA, refB); d != 0 {
				t.Errorf("interleaved on two slots: %d logits differ from each sequence's solo run", d)
			}
		})
	}
}

// recurrentSlotPlantShared binds slot 1 to slot 0's window and state, as one shared state would be.
func recurrentSlotPlantShared(r *cudaResident) {
	for l := range r.layers {
		if s := r.layers[l].recSlots; len(s) > 1 {
			s[1] = s[0]
		}
	}
}

// TestRecurrentSlots_sharedStateGoesRed is S-1's plant, past the first red: the slots exist, and slot 1 is bound to slot 0's pair.
func TestRecurrentSlots_sharedStateGoesRed(t *testing.T) {
	for _, fixture := range recurrentSlotFixtures {
		t.Run(fixture, func(t *testing.T) {
			_, two := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			recurrentSlotPlantShared(two)
			mo, one := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 1})
			A, B := recurrentSlotSeqs(mo, 12)
			refA, refB := recurrentSlotSolo(t, one, A), recurrentSlotSolo(t, one, B)
			d := recurrentSlotInterleaved(t, two, A, B, refA, refB)
			t.Logf("planted: %d logits differ", d)
			if d == 0 {
				t.Error("with one state shared by both slots, the sequences still equal their solo runs: the check does not see a shared state")
			}
		})
	}
}

// TestRecurrentSlots_expertCacheKeepsOne: expert streaming takes what VRAM is left, so an extra slot is expert slots lost, and a recurrent model keeps one slot under
// it though the caller asked for more. The same model without it gets what it asked for.
func TestRecurrentSlots_expertCacheKeepsOne(t *testing.T) {
	requireCUDADevice(t)
	const fixture = "../../decoder/testdata/qwen3_5_moe-tiny"
	if _, err := os.Stat(filepath.Join(fixture, "model.safetensors")); err != nil {
		t.Skipf("no weights at %s (gitignored; scripts/worktree_fixtures.sh places them)", fixture)
	}
	_, c := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2, MoECacheExperts: true})
	if !c.cacheExperts {
		t.Fatal("the load built no expert cache: this test reads nothing")
	}
	if n := c.KVSlots(); n != 1 {
		t.Errorf("a recurrent model with expert streaming that asked for 2 has %d KV slots; want 1", n)
	}
	_, u := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
	if n := u.KVSlots(); n != 2 {
		t.Errorf("the same model without expert streaming has %d KV slots; want 2", n)
	}
}

// TestRecurrentSlots_keepsOneSlotAtTheDefaultRequest: a recurrent model whose slot count is only the caller's default keeps one slot, so a default load keeps its bytes.
func TestRecurrentSlots_keepsOneSlotAtTheDefaultRequest(t *testing.T) {
	for _, fixture := range recurrentSlotFixtures {
		_, r := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 4, ResidentKVSlotsDefault: true})
		if n := r.KVSlots(); n != 1 {
			t.Errorf("%s at the caller's default request has %d KV slots; want 1", fixture, n)
		}
		for l := range r.layers {
			if len(r.layers[l].recSlots) != 0 {
				t.Errorf("%s at one slot: layer %d holds %d state pairs; want none past the upload's", fixture, l, len(r.layers[l].recSlots))
			}
		}
	}
}

// recurrentSlotDirty reads how many of slot s's window and state elements are non-zero.
func recurrentSlotDirty(t *testing.T, r *cudaResident, s int) int {
	t.Helper()
	n := 0
	for l := range r.layers {
		if !r.layers[l].isRecurrent() {
			continue
		}
		p := r.layers[l].recurrentOf(s)
		for _, b := range []Buffer{p.win, p.state} {
			if b.Len() == 0 {
				continue
			}
			got := make([]float32, b.Len())
			if err := r.do(func() error { return gpu.Download(b, got) }); err != nil {
				t.Fatal(err)
			}
			for _, v := range got {
				if v != 0 {
					n++
				}
			}
		}
	}
	return n
}

// recurrentSlotReset is S-2 on r: after each sequence dirties its slot, a reset with slot 1 bound zeroes slot 1 and changes nothing of
// slot 0, whose sequence then continues as if alone. It returns what it found wrong. duringReset, if not nil, is a plant to hold through that reset.
func recurrentSlotReset(t *testing.T, r *cudaResident, A, B, refA [][]float32, duringReset *bool) (wrong []string) {
	t.Helper()
	half := len(A) / 2
	recurrentSlotBind(t, r, 0)
	r.Reset()
	for i := range half {
		if _, err := r.Forward(A[i], i); err != nil {
			t.Fatal(err)
		}
	}
	recurrentSlotBind(t, r, 1)
	for i, emb := range B { // the slot is fresh, so no reset comes before the one under test
		if _, err := r.Forward(emb, i); err != nil {
			t.Fatal(err)
		}
	}
	before0 := recurrentSlotDirty(t, r, 0)
	if before0 == 0 || recurrentSlotDirty(t, r, 1) == 0 {
		t.Fatalf("after their sequences, slot 0 has %d and slot 1 %d non-zero state elements; want both dirty", before0, recurrentSlotDirty(t, r, 1))
	}
	if duringReset != nil { // a forward from position 0 resets too, so a plant holds for this reset alone
		*duringReset = true
		defer func() { *duringReset = false }()
	}
	r.Reset() // slot 1 is bound
	if duringReset != nil {
		*duringReset = false
	}
	if n := recurrentSlotDirty(t, r, 1); n != 0 {
		wrong = append(wrong, fmt.Sprintf("a reset left %d of the bound slot's state elements non-zero", n))
	}
	if n := recurrentSlotDirty(t, r, 0); n != before0 {
		wrong = append(wrong, fmt.Sprintf("a reset with slot 1 bound changed slot 0: %d non-zero elements, %d before", n, before0))
	}
	recurrentSlotBind(t, r, 0)
	for i := half; i < len(A); i++ {
		l, err := r.Forward(A[i], i)
		if err != nil {
			t.Fatal(err)
		}
		if d := logitsDiffer(l, refA[i]); d != 0 {
			wrong = append(wrong, fmt.Sprintf("slot 0's sequence at position %d: %d logits differ from its solo run", i, d))
			break
		}
	}
	return wrong
}

// TestRecurrentSlots_resetIsTheSlotsOwn is S-2. Its plant zeroes every slot's pair on a reset, with the slots otherwise apart, so only the reset can be what goes red.
// Mamba-2's own reset seam leaves the bound slot dirty: the check that the reset does anything.
func TestRecurrentSlots_resetIsTheSlotsOwn(t *testing.T) {
	for _, fixture := range recurrentSlotFixtures {
		t.Run(fixture, func(t *testing.T) {
			m, r := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			A, B := recurrentSlotSeqs(m, 12)
			_, one := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 1})
			refA := recurrentSlotSolo(t, one, A)
			for _, w := range recurrentSlotReset(t, r, A, B, refA, nil) {
				t.Error(w)
			}
			_, planted := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			every := recurrentSlotReset(t, planted, A, B, refA, &recurrentResetEverySlotForTest)
			t.Logf("planted, a reset zeroes every slot: %v", every)
			logits := false
			for _, w := range every {
				logits = logits || strings.Contains(w, "logits differ")
			}
			if !logits {
				t.Error("with a reset that zeroes every slot, slot 0's sequence still equals its solo run: the logit check does not see it")
			}
		})
	}
	t.Run("a reset that does nothing", func(t *testing.T) {
		m, r := recurrentSlotsLoad(t, "nemotron-tiny", decoder.Options{ResidentKVSlots: 2})
		A, B := recurrentSlotSeqs(m, 12)
		_, one := recurrentSlotsLoad(t, "nemotron-tiny", decoder.Options{ResidentKVSlots: 1})
		refA := recurrentSlotSolo(t, one, A)
		wrong := recurrentSlotReset(t, r, A, B, refA, &mambaResetSkipForTest)
		t.Logf("planted: %v", wrong)
		if len(wrong) == 0 {
			t.Error("with a reset that leaves the Mamba-2 state, the check found nothing")
		}
	})
}

// TestRecurrentSlots_stateLinePerSlot is S-3: the ledger's state line holds the plan's one slot's state once per slot, plus the
// checkpoint's copy, at 1, 2 and 4 slots. A build whose further slot cannot allocate its state gives the slot up, and leaves one pair.
func TestRecurrentSlots_stateLinePerSlot(t *testing.T) {
	stateLine := func(r *cudaResident) int64 {
		l, ok := r.dev.Leaser().(*deviceLedger)
		if !ok {
			t.Fatal("the device has no ledger")
		}
		return l.Usage().ByLine[mem.LineState]
	}
	for _, fixture := range recurrentSlotFixtures {
		for _, slots := range []int{1, 2, 4} {
			m, r := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: slots})
			if n := r.KVSlots(); n != slots {
				t.Fatalf("%s: %d slots, asked for %d", fixture, n, slots)
			}
			one := m.Plan("cuda", 1<<40, decoder.PlanRequest{Ctx: 1}).StateBytes
			if one <= 0 || one != r.recurrentStateCharged() {
				t.Errorf("%s: the plan prices %d bytes of state for a slot, the build charges %d", fixture, one, r.recurrentStateCharged())
			}
			want := int64(slots) * one
			if r.RecurrentCheckpointBytes() > 0 {
				want += one
			}
			got := stateLine(r)
			t.Logf("%s at %d slots: %d bytes on the state line, one slot's %d", fixture, slots, got, one)
			if got != want {
				t.Errorf("%s at %d slots: the state line holds %d bytes, want %d", fixture, slots, got, want)
			}
		}
		r := func() *cudaResident {
			recurrentSlotAllocFailForTest = true
			defer func() { recurrentSlotAllocFailForTest = false }()
			_, r := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			return r
		}()
		if n := r.KVSlots(); n != 1 {
			t.Errorf("%s: a further slot's state failed to allocate, and the build kept %d slots", fixture, n)
		}
		for l := range r.layers {
			if len(r.layers[l].recSlots) != 0 {
				t.Errorf("%s: layer %d kept %d pairs after its further slot was given up", fixture, l, len(r.layers[l].recSlots))
			}
		}
		if r.RecurrentCheckpointBytes() <= 0 {
			t.Errorf("%s: the load kept no checkpoint (%s), so the state line below reads less than this test means", fixture, r.ckptDeclined)
		}
		if got, want := stateLine(r), 2*r.recurrentStateCharged(); got != want {
			t.Errorf("%s: after the further slot was given up, the state line holds %d bytes, want %d, one slot's and the checkpoint's", fixture, got, want)
		}
	}
}

// TestRecurrentSlots_fitPricesTheState is S-3's fit, no device: one byte short of the second slot's KV and state keeps one slot.
func TestRecurrentSlots_fitPricesTheState(t *testing.T) {
	ls := []cudaLayer{{qDim: 64, hd: 16, kvDim: 64}}
	const c, reserve, state = 1024, int64(1000), int64(1 << 20)
	at := kvChargedBytes(c, 2, ls) + state + reserve
	if n := kvChargedSlotsFit(2, at, c, ls, reserve, state); n != 2 {
		t.Errorf("at the second slot's KV and state, %d slots fit; want 2", n)
	}
	if n := kvChargedSlotsFit(2, at-1, c, ls, reserve, state); n != 1 {
		t.Errorf("one byte short of the second slot's state, %d slots fit; want 1", n)
	}
	if n := kvChargedSlotsFit(2, at-1, c, ls, reserve, 0); n != 2 {
		t.Errorf("a model with no state: %d slots fit; want 2", n)
	}
}

// TestRecurrentSlots_checkpointStaysWithItsSlot is S-4: one copy, saved from the bound slot. A restore before any save, or with
// another slot bound, is refused; on its own slot it brings back that slot's state, so the sequence continues as if never moved.
func TestRecurrentSlots_checkpointStaysWithItsSlot(t *testing.T) {
	for _, fixture := range recurrentSlotFixtures {
		t.Run(fixture, func(t *testing.T) {
			m, r := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			if r.RecurrentCheckpointBytes() <= 0 {
				t.Fatalf("the load kept no recurrent checkpoint: %s", r.ckptDeclined)
			}
			_, one := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 1})
			A, B := recurrentSlotSeqs(m, 12)
			refA := recurrentSlotSolo(t, one, A)
			if err := r.RestoreRecurrentCheckpoint(); err == nil {
				t.Error("restoring a checkpoint before any save was not refused")
			}
			half := len(A) / 2
			recurrentSlotBind(t, r, 1)
			r.Reset()
			for i := range half {
				if _, err := r.Forward(A[i], i); err != nil {
					t.Fatal(err)
				}
			}
			if err := r.SaveRecurrentCheckpoint(); err != nil {
				t.Fatal(err)
			}
			recurrentSlotBind(t, r, 0)
			recurrentSlotSolo(t, r, B)
			dirty0 := recurrentSlotDirty(t, r, 0)
			if err := r.RestoreRecurrentCheckpoint(); err == nil {
				t.Error("restoring slot 1's checkpoint with slot 0 bound was not refused")
			}
			if n := recurrentSlotDirty(t, r, 0); n != dirty0 {
				t.Errorf("a refused restore changed slot 0: %d non-zero elements, %d before", n, dirty0)
			}
			recurrentSlotBind(t, r, 1)
			for i := half; i < half+2; i++ { // move slot 1's state past the save
				if _, err := r.Forward(B[i], i); err != nil {
					t.Fatal(err)
				}
			}
			if err := r.RestoreRecurrentCheckpoint(); err != nil {
				t.Fatalf("restoring the checkpoint on its own slot: %v", err)
			}
			for i := half; i < len(A); i++ {
				l, err := r.Forward(A[i], i)
				if err != nil {
					t.Fatal(err)
				}
				if d := logitsDiffer(l, refA[i]); d != 0 {
					t.Fatalf("after the restore, position %d: %d logits differ from the sequence's solo run", i, d)
				}
			}
		})
	}
}

// TestRecurrentSlots_decoderChoosesAmongSlots goes through Generate, which is how a slot gets bound: two conversations in turn keep
// their slots, each reply equals a cold one-slot model's, and each follow-up reuses its own conversation's prompt.
func TestRecurrentSlots_decoderChoosesAmongSlots(t *testing.T) {
	gen := func(m *decoder.Model, prompt []int) ([]int, *decoder.Generation) {
		t.Helper()
		ch, g := m.Generate(context.Background(), prompt, 6, decoder.SamplingParams{})
		var toks []int
		for tok := range ch {
			toks = append(toks, tok)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return toks, g
	}
	for _, fixture := range recurrentSlotFixtures {
		t.Run(fixture, func(t *testing.T) {
			m, _ := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 2})
			if got := m.ResidentKVSlots(); got != 2 {
				t.Fatalf("the decoder chooses among %d slots; want 2", got)
			}
			cold, _ := recurrentSlotsLoad(t, fixture, decoder.Options{ResidentKVSlots: 1})
			_, _, _, _, _, _, vocab := m.Dims()
			seq := func(n, a, b int) []int {
				s := make([]int, n)
				for i := range s {
					s[i] = (i*a + b) % vocab
				}
				return s
			}
			P, Q := seq(24, 31, 7), seq(24, 17, 101)
			hist := map[string][]int{"P": P, "Q": Q}
			for _, name := range []string{"P", "Q", "P", "Q", "P"} {
				prompt := hist[name]
				got, g := gen(m, prompt)
				cold.ForgetResidentForTest()
				want, _ := gen(cold, prompt)
				t.Logf("%s, %d tokens: reused %d; reply %v, cold %v", name, len(prompt), g.PrefillReused, got, want)
				if !slices.Equal(got, want) {
					t.Errorf("%s: the reply %v differs from a cold one-slot model's %v", name, got, want)
				}
				if len(prompt) > 24 && g.PrefillReused < 24 {
					t.Errorf("%s: a follow-up reused %d of its own conversation's %d tokens", name, g.PrefillReused, len(prompt))
				}
				hist[name] = append(append(slices.Clone(prompt), got...), 5, 6)
			}
		})
	}
}

// TestRecurrentSlots_snapshotHoldsTheSlotItNames is S-5, Gated-DeltaNet only: a snapshot taken and restored with another slot bound
// holds and writes the slot it names, so that slot's sequence continues as if never moved.
func TestRecurrentSlots_snapshotHoldsTheSlotItNames(t *testing.T) {
	const mark, n = 64, 72
	m, r := recurrentSlotsLoad(t, "qwen35-tiny", decoder.Options{ResidentKVSlots: 2})
	if !r.KVSnapshotRecurrent() {
		t.Fatal("a Gated-DeltaNet resident's snapshots do not carry its state")
	}
	_, one := recurrentSlotsLoad(t, "qwen35-tiny", decoder.Options{ResidentKVSlots: 1})
	A, B := recurrentSlotSeqs(m, n)
	refA := recurrentSlotSolo(t, one, A)
	recurrentSlotBind(t, r, 0)
	r.Reset()
	for i := range mark {
		if _, err := r.Forward(A[i], i); err != nil {
			t.Fatal(err)
		}
	}
	recurrentSlotBind(t, r, 1)
	r.Reset()
	for i := range 8 {
		if _, err := r.Forward(B[i], i); err != nil {
			t.Fatal(err)
		}
	}
	snap, err := r.SnapshotKV(0, mark) // slot 1 is bound
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Release()
	recurrentSlotBind(t, r, 0)
	for i := mark; i < n; i++ { // moves slot 0's state past the snapshot
		if _, err := r.Forward(B[i], i); err != nil {
			t.Fatal(err)
		}
	}
	recurrentSlotBind(t, r, 1)
	if err := r.RestoreKV(snap, 0, mark); err != nil {
		t.Fatal(err)
	}
	recurrentSlotBind(t, r, 0)
	for i := mark; i < n; i++ {
		l, err := r.Forward(A[i], i)
		if err != nil {
			t.Fatal(err)
		}
		if d := logitsDiffer(l, refA[i]); d != 0 {
			t.Fatalf("after the restore, position %d: %d logits differ from the sequence's solo run", i, d)
		}
	}
}

// TestRecurrentSlots_markedRowsFoldIntoTheirSlot: rows a speculative verify marked fold into the slot that marked them when another
// slot is bound, not into the new slot at its next verify, and the sequence then continues as it does on one slot.
func TestRecurrentSlots_markedRowsFoldIntoTheirSlot(t *testing.T) {
	const seed, verify, keep = 6, 4, 2
	run := func(r *cudaResident, embs [][]float32, leave func() error) []float32 {
		t.Helper()
		if err := r.PrepareRecurrentSpec(verify); err != nil {
			t.Fatal(err)
		}
		r.Reset()
		for i := range seed {
			if _, err := r.Forward(embs[i], i); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := r.VerifyLastNArgmax(embs[seed:seed+verify], seed); err != nil {
			t.Fatal(err)
		}
		if err := r.FoldVerified(keep); err != nil {
			t.Fatal(err)
		}
		if err := leave(); err != nil {
			t.Fatal(err)
		}
		l, err := r.Forward(embs[seed+keep], seed+keep)
		if err != nil {
			t.Fatal(err)
		}
		return slices.Clone(l)
	}
	m, two := recurrentSlotsLoad(t, "qwen35-tiny", decoder.Options{ResidentKVSlots: 2})
	_, one := recurrentSlotsLoad(t, "qwen35-tiny", decoder.Options{ResidentKVSlots: 1})
	A, _ := recurrentSlotSeqs(m, seed+verify)
	want := run(one, A, one.FlushFold)
	recurrentSlotBind(t, two, 0)
	got := run(two, A, func() error {
		if err := two.UseKVSlot(1); err != nil {
			return err
		}
		if n := two.dnSpec.pend; n != 0 {
			t.Errorf("%d marked rows still wait after another slot was bound", n)
		}
		if n := recurrentSlotDirty(t, two, 1); n != 0 {
			t.Errorf("the slot bound next holds %d non-zero state elements", n)
		}
		return two.UseKVSlot(0)
	})
	if d := logitsDiffer(got, want); d != 0 {
		t.Errorf("after the slot switch, %d logits differ from one slot's", d)
	}
}
