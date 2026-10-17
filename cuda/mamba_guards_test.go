//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
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

// withTwoSlots gives r a second slot entry that aliases the first's buffers, so the checks that run only on a multi-slot resident (the batched step, the
// copy) get past the slot count to the guard under test. Nothing is copied or run through it.
func withTwoSlots(r *cudaResident) {
	if r.KVSlots() < 2 {
		r.kvSlotBufs = []cudaKVSlot{{kc: r.kc, vc: r.vc}, {kc: r.kc, vc: r.vc}}
	}
}

// TestNemotronResidentCUDA_guards is OP2b's gate G on CUDA: a Nemotron-H resident is declined by each site that refuses recurrent state (the batched
// step, the prefix snapshot, the KV copy, the verify pass, and the batched prefill by name), and a Gated-DeltaNet one is declined by the same sites as before.
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
	if ctrl.batchHi < 2 || ctrl.copyDecl != "" || ctrl.snapDecl != "" || !ctrl.verifyBatched {
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
			if !strings.Contains(res.snapDecl, "recurrent") {
				t.Errorf("the prefix snapshot of %s: decline %q, want the recurrent-state decline", c.fixture, res.snapDecl)
			}
			if res.verifyBatched {
				t.Errorf("ForwardN's verify pass on %s reports a batched path", c.fixture)
			}
			static := r.prefillStaticDecline()
			if c.fixture == "nemotron-tiny" && (static == nil || !strings.Contains(static.Error(), "Mamba-2 mixer")) {
				t.Errorf("the batched prefill on nemotron-tiny: %v, want a decline naming the Mamba-2 mixer", static)
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
	if got := kvBytesForCap(ctx, r.layers); got != want {
		t.Errorf("kvBytesForCap(%d) = %d, the %d attention layers alone are %d", ctx, got, attn, want)
	}
	if got := m.ResidentKVBytes("cuda", ctx, r.kvF16, false); got != want {
		t.Errorf("the decoder's ResidentKVBytes(cuda, %d) = %d, want %d", ctx, got, want)
	}
	if got := mambaStateBytes(r.mamba, nMamba); got != stateBytes {
		t.Errorf("mambaStateBytes = %d, the %d Mamba layers' state buffers are %d bytes", got, nMamba, stateBytes)
	}
	t.Logf("nemotron-tiny: %d attention layers of %d price %d KV bytes at %d positions; %d Mamba layers' state is %d bytes", attn, len(r.layers), want, ctx, nMamba, stateBytes)
}
