//go:build cuda

package cuda

import (
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestWeightsFitDecision holds the weights check at its boundaries: an unknown reading passes, a fit exactly at the budget
// passes, one byte over refuses, and a companion attach is named in the refusal.
func TestWeightsFitDecision(t *testing.T) {
	const need, margin = int64(1000), int64(100)
	for _, c := range []struct {
		name        string
		extra, free int64
		ok          bool
		want        string // "" fits; else a substring of the refusal
	}{
		{"no reading", 0, 0, false, ""},
		{"exactly at the budget", 0, need + margin, true, ""},
		{"one byte over", 0, need + margin - 1, true, "dense weights need"},
		{"companion at the budget", 50, need + 50 + margin, true, ""},
		{"companion one byte over", 50, need + 50 + margin - 1, true, "reserved for a companion"},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := weightsFitDecision(need, c.extra, c.free, c.ok, margin)
			switch {
			case c.want == "" && err != nil:
				t.Errorf("refused: %v", err)
			case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
				t.Errorf("got %v, want a refusal containing %q", err, c.want)
			}
		})
	}
	if weightsFitDecision(0, 0, 1, true, margin) != nil {
		t.Error("an unknown need (0) must pass")
	}
}

// kvTestLayers is two full-attention layers with f32 KV, kvDim 256: what kvChargedBytes prices.
func kvTestLayers() []cudaLayer {
	return []cudaLayer{{idx: 0, kvDim: 256, nKV: 2, hd: 128}, {idx: 1, kvDim: 256, nKV: 2, hd: 128}}
}

// TestKVFitDecision holds checkKVFits' decision at its boundaries: an unknown reading grants one slot at the planned context;
// an explicit context fits exactly at the budget and refuses one byte under it with kvWontFitError; an unpinned context over
// the budget is trimmed, never below cudaCtxCapDefault, and refused when even that does not fit; extra slots are clamped to
// what fits.
func TestKVFitDecision(t *testing.T) {
	ls := kvTestLayers()
	const margin = int64(1 << 20)
	one := func(ctx int) int64 { return kvChargedBytes(ctx, 1, ls) }

	if d := kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 4, layers: ls, margin: margin}, 0, false); d.ctxCap != 8192 || d.slots != 1 || d.err != nil || len(d.logs) != 0 {
		t.Errorf("no reading: %+v, want ctx 8192, 1 slot, no refusal", d)
	}

	at := uint64(one(8192) + margin)
	if d := kvFitDecision(kvFitIn{ctxCap: 8192, explicit: true, slotsReq: 1, layers: ls, margin: margin}, at, true); d.err != nil || d.ctxCap != 8192 {
		t.Errorf("explicit, exactly at the budget: %+v", d)
	}
	d := kvFitDecision(kvFitIn{ctxCap: 8192, explicit: true, slotsReq: 1, layers: ls, margin: margin}, at-1, true)
	var wf *kvWontFitError
	if !errors.As(d.err, &wf) || d.ctxCap != 8192 {
		t.Errorf("explicit, one byte under: %+v, want kvWontFitError at ctx 8192", d)
	}

	d = kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 1, layers: ls, margin: margin}, at-1, true)
	if d.err != nil || d.ctxCap >= 8192 || d.ctxCap < cudaCtxCapDefault || len(d.logs) != 1 || !strings.Contains(d.logs[0], "trimmed from 8192") {
		t.Errorf("unpinned, one byte under: %+v, want a trim below 8192, at or above %d, logged once", d, cudaCtxCapDefault)
	}
	if one(d.ctxCap)+margin > int64(at-1) {
		t.Errorf("the trimmed context %d does not fit the budget it was trimmed to", d.ctxCap)
	}

	floor := uint64(one(cudaCtxCapDefault) + margin)
	d = kvFitDecision(kvFitIn{ctxCap: 8192, slotsReq: 1, layers: ls, margin: margin}, floor-1, true)
	if d.err == nil || !strings.Contains(d.err.Error(), "default resident context") || errors.As(d.err, &wf) {
		t.Errorf("unpinned, under the floor: %+v, want the default-context decline (not kvWontFitError)", d)
	}

	two := uint64(kvChargedBytes(cudaCtxCapDefault, 2, ls) + margin)
	d = kvFitDecision(kvFitIn{ctxCap: cudaCtxCapDefault, slotsReq: 4, layers: ls, margin: margin}, two, true)
	if d.err != nil || d.slots != 2 || len(d.logs) != 1 || !strings.Contains(d.logs[0], "2 resident KV slots of 4") {
		t.Errorf("slot clamp: %+v, want 2 slots of 4, logged once", d)
	}
}

// TestPlanMemory runs the memory plan on a tiny fixture without a device. With room everywhere it plans one slot at the
// default context; a short weights reading or KV reading refuses at that site; an unknown KV reading grants one slot.
func TestPlanMemory(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Skipf("fixture: %v", err)
	}
	defer m.Close()
	const room = int64(8 << 30)
	full := memReadings{WeightsOpen: room, Plan: room, Weights: room, KV: room, WeightsOpenOK: true, PlanOK: true, WeightsOK: true, KVOK: true}
	b := &cudaBackend{}
	p, err := b.planMemory(m, full)
	if err != nil {
		t.Fatal(err)
	}
	if p.Decline != "" || p.CtxCap < cudaCtxCapDefault || p.KVSlots < 1 || p.Margin <= 0 {
		t.Fatalf("room everywhere: %+v", p)
	}
	short := full
	short.Weights = 1
	if p, err := b.planMemory(m, short); err != nil || p.DeclineAt != "weights" {
		t.Errorf("short weights reading: %+v %v, want a refusal at weights", p, err)
	}
	short = full
	short.KV = 1
	if p, err := b.planMemory(m, short); err != nil || p.DeclineAt != "kv" {
		t.Errorf("short KV reading: %+v %v, want a refusal at kv", p, err)
	}
	unknown := full
	unknown.KVOK = false
	if p, err := b.planMemory(m, unknown); err != nil || p.Decline != "" || p.KVSlots != 1 {
		t.Errorf("unknown KV reading: %+v %v, want one slot and no refusal", p, err)
	}
}
