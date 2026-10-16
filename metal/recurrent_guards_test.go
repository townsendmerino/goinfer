//go:build darwin

package metal

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestRecurrentGuards_declineMamba is OP2b's guard gate (docs/tasks/task-forward-architecture-2026-10.md §4.7): every path
// that declined a Gated-DeltaNet model declines a Mamba-2 one, through the general recurrent check (residLayer.recurrent,
// resident.recurrentModel), on a resident built by hand with one Mamba-2 layer. Where the recurrent check is what decides, a
// control without the Mamba layer must not decline, so the guard is not passing on an earlier, unrelated exit. A guard that
// panics counts as admitting: it went past its decline.
func TestRecurrentGuards_declineMamba(t *testing.T) {
	mambaRes := func() *resident {
		return &resident{V: 1024, H: 64, mamba: &mambaParams{}, layers: []residLayer{{mamba: &mambaLayer{}}},
			decodeLaneW4F16: true, kvSlotBufs: make([]kvSlotBuf, 2), ctxCap: 4096}
	}
	plainRes := func() *resident {
		return &resident{V: 1024, H: 64, layers: []residLayer{{ffn: true}}, decodeLaneW4F16: true}
	}
	declines := func(f func() bool) (d bool) {
		defer func() {
			if recover() != nil {
				d = false
			}
		}()
		return f()
	}
	prevI8, prevMix := nativeInt8, nativeInt4Mix
	nativeInt8, nativeInt4Mix = true, true
	defer func() { nativeInt8, nativeInt4Mix = prevI8, prevMix }()

	guards := []struct {
		name    string
		decline func(r *resident) bool
		control bool // a resident without the Mamba layer is admitted by this guard
	}{
		{"greedy chain", func(r *resident) bool { return strings.Contains(r.greedyChainWhyNot(), "recurrent") }, true},
		{"KV copy", func(r *resident) bool {
			var d *decoder.KVCopyDecline
			err := (&metalResident{r: r}).CopyKV(0, 1, 0, 64)
			return errors.As(err, &d) && strings.Contains(d.Reason, "recurrent")
		}, false},
		{"batched step", func(r *resident) bool { return r.batchIneligible() != "" }, false},
		{"batched step attention-FA", func(r *resident) bool { return !r.canUseAttnFAAt(0, 4096) }, false},
		{"attention-FA", func(r *resident) bool { return !r.canUseAttnFA(0) }, false},
		{"f16 FFN lane", func(r *resident) bool { return !r.canUseF16Lane(0) }, true},
		{"native int8", func(r *resident) bool { return !w8Eligible(nil, r) }, false},
		{"native int8 attention", func(r *resident) bool { return !w8AttnEligible(nil, r) }, false},
		{"batched prefill", func(r *resident) bool { return !r.recurrentPrefillOK() }, true},
		{"q4k lane", func(r *resident) bool { return !r.q4kLaneShapeOK() }, true},
		{"E-model batched prefill", func(r *resident) bool { return !(&metalResident{r: r}).emodelBatched() }, false},
		{"image prefill", func(r *resident) bool {
			_, err := (&metalResident{r: r}).PrefillImageBlocksLast(context.Background(), [][]float32{{0}}, 0, [][2]int{{0, 1}})
			return err != nil
		}, false},
	}
	for _, g := range guards {
		if !declines(func() bool { return g.decline(mambaRes()) }) {
			t.Errorf("%s: a resident with a Mamba-2 layer is not declined", g.name)
		}
		if g.control && declines(func() bool { return g.decline(plainRes()) }) {
			t.Errorf("%s: the control without the Mamba-2 layer is declined too, so the recurrent check is not what decides", g.name)
		}
	}
	t.Logf("%d guards decline a Mamba-2 resident", len(guards))
}
