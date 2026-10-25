//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strings"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// SP2's speculative verify on a Gated-DeltaNet resident (decoder.ResidentRecurrentSpec; design and gates in
// docs/tasks/task-forward-architecture-2026-10.md §6a, "SP2 on nobara"). A verify pass runs the DeltaNet layers through the
// conv and scan twins that never store (delta_conv_rows_spec, delta_rule_rows_128_spec, delta_rule_rows_spec), and writes
// each row's recurrence inputs into dnSpecArena instead of the pass's scratch. FoldVerified then replays the committed rows
// through prefill's own kernels on the live window and state, in place, so the committed state is the bits a prefill of
// those rows would leave and no copy of the state is ever made.

var _ decoder.ResidentRecurrentSpec = (*cudaResident)(nil)

// dnSpecArena holds, per DeltaNet layer, the last verify's rows: the conv input, the conv output, the normed q and k and the
// gates, at most rows of each, in one device allocation (Buffer.At slices it). pend is how many rows the last verify left for
// FoldVerified; 0 when there is nothing to fold.
type dnSpecArena struct {
	buf   Buffer
	bytes int64
	rows  int
	slots map[int]dnPrefillBufs // layer index → that layer's slices (mix, conv, qn, kn, headP)
	core  Buffer                // the fold's recurrence output, [rows × valueDim]: read by nothing
	pend  int
}

// dnSpecPlantForTest plants a defect for SP2's identity gate to catch (G1's P1 and P2). Set by tests only.
var dnSpecPlantForTest int

const (
	dnSpecPlantFoldSkipsConv     = iota + 1 // the fold replays the scan but leaves the conv window where it was
	dnSpecPlantVerifyWritesState            // the verify commits every row, as before SP2, and the fold does nothing
)

// recurrentSpecFloats is the f32 count the arena keeps per row and DeltaNet layer: conv input and output (convDim each), normed
// q and k (keyDim each) and (beta, gt) per value head. decoder.RecurrentSpecArenaBytes prices the same formula.
func recurrentSpecFloats(dp *dnetParams) int { return 2*dp.convDim + 2*dp.keyDim + 2*dp.nv }

// allocRecurrentSpec allocates the arena for verify passes of up to rows rows, once, when a block drafter attaches. It keeps none,
// and says why, for a model whose recurrent layers are not all Gated DeltaNet or whose heads are wider than the scan twins hold;
// BlockSpec then refuses the model (decoder.Model.specRollbackSafe). Called on the caller's goroutine.
func (r *cudaResident) allocRecurrentSpec(rows int) {
	if r.dnet == nil || r.dnSpec != nil || rows < 1 {
		return
	}
	dp := r.dnet
	switch {
	case r.mamba != nil:
		fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: none kept (a Mamba-2 layer: SP2 covers Gated DeltaNet only)\n")
		return
	case dp.hk > 128 || dp.hv > 128:
		fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: none kept (head dims %d/%d past the scan twins' 128)\n", dp.hk, dp.hv)
		return
	case r.dnConvRowsSpec == (Pipeline{}) || r.dnRuleRowsSpec == (Pipeline{}) || r.dnRuleRows128Spec == (Pipeline{}):
		fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: none kept (the verify kernels did not load)\n")
		return
	}
	var dnLayers []int
	for l := range r.layers {
		if r.layers[l].isDeltaNet() {
			dnLayers = append(dnLayers, l)
		}
	}
	per := rows * recurrentSpecFloats(dp)
	total := len(dnLayers)*per + rows*dp.valueDim
	a := &dnSpecArena{rows: rows, slots: make(map[int]dnPrefillBufs, len(dnLayers))}
	err := r.do(func() (err error) {
		defer func() {
			if v := recover(); v != nil {
				if !strings.Contains(fmt.Sprint(v), "device allocation failed") {
					panic(v)
				}
				err = fmt.Errorf("the device allocation failed")
			}
		}()
		a.buf = r.af(total)
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: none kept (%v)\n", err)
		return
	}
	a.bytes = int64(total) * 4
	off := 0
	take := func(n int) Buffer {
		b := a.buf.At(off * 4)
		off += n
		return b
	}
	for _, l := range dnLayers {
		a.slots[l] = dnPrefillBufs{
			mix: take(rows * dp.convDim), conv: take(rows * dp.convDim),
			qn: take(rows * dp.keyDim), kn: take(rows * dp.keyDim), headP: take(rows * dp.nv * 2),
		}
	}
	a.core = take(rows * dp.valueDim)
	r.dnSpec = a
	fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: %.1f MiB on the device (%d rows x %d DeltaNet layers, SP2)\n",
		mib(a.bytes), rows, len(dnLayers))
}

// specBufs is the DeltaNet buffers layer l's mixer uses in this pass: b itself, or in a verify pass b with the row inputs
// pointed at the arena, so the fold can replay them.
func (r *cudaResident) specBufs(l int, b dnPrefillBufs) dnPrefillBufs {
	if !r.dnSpecOn {
		return b
	}
	s := r.dnSpec.slots[l]
	b.mix, b.conv, b.qn, b.kn, b.headP = s.mix, s.conv, s.qn, s.kn, s.headP
	return b
}

// RecurrentSpecRows (decoder.ResidentRecurrentSpec) is the widest verify the arena holds, 0 when this load kept none.
func (r *cudaResident) RecurrentSpecRows() int {
	if r.dnSpec == nil {
		return 0
	}
	return r.dnSpec.rows
}

// RecurrentSpecBytes is the arena's device bytes, 0 when this load kept none.
func (r *cudaResident) RecurrentSpecBytes() int64 {
	if r.dnSpec == nil {
		return 0
	}
	return r.dnSpec.bytes
}

// VerifyLastNArgmax (decoder.ResidentRecurrentSpec) is PrefillLastNArgmax's pass with every DeltaNet layer's committed state
// and conv window left as they were. Its rows wait in the arena for FoldVerified; a second verify replaces them.
func (r *cudaResident) VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	a := r.dnSpec
	switch {
	case a == nil:
		return nil, fmt.Errorf("cuda: no speculative recurrent rows kept on this load")
	case len(embeddings) < 1 || len(embeddings) > a.rows:
		return nil, fmt.Errorf("cuda: a verify of %d rows; the arena holds 1 to %d", len(embeddings), a.rows)
	case startPos < 1:
		// A pass from position 0 restarts the recurrence (prefillCore's resetState): there is no committed state to keep.
		return nil, fmt.Errorf("cuda: a speculative verify at position 0")
	}
	a.pend = 0
	r.dnSpecOn = dnSpecPlantForTest != dnSpecPlantVerifyWritesState
	defer func() { r.dnSpecOn = false }()
	ids, err := r.PrefillLastNArgmax(embeddings, startPos)
	if err == nil {
		a.pend = len(embeddings)
	}
	return ids, err
}

// FoldVerified (decoder.ResidentRecurrentSpec) replays the last verify's first n rows on every DeltaNet layer's live window and
// state through delta_conv_rows and the scan prefill runs: n rank-1 updates per value head, in place.
func (r *cudaResident) FoldVerified(n int) error {
	a := r.dnSpec
	if a == nil {
		return fmt.Errorf("cuda: no speculative recurrent rows kept on this load")
	}
	if n < 1 || n > a.pend {
		return fmt.Errorf("cuda: fold of %d rows; the last verify left %d", n, a.pend)
	}
	a.pend = 0
	if dnSpecPlantForTest == dnSpecPlantVerifyWritesState {
		return nil
	}
	dp := r.dnet
	return r.do(func() error {
		r.launchErr = nil
		for l := range r.layers {
			Ly := &r.layers[l]
			if !Ly.isDeltaNet() {
				continue
			}
			s := a.slots[l]
			// The conv output is written back over the row's own, with the same bits: the fold's window starts where the
			// verify's reads did.
			if dnSpecPlantForTest != dnSpecPlantFoldSkipsConv {
				if e := r.launch(r.dnConvRows, g1cfg(dp.convDim, 256),
					Arg(s.mix), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(s.conv),
					gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK)), gpu.ArgValue(int32(n))); e != nil {
					return e
				}
			}
			if e := r.launchRuleRows(false, s.qn, s.kn, s.conv, s.headP, Ly.dnState, a.core, n); e != nil {
				return e
			}
		}
		return r.launchErr
	})
}

// launchRuleRows launches the row scan for the model's geometry, the state-row-in-registers kernel for hk == hv == 128 and the
// generic one otherwise; spec takes their twins that never store the state.
func (r *cudaResident) launchRuleRows(spec bool, qn, kn, conv, headP, state, core Buffer, M int) error {
	dp := r.dnet
	if dp.hk == 128 && dp.hv == 128 {
		cfg := LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1}
		args := []KernelArg{Arg(qn), Arg(kn), Arg(conv), Arg(headP), Arg(state), Arg(core),
			gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2 * dp.keyDim)),
			gpu.ArgValue(int32(dp.keyDim)), gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))}
		if spec {
			return r.launch(r.dnRuleRows128Spec, cfg, args...)
		}
		return r.launch(r.dnRuleRows128, cfg, args...)
	}
	cfg := g1cfg(dp.valueDim, 128)
	args := []KernelArg{Arg(qn), Arg(kn), Arg(conv), Arg(headP), Arg(state), Arg(core),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2 * dp.keyDim)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))}
	if spec {
		return r.launch(r.dnRuleRowsSpec, cfg, args...)
	}
	return r.launch(r.dnRuleRows, cfg, args...)
}
