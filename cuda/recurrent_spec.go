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
// each row's recurrence inputs into dnSpecArena instead of the pass's scratch. FoldVerified marks the rows the loop commits; the
// next verify's twins fold them into the live window and state before scanning their own rows, and FlushFold folds them through
// prefill's own kernels when something other than a verify comes next. Either way the committed state is the bits a prefill of
// those rows would leave, and no copy of the state is ever made.

var _ decoder.ResidentRecurrentSpec = (*cudaResident)(nil)

// dnSpecArena holds, per DeltaNet layer, a verify's rows in two banks: the conv input, the conv output, the normed q and k and the
// gates, at most rows of each, in one device allocation (Buffer.At slices it). A verify writes one bank while the twins fold the
// previous verify's committed rows from the other. bank is the bank the last verify wrote; avail is how many of its rows
// FoldVerified may still mark; pend is how many it marked, waiting to fold.
type dnSpecArena struct {
	buf   Buffer
	bytes int64
	rows  int
	slots [2]map[int]dnPrefillBufs // bank → layer index → that layer's slices (mix, conv, qn, kn, headP)
	core  Buffer                   // FlushFold's recurrence output, [rows × valueDim]: read by nothing
	bank  int
	avail int
	pend  int
}

// dnSpecPlantForTest plants a defect for SP2's identity gate to catch (G1's P1 and P2). Set by tests only.
var dnSpecPlantForTest int

const (
	dnSpecPlantFoldSkipsConv     = iota + 1 // the fold replays the scan but leaves the conv window where it was
	dnSpecPlantVerifyWritesState            // the verify commits every row, as before SP2, and the fold does nothing
)

// recurrentSpecFloats is the f32 count one bank keeps per row and DeltaNet layer: conv input and output (convDim each), normed q
// and k (keyDim each) and (beta, gt) per value head. decoder.RecurrentSpecArenaBytes prices the same formula, for both banks.
func recurrentSpecFloats(dp *dnetParams) int { return 2*dp.convDim + 2*dp.keyDim + 2*dp.nv }

// PrepareRecurrentSpec (decoder.ResidentRecurrentSpec) allocates the arena for verify passes of up to rows rows, once, before a
// block drafter attaches. It keeps none, and says why, for a model whose recurrent layers are not all Gated DeltaNet, whose heads are
// wider than the scan twins hold, or whose card has no room for it. An arena already holding rows rows is kept; a narrower one is
// released and replaced. Called on the caller's goroutine.
func (r *cudaResident) PrepareRecurrentSpec(rows int) error {
	if a := r.dnSpec; a != nil {
		if a.rows >= rows {
			return nil
		}
		r.dnSpec = nil
		if err := r.do(func() error { r.dev.ReleaseBuf(a.buf); return nil }); err != nil {
			return err
		}
	}
	if err := r.allocRecurrentSpec(rows); err != nil {
		fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: none kept (%v)\n", err)
		return err
	}
	return nil
}

// allocRecurrentSpec is PrepareRecurrentSpec's allocation.
func (r *cudaResident) allocRecurrentSpec(rows int) error {
	dp := r.dnet
	switch {
	case dp == nil:
		return fmt.Errorf("no Gated-DeltaNet layer")
	case rows < 1:
		return fmt.Errorf("%d rows", rows)
	case r.mamba != nil:
		return fmt.Errorf("a Mamba-2 layer: the fold path covers Gated DeltaNet only")
	case dp.hk > 128 || dp.hv > 128:
		return fmt.Errorf("head dims %d/%d past the scan twins' 128", dp.hk, dp.hv)
	case r.dnConvRowsSpec == (Pipeline{}) || r.dnRuleRowsSpec == (Pipeline{}) || r.dnRuleRows128Spec == (Pipeline{}):
		return fmt.Errorf("the verify kernels did not load")
	}
	var dnLayers []int
	for l := range r.layers {
		if r.layers[l].isDeltaNet() {
			dnLayers = append(dnLayers, l)
		}
	}
	per := rows * recurrentSpecFloats(dp)
	total := 2*len(dnLayers)*per + rows*dp.valueDim
	a := &dnSpecArena{rows: rows}
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
		return err
	}
	a.bytes = int64(total) * 4
	off := 0
	take := func(n int) Buffer {
		b := a.buf.At(off * 4)
		off += n
		return b
	}
	for bank := range a.slots {
		a.slots[bank] = make(map[int]dnPrefillBufs, len(dnLayers))
		for _, l := range dnLayers {
			a.slots[bank][l] = dnPrefillBufs{
				mix: take(rows * dp.convDim), conv: take(rows * dp.convDim),
				qn: take(rows * dp.keyDim), kn: take(rows * dp.keyDim), headP: take(rows * dp.nv * 2),
			}
		}
	}
	a.core = take(rows * dp.valueDim)
	r.dnSpec = a
	fmt.Fprintf(os.Stderr, "cuda: speculative recurrent rows: %.1f MiB on the device (%d rows x %d DeltaNet layers, SP2)\n",
		mib(a.bytes), rows, len(dnLayers))
	return nil
}

// specBufs is the DeltaNet buffers layer l's mixer uses in this pass: b itself, or in a verify pass b with the row inputs pointed
// at the bank this verify writes, so its committed rows can fold later.
func (r *cudaResident) specBufs(l int, b dnPrefillBufs) dnPrefillBufs {
	if !r.dnSpecOn {
		return b
	}
	s := r.dnSpec.slots[r.dnSpec.bank^1][l]
	b.mix, b.conv, b.qn, b.kn, b.headP = s.mix, s.conv, s.qn, s.kn, s.headP
	return b
}

// specPending is what layer l's twins fold before a verify's own rows: the previous verify's bank and the rows marked in it. pM
// is 0 outside a verify pass and when nothing is marked.
func (r *cudaResident) specPending(l int) (p dnPrefillBufs, pM int) {
	if !r.dnSpecOn {
		return dnPrefillBufs{}, 0
	}
	a := r.dnSpec
	return a.slots[a.bank][l], a.pend
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

// VerifyLastNArgmax (decoder.ResidentRecurrentSpec) is PrefillLastNArgmax's pass with every DeltaNet layer's twins: the rows the
// last verify marked fold into the window and state first, then this pass's rows scan without writing them. Its rows wait in the
// other bank for FoldVerified. On an error the committed state is unknown, and the caller treats the sequence as lost.
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
	r.dnSpecOn = dnSpecPlantForTest != dnSpecPlantVerifyWritesState
	defer func() { r.dnSpecOn = false }()
	ids, err := r.PrefillLastNArgmax(embeddings, startPos)
	a.bank, a.avail, a.pend = a.bank^1, 0, 0
	if err == nil {
		a.avail = len(embeddings)
	}
	return ids, err
}

// FoldVerified (decoder.ResidentRecurrentSpec) marks the last verify's first n rows committed. They fold at the next verify, or at
// FlushFold.
func (r *cudaResident) FoldVerified(n int) error {
	a := r.dnSpec
	if a == nil {
		return fmt.Errorf("cuda: no speculative recurrent rows kept on this load")
	}
	if n < 1 || n > a.avail {
		return fmt.Errorf("cuda: fold of %d rows; the last verify left %d", n, a.avail)
	}
	a.avail = 0
	if dnSpecPlantForTest != dnSpecPlantVerifyWritesState {
		a.pend = n
	}
	return nil
}

// FlushFold (decoder.ResidentRecurrentSpec) folds the marked rows now, through delta_conv_rows and the scan prefill runs: n rank-1
// updates per value head per layer, in place. Nothing marked is a no-op.
func (r *cudaResident) FlushFold() error {
	a := r.dnSpec
	if a == nil || a.pend == 0 {
		return nil
	}
	n := a.pend
	a.pend = 0
	dp := r.dnet
	return r.do(func() error {
		r.launchErr = nil
		for l := range r.layers {
			Ly := &r.layers[l]
			if !Ly.isDeltaNet() {
				continue
			}
			s := a.slots[a.bank][l]
			// The conv output is written back over the row's own, with the same bits: the fold's window starts where the
			// verify's reads did.
			if dnSpecPlantForTest != dnSpecPlantFoldSkipsConv {
				if e := r.launch(r.dnConvRows, g1cfg(dp.convDim, 256),
					Arg(s.mix), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(s.conv),
					gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK)), gpu.ArgValue(int32(n))); e != nil {
					return e
				}
			}
			if e := r.launchRuleRows(false, s.qn, s.kn, s.conv, s.headP, Ly.dnState, a.core, n, dnPrefillBufs{}, 0); e != nil {
				return e
			}
		}
		return r.launchErr
	})
}

// launchRuleRows launches the row scan for the model's geometry, the state-row-in-registers kernel for hk == hv == 128 and the
// generic one otherwise; spec takes their twins, which fold pM rows of p into the state first and never store their own.
func (r *cudaResident) launchRuleRows(spec bool, qn, kn, conv, headP, state, core Buffer, M int, p dnPrefillBufs, pM int) error {
	dp := r.dnet
	if dp.hk == 128 && dp.hv == 128 {
		cfg := LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1}
		args := []KernelArg{Arg(qn), Arg(kn), Arg(conv), Arg(headP), Arg(state), Arg(core),
			gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2 * dp.keyDim)),
			gpu.ArgValue(int32(dp.keyDim)), gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))}
		if spec {
			return r.launch(r.dnRuleRows128Spec, cfg, append(args, specPendArgs(p, kn, conv, headP, pM)...)...)
		}
		return r.launch(r.dnRuleRows128, cfg, args...)
	}
	cfg := g1cfg(dp.valueDim, 128)
	args := []KernelArg{Arg(qn), Arg(kn), Arg(conv), Arg(headP), Arg(state), Arg(core),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2 * dp.keyDim)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(M))}
	if spec {
		return r.launch(r.dnRuleRowsSpec, cfg, append(args, specPendArgs(p, kn, conv, headP, pM)...)...)
	}
	return r.launch(r.dnRuleRows, cfg, args...)
}

// specPendArgs is the scan twins' trailing arguments: the pending rows' k, conv output (for v) and gates, and their count. With
// none pending the pass's own buffers stand in, unread.
func specPendArgs(p dnPrefillBufs, kn, conv, headP Buffer, pM int) []KernelArg {
	if pM == 0 {
		p.kn, p.conv, p.headP = kn, conv, headP
	}
	return []KernelArg{Arg(p.kn), Arg(p.conv), Arg(p.headP), gpu.ArgValue(int32(pM))}
}
