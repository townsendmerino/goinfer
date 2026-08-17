//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// residentDrafter is a block drafter's weights on the device, attached to an existing resident
// target (P10 / docs/spec/08).
//
// IT SHARES THE TARGET'S DEVICE, STREAM AND KERNELS rather than building its own context. The
// n-gram/EAGLE precedent (decoder/speculative.go) runs its draft as a separate *Model with its
// own context, which is right there because the draft IS a model. A block drafter is not: it has
// no embedding and no LM head (it borrows the target's), its context comes from the target's
// hidden states rather than from tokens, and it is FIVE layers against the target's thirty-six.
// Sharing means the drafter's block forward and the target's verify are already ordered on one
// stream, with no cross-context synchronisation between them — which matters because the two
// alternate every round.
//
// Built against decoder.BlockDrafterWeights, so DFlash and DSpark both work with no type switch.
type residentDrafter struct {
	r   *cudaResident // the target this is attached to; owns the device, stream and kernels
	geo decoder.DrafterGeometry

	fc         cudaWQ // [hidden, nTaps*hidden] — the fusion projection
	hiddenNorm Buffer // RMSNorm weight applied to the fused context
	finalNorm  Buffer

	// Scratch for the context fusion, sized to the widest ctx seen so far.
	ctxCap  int
	ctxIn   Buffer // [ctxCap, fcCols] f32 tap rows as handed in
	ctxQ    Buffer // quantized ctxIn
	ctxSc   Buffer
	ctxFuse Buffer // [ctxCap, hidden] fused + normed
}

// AttachDrafter uploads a block drafter's weights to the device this resident already owns.
//
// The geometry is CHECKED against the target rather than trusted: a drafter reads the target's
// residual stream, so a hidden-dim mismatch is not a resizing problem, it is the wrong pairing —
// and the failure mode is a drafter that runs and drafts noise. docs/spec/08 records that exact
// shape of bug costing a full measurement round.
func (r *cudaResident) AttachDrafter(w decoder.BlockDrafterWeights) (*residentDrafter, error) {
	geo := w.DrafterGeometry()
	if geo.Hidden != r.hidden {
		return nil, fmt.Errorf("cuda drafter: hidden %d != target hidden %d — wrong pairing", geo.Hidden, r.hidden)
	}
	if geo.Layers == 0 {
		return nil, fmt.Errorf("cuda drafter: zero layers")
	}
	d := &residentDrafter{r: r, geo: geo}
	err := r.do(func() error {
		fcw, e := packWeight(w.DrafterFC())
		if e != nil {
			return fmt.Errorf("cuda drafter: pack fc: %w", e)
		}
		d.fc = r.upW(fcw)
		d.hiddenNorm = r.up32(w.DrafterHiddenNorm())
		d.finalNorm = r.up32(w.DrafterFinalNorm())
		return r.stream.Sync()
	})
	if err != nil {
		return nil, err
	}
	return d, nil
}

// FuseContext projects the target's concatenated tap hidden states down to the trunk's width and
// norms them — the drafter's `fc` + `hiddenNorm`, batched over all rows in one pass.
//
// This is the one projection with no counterpart in a normal decoder layer, which is why it is
// the first thing built: if packWeight/upW mishandle the drafter's weights, it shows up here
// rather than eighteen kernels later.
//
// The rows are quantized to int8 on the way in, exactly as the target's own activations are, so
// the result is NOT bit-identical to the CPU f32 path — it is the same int8 arithmetic the
// resident target runs everywhere else, and the parity gate compares by cosine accordingly.
func (d *residentDrafter) FuseContext(rows [][]float32) ([][]float32, error) {
	n := len(rows)
	if n == 0 {
		return nil, fmt.Errorf("cuda drafter: empty context")
	}
	k := d.fc.K
	for i, row := range rows {
		if len(row) != k {
			return nil, fmt.Errorf("cuda drafter: context row %d is %d wide, want %d", i, len(row), k)
		}
	}
	hidden := d.geo.Hidden
	out := make([][]float32, n)
	err := d.r.do(func() error {
		if n > d.ctxCap {
			d.ctxIn = d.r.af(n * k)
			d.ctxQ = d.r.ai(n * (k / 4))
			d.ctxSc = d.r.af(n)
			d.ctxFuse = d.r.af(n * hidden)
			d.ctxCap = n
		}
		flat := make([]float32, 0, n*k)
		for _, row := range rows {
			flat = append(flat, row...)
		}
		if e := gpu.Upload(d.ctxIn, flat); e != nil {
			return e
		}
		// quantize each row, then ONE batched GEMV over all of them: the fc weights are read
		// once for the whole context instead of once per row.
		if e := d.r.launch(d.r.bQuant, LaunchConfig{GridX: uint32(n), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.ctxIn), gpu.ArgValue(int32(k)), Arg(d.ctxQ), Arg(d.ctxSc), gpu.ArgValue(int32(n))); e != nil {
			return e
		}
		if e := d.r.bGemvB(d.fc, d.ctxQ, d.ctxSc, ArgNull(), d.ctxFuse, n, 0); e != nil {
			return e
		}
		// NOT r.bNormF32B: that hardcodes the TARGET's eps and add-one flag. The drafter has
		// its own normEps, and its RMSNorm is plain (addOne=0) per blockTrunk's CPU path —
		// applying the target's would be a silent, plausible-looking wrong normalization.
		if e := d.r.launch(d.r.bNormF32, LaunchConfig{GridX: 1, GridY: uint32(n), GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.ctxFuse), Arg(d.hiddenNorm), gpu.ArgValue(int32(hidden)),
			gpu.ArgValue(float32(d.geo.NormEps)), gpu.ArgValue(int32(0)),
			gpu.ArgValue(int32(n))); e != nil {
			return e
		}
		if e := d.r.stream.Sync(); e != nil {
			return e
		}
		host := make([]float32, n*hidden)
		if e := gpu.Download(d.ctxFuse, host); e != nil {
			return e
		}
		for i := 0; i < n; i++ {
			out[i] = append([]float32(nil), host[i*hidden:(i+1)*hidden]...)
		}
		return d.r.launchErr
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
